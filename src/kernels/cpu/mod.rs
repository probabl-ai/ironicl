//! CPU kernels, chosen after runtime feature checks: the operations in
//! `ops`, with the instruction-set versions in `arch`.

#[allow(
    clippy::wildcard_imports,
    reason = "each file reaches its siblings through the cpu module; which items exist differs per target"
)]
mod arch {
    #[cfg(target_arch = "aarch64")]
    pub(crate) mod aarch64;
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    pub(crate) mod wasm32;
    #[cfg(target_arch = "x86_64")]
    pub(crate) mod x86_64;
}
#[allow(
    clippy::wildcard_imports,
    reason = "each file reaches its siblings through the cpu module; which items exist differs per target"
)]
mod ops {
    pub(crate) mod affine;
    pub(crate) mod attention;
    pub(crate) mod norm;
    pub(crate) mod pointwise;
}
#[cfg(test)]
mod tests;

use std::{fmt, str::FromStr};

use crate::{
    config::{ExecutionConfig, StorageDType},
    error::{Error, Result},
};

#[cfg(target_arch = "aarch64")]
pub(crate) use arch::aarch64::*;
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub(crate) use arch::wasm32::*;
#[cfg(target_arch = "x86_64")]
pub(crate) use arch::x86_64::*;
pub(crate) use ops::{affine::*, attention::*, norm::*, pointwise::*};

/// The instruction set the CPU kernels use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Portable,
    Neon,
    Avx512,
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Portable => "portable",
            Self::Neon => "neon",
            Self::Avx512 => "avx512",
        })
    }
}

impl Isa {
    /// The best instruction set this host supports, which the kernels use.
    #[must_use]
    pub fn detect() -> Self {
        Self::for_host(&CpuCapabilities::detect())
    }

    /// The best instruction set `host` supports.
    #[must_use]
    pub const fn for_host(host: &CpuCapabilities) -> Self {
        if host.neon {
            Self::Neon
        } else if host.avx512f {
            Self::Avx512
        } else {
            Self::Portable
        }
    }

    /// The instruction set `config` asks for on `host`, checked against the
    /// host and the storage types: BF16 storage on AVX-512 needs
    /// `AVX512_BF16`, and INT8 needs `AVX512_VNNI`.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is invalid or when an
    /// explicitly chosen instruction set lacks a required CPU capability.
    pub fn resolve(config: ExecutionConfig, host: &CpuCapabilities) -> Result<Self> {
        config.validate()?;
        match config.isa {
            IsaPreference::Auto => Ok(Self::for_host(host)),
            IsaPreference::Portable => Ok(Self::Portable),
            IsaPreference::Neon if host.neon => Ok(Self::Neon),
            IsaPreference::Neon => Err(Error::UnsupportedBackend("NEON needs an AArch64 CPU".into())),
            IsaPreference::Avx512 => {
                if !host.avx512f {
                    return Err(Error::UnsupportedBackend("AVX-512 needs AVX512F".into()));
                }
                let uses = |dtype| config.weight_dtype == dtype || config.kv_dtype == dtype;
                if uses(StorageDType::Bf16) && !host.avx512bf16 {
                    return Err(Error::UnsupportedBackend(
                        "BF16 storage on AVX-512 needs AVX512_BF16".into(),
                    ));
                }
                if uses(StorageDType::Int8) && !host.avx512vnni {
                    return Err(Error::UnsupportedBackend(
                        "INT8 storage on AVX-512 needs AVX512_VNNI".into(),
                    ));
                }
                Ok(Self::Avx512)
            }
        }
    }
}

/// The instruction set a configuration asks for, before [`Isa::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaPreference {
    Auto,
    Portable,
    Neon,
    Avx512,
}

impl fmt::Display for IsaPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Portable => "portable",
            Self::Neon => "neon",
            Self::Avx512 => "avx512",
        })
    }
}

impl FromStr for IsaPreference {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "portable" | "reference" => Ok(Self::Portable),
            "neon" => Ok(Self::Neon),
            "avx512" => Ok(Self::Avx512),
            _ => Err(Error::InvalidArgument(format!(
                "unknown ISA '{value}'; expected auto, portable, neon, or avx512"
            ))),
        }
    }
}

/// Runtime CPU capabilities relevant to Ironicl kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(clippy::struct_excessive_bools, reason = "one flag per detected CPU feature")]
pub struct CpuCapabilities {
    pub architecture: &'static str,
    pub operating_system: &'static str,
    pub neon: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512bf16: bool,
    pub avx512vnni: bool,
}

impl CpuCapabilities {
    #[must_use]
    pub fn detect() -> Self {
        #[cfg_attr(
            not(target_arch = "x86_64"),
            expect(unused_mut, reason = "only x86-64 fills in features after construction")
        )]
        let mut host = Self {
            architecture: std::env::consts::ARCH,
            operating_system: std::env::consts::OS,
            neon: cfg!(target_arch = "aarch64"),
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vl: false,
            avx512bf16: false,
            avx512vnni: false,
        };
        #[cfg(target_arch = "x86_64")]
        {
            host.avx2 = std::is_x86_feature_detected!("avx2");
            host.avx512f = std::is_x86_feature_detected!("avx512f");
            host.avx512bw = std::is_x86_feature_detected!("avx512bw");
            host.avx512vl = std::is_x86_feature_detected!("avx512vl");
            host.avx512bf16 = std::is_x86_feature_detected!("avx512bf16");
            host.avx512vnni = std::is_x86_feature_detected!("avx512vnni");
        }
        host
    }
}

impl fmt::Display for CpuCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let yes_no = |value: bool| if value { "yes" } else { "no" };
        writeln!(f, "architecture:    {}", self.architecture)?;
        writeln!(f, "operating system: {}", self.operating_system)?;
        writeln!(f, "neon:            {}", yes_no(self.neon))?;
        writeln!(f, "avx2:            {}", yes_no(self.avx2))?;
        writeln!(f, "avx512f:         {}", yes_no(self.avx512f))?;
        writeln!(f, "avx512bw:        {}", yes_no(self.avx512bw))?;
        writeln!(f, "avx512vl:        {}", yes_no(self.avx512vl))?;
        writeln!(f, "avx512bf16:      {}", yes_no(self.avx512bf16))?;
        write!(f, "avx512vnni:      {}", yes_no(self.avx512vnni))
    }
}

#[cfg(test)]
mod isa_tests {
    use super::{CpuCapabilities, Isa, IsaPreference};
    use crate::config::ExecutionConfig;

    fn avx512_host() -> CpuCapabilities {
        CpuCapabilities {
            architecture: "x86_64",
            operating_system: "linux",
            neon: false,
            avx2: true,
            avx512f: true,
            avx512bw: true,
            avx512vl: true,
            avx512bf16: true,
            avx512vnni: true,
        }
    }

    #[test]
    fn auto_picks_the_best_instruction_set() {
        assert_eq!(
            Isa::resolve(ExecutionConfig::default(), &avx512_host()),
            Ok(Isa::Avx512)
        );
        let mut host = avx512_host();
        host.avx512f = false;
        assert_eq!(Isa::resolve(ExecutionConfig::default(), &host), Ok(Isa::Portable));
    }

    #[test]
    fn avx512_needs_bf16_for_bf16_storage() {
        let config = ExecutionConfig {
            isa: IsaPreference::Avx512,
            ..ExecutionConfig::default()
        };
        let mut host = avx512_host();
        host.avx512bf16 = false;
        assert!(Isa::resolve(config, &host).is_err());
    }
}

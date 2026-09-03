use std::{fmt, str::FromStr};

use crate::{
    config::{ExecutionConfig, StorageDType},
    error::{Error, Result},
    platform::CpuCapabilities,
};

/// Backend selected after runtime capability checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Portable,
    AmdZen5Avx512,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Portable => f.write_str("portable"),
            Self::AmdZen5Avx512 => f.write_str("amd-zen5-avx512"),
        }
    }
}

/// User preference before runtime backend resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    Auto,
    Portable,
    AmdZen5Avx512,
}

impl fmt::Display for BackendPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Portable => f.write_str("portable"),
            Self::AmdZen5Avx512 => f.write_str("amd-zen5-avx512"),
        }
    }
}

impl FromStr for BackendPreference {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "portable" | "reference" => Ok(Self::Portable),
            "zen5" | "amd-zen5" | "amd-zen5-avx512" => Ok(Self::AmdZen5Avx512),
            _ => Err(Error::InvalidArgument(format!(
                "unknown backend '{value}'; expected auto, portable, or zen5"
            ))),
        }
    }
}

/// Resolve and validate a backend for a storage policy.
///
/// # Errors
///
/// Returns an error when the execution configuration is invalid or when an
/// explicitly selected backend lacks a required CPU capability.
pub fn resolve_backend(config: ExecutionConfig, host: &CpuCapabilities) -> Result<BackendKind> {
    config.validate()?;

    match config.backend {
        BackendPreference::Portable => Ok(BackendKind::Portable),
        BackendPreference::AmdZen5Avx512 => {
            validate_zen5(config, host)?;
            Ok(BackendKind::AmdZen5Avx512)
        }
        BackendPreference::Auto => {
            if validate_zen5(config, host).is_ok() {
                Ok(BackendKind::AmdZen5Avx512)
            } else {
                Ok(BackendKind::Portable)
            }
        }
    }
}

fn validate_zen5(config: ExecutionConfig, host: &CpuCapabilities) -> Result<()> {
    if !host.supports_zen5_base() {
        return Err(Error::UnsupportedBackend(
            "Zen 5 backend requires AVX2, AVX512F, AVX512BW, and AVX512VL".into(),
        ));
    }
    if (config.weight_dtype == StorageDType::Bf16 || config.kv_dtype == StorageDType::Bf16)
        && !host.avx512bf16
    {
        return Err(Error::UnsupportedBackend(
            "BF16 policy requires AVX512_BF16 on the optimized backend".into(),
        ));
    }
    if (config.weight_dtype == StorageDType::Int8 || config.kv_dtype == StorageDType::Int8)
        && !host.avx512vnni
    {
        return Err(Error::UnsupportedBackend(
            "INT8 policy requires AVX512_VNNI on the optimized backend".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zen5_host() -> CpuCapabilities {
        CpuCapabilities {
            architecture: "x86_64",
            operating_system: "linux",
            avx2: true,
            avx512f: true,
            avx512bw: true,
            avx512vl: true,
            avx512bf16: true,
            avx512vnni: true,
        }
    }

    #[test]
    fn resolves_zen5_when_required_features_exist() {
        let config = ExecutionConfig::default();
        assert_eq!(
            resolve_backend(config, &zen5_host()),
            Ok(BackendKind::AmdZen5Avx512)
        );
    }

    #[test]
    fn auto_falls_back_to_portable() {
        let mut host = zen5_host();
        host.avx512bf16 = false;
        assert_eq!(
            resolve_backend(ExecutionConfig::default(), &host),
            Ok(BackendKind::Portable)
        );
    }
}

//! CPU kernels, chosen after runtime feature checks: the operations in
//! `ops`, with the instruction-set versions in `arch`.

#![allow(
    dead_code,
    unsafe_code,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::missing_transmute_annotations
)]

mod arch {
    #[cfg(target_arch = "aarch64")]
    pub(crate) mod aarch64;
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    pub(crate) mod wasm32;
    #[cfg(target_arch = "x86_64")]
    pub(crate) mod x86_64;
}
mod ops {
    pub(crate) mod affine;
    pub(crate) mod attention;
    pub(crate) mod norm;
    pub(crate) mod pointwise;
}
#[cfg(test)]
mod tests;

#[cfg(target_arch = "aarch64")]
pub(crate) use arch::aarch64::*;
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub(crate) use arch::wasm32::*;
#[cfg(target_arch = "x86_64")]
pub(crate) use arch::x86_64::*;
pub(crate) use ops::{affine::*, attention::*, norm::*, pointwise::*};

/// The instruction set the CPU kernels use, detected once for an executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Isa {
    Portable,
    Neon,
    Avx512,
}

impl Isa {
    /// The best instruction set this host supports.
    #[must_use]
    pub(crate) fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                Self::Avx512
            } else {
                Self::Portable
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Portable
        }
    }
}

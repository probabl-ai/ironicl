//! Portable reference kernels and future architecture-specific backends.

pub(crate) mod cpu;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod metal;
pub mod reference;

//! `TabICLv2`.

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod metal;

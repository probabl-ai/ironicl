//! Layer norm.

#![allow(
    dead_code,
    unsafe_code,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::missing_transmute_annotations,
    clippy::wildcard_imports // Sibling items differ per target.
)]

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::kernels::cpu::*;

/// Apply `LayerNorm` with FP32 accumulation when the host supports it.
///
/// The caller validates dimensions and slices. Widths not divisible by 16 use
/// the portable path so the native implementation never needs a masked tail.
pub(crate) fn layer_norm_f32(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> bool {
    width.is_multiple_of(16) && layer_norm_for_target(input, rows, width, weight, bias, output)
}

/// Calculate FP32 `LayerNorm` row means and inverse standard deviations.
pub(crate) fn layer_norm_statistics_f32(
    input: &[f32],
    rows: usize,
    width: usize,
    mean: &mut [f32],
    scale: &mut [f32],
) -> bool {
    layer_norm_statistics_for_target(input, rows, width, mean, scale)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn layer_norm_for_target(
    _input: &[f32],
    _rows: usize,
    _width: usize,
    _weight: &[f32],
    _bias: Option<&[f32]>,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn layer_norm_statistics_for_target(
    _input: &[f32],
    _rows: usize,
    _width: usize,
    _mean: &mut [f32],
    _scale: &mut [f32],
) -> bool {
    false
}

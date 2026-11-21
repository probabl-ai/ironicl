//! Affine layers and their portable paths.

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_family = "wasm", target_feature = "simd128")
))]
use crate::kernels::cpu::*;

pub(crate) const MIN_SIMD_INPUT_DIM: usize = 16;

/// Smallest affine row count that goes to Accelerate on Apple Silicon. Thresholds
/// from 4 to 64 rows measured the same on an M1 Max.
pub(crate) const SGEMM_MIN_ROWS: usize = 16;

/// Whether an affine layer with `rows` input rows runs as one Accelerate
/// `sgemm`. Callers then skip their own row splitting: Accelerate uses the
/// matrix coprocessor, which all cores in a cluster share.
pub(crate) const fn affine_uses_sgemm(rows: usize) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64")) && rows >= SGEMM_MIN_ROWS
}

/// Run a native affine kernel when this host has a supported SIMD implementation.
///
/// The caller validates all slice lengths and dimensions before calling this function.
pub(crate) fn affine_f32(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    if input_dim < MIN_SIMD_INPUT_DIM {
        return false;
    }

    affine_for_target(input, rows, input_dim, weight, bias, output_dim, output)
}

/// Run a native `3 -> output_dim` affine kernel with input-major weights.
///
/// The caller validates `[rows, 3]` input, `[3, output_dim]` weight, bias, and
/// output lengths before calling this function.
pub(crate) fn affine_input3_transposed_f32(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    affine_input3_transposed_for_target(input, rows, weight, bias, output_dim, output)
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn affine_input3_transposed_for_target(
    _input: &[f32],
    _rows: usize,
    _weight: &[f32],
    _bias: &[f32],
    _output_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
pub(crate) fn affine_for_target(
    _input: &[f32],
    _rows: usize,
    _input_dim: usize,
    _weight: &[f32],
    _bias: Option<&[f32]>,
    _output_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

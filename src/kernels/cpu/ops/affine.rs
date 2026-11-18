//! Affine layers: packed FP32 and BF16 weights, and their portable paths.

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

use crate::kernels::cpu::*;

pub(crate) const MIN_SIMD_INPUT_DIM: usize = 16;

pub(crate) const F32_AFFINE_OUTPUT_BLOCK: usize = 16;

pub(crate) const F32_AFFINE_OUTPUT_GROUP_BLOCKS: usize = 4;

pub(crate) const F32_AFFINE_MAX_ROWS: usize = 4;

pub(crate) const BF16_AFFINE_OUTPUT_BLOCK: usize = 16;

pub(crate) const BF16_AFFINE_MAX_ROWS: usize = 8;

/// Smallest affine row count that goes to Accelerate on Apple Silicon. Thresholds
/// from 4 to 64 rows measured the same on an M1 Max.
pub(crate) const SGEMM_MIN_ROWS: usize = 16;

/// Whether an affine layer with `rows` input rows runs as one Accelerate
/// `sgemm`. Callers then skip their own row splitting: Accelerate uses the
/// matrix coprocessor, which all cores in a cluster share.
pub(crate) const fn affine_uses_sgemm(rows: usize) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64")) && rows >= SGEMM_MIN_ROWS
}

/// FP32 affine weights packed as `[output_block, input, output_lane]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PackedF32Affine {
    pub(crate) input_dim: usize,
    pub(crate) output_dim: usize,
    pub(crate) values: Vec<f32>,
}

impl PackedF32Affine {
    /// Pack PyTorch-layout `[output_dim, input_dim]` weights for output-lane SIMD.
    #[must_use]
    pub(crate) fn new(weight: &[f32], input_dim: usize, output_dim: usize) -> Option<Self> {
        if input_dim == 0
            || output_dim == 0
            || !output_dim.is_multiple_of(F32_AFFINE_OUTPUT_BLOCK)
            || weight.len() != input_dim.checked_mul(output_dim)?
        {
            return None;
        }

        let output_blocks = output_dim / F32_AFFINE_OUTPUT_BLOCK;
        let mut values = vec![0.0; weight.len()];
        for block in 0..output_blocks {
            for input in 0..input_dim {
                let destination = (block * input_dim + input) * F32_AFFINE_OUTPUT_BLOCK;
                for lane in 0..F32_AFFINE_OUTPUT_BLOCK {
                    let channel = block * F32_AFFINE_OUTPUT_BLOCK + lane;
                    values[destination + lane] = weight[channel * input_dim + input];
                }
            }
        }
        Some(Self {
            input_dim,
            output_dim,
            values,
        })
    }

    /// Execute the packed FP32 affine kernel.
    pub(crate) fn run(&self, input: &[f32], rows: usize, bias: Option<&[f32]>, output: &mut [f32]) -> bool {
        if rows == 0
            || input.len() != rows.saturating_mul(self.input_dim)
            || output.len() != rows.saturating_mul(self.output_dim)
            || bias.is_some_and(|values| values.len() != self.output_dim)
        {
            return false;
        }
        affine_packed_f32_for_target(input, rows, self, bias, output)
    }
}

/// First-order terms used to compensate BF16 affine quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bf16AffineCompensation {
    None,
    Input,
    Weight,
    FirstOrder,
}

impl Bf16AffineCompensation {
    const fn uses_input_residual(self) -> bool {
        matches!(self, Self::Input | Self::FirstOrder)
    }

    const fn uses_weight_residual(self) -> bool {
        matches!(self, Self::Weight | Self::FirstOrder)
    }
}

/// BF16 affine weights packed as `[output_block, input_pair, output_lane, pair]`.
///
/// Each 512-bit weight load contains both K values for 16 output channels, so
/// one `VDPBF16PS` updates a complete output-channel block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackedBf16Affine {
    pub(crate) input_dim: usize,
    pub(crate) output_dim: usize,
    pub(crate) values: Vec<u16>,
    pub(crate) residual_values: Vec<u16>,
}

impl PackedBf16Affine {
    /// Quantize and pack PyTorch-layout `[output_dim, input_dim]` FP32 weights.
    #[must_use]
    pub(crate) fn new(weight: &[f32], input_dim: usize, output_dim: usize) -> Option<Self> {
        if input_dim == 0
            || !input_dim.is_multiple_of(2)
            || output_dim == 0
            || !output_dim.is_multiple_of(BF16_AFFINE_OUTPUT_BLOCK)
            || weight.len() != input_dim.checked_mul(output_dim)?
        {
            return None;
        }

        let input_pairs = input_dim / 2;
        let output_blocks = output_dim / BF16_AFFINE_OUTPUT_BLOCK;
        let packed_len = output_blocks
            .checked_mul(input_pairs)?
            .checked_mul(BF16_AFFINE_OUTPUT_BLOCK * 2)?;
        let mut values = vec![0_u16; packed_len];
        let mut residual_values = vec![0_u16; packed_len];
        for block in 0..output_blocks {
            for pair in 0..input_pairs {
                let packed_pair = (block * input_pairs + pair) * BF16_AFFINE_OUTPUT_BLOCK * 2;
                for lane in 0..BF16_AFFINE_OUTPUT_BLOCK {
                    let channel = block * BF16_AFFINE_OUTPUT_BLOCK + lane;
                    let source = channel * input_dim + pair * 2;
                    let destination = packed_pair + lane * 2;
                    let first = f32_to_bf16_rne(weight[source]);
                    let second = f32_to_bf16_rne(weight[source + 1]);
                    values[destination] = first;
                    values[destination + 1] = second;
                    residual_values[destination] = f32_to_bf16_rne(weight[source] - bf16_bits_to_f32(first));
                    residual_values[destination + 1] =
                        f32_to_bf16_rne(weight[source + 1] - bf16_bits_to_f32(second));
                }
            }
        }
        Some(Self {
            input_dim,
            output_dim,
            values,
            residual_values,
        })
    }

    /// Quantize FP32 rows and execute the packed affine kernel with FP32 output.
    ///
    /// The caller owns `input_bf16`, which permits workspace reuse across calls.
    pub(crate) fn run_f32(
        &self,
        input: &[f32],
        rows: usize,
        bias: Option<&[f32]>,
        output: &mut [f32],
        input_bf16: &mut [u16],
    ) -> bool {
        if input.len() != rows.saturating_mul(self.input_dim) || input_bf16.len() != input.len() {
            return false;
        }
        if !convert_f32_to_bf16(input, input_bf16) {
            return false;
        }
        self.run_bf16(input_bf16, rows, bias, output)
    }

    /// Execute with optional first-order input and weight residual terms.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_f32_compensated(
        &self,
        input: &[f32],
        rows: usize,
        bias: Option<&[f32]>,
        output: &mut [f32],
        input_bf16: &mut [u16],
        input_residual_bf16: &mut [u16],
        compensation: Bf16AffineCompensation,
    ) -> bool {
        if input.len() != rows.saturating_mul(self.input_dim)
            || input_bf16.len() != input.len()
            || (compensation.uses_input_residual() && input_residual_bf16.len() != input.len())
        {
            return false;
        }
        if !convert_f32_to_bf16(input, input_bf16) {
            return false;
        }
        let input_residual = if compensation.uses_input_residual() {
            for ((value, high), low) in input.iter().zip(&*input_bf16).zip(&mut *input_residual_bf16) {
                *low = f32_to_bf16_rne(*value - bf16_bits_to_f32(*high));
            }
            Some(&*input_residual_bf16)
        } else {
            None
        };
        self.run_bf16_components(
            input_bf16,
            input_residual,
            rows,
            bias,
            output,
            compensation.uses_weight_residual(),
        )
    }

    /// Execute from caller-owned row-major BF16 activations with FP32 output.
    pub(crate) fn run_bf16(
        &self,
        input: &[u16],
        rows: usize,
        bias: Option<&[f32]>,
        output: &mut [f32],
    ) -> bool {
        if rows == 0
            || input.len() != rows.saturating_mul(self.input_dim)
            || output.len() != rows.saturating_mul(self.output_dim)
            || bias.is_some_and(|values| values.len() != self.output_dim)
        {
            return false;
        }
        self.run_bf16_components(input, None, rows, bias, output, false)
    }

    fn run_bf16_components(
        &self,
        input: &[u16],
        input_residual: Option<&[u16]>,
        rows: usize,
        bias: Option<&[f32]>,
        output: &mut [f32],
        use_weight_residual: bool,
    ) -> bool {
        if rows == 0
            || input.len() != rows.saturating_mul(self.input_dim)
            || input_residual.is_some_and(|values| values.len() != input.len())
            || output.len() != rows.saturating_mul(self.output_dim)
            || bias.is_some_and(|values| values.len() != self.output_dim)
        {
            return false;
        }
        affine_packed_bf16_for_target(
            input,
            input_residual,
            rows,
            self,
            bias,
            output,
            use_weight_residual,
        )
    }
}

/// Report whether packed BF16 affine kernels can execute on this host.
#[must_use]
pub(crate) fn packed_bf16_affine_supported() -> bool {
    packed_bf16_affine_supported_for_target()
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) const fn packed_bf16_affine_supported_for_target() -> bool {
    false
}

/// Report whether packed FP32 affine kernels can execute on this host.
#[must_use]
pub(crate) fn packed_f32_affine_supported() -> bool {
    packed_f32_affine_supported_for_target()
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) const fn packed_f32_affine_supported_for_target() -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn affine_packed_f32_for_target(
    _input: &[f32],
    _rows: usize,
    _packed: &PackedF32Affine,
    _bias: Option<&[f32]>,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn affine_packed_bf16_for_target(
    _input: &[u16],
    _input_residual: Option<&[u16]>,
    _rows: usize,
    _packed: &PackedBf16Affine,
    _bias: Option<&[f32]>,
    _output: &mut [f32],
    _use_weight_residual: bool,
) -> bool {
    false
}

/// Run a native affine kernel when this host has a supported SIMD implementation.
///
/// The caller validates all slice lengths and dimensions before calling this function.
#[allow(clippy::too_many_arguments)]
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

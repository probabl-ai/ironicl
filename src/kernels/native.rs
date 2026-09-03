//! Native FP32 kernels selected after runtime CPU feature checks.

#![allow(
    dead_code,
    unsafe_code,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::missing_transmute_annotations
)]

const MIN_SIMD_INPUT_DIM: usize = 16;
const F32_AFFINE_OUTPUT_BLOCK: usize = 16;
const F32_AFFINE_OUTPUT_GROUP_BLOCKS: usize = 4;
const F32_AFFINE_MAX_ROWS: usize = 4;
const BF16_AFFINE_OUTPUT_BLOCK: usize = 16;
const BF16_AFFINE_MAX_ROWS: usize = 8;
const MAX_PACKED_ATTENTION_QUERY_ROWS: usize = 16;
const SQRT_2: f32 = core::f32::consts::SQRT_2;

/// FP32 affine weights packed as `[output_block, input, output_lane]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PackedF32Affine {
    input_dim: usize,
    output_dim: usize,
    values: Vec<f32>,
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
    input_dim: usize,
    output_dim: usize,
    values: Vec<u16>,
    residual_values: Vec<u16>,
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

#[cfg(target_arch = "x86_64")]
fn packed_bf16_affine_supported_for_target() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512bf16")
}

#[cfg(not(target_arch = "x86_64"))]
const fn packed_bf16_affine_supported_for_target() -> bool {
    false
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    u16::try_from(bits.wrapping_add(rounding_bias) >> 16).expect("rounded BF16 bits fit u16")
}

fn bf16_bits_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// Native attention implementation selected once for an executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionBackend {
    Portable,
    Neon,
    Avx512,
}

/// Detect the best attention backend supported by this host.
#[must_use]
pub(crate) fn attention_backend() -> AttentionBackend {
    #[cfg(target_arch = "aarch64")]
    {
        AttentionBackend::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            AttentionBackend::Avx512
        } else {
            AttentionBackend::Portable
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        AttentionBackend::Portable
    }
}

/// Apply the native GELU approximation when the host supports it.
///
/// The caller validates that `input` and `output` have equal lengths.
pub(crate) fn gelu_f32(input: &[f32], output: &mut [f32]) -> bool {
    gelu_for_target(input, output)
}

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
    layer_norm_for_target(input, rows, width, weight, bias, output)
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

/// Apply `query * base_scale * (1 + tanh(mlp_output))` natively.
///
/// The caller validates the head-major slice lengths and dimensions.
pub(crate) fn qass_modulate_f32(
    queries: &[f32],
    base_scale: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    qass_modulate_for_target(queries, base_scale, heads, rows, head_dim, output)
}

/// Apply `output = weight * value + output` across one attention value row.
///
/// The caller validates equal slice lengths. Each lane performs one FP32 FMA,
/// so repeated source-row calls preserve the scalar accumulation order.
pub(crate) fn attention_value_fma_f32(
    backend: AttentionBackend,
    weight: f32,
    value: &[f32],
    output: &mut [f32],
) -> bool {
    attention_value_fma_for_target(backend, weight, value, output)
}

/// Compute one attention score dot product with native SIMD when available.
///
/// The caller validates equal slice lengths.
pub(crate) fn attention_dot_f32(backend: AttentionBackend, left: &[f32], right: &[f32]) -> Option<f32> {
    attention_dot_for_target(backend, left, right)
}

/// Compute one row-major `[query_rows, source_rows]` attention score tile.
///
/// Query and key rows are contiguous. The caller validates all slice lengths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_tile_f32(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    attention_score_tile_for_target(
        backend,
        query,
        query_rows,
        key,
        source_rows,
        head_dim,
        scale,
        scores,
    )
}

/// Compute one score tile directly from BF16 K storage with FP32 accumulation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_tile_bf16_f32(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    attention_score_tile_bf16_for_target(
        backend,
        query,
        query_rows,
        key,
        source_rows,
        head_dim,
        scale,
        scores,
    )
}

/// Compute scores from dimension-major packed K using SIMD lanes as source rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_packed_key_tile_f32(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    attention_score_packed_key_tile_for_target(
        backend,
        query,
        query_rows,
        packed_key,
        source_rows,
        head_dim,
        scale,
        scores,
    )
}

/// Compute scores from a dimension-major K tile padded to a SIMD source width.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_packed_key_padded_tile_f32(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    packed_source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    attention_score_packed_key_padded_tile_for_target(
        backend,
        query,
        query_rows,
        packed_key,
        source_rows,
        packed_source_rows,
        head_dim,
        scale,
        scores,
    )
}

/// Compute scores from dimension-major packed BF16 K with FP32 accumulation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_packed_key_bf16_tile_f32(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    packed_key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    attention_score_packed_key_bf16_tile_for_target(
        backend,
        query,
        query_rows,
        packed_key,
        source_rows,
        head_dim,
        scale,
        scores,
    )
}

/// Accumulate one row-major softmax-weight tile into query output rows.
///
/// The caller validates all slice lengths. Source order is preserved for each
/// query row, while one V row is reused across the complete query block.
pub(crate) fn attention_value_tile_fma_f32(
    backend: AttentionBackend,
    weights: &[f32],
    query_rows: usize,
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    attention_value_tile_fma_for_target(backend, weights, query_rows, value, source_rows, head_dim, output)
}

/// Accumulate one weighted-value tile directly from BF16 V storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_value_tile_bf16_fma_f32(
    backend: AttentionBackend,
    weights: &[f32],
    query_rows: usize,
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    attention_value_tile_bf16_fma_for_target(
        backend,
        weights,
        query_rows,
        value,
        source_rows,
        head_dim,
        output,
    )
}

/// Pack source-major BF16 V as `[dimension_block, source_pair, dimension_lane, pair]`.
pub(crate) fn pack_attention_value_bf16(
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    packed: &mut [u16],
) -> bool {
    if source_rows == 0
        || !source_rows.is_multiple_of(2)
        || head_dim == 0
        || !head_dim.is_multiple_of(16)
        || value.len() != source_rows.saturating_mul(head_dim)
        || packed.len() != value.len()
    {
        return false;
    }
    let source_pairs = source_rows / 2;
    let dimension_blocks = head_dim / 16;
    for block in 0..dimension_blocks {
        for pair in 0..source_pairs {
            let destination = (block * source_pairs + pair) * 32;
            for lane in 0..16 {
                let dimension = block * 16 + lane;
                packed[destination + lane * 2] = value[(pair * 2) * head_dim + dimension];
                packed[destination + lane * 2 + 1] = value[(pair * 2 + 1) * head_dim + dimension];
            }
        }
    }
    true
}

/// Accumulate FP32 softmax weights times packed BF16 V through `VDPBF16PS`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_value_packed_bf16_dot_f32(
    backend: AttentionBackend,
    weights: &[f32],
    query_rows: usize,
    packed_value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
    weights_bf16: &mut [u16],
) -> bool {
    if backend != AttentionBackend::Avx512
        || query_rows == 0
        || query_rows > 8
        || !source_rows.is_multiple_of(2)
        || !head_dim.is_multiple_of(16)
        || weights.len() != query_rows.saturating_mul(source_rows)
        || weights_bf16.len() != weights.len()
        || packed_value.len() != source_rows.saturating_mul(head_dim)
        || output.len() != query_rows.saturating_mul(head_dim)
        || !packed_bf16_affine_supported_for_target()
    {
        return false;
    }
    if !convert_f32_to_bf16(weights, weights_bf16) {
        return false;
    }
    attention_value_packed_bf16_dot_for_target(
        weights_bf16,
        query_rows,
        packed_value,
        source_rows,
        head_dim,
        output,
    )
}

/// Find an attention score maximum and report whether any lane is NaN.
pub(crate) fn attention_score_maximum_f32(backend: AttentionBackend, scores: &[f32]) -> Option<(f32, bool)> {
    attention_score_maximum_for_target(backend, scores)
}

/// Replace scores with `exp(score - maximum)` and return their FP32 sum.
///
/// Returns `None` when the selected backend has no validated vector path.
pub(crate) fn softmax_exp_sum_f32(
    backend: AttentionBackend,
    scores: &mut [f32],
    maximum: f32,
) -> Option<f32> {
    static EXACT_SOFTMAX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *EXACT_SOFTMAX.get_or_init(|| std::env::var_os("IRONICL_EXACT_SOFTMAX").is_some()) {
        return None;
    }
    softmax_exp_sum_for_target(backend, scores, maximum)
}

/// Expand BF16 storage values into FP32 tile scratch.
pub(crate) fn bf16_to_f32(input: &[u16], output: &mut [f32]) -> bool {
    if input.len() != output.len() {
        return false;
    }
    bf16_to_f32_for_target(input, output)
}

#[cfg(target_arch = "x86_64")]
fn bf16_to_f32_for_target(input: &[u16], output: &mut [f32]) -> bool {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        unsafe { bf16_to_f32_avx512(input, output) };
        true
    } else {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn bf16_to_f32_for_target(_input: &[u16], _output: &mut [f32]) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn bf16_to_f32_avx512(input: &[u16], output: &mut [f32]) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm512_castsi512_ps, _mm512_cvtepu16_epi32, _mm512_slli_epi32,
        _mm512_storeu_ps,
    };
    unsafe {
        let end = input.len() / 16 * 16;
        for offset in (0..end).step_by(16) {
            let packed = _mm256_loadu_si256(input.as_ptr().add(offset).cast::<__m256i>());
            let expanded = _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(packed));
            _mm512_storeu_ps(output.as_mut_ptr().add(offset), _mm512_castsi512_ps(expanded));
        }
        for offset in end..input.len() {
            output[offset] = f32::from_bits(u32::from(input[offset]) << 16);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn convert_f32_to_bf16(input: &[f32], output: &mut [u16]) -> bool {
    if input.len() != output.len() || !std::is_x86_feature_detected!("avx512bf16") {
        return false;
    }
    unsafe { convert_f32_to_bf16_avx512(input, output) };
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn convert_f32_to_bf16(_input: &[f32], _output: &mut [u16]) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn convert_f32_to_bf16_avx512(input: &[f32], output: &mut [u16]) {
    use core::arch::x86_64::{__m256i, _mm256_storeu_si256, _mm512_cvtneps_pbh, _mm512_loadu_ps};

    unsafe {
        let end = input.len() / 16 * 16;
        for offset in (0..end).step_by(16) {
            let packed: __m256i =
                core::mem::transmute(_mm512_cvtneps_pbh(_mm512_loadu_ps(input.as_ptr().add(offset))));
            _mm256_storeu_si256(output.as_mut_ptr().add(offset).cast::<__m256i>(), packed);
        }
        for offset in end..input.len() {
            output[offset] = f32_to_bf16_rne(input[offset]);
        }
    }
}

/// Report whether packed FP32 affine kernels can execute on this host.
#[must_use]
pub(crate) fn packed_f32_affine_supported() -> bool {
    packed_f32_affine_supported_for_target()
}

#[cfg(target_arch = "x86_64")]
fn packed_f32_affine_supported_for_target() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
const fn packed_f32_affine_supported_for_target() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn affine_packed_f32_for_target(
    input: &[f32],
    rows: usize,
    packed: &PackedF32Affine,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> bool {
    if packed_f32_affine_supported_for_target() {
        unsafe { affine_packed_f32_avx512(input, rows, packed, bias, output) };
        true
    } else {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn affine_packed_f32_for_target(
    _input: &[f32],
    _rows: usize,
    _packed: &PackedF32Affine,
    _bias: Option<&[f32]>,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn affine_packed_f32_avx512(
    input: &[f32],
    rows: usize,
    packed: &PackedF32Affine,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    unsafe {
        let output_blocks = packed.output_dim / F32_AFFINE_OUTPUT_BLOCK;
        for row_start in (0..rows).step_by(F32_AFFINE_MAX_ROWS) {
            let block_rows = (rows - row_start).min(F32_AFFINE_MAX_ROWS);
            for block_start in (0..output_blocks).step_by(F32_AFFINE_OUTPUT_GROUP_BLOCKS) {
                let group_blocks = (output_blocks - block_start).min(F32_AFFINE_OUTPUT_GROUP_BLOCKS);
                let mut accumulators =
                    [[_mm512_setzero_ps(); F32_AFFINE_OUTPUT_GROUP_BLOCKS]; F32_AFFINE_MAX_ROWS];
                for row in 0..block_rows {
                    for block in 0..group_blocks {
                        accumulators[row][block] = if let Some(bias) = bias {
                            _mm512_loadu_ps(
                                bias.as_ptr().add((block_start + block) * F32_AFFINE_OUTPUT_BLOCK),
                            )
                        } else {
                            _mm512_setzero_ps()
                        };
                    }
                }
                for column in 0..packed.input_dim {
                    let mut weights = [_mm512_setzero_ps(); F32_AFFINE_OUTPUT_GROUP_BLOCKS];
                    for block in 0..group_blocks {
                        weights[block] = _mm512_loadu_ps(packed.values.as_ptr().add(
                            ((block_start + block) * packed.input_dim + column) * F32_AFFINE_OUTPUT_BLOCK,
                        ));
                    }
                    for row in 0..block_rows {
                        let value = _mm512_set1_ps(input[(row_start + row) * packed.input_dim + column]);
                        for block in 0..group_blocks {
                            accumulators[row][block] =
                                _mm512_fmadd_ps(weights[block], value, accumulators[row][block]);
                        }
                    }
                }
                for row in 0..block_rows {
                    for block in 0..group_blocks {
                        _mm512_storeu_ps(
                            output.as_mut_ptr().add(
                                (row_start + row) * packed.output_dim
                                    + (block_start + block) * F32_AFFINE_OUTPUT_BLOCK,
                            ),
                            accumulators[row][block],
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn affine_packed_bf16_for_target(
    input: &[u16],
    input_residual: Option<&[u16]>,
    rows: usize,
    packed: &PackedBf16Affine,
    bias: Option<&[f32]>,
    output: &mut [f32],
    use_weight_residual: bool,
) -> bool {
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512bf16")
    {
        unsafe {
            affine_packed_bf16_avx512(
                input,
                input_residual,
                rows,
                packed,
                bias,
                output,
                use_weight_residual,
            );
        }
        true
    } else {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn affine_packed_bf16_for_target(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
unsafe fn affine_packed_bf16_avx512(
    input: &[u16],
    input_residual: Option<&[u16]>,
    rows: usize,
    packed: &PackedBf16Affine,
    bias: Option<&[f32]>,
    output: &mut [f32],
    use_weight_residual: bool,
) {
    use core::arch::x86_64::{
        __m512bh, _mm512_dpbf16_ps, _mm512_loadu_epi16, _mm512_loadu_ps, _mm512_set1_epi32,
        _mm512_setzero_ps, _mm512_storeu_ps,
    };

    unsafe {
        let input_pairs = packed.input_dim / 2;
        let output_blocks = packed.output_dim / BF16_AFFINE_OUTPUT_BLOCK;
        for row_start in (0..rows).step_by(BF16_AFFINE_MAX_ROWS) {
            let block_rows = (rows - row_start).min(BF16_AFFINE_MAX_ROWS);
            for block in 0..output_blocks {
                let initial = if let Some(bias) = bias {
                    _mm512_loadu_ps(bias.as_ptr().add(block * BF16_AFFINE_OUTPUT_BLOCK))
                } else {
                    _mm512_setzero_ps()
                };
                let mut accumulators = [initial; BF16_AFFINE_MAX_ROWS];
                for pair in 0..input_pairs {
                    let weight_offset = (block * input_pairs + pair) * BF16_AFFINE_OUTPUT_BLOCK * 2;
                    let weights: __m512bh = core::mem::transmute(_mm512_loadu_epi16(
                        packed.values.as_ptr().add(weight_offset).cast::<i16>(),
                    ));
                    let weight_residual: Option<__m512bh> = use_weight_residual.then(|| {
                        core::mem::transmute(_mm512_loadu_epi16(
                            packed.residual_values.as_ptr().add(weight_offset).cast::<i16>(),
                        ))
                    });
                    for row in 0..block_rows {
                        let input_offset = (row_start + row) * packed.input_dim + pair * 2;
                        let pair_bits = input.as_ptr().add(input_offset).cast::<u32>().read_unaligned();
                        let inputs: __m512bh = core::mem::transmute(_mm512_set1_epi32(i32::from_ne_bytes(
                            pair_bits.to_ne_bytes(),
                        )));
                        accumulators[row] = _mm512_dpbf16_ps(accumulators[row], inputs, weights);
                        if let Some(input_residual) = input_residual {
                            let pair_bits = input_residual
                                .as_ptr()
                                .add(input_offset)
                                .cast::<u32>()
                                .read_unaligned();
                            let residual: __m512bh = core::mem::transmute(_mm512_set1_epi32(
                                i32::from_ne_bytes(pair_bits.to_ne_bytes()),
                            ));
                            accumulators[row] = _mm512_dpbf16_ps(accumulators[row], residual, weights);
                        }
                        if let Some(weight_residual) = weight_residual {
                            accumulators[row] = _mm512_dpbf16_ps(accumulators[row], inputs, weight_residual);
                        }
                    }
                }
                for row in 0..block_rows {
                    _mm512_storeu_ps(
                        output
                            .as_mut_ptr()
                            .add((row_start + row) * packed.output_dim + block * BF16_AFFINE_OUTPUT_BLOCK),
                        accumulators[row],
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_score_tile_for_target(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512 && matches!(head_dim, 16 | 64) {
        unsafe {
            attention_score_tile_avx512(query, query_rows, key, source_rows, head_dim, scale, scores);
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_score_tile_bf16_for_target(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512
        && std::is_x86_feature_detected!("avx512bw")
        && query_rows <= MAX_PACKED_ATTENTION_QUERY_ROWS
        && matches!(head_dim, 16 | 64)
    {
        unsafe {
            attention_score_tile_bf16_avx512(query, query_rows, key, source_rows, head_dim, scale, scores);
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_score_packed_key_tile_for_target(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512
        && query_rows <= MAX_PACKED_ATTENTION_QUERY_ROWS
        && source_rows.is_multiple_of(16)
        && matches!(head_dim, 16 | 64)
    {
        unsafe {
            attention_score_packed_key_tile_avx512(
                query,
                query_rows,
                packed_key,
                source_rows,
                head_dim,
                scale,
                scores,
            );
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_score_packed_key_padded_tile_for_target(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    packed_source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512
        && query_rows <= MAX_PACKED_ATTENTION_QUERY_ROWS
        && source_rows > 0
        && source_rows <= packed_source_rows
        && packed_source_rows.is_multiple_of(16)
        && packed_source_rows == source_rows.div_ceil(16) * 16
        && matches!(head_dim, 16 | 64)
    {
        unsafe {
            attention_score_packed_key_padded_tile_avx512(
                query,
                query_rows,
                packed_key,
                source_rows,
                packed_source_rows,
                head_dim,
                scale,
                scores,
            );
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_score_packed_key_bf16_tile_for_target(
    backend: AttentionBackend,
    query: &[f32],
    query_rows: usize,
    packed_key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512
        && std::is_x86_feature_detected!("avx512bw")
        && query_rows <= 8
        && source_rows.is_multiple_of(16)
        && matches!(head_dim, 16 | 64)
    {
        unsafe {
            attention_score_packed_key_bf16_tile_avx512(
                query,
                query_rows,
                packed_key,
                source_rows,
                head_dim,
                scale,
                scores,
            );
        }
        true
    } else {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn attention_score_packed_key_bf16_tile_for_target(
    _backend: AttentionBackend,
    _query: &[f32],
    _query_rows: usize,
    _packed_key: &[u16],
    _source_rows: usize,
    _head_dim: usize,
    _scale: f32,
    _scores: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn attention_score_packed_key_tile_for_target(
    _backend: AttentionBackend,
    _query: &[f32],
    _query_rows: usize,
    _packed_key: &[f32],
    _source_rows: usize,
    _head_dim: usize,
    _scale: f32,
    _scores: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn attention_score_packed_key_padded_tile_for_target(
    _backend: AttentionBackend,
    _query: &[f32],
    _query_rows: usize,
    _packed_key: &[f32],
    _source_rows: usize,
    _packed_source_rows: usize,
    _head_dim: usize,
    _scale: f32,
    _scores: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn attention_score_tile_bf16_for_target(
    _backend: AttentionBackend,
    _query: &[f32],
    _query_rows: usize,
    _key: &[u16],
    _source_rows: usize,
    _head_dim: usize,
    _scale: f32,
    _scores: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn attention_score_tile_for_target(
    _backend: AttentionBackend,
    _query: &[f32],
    _query_rows: usize,
    _key: &[f32],
    _source_rows: usize,
    _head_dim: usize,
    _scale: f32,
    _scores: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_value_tile_fma_for_target(
    backend: AttentionBackend,
    weights: &[f32],
    query_rows: usize,
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512 && matches!(head_dim, 16 | 64) {
        unsafe {
            attention_value_tile_fma_avx512(weights, query_rows, value, source_rows, head_dim, output);
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_value_tile_bf16_fma_for_target(
    backend: AttentionBackend,
    weights: &[f32],
    query_rows: usize,
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512
        && std::is_x86_feature_detected!("avx512bw")
        && matches!(head_dim, 16 | 64)
    {
        unsafe {
            attention_value_tile_bf16_fma_avx512(weights, query_rows, value, source_rows, head_dim, output);
        }
        true
    } else {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn attention_value_tile_bf16_fma_for_target(
    _backend: AttentionBackend,
    _weights: &[f32],
    _query_rows: usize,
    _value: &[u16],
    _source_rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn softmax_exp_sum_for_target(backend: AttentionBackend, scores: &mut [f32], maximum: f32) -> Option<f32> {
    (backend == AttentionBackend::Avx512).then(|| unsafe { softmax_exp_sum_avx512(scores, maximum) })
}

#[cfg(not(target_arch = "x86_64"))]
fn softmax_exp_sum_for_target(_backend: AttentionBackend, _scores: &mut [f32], _maximum: f32) -> Option<f32> {
    None
}

#[cfg(target_arch = "x86_64")]
fn attention_score_maximum_for_target(backend: AttentionBackend, scores: &[f32]) -> Option<(f32, bool)> {
    (backend == AttentionBackend::Avx512).then(|| unsafe { attention_score_maximum_avx512(scores) })
}

#[cfg(not(target_arch = "x86_64"))]
fn attention_score_maximum_for_target(_backend: AttentionBackend, _scores: &[f32]) -> Option<(f32, bool)> {
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn attention_score_maximum_avx512(scores: &[f32]) -> (f32, bool) {
    use core::arch::x86_64::{
        _CMP_UNORD_Q, _mm512_cmp_ps_mask, _mm512_loadu_ps, _mm512_max_ps, _mm512_reduce_max_ps,
        _mm512_set1_ps,
    };

    unsafe {
        let mut maximum = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut has_nan = false;
        let end = scores.len() / 16 * 16;
        for offset in (0..end).step_by(16) {
            let values = _mm512_loadu_ps(scores.as_ptr().add(offset));
            has_nan |= _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(values, values) != 0;
            maximum = _mm512_max_ps(maximum, values);
        }
        let mut maximum = _mm512_reduce_max_ps(maximum);
        for value in &scores[end..] {
            has_nan |= value.is_nan();
            maximum = maximum.max(*value);
        }
        (maximum, has_nan)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_exp_sum_avx512(scores: &mut [f32], maximum: f32) -> f32 {
    use core::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _mm512_add_epi32, _mm512_castsi512_ps,
        _mm512_cvt_roundps_epi32, _mm512_cvtepi32_ps, _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps,
        _mm512_max_ps, _mm512_mul_ps, _mm512_reduce_add_ps, _mm512_set1_epi32, _mm512_set1_ps,
        _mm512_slli_epi32, _mm512_storeu_ps, _mm512_sub_ps,
    };
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    const LOG2_E: f32 = core::f32::consts::LOG2_E;
    const LN_2: f32 = core::f32::consts::LN_2;
    const MINIMUM: f32 = -87.0;
    unsafe {
        let maximum_vector = _mm512_set1_ps(maximum);
        let minimum = _mm512_set1_ps(MINIMUM);
        let log2_e = _mm512_set1_ps(LOG2_E);
        let ln_2 = _mm512_set1_ps(LN_2);
        let mut sum = _mm512_set1_ps(0.0);
        let end = scores.len() / 16 * 16;
        for offset in (0..end).step_by(16) {
            let x = _mm512_max_ps(
                _mm512_sub_ps(_mm512_loadu_ps(scores.as_ptr().add(offset)), maximum_vector),
                minimum,
            );
            let exponent = _mm512_cvt_roundps_epi32::<ROUND>(_mm512_mul_ps(x, log2_e));
            let remainder = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(exponent), ln_2, x);
            let mut polynomial = _mm512_set1_ps(1.0 / 5_040.0);
            for coefficient in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
                polynomial = _mm512_fmadd_ps(polynomial, remainder, _mm512_set1_ps(coefficient));
            }
            let power = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
                exponent,
                _mm512_set1_epi32(127),
            )));
            let values = _mm512_mul_ps(polynomial, power);
            _mm512_storeu_ps(scores.as_mut_ptr().add(offset), values);
            sum = _mm512_fmadd_ps(values, _mm512_set1_ps(1.0), sum);
        }
        let mut total = _mm512_reduce_add_ps(sum);
        for score in &mut scores[end..] {
            *score = fast_exp_negative(*score - maximum);
            total += *score;
        }
        total
    }
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn fast_exp_negative(value: f32) -> f32 {
    let value = value.max(-87.0);
    let exponent = (value * core::f32::consts::LOG2_E).round() as i32;
    let remainder = value - exponent as f32 * core::f32::consts::LN_2;
    let mut polynomial = 1.0_f32 / 5_040.0;
    for coefficient in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        polynomial = polynomial.mul_add(remainder, coefficient);
    }
    let exponent_bits = u32::try_from(exponent + 127).expect("clamped exp exponent is non-negative");
    polynomial * f32::from_bits(exponent_bits << 23)
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn attention_value_tile_fma_for_target(
    _backend: AttentionBackend,
    _weights: &[f32],
    _query_rows: usize,
    _value: &[f32],
    _source_rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_score_tile_avx512(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};
    unsafe {
        for source in 0..source_rows {
            let key = key.as_ptr().add(source * head_dim);
            for row in 0..query_rows {
                let query = query.as_ptr().add(row * head_dim);
                let mut accumulator = _mm512_setzero_ps();
                for dimension in (0..head_dim).step_by(16) {
                    accumulator = _mm512_fmadd_ps(
                        _mm512_loadu_ps(query.add(dimension)),
                        _mm512_loadu_ps(key.add(dimension)),
                        accumulator,
                    );
                }
                scores[row * source_rows + source] = _mm512_reduce_add_ps(accumulator) * scale;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_score_tile_bf16_avx512(
    query: &[f32],
    query_rows: usize,
    key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm512_castsi512_ps, _mm512_cvtepu16_epi32, _mm512_fmadd_ps,
        _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps, _mm512_slli_epi32,
    };
    unsafe {
        for source in 0..source_rows {
            let key = key.as_ptr().add(source * head_dim);
            let mut accumulators = [_mm512_setzero_ps(); MAX_PACKED_ATTENTION_QUERY_ROWS];
            for dimension in (0..head_dim).step_by(16) {
                let packed = _mm256_loadu_si256(key.add(dimension).cast::<__m256i>());
                let key = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(packed)));
                for row in 0..query_rows {
                    accumulators[row] = _mm512_fmadd_ps(
                        _mm512_loadu_ps(query.as_ptr().add(row * head_dim + dimension)),
                        key,
                        accumulators[row],
                    );
                }
            }
            for row in 0..query_rows {
                scores[row * source_rows + source] = _mm512_reduce_add_ps(accumulators[row]) * scale;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_score_packed_key_tile_avx512(
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };
    unsafe {
        if query_rows == 16 {
            attention_score_packed_key_tile_avx512_rows::<16>(
                query,
                packed_key,
                source_rows,
                head_dim,
                scale,
                scores,
            );
            return;
        }
        let scale = _mm512_set1_ps(scale);
        for source in (0..source_rows).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); MAX_PACKED_ATTENTION_QUERY_ROWS];
            for dimension in 0..head_dim {
                let key = _mm512_loadu_ps(packed_key.as_ptr().add(dimension * source_rows + source));
                for row in 0..query_rows {
                    accumulators[row] = _mm512_fmadd_ps(
                        key,
                        _mm512_set1_ps(query[row * head_dim + dimension]),
                        accumulators[row],
                    );
                }
            }
            for row in 0..query_rows {
                _mm512_storeu_ps(
                    scores.as_mut_ptr().add(row * source_rows + source),
                    _mm512_mul_ps(accumulators[row], scale),
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn attention_score_packed_key_tile_avx512_rows<const QUERY_ROWS: usize>(
    query: &[f32],
    packed_key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };
    unsafe {
        let scale = _mm512_set1_ps(scale);
        for source in (0..source_rows).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); QUERY_ROWS];
            for dimension in 0..head_dim {
                let key = _mm512_loadu_ps(packed_key.as_ptr().add(dimension * source_rows + source));
                for row in 0..QUERY_ROWS {
                    accumulators[row] = _mm512_fmadd_ps(
                        key,
                        _mm512_set1_ps(query[row * head_dim + dimension]),
                        accumulators[row],
                    );
                }
            }
            for row in 0..QUERY_ROWS {
                _mm512_storeu_ps(
                    scores.as_mut_ptr().add(row * source_rows + source),
                    _mm512_mul_ps(accumulators[row], scale),
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_score_packed_key_padded_tile_avx512(
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    packed_source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mask_storeu_ps, _mm512_mul_ps, _mm512_set1_ps,
        _mm512_setzero_ps,
    };

    unsafe {
        let scale = _mm512_set1_ps(scale);
        for source in (0..packed_source_rows).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); MAX_PACKED_ATTENTION_QUERY_ROWS];
            for dimension in 0..head_dim {
                let key = _mm512_loadu_ps(packed_key.as_ptr().add(dimension * packed_source_rows + source));
                for row in 0..query_rows {
                    accumulators[row] = _mm512_fmadd_ps(
                        key,
                        _mm512_set1_ps(query[row * head_dim + dimension]),
                        accumulators[row],
                    );
                }
            }
            let lanes = (source_rows - source).min(16);
            let mask = if lanes == 16 {
                u16::MAX
            } else {
                (1_u16 << lanes) - 1
            };
            for row in 0..query_rows {
                _mm512_mask_storeu_ps(
                    scores.as_mut_ptr().add(row * source_rows + source),
                    mask,
                    _mm512_mul_ps(accumulators[row], scale),
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_score_packed_key_bf16_tile_avx512(
    query: &[f32],
    query_rows: usize,
    packed_key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm512_castsi512_ps, _mm512_cvtepu16_epi32, _mm512_fmadd_ps,
        _mm512_mul_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_slli_epi32, _mm512_storeu_ps,
    };
    unsafe {
        let scale = _mm512_set1_ps(scale);
        for source in (0..source_rows).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); 8];
            for dimension in 0..head_dim {
                let packed = _mm256_loadu_si256(
                    packed_key
                        .as_ptr()
                        .add(dimension * source_rows + source)
                        .cast::<__m256i>(),
                );
                let key = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(packed)));
                for row in 0..query_rows {
                    accumulators[row] = _mm512_fmadd_ps(
                        key,
                        _mm512_set1_ps(query[row * head_dim + dimension]),
                        accumulators[row],
                    );
                }
            }
            for row in 0..query_rows {
                _mm512_storeu_ps(
                    scores.as_mut_ptr().add(row * source_rows + source),
                    _mm512_mul_ps(accumulators[row], scale),
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn attention_value_tile_fma_avx512(
    weights: &[f32],
    query_rows: usize,
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };
    unsafe {
        for dimension in (0..head_dim).step_by(16) {
            for row_start in (0..query_rows).step_by(8) {
                let rows = (query_rows - row_start).min(8);
                let mut accumulators = [_mm512_setzero_ps(); 8];
                for row in 0..rows {
                    accumulators[row] =
                        _mm512_loadu_ps(output.as_ptr().add((row_start + row) * head_dim + dimension));
                }
                for source in 0..source_rows {
                    let value = _mm512_loadu_ps(value.as_ptr().add(source * head_dim + dimension));
                    for row in 0..rows {
                        accumulators[row] = _mm512_fmadd_ps(
                            _mm512_set1_ps(weights[(row_start + row) * source_rows + source]),
                            value,
                            accumulators[row],
                        );
                    }
                }
                for row in 0..rows {
                    _mm512_storeu_ps(
                        output.as_mut_ptr().add((row_start + row) * head_dim + dimension),
                        accumulators[row],
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn attention_value_tile_bf16_fma_avx512(
    weights: &[f32],
    query_rows: usize,
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm512_castsi512_ps, _mm512_cvtepu16_epi32, _mm512_fmadd_ps,
        _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_slli_epi32, _mm512_storeu_ps,
    };
    unsafe {
        if query_rows == 16 {
            attention_value_tile_bf16_fma_avx512_rows::<16>(weights, value, source_rows, head_dim, output);
            return;
        }
        for dimension in (0..head_dim).step_by(16) {
            for row_start in (0..query_rows).step_by(8) {
                let rows = (query_rows - row_start).min(8);
                let mut accumulators = [_mm512_setzero_ps(); 8];
                for row in 0..rows {
                    accumulators[row] =
                        _mm512_loadu_ps(output.as_ptr().add((row_start + row) * head_dim + dimension));
                }
                for source in 0..source_rows {
                    let packed = _mm256_loadu_si256(
                        value
                            .as_ptr()
                            .add(source * head_dim + dimension)
                            .cast::<__m256i>(),
                    );
                    let value = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(packed)));
                    for row in 0..rows {
                        accumulators[row] = _mm512_fmadd_ps(
                            _mm512_set1_ps(weights[(row_start + row) * source_rows + source]),
                            value,
                            accumulators[row],
                        );
                    }
                }
                for row in 0..rows {
                    _mm512_storeu_ps(
                        output.as_mut_ptr().add((row_start + row) * head_dim + dimension),
                        accumulators[row],
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn attention_value_tile_bf16_fma_avx512_rows<const QUERY_ROWS: usize>(
    weights: &[f32],
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm512_castsi512_ps, _mm512_cvtepu16_epi32, _mm512_fmadd_ps,
        _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_slli_epi32, _mm512_storeu_ps,
    };
    unsafe {
        for dimension in (0..head_dim).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); QUERY_ROWS];
            for row in 0..QUERY_ROWS {
                accumulators[row] = _mm512_loadu_ps(output.as_ptr().add(row * head_dim + dimension));
            }
            for source in 0..source_rows {
                let packed = _mm256_loadu_si256(
                    value
                        .as_ptr()
                        .add(source * head_dim + dimension)
                        .cast::<__m256i>(),
                );
                let value = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(packed)));
                for row in 0..QUERY_ROWS {
                    accumulators[row] = _mm512_fmadd_ps(
                        _mm512_set1_ps(weights[row * source_rows + source]),
                        value,
                        accumulators[row],
                    );
                }
            }
            for row in 0..QUERY_ROWS {
                _mm512_storeu_ps(
                    output.as_mut_ptr().add(row * head_dim + dimension),
                    accumulators[row],
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn attention_value_packed_bf16_dot_for_target(
    weights: &[u16],
    query_rows: usize,
    packed_value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    unsafe {
        attention_value_packed_bf16_dot_avx512(
            weights,
            query_rows,
            packed_value,
            source_rows,
            head_dim,
            output,
        );
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn attention_value_packed_bf16_dot_for_target(
    _weights: &[u16],
    _query_rows: usize,
    _packed_value: &[u16],
    _source_rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
unsafe fn attention_value_packed_bf16_dot_avx512(
    weights: &[u16],
    query_rows: usize,
    packed_value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        __m512bh, _mm512_dpbf16_ps, _mm512_loadu_epi16, _mm512_loadu_ps, _mm512_set1_epi32,
        _mm512_setzero_ps, _mm512_storeu_ps,
    };

    unsafe {
        let source_pairs = source_rows / 2;
        let dimension_blocks = head_dim / 16;
        for block_start in (0..dimension_blocks).step_by(2) {
            let blocks = (dimension_blocks - block_start).min(2);
            let mut accumulators = [[_mm512_setzero_ps(); 2]; 8];
            for row in 0..query_rows {
                for block in 0..blocks {
                    accumulators[row][block] =
                        _mm512_loadu_ps(output.as_ptr().add(row * head_dim + (block_start + block) * 16));
                }
            }
            for pair in 0..source_pairs {
                let mut values = [_mm512_setzero_ps(); 2];
                for block in 0..blocks {
                    let offset = ((block_start + block) * source_pairs + pair) * 32;
                    let packed: __m512bh = core::mem::transmute(_mm512_loadu_epi16(
                        packed_value.as_ptr().add(offset).cast::<i16>(),
                    ));
                    values[block] = core::mem::transmute(packed);
                }
                for row in 0..query_rows {
                    let offset = row * source_rows + pair * 2;
                    let pair_bits = weights.as_ptr().add(offset).cast::<u32>().read_unaligned();
                    let weight: __m512bh =
                        core::mem::transmute(_mm512_set1_epi32(i32::from_ne_bytes(pair_bits.to_ne_bytes())));
                    for block in 0..blocks {
                        let value: __m512bh = core::mem::transmute(values[block]);
                        accumulators[row][block] = _mm512_dpbf16_ps(accumulators[row][block], weight, value);
                    }
                }
            }
            for row in 0..query_rows {
                for block in 0..blocks {
                    _mm512_storeu_ps(
                        output
                            .as_mut_ptr()
                            .add(row * head_dim + (block_start + block) * 16),
                        accumulators[row][block],
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::unnecessary_wraps)]
fn attention_dot_for_target(backend: AttentionBackend, left: &[f32], right: &[f32]) -> Option<f32> {
    (backend == AttentionBackend::Neon).then(|| unsafe { attention_dot_neon(left, right) })
}

#[cfg(target_arch = "x86_64")]
fn attention_dot_for_target(backend: AttentionBackend, left: &[f32], right: &[f32]) -> Option<f32> {
    (backend == AttentionBackend::Avx512).then(|| unsafe { attention_dot_avx512(left, right) })
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn attention_dot_for_target(_backend: AttentionBackend, _left: &[f32], _right: &[f32]) -> Option<f32> {
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attention_dot_neon(left: &[f32], right: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
    let end = left.len() / 4 * 4;
    let mut accumulator = vdupq_n_f32(0.0);
    unsafe {
        for offset in (0..end).step_by(4) {
            accumulator = vfmaq_f32(
                accumulator,
                vld1q_f32(left.as_ptr().add(offset)),
                vld1q_f32(right.as_ptr().add(offset)),
            );
        }
    }
    let mut sum = vaddvq_f32(accumulator);
    for offset in end..left.len() {
        sum = left[offset].mul_add(right[offset], sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn attention_dot_avx512(left: &[f32], right: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};
    let end = left.len() / 16 * 16;
    let mut accumulator = _mm512_setzero_ps();
    unsafe {
        for offset in (0..end).step_by(16) {
            accumulator = _mm512_fmadd_ps(
                _mm512_loadu_ps(left.as_ptr().add(offset)),
                _mm512_loadu_ps(right.as_ptr().add(offset)),
                accumulator,
            );
        }
    }
    let mut sum = _mm512_reduce_add_ps(accumulator);
    for offset in end..left.len() {
        sum = left[offset].mul_add(right[offset], sum);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
fn attention_value_fma_for_target(
    backend: AttentionBackend,
    weight: f32,
    value: &[f32],
    output: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Neon {
        unsafe { attention_value_fma_neon(weight, value, output) }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
fn attention_value_fma_for_target(
    backend: AttentionBackend,
    weight: f32,
    value: &[f32],
    output: &mut [f32],
) -> bool {
    if backend == AttentionBackend::Avx512 {
        unsafe { attention_value_fma_avx512(weight, value, output) }
        true
    } else {
        false
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn attention_value_fma_for_target(
    _backend: AttentionBackend,
    _weight: f32,
    _value: &[f32],
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attention_value_fma_neon(weight: f32, value: &[f32], output: &mut [f32]) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    let end = value.len() / 4 * 4;
    unsafe {
        let weight = vdupq_n_f32(weight);
        for offset in (0..end).step_by(4) {
            let value = vld1q_f32(value.as_ptr().add(offset));
            let current = vld1q_f32(output.as_ptr().add(offset));
            vst1q_f32(output.as_mut_ptr().add(offset), vfmaq_f32(current, value, weight));
        }
    }
    for offset in end..value.len() {
        output[offset] = weight.mul_add(value[offset], output[offset]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn attention_value_fma_avx512(weight: f32, value: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_storeu_ps};
    let end = value.len() / 16 * 16;
    unsafe {
        let weight = _mm512_set1_ps(weight);
        for offset in (0..end).step_by(16) {
            let value = _mm512_loadu_ps(value.as_ptr().add(offset));
            let current = _mm512_loadu_ps(output.as_ptr().add(offset));
            _mm512_storeu_ps(
                output.as_mut_ptr().add(offset),
                _mm512_fmadd_ps(weight, value, current),
            );
        }
    }
    for offset in end..value.len() {
        output[offset] = weight.mul_add(value[offset], output[offset]);
    }
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn bounded_exp(value: f32) -> f32 {
    let value = value.clamp(-40.0, 20.0);
    let exponent = (value * core::f32::consts::LOG2_E).round() as i32;
    let remainder = (-exponent as f32).mul_add(core::f32::consts::LN_2, value);
    let polynomial = remainder.mul_add(
        remainder.mul_add(
            remainder.mul_add(
                remainder.mul_add(
                    remainder.mul_add(
                        remainder.mul_add(remainder.mul_add(1.0 / 5040.0, 1.0 / 720.0), 1.0 / 120.0),
                        1.0 / 24.0,
                    ),
                    1.0 / 6.0,
                ),
                0.5,
            ),
            1.0,
        ),
        1.0,
    );
    polynomial * f32::from_bits(u32::try_from(exponent + 127).expect("bounded exponent is positive") << 23)
}

#[inline]
fn fast_erf(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude >= 6.0 {
        return value.signum();
    }
    if magnitude < 1e-4 {
        return value * core::f32::consts::FRAC_2_SQRT_PI;
    }
    let t = (0.327_591_1_f32.mul_add(magnitude, 1.0)).recip();
    let polynomial = 1.061_405_4_f32
        .mul_add(t, -1.453_152_1)
        .mul_add(t, 1.421_413_8)
        .mul_add(t, -0.284_496_72)
        .mul_add(t, 0.254_829_6)
        * t;
    value.signum() * (1.0 - polynomial * bounded_exp(-magnitude * magnitude))
}

#[inline]
fn fast_tanh(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude >= 9.0 {
        return value.signum();
    }
    let result = if magnitude <= 0.5 {
        let square = magnitude * magnitude;
        magnitude
            * square.mul_add(
                square.mul_add(
                    square.mul_add(
                        square.mul_add(square.mul_add(-0.008_863_236, 0.021_869_488), -0.053_968_254),
                        0.133_333_34,
                    ),
                    -1.0 / 3.0,
                ),
                1.0,
            )
    } else {
        let exponential = bounded_exp(2.0 * magnitude);
        (exponential - 1.0) / (exponential + 1.0)
    };
    value.signum() * result
}

#[inline]
fn fast_gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + fast_erf(value / SQRT_2))
}

#[cfg(target_arch = "aarch64")]
fn gelu_for_target(input: &[f32], output: &mut [f32]) -> bool {
    unsafe { gelu_neon(input, output) }
    true
}

#[cfg(target_arch = "x86_64")]
fn gelu_for_target(input: &[f32], output: &mut [f32]) -> bool {
    if std::is_x86_feature_detected!("avx512f") {
        unsafe { gelu_avx512(input, output) }
        true
    } else {
        false
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn gelu_for_target(_input: &[f32], _output: &mut [f32]) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn layer_norm_for_target(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> bool {
    if width.is_multiple_of(16) && std::is_x86_feature_detected!("avx512f") {
        unsafe { layer_norm_avx512(input, rows, width, weight, bias, output) }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
fn layer_norm_statistics_for_target(
    input: &[f32],
    rows: usize,
    width: usize,
    mean: &mut [f32],
    scale: &mut [f32],
) -> bool {
    if width.is_multiple_of(16) && std::is_x86_feature_detected!("avx512f") {
        unsafe { layer_norm_statistics_avx512(input, rows, width, mean, scale) }
        true
    } else {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn layer_norm_for_target(
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
fn layer_norm_statistics_for_target(
    _input: &[f32],
    _rows: usize,
    _width: usize,
    _mean: &mut [f32],
    _scale: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn qass_modulate_for_target(
    queries: &[f32],
    base_scale: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    unsafe { qass_modulate_neon(queries, base_scale, heads, rows, head_dim, output) }
    true
}

#[cfg(target_arch = "x86_64")]
fn qass_modulate_for_target(
    queries: &[f32],
    base_scale: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if std::is_x86_feature_detected!("avx512f") {
        unsafe { qass_modulate_avx512(queries, base_scale, heads, rows, head_dim, output) }
        true
    } else {
        false
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn qass_modulate_for_target(
    _queries: &[f32],
    _base_scale: &[f32],
    _heads: usize,
    _rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

// The explicit vector loads and stores keep contiguous activation data on the SIMD path.
// The approximation is evaluated lane-wise to retain the scalar approximation's bit behavior.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_neon(input: &[f32], output: &mut [f32]) {
    use core::arch::aarch64::{vld1q_f32, vst1q_f32};
    let end = input.len() / 4 * 4;
    unsafe {
        for offset in (0..end).step_by(4) {
            let values: [f32; 4] = core::mem::transmute(vld1q_f32(input.as_ptr().add(offset)));
            let result = values.map(fast_gelu);
            vst1q_f32(output.as_mut_ptr().add(offset), core::mem::transmute(result));
        }
    }
    for offset in end..input.len() {
        output[offset] = fast_gelu(input[offset]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_avx512(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::{
        _CMP_GE_OQ, _CMP_LT_OQ, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _mm512_add_epi32,
        _mm512_add_ps, _mm512_castsi512_ps, _mm512_cmp_ps_mask, _mm512_cvt_roundps_epi32, _mm512_cvtepi32_ps,
        _mm512_div_ps, _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_mask_mov_ps,
        _mm512_mask_sub_ps, _mm512_max_ps, _mm512_mul_ps, _mm512_set1_epi32, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_slli_epi32, _mm512_storeu_ps, _mm512_sub_ps,
    };
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let end = input.len() / 16 * 16;
    unsafe {
        let zero = _mm512_setzero_ps();
        let one = _mm512_set1_ps(1.0);
        let half = _mm512_set1_ps(0.5);
        let inverse_sqrt_two = _mm512_set1_ps(core::f32::consts::FRAC_1_SQRT_2);
        let tiny_scale = _mm512_set1_ps(core::f32::consts::FRAC_2_SQRT_PI);
        let minimum_exp = _mm512_set1_ps(-87.0);
        let log2_e = _mm512_set1_ps(core::f32::consts::LOG2_E);
        let ln_2 = _mm512_set1_ps(core::f32::consts::LN_2);
        for offset in (0..end).step_by(16) {
            let values = _mm512_loadu_ps(input.as_ptr().add(offset));
            let scaled = _mm512_mul_ps(values, inverse_sqrt_two);
            let magnitude = _mm512_max_ps(scaled, _mm512_sub_ps(zero, scaled));
            let t = _mm512_div_ps(one, _mm512_fmadd_ps(_mm512_set1_ps(0.327_591_1), magnitude, one));
            let mut erf_polynomial = _mm512_set1_ps(1.061_405_4);
            for coefficient in [-1.453_152_1, 1.421_413_8, -0.284_496_72, 0.254_829_6] {
                erf_polynomial = _mm512_fmadd_ps(erf_polynomial, t, _mm512_set1_ps(coefficient));
            }
            erf_polynomial = _mm512_mul_ps(erf_polynomial, t);

            let exponential_input = _mm512_max_ps(
                _mm512_sub_ps(zero, _mm512_mul_ps(magnitude, magnitude)),
                minimum_exp,
            );
            let exponent = _mm512_cvt_roundps_epi32::<ROUND>(_mm512_mul_ps(exponential_input, log2_e));
            let remainder = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(exponent), ln_2, exponential_input);
            let mut exp_polynomial = _mm512_set1_ps(1.0 / 5_040.0);
            for coefficient in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
                exp_polynomial = _mm512_fmadd_ps(exp_polynomial, remainder, _mm512_set1_ps(coefficient));
            }
            let power = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
                exponent,
                _mm512_set1_epi32(127),
            )));
            let exponential = _mm512_mul_ps(exp_polynomial, power);
            let erf_magnitude = _mm512_fnmadd_ps(erf_polynomial, exponential, one);
            let negative = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(scaled, zero);
            let mut erf = _mm512_mask_sub_ps(erf_magnitude, negative, zero, erf_magnitude);
            let tiny = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(magnitude, _mm512_set1_ps(1e-4));
            erf = _mm512_mask_mov_ps(erf, tiny, _mm512_mul_ps(scaled, tiny_scale));
            let large = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(magnitude, _mm512_set1_ps(6.0));
            let signed_one = _mm512_mask_sub_ps(one, negative, zero, one);
            erf = _mm512_mask_mov_ps(erf, large, signed_one);
            let result = _mm512_mul_ps(_mm512_mul_ps(half, values), _mm512_add_ps(one, erf));
            _mm512_storeu_ps(output.as_mut_ptr().add(offset), result);
        }
    }
    for offset in end..input.len() {
        output[offset] = fast_gelu(input[offset]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn layer_norm_avx512(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_reduce_add_ps, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_storeu_ps, _mm512_sub_ps,
    };
    unsafe {
        for row in 0..rows {
            let offset = row * width;
            let input = input.as_ptr().add(offset);
            let output = output.as_mut_ptr().add(offset);
            let mut sum = _mm512_setzero_ps();
            for column in (0..width).step_by(16) {
                sum = _mm512_add_ps(sum, _mm512_loadu_ps(input.add(column)));
            }
            let mean = _mm512_reduce_add_ps(sum) / width as f32;
            let mean_vector = _mm512_set1_ps(mean);
            let mut squared_sum = _mm512_setzero_ps();
            for column in (0..width).step_by(16) {
                let centered = _mm512_sub_ps(_mm512_loadu_ps(input.add(column)), mean_vector);
                squared_sum = _mm512_fmadd_ps(centered, centered, squared_sum);
            }
            let scale = (_mm512_reduce_add_ps(squared_sum) / width as f32 + 1e-5_f32)
                .sqrt()
                .recip();
            let scale = _mm512_set1_ps(scale);
            for column in (0..width).step_by(16) {
                let normalized = _mm512_mul_ps(
                    _mm512_sub_ps(_mm512_loadu_ps(input.add(column)), mean_vector),
                    scale,
                );
                let result = _mm512_mul_ps(normalized, _mm512_loadu_ps(weight.as_ptr().add(column)));
                let result = if let Some(bias) = bias {
                    _mm512_add_ps(result, _mm512_loadu_ps(bias.as_ptr().add(column)))
                } else {
                    result
                };
                _mm512_storeu_ps(output.add(column), result);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn layer_norm_statistics_avx512(
    input: &[f32],
    rows: usize,
    width: usize,
    mean: &mut [f32],
    scale: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_sub_ps,
    };
    unsafe {
        for row in 0..rows {
            let input = input.as_ptr().add(row * width);
            let mut sum = _mm512_setzero_ps();
            for column in (0..width).step_by(16) {
                sum = _mm512_add_ps(sum, _mm512_loadu_ps(input.add(column)));
            }
            let row_mean = _mm512_reduce_add_ps(sum) / width as f32;
            let mean_vector = _mm512_set1_ps(row_mean);
            let mut squared_sum = _mm512_setzero_ps();
            for column in (0..width).step_by(16) {
                let centered = _mm512_sub_ps(_mm512_loadu_ps(input.add(column)), mean_vector);
                squared_sum = _mm512_fmadd_ps(centered, centered, squared_sum);
            }
            mean[row] = row_mean;
            scale[row] = (_mm512_reduce_add_ps(squared_sum) / width as f32 + 1e-5_f32)
                .sqrt()
                .recip();
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn qass_modulate_neon(
    queries: &[f32],
    base_scale: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vld1q_f32, vst1q_f32};
    unsafe {
        for head in 0..heads {
            let scale = base_scale.as_ptr().add(head * head_dim);
            for row in 0..rows {
                let offset = (head * rows + row) * head_dim;
                let end = head_dim / 4 * 4;
                for dimension in (0..end).step_by(4) {
                    let query: [f32; 4] =
                        core::mem::transmute(vld1q_f32(queries.as_ptr().add(offset + dimension)));
                    let scale: [f32; 4] = core::mem::transmute(vld1q_f32(scale.add(dimension)));
                    let mlp: [f32; 4] =
                        core::mem::transmute(vld1q_f32(output.as_ptr().add(offset + dimension)));
                    let result: [f32; 4] =
                        std::array::from_fn(|lane| query[lane] * scale[lane] * (1.0 + fast_tanh(mlp[lane])));
                    vst1q_f32(
                        output.as_mut_ptr().add(offset + dimension),
                        core::mem::transmute(result),
                    );
                }
                for dimension in end..head_dim {
                    output[offset + dimension] = queries[offset + dimension]
                        * *scale.add(dimension)
                        * (1.0 + fast_tanh(output[offset + dimension]));
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn qass_modulate_avx512(
    queries: &[f32],
    base_scale: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_loadu_ps, _mm512_storeu_ps};
    unsafe {
        for head in 0..heads {
            let scale = base_scale.as_ptr().add(head * head_dim);
            for row in 0..rows {
                let offset = (head * rows + row) * head_dim;
                let end = head_dim / 16 * 16;
                for dimension in (0..end).step_by(16) {
                    let query: [f32; 16] =
                        core::mem::transmute(_mm512_loadu_ps(queries.as_ptr().add(offset + dimension)));
                    let scale: [f32; 16] = core::mem::transmute(_mm512_loadu_ps(scale.add(dimension)));
                    let mlp: [f32; 16] =
                        core::mem::transmute(_mm512_loadu_ps(output.as_ptr().add(offset + dimension)));
                    let result: [f32; 16] =
                        std::array::from_fn(|lane| query[lane] * scale[lane] * (1.0 + fast_tanh(mlp[lane])));
                    _mm512_storeu_ps(
                        output.as_mut_ptr().add(offset + dimension),
                        core::mem::transmute(result),
                    );
                }
                for dimension in end..head_dim {
                    output[offset + dimension] = queries[offset + dimension]
                        * *scale.add(dimension)
                        * (1.0 + fast_tanh(output[offset + dimension]));
                }
            }
        }
    }
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

#[cfg(target_arch = "x86_64")]
fn affine_input3_transposed_for_target(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    if std::is_x86_feature_detected!("avx512f") {
        unsafe {
            affine_input3_transposed_avx512(input, rows, weight, bias, output_dim, output);
        }
        return true;
    }
    false
}

#[cfg(not(target_arch = "x86_64"))]
fn affine_input3_transposed_for_target(
    _input: &[f32],
    _rows: usize,
    _weight: &[f32],
    _bias: &[f32],
    _output_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn affine_for_target(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    // Advanced SIMD is mandatory for AArch64, including Apple Silicon.
    unsafe {
        affine_neon(input, rows, input_dim, weight, bias, output_dim, output);
    }
    true
}

#[cfg(target_arch = "x86_64")]
fn affine_for_target(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    if std::is_x86_feature_detected!("avx512f") {
        unsafe {
            affine_avx512(input, rows, input_dim, weight, bias, output_dim, output);
        }
        return true;
    }
    false
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn affine_for_target(
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn affine_neon(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let vector_end = input_dim / 4 * 4;
        let mut row = 0;
        while row + 4 < rows {
            affine_neon_rows::<5>(input, row, input_dim, weight, bias, output_dim, output);
            row += 5;
        }
        while row + 3 < rows {
            affine_neon_rows::<4>(input, row, input_dim, weight, bias, output_dim, output);
            row += 4;
        }
        while row + 1 < rows {
            let input0 = input.as_ptr().add(row * input_dim);
            let input1 = input0.add(input_dim);
            let mut channel = 0;
            while channel + 3 < output_dim {
                let weight0 = weight.as_ptr().add(channel * input_dim);
                let weight1 = weight0.add(input_dim);
                let weight2 = weight1.add(input_dim);
                let weight3 = weight2.add(input_dim);
                let mut acc00 = vdupq_n_f32(0.0);
                let mut acc01 = vdupq_n_f32(0.0);
                let mut acc02 = vdupq_n_f32(0.0);
                let mut acc03 = vdupq_n_f32(0.0);
                let mut acc10 = vdupq_n_f32(0.0);
                let mut acc11 = vdupq_n_f32(0.0);
                let mut acc12 = vdupq_n_f32(0.0);
                let mut acc13 = vdupq_n_f32(0.0);
                let mut column = 0;
                while column < vector_end {
                    let inputs0 = vld1q_f32(input0.add(column));
                    let inputs1 = vld1q_f32(input1.add(column));
                    let weights0 = vld1q_f32(weight0.add(column));
                    let weights1 = vld1q_f32(weight1.add(column));
                    let weights2 = vld1q_f32(weight2.add(column));
                    let weights3 = vld1q_f32(weight3.add(column));
                    acc00 = vfmaq_f32(acc00, weights0, inputs0);
                    acc01 = vfmaq_f32(acc01, weights1, inputs0);
                    acc02 = vfmaq_f32(acc02, weights2, inputs0);
                    acc03 = vfmaq_f32(acc03, weights3, inputs0);
                    acc10 = vfmaq_f32(acc10, weights0, inputs1);
                    acc11 = vfmaq_f32(acc11, weights1, inputs1);
                    acc12 = vfmaq_f32(acc12, weights2, inputs1);
                    acc13 = vfmaq_f32(acc13, weights3, inputs1);
                    column += 4;
                }
                let mut sums0 = [
                    vaddvq_f32(acc00),
                    vaddvq_f32(acc01),
                    vaddvq_f32(acc02),
                    vaddvq_f32(acc03),
                ];
                let mut sums1 = [
                    vaddvq_f32(acc10),
                    vaddvq_f32(acc11),
                    vaddvq_f32(acc12),
                    vaddvq_f32(acc13),
                ];
                while column < input_dim {
                    for lane in 0..4 {
                        let value = *weight0.add(lane * input_dim + column);
                        sums0[lane] = value.mul_add(*input0.add(column), sums0[lane]);
                        sums1[lane] = value.mul_add(*input1.add(column), sums1[lane]);
                    }
                    column += 1;
                }
                for lane in 0..4 {
                    let offset = channel + lane;
                    output[row * output_dim + offset] =
                        bias.map_or(sums0[lane], |values| sums0[lane] + values[offset]);
                    output[(row + 1) * output_dim + offset] =
                        bias.map_or(sums1[lane], |values| sums1[lane] + values[offset]);
                }
                channel += 4;
            }
            while channel < output_dim {
                let weight_row = weight.as_ptr().add(channel * input_dim);
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut column = 0;
                while column < vector_end {
                    let weights = vld1q_f32(weight_row.add(column));
                    acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input0.add(column)));
                    acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input1.add(column)));
                    column += 4;
                }
                let mut sum0 = vaddvq_f32(acc0);
                let mut sum1 = vaddvq_f32(acc1);
                while column < input_dim {
                    let value = *weight_row.add(column);
                    sum0 = value.mul_add(*input0.add(column), sum0);
                    sum1 = value.mul_add(*input1.add(column), sum1);
                    column += 1;
                }
                output[row * output_dim + channel] = bias.map_or(sum0, |values| sum0 + values[channel]);
                output[(row + 1) * output_dim + channel] = bias.map_or(sum1, |values| sum1 + values[channel]);
                channel += 1;
            }
            row += 2;
        }
        if row < rows {
            affine_neon_single_row(input, row, input_dim, weight, bias, output_dim, output);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn affine_neon_rows<const ROWS: usize>(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let inputs: [*const f32; ROWS] =
            std::array::from_fn(|index| input.as_ptr().add((row + index) * input_dim));
        let vector_end = input_dim / 4 * 4;
        let mut channel = 0;
        while channel + 3 < output_dim {
            let weights = [
                weight.as_ptr().add(channel * input_dim),
                weight.as_ptr().add((channel + 1) * input_dim),
                weight.as_ptr().add((channel + 2) * input_dim),
                weight.as_ptr().add((channel + 3) * input_dim),
            ];
            let mut accumulators = [[vdupq_n_f32(0.0); 4]; ROWS];
            let mut column = 0;
            while column < vector_end {
                let input_vectors: [core::arch::aarch64::float32x4_t; ROWS] =
                    std::array::from_fn(|index| vld1q_f32(inputs[index].add(column)));
                let weight_vectors = [
                    vld1q_f32(weights[0].add(column)),
                    vld1q_f32(weights[1].add(column)),
                    vld1q_f32(weights[2].add(column)),
                    vld1q_f32(weights[3].add(column)),
                ];
                for input_index in 0..ROWS {
                    for weight_index in 0..4 {
                        accumulators[input_index][weight_index] = vfmaq_f32(
                            accumulators[input_index][weight_index],
                            weight_vectors[weight_index],
                            input_vectors[input_index],
                        );
                    }
                }
                column += 4;
            }
            let mut sums = [[0.0; 4]; ROWS];
            for input_index in 0..ROWS {
                for weight_index in 0..4 {
                    sums[input_index][weight_index] = vaddvq_f32(accumulators[input_index][weight_index]);
                }
            }
            while column < input_dim {
                for input_index in 0..ROWS {
                    for weight_index in 0..4 {
                        sums[input_index][weight_index] = (*weights[weight_index].add(column))
                            .mul_add(*inputs[input_index].add(column), sums[input_index][weight_index]);
                    }
                }
                column += 1;
            }
            for input_index in 0..ROWS {
                for weight_index in 0..4 {
                    let offset = channel + weight_index;
                    output[(row + input_index) * output_dim + offset] = bias
                        .map_or(sums[input_index][weight_index], |values| {
                            sums[input_index][weight_index] + values[offset]
                        });
                }
            }
            channel += 4;
        }
        while channel < output_dim {
            let weight_row = weight.as_ptr().add(channel * input_dim);
            let mut accumulators = [vdupq_n_f32(0.0); ROWS];
            let mut column = 0;
            while column < vector_end {
                let weight_vector = vld1q_f32(weight_row.add(column));
                for input_index in 0..ROWS {
                    accumulators[input_index] = vfmaq_f32(
                        accumulators[input_index],
                        weight_vector,
                        vld1q_f32(inputs[input_index].add(column)),
                    );
                }
                column += 4;
            }
            let mut sums = [0.0; ROWS];
            for input_index in 0..ROWS {
                sums[input_index] = vaddvq_f32(accumulators[input_index]);
            }
            while column < input_dim {
                let value = *weight_row.add(column);
                for input_index in 0..ROWS {
                    sums[input_index] = value.mul_add(*inputs[input_index].add(column), sums[input_index]);
                }
                column += 1;
            }
            for input_index in 0..ROWS {
                output[(row + input_index) * output_dim + channel] =
                    bias.map_or(sums[input_index], |values| sums[input_index] + values[channel]);
            }
            channel += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn affine_neon_single_row(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let input_row = input.as_ptr().add(row * input_dim);
        let vector_end = input_dim / 4 * 4;
        let mut channel = 0;
        while channel + 3 < output_dim {
            let weight0 = weight.as_ptr().add(channel * input_dim);
            let weight1 = weight0.add(input_dim);
            let weight2 = weight1.add(input_dim);
            let weight3 = weight2.add(input_dim);
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut column = 0;
            while column < vector_end {
                let inputs = vld1q_f32(input_row.add(column));
                acc0 = vfmaq_f32(acc0, vld1q_f32(weight0.add(column)), inputs);
                acc1 = vfmaq_f32(acc1, vld1q_f32(weight1.add(column)), inputs);
                acc2 = vfmaq_f32(acc2, vld1q_f32(weight2.add(column)), inputs);
                acc3 = vfmaq_f32(acc3, vld1q_f32(weight3.add(column)), inputs);
                column += 4;
            }
            let mut sums = [
                vaddvq_f32(acc0),
                vaddvq_f32(acc1),
                vaddvq_f32(acc2),
                vaddvq_f32(acc3),
            ];
            while column < input_dim {
                for lane in 0..4 {
                    sums[lane] =
                        (*weight0.add(lane * input_dim + column)).mul_add(*input_row.add(column), sums[lane]);
                }
                column += 1;
            }
            for lane in 0..4 {
                let offset = channel + lane;
                output[row * output_dim + offset] =
                    bias.map_or(sums[lane], |values| sums[lane] + values[offset]);
            }
            channel += 4;
        }
        while channel < output_dim {
            let weight_row = weight.as_ptr().add(channel * input_dim);
            let mut accumulator = vdupq_n_f32(0.0);
            let mut column = 0;
            while column < vector_end {
                accumulator = vfmaq_f32(
                    accumulator,
                    vld1q_f32(weight_row.add(column)),
                    vld1q_f32(input_row.add(column)),
                );
                column += 4;
            }
            let mut sum = vaddvq_f32(accumulator);
            while column < input_dim {
                sum = (*weight_row.add(column)).mul_add(*input_row.add(column), sum);
                column += 1;
            }
            output[row * output_dim + channel] = bias.map_or(sum, |values| sum + values[channel]);
            channel += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn affine_input3_transposed_avx512(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_storeu_ps};

    unsafe {
        let vector_end = output_dim / 16 * 16;
        for row in 0..rows {
            let input = input.as_ptr().add(row * 3);
            let input0 = _mm512_set1_ps(*input);
            let input1 = _mm512_set1_ps(*input.add(1));
            let input2 = _mm512_set1_ps(*input.add(2));
            let output = output.as_mut_ptr().add(row * output_dim);
            for channel in (0..vector_end).step_by(16) {
                let mut values = _mm512_loadu_ps(bias.as_ptr().add(channel));
                values = _mm512_fmadd_ps(_mm512_loadu_ps(weight.as_ptr().add(channel)), input0, values);
                values = _mm512_fmadd_ps(
                    _mm512_loadu_ps(weight.as_ptr().add(output_dim + channel)),
                    input1,
                    values,
                );
                values = _mm512_fmadd_ps(
                    _mm512_loadu_ps(weight.as_ptr().add(2 * output_dim + channel)),
                    input2,
                    values,
                );
                _mm512_storeu_ps(output.add(channel), values);
            }
            for channel in vector_end..output_dim {
                let value = weight[channel].mul_add(*input, bias[channel]);
                let value = weight[output_dim + channel].mul_add(*input.add(1), value);
                output
                    .add(channel)
                    .write(weight[2 * output_dim + channel].mul_add(*input.add(2), value));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn affine_avx512(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};

    unsafe {
        let vector_end = input_dim / 16 * 16;
        let mut row = 0;
        while row + 5 < rows {
            affine_avx512_rows::<6>(input, row, input_dim, weight, bias, output_dim, output);
            row += 6;
        }
        while row + 4 < rows {
            affine_avx512_rows::<5>(input, row, input_dim, weight, bias, output_dim, output);
            row += 5;
        }
        while row + 3 < rows {
            affine_avx512_rows::<4>(input, row, input_dim, weight, bias, output_dim, output);
            row += 4;
        }
        while row + 1 < rows {
            let input0 = input.as_ptr().add(row * input_dim);
            let input1 = input0.add(input_dim);
            let mut channel = 0;
            while channel + 3 < output_dim {
                let weight0 = weight.as_ptr().add(channel * input_dim);
                let weight1 = weight0.add(input_dim);
                let weight2 = weight1.add(input_dim);
                let weight3 = weight2.add(input_dim);
                let mut acc00 = _mm512_setzero_ps();
                let mut acc01 = _mm512_setzero_ps();
                let mut acc02 = _mm512_setzero_ps();
                let mut acc03 = _mm512_setzero_ps();
                let mut acc10 = _mm512_setzero_ps();
                let mut acc11 = _mm512_setzero_ps();
                let mut acc12 = _mm512_setzero_ps();
                let mut acc13 = _mm512_setzero_ps();
                let mut column = 0;
                while column < vector_end {
                    let inputs0 = _mm512_loadu_ps(input0.add(column));
                    let inputs1 = _mm512_loadu_ps(input1.add(column));
                    let weights0 = _mm512_loadu_ps(weight0.add(column));
                    let weights1 = _mm512_loadu_ps(weight1.add(column));
                    let weights2 = _mm512_loadu_ps(weight2.add(column));
                    let weights3 = _mm512_loadu_ps(weight3.add(column));
                    acc00 = _mm512_fmadd_ps(weights0, inputs0, acc00);
                    acc01 = _mm512_fmadd_ps(weights1, inputs0, acc01);
                    acc02 = _mm512_fmadd_ps(weights2, inputs0, acc02);
                    acc03 = _mm512_fmadd_ps(weights3, inputs0, acc03);
                    acc10 = _mm512_fmadd_ps(weights0, inputs1, acc10);
                    acc11 = _mm512_fmadd_ps(weights1, inputs1, acc11);
                    acc12 = _mm512_fmadd_ps(weights2, inputs1, acc12);
                    acc13 = _mm512_fmadd_ps(weights3, inputs1, acc13);
                    column += 16;
                }
                let mut sums0 = [
                    _mm512_reduce_add_ps(acc00),
                    _mm512_reduce_add_ps(acc01),
                    _mm512_reduce_add_ps(acc02),
                    _mm512_reduce_add_ps(acc03),
                ];
                let mut sums1 = [
                    _mm512_reduce_add_ps(acc10),
                    _mm512_reduce_add_ps(acc11),
                    _mm512_reduce_add_ps(acc12),
                    _mm512_reduce_add_ps(acc13),
                ];
                while column < input_dim {
                    for lane in 0..4 {
                        let value = *weight0.add(lane * input_dim + column);
                        sums0[lane] = value.mul_add(*input0.add(column), sums0[lane]);
                        sums1[lane] = value.mul_add(*input1.add(column), sums1[lane]);
                    }
                    column += 1;
                }
                for lane in 0..4 {
                    let offset = channel + lane;
                    output[row * output_dim + offset] =
                        bias.map_or(sums0[lane], |values| sums0[lane] + values[offset]);
                    output[(row + 1) * output_dim + offset] =
                        bias.map_or(sums1[lane], |values| sums1[lane] + values[offset]);
                }
                channel += 4;
            }
            while channel < output_dim {
                let weight_row = weight.as_ptr().add(channel * input_dim);
                let mut acc0 = _mm512_setzero_ps();
                let mut acc1 = _mm512_setzero_ps();
                let mut column = 0;
                while column < vector_end {
                    let weights = _mm512_loadu_ps(weight_row.add(column));
                    acc0 = _mm512_fmadd_ps(weights, _mm512_loadu_ps(input0.add(column)), acc0);
                    acc1 = _mm512_fmadd_ps(weights, _mm512_loadu_ps(input1.add(column)), acc1);
                    column += 16;
                }
                let mut sum0 = _mm512_reduce_add_ps(acc0);
                let mut sum1 = _mm512_reduce_add_ps(acc1);
                while column < input_dim {
                    let value = *weight_row.add(column);
                    sum0 = value.mul_add(*input0.add(column), sum0);
                    sum1 = value.mul_add(*input1.add(column), sum1);
                    column += 1;
                }
                output[row * output_dim + channel] = bias.map_or(sum0, |values| sum0 + values[channel]);
                output[(row + 1) * output_dim + channel] = bias.map_or(sum1, |values| sum1 + values[channel]);
                channel += 1;
            }
            row += 2;
        }
        if row < rows {
            affine_avx512_single_row(input, row, input_dim, weight, bias, output_dim, output);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn affine_avx512_rows<const ROWS: usize>(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};

    unsafe {
        let inputs: [*const f32; ROWS] =
            std::array::from_fn(|index| input.as_ptr().add((row + index) * input_dim));
        let vector_end = input_dim / 16 * 16;
        let mut channel = 0;
        while channel + 3 < output_dim {
            let weights = [
                weight.as_ptr().add(channel * input_dim),
                weight.as_ptr().add((channel + 1) * input_dim),
                weight.as_ptr().add((channel + 2) * input_dim),
                weight.as_ptr().add((channel + 3) * input_dim),
            ];
            let mut accumulators = [[_mm512_setzero_ps(); 4]; ROWS];
            let mut column = 0;
            while column < vector_end {
                let input_vectors: [core::arch::x86_64::__m512; ROWS] =
                    std::array::from_fn(|index| _mm512_loadu_ps(inputs[index].add(column)));
                let weight_vectors = [
                    _mm512_loadu_ps(weights[0].add(column)),
                    _mm512_loadu_ps(weights[1].add(column)),
                    _mm512_loadu_ps(weights[2].add(column)),
                    _mm512_loadu_ps(weights[3].add(column)),
                ];
                for input_index in 0..ROWS {
                    for weight_index in 0..4 {
                        accumulators[input_index][weight_index] = _mm512_fmadd_ps(
                            weight_vectors[weight_index],
                            input_vectors[input_index],
                            accumulators[input_index][weight_index],
                        );
                    }
                }
                column += 16;
            }
            let mut sums = [[0.0; 4]; ROWS];
            for input_index in 0..ROWS {
                for weight_index in 0..4 {
                    sums[input_index][weight_index] =
                        _mm512_reduce_add_ps(accumulators[input_index][weight_index]);
                }
            }
            while column < input_dim {
                for input_index in 0..ROWS {
                    for weight_index in 0..4 {
                        sums[input_index][weight_index] = (*weights[weight_index].add(column))
                            .mul_add(*inputs[input_index].add(column), sums[input_index][weight_index]);
                    }
                }
                column += 1;
            }
            for input_index in 0..ROWS {
                for weight_index in 0..4 {
                    let offset = channel + weight_index;
                    output[(row + input_index) * output_dim + offset] = bias
                        .map_or(sums[input_index][weight_index], |values| {
                            sums[input_index][weight_index] + values[offset]
                        });
                }
            }
            channel += 4;
        }
        while channel < output_dim {
            let weight_row = weight.as_ptr().add(channel * input_dim);
            let mut accumulators = [_mm512_setzero_ps(); ROWS];
            let mut column = 0;
            while column < vector_end {
                let weight_vector = _mm512_loadu_ps(weight_row.add(column));
                for input_index in 0..ROWS {
                    accumulators[input_index] = _mm512_fmadd_ps(
                        weight_vector,
                        _mm512_loadu_ps(inputs[input_index].add(column)),
                        accumulators[input_index],
                    );
                }
                column += 16;
            }
            let mut sums = [0.0; ROWS];
            for input_index in 0..ROWS {
                sums[input_index] = _mm512_reduce_add_ps(accumulators[input_index]);
            }
            while column < input_dim {
                let value = *weight_row.add(column);
                for input_index in 0..ROWS {
                    sums[input_index] = value.mul_add(*inputs[input_index].add(column), sums[input_index]);
                }
                column += 1;
            }
            for input_index in 0..ROWS {
                output[(row + input_index) * output_dim + channel] =
                    bias.map_or(sums[input_index], |values| sums[input_index] + values[channel]);
            }
            channel += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn affine_avx512_single_row(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};

    unsafe {
        let input_row = input.as_ptr().add(row * input_dim);
        let vector_end = input_dim / 16 * 16;
        let mut channel = 0;
        while channel + 3 < output_dim {
            let weight0 = weight.as_ptr().add(channel * input_dim);
            let weight1 = weight0.add(input_dim);
            let weight2 = weight1.add(input_dim);
            let weight3 = weight2.add(input_dim);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut column = 0;
            while column < vector_end {
                let inputs = _mm512_loadu_ps(input_row.add(column));
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(weight0.add(column)), inputs, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(weight1.add(column)), inputs, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(weight2.add(column)), inputs, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(weight3.add(column)), inputs, acc3);
                column += 16;
            }
            let mut sums = [
                _mm512_reduce_add_ps(acc0),
                _mm512_reduce_add_ps(acc1),
                _mm512_reduce_add_ps(acc2),
                _mm512_reduce_add_ps(acc3),
            ];
            while column < input_dim {
                for lane in 0..4 {
                    sums[lane] =
                        (*weight0.add(lane * input_dim + column)).mul_add(*input_row.add(column), sums[lane]);
                }
                column += 1;
            }
            for lane in 0..4 {
                let offset = channel + lane;
                output[row * output_dim + offset] =
                    bias.map_or(sums[lane], |values| sums[lane] + values[offset]);
            }
            channel += 4;
        }
        while channel < output_dim {
            let weight_row = weight.as_ptr().add(channel * input_dim);
            let mut accumulator = _mm512_setzero_ps();
            let mut column = 0;
            while column < vector_end {
                accumulator = _mm512_fmadd_ps(
                    _mm512_loadu_ps(weight_row.add(column)),
                    _mm512_loadu_ps(input_row.add(column)),
                    accumulator,
                );
                column += 16;
            }
            let mut sum = _mm512_reduce_add_ps(accumulator);
            while column < input_dim {
                sum = (*weight_row.add(column)).mul_add(*input_row.add(column), sum);
                column += 1;
            }
            output[row * output_dim + channel] = bias.map_or(sum, |values| sum + values[channel]);
            channel += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use super::{Bf16AffineCompensation, attention_score_maximum_f32};
    use super::{
        PackedBf16Affine, PackedF32Affine, affine_f32, attention_backend, f32_to_bf16_rne, fast_erf,
        fast_exp_negative, fast_gelu, fast_tanh, gelu_f32, layer_norm_f32, pack_attention_value_bf16,
        qass_modulate_f32, softmax_exp_sum_f32,
    };

    fn bf16_to_f32(value: u16) -> f32 {
        f32::from_bits(u32::from(value) << 16)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(clippy::float_cmp)]
    fn native_attention_maximum_handles_tails_and_nan() {
        let backend = attention_backend();
        let mut scores = (0..35).map(|index| index as f32 * 0.25 - 3.0).collect::<Vec<_>>();
        let Some((maximum, has_nan)) = attention_score_maximum_f32(backend, &scores) else {
            return;
        };
        assert_eq!(maximum, 5.5);
        assert!(!has_nan);
        scores[17] = f32::NAN;
        let (_, has_nan) = attention_score_maximum_f32(backend, &scores).expect("native maximum");
        assert!(has_nan);
    }

    #[test]
    fn packed_bf16_value_layout_groups_source_pairs_by_dimension_lane() {
        let source_rows = 2;
        let head_dim = 16;
        let value = (0..source_rows * head_dim)
            .map(|index| u16::try_from(index).expect("fixture fits u16"))
            .collect::<Vec<_>>();
        let mut packed = vec![0_u16; value.len()];
        assert!(pack_attention_value_bf16(
            &value,
            source_rows,
            head_dim,
            &mut packed
        ));
        for dimension in 0..head_dim {
            assert_eq!(packed[dimension * 2], value[dimension]);
            assert_eq!(packed[dimension * 2 + 1], value[head_dim + dimension]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn packed_bf16_value_dot_matches_quantized_weight_reference() {
        use super::{AttentionBackend, attention_value_packed_bf16_dot_f32};

        if !std::is_x86_feature_detected!("avx512bf16") {
            return;
        }
        let (rows, source_rows, head_dim) = (8, 128, 64);
        let weights = (0..rows * source_rows)
            .map(|index| ((index % 31) as f32 + 1.0) / 32.0)
            .collect::<Vec<_>>();
        let value = (0..source_rows * head_dim)
            .map(|index| f32_to_bf16_rne(((index % 37) as f32 - 18.0) * 0.0113))
            .collect::<Vec<_>>();
        let mut packed = vec![0_u16; value.len()];
        assert!(pack_attention_value_bf16(
            &value,
            source_rows,
            head_dim,
            &mut packed
        ));
        let mut output = vec![0.25; rows * head_dim];
        let mut weights_bf16 = vec![0_u16; weights.len()];
        assert!(attention_value_packed_bf16_dot_f32(
            AttentionBackend::Avx512,
            &weights,
            rows,
            &packed,
            source_rows,
            head_dim,
            &mut output,
            &mut weights_bf16,
        ));
        for row in 0..rows {
            for dimension in 0..head_dim {
                let mut expected = 0.25;
                for pair in 0..source_rows / 2 {
                    let first = pair * 2;
                    expected += bf16_to_f32(weights_bf16[row * source_rows + first]).mul_add(
                        bf16_to_f32(value[first * head_dim + dimension]),
                        bf16_to_f32(weights_bf16[row * source_rows + first + 1])
                            * bf16_to_f32(value[(first + 1) * head_dim + dimension]),
                    );
                }
                assert!((output[row * head_dim + dimension] - expected).abs() <= 2e-4);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "release-only Zen 5 kernel benchmark"]
    #[allow(clippy::cast_precision_loss)]
    fn benchmark_bf16_value_dot_8x128x64() {
        use std::{hint::black_box, time::Instant};

        use super::{
            AttentionBackend, attention_value_packed_bf16_dot_f32, attention_value_tile_bf16_fma_f32,
        };

        if !std::is_x86_feature_detected!("avx512bf16") {
            eprintln!("skipped: AVX512_BF16 is unavailable");
            return;
        }
        fn measure(mut operation: impl FnMut(), iterations: usize) -> u64 {
            for _ in 0..64 {
                operation();
            }
            let mut samples = Vec::with_capacity(21);
            for _ in 0..21 {
                let started = Instant::now();
                for _ in 0..iterations {
                    operation();
                }
                samples.push(
                    u64::try_from(started.elapsed().as_nanos() / iterations as u128)
                        .expect("nanoseconds fit u64"),
                );
            }
            samples.sort_unstable();
            samples[samples.len() / 2]
        }

        let (rows, source_rows, head_dim) = (8, 128, 64);
        let weights = (0..rows * source_rows)
            .map(|index| ((index % 31) as f32 + 1.0) / 32.0)
            .collect::<Vec<_>>();
        let value = (0..source_rows * head_dim)
            .map(|index| f32_to_bf16_rne(((index % 37) as f32 - 18.0) * 0.0113))
            .collect::<Vec<_>>();
        let mut packed = vec![0_u16; value.len()];
        assert!(pack_attention_value_bf16(
            &value,
            source_rows,
            head_dim,
            &mut packed
        ));
        let mut baseline_output = vec![0.0; rows * head_dim];
        let mut dot_output = baseline_output.clone();
        let mut weights_bf16 = vec![0_u16; weights.len()];
        let iterations = 2_000;
        let baseline = measure(
            || {
                baseline_output.fill(0.0);
                assert!(attention_value_tile_bf16_fma_f32(
                    AttentionBackend::Avx512,
                    black_box(&weights),
                    rows,
                    black_box(&value),
                    source_rows,
                    head_dim,
                    black_box(&mut baseline_output),
                ));
            },
            iterations,
        );
        let dot = measure(
            || {
                dot_output.fill(0.0);
                assert!(attention_value_packed_bf16_dot_f32(
                    AttentionBackend::Avx512,
                    black_box(&weights),
                    rows,
                    black_box(&packed),
                    source_rows,
                    head_dim,
                    black_box(&mut dot_output),
                    black_box(&mut weights_bf16),
                ));
            },
            iterations,
        );
        black_box((&baseline_output, &dot_output));
        println!(
            "bf16_value_8x128x64 baseline_p50_ns={baseline} dot_p50_ns={dot} speedup={:.3}",
            baseline as f64 / dot as f64,
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn packed_f32_affine_layout_groups_output_lanes_by_input() {
        let input_dim = 3;
        let output_dim = 16;
        let weight = (0..output_dim * input_dim)
            .map(|index| index as f32 + 0.25)
            .collect::<Vec<_>>();
        let packed = PackedF32Affine::new(&weight, input_dim, output_dim).expect("supported shape");
        assert_eq!(packed.input_dim, input_dim);
        assert_eq!(packed.output_dim, output_dim);
        assert_eq!(packed.values.len(), weight.len());
        for input in 0..input_dim {
            for channel in 0..output_dim {
                assert_eq!(
                    packed.values[input * output_dim + channel],
                    weight[channel * input_dim + input]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    fn packed_f32_affine_matches_scalar_order() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let (rows, input_dim, output_dim) = (7, 19, 32);
        let input = (0..rows * input_dim)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.0137)
            .collect::<Vec<_>>();
        let weight = (0..output_dim * input_dim)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.0073)
            .collect::<Vec<_>>();
        let bias = (0..output_dim)
            .map(|index| (index as f32 - 11.0) * 0.0031)
            .collect::<Vec<_>>();
        let packed = PackedF32Affine::new(&weight, input_dim, output_dim).expect("supported shape");
        let mut actual = vec![0.0; rows * output_dim];
        assert!(packed.run(&input, rows, Some(&bias), &mut actual));
        for row in 0..rows {
            for channel in 0..output_dim {
                let mut expected = bias[channel];
                for column in 0..input_dim {
                    expected = weight[channel * input_dim + column]
                        .mul_add(input[row * input_dim + column], expected);
                }
                assert_eq!(actual[row * output_dim + channel], expected);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "release-only Zen 5 kernel benchmark"]
    #[allow(clippy::cast_precision_loss)]
    fn benchmark_packed_f32_affine_transformer_shapes() {
        use std::hint::black_box;
        use std::time::Instant;

        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipped: AVX512F is unavailable");
            return;
        }

        fn measure(mut operation: impl FnMut(), iterations: usize) -> u64 {
            for _ in 0..8 {
                operation();
            }
            let mut samples = Vec::with_capacity(11);
            for _ in 0..11 {
                let started = Instant::now();
                for _ in 0..iterations {
                    operation();
                }
                let per_iteration = started.elapsed().as_nanos() / iterations as u128;
                samples.push(u64::try_from(per_iteration).expect("nanoseconds fit u64"));
            }
            samples.sort_unstable();
            samples[samples.len() / 2]
        }

        for (rows, input_dim, output_dim, iterations) in [
            (128, 128, 256, 80),
            (128, 512, 512, 20),
            (128, 512, 1024, 10),
            (128, 1024, 512, 10),
        ] {
            let input = (0..rows * input_dim)
                .map(|index| ((index % 37) as f32 - 18.0) * 0.0113)
                .collect::<Vec<_>>();
            let weight = (0..output_dim * input_dim)
                .map(|index| ((index % 41) as f32 - 20.0) * 0.0067)
                .collect::<Vec<_>>();
            let bias = (0..output_dim)
                .map(|index| (index as f32 - 7.0) * 0.0029)
                .collect::<Vec<_>>();
            let packed = PackedF32Affine::new(&weight, input_dim, output_dim).expect("supported shape");
            let mut baseline_output = vec![0.0; rows * output_dim];
            let mut packed_output = baseline_output.clone();
            let baseline = measure(
                || {
                    assert!(affine_f32(
                        black_box(&input),
                        rows,
                        input_dim,
                        black_box(&weight),
                        Some(black_box(&bias)),
                        output_dim,
                        black_box(&mut baseline_output),
                    ));
                },
                iterations,
            );
            let packed_time = measure(
                || {
                    assert!(packed.run(
                        black_box(&input),
                        rows,
                        Some(black_box(&bias)),
                        black_box(&mut packed_output),
                    ));
                },
                iterations,
            );
            black_box((&baseline_output, &packed_output));
            println!(
                "f32_dense rows={rows} input_dim={input_dim} output_dim={output_dim} baseline_p50_ns={baseline} packed_p50_ns={packed_time} speedup={:.3}",
                baseline as f64 / packed_time as f64,
            );
        }
    }

    #[test]
    fn packed_bf16_affine_layout_groups_input_pairs_by_output_lane() {
        let input_dim = 2;
        let output_dim = 16;
        let weight = (0..output_dim * input_dim)
            .map(|index| index as f32 + 0.25)
            .collect::<Vec<_>>();
        let packed = PackedBf16Affine::new(&weight, input_dim, output_dim).expect("supported shape");
        assert_eq!(packed.input_dim, input_dim);
        assert_eq!(packed.output_dim, output_dim);
        assert_eq!(packed.values.len(), weight.len());
        assert_eq!(packed.residual_values.len(), weight.len());
        for channel in 0..output_dim {
            assert_eq!(packed.values[channel * 2], f32_to_bf16_rne(weight[channel * 2]));
            assert_eq!(
                packed.values[channel * 2 + 1],
                f32_to_bf16_rne(weight[channel * 2 + 1])
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn first_order_bf16_affine_tracks_fp32() {
        if !std::is_x86_feature_detected!("avx512bf16") {
            return;
        }
        let (rows, input_dim, output_dim) = (8, 512, 16);
        let input = (0..rows * input_dim)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.0113)
            .collect::<Vec<_>>();
        let weight = (0..output_dim * input_dim)
            .map(|index| ((index % 41) as f32 - 20.0) * 0.0067)
            .collect::<Vec<_>>();
        let bias = (0..output_dim)
            .map(|index| (index as f32 - 7.0) * 0.0029)
            .collect::<Vec<_>>();
        let packed = PackedBf16Affine::new(&weight, input_dim, output_dim).expect("supported shape");
        let mut expected = vec![0.0; rows * output_dim];
        assert!(affine_f32(
            &input,
            rows,
            input_dim,
            &weight,
            Some(&bias),
            output_dim,
            &mut expected,
        ));
        let mut actual = vec![0.0; expected.len()];
        let mut input_bf16 = vec![0_u16; input.len()];
        let mut input_residual_bf16 = vec![0_u16; input.len()];
        assert!(packed.run_f32_compensated(
            &input,
            rows,
            Some(&bias),
            &mut actual,
            &mut input_bf16,
            &mut input_residual_bf16,
            Bf16AffineCompensation::FirstOrder,
        ));
        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 2e-5 + 2e-5 * expected.abs();
            assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn packed_bf16_affine_rejects_unsupported_shapes() {
        assert!(PackedBf16Affine::new(&[0.0; 30], 2, 15).is_none());
        assert!(PackedBf16Affine::new(&[0.0; 48], 3, 16).is_none());
        assert!(PackedBf16Affine::new(&[0.0; 31], 2, 16).is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn packed_bf16_affine_matches_expanded_bf16_reference() {
        if !std::is_x86_feature_detected!("avx512bf16") {
            return;
        }
        let (rows, input_dim, output_dim) = (11, 512, 32);
        let input = (0..rows * input_dim)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.0137)
            .collect::<Vec<_>>();
        let weight = (0..output_dim * input_dim)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.0073)
            .collect::<Vec<_>>();
        let bias = (0..output_dim)
            .map(|index| (index as f32 - 11.0) * 0.0031)
            .collect::<Vec<_>>();
        let packed = PackedBf16Affine::new(&weight, input_dim, output_dim).expect("supported shape");
        let mut input_bf16 = vec![0_u16; input.len()];
        let mut actual = vec![0.0; rows * output_dim];
        assert!(packed.run_f32(&input, rows, Some(&bias), &mut actual, &mut input_bf16));
        for (actual, input) in input_bf16.iter().zip(&input) {
            assert_eq!(*actual, f32_to_bf16_rne(*input));
        }

        for row in 0..rows {
            for channel in 0..output_dim {
                let mut expected = bias[channel];
                for pair in 0..input_dim / 2 {
                    let first = pair * 2;
                    let input0 = bf16_to_f32(input_bf16[row * input_dim + first]);
                    let input1 = bf16_to_f32(input_bf16[row * input_dim + first + 1]);
                    let weight0 = bf16_to_f32(f32_to_bf16_rne(weight[channel * input_dim + first]));
                    let weight1 = bf16_to_f32(f32_to_bf16_rne(weight[channel * input_dim + first + 1]));
                    expected += input0.mul_add(weight0, input1 * weight1);
                }
                let difference = (actual[row * output_dim + channel] - expected).abs();
                assert!(
                    difference <= 2e-4,
                    "row={row} channel={channel} difference={difference}"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "release-only Zen 5 kernel benchmark"]
    #[allow(clippy::cast_precision_loss)]
    fn benchmark_packed_bf16_affine_8x16x512() {
        use std::hint::black_box;
        use std::time::Instant;

        use super::convert_f32_to_bf16;

        if !std::is_x86_feature_detected!("avx512bf16") {
            eprintln!("skipped: AVX512_BF16 is unavailable");
            return;
        }

        fn measure(mut operation: impl FnMut(), iterations: usize) -> (u64, u64) {
            for _ in 0..128 {
                operation();
            }
            let mut samples = Vec::with_capacity(21);
            for _ in 0..21 {
                let started = Instant::now();
                for _ in 0..iterations {
                    operation();
                }
                let per_iteration = started.elapsed().as_nanos() / iterations as u128;
                samples.push(u64::try_from(per_iteration).expect("nanoseconds fit u64"));
            }
            samples.sort_unstable();
            (samples[samples.len() / 2], samples[samples.len() * 95 / 100])
        }

        let (rows, input_dim, output_dim) = (8, 512, 16);
        let input = (0..rows * input_dim)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.0113)
            .collect::<Vec<_>>();
        let weight = (0..output_dim * input_dim)
            .map(|index| ((index % 41) as f32 - 20.0) * 0.0067)
            .collect::<Vec<_>>();
        let bias = (0..output_dim)
            .map(|index| (index as f32 - 7.0) * 0.0029)
            .collect::<Vec<_>>();
        let packed = PackedBf16Affine::new(&weight, input_dim, output_dim).expect("supported shape");
        let mut input_bf16 = vec![0_u16; input.len()];
        assert!(convert_f32_to_bf16(&input, &mut input_bf16));
        let mut fp32_output = vec![0.0; rows * output_dim];
        let mut bf16_output = fp32_output.clone();
        let mut converted_output = fp32_output.clone();
        let mut input_compensated_output = fp32_output.clone();
        let mut weight_compensated_output = fp32_output.clone();
        let mut first_order_output = fp32_output.clone();
        let mut conversion_scratch = input_bf16.clone();
        let mut compensation_high = input_bf16.clone();
        let mut compensation_low = input_bf16.clone();
        let iterations = 2_000;

        let fp32 = measure(
            || {
                assert!(affine_f32(
                    black_box(&input),
                    rows,
                    input_dim,
                    black_box(&weight),
                    Some(black_box(&bias)),
                    output_dim,
                    black_box(&mut fp32_output),
                ));
            },
            iterations,
        );
        let bf16 = measure(
            || {
                assert!(packed.run_bf16(
                    black_box(&input_bf16),
                    rows,
                    Some(black_box(&bias)),
                    black_box(&mut bf16_output),
                ));
            },
            iterations,
        );
        let converted = measure(
            || {
                assert!(packed.run_f32(
                    black_box(&input),
                    rows,
                    Some(black_box(&bias)),
                    black_box(&mut converted_output),
                    black_box(&mut conversion_scratch),
                ));
            },
            iterations,
        );
        let input_compensated = measure(
            || {
                assert!(packed.run_f32_compensated(
                    black_box(&input),
                    rows,
                    Some(black_box(&bias)),
                    black_box(&mut input_compensated_output),
                    black_box(&mut compensation_high),
                    black_box(&mut compensation_low),
                    Bf16AffineCompensation::Input,
                ));
            },
            iterations,
        );
        let weight_compensated = measure(
            || {
                assert!(packed.run_f32_compensated(
                    black_box(&input),
                    rows,
                    Some(black_box(&bias)),
                    black_box(&mut weight_compensated_output),
                    black_box(&mut compensation_high),
                    black_box(&mut []),
                    Bf16AffineCompensation::Weight,
                ));
            },
            iterations,
        );
        let first_order = measure(
            || {
                assert!(packed.run_f32_compensated(
                    black_box(&input),
                    rows,
                    Some(black_box(&bias)),
                    black_box(&mut first_order_output),
                    black_box(&mut compensation_high),
                    black_box(&mut compensation_low),
                    Bf16AffineCompensation::FirstOrder,
                ));
            },
            iterations,
        );
        black_box((
            &fp32_output,
            &bf16_output,
            &converted_output,
            &input_compensated_output,
            &weight_compensated_output,
            &first_order_output,
        ));

        println!(
            "bf16_dense_8x16x512 fp32_p50_ns={} fp32_p95_ns={} bf16_p50_ns={} bf16_p95_ns={} bf16_with_conversion_p50_ns={} bf16_with_conversion_p95_ns={} input_compensated_p50_ns={} weight_compensated_p50_ns={} first_order_p50_ns={} kernel_speedup={:.3} end_to_end_speedup={:.3} first_order_speedup={:.3}",
            fp32.0,
            fp32.1,
            bf16.0,
            bf16.1,
            converted.0,
            converted.1,
            input_compensated.0,
            weight_compensated.0,
            first_order.0,
            fp32.0 as f64 / bf16.0 as f64,
            fp32.0 as f64 / converted.0 as f64,
            fp32.0 as f64 / first_order.0 as f64,
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn softmax_exponential_tracks_exact_exp() {
        let mut maximum_relative_error = 0.0_f32;
        for index in -80_000..=0 {
            let value = index as f32 / 1_000.0;
            let exact = value.exp();
            maximum_relative_error =
                maximum_relative_error.max((fast_exp_negative(value) - exact).abs() / exact);
        }
        assert!(
            maximum_relative_error <= 5e-6,
            "exp relative error: {maximum_relative_error}"
        );

        let mut scores = (0..35).map(|index| index as f32 * 0.17 - 7.0).collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let expected = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .collect::<Vec<_>>();
        let Some(sum) = softmax_exp_sum_f32(attention_backend(), &mut scores, maximum) else {
            return;
        };
        for (actual, expected) in scores.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 5e-6 * expected.max(1e-20));
        }
        let expected_sum = expected.iter().sum::<f32>();
        assert!((sum - expected_sum).abs() <= 1e-5 * expected_sum);
    }

    #[test]
    fn activation_approximations_track_libm_across_the_useful_domain() {
        let mut maximum_erf_error = 0.0_f32;
        let mut maximum_tanh_error = 0.0_f32;
        let mut maximum_gelu_error = 0.0_f32;
        for index in -10_000..=10_000 {
            let value = index as f32 / 1_000.0;
            maximum_erf_error = maximum_erf_error.max((fast_erf(value) - libm::erff(value)).abs());
            maximum_tanh_error = maximum_tanh_error.max((fast_tanh(value) - libm::tanhf(value)).abs());
            let exact_gelu = 0.5 * value * (1.0 + libm::erff(value / core::f32::consts::SQRT_2));
            maximum_gelu_error = maximum_gelu_error.max((fast_gelu(value) - exact_gelu).abs());
        }
        assert!(maximum_erf_error <= 6e-7, "erf error: {maximum_erf_error}");
        assert!(maximum_tanh_error <= 1e-6, "tanh error: {maximum_tanh_error}");
        assert!(maximum_gelu_error <= 1e-6, "GELU error: {maximum_gelu_error}");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn native_activations_match_approximations_with_tails() {
        let input = (0..35)
            .map(|index| (index as f32 - 17.0) * 0.37)
            .collect::<Vec<_>>();
        let mut actual = vec![0.0; input.len()];
        if !gelu_f32(&input, &mut actual) {
            return;
        }
        for (actual, input) in actual.iter().zip(input) {
            assert!((*actual - fast_gelu(input)).abs() <= 2e-6);
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn native_layer_norm_tracks_scalar_rows() {
        let (rows, width) = (3, 128);
        let input = (0..rows * width)
            .map(|index| (index as f32 * 0.03125).sin() + (index % 11) as f32 * 0.125)
            .collect::<Vec<_>>();
        let weight = (0..width)
            .map(|index| 0.75 + index as f32 * 0.001)
            .collect::<Vec<_>>();
        let bias = (0..width)
            .map(|index| (index as f32 - 64.0) * 0.0005)
            .collect::<Vec<_>>();
        let mut actual = vec![0.0; input.len()];
        if !layer_norm_f32(&input, rows, width, &weight, Some(&bias), &mut actual) {
            return;
        }
        for row in 0..rows {
            let values = &input[row * width..][..width];
            let mean = values.iter().sum::<f32>() / width as f32;
            let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / width as f32;
            let scale = (variance + 1e-5).sqrt().recip();
            for column in 0..width {
                let expected = (values[column] - mean) * scale * weight[column] + bias[column];
                assert!((actual[row * width + column] - expected).abs() <= 2e-5);
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    fn native_qass_modulation_matches_approximation_with_head_tails() {
        let (heads, rows, head_dim) = (3, 2, 19);
        let queries = (0..heads * rows * head_dim)
            .map(|index| (index as f32 - 41.0) * 0.03125)
            .collect::<Vec<_>>();
        let base_scale = (0..heads * head_dim)
            .map(|index| (index as f32 - 17.0) * 0.0625)
            .collect::<Vec<_>>();
        let mut actual = (0..queries.len())
            .map(|index| (index as f32 - 29.0) * 0.125)
            .collect::<Vec<_>>();
        let expected = actual.clone();
        if !qass_modulate_f32(&queries, &base_scale, heads, rows, head_dim, &mut actual) {
            return;
        }
        for head in 0..heads {
            for row in 0..rows {
                for dimension in 0..head_dim {
                    let offset = (head * rows + row) * head_dim + dimension;
                    assert_eq!(
                        actual[offset],
                        queries[offset]
                            * base_scale[head * head_dim + dimension]
                            * (1.0 + fast_tanh(expected[offset]))
                    );
                }
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn native_affine_matches_scalar_with_a_tail() {
        let rows = 5;
        let input_dim = 19;
        let output_dim = 7;
        let input = (0..rows * input_dim)
            .map(|index| (index as f32 - 13.0) * 0.03125)
            .collect::<Vec<_>>();
        let weight = (0..output_dim * input_dim)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.015_625)
            .collect::<Vec<_>>();
        let bias = (0..output_dim)
            .map(|index| (index as f32 - 2.0) * 0.125)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; rows * output_dim];
        for row in 0..rows {
            for channel in 0..output_dim {
                let mut sum = bias[channel];
                for column in 0..input_dim {
                    sum = weight[channel * input_dim + column].mul_add(input[row * input_dim + column], sum);
                }
                expected[row * output_dim + channel] = sum;
            }
        }

        let mut actual = vec![0.0; expected.len()];
        if !affine_f32(
            &input,
            rows,
            input_dim,
            &weight,
            Some(&bias),
            output_dim,
            &mut actual,
        ) {
            return;
        }
        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 1e-5 + 1e-5 * expected.abs();
            assert!((actual - expected).abs() <= tolerance);
        }
    }
}

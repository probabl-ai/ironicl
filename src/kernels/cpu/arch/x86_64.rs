//! x86-64 kernels: AVX2 and AVX-512.
//!
//! Safety contract. Every `*_avx512` kernel is an `unsafe fn` with `#[target_feature]`; its
//! `# Safety` section lists the CPU features and slice lengths it needs. The safe `*_for_target`
//! dispatchers check the CPU features, directly or through [`avx512`], and the lengths their
//! kernels need, falling back to the scalar path otherwise. Each SAFETY comment names the
//! check it relies on.

#![expect(
    unsafe_code,
    reason = "AVX-512 intrinsics and raw pointer loads; each unsafe block states its invariant"
)]
#![expect(
    clippy::needless_range_loop,
    reason = "loops index fixed-size accumulator and lane arrays in step with each other"
)]

use crate::kernels::cpu::*;

/// Whether `backend` selects the AVX-512 kernels on this CPU. `Isa` is a public type anyone can
/// build, so the feature is checked here (a cached load) rather than trusted.
fn avx512(backend: Isa) -> bool {
    backend == Isa::Avx512 && std::is_x86_feature_detected!("avx512f")
}

pub(crate) const F32_AFFINE_OUTPUT_GROUP_BLOCKS: usize = 4;

pub(crate) const F32_AFFINE_MAX_ROWS: usize = 4;

pub(crate) const BF16_AFFINE_MAX_ROWS: usize = 8;

pub(crate) const MAX_PACKED_ATTENTION_QUERY_ROWS: usize = 16;

#[inline]
#[expect(clippy::cast_possible_truncation, reason = "the clamped exponent fits in i32")]
#[expect(
    clippy::cast_precision_loss,
    reason = "the exponent is clamped to within ±128, exact in f32"
)]
pub(crate) fn fast_exp_negative(value: f32) -> f32 {
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn packed_bf16_affine_supported_for_target() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512bf16")
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn bf16_storage_supported_for_target() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512bf16")
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn convert_f32_to_bf16(input: &[f32], output: &mut [u16]) -> bool {
    if input.len() != output.len()
        || !std::is_x86_feature_detected!("avx512f")
        || !std::is_x86_feature_detected!("avx512bf16")
    {
        return false;
    }
    // SAFETY: avx512f and avx512bf16 were detected and `input.len() == output.len()` checked
    // just above.
    unsafe { convert_f32_to_bf16_avx512(input, output) };
    true
}

/// Converts FP32 to BF16 with round-to-nearest-even: 16 lanes per step, then a scalar tail.
///
/// # Safety
///
/// The CPU must support `avx512f` and `avx512bf16`, and `output.len() >= input.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
pub(crate) unsafe fn convert_f32_to_bf16_avx512(input: &[f32], output: &mut [u16]) {
    use core::arch::x86_64::{__m256i, _mm256_storeu_si256, _mm512_cvtneps_pbh, _mm512_loadu_ps};

    // SAFETY: each step reads `input[offset..offset + 16]` and writes 16 `u16` (32 bytes) to the
    // same range of `output`; `offset + 16 <= end <= input.len() <= output.len()`. The transmute
    // is `__m256bh` to `__m256i`, both 32 bytes. The tail uses checked indexing.
    unsafe {
        let end = input.len() / 16 * 16;
        for offset in (0..end).step_by(16) {
            let packed: __m256i =
                core::mem::transmute(_mm512_cvtneps_pbh(_mm512_loadu_ps(input.as_ptr().add(offset))));
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "This intrinsic accepts unaligned pointers."
            )]
            _mm256_storeu_si256(output.as_mut_ptr().add(offset).cast::<__m256i>(), packed);
        }
        for offset in end..input.len() {
            output[offset] = f32_to_bf16_rne(input[offset]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn packed_f32_affine_supported_for_target() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn affine_packed_f32_for_target(
    input: &[f32],
    rows: usize,
    packed: &PackedF32Affine,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> bool {
    if packed_f32_affine_supported_for_target() {
        // SAFETY: avx512f was detected by `packed_f32_affine_supported_for_target`. The lengths are
        // checked by `PackedF32Affine::run`, the only caller.
        unsafe { affine_packed_f32_avx512(input, rows, packed, bias, output) };
        true
    } else {
        false
    }
}

/// Runs `output = input * weight^T + bias` from weights packed by `PackedF32Affine::new`.
///
/// # Safety
///
/// The CPU must support `avx512f`. `packed.values.len() >= packed.input_dim * packed.output_dim`
/// (as `PackedF32Affine::new` builds it), `input.len() >= rows * packed.input_dim`,
/// `output.len() >= rows * packed.output_dim`, and `bias`, if any, holds `packed.output_dim`
/// values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn affine_packed_f32_avx512(
    input: &[f32],
    rows: usize,
    packed: &PackedF32Affine,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    // SAFETY: for block `b = block_start + block < output_dim / 16`, the bias load reads
    // `bias[b * 16..][..16]`, the weight load reads `values[(b * input_dim + column) * 16..][..16]`
    // (below `output_dim * input_dim`), and the store writes 16 values of output row
    // `row_start + row < rows` at column `b * 16`. The input read uses checked indexing.
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
pub(crate) fn affine_packed_bf16_for_target(
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
        // SAFETY: avx512f, avx512bw, and avx512bf16 were detected just above. The lengths are
        // checked by `PackedBf16Affine::run_bf16_components`, the only caller.
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

/// Runs a BF16 affine layer with FP32 accumulation through `VDPBF16PS`, from weights packed by
/// `PackedBf16Affine::new`, optionally adding input or weight residual terms.
///
/// # Safety
///
/// The CPU must support `avx512f`, `avx512bw`, and `avx512bf16`. `packed.values` (and
/// `packed.residual_values` when `use_weight_residual`) hold at least
/// `packed.input_dim / 2 * packed.output_dim * 2` values (as `PackedBf16Affine::new` builds
/// them); `input` and `input_residual`, if any, hold `rows * packed.input_dim`; `output` holds
/// `rows * packed.output_dim`; and `bias`, if any, holds `packed.output_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
pub(crate) unsafe fn affine_packed_bf16_avx512(
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

    // SAFETY: for `block < output_dim / 16` and `pair < input_dim / 2`, each weight load reads 32
    // `u16` at `(block * input_pairs + pair) * 32`, inside `values`/`residual_values`; each input
    // read takes 2 `u16` at `row * input_dim + pair * 2`, inside row `row_start + row < rows`, via
    // `read_unaligned`; each store writes 16 values inside that output row. The bias load reads
    // `bias[block * 16..][..16]`. The transmutes are between 64-byte vector types.
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
pub(crate) fn attention_score_tile_for_target(
    backend: Isa,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if avx512(backend) && matches!(head_dim, 16 | 64) {
        // SAFETY: `avx512(backend)` detected avx512f, and
        // `head_dim` is 16 or 64. Callers pass `query_rows * head_dim` query, `source_rows *
        // head_dim` key, and `query_rows * source_rows` scores (`reference.rs` slices the tiles).
        unsafe {
            attention_score_tile_avx512(query, query_rows, key, source_rows, head_dim, scale, scores);
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
pub(crate) fn attention_score_tile_bf16_for_target(
    backend: Isa,
    query: &[f32],
    query_rows: usize,
    key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if avx512(backend)
        && std::is_x86_feature_detected!("avx512bw")
        && query_rows <= MAX_PACKED_ATTENTION_QUERY_ROWS
        && matches!(head_dim, 16 | 64)
    {
        // SAFETY: as in `attention_score_tile_for_target`; avx512bw was detected here.
        unsafe {
            attention_score_tile_bf16_avx512(query, query_rows, key, source_rows, head_dim, scale, scores);
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
pub(crate) fn attention_score_packed_key_tile_for_target(
    backend: Isa,
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if avx512(backend)
        && query_rows <= MAX_PACKED_ATTENTION_QUERY_ROWS
        && source_rows.is_multiple_of(16)
        && matches!(head_dim, 16 | 64)
    {
        // SAFETY: avx512f via `avx512(backend)`; `source_rows % 16 == 0` and `head_dim` are
        // checked here. Callers pass `head_dim * source_rows` packed key and `query_rows *
        // source_rows` scores (`source_major_online_attention_with_backend_f32`).
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
pub(crate) fn attention_score_packed_key_padded_tile_for_target(
    backend: Isa,
    query: &[f32],
    query_rows: usize,
    packed_key: &[f32],
    source_rows: usize,
    packed_source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if avx512(backend)
        && query_rows <= MAX_PACKED_ATTENTION_QUERY_ROWS
        && source_rows > 0
        && source_rows <= packed_source_rows
        && packed_source_rows.is_multiple_of(16)
        && packed_source_rows == source_rows.div_ceil(16) * 16
        && matches!(head_dim, 16 | 64)
    {
        // SAFETY: avx512f via `avx512(backend)`; the padding relation between `source_rows`
        // and `packed_source_rows` is checked here. Callers pass `head_dim * packed_source_rows`
        // packed key and `query_rows * source_rows` scores.
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
pub(crate) fn attention_score_packed_key_bf16_tile_for_target(
    backend: Isa,
    query: &[f32],
    query_rows: usize,
    packed_key: &[u16],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    if avx512(backend)
        && std::is_x86_feature_detected!("avx512bw")
        && query_rows <= 8
        && source_rows.is_multiple_of(16)
        && matches!(head_dim, 16 | 64)
    {
        // SAFETY: avx512f via `avx512(backend)`, avx512bw detected here, and
        // `source_rows % 16 == 0`. Callers pass `head_dim * source_rows` packed key and
        // `query_rows * source_rows` scores (`source_major_online_attention_bf16_with_backend_f32`).
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_value_tile_fma_for_target(
    backend: Isa,
    weights: &[f32],
    query_rows: usize,
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if avx512(backend) && matches!(head_dim, 16 | 64) {
        // SAFETY: avx512f via `avx512(backend)` and `head_dim` is 16 or 64. Callers pass
        // `query_rows * source_rows` weights, `source_rows * head_dim` value, and
        // `query_rows * head_dim` output (`reference.rs` slices the tiles).
        unsafe {
            attention_value_tile_fma_avx512(weights, query_rows, value, source_rows, head_dim, output);
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_value_tile_bf16_fma_for_target(
    backend: Isa,
    weights: &[f32],
    query_rows: usize,
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if avx512(backend) && std::is_x86_feature_detected!("avx512bw") && matches!(head_dim, 16 | 64) {
        // SAFETY: as in `attention_value_tile_fma_for_target`; avx512bw was detected here.
        unsafe {
            attention_value_tile_bf16_fma_avx512(weights, query_rows, value, source_rows, head_dim, output);
        }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn softmax_exp_sum_for_target(backend: Isa, scores: &mut [f32], maximum: f32) -> Option<f32> {
    // SAFETY: `avx512(backend)` detected avx512f; the kernel
    // has no length preconditions.
    avx512(backend).then(|| unsafe { softmax_exp_sum_avx512(scores, maximum) })
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_score_maximum_for_target(backend: Isa, scores: &[f32]) -> Option<(f32, bool)> {
    // SAFETY: as in `softmax_exp_sum_for_target`.
    avx512(backend).then(|| unsafe { attention_score_maximum_avx512(scores) })
}

/// Returns the maximum of `scores` and whether any score is NaN.
///
/// # Safety
///
/// The CPU must support `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_score_maximum_avx512(scores: &[f32]) -> (f32, bool) {
    use core::arch::x86_64::{
        _CMP_UNORD_Q, _mm512_cmp_ps_mask, _mm512_loadu_ps, _mm512_max_ps, _mm512_reduce_max_ps,
        _mm512_set1_ps,
    };

    // SAFETY: each load reads `scores[offset..offset + 16]` with `offset + 16 <= end <= len`.
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

/// Replaces each score with `exp(score - maximum)` and returns their sum.
///
/// # Safety
///
/// The CPU must support `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn softmax_exp_sum_avx512(scores: &mut [f32], maximum: f32) -> f32 {
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
    // SAFETY: each load and store covers `scores[offset..offset + 16]` with
    // `offset + 16 <= end <= scores.len()`. The tail uses checked indexing.
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

/// Computes a row-major `[query_rows, source_rows]` score tile from row-major FP32 Q and K.
///
/// # Safety
///
/// The CPU must support `avx512f`; `head_dim` is a multiple of 16; `query.len() >=
/// query_rows * head_dim`; and `key.len() >= source_rows * head_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_score_tile_avx512(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};
    // SAFETY: the loads read `query[row * head_dim + dimension..][..16]` and
    // `key[source * head_dim + dimension..][..16]` with `dimension + 16 <= head_dim`,
    // `row < query_rows`, and `source < source_rows`. The score write uses checked indexing.
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

/// Computes a row-major score tile from FP32 Q and row-major BF16 K.
///
/// # Safety
///
/// The CPU must support `avx512f` and `avx512bw`; `head_dim` is a multiple of 16;
/// `query.len() >= query_rows * head_dim`; and `key.len() >= source_rows * head_dim`.
/// (`query_rows > 16` panics on the accumulator index.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn attention_score_tile_bf16_avx512(
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
    // SAFETY: the key load reads 16 `u16` at `source * head_dim + dimension` and the query load 16
    // values at `row * head_dim + dimension`, with `dimension + 16 <= head_dim`, `source <
    // source_rows`, and `row < query_rows`. The score write uses checked indexing.
    unsafe {
        for source in 0..source_rows {
            let key = key.as_ptr().add(source * head_dim);
            let mut accumulators = [_mm512_setzero_ps(); MAX_PACKED_ATTENTION_QUERY_ROWS];
            for dimension in (0..head_dim).step_by(16) {
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "This intrinsic accepts unaligned pointers."
                )]
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

/// Computes a row-major score tile from dimension-major packed FP32 K (`[head_dim,
/// source_rows]`), with SIMD lanes over source rows.
///
/// # Safety
///
/// The CPU must support `avx512f`; `source_rows % 16 == 0`; `packed_key.len() >= head_dim *
/// source_rows`; and `scores.len() >= query_rows * source_rows`. (`query_rows > 16` panics.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_score_packed_key_tile_avx512(
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
    // SAFETY: with `source + 16 <= source_rows` (a multiple of 16), the key load reads
    // `packed_key[dimension * source_rows + source..][..16]` with `dimension < head_dim`, and the
    // store writes `scores[row * source_rows + source..][..16]` with `row < query_rows`. The
    // 16-row call has the same contract. The query read uses checked indexing.
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

/// `attention_score_packed_key_tile_avx512` unrolled for `QUERY_ROWS` query rows.
///
/// # Safety
///
/// As for `attention_score_packed_key_tile_avx512` with `query_rows == QUERY_ROWS`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_score_packed_key_tile_avx512_rows<const QUERY_ROWS: usize>(
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
    // SAFETY: as in `attention_score_packed_key_tile_avx512`, with `row < QUERY_ROWS`.
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

/// Computes a row-major score tile from packed FP32 K padded to `packed_source_rows`, storing
/// only the first `source_rows` scores of each row.
///
/// # Safety
///
/// The CPU must support `avx512f`; `packed_source_rows == source_rows.div_ceil(16) * 16` and
/// `source_rows > 0`; `packed_key.len() >= head_dim * packed_source_rows`; and
/// `scores.len() >= query_rows * source_rows`. (`query_rows > 16` panics.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
pub(crate) unsafe fn attention_score_packed_key_padded_tile_avx512(
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

    // SAFETY: the key load reads `packed_key[dimension * packed_source_rows + source..][..16]`
    // with `source + 16 <= packed_source_rows`. `source` is a multiple of 16 below
    // `packed_source_rows`, so `source < source_rows`, `lanes >= 1`, and the masked store writes
    // only `scores[row * source_rows + source..][..lanes]`, inside row `row < query_rows`.
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

/// Computes a row-major score tile from dimension-major packed BF16 K.
///
/// # Safety
///
/// The CPU must support `avx512f` and `avx512bw`; `source_rows % 16 == 0`;
/// `packed_key.len() >= head_dim * source_rows`; and `scores.len() >= query_rows *
/// source_rows`. (`query_rows > 8` panics on the accumulator index.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn attention_score_packed_key_bf16_tile_avx512(
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
    // SAFETY: as in `attention_score_packed_key_tile_avx512`: the key load reads 16 `u16` at
    // `dimension * source_rows + source` with `source + 16 <= source_rows`, and the store writes
    // `scores[row * source_rows + source..][..16]` with `row < query_rows`.
    unsafe {
        let scale = _mm512_set1_ps(scale);
        for source in (0..source_rows).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); 8];
            for dimension in 0..head_dim {
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "This intrinsic accepts unaligned pointers."
                )]
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

/// Accumulates `output += weights * value` for a `[query_rows, source_rows]` weight tile and
/// row-major FP32 V.
///
/// # Safety
///
/// The CPU must support `avx512f`; `head_dim` is a multiple of 16; `value.len() >= source_rows *
/// head_dim`; and `output.len() >= query_rows * head_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_value_tile_fma_avx512(
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
    // SAFETY: with `dimension + 16 <= head_dim`, the output loads and stores cover
    // `output[(row_start + row) * head_dim + dimension..][..16]` with `row_start + row <
    // query_rows`, and the value load covers `value[source * head_dim + dimension..][..16]` with
    // `source < source_rows`. The weight read uses checked indexing.
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

/// Accumulates `output += weights * value` from row-major BF16 V.
///
/// # Safety
///
/// The CPU must support `avx512f` and `avx512bw`; otherwise as for
/// `attention_value_tile_fma_avx512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn attention_value_tile_bf16_fma_avx512(
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
    // SAFETY: as in `attention_value_tile_fma_avx512`; the value load reads 16 `u16` at
    // `source * head_dim + dimension`. The 16-row call has the same contract.
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
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "This intrinsic accepts unaligned pointers."
                    )]
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

/// `attention_value_tile_bf16_fma_avx512` unrolled for `QUERY_ROWS` query rows.
///
/// # Safety
///
/// As for `attention_value_tile_bf16_fma_avx512` with `query_rows == QUERY_ROWS`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn attention_value_tile_bf16_fma_avx512_rows<const QUERY_ROWS: usize>(
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
    // SAFETY: as in `attention_value_tile_bf16_fma_avx512`, with `row < QUERY_ROWS`.
    unsafe {
        for dimension in (0..head_dim).step_by(16) {
            let mut accumulators = [_mm512_setzero_ps(); QUERY_ROWS];
            for row in 0..QUERY_ROWS {
                accumulators[row] = _mm512_loadu_ps(output.as_ptr().add(row * head_dim + dimension));
            }
            for source in 0..source_rows {
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "This intrinsic accepts unaligned pointers."
                )]
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
pub(crate) fn attention_value_packed_bf16_dot_for_target(
    weights: &[u16],
    query_rows: usize,
    packed_value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    let fits = |count: Option<usize>, len: usize| count.is_some_and(|count| len >= count);
    if !packed_bf16_affine_supported_for_target()
        || !(1..=8).contains(&query_rows)
        || !source_rows.is_multiple_of(2)
        || !head_dim.is_multiple_of(16)
        || !fits(query_rows.checked_mul(source_rows), weights.len())
        || !fits(source_rows.checked_mul(head_dim), packed_value.len())
        || !fits(query_rows.checked_mul(head_dim), output.len())
    {
        return false;
    }
    // SAFETY: checked above: the CPU features (f, bw, bf16), `query_rows` in 1..=8, even
    // `source_rows`, `head_dim % 16 == 0`, and every length.
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

/// Accumulates BF16 softmax weights times packed BF16 V (`pack_attention_value_bf16` layout)
/// into FP32 output through `VDPBF16PS`.
///
/// # Safety
///
/// The CPU must support `avx512f`, `avx512bw`, and `avx512bf16`; `head_dim % 16 == 0`;
/// `source_rows` is even; `packed_value.len() >= source_rows * head_dim`; `weights.len() >=
/// query_rows * source_rows`; and `output.len() >= query_rows * head_dim`. (`query_rows > 8`
/// panics on the accumulator index.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bf16")]
pub(crate) unsafe fn attention_value_packed_bf16_dot_avx512(
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

    // SAFETY: for block `b < head_dim / 16`, the output loads and stores cover 16 values at
    // `row * head_dim + b * 16`; the value load reads 32 `u16` at `(b * source_pairs + pair) * 32`,
    // below `head_dim * source_rows`; the weight read takes 2 `u16` at `row * source_rows +
    // pair * 2` via `read_unaligned`. The transmutes are between 64-byte vector types.
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
                    values[block] = core::mem::transmute::<__m512bh, core::arch::x86_64::__m512>(packed);
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_dot_for_target(backend: Isa, left: &[f32], right: &[f32]) -> Option<f32> {
    if right.len() < left.len() {
        return None;
    }
    // SAFETY: `avx512(backend)` detected avx512f, and
    // `right.len() >= left.len()` was checked above.
    avx512(backend).then(|| unsafe { attention_dot_avx512(left, right) })
}

/// Returns the dot product of `left` and `right`.
///
/// # Safety
///
/// The CPU must support `avx512f`, and `right.len() >= left.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_dot_avx512(left: &[f32], right: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};
    let end = left.len() / 16 * 16;
    let mut accumulator = _mm512_setzero_ps();
    // SAFETY: both loads read `[offset..offset + 16]` with `offset + 16 <= end <= left.len() <=
    // right.len()`.
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_value_fma_for_target(
    backend: Isa,
    weight: f32,
    value: &[f32],
    output: &mut [f32],
) -> bool {
    if output.len() < value.len() {
        return false;
    }
    if avx512(backend) {
        // SAFETY: `avx512(backend)` detected avx512f, and
        // `output.len() >= value.len()` was checked above.
        unsafe { attention_value_fma_avx512(weight, value, output) }
        true
    } else {
        false
    }
}

/// Applies `output = weight * value + output` elementwise.
///
/// # Safety
///
/// The CPU must support `avx512f`, and `output.len() >= value.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_value_fma_avx512(weight: f32, value: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_storeu_ps};
    let end = value.len() / 16 * 16;
    // SAFETY: the loads and the store cover `[offset..offset + 16]` with `offset + 16 <= end <=
    // value.len() <= output.len()`.
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn gelu_for_target(input: &[f32], output: &mut [f32]) -> bool {
    if output.len() < input.len() {
        return false;
    }
    if std::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f was detected just above, and `output.len() >= input.len()` above that.
        unsafe { gelu_avx512(input, output) }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn layer_norm_for_target(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> bool {
    let fits = rows
        .checked_mul(width)
        .is_some_and(|len| input.len() >= len && output.len() >= len)
        && weight.len() >= width
        && bias.is_none_or(|bias| bias.len() >= width);
    if !width.is_multiple_of(16) || !fits {
        return false;
    }
    if std::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f was detected just above; `width % 16 == 0` and every length were
        // checked above that.
        unsafe { layer_norm_avx512(input, rows, width, weight, bias, output) }
        true
    } else {
        false
    }
}

/// Applies the erf-based GELU approximation.
///
/// # Safety
///
/// The CPU must support `avx512f`, and `output.len() >= input.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn gelu_avx512(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::{
        _CMP_GE_OQ, _CMP_LT_OQ, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _mm512_add_epi32,
        _mm512_add_ps, _mm512_castsi512_ps, _mm512_cmp_ps_mask, _mm512_cvt_roundps_epi32, _mm512_cvtepi32_ps,
        _mm512_div_ps, _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_mask_mov_ps,
        _mm512_mask_sub_ps, _mm512_max_ps, _mm512_mul_ps, _mm512_set1_epi32, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_slli_epi32, _mm512_storeu_ps, _mm512_sub_ps,
    };
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let end = input.len() / 16 * 16;
    // SAFETY: the load reads `input[offset..offset + 16]` and the store writes the same range of
    // `output`, with `offset + 16 <= end <= input.len() <= output.len()`.
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

/// Applies `LayerNorm` to `rows` rows of `width` values.
///
/// # Safety
///
/// The CPU must support `avx512f`; `width % 16 == 0`; `input.len()` and `output.len()` are at
/// least `rows * width`; and `weight` and `bias`, if any, hold at least `width` values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[expect(
    clippy::cast_precision_loss,
    reason = "width is at most a few thousand, exact in f32"
)]
pub(crate) unsafe fn layer_norm_avx512(
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
    // SAFETY: with `row < rows` and `column + 16 <= width` (a multiple of 16), the loads read
    // `input[row * width + column..][..16]`, `weight[column..][..16]`, and `bias[column..][..16]`,
    // and the store writes `output[row * width + column..][..16]`.
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
pub(crate) fn affine_input3_transposed_for_target(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    if std::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f was detected just above. The lengths are checked by
        // `reference::affine_input3_transposed_f32`, the only caller.
        unsafe {
            affine_input3_transposed_avx512(input, rows, weight, bias, output_dim, output);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn affine_for_target(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    let fits = |count: Option<usize>, len: usize| count.is_some_and(|count| len >= count);
    if !fits(rows.checked_mul(input_dim), input.len())
        || !fits(output_dim.checked_mul(input_dim), weight.len())
        || !fits(rows.checked_mul(output_dim), output.len())
        || bias.is_some_and(|bias| bias.len() < output_dim)
    {
        return false;
    }
    if std::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f was detected just above, and every length checked above that.
        unsafe {
            affine_avx512(input, rows, input_dim, weight, bias, output_dim, output);
        }
        return true;
    }
    false
}

/// Runs a `3 -> output_dim` affine layer with input-major `[3, output_dim]` weights.
///
/// # Safety
///
/// The CPU must support `avx512f`; `input.len() >= rows * 3`; `weight.len() >= 3 * output_dim`;
/// `bias.len() >= output_dim`; and `output.len() >= rows * output_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn affine_input3_transposed_avx512(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_storeu_ps};

    // SAFETY: with `row < rows`, the input reads are `input[row * 3..][..3]`; vector loads and
    // stores at `channel + 16 <= vector_end <= output_dim` stay inside `bias`, each weight row,
    // and output row `row`; the scalar tail writes `channel < output_dim` of that row.
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

/// Runs `output = input * weight^T + bias` with PyTorch-layout `[output_dim, input_dim]`
/// weights.
///
/// # Safety
///
/// The CPU must support `avx512f`; `input.len() >= rows * input_dim`; `weight.len() >=
/// output_dim * input_dim`; and `output.len() >= rows * output_dim`. `bias` uses checked indexing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "register-blocked kernel: accumulators and loads named by row and column"
)]
pub(crate) unsafe fn affine_avx512(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};

    // SAFETY: each `affine_avx512_rows::<N>` call has `row + N <= rows` by its loop condition and
    // the final single-row call has `row < rows`. The two-row loop reads input rows `row` and
    // `row + 1 < rows` and weight rows `channel + lane < output_dim`, at columns below `input_dim`
    // (vector loads end at `vector_end <= input_dim`). Output writes use checked indexing.
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

/// `affine_avx512` for input rows `row..row + ROWS`.
///
/// # Safety
///
/// As for `affine_avx512`, and `row + ROWS <= rows`, so `input.len() >= (row + ROWS) *
/// input_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn affine_avx512_rows<const ROWS: usize>(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};

    // SAFETY: input pointers address rows `row..row + ROWS`; weight pointers address rows
    // `channel..channel + 4 <= output_dim` (or `channel < output_dim`); vector loads end at
    // `vector_end <= input_dim` and scalar reads at `input_dim`. Output writes use checked indexing.
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

/// `affine_avx512` for input row `row` only.
///
/// # Safety
///
/// As for `affine_avx512`, and `row < rows`, so `input.len() >= (row + 1) * input_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn affine_avx512_single_row(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps};

    // SAFETY: as in `affine_avx512_rows` with one input row `row`.
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

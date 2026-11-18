//! x86-64 kernels: AVX2 and AVX-512.

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
pub(crate) fn bf16_to_f32_for_target(input: &[u16], output: &mut [f32]) -> bool {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        unsafe { bf16_to_f32_avx512(input, output) };
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bf16_to_f32_avx512(input: &[u16], output: &mut [f32]) {
    use core::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm512_castsi512_ps, _mm512_cvtepu16_epi32, _mm512_slli_epi32,
        _mm512_storeu_ps,
    };
    unsafe {
        let end = input.len() / 16 * 16;
        for offset in (0..end).step_by(16) {
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "This intrinsic accepts unaligned pointers."
            )]
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
pub(crate) fn convert_f32_to_bf16(input: &[f32], output: &mut [u16]) -> bool {
    if input.len() != output.len() || !std::is_x86_feature_detected!("avx512bf16") {
        return false;
    }
    unsafe { convert_f32_to_bf16_avx512(input, output) };
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
pub(crate) unsafe fn convert_f32_to_bf16_avx512(input: &[f32], output: &mut [u16]) {
    use core::arch::x86_64::{__m256i, _mm256_storeu_si256, _mm512_cvtneps_pbh, _mm512_loadu_ps};

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
        unsafe { affine_packed_f32_avx512(input, rows, packed, bias, output) };
        true
    } else {
        false
    }
}

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
    if backend == Isa::Avx512 && matches!(head_dim, 16 | 64) {
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
    if backend == Isa::Avx512
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
    if backend == Isa::Avx512
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
    if backend == Isa::Avx512
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
    if backend == Isa::Avx512
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

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_value_tile_fma_for_target(
    backend: Isa,
    weights: &[f32],
    query_rows: usize,
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if backend == Isa::Avx512 && matches!(head_dim, 16 | 64) {
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
pub(crate) fn attention_value_tile_bf16_fma_for_target(
    backend: Isa,
    weights: &[f32],
    query_rows: usize,
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if backend == Isa::Avx512 && std::is_x86_feature_detected!("avx512bw") && matches!(head_dim, 16 | 64) {
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
    (backend == Isa::Avx512).then(|| unsafe { softmax_exp_sum_avx512(scores, maximum) })
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_score_maximum_for_target(backend: Isa, scores: &[f32]) -> Option<(f32, bool)> {
    (backend == Isa::Avx512).then(|| unsafe { attention_score_maximum_avx512(scores) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_score_maximum_avx512(scores: &[f32]) -> (f32, bool) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_dot_for_target(backend: Isa, left: &[f32], right: &[f32]) -> Option<f32> {
    (backend == Isa::Avx512).then(|| unsafe { attention_dot_avx512(left, right) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_dot_avx512(left: &[f32], right: &[f32]) -> f32 {
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn attention_value_fma_for_target(
    backend: Isa,
    weight: f32,
    value: &[f32],
    output: &mut [f32],
) -> bool {
    if backend == Isa::Avx512 {
        unsafe { attention_value_fma_avx512(weight, value, output) }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn attention_value_fma_avx512(weight: f32, value: &[f32], output: &mut [f32]) {
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn gelu_for_target(input: &[f32], output: &mut [f32]) -> bool {
    if std::is_x86_feature_detected!("avx512f") {
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
    if std::is_x86_feature_detected!("avx512f") {
        unsafe { layer_norm_avx512(input, rows, width, weight, bias, output) }
        true
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn layer_norm_statistics_for_target(
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn qass_modulate_for_target(
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
pub(crate) unsafe fn layer_norm_statistics_avx512(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn qass_modulate_avx512(
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
    if std::is_x86_feature_detected!("avx512f") {
        unsafe {
            affine_avx512(input, rows, input_dim, weight, bias, output_dim, output);
        }
        return true;
    }
    false
}

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

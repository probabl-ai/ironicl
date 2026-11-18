//! Attention: score and value tiles, softmax, and packed BF16 values.

#![allow(
    unsafe_code,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::missing_transmute_annotations,
    clippy::wildcard_imports // Sibling items differ per target.
)]

use crate::kernels::cpu::*;

/// Run head-major FP32 attention natively unless `backend` is `Portable`:
/// through Accelerate when it is large enough, else with the short-source
/// NEON kernel. Returns false when the caller must use its own kernel.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
pub(crate) fn attention_f32(
    backend: Isa,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if backend == Isa::Portable {
        return false;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if query_rows * source_rows >= SGEMM_MIN_ATTENTION_SCORES
        && accelerate::attention(
            query,
            query_rows,
            key,
            value,
            source_rows,
            heads,
            head_dim,
            output,
        )
    {
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if small_attention(
        query,
        query_rows,
        key,
        value,
        source_rows,
        heads,
        head_dim,
        output,
    ) {
        return true;
    }
    false
}

/// Apply `output = weight * value + output` across one attention value row.
///
/// The caller validates equal slice lengths. Each lane performs one FP32 FMA,
/// so repeated source-row calls preserve the scalar accumulation order.
pub(crate) fn attention_value_fma_f32(backend: Isa, weight: f32, value: &[f32], output: &mut [f32]) -> bool {
    attention_value_fma_for_target(backend, weight, value, output)
}

/// Compute one attention score dot product with native SIMD when available.
///
/// The caller validates equal slice lengths.
pub(crate) fn attention_dot_f32(backend: Isa, left: &[f32], right: &[f32]) -> Option<f32> {
    attention_dot_for_target(backend, left, right)
}

/// Compute one row-major `[query_rows, source_rows]` attention score tile.
///
/// Query and key rows are contiguous. The caller validates all slice lengths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_tile_f32(
    backend: Isa,
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
    backend: Isa,
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
    backend: Isa,
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
    backend: Isa,
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
    backend: Isa,
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
    backend: Isa,
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
    backend: Isa,
    weights: &[f32],
    query_rows: usize,
    packed_value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
    weights_bf16: &mut [u16],
) -> bool {
    if backend != Isa::Avx512
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
pub(crate) fn attention_score_maximum_f32(backend: Isa, scores: &[f32]) -> Option<(f32, bool)> {
    attention_score_maximum_for_target(backend, scores)
}

/// Replace scores with `exp(score - maximum)` and return their FP32 sum.
///
/// Returns `None` when the selected backend has no validated vector path.
pub(crate) fn softmax_exp_sum_f32(backend: Isa, scores: &mut [f32], maximum: f32) -> Option<f32> {
    if cfg!(feature = "exact-math") {
        return None;
    }
    softmax_exp_sum_for_target(backend, scores, maximum)
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_packed_key_bf16_tile_for_target(
    _backend: Isa,
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
pub(crate) fn attention_score_packed_key_tile_for_target(
    _backend: Isa,
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
pub(crate) fn attention_score_packed_key_padded_tile_for_target(
    _backend: Isa,
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
pub(crate) fn attention_score_tile_bf16_for_target(
    _backend: Isa,
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

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_tile_for_target(
    _backend: Isa,
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

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_value_tile_bf16_fma_for_target(
    _backend: Isa,
    _weights: &[f32],
    _query_rows: usize,
    _value: &[u16],
    _source_rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn softmax_exp_sum_for_target(_backend: Isa, _scores: &mut [f32], _maximum: f32) -> Option<f32> {
    None
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn attention_score_maximum_for_target(_backend: Isa, _scores: &[f32]) -> Option<(f32, bool)> {
    None
}

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_value_tile_fma_for_target(
    _backend: Isa,
    _weights: &[f32],
    _query_rows: usize,
    _value: &[f32],
    _source_rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn attention_value_packed_bf16_dot_for_target(
    _weights: &[u16],
    _query_rows: usize,
    _packed_value: &[u16],
    _source_rows: usize,
    _head_dim: usize,
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
pub(crate) fn attention_dot_for_target(_backend: Isa, _left: &[f32], _right: &[f32]) -> Option<f32> {
    None
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
pub(crate) fn attention_value_fma_for_target(
    _backend: Isa,
    _weight: f32,
    _value: &[f32],
    _output: &mut [f32],
) -> bool {
    false
}

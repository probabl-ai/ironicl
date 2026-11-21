//! Attention: score and value tiles and softmax.

use crate::kernels::cpu::*;

/// Run head-major FP32 attention natively unless `backend` is `Portable`:
/// through Accelerate when it is large enough, else with the short-source
/// NEON kernel. Returns false when the caller must use its own kernel.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
#[cfg_attr(
    not(target_arch = "aarch64"),
    expect(unused_variables, reason = "only the AArch64 version uses every argument")
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes its buffers and their shape as plain arguments"
)]
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

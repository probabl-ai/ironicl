//! `WebAssembly` kernels: SIMD128.

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

/// Scores are an affine product of query rows with key rows, then scaled.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_score_tile_for_target(
    _backend: Isa,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    source_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) -> bool {
    let scores = &mut scores[..query_rows * source_rows];
    affine_for_target(query, query_rows, head_dim, key, None, source_rows, scores);
    for score in scores {
        *score *= scale;
    }
    true
}

/// `output[row] += weights[row, source] * value[source]` over the tile, with
/// blocks of four query rows by sixteen dimensions held in registers. Sources
/// accumulate in order, as in the portable loop, but each product rounds
/// before its sum: WebAssembly has no fused multiply-add.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_value_tile_fma_for_target(
    _backend: Isa,
    weights: &[f32],
    query_rows: usize,
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> bool {
    if !head_dim.is_multiple_of(16) {
        return false;
    }
    let mut row = 0;
    while row + 4 <= query_rows {
        attention_value_simd128_rows::<4>(weights, row, value, source_rows, head_dim, output);
        row += 4;
    }
    while row < query_rows {
        attention_value_simd128_rows::<1>(weights, row, value, source_rows, head_dim, output);
        row += 1;
    }
    true
}

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub(crate) fn attention_value_simd128_rows<const ROWS: usize>(
    weights: &[f32],
    row: usize,
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, v128};

    let weights: [&[f32]; ROWS] =
        std::array::from_fn(|index| &weights[(row + index) * source_rows..][..source_rows]);
    for chunk in (0..head_dim).step_by(16) {
        let mut accumulators: [[v128; 4]; ROWS] = std::array::from_fn(|index| {
            std::array::from_fn(|lane| wasm_load(output, (row + index) * head_dim + chunk + lane * 4))
        });
        for source in 0..source_rows {
            let values: [v128; 4] =
                std::array::from_fn(|lane| wasm_load(value, source * head_dim + chunk + lane * 4));
            for (accumulators, weights) in accumulators.iter_mut().zip(&weights) {
                let weight = f32x4_splat(weights[source]);
                for (accumulator, value) in accumulators.iter_mut().zip(&values) {
                    *accumulator = f32x4_add(*accumulator, f32x4_mul(weight, *value));
                }
            }
        }
        for (index, accumulators) in accumulators.iter().enumerate() {
            for (lane, accumulator) in accumulators.iter().enumerate() {
                wasm_store(output, (row + index) * head_dim + chunk + lane * 4, *accumulator);
            }
        }
    }
}

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
#[allow(clippy::unnecessary_wraps)] // The other targets' signature.
pub(crate) fn attention_dot_for_target(_backend: Isa, left: &[f32], right: &[f32]) -> Option<f32> {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat};

    let end = left.len() / 4 * 4;
    let mut accumulator = f32x4_splat(0.0);
    for offset in (0..end).step_by(4) {
        accumulator = f32x4_add(
            accumulator,
            f32x4_mul(wasm_load(left, offset), wasm_load(right, offset)),
        );
    }
    let mut sum = wasm_sum(accumulator);
    for (left, right) in left[end..].iter().zip(&right[end..]) {
        sum += left * right;
    }
    Some(sum)
}

/// Multiplies and adds round separately: WebAssembly has no fused multiply-add.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub(crate) fn attention_value_fma_for_target(
    _backend: Isa,
    weight: f32,
    value: &[f32],
    output: &mut [f32],
) -> bool {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat};

    let weights = f32x4_splat(weight);
    let end = value.len() / 4 * 4;
    for offset in (0..end).step_by(4) {
        let sum = f32x4_add(
            wasm_load(output, offset),
            f32x4_mul(weights, wasm_load(value, offset)),
        );
        wasm_store(output, offset, sum);
    }
    for (value, output) in value[end..].iter().zip(&mut output[end..]) {
        *output += weight * value;
    }
    true
}

/// Four lanes from `values[offset..offset + 4]`.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
#[inline]
pub(crate) fn wasm_load(values: &[f32], offset: usize) -> core::arch::wasm32::v128 {
    let values = &values[offset..offset + 4];
    // SAFETY: the checked reslice holds exactly four `f32` (16 bytes), and `v128_load` reads
    // unaligned (`read_unaligned`); `simd128` is enabled at compile time by the `cfg`.
    unsafe { core::arch::wasm32::v128_load(values.as_ptr().cast()) }
}

/// Store four lanes into `values[offset..offset + 4]`.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
#[inline]
pub(crate) fn wasm_store(values: &mut [f32], offset: usize, vector: core::arch::wasm32::v128) {
    let values = &mut values[offset..offset + 4];
    // SAFETY: the checked reslice holds exactly four `f32` (16 bytes) borrowed mutably, and
    // `v128_store` writes unaligned (`write_unaligned`); `simd128` is enabled by the `cfg`.
    unsafe { core::arch::wasm32::v128_store(values.as_mut_ptr().cast(), vector) }
}

/// Sum of four lanes, in pairs.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
#[inline]
pub(crate) fn wasm_sum(vector: core::arch::wasm32::v128) -> f32 {
    use core::arch::wasm32::f32x4_extract_lane;
    (f32x4_extract_lane::<0>(vector) + f32x4_extract_lane::<1>(vector))
        + (f32x4_extract_lane::<2>(vector) + f32x4_extract_lane::<3>(vector))
}

/// WebAssembly SIMD: blocks of four rows by four output channels, four lanes
/// each. WebAssembly has no fused multiply-add, so products and sums round
/// separately; `f32::mul_add` would call a slow software `fmaf`.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub(crate) fn affine_for_target(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) -> bool {
    let mut row = 0;
    while row + 4 <= rows {
        affine_simd128_rows::<4>(input, row, input_dim, weight, bias, output_dim, output);
        row += 4;
    }
    while row < rows {
        affine_simd128_rows::<1>(input, row, input_dim, weight, bias, output_dim, output);
        row += 1;
    }
    true
}

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub(crate) fn affine_simd128_rows<const ROWS: usize>(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) {
    let mut channel = 0;
    while channel + 4 <= output_dim {
        affine_simd128_block::<ROWS, 4>(input, row, input_dim, weight, bias, channel, output_dim, output);
        channel += 4;
    }
    while channel < output_dim {
        affine_simd128_block::<ROWS, 1>(input, row, input_dim, weight, bias, channel, output_dim, output);
        channel += 1;
    }
}

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn affine_simd128_block<const ROWS: usize, const CHANNELS: usize>(
    input: &[f32],
    row: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    channel: usize,
    output_dim: usize,
    output: &mut [f32],
) {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, v128};

    let inputs: [&[f32]; ROWS] =
        std::array::from_fn(|index| &input[(row + index) * input_dim..][..input_dim]);
    let weights: [&[f32]; CHANNELS] =
        std::array::from_fn(|index| &weight[(channel + index) * input_dim..][..input_dim]);
    let mut accumulators = [[f32x4_splat(0.0); CHANNELS]; ROWS];
    let vector_end = input_dim / 4 * 4;
    let mut column = 0;
    while column < vector_end {
        let weight_vectors: [v128; CHANNELS] = std::array::from_fn(|index| wasm_load(weights[index], column));
        for (accumulators, input) in accumulators.iter_mut().zip(&inputs) {
            let input_vector = wasm_load(input, column);
            for (accumulator, weight_vector) in accumulators.iter_mut().zip(&weight_vectors) {
                *accumulator = f32x4_add(*accumulator, f32x4_mul(*weight_vector, input_vector));
            }
        }
        column += 4;
    }
    for (index, (accumulators, input)) in accumulators.iter().zip(&inputs).enumerate() {
        for (lane, (accumulator, weight_row)) in accumulators.iter().zip(&weights).enumerate() {
            let mut sum = wasm_sum(*accumulator);
            for column in vector_end..input_dim {
                sum += weight_row[column] * input[column];
            }
            let offset = channel + lane;
            output[(row + index) * output_dim + offset] = bias.map_or(sum, |values| sum + values[offset]);
        }
    }
}

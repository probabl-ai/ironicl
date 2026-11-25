#![allow(
    clippy::cast_precision_loss,
    reason = "test data from small integer indices, exact in f32"
)]

#[cfg(target_arch = "x86_64")]
use super::{Isa, attention_score_maximum_f32, fast_exp_negative, softmax_exp_sum_f32};
use super::{affine_f32, fast_erf, fast_gelu, gelu_f32, layer_norm_f32};

#[cfg(target_arch = "x86_64")]
#[test]
fn native_attention_maximum_handles_tails_and_nan() {
    let backend = Isa::detect();
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

#[cfg(target_arch = "x86_64")]
#[test]
fn softmax_exponential_tracks_exact_exp() {
    let mut maximum_relative_error = 0.0_f32;
    for index in -80_000..=0 {
        let value = index as f32 / 1_000.0;
        let exact = value.exp();
        maximum_relative_error = maximum_relative_error.max((fast_exp_negative(value) - exact).abs() / exact);
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
    let Some(sum) = softmax_exp_sum_f32(Isa::detect(), &mut scores, maximum) else {
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
    let mut maximum_gelu_error = 0.0_f32;
    for index in -10_000..=10_000 {
        let value = index as f32 / 1_000.0;
        maximum_erf_error = maximum_erf_error.max((fast_erf(value) - libm::erff(value)).abs());
        let exact_gelu = 0.5 * value * (1.0 + libm::erff(value / core::f32::consts::SQRT_2));
        maximum_gelu_error = maximum_gelu_error.max((fast_gelu(value) - exact_gelu).abs());
    }
    assert!(maximum_erf_error <= 6e-7, "erf error: {maximum_erf_error}");
    assert!(maximum_gelu_error <= 1e-6, "GELU error: {maximum_gelu_error}");
}

#[test]
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

#[cfg(target_arch = "aarch64")]
fn sine_values(count: usize, seed: f32) -> Vec<f32> {
    (0..count).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
}

/// Run `kernel` on `[heads, rows, head_dim]` inputs and compare it with a
/// plain softmax attention loop.
#[cfg(target_arch = "aarch64")]
fn assert_attention_matches_naive(
    (heads, q_rows, source_rows, head_dim): (usize, usize, usize, usize),
    kernel: impl Fn(&[f32], &[f32], &[f32], &mut [f32]) -> bool,
) {
    let query = sine_values(heads * q_rows * head_dim, 4.0);
    let key = sine_values(heads * source_rows * head_dim, 5.0);
    let value = sine_values(heads * source_rows * head_dim, 6.0);
    let mut actual = vec![0.0; query.len()];
    assert!(kernel(&query, &key, &value, &mut actual));
    let scale = (head_dim as f32).sqrt().recip();
    for head in 0..heads {
        for row in 0..q_rows {
            let q = &query[(head * q_rows + row) * head_dim..][..head_dim];
            let scores = (0..source_rows)
                .map(|source| {
                    let k = &key[(head * source_rows + source) * head_dim..][..head_dim];
                    q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            for dim in 0..head_dim {
                let expected = (0..source_rows)
                    .map(|source| weights[source] * value[(head * source_rows + source) * head_dim + dim])
                    .sum::<f32>()
                    / total;
                assert!((actual[(head * q_rows + row) * head_dim + dim] - expected).abs() < 1e-5);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn small_attention_matches_naive() {
    // The row stage's shape, and the longest supported source.
    for (q_rows, source_rows) in [(26, 26), (4, 26), (3, 128)] {
        let shape = (8, q_rows, source_rows, 16);
        assert_attention_matches_naive(shape, |query, key, value, output| {
            super::small_attention(query, q_rows, key, value, source_rows, 8, 16, output)
        });
    }
    assert!(!super::small_attention(&[], 1, &[], &[], 129, 1, 16, &mut []));
    // A portable request never takes a native kernel.
    let (query, mut output) = (vec![0.5; 16], vec![0.0; 16]);
    assert!(!super::attention_f32(
        super::Isa::Portable,
        &query,
        1,
        &query,
        &query,
        1,
        1,
        16,
        &mut output
    ));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn accelerate_matches_naive_affine_and_attention() {
    let (rows, input_dim, output_dim) = (33, 20, 7);
    let (input, weight, bias) = (
        sine_values(rows * input_dim, 1.0),
        sine_values(output_dim * input_dim, 2.0),
        sine_values(output_dim, 3.0),
    );
    let mut actual = vec![0.0; rows * output_dim];
    assert!(super::accelerate::affine(
        &input,
        rows,
        input_dim,
        &weight,
        Some(&bias),
        output_dim,
        &mut actual
    ));
    for row in 0..rows {
        for out in 0..output_dim {
            let expected = bias[out]
                + (0..input_dim)
                    .map(|i| input[row * input_dim + i] * weight[out * input_dim + i])
                    .sum::<f32>();
            assert!((actual[row * output_dim + out] - expected).abs() < 1e-4);
        }
    }

    // More query rows than one 256-row tile, two heads.
    let (heads, q_rows, source_rows, head_dim) = (2, 300, 9, 4);
    assert_attention_matches_naive(
        (heads, q_rows, source_rows, head_dim),
        |query, key, value, output| {
            super::accelerate::attention(query, q_rows, key, value, source_rows, heads, head_dim, output)
        },
    );

    // NaN scores must fall back instead of silently succeeding with NaN outputs.
    let mut nan_query = vec![1.0_f32; 4 * 4];
    nan_query[0] = f32::NAN;
    let key = vec![1.0_f32; 4 * 4];
    let value = vec![1.0_f32; 4 * 4];
    let mut out = vec![0.0_f32; 4 * 4];
    assert!(!super::small_attention(
        &nan_query, 4, &key, &value, 4, 1, 4, &mut out
    ));
    assert!(!super::accelerate::attention(
        &nan_query, 4, &key, &value, 4, 1, 4, &mut out
    ));
}

#![allow(
    unsafe_code,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::missing_transmute_annotations
)]

#[cfg(target_arch = "x86_64")]
use super::{
    Bf16AffineCompensation, Isa, attention_score_maximum_f32, fast_exp_negative, softmax_exp_sum_f32,
};
use super::{
    PackedBf16Affine, PackedF32Affine, affine_f32, f32_to_bf16_rne, fast_erf, fast_gelu, gelu_f32,
    layer_norm_f32, pack_attention_value_bf16,
};

#[cfg(target_arch = "x86_64")]
fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(clippy::float_cmp)]
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
    use super::{Isa, attention_value_packed_bf16_dot_f32};

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
        Isa::Avx512,
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

    use super::{Isa, attention_value_packed_bf16_dot_f32, attention_value_tile_bf16_fma_f32};

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

    if !std::is_x86_feature_detected!("avx512bf16") {
        eprintln!("skipped: AVX512_BF16 is unavailable");
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
    let mut baseline_output = vec![0.0; rows * head_dim];
    let mut dot_output = baseline_output.clone();
    let mut weights_bf16 = vec![0_u16; weights.len()];
    let iterations = 2_000;
    let baseline = measure(
        || {
            baseline_output.fill(0.0);
            assert!(attention_value_tile_bf16_fma_f32(
                Isa::Avx512,
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
                Isa::Avx512,
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
                expected =
                    weight[channel * input_dim + column].mul_add(input[row * input_dim + column], expected);
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

    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("skipped: AVX512F is unavailable");
        return;
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

    if !std::is_x86_feature_detected!("avx512bf16") {
        eprintln!("skipped: AVX512_BF16 is unavailable");
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

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(clippy::cast_precision_loss)]
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
}

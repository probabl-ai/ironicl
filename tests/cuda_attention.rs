//! ICL attention on an NVIDIA GPU against an FP64 reference. Skips without a
//! CUDA device.
#![cfg(feature = "cuda")]

use half::f16;
use ironicl::cuda::{Device, HEAD_DIM, Precision};

const D: usize = HEAD_DIM;

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 40) as f32 / (1u64 << 23) as f32;
            unit - 1.0
        })
        .collect()
}

/// FP16-rounded values, as the device sees them.
fn rounded(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

fn bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| f16::from_f32(v).to_bits()).collect()
}

fn reference(q: &[f32], k: &[f32], v: &[f32], rows: usize, scale: f64) -> Vec<f64> {
    let scores: Vec<f64> = (0..rows)
        .map(|j| {
            (0..D)
                .map(|d| f64::from(q[d]) * f64::from(k[j * D + d]))
                .sum::<f64>()
                * scale
        })
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    (0..D)
        .map(|d| {
            (0..rows)
                .map(|j| weights[j] * f64::from(v[j * D + d]))
                .sum::<f64>()
                / total
        })
        .collect()
}

#[test]
fn icl_attention_matches_an_fp64_reference() {
    let Some(device) = Device::new().expect("load CUDA kernels") else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let pairs = 64;
    let scale = 0.125_f64; // 1 / sqrt(HEAD_DIM)
    // Rows below, at, and past a tile, and query counts around both modes.
    for rows in [1usize, 63, 64, 65, 2_000, 10_007] {
        let k = rounded(&values(pairs * rows * D, 2));
        let v = rounded(&values(pairs * rows * D, 3));
        let (k_bits, v_bits) = (bits(&k), bits(&v));
        for (query_rows, precision) in [1usize, 5, 16, 17, 64, 100].into_iter().flat_map(|rows| {
            [
                (rows, Precision::Fp32),
                (rows, Precision::Split),
                (rows, Precision::Half),
            ]
        }) {
            let q = rounded(&values(pairs * query_rows * D, 1));
            let out = device
                .icl_attention(
                    precision,
                    &bits(&q),
                    (&k_bits, &v_bits),
                    (pairs, query_rows, rows),
                )
                .expect("attention");
            let mut max_error = 0.0f64;
            for pair in [0, 17, 63] {
                for row in [0, query_rows / 2, query_rows - 1] {
                    let expected = reference(
                        &q[(pair * query_rows + row) * D..][..D],
                        &k[pair * rows * D..],
                        &v[pair * rows * D..],
                        rows,
                        scale,
                    );
                    let got = &out[(pair * query_rows + row) * D..][..D];
                    for d in 0..D {
                        max_error = max_error.max((f64::from(got[d]) - expected[d]).abs());
                    }
                }
            }
            // FP32 keeps every value and weight in FP32. Split rounds the
            // softmax weights to FP16 before P V. Under Half, keys and values
            // are FP8 (3 mantissa bits), and wide batches accumulate each
            // 64-key stage in FP16.
            let tolerance = match precision {
                Precision::Fp32 => 1e-5,
                Precision::Split => 1e-4,
                Precision::Half => 3e-2,
            };
            assert!(
                max_error < tolerance,
                "rows {rows}, queries {query_rows}, {precision:?}: max error {max_error:.2e}"
            );
        }
    }
}

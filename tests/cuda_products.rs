//! The products of CUDA prediction, every precision, against an FP64
//! reference: two chained products, the second reading the first's output in
//! the operand form the first wrote. Skips without a CUDA device.
#![cfg(feature = "cuda")]

use half::f16;
use ironicl::models::tabicl_v2::cuda::{Device, Precision};

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(
                clippy::cast_precision_loss,
                reason = "test data from small integer indices, exact in f32"
            )]
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            (unit - 0.5) * 0.2
        })
        .collect()
}

fn gelu(v: f64) -> f64 {
    0.5 * v * (1.0 + libm::erf(v / std::f64::consts::SQRT_2))
}

/// `act(x wᵀ + b)` in FP64, its inputs first rounded to FP16 when `half`.
#[allow(
    clippy::many_single_char_names,
    reason = "m, n, k name the product's dimensions"
)]
fn product(x: &[f64], w: &[f32], b: &[f32], (m, n, k): (usize, usize, usize), half: bool) -> Vec<f64> {
    let round = |v: f64| {
        if half {
            f64::from(f16::from_f64(v).to_f32())
        } else {
            v
        }
    };
    (0..m * n)
        .map(|at| {
            let (i, j) = (at / n, at % n);
            (0..k)
                .map(|t| round(x[i * k + t]) * round(f64::from(w[j * k + t])))
                .sum::<f64>()
                + f64::from(b[j])
        })
        .collect()
}

#[test]
fn tensor_products_match_an_fp64_reference() {
    let Some(device) = Device::new().expect("load CUDA kernels") else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    // Ragged rows, a 10-wide output, and k of 64, 512, and 1024.
    for (m, n, k, n2) in [
        (65, 64, 64, 64),
        (300, 512, 512, 10),
        (513, 1024, 512, 512),
        (128, 512, 1024, 512),
    ] {
        let x = values(m * k, 1);
        let (w1, b1) = (values(n * k, 2), values(n, 3));
        let (w2, b2) = (values(n2 * n, 4), values(n2, 5));
        for precision in [Precision::Fp32, Precision::Split, Precision::Half] {
            let half = precision == Precision::Half;
            let (y, z) = device
                .tensor_products(precision, &x, (&w1, &b1), (&w2, &b2), (m, n, k, n2))
                .expect("products");
            let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            let y_ref: Vec<f64> = product(&x64, &w1, &b1, (m, n, k), half)
                .into_iter()
                .map(gelu)
                .collect();
            let mut z_ref = product(&y_ref, &w2, &b2, (m, n2, n), half);
            if n2 == n {
                z_ref.iter_mut().zip(&y_ref).for_each(|(z, y)| *z += y);
            }
            let error = |got: &[f32], want: &[f64]| {
                let peak = want.iter().fold(0.0f64, |p, v| p.max(v.abs()));
                got.iter()
                    .zip(want)
                    .map(|(g, w)| (f64::from(*g) - w).abs())
                    .fold(0.0, f64::max)
                    / peak
            };
            // FP32 and split parts are close to FP32; FP16 rounds the second
            // product's operand, which the reference rounds from the exact
            // first output.
            let tolerance = if half { 2e-3 } else { 1e-5 };
            let (ey, ez) = (error(&y, &y_ref), error(&z, &z_ref));
            assert!(
                ey < tolerance && ez < tolerance,
                "{precision:?} m {m} n {n} k {k} n2 {n2}: relative errors {ey:.2e}, {ez:.2e}"
            );
        }
    }
}

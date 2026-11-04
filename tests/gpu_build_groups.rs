#![cfg(all(
    feature = "checkpoint",
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64"
))]
// The only test in this binary, so changing the environment races nothing.
#![allow(unsafe_code)]

use std::path::Path;

use ironicl::{ClassifierExecutor, ClassifierQueryRequest, Optimize, load_model_artifact};

/// A GPU build that runs one estimator per graph must match a build that runs
/// every estimator at once. The graph shapes differ, so the GPU may pick
/// other kernels. On an M1 Max: 2.4e-5 apart for `Safe` and 1.2e-4 for
/// `Fast`; a group copied to the wrong place is 1.6e-2 apart.
#[test]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn grouped_gpu_build_matches_one_graph_run() {
    let Some(directory) = std::env::var_os("IRONICL_CHECKPOINT_FIXTURES") else {
        return;
    };
    let model = load_model_artifact(&Path::new(&directory).join("tabicl-classifier-v2.ironicl"))
        .expect("load classifier artifact");
    let executor = ClassifierExecutor::new(model);
    let (b, n, q, h) = (4, 240, 50, 5);
    let value = |index: usize| ((index * 7919) % 1000) as f32 / 250.0 - 2.0;
    let train: Vec<f32> = (0..b * n * h).map(value).collect();
    let query: Vec<f32> = (0..b * q * h).map(|index| value(index + 17)).collect();
    let y: Vec<f32> = (0..b * n).map(|row| (row % 3) as f32).collect();
    let logits = |optimize| {
        executor
            .build_context(
                b,
                n,
                h,
                &train,
                &y,
                (0..b).flat_map(|_| 0..3).collect(),
                (0..b).flat_map(|_| 0..h as u64).collect(),
                optimize,
            )
            .expect("build context")
            .predict_logits(ClassifierQueryRequest::new(b, q, h, &query).expect("valid query"))
            .expect("logits")
            .values()
            .to_vec()
    };
    for (optimize, tolerance) in [(Optimize::Safe, 1e-4), (Optimize::Fast, 1e-3)] {
        let together = logits(optimize);
        // About 17 MB per estimator at 240 x 5: one estimator per run.
        // SAFETY: no other thread reads the environment.
        unsafe { std::env::set_var("IRONICL_GPU_BUILD_BUDGET", "20000000") };
        let grouped = logits(optimize);
        unsafe { std::env::remove_var("IRONICL_GPU_BUILD_BUDGET") };
        for (index, (together, grouped)) in together.iter().zip(&grouped).enumerate() {
            assert!(
                (together - grouped).abs() <= tolerance,
                "{optimize:?} logit {index}: grouped {grouped}, together {together}"
            );
        }
    }
}

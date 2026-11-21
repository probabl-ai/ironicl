#![cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]

use std::path::Path;

use ironicl::{
    ClassifierAggregation, Executor, Optimize, QueryRequest, StaticExecutor, load_model_artifact,
    memory::MemoryLimits,
};

/// The all-missing marker. A row whose every feature holds it runs through the
/// CPU column stage only, in a build or in a prediction on the GPU.
const MISSING: f32 = -100.0;

/// Builds `safe` and `auto` contexts of the same training rows and checks that
/// they predict the same probabilities. Rows listed in `missing_training` and
/// `missing_query` hold the all-missing marker in every estimator.
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "test data from small integer indices; b, q, h, n name the model's dimensions"
)]
fn assert_safe_matches_auto(missing_training: &[usize], missing_query: &[usize]) {
    let directory = std::env::var_os("IRONICL_CHECKPOINT_FIXTURES")
        .expect("set IRONICL_CHECKPOINT_FIXTURES to run this ignored test; see CONTRIBUTING.md");
    let model = load_model_artifact(&Path::new(&directory).join("tabicl-classifier-v2.irmodel"))
        .expect("load classifier artifact");
    let executor = Executor::new(model);
    let (b, n, q, h) = (4, 2000, 100, 8);
    let value = |index: usize| ((index * 7919) % 1000) as f32 / 250.0 - 2.0;
    let mark = |rows: &mut [f32], count: usize, missing: &[usize]| {
        for estimator in 0..b {
            for &row in missing {
                rows[(estimator * count + row) * h..][..h].fill(MISSING);
            }
        }
    };
    let mut train: Vec<f32> = (0..b * n * h).map(value).collect();
    mark(&mut train, n, missing_training);
    let mut query: Vec<f32> = (0..b * q * h).map(|index| value(index + 17)).collect();
    mark(&mut query, q, missing_query);
    let y: Vec<f32> = (0..b * n).map(|row| (row % 3) as f32).collect();
    let build = |optimize| -> StaticExecutor {
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
                None,
                MemoryLimits::default(),
                None,
            )
            .expect("build context")
    };
    let (safe, auto) = (build(Optimize::Safe), build(Optimize::Auto));
    if cfg!(feature = "cuda") {
        assert_eq!(auto.device(), "cuda");
    }
    let proba = |context: &StaticExecutor| {
        let aggregation = ClassifierAggregation {
            softmax_temperature: 0.9,
            average_logits: true,
        };
        context
            .predict_probabilities(
                QueryRequest::new(b, q, h, &query).expect("valid query"),
                aggregation,
            )
            .expect("probabilities")
            .values()
            .to_vec()
    };
    for (index, (safe, auto)) in proba(&safe).iter().zip(&proba(&auto)).enumerate() {
        assert!(
            (safe - auto).abs() <= 1e-3,
            "probability {index}: safe {safe}, auto {auto}"
        );
    }
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn all_missing_training_rows_build_on_the_cpu() {
    assert_safe_matches_auto(&[3, 70], &[]);
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn all_missing_query_rows_match_the_cpu_path() {
    assert_safe_matches_auto(&[], &[3, 70]);
}

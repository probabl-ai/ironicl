use std::path::{Path, PathBuf};

use ironicl::{
    CheckpointCacheOptions, CheckpointProfile, Executor, Optimize, PreparedCheckpoint, QueryRequest, Request,
    TaskKind, import_checkpoint, inspect_checkpoint, inspect_model_artifact, load_model_artifact,
    prepare_checkpoint,
};
use tempfile::tempdir;

const CLASSIFIER_SHA256: &str = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0";
const REGRESSOR_SHA256: &str = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a";

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn inspects_official_classifier_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-classifier-v2-20260212.ckpt") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let summary = inspect_checkpoint(&path).expect("official classifier checkpoint");

    assert_eq!(summary.source_sha256, CLASSIFIER_SHA256);
    assert_eq!(summary.profile, CheckpointProfile::TabIclV2Classifier);
    assert_eq!(summary.tensor_count, 391);
    assert_eq!(summary.parameter_count, 27_552_258);
    assert_eq!(summary.tensor_bytes, 110_209_032);
    assert_eq!(summary.dtype_counts.get("f32"), Some(&391));
    assert_eq!(
        summary.config.and_then(|config| config.task()),
        Some(TaskKind::Classifier)
    );
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn inspects_official_regressor_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-regressor-v2-20260212.ckpt") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let summary = inspect_checkpoint(&path).expect("official regressor checkpoint");

    assert_eq!(summary.source_sha256, REGRESSOR_SHA256);
    assert_eq!(summary.profile, CheckpointProfile::TabIclV2Regressor);
    assert_eq!(summary.tensor_count, 347);
    assert_eq!(summary.parameter_count, 28_544_991);
    assert_eq!(summary.tensor_bytes, 114_179_964);
    assert_eq!(summary.dtype_counts.get("f32"), Some(&347));
    assert_eq!(
        summary.config.and_then(|config| config.task()),
        Some(TaskKind::Regressor)
    );
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn imports_official_classifier_and_reuses_automatic_cache() {
    let Some(checkpoint) = fixture("tabicl-classifier-v2-20260212.ckpt") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let directory = tempdir().expect("temporary cache directory");
    let explicit_model = directory.path().join("classifier.irmodel");
    let imported = import_checkpoint(&checkpoint, &explicit_model).expect("import classifier");
    assert_eq!(imported.source_sha256, CLASSIFIER_SHA256);
    assert_eq!(imported.profile, ironicl::ModelProfile::TabIclV2Classifier);
    assert_eq!(
        inspect_model_artifact(&explicit_model).expect("inspect imported model"),
        imported
    );

    let options = CheckpointCacheOptions {
        cache_dir: Some(directory.path().join("cache")),
        ..CheckpointCacheOptions::default()
    };
    let created = prepare_checkpoint(&checkpoint, &options).expect("create model cache");
    assert!(created.created, "expected a newly created cache entry");
    assert_eq!(created.summary.source_sha256, CLASSIFIER_SHA256);

    let reused = prepare_checkpoint(&checkpoint, &options).expect("reuse model cache");
    assert_eq!(
        reused,
        PreparedCheckpoint {
            created: false,
            ..created
        }
    );
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn loads_official_classifier_artifact_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-classifier-v2.irmodel") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let model = load_model_artifact(&path).expect("load official classifier artifact");

    assert_eq!(model.summary().profile, ironicl::ModelProfile::TabIclV2Classifier);
    let tensor = model
        .tensor_f32("icl_predictor.decoder.2.bias")
        .expect("official decoder bias");
    assert_eq!(tensor.dimensions(), &[10]);
    assert_eq!(tensor.values().len(), 10);
    assert!(model.tensor_f32("missing.tensor").is_err());
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "test data from small integer indices; b, q, h, n name the model's dimensions"
)]
fn built_context_matches_the_uncached_forward_pass() {
    let Some(model_path) = fixture("tabicl-classifier-v2.irmodel") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let executor = Executor::new(load_model_artifact(&model_path).expect("load classifier artifact"));
    // With the `metal` feature, 240 training rows build on the GPU, and 250
    // query rows are padded to the GPU's 256-row bucket.
    let (b, n, q, h, classes) = (2, 240, 250, 5, 3_u64);
    let mut state = 0x2545_f491_u32;
    let mut x = vec![0.0_f32; b * (n + q) * h];
    for value in &mut x {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *value = (state >> 8) as f32 / (1 << 24) as f32 * 4.0 - 2.0;
    }
    // Each estimator shifts the classes, as TabICL's cyclic class shuffle does.
    let shifts: Vec<Vec<u64>> = (0..b as u64)
        .map(|e| (0..classes).map(|c| (c + e) % classes).collect())
        .collect();
    let y: Vec<f32> = (0..b)
        .flat_map(|e| (0..n).map(move |row| (row as u64 % classes, e)))
        .map(|(c, e)| shifts[e][c as usize] as f32)
        .collect();
    let rows = |start: usize, count: usize| -> Vec<f32> {
        (0..b)
            .flat_map(|e| x[(e * (n + q) + start) * h..][..count * h].to_vec())
            .collect()
    };

    let expected = executor
        .predict_logits(Request::new(b, n + q, h, n, &x, &y).expect("valid request"))
        .expect("uncached logits");
    let query = rows(n, q);
    // Logit tolerances per level; reduced levels only differ from `Safe` on
    // GPU-built contexts. Probability errors are several times smaller.
    let mut safe = Vec::new();
    for (optimize, tolerance) in [
        (Optimize::Safe, 1e-4),
        // Measured on an M1 Max with FP16 caches and attention: 4.6e-3.
        (Optimize::Auto, 1e-2),
        (Optimize::Fast, 5e-2),
    ] {
        let context = executor
            .build_context(
                b,
                n,
                h,
                &rows(0, n),
                &y,
                shifts.concat(),
                (0..b).flat_map(|_| 0..h as u64).collect(),
                optimize,
                None,
                ironicl::memory::MemoryLimits::default(),
                None,
            )
            .expect("build context");
        let actual = context
            .predict_logits(QueryRequest::new(b, q, h, &query).expect("valid query"))
            .expect("cached logits");
        assert_eq!(actual.shape(), expected.shape());
        for (index, (actual, expected)) in actual.values().iter().zip(expected.values()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{optimize:?} logit {index}: cached {actual}, uncached {expected}"
            );
        }
        if optimize == Optimize::Safe {
            safe = actual.values().to_vec();
        } else if cfg!(all(
            feature = "metal",
            target_os = "macos",
            target_arch = "aarch64"
        )) && std::env::var_os("IRONICL_NO_METAL").is_none()
        {
            assert!(
                actual
                    .values()
                    .iter()
                    .zip(&safe)
                    .any(|(left, right)| left.to_bits() != right.to_bits()),
                "{optimize:?} must store reduced-precision keys and values"
            );
        }
    }
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "test data from small integer indices; b, q, h, n name the model's dimensions"
)]
fn large_cached_query_runs_in_chunks() {
    let Some(model_path) = fixture("tabicl-classifier-v2.irmodel") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let executor = Executor::new(load_model_artifact(&model_path).expect("load classifier artifact"));
    // 3000 query rows exceed the chunk size; each estimator's rows must land
    // in place when the chunks are joined.
    let (b, n, q, h) = (2, 240, 3000, 5);
    let value = |index: usize| ((index * 7919) % 1000) as f32 / 250.0 - 2.0;
    let train: Vec<f32> = (0..b * n * h).map(value).collect();
    let query: Vec<f32> = (0..b * q * h).map(|index| value(index + 17)).collect();
    let y: Vec<f32> = (0..b * n).map(|row| (row % 3) as f32).collect();
    let context = executor
        .build_context(
            b,
            n,
            h,
            &train,
            &y,
            (0..b).flat_map(|_| 0..3).collect(),
            (0..b).flat_map(|_| 0..h as u64).collect(),
            Optimize::Safe,
            None,
            ironicl::memory::MemoryLimits::default(),
            None,
        )
        .expect("build context");
    let all = context
        .predict_logits(QueryRequest::new(b, q, h, &query).expect("valid query"))
        .expect("chunked logits");
    let part = q / 3;
    for start in (0..q).step_by(part) {
        let rows: Vec<f32> = (0..b)
            .flat_map(|e| query[(e * q + start) * h..][..part * h].to_vec())
            .collect();
        let expected = context
            .predict_logits(QueryRequest::new(b, part, h, &rows).expect("valid query"))
            .expect("part logits");
        for e in 0..b {
            let actual = &all.values()[(e * q + start) * 3..][..part * 3];
            let expected = &expected.values()[e * part * 3..][..part * 3];
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "chunked {actual}, alone {expected}"
                );
            }
        }
    }
}

/// A query row whose grouped features are all the missing marker runs on the
/// CPU, which must read the FP16 caches of an `Auto` context.
#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "test data from small integer indices; b, q, h, n name the model's dimensions"
)]
fn reduced_precision_context_predicts_all_missing_rows() {
    let Some(model_path) = fixture("tabicl-classifier-v2.irmodel") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let executor = Executor::new(load_model_artifact(&model_path).expect("load classifier artifact"));
    let (b, n, q, h) = (2, 240, 4, 5);
    let value = |index: usize| ((index * 7919) % 1000) as f32 / 250.0 - 2.0;
    let train: Vec<f32> = (0..b * n * h).map(value).collect();
    let mut query: Vec<f32> = (0..b * q * h).map(|index| value(index + 17)).collect();
    for e in 0..b {
        query[e * q * h..][..h].fill(-100.0);
    }
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
                None,
                ironicl::memory::MemoryLimits::default(),
                None,
            )
            .expect("build context")
            .predict_logits(QueryRequest::new(b, q, h, &query).expect("valid query"))
            .expect("logits")
            .values()
            .to_vec()
    };
    let safe = logits(Optimize::Safe);
    for (index, (auto, safe)) in logits(Optimize::Auto).iter().zip(&safe).enumerate() {
        assert!(
            (auto - safe).abs() <= 1e-2,
            "logit {index}: auto {auto}, safe {safe}"
        );
    }
}

fn fixture(name: &str) -> Option<PathBuf> {
    let directory = std::env::var_os("IRONICL_CHECKPOINT_FIXTURES")?;
    let path = Path::new(&directory).join(name);
    assert!(
        path.is_file(),
        "checkpoint fixture is missing: {}",
        path.display()
    );
    Some(path)
}

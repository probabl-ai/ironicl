use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use ironicl::{
    CheckpointCacheOptions, CheckpointProfile, Executor, InputBundle, Optimize, PreparedCheckpoint,
    QueryRequest, TaskKind, import_checkpoint, inspect_checkpoint, inspect_model_artifact,
    load_model_artifact, prepare_checkpoint,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

const CLASSIFIER_SHA256: &str = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0";
const REGRESSOR_SHA256: &str = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a";
const ORACLE_INPUT_SHA256: &str = "dfaa40fda773a7de5c1216ca0051e447b05f16e4fafa315624c5e4d5b084a0e1";
const ORACLE_LOGITS_SHA256: &str = "be53800fb2262a0c4a5d2179514c5fe9cfbaf0dc3baf7269c780a8de93ccd2c1";

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
    let PreparedCheckpoint::Cached {
        path,
        created: true,
        summary,
    } = created
    else {
        panic!("expected a newly created cache entry");
    };
    assert_eq!(summary.source_sha256, CLASSIFIER_SHA256);

    let reused = prepare_checkpoint(&checkpoint, &options).expect("reuse model cache");
    assert_eq!(
        reused,
        PreparedCheckpoint::Cached {
            path,
            created: false,
            summary,
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
fn runs_official_classifier_artifact_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-classifier-v2.irmodel") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let directory = tempdir().expect("temporary input directory");
    let input_path = directory.path().join("request.irinput");
    let request = InputBundle::new(1, 3, 1, 2, vec![0.0, 1.0, -1.0], vec![0.0, 1.0])
        .expect("tiny valid classifier request");
    std::fs::write(&input_path, request.to_bytes()).expect("write input bundle");

    let output = Command::new(env!("CARGO_BIN_EXE_ironicl"))
        .args(["run", "--model"])
        .arg(path)
        .args(["--input"])
        .arg(input_path)
        .output()
        .expect("run CLI");
    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 CLI output");
    let lines = stdout.lines().collect::<Vec<_>>();
    assert_eq!(lines.first(), Some(&"shape: 1 1 2"));
    assert_eq!(lines.len(), 2);
    assert!(lines[1].starts_with("logits[0,0]: "));
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn matches_python_classifier_oracle_when_fixtures_are_available() {
    let Some(oracle) = oracle_fixtures() else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let Some(model_path) = fixture("tabicl-classifier-v2.irmodel") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };

    let input_bytes = fs::read(&oracle.input).expect("read Python oracle input");
    assert_eq!(sha256_hex(&input_bytes), ORACLE_INPUT_SHA256);
    let input = InputBundle::from_bytes(&input_bytes).expect("validate Python oracle input");

    let logits_bytes = fs::read(&oracle.logits).expect("read Python oracle logits");
    assert_eq!(sha256_hex(&logits_bytes), ORACLE_LOGITS_SHA256);
    let expected = read_irlogits(&logits_bytes).expect("validate Python oracle logits");

    let actual = Executor::new(load_model_artifact(&model_path).expect("load official classifier artifact"))
        .predict_logits(input.request())
        .expect("run official classifier artifact");
    assert_eq!(actual.shape(), expected.shape, "logit dimensions differ");
    assert_eq!(
        actual.values().len(),
        expected.values.len(),
        "logit lengths differ"
    );

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (index, (actual, expected)) in actual.values().iter().zip(&expected.values).enumerate() {
        assert!(
            actual.is_finite(),
            "Ironicl logit {index} is not finite: {actual}"
        );
        let abs = (actual - expected).abs();
        let rel = if *expected == 0.0 {
            if abs == 0.0 { 0.0 } else { f32::INFINITY }
        } else {
            abs / expected.abs()
        };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            abs <= 1e-4 + 1e-4 * expected.abs(),
            "logit {index} differs: actual={actual}, expected={expected}, abs={abs}, rel={rel}"
        );
    }
    println!("Python oracle parity: max_abs={max_abs:.9e}, max_rel={max_rel:.9e}");
}

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
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

    let bundle = InputBundle::new(b, n + q, h, n, x.clone(), y.clone()).expect("valid bundle");
    let expected = executor
        .predict_logits(bundle.request())
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
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
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
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
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

#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
fn no_cache_validates_without_writing() {
    let Some(checkpoint) = fixture("tabicl-classifier-v2-20260212.ckpt") else {
        panic!("fixtures required to run this test; set the fixture env vars. See CONTRIBUTING.md");
    };
    let prepared = prepare_checkpoint(
        &checkpoint,
        &CheckpointCacheOptions {
            no_cache: true,
            ..CheckpointCacheOptions::default()
        },
    )
    .expect("validate checkpoint without cache");

    assert_eq!(
        prepared,
        PreparedCheckpoint::Direct {
            source_sha256: CLASSIFIER_SHA256.into(),
            profile: ironicl::ModelProfile::TabIclV2Classifier,
        }
    );
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

struct OracleFixtures {
    input: PathBuf,
    logits: PathBuf,
}

fn oracle_fixtures() -> Option<OracleFixtures> {
    let directory = PathBuf::from(std::env::var_os("IRONICL_ORACLE_FIXTURES")?);
    let fixtures = OracleFixtures {
        input: directory.join("input.irinput"),
        logits: directory.join("logits.irlogits"),
    };
    assert!(
        fixtures.input.is_file() && fixtures.logits.is_file(),
        "oracle fixtures are missing: {} and {}",
        fixtures.input.display(),
        fixtures.logits.display()
    );
    Some(fixtures)
}

struct OracleLogits {
    shape: [usize; 3],
    values: Vec<f32>,
}

fn read_irlogits(bytes: &[u8]) -> std::result::Result<OracleLogits, String> {
    const HEADER_BYTES: usize = 48;
    const CHECKSUM_BYTES: usize = 32;
    if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
        return Err("IRLOGITS file is truncated".into());
    }
    let checksum_offset = bytes.len() - CHECKSUM_BYTES;
    if Sha256::digest(&bytes[..checksum_offset]).as_slice() != &bytes[checksum_offset..] {
        return Err("IRLOGITS checksum does not match".into());
    }
    let mut reader = OracleReader::new(&bytes[..HEADER_BYTES]);
    if reader.array::<8>()? != *b"IRLOGITS" {
        return Err("IRLOGITS magic does not match".into());
    }
    if reader.u32()? != 1 {
        return Err("IRLOGITS version is unsupported".into());
    }
    if reader.u32()? != 1 {
        return Err("IRLOGITS kind is not classifier logits".into());
    }
    let shape = [
        reader.usize("B")?,
        reader.usize("test rows")?,
        reader.usize("classes")?,
    ];
    let elements = reader.usize("element count")?;
    if !reader.is_empty() {
        return Err("IRLOGITS header has trailing bytes".into());
    }
    if shape.contains(&0) {
        return Err("IRLOGITS dimensions must be nonzero".into());
    }
    let expected_elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or("IRLOGITS dimensions overflow usize")
    })?;
    if elements != expected_elements {
        return Err("IRLOGITS element count does not match dimensions".into());
    }
    let payload_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("IRLOGITS payload length overflows usize")?;
    if checksum_offset != HEADER_BYTES + payload_bytes {
        return Err("IRLOGITS lengths do not match whole-file length".into());
    }
    let values = bytes[HEADER_BYTES..checksum_offset]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("FP32 chunks have four bytes")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("IRLOGITS contains a non-finite value".into());
    }
    Ok(OracleLogits { shape, values })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut encoded, "{byte:02x}").expect("write SHA-256 hex");
    }
    encoded
}

struct OracleReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> OracleReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn array<const N: usize>(&mut self) -> std::result::Result<[u8; N], String> {
        let end = self
            .position
            .checked_add(N)
            .ok_or("IRLOGITS field offset overflows usize")?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or("IRLOGITS file is truncated")?
            .try_into()
            .map_err(|_| "IRLOGITS field has an invalid length")?;
        self.position = end;
        Ok(value)
    }

    fn u32(&mut self) -> std::result::Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn usize(&mut self, name: &str) -> std::result::Result<usize, String> {
        usize::try_from(u64::from_le_bytes(self.array()?))
            .map_err(|_| format!("IRLOGITS {name} exceeds usize"))
    }
}

#![cfg(feature = "checkpoint")]

use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use ironicl::{
    CheckpointCacheOptions, CheckpointProfile, ClassifierExecutor, ClassifierInputBundle,
    ClassifierQueryRequest, KvDtype, PreparedCheckpoint, StaticClassifierExecutor, StaticContextArtifact,
    StaticContextOptions, StaticContextRequest, TaskKind, build_static_context, build_static_context_file,
    import_checkpoint, inspect_checkpoint, inspect_model_artifact, load_model_artifact, prepare_checkpoint,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

const CLASSIFIER_SHA256: &str = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0";
const REGRESSOR_SHA256: &str = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a";
const ORACLE_INPUT_SHA256: &str = "dfaa40fda773a7de5c1216ca0051e447b05f16e4fafa315624c5e4d5b084a0e1";
const ORACLE_LOGITS_SHA256: &str = "be53800fb2262a0c4a5d2179514c5fe9cfbaf0dc3baf7269c780a8de93ccd2c1";

#[test]
fn inspects_official_classifier_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-classifier-v2-20260212.ckpt") else {
        return;
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
fn inspects_official_regressor_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-regressor-v2-20260212.ckpt") else {
        return;
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
fn imports_official_classifier_and_reuses_automatic_cache() {
    let Some(checkpoint) = fixture("tabicl-classifier-v2-20260212.ckpt") else {
        return;
    };
    let directory = tempdir().expect("temporary cache directory");
    let explicit_model = directory.path().join("classifier.ironicl");
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
fn loads_official_classifier_artifact_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-classifier-v2.ironicl") else {
        return;
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
fn runs_official_classifier_artifact_when_fixture_is_available() {
    let Some(path) = fixture("tabicl-classifier-v2.ironicl") else {
        return;
    };
    let directory = tempdir().expect("temporary input directory");
    let input_path = directory.path().join("request.irinput");
    let request = ClassifierInputBundle::new(1, 3, 1, 2, vec![0.0, 1.0, -1.0], vec![0.0, 1.0])
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
fn matches_python_classifier_oracle_when_fixtures_are_available() {
    let Some(oracle) = oracle_fixtures() else {
        return;
    };
    let Some(model_path) = fixture("tabicl-classifier-v2.ironicl") else {
        return;
    };

    let input_bytes = fs::read(&oracle.input).expect("read Python oracle input");
    assert_eq!(sha256_hex(&input_bytes), ORACLE_INPUT_SHA256);
    let input = ClassifierInputBundle::from_bytes(&input_bytes).expect("validate Python oracle input");

    let logits_bytes = fs::read(&oracle.logits).expect("read Python oracle logits");
    assert_eq!(sha256_hex(&logits_bytes), ORACLE_LOGITS_SHA256);
    let expected = read_irlogits(&logits_bytes).expect("validate Python oracle logits");

    let actual =
        ClassifierExecutor::new(load_model_artifact(&model_path).expect("load official classifier artifact"))
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
            "IronICL logit {index} is not finite: {actual}"
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
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]
fn native_static_context_matches_uncached_executor() {
    let Some(model_path) = fixture("tabicl-classifier-v2.ironicl") else {
        return;
    };
    let executor =
        ClassifierExecutor::new(load_model_artifact(&model_path).expect("load official classifier"));
    let (b, n, q, h, c) = (2_usize, 40_usize, 6_usize, 5_usize, 3_usize);
    let t = n + q;
    let mut state = 12_345_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1_u32 << 24) as f32 * 4.0 - 2.0
    };
    let x = (0..b * t * h).map(|_| next()).collect::<Vec<_>>();
    let y = (0..b * n).map(|index| (index % c) as f32).collect::<Vec<_>>();
    let bundle = ClassifierInputBundle::new(b, t, h, n, x.clone(), y.clone()).expect("synthetic bundle");
    let expected = executor
        .predict_logits(bundle.request())
        .expect("uncached logits");
    assert_eq!(expected.shape(), [b, q, c]);

    let class_permutations = (0..b).flat_map(|_| 0..c as u64).collect::<Vec<_>>();
    let feature_permutations = (0..b).flat_map(|_| 0..h as u64).collect::<Vec<_>>();
    let request = StaticContextRequest::new(
        b,
        n,
        t,
        h,
        &x,
        &y,
        &class_permutations,
        &feature_permutations,
        [7; 32],
    )
    .expect("static context request");
    let mut queries = Vec::with_capacity(b * q * h);
    for ensemble in 0..b {
        queries.extend_from_slice(&x[(ensemble * t + n) * h..][..q * h]);
    }
    let directory = tempdir().expect("temporary directory");
    let fp32_path = directory.path().join("fp32.ironiclctx");
    let cases = [
        ("fp32", StaticContextOptions::default(), 1e-5_f32),
        (
            "packed",
            StaticContextOptions {
                pack_keys: true,
                ..StaticContextOptions::default()
            },
            1e-5,
        ),
        (
            "indexed",
            StaticContextOptions {
                sparse_index_clusters: Some(8),
                ..StaticContextOptions::default()
            },
            1e-5,
        ),
        (
            "bf16",
            StaticContextOptions {
                kv_dtype: KvDtype::Bf16,
                ..StaticContextOptions::default()
            },
            5e-2,
        ),
    ];
    for (name, options, tolerance) in cases {
        let path = if name == "fp32" {
            fp32_path.clone()
        } else {
            directory.path().join(format!("{name}.ironiclctx"))
        };
        let summary = build_static_context_file(&executor, &request, &options, &path).expect("build context");
        assert_eq!(
            summary.file_bytes,
            fs::metadata(&path).expect("context file").len()
        );
        let context = StaticContextArtifact::read_mapped(&path).expect("map built context");
        assert_eq!(context.metadata().training_rows, n);
        assert_eq!(context.has_packed_icl_keys(), options.pack_keys);
        let cached = StaticClassifierExecutor::new(
            load_model_artifact(&model_path).expect("reload official classifier"),
            context,
        )
        .expect("cached executor");
        let actual = cached
            .predict_logits(ClassifierQueryRequest::new(b, q, h, &queries).expect("query request"))
            .expect("cached logits");
        assert_eq!(actual.shape(), expected.shape());
        let mut max_abs = 0.0_f32;
        for (index, (actual, expected)) in actual.values().iter().zip(expected.values()).enumerate() {
            let abs = (actual - expected).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= tolerance + tolerance * expected.abs(),
                "{name}: logit {index} differs: actual={actual}, expected={expected}, abs={abs}"
            );
        }
        println!("native static context ({name}): max_abs={max_abs:.3e}");
        if name == "bf16" {
            for (actual, expected) in actual.values().chunks(c).zip(expected.values().chunks(c)) {
                let argmax = |row: &[f32]| {
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(index, _)| index)
                };
                assert_eq!(argmax(actual), argmax(expected));
            }
        }
    }
    let owned = build_static_context(&executor, &request).expect("in-memory context");
    assert_eq!(
        owned,
        StaticContextArtifact::read(&fp32_path).expect("read fp32 context")
    );
}

#[test]
fn no_cache_validates_without_writing() {
    let Some(checkpoint) = fixture("tabicl-classifier-v2-20260212.ckpt") else {
        return;
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
    let directory = std::env::var_os("IRONICL_CHECKPOINT_DIR")?;
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
    let directory = PathBuf::from(std::env::var_os("IRONICL_CHECKPOINT_DIR")?).join("oracle");
    let fixtures = OracleFixtures {
        input: directory.join("input.irinput"),
        logits: directory.join("logits.irlogits"),
    };
    (fixtures.input.is_file() && fixtures.logits.is_file()).then_some(fixtures)
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

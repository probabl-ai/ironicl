use std::{collections::BTreeMap, fmt, fs, path::Path, sync::Arc};

use crate::{
    TaskKind,
    error::{Error, Result},
    format::{
        artifact_error,
        container::{
            self, DISK_ALIGNMENT, DType, Entry, Header, Kind, MAX_TENSOR_COUNT, MEMORY_ALIGNMENT, Writer,
            decode_le,
        },
        encode_sha256, u64_to_usize,
    },
    models::kumo_tabular::Release,
};

pub(crate) use container::DType as ModelTensorDType;
pub use container::{MAGIC as MODEL_ARTIFACT_MAGIC, VERSION as MODEL_ARTIFACT_VERSION};

/// Kumo-Tabular large is about 860 MB.
const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_TENSOR_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOTAL_TENSOR_BYTES: u64 = 1000 * 1024 * 1024;
const CLASSIFIER_SHA256: [u8; 32] = [
    0xbd, 0xc7, 0xdb, 0xd5, 0xe4, 0xff, 0x21, 0xf8, 0xf0, 0x45, 0x6f, 0xcf, 0x90, 0xc6, 0xb7, 0xcd, 0xf7,
    0x2d, 0xbe, 0xa9, 0x60, 0xf2, 0xd0, 0x5b, 0x19, 0xbe, 0xc1, 0x9f, 0x9b, 0x3d, 0x4e, 0xd0,
];
const REGRESSOR_SHA256: [u8; 32] = [
    0x0d, 0xb9, 0xcb, 0x53, 0x8f, 0x11, 0x4e, 0x79, 0x02, 0x6b, 0xf0, 0x8f, 0x45, 0xf4, 0x1a, 0xd8, 0xdd,
    0x7a, 0xd2, 0xde, 0x2a, 0xac, 0xa9, 0xa5, 0xca, 0x8c, 0xd3, 0xbd, 0x97, 0x48, 0xae, 0x7a,
];
/// A released model identity or an otherwise valid custom model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelProfile {
    TabIclV2Classifier,
    TabIclV2Regressor,
    KumoTabular(Release),
    Unknown,
}

impl fmt::Display for ModelProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TabIclV2Classifier => f.write_str("TabICLv2 classifier"),
            Self::TabIclV2Regressor => f.write_str("TabICLv2 regressor"),
            Self::KumoTabular(release) => release.fmt(f),
            Self::Unknown => f.write_str("unknown/custom"),
        }
    }
}

/// A model's architecture and its values, as the artifact header stores them.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelConfig {
    #[serde(rename = "tabicl_v2")]
    TabIclV2(TabIclV2Config),
    /// A released checkpoint, whose size fixes every value.
    KumoTabular(Release),
}

impl ModelConfig {
    #[must_use]
    pub fn task(&self) -> Option<TaskKind> {
        match self {
            Self::TabIclV2(config) => config.task(),
            Self::KumoTabular(release) => Some(release.task),
        }
    }
}

/// `TabICLv2`'s architecture values.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TabIclV2Config {
    pub max_classes: Option<u64>,
    pub num_quantiles: Option<u64>,
    pub embed_dim: Option<u64>,
    pub column_blocks: Option<u64>,
    pub row_blocks: Option<u64>,
    pub row_cls_tokens: Option<u64>,
    pub icl_blocks: Option<u64>,
}

impl TabIclV2Config {
    #[must_use]
    pub fn task(&self) -> Option<TaskKind> {
        self.max_classes.map(|classes| {
            if classes == 0 {
                TaskKind::Regressor
            } else {
                TaskKind::Classifier
            }
        })
    }

    fn required_values(&self) -> Result<[u64; 7]> {
        Ok([
            required_config(self.max_classes, "max_classes")?,
            required_config(self.num_quantiles, "num_quantiles")?,
            required_config(self.embed_dim, "embed_dim")?,
            required_config(self.column_blocks, "column_blocks")?,
            required_config(self.row_blocks, "row_blocks")?,
            required_config(self.row_cls_tokens, "row_cls_tokens")?,
            required_config(self.icl_blocks, "icl_blocks")?,
        ])
    }

    fn validate(&self) -> Result<()> {
        let values = self.required_values()?;
        if values[1..].contains(&0) {
            return Err(Error::Artifact(
                "model configuration dimensions must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelTensor {
    pub name: String,
    pub dtype: ModelTensorDType,
    pub dimensions: Vec<u64>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelArtifactData {
    pub source_sha256: [u8; 32],
    pub config: ModelConfig,
    pub tensors: Vec<ModelTensor>,
}

/// Verified metadata for a portable Ironicl model artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactSummary {
    pub source_sha256: String,
    pub profile: ModelProfile,
    pub tensor_count: usize,
    pub parameter_count: u64,
    pub tensor_bytes: u64,
    pub dtype_counts: BTreeMap<String, usize>,
    pub config: ModelConfig,
}

/// A logical FP32 tensor backed by owned values or an immutable mapping.
#[derive(Debug, Clone)]
pub struct ModelTensorF32 {
    dimensions: Vec<u64>,
    storage: ModelStorage,
}

#[derive(Debug, Clone)]
enum ModelStorage {
    Owned(Vec<f32>),
    Mapped(Arc<memmap2::Mmap>, std::ops::Range<usize>),
}
impl PartialEq for ModelTensorF32 {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.values() == other.values()
    }
}

impl ModelTensorF32 {
    /// Logical tensor dimensions in checkpoint order.
    #[must_use]
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    /// Tensor values decoded from the artifact's little-endian FP32 payload.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        match &self.storage {
            ModelStorage::Owned(v) => v,
            ModelStorage::Mapped(m, r) => bytemuck::cast_slice(&m[r.clone()]),
        }
    }
}

/// An owned, executable official `TabICLv2` classifier or regressor model.
///
/// The loader validates the complete official tensor manifest before it returns
/// this type. File loads retain the immutable mapping on little-endian hosts.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelArtifact {
    summary: ModelArtifactSummary,
    task: TaskKind,
    tensors: BTreeMap<String, ModelTensorF32>,
}

impl ModelArtifact {
    /// Read, verify, and prepare an official `TabICLv2` model artifact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the file is invalid, is not an
    /// official model, or does not have its exact tensor manifest.
    ///
    /// The file must remain immutable while loaded. Publish changes by rename.
    pub fn load(path: &Path) -> Result<Self> {
        // WebAssembly filesystems need not support file-backed mmap.
        #[cfg(target_family = "wasm")]
        {
            let bytes = crate::format::read_bounded(path, MAX_ARTIFACT_BYTES).map_err(artifact_error)?;
            Self::from_bytes(&bytes)
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let mapping = Arc::new(container::map_immutable(path, MAX_ARTIFACT_BYTES)?);
            Self::decode(&mapping, Some(&mapping))
        }
    }

    /// Verify and prepare bytes from an official `TabICLv2` model artifact.
    ///
    /// This owns decoded tensor storage, so the returned model does not borrow
    /// the artifact bytes. The regressor's bias-free layer norms get zero
    /// biases, so both models run the same layer-norm code.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the bytes are invalid or do not match
    /// the official manifest.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::decode(bytes, None)
    }

    fn decode(bytes: &[u8], mapping: Option<&Arc<memmap2::Mmap>>) -> Result<Self> {
        let (header, summary) = parse_model_artifact_bytes(bytes)?;
        let (task, manifest) = match summary.profile {
            ModelProfile::TabIclV2Classifier => (TaskKind::Classifier, tensor_manifest(TaskKind::Classifier)),
            ModelProfile::TabIclV2Regressor => (TaskKind::Regressor, tensor_manifest(TaskKind::Regressor)),
            ModelProfile::KumoTabular(release) => (release.task, release.manifest()),
            ModelProfile::Unknown => {
                return Err(Error::Artifact(
                    "executable model must be an official TabICLv2 or Kumo-Tabular checkpoint".into(),
                ));
            }
        };
        if !matches!(summary.profile, ModelProfile::KumoTabular(_)) {
            validate_config(&summary.config, task)?;
        }
        validate_manifest(&header.tensors, &manifest, task)?;

        let mut tensors = BTreeMap::new();
        for entry in &header.tensors {
            let range = header.range(entry);
            let payload = &bytes[range.clone()];
            if !entry.dtype.all_finite(payload)? {
                return Err(artifact_error(format!(
                    "tensor '{}' contains a non-finite value",
                    entry.name
                )));
            }
            let storage = if let Some(mapping) = mapping.filter(|_| cfg!(target_endian = "little")) {
                bytemuck::try_cast_slice::<u8, f32>(payload).map_err(artifact_error)?;
                ModelStorage::Mapped(Arc::clone(mapping), range)
            } else {
                ModelStorage::Owned(decode_le(payload, f32::from_le_bytes))
            };
            tensors.insert(
                entry.name.clone(),
                ModelTensorF32 {
                    dimensions: entry.shape.clone(),
                    storage,
                },
            );
        }
        if summary.profile == ModelProfile::TabIclV2Regressor {
            for (name, dimensions) in tensor_manifest(TaskKind::Classifier) {
                if let std::collections::btree_map::Entry::Vacant(entry) = tensors.entry(name) {
                    let values = vec![0.0; u64_to_usize(dimensions.iter().product(), "layer-norm bias")?];
                    entry.insert(ModelTensorF32 {
                        dimensions,
                        storage: ModelStorage::Owned(values),
                    });
                }
            }
        }
        Ok(Self {
            summary,
            task,
            tensors,
        })
    }

    /// Whether this is the classifier or the regressor.
    #[must_use]
    pub fn task(&self) -> TaskKind {
        self.task
    }

    /// Verified model metadata.
    #[must_use]
    pub fn summary(&self) -> &ModelArtifactSummary {
        &self.summary
    }

    /// Get a validated official tensor by its exact checkpoint name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when `name` is not in the official manifest.
    pub fn tensor_f32(&self, name: &str) -> Result<&ModelTensorF32> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Artifact(format!("official model tensor '{name}' is missing")))
    }
}

/// Load an executable official `TabICLv2` model artifact.
///
/// # Errors
///
/// Returns [`Error::Artifact`] when the artifact cannot be used for execution.
pub fn load_model_artifact(path: &Path) -> Result<ModelArtifact> {
    ModelArtifact::load(path)
}

/// Read and verify a portable Ironicl model artifact.
///
/// # Errors
///
/// Returns [`Error::Artifact`] when the file is malformed, too large, has an
/// unsupported version, or contains inconsistent tensor metadata.
pub fn inspect_model_artifact(path: &Path) -> Result<ModelArtifactSummary> {
    summarize_header(&Header::open(path, Kind::Model, MAX_ARTIFACT_BYTES)?)
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    #[serde(with = "container::digest")]
    source_sha256: [u8; 32],
    config: ModelConfig,
}

fn writer(model: &ModelArtifactData, alignment: usize) -> Result<Writer<'_>> {
    if model.source_sha256 == [0; 32] || model.tensors.is_empty() || model.tensors.len() > MAX_TENSOR_COUNT {
        return Err(artifact_error("invalid model identity or tensor count"));
    }
    if let ModelConfig::TabIclV2(config) = &model.config {
        config.validate()?;
    }
    let mut tensors: Vec<_> = model.tensors.iter().collect();
    tensors.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    let metadata = Metadata {
        source_sha256: model.source_sha256,
        config: model.config.clone(),
    };
    let mut total = 0_u64;
    for t in &tensors {
        total = total
            .checked_add(t.data.len() as u64)
            .ok_or_else(|| artifact_error("model size overflow"))?;
        if t.data.len() as u64 > MAX_TENSOR_BYTES || total > MAX_TOTAL_TENSOR_BYTES {
            return Err(artifact_error("model tensor size exceeds limit"));
        }
    }
    let writer = Writer::new(
        Kind::Model,
        alignment,
        metadata,
        tensors
            .into_iter()
            .map(|t| (t.name.clone(), t.dtype, t.dimensions.clone(), t.data.as_slice())),
    )?;
    if writer.len as u64 > MAX_ARTIFACT_BYTES {
        return Err(artifact_error("model exceeds size limit"));
    }
    Ok(writer)
}

pub(crate) fn write_model_artifact(path: &Path, model: &ModelArtifactData) -> Result<ModelArtifactSummary> {
    let parent = super::parent_dir(path);
    fs::create_dir_all(parent).map_err(artifact_error)?;
    let actual = super::write_atomic(path, |file| {
        writer(model, DISK_ALIGNMENT)?.write(file)?;
        inspect_model_artifact(file.path())
    })?;
    #[cfg(unix)]
    fs::File::open(parent)
        .and_then(|dir| dir.sync_all())
        .map_err(artifact_error)?;
    Ok(actual)
}

pub(crate) fn encode_model_artifact(model: &ModelArtifactData) -> Result<Vec<u8>> {
    writer(model, MEMORY_ALIGNMENT)?.bytes()
}

fn parse_model_artifact_bytes(bytes: &[u8]) -> Result<(Header<Metadata>, ModelArtifactSummary)> {
    if bytes.len() as u64 > MAX_ARTIFACT_BYTES {
        return Err(artifact_error("model exceeds size limit"));
    }
    let header = Header::from_bytes(bytes, Kind::Model)?;
    let summary = summarize_header(&header)?;
    Ok((header, summary))
}

fn summarize_header(header: &Header<Metadata>) -> Result<ModelArtifactSummary> {
    let tensor_count = header.tensors.len();
    let Metadata {
        source_sha256,
        config,
    } = &header.metadata;
    if *source_sha256 == [0; 32] || tensor_count == 0 {
        return Err(artifact_error("invalid model identity or tensor count"));
    }
    if let ModelConfig::TabIclV2(config) = config {
        config.validate()?;
    }
    let mut parameter_count = 0_u64;
    let mut tensor_bytes = 0_u64;
    let mut dtype_counts = BTreeMap::new();
    for t in &header.tensors {
        if t.bytes as u64 > MAX_TENSOR_BYTES || matches!(t.dtype, DType::I32 | DType::F8E4M3) {
            return Err(artifact_error("invalid model tensor size or dtype"));
        }
        parameter_count = parameter_count
            .checked_add(t.bytes as u64 / t.dtype.size_in_bytes() as u64)
            .ok_or_else(|| artifact_error("parameter count overflow"))?;
        tensor_bytes += t.bytes as u64;
        *dtype_counts.entry(t.dtype.as_str().to_owned()).or_insert(0) += 1;
    }
    if tensor_bytes > MAX_TOTAL_TENSOR_BYTES {
        return Err(artifact_error("model payload exceeds size limit"));
    }
    let profile = identify_profile(source_sha256, config, tensor_count, parameter_count, tensor_bytes);
    Ok(ModelArtifactSummary {
        source_sha256: encode_sha256(source_sha256),
        profile,
        tensor_count,
        parameter_count,
        tensor_bytes,
        dtype_counts,
        config: config.clone(),
    })
}

fn validate_manifest(tensors: &[Entry], expected: &BTreeMap<String, Vec<u64>>, task: TaskKind) -> Result<()> {
    if tensors.len() != expected.len() {
        return Err(Error::Artifact(format!(
            "official {task} has {} tensors; expected {}",
            tensors.len(),
            expected.len()
        )));
    }
    for tensor in tensors {
        let Some(dimensions) = expected.get(&tensor.name) else {
            return Err(Error::Artifact(format!(
                "unexpected official {task} tensor '{}'",
                tensor.name
            )));
        };
        if tensor.dtype != DType::F32 {
            return Err(Error::Artifact(format!(
                "official {task} tensor '{}' has dtype {}; expected f32",
                tensor.name,
                tensor.dtype.as_str()
            )));
        }
        if tensor.shape != *dimensions {
            return Err(Error::Artifact(format!(
                "official {task} tensor '{}' has shape {:?}; expected {:?}",
                tensor.name, tensor.shape, dimensions
            )));
        }
    }
    Ok(())
}

fn validate_config(config: &ModelConfig, task: TaskKind) -> Result<()> {
    let ModelConfig::TabIclV2(config) = config else {
        return Err(Error::Artifact("model is not TabICLv2".into()));
    };
    let actual = config.required_values()?;
    let max_classes = match task {
        TaskKind::Classifier => 10,
        TaskKind::Regressor => 0,
    };
    let expected = [max_classes, 999, 128, 3, 3, 4, 12];
    if actual != expected {
        return Err(Error::Artifact(format!(
            "official {task} configuration is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

/// The official tensor names and shapes. The regressor encodes a scalar
/// target, decodes 999 quantiles, and has no layer-norm biases.
fn tensor_manifest(task: TaskKind) -> BTreeMap<String, Vec<u64>> {
    let (targets, outputs) = match task {
        TaskKind::Classifier => (10, 10),
        TaskKind::Regressor => (1, 999),
    };
    let mut tensors = BTreeMap::new();
    let mut insert = |name: String, dimensions: &[u64]| {
        let layer_norm_bias =
            name.ends_with("norm1.bias") || name.ends_with("norm2.bias") || name.ends_with("ln.bias");
        if task == TaskKind::Regressor && layer_norm_bias {
            return;
        }
        assert!(tensors.insert(name, dimensions.to_vec()).is_none());
    };
    insert("col_embedder.in_linear.bias".into(), &[128]);
    insert("col_embedder.in_linear.weight".into(), &[128, 3]);
    insert("col_embedder.y_encoder.bias".into(), &[128]);
    insert("col_embedder.y_encoder.weight".into(), &[128, targets]);
    for block in 0..3 {
        let prefix = format!("col_embedder.tf_col.blocks.{block}");
        insert(format!("{prefix}.ind_vectors"), &[128, 128]);
        insert_attention_manifest(&mut insert, &format!("{prefix}.multihead_attn1"), 128, true);
        insert_attention_manifest(&mut insert, &format!("{prefix}.multihead_attn2"), 128, false);
    }

    insert("icl_predictor.decoder.0.bias".into(), &[1024]);
    insert("icl_predictor.decoder.0.weight".into(), &[1024, 512]);
    insert("icl_predictor.decoder.2.bias".into(), &[outputs]);
    insert("icl_predictor.decoder.2.weight".into(), &[outputs, 1024]);
    insert("icl_predictor.ln.bias".into(), &[512]);
    insert("icl_predictor.ln.weight".into(), &[512]);
    insert("icl_predictor.y_encoder.bias".into(), &[512]);
    insert("icl_predictor.y_encoder.weight".into(), &[512, targets]);
    for block in 0..12 {
        insert_attention_manifest(
            &mut insert,
            &format!("icl_predictor.tf_icl.blocks.{block}"),
            512,
            true,
        );
    }

    insert("row_interactor.cls_tokens".into(), &[4, 128]);
    insert("row_interactor.out_ln.bias".into(), &[128]);
    insert("row_interactor.out_ln.weight".into(), &[128]);
    insert("row_interactor.tf_row.rope.freqs".into(), &[8]);
    for block in 0..3 {
        insert_attention_manifest(
            &mut insert,
            &format!("row_interactor.tf_row.blocks.{block}"),
            128,
            false,
        );
    }
    debug_assert_eq!(
        tensors.len(),
        match task {
            TaskKind::Classifier => 391,
            TaskKind::Regressor => 347,
        }
    );
    tensors
}

fn insert_attention_manifest(
    insert: &mut impl FnMut(String, &[u64]),
    prefix: &str,
    embed_dim: u64,
    has_ssmax: bool,
) {
    let attn = format!("{prefix}.attn");
    insert(format!("{attn}.in_proj_bias"), &[3 * embed_dim]);
    insert(format!("{attn}.in_proj_weight"), &[3 * embed_dim, embed_dim]);
    insert(format!("{attn}.out_proj.bias"), &[embed_dim]);
    insert(format!("{attn}.out_proj.weight"), &[embed_dim, embed_dim]);
    if has_ssmax {
        let query_dim = embed_dim / 8;
        for (layer, dimensions) in [
            ("base_mlp.0", vec![64, 1]),
            ("base_mlp.2", vec![embed_dim, 64]),
            ("query_mlp.0", vec![64, query_dim]),
            ("query_mlp.2", vec![query_dim, 64]),
        ] {
            insert(format!("{attn}.ssmax_layer.{layer}.bias"), &[dimensions[0]]);
            insert(format!("{attn}.ssmax_layer.{layer}.weight"), &dimensions);
        }
    }
    insert(format!("{prefix}.linear1.bias"), &[2 * embed_dim]);
    insert(format!("{prefix}.linear1.weight"), &[2 * embed_dim, embed_dim]);
    insert(format!("{prefix}.linear2.bias"), &[embed_dim]);
    insert(format!("{prefix}.linear2.weight"), &[embed_dim, 2 * embed_dim]);
    insert(format!("{prefix}.norm1.bias"), &[embed_dim]);
    insert(format!("{prefix}.norm1.weight"), &[embed_dim]);
    insert(format!("{prefix}.norm2.bias"), &[embed_dim]);
    insert(format!("{prefix}.norm2.weight"), &[embed_dim]);
}

fn identify_profile(
    source_sha256: &[u8; 32],
    config: &ModelConfig,
    tensor_count: usize,
    parameter_count: u64,
    tensor_bytes: u64,
) -> ModelProfile {
    if let ModelConfig::KumoTabular(release) = config {
        let manifest = release.manifest();
        let elements: u64 = manifest
            .values()
            .map(|dimensions| dimensions.iter().product::<u64>())
            .sum();
        // Any weights of the release's exact tensors, NVIDIA's or fine-tuned.
        let complete =
            (tensor_count, parameter_count, tensor_bytes) == (manifest.len(), elements, 4 * elements);
        return if complete {
            ModelProfile::KumoTabular(*release)
        } else {
            ModelProfile::Unknown
        };
    }
    match (
        source_sha256,
        config.task(),
        tensor_count,
        parameter_count,
        tensor_bytes,
    ) {
        (&CLASSIFIER_SHA256, Some(TaskKind::Classifier), 391, 27_552_258, 110_209_032) => {
            ModelProfile::TabIclV2Classifier
        }
        (&REGRESSOR_SHA256, Some(TaskKind::Regressor), 347, 28_544_991, 114_179_964) => {
            ModelProfile::TabIclV2Regressor
        }
        _ => ModelProfile::Unknown,
    }
}

fn required_config(value: Option<u64>, name: &str) -> Result<u64> {
    value.ok_or_else(|| Error::Artifact(format!("model configuration '{name}' is missing")))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn model_artifact_round_trips_and_detects_corruption() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("model.ir");
        let model = fixture_model();

        let written = write_model_artifact(&path, &model).expect("write model artifact");
        let inspected = inspect_model_artifact(&path).expect("inspect model artifact");
        assert_eq!(written, inspected);
        assert_eq!(inspected.tensor_count, 2);
        assert_eq!(inspected.parameter_count, 6);
        assert_eq!(inspected.tensor_bytes, 24);

        let mut bytes = fs::read(&path).expect("read artifact");
        bytes[16] ^= 1;
        fs::write(&path, bytes).expect("corrupt artifact");
        assert!(inspect_model_artifact(&path).is_err());
    }

    #[test]
    fn parsed_tensor_ranges_include_the_payload_section_offset() {
        let model = fixture_model();
        let bytes = encode_model_artifact(&model).expect("encode fixture");
        let (header, _) = parse_model_artifact_bytes(&bytes).expect("parse fixture");
        let tensor = header
            .tensors
            .iter()
            .find(|tensor| tensor.name == "a.weight")
            .expect("fixture tensor");
        let range = header.range(tensor);

        assert_eq!(&bytes[range.clone()], &[0; 16]);
        assert!(range.start >= MEMORY_ALIGNMENT);
    }

    #[test]
    fn manifests_have_all_official_tensors() {
        let manifest = tensor_manifest(TaskKind::Classifier);
        assert_eq!(manifest.len(), 391);
        assert_eq!(
            manifest.get("icl_predictor.tf_icl.blocks.11.linear2.weight"),
            Some(&vec![512, 1024])
        );
        assert_eq!(manifest.get("row_interactor.tf_row.rope.freqs"), Some(&vec![8]));

        let manifest = tensor_manifest(TaskKind::Regressor);
        assert_eq!(manifest.len(), 347);
        assert_eq!(
            manifest.get("icl_predictor.decoder.2.weight"),
            Some(&vec![999, 1024])
        );
        assert_eq!(manifest.get("col_embedder.y_encoder.weight"), Some(&vec![128, 1]));
        assert!(!manifest.contains_key("icl_predictor.ln.bias"));
        assert!(!manifest.contains_key("row_interactor.tf_row.blocks.0.norm1.bias"));
    }

    #[test]
    fn manifest_rejects_wrong_shape_and_unexpected_tensor() {
        let manifest = tensor_manifest(TaskKind::Classifier);
        let mut tensors = manifest
            .into_iter()
            .map(|(name, shape)| Entry {
                name,
                dtype: DType::F32,
                shape,
                offset: 0,
                bytes: 0,
            })
            .collect::<Vec<_>>();
        tensors[0].shape = vec![1];
        assert!(
            validate_manifest(
                &tensors,
                &tensor_manifest(TaskKind::Classifier),
                TaskKind::Classifier
            )
            .is_err()
        );

        tensors[0].shape = tensor_manifest(TaskKind::Classifier)
            .get(&tensors[0].name)
            .expect("known tensor")
            .clone();
        tensors[0].name = "unexpected.tensor".into();
        assert!(
            validate_manifest(
                &tensors,
                &tensor_manifest(TaskKind::Classifier),
                TaskKind::Classifier
            )
            .is_err()
        );

        tensors[0].name = tensor_manifest(TaskKind::Classifier)
            .keys()
            .next()
            .expect("manifest has tensors")
            .clone();
        tensors[0].dtype = DType::F64;
        assert!(
            validate_manifest(
                &tensors,
                &tensor_manifest(TaskKind::Classifier),
                TaskKind::Classifier
            )
            .is_err()
        );
    }

    #[test]
    fn configuration_must_match_the_release() {
        let ModelConfig::TabIclV2(mut config) = fixture_model().config else {
            unreachable!("the fixture is TabICLv2")
        };
        config.num_quantiles = Some(998);
        let check =
            |config: &TabIclV2Config, task| validate_config(&ModelConfig::TabIclV2(config.clone()), task);
        assert!(check(&config, TaskKind::Classifier).is_err());
        config.num_quantiles = Some(999);
        assert!(check(&config, TaskKind::Classifier).is_ok());
        assert!(check(&config, TaskKind::Regressor).is_err());
        let kumo = ModelConfig::KumoTabular(Release::ALL[0]);
        assert!(validate_config(&kumo, TaskKind::Classifier).is_err());
    }

    #[test]
    fn kumo_tabular_configs_round_trip_through_the_header() {
        for release in Release::ALL {
            let config = ModelConfig::KumoTabular(release);
            let mut data = fixture_model();
            data.config = config.clone();
            let bytes = encode_model_artifact(&data).unwrap();
            assert_eq!(parse_model_artifact_bytes(&bytes).unwrap().1.config, config);
        }
    }

    fn fixture_model() -> ModelArtifactData {
        ModelArtifactData {
            source_sha256: [1; 32],
            config: ModelConfig::TabIclV2(TabIclV2Config {
                max_classes: Some(10),
                num_quantiles: Some(999),
                embed_dim: Some(128),
                column_blocks: Some(3),
                row_blocks: Some(3),
                row_cls_tokens: Some(4),
                icl_blocks: Some(12),
            }),
            tensors: vec![
                ModelTensor {
                    name: "b.bias".into(),
                    dtype: ModelTensorDType::F32,
                    dimensions: vec![2],
                    data: vec![0; 8],
                },
                ModelTensor {
                    name: "a.weight".into(),
                    dtype: ModelTensorDType::F32,
                    dimensions: vec![2, 2],
                    data: vec![0; 16],
                },
            ],
        }
    }
}

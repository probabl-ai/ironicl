use std::{collections::BTreeMap, fmt, fs::File, io::Read, path::Path};

#[cfg(feature = "checkpoint")]
use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use sha2::{Digest, Sha256};

use crate::{
    TaskKind,
    error::{Error, Result},
};

pub const MODEL_ARTIFACT_MAGIC: [u8; 8] = *b"IRMODEL\0";
pub const MODEL_ARTIFACT_VERSION: u32 = 1;

const MODEL_ARTIFACT_KIND: u32 = 1;
const HEADER_BYTES: usize = 144;
const CHECKSUM_BYTES: usize = 32;
const TENSOR_ALIGNMENT: usize = 64;
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DIRECTORY_BYTES: u64 = 4 * 1024 * 1024;
const MAX_TENSOR_COUNT: usize = 4096;
const MAX_TENSOR_RANK: usize = 16;
const MAX_TENSOR_NAME_BYTES: usize = 4096;
const MAX_TENSOR_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOTAL_TENSOR_BYTES: u64 = 256 * 1024 * 1024;
const CLASSIFIER_SHA256: [u8; 32] = [
    0xbd, 0xc7, 0xdb, 0xd5, 0xe4, 0xff, 0x21, 0xf8, 0xf0, 0x45, 0x6f, 0xcf, 0x90, 0xc6, 0xb7, 0xcd, 0xf7,
    0x2d, 0xbe, 0xa9, 0x60, 0xf2, 0xd0, 0x5b, 0x19, 0xbe, 0xc1, 0x9f, 0x9b, 0x3d, 0x4e, 0xd0,
];
const REGRESSOR_SHA256: [u8; 32] = [
    0x0d, 0xb9, 0xcb, 0x53, 0x8f, 0x11, 0x4e, 0x79, 0x02, 0x6b, 0xf0, 0x8f, 0x45, 0xf4, 0x1a, 0xd8, 0xdd,
    0x7a, 0xd2, 0xde, 0x2a, 0xac, 0xa9, 0xa5, 0xca, 0x8c, 0xd3, 0xbd, 0x97, 0x48, 0xae, 0x7a,
];
const HEX: &[u8; 16] = b"0123456789abcdef";

/// A released model identity or an otherwise valid custom model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelProfile {
    TabIclV2Classifier,
    TabIclV2Regressor,
    Unknown,
}

impl fmt::Display for ModelProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TabIclV2Classifier => f.write_str("TabICLv2 classifier"),
            Self::TabIclV2Regressor => f.write_str("TabICLv2 regressor"),
            Self::Unknown => f.write_str("unknown/custom"),
        }
    }
}

/// Architecture values stored in a portable model artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub max_classes: Option<u64>,
    pub num_quantiles: Option<u64>,
    pub embed_dim: Option<u64>,
    pub column_blocks: Option<u64>,
    pub row_blocks: Option<u64>,
    pub row_cls_tokens: Option<u64>,
    pub icl_blocks: Option<u64>,
}

impl ModelConfig {
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

/// A storage dtype supported by the portable model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelTensorDType {
    U8,
    I64,
    Bf16,
    F16,
    F32,
    F64,
}

impl ModelTensorDType {
    #[cfg(feature = "checkpoint")]
    fn id(self) -> u8 {
        match self {
            Self::U8 => 1,
            Self::I64 => 2,
            Self::Bf16 => 3,
            Self::F16 => 4,
            Self::F32 => 5,
            Self::F64 => 6,
        }
    }

    fn from_id(id: u8) -> Result<Self> {
        match id {
            1 => Ok(Self::U8),
            2 => Ok(Self::I64),
            3 => Ok(Self::Bf16),
            4 => Ok(Self::F16),
            5 => Ok(Self::F32),
            6 => Ok(Self::F64),
            _ => Err(Error::Artifact(format!("unsupported tensor dtype ID {id}"))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I64 => "i64",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub(crate) fn size_in_bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(feature = "checkpoint")]
pub(crate) struct ModelTensor {
    pub name: String,
    pub dtype: ModelTensorDType,
    pub dimensions: Vec<u64>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(feature = "checkpoint")]
pub(crate) struct ModelArtifactData {
    pub source_sha256: [u8; 32],
    pub config: ModelConfig,
    pub tensors: Vec<ModelTensor>,
}

/// Verified metadata for a portable `IronICL` model artifact.
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

/// An owned FP32 tensor from a validated executable model artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelTensorF32 {
    dimensions: Vec<u64>,
    values: Vec<f32>,
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
        &self.values
    }
}

/// An owned, executable official `TabICLv2` classifier model.
///
/// The loader validates the complete official tensor manifest before it returns
/// this type. Tensor values are stored in owned, host-native FP32 vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelArtifact {
    summary: ModelArtifactSummary,
    tensors: BTreeMap<String, ModelTensorF32>,
}

impl ModelArtifact {
    /// Read, verify, and prepare the official `TabICLv2` classifier artifact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the file is invalid, is not the
    /// official classifier, or does not have its exact tensor manifest.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_tabicl_v2_classifier(path)
    }

    /// Read, verify, and prepare the official `TabICLv2` classifier artifact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the file is invalid, is not the
    /// official classifier, or does not have its exact tensor manifest.
    pub fn load_tabicl_v2_classifier(path: &Path) -> Result<Self> {
        let bytes = read_artifact_bytes(path)?;
        Self::from_tabicl_v2_classifier_bytes(&bytes)
    }

    /// Verify and prepare bytes from an official `TabICLv2` classifier artifact.
    ///
    /// This owns decoded tensor storage, so the returned model does not borrow
    /// the artifact bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the bytes are invalid or do not match
    /// the official classifier manifest.
    pub fn from_tabicl_v2_classifier_bytes(bytes: &[u8]) -> Result<Self> {
        let parsed = parse_model_artifact_bytes(bytes)?;
        if parsed.summary.profile != ModelProfile::TabIclV2Classifier {
            return Err(Error::Artifact(format!(
                "executable model must be the official TabICLv2 classifier; found {}",
                parsed.summary.profile
            )));
        }
        validate_classifier_config(&parsed.summary.config)?;
        validate_classifier_manifest(&parsed.tensors)?;

        let mut tensors = BTreeMap::new();
        for tensor in parsed.tensors {
            let payload = bytes
                .get(tensor.payload_range)
                .ok_or_else(|| Error::Artifact(format!("tensor '{}' payload is missing", tensor.name)))?;
            let values = payload
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes = chunk.try_into().map_err(|_| {
                        Error::Artifact(format!("tensor '{}' has an invalid FP32 payload", tensor.name))
                    })?;
                    Ok(f32::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<_>>>()?;
            tensors.insert(
                tensor.name,
                ModelTensorF32 {
                    dimensions: tensor.dimensions,
                    values,
                },
            );
        }
        Ok(Self {
            summary: parsed.summary,
            tensors,
        })
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
            .ok_or_else(|| Error::Artifact(format!("official classifier tensor '{name}' is missing")))
    }
}

/// Load an executable official `TabICLv2` classifier model artifact.
///
/// # Errors
///
/// Returns [`Error::Artifact`] when the artifact cannot be used for execution.
pub fn load_model_artifact(path: &Path) -> Result<ModelArtifact> {
    ModelArtifact::load(path)
}

#[derive(Debug)]
struct ParsedTensor {
    name: String,
    dtype: ModelTensorDType,
    dimensions: Vec<u64>,
    payload_range: std::ops::Range<usize>,
}

#[derive(Debug)]
struct ParsedModelArtifact {
    summary: ModelArtifactSummary,
    tensors: Vec<ParsedTensor>,
}

/// Read and verify a portable `IronICL` model artifact.
///
/// # Errors
///
/// Returns [`Error::Artifact`] when the file is malformed, too large, has an
/// invalid checksum, or contains inconsistent tensor metadata.
pub fn inspect_model_artifact(path: &Path) -> Result<ModelArtifactSummary> {
    let bytes = read_artifact_bytes(path)?;
    Ok(parse_model_artifact_bytes(&bytes)?.summary)
}

#[cfg(feature = "checkpoint")]
pub(crate) fn write_model_artifact(path: &Path, model: &ModelArtifactData) -> Result<ModelArtifactSummary> {
    let bytes = encode_model_artifact(model)?;
    let expected = parse_model_artifact_bytes(&bytes)?.summary;
    let parent = usable_parent(path);
    fs::create_dir_all(parent).map_err(artifact_error)?;
    let temporary = create_temporary_path(path)?;

    let write_result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(artifact_error)?;
        file.write_all(&bytes).map_err(artifact_error)?;
        file.sync_all().map_err(artifact_error)?;
        drop(file);
        let actual = inspect_model_artifact(&temporary)?;
        if actual != expected {
            return Err(Error::Artifact(
                "temporary artifact verification changed metadata".into(),
            ));
        }
        fs::rename(&temporary, path).map_err(artifact_error)?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(artifact_error)?;
        inspect_model_artifact(path)
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

#[cfg(feature = "checkpoint")]
fn encode_model_artifact(model: &ModelArtifactData) -> Result<Vec<u8>> {
    model.config.validate()?;
    if model.source_sha256 == [0; 32] {
        return Err(Error::Artifact("source checkpoint hash is empty".into()));
    }
    if model.tensors.is_empty() || model.tensors.len() > MAX_TENSOR_COUNT {
        return Err(Error::Artifact(format!(
            "model has {} tensors; expected 1..={MAX_TENSOR_COUNT}",
            model.tensors.len()
        )));
    }

    let mut tensors = model.tensors.iter().collect::<Vec<_>>();
    tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    let mut names = BTreeSet::new();
    let mut directory = Vec::new();
    let mut payload = Vec::new();
    let mut parameter_count = 0_u64;
    let mut tensor_bytes = 0_u64;
    for tensor in tensors {
        validate_tensor(tensor)?;
        if !names.insert(tensor.name.as_str()) {
            return Err(Error::Artifact(format!(
                "duplicate tensor name '{}'",
                tensor.name
            )));
        }
        pad_to_alignment(&mut payload, TENSOR_ALIGNMENT)?;
        let payload_offset = usize_to_u64(payload.len(), "tensor payload offset")?;
        let byte_len = usize_to_u64(tensor.data.len(), "tensor byte length")?;
        write_u32(
            &mut directory,
            usize_to_u32(tensor.name.len(), "tensor name length")?,
        );
        write_u16(
            &mut directory,
            u16::try_from(tensor.dimensions.len())
                .map_err(|_| Error::Artifact("tensor rank exceeds u16".into()))?,
        );
        directory.push(tensor.dtype.id());
        directory.push(0);
        write_u64(&mut directory, payload_offset);
        write_u64(&mut directory, byte_len);
        for dimension in &tensor.dimensions {
            write_u64(&mut directory, *dimension);
        }
        directory.extend_from_slice(tensor.name.as_bytes());
        payload.extend_from_slice(&tensor.data);

        let elements = checked_elements(&tensor.name, &tensor.dimensions)?;
        parameter_count = parameter_count
            .checked_add(elements)
            .ok_or_else(|| Error::Artifact("model parameter count overflows u64".into()))?;
        tensor_bytes = tensor_bytes
            .checked_add(byte_len)
            .ok_or_else(|| Error::Artifact("model tensor byte count overflows u64".into()))?;
    }
    if tensor_bytes > MAX_TOTAL_TENSOR_BYTES {
        return Err(Error::Artifact(format!(
            "model tensors are {tensor_bytes} bytes; limit is {MAX_TOTAL_TENSOR_BYTES}"
        )));
    }
    if usize_to_u64(directory.len(), "model directory length")? > MAX_DIRECTORY_BYTES {
        return Err(Error::Artifact("model tensor directory is too large".into()));
    }

    let mut bytes = Vec::with_capacity(
        HEADER_BYTES
            .checked_add(directory.len())
            .and_then(|size| size.checked_add(TENSOR_ALIGNMENT))
            .and_then(|size| size.checked_add(payload.len()))
            .and_then(|size| size.checked_add(CHECKSUM_BYTES))
            .ok_or_else(|| Error::Artifact("artifact size overflows usize".into()))?,
    );
    bytes.extend_from_slice(&MODEL_ARTIFACT_MAGIC);
    write_u32(&mut bytes, MODEL_ARTIFACT_VERSION);
    write_u32(&mut bytes, MODEL_ARTIFACT_KIND);
    bytes.extend_from_slice(&model.source_sha256);
    write_u32(&mut bytes, usize_to_u32(model.tensors.len(), "tensor count")?);
    write_u64(&mut bytes, parameter_count);
    write_u64(&mut bytes, tensor_bytes);
    for value in model.config.required_values()? {
        write_u64(&mut bytes, value);
    }
    write_u64(&mut bytes, usize_to_u64(directory.len(), "directory length")?);
    write_u64(&mut bytes, usize_to_u64(payload.len(), "payload length")?);
    write_u32(&mut bytes, 0);
    if bytes.len() != HEADER_BYTES {
        return Err(Error::Artifact("internal artifact header size mismatch".into()));
    }
    bytes.extend_from_slice(&directory);
    pad_to_alignment(&mut bytes, TENSOR_ALIGNMENT)?;
    bytes.extend_from_slice(&payload);
    let checksum = sha256(&bytes);
    bytes.extend_from_slice(&checksum);
    if usize_to_u64(bytes.len(), "artifact length")? > MAX_ARTIFACT_BYTES {
        return Err(Error::Artifact(format!(
            "artifact is {} bytes; limit is {MAX_ARTIFACT_BYTES}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

#[allow(clippy::too_many_lines)]
fn parse_model_artifact_bytes(bytes: &[u8]) -> Result<ParsedModelArtifact> {
    if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
        return Err(Error::Artifact("artifact is truncated".into()));
    }
    if usize_to_u64(bytes.len(), "artifact length")? > MAX_ARTIFACT_BYTES {
        return Err(Error::Artifact(format!(
            "artifact is {} bytes; limit is {MAX_ARTIFACT_BYTES}",
            bytes.len()
        )));
    }
    let checksum_offset = bytes.len() - CHECKSUM_BYTES;
    if sha256(&bytes[..checksum_offset]).as_slice() != &bytes[checksum_offset..] {
        return Err(Error::Artifact("artifact checksum does not match".into()));
    }

    let mut header = ByteReader::new(&bytes[..HEADER_BYTES]);
    if header.read_array::<8>()? != MODEL_ARTIFACT_MAGIC {
        return Err(Error::Artifact("model artifact magic does not match".into()));
    }
    let version = header.read_u32()?;
    if version != MODEL_ARTIFACT_VERSION {
        return Err(Error::Artifact(format!(
            "model artifact version {version} is unsupported; expected {MODEL_ARTIFACT_VERSION}"
        )));
    }
    let kind = header.read_u32()?;
    if kind != MODEL_ARTIFACT_KIND {
        return Err(Error::Artifact(format!(
            "artifact kind {kind} is not a portable model"
        )));
    }
    let source_sha256 = header.read_array::<32>()?;
    if source_sha256 == [0; 32] {
        return Err(Error::Artifact("source checkpoint hash is empty".into()));
    }
    let tensor_count = usize::try_from(header.read_u32()?)
        .map_err(|_| Error::Artifact("tensor count exceeds usize".into()))?;
    if tensor_count == 0 || tensor_count > MAX_TENSOR_COUNT {
        return Err(Error::Artifact(format!(
            "artifact has {tensor_count} tensors; expected 1..={MAX_TENSOR_COUNT}"
        )));
    }
    let declared_parameters = header.read_u64()?;
    let declared_tensor_bytes = header.read_u64()?;
    let config = ModelConfig {
        max_classes: Some(header.read_u64()?),
        num_quantiles: Some(header.read_u64()?),
        embed_dim: Some(header.read_u64()?),
        column_blocks: Some(header.read_u64()?),
        row_blocks: Some(header.read_u64()?),
        row_cls_tokens: Some(header.read_u64()?),
        icl_blocks: Some(header.read_u64()?),
    };
    config.validate()?;
    let directory_length = header.read_u64()?;
    let payload_length = header.read_u64()?;
    if header.read_u32()? != 0 || !header.is_empty() {
        return Err(Error::Artifact(
            "artifact header reserved bytes are invalid".into(),
        ));
    }
    if directory_length > MAX_DIRECTORY_BYTES || payload_length > MAX_TOTAL_TENSOR_BYTES + 64 * 4096 {
        return Err(Error::Artifact(
            "artifact section length exceeds its limit".into(),
        ));
    }

    let directory_end = HEADER_BYTES
        .checked_add(u64_to_usize(directory_length, "directory length")?)
        .ok_or_else(|| Error::Artifact("directory end overflows usize".into()))?;
    let payload_start = align_up(directory_end, TENSOR_ALIGNMENT)?;
    let payload_end = payload_start
        .checked_add(u64_to_usize(payload_length, "payload length")?)
        .ok_or_else(|| Error::Artifact("payload end overflows usize".into()))?;
    if payload_end != checksum_offset || directory_end > payload_start {
        return Err(Error::Artifact(
            "artifact section offsets are inconsistent".into(),
        ));
    }
    if bytes[directory_end..payload_start].iter().any(|byte| *byte != 0) {
        return Err(Error::Artifact("artifact alignment padding is not zero".into()));
    }

    let mut directory = ByteReader::new(&bytes[HEADER_BYTES..directory_end]);
    let mut previous_name: Option<String> = None;
    let mut ranges = Vec::with_capacity(tensor_count);
    let mut tensors = Vec::with_capacity(tensor_count);
    let mut parameter_count = 0_u64;
    let mut tensor_bytes = 0_u64;
    let mut dtype_counts = BTreeMap::new();
    for _ in 0..tensor_count {
        let name_length = usize::try_from(directory.read_u32()?)
            .map_err(|_| Error::Artifact("tensor name length exceeds usize".into()))?;
        if name_length == 0 || name_length > MAX_TENSOR_NAME_BYTES {
            return Err(Error::Artifact(format!(
                "invalid tensor name length {name_length}"
            )));
        }
        let rank = usize::from(directory.read_u16()?);
        if rank > MAX_TENSOR_RANK {
            return Err(Error::Artifact(format!(
                "tensor rank {rank} exceeds {MAX_TENSOR_RANK}"
            )));
        }
        let dtype = ModelTensorDType::from_id(directory.read_u8()?)?;
        if directory.read_u8()? != 0 {
            return Err(Error::Artifact(
                "tensor directory reserved byte is not zero".into(),
            ));
        }
        let payload_offset = directory.read_u64()?;
        let byte_length = directory.read_u64()?;
        if byte_length > MAX_TENSOR_BYTES || payload_offset % TENSOR_ALIGNMENT as u64 != 0 {
            return Err(Error::Artifact(
                "tensor payload offset or length is invalid".into(),
            ));
        }
        let mut dimensions = Vec::with_capacity(rank);
        for _ in 0..rank {
            dimensions.push(directory.read_u64()?);
        }
        let name = std::str::from_utf8(directory.read_bytes(name_length)?)
            .map_err(|error| Error::Artifact(format!("tensor name is not UTF-8: {error}")))?
            .to_owned();
        if previous_name.as_ref().is_some_and(|previous| previous >= &name) {
            return Err(Error::Artifact("tensor names are not unique and sorted".into()));
        }
        previous_name = Some(name.clone());

        let elements = checked_elements(&name, &dimensions)?;
        let expected_bytes = elements
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Artifact(format!("tensor '{name}' byte count overflows u64")))?;
        if expected_bytes != byte_length {
            return Err(Error::Artifact(format!(
                "tensor '{name}' is {byte_length} bytes; expected {expected_bytes}"
            )));
        }
        let start = u64_to_usize(payload_offset, "tensor payload offset")?;
        let end = start
            .checked_add(u64_to_usize(byte_length, "tensor byte length")?)
            .ok_or_else(|| Error::Artifact(format!("tensor '{name}' payload end overflows usize")))?;
        if end > u64_to_usize(payload_length, "payload length")? {
            return Err(Error::Artifact(format!(
                "tensor '{name}' exceeds the payload section"
            )));
        }
        ranges.push((start, end));
        let payload_range = payload_start
            .checked_add(start)
            .zip(payload_start.checked_add(end))
            .ok_or_else(|| {
                Error::Artifact(format!("tensor '{name}' absolute payload range overflows usize"))
            })?;
        tensors.push(ParsedTensor {
            name,
            dtype,
            dimensions,
            payload_range: payload_range.0..payload_range.1,
        });
        parameter_count = parameter_count
            .checked_add(elements)
            .ok_or_else(|| Error::Artifact("model parameter count overflows u64".into()))?;
        tensor_bytes = tensor_bytes
            .checked_add(byte_length)
            .ok_or_else(|| Error::Artifact("model tensor byte count overflows u64".into()))?;
        *dtype_counts.entry(dtype.as_str().to_owned()).or_insert(0) += 1;
    }
    if !directory.is_empty() {
        return Err(Error::Artifact("tensor directory has trailing bytes".into()));
    }
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(Error::Artifact("tensor payload ranges overlap".into()));
        }
    }
    if parameter_count != declared_parameters || tensor_bytes != declared_tensor_bytes {
        return Err(Error::Artifact(
            "artifact aggregate tensor metadata does not match".into(),
        ));
    }

    let profile = identify_profile(
        &source_sha256,
        config.task(),
        tensor_count,
        parameter_count,
        tensor_bytes,
    );
    Ok(ParsedModelArtifact {
        summary: ModelArtifactSummary {
            source_sha256: encode_sha256(&source_sha256),
            profile,
            tensor_count,
            parameter_count,
            tensor_bytes,
            dtype_counts,
            config,
        },
        tensors,
    })
}

#[cfg(feature = "checkpoint")]
fn validate_tensor(tensor: &ModelTensor) -> Result<()> {
    if tensor.name.is_empty() || tensor.name.len() > MAX_TENSOR_NAME_BYTES {
        return Err(Error::Artifact(format!("invalid tensor name '{}'", tensor.name)));
    }
    if tensor.dimensions.len() > MAX_TENSOR_RANK {
        return Err(Error::Artifact(format!(
            "tensor '{}' rank {} exceeds {MAX_TENSOR_RANK}",
            tensor.name,
            tensor.dimensions.len()
        )));
    }
    let expected = checked_elements(&tensor.name, &tensor.dimensions)?
        .checked_mul(tensor.dtype.size_in_bytes())
        .ok_or_else(|| Error::Artifact(format!("tensor '{}' byte count overflows u64", tensor.name)))?;
    let actual = usize_to_u64(tensor.data.len(), "tensor byte length")?;
    if actual != expected || actual > MAX_TENSOR_BYTES {
        return Err(Error::Artifact(format!(
            "tensor '{}' is {actual} bytes; expected {expected} with limit {MAX_TENSOR_BYTES}",
            tensor.name
        )));
    }
    Ok(())
}

fn checked_elements(name: &str, dimensions: &[u64]) -> Result<u64> {
    dimensions.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| Error::Artifact(format!("tensor '{name}' element count overflows u64")))
    })
}

fn validate_classifier_manifest(tensors: &[ParsedTensor]) -> Result<()> {
    let expected = classifier_tensor_manifest();
    if tensors.len() != expected.len() {
        return Err(Error::Artifact(format!(
            "official classifier has {} tensors; expected {}",
            tensors.len(),
            expected.len()
        )));
    }
    for tensor in tensors {
        let Some(dimensions) = expected.get(&tensor.name) else {
            return Err(Error::Artifact(format!(
                "unexpected official classifier tensor '{}'",
                tensor.name
            )));
        };
        if tensor.dtype != ModelTensorDType::F32 {
            return Err(Error::Artifact(format!(
                "official classifier tensor '{}' has dtype {}; expected f32",
                tensor.name,
                tensor.dtype.as_str()
            )));
        }
        if tensor.dimensions != *dimensions {
            return Err(Error::Artifact(format!(
                "official classifier tensor '{}' has shape {:?}; expected {:?}",
                tensor.name, tensor.dimensions, dimensions
            )));
        }
    }
    Ok(())
}

fn validate_classifier_config(config: &ModelConfig) -> Result<()> {
    let actual = config.required_values()?;
    let expected = [10, 999, 128, 3, 3, 4, 12];
    if actual != expected {
        return Err(Error::Artifact(format!(
            "official classifier configuration is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn classifier_tensor_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();
    let mut insert = |name: String, dimensions: &[u64]| {
        assert!(tensors.insert(name, dimensions.to_vec()).is_none());
    };
    insert("col_embedder.in_linear.bias".into(), &[128]);
    insert("col_embedder.in_linear.weight".into(), &[128, 3]);
    insert("col_embedder.y_encoder.bias".into(), &[128]);
    insert("col_embedder.y_encoder.weight".into(), &[128, 10]);
    for block in 0..3 {
        let prefix = format!("col_embedder.tf_col.blocks.{block}");
        insert(format!("{prefix}.ind_vectors"), &[128, 128]);
        insert_attention_manifest(&mut insert, &format!("{prefix}.multihead_attn1"), 128, true);
        insert_attention_manifest(&mut insert, &format!("{prefix}.multihead_attn2"), 128, false);
    }

    insert("icl_predictor.decoder.0.bias".into(), &[1024]);
    insert("icl_predictor.decoder.0.weight".into(), &[1024, 512]);
    insert("icl_predictor.decoder.2.bias".into(), &[10]);
    insert("icl_predictor.decoder.2.weight".into(), &[10, 1024]);
    insert("icl_predictor.ln.bias".into(), &[512]);
    insert("icl_predictor.ln.weight".into(), &[512]);
    insert("icl_predictor.y_encoder.bias".into(), &[512]);
    insert("icl_predictor.y_encoder.weight".into(), &[512, 10]);
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
    debug_assert_eq!(tensors.len(), 391);
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
    task: Option<TaskKind>,
    tensor_count: usize,
    parameter_count: u64,
    tensor_bytes: u64,
) -> ModelProfile {
    match (source_sha256, task, tensor_count, parameter_count, tensor_bytes) {
        (&CLASSIFIER_SHA256, Some(TaskKind::Classifier), 391, 27_552_258, 110_209_032) => {
            ModelProfile::TabIclV2Classifier
        }
        (&REGRESSOR_SHA256, Some(TaskKind::Regressor), 347, 28_544_991, 114_179_964) => {
            ModelProfile::TabIclV2Regressor
        }
        _ => ModelProfile::Unknown,
    }
}

fn read_artifact_bytes(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(artifact_error)?;
    let length = file.metadata().map_err(artifact_error)?.len();
    if length > MAX_ARTIFACT_BYTES {
        return Err(Error::Artifact(format!(
            "artifact is {length} bytes; limit is {MAX_ARTIFACT_BYTES}"
        )));
    }
    let capacity = u64_to_usize(length, "artifact length")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(MAX_ARTIFACT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(artifact_error)?;
    if usize_to_u64(bytes.len(), "artifact length")? > MAX_ARTIFACT_BYTES {
        return Err(Error::Artifact(
            "artifact grew beyond its size limit while reading".into(),
        ));
    }
    Ok(bytes)
}

#[cfg(feature = "checkpoint")]
fn create_temporary_path(path: &Path) -> Result<PathBuf> {
    let parent = usable_parent(path);
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::Artifact("output path has no valid UTF-8 file name".into()))?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::Artifact(format!("system clock is before UNIX epoch: {error}")))?
        .as_nanos();
    for attempt in 0..100_u32 {
        let candidate = parent.join(format!(
            ".{filename}.tmp-{}-{timestamp}-{attempt}",
            std::process::id()
        ));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(Error::Artifact(
        "could not allocate a temporary output path".into(),
    ))
}

#[cfg(feature = "checkpoint")]
fn usable_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(feature = "checkpoint")]
fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) -> Result<()> {
    let aligned = align_up(bytes.len(), alignment)?;
    bytes.resize(aligned, 0);
    Ok(())
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| Error::Artifact("alignment calculation overflows usize".into()))
    }
}

fn required_config(value: Option<u64>, name: &str) -> Result<u64> {
    value.ok_or_else(|| Error::Artifact(format!("model configuration '{name}' is missing")))
}

#[cfg(feature = "checkpoint")]
fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "checkpoint")]
fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "checkpoint")]
fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "checkpoint")]
fn usize_to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Artifact(format!("{name} exceeds u32")))
}

fn usize_to_u64(value: usize, name: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::Artifact(format!("{name} exceeds u64")))
}

fn u64_to_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::Artifact(format!("{name} exceeds usize")))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Encode a SHA-256 digest as 64 lowercase hexadecimal characters.
#[must_use]
pub fn encode_sha256(digest: &[u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// Decode a 64-character hexadecimal SHA-256 digest.
///
/// # Errors
///
/// Returns [`Error::Artifact`] when `encoded` is not 64 hexadecimal characters.
pub fn decode_sha256(encoded: &str) -> Result<[u8; 32]> {
    let bytes = encoded.as_bytes();
    if bytes.len() != 64 {
        return Err(artifact_error(
            "SHA-256 digest must have 64 hexadecimal characters",
        ));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let nibble = |byte: u8| match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            b'A'..=b'F' => Ok(byte - b'A' + 10),
            _ => Err(artifact_error("SHA-256 digest is not hexadecimal")),
        };
        digest[index] = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Ok(digest)
}

fn artifact_error(error: impl fmt::Display) -> Error {
    Error::Artifact(error.to_string())
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.read_bytes(N)?
            .try_into()
            .map_err(|_| Error::Artifact("artifact field has an invalid length".into()))
    }

    fn read_bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| Error::Artifact("artifact offset overflows usize".into()))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::Artifact("artifact is truncated".into()))?;
        self.position = end;
        Ok(bytes)
    }
}

#[cfg(all(test, feature = "checkpoint"))]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn model_artifact_round_trips_and_detects_corruption() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("model.ironicl");
        let model = fixture_model();

        let written = write_model_artifact(&path, &model).expect("write model artifact");
        let inspected = inspect_model_artifact(&path).expect("inspect model artifact");
        assert_eq!(written, inspected);
        assert_eq!(inspected.tensor_count, 2);
        assert_eq!(inspected.parameter_count, 6);
        assert_eq!(inspected.tensor_bytes, 24);

        let mut bytes = fs::read(&path).expect("read artifact");
        bytes[HEADER_BYTES] ^= 1;
        fs::write(&path, bytes).expect("corrupt artifact");
        assert!(inspect_model_artifact(&path).is_err());
    }

    #[test]
    fn parsed_tensor_ranges_include_the_payload_section_offset() {
        let model = fixture_model();
        let bytes = encode_model_artifact(&model).expect("encode fixture");
        let parsed = parse_model_artifact_bytes(&bytes).expect("parse fixture");
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == "a.weight")
            .expect("fixture tensor");

        assert_eq!(&bytes[tensor.payload_range.clone()], &[0; 16]);
        assert!(tensor.payload_range.start >= HEADER_BYTES);
    }

    #[test]
    fn classifier_manifest_has_all_official_tensors() {
        let manifest = classifier_tensor_manifest();
        assert_eq!(manifest.len(), 391);
        assert_eq!(
            manifest.get("icl_predictor.tf_icl.blocks.11.linear2.weight"),
            Some(&vec![512, 1024])
        );
        assert_eq!(manifest.get("row_interactor.tf_row.rope.freqs"), Some(&vec![8]));
    }

    #[test]
    fn classifier_manifest_rejects_wrong_shape_and_unexpected_tensor() {
        let manifest = classifier_tensor_manifest();
        let mut tensors = manifest
            .into_iter()
            .map(|(name, dimensions)| ParsedTensor {
                name,
                dtype: ModelTensorDType::F32,
                dimensions,
                payload_range: 0..0,
            })
            .collect::<Vec<_>>();
        tensors[0].dimensions = vec![1];
        assert!(validate_classifier_manifest(&tensors).is_err());

        tensors[0].dimensions = classifier_tensor_manifest()
            .get(&tensors[0].name)
            .expect("known tensor")
            .clone();
        tensors[0].name = "unexpected.tensor".into();
        assert!(validate_classifier_manifest(&tensors).is_err());

        tensors[0].name = classifier_tensor_manifest()
            .keys()
            .next()
            .expect("manifest has tensors")
            .clone();
        tensors[0].dtype = ModelTensorDType::F64;
        assert!(validate_classifier_manifest(&tensors).is_err());
    }

    #[test]
    fn classifier_configuration_must_match_the_release() {
        let mut config = fixture_model().config;
        config.num_quantiles = Some(998);
        assert!(validate_classifier_config(&config).is_err());
    }

    fn fixture_model() -> ModelArtifactData {
        ModelArtifactData {
            source_sha256: [1; 32],
            config: ModelConfig {
                max_classes: Some(10),
                num_quantiles: Some(999),
                embed_dim: Some(128),
                column_blocks: Some(3),
                row_blocks: Some(3),
                row_cls_tokens: Some(4),
                icl_blocks: Some(12),
            },
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

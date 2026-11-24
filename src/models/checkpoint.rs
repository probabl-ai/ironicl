use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
};

use candle_core::{
    DType, Layout, Shape,
    pickle::{Object, Stack, TensorInfo},
};
use sha2::{Digest, Sha256};

use crate::{
    ModelArtifact, TaskKind,
    error::{Error, Result},
    format::model::{
        ModelArtifactData, ModelConfig, ModelTensor, ModelTensorDType, TabIclV2Config, encode_model_artifact,
        write_model_artifact,
    },
    models::kumo_tabular::Release,
};

/// Kumo-Tabular large is about 860 MB.
const MAX_ARCHIVE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 4096;
const MAX_PICKLE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_PICKLE_OPS: usize = 1_000_000;
const MAX_PICKLE_STACK_DEPTH: usize = 65_536;
const MAX_PICKLE_MEMO_ENTRIES: usize = 65_536;
const MAX_PICKLE_OBJECT_DEPTH: usize = 64;
const MAX_PICKLE_OBJECT_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TENSOR_COUNT: usize = 4096;
const MAX_TENSOR_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOTAL_TENSOR_BYTES: u64 = 1000 * 1024 * 1024;
const MAX_TENSOR_RANK: usize = 16;

const CLASSIFIER_SHA256: &str = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0";
const REGRESSOR_SHA256: &str = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a";
const HEX: &[u8; 16] = b"0123456789abcdef";

type CheckpointArchive = zip::ZipArchive<Cursor<Vec<u8>>>;

/// A recognized released checkpoint or an otherwise parseable checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointProfile {
    TabIclV2Classifier,
    TabIclV2Regressor,
    KumoTabular(Release),
    Unknown,
}

impl fmt::Display for CheckpointProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TabIclV2Classifier => f.write_str("TabICLv2 classifier"),
            Self::TabIclV2Regressor => f.write_str("TabICLv2 regressor"),
            Self::KumoTabular(release) => release.fmt(f),
            Self::Unknown => f.write_str("unknown/custom"),
        }
    }
}

/// Selected architecture values retained in a `TabICLv2` checkpoint's primitive config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointModelConfig {
    pub max_classes: Option<u64>,
    pub num_quantiles: Option<u64>,
    pub embed_dim: Option<u64>,
    pub column_blocks: Option<u64>,
    pub row_blocks: Option<u64>,
    pub row_cls_tokens: Option<u64>,
    pub icl_blocks: Option<u64>,
}

impl CheckpointModelConfig {
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
}

/// Safe, non-executing summary of a `PyTorch` ZIP checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSummary {
    pub source_sha256: String,
    pub profile: CheckpointProfile,
    pub tensor_count: usize,
    pub parameter_count: u64,
    pub tensor_bytes: u64,
    pub dtype_counts: BTreeMap<String, usize>,
    pub config: Option<CheckpointModelConfig>,
    pub warnings: Vec<String>,
}

/// What a checkpoint holds, as its caller knows: a `TabICLv2` checkpoint
/// carries its own config; a Kumo-Tabular one is a bare state dict of exactly
/// one release's tensors, `release`'s or, for `None`, the release they match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointKind {
    TabIclV2,
    KumoTabular(Option<Release>),
}

impl CheckpointKind {
    /// Whether imported weights with `profile` are this kind's.
    pub(crate) fn holds(self, profile: crate::ModelProfile) -> bool {
        match self {
            Self::TabIclV2 => !matches!(profile, crate::ModelProfile::KumoTabular(_)),
            Self::KumoTabular(None) => matches!(profile, crate::ModelProfile::KumoTabular(_)),
            Self::KumoTabular(Some(release)) => profile == crate::ModelProfile::KumoTabular(release),
        }
    }
}

/// Parse and summarize a `PyTorch` ZIP checkpoint of `kind` without executing
/// pickle code. Without a `kind`, a checkpoint with a config is `TabICLv2`'s
/// and one without is Kumo-Tabular's.
///
/// Released `TabICLv2` profiles require both the official SHA-256 and the
/// expected model metadata; other valid ones are reported as
/// [`CheckpointProfile::Unknown`].
///
/// # Errors
///
/// Returns [`Error::Checkpoint`] when the archive, pickle structure, tensor
/// metadata, or size calculations are invalid, or the tensors are not
/// `kind`'s.
pub fn inspect_checkpoint(path: &Path, kind: Option<CheckpointKind>) -> Result<CheckpointSummary> {
    Ok(load_checkpoint(path, kind)?.summary)
}

pub(crate) fn checkpoint_sha256(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(checkpoint_error)?;
    let file_size = file.metadata().map_err(checkpoint_error)?.len();
    if file_size > MAX_ARCHIVE_BYTES {
        return Err(Error::Checkpoint(format!(
            "archive is {file_size} bytes; limit is {MAX_ARCHIVE_BYTES}"
        )));
    }
    let mut limited = file.take(MAX_ARCHIVE_BYTES + 1);
    let digest = sha256_reader(&mut limited)?;
    if limited.limit() == 0 {
        return Err(Error::Checkpoint(format!(
            "archive grew beyond the {MAX_ARCHIVE_BYTES}-byte limit while hashing"
        )));
    }
    Ok(digest)
}

/// Import a validated checkpoint into the portable Ironicl model format.
///
/// # Errors
///
/// Returns [`Error::Checkpoint`] when the checkpoint is invalid and
/// [`Error::Artifact`] when the output cannot be written or verified.
pub fn import_checkpoint(
    path: &Path,
    kind: CheckpointKind,
    output: &Path,
) -> Result<crate::ModelArtifactSummary> {
    if same_file_path(path, output)? {
        return Err(Error::InvalidArgument(
            "checkpoint input and model output must be different files".into(),
        ));
    }
    write_model_artifact(output, &model_data(path, kind)?)
}

/// Convert a validated checkpoint into an executable model in memory, as
/// [`import_checkpoint`] would write it, without writing a file.
///
/// # Errors
///
/// Returns [`Error::Checkpoint`] when the checkpoint is invalid and
/// [`Error::Artifact`] when it is not an official model.
pub(crate) fn model_from_checkpoint(path: &Path, kind: CheckpointKind) -> Result<ModelArtifact> {
    ModelArtifact::from_bytes(&encode_model_artifact(&model_data(path, kind)?)?)
}

/// The portable model of a validated checkpoint.
fn model_data(path: &Path, kind: CheckpointKind) -> Result<ModelArtifactData> {
    let loaded = load_checkpoint(path, Some(kind))?;
    let source_sha256 = decode_sha256(&loaded.summary.source_sha256)?;
    let config = match (loaded.summary.profile, &loaded.summary.config) {
        (CheckpointProfile::KumoTabular(release), _) => ModelConfig::KumoTabular(release),
        (_, Some(config)) => ModelConfig::TabIclV2(TabIclV2Config {
            max_classes: config.max_classes,
            num_quantiles: config.num_quantiles,
            embed_dim: config.embed_dim,
            column_blocks: config.column_blocks,
            row_blocks: config.row_blocks,
            row_cls_tokens: config.row_cls_tokens,
            icl_blocks: config.icl_blocks,
        }),
        (_, None) => {
            return Err(Error::Checkpoint(
                "validated checkpoint has no model config".into(),
            ));
        }
    };
    let tensors = loaded
        .tensors
        .into_iter()
        .map(|tensor| {
            let dimensions = tensor
                .info
                .layout
                .dims()
                .iter()
                .map(|dimension| {
                    u64::try_from(*dimension).map_err(|_| {
                        Error::Checkpoint("tensor dimension exceeds portable artifact limits".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(ModelTensor {
                name: tensor.info.name,
                dtype: model_dtype(tensor.info.dtype),
                dimensions,
                data: tensor.data,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ModelArtifactData {
        source_sha256,
        config,
        tensors,
    })
}

struct LoadedCheckpoint {
    summary: CheckpointSummary,
    tensors: Vec<LoadedTensor>,
}

struct LoadedTensor {
    info: TensorInfo,
    data: Vec<u8>,
}

fn load_checkpoint(path: &Path, kind: Option<CheckpointKind>) -> Result<LoadedCheckpoint> {
    let archive_bytes = read_checkpoint_bytes(path)?;
    let source_sha256 = sha256_reader(&mut Cursor::new(&archive_bytes))?;
    let (root, storage_directory, mut archive) = read_pickle_root(archive_bytes)?;
    // A Kumo-Tabular checkpoint is a bare state dict; a TabICLv2 one also has its config.
    let has_config = unique_dict_value(&root, "config")?.is_some();
    let kind = match kind {
        Some(CheckpointKind::KumoTabular(_)) if has_config => {
            return Err(Error::Checkpoint(
                "not a Kumo-Tabular checkpoint: it has a config, as TabICLv2's do".into(),
            ));
        }
        Some(CheckpointKind::TabIclV2) if !has_config => {
            return Err(Error::Checkpoint(
                "not a TabICLv2 checkpoint: it has no config".into(),
            ));
        }
        Some(kind) => kind,
        None if has_config => CheckpointKind::TabIclV2,
        None => CheckpointKind::KumoTabular(None),
    };
    let (state_dict, config) = if matches!(kind, CheckpointKind::KumoTabular(_)) {
        (&root, None)
    } else {
        let config_object = unique_dict_value(&root, "config")?
            .ok_or_else(|| Error::Checkpoint("top-level 'config' mapping is missing".into()))?;
        let state_dict = unique_dict_value(&root, "state_dict")?
            .ok_or_else(|| Error::Checkpoint("top-level 'state_dict' mapping is missing".into()))?;
        (state_dict, Some(extract_model_config(config_object)?))
    };
    let tensors = tensor_metadata(state_dict, &storage_directory)?;
    let kumo = match kind {
        CheckpointKind::TabIclV2 => None,
        CheckpointKind::KumoTabular(Some(release)) => {
            check_kumo_manifest(release, &tensors)?;
            Some(release)
        }
        CheckpointKind::KumoTabular(None) => Some(kumo_release(&tensors)?),
    };
    let tensor_data = read_storage_entries(&mut archive, &tensors)?;
    let mut summary = summarize_tensors(source_sha256, &tensors, config)?;
    if let Some(release) = kumo {
        summary.profile = CheckpointProfile::KumoTabular(release);
    }
    let tensors = tensors
        .into_iter()
        .map(|info| {
            let storage = tensor_data
                .get(&info.path)
                .ok_or_else(|| Error::Checkpoint(format!("missing loaded storage '{}'", info.path)))?;
            let tensor_bytes = checked_element_count(&info.name, info.layout.dims())?
                .checked_mul(u64::try_from(info.dtype.size_in_bytes()).map_err(|_| {
                    Error::Checkpoint(format!("tensor '{}' dtype size exceeds u64", info.name))
                })?)
                .ok_or_else(|| {
                    Error::Checkpoint(format!("tensor '{}' byte count overflows u64", info.name))
                })?;
            let tensor_bytes = usize::try_from(tensor_bytes)
                .map_err(|_| Error::Checkpoint(format!("tensor '{}' byte count exceeds usize", info.name)))?;
            let truncated = || Error::Checkpoint(format!("tensor '{}' storage is truncated", info.name));
            let data = if info.layout.is_contiguous() && info.layout.start_offset() == 0 {
                storage.get(..tensor_bytes).ok_or_else(truncated)?.to_vec()
            } else {
                // Copy a strided view, such as a transposed weight, into row-major order.
                let size = info.dtype.size_in_bytes();
                let mut data = Vec::with_capacity(tensor_bytes);
                for element in 0..tensor_bytes / size {
                    let index = storage_index(&info.layout, element);
                    data.extend_from_slice(
                        storage
                            .get(index * size..(index + 1) * size)
                            .ok_or_else(truncated)?,
                    );
                }
                data
            };
            Ok(LoadedTensor { info, data })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LoadedCheckpoint { summary, tensors })
}

/// The storage index of a strided tensor's `element`th value in row-major order.
fn storage_index(layout: &Layout, element: usize) -> usize {
    let (mut rest, mut index) = (element, layout.start_offset());
    for (dimension, stride) in layout.dims().iter().zip(layout.stride()).rev() {
        index += rest % dimension * stride;
        rest /= dimension;
    }
    index
}

/// The Kumo-Tabular release whose tensors a checkpoint holds.
fn kumo_release(tensors: &[TensorInfo]) -> Result<Release> {
    Release::ALL
        .into_iter()
        .find(|&release| check_kumo_manifest(release, tensors).is_ok())
        .ok_or_else(|| {
            Error::Checkpoint("not a Kumo-Tabular checkpoint: its tensors are no release's".into())
        })
}

/// Check that a Kumo-Tabular checkpoint holds exactly its release's FP32 tensors.
fn check_kumo_manifest(release: Release, tensors: &[TensorInfo]) -> Result<()> {
    let mismatch = |detail: String| Err(Error::Checkpoint(format!("not a {release} checkpoint: {detail}")));
    let expected = release.manifest();
    for tensor in tensors {
        let dimensions: Vec<u64> = tensor
            .layout
            .dims()
            .iter()
            .map(|&dimension| dimension as u64)
            .collect();
        match expected.get(&tensor.name) {
            None => return mismatch(format!("unexpected tensor '{}'", tensor.name)),
            Some(shape) if *shape != dimensions => {
                return mismatch(format!("'{}' is {dimensions:?}, expected {shape:?}", tensor.name));
            }
            Some(_) if tensor.dtype != DType::F32 => {
                return mismatch(format!("'{}' is not FP32", tensor.name));
            }
            Some(_) => {}
        }
    }
    let found: BTreeSet<&str> = tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    match expected.keys().find(|name| !found.contains(name.as_str())) {
        Some(name) => mismatch(format!("'{name}' is missing")),
        None if found.len() < tensors.len() => mismatch("a tensor name repeats".into()),
        None => Ok(()),
    }
}

fn read_checkpoint_bytes(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(checkpoint_error)?;
    let file_size = file.metadata().map_err(checkpoint_error)?.len();
    if file_size > MAX_ARCHIVE_BYTES {
        return Err(Error::Checkpoint(format!(
            "archive is {file_size} bytes; limit is {MAX_ARCHIVE_BYTES}"
        )));
    }
    let capacity =
        usize::try_from(file_size).map_err(|_| Error::Checkpoint("archive size exceeds usize".into()))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(MAX_ARCHIVE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(checkpoint_error)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_ARCHIVE_BYTES {
        return Err(Error::Checkpoint(format!(
            "archive grew beyond the {MAX_ARCHIVE_BYTES}-byte limit while reading"
        )));
    }
    Ok(bytes)
}

fn read_pickle_root(bytes: Vec<u8>) -> Result<(Object, String, CheckpointArchive)> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(checkpoint_error)?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        return Err(Error::Checkpoint(format!(
            "archive has {} entries; limit is {MAX_ARCHIVE_ENTRIES}",
            archive.len()
        )));
    }

    let mut entry_names = BTreeSet::new();
    let mut pickle_names = Vec::new();
    for index in 0..archive.len() {
        let entry = archive.by_index(index).map_err(checkpoint_error)?;
        let name = entry.name().map_err(checkpoint_error)?.into_owned();
        if entry.enclosed_name().is_none() {
            return Err(Error::Checkpoint(format!(
                "archive entry '{name}' has an unsafe path"
            )));
        }
        if !entry_names.insert(name.clone()) {
            return Err(Error::Checkpoint(format!(
                "archive contains duplicate entry '{name}'"
            )));
        }
        if name == "data.pkl" || name.ends_with("/data.pkl") {
            pickle_names.push(name);
        }
    }
    let [pickle_name] = pickle_names.as_slice() else {
        return Err(Error::Checkpoint(format!(
            "expected one data.pkl entry, found {}",
            pickle_names.len()
        )));
    };
    let storage_directory = pickle_name
        .strip_suffix(".pkl")
        .ok_or_else(|| Error::Checkpoint("invalid data.pkl entry name".into()))?
        .to_owned();
    let mut pickle = archive.by_name(pickle_name).map_err(checkpoint_error)?;
    if pickle.size() > MAX_PICKLE_BYTES {
        return Err(Error::Checkpoint(format!(
            "data.pkl is {} bytes; limit is {MAX_PICKLE_BYTES}",
            pickle.size()
        )));
    }

    let pickle_size = usize::try_from(pickle.size())
        .map_err(|_| Error::Checkpoint("data.pkl size exceeds usize".into()))?;
    let mut pickle_bytes = Vec::with_capacity(pickle_size);
    pickle.read_to_end(&mut pickle_bytes).map_err(checkpoint_error)?;
    drop(pickle);
    preflight_pickle(&pickle_bytes)?;

    let mut stack = Stack::empty();
    stack
        .read_loop(&mut BufReader::new(Cursor::new(&pickle_bytes)))
        .map_err(checkpoint_error)?;
    let root = stack.finalize().map_err(checkpoint_error)?;
    validate_object_tree(&root, 0, &mut 0, &mut BTreeSet::new())?;
    Ok((root, storage_directory, archive))
}

fn tensor_metadata(state_dict: &Object, storage_directory: &str) -> Result<Vec<TensorInfo>> {
    let Object::Dict(entries) = state_dict else {
        return Err(Error::Checkpoint(
            "top-level 'state_dict' is not a mapping".into(),
        ));
    };
    if entries.len() > MAX_TENSOR_COUNT {
        return Err(Error::Checkpoint(format!(
            "state_dict has {} entries; limit is {MAX_TENSOR_COUNT}",
            entries.len()
        )));
    }

    let mut names = BTreeSet::new();
    let mut tensors = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        let Object::Unicode(name) = name else {
            return Err(Error::Checkpoint("state_dict contains a non-string key".into()));
        };
        if !names.insert(name.as_str()) {
            return Err(Error::Checkpoint(format!("duplicate tensor name '{name}'")));
        }
        if name == "_metadata" {
            if !matches!(value, Object::Dict(_)) {
                return Err(Error::Checkpoint(
                    "state_dict '_metadata' attribute is not a mapping".into(),
                ));
            }
            continue;
        }
        if name.starts_with('_') {
            return Err(Error::Checkpoint(format!(
                "state_dict contains unsupported attribute '{name}'"
            )));
        }
        let tensor = decode_tensor_info(name, value, storage_directory)?;
        validate_tensor(&tensor)?;
        tensors.push(tensor);
    }
    if tensors.is_empty() {
        return Err(Error::Checkpoint("state_dict contains no tensors".into()));
    }
    Ok(tensors)
}

fn decode_tensor_info(name: &str, value: &Object, storage_directory: &str) -> Result<TensorInfo> {
    let Object::Reduce { callable, args } = value else {
        return Err(Error::Checkpoint(format!(
            "state_dict value '{name}' is not a tensor reduction"
        )));
    };
    let Object::Class {
        module_name,
        class_name,
    } = callable.as_ref()
    else {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' reduction target is not a global"
        )));
    };
    if module_name != "torch._utils" || class_name != "_rebuild_tensor_v2" {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' uses unsupported reduction {module_name}.{class_name}"
        )));
    }

    let arguments = object_tuple(args, name, "rebuild arguments")?;
    if arguments.len() < 4 {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' rebuild has {} arguments; expected at least 4",
            arguments.len()
        )));
    }
    let storage = object_persistent_load(&arguments[0], name)?;
    let storage = object_tuple(storage, name, "storage descriptor")?;
    if storage.len() < 5 {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' storage descriptor has {} fields; expected at least 5",
            storage.len()
        )));
    }
    if object_string(&storage[0], name, "storage kind")? != "storage" {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' has unsupported persistent storage kind"
        )));
    }
    let dtype = storage_dtype(&storage[1], name)?;
    let storage_key = object_string(&storage[2], name, "storage key")?;
    if storage_key.is_empty()
        || storage_key == "."
        || storage_key == ".."
        || storage_key.contains('/')
        || storage_key.contains('\\')
    {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' has invalid storage key '{storage_key}'"
        )));
    }
    let storage_size = object_usize(&storage[4], name, "storage size")?;
    let offset_elements = object_usize(&arguments[1], name, "storage offset")?;
    let dimensions = object_usize_tuple(&arguments[2], name, "shape")?;
    let stride = object_usize_tuple(&arguments[3], name, "stride")?;
    if dimensions.len() != stride.len() {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' shape rank {} does not match stride rank {}",
            dimensions.len(),
            stride.len()
        )));
    }
    let start_offset = offset_elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Checkpoint(format!("tensor '{name}' byte offset overflows usize")))?;
    let path = Path::new(storage_directory)
        .join(storage_key)
        .to_string_lossy()
        .into_owned();
    Ok(TensorInfo {
        name: name.to_owned(),
        dtype,
        layout: Layout::new(Shape::from(dimensions), stride, start_offset),
        path,
        storage_size,
    })
}

fn object_tuple<'a>(object: &'a Object, name: &str, field: &str) -> Result<&'a [Object]> {
    let Object::Tuple(values) = object else {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' {field} is not a tuple"
        )));
    };
    Ok(values)
}

fn object_persistent_load<'a>(object: &'a Object, name: &str) -> Result<&'a Object> {
    let Object::PersistentLoad(value) = object else {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' storage is not a persistent reference"
        )));
    };
    Ok(value)
}

fn object_string<'a>(object: &'a Object, name: &str, field: &str) -> Result<&'a str> {
    let Object::Unicode(value) = object else {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' {field} is not a string"
        )));
    };
    Ok(value)
}

fn object_usize(object: &Object, name: &str, field: &str) -> Result<usize> {
    let value = match object {
        Object::Int(value) => i64::from(*value),
        Object::Long(value) => *value,
        _ => {
            return Err(Error::Checkpoint(format!(
                "tensor '{name}' {field} is not an integer"
            )));
        }
    };
    usize::try_from(value)
        .map_err(|_| Error::Checkpoint(format!("tensor '{name}' {field} is negative or too large")))
}

fn object_usize_tuple(object: &Object, name: &str, field: &str) -> Result<Vec<usize>> {
    object_tuple(object, name, field)?
        .iter()
        .map(|value| object_usize(value, name, field))
        .collect()
}

fn storage_dtype(object: &Object, name: &str) -> Result<DType> {
    let Object::Class {
        module_name,
        class_name,
    } = object
    else {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' storage dtype is not a global"
        )));
    };
    if module_name != "torch" {
        return Err(Error::Checkpoint(format!(
            "tensor '{name}' storage dtype module '{module_name}' is unsupported"
        )));
    }
    match class_name.as_str() {
        "FloatStorage" => Ok(DType::F32),
        "DoubleStorage" => Ok(DType::F64),
        "HalfStorage" => Ok(DType::F16),
        "BFloat16Storage" => Ok(DType::BF16),
        "ByteStorage" => Ok(DType::U8),
        "LongStorage" => Ok(DType::I64),
        _ => Err(Error::Checkpoint(format!(
            "tensor '{name}' storage type '{class_name}' is unsupported"
        ))),
    }
}

fn read_storage_entries(
    archive: &mut CheckpointArchive,
    tensors: &[TensorInfo],
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut storages = BTreeMap::new();
    for tensor in tensors {
        let storage_bytes = u64::try_from(tensor.storage_size)
            .ok()
            .and_then(|size| size.checked_mul(u64::try_from(tensor.dtype.size_in_bytes()).ok()?))
            .ok_or_else(|| {
                Error::Checkpoint(format!("tensor '{}' storage size overflows u64", tensor.name))
            })?;
        match storages.insert(tensor.path.as_str(), storage_bytes) {
            Some(previous) if previous != storage_bytes => {
                return Err(Error::Checkpoint(format!(
                    "storage '{}' has conflicting declared sizes",
                    tensor.path
                )));
            }
            _ => {}
        }
    }

    let total_storage_bytes = storages.values().try_fold(0_u64, |total, size| {
        total
            .checked_add(*size)
            .ok_or_else(|| Error::Checkpoint("checkpoint storage byte count overflows u64".into()))
    })?;
    if total_storage_bytes > MAX_TOTAL_TENSOR_BYTES {
        return Err(Error::Checkpoint(format!(
            "checkpoint storage is {total_storage_bytes} bytes; limit is {MAX_TOTAL_TENSOR_BYTES}"
        )));
    }

    let mut storage_data = BTreeMap::new();
    for (path, expected_bytes) in storages {
        let mut entry = archive
            .by_name(path)
            .map_err(|error| Error::Checkpoint(format!("cannot read tensor storage '{path}': {error}")))?;
        if !entry.is_file() {
            return Err(Error::Checkpoint(format!(
                "tensor storage '{path}' is not a file"
            )));
        }
        if entry.size() != expected_bytes {
            return Err(Error::Checkpoint(format!(
                "tensor storage '{path}' is {} bytes; expected {expected_bytes}",
                entry.size()
            )));
        }
        let capacity = usize::try_from(expected_bytes)
            .map_err(|_| Error::Checkpoint(format!("tensor storage '{path}' exceeds usize")))?;
        let mut data = Vec::with_capacity(capacity);
        entry
            .read_to_end(&mut data)
            .map_err(|error| Error::Checkpoint(format!("cannot verify tensor storage '{path}': {error}")))?;
        if u64::try_from(data.len()).unwrap_or(u64::MAX) != expected_bytes {
            return Err(Error::Checkpoint(format!(
                "tensor storage '{path}' yielded {} bytes; expected {expected_bytes}",
                data.len()
            )));
        }
        storage_data.insert(path.to_owned(), data);
    }
    Ok(storage_data)
}

fn model_dtype(dtype: DType) -> ModelTensorDType {
    match dtype {
        DType::U8 => ModelTensorDType::U8,
        DType::I64 => ModelTensorDType::I64,
        DType::BF16 => ModelTensorDType::Bf16,
        DType::F16 => ModelTensorDType::F16,
        DType::F32 => ModelTensorDType::F32,
        DType::F64 => ModelTensorDType::F64,
        _ => unreachable!("storage_dtype accepts only the dtypes above"),
    }
}

pub(crate) fn decode_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(Error::Checkpoint("source SHA-256 has an invalid length".into()));
    }
    let mut digest = [0_u8; 32];
    for (output, pair) in digest.iter_mut().zip(value.as_bytes().as_chunks::<2>().0) {
        let high = decode_hex(pair[0])?;
        let low = decode_hex(pair[1])?;
        *output = (high << 4) | low;
    }
    Ok(digest)
}

fn decode_hex(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(Error::Checkpoint(
            "source SHA-256 is not lowercase hexadecimal".into(),
        )),
    }
}

fn same_file_path(left: &Path, right: &Path) -> Result<bool> {
    if left == right {
        return Ok(true);
    }
    let left = left.canonicalize().map_err(checkpoint_error)?;
    match right.canonicalize() {
        Ok(right) => Ok(left == right),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(checkpoint_error(error)),
    }
}

fn validate_tensor(tensor: &TensorInfo) -> Result<()> {
    let dimensions = tensor.layout.dims();
    if dimensions.len() > MAX_TENSOR_RANK {
        return Err(Error::Checkpoint(format!(
            "tensor '{}' rank {} exceeds limit {MAX_TENSOR_RANK}",
            tensor.name,
            dimensions.len()
        )));
    }
    let elements = checked_element_count(&tensor.name, dimensions)?;
    let tensor_bytes = elements
        .checked_mul(
            u64::try_from(tensor.dtype.size_in_bytes())
                .map_err(|_| Error::Checkpoint(format!("tensor '{}' dtype size exceeds u64", tensor.name)))?,
        )
        .ok_or_else(|| Error::Checkpoint(format!("tensor '{}' byte count overflows u64", tensor.name)))?;
    if tensor_bytes > MAX_TENSOR_BYTES {
        return Err(Error::Checkpoint(format!(
            "tensor '{}' is {tensor_bytes} bytes; limit is {MAX_TENSOR_BYTES}",
            tensor.name
        )));
    }
    let storage_bytes = u64::try_from(tensor.storage_size)
        .ok()
        .and_then(|size| size.checked_mul(u64::try_from(tensor.dtype.size_in_bytes()).ok()?))
        .ok_or_else(|| Error::Checkpoint(format!("tensor '{}' storage size overflows u64", tensor.name)))?;
    if tensor_bytes > storage_bytes {
        return Err(Error::Checkpoint(format!(
            "tensor '{}' requires {tensor_bytes} bytes but its storage declares {storage_bytes}",
            tensor.name
        )));
    }
    // A strided view, such as a transposed weight, must stay inside its storage.
    let last = dimensions.iter().zip(tensor.layout.stride()).try_fold(
        tensor.layout.start_offset(),
        |last, (&dimension, &stride)| {
            dimension
                .checked_sub(1)
                .map_or(Some(last), |steps| last.checked_add(steps.checked_mul(stride)?))
        },
    );
    if elements > 0 && last.is_none_or(|last| last >= tensor.storage_size) {
        return Err(Error::Checkpoint(format!(
            "tensor '{}' reaches past its storage",
            tensor.name
        )));
    }
    Ok(())
}

fn summarize_tensors(
    source_sha256: String,
    tensors: &[TensorInfo],
    config: Option<CheckpointModelConfig>,
) -> Result<CheckpointSummary> {
    let mut parameter_count = 0_u64;
    let mut tensor_bytes = 0_u64;
    let mut dtype_counts = BTreeMap::new();
    for tensor in tensors {
        let elements = checked_element_count(&tensor.name, tensor.layout.dims())?;
        parameter_count = parameter_count
            .checked_add(elements)
            .ok_or_else(|| Error::Checkpoint("checkpoint parameter count overflows u64".into()))?;
        let bytes = elements
            .checked_mul(
                u64::try_from(tensor.dtype.size_in_bytes()).map_err(|_| {
                    Error::Checkpoint(format!("tensor '{}' dtype size exceeds u64", tensor.name))
                })?,
            )
            .ok_or_else(|| Error::Checkpoint(format!("tensor '{}' byte count overflows u64", tensor.name)))?;
        tensor_bytes = tensor_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Checkpoint("checkpoint tensor byte count overflows u64".into()))?;
        *dtype_counts.entry(tensor.dtype.as_str().to_owned()).or_insert(0) += 1;
    }
    if tensor_bytes > MAX_TOTAL_TENSOR_BYTES {
        return Err(Error::Checkpoint(format!(
            "checkpoint tensors are {tensor_bytes} bytes; limit is {MAX_TOTAL_TENSOR_BYTES}"
        )));
    }

    let profile = identify_profile(
        &source_sha256,
        tensors.len(),
        parameter_count,
        tensor_bytes,
        config.as_ref(),
    );
    Ok(CheckpointSummary {
        source_sha256,
        profile,
        tensor_count: tensors.len(),
        parameter_count,
        tensor_bytes,
        dtype_counts,
        config,
        warnings: Vec::new(),
    })
}

fn checked_element_count(name: &str, dimensions: &[usize]) -> Result<u64> {
    dimensions.iter().try_fold(1_u64, |product, dimension| {
        let dimension = u64::try_from(*dimension)
            .map_err(|_| Error::Checkpoint(format!("tensor '{name}' dimension exceeds u64")))?;
        product
            .checked_mul(dimension)
            .ok_or_else(|| Error::Checkpoint(format!("tensor '{name}' element count overflows u64")))
    })
}

fn extract_model_config(config: &Object) -> Result<CheckpointModelConfig> {
    if !matches!(config, Object::Dict(_)) {
        return Err(Error::Checkpoint("top-level 'config' is not a mapping".into()));
    }
    Ok(CheckpointModelConfig {
        max_classes: Some(required_dict_u64(config, "max_classes")?),
        num_quantiles: Some(required_dict_u64(config, "num_quantiles")?),
        embed_dim: Some(required_dict_u64(config, "embed_dim")?),
        column_blocks: Some(required_dict_u64(config, "col_num_blocks")?),
        row_blocks: Some(required_dict_u64(config, "row_num_blocks")?),
        row_cls_tokens: Some(required_dict_u64(config, "row_num_cls")?),
        icl_blocks: Some(required_dict_u64(config, "icl_num_blocks")?),
    })
}

fn unique_dict_value<'a>(object: &'a Object, key: &str) -> Result<Option<&'a Object>> {
    let Object::Dict(entries) = object else {
        return Err(Error::Checkpoint("checkpoint root is not a mapping".into()));
    };
    let mut matches = entries.iter().filter_map(|(entry_key, value)| {
        matches!(entry_key, Object::Unicode(entry_key) if entry_key == key).then_some(value)
    });
    let value = matches.next();
    if matches.next().is_some() {
        return Err(Error::Checkpoint(format!(
            "checkpoint contains duplicate top-level '{key}' keys"
        )));
    }
    Ok(value)
}

fn dict_u64(object: &Object, key: &str) -> Result<Option<u64>> {
    let Object::Dict(entries) = object else {
        return Err(Error::Checkpoint("model config is not a mapping".into()));
    };
    let mut matches = entries.iter().filter_map(|(entry_key, value)| {
        matches!(entry_key, Object::Unicode(entry_key) if entry_key == key).then_some(value)
    });
    let Some(value) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(Error::Checkpoint(format!(
            "model config contains duplicate '{key}' keys"
        )));
    }
    let value = match value {
        Object::Int(value) => i64::from(*value),
        Object::Long(value) => *value,
        _ => {
            return Err(Error::Checkpoint(format!(
                "model config '{key}' is not an integer"
            )));
        }
    };
    Ok(Some(u64::try_from(value).map_err(|_| {
        Error::Checkpoint(format!("model config '{key}' is negative or too large"))
    })?))
}

fn required_dict_u64(object: &Object, key: &str) -> Result<u64> {
    dict_u64(object, key)?.ok_or_else(|| Error::Checkpoint(format!("model config '{key}' is missing")))
}

fn identify_profile(
    source_sha256: &str,
    tensor_count: usize,
    parameter_count: u64,
    tensor_bytes: u64,
    config: Option<&CheckpointModelConfig>,
) -> CheckpointProfile {
    match (
        source_sha256,
        config.and_then(CheckpointModelConfig::task),
        tensor_count,
        parameter_count,
        tensor_bytes,
    ) {
        (CLASSIFIER_SHA256, Some(TaskKind::Classifier), 391, 27_552_258, 110_209_032) => {
            CheckpointProfile::TabIclV2Classifier
        }
        (REGRESSOR_SHA256, Some(TaskKind::Regressor), 347, 28_544_991, 114_179_964) => {
            CheckpointProfile::TabIclV2Regressor
        }
        _ => CheckpointProfile::Unknown,
    }
}

#[derive(Debug, Clone, Copy)]
struct PickleValueShape {
    bytes: u64,
    depth: usize,
}

impl PickleValueShape {
    const SCALAR: Self = Self { bytes: 32, depth: 1 };

    fn string(length: u64) -> Result<Self> {
        Ok(Self {
            bytes: 32_u64
                .checked_add(length)
                .ok_or_else(|| Error::Checkpoint("pickle object size overflows u64".into()))?,
            depth: 1,
        })
    }

    fn container(values: impl IntoIterator<Item = Self>) -> Result<Self> {
        let mut bytes = 32_u64;
        let mut depth = 1_usize;
        for value in values {
            bytes = bytes
                .checked_add(value.bytes)
                .ok_or_else(|| Error::Checkpoint("pickle object size overflows u64".into()))?;
            depth = depth.max(
                value
                    .depth
                    .checked_add(1)
                    .ok_or_else(|| Error::Checkpoint("pickle object depth overflows usize".into()))?,
            );
        }
        if depth > MAX_PICKLE_OBJECT_DEPTH {
            return Err(Error::Checkpoint(format!(
                "pickle object depth exceeds {MAX_PICKLE_OBJECT_DEPTH}"
            )));
        }
        Ok(Self { bytes, depth })
    }
}

struct PicklePreflight {
    stack: Vec<PickleValueShape>,
    marks: Vec<usize>,
    memo: BTreeMap<u32, PickleValueShape>,
    next_memo: u32,
    live_bytes: u64,
}

impl PicklePreflight {
    fn new() -> Self {
        Self {
            stack: Vec::new(),
            marks: Vec::new(),
            memo: BTreeMap::new(),
            next_memo: 0,
            live_bytes: 0,
        }
    }

    fn push(&mut self, value: PickleValueShape) -> Result<()> {
        self.live_bytes = self
            .live_bytes
            .checked_add(value.bytes)
            .ok_or_else(|| Error::Checkpoint("pickle object size overflows u64".into()))?;
        self.check_limits()?;
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self, offset: usize) -> Result<PickleValueShape> {
        let value = self
            .stack
            .pop()
            .ok_or_else(|| Error::Checkpoint(format!("pickle stack underflow at byte {offset}")))?;
        self.live_bytes = self
            .live_bytes
            .checked_sub(value.bytes)
            .ok_or_else(|| Error::Checkpoint("pickle object accounting underflowed".into()))?;
        Ok(value)
    }

    fn pop_many(&mut self, count: usize, offset: usize) -> Result<Vec<PickleValueShape>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.pop(offset)?);
        }
        Ok(values)
    }

    fn mark(&mut self) -> Result<()> {
        self.marks.push(self.stack.len());
        self.push(PickleValueShape::SCALAR)
    }

    fn pop_mark(&mut self, offset: usize) -> Result<Vec<PickleValueShape>> {
        let mark = self.marks.pop().ok_or_else(|| {
            Error::Checkpoint(format!("pickle MARK is missing for opcode at byte {offset}"))
        })?;
        if mark >= self.stack.len() {
            return Err(Error::Checkpoint(format!(
                "pickle MARK at byte {offset} is invalid"
            )));
        }
        let values = self.stack.split_off(mark + 1);
        let marker = self
            .stack
            .pop()
            .ok_or_else(|| Error::Checkpoint(format!("pickle MARK at byte {offset} is missing")))?;
        let removed_bytes = values.iter().try_fold(marker.bytes, |total, value| {
            total
                .checked_add(value.bytes)
                .ok_or_else(|| Error::Checkpoint("pickle object size overflows u64".into()))
        })?;
        self.live_bytes = self
            .live_bytes
            .checked_sub(removed_bytes)
            .ok_or_else(|| Error::Checkpoint("pickle object accounting underflowed".into()))?;
        Ok(values)
    }

    fn combine(&mut self, pop: usize, offset: usize) -> Result<()> {
        let values = self.pop_many(pop, offset)?;
        self.push(PickleValueShape::container(values)?)
    }

    fn combine_mark(&mut self, offset: usize) -> Result<()> {
        let values = self.pop_mark(offset)?;
        self.push(PickleValueShape::container(values)?)
    }

    fn append(&mut self, count: usize, offset: usize) -> Result<()> {
        let values = self.pop_many(count, offset)?;
        let container = self.pop(offset)?;
        self.push(PickleValueShape::container(
            std::iter::once(container).chain(values),
        )?)
    }

    fn append_mark(&mut self, offset: usize) -> Result<()> {
        let values = self.pop_mark(offset)?;
        let container = self.pop(offset)?;
        self.push(PickleValueShape::container(
            std::iter::once(container).chain(values),
        )?)
    }

    fn memo_put(&mut self, id: u32, offset: usize) -> Result<()> {
        let value = *self
            .stack
            .last()
            .ok_or_else(|| Error::Checkpoint(format!("pickle memo write on empty stack at byte {offset}")))?;
        if let Some(previous) = self.memo.insert(id, value) {
            self.live_bytes = self
                .live_bytes
                .checked_sub(previous.bytes)
                .ok_or_else(|| Error::Checkpoint("pickle object accounting underflowed".into()))?;
        }
        self.live_bytes = self
            .live_bytes
            .checked_add(value.bytes)
            .ok_or_else(|| Error::Checkpoint("pickle object size overflows u64".into()))?;
        self.next_memo = self.next_memo.max(id.saturating_add(1));
        self.check_limits()
    }

    fn memoize(&mut self, offset: usize) -> Result<()> {
        let id = self.next_memo;
        self.next_memo = self
            .next_memo
            .checked_add(1)
            .ok_or_else(|| Error::Checkpoint("pickle memo index overflows u32".into()))?;
        self.memo_put(id, offset)
    }

    fn memo_get(&mut self, id: u32) -> Result<()> {
        let value = *self
            .memo
            .get(&id)
            .ok_or_else(|| Error::Checkpoint(format!("pickle references missing memo entry {id}")))?;
        self.push(value)
    }

    fn finish(&self, cursor_empty: bool, offset: usize) -> Result<()> {
        if self.stack.is_empty() {
            return Err(Error::Checkpoint(format!(
                "pickle STOP at byte {offset} has no result"
            )));
        }
        if !self.marks.is_empty() {
            return Err(Error::Checkpoint("pickle stopped with an open MARK".into()));
        }
        if !cursor_empty {
            return Err(Error::Checkpoint("pickle contains data after STOP".into()));
        }
        Ok(())
    }

    fn check_limits(&self) -> Result<()> {
        if self.stack.len() > MAX_PICKLE_STACK_DEPTH {
            return Err(Error::Checkpoint(format!(
                "pickle stack depth exceeds {MAX_PICKLE_STACK_DEPTH}"
            )));
        }
        if self.memo.len() > MAX_PICKLE_MEMO_ENTRIES {
            return Err(Error::Checkpoint(format!(
                "pickle memo has more than {MAX_PICKLE_MEMO_ENTRIES} entries"
            )));
        }
        if self.live_bytes > MAX_PICKLE_OBJECT_BYTES {
            return Err(Error::Checkpoint(format!(
                "pickle object allocation exceeds {MAX_PICKLE_OBJECT_BYTES} bytes"
            )));
        }
        Ok(())
    }
}

#[expect(clippy::too_many_lines, reason = "one flat match over the pickle opcodes")]
fn preflight_pickle(bytes: &[u8]) -> Result<()> {
    let mut cursor = PickleCursor::new(bytes);
    let mut state = PicklePreflight::new();
    let mut op_count = 0_usize;

    while !cursor.is_empty() {
        op_count = op_count
            .checked_add(1)
            .ok_or_else(|| Error::Checkpoint("pickle opcode count overflows usize".into()))?;
        if op_count > MAX_PICKLE_OPS {
            return Err(Error::Checkpoint(format!(
                "pickle has more than {MAX_PICKLE_OPS} opcodes"
            )));
        }
        let offset = cursor.position();
        let opcode = cursor.read_u8()?;
        match opcode {
            0x80 => {
                let protocol = cursor.read_u8()?;
                if !(2..=3).contains(&protocol) {
                    return Err(Error::Checkpoint(format!(
                        "pickle protocol {protocol} is unsupported"
                    )));
                }
            }
            b'c' => {
                let module = cursor.read_line()?;
                let name = cursor.read_line()?;
                if !supported_pickle_global(module, name) {
                    let module = String::from_utf8_lossy(module);
                    let name = String::from_utf8_lossy(name);
                    return Err(Error::Checkpoint(format!(
                        "pickle global '{module}.{name}' at byte {offset} is unsupported"
                    )));
                }
                state.push(PickleValueShape::string(
                    usize_to_u64(module.len())?.saturating_add(usize_to_u64(name.len())?),
                )?)?;
            }
            b'q' => {
                let id = u32::from(cursor.read_u8()?);
                state.memo_put(id, offset)?;
            }
            b'r' => {
                let id = cursor.read_u32_le()?;
                state.memo_put(id, offset)?;
            }
            0x94 => state.memoize(offset)?,
            b'h' => {
                let id = u32::from(cursor.read_u8()?);
                state.memo_get(id)?;
            }
            b'j' => {
                let id = cursor.read_u32_le()?;
                state.memo_get(id)?;
            }
            b'(' => state.mark()?,
            b')' | b'}' | b']' | b'N' | 0x88 | 0x89 => state.push(PickleValueShape::SCALAR)?,
            b'J' | b'G' => {
                cursor.skip(if opcode == b'J' { 4 } else { 8 })?;
                state.push(PickleValueShape::SCALAR)?;
            }
            b'K' => {
                cursor.skip(1)?;
                state.push(PickleValueShape::SCALAR)?;
            }
            b'M' => {
                cursor.skip(2)?;
                state.push(PickleValueShape::SCALAR)?;
            }
            b'X' => {
                let length = cursor.read_u32_le()?;
                cursor.skip(u64::from(length))?;
                state.push(PickleValueShape::string(u64::from(length))?)?;
            }
            0x8a => {
                let length = cursor.read_u8()?;
                if length > 8 {
                    return Err(Error::Checkpoint(format!(
                        "LONG1 at byte {offset} uses {length} bytes; limit is 8"
                    )));
                }
                cursor.skip(u64::from(length))?;
                state.push(PickleValueShape::string(u64::from(length))?)?;
            }
            b'Q' | 0x85 => state.combine(1, offset)?,
            b'R' | b'b' | 0x81 | 0x86 => state.combine(2, offset)?,
            0x87 => state.combine(3, offset)?,
            b'a' => state.append(1, offset)?,
            b's' => state.append(2, offset)?,
            b't' | b'd' => state.combine_mark(offset)?,
            b'e' | b'u' => state.append_mark(offset)?,
            b'.' => {
                state.finish(cursor.is_empty(), offset)?;
                return Ok(());
            }
            _ => {
                return Err(Error::Checkpoint(format!(
                    "pickle opcode 0x{opcode:02x} at byte {offset} is unsupported"
                )));
            }
        }
    }
    Err(Error::Checkpoint("pickle stream ended without STOP".into()))
}

fn validate_object_tree(
    object: &Object,
    depth: usize,
    object_bytes: &mut u64,
    tensor_storage_keys: &mut BTreeSet<String>,
) -> Result<()> {
    if depth > MAX_PICKLE_OBJECT_DEPTH {
        return Err(Error::Checkpoint(format!(
            "pickle object depth exceeds {MAX_PICKLE_OBJECT_DEPTH}"
        )));
    }
    add_object_bytes(object_bytes, 32)?;
    match object {
        Object::Class {
            module_name,
            class_name,
        } => {
            add_object_bytes(object_bytes, usize_to_u64(module_name.len())?)?;
            add_object_bytes(object_bytes, usize_to_u64(class_name.len())?)?;
        }
        Object::Unicode(value) => add_object_bytes(object_bytes, usize_to_u64(value.len())?)?,
        Object::Tuple(values) | Object::List(values) => {
            for value in values {
                validate_object_tree(value, depth + 1, object_bytes, tensor_storage_keys)?;
            }
        }
        Object::Dict(entries) => {
            for (key, value) in entries {
                validate_object_tree(key, depth + 1, object_bytes, tensor_storage_keys)?;
                validate_object_tree(value, depth + 1, object_bytes, tensor_storage_keys)?;
            }
        }
        Object::Reduce { callable, args } | Object::Build { callable, args } => {
            validate_object_tree(callable, depth + 1, object_bytes, tensor_storage_keys)?;
            validate_object_tree(args, depth + 1, object_bytes, tensor_storage_keys)?;
        }
        Object::PersistentLoad(value) => {
            validate_storage_descriptor(value, tensor_storage_keys)?;
            validate_object_tree(value, depth + 1, object_bytes, tensor_storage_keys)?;
        }
        Object::Int(_) | Object::Long(_) | Object::Float(_) | Object::Bool(_) | Object::None => {}
        Object::Mark => return Err(Error::Checkpoint("pickle object contains a residual MARK".into())),
    }
    Ok(())
}

fn supported_pickle_global(module: &[u8], name: &[u8]) -> bool {
    matches!(
        (module, name),
        (b"collections", b"OrderedDict")
            | (b"torch._utils", b"_rebuild_tensor_v2")
            | (
                b"torch",
                b"FloatStorage"
                    | b"DoubleStorage"
                    | b"HalfStorage"
                    | b"BFloat16Storage"
                    | b"ByteStorage"
                    | b"LongStorage"
            )
    )
}

fn validate_storage_descriptor(object: &Object, storage_keys: &mut BTreeSet<String>) -> Result<()> {
    let Object::Tuple(fields) = object else {
        return Err(Error::Checkpoint("persistent ID is not a tuple".into()));
    };
    if fields.len() < 5 {
        return Err(Error::Checkpoint(
            "persistent ID has fewer than five fields".into(),
        ));
    }
    if !matches!(&fields[0], Object::Unicode(kind) if kind == "storage") {
        return Err(Error::Checkpoint("persistent ID kind is not 'storage'".into()));
    }
    let Object::Unicode(key) = &fields[2] else {
        return Err(Error::Checkpoint("persistent storage key is not a string".into()));
    };
    storage_keys.insert(key.clone());
    if storage_keys.len() > MAX_TENSOR_COUNT {
        return Err(Error::Checkpoint(format!(
            "pickle references more than {MAX_TENSOR_COUNT} tensor storages"
        )));
    }
    Ok(())
}

fn add_object_bytes(total: &mut u64, bytes: u64) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| Error::Checkpoint("pickle object size overflows u64".into()))?;
    if *total > MAX_PICKLE_OBJECT_BYTES {
        return Err(Error::Checkpoint(format!(
            "parsed pickle objects exceed {MAX_PICKLE_OBJECT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn usize_to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::Checkpoint("pickle object size exceeds u64".into()))
}

struct PickleCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> PickleCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn position(&self) -> usize {
        self.position
    }

    fn read_u8(&mut self) -> Result<u8> {
        let value = *self
            .bytes
            .get(self.position)
            .ok_or_else(|| Error::Checkpoint("unexpected end of pickle".into()))?;
        self.position += 1;
        Ok(value)
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        let end = self
            .position
            .checked_add(4)
            .ok_or_else(|| Error::Checkpoint("pickle offset overflows usize".into()))?;
        let bytes: [u8; 4] = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::Checkpoint("unexpected end of pickle".into()))?
            .try_into()
            .map_err(|_| Error::Checkpoint("invalid pickle integer".into()))?;
        self.position = end;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_line(&mut self) -> Result<&'a [u8]> {
        let start = self.position;
        let relative_end = self.bytes[self.position..]
            .iter()
            .position(|byte| *byte == b'\n')
            .ok_or_else(|| Error::Checkpoint("unterminated pickle GLOBAL line".into()))?;
        self.position = self
            .position
            .checked_add(relative_end + 1)
            .ok_or_else(|| Error::Checkpoint("pickle offset overflows usize".into()))?;
        Ok(&self.bytes[start..start + relative_end])
    }

    fn skip(&mut self, count: u64) -> Result<()> {
        let count = usize::try_from(count)
            .map_err(|_| Error::Checkpoint("pickle field length exceeds usize".into()))?;
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| Error::Checkpoint("pickle offset overflows usize".into()))?;
        if end > self.bytes.len() {
            return Err(Error::Checkpoint("unexpected end of pickle".into()));
        }
        self.position = end;
        Ok(())
    }
}

fn sha256_reader(reader: &mut impl Read) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let bytes_read = reader.read(&mut buffer).map_err(checkpoint_error)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let mut encoded = String::with_capacity(64);
    for byte in hasher.finalize() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    Ok(encoded)
}

fn checkpoint_error(error: impl fmt::Display) -> Error {
    Error::Checkpoint(error.to_string())
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Layout};

    use super::*;

    #[test]
    fn strided_tensors_read_in_row_major_order() {
        // A [2, 3] view of a [3, 2] buffer, transposed as torch saves `weight.T`.
        let transposed = Layout::new(Shape::from_dims(&[2, 3]), vec![1, 2], 0);
        let order: Vec<_> = (0..6)
            .map(|element| storage_index(&transposed, element))
            .collect();
        assert_eq!(order, [0, 2, 4, 1, 3, 5]);
        let offset = Layout::contiguous_with_offset(Shape::from_dims(&[2, 2]), 3);
        assert_eq!(storage_index(&offset, 3), 6);
    }

    #[test]
    fn recognizes_classifier_metadata() {
        let tensors = classifier_tensor_manifest();
        let config = CheckpointModelConfig {
            max_classes: Some(10),
            num_quantiles: Some(999),
            embed_dim: Some(128),
            column_blocks: Some(3),
            row_blocks: Some(3),
            row_cls_tokens: Some(4),
            icl_blocks: Some(12),
        };
        let summary = summarize_tensors(CLASSIFIER_SHA256.into(), &tensors, Some(config))
            .expect("valid checkpoint summary");

        assert_eq!(summary.profile, CheckpointProfile::TabIclV2Classifier);
        assert_eq!(summary.parameter_count, 27_552_258);
        assert_eq!(summary.tensor_bytes, 110_209_032);
        assert_eq!(summary.dtype_counts.get("f32"), Some(&391));
        assert_eq!(
            summary.config.and_then(|config| config.task()),
            Some(TaskKind::Classifier)
        );
    }

    #[test]
    fn extracts_integer_config_values() {
        let config = Object::Dict(vec![
            (Object::Unicode("max_classes".into()), Object::Int(0)),
            (Object::Unicode("num_quantiles".into()), Object::Long(999)),
            (Object::Unicode("embed_dim".into()), Object::Int(128)),
            (Object::Unicode("col_num_blocks".into()), Object::Int(3)),
            (Object::Unicode("row_num_blocks".into()), Object::Int(3)),
            (Object::Unicode("row_num_cls".into()), Object::Int(4)),
            (Object::Unicode("icl_num_blocks".into()), Object::Int(12)),
        ]);

        let extracted = extract_model_config(&config).expect("valid config");
        assert_eq!(extracted.max_classes, Some(0));
        assert_eq!(extracted.num_quantiles, Some(999));
        assert_eq!(extracted.task(), Some(TaskKind::Regressor));
    }

    #[test]
    fn strided_tensors_must_stay_inside_their_storage() {
        let tensor = |layout, storage_size| TensorInfo {
            name: "weight".into(),
            dtype: DType::F32,
            layout,
            path: "model/data/0".into(),
            storage_size,
        };
        let transposed = || Layout::new(vec![2, 2].into(), vec![1, 2], 0);
        assert!(validate_tensor(&tensor(transposed(), 4)).is_ok());
        assert!(validate_tensor(&tensor(Layout::new(vec![2, 2].into(), vec![1, 3], 0), 4)).is_err());
        assert!(validate_tensor(&tensor(Layout::new(vec![2, 2].into(), vec![2, 1], 1), 4)).is_err());
    }

    #[test]
    fn rejects_oversized_long_before_candle_parses_it() {
        let pickle = [0x80, 2, 0x8a, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, b'.'];
        assert!(preflight_pickle(&pickle).is_err());
    }

    #[test]
    fn rejects_memo_clone_allocation_before_candle_parses_it() {
        let string_length = 1024_u32;
        let mut pickle = vec![0x80, 2, b'X'];
        pickle.extend(string_length.to_le_bytes());
        pickle.resize(pickle.len() + string_length as usize, b'x');
        pickle.extend([b'q', 0]);
        for _ in 0..=(MAX_PICKLE_OBJECT_BYTES / u64::from(string_length)) {
            pickle.extend([b'h', 0]);
        }
        pickle.push(b'.');

        assert!(preflight_pickle(&pickle).is_err());
    }

    #[test]
    fn rejects_deep_object_before_candle_parses_it() {
        let mut pickle = vec![0x80, 2, b'N'];
        pickle.extend(std::iter::repeat_n(0x85, MAX_PICKLE_OBJECT_DEPTH));
        pickle.push(b'.');

        assert!(preflight_pickle(&pickle).is_err());
    }

    #[test]
    fn profile_requires_official_hash() {
        let tensors = classifier_tensor_manifest();
        let config = CheckpointModelConfig {
            max_classes: Some(10),
            num_quantiles: Some(999),
            embed_dim: Some(128),
            column_blocks: Some(3),
            row_blocks: Some(3),
            row_cls_tokens: Some(4),
            icl_blocks: Some(12),
        };

        let summary = summarize_tensors("0".repeat(64), &tensors, Some(config))
            .expect("valid custom checkpoint summary");
        assert_eq!(summary.profile, CheckpointProfile::Unknown);
    }

    fn classifier_tensor_manifest() -> Vec<TensorInfo> {
        let mut tensors = Vec::with_capacity(391);
        tensors.push(tensor("large", 27_551_868));
        for index in 0..390 {
            tensors.push(tensor(&format!("scalar.{index}"), 1));
        }
        tensors
    }

    fn tensor(name: &str, elements: usize) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype: DType::F32,
            layout: Layout::contiguous(vec![elements]),
            path: format!("model/data/{name}"),
            storage_size: elements,
        }
    }
}

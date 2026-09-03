//! Checked static-context artifacts for exact exported `TabICLv2` cache tensors.
//!
//! The format is separate from model artifacts. It stores fixed context metadata,
//! class and feature permutations, named little-endian cache tensors, and a
//! trailing SHA-256 digest. Version 1 stores FP32. Version 2 adds explicit FP32
//! and BF16 tensor dtypes. Version 3 adds source-tile, dimension-major ICL K.

use std::{collections::BTreeMap, fs::File, path::Path, sync::Arc};

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Magic bytes at the start of a static-context artifact.
pub const STATIC_CONTEXT_MAGIC: [u8; 8] = *b"IRONICL\0";
/// Latest supported static-context artifact format version.
pub const STATIC_CONTEXT_VERSION: u32 = 3;

const STATIC_CONTEXT_KIND: u32 = 1;
const STATIC_CONTEXT_FP32_VERSION: u32 = 1;
const STATIC_CONTEXT_BF16_VERSION: u32 = 2;
const TENSOR_DTYPE_FP32: u16 = 1;
const TENSOR_DTYPE_BF16: u16 = 2;
const HEADER_BYTES: usize = 164;
const CHECKSUM_BYTES: usize = 32;
const TENSOR_ALIGNMENT: usize = 64;
const MAX_ARTIFACT_BYTES: usize = 48 * 1024 * 1024 * 1024;
const MAX_OWNED_ARTIFACT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_DIRECTORY_BYTES: usize = 4 * 1024 * 1024;
const MAX_TENSOR_COUNT: usize = 4096;
const MAX_TENSOR_RANK: usize = 16;
const MAX_TENSOR_NAME_BYTES: usize = 4096;
const MAX_TENSOR_BYTES: usize = 4 * 1024 * 1024 * 1024;
const MAX_TOTAL_TENSOR_BYTES: usize = 48 * 1024 * 1024 * 1024;
const MAX_ENSEMBLE_COUNT: usize = 4096;
const MAX_TRAINING_ROWS: usize = 1_000_000;
const MAX_FEATURE_COUNT: usize = 65_536;
const MAX_CLASS_COUNT: usize = 4096;

/// Metadata that identifies and describes one static training context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticContextMetadata {
    /// SHA-256 of the source checkpoint recorded by the model artifact.
    pub model_sha256: [u8; 32],
    /// SHA-256 identity of the source static context and export inputs.
    pub context_sha256: [u8; 32],
    /// Ensemble count `B`.
    pub ensemble_count: usize,
    /// Training-row count `N`.
    pub training_rows: usize,
    /// Feature count `H`.
    pub feature_count: usize,
    /// Class count.
    pub class_count: usize,
    /// Estimator-major class orders with shape `[B, class_count]`.
    pub class_permutations: Vec<u64>,
    /// Estimator-major feature orders with shape `[B, H]`.
    pub feature_permutations: Vec<u64>,
}

/// An owned named FP32 tensor in logical row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticContextTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub values: Vec<f32>,
}

/// Borrowed static-context tensor values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StaticContextTensorData<'a> {
    Fp32(&'a [f32]),
    Bf16(&'a [u16]),
}

impl StaticContextTensorData<'_> {
    #[must_use]
    pub const fn len(self) -> usize {
        match self {
            Self::Fp32(values) => values.len(),
            Self::Bf16(values) => values.len(),
        }
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn slice(self, start: usize, length: usize) -> Self {
        match self {
            Self::Fp32(values) => Self::Fp32(&values[start..][..length]),
            Self::Bf16(values) => Self::Bf16(&values[start..][..length]),
        }
    }
}

/// A borrowed named tensor from owned or mapped context storage.
#[derive(Debug, Clone, Copy)]
pub struct StaticContextTensorRef<'a> {
    pub name: &'a str,
    pub dimensions: &'a [u64],
    pub data: StaticContextTensorData<'a>,
}

impl<'a> StaticContextTensorRef<'a> {
    #[must_use]
    pub const fn fp32_values(self) -> Option<&'a [f32]> {
        match self.data {
            StaticContextTensorData::Fp32(values) => Some(values),
            StaticContextTensorData::Bf16(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum StaticContextTensorValues {
    Owned(Vec<f32>),
    MappedFp32 { byte_offset: usize, byte_length: usize },
    MappedBf16 { byte_offset: usize, byte_length: usize },
}

#[derive(Debug, Clone, PartialEq)]
struct StaticContextTensorRecord {
    dimensions: Vec<u64>,
    values: StaticContextTensorValues,
}

#[derive(Debug, Clone)]
enum StaticContextStorage {
    Owned,
    Mapped(Arc<Mmap>),
}

/// A verified static context backed by owned or read-only mapped tensor storage.
#[derive(Debug, Clone)]
pub struct StaticContextArtifact {
    version: u32,
    metadata: StaticContextMetadata,
    tensors: BTreeMap<String, StaticContextTensorRecord>,
    storage: StaticContextStorage,
}

impl PartialEq for StaticContextArtifact {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.metadata == other.metadata
            && self.tensors.keys().eq(other.tensors.keys())
            && self.tensors.keys().all(|name| {
                let left = self.tensor(name).expect("existing left tensor");
                let right = other.tensor(name).expect("existing right tensor");
                left.dimensions == right.dimensions && left.data == right.data
            })
    }
}

impl StaticContextArtifact {
    /// Validate and own static-context metadata and exact FP32 cache tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when metadata, names, shapes, or values are invalid.
    pub fn new(metadata: StaticContextMetadata, tensors: Vec<StaticContextTensor>) -> Result<Self> {
        validate_metadata(&metadata)?;
        if tensors.is_empty() || tensors.len() > MAX_TENSOR_COUNT {
            return Err(artifact_error(format!(
                "static context has {} tensors; expected 1..={MAX_TENSOR_COUNT}",
                tensors.len()
            )));
        }
        let mut named = BTreeMap::new();
        let mut total_bytes = 0_usize;
        for tensor in tensors {
            validate_tensor(&tensor)?;
            total_bytes = total_bytes
                .checked_add(
                    tensor
                        .values
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| artifact_error("tensor byte length overflows usize"))?,
                )
                .ok_or_else(|| artifact_error("total tensor byte length overflows usize"))?;
            if named
                .insert(
                    tensor.name.clone(),
                    StaticContextTensorRecord {
                        dimensions: tensor.dimensions,
                        values: StaticContextTensorValues::Owned(tensor.values),
                    },
                )
                .is_some()
            {
                return Err(artifact_error("static context tensor names must be unique"));
            }
        }
        if total_bytes > MAX_TOTAL_TENSOR_BYTES {
            return Err(artifact_error(
                "static context tensor payload exceeds the resource limit",
            ));
        }
        Ok(Self {
            version: STATIC_CONTEXT_FP32_VERSION,
            metadata,
            tensors: named,
            storage: StaticContextStorage::Owned,
        })
    }

    /// Decode, verify, and own a complete static-context artifact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] for malformed, corrupt, or invalid input.
    #[allow(clippy::too_many_lines)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES || bytes.len() > MAX_OWNED_ARTIFACT_BYTES {
            return Err(artifact_error(
                "static context length is outside the resource limit",
            ));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        if sha256(&bytes[..checksum_offset]).as_slice() != &bytes[checksum_offset..] {
            return Err(artifact_error("static context checksum does not match"));
        }
        let mut header = ByteReader::new(&bytes[..HEADER_BYTES]);
        if header.read_array::<8>()? != STATIC_CONTEXT_MAGIC {
            return Err(artifact_error("static context magic does not match"));
        }
        let version = header.read_u32()?;
        if version != STATIC_CONTEXT_FP32_VERSION {
            return Err(artifact_error(format!(
                "owned static context version {version} is unsupported; expected {STATIC_CONTEXT_FP32_VERSION}"
            )));
        }
        if header.read_u32()? != STATIC_CONTEXT_KIND {
            return Err(artifact_error("artifact kind is not a static context"));
        }
        let model_sha256 = header.read_array()?;
        let context_sha256 = header.read_array()?;
        let ensemble_count = to_usize(header.read_u64()?, "ensemble count")?;
        let training_rows = to_usize(header.read_u64()?, "training rows")?;
        let feature_count = to_usize(header.read_u64()?, "feature count")?;
        let class_count = to_usize(header.read_u64()?, "class count")?;
        let architecture = [
            header.read_u32()?,
            header.read_u32()?,
            header.read_u32()?,
            header.read_u32()?,
            header.read_u32()?,
        ];
        if architecture != [3, 12, 8, 16, 64] {
            return Err(artifact_error(
                "static context architecture metadata is unsupported",
            ));
        }
        let tensor_count = to_usize(u64::from(header.read_u32()?), "tensor count")?;
        let class_permutation_count = to_usize(u64::from(header.read_u32()?), "class permutation count")?;
        let feature_permutation_count = to_usize(u64::from(header.read_u32()?), "feature permutation count")?;
        let directory_length = to_usize(header.read_u64()?, "directory length")?;
        let payload_length = to_usize(header.read_u64()?, "payload length")?;
        if header.read_u32()? != 0 || !header.is_empty() {
            return Err(artifact_error("static context header reserved bytes are invalid"));
        }
        if tensor_count == 0
            || tensor_count > MAX_TENSOR_COUNT
            || directory_length > MAX_DIRECTORY_BYTES
            || payload_length > MAX_TOTAL_TENSOR_BYTES
        {
            return Err(artifact_error(
                "static context section exceeds the resource limit",
            ));
        }
        let expected_class_permutations = ensemble_count
            .checked_mul(class_count)
            .ok_or_else(|| artifact_error("class permutation count overflows usize"))?;
        let expected_feature_permutations = ensemble_count
            .checked_mul(feature_count)
            .ok_or_else(|| artifact_error("feature permutation count overflows usize"))?;
        if class_permutation_count != expected_class_permutations
            || feature_permutation_count != expected_feature_permutations
        {
            return Err(artifact_error(
                "static context permutation lengths do not match metadata",
            ));
        }
        let permutation_bytes = class_permutation_count
            .checked_add(feature_permutation_count)
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| artifact_error("permutation length overflows usize"))?;
        let permutations_end = HEADER_BYTES
            .checked_add(permutation_bytes)
            .ok_or_else(|| artifact_error("permutation section end overflows usize"))?;
        let directory_end = permutations_end
            .checked_add(directory_length)
            .ok_or_else(|| artifact_error("directory end overflows usize"))?;
        let payload_start = align_up(directory_end, TENSOR_ALIGNMENT)?;
        let payload_end = payload_start
            .checked_add(payload_length)
            .ok_or_else(|| artifact_error("payload end overflows usize"))?;
        if payload_end != checksum_offset || directory_end > payload_start {
            return Err(artifact_error("static context section offsets are inconsistent"));
        }
        if bytes[directory_end..payload_start].iter().any(|byte| *byte != 0) {
            return Err(artifact_error("static context alignment padding is not zero"));
        }
        let mut permutations = ByteReader::new(&bytes[HEADER_BYTES..permutations_end]);
        let class_permutations = read_permutation(&mut permutations, class_permutation_count)?;
        let feature_permutations = read_permutation(&mut permutations, feature_permutation_count)?;
        let metadata = StaticContextMetadata {
            model_sha256,
            context_sha256,
            ensemble_count,
            training_rows,
            feature_count,
            class_count,
            class_permutations,
            feature_permutations,
        };
        validate_metadata(&metadata)?;

        let mut directory = ByteReader::new(&bytes[permutations_end..directory_end]);
        let mut previous_name = None;
        let mut ranges = Vec::with_capacity(tensor_count);
        let mut tensors = BTreeMap::new();
        let mut total_bytes = 0_usize;
        for _ in 0..tensor_count {
            let name_length = to_usize(u64::from(directory.read_u32()?), "tensor name length")?;
            let rank = usize::from(directory.read_u16()?);
            if name_length == 0
                || name_length > MAX_TENSOR_NAME_BYTES
                || rank == 0
                || rank > MAX_TENSOR_RANK
                || directory.read_u16()? != 0
            {
                return Err(artifact_error("static context tensor directory entry is invalid"));
            }
            let offset = to_usize(directory.read_u64()?, "tensor payload offset")?;
            let byte_length = to_usize(directory.read_u64()?, "tensor byte length")?;
            if offset % TENSOR_ALIGNMENT != 0 || byte_length > MAX_TENSOR_BYTES {
                return Err(artifact_error(
                    "static context tensor offset or length is invalid",
                ));
            }
            let mut dimensions = Vec::with_capacity(rank);
            for _ in 0..rank {
                dimensions.push(directory.read_u64()?);
            }
            let name = std::str::from_utf8(directory.read_bytes(name_length)?)
                .map_err(|_| artifact_error("static context tensor name is not UTF-8"))?
                .to_owned();
            if previous_name
                .as_ref()
                .is_some_and(|previous: &String| previous >= &name)
            {
                return Err(artifact_error(
                    "static context tensor names are not unique and sorted",
                ));
            }
            previous_name = Some(name.clone());
            let elements = checked_elements(&dimensions)?;
            let expected_bytes = elements
                .checked_mul(4)
                .ok_or_else(|| artifact_error("tensor byte length overflows usize"))?;
            if byte_length != expected_bytes {
                return Err(artifact_error(format!(
                    "static context tensor '{name}' byte length does not match its shape"
                )));
            }
            let end = offset
                .checked_add(byte_length)
                .ok_or_else(|| artifact_error("tensor payload end overflows usize"))?;
            if end > payload_length {
                return Err(artifact_error(format!(
                    "static context tensor '{name}' exceeds the payload section"
                )));
            }
            total_bytes = total_bytes
                .checked_add(byte_length)
                .ok_or_else(|| artifact_error("total tensor byte length overflows usize"))?;
            ranges.push((offset, end));
            let absolute_start = payload_start
                .checked_add(offset)
                .ok_or_else(|| artifact_error("absolute tensor payload start overflows usize"))?;
            let absolute_end = payload_start
                .checked_add(end)
                .ok_or_else(|| artifact_error("absolute tensor payload end overflows usize"))?;
            let payload = bytes
                .get(absolute_start..absolute_end)
                .ok_or_else(|| artifact_error("static context tensor payload is truncated"))?;
            let values = decode_f32(payload)?;
            tensors.insert(
                name,
                StaticContextTensorRecord {
                    dimensions,
                    values: StaticContextTensorValues::Owned(values),
                },
            );
        }
        if !directory.is_empty() {
            return Err(artifact_error(
                "static context tensor directory has trailing bytes",
            ));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) || total_bytes > MAX_TOTAL_TENSOR_BYTES {
            return Err(artifact_error("static context tensor payload ranges are invalid"));
        }
        Ok(Self {
            version: STATIC_CONTEXT_FP32_VERSION,
            metadata,
            tensors,
            storage: StaticContextStorage::Owned,
        })
    }

    /// Read, decode, and verify an artifact from a bounded file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the file cannot be read or is invalid.
    pub fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(artifact_error)?;
        let length = to_usize(
            file.metadata().map_err(artifact_error)?.len(),
            "static context length",
        )?;
        if length > MAX_ARTIFACT_BYTES {
            return Err(artifact_error("static context exceeds the resource limit"));
        }
        Self::read_mapped_file(&file, length)
    }

    /// Map, verify, and borrow tensors directly from an immutable artifact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the file or mapped tensor layout is invalid.
    pub fn read_mapped(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(artifact_error)?;
        let length = to_usize(
            file.metadata().map_err(artifact_error)?.len(),
            "static context length",
        )?;
        Self::read_mapped_file(&file, length)
    }

    #[allow(unsafe_code, clippy::too_many_lines)]
    fn read_mapped_file(file: &File, length: usize) -> Result<Self> {
        if !cfg!(target_endian = "little") {
            return Err(artifact_error(
                "mapped FP32 contexts require a little-endian host",
            ));
        }
        if !(HEADER_BYTES + CHECKSUM_BYTES..=MAX_ARTIFACT_BYTES).contains(&length) {
            return Err(artifact_error(
                "static context length is outside the resource limit",
            ));
        }
        // SAFETY: the mapping is read-only and retained by the artifact for every
        // tensor borrow. Serving artifacts must remain immutable while mapped.
        let mapping = unsafe { MmapOptions::new().map(file) }.map_err(artifact_error)?;
        let checksum_offset = mapping.len() - CHECKSUM_BYTES;
        if sha256(&mapping[..checksum_offset]).as_slice() != &mapping[checksum_offset..] {
            return Err(artifact_error("static context checksum does not match"));
        }
        let mut header = ByteReader::new(&mapping[..HEADER_BYTES]);
        if header.read_array::<8>()? != STATIC_CONTEXT_MAGIC {
            return Err(artifact_error("static context magic does not match"));
        }
        let version = header.read_u32()?;
        if !matches!(
            version,
            STATIC_CONTEXT_FP32_VERSION | STATIC_CONTEXT_BF16_VERSION | STATIC_CONTEXT_VERSION
        ) {
            return Err(artifact_error(format!(
                "static context version {version} is unsupported"
            )));
        }
        if header.read_u32()? != STATIC_CONTEXT_KIND {
            return Err(artifact_error("artifact kind is not a static context"));
        }
        let model_sha256 = header.read_array()?;
        let context_sha256 = header.read_array()?;
        let ensemble_count = to_usize(header.read_u64()?, "ensemble count")?;
        let training_rows = to_usize(header.read_u64()?, "training rows")?;
        let feature_count = to_usize(header.read_u64()?, "feature count")?;
        let class_count = to_usize(header.read_u64()?, "class count")?;
        let architecture = [
            header.read_u32()?,
            header.read_u32()?,
            header.read_u32()?,
            header.read_u32()?,
            header.read_u32()?,
        ];
        if architecture != [3, 12, 8, 16, 64] {
            return Err(artifact_error(
                "static context architecture metadata is unsupported",
            ));
        }
        let tensor_count = to_usize(u64::from(header.read_u32()?), "tensor count")?;
        let class_permutation_count = to_usize(u64::from(header.read_u32()?), "class permutation count")?;
        let feature_permutation_count = to_usize(u64::from(header.read_u32()?), "feature permutation count")?;
        let directory_length = to_usize(header.read_u64()?, "directory length")?;
        let payload_length = to_usize(header.read_u64()?, "payload length")?;
        if header.read_u32()? != 0 || !header.is_empty() {
            return Err(artifact_error("static context header reserved bytes are invalid"));
        }
        if tensor_count == 0
            || tensor_count > MAX_TENSOR_COUNT
            || directory_length > MAX_DIRECTORY_BYTES
            || payload_length > MAX_TOTAL_TENSOR_BYTES
        {
            return Err(artifact_error(
                "static context section exceeds the resource limit",
            ));
        }
        let expected_class_permutations = ensemble_count
            .checked_mul(class_count)
            .ok_or_else(|| artifact_error("class permutation count overflows usize"))?;
        let expected_feature_permutations = ensemble_count
            .checked_mul(feature_count)
            .ok_or_else(|| artifact_error("feature permutation count overflows usize"))?;
        if class_permutation_count != expected_class_permutations
            || feature_permutation_count != expected_feature_permutations
        {
            return Err(artifact_error(
                "static context permutation lengths do not match metadata",
            ));
        }
        let permutation_bytes = class_permutation_count
            .checked_add(feature_permutation_count)
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| artifact_error("permutation length overflows usize"))?;
        let permutations_end = HEADER_BYTES
            .checked_add(permutation_bytes)
            .ok_or_else(|| artifact_error("permutation section end overflows usize"))?;
        let directory_end = permutations_end
            .checked_add(directory_length)
            .ok_or_else(|| artifact_error("directory end overflows usize"))?;
        let payload_start = align_up(directory_end, TENSOR_ALIGNMENT)?;
        let payload_end = payload_start
            .checked_add(payload_length)
            .ok_or_else(|| artifact_error("payload end overflows usize"))?;
        if payload_end != checksum_offset || directory_end > payload_start {
            return Err(artifact_error("static context section offsets are inconsistent"));
        }
        if mapping[directory_end..payload_start]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(artifact_error("static context alignment padding is not zero"));
        }
        let mut permutations = ByteReader::new(&mapping[HEADER_BYTES..permutations_end]);
        let class_permutations = read_permutation(&mut permutations, class_permutation_count)?;
        let feature_permutations = read_permutation(&mut permutations, feature_permutation_count)?;
        let metadata = StaticContextMetadata {
            model_sha256,
            context_sha256,
            ensemble_count,
            training_rows,
            feature_count,
            class_count,
            class_permutations,
            feature_permutations,
        };
        validate_metadata(&metadata)?;

        let mut directory = ByteReader::new(&mapping[permutations_end..directory_end]);
        let mut previous_name = None;
        let mut ranges = Vec::with_capacity(tensor_count);
        let mut tensors = BTreeMap::new();
        let mut total_bytes = 0_usize;
        for _ in 0..tensor_count {
            let name_length = to_usize(u64::from(directory.read_u32()?), "tensor name length")?;
            let rank = usize::from(directory.read_u16()?);
            let dtype = directory.read_u16()?;
            if name_length == 0
                || name_length > MAX_TENSOR_NAME_BYTES
                || rank == 0
                || rank > MAX_TENSOR_RANK
                || (version == STATIC_CONTEXT_FP32_VERSION && dtype != 0)
                || (version != STATIC_CONTEXT_FP32_VERSION
                    && !matches!(dtype, TENSOR_DTYPE_FP32 | TENSOR_DTYPE_BF16))
            {
                return Err(artifact_error("static context tensor directory entry is invalid"));
            }
            let offset = to_usize(directory.read_u64()?, "tensor payload offset")?;
            let byte_length = to_usize(directory.read_u64()?, "tensor byte length")?;
            if offset % TENSOR_ALIGNMENT != 0 || byte_length > MAX_TENSOR_BYTES {
                return Err(artifact_error(
                    "static context tensor offset or length is invalid",
                ));
            }
            let mut dimensions = Vec::with_capacity(rank);
            for _ in 0..rank {
                dimensions.push(directory.read_u64()?);
            }
            let name = std::str::from_utf8(directory.read_bytes(name_length)?)
                .map_err(|_| artifact_error("static context tensor name is not UTF-8"))?
                .to_owned();
            if previous_name
                .as_ref()
                .is_some_and(|previous: &String| previous >= &name)
            {
                return Err(artifact_error(
                    "static context tensor names are not unique and sorted",
                ));
            }
            previous_name = Some(name.clone());
            let elements = checked_elements(&dimensions)?;
            let element_bytes = if version == STATIC_CONTEXT_FP32_VERSION || dtype == TENSOR_DTYPE_FP32 {
                4
            } else {
                2
            };
            let expected_bytes = elements
                .checked_mul(element_bytes)
                .ok_or_else(|| artifact_error("tensor byte length overflows usize"))?;
            if byte_length != expected_bytes {
                return Err(artifact_error(format!(
                    "static context tensor '{name}' byte length does not match its shape"
                )));
            }
            let end = offset
                .checked_add(byte_length)
                .ok_or_else(|| artifact_error("tensor payload end overflows usize"))?;
            if end > payload_length {
                return Err(artifact_error(format!(
                    "static context tensor '{name}' exceeds the payload section"
                )));
            }
            total_bytes = total_bytes
                .checked_add(byte_length)
                .ok_or_else(|| artifact_error("total tensor byte length overflows usize"))?;
            ranges.push((offset, end));
            let byte_offset = payload_start
                .checked_add(offset)
                .ok_or_else(|| artifact_error("absolute tensor payload start overflows usize"))?;
            let bytes = mapping
                .get(byte_offset..byte_offset + byte_length)
                .ok_or_else(|| artifact_error("static context tensor payload is truncated"))?;
            let values = if element_bytes == 4 {
                let values = bytemuck::try_cast_slice::<u8, f32>(bytes)
                    .map_err(|_| artifact_error("mapped FP32 tensor alignment is invalid"))?;
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(artifact_error(
                        "static context tensor contains a non-finite value",
                    ));
                }
                StaticContextTensorValues::MappedFp32 {
                    byte_offset,
                    byte_length,
                }
            } else {
                let values = bytemuck::try_cast_slice::<u8, u16>(bytes)
                    .map_err(|_| artifact_error("mapped BF16 tensor alignment is invalid"))?;
                if values
                    .iter()
                    .any(|value| !f32::from_bits(u32::from(*value) << 16).is_finite())
                {
                    return Err(artifact_error(
                        "static context tensor contains a non-finite BF16 value",
                    ));
                }
                StaticContextTensorValues::MappedBf16 {
                    byte_offset,
                    byte_length,
                }
            };
            tensors.insert(name, StaticContextTensorRecord { dimensions, values });
        }
        if !directory.is_empty() {
            return Err(artifact_error(
                "static context tensor directory has trailing bytes",
            ));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) || total_bytes > MAX_TOTAL_TENSOR_BYTES {
            return Err(artifact_error("static context tensor payload ranges are invalid"));
        }
        Ok(Self {
            version,
            metadata,
            tensors,
            storage: StaticContextStorage::Mapped(Arc::new(mapping)),
        })
    }

    /// Encode the artifact deterministically for tests and external exporters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] if a validated size cannot be encoded.
    #[allow(clippy::too_many_lines)]
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut directory = Vec::new();
        let mut payload = Vec::new();
        for (name, tensor) in &self.tensors {
            let StaticContextTensorData::Fp32(values) = self.tensor_data(tensor) else {
                return Err(artifact_error(
                    "cannot encode BF16 tensor as a version 1 artifact",
                ));
            };
            pad_to_alignment(&mut payload, TENSOR_ALIGNMENT)?;
            let offset = payload.len();
            let byte_length = values
                .len()
                .checked_mul(4)
                .ok_or_else(|| artifact_error("tensor byte length overflows usize"))?;
            write_u32(
                &mut directory,
                u32::try_from(name.len()).map_err(|_| artifact_error("tensor name length exceeds u32"))?,
            );
            write_u16(
                &mut directory,
                u16::try_from(tensor.dimensions.len())
                    .map_err(|_| artifact_error("tensor rank exceeds u16"))?,
            );
            write_u16(&mut directory, 0);
            write_u64(
                &mut directory,
                u64::try_from(offset).map_err(|_| artifact_error("tensor offset exceeds u64"))?,
            );
            write_u64(
                &mut directory,
                u64::try_from(byte_length).map_err(|_| artifact_error("tensor byte length exceeds u64"))?,
            );
            for dimension in &tensor.dimensions {
                write_u64(&mut directory, *dimension);
            }
            directory.extend_from_slice(name.as_bytes());
            for value in values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        if directory.len() > MAX_DIRECTORY_BYTES || payload.len() > MAX_TOTAL_TENSOR_BYTES {
            return Err(artifact_error(
                "static context section exceeds the resource limit",
            ));
        }
        let capacity = HEADER_BYTES
            .checked_add(
                (self.metadata.class_permutations.len() + self.metadata.feature_permutations.len())
                    .checked_mul(8)
                    .ok_or_else(|| artifact_error("permutation length overflows usize"))?,
            )
            .and_then(|size| size.checked_add(directory.len()))
            .and_then(|size| size.checked_add(TENSOR_ALIGNMENT))
            .and_then(|size| size.checked_add(payload.len()))
            .and_then(|size| size.checked_add(CHECKSUM_BYTES))
            .ok_or_else(|| artifact_error("static context length overflows usize"))?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&STATIC_CONTEXT_MAGIC);
        write_u32(&mut bytes, STATIC_CONTEXT_FP32_VERSION);
        write_u32(&mut bytes, STATIC_CONTEXT_KIND);
        bytes.extend_from_slice(&self.metadata.model_sha256);
        bytes.extend_from_slice(&self.metadata.context_sha256);
        for value in [
            self.metadata.ensemble_count,
            self.metadata.training_rows,
            self.metadata.feature_count,
            self.metadata.class_count,
        ] {
            write_u64(
                &mut bytes,
                u64::try_from(value).map_err(|_| artifact_error("metadata value exceeds u64"))?,
            );
        }
        for value in [3_u32, 12, 8, 16, 64] {
            write_u32(&mut bytes, value);
        }
        write_u32(
            &mut bytes,
            u32::try_from(self.tensors.len()).map_err(|_| artifact_error("tensor count exceeds u32"))?,
        );
        write_u32(
            &mut bytes,
            u32::try_from(self.metadata.class_permutations.len())
                .map_err(|_| artifact_error("class permutation count exceeds u32"))?,
        );
        write_u32(
            &mut bytes,
            u32::try_from(self.metadata.feature_permutations.len())
                .map_err(|_| artifact_error("feature permutation count exceeds u32"))?,
        );
        write_u64(
            &mut bytes,
            u64::try_from(directory.len()).map_err(|_| artifact_error("directory length exceeds u64"))?,
        );
        write_u64(
            &mut bytes,
            u64::try_from(payload.len()).map_err(|_| artifact_error("payload length exceeds u64"))?,
        );
        write_u32(&mut bytes, 0);
        debug_assert_eq!(bytes.len(), HEADER_BYTES);
        for value in self
            .metadata
            .class_permutations
            .iter()
            .chain(&self.metadata.feature_permutations)
        {
            write_u64(&mut bytes, *value);
        }
        bytes.extend_from_slice(&directory);
        pad_to_alignment(&mut bytes, TENSOR_ALIGNMENT)?;
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&sha256(&bytes));
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(artifact_error("static context exceeds the resource limit"));
        }
        Ok(bytes)
    }

    /// Verified context metadata.
    #[must_use]
    pub fn metadata(&self) -> &StaticContextMetadata {
        &self.metadata
    }

    /// Whether ICL K tensors use source-tile, dimension-major physical packing.
    #[must_use]
    pub const fn has_packed_icl_keys(&self) -> bool {
        self.version >= STATIC_CONTEXT_VERSION
    }

    /// Get a named exact cache tensor.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<StaticContextTensorRef<'_>> {
        self.tensors
            .get_key_value(name)
            .map(|(stored_name, tensor)| StaticContextTensorRef {
                name: stored_name,
                dimensions: &tensor.dimensions,
                data: self.tensor_data(tensor),
            })
    }

    fn tensor_data<'a>(&'a self, tensor: &'a StaticContextTensorRecord) -> StaticContextTensorData<'a> {
        match (&self.storage, &tensor.values) {
            (StaticContextStorage::Owned, StaticContextTensorValues::Owned(values)) => {
                StaticContextTensorData::Fp32(values)
            }
            (
                StaticContextStorage::Mapped(mapping),
                StaticContextTensorValues::MappedFp32 {
                    byte_offset,
                    byte_length,
                },
            ) => StaticContextTensorData::Fp32(
                bytemuck::try_cast_slice(&mapping[*byte_offset..*byte_offset + *byte_length])
                    .expect("mapped FP32 tensor alignment was validated"),
            ),
            (
                StaticContextStorage::Mapped(mapping),
                StaticContextTensorValues::MappedBf16 {
                    byte_offset,
                    byte_length,
                },
            ) => StaticContextTensorData::Bf16(
                bytemuck::try_cast_slice(&mapping[*byte_offset..*byte_offset + *byte_length])
                    .expect("mapped BF16 tensor alignment was validated"),
            ),
            _ => unreachable!("static context tensor storage must match artifact storage"),
        }
    }
}

fn validate_metadata(metadata: &StaticContextMetadata) -> Result<()> {
    if metadata.model_sha256 == [0; 32] || metadata.context_sha256 == [0; 32] {
        return Err(artifact_error("static context SHA-256 identity is empty"));
    }
    if metadata.ensemble_count == 0
        || metadata.ensemble_count > MAX_ENSEMBLE_COUNT
        || metadata.training_rows == 0
        || metadata.training_rows > MAX_TRAINING_ROWS
        || metadata.feature_count == 0
        || metadata.feature_count > MAX_FEATURE_COUNT
        || metadata.class_count == 0
        || metadata.class_count > MAX_CLASS_COUNT
    {
        return Err(artifact_error(
            "static context dimensions are outside the resource limits",
        ));
    }
    validate_estimator_permutations(
        &metadata.class_permutations,
        metadata.ensemble_count,
        metadata.class_count,
        "class",
    )?;
    validate_estimator_permutations(
        &metadata.feature_permutations,
        metadata.ensemble_count,
        metadata.feature_count,
        "feature",
    )
}

fn validate_estimator_permutations(
    values: &[u64],
    ensemble_count: usize,
    width: usize,
    name: &str,
) -> Result<()> {
    let expected = ensemble_count
        .checked_mul(width)
        .ok_or_else(|| artifact_error(format!("{name} permutation count overflows usize")))?;
    if values.len() != expected {
        return Err(artifact_error(format!(
            "{name} permutation count does not match estimator metadata"
        )));
    }
    for permutation in values.chunks_exact(width) {
        validate_permutation(permutation, width, name)?;
    }
    Ok(())
}

fn validate_permutation(values: &[u64], expected_length: usize, name: &str) -> Result<()> {
    if values.len() != expected_length {
        return Err(artifact_error(format!(
            "{name} permutation length does not match metadata"
        )));
    }
    let mut seen = vec![false; expected_length];
    for value in values {
        let index = to_usize(*value, name)?;
        if index >= expected_length || std::mem::replace(&mut seen[index], true) {
            return Err(artifact_error(format!("{name} permutation is invalid")));
        }
    }
    Ok(())
}

fn validate_tensor(tensor: &StaticContextTensor) -> Result<()> {
    if tensor.name.is_empty()
        || tensor.name.len() > MAX_TENSOR_NAME_BYTES
        || tensor.dimensions.is_empty()
        || tensor.dimensions.len() > MAX_TENSOR_RANK
    {
        return Err(artifact_error("static context tensor name or rank is invalid"));
    }
    let elements = checked_elements(&tensor.dimensions)?;
    if tensor.values.len() != elements || tensor.values.iter().any(|value| !value.is_finite()) {
        return Err(artifact_error(format!(
            "static context tensor '{}' has an invalid shape or non-finite value",
            tensor.name
        )));
    }
    if tensor
        .values
        .len()
        .checked_mul(4)
        .ok_or_else(|| artifact_error("tensor byte length overflows usize"))?
        > MAX_TENSOR_BYTES
    {
        return Err(artifact_error(format!(
            "static context tensor '{}' exceeds the resource limit",
            tensor.name
        )));
    }
    Ok(())
}

fn checked_elements(dimensions: &[u64]) -> Result<usize> {
    dimensions.iter().try_fold(1_usize, |elements, dimension| {
        let dimension = to_usize(*dimension, "tensor dimension")?;
        if dimension == 0 {
            return Err(artifact_error("static context tensor dimension is zero"));
        }
        elements
            .checked_mul(dimension)
            .ok_or_else(|| artifact_error("tensor element count overflows usize"))
    })
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(artifact_error("static context FP32 payload length is invalid"));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("FP32 chunk has four bytes")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(artifact_error(
            "static context tensor contains a non-finite value",
        ));
    }
    Ok(values)
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| artifact_error("alignment overflows usize"))
}
fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) -> Result<()> {
    let end = align_up(bytes.len(), alignment)?;
    bytes.resize(end, 0);
    Ok(())
}
fn read_permutation(reader: &mut ByteReader<'_>, count: usize) -> Result<Vec<u64>> {
    (0..count).map(|_| reader.read_u64()).collect()
}
fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn to_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| artifact_error(format!("{name} exceeds usize")))
}
fn artifact_error(message: impl std::fmt::Display) -> Error {
    Error::Artifact(format!("static context: {message}"))
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
            .map_err(|_| artifact_error("static context field length is invalid"))
    }
    fn read_bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| artifact_error("static context offset overflows usize"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| artifact_error("static context is truncated"))?;
        self.position = end;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> StaticContextArtifact {
        StaticContextArtifact::new(
            StaticContextMetadata {
                model_sha256: [1; 32],
                context_sha256: [2; 32],
                ensemble_count: 2,
                training_rows: 3,
                feature_count: 2,
                class_count: 3,
                class_permutations: vec![2, 0, 1, 1, 2, 0],
                feature_permutations: vec![1, 0, 0, 1],
            },
            vec![
                StaticContextTensor {
                    name: "column.k.0".into(),
                    dimensions: vec![2, 8, 3, 16],
                    values: vec![1.0; 768],
                },
                StaticContextTensor {
                    name: "icl.v.0".into(),
                    dimensions: vec![2, 8, 3, 64],
                    values: vec![2.0; 3072],
                },
            ],
        )
        .expect("valid fixture")
    }

    fn resign(bytes: &mut [u8]) {
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let checksum = sha256(&bytes[..checksum_offset]);
        bytes[checksum_offset..].copy_from_slice(&checksum);
    }

    #[test]
    fn round_trip_is_deterministic_and_owned() {
        let artifact = fixture();
        let bytes = artifact.to_bytes().expect("encode");
        assert_eq!(bytes, artifact.to_bytes().expect("encode again"));
        let decoded = StaticContextArtifact::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, artifact);
        assert_eq!(
            decoded
                .tensor("icl.v.0")
                .expect("tensor")
                .fp32_values()
                .expect("FP32")[0]
                .to_bits(),
            2.0_f32.to_bits()
        );
    }

    #[test]
    fn mapped_round_trip_retains_validated_tensor_storage() {
        let artifact = fixture();
        let bytes = artifact.to_bytes().expect("encode");
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("context.ironiclctx");
        std::fs::write(&path, &bytes).expect("write context");

        let mapped = StaticContextArtifact::read_mapped(&path).expect("map context");
        assert_eq!(mapped, artifact);
        let clone = mapped.clone();
        drop(mapped);
        assert_eq!(
            clone
                .tensor("icl.v.0")
                .expect("tensor")
                .fp32_values()
                .expect("FP32")[0]
                .to_bits(),
            2.0_f32.to_bits()
        );
        assert_eq!(clone.to_bytes().expect("re-encode mapped context"), bytes);
    }

    #[test]
    fn rejects_corruption_and_invalid_shapes() {
        let mut corrupt = fixture().to_bytes().expect("encode");
        corrupt[HEADER_BYTES] ^= 1;
        assert!(StaticContextArtifact::from_bytes(&corrupt).is_err());
        assert!(
            StaticContextArtifact::new(
                fixture().metadata().clone(),
                vec![StaticContextTensor {
                    name: "bad".into(),
                    dimensions: vec![2],
                    values: vec![f32::NAN, 1.0]
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_resource_and_directory_metadata() {
        let mut malformed = fixture().to_bytes().expect("encode");
        let tensor_count_offset = 8 + 4 + 4 + 32 + 32 + 32 + 20;
        malformed[tensor_count_offset..tensor_count_offset + 4].copy_from_slice(&0_u32.to_le_bytes());
        resign(&mut malformed);
        assert!(StaticContextArtifact::from_bytes(&malformed).is_err());

        let metadata = StaticContextMetadata {
            feature_permutations: vec![0; MAX_FEATURE_COUNT + 1],
            feature_count: MAX_FEATURE_COUNT + 1,
            ..fixture().metadata().clone()
        };
        assert!(
            StaticContextArtifact::new(
                metadata,
                vec![StaticContextTensor {
                    name: "x".into(),
                    dimensions: vec![1],
                    values: vec![0.0]
                }]
            )
            .is_err()
        );
    }
}

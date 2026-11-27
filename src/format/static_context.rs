//! Fitted cache tensors backed by owned values, anonymous pages, or an immutable file.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use memmap2::{Mmap, MmapMut, MmapOptions};

use crate::{
    Optimize,
    error::{Error, Result},
    format::container::{
        self, DISK_ALIGNMENT, DType, Entry, Header, Kind, MAX_TENSOR_COUNT, MAX_TENSOR_NAME_BYTES,
        MAX_TENSOR_RANK, MEMORY_ALIGNMENT, Writer,
    },
    models::kumo_tabular::Release,
};

pub use container::{MAGIC as STATIC_CONTEXT_MAGIC, VERSION as STATIC_CONTEXT_VERSION};
const MAX_ARTIFACT_BYTES: usize = byte_limit(48 << 30);
const MAX_TENSOR_BYTES: usize = byte_limit(4 << 30);
const MAX_TOTAL_TENSOR_BYTES: usize = byte_limit(48 << 30);
const MAX_ENSEMBLE_COUNT: usize = 4096;
const MAX_TRAINING_ROWS: usize = 1_000_000;
const MAX_FEATURE_COUNT: usize = 65_536;
const MAX_CLASS_COUNT: usize = 4096;

/// `bytes`, or the largest possible allocation (`isize::MAX`) where that is
/// smaller, as on 32-bit WebAssembly.
#[expect(clippy::cast_possible_truncation, reason = "Only after checking that it fits")]
const fn byte_limit(bytes: u64) -> usize {
    const LARGEST: usize = usize::MAX >> 1;
    if bytes > LARGEST as u64 {
        LARGEST
    } else {
        bytes as usize
    }
}

/// The model a static context is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextArchitecture {
    #[serde(rename = "tabicl_v2")]
    TabIclV2,
    /// Kumo-Tabular: no permutations, one context per
    /// member and codebook task.
    KumoTabular(Release),
}

/// Metadata that identifies and describes one static training context.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticContextMetadata {
    /// The model the context is for.
    pub architecture: ContextArchitecture,
    /// SHA-256 of the source checkpoint recorded by the model artifact.
    #[serde(rename = "checkpoint_sha256", with = "container::digest")]
    pub model_sha256: [u8; 32],
    /// SHA-256 identity of the source static context and export inputs.
    #[serde(rename = "identity", with = "container::digest")]
    pub context_sha256: [u8; 32],
    /// Ensemble count `B`.
    pub ensemble_count: usize,
    /// Training-row count `N`.
    pub training_rows: usize,
    /// Feature count `H`.
    pub feature_count: usize,
    /// Class count; 0 for a regression context, which has no class order.
    pub class_count: usize,
    /// Estimator-major class orders with shape `[B, class_count]`.
    pub class_permutations: Vec<u64>,
    /// Estimator-major feature orders with shape `[B, H]`.
    pub feature_permutations: Vec<u64>,
    /// The level the context was built at.
    pub optimize: Optimize,
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
    /// IEEE half precision, written by GPU-built contexts under `optimize="auto"`.
    F16(&'a [u16]),
    /// Finite-number E4M3, stored verbatim; any scales are separate tensors.
    F8E4M3(&'a [u8]),
}

impl<'a> StaticContextTensorData<'a> {
    pub(crate) fn encoded(self) -> (DType, &'a [u8]) {
        match self {
            Self::Fp32(v) => (DType::F32, bytemuck::cast_slice(v)),
            Self::Bf16(v) => (DType::Bf16, bytemuck::cast_slice(v)),
            Self::F16(v) => (DType::F16, bytemuck::cast_slice(v)),
            Self::F8E4M3(v) => (DType::F8E4M3, v),
        }
    }

    #[must_use]
    pub const fn len(self) -> usize {
        match self {
            Self::Fp32(values) => values.len(),
            Self::F8E4M3(values) => values.len(),
            Self::Bf16(values) | Self::F16(values) => values.len(),
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
            Self::F16(values) => Self::F16(&values[start..][..length]),
            Self::F8E4M3(values) => Self::F8E4M3(&values[start..][..length]),
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
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum StaticContextTensorValues {
    Owned(Vec<f32>),
    Mapped(DType, std::ops::Range<usize>),
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
    /// A mapping where every tensor starts on a page and is padded to a
    /// page multiple, so a GPU can wrap each tensor without copying it.
    #[cfg_attr(
        not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
        allow(dead_code, reason = "only the Metal build uses page-aligned storage")
    )]
    PageAligned(Arc<Mmap>),
}

/// Page size for page-aligned tensors: the 16 KiB Apple Silicon page, a
/// multiple of 4 KiB pages elsewhere.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code, reason = "only the Metal build uses page-aligned storage")
)]
pub(crate) const TENSOR_PAGE_BYTES: usize = DISK_ALIGNMENT;

/// FP32, FP16, or BF16 tensors being written into page-aligned anonymous memory, for
/// [`StaticContextArtifact::from_page_aligned`].
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code, reason = "only the Metal build uses page-aligned storage")
)]
pub(crate) struct PageAlignedTensors {
    mapping: MmapMut,
    /// Name, dimensions, element type, byte offset, and byte length of each tensor.
    layout: Vec<(String, Vec<u64>, DType, usize, usize)>,
}

const fn is_cache_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::Bf16 | DType::F8E4M3)
}

fn view_cache(dtype: DType, bytes: &[u8]) -> Result<StaticContextTensorData<'_>> {
    Ok(match dtype {
        DType::F32 => StaticContextTensorData::Fp32(bytemuck::try_cast_slice(bytes).map_err(artifact_error)?),
        DType::Bf16 => {
            StaticContextTensorData::Bf16(bytemuck::try_cast_slice(bytes).map_err(artifact_error)?)
        }
        DType::F16 => StaticContextTensorData::F16(bytemuck::try_cast_slice(bytes).map_err(artifact_error)?),
        DType::F8E4M3 => StaticContextTensorData::F8E4M3(bytes),
        _ => return Err(artifact_error("invalid cache dtype")),
    })
}

#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code, reason = "only the Metal build uses page-aligned storage")
)]
impl PageAlignedTensors {
    /// Reserve zeroed, page-aligned space for each `(name, dimensions, dtype)`.
    pub(crate) fn new(specs: Vec<(String, Vec<u64>, DType)>) -> Result<Self> {
        let mut layout = Vec::with_capacity(specs.len());
        let mut offset = 0_usize;
        for (name, dimensions, dtype) in specs {
            if !is_cache_dtype(dtype) {
                return Err(artifact_error("invalid cache dtype"));
            }
            let bytes = checked_elements(&dimensions)?
                .checked_mul(dtype.size_in_bytes())
                .ok_or_else(|| artifact_error("tensor byte length overflows usize"))?;
            if bytes > MAX_TENSOR_BYTES {
                return Err(artifact_error("tensor exceeds size limit"));
            }
            layout.push((name, dimensions, dtype, offset, bytes));
            offset = offset
                .checked_add(bytes.next_multiple_of(TENSOR_PAGE_BYTES))
                .ok_or_else(|| artifact_error("total tensor byte length overflows usize"))?;
        }
        if offset > MAX_TOTAL_TENSOR_BYTES {
            return Err(artifact_error(
                "static context tensor payload exceeds the resource limit",
            ));
        }
        let mapping = MmapOptions::new()
            .len(offset.max(TENSOR_PAGE_BYTES))
            .map_anon()
            .map_err(artifact_error)?;
        Ok(Self { mapping, layout })
    }

    /// Copy every tensor of `group`, which holds the same tensors with a
    /// smaller outermost dimension, into slot `index` of that dimension.
    pub(crate) fn copy_group(&mut self, group: &Self, index: usize) {
        for ((.., offset, bytes), (.., source, length)) in self.layout.iter().zip(&group.layout) {
            debug_assert!((index + 1) * length <= *bytes);
            self.mapping[offset + index * length..][..*length]
                .copy_from_slice(&group.mapping[*source..][..*length]);
        }
    }

    /// Each tensor's page-aligned start and page-padded byte length, in the
    /// order given to [`Self::new`]. The pointers stay valid while `self` and
    /// the artifact made from it live.
    pub(crate) fn regions(&mut self) -> Vec<(std::ptr::NonNull<u8>, usize)> {
        // One pointer to the whole mapping, so each region's pointer may reach its full length.
        let (base, length) = (self.mapping.as_mut_ptr(), self.mapping.len());
        self.layout
            .iter()
            .map(|(_, _, _, offset, bytes)| {
                let padded = bytes.next_multiple_of(TENSOR_PAGE_BYTES);
                assert!(
                    offset + padded <= length,
                    "page-aligned tensors lie within the mapping"
                );
                let start = std::ptr::NonNull::new(base.wrapping_add(*offset));
                (start.expect("a mapping's pointers are non-null"), padded)
            })
            .collect()
    }

    /// Each tensor's bytes, in the order given to [`Self::new`], for writing
    /// in its storage type.
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "only the CUDA build fills tensors in place")
    )]
    pub(crate) fn tensors_mut(&mut self) -> Vec<&mut [u8]> {
        let mut rest = &mut self.mapping[..];
        let mut at = 0;
        self.layout
            .iter()
            .map(|(.., offset, bytes)| {
                let (_padding, tensor) = (rest.split_off_mut(..offset - at), rest.split_off_mut(..*bytes));
                at = offset + bytes;
                tensor.expect("tensors lie within the mapping")
            })
            .collect()
    }

    /// Write a tensor's already-encoded storage bytes into its region. The
    /// caller converts to the tensor's [`TensorStorage`] and must pass exactly
    /// the region's byte length.
    pub(crate) fn fill(&mut self, index: usize, bytes: &[u8]) {
        let (offset, length) = {
            let (.., offset, length) = &self.layout[index];
            (*offset, *length)
        };
        assert_eq!(bytes.len(), length, "page-aligned tensor length mismatch");
        self.mapping[offset..offset + length].copy_from_slice(bytes);
    }
}

/// A verified static context backed by owned or read-only mapped tensor storage.
#[derive(Debug, Clone)]
pub struct StaticContextArtifact {
    metadata: StaticContextMetadata,
    tensors: BTreeMap<String, StaticContextTensorRecord>,
    storage: StaticContextStorage,
}

impl PartialEq for StaticContextArtifact {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata
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
            metadata,
            tensors: named,
            storage: StaticContextStorage::Owned,
        })
    }

    /// Own tensors written into page-aligned memory without copying; the
    /// memory becomes read-only. Only names and shapes are checked here: the
    /// writer must already have proven every value finite. Tensors may be
    /// FP32, FP16, or BF16.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when metadata, names, or shapes are invalid.
    #[cfg_attr(
        not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
        allow(dead_code, reason = "only the Metal build uses page-aligned storage")
    )]
    pub(crate) fn from_page_aligned(
        metadata: StaticContextMetadata,
        tensors: PageAlignedTensors,
    ) -> Result<Self> {
        validate_metadata(&metadata)?;
        if tensors.layout.is_empty() || tensors.layout.len() > MAX_TENSOR_COUNT {
            return Err(artifact_error("static context tensor count is invalid"));
        }
        let mut named = BTreeMap::new();
        for (name, dimensions, dtype, byte_offset, byte_length) in tensors.layout {
            let elements = byte_length / dtype.size_in_bytes();
            validate_tensor_shape(&name, &dimensions, elements, dtype.size_in_bytes())?;
            let values = StaticContextTensorValues::Mapped(dtype, byte_offset..byte_offset + byte_length);
            if named
                .insert(name, StaticContextTensorRecord { dimensions, values })
                .is_some()
            {
                return Err(artifact_error("static context tensor names must be unique"));
            }
        }
        Ok(Self {
            metadata,
            tensors: named,
            storage: StaticContextStorage::PageAligned(Arc::new(
                tensors.mapping.make_read_only().map_err(artifact_error)?,
            )),
        })
    }

    /// The page-aligned memory holding every tensor, if the artifact has it.
    #[cfg_attr(
        not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
        allow(dead_code, reason = "only the Metal build uses page-aligned storage")
    )]
    pub(crate) fn page_aligned_storage(&self) -> Option<&Arc<Mmap>> {
        match &self.storage {
            StaticContextStorage::PageAligned(mapping) => Some(mapping),
            _ => None,
        }
    }

    /// Decode and verify a complete static-context artifact,
    /// copied into anonymous memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] for malformed, corrupt, or invalid input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mapping = container::map_bytes(bytes, MAX_ARTIFACT_BYTES)?;
        let header = context_header(&mapping)?;
        Self::from_mapping(mapping, header)
    }

    /// Map, verify, and borrow tensors directly from an immutable artifact.
    ///
    /// The file is memory-mapped, not copied: it must not change while the artifact lives.
    /// Another process writing it makes the mapped tensors change underneath Rust references,
    /// and truncating it makes the next read fault (`SIGBUS`). Serving artifacts are written
    /// once and replaced atomically, never edited in place.
    ///
    /// The mapping is prefaulted so prediction runs at RAM speed from the first query instead
    /// of paying page-fault latency. The hint is best-effort and ignored where the platform
    /// does not support it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] when the file or mapped tensor layout is invalid.
    pub fn read_mapped_prefault(path: &Path) -> Result<Self> {
        let mapping = container::map_immutable(path, MAX_ARTIFACT_BYTES as u64)?;
        let header = context_header(&mapping)?;
        container::prefault(&mapping);
        Self::from_mapping(mapping, header)
    }

    fn from_mapping(mapping: Mmap, header: Header<StaticContextMetadata>) -> Result<Self> {
        if !cfg!(target_endian = "little") {
            return Err(artifact_error("mapped contexts require a little-endian host"));
        }
        // Each range, including its page-rounded length, is inside this same mapping.
        let page_aligned = header.alignment >= TENSOR_PAGE_BYTES
            && mapping.as_ptr().addr().is_multiple_of(TENSOR_PAGE_BYTES);
        let mut tensors = BTreeMap::new();
        for t in &header.tensors {
            let range = header.range(t);
            let slice = &mapping[range.clone()];
            if !t.dtype.all_finite(slice)? {
                return Err(artifact_error(format!("non-finite cache tensor '{}'", t.name)));
            }
            // Verify alignment for typed borrowing before exposing the mapping.
            view_cache(t.dtype, slice)?;
            tensors.insert(
                t.name.clone(),
                StaticContextTensorRecord {
                    dimensions: t.shape.clone(),
                    values: StaticContextTensorValues::Mapped(t.dtype, range),
                },
            );
        }
        let mapping = Arc::new(mapping);
        Ok(Self {
            metadata: header.metadata,
            tensors,
            storage: if page_aligned {
                StaticContextStorage::PageAligned(mapping)
            } else {
                StaticContextStorage::Mapped(mapping)
            },
        })
    }

    /// Encode compact, deterministic bytes for an in-memory consumer.
    /// # Errors
    /// If a size or the host byte order is unsupported.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.writer(MEMORY_ALIGNMENT)?.bytes()
    }

    /// The exact size of [`Self::encode`]'s output.
    /// # Errors
    /// If the container exceeds a resource limit.
    pub fn encoded_len(&self) -> Result<usize> {
        Ok(self.writer(MEMORY_ALIGNMENT)?.len)
    }

    /// Write compact bytes without buffering another copy of the tensors.
    /// # Errors
    /// If the container cannot be encoded or written.
    pub fn encode(&self, sink: &mut impl std::io::Write) -> Result<()> {
        self.writer(MEMORY_ALIGNMENT)?.write(sink)
    }

    fn writer(&self, alignment: usize) -> Result<Writer<'_>> {
        if !cfg!(target_endian = "little") {
            return Err(artifact_error("encoding contexts requires a little-endian host"));
        }
        let mut tensors: Vec<_> = self.tensors.iter().collect();
        tensors.sort_by_cached_key(|(name, _)| execution_key(name));
        let writer = Writer::new(
            Kind::Context,
            alignment,
            &self.metadata,
            tensors.into_iter().map(|(name, t)| {
                let (dtype, bytes) = self.tensor_data(t).encoded();
                (name.clone(), dtype, t.dimensions.clone(), bytes)
            }),
        )?;
        if writer.len > MAX_ARTIFACT_BYTES {
            return Err(artifact_error("context exceeds size limit"));
        }
        Ok(writer)
    }

    /// Encode and write this context to `path` atomically: write a new temporary
    /// file in the same directory, sync it, then rename it into place, so a
    /// reader never observes a partial file and concurrent writers do not
    /// clobber each other. The file is readable only by its owner on Unix,
    /// because a context encodes its training data. The payload streams to the
    /// file without a copy in memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] if encoding or any filesystem step fails.
    pub fn write(&self, path: &Path) -> Result<()> {
        super::write_atomic(path, |file| {
            use std::io::{BufWriter, Write};
            let mut writer = BufWriter::new(file);
            self.writer(DISK_ALIGNMENT)?.write(&mut writer)?;
            writer.flush().map_err(artifact_error)
        })
    }

    /// The verified artifact at `path`, prefaulted as by
    /// [`Self::read_mapped_prefault`], if its metadata is exactly `expected`.
    /// `None` when the file is missing, unreadable, or invalid, or describes
    /// another context.
    pub(crate) fn read_cached(path: &Path, expected: &StaticContextMetadata) -> Option<Self> {
        Self::read_mapped_prefault(path)
            .ok()
            .filter(|context| context.metadata() == expected)
    }

    /// The tensor `name`, after checking that it exists with `dimensions`.
    ///
    /// # Errors
    ///
    /// When the tensor is missing or has other dimensions.
    pub(crate) fn check_tensor(&self, name: &str, dimensions: &[u64]) -> Result<StaticContextTensorRef<'_>> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| artifact_error(format!("static context cache tensor '{name}' is missing")))?;
        if tensor.dimensions != dimensions {
            return Err(artifact_error(format!(
                "static context cache tensor '{name}' has shape {:?}; expected {dimensions:?}",
                tensor.dimensions
            )));
        }
        Ok(tensor)
    }

    /// How many tensors the context holds.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Verified context metadata.
    #[must_use]
    pub fn metadata(&self) -> &StaticContextMetadata {
        &self.metadata
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
                StaticContextStorage::Mapped(mapping) | StaticContextStorage::PageAligned(mapping),
                StaticContextTensorValues::Mapped(dtype, range),
            ) => view_cache(*dtype, &mapping[range.clone()]).expect("validated tensor alignment and length"),
            _ => unreachable!("static context tensor storage must match artifact storage"),
        }
    }
}

// Sort layer, member and task indices numerically. Keep K and V together per
// stage, placing any scale tensor immediately after its tensor.
fn execution_key(name: &str) -> (usize, usize, usize, usize, usize) {
    let mut parts = name.split('.').peekable();
    let mut index = || {
        parts
            .next_if(|s| s.parse::<usize>().is_ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    };
    let member = index();
    let task = index();
    let stage = match parts.next() {
        Some("means") => 0,
        Some("categorical") => 1,
        Some("column" | "columns") => 2,
        Some("icl") => 3,
        _ => 4,
    };
    let block = parts
        .next_if(|s| s.parse::<usize>().is_ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let kv = match parts.next() {
        Some("key") => 0,
        Some("key_scale") => 1,
        Some("value") => 2,
        Some("value_scale") => 3,
        _ => 4,
    };
    (member, task, stage, block, kv)
}

fn context_header(bytes: &[u8]) -> Result<Header<StaticContextMetadata>> {
    let header = Header::from_bytes(bytes, Kind::Context)?;
    validate_header(&header)?;
    Ok(header)
}

fn validate_header(header: &Header<StaticContextMetadata>) -> Result<()> {
    validate_metadata(&header.metadata)?;
    if header.tensors.is_empty()
        || header.tensors.iter().any(|t| {
            !is_cache_dtype(t.dtype)
                || t.shape.is_empty()
                || t.shape.contains(&0)
                || t.bytes > MAX_TENSOR_BYTES
        })
    {
        return Err(artifact_error("invalid cache tensor count, shape, size or dtype"));
    }
    Ok(())
}

/// Validated context metadata and tensor directory. Inspection never reads the payloads.
pub struct ContextHeader {
    pub metadata: StaticContextMetadata,
    pub tensors: Vec<Entry>,
    pub file_len: u64,
}
impl ContextHeader {
    /// Open an immutable context and validate its metadata and physical layout.
    /// # Errors
    /// If the file cannot be opened or the header is invalid.
    pub fn open(path: &Path) -> Result<Self> {
        let header = Header::open(path, Kind::Context, MAX_ARTIFACT_BYTES as u64)?;
        validate_header(&header)?;
        Ok(Self {
            metadata: header.metadata,
            tensors: header.tensors,
            file_len: header.file_len,
        })
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
        || metadata.class_count > MAX_CLASS_COUNT
    {
        return Err(artifact_error(
            "static context dimensions are outside the resource limits",
        ));
    }
    if let ContextArchitecture::KumoTabular(_) = metadata.architecture {
        if metadata.class_count != 0
            || !metadata.class_permutations.is_empty()
            || !metadata.feature_permutations.is_empty()
        {
            return Err(artifact_error(
                "a Kumo-Tabular context has no class count or permutations",
            ));
        }
        return Ok(());
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
    if width > 0 {
        for permutation in values.chunks_exact(width) {
            validate_permutation(permutation, width, name)?;
        }
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
    let name = &tensor.name;
    validate_tensor_shape(name, &tensor.dimensions, tensor.values.len(), 4)?;
    if tensor.values.iter().any(|value| !value.is_finite()) {
        return Err(artifact_error(format!(
            "static context tensor '{name}' has a non-finite value"
        )));
    }
    Ok(())
}

fn validate_tensor_shape(name: &str, dimensions: &[u64], length: usize, element_bytes: usize) -> Result<()> {
    if name.is_empty()
        || name.len() > MAX_TENSOR_NAME_BYTES
        || dimensions.is_empty()
        || dimensions.len() > MAX_TENSOR_RANK
    {
        return Err(artifact_error("static context tensor name or rank is invalid"));
    }
    if length != checked_elements(dimensions)? {
        return Err(artifact_error(format!(
            "static context tensor '{name}' has an invalid shape"
        )));
    }
    if length
        .checked_mul(element_bytes)
        .ok_or_else(|| artifact_error("tensor byte length overflows usize"))?
        > MAX_TENSOR_BYTES
    {
        return Err(artifact_error(format!(
            "static context tensor '{name}' exceeds the resource limit"
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

fn to_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| artifact_error(format!("{name} exceeds usize")))
}
fn artifact_error(message: impl std::fmt::Display) -> Error {
    Error::Artifact(format!("static context: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> StaticContextArtifact {
        StaticContextArtifact::new(
            StaticContextMetadata {
                architecture: ContextArchitecture::TabIclV2,
                model_sha256: [1; 32],
                context_sha256: [2; 32],
                ensemble_count: 2,
                training_rows: 3,
                feature_count: 2,
                class_count: 3,
                class_permutations: vec![2, 0, 1, 1, 2, 0],
                feature_permutations: vec![1, 0, 0, 1],
                optimize: Optimize::Auto,
            },
            fixture_tensors(),
        )
        .expect("valid fixture")
    }

    fn fixture_tensors() -> Vec<StaticContextTensor> {
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
        ]
    }

    #[test]
    fn round_trip_is_deterministic() {
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
    fn disk_layout_aligns_tensors_in_execution_order() {
        let names = [
            "icl.10.key",
            "icl.2.value_scale",
            "icl.2.value",
            "column.1.key",
            "icl.2.key_scale",
            "icl.2.key",
        ];
        let tensors = names
            .iter()
            .enumerate()
            .map(|(i, name)| StaticContextTensor {
                name: (*name).into(),
                dimensions: vec![2],
                values: vec![0.0, f32::from(u8::try_from(i).unwrap())],
            })
            .collect();
        let artifact = StaticContextArtifact::new(fixture().metadata().clone(), tensors).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("context.ir");
        artifact.write(&path).unwrap();
        let header = ContextHeader::open(&path).unwrap();
        assert_eq!(&header.metadata, artifact.metadata());
        assert_eq!(header.file_len, (7 * DISK_ALIGNMENT) as u64);
        assert_eq!(
            header.tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            [
                "column.1.key",
                "icl.2.key",
                "icl.2.key_scale",
                "icl.2.value",
                "icl.2.value_scale",
                "icl.10.key",
            ]
        );
        for (i, tensor) in header.tensors.iter().enumerate() {
            assert_eq!(tensor.offset, i * DISK_ALIGNMENT);
        }
        let mapped = StaticContextArtifact::read_mapped_prefault(&path).unwrap();
        assert_eq!(mapped, artifact);
        // On 16 KiB-page Apple Silicon, disk-written files qualify for zero-copy Metal wrapping.
        // 4 KiB-page hosts do not guarantee 16 KiB mmap base alignment.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(mapped.page_aligned_storage().is_some());
        assert_eq!(
            artifact.to_bytes().unwrap().len(),
            artifact.encoded_len().unwrap()
        );
    }

    #[test]
    fn regression_context_without_classes_round_trips() {
        let artifact = StaticContextArtifact::new(
            StaticContextMetadata {
                architecture: ContextArchitecture::TabIclV2,
                class_count: 0,
                class_permutations: Vec::new(),
                ..fixture().metadata().clone()
            },
            fixture_tensors(),
        )
        .expect("valid regression context");
        let bytes = artifact.to_bytes().expect("encode");
        assert_eq!(
            StaticContextArtifact::from_bytes(&bytes).expect("decode"),
            artifact
        );
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("context.ir");
        std::fs::write(&path, &bytes).expect("write context");
        assert_eq!(
            StaticContextArtifact::read_mapped_prefault(&path).expect("map context"),
            artifact
        );
    }

    #[test]
    fn mapped_round_trip_retains_validated_tensor_storage() {
        let artifact = fixture();
        let bytes = artifact.to_bytes().expect("encode");
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("context.ir");
        std::fs::write(&path, &bytes).expect("write context");

        let mapped = StaticContextArtifact::read_mapped_prefault(&path).expect("map context");
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
    fn round_trip_preserves_encoded_context_dtypes() {
        let specs = vec![
            ("column.k.0".to_string(), vec![2, 8, 3, 16], DType::F32),
            ("icl.k.0".to_string(), vec![2, 8, 3, 64], DType::Bf16),
            ("icl.v.0".to_string(), vec![2, 8, 3, 64], DType::F16),
            ("encoded".to_string(), vec![4], DType::F8E4M3),
        ];
        let mut tensors = PageAlignedTensors::new(specs).expect("page-aligned allocation");
        let fp32: Vec<f32> = (0..768u16).map(|i| f32::from(i) * 0.5).collect();
        let bf16: Vec<u16> = (0..3072u16).map(|i| i % 97 + 1).collect();
        let f16: Vec<u16> = (0..3072u16).map(|i| i % 89 + 1).collect();
        tensors.fill(0, bytemuck::cast_slice(&fp32));
        tensors.fill(1, bytemuck::cast_slice(&bf16));
        tensors.fill(2, bytemuck::cast_slice(&f16));
        let fp8 = [0, 0x80, 0x7e, 0xfe];
        tensors.fill(3, &fp8);
        let artifact = StaticContextArtifact::from_page_aligned(fixture().metadata().clone(), tensors)
            .expect("page-aligned artifact");

        let bytes = artifact.to_bytes().expect("encode");
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("mixed.ir");
        std::fs::write(&path, &bytes).expect("write context");
        let decoded = StaticContextArtifact::read_mapped_prefault(&path).expect("map context");

        assert!(matches!(
            decoded.tensor("column.k.0").expect("fp32 tensor").data,
            StaticContextTensorData::Fp32(_)
        ));
        match decoded.tensor("icl.k.0").expect("bf16 tensor").data {
            StaticContextTensorData::Bf16(values) => assert_eq!(values, bf16.as_slice()),
            other => panic!("expected BF16, got {other:?}"),
        }
        match decoded.tensor("icl.v.0").expect("f16 tensor").data {
            StaticContextTensorData::F16(values) => assert_eq!(values, f16.as_slice()),
            other => panic!("expected FP16, got {other:?}"),
        }
        assert_eq!(
            decoded.tensor("encoded").unwrap().data,
            StaticContextTensorData::F8E4M3(&fp8)
        );
        let header = super::context_header(&bytes).unwrap();
        let encoded = header.tensors.iter().find(|t| t.name == "encoded").unwrap();
        for nan in [0x7f, 0xff] {
            let mut bad = bytes.clone();
            bad[header.range(encoded).start] = nan;
            assert!(StaticContextArtifact::from_bytes(&bad).is_err());
        }
        // Re-encoding the mapped context is byte-stable.
        assert_eq!(decoded.to_bytes().expect("re-encode"), bytes);
        let from_bytes = StaticContextArtifact::from_bytes(&bytes).expect("decode bytes");
        assert_eq!(from_bytes.to_bytes().expect("re-encode bytes"), bytes);
    }

    #[test]
    fn rejects_legacy_contexts() {
        let mut bytes = fixture().to_bytes().unwrap();
        bytes[..8].copy_from_slice(b"IRCNTXT\0");
        assert!(StaticContextArtifact::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_corruption_and_invalid_shapes() {
        let mut corrupt = fixture().to_bytes().expect("encode");
        corrupt[16] ^= 1;
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
        malformed.truncate(malformed.len() - 1);
        assert!(StaticContextArtifact::from_bytes(&malformed).is_err());

        let metadata = StaticContextMetadata {
            architecture: ContextArchitecture::TabIclV2,
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

    fn temporary_files(directory: &Path) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(directory)
            .expect("list directory")
            .map(|entry| entry.expect("directory entry").path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "tmp"))
            .collect()
    }

    #[test]
    fn cached_read_accepts_only_the_expected_context() {
        let artifact = fixture();
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("context.ir");
        assert!(StaticContextArtifact::read_cached(&path, artifact.metadata()).is_none());

        artifact.write(&path).expect("write context");
        assert_eq!(
            StaticContextArtifact::read_cached(&path, artifact.metadata()),
            Some(artifact.clone())
        );
        // A valid entry of another context, as if it were stored under the wrong name.
        let other = StaticContextMetadata {
            architecture: ContextArchitecture::TabIclV2,
            context_sha256: [3; 32],
            ..artifact.metadata().clone()
        };
        assert!(StaticContextArtifact::read_cached(&path, &other).is_none());

        let bytes = std::fs::read(&path).expect("read context");
        std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncate context");
        assert!(StaticContextArtifact::read_cached(&path, artifact.metadata()).is_none());
    }

    #[test]
    fn concurrent_writes_leave_one_private_context() {
        let artifact = fixture();
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("context.ir");
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| artifact.write(&path).expect("write context"));
            }
        });
        assert_eq!(
            StaticContextArtifact::read_mapped_prefault(&path).expect("read context"),
            artifact
        );
        assert_eq!(
            temporary_files(directory.path()),
            Vec::<std::path::PathBuf>::new()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&path).expect("metadata").permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn failed_write_removes_its_temporary_file() {
        let directory = tempfile::tempdir().expect("temporary directory");
        // Renaming a file over a nonempty directory fails on every platform.
        let path = directory.path().join("context.ir");
        std::fs::create_dir(&path).expect("create directory");
        std::fs::write(path.join("kept"), b"kept").expect("write file");
        assert!(fixture().write(&path).is_err());
        assert_eq!(
            temporary_files(directory.path()),
            Vec::<std::path::PathBuf>::new()
        );
        assert_eq!(std::fs::read(path.join("kept")).expect("read file"), b"kept");
    }
}

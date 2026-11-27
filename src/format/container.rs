//! Shared aligned container for weights, fitted preprocessing, and static contexts.
//!
//! Layout: `b"IRONICL\0"`, little-endian `u64` JSON header length (<= 16 MiB),
//! UTF-8 JSON header, zero padding to `alignment` (`data_start`), then contiguous
//! row-major little-endian tensor payloads in directory order, each padded to
//! `alignment` (including the final tensor). Offsets are relative to `data_start`.

use std::{
    collections::HashSet,
    fs::File,
    io::{Read, Write},
    ops::Range,
    path::Path,
};

#[cfg(unix)]
use memmap2::Advice;
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::artifact_error;
use crate::Result;

pub const MAGIC: [u8; 8] = *b"IRONICL\0";
pub const VERSION: u32 = 1;
pub(crate) const MEMORY_ALIGNMENT: usize = 64;
pub(crate) const DISK_ALIGNMENT: usize = 16_384;
const MAX_HEADER_BYTES: usize = 16 << 20;
pub(crate) const MAX_TENSOR_COUNT: usize = 4096;
pub(crate) const MAX_TENSOR_RANK: usize = 16;
pub(crate) const MAX_TENSOR_NAME_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Model,
    Preprocessing,
    Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    U8,
    I64,
    I32,
    Bf16,
    F16,
    F32,
    F64,
    #[serde(rename = "f8_e4m3")]
    F8E4M3,
}

impl DType {
    pub(crate) const fn size_in_bytes(self) -> usize {
        match self {
            Self::U8 | Self::F8E4M3 => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F8E4M3 => "f8_e4m3",
        }
    }

    pub(crate) fn all_finite(self, bytes: &[u8]) -> Result<bool> {
        if !bytes.len().is_multiple_of(self.size_in_bytes()) {
            return Err(artifact_error("tensor length is not a multiple of its dtype"));
        }
        Ok(match self {
            Self::U8 | Self::I32 | Self::I64 => true,
            Self::F8E4M3 => bytes.iter().all(|&x| x & 0x7f != 0x7f),
            Self::Bf16 => all_le(bytes, u16::from_le_bytes, |x| x & 0x7f80 != 0x7f80),
            Self::F16 => all_le(bytes, u16::from_le_bytes, |x| x & 0x7c00 != 0x7c00),
            Self::F32 => all_le(bytes, f32::from_le_bytes, f32::is_finite),
            Self::F64 => all_le(bytes, f64::from_le_bytes, f64::is_finite),
        })
    }
}

fn all_le<T: bytemuck::Pod, const N: usize>(
    bytes: &[u8],
    decode: fn([u8; N]) -> T,
    ok: impl Fn(T) -> bool,
) -> bool {
    if cfg!(target_endian = "little")
        && let Ok(values) = bytemuck::try_cast_slice::<u8, T>(bytes)
    {
        values.iter().copied().all(ok)
    } else {
        bytes.as_chunks::<N>().0.iter().all(|&w| ok(decode(w)))
    }
}

pub(crate) fn decode_le<T, const N: usize>(bytes: &[u8], decode: fn([u8; N]) -> T) -> Vec<T> {
    bytes.as_chunks::<N>().0.iter().copied().map(decode).collect()
}

pub(crate) fn native_bytes<T: bytemuck::Pod>(values: &[T]) -> Result<&[u8]> {
    if !cfg!(target_endian = "little") {
        return Err(artifact_error("encoding tensors requires a little-endian host"));
    }
    Ok(bytemuck::cast_slice(values))
}

pub(crate) fn checked_tensor_bytes(dtype: DType, shape: &[u64]) -> Result<usize> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes() as u64, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| artifact_error("tensor byte length overflow"))?;
    usize::try_from(bytes).map_err(artifact_error)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Header<M> {
    format: u32,
    pub kind: Kind,
    pub alignment: usize,
    pub metadata: M,
    pub tensors: Vec<Entry>,
    #[serde(skip)]
    pub data_start: usize,
    #[serde(skip)]
    pub file_len: u64,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| artifact_error("container length overflow"))
}

fn align(n: usize, alignment: usize) -> Result<usize> {
    n.checked_next_multiple_of(alignment)
        .ok_or_else(|| artifact_error("aligned length overflow"))
}

impl<M> Header<M> {
    /// Validate header and directory invariants; writers and readers share this check.
    fn validate(&mut self, json_len: usize) -> Result<usize> {
        if self.format != VERSION
            || json_len > MAX_HEADER_BYTES
            || !self.alignment.is_power_of_two()
            || !(MEMORY_ALIGNMENT..=2 << 20).contains(&self.alignment)
            || self.tensors.len() > MAX_TENSOR_COUNT
        {
            return Err(artifact_error(
                "unsupported container version, alignment or directory size",
            ));
        }
        self.data_start = align(16 + json_len, self.alignment)?;
        let mut names = HashSet::with_capacity(self.tensors.len());
        let mut next = 0_usize;
        for t in &self.tensors {
            if t.name.is_empty()
                || t.name.len() > MAX_TENSOR_NAME_BYTES
                || !names.insert(&t.name)
                || t.shape.len() > MAX_TENSOR_RANK
                || checked_tensor_bytes(t.dtype, &t.shape).ok() != Some(t.bytes)
                || t.offset != next
            {
                return Err(artifact_error(format!(
                    "invalid container tensor entry {:?}",
                    t.name
                )));
            }
            next = align(add(next, t.bytes)?, self.alignment)?;
        }
        let total = add(self.data_start, next)?;
        self.file_len = total as u64;
        Ok(total)
    }

    pub const fn range(&self, entry: &Entry) -> Range<usize> {
        let start = self.data_start + entry.offset;
        start..start + entry.bytes
    }

    pub fn check_bytes<'a>(
        &self,
        index: usize,
        name: &str,
        dtype: DType,
        shape: &[u64],
        bytes: &'a [u8],
    ) -> Result<&'a [u8]> {
        let entry = self
            .tensors
            .get(index)
            .filter(|t| t.name == name && t.dtype == dtype && t.shape == shape)
            .ok_or_else(|| artifact_error(format!("unexpected container tensor: expected {name}")))?;
        Ok(&bytes[self.range(entry)])
    }
}

fn header_length(prefix: &[u8], file_len: usize) -> Result<usize> {
    if prefix.len() != 16 || prefix[..8] != MAGIC {
        return Err(artifact_error(
            "not an IRONICL container; regenerate legacy artifacts",
        ));
    }
    let len = u64::from_le_bytes(prefix[8..].try_into().expect("eight bytes"));
    if len > MAX_HEADER_BYTES as u64 || len.saturating_add(16) > file_len as u64 {
        return Err(artifact_error("invalid JSON header length"));
    }
    usize::try_from(len).map_err(artifact_error)
}

impl<M: DeserializeOwned> Header<M> {
    fn parse(json: &[u8], len: usize, expected: Option<Kind>) -> Result<Self> {
        let mut header: Self = serde_json::from_slice(json).map_err(artifact_error)?;
        if expected.is_some_and(|kind| kind != header.kind) || header.validate(json.len())? != len {
            return Err(artifact_error("wrong artifact kind or file length"));
        }
        Ok(header)
    }

    pub fn from_bytes(bytes: &[u8], kind: Kind) -> Result<Self> {
        let n = header_length(bytes.get(..16).unwrap_or_default(), bytes.len())?;
        Self::parse(&bytes[16..16 + n], bytes.len(), Some(kind))
    }

    pub fn open(path: &Path, kind: impl Into<Option<Kind>>, max: u64) -> Result<Self> {
        let mut file = File::open(path).map_err(artifact_error)?;
        let raw_len = file.metadata().map_err(artifact_error)?.len();
        if raw_len > max {
            return Err(artifact_error("artifact exceeds size limit"));
        }
        let len = usize::try_from(raw_len).map_err(artifact_error)?;
        let mut prefix = [0; 16];
        file.read_exact(&mut prefix).map_err(artifact_error)?;
        let mut json = vec![0; header_length(&prefix, len)?];
        file.read_exact(&mut json).map_err(artifact_error)?;
        Self::parse(&json, len, kind.into())
    }
}

/// Inspect the envelope without decoding model-specific metadata or reading tensor data.
///
/// # Errors
/// If the file cannot be opened or its container layout is invalid.
pub fn inspect(path: &Path) -> Result<(Kind, usize, usize)> {
    let header = Header::<serde::de::IgnoredAny>::open(path, None, u64::MAX)?;
    Ok((header.kind, header.alignment, header.tensors.len()))
}

/// Map one immutable artifact read-only. Writers must publish by atomic rename,
/// never truncate or edit a mapped file in place.
#[expect(unsafe_code, reason = "read-only mapping of an immutable artifact")]
pub(crate) fn map_immutable(path: &Path, max: u64) -> Result<Mmap> {
    let file = File::open(path).map_err(artifact_error)?;
    let len = file.metadata().map_err(artifact_error)?.len();
    if !(16..=max).contains(&len) {
        return Err(artifact_error("invalid mapped artifact length"));
    }
    // SAFETY: artifacts are immutable while mapped; all writers replace by atomic rename.
    unsafe {
        MmapOptions::new()
            .len(usize::try_from(len).map_err(artifact_error)?)
            .map(&file)
    }
    .map_err(artifact_error)
}

/// Copy an in-memory container into a read-only, page-aligned anonymous mapping.
pub(crate) fn map_bytes(bytes: &[u8], max: usize) -> Result<Mmap> {
    if !(16..=max).contains(&bytes.len()) {
        return Err(artifact_error("artifact length is outside the resource limit"));
    }
    let mut mapping = MmapOptions::new()
        .len(bytes.len())
        .map_anon()
        .map_err(artifact_error)?;
    mapping.copy_from_slice(bytes);
    mapping.make_read_only().map_err(artifact_error)
}

/// Best-effort OS read-ahead hint so the first read runs at RAM speed.
pub(crate) fn prefault(mapping: &Mmap) {
    #[cfg(unix)]
    let _ = mapping.advise(Advice::WillNeed);
    #[cfg(windows)]
    prefetch_windows(mapping);
    #[cfg(not(any(unix, windows)))]
    let _ = mapping;
}

#[cfg(windows)]
#[expect(unsafe_code, reason = "One Win32 call; memmap2 does not expose it")]
fn prefetch_windows(mapping: &Mmap) {
    use windows_sys::Win32::System::{
        Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY},
        Threading::GetCurrentProcess,
    };
    let range = WIN32_MEMORY_RANGE_ENTRY {
        VirtualAddress: mapping.as_ptr().cast_mut().cast(),
        NumberOfBytes: mapping.len(),
    };
    // SAFETY: `range` describes this process's live mapping for the duration of the call.
    let _ = unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &raw const range, 0) };
}

/// Payload slices stay with the caller; only the header and directory are buffered.
pub(crate) struct Writer<'a> {
    json: Vec<u8>,
    payloads: Vec<&'a [u8]>,
    alignment: usize,
    pub len: usize,
}

impl<'a> Writer<'a> {
    pub fn new<M: Serialize>(
        kind: Kind,
        alignment: usize,
        metadata: M,
        tensors: impl IntoIterator<Item = (String, DType, Vec<u64>, &'a [u8])>,
    ) -> Result<Self> {
        if !matches!(alignment, MEMORY_ALIGNMENT | DISK_ALIGNMENT) {
            return Err(artifact_error("invalid writer alignment"));
        }
        let mut header = Header {
            format: VERSION,
            kind,
            alignment,
            metadata,
            tensors: Vec::new(),
            data_start: 0,
            file_len: 0,
        };
        let (mut offset, mut payloads) = (0_usize, Vec::new());
        for (name, dtype, shape, payload) in tensors {
            let bytes = payload.len();
            header.tensors.push(Entry {
                name,
                dtype,
                shape,
                offset,
                bytes,
            });
            payloads.push(payload);
            offset = align(add(offset, bytes)?, alignment)?;
        }
        let json = serde_json::to_vec(&header).map_err(artifact_error)?;
        let len = header.validate(json.len())?;
        Ok(Self {
            json,
            payloads,
            alignment,
            len,
        })
    }

    pub fn write(&self, w: &mut impl Write) -> Result<()> {
        static ZERO: [u8; DISK_ALIGNMENT] = [0; DISK_ALIGNMENT];
        let pad = |n: usize| align(n, self.alignment).map(|end| &ZERO[..end - n]);
        w.write_all(&MAGIC).map_err(artifact_error)?;
        w.write_all(&(self.json.len() as u64).to_le_bytes())
            .map_err(artifact_error)?;
        w.write_all(&self.json).map_err(artifact_error)?;
        w.write_all(pad(16 + self.json.len())?).map_err(artifact_error)?;
        for &bytes in &self.payloads {
            w.write_all(bytes).map_err(artifact_error)?;
            w.write_all(pad(bytes.len())?).map_err(artifact_error)?;
        }
        Ok(())
    }

    pub fn bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(self.len);
        self.write(&mut bytes)?;
        Ok(bytes)
    }
}

/// Hex strings on disk, fixed-size digests in memory. Identifies inputs, not payload integrity.
pub(crate) mod digest {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&crate::format::encode_sha256(value))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        crate::models::checkpoint::decode_sha256(&String::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    #[test]
    fn rejects_malformed_directories_without_reading_payloads() {
        let t = json!({"name":"k","dtype":"f32","shape":[1],"offset":0,"bytes":4});
        let base = json!({"format":1,"kind":"context","alignment":4096,"metadata":null,"tensors":[t]});
        let parse =
            |h: &Value| Header::<()>::parse(&serde_json::to_vec(h).unwrap(), 8192, Some(Kind::Context));
        assert!(parse(&base).is_ok());
        for (field, value) in [
            ("name", json!("")),
            ("dtype", json!("unknown")),
            ("shape", json!([u64::MAX, 2])),
            ("shape", json!(vec![1; 17])),
            ("offset", json!(1)),
            ("bytes", json!(3)),
        ] {
            let mut h = base.clone();
            h["tensors"][0][field] = value;
            assert!(parse(&h).is_err(), "{field}");
        }
        for (field, value) in [
            ("format", json!(0)),
            ("kind", json!("model")),
            ("alignment", json!(0)),
            ("alignment", json!(48)),
        ] {
            let mut h = base.clone();
            h[field] = value;
            assert!(parse(&h).is_err(), "{field}");
        }
        let mut two = base;
        let mut t2 = t;
        t2["offset"] = json!(4096);
        two["tensors"].as_array_mut().unwrap().push(t2);
        let parse_two =
            |h: &Value| Header::<()>::parse(&serde_json::to_vec(h).unwrap(), 12288, Some(Kind::Context));
        assert!(parse_two(&two).is_err(), "duplicate name");
        two["tensors"][1]["name"] = json!("v");
        assert!(parse_two(&two).is_ok());
        two["tensors"][1]["offset"] = json!(0);
        assert!(parse_two(&two).is_err(), "overlap");
        let mut prefix = [0; 16];
        prefix[..8].copy_from_slice(&MAGIC);
        prefix[8..].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(header_length(&prefix, usize::MAX).is_err());
    }

    #[test]
    fn finiteness_checks_every_float_dtype_including_unaligned_buffers() {
        let check = |dtype: DType, payload: &[u8], expected: bool| {
            let mut unaligned = vec![0xaa];
            unaligned.extend_from_slice(payload);
            assert_eq!(dtype.all_finite(&unaligned[1..]).unwrap(), expected, "{dtype:?}");
        };
        check(DType::F32, &1.0_f32.to_le_bytes(), true);
        check(DType::F32, &f32::NAN.to_le_bytes(), false);
        check(DType::F32, &f32::INFINITY.to_le_bytes(), false);
        check(DType::F64, &(-42.5_f64).to_le_bytes(), true);
        check(DType::F64, &f64::NEG_INFINITY.to_le_bytes(), false);
        check(DType::F16, &0x7bff_u16.to_le_bytes(), true);
        check(DType::F16, &0x7c00_u16.to_le_bytes(), false);
        check(DType::Bf16, &0x7f7f_u16.to_le_bytes(), true);
        check(DType::Bf16, &0x7f80_u16.to_le_bytes(), false);
        check(DType::F8E4M3, &[0x7e, 0xfe], true);
        check(DType::F8E4M3, &[0x7f], false);
        check(DType::F8E4M3, &[0xff], false);
        assert!(DType::F32.all_finite(&[0; 3]).is_err());
    }
}

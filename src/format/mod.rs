//! Shared primitives for Ironicl's on-disk artifact containers.
//!
//! Every artifact format — the model artifact, the preprocessor, and the static
//! context — is a little-endian container: a fixed header, an optional
//! 64-byte-aligned tensor payload, and a trailing SHA-256 digest over all
//! preceding bytes. This module holds the byte-level primitives they all share.
//! Each format still owns its own header layout, magic, version, and dtype
//! numbering; only the mechanics live here.

pub mod container;
pub mod model;
pub mod preprocessor;
pub mod static_context;

use std::{
    fmt,
    fs::File,
    io::{self, Read},
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Length of the trailing SHA-256 digest, in bytes.
pub(crate) const CHECKSUM_BYTES: usize = 32;
/// Alignment for tensor payloads: the widest SIMD/mmap requirement on the
/// supported targets.
pub(crate) const TENSOR_ALIGNMENT: usize = 64;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// SHA-256 digest of `bytes`.
pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Read a complete small artifact without trusting either the reported file size or EOF.
pub(crate) fn read_bounded(path: &Path, limit: u64) -> io::Result<Vec<u8>> {
    let limit = limit.min(isize::MAX as u64);
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    if length > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("file is {length} bytes; limit is {limit}"),
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("file grew beyond the {limit}-byte limit while reading"),
        ));
    }
    Ok(bytes)
}

pub(crate) fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Write and validate a new owner-only file before publishing it beside the destination.
/// Close and remove the temporary on any error; never modify the old mapping in place.
pub(crate) fn write_atomic<T>(
    path: &Path,
    write: impl FnOnce(&mut tempfile::NamedTempFile) -> Result<T>,
) -> Result<T> {
    let io_error = |e| artifact_error(format!("{}: {e}", path.display()));
    let mut file = tempfile::Builder::new()
        .prefix(".ironicl-")
        .suffix(".tmp")
        .tempfile_in(parent_dir(path))
        .map_err(io_error)?;
    let result = write(&mut file)?;
    file.as_file().sync_all().map_err(io_error)?;
    persist_atomic(file, path).map_err(io_error)?;
    Ok(result)
}

/// Atomically move `file` onto `path`. On Windows, concurrent writers replacing
/// the same destination can transiently return `ERROR_ACCESS_DENIED` (5) or
/// `ERROR_SHARING_VIOLATION` (32), so retry briefly unless `path` is a directory.
pub(crate) fn persist_atomic(file: tempfile::NamedTempFile, path: &std::path::Path) -> std::io::Result<()> {
    #[cfg(not(windows))]
    {
        file.persist(path).map(drop).map_err(|err| err.error)
    }
    #[cfg(windows)]
    {
        const RETRIES: u32 = 20;
        let mut current = file;
        for attempt in 0..=RETRIES {
            match current.persist(path) {
                Ok(_) => return Ok(()),
                Err(err) => {
                    let transient = err.error.kind() == std::io::ErrorKind::PermissionDenied
                        || err.error.raw_os_error() == Some(32);
                    if attempt == RETRIES || !transient || path.is_dir() {
                        return Err(err.error);
                    }
                    current = err.file;
                    std::thread::sleep(std::time::Duration::from_millis(5 * u64::from(attempt + 1)));
                }
            }
        }
        unreachable!("loop returns on attempt == RETRIES")
    }
}

/// An [`Error::Artifact`] from any displayable cause.
pub(crate) fn artifact_error(error: impl fmt::Display) -> Error {
    Error::Artifact(error.to_string())
}

/// Lowercase hex encoding of a 32-byte digest.
pub(crate) fn encode_sha256(digest: &[u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

pub(crate) fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn u64_to_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| artifact_error(format!("{name} exceeds usize")))
}

/// Round `value` up to the next multiple of `alignment`.
pub(crate) fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| artifact_error("alignment calculation overflows usize"))
    }
}

/// Pad `bytes` with zeros up to the next `alignment` boundary.
pub(crate) fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) -> Result<()> {
    let aligned = align_up(bytes.len(), alignment)?;
    bytes.resize(aligned, 0);
    Ok(())
}

/// A cursor over a little-endian byte buffer. Every read is bounds-checked and
/// returns [`Error::Artifact`] on truncation.
pub(crate) struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    pub(crate) fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.read_bytes(N)?
            .try_into()
            .map_err(|_| artifact_error("artifact field has an invalid length"))
    }

    pub(crate) fn read_bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| artifact_error("artifact offset overflows usize"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| artifact_error("artifact is truncated"))?;
        self.position = end;
        Ok(bytes)
    }
}

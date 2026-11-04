//! Shared primitives for Ironicl's on-disk artifact containers.
//!
//! Every artifact format — the model artifact, the static context, the query
//! bundle, and the input bundle — is a little-endian container: a fixed header,
//! an optional 64-byte-aligned tensor payload, and a trailing SHA-256 digest
//! over all preceding bytes. This module holds the byte-level primitives they
//! all share. Each format still owns its own header layout, magic, version, and
//! dtype numbering; only the mechanics live here.

use std::fmt;

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

pub(crate) fn usize_to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| artifact_error(format!("{name} exceeds u32")))
}

pub(crate) fn usize_to_u64(value: usize, name: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| artifact_error(format!("{name} exceeds u64")))
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

    pub(crate) fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
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

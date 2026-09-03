//! Checked query-only bundles for one exact static context.

use std::{fs::File, io::Read, path::Path};

use sha2::{Digest, Sha256};

use crate::{ClassifierQueryRequest, Error, Result};

/// Magic bytes at the start of a static-context query bundle.
pub const STATIC_QUERY_BUNDLE_MAGIC: [u8; 8] = *b"IRQUERY\0";
/// Supported static-context query bundle version.
pub const STATIC_QUERY_BUNDLE_VERSION: u32 = 1;

const STATIC_QUERY_BUNDLE_KIND: u32 = 1;
const HEADER_BYTES: usize = 112;
const CHECKSUM_BYTES: usize = 32;
const MAX_BUNDLE_BYTES: usize = 512 * 1024 * 1024;
const MAX_ENSEMBLE_COUNT: usize = 4096;
const MAX_QUERY_ROWS: usize = 1_000_000;
const MAX_FEATURE_COUNT: usize = 65_536;

/// Owned preprocessed queries with shape `[B,Q,H]` and exact artifact identities.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticQueryBundle {
    model_sha256: [u8; 32],
    context_sha256: [u8; 32],
    ensemble_count: usize,
    query_rows: usize,
    feature_count: usize,
    x: Vec<f32>,
}

impl StaticQueryBundle {
    /// Decode and verify a complete IRQUERY v1 bundle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InputBundle`] for malformed, corrupt, or invalid input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES || bytes.len() > MAX_BUNDLE_BYTES {
            return Err(query_error("query bundle length is outside the resource limit"));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        if sha256(&bytes[..checksum_offset]).as_slice() != &bytes[checksum_offset..] {
            return Err(query_error("query bundle checksum does not match"));
        }
        let mut reader = ByteReader::new(&bytes[..HEADER_BYTES]);
        if reader.read_array::<8>()? != STATIC_QUERY_BUNDLE_MAGIC {
            return Err(query_error("query bundle magic does not match"));
        }
        if reader.read_u32()? != STATIC_QUERY_BUNDLE_VERSION || reader.read_u32()? != STATIC_QUERY_BUNDLE_KIND
        {
            return Err(query_error("query bundle version or kind is unsupported"));
        }
        let model_sha256 = reader.read_array()?;
        let context_sha256 = reader.read_array()?;
        if model_sha256 == [0; 32] || context_sha256 == [0; 32] {
            return Err(query_error("query bundle identity is empty"));
        }
        let ensemble_count = to_usize(reader.read_u64()?, "ensemble count")?;
        let query_rows = to_usize(reader.read_u64()?, "query rows")?;
        let feature_count = to_usize(reader.read_u64()?, "feature count")?;
        let elements = to_usize(reader.read_u64()?, "query element count")?;
        debug_assert!(reader.is_empty());
        if ensemble_count == 0
            || ensemble_count > MAX_ENSEMBLE_COUNT
            || query_rows == 0
            || query_rows > MAX_QUERY_ROWS
            || feature_count == 0
            || feature_count > MAX_FEATURE_COUNT
        {
            return Err(query_error(
                "query bundle dimensions are outside the resource limits",
            ));
        }
        let expected_elements = ensemble_count
            .checked_mul(query_rows)
            .and_then(|value| value.checked_mul(feature_count))
            .ok_or_else(|| query_error("query element count overflows usize"))?;
        let expected_length = HEADER_BYTES
            .checked_add(
                expected_elements
                    .checked_mul(4)
                    .ok_or_else(|| query_error("query byte length overflows usize"))?,
            )
            .and_then(|value| value.checked_add(CHECKSUM_BYTES))
            .ok_or_else(|| query_error("query bundle length overflows usize"))?;
        if elements != expected_elements || bytes.len() != expected_length {
            return Err(query_error("query bundle dimensions and length do not match"));
        }
        let mut x = Vec::with_capacity(elements);
        for chunk in bytes[HEADER_BYTES..checksum_offset].chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !value.is_finite() {
                return Err(query_error("query bundle contains a non-finite value"));
            }
            x.push(value);
        }
        Ok(Self {
            model_sha256,
            context_sha256,
            ensemble_count,
            query_rows,
            feature_count,
            x,
        })
    }

    /// Read and verify a bounded IRQUERY v1 file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InputBundle`] when the file cannot be read or is invalid.
    pub fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(query_error)?;
        let length = to_usize(file.metadata().map_err(query_error)?.len(), "query bundle length")?;
        if length > MAX_BUNDLE_BYTES {
            return Err(query_error("query bundle exceeds the resource limit"));
        }
        let mut bytes = Vec::with_capacity(length);
        file.take((MAX_BUNDLE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(query_error)?;
        Self::from_bytes(&bytes)
    }

    /// Borrow this bundle as an executor request.
    ///
    /// # Errors
    ///
    /// Returns an error only if internal validated dimensions cannot form a request.
    pub fn request(&self) -> Result<ClassifierQueryRequest<'_>> {
        ClassifierQueryRequest::new(self.ensemble_count, self.query_rows, self.feature_count, &self.x)
    }

    /// Source-checkpoint identity of the required model.
    #[must_use]
    pub fn model_sha256(&self) -> &[u8; 32] {
        &self.model_sha256
    }

    /// Identity of the required static context.
    #[must_use]
    pub fn context_sha256(&self) -> &[u8; 32] {
        &self.context_sha256
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn to_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| query_error(format!("{name} exceeds usize")))
}

fn query_error(message: impl std::fmt::Display) -> Error {
    Error::InputBundle(format!("static query: {message}"))
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| query_error("query offset overflows usize"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| query_error("query bundle is truncated"))?;
        self.offset = end;
        value
            .try_into()
            .map_err(|_| query_error("query field length is invalid"))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&STATIC_QUERY_BUNDLE_MAGIC);
        bytes.extend_from_slice(&STATIC_QUERY_BUNDLE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&STATIC_QUERY_BUNDLE_KIND.to_le_bytes());
        bytes.extend_from_slice(&[1; 32]);
        bytes.extend_from_slice(&[2; 32]);
        for value in [1_u64, 2, 3, 6] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&sha256(&bytes));
        bytes
    }

    #[test]
    fn decodes_checked_query() {
        let query = StaticQueryBundle::from_bytes(&fixture()).expect("valid query");
        let request = query.request().expect("valid request");
        assert_eq!(
            (
                request.ensemble_count(),
                request.query_rows(),
                request.feature_count()
            ),
            (1, 2, 3)
        );
        assert_eq!(query.model_sha256(), &[1; 32]);
        assert_eq!(query.context_sha256(), &[2; 32]);
    }

    #[test]
    fn rejects_corrupt_and_non_finite_queries() {
        let mut corrupt = fixture();
        corrupt[HEADER_BYTES] ^= 1;
        assert!(StaticQueryBundle::from_bytes(&corrupt).is_err());

        let mut non_finite = fixture();
        non_finite[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let checksum_offset = non_finite.len() - CHECKSUM_BYTES;
        let checksum = sha256(&non_finite[..checksum_offset]);
        non_finite[checksum_offset..].copy_from_slice(&checksum);
        assert!(StaticQueryBundle::from_bytes(&non_finite).is_err());
    }
}

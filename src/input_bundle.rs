//! Checked binary input bundles for preprocessed classifier tensors.
//!
//! The format stores little-endian FP32 values in this order: a fixed header,
//! `X` with shape `[B, T, H]`, `y_train` with shape `[B, N]`, and a trailing
//! SHA-256 digest over all preceding bytes. It is independent of model execution.

use std::{fs::File, io::Read, path::Path};

use sha2::{Digest, Sha256};

use crate::{
    error::{Error, Result},
    model_artifact::encode_sha256,
};

/// Magic bytes at the start of a classifier input bundle.
pub const CLASSIFIER_INPUT_BUNDLE_MAGIC: [u8; 8] = *b"IRINPUT\0";
/// Supported classifier input bundle format version.
pub const CLASSIFIER_INPUT_BUNDLE_VERSION: u32 = 1;
/// Format kind for classifier input tensors.
pub const CLASSIFIER_INPUT_BUNDLE_KIND: u32 = 1;

const HEADER_BYTES: usize = 80;
const CHECKSUM_BYTES: usize = 32;
const MAX_BUNDLE_BYTES: usize = 512 * 1024 * 1024;
const MAX_ENSEMBLE_COUNT: usize = 4096;
const MAX_TOTAL_ROWS: usize = 1_000_000;
const MAX_FEATURE_COUNT: usize = 65_536;
const MAX_TRAINING_ROWS: usize = 1_000_000;

/// Owned, validated preprocessed tensors for one classifier request bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierInputBundle {
    ensemble_count: usize,
    total_rows: usize,
    feature_count: usize,
    training_rows: usize,
    x: Vec<f32>,
    y_train: Vec<f32>,
}

impl ClassifierInputBundle {
    /// Validate and own classifier tensors with shapes `[B, T, H]` and `[B, N]`.
    ///
    /// Labels must be finite integer-valued FP32 values from 0 through 9.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InputBundle`] for invalid dimensions, lengths, tensor
    /// values, or labels.
    pub fn new(
        ensemble_count: usize,
        total_rows: usize,
        feature_count: usize,
        training_rows: usize,
        x: Vec<f32>,
        y_train: Vec<f32>,
    ) -> Result<Self> {
        validate_dimensions(ensemble_count, total_rows, feature_count, training_rows)?;
        validate_lengths(
            ensemble_count,
            total_rows,
            feature_count,
            training_rows,
            x.len(),
            y_train.len(),
        )?;
        validate_values(&x, &y_train)?;
        Ok(Self {
            ensemble_count,
            total_rows,
            feature_count,
            training_rows,
            x,
            y_train,
        })
    }

    /// Decode and verify a complete bundle from bytes. The returned bundle owns its tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InputBundle`] for malformed, truncated, corrupt, or
    /// invalid input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(input_error("bundle is truncated"));
        }
        if bytes.len() > MAX_BUNDLE_BYTES {
            return Err(input_error("bundle exceeds the resource limit"));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        if sha256(&bytes[..checksum_offset]).as_slice() != &bytes[checksum_offset..] {
            return Err(input_error("bundle checksum does not match"));
        }

        let mut header = ByteReader::new(&bytes[..HEADER_BYTES]);
        if header.read_array::<8>()? != CLASSIFIER_INPUT_BUNDLE_MAGIC {
            return Err(input_error("bundle magic does not match"));
        }
        let version = header.read_u32()?;
        if version != CLASSIFIER_INPUT_BUNDLE_VERSION {
            return Err(input_error(format!(
                "bundle version {version} is unsupported; expected {CLASSIFIER_INPUT_BUNDLE_VERSION}"
            )));
        }
        if header.read_u32()? != CLASSIFIER_INPUT_BUNDLE_KIND {
            return Err(input_error("bundle kind is not a classifier input"));
        }
        let ensemble_count = to_usize(header.read_u64()?, "ensemble count")?;
        let total_rows = to_usize(header.read_u64()?, "total rows")?;
        let feature_count = to_usize(header.read_u64()?, "feature count")?;
        let training_rows = to_usize(header.read_u64()?, "training rows")?;
        let x_elements = to_usize(header.read_u64()?, "X element count")?;
        let y_elements = to_usize(header.read_u64()?, "y_train element count")?;
        let x_bytes = to_usize(header.read_u64()?, "X byte length")?;
        let y_bytes = to_usize(header.read_u64()?, "y_train byte length")?;
        debug_assert!(header.is_empty());

        validate_dimensions(ensemble_count, total_rows, feature_count, training_rows)?;
        let expected_x = checked_product(&[ensemble_count, total_rows, feature_count], "X element count")?;
        let expected_y = checked_product(&[ensemble_count, training_rows], "y_train element count")?;
        if x_elements != expected_x || y_elements != expected_y {
            return Err(input_error(
                "declared tensor element counts do not match dimensions",
            ));
        }
        let expected_feature_bytes = checked_bytes(x_elements, "X byte length")?;
        let expected_label_bytes = checked_bytes(y_elements, "y_train byte length")?;
        if x_bytes != expected_feature_bytes || y_bytes != expected_label_bytes {
            return Err(input_error(
                "declared tensor byte lengths do not match element counts",
            ));
        }
        let expected_length = HEADER_BYTES
            .checked_add(x_bytes)
            .and_then(|length| length.checked_add(y_bytes))
            .and_then(|length| length.checked_add(CHECKSUM_BYTES))
            .ok_or_else(|| input_error("bundle length overflows usize"))?;
        if bytes.len() != expected_length {
            return Err(input_error(
                "bundle section lengths do not match whole-file length",
            ));
        }
        let x_end = HEADER_BYTES + x_bytes;
        let x = decode_f32(&bytes[HEADER_BYTES..x_end]);
        let y_train = decode_f32(&bytes[x_end..checksum_offset]);
        Self::new(
            ensemble_count,
            total_rows,
            feature_count,
            training_rows,
            x,
            y_train,
        )
    }

    /// Read, decode, and verify a bundle from a bounded file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InputBundle`] when the file cannot be read or its bytes
    /// do not form a valid bundle.
    pub fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(input_error)?;
        let length = file.metadata().map_err(input_error)?.len();
        let length = to_usize(length, "bundle length")?;
        if length > MAX_BUNDLE_BYTES {
            return Err(input_error("bundle exceeds the resource limit"));
        }
        let mut bytes = Vec::with_capacity(length);
        file.take((MAX_BUNDLE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(input_error)?;
        Self::from_bytes(&bytes)
    }

    /// Encode this bundle deterministically for tests and external oracle tools.
    ///
    /// The output always uses little-endian FP32 values and ends with a SHA-256 digest.
    ///
    /// # Panics
    ///
    /// Panics if a platform cannot represent a validated bundle length as `u64`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes =
            Vec::with_capacity(HEADER_BYTES + (self.x.len() + self.y_train.len()) * 4 + CHECKSUM_BYTES);
        bytes.extend_from_slice(&CLASSIFIER_INPUT_BUNDLE_MAGIC);
        write_u32(&mut bytes, CLASSIFIER_INPUT_BUNDLE_VERSION);
        write_u32(&mut bytes, CLASSIFIER_INPUT_BUNDLE_KIND);
        for value in [
            self.ensemble_count,
            self.total_rows,
            self.feature_count,
            self.training_rows,
            self.x.len(),
            self.y_train.len(),
            self.x.len() * 4,
            self.y_train.len() * 4,
        ] {
            write_u64(&mut bytes, u64::try_from(value).expect("validated size fits u64"));
        }
        for value in self.x.iter().chain(&self.y_train) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&sha256(&bytes));
        bytes
    }

    /// Borrow the tensors and dimensions needed to form a classifier request.
    #[must_use]
    pub fn request(&self) -> ClassifierRequest<'_> {
        ClassifierRequest {
            ensemble_count: self.ensemble_count,
            total_rows: self.total_rows,
            feature_count: self.feature_count,
            training_rows: self.training_rows,
            x: &self.x,
            y_train: &self.y_train,
        }
    }

    /// SHA-256 digest of the deterministic encoded bundle, in lowercase hexadecimal.
    ///
    /// # Panics
    ///
    /// Panics if a platform cannot represent a validated bundle length as `u64`.
    #[must_use]
    pub fn sha256(&self) -> String {
        let bytes = self.to_bytes();
        let digest: [u8; 32] = bytes[bytes.len() - CHECKSUM_BYTES..]
            .try_into()
            .expect("encoded bundle has a checksum");
        encode_sha256(&digest)
    }
}

/// Borrowed classifier tensors and dimensions from a [`ClassifierInputBundle`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifierRequest<'a> {
    ensemble_count: usize,
    total_rows: usize,
    feature_count: usize,
    training_rows: usize,
    x: &'a [f32],
    y_train: &'a [f32],
}

/// Borrowed preprocessed test queries with shape `[B, Q, H]`.
///
/// This request is for a [`crate::StaticClassifierExecutor`]. The static
/// context supplies the training rows and labels, so query execution accepts
/// neither of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifierQueryRequest<'a> {
    ensemble_count: usize,
    query_rows: usize,
    feature_count: usize,
    x: &'a [f32],
}

impl<'a> ClassifierQueryRequest<'a> {
    /// Validate borrowed FP32 query values with shape `[B, Q, H]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when dimensions, length, or values
    /// are invalid.
    pub fn new(ensemble_count: usize, query_rows: usize, feature_count: usize, x: &'a [f32]) -> Result<Self> {
        if ensemble_count == 0
            || ensemble_count > MAX_ENSEMBLE_COUNT
            || query_rows == 0
            || query_rows > MAX_TOTAL_ROWS
            || feature_count == 0
            || feature_count > MAX_FEATURE_COUNT
        {
            return Err(Error::InvalidArgument(format!(
                "invalid query dimensions B={ensemble_count}, Q={query_rows}, H={feature_count}"
            )));
        }
        if x.len()
            != checked_product(
                &[ensemble_count, query_rows, feature_count],
                "query X element count",
            )?
        {
            return Err(Error::InvalidArgument(
                "query X length does not match dimensions".into(),
            ));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "query X contains a non-finite value".into(),
            ));
        }
        Ok(Self {
            ensemble_count,
            query_rows,
            feature_count,
            x,
        })
    }

    /// Ensemble count `B`.
    #[must_use]
    pub fn ensemble_count(self) -> usize {
        self.ensemble_count
    }

    /// Query row count `Q`.
    #[must_use]
    pub fn query_rows(self) -> usize {
        self.query_rows
    }

    /// Feature count `H`.
    #[must_use]
    pub fn feature_count(self) -> usize {
        self.feature_count
    }

    /// FP32 query values in `[B, Q, H]` row-major order.
    #[must_use]
    pub fn x(self) -> &'a [f32] {
        self.x
    }
}

impl<'a> ClassifierRequest<'a> {
    /// Ensemble count `B`.
    #[must_use]
    pub fn ensemble_count(self) -> usize {
        self.ensemble_count
    }
    /// Total row count `T`.
    #[must_use]
    pub fn total_rows(self) -> usize {
        self.total_rows
    }
    /// Feature count `H`.
    #[must_use]
    pub fn feature_count(self) -> usize {
        self.feature_count
    }
    /// Training row count `N`.
    #[must_use]
    pub fn training_rows(self) -> usize {
        self.training_rows
    }
    /// FP32 feature values in `[B, T, H]` row-major order.
    #[must_use]
    pub fn x(self) -> &'a [f32] {
        self.x
    }
    /// FP32 class labels in `[B, N]` row-major order.
    #[must_use]
    pub fn y_train(self) -> &'a [f32] {
        self.y_train
    }
}

fn validate_dimensions(b: usize, t: usize, h: usize, n: usize) -> Result<()> {
    if b == 0
        || b > MAX_ENSEMBLE_COUNT
        || t == 0
        || t > MAX_TOTAL_ROWS
        || h == 0
        || h > MAX_FEATURE_COUNT
        || n == 0
        || n > MAX_TRAINING_ROWS
        || t <= n
    {
        return Err(input_error(format!(
            "invalid dimensions B={b}, T={t}, H={h}, N={n}; require B, H, N > 0, T > N, and resource limits"
        )));
    }
    Ok(())
}

fn validate_lengths(b: usize, t: usize, h: usize, n: usize, x_length: usize, y_length: usize) -> Result<()> {
    if x_length != checked_product(&[b, t, h], "X element count")?
        || y_length != checked_product(&[b, n], "y_train element count")?
    {
        return Err(input_error("tensor lengths do not match dimensions"));
    }
    let x_bytes = checked_bytes(x_length, "X byte length")?;
    let y_bytes = checked_bytes(y_length, "y_train byte length")?;
    let total_bytes = HEADER_BYTES
        .checked_add(x_bytes)
        .and_then(|length| length.checked_add(y_bytes))
        .and_then(|length| length.checked_add(CHECKSUM_BYTES))
        .ok_or_else(|| input_error("bundle length overflows usize"))?;
    if total_bytes > MAX_BUNDLE_BYTES {
        return Err(input_error("bundle exceeds the resource limit"));
    }
    Ok(())
}

fn validate_values(x: &[f32], y_train: &[f32]) -> Result<()> {
    if x.iter().any(|value| !value.is_finite()) {
        return Err(input_error("X contains a non-finite value"));
    }
    if y_train
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || *value > 9.0 || value.fract() != 0.0)
    {
        return Err(input_error(
            "y_train labels must be finite integer-valued FP32 values in 0..=9",
        ));
    }
    Ok(())
}

fn checked_product(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| input_error(format!("{name} overflows usize")))
    })
}

fn checked_bytes(elements: usize, name: &str) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| input_error(format!("{name} overflows usize")))
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("FP32 chunk has four bytes")))
        .collect()
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn to_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| input_error(format!("{name} exceeds usize")))
}
fn input_error(message: impl std::fmt::Display) -> Error {
    Error::InputBundle(message.to_string())
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
    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }
    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.read_bytes(N)?
            .try_into()
            .map_err(|_| input_error("bundle field has an invalid length"))
    }
    fn read_bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| input_error("bundle offset overflows usize"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| input_error("bundle is truncated"))?;
        self.position = end;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ClassifierInputBundle {
        ClassifierInputBundle::new(2, 3, 2, 2, vec![0.0; 12], vec![0.0, 9.0, 1.0, 8.0])
            .expect("valid fixture")
    }

    #[test]
    fn round_trip_is_deterministic() {
        let bundle = fixture();
        let bytes = bundle.to_bytes();
        assert_eq!(bytes, bundle.to_bytes());
        assert_eq!(
            ClassifierInputBundle::from_bytes(&bytes).expect("read bundle"),
            bundle
        );
        let request = bundle.request();
        assert_eq!(
            (
                request.ensemble_count(),
                request.total_rows(),
                request.feature_count(),
                request.training_rows()
            ),
            (2, 3, 2, 2)
        );
    }

    #[test]
    fn rejects_corruption_and_truncation() {
        let mut corrupt = fixture().to_bytes();
        corrupt[HEADER_BYTES] ^= 1;
        assert!(ClassifierInputBundle::from_bytes(&corrupt).is_err());
        let truncated = fixture().to_bytes();
        assert!(ClassifierInputBundle::from_bytes(&truncated[..truncated.len() - 1]).is_err());
    }

    #[test]
    fn rejects_invalid_shapes_and_noninteger_labels() {
        assert!(ClassifierInputBundle::new(1, 2, 1, 2, vec![0.0; 2], vec![0.0; 2]).is_err());
        assert!(ClassifierInputBundle::new(1, 2, 1, 1, vec![0.0; 2], vec![1.5]).is_err());

        let mut malformed = fixture().to_bytes();
        malformed[24..32].copy_from_slice(&2_u64.to_le_bytes());
        let checksum_offset = malformed.len() - CHECKSUM_BYTES;
        let checksum = sha256(&malformed[..checksum_offset]);
        malformed[checksum_offset..].copy_from_slice(&checksum);
        assert!(ClassifierInputBundle::from_bytes(&malformed).is_err());
    }

    #[test]
    fn query_request_rejects_shape_and_nonfinite_values() {
        assert!(ClassifierQueryRequest::new(1, 2, 3, &[0.0; 6]).is_ok());
        assert!(ClassifierQueryRequest::new(1, 2, 3, &[0.0; 5]).is_err());
        assert!(ClassifierQueryRequest::new(1, 2, 3, &[0.0, 0.0, 0.0, 0.0, 0.0, f32::NAN]).is_err());
    }
}

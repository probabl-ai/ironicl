use crate::{
    config::{CacheMode, StorageDType, TaskKind},
    error::{Error, Result},
};

pub const ARTIFACT_MAGIC: [u8; 8] = *b"IRONICL\0";
pub const ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Metadata that must be validated before a persisted static context is used.
/// Binary serialization is intentionally deferred until the tensor layout is
/// proven against the `PyTorch` oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactDescriptor {
    pub format_version: u32,
    pub task: TaskKind,
    pub cache_mode: CacheMode,
    pub weight_dtype: StorageDType,
    pub kv_dtype: StorageDType,
    pub training_rows: u64,
    pub features: u64,
    pub estimators: u64,
    pub model_sha256: [u8; 32],
    pub context_sha256: [u8; 32],
}

impl ArtifactDescriptor {
    /// Validate dimensions, hashes, and the artifact format version.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] when metadata cannot describe a valid
    /// artifact for this reader.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != ARTIFACT_FORMAT_VERSION {
            return Err(Error::InvalidConfig(format!(
                "artifact version {} is not supported; expected {}",
                self.format_version, ARTIFACT_FORMAT_VERSION
            )));
        }
        if self.training_rows == 0 || self.features == 0 || self.estimators == 0 {
            return Err(Error::InvalidConfig(
                "artifact dimensions must all be greater than zero".into(),
            ));
        }
        if self.model_sha256 == [0; 32] || self.context_sha256 == [0; 32] {
            return Err(Error::InvalidConfig("artifact hashes must be populated".into()));
        }
        Ok(())
    }
}

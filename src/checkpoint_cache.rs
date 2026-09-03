use std::path::{Path, PathBuf};

#[cfg(feature = "checkpoint")]
use std::env;

use crate::{
    CheckpointCacheOptions, ModelArtifactSummary, ModelProfile,
    error::{Error, Result},
};

#[cfg(feature = "checkpoint")]
use crate::{
    checkpoint::{checkpoint_sha256, inspect_checkpoint},
    import_checkpoint, inspect_model_artifact,
};

/// Result of preparing a checkpoint for execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedCheckpoint {
    Direct {
        source_sha256: String,
        profile: ModelProfile,
    },
    Cached {
        path: PathBuf,
        created: bool,
        summary: ModelArtifactSummary,
    },
}

/// Validate a checkpoint and optionally prepare its content-addressed model cache.
///
/// # Errors
///
/// Returns an error when checkpoint validation, cache path resolution, import,
/// or cached artifact verification fails.
#[cfg(feature = "checkpoint")]
pub fn prepare_checkpoint(path: &Path, options: &CheckpointCacheOptions) -> Result<PreparedCheckpoint> {
    if options.no_cache {
        let checkpoint = inspect_checkpoint(path)?;
        return Ok(PreparedCheckpoint::Direct {
            source_sha256: checkpoint.source_sha256,
            profile: checkpoint.profile.into(),
        });
    }

    let source_sha256 = checkpoint_sha256(path)?;
    let cache_root = options.cache_dir.clone().map_or_else(default_cache_dir, Ok)?;
    let cache_path = cache_model_path(&cache_root, &source_sha256)?;
    if !options.rebuild_cache {
        match inspect_model_artifact(&cache_path) {
            Ok(summary) if summary.source_sha256 == source_sha256 => {
                return Ok(PreparedCheckpoint::Cached {
                    path: cache_path,
                    created: false,
                    summary,
                });
            }
            Ok(_) | Err(Error::Artifact(_)) => {}
            Err(error) => return Err(error),
        }
    }

    let summary = import_checkpoint(path, &cache_path)?;
    if summary.source_sha256 != source_sha256 {
        return Err(Error::Cache(
            "imported model source hash does not match the checkpoint".into(),
        ));
    }
    Ok(PreparedCheckpoint::Cached {
        path: cache_path,
        created: true,
        summary,
    })
}

/// Return an explicit error when checkpoint support is disabled.
///
/// # Errors
///
/// Always returns [`Error::NotImplemented`].
#[cfg(not(feature = "checkpoint"))]
pub fn prepare_checkpoint(_path: &Path, _options: &CheckpointCacheOptions) -> Result<PreparedCheckpoint> {
    Err(Error::NotImplemented(
        "checkpoint import/cache is disabled; rebuild with the 'checkpoint' feature".into(),
    ))
}

#[cfg(feature = "checkpoint")]
fn default_cache_dir() -> Result<PathBuf> {
    if let Some(path) = env::var_os("XDG_CACHE_HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path).join("ironicl"));
    }
    if let Some(path) = env::var_os("HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path).join(".cache/ironicl"));
    }
    Err(Error::Cache(
        "cannot resolve cache directory; set --cache-dir, XDG_CACHE_HOME, or HOME".into(),
    ))
}

#[cfg(any(feature = "checkpoint", test))]
fn cache_model_path(cache_root: &Path, source_sha256: &str) -> Result<PathBuf> {
    if source_sha256.len() != 64
        || !source_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::Cache("checkpoint SHA-256 is invalid".into()));
    }
    Ok(cache_root
        .join("models")
        .join(format!("v{}", crate::MODEL_ARTIFACT_VERSION))
        .join(format!("{source_sha256}.ironicl")))
}

#[cfg(feature = "checkpoint")]
impl From<crate::CheckpointProfile> for ModelProfile {
    fn from(profile: crate::CheckpointProfile) -> Self {
        match profile {
            crate::CheckpointProfile::TabIclV2Classifier => Self::TabIclV2Classifier,
            crate::CheckpointProfile::TabIclV2Regressor => Self::TabIclV2Regressor,
            crate::CheckpointProfile::Unknown => Self::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_content_addressed_cache_path() {
        let digest = "a".repeat(64);
        assert_eq!(
            cache_model_path(Path::new("cache"), &digest),
            Ok(PathBuf::from(format!("cache/models/v1/{digest}.ironicl")))
        );
    }

    #[test]
    fn rejects_invalid_cache_key() {
        assert!(cache_model_path(Path::new("cache"), "../model").is_err());
    }
}

use std::{
    env,
    path::{Path, PathBuf},
};

use crate::{
    ModelArtifact, ModelArtifactSummary,
    error::{Error, Result},
    format::{encode_sha256, model::write_model_artifact},
    inspect_model_artifact, load_model_artifact,
    models::checkpoint::{
        CheckpointKind, checkpoint_sha256, model_data, model_from_checkpoint, same_file_path,
    },
};

/// Controls where and whether [`prepare_checkpoint`] reuses an imported model.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CheckpointCacheOptions {
    pub rebuild_cache: bool,
    pub cache_dir: Option<PathBuf>,
}

/// A checkpoint's imported model in the content-addressed cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCheckpoint {
    pub path: PathBuf,
    pub created: bool,
    pub summary: ModelArtifactSummary,
}

/// Validate a checkpoint of `kind` and prepare its content-addressed model
/// cache.
///
/// # Errors
///
/// Returns an error when checkpoint validation, cache path resolution, import,
/// or cached artifact verification fails.
pub fn prepare_checkpoint(
    path: &Path,
    kind: CheckpointKind,
    options: &CheckpointCacheOptions,
) -> Result<PreparedCheckpoint> {
    let source_sha256 = checkpoint_sha256(path)?;
    let cache_root = options.cache_dir.clone().map_or_else(default_cache_dir, Ok)?;
    let cache_path = cache_model_path(&cache_root, &source_sha256)?;
    if !options.rebuild_cache {
        match inspect_model_artifact(&cache_path) {
            Ok(summary) if summary.source_sha256 == source_sha256 && kind.holds(summary.profile) => {
                return Ok(PreparedCheckpoint {
                    path: cache_path,
                    created: false,
                    summary,
                });
            }
            Ok(_) | Err(Error::Artifact(_)) => {}
            Err(error) => return Err(error),
        }
    }

    if same_file_path(path, &cache_path)? {
        return Err(Error::InvalidArgument(
            "checkpoint input and model output must be different files".into(),
        ));
    }
    let data = model_data(path, kind)?;
    if encode_sha256(&data.source_sha256) != source_sha256 {
        return Err(Error::Cache(
            "imported model source hash does not match the checkpoint".into(),
        ));
    }
    let summary = write_model_artifact(&cache_path, &data)?;
    Ok(PreparedCheckpoint {
        path: cache_path,
        created: true,
        summary,
    })
}

/// Load the executable model of a checkpoint of `kind` through the default
/// model cache. When
/// the cache cannot be written or read, as in a read-only home directory, the
/// checkpoint is converted in memory instead, on every load.
///
/// # Errors
///
/// Returns [`Error::Checkpoint`] when the checkpoint is invalid and
/// [`Error::Artifact`] when it is not an official model.
pub fn load_checkpoint_model(path: &Path, kind: CheckpointKind) -> Result<ModelArtifact> {
    match prepare_checkpoint(path, kind, &CheckpointCacheOptions::default()) {
        Ok(prepared) => load_model_artifact(&prepared.path),
        Err(Error::Checkpoint(message)) => Err(Error::Checkpoint(message)),
        Err(_) => model_from_checkpoint(path, kind),
    }
}

/// The folder of imported weights, `<checkpoint SHA-256>.irmodel` files, that
/// loading a checkpoint fills: `models/v<version>` in Ironicl's cache, under
/// `XDG_CACHE_HOME` or `~/.cache`.
///
/// # Errors
///
/// When neither `XDG_CACHE_HOME` nor `HOME` is set.
pub fn default_weights_dir() -> Result<PathBuf> {
    Ok(default_cache_dir()?
        .join("models")
        .join(format!("v{}", crate::format::model::MODEL_ARTIFACT_VERSION)))
}

/// Ironicl's cache folder: `$XDG_CACHE_HOME/ironicl`, else `.cache/ironicl`
/// in the home folder (`HOME`, else the account's, or `USERPROFILE` on Windows).
///
/// # Errors
///
/// When `XDG_CACHE_HOME` is unset and there is no home folder.
pub fn default_cache_dir() -> Result<PathBuf> {
    if let Some(path) = env::var_os("XDG_CACHE_HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path).join("ironicl"));
    }
    if let Some(path) = env::home_dir().filter(|path| !path.as_os_str().is_empty()) {
        return Ok(path.join(".cache").join("ironicl"));
    }
    Err(Error::Cache(
        "cannot resolve cache directory; set XDG_CACHE_HOME or HOME".into(),
    ))
}

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
        .join(format!("v{}", crate::format::model::MODEL_ARTIFACT_VERSION))
        .join(format!("{source_sha256}.irmodel")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_content_addressed_cache_path() {
        let digest = "a".repeat(64);
        assert_eq!(
            cache_model_path(Path::new("cache"), &digest),
            Ok(PathBuf::from(format!("cache/models/v1/{digest}.irmodel")))
        );
    }

    #[test]
    fn rejects_invalid_cache_key() {
        assert!(cache_model_path(Path::new("cache"), "../model").is_err());
    }
}

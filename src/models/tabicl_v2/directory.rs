//! A model directory: what the Python estimators' `export` writes, and what
//! `ironicl prepare` and `ironicl serve` read.
//!
//! - `ironicl.json`: the metadata, written last, so a directory without it is
//!   incomplete. `format` and `version` are checked; the rest describes the
//!   model, and the other files stay the source of truth.
//! - `preprocessing.irprep`: the fitted preprocessing with its training rows.
//! - `context.ircntxt`: the KV cache, optional. [`prepare`] builds it when it
//!   is missing or describes another context.
//!
//! The weights stay out of it, in a shared folder of `<checkpoint
//! SHA-256>.irmodel` files: by default the model cache that loading a
//! checkpoint fills ([`default_weights_dir`]).

use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    Error, Executor, FittedModel, Labels, Optimize, Result, StaticContextArtifact,
    format::{encode_sha256, model::load_model_artifact, preprocessor},
    memory::MemoryLimits,
};

pub use super::checkpoint_cache::default_weights_dir;

pub const METADATA: &str = "ironicl.json";
pub const PREPROCESSING: &str = "preprocessing.irprep";
pub const CONTEXT: &str = "context.ircntxt";
/// The metadata's `format` and `version` for this layout.
const FORMAT: &str = "ironicl-model";
const VERSION: u64 = 1;

fn artifact_error(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::Artifact(format!("{}: {error}", path.display()))
}

/// Write `bytes` to a new owner-only file beside `path`, then rename it into
/// place, so a reader never sees a partial file.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let mut file = tempfile::Builder::new()
        .prefix(".ironicl-")
        .suffix(".tmp")
        .tempfile_in(directory)
        .map_err(|error| artifact_error(directory, error))?;
    file.write_all(bytes)
        .and_then(|()| file.as_file().sync_all())
        .map_err(|error| artifact_error(path, error))?;
    file.persist(path)
        .map(drop)
        .map_err(|error| artifact_error(path, error))
}

fn remove(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(artifact_error(path, error)),
        _ => Ok(()),
    }
}

/// Write `model` to the directory `dir`, with its KV cache when `kv_cache` is
/// set. Each file replaces the old one whole, and the metadata goes last.
///
/// # Errors
///
/// When the model kept no training rows, or a file cannot be written.
pub fn write(dir: &Path, model: &FittedModel, kv_cache: bool) -> Result<()> {
    if !model.preprocessed.has_training() {
        return Err(Error::InvalidArgument(
            "the model kept no training rows, so it cannot be written as a model directory".into(),
        ));
    }
    std::fs::create_dir_all(dir).map_err(|error| artifact_error(dir, error))?;
    remove(&dir.join(METADATA))?;
    write_atomic(&dir.join(PREPROCESSING), &model.preprocessor_bytes(true)?)?;
    let context = dir.join(CONTEXT);
    if kv_cache {
        model.context.save_context(&context)?;
    } else {
        remove(&context)?;
    }
    write_atomic(&dir.join(METADATA), &metadata(model)?)
}

fn metadata(model: &FittedModel) -> Result<Vec<u8>> {
    let preprocessor = model.preprocessed.preprocessor();
    let classes = preprocessor.classes().map(|labels| match labels {
        Labels::Int(values) => serde_json::json!(values),
        Labels::Float(values) => serde_json::json!(values),
        Labels::Bool(values) => serde_json::json!(values),
        Labels::Text(values) => serde_json::json!(values),
    });
    let metadata = serde_json::json!({
        "format": FORMAT,
        "version": VERSION,
        "ironicl": env!("CARGO_PKG_VERSION"),
        "task": model.task().to_string(),
        "n_features": preprocessor.n_features_in(),
        "feature_names": preprocessor.feature_names,
        "classes": classes,
        "weights": encode_sha256(&model.context.context_metadata().model_sha256),
    });
    let mut bytes =
        serde_json::to_vec_pretty(&metadata).map_err(|error| Error::Artifact(error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// The metadata of the model directory `dir`, checked to be complete and in
/// this layout.
///
/// # Errors
///
/// When the metadata is missing, invalid, or for another layout.
pub fn read_metadata(dir: &Path) -> Result<serde_json::Value> {
    let path = dir.join(METADATA);
    let bytes = std::fs::read(&path).map_err(|error| {
        Error::Artifact(format!(
            "{} is not a complete Ironicl model directory ({METADATA}: {error})",
            dir.display()
        ))
    })?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| artifact_error(&path, error))?;
    if metadata["format"] != FORMAT || metadata["version"] != VERSION {
        return Err(Error::Artifact(format!(
            "{} is not an Ironicl model directory of version {VERSION}; export the model again",
            dir.display()
        )));
    }
    Ok(metadata)
}

/// The weights file in `weights` that the preprocessing `bytes` were fitted
/// with.
fn weights_path(bytes: &[u8], weights: &Path) -> Result<PathBuf> {
    let sha256 = encode_sha256(&preprocessor::checkpoint_sha256(bytes)?);
    Ok(weights.join(format!("{sha256}.irmodel")))
}

fn executor(path: &Path) -> Result<Executor> {
    if !path.is_file() {
        return Err(Error::Artifact(format!(
            "the weights {} are missing; import the model's checkpoint with \
             `ironicl import CHECKPOINT`, or pass --weights",
            path.display()
        )));
    }
    let model = load_model_artifact(path).map_err(|error| match error {
        Error::Artifact(message) => artifact_error(path, message),
        error => error,
    })?;
    Ok(Executor::new(model))
}

/// A model directory's model, ready to predict.
pub struct Prepared {
    pub model: FittedModel,
    pub optimize: Optimize,
    /// Whether the KV cache was built, rather than loaded from `context.ircntxt`.
    pub built: bool,
}

/// Load the model directory `dir` with the weights in `weights`, and its KV
/// cache at `optimize`, by default the level of the saved cache, else `safe`.
/// A saved cache that is not exactly the one this model builds at that level
/// is rebuilt on this machine's device and replaces `context.ircntxt`.
///
/// # Errors
///
/// When a file is missing or invalid, the weights are not the model's, or the
/// cache cannot be built or written.
pub fn prepare(
    dir: &Path,
    weights: &Path,
    optimize: Option<Optimize>,
    threads: Option<usize>,
    limits: MemoryLimits,
) -> Result<Prepared> {
    read_metadata(dir)?;
    let path = dir.join(PREPROCESSING);
    let bytes = std::fs::read(&path).map_err(|error| artifact_error(&path, error))?;
    let executor = executor(&weights_path(&bytes, weights)?)?;
    let (_, preprocessed) = executor
        .decode_preprocessed(&bytes)
        .map_err(|error| artifact_error(&path, error))?;
    let preprocessed = Arc::new(preprocessed);
    let path = dir.join(CONTEXT);
    // An unreadable or invalid cache is rebuilt like a mismatched one.
    let saved = StaticContextArtifact::read_mapped_prefault(&path).ok();
    let optimize = optimize
        .or(saved.as_ref().map(|context| context.metadata().optimize))
        .unwrap_or_default();
    if let Some(context) = saved
        && *context.metadata() == executor.fitted_context_metadata(&preprocessed, optimize)?
    {
        let context = executor.load_context_artifact(context, optimize, threads, limits)?;
        let model = FittedModel {
            preprocessed,
            context: Arc::new(context),
        };
        return Ok(Prepared {
            model,
            optimize,
            built: false,
        });
    }
    let model = executor.build_fitted(preprocessed, optimize, threads, limits, None)?;
    model.context.save_context(&path)?;
    Ok(Prepared {
        model,
        optimize,
        built: true,
    })
}

/// What `ironicl inspect` shows of a model directory.
pub struct Summary {
    pub metadata: serde_json::Value,
    /// The weights file the model needs, and whether it is there.
    pub weights: (PathBuf, bool),
    /// The saved KV cache's level and size, when there is a valid one.
    pub cache: Option<(Optimize, u64)>,
}

/// Describe the model directory `dir`, looking for its weights in `weights`.
///
/// # Errors
///
/// When the metadata or the preprocessing is missing or invalid.
pub fn inspect(dir: &Path, weights: &Path) -> Result<Summary> {
    let metadata = read_metadata(dir)?;
    let path = dir.join(PREPROCESSING);
    let bytes = std::fs::read(&path).map_err(|error| artifact_error(&path, error))?;
    let weights = weights_path(&bytes, weights)?;
    let found = weights.is_file();
    let path = dir.join(CONTEXT);
    let cache = StaticContextArtifact::read_mapped_prefault(&path)
        .ok()
        .map(|context| {
            (
                context.metadata().optimize,
                std::fs::metadata(&path).map_or(0, |file| file.len()),
            )
        });
    Ok(Summary {
        metadata,
        weights: (weights, found),
        cache,
    })
}

#[cfg(test)]
mod tests {
    use super::{METADATA, read_metadata};

    #[test]
    fn a_model_directory_needs_metadata_of_this_layout() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_metadata(dir.path()).unwrap_err().to_string();
        assert!(
            error.contains("not a complete Ironicl model directory"),
            "{error}"
        );
        for (metadata, ok) in [
            (
                r#"{"format": "ironicl-model", "version": 1, "task": "classifier"}"#,
                true,
            ),
            (r#"{"format": "ironicl-model", "version": 2}"#, false),
            (r#"{"format": "other", "version": 1}"#, false),
            ("not json", false),
        ] {
            std::fs::write(dir.path().join(METADATA), metadata).unwrap();
            assert_eq!(read_metadata(dir.path()).is_ok(), ok, "{metadata}");
        }
    }
}

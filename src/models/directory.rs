//! A model directory: what the Python estimators' `export` writes, and what
//! `ironicl prepare` and `ironicl serve` read.
//!
//! - `ironicl.json`: the metadata, written last, so a directory without it is
//!   incomplete. `format` and `version` are checked; the rest describes the
//!   model, and the other files stay the source of truth.
//! - `preprocessing.ir`: the fitted preprocessing with its training rows,
//!   or for Kumo-Tabular the training data, from which it fits its
//!   preprocessing again.
//! - `context.ircntxt`: the KV cache, optional. [`prepare`] builds it when it
//!   is missing or describes another context.
//!
//! The weights stay out of it, in a shared folder of `<checkpoint
//! SHA-256>.ir` files: by default the model cache that loading a
//! checkpoint fills ([`default_weights_dir`]).

use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Serialize};

use crate::{
    ClassifierAggregation, Error, Executor, Features, FittedModel, Labels, Optimize, RegressionOutput,
    Result, StaticContextArtifact, TaskKind,
    format::{encode_sha256, model::load_model_artifact, preprocessor, read_bounded},
    memory::MemoryLimits,
    models::{ModelFamily, kumo_tabular},
};

pub use super::checkpoint_cache::default_weights_dir;

pub const METADATA: &str = "ironicl.json";
pub const PREPROCESSING: &str = "preprocessing.ir";
pub const CONTEXT: &str = "context.ircntxt";
/// The metadata's `format` and `version` for this layout.
const FORMAT: &str = "ironicl-model";
const VERSION: u64 = 1;
const MAX_METADATA_BYTES: u64 = 4 * 1024 * 1024;
const MAX_PREPROCESSING_BYTES: u64 = 4 * 1024 * 1024 * 1024;

fn artifact_error(path: &Path, error: impl std::fmt::Display) -> Error {
    Error::Artifact(format!("{}: {error}", path.display()))
}

/// Write `bytes` to a new owner-only file beside `path`, then rename it into
/// place, so a reader never sees a partial file.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::format::write_atomic(path, |file| {
        file.write_all(bytes).map_err(|e| artifact_error(path, e))
    })
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
    write_files(
        dir,
        &model.preprocessor_bytes(true)?,
        kv_cache.then_some(|path: &Path| model.context.save_context(path)),
        &Metadata::of(model),
    )
}

/// Write the Kumo-Tabular `model` to the directory `dir`, with its contexts
/// when `kv_cache`, as [`write`] does.
///
/// # Errors
///
/// When a file cannot be written.
pub fn write_kumo_tabular(dir: &Path, model: &kumo_tabular::fitted::Fitted, kv_cache: bool) -> Result<()> {
    let training = model.training();
    let metadata = Metadata {
        format: FORMAT.to_owned(),
        version: VERSION,
        ironicl: env!("CARGO_PKG_VERSION").to_owned(),
        task: model.task(),
        n_features: training.categories.len(),
        feature_names: training.feature_names.clone(),
        classes: model.classes().cloned(),
        weights: encode_sha256(&model.checkpoint_sha256()),
        optimize: Some(model.optimize()),
    };
    let context = |path: &Path| write_atomic(path, &model.context_bytes()?);
    write_files(dir, &model.to_bytes()?, kv_cache.then_some(context), &metadata)
}

fn write_files(
    dir: &Path,
    preprocessing: &[u8],
    context: Option<impl FnOnce(&Path) -> Result<()>>,
    metadata: &Metadata,
) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|error| artifact_error(dir, error))?;
    remove(&dir.join(METADATA))?;
    write_atomic(&dir.join(PREPROCESSING), preprocessing)?;
    let path = dir.join(CONTEXT);
    match context {
        Some(save) => save(&path)?,
        None => remove(&path)?,
    }
    let mut bytes =
        serde_json::to_vec_pretty(metadata).map_err(|error| Error::Artifact(error.to_string()))?;
    bytes.push(b'\n');
    write_atomic(&dir.join(METADATA), &bytes)
}

/// `ironicl.json`: what the model is. It describes the model; the other files
/// stay the source of truth.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    /// `"ironicl-model"`.
    pub format: String,
    /// The layout's version.
    pub version: u64,
    /// The Ironicl version that wrote the directory.
    pub ironicl: String,
    pub task: TaskKind,
    pub n_features: usize,
    /// The training data's column names, when it had them.
    pub feature_names: Option<Vec<String>>,
    /// A classifier's classes, in the order of its probabilities.
    pub classes: Option<Labels>,
    /// The SHA-256 of the checkpoint the weights come from, which names their
    /// file in the weights folder.
    pub weights: String,
    /// Fallback when no cache records the level (older exports default to `safe`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimize: Option<Optimize>,
}

impl Metadata {
    fn of(model: &FittedModel) -> Self {
        let preprocessor = model.preprocessed.preprocessor();
        Self {
            format: FORMAT.to_owned(),
            version: VERSION,
            ironicl: env!("CARGO_PKG_VERSION").to_owned(),
            task: model.task(),
            n_features: preprocessor.n_features_in(),
            feature_names: preprocessor.feature_names.clone(),
            classes: preprocessor.classes().cloned(),
            weights: encode_sha256(&model.context.context_metadata().model_sha256),
            optimize: Some(model.context.context_metadata().optimize),
        }
    }
}

/// The metadata of the model directory `dir`, checked to be complete and in
/// this layout.
///
/// # Errors
///
/// When the metadata is missing, invalid, or for another layout.
pub fn read_metadata(dir: &Path) -> Result<Metadata> {
    let path = dir.join(METADATA);
    let bytes = read_bounded(&path, MAX_METADATA_BYTES).map_err(|error| {
        Error::Artifact(format!(
            "{} is not a complete Ironicl model directory ({METADATA}: {error})",
            dir.display()
        ))
    })?;
    match serde_json::from_slice::<Metadata>(&bytes) {
        Ok(metadata) if metadata.format == FORMAT && metadata.version == VERSION => Ok(metadata),
        Ok(_) => Err(Error::Artifact(format!(
            "{} is not an Ironicl model directory of version {VERSION}; export the model again",
            dir.display()
        ))),
        Err(error) => Err(artifact_error(&path, error)),
    }
}

/// Shared weights addressed by the source checkpoint, not the container bytes.
fn weights_path(sha: &[u8; 32], weights: &Path) -> PathBuf {
    weights.join(format!("{}.ir", encode_sha256(sha)))
}

fn executor(path: &Path) -> Result<Executor> {
    let model = load_weights(path)?;
    if let crate::ModelProfile::KumoTabular(_) = model.summary().profile {
        return Err(artifact_error(
            path,
            "the weights are Kumo-Tabular's, not TabICLv2's",
        ));
    }
    Ok(Executor::new(model))
}

fn load_weights(path: &Path) -> Result<crate::ModelArtifact> {
    if !path.is_file() {
        return Err(Error::Artifact(format!(
            "the weights {} are missing; import the model's checkpoint with \
             `ironicl import CHECKPOINT` (add --model kumo-tabular for Kumo-Tabular), or pass --weights",
            path.display()
        )));
    }
    load_model_artifact(path).map_err(|error| match error {
        Error::Artifact(message) => artifact_error(path, message),
        error => error,
    })
}

/// A model directory's fitted model.
pub enum Fitted {
    TabIclV2(FittedModel),
    KumoTabular(Box<kumo_tabular::fitted::Fitted>),
}

impl Fitted {
    #[must_use]
    pub fn task(&self) -> TaskKind {
        match self {
            Self::TabIclV2(model) => model.task(),
            Self::KumoTabular(model) => model.task(),
        }
    }

    #[must_use]
    pub fn device(&self) -> &'static str {
        match self {
            Self::TabIclV2(model) => model.device(),
            Self::KumoTabular(model) => model.device(),
        }
    }

    /// Each input feature's categories, `None` for a numeric one.
    #[must_use]
    pub fn categories(&self) -> &[Option<Labels>] {
        match self {
            Self::TabIclV2(model) => model.preprocessed().preprocessor().categories(),
            Self::KumoTabular(model) => &model.training().categories,
        }
    }

    #[must_use]
    pub fn classes(&self) -> Option<&Labels> {
        match self {
            Self::TabIclV2(model) => model.preprocessed().preprocessor().classes(),
            Self::KumoTabular(model) => model.classes(),
        }
    }

    #[must_use]
    pub fn feature_names(&self) -> Option<&[String]> {
        match self {
            Self::TabIclV2(model) => model.preprocessed().preprocessor().feature_names.as_deref(),
            Self::KumoTabular(model) => model.training().feature_names.as_deref(),
        }
    }

    /// Class probabilities `[rows, classes]` for the encoded `[rows, features]` rows `x`.
    ///
    /// # Errors
    ///
    /// As the model's prediction.
    pub fn predict_probabilities(&self, x: &[f64]) -> Result<Vec<f32>> {
        match self {
            Self::TabIclV2(model) => {
                let aggregation: ClassifierAggregation = model
                    .preprocessed()
                    .preprocessor()
                    .aggregation
                    .unwrap_or_default();
                let columns = self.categories().len();
                Ok(model
                    .predict_probabilities(Features::F64(x), columns, aggregation)?
                    .values()
                    .to_vec())
            }
            Self::KumoTabular(model) => model.predict_proba(x),
        }
    }

    /// A regressor's `outputs` for the encoded `[rows, features]` rows `x`.
    ///
    /// # Errors
    ///
    /// As the model's prediction, or for a classifier.
    pub fn predict_regression(
        &self,
        x: &[f64],
        outputs: &[RegressionOutput],
    ) -> Result<Vec<(Vec<f32>, usize)>> {
        match self {
            Self::TabIclV2(model) => {
                model.predict_regression(Features::F64(x), self.categories().len(), outputs)
            }
            Self::KumoTabular(model) => model.predict_regression(x, outputs),
        }
    }

    /// A regressor's outputs for a batch of queries whose row counts sum to
    /// `x`'s rows, summarizing each query only over its own rows.
    ///
    /// # Errors
    ///
    /// As the model's prediction, or for a classifier.
    #[cfg(feature = "server")]
    pub(crate) fn predict_regression_batch(
        &self,
        x: &[f64],
        queries: &[(usize, &[RegressionOutput])],
    ) -> Result<Vec<crate::models::quantiles::RegressionOutputs>> {
        match self {
            Self::TabIclV2(model) => {
                model.predict_regression_batch(Features::F64(x), self.categories().len(), queries)
            }
            Self::KumoTabular(model) => model.predict_regression_batch(x, queries),
        }
    }
}

/// A model directory's model, ready to predict.
pub struct Prepared {
    pub model: Fitted,
    pub optimize: Optimize,
    /// Whether the KV cache was built, rather than loaded from `context.ircntxt`.
    pub built: bool,
    /// Warnings from loading the model or rebuilding its cache.
    pub warning: Option<String>,
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
    let prepared = load(dir, weights, optimize, threads, limits)?;
    prepared.save_cache_if_built(dir)?;
    Ok(prepared)
}

impl Prepared {
    pub(crate) fn save_cache_if_built(&self, dir: &Path) -> Result<()> {
        if !self.built {
            return Ok(());
        }
        let path = dir.join(CONTEXT);
        match &self.model {
            Fitted::TabIclV2(model) => model.context.save_context(&path),
            Fitted::KumoTabular(model) => write_atomic(&path, &model.context_bytes()?),
        }
    }
}

/// Load a fitted model, rebuilding its cache in memory when necessary.
pub(crate) fn load(
    dir: &Path,
    weights: &Path,
    optimize: Option<Optimize>,
    threads: Option<usize>,
    limits: MemoryLimits,
) -> Result<Prepared> {
    let metadata = read_metadata(dir)?;
    let path = dir.join(PREPROCESSING);
    let bytes = read_bounded(&path, MAX_PREPROCESSING_BYTES).map_err(|error| artifact_error(&path, error))?;
    let identity = preprocessor::identity(&bytes)?;
    if identity.family == ModelFamily::KumoTabular
        && let Some(limit) = limits.host.or(limits.device)
    {
        return Err(Error::InvalidArgument(format!(
            "Kumo-Tabular does not support memory limits ({} is set)",
            limit.source
        )));
    }
    let path = dir.join(CONTEXT);
    // A missing, unreadable, or invalid cache is rebuilt like a mismatched
    // one; one that exists but cannot be read is reported.
    let mut warning = None;
    let saved = match StaticContextArtifact::read_mapped_prefault(&path) {
        Ok(context) => Some(context),
        Err(error) => {
            if path.exists() {
                warning = Some(format!(
                    "could not read the saved KV cache {}, so rebuilt it: {error}",
                    path.display()
                ));
            }
            None
        }
    };
    let optimize = optimize
        .or(saved.as_ref().map(|context| context.metadata().optimize))
        .or(metadata.optimize)
        .unwrap_or_default();
    let weights = weights_path(&identity.checkpoint_sha256, weights);
    match identity.family {
        ModelFamily::KumoTabular => {
            let model = Arc::new(kumo_tabular::cpu::Model::with_optimize(
                load_weights(&weights)?,
                optimize,
            )?);
            if let Some(cache) = saved.filter(|c| c.metadata().optimize == optimize) {
                match kumo_tabular::fitted::Fitted::from_artifact(
                    Arc::clone(&model),
                    &bytes,
                    Some(cache),
                    threads,
                ) {
                    Ok(fitted) => {
                        return Ok(Prepared {
                            model: Fitted::KumoTabular(Box::new(fitted)),
                            optimize,
                            built: false,
                            warning,
                        });
                    }
                    Err(error) => {
                        warning = Some(format!(
                            "could not load the saved KV cache {}, so rebuilt it: {error}",
                            path.display()
                        ));
                    }
                }
            }
            let fitted = kumo_tabular::fitted::Fitted::from_bytes(model, &bytes, None, threads)
                .map_err(|error| artifact_error(&dir.join(PREPROCESSING), error))?;
            Ok(Prepared {
                model: Fitted::KumoTabular(Box::new(fitted)),
                optimize,
                built: true,
                warning,
            })
        }
        ModelFamily::TabIclV2 => {
            let executor = executor(&weights)?;
            let (preprocessed, _) = executor
                .decode_preprocessed(&bytes)
                .map_err(|error| artifact_error(&dir.join(PREPROCESSING), error))?;
            let preprocessed = Arc::new(preprocessed);
            if let Some(context) = saved
                && *context.metadata() == executor.fitted_context_metadata(&preprocessed, optimize)?
            {
                let context = executor.load_context_artifact(context, optimize, threads, limits)?;
                let model = Fitted::TabIclV2(FittedModel {
                    preprocessed,
                    context: Arc::new(context),
                });
                return Ok(Prepared {
                    model,
                    optimize,
                    built: false,
                    warning,
                });
            }
            let model = executor.build_fitted(preprocessed, optimize, threads, limits, None)?;
            Ok(Prepared {
                model: Fitted::TabIclV2(model),
                optimize,
                built: true,
                warning,
            })
        }
    }
}

/// What `ironicl inspect` shows of a model directory.
pub struct Summary {
    pub metadata: Metadata,
    /// The weights file the model needs, and whether it is there.
    pub weights: (PathBuf, bool),
    /// The saved KV cache's level and size, or why it cannot be read, when
    /// there is one.
    pub cache: Option<Result<(Optimize, u64)>>,
}

/// Describe the model directory `dir`, looking for its weights in `weights`.
///
/// # Errors
///
/// When the metadata or the preprocessing is missing or invalid.
pub fn inspect(dir: &Path, weights: &Path) -> Result<Summary> {
    let metadata = read_metadata(dir)?;
    let path = dir.join(PREPROCESSING);
    let identity = preprocessor::file_identity(&path, MAX_PREPROCESSING_BYTES)
        .map_err(|error| artifact_error(&path, error))?;
    let weights = weights_path(&identity.checkpoint_sha256, weights);
    let found = weights.is_file();
    let path = dir.join(CONTEXT);
    let cache = path.exists().then(|| {
        StaticContextArtifact::read_mapped_prefault(&path).map(|context| {
            (
                context.metadata().optimize,
                std::fs::metadata(&path).map_or(0, |file| file.len()),
            )
        })
    });
    Ok(Summary {
        metadata,
        weights: (weights, found),
        cache,
    })
}

#[cfg(test)]
mod tests {
    use super::{METADATA, Metadata, read_metadata};
    use crate::{Labels, TaskKind};

    #[test]
    fn a_model_directory_needs_metadata_of_this_layout() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_metadata(dir.path()).unwrap_err().to_string();
        assert!(
            error.contains("not a complete Ironicl model directory"),
            "{error}"
        );
        let metadata = Metadata {
            format: "ironicl-model".into(),
            version: 1,
            ironicl: "0.1.0".into(),
            task: TaskKind::Classifier,
            n_features: 2,
            feature_names: Some(vec!["a".into(), "b".into()]),
            classes: Some(Labels::Text(vec!["no".into(), "yes".into()])),
            weights: "00".repeat(32),
            optimize: None,
        };
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains(r#""task":"classifier","n_features":2"#), "{json}");
        assert!(json.contains(r#""classes":["no","yes"]"#), "{json}");
        std::fs::write(dir.path().join(METADATA), &json).unwrap();
        assert_eq!(read_metadata(dir.path()).unwrap(), metadata);
        for (bad, message) in [
            (json.replace(r#""version":1"#, r#""version":2"#), "of version 1"),
            (json.replace("ironicl-model", "other"), "of version 1"),
            (
                json.replace(r#""task":"classifier""#, r#""task":"ranker""#),
                "unknown variant",
            ),
            ("not json".into(), "expected ident"),
        ] {
            std::fs::write(dir.path().join(METADATA), &bad).unwrap();
            let error = read_metadata(dir.path()).unwrap_err().to_string();
            assert!(error.contains(message), "{bad}: {error}");
        }
        let oversized = vec![b' '; usize::try_from(super::MAX_METADATA_BYTES).unwrap() + 1];
        std::fs::write(dir.path().join(METADATA), oversized).unwrap();
        let error = read_metadata(dir.path()).unwrap_err().to_string();
        assert!(error.contains("limit"), "{error}");
    }
}

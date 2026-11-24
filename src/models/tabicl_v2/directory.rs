//! A model directory: what the Python estimators' `export` writes, and what
//! `ironicl prepare` and `ironicl serve` read.
//!
//! - `ironicl.json`: the metadata, written last, so a directory without it is
//!   incomplete. `format` and `version` are checked; the rest describes the
//!   model, and the other files stay the source of truth.
//! - `preprocessing.irprep`: the fitted preprocessing with its training rows,
//!   or for Kumo-Tabular the training data, from which [`prepare`] fits again.
//! - `context.ircntxt`: the `TabICLv2` KV cache, optional. [`prepare`] builds
//!   it when it is missing or describes another context.
//!
//! The weights stay out of it, in a shared folder of `<checkpoint
//! SHA-256>.irmodel` files: by default the model cache that loading a
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
    format::{encode_sha256, model::load_model_artifact, preprocessor},
    memory::MemoryLimits,
    models::kumo_tabular,
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
    write_files(
        dir,
        &model.preprocessor_bytes(true)?,
        kv_cache.then_some(|path: &Path| model.context.save_context(path)),
        &Metadata::of(model),
    )
}

/// Write the Kumo-Tabular `model` to the directory `dir`, as [`write`] does.
///
/// # Errors
///
/// When a file cannot be written.
pub fn write_kumo_tabular(dir: &Path, model: &kumo_tabular::fitted::Fitted) -> Result<()> {
    let training = model.training();
    let metadata = Metadata {
        format: FORMAT.to_owned(),
        version: VERSION,
        ironicl: env!("CARGO_PKG_VERSION").to_owned(),
        task: TaskKind::Classifier,
        n_features: training.categories.len(),
        feature_names: training.feature_names.clone(),
        classes: Some(model.classes().clone()),
        weights: encode_sha256(&model.checkpoint_sha256()),
    };
    write_files(
        dir,
        &model.to_bytes()?,
        None::<fn(&Path) -> Result<()>>,
        &metadata,
    )
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
    let bytes = std::fs::read(&path).map_err(|error| {
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

/// The weights file in `weights` that the preprocessing `bytes` were fitted
/// with.
fn weights_path(bytes: &[u8], weights: &Path) -> Result<PathBuf> {
    let sha256 = encode_sha256(&preprocessor::checkpoint_sha256(bytes)?);
    Ok(weights.join(format!("{sha256}.irmodel")))
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
             `ironicl import CHECKPOINT`, or pass --weights",
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
            Self::KumoTabular(_) => TaskKind::Classifier,
        }
    }

    #[must_use]
    pub fn device(&self) -> &'static str {
        match self {
            Self::TabIclV2(model) => model.device(),
            Self::KumoTabular(_) => "cpu",
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
            Self::KumoTabular(model) => Some(model.classes()),
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
            Self::KumoTabular(_) => Err(Error::InvalidArgument(
                "a classifier has no regression outputs".into(),
            )),
        }
    }
}

/// A model directory's model, ready to predict.
pub struct Prepared {
    pub model: Fitted,
    pub optimize: Optimize,
    /// Whether the KV cache was built, rather than loaded from `context.ircntxt`.
    pub built: bool,
    /// A warning that a saved KV cache exists but could not be read, so was
    /// rebuilt.
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
    read_metadata(dir)?;
    let path = dir.join(PREPROCESSING);
    let bytes = std::fs::read(&path).map_err(|error| artifact_error(&path, error))?;
    if preprocessor::architecture(&bytes)? == preprocessor::KUMO_TABULAR {
        let model = Arc::new(kumo_tabular::cpu::Model::new(load_weights(&weights_path(
            &bytes, weights,
        )?)?)?);
        let model = kumo_tabular::fitted::Fitted::from_bytes(model, &bytes, threads)
            .map_err(|error| artifact_error(&path, error))?;
        return Ok(Prepared {
            model: Fitted::KumoTabular(Box::new(model)),
            optimize: Optimize::default(),
            built: true,
            warning: None,
        });
    }
    let executor = executor(&weights_path(&bytes, weights)?)?;
    let (_, preprocessed) = executor
        .decode_preprocessed(&bytes)
        .map_err(|error| artifact_error(&path, error))?;
    let preprocessed = Arc::new(preprocessed);
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
        .unwrap_or_default();
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
    model.context.save_context(&path)?;
    Ok(Prepared {
        model: Fitted::TabIclV2(model),
        optimize,
        built: true,
        warning,
    })
}

/// What `ironicl inspect` shows of a model directory.
pub struct Summary {
    pub metadata: Metadata,
    /// The weights file the model needs, and whether it is there.
    pub weights: (PathBuf, bool),
    /// The saved KV cache's level and size, or why it cannot be read, when
    /// there is one.
    pub cache: Option<Result<(Optimize, u64)>>,
    /// Whether the model is fitted again when loaded, keeping no KV cache, as
    /// Kumo-Tabular is.
    pub refits: bool,
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
    let refits = preprocessor::architecture(&bytes)? == preprocessor::KUMO_TABULAR;
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
        refits,
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
    }
}

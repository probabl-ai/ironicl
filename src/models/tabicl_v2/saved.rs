//! An exported model: the directory the Python estimators' `export` writes,
//! which `ironicl prepare` and `ironicl serve` read. `export` writes the
//! manifest last, so a directory without one is incomplete.

use std::path::{Path, PathBuf};

use super::preprocess::{Features, Targets, invalid};
use crate::{
    Error, Executor, FittedModel, Optimize, Result, StaticContextArtifact,
    format::{
        model::load_model_artifact,
        npy::{self, Npy},
        preprocessor,
    },
    memory::MemoryLimits,
};

pub const MANIFEST: &str = "ironicl.json";
pub const MODEL: &str = "model.irmodel";
pub const PREPROCESSOR: &str = "preprocessing.irprep";
pub const CONTEXT: &str = "context.ircntxt";
const TRAIN_X: &str = "train_x.npy";
const TRAIN_Y: &str = "train_y.npy";
/// The manifest's `format` and `version` for this layout.
const FORMAT: &str = "ironicl-export";
const VERSION: u64 = 1;

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| Error::Artifact(format!("{}: {error}", path.display())))
}

/// Check that `dir` holds a complete export in this layout. The manifest's
/// other fields describe the model for people and tools; the files are the
/// source of truth.
fn check_manifest(dir: &Path) -> Result<()> {
    let path = dir.join(MANIFEST);
    let bytes = std::fs::read(&path).map_err(|error| {
        Error::Artifact(format!(
            "{} is not a complete Ironicl export ({MANIFEST}: {error}); export the model again",
            dir.display()
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| Error::Artifact(format!("{}: {error}", path.display())))?;
    if manifest["format"] != FORMAT || manifest["version"] != VERSION {
        return Err(Error::Artifact(format!(
            "{} is not an Ironicl export of version {VERSION}; export the model again",
            dir.display()
        )));
    }
    Ok(())
}

fn executor(dir: &Path) -> Result<Executor> {
    let path = dir.join(MODEL);
    let model = load_model_artifact(&path).map_err(|error| match error {
        Error::Artifact(message) => Error::Artifact(format!("{}: {message}", path.display())),
        error => error,
    })?;
    Ok(Executor::new(model))
}

/// Build the KV cache of the model saved in `dir` at `optimize`, on this
/// machine's device, and write it to `dir`'s `context.ircntxt`.
///
/// # Errors
///
/// When a file is missing or invalid, or as [`Executor::load_fitted`].
pub fn prepare(dir: &Path, optimize: Optimize) -> Result<(FittedModel, PathBuf)> {
    check_manifest(dir)?;
    let executor = executor(dir)?;
    let preprocessor = read(&dir.join(PREPROCESSOR))?;
    let (_, decoded) = preprocessor::decode(&preprocessor)?;
    let (x, shape) = npy::read(&dir.join(TRAIN_X))?;
    let (y, y_shape) = npy::read(&dir.join(TRAIN_Y))?;
    let &[rows, columns] = shape.as_slice() else {
        return Err(invalid("train_x.npy must be a matrix"));
    };
    if y_shape != [rows] {
        return Err(invalid("train_y.npy must hold one target per row of train_x.npy"));
    }
    let targets = match (y, decoded.classes()) {
        (Npy::I64(codes), Some(classes)) => {
            let indices = codes
                .iter()
                .map(|&code| usize::try_from(code).ok().filter(|&index| index < classes.len()))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("train_y.npy holds a class index out of range"))?;
            Targets::Labels(classes.select(&indices))
        }
        (Npy::F32(values), None) => Targets::Values(values),
        _ => return Err(invalid("train_y.npy does not match the model's task")),
    };
    let x = match &x {
        Npy::F32(values) => Features::F32(values),
        Npy::F64(values) => Features::F64(values),
        Npy::I64(_) => return Err(invalid("train_x.npy must hold floats")),
    };
    let limits = MemoryLimits::resolve(None, None)?;
    let fitted = executor.load_fitted(
        &preprocessor,
        None,
        Some((x, columns, &targets)),
        optimize,
        None,
        limits,
        None,
    )?;
    let path = dir.join(CONTEXT);
    fitted.context.save_context(&path)?;
    Ok((fitted, path))
}

/// Load the model saved in `dir` with its KV cache, at the level the cache
/// was built at.
///
/// # Errors
///
/// When a file is missing or invalid, or the files do not belong together.
pub fn load(dir: &Path, threads: Option<usize>, limits: MemoryLimits) -> Result<FittedModel> {
    check_manifest(dir)?;
    let executor = executor(dir)?;
    let path = dir.join(CONTEXT);
    let context = StaticContextArtifact::read_mapped_prefault(&path)
        .map_err(|error| Error::Artifact(format!("{}: {error}; run ironicl prepare", path.display())))?;
    let optimize = context.metadata().optimize;
    let context = executor.load_context_artifact(context, optimize, threads, limits)?;
    executor.fitted_with_context(&read(&dir.join(PREPROCESSOR))?, context)
}

#[cfg(test)]
mod tests {
    use super::{MANIFEST, check_manifest};

    #[test]
    fn an_export_needs_a_manifest_of_this_layout() {
        let dir = tempfile::tempdir().unwrap();
        let error = check_manifest(dir.path()).unwrap_err().to_string();
        assert!(error.contains(MANIFEST), "{error}");
        for (manifest, ok) in [
            (
                r#"{"format": "ironicl-export", "version": 1, "task": "classifier"}"#,
                true,
            ),
            (r#"{"format": "ironicl-export", "version": 2}"#, false),
            (r#"{"format": "other", "version": 1}"#, false),
            ("not json", false),
        ] {
            std::fs::write(dir.path().join(MANIFEST), manifest).unwrap();
            assert_eq!(check_manifest(dir.path()).is_ok(), ok, "{manifest}");
        }
    }
}

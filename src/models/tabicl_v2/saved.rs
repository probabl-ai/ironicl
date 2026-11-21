//! An exported model: the directory the Python estimators' `export` writes,
//! which `ironicl prepare` and `ironicl serve` read.

use std::path::{Path, PathBuf};

use super::preprocess::{Features, Targets, invalid};
use crate::{
    Error, Executor, FittedModel, Optimize, Result,
    format::{
        model::load_model_artifact,
        npy::{self, Npy},
        preprocessor,
    },
    memory::MemoryLimits,
};

pub const MODEL: &str = "model.irmodel";
pub const PREPROCESSOR: &str = "prep.irprep";
pub const CONTEXT: &str = "context.ircntxt";
const TRAIN_X: &str = "train_x.npy";
const TRAIN_Y: &str = "train_y.npy";

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| Error::Artifact(format!("{}: {error}", path.display())))
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

//! `ironicl._core`: the private Python extension module.
//!
//! Python owns validation, preprocessing, and aggregation. This module only
//! loads the model and runs the forward pass.

#![allow(clippy::needless_pass_by_value)] // PyO3 and `map_err` pass arguments by value.

use std::path::PathBuf;

use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

use crate::{
    CheckpointCacheOptions, ClassifierExecutor, ClassifierInputBundle, Error, PreparedCheckpoint,
    load_model_artifact, prepare_checkpoint,
};

fn py_error(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::InvalidArgument(_) | Error::InputBundle(_) => PyValueError::new_err(message),
        Error::NotImplemented(_) => PyNotImplementedError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// A loaded `TabICLv2` classifier.
#[pyclass(frozen)]
struct Model(ClassifierExecutor);

#[pymethods]
impl Model {
    /// Import a checkpoint through the content-addressed model cache and load it.
    #[new]
    fn new(py: Python<'_>, checkpoint: PathBuf) -> PyResult<Self> {
        py.detach(|| {
            let PreparedCheckpoint::Cached { path, .. } =
                prepare_checkpoint(&checkpoint, &CheckpointCacheOptions::default())?
            else {
                unreachable!("the default options always use the cache");
            };
            Ok(Self(ClassifierExecutor::new(load_model_artifact(&path)?)))
        })
        .map_err(py_error)
    }

    /// Run the uncached forward pass on FP32 `x[B, T, H]` and `y_train[B, N]`.
    ///
    /// Returns test-row logits as FP32 bytes and their `[B, T - N, C]` shape.
    fn predict_logits<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f32>,
        y_train: PyBuffer<f32>,
    ) -> PyResult<(Bound<'py, PyBytes>, [usize; 3])> {
        let (&[ensembles, rows, features], &[_, train_rows]) = (x.shape(), y_train.shape()) else {
            return Err(PyValueError::new_err(
                "x must have shape [B, T, H] and y_train [B, N]",
            ));
        };
        let bundle = ClassifierInputBundle::new(
            ensembles,
            rows,
            features,
            train_rows,
            x.to_vec(py)?,
            y_train.to_vec(py)?,
        )
        .map_err(py_error)?;
        let logits = py
            .detach(|| self.0.predict_logits(bundle.request()))
            .map_err(py_error)?;
        Ok((
            PyBytes::new(py, bytemuck::cast_slice(logits.values())),
            logits.shape(),
        ))
    }
}

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()
}

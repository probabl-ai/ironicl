//! `ironicl._core`: the private Python extension module.
//!
//! Python owns validation and preprocessing. This module loads the model,
//! builds the training context, and predicts against it.

#![allow(clippy::needless_pass_by_value)] // PyO3 and `map_err` pass arguments by value.

use std::path::PathBuf;

use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

use crate::{
    CheckpointCacheOptions, ClassifierAggregation, ClassifierExecutor, ClassifierQueryRequest, Error,
    Optimize, PreparedCheckpoint, StaticClassifierExecutor, load_model_artifact, prepare_checkpoint,
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

    /// Run FP32 training rows `x[B, N, H]` with labels `y_train[B, N]` once and
    /// keep their attention keys and values. Permutations are `[B, C]` and
    /// `[B, H]`; `optimize` is `"safe"`, `"auto"`, or `"fast"`.
    fn build_context(
        &self,
        py: Python<'_>,
        x: PyBuffer<f32>,
        y_train: PyBuffer<f32>,
        class_permutations: PyBuffer<u64>,
        feature_permutations: PyBuffer<u64>,
        optimize: &str,
    ) -> PyResult<Context> {
        let optimize = optimize.parse::<Optimize>().map_err(py_error)?;
        let &[ensembles, rows, features] = x.shape() else {
            return Err(PyValueError::new_err("x must have shape [B, N, H]"));
        };
        let (x, y_train) = (x.to_vec(py)?, y_train.to_vec(py)?);
        let (class_permutations, feature_permutations) =
            (class_permutations.to_vec(py)?, feature_permutations.to_vec(py)?);
        py.detach(|| {
            self.0.build_context(
                ensembles,
                rows,
                features,
                &x,
                &y_train,
                class_permutations,
                feature_permutations,
                optimize,
            )
        })
        .map(Context)
        .map_err(py_error)
    }
}

/// A fitted training context. Prediction runs only the query rows.
#[pyclass(frozen)]
struct Context(StaticClassifierExecutor);

#[pymethods]
impl Context {
    /// Predict FP32 query rows `x[B, Q, H]`. Returns probabilities as FP32
    /// bytes and their `[Q, C]` shape.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f32>,
        softmax_temperature: f32,
        average_logits: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, [usize; 2])> {
        let &[ensembles, rows, features] = x.shape() else {
            return Err(PyValueError::new_err("x must have shape [B, Q, H]"));
        };
        let x = x.to_vec(py)?;
        let aggregation = ClassifierAggregation {
            softmax_temperature,
            average_logits,
        };
        let probabilities = py
            .detach(|| {
                let request = ClassifierQueryRequest::new(ensembles, rows, features, &x)?;
                self.0.predict_probabilities(request, aggregation)
            })
            .map_err(py_error)?;
        Ok((
            PyBytes::new(py, bytemuck::cast_slice(probabilities.values())),
            probabilities.shape(),
        ))
    }
}

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<Context>()
}

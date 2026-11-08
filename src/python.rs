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
    types::{PyBytes, PyDict},
};

use crate::{
    CheckpointCacheOptions, ClassifierAggregation, Error, Executor, Optimize, PreparedCheckpoint,
    QueryRequest, RegressionOutput, StaticExecutor, TaskKind, load_model_artifact,
    memory::{Builder, MemoryLimits, Verdict},
    plan_context, prepare_checkpoint,
};

fn py_error(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::InvalidArgument(_) | Error::InputBundle(_) => PyValueError::new_err(message),
        Error::NotImplemented(_) => PyNotImplementedError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// A loaded `TabICLv2` classifier or regressor.
#[pyclass(frozen)]
struct Model(Executor);

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
            Ok(Self(Executor::new(load_model_artifact(&path)?)))
        })
        .map_err(py_error)
    }

    /// Run FP32 training rows `x[B, N, H]` with targets `y_train[B, N]` once and
    /// keep their attention keys and values. Targets are class labels for the
    /// classifier and standardized values for the regressor. Permutations are
    /// `[B, C]` (empty for the regressor) and `[B, H]`; `optimize` is `"safe"`,
    /// `"auto"`, or `"fast"`. The memory limits are sizes such as `"16G"`;
    /// `None` reads `IRONICL_MEMORY_LIMIT` and `IRONICL_DEVICE_MEMORY_LIMIT`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y_train, class_permutations, feature_permutations, optimize, threads,
        memory_limit=None, device_memory_limit=None,
    ))]
    fn build_context(
        &self,
        py: Python<'_>,
        x: PyBuffer<f32>,
        y_train: PyBuffer<f32>,
        class_permutations: PyBuffer<u64>,
        feature_permutations: PyBuffer<u64>,
        optimize: &str,
        threads: Option<usize>,
        memory_limit: Option<String>,
        device_memory_limit: Option<String>,
    ) -> PyResult<Context> {
        let optimize = optimize.parse::<Optimize>().map_err(py_error)?;
        let limits = MemoryLimits::resolve(memory_limit.as_deref(), device_memory_limit.as_deref())
            .map_err(py_error)?;
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
                threads,
                limits,
            )
        })
        .map(Context)
        .map_err(py_error)
    }

    /// `"classifier"` or `"regressor"`.
    #[getter]
    fn task(&self) -> String {
        self.0.task().to_string()
    }

    /// The source-checkpoint SHA-256 that identifies this model.
    #[getter]
    fn source_sha256(&self) -> String {
        self.0.source_sha256().to_owned()
    }

    /// Load a persisted static context for this model from `path`. `threads` and
    /// the memory limits mean what they mean for `build_context`.
    #[pyo3(signature = (path, threads, memory_limit=None, device_memory_limit=None))]
    fn load_context(
        &self,
        py: Python<'_>,
        path: PathBuf,
        threads: Option<usize>,
        memory_limit: Option<String>,
        device_memory_limit: Option<String>,
    ) -> PyResult<Context> {
        let limits = MemoryLimits::resolve(memory_limit.as_deref(), device_memory_limit.as_deref())
            .map_err(py_error)?;
        py.detach(|| self.0.load_context(&path, threads, limits))
            .map(Context)
            .map_err(py_error)
    }

    /// Load a context from the bytes `Context.to_bytes` returned, as
    /// `load_context` does from a file.
    #[pyo3(signature = (data, threads, memory_limit=None, device_memory_limit=None))]
    fn context_from_bytes(
        &self,
        py: Python<'_>,
        data: &[u8],
        threads: Option<usize>,
        memory_limit: Option<String>,
        device_memory_limit: Option<String>,
    ) -> PyResult<Context> {
        let limits = MemoryLimits::resolve(memory_limit.as_deref(), device_memory_limit.as_deref())
            .map_err(py_error)?;
        py.detach(|| self.0.load_context_bytes(data, threads, limits))
            .map(Context)
            .map_err(py_error)
    }
}

/// A fitted training context. Prediction runs only the query rows.
#[pyclass(frozen)]
struct Context(StaticExecutor);

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
                let request = QueryRequest::new(ensembles, rows, features, &x)?;
                self.0.predict_probabilities(request, aggregation)
            })
            .map_err(py_error)?;
        Ok((
            PyBytes::new(py, bytemuck::cast_slice(probabilities.values())),
            probabilities.shape(),
        ))
    }

    /// Predict FP32 query rows `x[B, Q, H]` against a regression context. Returns
    /// each `output_types` entry as FP32 bytes and its `[Q, columns]` shape, in
    /// standardized target units. `alphas` are the levels for `"quantiles"`.
    fn predict_regression<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f32>,
        output_types: Vec<String>,
        alphas: Vec<f32>,
    ) -> PyResult<Vec<(Bound<'py, PyBytes>, [usize; 2])>> {
        let &[ensembles, rows, features] = x.shape() else {
            return Err(PyValueError::new_err("x must have shape [B, Q, H]"));
        };
        let outputs = output_types
            .iter()
            .map(|name| RegressionOutput::parse(name, &alphas))
            .collect::<Result<Vec<_>, _>>()
            .map_err(py_error)?;
        let x = x.to_vec(py)?;
        let results = py
            .detach(|| {
                let request = QueryRequest::new(ensembles, rows, features, &x)?;
                self.0.predict_regression(request, &outputs)
            })
            .map_err(py_error)?;
        Ok(results
            .into_iter()
            .map(|(values, columns)| (PyBytes::new(py, bytemuck::cast_slice(&values)), [rows, columns]))
            .collect())
    }

    /// Write this fitted context to `path` atomically, for reuse through
    /// `Model.load_context`.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.0.save_context(&path)).map_err(py_error)
    }

    /// This context encoded as `save` writes it, for pickling.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.0.context_bytes()).map_err(py_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Plan the memory to build a `task` context of `estimators x rows x features`
/// with `optimize`, as `build_context` would on this machine. The memory limits
/// are sizes such as `"16G"`; `None` reads `IRONICL_MEMORY_LIMIT` and
/// `IRONICL_DEVICE_MEMORY_LIMIT`. Returns the plan as a dict.
#[pyfunction]
#[pyo3(signature = (task, estimators, rows, features, optimize, memory_limit=None, device_memory_limit=None))]
#[allow(clippy::too_many_arguments)]
fn plan_memory<'py>(
    py: Python<'py>,
    task: &str,
    estimators: usize,
    rows: usize,
    features: usize,
    optimize: &str,
    memory_limit: Option<String>,
    device_memory_limit: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let task = task.parse::<TaskKind>().map_err(py_error)?;
    let optimize = optimize.parse::<Optimize>().map_err(py_error)?;
    let limits =
        MemoryLimits::resolve(memory_limit.as_deref(), device_memory_limit.as_deref()).map_err(py_error)?;
    let plan = py
        .detach(|| plan_context(task, (estimators, rows, features), optimize, limits))
        .map_err(py_error)?;
    let dict = PyDict::new(py);
    dict.set_item(
        "verdict",
        match plan.verdict() {
            Verdict::Fits => "fits",
            Verdict::Overflow => "overflow",
            Verdict::CannotComplete => "cannot_complete",
            Verdict::Unknown => "unknown",
        },
    )?;
    dict.set_item("report", plan.to_string())?;
    dict.set_item("peak_bytes", plan.peak_bytes)?;
    dict.set_item("available_bytes", plan.available_bytes)?;
    dict.set_item("context_bytes", plan.context_bytes)?;
    dict.set_item("weight_bytes", plan.weight_bytes)?;
    dict.set_item("build_group", plan.build_group)?;
    dict.set_item(
        "builder",
        match plan.builder {
            Builder::Cpu => "cpu",
            Builder::Gpu => "gpu",
        },
    )?;
    dict.set_item("storage", plan.storage.name)?;
    dict.set_item("memory_limit", limits.host.map(|limit| limit.bytes))?;
    dict.set_item("device_memory_limit", limits.device.map(|limit| limit.bytes))?;
    Ok(dict)
}

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<Context>()?;
    module.add_function(wrap_pyfunction!(plan_memory, module)?)
}

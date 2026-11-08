//! `ironicl._core`: the private Python extension module.
//!
//! Python owns validation and preprocessing. This module loads the model,
//! builds the training context, and predicts against it.

#![allow(clippy::needless_pass_by_value)] // PyO3 and `map_err` pass arguments by value.

use std::{
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex, PoisonError},
};

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
    progress::{self, Hook, Stage},
};

fn py_error(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::InvalidArgument(_) | Error::InputBundle(_) => PyValueError::new_err(message),
        Error::NotImplemented(_) => PyNotImplementedError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// Run `work` without the GIL. Between steps of a build or prediction it
/// reports `(stage, done, total)` to `progress`, and on the main thread checks
/// for Ctrl-C, so the work stops with `KeyboardInterrupt` instead of running
/// to the end. An exception from `progress` stops it too.
fn run<T: Send>(
    py: Python<'_>,
    progress: Option<Py<PyAny>>,
    work: impl FnOnce() -> crate::Result<T> + Send,
) -> PyResult<T> {
    let threading = py.import("threading")?;
    let main = threading
        .call_method0("current_thread")?
        .is(&threading.call_method0("main_thread")?);
    if progress.is_none() && !main {
        return py.detach(work).map_err(py_error);
    }
    let raised = Arc::new(Mutex::new(None::<PyErr>));
    let sink = Arc::clone(&raised);
    let result = py.detach(move || {
        let hook: Hook = Rc::new(move |stage: Stage, done, total| {
            Python::attach(|py| {
                if main {
                    py.check_signals()?;
                }
                if let Some(progress) = &progress {
                    progress.call1(py, (stage.name(), done, total))?;
                }
                Ok(())
            })
            .map_err(|error: PyErr| {
                *sink.lock().unwrap_or_else(PoisonError::into_inner) = Some(error);
                Error::Interrupted
            })
        });
        progress::with_hook(hook, work)
    });
    result.map_err(|error| match error {
        Error::Interrupted => raised
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .unwrap_or_else(|| py_error(Error::Interrupted)),
        error => py_error(error),
    })
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
    /// `progress` receives `("build", estimators built, estimators)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y_train, class_permutations, feature_permutations, optimize, threads,
        memory_limit=None, device_memory_limit=None, progress=None,
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
        progress: Option<Py<PyAny>>,
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
        run(py, progress, || {
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
    /// bytes and their `[Q, C]` shape. `progress` receives `("predict", rows
    /// done, rows)` when the query runs in pieces.
    #[pyo3(signature = (x, softmax_temperature, average_logits, progress=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f32>,
        softmax_temperature: f32,
        average_logits: bool,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyBytes>, [usize; 2])> {
        let &[ensembles, rows, features] = x.shape() else {
            return Err(PyValueError::new_err("x must have shape [B, Q, H]"));
        };
        let x = x.to_vec(py)?;
        let aggregation = ClassifierAggregation {
            softmax_temperature,
            average_logits,
        };
        let probabilities = run(py, progress, || {
            let request = QueryRequest::new(ensembles, rows, features, &x)?;
            self.0.predict_probabilities(request, aggregation)
        })?;
        Ok((
            PyBytes::new(py, bytemuck::cast_slice(probabilities.values())),
            probabilities.shape(),
        ))
    }

    /// Predict FP32 query rows `x[B, Q, H]` against a regression context. Returns
    /// each `output_types` entry as FP32 bytes and its `[Q, columns]` shape, in
    /// standardized target units. `alphas` are the levels for `"quantiles"`.
    /// `progress` is as for `predict_proba`.
    #[pyo3(signature = (x, output_types, alphas, progress=None))]
    fn predict_regression<'py>(
        &self,
        py: Python<'py>,
        x: PyBuffer<f32>,
        output_types: Vec<String>,
        alphas: Vec<f32>,
        progress: Option<Py<PyAny>>,
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
        let results = run(py, progress, || {
            let request = QueryRequest::new(ensembles, rows, features, &x)?;
            self.0.predict_regression(request, &outputs)
        })?;
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

    /// This context encoded as `save` writes it, for pickling. Encodes straight
    /// into the returned bytes, without another copy.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let length = self.0.context_encoded_len().map_err(py_error)?;
        PyBytes::new_with(py, length, |mut buffer: &mut [u8]| {
            py.detach(|| self.0.encode_context(&mut buffer))
                .map_err(py_error)?;
            if buffer.is_empty() {
                Ok(())
            } else {
                Err(PyRuntimeError::new_err(
                    "the encoded context is shorter than its length",
                ))
            }
        })
    }
}

/// Plan the memory to build a `task` context of `estimators x rows x features`
/// with `optimize`, as `build_context` would on this machine. The memory limits
/// are sizes such as `"16G"`; `None` reads `IRONICL_MEMORY_LIMIT` and
/// `IRONICL_DEVICE_MEMORY_LIMIT`. With `load`, plans loading the built context
/// instead of building it. Returns the plan as a dict.
#[pyfunction]
#[pyo3(signature = (
    task, estimators, rows, features, optimize, memory_limit=None, device_memory_limit=None, load=false,
))]
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
    load: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let task = task.parse::<TaskKind>().map_err(py_error)?;
    let optimize = optimize.parse::<Optimize>().map_err(py_error)?;
    let limits =
        MemoryLimits::resolve(memory_limit.as_deref(), device_memory_limit.as_deref()).map_err(py_error)?;
    let plan = py
        .detach(|| plan_context(task, (estimators, rows, features), optimize, limits))
        .map_err(py_error)?;
    let plan = if load { plan.for_load() } else { plan };
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
    dict.set_item("estimators", estimators)?;
    dict.set_item("rows", rows)?;
    dict.set_item("features", features)?;
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

/// A size such as `"16G"` in bytes, as the memory limits read it.
#[pyfunction]
fn parse_size(text: &str) -> PyResult<u64> {
    crate::memory::parse_bytes(text).map_err(py_error)
}

/// What Ironicl detects on this machine, for `ironicl.show_config`.
#[pyfunction]
fn system_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
    let metal = cfg!(all(
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    ));
    dict.set_item("metal", metal)?;
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    {
        let gpu = crate::metal::system_gpu();
        dict.set_item("gpu", gpu.as_ref().map(|(name, _)| name.clone()))?;
        dict.set_item("gpu_working_set", gpu.map(|(_, bytes)| bytes))?;
        dict.set_item("gpu_min_build_rows", crate::metal::GPU_MIN_BUILD_ROWS)?;
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    for key in ["gpu", "gpu_working_set", "gpu_min_build_rows"] {
        dict.set_item(key, py.None())?;
    }
    let cpu = crate::platform::CpuCapabilities::detect();
    dict.set_item("architecture", cpu.architecture)?;
    dict.set_item("operating_system", cpu.operating_system)?;
    dict.set_item("bf16_storage", crate::kernels::native::bf16_storage_supported())?;
    dict.set_item("threads", crate::executor::executor_threads())?;
    let host = crate::memory::host_memory();
    dict.set_item("memory_total", host.map(|host| host.total))?;
    dict.set_item("memory_available", host.map(|host| host.available))?;
    dict.set_item("memory_hard_limit", host.and_then(|host| host.hard_limit))?;
    Ok(dict)
}

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<Context>()?;
    module.add_function(wrap_pyfunction!(parse_size, module)?)?;
    module.add_function(wrap_pyfunction!(system_info, module)?)?;
    module.add_function(wrap_pyfunction!(plan_memory, module)?)
}

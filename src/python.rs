//! `ironicl._core`: the private Python extension module.
//!
//! Python owns scikit-learn's interface, input conversion, and the checkpoint
//! download. This module preprocesses, builds the training context, and
//! predicts ([`FittedModel`]).

#![expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 and `map_err` pass arguments by value"
)]

use std::{
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex, PoisonError},
};

use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList, PyTuple},
};

use crate::{
    ClassifierAggregation, DEFAULT_ALPHAS, Error, Executor, Features, Labels, NormMethod, Optimize,
    PreprocessOptions, RegressionOutput, StaticExecutor, Targets, TaskKind, load_checkpoint_model,
    memory::{Builder, MemoryLimits, Verdict},
    models::kumo_tabular::{
        cpu,
        fitted::{Fitted, Training},
    },
    models::tabicl_v2::preprocess::flatten_orders,
    plan_context,
    preprocess::columns::{self, Column},
    progress::{self, Hook, Stage},
};

fn py_error(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::InvalidArgument(_) => PyValueError::new_err(message),
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
    /// Import a checkpoint through the content-addressed model cache, or in
    /// memory when the cache is not writable, and load it.
    #[new]
    fn new(py: Python<'_>, checkpoint: PathBuf) -> PyResult<Self> {
        py.detach(|| {
            let model = load_checkpoint_model(&checkpoint)?;
            if let crate::ModelProfile::KumoTabular(_) = model.summary().profile {
                return Err(Error::InvalidArgument(
                    "the checkpoint is Kumo-Tabular's; use KumoTabularClassifier".into(),
                ));
            }
            Ok(Self(Executor::new(model)))
        })
        .map_err(py_error)
    }

    /// Build the context of `preprocess`'s result. `optimize` is `"safe"`, `"auto"`,
    /// or `"fast"`. The memory limits are sizes such as `"16G"`; `None` reads
    /// `IRONICL_MEMORY_LIMIT` and `IRONICL_DEVICE_MEMORY_LIMIT`. `progress`
    /// receives `("build", estimators built, estimators)`. With `cache_dir`, a
    /// matching context there is loaded instead of built, and a built one is
    /// written there when possible.
    #[expect(clippy::too_many_arguments, reason = "mirrors the Python call's parameters")]
    #[pyo3(signature = (
        preprocessed, optimize, threads, memory_limit=None, device_memory_limit=None, progress=None, cache_dir=None,
    ))]
    fn build(
        &self,
        py: Python<'_>,
        preprocessed: &Preprocessed,
        optimize: &str,
        threads: Option<usize>,
        memory_limit: Option<String>,
        device_memory_limit: Option<String>,
        progress: Option<Py<PyAny>>,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<FittedModel> {
        let optimize = optimize.parse::<Optimize>().map_err(py_error)?;
        let limits = MemoryLimits::resolve(memory_limit.as_deref(), device_memory_limit.as_deref())
            .map_err(py_error)?;
        let preprocessed = Arc::clone(&preprocessed.0);
        run(py, progress, || {
            self.0
                .build_fitted(preprocessed, optimize, threads, limits, cache_dir.as_deref())
        })
        .map(FittedModel)
    }

    /// Load a fitted model from `FittedModel.preprocessor_bytes` (`IRPREP`) and
    /// `context`, the bytes of `FittedModel.context_bytes`; without a context,
    /// it is built from the training rows in the `IRPREP` bytes, as `build`
    /// builds it. The other arguments are as for `build`.
    #[expect(clippy::too_many_arguments, reason = "mirrors the Python call's parameters")]
    #[pyo3(signature = (
        preprocessor, optimize, threads, memory_limit=None, device_memory_limit=None, progress=None,
        cache_dir=None, *, context=None,
    ))]
    fn load_fitted(
        &self,
        py: Python<'_>,
        preprocessor: &[u8],
        optimize: &str,
        threads: Option<usize>,
        memory_limit: Option<String>,
        device_memory_limit: Option<String>,
        progress: Option<Py<PyAny>>,
        cache_dir: Option<PathBuf>,
        context: Option<&[u8]>,
    ) -> PyResult<FittedModel> {
        let optimize = optimize.parse::<Optimize>().map_err(py_error)?;
        let limits = MemoryLimits::resolve(memory_limit.as_deref(), device_memory_limit.as_deref())
            .map_err(py_error)?;
        run(py, progress, || {
            self.0.load_fitted(
                preprocessor,
                context,
                optimize,
                threads,
                limits,
                cache_dir.as_deref(),
            )
        })
        .map(FittedModel)
    }

    /// `"classifier"` or `"regressor"`.
    #[getter]
    fn task(&self) -> String {
        self.0.task().to_string()
    }
}

/// `context` encoded as the context cache writes it, straight into the
/// returned bytes, without another copy.
fn context_bytes<'py>(py: Python<'py>, context: &StaticExecutor) -> PyResult<Bound<'py, PyBytes>> {
    let length = context.context_encoded_len().map_err(py_error)?;
    PyBytes::new_with(py, length, |mut buffer: &mut [u8]| {
        py.detach(|| context.encode_context(&mut buffer))
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

/// Plan the memory to build a `task` context of `estimators x rows x features`
/// with `optimize`, as `Model.build` would on this machine. The memory limits
/// are sizes such as `"16G"`; `None` reads `IRONICL_MEMORY_LIMIT` and
/// `IRONICL_DEVICE_MEMORY_LIMIT`. With `load`, plans loading the built context
/// instead of building it. Returns the plan as a dict.
#[pyfunction]
#[pyo3(signature = (
    task, estimators, rows, features, optimize, memory_limit=None, device_memory_limit=None, load=false,
))]
#[expect(clippy::too_many_arguments, reason = "mirrors the Python call's parameters")]
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

/// Ironicl's cache folder: `$XDG_CACHE_HOME/ironicl`, else `~/.cache/ironicl`.
#[pyfunction]
fn cache_dir() -> PyResult<PathBuf> {
    crate::default_cache_dir().map_err(py_error)
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
        let gpu = crate::kernels::metal::system_gpu();
        dict.set_item("gpu", gpu.as_ref().map(|(name, _)| name.clone()))?;
        dict.set_item("gpu_working_set", gpu.map(|(_, bytes)| bytes))?;
        dict.set_item(
            "gpu_min_build_rows",
            crate::models::tabicl_v2::metal::GPU_MIN_BUILD_ROWS,
        )?;
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    for key in ["gpu", "gpu_working_set", "gpu_min_build_rows"] {
        dict.set_item(key, py.None())?;
    }
    let cpu = crate::kernels::cpu::CpuCapabilities::detect();
    dict.set_item("architecture", cpu.architecture)?;
    dict.set_item("operating_system", cpu.operating_system)?;
    dict.set_item("bf16_storage", crate::kernels::cpu::bf16_storage_supported())?;
    dict.set_item("threads", crate::models::tabicl_v2::cpu::executor_threads())?;
    let host = crate::memory::host_memory();
    dict.set_item("memory_total", host.map(|host| host.total))?;
    dict.set_item("memory_available", host.map(|host| host.available))?;
    dict.set_item("memory_hard_limit", host.and_then(|host| host.hard_limit))?;
    Ok(dict)
}

/// A C-ordered float32 or float64 matrix with `columns` columns.
enum Matrix {
    F32 { values: Vec<f32>, columns: usize },
    F64 { values: Vec<f64>, columns: usize },
}

impl Matrix {
    /// A copy of a two-dimensional float64 or float32 buffer.
    fn read(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Self> {
        let columns = |shape: &[usize]| match *shape {
            [_, columns] => Ok(columns),
            _ => Err(PyValueError::new_err("X must be two-dimensional")),
        };
        if let Ok(buffer) = PyBuffer::<f64>::get(x) {
            let columns = columns(buffer.shape())?;
            return Ok(Self::F64 {
                values: buffer.to_vec(py)?,
                columns,
            });
        }
        let buffer = PyBuffer::<f32>::get(x)?;
        let columns = columns(buffer.shape())?;
        Ok(Self::F32 {
            values: buffer.to_vec(py)?,
            columns,
        })
    }

    fn features(&self) -> Features<'_> {
        match self {
            Self::F32 { values, .. } => Features::F32(values),
            Self::F64 { values, .. } => Features::F64(values),
        }
    }

    fn columns(&self) -> usize {
        match self {
            Self::F32 { columns, .. } | Self::F64 { columns, .. } => *columns,
        }
    }

    /// The values as float64, exactly.
    fn into_f64(self) -> Vec<f64> {
        match self {
            Self::F32 { values, .. } => values.into_iter().map(f64::from).collect(),
            Self::F64 { values, .. } => values,
        }
    }
}

/// X as Python passes it, before it becomes a matrix: a matrix already, or a
/// list of columns, each a float64 array or a categorical column's `(kind,
/// categories, codes)`, with the categories as `read_labels` takes them and
/// int64 codes, -1 when missing.
enum RawFeatures {
    Matrix(Matrix),
    Columns(Vec<Column>),
}

impl RawFeatures {
    fn read(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Self> {
        let Ok(list) = x.cast::<PyList>() else {
            return Ok(Self::Matrix(Matrix::read(py, x)?));
        };
        let columns = list
            .iter()
            .map(|column| {
                if column.is_instance_of::<PyTuple>() {
                    let (kind, categories, codes) =
                        column.extract::<(String, Bound<'_, PyAny>, PyBuffer<i64>)>()?;
                    Ok(Column::Categorical {
                        categories: read_labels(py, (kind, categories))?,
                        codes: codes.to_vec(py)?,
                    })
                } else {
                    Ok(Column::Numeric(PyBuffer::<f64>::get(&column)?.to_vec(py)?))
                }
            })
            .collect::<PyResult<_>>()?;
        Ok(Self::Columns(columns))
    }

    /// The matrix to fit on, and its features' categories.
    fn fit(self) -> crate::Result<(Matrix, Vec<Option<Labels>>)> {
        match self {
            Self::Matrix(matrix) => {
                let columns = matrix.columns();
                Ok((matrix, vec![None; columns]))
            }
            Self::Columns(columns) => {
                let categories = columns::categories(&columns)?;
                let values = columns::encode(&columns, &categories)?;
                Ok((
                    Matrix::F64 {
                        values,
                        columns: columns.len(),
                    },
                    categories,
                ))
            }
        }
    }

    /// The matrix to predict from, with the fitted `categories`.
    fn encode(self, categories: &[Option<Labels>]) -> crate::Result<Matrix> {
        match self {
            Self::Matrix(matrix) => Ok(matrix),
            Self::Columns(columns) => Ok(Matrix::F64 {
                values: columns::encode(&columns, categories)?,
                columns: columns.len(),
            }),
        }
    }
}

/// Labels from Python as `(kind, values)`: `"int"`, `"float"`, or `"bool"` with
/// an int64, float64, or uint8 buffer, or `"text"` with a list of str.
fn read_labels(py: Python<'_>, (kind, values): (String, Bound<'_, PyAny>)) -> PyResult<Labels> {
    Ok(match kind.as_str() {
        "int" => Labels::Int(PyBuffer::<i64>::get(&values)?.to_vec(py)?),
        "float" => Labels::Float(PyBuffer::<f64>::get(&values)?.to_vec(py)?),
        "bool" => Labels::Bool(
            PyBuffer::<u8>::get(&values)?
                .to_vec(py)?
                .into_iter()
                .map(|value| value != 0)
                .collect(),
        ),
        "text" => Labels::Text(values.extract()?),
        _ => return Err(PyValueError::new_err(format!("unknown label kind {kind:?}"))),
    })
}

/// A classifier's `labels` or a regressor's float32 `values`, exactly one of them.
fn read_targets(
    py: Python<'_>,
    labels: Option<(String, Bound<'_, PyAny>)>,
    values: Option<PyBuffer<f32>>,
) -> PyResult<Targets> {
    match (labels, values) {
        (Some(labels), None) => Ok(Targets::Labels(read_labels(py, labels)?)),
        (None, Some(values)) => Ok(Targets::Values(values.to_vec(py)?)),
        _ => Err(PyValueError::new_err(
            "pass labels for a classifier or values for a regressor",
        )),
    }
}

/// Index rows of equal length as little-endian uint64 bytes and their shape.
fn index_rows<'py>(py: Python<'py>, rows: &[Vec<usize>]) -> (Bound<'py, PyBytes>, [usize; 2]) {
    let values = flatten_orders(rows);
    let width = rows.first().map_or(0, Vec::len);
    (
        PyBytes::new(py, bytemuck::cast_slice(&values)),
        [rows.len(), width],
    )
}

/// Little-endian bytes of `values`.
fn f32_bytes<'py>(py: Python<'py>, values: &[f32]) -> Bound<'py, PyBytes> {
    PyBytes::new(py, bytemuck::cast_slice(values))
}

/// Fit preprocessing on `x`, a float32 or float64 `[rows, columns]` array or
/// a list of columns as `RawFeatures` reads them, for
/// a classifier with `labels` (`(kind, values)`, as `read_labels` takes) or a
/// regressor with float32 `values`. The result's `shape` plans the memory to
/// build its context with `Model.build`.
#[pyfunction]
#[pyo3(signature = (
    x, *, labels=None, values=None, n_estimators, random_state, norm_methods, outlier_threshold,
    feature_names=None, softmax_temperature=0.9, average_logits=true,
))]
#[expect(clippy::too_many_arguments, reason = "mirrors the Python call's parameters")]
fn preprocess(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    labels: Option<(String, Bound<'_, PyAny>)>,
    values: Option<PyBuffer<f32>>,
    n_estimators: usize,
    random_state: i128,
    norm_methods: Vec<String>,
    outlier_threshold: f64,
    feature_names: Option<Vec<String>>,
    softmax_temperature: f32,
    average_logits: bool,
) -> PyResult<Preprocessed> {
    if !(softmax_temperature.is_finite() && softmax_temperature > 0.0) {
        return Err(PyValueError::new_err(
            "softmax_temperature must be positive and finite",
        ));
    }
    let x = RawFeatures::read(py, x)?;
    let targets = read_targets(py, labels, values)?;
    let options = PreprocessOptions {
        n_estimators,
        random_state,
        norm_methods: norm_methods
            .iter()
            .map(|name| NormMethod::from_name(name))
            .collect::<crate::Result<_>>()
            .map_err(py_error)?,
        outlier_threshold,
    };
    let mut preprocessed = py
        .detach(|| {
            let (x, categories) = x.fit()?;
            crate::Preprocessed::fit(&options, x.features(), x.columns(), categories, &targets)
        })
        .map_err(py_error)?;
    // Saved for serving; Python passes its own settings to every prediction.
    let preprocessor = &mut preprocessed.preprocessor;
    preprocessor.feature_names = feature_names;
    if let Some(aggregation) = &mut preprocessor.aggregation {
        *aggregation = ClassifierAggregation {
            softmax_temperature,
            average_logits,
        };
    }
    Ok(Preprocessed(Arc::new(preprocessed)))
}

/// Fitted preprocessing and its training rows, before or within a `FittedModel`.
#[pyclass(frozen)]
struct Preprocessed(Arc<crate::Preprocessed>);

#[pymethods]
impl Preprocessed {
    /// The context's `[B, N, H]` shape, or `None` without training rows.
    #[getter]
    fn shape(&self) -> Option<[usize; 3]> {
        self.0.shape()
    }

    #[getter]
    fn has_training(&self) -> bool {
        self.0.has_training()
    }

    #[getter]
    fn feature_mask(&self) -> Vec<bool> {
        self.0.preprocessor().feature_mask().to_vec()
    }

    /// Whether each input feature is categorical.
    #[getter]
    fn categorical(&self) -> Vec<bool> {
        self.0
            .preprocessor()
            .categories()
            .iter()
            .map(Option::is_some)
            .collect()
    }

    /// The normalization method of each group, in group order.
    #[getter]
    fn group_names(&self) -> Vec<&'static str> {
        self.0
            .preprocessor()
            .group_methods()
            .iter()
            .map(|method| method.name())
            .collect()
    }

    /// Each member's group.
    #[getter]
    fn member_groups(&self) -> Vec<usize> {
        self.0.preprocessor().member_groups().to_vec()
    }

    /// Each member's feature order, as uint64 bytes, and their `[B, H]` shape.
    #[getter]
    fn feature_permutations<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, [usize; 2]) {
        index_rows(py, self.0.preprocessor().feature_permutations())
    }

    /// Each member's class order, as uint64 bytes, and their `[B, classes]` shape.
    #[getter]
    fn class_permutations<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, [usize; 2]) {
        index_rows(py, self.0.preprocessor().class_permutations())
    }

    /// What the context build takes: FP32 `[B, N, H]` rows and `[B, N]` targets,
    /// and uint64 class and feature orders, each as bytes and a shape.
    #[expect(clippy::type_complexity, reason = "four arrays as bytes and shapes")]
    fn training_inputs<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        (Bound<'py, PyBytes>, [usize; 3]),
        (Bound<'py, PyBytes>, [usize; 2]),
        (Bound<'py, PyBytes>, [usize; 2]),
        (Bound<'py, PyBytes>, [usize; 2]),
    )> {
        let inputs = py.detach(|| self.0.training_inputs()).map_err(py_error)?;
        let [members, rows, _] = inputs.shape;
        let preprocessor = self.0.preprocessor();
        Ok((
            (f32_bytes(py, &inputs.rows), inputs.shape),
            (f32_bytes(py, &inputs.targets), [members, rows]),
            index_rows(py, preprocessor.class_permutations()),
            index_rows(py, preprocessor.feature_permutations()),
        ))
    }

    /// Each member's preprocessed rows of `x`, `[rows, columns]`, as FP32 bytes
    /// and their `[B, rows, H]` shape.
    fn query_rows<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyBytes>, [usize; 3])> {
        let x = RawFeatures::read(py, x)?;
        let (rows, shape) = py
            .detach(|| {
                let x = x.encode(self.0.preprocessor().categories())?;
                self.0.query_rows(x.features(), x.columns())
            })
            .map_err(py_error)?;
        Ok((f32_bytes(py, &rows), shape))
    }
}

/// A fitted model: preprocessing and its training context. Predictions take
/// raw features and return the caller's units.
#[pyclass(frozen)]
struct FittedModel(crate::FittedModel);

#[pymethods]
impl FittedModel {
    /// Class probabilities for `x`, a float32 or float64 `[rows, columns]`
    /// array, as FP32 bytes and their `[rows, classes]` shape. `progress`
    /// receives `("predict", rows done, rows)`.
    #[pyo3(signature = (x, softmax_temperature, average_logits, progress=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        softmax_temperature: f32,
        average_logits: bool,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyBytes>, [usize; 2])> {
        let x = RawFeatures::read(py, x)?;
        let aggregation = ClassifierAggregation {
            softmax_temperature,
            average_logits,
        };
        let probabilities = run(py, progress, || {
            let x = x.encode(self.0.preprocessed().preprocessor().categories())?;
            self.0
                .predict_probabilities(x.features(), x.columns(), aggregation)
        })?;
        Ok((f32_bytes(py, probabilities.values()), probabilities.shape()))
    }

    /// Each `output_types` entry for `x` in the target's units, as FP32 bytes
    /// and its `[rows, columns]` shape. `alphas` are the levels for
    /// `"quantiles"`, `TabICL`'s default when `None`; `progress` is as for
    /// `predict_proba`.
    #[pyo3(signature = (x, output_types, alphas=None, progress=None))]
    fn predict_regression<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        output_types: Vec<String>,
        alphas: Option<Vec<f32>>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Vec<(Bound<'py, PyBytes>, [usize; 2])>> {
        let x = RawFeatures::read(py, x)?;
        let outputs = output_types
            .iter()
            .map(|name| RegressionOutput::parse(name, alphas.as_deref().unwrap_or(&DEFAULT_ALPHAS)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(py_error)?;
        let (results, rows) = run(py, progress, || {
            let x = x.encode(self.0.preprocessed().preprocessor().categories())?;
            let rows = x.features().len() / x.columns().max(1);
            Ok((
                self.0.predict_regression(x.features(), x.columns(), &outputs)?,
                rows,
            ))
        })?;
        Ok(results
            .into_iter()
            .map(|(values, width)| (f32_bytes(py, &values), [rows, width]))
            .collect())
    }

    #[getter]
    fn preprocessed(&self) -> Preprocessed {
        Preprocessed(Arc::clone(self.0.preprocessed()))
    }

    /// Where predictions run: `"metal"`, `"cuda"` with the ICL cache on the
    /// GPU, `"cuda-streamed"` with it streamed from host memory on every pass,
    /// or `"cpu"`.
    #[getter]
    fn device(&self) -> &'static str {
        self.0.device()
    }

    /// Query rows a prediction call should bring at once: one pass's when the
    /// ICL cache streams, so each copy of it serves as many rows as fit,
    /// else `None`.
    #[getter]
    fn batch_rows(&self) -> Option<usize> {
        self.0.batch_rows()
    }

    /// The fitted preprocessing as `IRPREP` bytes, for `Model.load_fitted`, with
    /// the training rows when `training` is set and they were kept.
    #[pyo3(signature = (training=true))]
    fn preprocessor_bytes<'py>(&self, py: Python<'py>, training: bool) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.0.preprocessor_bytes(training).map_err(py_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Write the model directory `path` for `ironicl serve`, with the KV cache
    /// when `kv_cache` is set, and import `checkpoint`, the model's, into the
    /// shared weights folder when it is not there.
    fn export(&self, py: Python<'_>, path: PathBuf, checkpoint: PathBuf, kv_cache: bool) -> PyResult<()> {
        py.detach(|| {
            crate::prepare_checkpoint(&checkpoint, &crate::CheckpointCacheOptions::default())?;
            crate::models::tabicl_v2::directory::write(&path, &self.0, kv_cache)
        })
        .map_err(py_error)
    }

    /// The training context as `IRCNTXT` bytes, for `Model.load_fitted`.
    fn context_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        context_bytes(py, &self.0.context)
    }
}

/// The engine's phase timings since the start or the last `profile_reset`, as
/// `(category, calls, total seconds)` tuples. Empty unless the extension is
/// built with the `profile` feature and `IRONICL_PROFILE` is set before the
/// first fit or prediction.
#[pyfunction]
fn profile_report() -> Vec<(&'static str, u64, f64)> {
    crate::profile::report()
}

/// Zero the timings `profile_report` returns.
#[pyfunction]
fn profile_reset() {
    crate::profile::reset();
}

/// A loaded Kumo-Tabular checkpoint.
#[pyclass(frozen)]
struct KumoTabular(Arc<cpu::Model>);

#[pymethods]
impl KumoTabular {
    /// Import a checkpoint as `Model` does, and load it.
    #[new]
    fn new(py: Python<'_>, checkpoint: PathBuf) -> PyResult<Self> {
        py.detach(|| load_checkpoint_model(&checkpoint).and_then(cpu::Model::new))
            .map(|model| Self(Arc::new(model)))
            .map_err(py_error)
    }

    #[getter]
    fn task(&self) -> String {
        self.0.release.task.to_string()
    }

    /// Fit on `x`, as `preprocess` reads it, and the class `labels`
    /// (`(kind, values)`). `threads` sets the worker threads, `None` all.
    /// `progress` receives `("build", members built, members)`.
    #[expect(clippy::too_many_arguments, reason = "mirrors the Python call's parameters")]
    #[pyo3(signature = (x, labels, n_estimators, random_state, threads=None, feature_names=None, progress=None))]
    fn fit(
        &self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        labels: (String, Bound<'_, PyAny>),
        n_estimators: usize,
        random_state: i128,
        threads: Option<usize>,
        feature_names: Option<Vec<String>>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<KumoFitted> {
        let raw = RawFeatures::read(py, &x)?;
        let labels = read_labels(py, labels)?;
        let model = Arc::clone(&self.0);
        run(py, progress, move || {
            let (matrix, categories) = raw.fit()?;
            let training = Training {
                x: matrix.into_f64(),
                categories,
                labels,
                members: n_estimators,
                seed: random_state,
                feature_names,
            };
            Fitted::fit(model, training, threads)
        })
        .map(KumoFitted)
    }

    /// Load what `KumoFitted.to_bytes` saved, fitting again, as `fit` does.
    #[pyo3(signature = (data, threads=None, progress=None))]
    fn load(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        threads: Option<usize>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<KumoFitted> {
        let model = Arc::clone(&self.0);
        run(py, progress, move || Fitted::from_bytes(model, &data, threads)).map(KumoFitted)
    }
}

/// A fitted Kumo-Tabular classifier.
#[pyclass(frozen)]
struct KumoFitted(Fitted);

#[pymethods]
impl KumoFitted {
    /// Which input features are categorical.
    #[getter]
    fn categorical(&self) -> Vec<bool> {
        self.0.training().categories.iter().map(Option::is_some).collect()
    }

    /// `(probabilities as float32 bytes, [rows, classes])` for `x`, as `fit` reads it.
    #[pyo3(signature = (x, progress=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'_, PyAny>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyBytes>, [usize; 2])> {
        let raw = RawFeatures::read(py, &x)?;
        let categories = &self.0.training().categories;
        let classes = self.0.classes().len();
        let values = run(py, progress, || {
            let matrix = raw.encode(categories)?;
            if matrix.columns() != categories.len() {
                return Err(Error::InvalidArgument(format!(
                    "X has {} features; expected {}",
                    matrix.columns(),
                    categories.len()
                )));
            }
            self.0.predict_proba(&matrix.into_f64())
        })?;
        Ok((f32_bytes(py, &values), [values.len() / classes, classes]))
    }

    /// Write the model directory `path` for `ironicl serve`, and import
    /// `checkpoint`, the model's, into the shared weights folder when it is
    /// not there.
    fn export(&self, py: Python<'_>, path: PathBuf, checkpoint: PathBuf) -> PyResult<()> {
        py.detach(|| {
            crate::prepare_checkpoint(&checkpoint, &crate::CheckpointCacheOptions::default())?;
            crate::models::tabicl_v2::directory::write_kumo_tabular(&path, &self.0)
        })
        .map_err(py_error)
    }

    /// The `IRPREP` bytes `KumoTabular.load` loads.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.0.to_bytes().map_err(py_error)?))
    }
}

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<Preprocessed>()?;
    module.add_class::<FittedModel>()?;
    module.add_class::<KumoTabular>()?;
    module.add_class::<KumoFitted>()?;
    module.add_function(wrap_pyfunction!(preprocess, module)?)?;
    module.add_function(wrap_pyfunction!(parse_size, module)?)?;
    module.add_function(wrap_pyfunction!(cache_dir, module)?)?;
    module.add_function(wrap_pyfunction!(system_info, module)?)?;
    module.add_function(wrap_pyfunction!(profile_report, module)?)?;
    module.add_function(wrap_pyfunction!(profile_reset, module)?)?;
    module.add_function(wrap_pyfunction!(plan_memory, module)?)
}

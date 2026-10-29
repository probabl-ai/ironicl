//! Python extension module `ironicl._ironicl`, compiled with the `python` feature.
//!
//! The module exposes the resident cached classifier and checkpoint import to
//! the `ironicl` Python package. Inference and artifact loading run with the
//! GIL released so scikit-learn and joblib threads are not serialised.

#![allow(
    clippy::needless_pass_by_value,
    clippy::must_use_candidate,
    clippy::unused_self
)]

use std::path::{Path, PathBuf};

use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::{
    create_exception,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use crate::{
    CheckpointCacheOptions, ClassifierExecutor, ClassifierQueryRequest, Error, KvDtype, PreparedCheckpoint,
    StaticClassifierExecutor, StaticContextArtifact, StaticContextOptions, StaticContextRequest,
    build_static_context_file, encode_sha256, inspect_model_artifact, load_model_artifact,
    prepare_checkpoint,
};

create_exception!(
    _ironicl,
    IronICLError,
    PyRuntimeError,
    "The IronICL engine failed to load an artifact or run a query."
);

fn to_py_error(error: Error) -> PyErr {
    match error {
        Error::InvalidArgument(message) => PyValueError::new_err(message),
        other => IronICLError::new_err(other.to_string()),
    }
}

#[derive(Clone, Copy)]
struct Dimensions {
    ensemble_count: usize,
    training_rows: usize,
    feature_count: usize,
    class_count: usize,
}

/// A resident cached classifier: one model artifact plus one static context.
#[pyclass(name = "StaticExecutor", module = "ironicl._ironicl", frozen)]
pub struct PyStaticExecutor {
    executor: StaticClassifierExecutor,
    dimensions: Dimensions,
}

impl PyStaticExecutor {
    fn build(model_path: &Path, context: StaticContextArtifact) -> PyResult<Self> {
        let model = load_model_artifact(model_path).map_err(to_py_error)?;
        let metadata = context.metadata();
        let dimensions = Dimensions {
            ensemble_count: metadata.ensemble_count,
            training_rows: metadata.training_rows,
            feature_count: metadata.feature_count,
            class_count: metadata.class_count,
        };
        let executor = StaticClassifierExecutor::new(model, context).map_err(to_py_error)?;
        Ok(Self { executor, dimensions })
    }

    /// Run `request_count` requests of shape `[b, q, h]` stored back to back in
    /// `values`; return the flat `[request_count, q, classes]` probabilities.
    fn run(
        &self,
        request_count: usize,
        b: usize,
        q: usize,
        h: usize,
        values: &[f32],
    ) -> PyResult<(Vec<f32>, usize)> {
        let request_length = b
            .checked_mul(q)
            .and_then(|length| length.checked_mul(h))
            .filter(|length| *length > 0)
            .ok_or_else(|| PyValueError::new_err("variants must have non-zero [B, Q, H] dimensions"))?;
        if request_length * request_count != values.len() {
            return Err(PyValueError::new_err("variants length does not match its shape"));
        }
        let outputs = if request_count == 1 {
            let request = ClassifierQueryRequest::new(b, q, h, values).map_err(to_py_error)?;
            vec![
                self.executor
                    .predict_probabilities(request)
                    .map_err(to_py_error)?,
            ]
        } else {
            let requests = values
                .chunks_exact(request_length)
                .map(|chunk| ClassifierQueryRequest::new(b, q, h, chunk))
                .collect::<crate::Result<Vec<_>>>()
                .map_err(to_py_error)?;
            self.executor
                .predict_probabilities_batch(&requests)
                .map_err(to_py_error)?
        };
        let classes = outputs[0].shape()[1];
        let mut flat = Vec::with_capacity(request_count * q * classes);
        for output in &outputs {
            flat.extend_from_slice(output.values());
        }
        Ok((flat, classes))
    }
}

fn contiguous<'a, E>(values: Result<&'a [f32], E>, expected: &str) -> PyResult<&'a [f32]> {
    values.map_err(|_| {
        PyValueError::new_err(format!(
            "variants must be a C-contiguous float32 array of shape {expected}"
        ))
    })
}

fn contiguous_array<'a, T, E>(
    values: Result<&'a [T], E>,
    name: &str,
    dtype: &str,
    expected: &str,
) -> PyResult<&'a [T]> {
    values.map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a C-contiguous {dtype} array of shape {expected}"
        ))
    })
}

#[pymethods]
impl PyStaticExecutor {
    #[new]
    fn new(py: Python<'_>, model_path: PathBuf, context_path: PathBuf) -> PyResult<Self> {
        py.detach(|| {
            let context = StaticContextArtifact::read(&context_path).map_err(to_py_error)?;
            Self::build(&model_path, context)
        })
    }

    /// Load the model from disk and the static context from in-memory bytes.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, model_path: PathBuf, context: &[u8]) -> PyResult<Self> {
        py.detach(|| {
            let context = StaticContextArtifact::from_bytes(context).map_err(to_py_error)?;
            Self::build(&model_path, context)
        })
    }

    #[getter]
    fn ensemble_count(&self) -> usize {
        self.dimensions.ensemble_count
    }

    #[getter]
    fn training_rows(&self) -> usize {
        self.dimensions.training_rows
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.dimensions.feature_count
    }

    #[getter]
    fn class_count(&self) -> usize {
        self.dimensions.class_count
    }

    /// Run one `[B, Q, H]` request and return `[Q, C]` probabilities.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        variants: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = variants.shape();
        let (b, q, h) = (shape[0], shape[1], shape[2]);
        let values = contiguous(variants.as_slice(), "[B, Q, H]")?;
        let (flat, classes) = py.detach(|| self.run(1, b, q, h, values))?;
        PyArray1::from_vec(py, flat).reshape([q, classes])
    }

    /// Run `[R, B, Q, H]` requests as one cache-sharing microbatch; returns `[R, Q, C]`.
    fn predict_proba_batch<'py>(
        &self,
        py: Python<'py>,
        variants: PyReadonlyArray4<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let shape = variants.shape();
        let (r, b, q, h) = (shape[0], shape[1], shape[2], shape[3]);
        if r == 0 {
            return Err(PyValueError::new_err(
                "microbatch must contain at least one request",
            ));
        }
        let values = contiguous(variants.as_slice(), "[R, B, Q, H]")?;
        let (flat, classes) = py.detach(|| self.run(r, b, q, h, values))?;
        PyArray1::from_vec(py, flat).reshape([r, q, classes])
    }

    /// Sparse-attention decisions taken since the last reset.
    fn sparse_attention_statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let statistics = self.executor.sparse_attention_statistics();
        let dict = PyDict::new(py);
        dict.set_item("attempted", statistics.attempted)?;
        dict.set_item("accepted", statistics.accepted)?;
        dict.set_item("fallback", statistics.fallback)?;
        Ok(dict)
    }

    fn reset_sparse_attention_statistics(&self) {
        self.executor.reset_sparse_attention_statistics();
    }
}

/// Import a `TabICLv2` checkpoint into a content-addressed model artifact and
/// return the artifact path. Re-uses an existing import unless `rebuild` is set.
#[pyfunction]
#[pyo3(signature = (checkpoint_path, cache_dir=None, rebuild=false))]
fn prepare_model(
    py: Python<'_>,
    checkpoint_path: PathBuf,
    cache_dir: Option<PathBuf>,
    rebuild: bool,
) -> PyResult<String> {
    let options = CheckpointCacheOptions {
        no_cache: false,
        rebuild_cache: rebuild,
        cache_dir,
    };
    let prepared = py
        .detach(|| prepare_checkpoint(&checkpoint_path, &options))
        .map_err(to_py_error)?;
    match prepared {
        PreparedCheckpoint::Cached { path, .. } => Ok(path.to_string_lossy().into_owned()),
        PreparedCheckpoint::Direct { .. } => Err(IronICLError::new_err(
            "checkpoint import did not produce a cached model artifact",
        )),
    }
}

/// Build the static context of a fitted training table natively and write it
/// to `output_path`; returns the file description the Python package records.
///
/// `x` holds the `[B, N, H]` FP32 ensemble variants of the training rows, `y`
/// the `[B, N]` class indices as FP32, and the permutations are `[B, C]` and
/// `[B, H]` unsigned 64-bit arrays, all as the upstream preprocessing produces
/// them. `context_sha256` is the 32-byte identity of the training table and
/// settings.
#[pyfunction]
#[pyo3(signature = (model_path, x, y, class_permutations, feature_permutations, context_sha256, output_path, *, kv_dtype="fp32", pack_keys=false, index_clusters=None))]
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn build_context<'py>(
    py: Python<'py>,
    model_path: PathBuf,
    x: PyReadonlyArray3<'py, f32>,
    y: PyReadonlyArray2<'py, f32>,
    class_permutations: PyReadonlyArray2<'py, u64>,
    feature_permutations: PyReadonlyArray2<'py, u64>,
    context_sha256: Vec<u8>,
    output_path: PathBuf,
    kv_dtype: &str,
    pack_keys: bool,
    index_clusters: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let (b, n, h) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    if y.shape() != [b, n] {
        return Err(PyValueError::new_err("y must have shape [B, N] matching x"));
    }
    if class_permutations.shape()[0] != b || feature_permutations.shape() != [b, h] {
        return Err(PyValueError::new_err(
            "permutations must have shapes [B, C] and [B, H] matching x",
        ));
    }
    let context_sha256: [u8; 32] = context_sha256
        .try_into()
        .map_err(|_| PyValueError::new_err("context_sha256 must be 32 bytes"))?;
    let kv_dtype = kv_dtype.parse::<KvDtype>().map_err(to_py_error)?;
    let x = contiguous_array(x.as_slice(), "x", "float32", "[B, N, H]")?;
    let y = contiguous_array(y.as_slice(), "y", "float32", "[B, N]")?;
    let class_permutations = contiguous_array(
        class_permutations.as_slice(),
        "class_permutations",
        "uint64",
        "[B, C]",
    )?;
    let feature_permutations = contiguous_array(
        feature_permutations.as_slice(),
        "feature_permutations",
        "uint64",
        "[B, H]",
    )?;
    let request = StaticContextRequest::new(
        b,
        n,
        n,
        h,
        x,
        y,
        class_permutations,
        feature_permutations,
        context_sha256,
    )
    .map_err(to_py_error)?;
    let options = StaticContextOptions {
        kv_dtype,
        pack_keys,
        sparse_index_clusters: index_clusters,
    };
    let summary = py
        .detach(|| {
            let executor = ClassifierExecutor::new(load_model_artifact(&model_path)?);
            build_static_context_file(&executor, &request, &options, &output_path)
        })
        .map_err(to_py_error)?;
    let dict = PyDict::new(py);
    dict.set_item("B", summary.ensemble_count)?;
    dict.set_item("N", summary.training_rows)?;
    dict.set_item("H", summary.feature_count)?;
    dict.set_item("C", summary.class_count)?;
    dict.set_item("static_context_version", summary.version)?;
    dict.set_item("kv_dtype", summary.kv_dtype.name())?;
    dict.set_item("packed_icl_keys", summary.packed_icl_keys)?;
    dict.set_item("index_clusters", summary.sparse_index_clusters)?;
    dict.set_item("context_file_bytes", summary.file_bytes)?;
    dict.set_item("context_file_sha256", encode_sha256(&summary.file_sha256))?;
    Ok(dict)
}

/// Summarise a verified model artifact.
#[pyfunction]
fn inspect_model(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyDict>> {
    let summary = py.detach(|| inspect_model_artifact(&path)).map_err(to_py_error)?;
    let dict = PyDict::new(py);
    dict.set_item("source_sha256", summary.source_sha256)?;
    dict.set_item("profile", format!("{:?}", summary.profile))?;
    dict.set_item("tensor_count", summary.tensor_count)?;
    dict.set_item("parameter_count", summary.parameter_count)?;
    dict.set_item("tensor_bytes", summary.tensor_bytes)?;
    Ok(dict)
}

#[pymodule(name = "_ironicl")]
fn ironicl_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStaticExecutor>()?;
    m.add_function(wrap_pyfunction!(prepare_model, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_model, m)?)?;
    m.add_function(wrap_pyfunction!(build_context, m)?)?;
    m.add("IronICLError", m.py().get_type::<IronICLError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

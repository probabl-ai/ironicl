//! Uncached scalar FP32 execution for the official `TabICLv2` classifier.

use std::sync::{Arc, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use sha2::{Digest, Sha256};

use crate::progress::Stage;
use crate::{
    Error, ModelArtifact, QueryRequest, RegressionOutput, Request, Result, StaticContextArtifact,
    StaticContextMetadata, TaskKind,
    format::static_context::{
        PageAlignedTensors, StaticContextTensor, StaticContextTensorData, TensorStorage,
    },
    kernels::reference::{
        OnlineAttentionScratch, QassMaxWeights, affine_f32, affine_input3_transposed_f32, gelu_f32,
        layer_norm_f32, one_hot_linear_f32, qassmax_f32, rope_non_interleaved_f32,
        source_major_online_attention_f32_key_bf16_value_with_backend_f32,
        source_major_online_attention_with_backend_f32, tiled_online_scaled_dot_product_attention_f32,
    },
    memory::{Builder, ContextShape, MemoryLimits, Plan, Pools, Storage},
    models::tabicl_v2::Device,
    models::tabicl_v2::execution_plan::ExecutionPlan,
    models::tabicl_v2::quantiles::summarize,
    profile::{self, Category},
};

pub(crate) const COLUMN_DIM: usize = 128;
pub(crate) const ICL_DIM: usize = 512;
pub(crate) const HEADS: usize = 8;
pub(crate) const INDUCING: usize = 128;
pub(crate) const COLUMN_BLOCKS: usize = 3;
pub(crate) const ICL_BLOCKS: usize = 12;
/// CLS streams ahead of the features in the row stage.
pub(crate) const RESERVED: usize = 4;
const ATTENTION_QUERY_BLOCK_ROWS: usize = 16;
const ATTENTION_KEY_TILE_ROWS: usize = 128;
const MAX_ACTIVATION_BYTES: usize = 1536 * 1024 * 1024;
const MAX_ACTIVATION_ELEMENTS: usize = MAX_ACTIVATION_BYTES / std::mem::size_of::<f32>();
/// Cached prediction splits larger queries. The GPU keeps the buffers and
/// compiled graph of every query bucket up to this size.
pub(crate) const QUERY_CHUNK_ROWS: usize = 1024;
static EXECUTOR_THREADS: OnceLock<usize> = OnceLock::new();

/// Owned classifier logits in row-major `[B, test_rows, C]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderOutputs {
    shape: [usize; 3],
    values: Vec<f32>,
}

/// How far Ironicl may trade exactness for speed.
///
/// On the Apple Silicon GPU, `Auto` and `Fast` store ICL caches in FP16. On a
/// BF16-capable CPU (`AVX512_BF16`), `Fast` stores ICL values in BF16. Elsewhere
/// every level is FP32.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Optimize {
    /// FP32 throughout; matches `TabICL` within `1e-4`.
    #[default]
    Safe,
    /// On the Apple GPU: ICL keys and values in FP16, ICL attention in FP16;
    /// everything else FP32 (measured probability error below 5e-4, labels
    /// unchanged). On the CPU: FP32, same as `Safe`.
    Auto,
    /// Everything `Auto` does, plus, on a BF16-capable CPU, ICL values stored in
    /// BF16 (FP32 keys). Further reduced-precision work is allowed here first.
    Fast,
}

impl std::str::FromStr for Optimize {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "safe" => Ok(Self::Safe),
            "auto" => Ok(Self::Auto),
            "fast" => Ok(Self::Fast),
            _ => Err(Error::InvalidArgument(format!(
                "optimize must be \"safe\", \"auto\", or \"fast\", not {value:?}"
            ))),
        }
    }
}

/// How ensemble logits become class probabilities, matching `TabICLClassifier`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifierAggregation {
    /// Divides logits before softmax. Must be positive and finite.
    pub softmax_temperature: f32,
    /// Average logits before softmax when true; otherwise average probabilities.
    pub average_logits: bool,
}

impl Default for ClassifierAggregation {
    fn default() -> Self {
        Self {
            softmax_temperature: 0.9,
            average_logits: true,
        }
    }
}

/// Aggregated classifier probabilities in row-major `[test_rows, C]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierProbabilities {
    shape: [usize; 2],
    values: Vec<f32>,
}

impl ClassifierProbabilities {
    pub(crate) fn from_rows(values: Vec<f32>, classes: usize) -> Self {
        Self {
            shape: [values.len() / classes.max(1), classes],
            values,
        }
    }

    /// Output dimensions `[test_rows, C]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Output probabilities in row-major `[test_rows, C]` order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

impl DecoderOutputs {
    /// Wrap row-major `[B, rows, outputs]` values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `values` does not fill `shape`.
    pub fn new(shape: [usize; 3], values: Vec<f32>) -> Result<Self> {
        if checked_product(&shape, "decoder outputs")? != values.len() {
            return Err(Error::InvalidArgument(
                "decoder output values do not match their shape".into(),
            ));
        }
        Ok(Self { shape, values })
    }

    /// Output dimensions `[B, test_rows, C]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Output values in row-major `[B, test_rows, C]` order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Immutable, uncached executor for a validated official classifier artifact.
#[derive(Debug, Clone)]
pub struct Executor {
    model: Arc<ModelArtifact>,
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    gpu_weights: Arc<OnceLock<Option<Arc<crate::kernels::metal::ModelWeights>>>>,
    #[cfg(feature = "cuda")]
    cuda_weights: Arc<OnceLock<Result<Option<Arc<crate::models::tabicl_v2::cuda::ModelWeights>>>>>,
}

/// Attention keys and values in head-major `[heads, rows, head_dim]` order.
type KeyValue = (Vec<f32>, Vec<f32>);

/// One estimator's captured keys and values: per column block, every feature's
/// in feature order; per ICL block, the training rows'.
type EstimatorCaches = (Vec<KeyValue>, Vec<KeyValue>);

/// One embedded column stream, with its keys and values when captured.
type ColumnStream = (usize, Vec<f32>, Vec<KeyValue>);

/// Immutable query-only executor backed by one exact full-K/V static context.
///
/// It can return raw estimator logits in `[B, Q, C]` order or aggregate them
/// into final class probabilities in `[Q, C]` order.
#[derive(Debug, Clone)]
pub struct StaticExecutor {
    model: Executor,
    context: StaticContextArtifact,
    affine_pool: Arc<WorkerPool>,
    column_input_weight: Arc<Vec<f32>>,
    qass_base_scales: Arc<Vec<Vec<f32>>>,
    isa: crate::kernels::cpu::Isa,
    device: Device,
}

impl Executor {
    /// Take ownership of an exact-manifest validated classifier model.
    #[must_use]
    pub fn new(model: ModelArtifact) -> Self {
        WorkerPool::init_global();
        Self {
            model: Arc::new(model),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            gpu_weights: Arc::default(),
            #[cfg(feature = "cuda")]
            cuda_weights: Arc::default(),
        }
    }

    /// Load a persisted static context and pair it with this model's weights,
    /// prefaulting the mapping so the first prediction runs at RAM speed. The
    /// model weights are shared, not copied. `limits` size the prediction
    /// buffers, as in [`Self::build_context`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file is unreadable or invalid, or if the context
    /// does not match this model's serving contract.
    pub fn load_context(
        &self,
        path: &std::path::Path,
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
    ) -> Result<StaticExecutor> {
        let context = StaticContextArtifact::read_mapped_prefault(path)?;
        StaticExecutor::with_executor(self.clone(), context, optimize, threads, limits)
    }

    /// Load a context from the bytes [`StaticExecutor::encode_context`] wrote,
    /// as [`Self::load_context`] does from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are invalid, or if the context does not
    /// match this model's serving contract.
    pub fn load_context_bytes(
        &self,
        bytes: &[u8],
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
    ) -> Result<StaticExecutor> {
        let context = StaticContextArtifact::from_bytes(bytes)?;
        StaticExecutor::with_executor(self.clone(), context, optimize, threads, limits)
    }

    /// Whether the model is the classifier or the regressor.
    #[must_use]
    pub fn task(&self) -> TaskKind {
        self.model.task()
    }

    /// The source-checkpoint SHA-256 that identifies this model.
    #[must_use]
    pub fn source_sha256(&self) -> &str {
        &self.model.summary().source_sha256
    }

    /// The CUDA device and the model's weights on it, created on first use
    /// and shared by every context of the model. `None` without a CUDA device
    /// or with `IRONICL_NO_CUDA` set.
    ///
    /// # Errors
    ///
    /// Returns an error if a CUDA device exists but the kernels or weights
    /// fail to load; set `IRONICL_NO_CUDA` to use the CPU instead.
    #[cfg(feature = "cuda")]
    fn cuda_weights(&self) -> Result<Option<Arc<crate::models::tabicl_v2::cuda::ModelWeights>>> {
        self.cuda_weights
            .get_or_init(|| {
                crate::models::tabicl_v2::cuda::Device::new()?
                    .map(|device| {
                        crate::models::tabicl_v2::cuda::ModelWeights::new(device, |name| self.tensor(name))
                            .map(Arc::new)
                    })
                    .transpose()
            })
            .clone()
    }

    /// The GPU and the model's weights in shared GPU buffers, created on first
    /// use and shared by every context of the model. `None` without a GPU or
    /// with `IRONICL_NO_METAL` set.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn gpu_weights(&self) -> Option<Arc<crate::kernels::metal::ModelWeights>> {
        if std::env::var_os("IRONICL_NO_METAL").is_some() {
            return None;
        }
        self.gpu_weights
            .get_or_init(|| {
                let gpu = crate::kernels::metal::Gpu::new().ok().flatten()?;
                Some(Arc::new(crate::kernels::metal::ModelWeights::new(gpu)))
            })
            .clone()
    }

    /// Build every estimator's keys and values on the GPU, straight into
    /// page-aligned context tensors. Returns `None` without a GPU, below
    /// `metal::GPU_MIN_BUILD_ROWS`, when a grouped triple is the all-missing
    /// marker that only the CPU path handles, or when one estimator exceeds the
    /// build's memory budget.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn build_tensors_gpu(
        &self,
        (b, n, h): (usize, usize, usize),
        x: &[f32],
        y_train: &[f32],
        optimize: Optimize,
        limits: MemoryLimits,
    ) -> Result<Option<PageAlignedTensors>> {
        let icl_storage = match optimize {
            Optimize::Safe => TensorStorage::Fp32,
            // BF16 had no native GPU arithmetic on an M1 Max: slower and less
            // accurate than FP16, so `Fast` uses FP16 until it gets more.
            Optimize::Auto | Optimize::Fast => TensorStorage::F16,
        };
        if n < crate::models::tabicl_v2::metal::GPU_MIN_BUILD_ROWS {
            return Ok(None);
        }
        let Some(gpu_weights) = self.gpu_weights() else {
            return Ok(None);
        };
        // Each feature grouped with its neighbors, `[B * H, N, 3]`.
        let Some(grouped) = x
            .chunks_exact(n * h)
            .map(|x| crate::models::tabicl_v2::metal::group_features(x, n, h))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let grouped = grouped.concat();
        let (column_labels, icl_labels) = self.label_embeddings(n, y_train)?;
        // The memory plan sizes the estimator groups: half the host's available
        // RAM (unified memory is the host's RAM), capped by Metal's working set
        // and by the memory limit. When one estimator does not fit the device,
        // the CPU builds the context.
        let pools = shared_gpu_pools(gpu_weights.gpu(), limits);
        let estimator_bytes = u64::try_from(crate::models::tabicl_v2::metal::gpu_build_estimator_bytes(n, h))
            .unwrap_or(u64::MAX);
        let Some(group) = crate::memory::gpu_build_group(b as u64, estimator_bytes, pools) else {
            return Ok(None);
        };
        let group = usize::try_from(group).map_err(|_| Error::ArithmeticOverflow("GPU build group"))?;
        let column_scales = (0..COLUMN_BLOCKS)
            .map(|block| {
                qass_base_scale(
                    self,
                    &format!("col_embedder.tf_col.blocks.{block}.multihead_attn1.attn"),
                    n,
                    COLUMN_DIM,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let icl_scales = self.icl_base_scales(n)?;
        // Only the ICL caches, almost all of a context, take the reduced type.
        let tensors = |ensembles: usize| context_page(ensembles, n, h, icl_storage);
        let mut context = tensors(b)?;
        let mut scratch = if group == b { None } else { Some(tensors(group)?) };
        crate::progress::report(Stage::Build, 0, b)?;
        for start in (0..b).step_by(group) {
            let output = scratch.as_mut().unwrap_or(&mut context);
            crate::models::tabicl_v2::metal::build_context(
                &gpu_weights,
                &|name| self.tensor(name),
                crate::models::tabicl_v2::metal::Shape {
                    ensembles: group,
                    features: h,
                    training_rows: n,
                },
                &column_scales,
                &icl_scales,
                &grouped[start * h * n * 3..][..group * h * n * 3],
                &column_labels[start * n * COLUMN_DIM..][..group * n * COLUMN_DIM],
                &icl_labels[start * n * ICL_DIM..][..group * n * ICL_DIM],
                icl_storage,
                &output.regions(),
            )?;
            // Estimators are the outermost dimension, so a group's slice of
            // each tensor is contiguous.
            if let Some(scratch) = &scratch {
                context.copy_group(scratch, start / group);
            }
            crate::progress::report(Stage::Build, start + group, b)?;
        }
        Ok(Some(context))
    }

    /// Each estimator's training labels `y_train` `[B, N]` embedded for the
    /// column stage, `[B, N, 128]`, and for the ICL stage, `[B, N, 512]`.
    #[cfg(any(
        feature = "cuda",
        all(feature = "metal", target_os = "macos", target_arch = "aarch64")
    ))]
    fn label_embeddings(&self, n: usize, y_train: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut column = vec![0.0; y_train.len() * COLUMN_DIM];
        let mut icl = vec![0.0; y_train.len() * ICL_DIM];
        column
            .par_chunks_mut(n * COLUMN_DIM)
            .zip(icl.par_chunks_mut(n * ICL_DIM))
            .zip(y_train.par_chunks(n))
            .try_for_each(|((column, icl), labels)| {
                self.target_embeddings("col_embedder", labels, COLUMN_DIM, column)?;
                self.target_embeddings("icl_predictor", labels, ICL_DIM, icl)
            })?;
        Ok((column, icl))
    }

    /// Each ICL block's `QASSMax` base scale for `n` training rows, `[8, 64]`.
    fn icl_base_scales(&self, n: usize) -> Result<Vec<Vec<f32>>> {
        (0..ICL_BLOCKS)
            .map(|block| {
                qass_base_scale(
                    self,
                    &format!("icl_predictor.tf_icl.blocks.{block}.attn"),
                    n,
                    ICL_DIM,
                )
            })
            .collect()
    }

    /// Run the training rows once and keep their attention keys and values as
    /// an exact static context for query-only prediction.
    ///
    /// `x` holds preprocessed FP32 `[B, N, H]` training rows and `y_train` the
    /// `[B, N]` class-permuted labels. The permutations are stored as context
    /// metadata; `class_permutations` is used to undo each estimator's class
    /// order when probabilities are aggregated. `optimize` selects the ICL
    /// cache storage: FP16 on the Apple GPU under `Auto`/`Fast`, BF16 values on
    /// a BF16-capable CPU under `Fast`, FP32 otherwise. `limits` lower the
    /// memory the build plans its estimator groups for; pass
    /// `MemoryLimits::resolve(None, None)?` to read them from the environment.
    ///
    /// With `cache_dir`, the context is loaded from `<identity>.irctx` there
    /// when that file holds exactly this context, and otherwise built and
    /// written there. The identity covers the Ironicl version, the model,
    /// `optimize`, and the inputs. A missing, invalid, or mismatched entry is
    /// rebuilt, and a failed write leaves the built context in memory only.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent dimensions or labels, invalid
    /// permutations, or when the context does not fit in memory.
    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    pub fn build_context(
        &self,
        ensembles: usize,
        training_rows: usize,
        features: usize,
        x: &[f32],
        y_train: &[f32],
        class_permutations: Vec<u64>,
        feature_permutations: Vec<u64>,
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
        cache_dir: Option<&std::path::Path>,
    ) -> Result<StaticExecutor> {
        let shape = (ensembles, training_rows, features);
        let metadata = self.context_metadata(
            shape,
            x,
            y_train,
            class_permutations,
            feature_permutations,
            optimize,
        )?;
        let Some(directory) = cache_dir else {
            return self.build_uncached(metadata, shape, x, y_train, optimize, threads, limits);
        };
        let path = directory.join(format!(
            "{}.irctx",
            crate::format::encode_sha256(&metadata.context_sha256)
        ));
        if let Some(context) = StaticContextArtifact::read_cached(&path, &metadata)
            .filter(|context| validate_static_context(&self.model, context).is_ok())
        {
            // The entry's last use, for `ironicl.clear_cache`.
            let _ = std::fs::File::options()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_modified(std::time::SystemTime::now()));
            return StaticExecutor::with_executor(self.clone(), context, optimize, threads, limits);
        }
        let executor = self.build_uncached(metadata, shape, x, y_train, optimize, threads, limits)?;
        // Caching is best effort: a context that cannot be written is still returned.
        if std::fs::create_dir_all(directory).is_ok() {
            let _ = executor.save_context(&path);
        }
        Ok(executor)
    }

    /// Build the context that validated `metadata` describes, without the cache.
    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    fn build_uncached(
        &self,
        metadata: StaticContextMetadata,
        shape: (usize, usize, usize),
        x: &[f32],
        y_train: &[f32],
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
    ) -> Result<StaticExecutor> {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(tensors) = self.build_tensors_gpu(shape, x, y_train, optimize, limits)? {
            // The build graph checked every value on the GPU.
            let context = StaticContextArtifact::from_page_aligned(metadata, tensors)?;
            return StaticExecutor::with_executor(self.clone(), context, optimize, threads, limits);
        }
        // With a CUDA GPU, every level builds on it and stores the ICL keys and
        // values as the GPU predicts from them: FP32 for `safe`, FP16 for
        // `auto` and `fast`, as on Apple GPUs.
        #[cfg(feature = "cuda")]
        if let Some(weights) = self.cuda_weights()? {
            let precision = crate::models::tabicl_v2::cuda::Precision::for_optimize(optimize);
            let built = self.build_tensors_cuda(&weights, shape, x, y_train, precision, limits)?;
            let (tensors, built) = if let Some(built) = built {
                built
            } else {
                let tensors = self.build_tensors_cpu(shape, x, y_train, limits)?;
                let icl = |name: &str| name.starts_with("icl.");
                (page_aligned(&tensors, icl, cuda_icl_storage(precision))?, None)
            };
            let context = StaticContextArtifact::from_page_aligned(metadata, tensors)?;
            return StaticExecutor::prepared(self.clone(), context, threads, limits)?
                .with_cuda(optimize, limits, built);
        }
        // On a BF16-capable CPU, `fast` stores the ICL values in BF16 (FP32
        // keys, the retained serving layout); every other case is FP32.
        if optimize == Optimize::Fast && crate::kernels::cpu::bf16_storage_supported() {
            let tensors = self.build_tensors_cpu(shape, x, y_train, limits)?;
            let values = |name: &str| name.starts_with("icl.") && name.rsplit('.').next() == Some("value");
            let context = StaticContextArtifact::from_page_aligned(
                metadata,
                page_aligned(&tensors, values, TensorStorage::Bf16)?,
            )?;
            return StaticExecutor::with_executor(self.clone(), context, optimize, threads, limits);
        }
        let context =
            StaticContextArtifact::new(metadata, self.build_tensors_cpu(shape, x, y_train, limits)?)?;
        StaticExecutor::with_executor(self.clone(), context, optimize, threads, limits)
    }

    /// Build every estimator's keys and values on the CUDA device, with
    /// products at `precision`, into page-aligned context tensors, the ICL ones
    /// as [`cuda_icl_storage`] says, and return them with the ICL ones the build left on the device
    /// for prediction. Returns `None` when a grouped triple is the all-missing
    /// marker that only the CPU path handles, or when the context does not
    /// [`crate::models::tabicl_v2::cuda::fits`] the device.
    #[cfg(feature = "cuda")]
    fn build_tensors_cuda(
        &self,
        weights: &crate::models::tabicl_v2::cuda::ModelWeights,
        (b, n, h): (usize, usize, usize),
        x: &[f32],
        y_train: &[f32],
        precision: crate::models::tabicl_v2::cuda::Precision,
        limits: MemoryLimits,
    ) -> Result<
        Option<(
            PageAlignedTensors,
            Option<crate::models::tabicl_v2::cuda::BuiltCache>,
        )>,
    > {
        use crate::models::tabicl_v2::cuda::{BuildInputs, BuildOutputs, Shape};
        let shape = Shape {
            ensembles: b,
            features: h,
            rows: n,
            outputs: self.decoder_width()?,
        };
        let budget = weights.budget(limits.device.map(|limit| limit.bytes))?;
        if !crate::models::tabicl_v2::cuda::fits(budget, precision, shape) || has_missing_marker(x, h) {
            return Ok(None);
        }
        let (column_labels, icl_labels) = self.label_embeddings(n, y_train)?;
        let inducing_queries = (0..COLUMN_BLOCKS)
            .map(|block| {
                let prefix = format!("col_embedder.tf_col.blocks.{block}");
                let inducing = self.tensor(&format!("{prefix}.ind_vectors"))?;
                let prefix = format!("{prefix}.multihead_attn1");
                let heads = self.query_heads(inducing, INDUCING, n, COLUMN_DIM, &prefix, true)?;
                Ok(heads_to_tokens(&heads, INDUCING, COLUMN_DIM, HEADS))
            })
            .collect::<Result<Vec<_>>>()?;
        let base_scales = self.icl_base_scales(n)?;
        let mut page = context_page(b, n, h, cuda_icl_storage(precision))?;
        // `context_tensors` order: the column blocks, then the ICL blocks, each
        // its key then its value.
        let mut tensors = page.tensors_mut().into_iter();
        let mut pairs = std::iter::from_fn(|| Some((tensors.next()?, tensors.next()?)));
        let mut outputs = BuildOutputs {
            column: pairs
                .by_ref()
                .take(COLUMN_BLOCKS)
                .map(|(key, value)| (bytemuck::cast_slice_mut(key), bytemuck::cast_slice_mut(value)))
                .collect(),
            icl: pairs.collect(),
        };
        let built = crate::models::tabicl_v2::cuda::build(
            weights,
            precision,
            &BuildInputs {
                shape,
                x,
                column_labels: &column_labels,
                icl_labels: &icl_labels,
                inducing_queries: &inducing_queries,
                base_scales: &base_scales,
                rope_frequencies: self.tensor("row_interactor.tf_row.rope.freqs")?,
                cls_tokens: self.tensor("row_interactor.cls_tokens")?,
                budget,
            },
            &mut outputs,
        )?;
        drop(outputs);
        Ok(Some((page, built)))
    }

    /// Validate a context's inputs and describe it: model and content hashes,
    /// shape, class count, and permutations. The content hash is the context's
    /// cache identity: it covers everything the stored tensors depend on.
    fn context_metadata(
        &self,
        (b, n, h): (usize, usize, usize),
        x: &[f32],
        y_train: &[f32],
        class_permutations: Vec<u64>,
        feature_permutations: Vec<u64>,
        optimize: Optimize,
    ) -> Result<StaticContextMetadata> {
        if b == 0
            || n == 0
            || h == 0
            || x.len() != checked_product(&[b, n, h], "context features")?
            || y_train.len() != checked_product(&[b, n], "context labels")?
        {
            return Err(Error::InvalidArgument(format!(
                "context needs nonempty x[B, N, H] and y_train[B, N] for B={b}, N={n}, H={h}"
            )));
        }
        let class_count = self.validate_targets(y_train, b, n)?;
        let mut model_sha256 = [0; 32];
        let source = self.model.summary().source_sha256.as_bytes();
        for (byte, pair) in model_sha256.iter_mut().zip(source.as_chunks::<2>().0) {
            *byte = std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
                .ok_or_else(|| Error::Artifact("model source SHA-256 is not hexadecimal".into()))?;
        }
        // Bump the prefix when built contexts change within an Ironicl version.
        let mut context_sha256 = Sha256::new_with_prefix(b"ironicl-context-v2\0");
        let level = match optimize {
            Optimize::Safe => "safe",
            Optimize::Auto => "auto",
            Optimize::Fast => "fast",
        };
        for part in [
            env!("CARGO_PKG_VERSION"),
            &self.model.summary().source_sha256,
            level,
        ] {
            context_sha256.update(part.as_bytes());
            context_sha256.update([0]);
        }
        for value in [b, n, h] {
            context_sha256.update((value as u64).to_le_bytes());
        }
        context_sha256.update(bytemuck::cast_slice::<f32, u8>(x));
        context_sha256.update(bytemuck::cast_slice::<f32, u8>(y_train));
        context_sha256.update(bytemuck::cast_slice::<u64, u8>(&class_permutations));
        context_sha256.update(bytemuck::cast_slice::<u64, u8>(&feature_permutations));
        Ok(StaticContextMetadata {
            model_sha256,
            context_sha256: context_sha256.finalize().into(),
            ensemble_count: b,
            training_rows: n,
            feature_count: h,
            class_count,
            class_permutations,
            feature_permutations,
            optimize: Some(optimize),
        })
    }

    /// Build every estimator's keys and values on the CPU, in the order of
    /// [`context_tensors`].
    fn build_tensors_cpu(
        &self,
        (b, n, h): (usize, usize, usize),
        x: &[f32],
        y_train: &[f32],
        limits: MemoryLimits,
    ) -> Result<Vec<StaticContextTensor>> {
        let streams = h + RESERVED;
        let column_elements = HEADS * INDUCING * 16;
        // The context is the one allocation that grows with N x B; running
        // out of memory for it is an error, not an abort.
        let reserve = |capacity: usize| -> Result<KeyValue> {
            let (mut keys, mut values) = (Vec::new(), Vec::new());
            for buffer in [&mut keys, &mut values] {
                buffer.try_reserve_exact(capacity).map_err(|_| {
                    Error::InvalidArgument(format!("not enough memory for a {b} x {n} x {h} context"))
                })?;
            }
            Ok((keys, values))
        };
        let mut column = (0..COLUMN_BLOCKS)
            .map(|_| reserve(b * streams * column_elements))
            .collect::<Result<Vec<_>>>()?;
        let mut icl = (0..ICL_BLOCKS)
            .map(|_| reserve(b * n * ICL_DIM))
            .collect::<Result<Vec<_>>>()?;
        // Build as many estimators at once as the memory plan allows, and at
        // least one: a large context runs serially instead of failing.
        let estimate = cpu_build_estimator_bytes(n, h, self.decoder_width()?)?;
        let group = crate::memory::cpu_build_group(
            b as u64,
            u64::try_from(estimate).unwrap_or(u64::MAX),
            Pools::detect().with_limits(limits).host_available(),
        );
        let group = usize::try_from(group).map_err(|_| Error::ArithmeticOverflow("CPU build group"))?;
        crate::progress::report(Stage::Build, 0, b)?;
        for members in (0..b).collect::<Vec<_>>().chunks(group) {
            let estimators: Vec<EstimatorCaches> = members
                .par_iter()
                .map(|&ensemble| {
                    let x = &x[ensemble * n * h..][..n * h];
                    let labels = &y_train[ensemble * n..][..n];
                    let (columns, feature_caches) = self.column_embeddings(x, labels, n, h, n, true)?;
                    let rows = self.row_embeddings(&columns, n, streams)?;
                    let mut caches = Vec::with_capacity(12);
                    self.icl_rows(rows, labels, n, n, Some(&mut caches))?;
                    // Per block, every feature's keys and values in feature order.
                    let column = (0..COLUMN_BLOCKS)
                        .map(|block| {
                            let keys = feature_caches
                                .iter()
                                .flat_map(|caches| caches[block].0.iter().copied());
                            let values = feature_caches
                                .iter()
                                .flat_map(|caches| caches[block].1.iter().copied());
                            (keys.collect(), values.collect())
                        })
                        .collect();
                    Ok((column, caches))
                })
                .collect::<Result<Vec<_>>>()?;
            for (column_caches, caches) in estimators {
                for ((keys, values), (key, value)) in column.iter_mut().zip(column_caches) {
                    // The cached path never reads the reserved CLS streams.
                    keys.resize(keys.len() + RESERVED * column_elements, 0.0);
                    values.resize(values.len() + RESERVED * column_elements, 0.0);
                    keys.extend(key);
                    values.extend(value);
                }
                for ((key, value), (keys, values)) in caches.into_iter().zip(&mut icl) {
                    keys.extend(key);
                    values.extend(value);
                }
            }
            crate::progress::report(Stage::Build, members[members.len() - 1] + 1, b)?;
        }
        Ok(column
            .into_iter()
            .chain(icl)
            .flat_map(|(keys, values)| [keys, values])
            .zip(context_tensors(b, n, h))
            .map(|(values, (name, dimensions))| StaticContextTensor {
                name,
                dimensions,
                values,
            })
            .collect())
    }

    /// Execute the complete FP32 classifier forward graph without persistent caches.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid labels, arithmetic overflow, excessive
    /// baseline activation memory or an
    /// unavailable validated model tensor. This uncached scalar executor is
    /// intended for small baseline runs, not 50K-row serving workloads.
    #[expect(
        clippy::many_single_char_names,
        reason = "b, q, h, n, c name the model's dimensions"
    )]
    pub fn predict_logits(&self, request: Request<'_>) -> Result<DecoderOutputs> {
        let (b, t, h, n) = (
            request.ensemble_count(),
            request.total_rows(),
            request.feature_count(),
            request.training_rows(),
        );
        let width = self.decoder_width()?;
        admit_request(b, t, h, n, width)?;
        // A regression fit keeps every decoded quantile.
        let classes = match self.validate_targets(request.y_train(), b, n)? {
            0 => width,
            classes => classes,
        };
        let test = t.checked_sub(n).ok_or(Error::ArithmeticOverflow("test rows"))?;
        let output_len = checked_product(&[b, test, classes], "decoder output")?;
        let mut values = Vec::with_capacity(output_len);
        for ensemble in 0..b {
            let x = &request.x()[checked_product(&[ensemble, t, h], "feature offset")?..]
                [..checked_product(&[t, h], "feature rows")?];
            let labels = &request.y_train()[ensemble * n..][..n];
            let (columns, _) = self.column_embeddings(x, labels, t, h, n, false)?;
            let rows = self.row_embeddings(&columns, t, h + RESERVED)?;
            let logits = self.icl_logits(rows, labels, t, n)?;
            for row in n..t {
                values.extend_from_slice(&logits[row * width..][..classes]);
            }
        }
        Ok(DecoderOutputs {
            shape: [b, test, classes],
            values,
        })
    }

    /// Embed every feature stream. With `capture`, also return each feature's
    /// three `multihead_attn2` key/value pairs, in feature order.
    #[expect(
        clippy::many_single_char_names,
        reason = "b, q, h, n, c name the model's dimensions"
    )]
    fn column_embeddings(
        &self,
        x: &[f32],
        labels: &[f32],
        t: usize,
        h: usize,
        n: usize,
        capture: bool,
    ) -> Result<(Vec<f32>, Vec<Vec<KeyValue>>)> {
        let _profile = profile::scope(Category::Column);
        let streams = h
            .checked_add(RESERVED)
            .ok_or(Error::ArithmeticOverflow("column streams"))?;
        let mut output = vec![0.0; checked_product(&[t, streams, COLUMN_DIM], "column embeddings")?];
        let cls = self.tensor("row_interactor.cls_tokens")?;
        for row in 0..t {
            for token in 0..RESERVED {
                output[(row * streams + token) * COLUMN_DIM..][..COLUMN_DIM]
                    .copy_from_slice(&cls[token * COLUMN_DIM..][..COLUMN_DIM]);
            }
        }
        let in_weight = self.tensor("col_embedder.in_linear.weight")?;
        let in_bias = self.tensor("col_embedder.in_linear.bias")?;
        let mut y = vec![0.0; checked_product(&[n, COLUMN_DIM], "column target embeddings")?];
        self.target_embeddings("col_embedder", labels, COLUMN_DIM, &mut y)?;
        // RowInteraction replaces the reserved streams with CLS tokens before
        // its first block, so running the column transformer for them is dead work.
        let y = y.as_slice();
        let results = scoped_chunks(h, executor_threads(), |features| -> Result<Vec<ColumnStream>> {
            let mut results = Vec::with_capacity(features.len());
            for feature in features {
                let (embedded, caches) =
                    self.column_stream(x, y, t, h, n, feature, in_weight, in_bias, capture)?;
                results.push((feature + RESERVED, embedded, caches));
            }
            Ok(results)
        })?;
        let mut feature_caches = Vec::with_capacity(if capture { h } else { 0 });
        for (stream, embedded, caches) in results.into_iter().flatten() {
            for row in 0..t {
                output[(row * streams + stream) * COLUMN_DIM..][..COLUMN_DIM]
                    .copy_from_slice(&embedded[row * COLUMN_DIM..][..COLUMN_DIM]);
            }
            if capture {
                feature_caches.push(caches);
            }
        }
        Ok((output, feature_caches))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    #[expect(
        clippy::float_cmp,
        clippy::many_single_char_names,
        reason = "-100.0 is an exact missing-value marker; b, q, h, n, c name the model's dimensions"
    )]
    fn column_stream(
        &self,
        x: &[f32],
        y: &[f32],
        t: usize,
        h: usize,
        n: usize,
        feature: usize,
        in_weight: &[f32],
        in_bias: &[f32],
        capture: bool,
    ) -> Result<(Vec<f32>, Vec<KeyValue>)> {
        let mut source = vec![0.0; checked_product(&[t, 3], "column grouped features")?];
        for row in 0..t {
            let target = &mut source[row * 3..][..3];
            for (part, offset) in [1_usize, 2, 4].iter().enumerate() {
                target[part] = x[row * h + (feature + offset) % h];
            }
        }
        let mut embedded = vec![0.0; checked_product(&[t, COLUMN_DIM], "column stream embeddings")?];
        affine_f32(&source, t, 3, in_weight, Some(in_bias), COLUMN_DIM, &mut embedded)?;
        for row in 0..t {
            if source[row * 3..][..3].iter().all(|value| *value == -100.0) {
                embedded[row * COLUMN_DIM..][..COLUMN_DIM].fill(-100.0);
            }
        }
        for row in 0..n {
            for dim in 0..COLUMN_DIM {
                embedded[row * COLUMN_DIM + dim] += y[row * COLUMN_DIM + dim];
            }
        }
        let mut caches = Vec::with_capacity(if capture { 3 } else { 0 });
        for block in 0..COLUMN_BLOCKS {
            let cache;
            (embedded, cache) = self.isab(embedded, t, n, block)?;
            if capture {
                // The cached path has no equivalent of the all-missing shortcut.
                caches.push(cache.ok_or_else(|| {
                    Error::NotImplemented("a static context for an all-missing feature column".into())
                })?);
            }
        }
        Ok((embedded, caches))
    }

    /// Return the block output and the `multihead_attn2` keys and values, or no
    /// keys and values when the all-missing stream passes through unchanged.
    #[expect(clippy::float_cmp, reason = "-100.0 is an exact missing-value marker")]
    fn isab(
        &self,
        source: Vec<f32>,
        t: usize,
        n: usize,
        block: usize,
    ) -> Result<(Vec<f32>, Option<KeyValue>)> {
        if source.iter().all(|value| *value == -100.0) {
            return Ok((source, None));
        }
        let prefix = format!("col_embedder.tf_col.blocks.{block}");
        let inducing = self.tensor(&format!("{prefix}.ind_vectors"))?;
        let hidden = self.block(
            inducing,
            &source[..n * COLUMN_DIM],
            &source[..n * COLUMN_DIM],
            INDUCING,
            n,
            COLUMN_DIM,
            &format!("{prefix}.multihead_attn1"),
            true,
            None,
        )?;
        let attn2 = format!("{prefix}.multihead_attn2");
        let cache = self.source_heads(&hidden, &hidden, INDUCING, COLUMN_DIM, &attn2, None)?;
        let output = self.attend(&source, t, &cache, INDUCING, COLUMN_DIM, &attn2, false, None)?;
        Ok((output, Some(cache)))
    }

    fn row_embeddings(&self, columns: &[f32], t: usize, streams: usize) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Row);
        let cls = self.tensor("row_interactor.cls_tokens")?;
        let chunks = scoped_chunks(t, executor_threads(), |rows| -> Result<Vec<f32>> {
            let mut output = Vec::with_capacity(rows.len() * ICL_DIM);
            for row in rows {
                output.extend_from_slice(&self.row_embedding(columns, cls, row, streams)?);
            }
            Ok(output)
        })?;
        let mut rows = Vec::with_capacity(t * ICL_DIM);
        for chunk in chunks {
            rows.extend_from_slice(&chunk);
        }
        Ok(rows)
    }

    fn row_embedding(&self, columns: &[f32], cls: &[f32], row: usize, streams: usize) -> Result<Vec<f32>> {
        let mut sequence = columns[row * streams * COLUMN_DIM..][..streams * COLUMN_DIM].to_vec();
        for token in 0..RESERVED {
            sequence[token * COLUMN_DIM..][..COLUMN_DIM]
                .copy_from_slice(&cls[token * COLUMN_DIM..][..COLUMN_DIM]);
        }
        for block in 0..2 {
            sequence = self.block(
                &sequence,
                &sequence,
                &sequence,
                streams,
                streams,
                COLUMN_DIM,
                &format!("row_interactor.tf_row.blocks.{block}"),
                false,
                Some(0),
            )?;
        }
        let final_cls = self.block(
            &sequence[..RESERVED * COLUMN_DIM],
            &sequence,
            &sequence,
            RESERVED,
            streams,
            COLUMN_DIM,
            "row_interactor.tf_row.blocks.2",
            false,
            Some(0),
        )?;
        let mut normalized = vec![0.0; ICL_DIM];
        layer_norm_f32(
            &final_cls,
            RESERVED,
            COLUMN_DIM,
            self.tensor("row_interactor.out_ln.weight")?,
            Some(self.tensor("row_interactor.out_ln.bias")?),
            &mut normalized,
        )?;
        Ok(normalized)
    }

    /// Add the training targets and run the twelve ICL blocks. With `capture`,
    /// also collect each block's training-row keys and values.
    fn icl_rows(
        &self,
        mut rows: Vec<f32>,
        labels: &[f32],
        t: usize,
        n: usize,
        mut capture: Option<&mut Vec<KeyValue>>,
    ) -> Result<Vec<f32>> {
        let mut targets = vec![0.0; checked_product(&[n, ICL_DIM], "ICL target embeddings")?];
        self.target_embeddings("icl_predictor", labels, ICL_DIM, &mut targets)?;
        for row in 0..n {
            for dim in 0..ICL_DIM {
                rows[row * ICL_DIM + dim] += targets[row * ICL_DIM + dim];
            }
        }
        for block in 0..ICL_BLOCKS {
            let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
            let source = &rows[..n * ICL_DIM];
            let cache = self.source_heads(source, source, n, ICL_DIM, &prefix, None)?;
            rows = self.attend(&rows, t, &cache, n, ICL_DIM, &prefix, true, None)?;
            if let Some(caches) = capture.as_deref_mut() {
                caches.push(cache);
            }
        }
        Ok(rows)
    }

    fn icl_logits(&self, rows: Vec<f32>, labels: &[f32], t: usize, n: usize) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Icl);
        let rows = self.icl_rows(rows, labels, t, n, None)?;
        let mut normalized = vec![0.0; checked_product(&[t, ICL_DIM], "ICL normalized rows")?];
        layer_norm_f32(
            &rows,
            t,
            ICL_DIM,
            self.tensor("icl_predictor.ln.weight")?,
            Some(self.tensor("icl_predictor.ln.bias")?),
            &mut normalized,
        )?;
        let mut hidden = vec![0.0; checked_product(&[t, 1024], "decoder hidden rows")?];
        affine_f32(
            &normalized,
            t,
            ICL_DIM,
            self.tensor("icl_predictor.decoder.0.weight")?,
            Some(self.tensor("icl_predictor.decoder.0.bias")?),
            1024,
            &mut hidden,
        )?;
        let mut activated = vec![0.0; hidden.len()];
        gelu_f32(&hidden, &mut activated)?;
        let width = self.decoder_width()?;
        let mut logits = vec![0.0; checked_product(&[t, width], "decoder logits")?];
        affine_f32(
            &activated,
            t,
            1024,
            self.tensor("icl_predictor.decoder.2.weight")?,
            Some(self.tensor("icl_predictor.decoder.2.bias")?),
            width,
            &mut logits,
        )?;
        Ok(logits)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    fn block(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_rows: usize,
        source_rows: usize,
        dim: usize,
        prefix: &str,
        qass: bool,
        rope: Option<usize>,
    ) -> Result<Vec<f32>> {
        let cache = self.source_heads(k, v, source_rows, dim, prefix, rope)?;
        self.attend(q, q_rows, &cache, source_rows, dim, prefix, qass, rope)
    }

    /// Normalize and project the attention sources into head-major keys and values.
    fn source_heads(
        &self,
        k: &[f32],
        v: &[f32],
        source_rows: usize,
        dim: usize,
        prefix: &str,
        rope: Option<usize>,
    ) -> Result<KeyValue> {
        let _profile = profile::scope(Category::Block);
        let norm_weight = self.tensor(&format!("{prefix}.norm1.weight"))?;
        let norm_bias = self.tensor(&format!("{prefix}.norm1.bias"))?;
        let mut k_norm = vec![0.0; k.len()];
        layer_norm_f32(k, source_rows, dim, norm_weight, Some(norm_bias), &mut k_norm)?;
        let v_norm = if std::ptr::eq(k.as_ptr(), v.as_ptr()) {
            k_norm.clone()
        } else {
            let mut x = vec![0.0; v.len()];
            layer_norm_f32(v, source_rows, dim, norm_weight, Some(norm_bias), &mut x)?;
            x
        };
        let attn_prefix = format!("{prefix}.attn");
        let weight = self.tensor(&format!("{attn_prefix}.in_proj_weight"))?;
        let bias = self.tensor(&format!("{attn_prefix}.in_proj_bias"))?;
        let mut k_proj = vec![0.0; k.len()];
        affine_f32(
            &k_norm,
            source_rows,
            dim,
            &weight[dim * dim..2 * dim * dim],
            Some(&bias[dim..2 * dim]),
            dim,
            &mut k_proj,
        )?;
        let mut v_proj = vec![0.0; v.len()];
        affine_f32(
            &v_norm,
            source_rows,
            dim,
            &weight[2 * dim * dim..],
            Some(&bias[2 * dim..]),
            dim,
            &mut v_proj,
        )?;
        let mut k_heads = tokens_to_heads(&k_proj, source_rows, dim, HEADS);
        let v_heads = tokens_to_heads(&v_proj, source_rows, dim, HEADS);
        if rope.is_some() {
            let mut rotated = vec![0.0; k_heads.len()];
            rope_non_interleaved_f32(
                &k_heads,
                HEADS,
                source_rows,
                dim / HEADS,
                self.tensor("row_interactor.tf_row.rope.freqs")?,
                0,
                &mut rotated,
            )?;
            k_heads = rotated;
        }
        Ok((k_heads, v_heads))
    }

    /// Block `prefix`'s head-major queries for `q` `[q_rows, dim]`: `norm1`
    /// and the query projection, then, with `qass`, `QASSMax` for
    /// `source_rows` keys.
    fn query_heads(
        &self,
        q: &[f32],
        q_rows: usize,
        source_rows: usize,
        dim: usize,
        prefix: &str,
        qass: bool,
    ) -> Result<Vec<f32>> {
        let mut q_norm = vec![0.0; q.len()];
        layer_norm_f32(
            q,
            q_rows,
            dim,
            self.tensor(&format!("{prefix}.norm1.weight"))?,
            Some(self.tensor(&format!("{prefix}.norm1.bias"))?),
            &mut q_norm,
        )?;
        let attn_prefix = format!("{prefix}.attn");
        let weight = self.tensor(&format!("{attn_prefix}.in_proj_weight"))?;
        let bias = self.tensor(&format!("{attn_prefix}.in_proj_bias"))?;
        let mut q_proj = vec![0.0; q.len()];
        affine_f32(
            &q_norm,
            q_rows,
            dim,
            &weight[..dim * dim],
            Some(&bias[..dim]),
            dim,
            &mut q_proj,
        )?;
        let q_heads = tokens_to_heads(&q_proj, q_rows, dim, HEADS);
        if !qass {
            return Ok(q_heads);
        }
        let mut gated = vec![0.0; q_heads.len()];
        self.qass(
            &attn_prefix,
            &q_heads,
            q_rows,
            source_rows,
            HEADS,
            dim / HEADS,
            &mut gated,
        )?;
        Ok(gated)
    }

    /// Attend from `q` to projected sources, then apply the residual feed-forward block.
    #[expect(clippy::too_many_lines, reason = "one attention, step by step")]
    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    fn attend(
        &self,
        q: &[f32],
        q_rows: usize,
        (k_heads, v_heads): &KeyValue,
        source_rows: usize,
        dim: usize,
        prefix: &str,
        qass: bool,
        rope: Option<usize>,
    ) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Block);
        let attn_prefix = format!("{prefix}.attn");
        let heads = HEADS;
        let head_dim = dim / heads;
        let mut q_heads = self.query_heads(q, q_rows, source_rows, dim, prefix, qass)?;
        if rope.is_some() {
            let mut rotated = vec![0.0; q_heads.len()];
            rope_non_interleaved_f32(
                &q_heads,
                heads,
                q_rows,
                head_dim,
                self.tensor("row_interactor.tf_row.rope.freqs")?,
                0,
                &mut rotated,
            )?;
            q_heads = rotated;
        }
        let mut attended = vec![0.0; q_heads.len()];
        if rope.is_some() && row_source_major_attention_enabled() {
            let query_block_rows = ATTENTION_QUERY_BLOCK_ROWS;
            let scratch_rows = q_rows.min(query_block_rows);
            let scratch_sources = source_rows.min(ATTENTION_KEY_TILE_ROWS);
            let mut scores = vec![0.0; scratch_rows * scratch_sources];
            let mut maximum = vec![0.0; heads * q_rows];
            let mut denominator = vec![0.0; heads * q_rows];
            source_major_online_attention_with_backend_f32(
                &q_heads,
                q_rows,
                k_heads,
                v_heads,
                source_rows,
                heads,
                head_dim,
                query_block_rows,
                ATTENTION_KEY_TILE_ROWS,
                crate::kernels::cpu::Isa::detect(),
                &mut OnlineAttentionScratch {
                    scores: &mut scores,
                    maximum: &mut maximum,
                    denominator: &mut denominator,
                },
                &mut attended,
            )?;
        } else {
            let scratch_rows = q_rows.min(ATTENTION_QUERY_BLOCK_ROWS);
            let scratch_sources = source_rows.min(ATTENTION_KEY_TILE_ROWS);
            let mut scores = vec![0.0; attention_scratch_scores(scratch_rows, scratch_sources)?];
            let mut maximum = vec![0.0; attention_scratch_state(scratch_rows)?];
            let mut denominator = vec![0.0; attention_scratch_state(scratch_rows)?];
            tiled_online_scaled_dot_product_attention_f32(
                &q_heads,
                q_rows,
                k_heads,
                v_heads,
                source_rows,
                heads,
                head_dim,
                ATTENTION_QUERY_BLOCK_ROWS,
                ATTENTION_KEY_TILE_ROWS,
                &mut OnlineAttentionScratch {
                    scores: &mut scores,
                    maximum: &mut maximum,
                    denominator: &mut denominator,
                },
                &mut attended,
            )?;
        }
        let merged = heads_to_tokens(&attended, q_rows, dim, heads);
        let mut projected = vec![0.0; q.len()];
        affine_f32(
            &merged,
            q_rows,
            dim,
            self.tensor(&format!("{attn_prefix}.out_proj.weight"))?,
            Some(self.tensor(&format!("{attn_prefix}.out_proj.bias"))?),
            dim,
            &mut projected,
        )?;
        let mut residual = q.iter().zip(&projected).map(|(a, b)| a + b).collect::<Vec<_>>();
        let mut ff_norm = vec![0.0; residual.len()];
        layer_norm_f32(
            &residual,
            q_rows,
            dim,
            self.tensor(&format!("{prefix}.norm2.weight"))?,
            Some(self.tensor(&format!("{prefix}.norm2.bias"))?),
            &mut ff_norm,
        )?;
        let mut ff = vec![0.0; checked_product(&[q_rows, dim, 2], "feedforward hidden rows")?];
        affine_f32(
            &ff_norm,
            q_rows,
            dim,
            self.tensor(&format!("{prefix}.linear1.weight"))?,
            Some(self.tensor(&format!("{prefix}.linear1.bias"))?),
            dim * 2,
            &mut ff,
        )?;
        let mut activated = vec![0.0; ff.len()];
        gelu_f32(&ff, &mut activated)?;
        let mut ff_out = vec![0.0; q.len()];
        affine_f32(
            &activated,
            q_rows,
            dim * 2,
            self.tensor(&format!("{prefix}.linear2.weight"))?,
            Some(self.tensor(&format!("{prefix}.linear2.bias"))?),
            dim,
            &mut ff_out,
        )?;
        for (value, addition) in residual.iter_mut().zip(ff_out) {
            *value += addition;
        }
        Ok(residual)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    fn qass(
        &self,
        prefix: &str,
        input: &[f32],
        rows: usize,
        source: usize,
        heads: usize,
        head_dim: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let name = |part: &str| format!("{prefix}.ssmax_layer.{part}");
        let weights = QassMaxWeights {
            base_first_weight: self.tensor(&name("base_mlp.0.weight"))?,
            base_first_bias: self.tensor(&name("base_mlp.0.bias"))?,
            base_second_weight: self.tensor(&name("base_mlp.2.weight"))?,
            base_second_bias: self.tensor(&name("base_mlp.2.bias"))?,
            query_first_weight: self.tensor(&name("query_mlp.0.weight"))?,
            query_first_bias: self.tensor(&name("query_mlp.0.bias"))?,
            query_second_weight: self.tensor(&name("query_mlp.2.weight"))?,
            query_second_bias: self.tensor(&name("query_mlp.2.bias"))?,
        };
        let mut base_scale = vec![0.0; checked_product(&[heads, head_dim], "QASSMax scale")?];
        let mut query_hidden = vec![0.0; checked_product(&[heads, rows, 64], "QASSMax query hidden")?];
        qassmax_f32(
            input,
            heads,
            rows,
            head_dim,
            source,
            64,
            weights,
            &mut [0.0; 64],
            &mut base_scale,
            &mut query_hidden,
            output,
        )
    }

    fn tensor(&self, name: &str) -> Result<&[f32]> {
        Ok(self.model.tensor_f32(name)?.values())
    }

    /// Outputs per decoded row: the classifier's class capacity or the
    /// regressor's quantile count.
    fn decoder_width(&self) -> Result<usize> {
        Ok(self.tensor("icl_predictor.decoder.2.bias")?.len())
    }

    /// Embed `[N]` training targets with `{prefix}.y_encoder` into `[N, width]`:
    /// one-hot classes for the classifier, the scalar target for the regressor.
    fn target_embeddings(
        &self,
        prefix: &str,
        targets: &[f32],
        width: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let weight = self.tensor(&format!("{prefix}.y_encoder.weight"))?;
        let bias = self.tensor(&format!("{prefix}.y_encoder.bias"))?;
        match self.model.task() {
            TaskKind::Classifier => {
                one_hot_linear_f32(&label_indices(targets)?, weight, bias, 10, width, output)
            }
            TaskKind::Regressor => {
                for (row, &target) in output.chunks_exact_mut(width).zip(targets) {
                    for ((value, &weight), &bias) in row.iter_mut().zip(weight).zip(bias) {
                        *value = target * weight + bias;
                    }
                }
                Ok(())
            }
        }
    }

    /// Check `[B, N]` training targets and return the class count: 2..=10
    /// contiguous labels for the classifier, 0 for the regressor's finite targets.
    fn validate_targets(&self, targets: &[f32], b: usize, n: usize) -> Result<usize> {
        match self.model.task() {
            TaskKind::Classifier => validate_labels(targets, b, n),
            TaskKind::Regressor if targets.iter().all(|target| target.is_finite()) => Ok(0),
            TaskKind::Regressor => Err(Error::InvalidArgument("regression targets must be finite".into())),
        }
    }
}

impl StaticExecutor {
    /// Write this context to `path` atomically, for reuse by [`Executor::load_context`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] if encoding or the atomic write fails.
    pub fn save_context(&self, path: &std::path::Path) -> Result<()> {
        self.context.write(path)
    }

    /// The context's shape, permutations, and identities.
    pub(crate) fn context_metadata(&self) -> &StaticContextMetadata {
        self.context.metadata()
    }

    /// The length of [`Self::encode_context`]'s output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] if a validated size cannot be encoded.
    pub fn context_encoded_len(&self) -> Result<usize> {
        self.context.encoded_len()
    }

    /// Encode this context into `sink` as [`Self::save_context`] writes it,
    /// for [`Executor::load_context_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] if encoding or writing fails.
    pub fn encode_context(&self, sink: &mut impl std::io::Write) -> Result<()> {
        self.context.encode(sink)
    }

    /// Pair a context with an executor whose model weights it shares.
    fn with_executor(
        model: Executor,
        context: StaticContextArtifact,
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
    ) -> Result<Self> {
        let executor = Self::prepared(model, context, threads, limits)?;
        #[cfg(feature = "cuda")]
        let executor = executor.with_cuda(optimize, limits, None)?;
        #[cfg(not(feature = "cuda"))]
        let _ = optimize;
        Ok(executor)
    }

    /// [`Self::with_executor`] up to the CUDA predictor.
    fn prepared(
        model: Executor,
        context: StaticContextArtifact,
        threads: Option<usize>,
        limits: MemoryLimits,
    ) -> Result<Self> {
        let threads = threads.map_or_else(executor_threads, |threads| threads.max(1));
        validate_static_context(&model.model, &context)?;
        let input_weight = model.tensor("col_embedder.in_linear.weight")?;
        let mut column_input_weight = vec![0.0; input_weight.len()];
        for output in 0..COLUMN_DIM {
            for input in 0..3 {
                column_input_weight[input * COLUMN_DIM + output] = input_weight[output * 3 + input];
            }
        }
        let qass_base_scales = model.icl_base_scales(context.metadata().training_rows)?;
        let affine_pool = WorkerPool::new(threads)?;
        Self {
            model,
            context,
            affine_pool: Arc::new(affine_pool),
            column_input_weight: Arc::new(column_input_weight),
            qass_base_scales: Arc::new(qass_base_scales),
            isa: crate::kernels::cpu::Isa::detect(),
            device: Device::Cpu,
        }
        .with_gpu(limits)
    }

    /// Where predictions run: `"metal"`, `"cuda"` with the ICL cache on the
    /// GPU, `"cuda-streamed"` with it streamed from host memory on every pass,
    /// or `"cpu"`.
    #[must_use]
    pub fn device(&self) -> &'static str {
        self.device.name()
    }

    /// Query rows a prediction should bring at once: a streamed pass's, so
    /// each copy of the ICL cache serves as many rows as fit, else `None`.
    #[must_use]
    pub fn batch_rows(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(predictor) = &self.device
            && predictor.streamed()
        {
            return Some(predictor.pass_rows());
        }
        None
    }

    /// Upload the context for CUDA prediction at `optimize` when a CUDA
    /// device is available and the context stores its ICL caches as that
    /// level's GPU reads them: FP32 for `safe`, FP16 otherwise. `built`, the
    /// ICL keys and values the context's build left on the device, replaces
    /// their upload. Other contexts stay on the CPU.
    #[cfg(feature = "cuda")]
    fn with_cuda(
        mut self,
        optimize: Optimize,
        limits: MemoryLimits,
        built: Option<crate::models::tabicl_v2::cuda::BuiltCache>,
    ) -> Result<Self> {
        let precision = crate::models::tabicl_v2::cuda::Precision::for_optimize(optimize);
        let Some(weights) = self.model.cuda_weights()? else {
            return Ok(self);
        };
        let Some(icl) = ContextIcl::new(&self.context, precision) else {
            return Ok(self);
        };
        let column = (0..COLUMN_BLOCKS)
            .map(|block| {
                let tensor = |kind: &str| {
                    self.context
                        .tensor(&format!("column.{block}.{kind}"))?
                        .fp32_values()
                };
                Some((tensor("key")?, tensor("value")?))
            })
            .collect::<Option<Vec<_>>>();
        let Some(column) = column else {
            return Ok(self);
        };
        let metadata = self.context.metadata();
        let budget = weights.budget(limits.device.map(|limit| limit.bytes))?;
        let predictor = crate::models::tabicl_v2::cuda::Predictor::new(
            weights,
            precision,
            &crate::models::tabicl_v2::cuda::ContextTensors {
                shape: crate::models::tabicl_v2::cuda::Shape {
                    ensembles: metadata.ensemble_count,
                    features: metadata.feature_count,
                    rows: metadata.training_rows,
                    outputs: self.model.decoder_width()?,
                },
                column,
                icl: Arc::new(icl),
                base_scales: &self.qass_base_scales,
                rope_frequencies: self.model.tensor("row_interactor.tf_row.rope.freqs")?,
                cls_tokens: self.model.tensor("row_interactor.cls_tokens")?,
                budget,
            },
            built,
        )?;
        // Without room even to stream the ICL cache, the CPU predicts.
        if let Some(predictor) = predictor {
            self.device = Device::Cuda(Arc::new(predictor));
        }
        Ok(self)
    }

    /// Build the context's GPU predict graph when Metal is available, with
    /// queries split to keep its buckets within the device's share of memory
    /// under `limits`, as for builds. Contexts with mixed ICL storage stay on
    /// the CPU.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn with_gpu(mut self, limits: MemoryLimits) -> Result<Self> {
        let model = &self.model;
        let weights = |name: &str| model.tensor(name);
        let Some(gpu_weights) = model.gpu_weights() else {
            return Ok(self);
        };
        let column = |kind: &str| -> Option<Vec<&[f32]>> {
            (0..COLUMN_BLOCKS)
                .map(|block| {
                    self.context
                        .tensor(&format!("column.{block}.{kind}"))?
                        .fp32_values()
                })
                .collect()
        };
        // The ICL caches as raw bytes, in FP32 or FP16 as stored. A context
        // with BF16 values (and FP32 keys) stays on the CPU.
        let icl = |kind: &str| -> Option<(Vec<&[u8]>, crate::format::static_context::TensorStorage)> {
            use crate::format::static_context::TensorStorage;
            let mut storage = None;
            let blocks = (0..ICL_BLOCKS)
                .map(|block| {
                    let (bytes, block_storage) =
                        match self.context.tensor(&format!("icl.{block}.{kind}"))?.data {
                            StaticContextTensorData::Fp32(values) => {
                                (bytemuck::cast_slice(values), TensorStorage::Fp32)
                            }
                            StaticContextTensorData::F16(values) => {
                                (bytemuck::cast_slice(values), TensorStorage::F16)
                            }
                            StaticContextTensorData::Bf16(_) => return None,
                        };
                    (*storage.get_or_insert(block_storage) == block_storage).then_some(bytes)
                })
                .collect::<Option<Vec<_>>>()?;
            Some((blocks, storage?))
        };
        let (
            Some((keys, icl_storage)),
            Some((values, value_storage)),
            Some(column_keys),
            Some(column_values),
        ) = (icl("key"), icl("value"), column("key"), column("value"))
        else {
            return Ok(self);
        };
        if icl_storage != value_storage {
            return Ok(self);
        }
        let metadata = self.context.metadata();
        let predict = crate::models::tabicl_v2::metal::PredictGraph::new(
            &gpu_weights,
            &weights,
            crate::models::tabicl_v2::metal::Shape {
                ensembles: metadata.ensemble_count,
                features: metadata.feature_count,
                training_rows: metadata.training_rows,
            },
            &column_keys,
            &column_values,
            (&keys, &values, icl_storage),
            &self.qass_base_scales,
            self.context.page_aligned_storage().is_some(),
        )?;
        let (budget, _) = crate::memory::device_share(shared_gpu_pools(gpu_weights.gpu(), limits));
        self.device = Device::Metal(Arc::new(crate::models::tabicl_v2::metal::GpuContext {
            split: predict.query_split(budget),
            predict,
            _storage: self.context.page_aligned_storage().cloned(),
        }));
        Ok(self)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "same signature as the Metal version, which can fail"
    )]
    fn with_gpu(self, _limits: MemoryLimits) -> Result<Self> {
        Ok(self)
    }

    /// Run every stage for every estimator on the GPU in one graph run. The
    /// CPU only groups each feature with its neighbors. Returns false, with nothing written,
    /// when a grouped triple is the all-missing marker that only the CPU path
    /// handles.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn predict_logits_gpu(
        &self,
        gpu: &crate::models::tabicl_v2::metal::GpuContext,
        request: QueryRequest<'_>,
        values: &mut [f32],
    ) -> Result<bool> {
        let metadata = self.context.metadata();
        let (q, h, classes) = (request.query_rows(), metadata.feature_count, self.outputs()?);
        let x = request.x();
        let Some(logits) = gpu.predict.run(q, |input, stride| {
            input
                .chunks_exact_mut(h * stride * 3)
                .enumerate()
                .all(|(ensemble, input)| {
                    crate::models::tabicl_v2::metal::group_features_into(
                        &x[ensemble * q * h..][..q * h],
                        (q, h),
                        stride,
                        input,
                    )
                })
        })?
        else {
            return Ok(false);
        };
        let width = self.model.decoder_width()?;
        for (output, logits) in values.chunks_exact_mut(classes).zip(logits.chunks_exact(width)) {
            output.copy_from_slice(&logits[..classes]);
        }
        Ok(true)
    }

    /// Execute query-only FP32 inference for preprocessed `[B, Q, H]` values.
    ///
    /// # Errors
    ///
    /// Returns an error when the query shape does not match the static context
    /// or a checked kernel cannot execute the request.
    pub fn predict_logits(&self, request: QueryRequest<'_>) -> Result<DecoderOutputs> {
        let metadata = self.context.metadata();
        if request.ensemble_count() != metadata.ensemble_count
            || request.feature_count() != metadata.feature_count
        {
            return Err(Error::InvalidArgument(format!(
                "query shape [B, Q, H]=[{}, {}, {}] does not match static context B={} and H={}",
                request.ensemble_count(),
                request.query_rows(),
                request.feature_count(),
                metadata.ensemble_count,
                metadata.feature_count
            )));
        }
        let classes = self.outputs()?;
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = &self.device {
            let (b, q, h) = (
                metadata.ensemble_count,
                request.query_rows(),
                metadata.feature_count,
            );
            let prediction = cuda.predict(request.x(), q)?;
            let width = self.model.decoder_width()?;
            let mut values: Vec<f32> = prediction
                .logits
                .chunks_exact(width)
                .flat_map(|row| row[..classes].iter().copied())
                .collect();
            // Rows with the all-missing marker run on the CPU path.
            if !prediction.cpu_rows.is_empty() {
                let rows = &prediction.cpu_rows;
                let mut x = Vec::with_capacity(b * rows.len() * h);
                for ensemble in 0..b {
                    for &row in rows {
                        x.extend_from_slice(&request.x()[(ensemble * q + row) * h..][..h]);
                    }
                }
                let cpu = self.predict_cpu(QueryRequest::new(b, rows.len(), h, &x)?, classes)?;
                for (ensemble, part) in cpu.values.chunks_exact(rows.len() * classes).enumerate() {
                    for (&row, value) in rows.iter().zip(part.chunks_exact(classes)) {
                        values[(ensemble * q + row) * classes..][..classes].copy_from_slice(value);
                    }
                }
            }
            return Ok(DecoderOutputs {
                shape: [b, q, classes],
                values,
            });
        }
        self.predict_cpu(request, classes)
    }

    /// [`Self::predict_logits`] on the CPU, or on Metal where available.
    fn predict_cpu(&self, request: QueryRequest<'_>, classes: usize) -> Result<DecoderOutputs> {
        let (b, q, h) = (
            request.ensemble_count(),
            request.query_rows(),
            request.feature_count(),
        );
        // Rows are independent, so a large query runs in pieces that fit the
        // activation budget and the GPU's. Only a single row that does not fit
        // is an error.
        let (mut chunk, small) = self.query_split();
        while let Err(error) = self.admit_query(chunk.min(q)) {
            if chunk == 1 {
                return Err(error);
            }
            chunk /= 2;
        }
        let small = small.min(chunk);
        // Whole chunks, then powers of two, until at most `small` rows remain.
        let piece = |left: usize| match left {
            left if left > chunk => chunk,
            left if left <= small => left,
            left => 1 << left.ilog2(),
        };
        if piece(q) == q {
            return self.predict_chunk(request);
        }
        let mut values = vec![0.0; checked_product(&[b, q, classes], "static decoder output")?];
        let mut x = Vec::with_capacity(checked_product(&[b, chunk, h], "query chunk")?);
        let mut start = 0;
        while start < q {
            let len = piece(q - start);
            let logits = self.predict_chunk(query_rows(request, start, len, &mut x)?)?;
            for (ensemble, part) in logits.values.chunks_exact(len * classes).enumerate() {
                values[(ensemble * q + start) * classes..][..len * classes].copy_from_slice(part);
            }
            start += len;
            crate::progress::report(Stage::Predict, start, q)?;
        }
        Ok(DecoderOutputs {
            shape: [b, q, classes],
            values,
        })
    }

    /// How queries split into runs, `(chunk, small)`: the GPU's split within
    /// its budget, or chunks of [`QUERY_CHUNK_ROWS`] on the CPU.
    #[cfg_attr(
        not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
        expect(clippy::unused_self, reason = "only the Metal build reads `self`")
    )]
    fn query_split(&self) -> (usize, usize) {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Device::Metal(gpu) = &self.device {
            return gpu.split;
        }
        (QUERY_CHUNK_ROWS, QUERY_CHUNK_ROWS)
    }

    /// Decoder outputs kept per query row: the class count, or every quantile
    /// for a regression context.
    fn outputs(&self) -> Result<usize> {
        match self.context.metadata().class_count {
            0 => self.model.decoder_width(),
            classes => Ok(classes),
        }
    }

    fn plan(&self) -> ExecutionPlan {
        let ensembles = self.context.metadata().ensemble_count;
        static_execution_plan(ensembles, self.affine_pool.current_num_threads())
    }

    fn admit_query(&self, rows: usize) -> Result<()> {
        let metadata = self.context.metadata();
        let plan = self.plan();
        admit_static_request(
            metadata.ensemble_count,
            rows,
            metadata.feature_count,
            self.outputs()?,
            if plan.flat_pool_ensembles {
                metadata.ensemble_count
            } else {
                plan.ensemble_workers
            },
        )
    }

    /// Run one query chunk that [`Self::admit_query`] accepted.
    fn predict_chunk(&self, request: QueryRequest<'_>) -> Result<DecoderOutputs> {
        let metadata = self.context.metadata();
        let plan = self.plan();
        let shape = [metadata.ensemble_count, request.query_rows(), self.outputs()?];
        let mut values = vec![0.0; checked_product(&shape, "static decoder output")?];
        let ensemble_output_elements =
            checked_product(&[request.query_rows(), shape[2]], "static ensemble output")?;
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Device::Metal(gpu) = &self.device
            && self.predict_logits_gpu(gpu, request, &mut values)?
        {
            return Ok(DecoderOutputs { shape, values });
        }
        if plan.ensemble_workers == 1 {
            for (ensemble, output) in values.chunks_mut(ensemble_output_elements).enumerate() {
                self.predict_estimator(request, ensemble, plan.inner_workers, output)?;
            }
        } else if plan.flat_pool_ensembles {
            self.affine_pool.install(|| {
                values
                    .par_chunks_mut(ensemble_output_elements)
                    .enumerate()
                    .try_for_each(|(ensemble, output)| {
                        self.predict_estimator(request, ensemble, plan.inner_workers, output)
                    })
            })?;
        } else {
            let ensembles_per_worker = metadata.ensemble_count.div_ceil(plan.ensemble_workers);
            let output_elements_per_worker = checked_product(
                &[ensembles_per_worker, ensemble_output_elements],
                "static worker output",
            )?;
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(plan.ensemble_workers);
                for (worker, outputs) in values.chunks_mut(output_elements_per_worker).enumerate() {
                    handles.push(scope.spawn(move || {
                        let first_ensemble = worker * ensembles_per_worker;
                        for (offset, output) in outputs.chunks_mut(ensemble_output_elements).enumerate() {
                            self.predict_estimator(
                                request,
                                first_ensemble + offset,
                                plan.inner_workers,
                                output,
                            )?;
                        }
                        Ok(())
                    }));
                }
                for handle in handles {
                    handle
                        .join()
                        .map_err(|_| Error::InvalidArgument("static ensemble worker panicked".into()))??;
                }
                Ok(())
            })?;
        }
        Ok(DecoderOutputs { shape, values })
    }

    fn predict_estimator(
        &self,
        request: QueryRequest<'_>,
        ensemble: usize,
        workers: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let rows = self.cached_frontend(request, ensemble, workers)?;
        let logits = self.cached_icl_logits(rows, ensemble, request.query_rows(), workers)?;
        let (width, outputs) = (self.model.decoder_width()?, self.outputs()?);
        for row in 0..request.query_rows() {
            output[row * outputs..][..outputs].copy_from_slice(&logits[row * width..][..outputs]);
        }
        Ok(())
    }

    fn cached_frontend(
        &self,
        request: QueryRequest<'_>,
        ensemble: usize,
        workers: usize,
    ) -> Result<Vec<f32>> {
        let metadata = self.context.metadata();
        let offset = checked_product(
            &[ensemble, request.query_rows(), metadata.feature_count],
            "query offset",
        )?;
        let x = &request.x()[offset..]
            [..checked_product(&[request.query_rows(), metadata.feature_count], "query rows")?];
        let columns = self.cached_column_embeddings(x, ensemble, request.query_rows(), workers)?;
        self.row_embeddings(
            &columns,
            request.query_rows(),
            metadata.feature_count + RESERVED,
            workers,
        )
    }

    /// Predict each requested regression output for `[B, Q, H]` query rows,
    /// in the standardized target units the model works in. Rows run in
    /// chunks, so the `[B, rows, 999]` quantiles stay small.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::predict_logits`], and an error when
    /// the context is not a regression fit.
    pub fn predict_regression(
        &self,
        request: QueryRequest<'_>,
        outputs: &[RegressionOutput],
    ) -> Result<Vec<(Vec<f32>, usize)>> {
        if self.context.metadata().class_count != 0 {
            return Err(Error::InvalidArgument(
                "the context is a classification fit".into(),
            ));
        }
        let q = request.query_rows();
        let mut results = vec![(Vec::new(), 0); outputs.len()];
        let mut x = Vec::new();
        for start in (0..q).step_by(QUERY_CHUNK_ROWS) {
            let len = QUERY_CHUNK_ROWS.min(q - start);
            let decoded = self.predict_logits(query_rows(request, start, len, &mut x)?)?;
            for ((values, columns), (chunk, chunk_columns)) in
                results.iter_mut().zip(summarize(&decoded, outputs)?)
            {
                values.extend(chunk);
                *columns = chunk_columns;
            }
        }
        Ok(results)
    }

    /// Execute cached inference, undo class permutations, and aggregate the
    /// ensemble with temperature-scaled softmax.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::predict_logits`], and an error for an
    /// invalid softmax temperature.
    pub fn predict_probabilities(
        &self,
        request: QueryRequest<'_>,
        aggregation: ClassifierAggregation,
    ) -> Result<ClassifierProbabilities> {
        if self.context.metadata().class_count == 0 {
            return Err(Error::InvalidArgument("the context is a regression fit".into()));
        }
        let logits = self.predict_logits(request)?;
        aggregate_classifier_probabilities(&logits, &self.context.metadata().class_permutations, aggregation)
    }

    fn cached_column_embeddings(
        &self,
        x: &[f32],
        ensemble: usize,
        q: usize,
        workers: usize,
    ) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Column);
        let h = self.context.metadata().feature_count;
        let streams = h + RESERVED;
        let mut output = vec![0.0; checked_product(&[q, streams, COLUMN_DIM], "cached column embeddings")?];
        let in_weight = self.column_input_weight.as_slice();
        let in_bias = self.tensor("col_embedder.in_linear.bias")?;
        let workers = workers.min(h);
        let chunk = h.div_ceil(workers);
        let results = if workers == 1 {
            let mut results = Vec::with_capacity(h);
            for feature in 0..h {
                results.push((
                    feature + RESERVED,
                    self.cached_column_stream(x, ensemble, q, h, feature, in_weight, in_bias, workers)?,
                ));
            }
            vec![results]
        } else {
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(workers);
                for start in (0..h).step_by(chunk) {
                    let end = (start + chunk).min(h);
                    handles.push(scope.spawn(move || -> Result<Vec<(usize, Vec<f32>)>> {
                        let mut results = Vec::with_capacity(end - start);
                        for feature in start..end {
                            results.push((
                                feature + RESERVED,
                                self.cached_column_stream(
                                    x, ensemble, q, h, feature, in_weight, in_bias, workers,
                                )?,
                            ));
                        }
                        Ok(results)
                    }));
                }
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("cached column worker must not panic"))
                    .collect::<Result<Vec<_>>>()
            })?
        };
        for (stream, embedded) in results.into_iter().flatten() {
            for row in 0..q {
                output[(row * streams + stream) * COLUMN_DIM..][..COLUMN_DIM]
                    .copy_from_slice(&embedded[row * COLUMN_DIM..][..COLUMN_DIM]);
            }
        }
        Ok(output)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    #[expect(clippy::float_cmp, reason = "-100.0 is an exact missing-value marker")]
    fn cached_column_stream(
        &self,
        x: &[f32],
        ensemble: usize,
        q: usize,
        h: usize,
        feature: usize,
        in_weight: &[f32],
        in_bias: &[f32],
        workers: usize,
    ) -> Result<Vec<f32>> {
        let mut source = vec![0.0; checked_product(&[q, 3], "cached column grouped features")?];
        for row in 0..q {
            for (part, offset) in [1_usize, 2, 4].iter().enumerate() {
                source[row * 3 + part] = x[row * h + (feature + offset) % h];
            }
        }
        let mut embedded = vec![0.0; q * COLUMN_DIM];
        affine_input3_transposed_f32(&source, q, in_weight, in_bias, COLUMN_DIM, &mut embedded)?;
        for row in 0..q {
            if source[row * 3..][..3].iter().all(|value| *value == -100.0) {
                embedded[row * COLUMN_DIM..][..COLUMN_DIM].fill(-100.0);
            }
        }
        for block in 0..COLUMN_BLOCKS {
            let prefix = format!("col_embedder.tf_col.blocks.{block}.multihead_attn2");
            embedded = self.cached_block(
                &embedded,
                q,
                COLUMN_DIM,
                &prefix,
                &format!("{prefix}.attn"),
                None,
                StaticContextTensorData::Fp32(self.column_cache(
                    block,
                    "key",
                    ensemble,
                    feature + RESERVED,
                )?),
                StaticContextTensorData::Fp32(self.column_cache(
                    block,
                    "value",
                    ensemble,
                    feature + RESERVED,
                )?),
                INDUCING,
                16,
                workers,
            )?;
        }
        Ok(embedded)
    }

    fn cached_icl_logits(
        &self,
        mut rows: Vec<f32>,
        ensemble: usize,
        q: usize,
        workers: usize,
    ) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Icl);
        for block in 0..ICL_BLOCKS {
            rows = self.cached_icl_block(&rows, ensemble, q, block, workers)?;
        }
        self.cached_icl_decoder(&rows, q, workers)
    }

    fn cached_icl_block(
        &self,
        rows: &[f32],
        ensemble: usize,
        q: usize,
        block: usize,
        workers: usize,
    ) -> Result<Vec<f32>> {
        self.cached_block(
            rows,
            q,
            ICL_DIM,
            &format!("icl_predictor.tf_icl.blocks.{block}"),
            &format!("icl_predictor.tf_icl.blocks.{block}.attn"),
            Some(&self.qass_base_scales[block]),
            self.cache("icl", block, "key", ensemble)?,
            self.cache("icl", block, "value", ensemble)?,
            self.context.metadata().training_rows,
            64,
            workers,
        )
    }

    fn cached_icl_decoder(&self, rows: &[f32], q: usize, workers: usize) -> Result<Vec<f32>> {
        let mut normalized = vec![0.0; q * ICL_DIM];
        layer_norm_f32(
            rows,
            q,
            ICL_DIM,
            self.tensor("icl_predictor.ln.weight")?,
            Some(self.tensor("icl_predictor.ln.bias")?),
            &mut normalized,
        )?;
        let mut hidden = vec![0.0; q * 1024];
        self.parallel_affine(
            &normalized,
            q,
            ICL_DIM,
            self.tensor("icl_predictor.decoder.0.weight")?,
            Some(self.tensor("icl_predictor.decoder.0.bias")?),
            1024,
            &mut hidden,
            workers,
        )?;
        let mut activated = vec![0.0; hidden.len()];
        self.parallel_gelu(&hidden, &mut activated, workers)?;
        let width = self.model.decoder_width()?;
        let mut logits = vec![0.0; q * width];
        self.parallel_affine(
            &activated,
            q,
            1024,
            self.tensor("icl_predictor.decoder.2.weight")?,
            Some(self.tensor("icl_predictor.decoder.2.bias")?),
            width,
            &mut logits,
            workers,
        )?;
        Ok(logits)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    #[expect(clippy::too_many_lines, reason = "one transformer block, step by step")]
    fn cached_block(
        &self,
        q: &[f32],
        q_rows: usize,
        dim: usize,
        prefix: &str,
        attn_prefix: &str,
        qass_base_scale: Option<&[f32]>,
        key: StaticContextTensorData<'_>,
        value: StaticContextTensorData<'_>,
        source_rows: usize,
        head_dim: usize,
        workers: usize,
    ) -> Result<Vec<f32>> {
        let weight = self.tensor(&format!("{attn_prefix}.in_proj_weight"))?;
        let bias = self.tensor(&format!("{attn_prefix}.in_proj_bias"))?;
        let mut q_proj = vec![0.0; q.len()];
        let mut q_norm = vec![0.0; q.len()];
        layer_norm_f32(
            q,
            q_rows,
            dim,
            self.tensor(&format!("{prefix}.norm1.weight"))?,
            Some(self.tensor(&format!("{prefix}.norm1.bias"))?),
            &mut q_norm,
        )?;
        self.parallel_affine(
            &q_norm,
            q_rows,
            dim,
            &weight[..dim * dim],
            Some(&bias[..dim]),
            dim,
            &mut q_proj,
            workers,
        )?;
        let mut q_heads = tokens_to_heads(&q_proj, q_rows, dim, HEADS);
        if let Some(base_scale) = qass_base_scale {
            let original = q_heads.clone();
            self.qass(
                attn_prefix,
                &original,
                q_rows,
                HEADS,
                head_dim,
                base_scale,
                &mut q_heads,
                workers,
            )?;
        }
        let mut attended = vec![0.0; q_heads.len()];
        if qass_base_scale.is_some() {
            match (key, value) {
                (StaticContextTensorData::Fp32(key), StaticContextTensorData::Fp32(value)) => {
                    parallel_cached_attention(
                        self.isa,
                        &q_heads,
                        q_rows,
                        key,
                        value,
                        source_rows,
                        head_dim,
                        &mut attended,
                        workers,
                    )?;
                }
                (StaticContextTensorData::Fp32(key), StaticContextTensorData::Bf16(value)) => {
                    parallel_cached_attention_f32_key_bf16_value(
                        self.isa,
                        &q_heads,
                        q_rows,
                        key,
                        value,
                        source_rows,
                        head_dim,
                        &mut attended,
                        workers,
                    )?;
                }
                // Only queries the GPU hands back reach this: widen the
                // block's FP16 caches and run the FP32 kernel.
                (StaticContextTensorData::F16(key), StaticContextTensorData::F16(value)) => {
                    let widen =
                        |values: &[u16]| values.iter().map(|&bits| f16_to_f32(bits)).collect::<Vec<_>>();
                    parallel_cached_attention(
                        self.isa,
                        &q_heads,
                        q_rows,
                        &widen(key),
                        &widen(value),
                        source_rows,
                        head_dim,
                        &mut attended,
                        workers,
                    )?;
                }
                _ => {
                    return Err(Error::Artifact(
                        "static context attention K/V dtypes do not match".into(),
                    ));
                }
            }
        } else {
            let (StaticContextTensorData::Fp32(key), StaticContextTensorData::Fp32(value)) = (key, value)
            else {
                return Err(Error::Artifact(
                    "column static-context caches must use FP32".into(),
                ));
            };
            if column_source_major_attention_enabled(self.isa) {
                let query_block_rows = ATTENTION_QUERY_BLOCK_ROWS;
                let scratch_query_rows = q_rows.min(query_block_rows);
                let scratch_source_rows = source_rows.min(ATTENTION_KEY_TILE_ROWS);
                let mut scores = vec![0.0; scratch_query_rows * scratch_source_rows];
                let mut maximum = vec![0.0; HEADS * q_rows];
                let mut denominator = vec![0.0; HEADS * q_rows];
                source_major_online_attention_with_backend_f32(
                    &q_heads,
                    q_rows,
                    key,
                    value,
                    source_rows,
                    HEADS,
                    head_dim,
                    query_block_rows,
                    ATTENTION_KEY_TILE_ROWS,
                    self.isa,
                    &mut OnlineAttentionScratch {
                        scores: &mut scores,
                        maximum: &mut maximum,
                        denominator: &mut denominator,
                    },
                    &mut attended,
                )?;
            } else {
                let mut scores = vec![
                    0.0;
                    attention_scratch_scores(
                        q_rows.min(ATTENTION_QUERY_BLOCK_ROWS),
                        source_rows.min(ATTENTION_KEY_TILE_ROWS)
                    )?
                ];
                let mut maximum = vec![0.0; attention_scratch_state(q_rows.min(ATTENTION_QUERY_BLOCK_ROWS))?];
                let mut denominator =
                    vec![0.0; attention_scratch_state(q_rows.min(ATTENTION_QUERY_BLOCK_ROWS))?];
                tiled_online_scaled_dot_product_attention_f32(
                    &q_heads,
                    q_rows,
                    key,
                    value,
                    source_rows,
                    HEADS,
                    head_dim,
                    ATTENTION_QUERY_BLOCK_ROWS,
                    ATTENTION_KEY_TILE_ROWS,
                    &mut OnlineAttentionScratch {
                        scores: &mut scores,
                        maximum: &mut maximum,
                        denominator: &mut denominator,
                    },
                    &mut attended,
                )?;
            }
        }
        let merged = heads_to_tokens(&attended, q_rows, dim, HEADS);
        let mut projected = vec![0.0; q.len()];
        self.parallel_affine(
            &merged,
            q_rows,
            dim,
            self.tensor(&format!("{attn_prefix}.out_proj.weight"))?,
            Some(self.tensor(&format!("{attn_prefix}.out_proj.bias"))?),
            dim,
            &mut projected,
            workers,
        )?;
        let mut residual = q
            .iter()
            .zip(projected)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let mut ff = vec![0.0; q_rows * dim * 2];
        let ff_name = format!("{prefix}.linear1.weight");
        let mut ff_norm = vec![0.0; residual.len()];
        layer_norm_f32(
            &residual,
            q_rows,
            dim,
            self.tensor(&format!("{prefix}.norm2.weight"))?,
            Some(self.tensor(&format!("{prefix}.norm2.bias"))?),
            &mut ff_norm,
        )?;
        self.parallel_affine(
            &ff_norm,
            q_rows,
            dim,
            self.tensor(&ff_name)?,
            Some(self.tensor(&format!("{prefix}.linear1.bias"))?),
            dim * 2,
            &mut ff,
            workers,
        )?;
        let mut activated = vec![0.0; ff.len()];
        self.parallel_gelu(&ff, &mut activated, workers)?;
        let mut ff_out = vec![0.0; q.len()];
        self.parallel_affine(
            &activated,
            q_rows,
            dim * 2,
            self.tensor(&format!("{prefix}.linear2.weight"))?,
            Some(self.tensor(&format!("{prefix}.linear2.bias"))?),
            dim,
            &mut ff_out,
            workers,
        )?;
        for (value, addition) in residual.iter_mut().zip(ff_out) {
            *value += addition;
        }
        Ok(residual)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    fn parallel_affine(
        &self,
        input: &[f32],
        rows: usize,
        input_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output_dim: usize,
        output: &mut [f32],
        workers: usize,
    ) -> Result<()> {
        let workers = workers.min(self.affine_pool.current_num_threads()).min(rows);
        // Apple Silicon always returns here: fewer than 16 rows run on one
        // thread, and more run as one Accelerate `sgemm`.
        if workers <= 1 || rows < 16 || crate::kernels::cpu::affine_uses_sgemm(rows) {
            return affine_f32(input, rows, input_dim, weight, bias, output_dim, output);
        }
        let rows_per_job = rows.div_ceil(workers);
        self.affine_pool.install(|| {
            input
                .par_chunks(input_dim * rows_per_job)
                .zip(output.par_chunks_mut(output_dim * rows_per_job))
                .try_for_each(|(input, output)| {
                    affine_f32(
                        input,
                        input.len() / input_dim,
                        input_dim,
                        weight,
                        bias,
                        output_dim,
                        output,
                    )
                })
        })
    }

    fn parallel_gelu(&self, input: &[f32], output: &mut [f32], workers: usize) -> Result<()> {
        let workers = workers.min(self.affine_pool.current_num_threads());
        if workers <= 1 || input.len() < 16_384 {
            return gelu_f32(input, output);
        }
        let chunk = input.len().div_ceil(workers);
        self.affine_pool.install(|| {
            input
                .par_chunks(chunk)
                .zip(output.par_chunks_mut(chunk))
                .try_for_each(|(input, output)| gelu_f32(input, output))
        })
    }

    fn row_embeddings(
        &self,
        columns: &[f32],
        rows: usize,
        streams: usize,
        workers: usize,
    ) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Row);
        let cls = self.tensor("row_interactor.cls_tokens")?;
        let workers = workers.min(rows);
        if workers == 1 {
            let mut output = Vec::with_capacity(rows * ICL_DIM);
            for row in 0..rows {
                output.extend_from_slice(&self.row_embedding(columns, cls, row, streams)?);
            }
            return Ok(output);
        }
        let chunk = rows.div_ceil(workers);
        let chunks = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for start in (0..rows).step_by(chunk) {
                let end = (start + chunk).min(rows);
                handles.push(scope.spawn(move || -> Result<Vec<f32>> {
                    let mut output = Vec::with_capacity((end - start) * ICL_DIM);
                    for row in start..end {
                        output.extend_from_slice(&self.row_embedding(columns, cls, row, streams)?);
                    }
                    Ok(output)
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("cached row worker must not panic"))
                .collect::<Result<Vec<_>>>()
        })?;
        let mut output = Vec::with_capacity(rows * ICL_DIM);
        for chunk in chunks {
            output.extend_from_slice(&chunk);
        }
        Ok(output)
    }

    fn row_embedding(&self, columns: &[f32], cls: &[f32], row: usize, streams: usize) -> Result<Vec<f32>> {
        let mut sequence = columns[row * streams * COLUMN_DIM..][..streams * COLUMN_DIM].to_vec();
        for token in 0..RESERVED {
            sequence[token * COLUMN_DIM..][..COLUMN_DIM]
                .copy_from_slice(&cls[token * COLUMN_DIM..][..COLUMN_DIM]);
        }
        for block in 0..2 {
            sequence = self.uncached_block(
                &sequence,
                &sequence,
                streams,
                streams,
                COLUMN_DIM,
                &format!("row_interactor.tf_row.blocks.{block}"),
                false,
                true,
            )?;
        }
        let final_cls = self.uncached_block(
            &sequence[..RESERVED * COLUMN_DIM],
            &sequence,
            RESERVED,
            streams,
            COLUMN_DIM,
            "row_interactor.tf_row.blocks.2",
            false,
            true,
        )?;
        let mut normalized = vec![0.0; ICL_DIM];
        layer_norm_f32(
            &final_cls,
            RESERVED,
            COLUMN_DIM,
            self.tensor("row_interactor.out_ln.weight")?,
            Some(self.tensor("row_interactor.out_ln.bias")?),
            &mut normalized,
        )?;
        Ok(normalized)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    fn uncached_block(
        &self,
        q: &[f32],
        source: &[f32],
        q_rows: usize,
        source_rows: usize,
        dim: usize,
        prefix: &str,
        qass: bool,
        rope: bool,
    ) -> Result<Vec<f32>> {
        self.model.block(
            q,
            source,
            source,
            q_rows,
            source_rows,
            dim,
            prefix,
            qass,
            rope.then_some(0),
        )
    }

    fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.model.tensor(name)
    }

    fn cache(
        &self,
        stage: &str,
        layer: usize,
        kind: &str,
        ensemble: usize,
    ) -> Result<StaticContextTensorData<'_>> {
        let name = format!("{stage}.{layer}.{kind}");
        let tensor = self
            .context
            .tensor(&name)
            .ok_or_else(|| Error::Artifact(format!("static context cache tensor '{name}' is missing")))?;
        let per_ensemble = tensor.data.len() / self.context.metadata().ensemble_count;
        Ok(tensor.data.slice(ensemble * per_ensemble, per_ensemble))
    }

    fn column_cache(&self, layer: usize, kind: &str, ensemble: usize, stream: usize) -> Result<&[f32]> {
        let name = format!("column.{layer}.{kind}");
        let tensor = self
            .context
            .tensor(&name)
            .ok_or_else(|| Error::Artifact(format!("static context cache tensor '{name}' is missing")))?;
        let StaticContextTensorData::Fp32(values) = tensor.data else {
            return Err(Error::Artifact(format!(
                "static context column cache tensor '{name}' must use FP32"
            )));
        };
        let stream_elements = HEADS * INDUCING * 16;
        let ensemble_elements = (self.context.metadata().feature_count + RESERVED) * stream_elements;
        let offset = ensemble * ensemble_elements + stream * stream_elements;
        Ok(&values[offset..][..stream_elements])
    }
}

/// `QASSMax` base scales for `source` keys: `[heads * head_dim]`, the same for
/// every query, so they are computed once per source length.
#[expect(clippy::cast_precision_loss, reason = "ln(source) needs no more precision")]
fn qass_base_scale(model: &Executor, prefix: &str, source: usize, width: usize) -> Result<Vec<f32>> {
    let name = |part: &str| format!("{prefix}.ssmax_layer.{part}");
    let first_weight = model.tensor(&name("base_mlp.0.weight"))?;
    let first_bias = model.tensor(&name("base_mlp.0.bias"))?;
    let second_weight = model.tensor(&name("base_mlp.2.weight"))?;
    let second_bias = model.tensor(&name("base_mlp.2.bias"))?;
    let log_source = (source.max(1) as f32).ln();
    let mut hidden = [0.0; 64];
    for index in 0..64 {
        let value = first_weight[index].mul_add(log_source, first_bias[index]);
        hidden[index] = 0.5 * value * (1.0 + libm::erff(value / core::f32::consts::SQRT_2));
    }
    let mut scale = vec![0.0; width];
    affine_f32(
        &hidden,
        1,
        64,
        second_weight,
        Some(second_bias),
        width,
        &mut scale,
    )?;
    Ok(scale)
}

impl StaticExecutor {
    #[expect(
        clippy::too_many_arguments,
        reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
    )]
    fn qass(
        &self,
        prefix: &str,
        input: &[f32],
        rows: usize,
        heads: usize,
        head_dim: usize,
        base_scale: &[f32],
        output: &mut [f32],
        workers: usize,
    ) -> Result<()> {
        let name = |part: &str| format!("{prefix}.ssmax_layer.{part}");
        let query_first_name = name("query_mlp.0.weight");
        let query_second_name = name("query_mlp.2.weight");
        let query_first_weight = self.tensor(&query_first_name)?;
        let query_first_bias = self.tensor(&name("query_mlp.0.bias"))?;
        let query_second_weight = self.tensor(&query_second_name)?;
        let query_second_bias = self.tensor(&name("query_mlp.2.bias"))?;
        let query_rows = heads * rows;
        let mut query_hidden = vec![0.0; query_rows * 64];
        self.parallel_affine(
            input,
            query_rows,
            head_dim,
            query_first_weight,
            Some(query_first_bias),
            64,
            &mut query_hidden,
            workers,
        )?;
        let mut activated = vec![0.0; query_hidden.len()];
        self.parallel_gelu(&query_hidden, &mut activated, workers)?;
        self.parallel_affine(
            &activated,
            query_rows,
            64,
            query_second_weight,
            Some(query_second_bias),
            head_dim,
            output,
            workers,
        )?;
        for head in 0..heads {
            for row in 0..rows {
                let offset = (head * rows + row) * head_dim;
                for dimension in 0..head_dim {
                    output[offset + dimension] = input[offset + dimension]
                        * base_scale[head * head_dim + dimension]
                        * (1.0 + output[offset + dimension].tanh());
                }
            }
        }
        Ok(())
    }
}

/// IEEE half-precision bits as `f32`, exactly. `f16` is not stable Rust yet.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * (-24.0_f32).exp2(),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mantissa / 1024.0) * (f32::from(exponent) - 15.0).exp2(),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "permutation entries are class indices"
)]
fn aggregate_classifier_probabilities(
    logits: &DecoderOutputs,
    class_permutations: &[u64],
    aggregation: ClassifierAggregation,
) -> Result<ClassifierProbabilities> {
    let temperature = aggregation.softmax_temperature;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::InvalidArgument(
            "softmax temperature must be positive and finite".into(),
        ));
    }
    let [ensembles, rows, classes] = logits.shape;
    let mut values = vec![0.0_f32; rows * classes];
    let mut member = vec![0.0_f32; classes];
    let ensemble_scale =
        1.0 / f32::from(u16::try_from(ensembles).expect("validated ensemble count fits u16"));
    for ensemble in 0..ensembles {
        for row in 0..rows {
            for (class, value) in member.iter_mut().enumerate() {
                let shuffled_class = class_permutations[ensemble * classes + class] as usize;
                *value = logits.values[(ensemble * rows + row) * classes + shuffled_class];
            }
            if !aggregation.average_logits {
                softmax(&mut member, temperature);
            }
            for (total, value) in values[row * classes..][..classes].iter_mut().zip(&member) {
                *total += value * ensemble_scale;
            }
        }
    }
    for row in values.chunks_exact_mut(classes) {
        if aggregation.average_logits {
            softmax(row, temperature);
        }
        // Upstream renormalizes after either averaging mode.
        let sum = row.iter().sum::<f32>();
        for value in row {
            *value /= sum;
        }
    }
    Ok(ClassifierProbabilities {
        shape: [rows, classes],
        values,
    })
}

fn softmax(values: &mut [f32], temperature: f32) {
    let maximum = values
        .iter()
        .map(|value| value / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0_f32;
    for value in &mut *values {
        *value = (*value / temperature - maximum).exp();
        denominator += *value;
    }
    for value in values {
        *value /= denominator;
    }
}

/// Copy rows `start..start + len` of every estimator into `x` as a `[B, len, H]` request.
fn query_rows<'a>(
    request: QueryRequest<'_>,
    start: usize,
    len: usize,
    x: &'a mut Vec<f32>,
) -> Result<QueryRequest<'a>> {
    let (b, q, h) = (
        request.ensemble_count(),
        request.query_rows(),
        request.feature_count(),
    );
    x.clear();
    for ensemble in 0..b {
        x.extend_from_slice(&request.x()[(ensemble * q + start) * h..][..len * h]);
    }
    QueryRequest::new(b, len, h, x)
}

fn validate_static_context(model: &ModelArtifact, context: &StaticContextArtifact) -> Result<()> {
    if !model_hash_matches(
        model.summary().source_sha256.as_bytes(),
        &context.metadata().model_sha256,
    ) {
        return Err(Error::Artifact(
            "static context model SHA-256 does not match model artifact".into(),
        ));
    }
    validate_class_count(model.task(), context.metadata().class_count)?;
    validate_static_cache_contract(context)
}

/// A classifier context has 2 to 10 classes; a regression context has none.
fn validate_class_count(task: TaskKind, classes: usize) -> Result<()> {
    let (supported, expected) = match task {
        TaskKind::Classifier => ((2..=10).contains(&classes), "2..=10"),
        TaskKind::Regressor => (classes == 0, "0"),
    };
    if !supported {
        return Err(Error::Artifact(format!(
            "static context class count {classes} is unsupported; expected {expected}"
        )));
    }
    Ok(())
}

fn model_hash_matches(model_hash: &[u8], context_hash: &[u8; 32]) -> bool {
    model_hash == crate::format::encode_sha256(context_hash).as_bytes()
}

/// Pack FP32 context tensors into page-aligned memory: the tensors `reduced`
/// names in `storage`, FP16 or BF16, the rest in FP32.
fn page_aligned(
    tensors: &[StaticContextTensor],
    reduced: impl Fn(&str) -> bool,
    storage: TensorStorage,
) -> Result<PageAlignedTensors> {
    let storage_of = |name: &str| {
        if reduced(name) {
            storage
        } else {
            TensorStorage::Fp32
        }
    };
    let mut page = PageAlignedTensors::new(
        tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    tensor.dimensions.clone(),
                    storage_of(&tensor.name),
                )
            })
            .collect(),
    )?;
    for (index, tensor) in tensors.iter().enumerate() {
        let storage = storage_of(&tensor.name);
        let holds =
            |value: &f32| value.is_finite() && (storage != TensorStorage::F16 || value.abs() <= 65504.0);
        if !tensor.values.iter().all(holds) {
            return Err(Error::InvalidArgument(format!(
                "static context tensor '{}' has a value {storage:?} cannot hold",
                tensor.name
            )));
        }
        let values = tensor.values.iter().copied();
        let bytes: Vec<u8> = match storage {
            TensorStorage::Fp32 => values.flat_map(f32::to_le_bytes).collect(),
            TensorStorage::Bf16 => values
                .flat_map(|value| crate::kernels::cpu::f32_to_bf16_rne(value).to_le_bytes())
                .collect(),
            TensorStorage::F16 => values
                .flat_map(|value| half::f16::from_f32(value).to_bits().to_le_bytes())
                .collect(),
        };
        page.fill(index, &bytes);
    }
    Ok(page)
}

/// How a context for the CUDA GPU at `precision` stores its ICL keys and
/// values: as the GPU caches them, FP32 for `safe` and FP16 otherwise.
#[cfg(feature = "cuda")]
const fn cuda_icl_storage(precision: crate::models::tabicl_v2::cuda::Precision) -> TensorStorage {
    match precision {
        crate::models::tabicl_v2::cuda::Precision::Fp32 => TensorStorage::Fp32,
        crate::models::tabicl_v2::cuda::Precision::Split
        | crate::models::tabicl_v2::cuda::Precision::Half => TensorStorage::F16,
    }
}

/// A context's page-aligned tensors, as [`context_tensors`] lists them, with
/// the ICL keys and values in `icl`.
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
fn context_page(b: usize, n: usize, h: usize, icl: TensorStorage) -> Result<PageAlignedTensors> {
    PageAlignedTensors::new(
        context_tensors(b, n, h)
            .into_iter()
            .map(|(name, dimensions)| {
                let storage = if name.starts_with("icl.") {
                    icl
                } else {
                    TensorStorage::Fp32
                };
                (name, dimensions, storage)
            })
            .collect(),
    )
}

/// Whether a row of `x` `[_, H]` groups a feature's neighbors into the
/// all-missing marker, which only the CPU column stage handles.
#[cfg(feature = "cuda")]
#[expect(clippy::float_cmp, reason = "The all-missing marker is exactly -100")]
fn has_missing_marker(x: &[f32], h: usize) -> bool {
    x.par_chunks(h).any(|row| {
        (0..h).any(|feature| {
            [1, 2, 4]
                .iter()
                .all(|offset| row[(feature + offset) % h] == -100.0)
        })
    })
}

/// `RoPE` tables of the row stage for `len` sequence positions: cos twice,
/// and `[-sin, sin]`, per 16-dimensional head, as the GPU paths read them.
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
#[expect(clippy::cast_precision_loss, reason = "Positions are small")]
pub(crate) fn rope_tables(len: usize, frequencies: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let head_dim = COLUMN_DIM / HEADS;
    let half = head_dim / 2;
    let (mut cos, mut sin) = (vec![0.0; len * head_dim], vec![0.0; len * head_dim]);
    for row in 0..len {
        for (index, frequency) in frequencies[..half].iter().enumerate() {
            let (s, c) = (row as f32 * frequency).sin_cos();
            let at = row * head_dim + index;
            (cos[at], cos[at + half]) = (c, c);
            (sin[at], sin[at + half]) = (-s, s);
        }
    }
    (cos, sin)
}

/// A context's cache tensors in order: each column block's key and value as
/// `[B, H + 4, 8, 128, 16]`, then each ICL block's as `[B, 8, N, 64]`.
fn context_tensors(b: usize, n: usize, h: usize) -> Vec<(String, Vec<u64>)> {
    let dimensions = |values: &[usize]| values.iter().map(|&value| value as u64).collect::<Vec<_>>();
    [
        (
            "column",
            3,
            dimensions(&[b, h + RESERVED, HEADS, INDUCING, COLUMN_DIM / HEADS]),
        ),
        ("icl", 12, dimensions(&[b, HEADS, n, ICL_DIM / HEADS])),
    ]
    .into_iter()
    .flat_map(|(stage, blocks, dimensions)| {
        (0..blocks).flat_map(move |block| {
            ["key", "value"].map(|kind| (format!("{stage}.{block}.{kind}"), dimensions.clone()))
        })
    })
    .collect()
}

fn validate_static_cache_contract(context: &StaticContextArtifact) -> Result<()> {
    let metadata = context.metadata();
    for (name, expected_shape) in context_tensors(
        metadata.ensemble_count,
        metadata.training_rows,
        metadata.feature_count,
    ) {
        let tensor = context
            .tensor(&name)
            .ok_or_else(|| Error::Artifact(format!("static context cache tensor '{name}' is missing")))?;
        if tensor.dimensions != expected_shape {
            return Err(Error::Artifact(format!(
                "static context cache tensor '{name}' has shape {:?}; expected {expected_shape:?}",
                tensor.dimensions
            )));
        }
        // Only the removed Python exporter wrote BF16 ICL keys.
        if name.starts_with("icl.")
            && name.rsplit('.').next() == Some("key")
            && matches!(tensor.data, StaticContextTensorData::Bf16(_))
        {
            return Err(Error::Artifact(format!(
                "static context cache tensor '{name}' stores BF16 keys, which are no longer supported; \
                 rebuild the context"
            )));
        }
    }
    Ok(())
}

/// The executor's own worker pool, or none where the target cannot start
/// threads (WebAssembly without atomics, as in Pyodide). Work then runs on the
/// calling thread, which [`WorkerPool::init_global`] makes rayon's only worker.
#[derive(Debug)]
struct WorkerPool(Option<ThreadPool>);

impl WorkerPool {
    const THREADS: bool = !cfg!(all(target_family = "wasm", not(target_feature = "atomics")));

    fn new(threads: usize) -> Result<Self> {
        if !Self::THREADS {
            return Ok(Self(None));
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("ironicl-affine-{index}"))
            .build()
            .map_err(|error| Error::InvalidArgument(format!("cannot create affine worker pool: {error}")))?;
        Ok(Self(Some(pool)))
    }

    /// Without threads, run rayon's global pool on the calling thread. Rayon
    /// falls back to this only when spawning reports `Unsupported`, and
    /// Emscripten reports `WouldBlock`.
    fn init_global() {
        if !Self::THREADS {
            // An error means the global pool already exists.
            let _ = ThreadPoolBuilder::new()
                .num_threads(1)
                .use_current_thread()
                .build_global();
        }
    }

    fn current_num_threads(&self) -> usize {
        self.0.as_ref().map_or(1, ThreadPool::current_num_threads)
    }

    fn install<R: Send>(&self, work: impl FnOnce() -> R + Send) -> R {
        match &self.0 {
            Some(pool) => pool.install(work),
            None => work(),
        }
    }
}

/// Split `0..items` into at most `workers` contiguous ranges and run `work` on
/// each, one scoped thread per range, or on the calling thread when there is
/// only one.
fn scoped_chunks<T: Send>(
    items: usize,
    workers: usize,
    work: impl Fn(std::ops::Range<usize>) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let chunk = items.div_ceil(workers.clamp(1, items.max(1)));
    if chunk >= items {
        return Ok(vec![work(0..items)?]);
    }
    let work = &work;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..items)
            .step_by(chunk)
            .map(|start| scope.spawn(move || work(start..(start + chunk).min(items))))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker must not panic"))
            .collect()
    })
}

pub(crate) fn executor_threads() -> usize {
    *EXECUTOR_THREADS.get_or_init(|| {
        if !WorkerPool::THREADS {
            return 1;
        }
        if let Some(value) = std::env::var_os("IRONICL_THREADS") {
            return value
                .to_string_lossy()
                .parse::<usize>()
                .ok()
                .filter(|threads| *threads > 0)
                .unwrap_or(1);
        }
        if let Some(cores) = performance_cores() {
            return cores;
        }
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        if cfg!(target_arch = "x86_64") {
            logical.div_ceil(2)
        } else {
            logical
        }
    })
}

/// Performance-core count on Apple Silicon. Using only these cores keeps workers off
/// the efficiency cores, and on an M1 Max (8 + 2 cores) it lets 8 estimators
/// run in parallel instead of one at a time with each split 10 ways.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[expect(unsafe_code, reason = "One libSystem call; std has no portable equivalent")]
fn performance_cores() -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            value: *mut std::ffi::c_void,
            length: *mut usize,
            new_value: *const std::ffi::c_void,
            new_length: usize,
        ) -> i32;
    }
    let mut cores = 0_i32;
    let mut length = std::mem::size_of::<i32>();
    // SAFETY: the name is a NUL-terminated C string, `cores` is a live `i32` and `length`
    // starts at its size, so sysctl writes at most 4 bytes; there is no new value to set.
    let status = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut cores).cast(),
            &raw mut length,
            std::ptr::null(),
            0,
        )
    };
    (status == 0 && length == std::mem::size_of::<i32>())
        .then(|| usize::try_from(cores).ok())
        .flatten()
        .filter(|cores| *cores > 0)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
const fn performance_cores() -> Option<usize> {
    None
}

fn column_source_major_attention_enabled(backend: crate::kernels::cpu::Isa) -> bool {
    backend == crate::kernels::cpu::Isa::Avx512
}

fn row_source_major_attention_enabled() -> bool {
    crate::kernels::cpu::Isa::detect() == crate::kernels::cpu::Isa::Avx512
}

fn static_execution_plan(ensembles: usize, total_threads: usize) -> ExecutionPlan {
    ExecutionPlan::automatic(ensembles, total_threads, false)
}

#[expect(
    clippy::too_many_arguments,
    reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
)]
fn parallel_cached_attention(
    backend: crate::kernels::cpu::Isa,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
    workers: usize,
) -> Result<()> {
    let query_head_elements = checked_product(&[query_rows, head_dim], "cached query head")?;
    let source_head_elements = checked_product(&[source_rows, head_dim], "cached source head")?;
    let workers = workers.min(HEADS);
    if workers == 1 {
        let scratch_query_rows = query_rows.min(ATTENTION_QUERY_BLOCK_ROWS);
        let scratch_source_rows = source_rows.min(ATTENTION_KEY_TILE_ROWS);
        let mut scores = vec![0.0; scratch_query_rows * scratch_source_rows];
        let mut maximum = vec![0.0; HEADS * query_rows];
        let mut denominator = vec![0.0; HEADS * query_rows];
        return source_major_online_attention_with_backend_f32(
            query,
            query_rows,
            key,
            value,
            source_rows,
            HEADS,
            head_dim,
            ATTENTION_QUERY_BLOCK_ROWS,
            ATTENTION_KEY_TILE_ROWS,
            backend,
            &mut OnlineAttentionScratch {
                scores: &mut scores,
                maximum: &mut maximum,
                denominator: &mut denominator,
            },
            output,
        );
    }
    let heads_per_worker = HEADS.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut remaining_output = output;
        let mut handles = Vec::with_capacity(workers);
        for start in (0..HEADS).step_by(heads_per_worker) {
            let heads = heads_per_worker.min(HEADS - start);
            let output_elements = heads * query_head_elements;
            let (worker_output, rest) = remaining_output.split_at_mut(output_elements);
            remaining_output = rest;
            let query = &query[start * query_head_elements..][..output_elements];
            let source_elements = heads * source_head_elements;
            let key = &key[start * source_head_elements..][..source_elements];
            let value = &value[start * source_head_elements..][..source_elements];
            handles.push(scope.spawn(move || {
                let scratch_query_rows = query_rows.min(ATTENTION_QUERY_BLOCK_ROWS);
                let scratch_source_rows = source_rows.min(ATTENTION_KEY_TILE_ROWS);
                let mut scores = vec![0.0; scratch_query_rows * scratch_source_rows];
                let mut maximum = vec![0.0; heads * query_rows];
                let mut denominator = vec![0.0; heads * query_rows];
                source_major_online_attention_with_backend_f32(
                    query,
                    query_rows,
                    key,
                    value,
                    source_rows,
                    heads,
                    head_dim,
                    ATTENTION_QUERY_BLOCK_ROWS,
                    ATTENTION_KEY_TILE_ROWS,
                    backend,
                    &mut OnlineAttentionScratch {
                        scores: &mut scores,
                        maximum: &mut maximum,
                        denominator: &mut denominator,
                    },
                    worker_output,
                )
            }));
        }
        for handle in handles {
            handle.join().expect("cached attention worker must not panic")?;
        }
        Ok(())
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "a model stage takes its inputs, weights, caches, and shape as plain arguments"
)]
fn parallel_cached_attention_f32_key_bf16_value(
    backend: crate::kernels::cpu::Isa,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    output: &mut [f32],
    workers: usize,
) -> Result<()> {
    let query_head_elements = checked_product(&[query_rows, head_dim], "cached query head")?;
    let source_head_elements = checked_product(&[source_rows, head_dim], "cached source head")?;
    let key_tile_rows = ATTENTION_KEY_TILE_ROWS;
    let workers = workers.min(HEADS);
    if workers == 1 {
        let scratch_query_rows = query_rows.min(ATTENTION_QUERY_BLOCK_ROWS);
        let scratch_source_rows = source_rows.min(key_tile_rows);
        let mut scores = vec![0.0; scratch_query_rows * scratch_source_rows];
        let mut maximum = vec![0.0; HEADS * query_rows];
        let mut denominator = vec![0.0; HEADS * query_rows];
        return source_major_online_attention_f32_key_bf16_value_with_backend_f32(
            query,
            query_rows,
            key,
            value,
            source_rows,
            HEADS,
            head_dim,
            ATTENTION_QUERY_BLOCK_ROWS,
            key_tile_rows,
            backend,
            &mut OnlineAttentionScratch {
                scores: &mut scores,
                maximum: &mut maximum,
                denominator: &mut denominator,
            },
            output,
        );
    }
    let heads_per_worker = HEADS.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut remaining_output = output;
        let mut handles = Vec::with_capacity(workers);
        for start in (0..HEADS).step_by(heads_per_worker) {
            let heads = heads_per_worker.min(HEADS - start);
            let output_elements = heads * query_head_elements;
            let (worker_output, rest) = remaining_output.split_at_mut(output_elements);
            remaining_output = rest;
            let query = &query[start * query_head_elements..][..output_elements];
            let source_elements = heads * source_head_elements;
            let key = &key[start * source_head_elements..][..source_elements];
            let value = &value[start * source_head_elements..][..source_elements];
            handles.push(scope.spawn(move || {
                let scratch_query_rows = query_rows.min(ATTENTION_QUERY_BLOCK_ROWS);
                let scratch_source_rows = source_rows.min(key_tile_rows);
                let mut scores = vec![0.0; scratch_query_rows * scratch_source_rows];
                let mut maximum = vec![0.0; heads * query_rows];
                let mut denominator = vec![0.0; heads * query_rows];
                source_major_online_attention_f32_key_bf16_value_with_backend_f32(
                    query,
                    query_rows,
                    key,
                    value,
                    source_rows,
                    heads,
                    head_dim,
                    ATTENTION_QUERY_BLOCK_ROWS,
                    key_tile_rows,
                    backend,
                    &mut OnlineAttentionScratch {
                        scores: &mut scores,
                        maximum: &mut maximum,
                        denominator: &mut denominator,
                    },
                    worker_output,
                )
            }));
        }
        for handle in handles {
            handle
                .join()
                .expect("cached FP32-K/BF16-V attention worker must not panic")?;
        }
        Ok(())
    })
}

fn validate_labels(labels: &[f32], b: usize, n: usize) -> Result<usize> {
    let mut expected_classes = None;
    for ensemble in 0..b {
        let mut present = [false; 10];
        for label in &labels[ensemble * n..][..n] {
            present[label_index(*label)?] = true;
        }
        let classes = present.iter().take_while(|value| **value).count();
        if !(2..=10).contains(&classes) || present[classes..].iter().any(|value| *value) {
            return Err(Error::InvalidArgument(
                "each ensemble's labels must be contiguous 0..C-1 with C in 2..=10".into(),
            ));
        }
        if expected_classes
            .replace(classes)
            .is_some_and(|expected| expected != classes)
        {
            return Err(Error::InvalidArgument(
                "all ensembles must have the same class count".into(),
            ));
        }
    }
    expected_classes.ok_or_else(|| Error::InvalidArgument("labels are empty".into()))
}

fn checked_product(values: &[usize], name: &'static str) -> Result<usize> {
    values.iter().try_fold(1_usize, |result, value| {
        result.checked_mul(*value).ok_or(Error::ArithmeticOverflow(name))
    })
}

/// Estimate the largest persistent and intermediate FP32 activation set for one ensemble.
///
/// The executor processes ensembles sequentially. The estimates intentionally retain
/// all operands that can coexist during each stage rather than assuming allocator reuse.
fn admit_request(b: usize, t: usize, h: usize, n: usize, width: usize) -> Result<()> {
    if request_activation_elements(b, t, h, n, width)? > MAX_ACTIVATION_ELEMENTS {
        return Err(Error::InvalidArgument(format!(
            "request exceeds the uncached scalar activation limit of {MAX_ACTIVATION_BYTES} bytes"
        )));
    }
    Ok(())
}

/// The memory plan for building a `task` context of `estimators x rows x
/// features` with `optimize`: the ICL storage, builder, and pools the build
/// itself would pick, under `limits`. Faster `optimize` levels with smaller
/// storage are offered as suggestions.
///
/// # Errors
///
/// Returns an error for an empty shape or a size that overflows.
pub fn plan_context(
    task: TaskKind,
    (estimators, rows, features): (usize, usize, usize),
    optimize: Optimize,
    limits: MemoryLimits,
) -> Result<Plan> {
    let working_set = gpu_working_set(rows);
    let bf16 = crate::kernels::cpu::bf16_storage_supported();
    // As in `Executor::build_context`.
    let storage = |optimize| match optimize {
        Optimize::Safe => Storage::FP32,
        _ if working_set.is_some() => Storage::FP16,
        Optimize::Fast if bf16 => Storage::FP32_KEY_BF16_VALUE,
        Optimize::Auto | Optimize::Fast => Storage::FP32,
    };
    let current = storage(optimize);
    let element_bytes = |storage: Storage| storage.key_bytes + storage.value_bytes;
    let mut alternatives: Vec<(String, Storage)> = Vec::new();
    for (level, name) in [(Optimize::Auto, "auto"), (Optimize::Fast, "fast")] {
        let smaller = storage(level);
        if element_bytes(smaller) < element_bytes(current) && alternatives.iter().all(|(_, s)| *s != smaller)
        {
            alternatives.push((format!("optimize=\"{name}\""), smaller));
        }
    }
    let mut pools = Pools::detect().with_limits(limits);
    pools.device = working_set.map(|limit| crate::memory::DeviceMemory::Shared { limit });
    let shape = ContextShape {
        task,
        estimators: estimators as u64,
        rows: rows as u64,
        features: features as u64,
    };
    let builder = if working_set.is_some() {
        Builder::Gpu
    } else {
        Builder::Cpu
    };
    crate::memory::plan(shape, current, builder, pools, &alternatives)
}

/// This machine's memory pools with `gpu` as a device that shares the host's
/// RAM, limited by Metal's recommended working set and by `limits`.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn shared_gpu_pools(gpu: &crate::kernels::metal::Gpu, limits: MemoryLimits) -> Pools {
    let working_set = u64::try_from(gpu.working_set_bytes()).unwrap_or(u64::MAX);
    Pools {
        device: Some(crate::memory::DeviceMemory::Shared { limit: working_set }),
        ..Pools::detect().with_limits(limits)
    }
}

/// Metal's recommended working set when the GPU would build `rows` training
/// rows, or `None` when the CPU builds: no GPU, `IRONICL_NO_METAL` set, or too
/// few rows.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    expect(unused_variables, reason = "only the Metal build reads `rows`")
)]
fn gpu_working_set(rows: usize) -> Option<u64> {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if std::env::var_os("IRONICL_NO_METAL").is_none()
        && rows >= crate::models::tabicl_v2::metal::GPU_MIN_BUILD_ROWS
    {
        return crate::kernels::metal::system_working_set();
    }
    None
}

/// Transient bytes the CPU build estimates for one estimator, for the memory
/// planner. Like the build itself, this is 2-3x the measured peak.
pub(crate) fn cpu_build_estimator_bytes(n: usize, h: usize, width: usize) -> Result<usize> {
    checked_product(
        &[
            request_activation_elements(1, n, h, n, width)?,
            std::mem::size_of::<f32>(),
        ],
        "CPU build bytes",
    )
}

/// The CPU build's budget for estimators built together: four activation limits.
pub(crate) const fn cpu_build_budget_bytes() -> u64 {
    4 * MAX_ACTIVATION_BYTES as u64
}

/// The FP32 activation elements that [`admit_request`] checks.
fn request_activation_elements(b: usize, t: usize, h: usize, n: usize, width: usize) -> Result<usize> {
    let streams = h
        .checked_add(RESERVED)
        .ok_or(Error::ArithmeticOverflow("column streams"))?;
    let test = t.checked_sub(n).ok_or(Error::ArithmeticOverflow("test rows"))?;

    let output = checked_product(&[b, test, width], "decoder output")?;
    let columns = checked_product(&[t, streams, COLUMN_DIM], "column embeddings")?;
    let column_targets = checked_product(&[n, COLUMN_DIM], "column target embeddings")?;
    let rows = checked_product(&[t, ICL_DIM], "row embeddings")?;
    let icl_targets = checked_product(&[n, ICL_DIM], "ICL target embeddings")?;
    let attention_scratch = attention_scratch_elements()?;

    // The block working-set multipliers retain all block vectors that can stay
    // live together. They deliberately exceed the current implementation.
    let column_peak = checked_sum(
        &[
            columns,
            column_targets,
            checked_product(&[40, t, COLUMN_DIM], "column block activations")?,
            attention_scratch,
        ],
        "column activation peak",
    )?;
    let row_peak = checked_sum(
        &[
            columns,
            rows,
            checked_product(&[40, streams, COLUMN_DIM], "row block activations")?,
            attention_scratch,
        ],
        "row activation peak",
    )?;
    let icl_peak = checked_sum(
        &[
            checked_product(&[40, t, ICL_DIM], "ICL block activations")?,
            checked_product(&[16, n, ICL_DIM], "ICL source activations")?,
            icl_targets,
            attention_scratch,
        ],
        "ICL activation peak",
    )?;
    checked_sum(&[output, column_peak, row_peak, icl_peak], "activation peak")
}

fn admit_static_request(
    b: usize,
    q: usize,
    h: usize,
    classes: usize,
    concurrent_estimators: usize,
) -> Result<()> {
    let streams = h
        .checked_add(RESERVED)
        .ok_or(Error::ArithmeticOverflow("cached column streams"))?;
    let per_estimator_peak = checked_sum(
        &[
            checked_product(&[q, streams, COLUMN_DIM], "cached column embeddings")?,
            checked_product(&[16, q, ICL_DIM], "cached ICL block activations")?,
            checked_product(&[40, streams, COLUMN_DIM], "cached row block activations")?,
            attention_scratch_elements()?,
        ],
        "cached estimator activation peak",
    )?;
    let activation_peak = checked_sum(
        &[
            checked_product(&[b, q, classes], "cached classifier output")?,
            checked_product(
                &[concurrent_estimators, per_estimator_peak],
                "concurrent cached estimator activations",
            )?,
        ],
        "cached activation peak",
    )?;
    if activation_peak > MAX_ACTIVATION_ELEMENTS {
        return Err(Error::InvalidArgument(format!(
            "query exceeds the cached FP32 activation limit of {MAX_ACTIVATION_BYTES} bytes"
        )));
    }
    Ok(())
}

fn checked_sum(values: &[usize], name: &'static str) -> Result<usize> {
    values.iter().try_fold(0_usize, |sum, value| {
        sum.checked_add(*value).ok_or(Error::ArithmeticOverflow(name))
    })
}

fn attention_scratch_scores(q_rows: usize, source_rows: usize) -> Result<usize> {
    checked_product(&[HEADS, q_rows, source_rows], "attention scratch")
}

fn attention_scratch_state(q_rows: usize) -> Result<usize> {
    checked_product(&[HEADS, q_rows], "attention online state")
}

fn attention_scratch_elements() -> Result<usize> {
    let query_block_rows = ATTENTION_QUERY_BLOCK_ROWS;
    checked_sum(
        &[
            attention_scratch_scores(query_block_rows, ATTENTION_KEY_TILE_ROWS)?,
            attention_scratch_state(query_block_rows)?,
            attention_scratch_state(query_block_rows)?,
        ],
        "attention scratch",
    )
}

fn label_indices(labels: &[f32]) -> Result<Vec<usize>> {
    labels.iter().copied().map(label_index).collect()
}

fn label_index(value: f32) -> Result<usize> {
    match value {
        0.0 => Ok(0),
        1.0 => Ok(1),
        2.0 => Ok(2),
        3.0 => Ok(3),
        4.0 => Ok(4),
        5.0 => Ok(5),
        6.0 => Ok(6),
        7.0 => Ok(7),
        8.0 => Ok(8),
        9.0 => Ok(9),
        _ => Err(Error::InvalidArgument("labels must be integers in 0..=9".into())),
    }
}

fn tokens_to_heads(input: &[f32], rows: usize, dim: usize, heads: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    tokens_to_heads_into(input, rows, dim, heads, &mut output);
    output
}
fn tokens_to_heads_into(input: &[f32], rows: usize, dim: usize, heads: usize, output: &mut [f32]) {
    let head = dim / heads;
    for row in 0..rows {
        for h in 0..heads {
            output[(h * rows + row) * head..][..head].copy_from_slice(&input[row * dim + h * head..][..head]);
        }
    }
}

fn heads_to_tokens(input: &[f32], rows: usize, dim: usize, heads: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    heads_to_tokens_into(input, rows, dim, heads, &mut output);
    output
}
fn heads_to_tokens_into(input: &[f32], rows: usize, dim: usize, heads: usize, output: &mut [f32]) {
    let head = dim / heads;
    for row in 0..rows {
        for h in 0..heads {
            output[row * dim + h * head..][..head].copy_from_slice(&input[(h * rows + row) * head..][..head]);
        }
    }
}

/// A context's FP16 ICL keys and values, which the CUDA predictor uploads or
/// streams. Holds the context, so its storage outlives the predictor.
#[cfg(feature = "cuda")]
struct ContextIcl(StaticContextArtifact);

#[cfg(feature = "cuda")]
impl ContextIcl {
    /// `None` unless every ICL block's keys and values are stored as a cache
    /// at `precision` holds them.
    fn new(
        context: &StaticContextArtifact,
        precision: crate::models::tabicl_v2::cuda::Precision,
    ) -> Option<Self> {
        let context = Self(context.clone());
        (0..ICL_BLOCKS)
            .all(|block| {
                context
                    .tensors(block)
                    .is_some_and(|(keys, values)| keys.stored_at(precision) && values.stored_at(precision))
            })
            .then_some(context)
    }

    fn tensors(
        &self,
        block: usize,
    ) -> Option<(
        crate::models::tabicl_v2::cuda::IclValues<'_>,
        crate::models::tabicl_v2::cuda::IclValues<'_>,
    )> {
        let tensor = |kind: &str| match self.0.tensor(&format!("icl.{block}.{kind}"))?.data {
            StaticContextTensorData::Fp32(values) => {
                Some(crate::models::tabicl_v2::cuda::IclValues::F32(values))
            }
            StaticContextTensorData::F16(values) => {
                Some(crate::models::tabicl_v2::cuda::IclValues::F16(values))
            }
            StaticContextTensorData::Bf16(_) => None,
        };
        Some((tensor("key")?, tensor("value")?))
    }
}

#[cfg(feature = "cuda")]
impl crate::models::tabicl_v2::cuda::IclSource for ContextIcl {
    fn block(
        &self,
        block: usize,
    ) -> (
        crate::models::tabicl_v2::cuda::IclValues<'_>,
        crate::models::tabicl_v2::cuda::IclValues<'_>,
    ) {
        self.tensors(block).expect("ContextIcl::new checked every block")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClassifierAggregation, DecoderOutputs, admit_request, admit_static_request,
        aggregate_classifier_probabilities, attention_scratch_scores, attention_scratch_state,
        model_hash_matches, page_aligned, parallel_cached_attention_f32_key_bf16_value, tokens_to_heads,
        validate_class_count, validate_labels, validate_static_cache_contract,
    };
    use crate::kernels::reference::{
        OnlineAttentionScratch, affine_f32, tiled_online_scaled_dot_product_attention_f32,
    };
    use crate::{StaticContextArtifact, StaticContextMetadata, format::static_context::StaticContextTensor};

    fn context(tensors: Vec<StaticContextTensor>) -> StaticContextArtifact {
        StaticContextArtifact::new(
            StaticContextMetadata {
                model_sha256: [1; 32],
                context_sha256: [2; 32],
                ensemble_count: 1,
                training_rows: 2,
                feature_count: 1,
                class_count: 2,
                class_permutations: vec![0, 1],
                feature_permutations: vec![0],
                optimize: None,
            },
            tensors,
        )
        .expect("valid static context")
    }

    #[test]
    #[expect(clippy::cast_possible_truncation, reason = "test shapes are small")]
    fn cached_context_rejects_missing_and_wrong_shaped_tensors() {
        assert!(
            validate_static_cache_contract(&context(vec![StaticContextTensor {
                name: "unrelated".into(),
                dimensions: vec![1],
                values: vec![0.0],
            }]))
            .is_err()
        );
        let mut tensors = Vec::new();
        for (stage, layers, sources, head_dim) in [("column", 3, 128, 16), ("icl", 12, 2, 64)] {
            for layer in 0..layers {
                for kind in ["key", "value"] {
                    let dimensions = if stage == "column" {
                        vec![1, 5, 8, sources, head_dim]
                    } else {
                        vec![1, 8, sources, head_dim]
                    };
                    let elements = dimensions.iter().product::<u64>() as usize;
                    tensors.push(StaticContextTensor {
                        name: format!("{stage}.{layer}.{kind}"),
                        dimensions,
                        values: vec![0.0; elements],
                    });
                }
            }
        }
        let valid = context(tensors.clone());
        assert!(validate_static_cache_contract(&valid).is_ok());
        // FP32 keys with BF16 values, as `fast` writes, pass; BF16 keys do not.
        let icl = |kind: &'static str| {
            move |name: &str| name.starts_with("icl.") && name.rsplit('.').next() == Some(kind)
        };
        let with_bf16 = |kind| {
            StaticContextArtifact::from_page_aligned(
                valid.metadata().clone(),
                page_aligned(
                    &tensors,
                    icl(kind),
                    crate::format::static_context::TensorStorage::Bf16,
                )
                .expect("page-aligned tensors"),
            )
            .expect("page-aligned context")
        };
        assert!(validate_static_cache_contract(&with_bf16("value")).is_ok());
        let error = validate_static_cache_contract(&with_bf16("key")).expect_err("BF16 keys are rejected");
        assert!(error.to_string().contains("BF16 keys"), "{error}");
        tensors[0].dimensions[1] = 4;
        tensors[0].values.truncate(4 * 8 * 128 * 16);
        assert!(validate_static_cache_contract(&context(tensors)).is_err());
    }

    #[test]
    fn context_class_count_must_match_the_model_task() {
        use crate::TaskKind::{Classifier, Regressor};
        assert!(validate_class_count(Classifier, 2).is_ok());
        assert!(validate_class_count(Classifier, 10).is_ok());
        assert!(validate_class_count(Classifier, 0).is_err());
        assert!(validate_class_count(Classifier, 11).is_err());
        assert!(validate_class_count(Regressor, 0).is_ok());
        assert!(validate_class_count(Regressor, 2).is_err());
    }

    #[test]
    fn cached_request_admission_is_bounded() {
        assert!(admit_static_request(1, 64, 54, 7, 1).is_ok());
        assert!(admit_static_request(8, 64, 54, 7, 8).is_ok());
        assert!(admit_static_request(1, 1_000_000, 1, 2, 1).is_err());
    }

    #[test]
    fn cache_identity_rejects_a_different_model() {
        assert!(model_hash_matches(
            b"0101010101010101010101010101010101010101010101010101010101010101",
            &[1; 32]
        ));
        assert!(!model_hash_matches(
            b"0101010101010101010101010101010101010101010101010101010101010101",
            &[2; 32]
        ));
    }

    #[test]
    fn cached_attention_matches_source_projected_key_value() {
        let (rows, source_rows, dim, heads) = (2, 3, 8, 2);
        let head_dim = dim / heads;
        #[expect(
            clippy::cast_precision_loss,
            reason = "row counts are at most a few million, close enough in f32"
        )]
        let source = (0..source_rows * dim)
            .map(|index| index as f32 * 0.03125 - 0.2)
            .collect::<Vec<_>>();
        #[expect(
            clippy::cast_precision_loss,
            reason = "row counts are at most a few million, close enough in f32"
        )]
        let query = (0..rows * dim)
            .map(|index| index as f32 * -0.0625 + 0.4)
            .collect::<Vec<_>>();
        #[expect(
            clippy::cast_precision_loss,
            reason = "row counts are at most a few million, close enough in f32"
        )]
        let weight = (0..dim * dim)
            .map(|index| index as f32 * 0.007_812_5 - 0.1)
            .collect::<Vec<_>>();
        let mut key_tokens = vec![0.0; source.len()];
        let mut value_tokens = vec![0.0; source.len()];
        affine_f32(&source, source_rows, dim, &weight, None, dim, &mut key_tokens).expect("key projection");
        affine_f32(&source, source_rows, dim, &weight, None, dim, &mut value_tokens)
            .expect("value projection");
        let key = tokens_to_heads(&key_tokens, source_rows, dim, heads);
        let value = tokens_to_heads(&value_tokens, source_rows, dim, heads);
        let query = tokens_to_heads(&query, rows, dim, heads);
        let mut expected = vec![0.0; query.len()];
        let mut actual = vec![0.0; query.len()];
        let mut scores = vec![0.0; heads * rows * source_rows];
        crate::kernels::reference::scaled_dot_product_attention_f32(
            &query,
            rows,
            &key,
            &value,
            source_rows,
            heads,
            head_dim,
            &mut scores,
            &mut expected,
        )
        .expect("source attention");
        let mut scores = vec![0.0; attention_scratch_scores(rows, source_rows).expect("scratch")];
        let mut maximum = vec![0.0; attention_scratch_state(rows).expect("scratch")];
        let mut denominator = vec![0.0; attention_scratch_state(rows).expect("scratch")];
        tiled_online_scaled_dot_product_attention_f32(
            &query,
            rows,
            &key,
            &value,
            source_rows,
            heads,
            head_dim,
            rows,
            source_rows,
            &mut OnlineAttentionScratch {
                scores: &mut scores,
                maximum: &mut maximum,
                denominator: &mut denominator,
            },
            &mut actual,
        )
        .expect("cached attention");
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "row counts are at most a few million, close enough in f32"
    )]
    fn serial_hybrid_attention_matches_head_parallel_attention() {
        let (query_rows, source_rows, head_dim) = (2, 16, 64);
        let query = (0..super::HEADS * query_rows * head_dim)
            .map(|index| index as f32 * 0.000_31 - 0.2)
            .collect::<Vec<_>>();
        let key = (0..super::HEADS * source_rows * head_dim)
            .map(|index| index as f32 * -0.000_17 + 0.1)
            .collect::<Vec<_>>();
        let value = (0..super::HEADS * source_rows * head_dim)
            .map(|index| {
                let bits = (index as f32 * 0.000_23 - 0.3).to_bits();
                u16::try_from((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).expect("upper BF16 bits")
            })
            .collect::<Vec<_>>();
        let mut serial = vec![0.0; query.len()];
        let mut parallel = vec![0.0; query.len()];
        parallel_cached_attention_f32_key_bf16_value(
            crate::kernels::cpu::Isa::Portable,
            &query,
            query_rows,
            &key,
            &value,
            source_rows,
            head_dim,
            &mut serial,
            1,
        )
        .expect("serial hybrid attention");
        parallel_cached_attention_f32_key_bf16_value(
            crate::kernels::cpu::Isa::Portable,
            &query,
            query_rows,
            &key,
            &value,
            source_rows,
            head_dim,
            &mut parallel,
            2,
        )
        .expect("parallel hybrid attention");
        assert_eq!(serial, parallel);
    }

    #[test]
    fn labels_require_contiguous_shared_classes() {
        assert_eq!(validate_labels(&[0.0, 1.0, 0.0, 1.0], 1, 4), Ok(2));
        assert!(validate_labels(&[0.0, 2.0], 1, 2).is_err());
        assert!(validate_labels(&[0.0, 1.0, 0.0, 1.0, 2.0, 2.0], 2, 3).is_err());
    }

    #[test]
    fn probabilities_reverse_class_permutations_and_follow_aggregation_options() {
        let logits = DecoderOutputs {
            shape: [2, 1, 3],
            values: vec![1.0, 2.0, 3.0, 4.0, 8.0, 6.0],
        };
        let permutations = [2, 0, 1, 1, 2, 0];
        // After undoing the permutations, the members' logits are these rows.
        let members = [[3.0_f32, 1.0, 2.0], [8.0, 6.0, 4.0]];
        let reference = |values: &[f32], temperature: f32| -> Vec<f32> {
            let exp = values
                .iter()
                .map(|value| (value / temperature).exp())
                .collect::<Vec<_>>();
            let sum = exp.iter().sum::<f32>();
            exp.iter().map(|value| value / sum).collect()
        };
        for (temperature, average_logits) in [(0.9, true), (0.5, true), (0.9, false), (2.0, false)] {
            let aggregation = ClassifierAggregation {
                softmax_temperature: temperature,
                average_logits,
            };
            let expected = if average_logits {
                let mean = [0, 1, 2].map(|class| f32::midpoint(members[0][class], members[1][class]));
                reference(&mean, temperature)
            } else {
                let (first, second) = (
                    reference(&members[0], temperature),
                    reference(&members[1], temperature),
                );
                first
                    .iter()
                    .zip(&second)
                    .map(|(a, b)| f32::midpoint(*a, *b))
                    .collect()
            };
            let probabilities = aggregate_classifier_probabilities(&logits, &permutations, aggregation)
                .expect("valid aggregation");
            assert_eq!(probabilities.shape(), [1, 3]);
            for (actual, expected) in probabilities.values().iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "{aggregation:?}: {actual} != {expected}"
                );
            }
        }
        for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let aggregation = ClassifierAggregation {
                softmax_temperature: temperature,
                average_logits: true,
            };
            assert!(aggregate_classifier_probabilities(&logits, &permutations, aggregation).is_err());
        }
    }

    #[test]
    fn admission_allows_a_small_baseline_run() {
        assert!(admit_request(8, 1024, 100, 512, 10).is_ok());
    }

    #[test]
    fn admission_allows_bounded_attention_scratch() {
        // The former [heads, query_rows, source_rows] score allocation here
        // exceeded the attention limit; the remaining activation estimate fits.
        assert!(admit_request(1, 11_000, 1, 3_000, 10).is_ok());
    }

    #[test]
    fn admission_rejects_huge_dimensions_without_input_allocation() {
        assert!(admit_request(1, 1_000_000, 1, 1, 10).is_err());
        assert!(admit_request(1, usize::MAX, usize::MAX, 1, 10).is_err());
    }

    #[test]
    fn widens_half_precision_exactly() {
        for (bits, expected) in [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x3555, 0.333_251_95),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x0000, 0.0),
            (0x7c00, f32::INFINITY),
        ] {
            assert_eq!(
                super::f16_to_f32(bits).to_bits(),
                f32::to_bits(expected),
                "{bits:#06x}"
            );
        }
        assert!(super::f16_to_f32(0x7e00).is_nan());
    }
}

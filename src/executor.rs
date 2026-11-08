//! Uncached scalar FP32 execution for the official `TabICLv2` classifier.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use sha2::{Digest, Sha256};

use crate::{
    Error, ModelArtifact, QueryRequest, RegressionOutput, Request, Result, StaticContextArtifact,
    StaticContextMetadata, StaticContextTensor, StaticContextTensorData, StaticQueryBundle, TaskKind,
    execution_plan::ExecutionPlan,
    kernels::{
        native::{
            Bf16AffineCompensation, PackedBf16Affine, PackedF32Affine, packed_bf16_affine_supported,
            packed_f32_affine_supported,
        },
        reference::{
            OnlineAttentionScratch, QassMaxWeights, affine_f32, affine_input3_transposed_f32, gelu_f32,
            layer_norm_f32, one_hot_linear_f32, qassmax_f32, rope_non_interleaved_f32,
            source_major_online_attention_bf16_with_backend_f32,
            source_major_online_attention_f32_key_bf16_value_with_backend_f32,
            source_major_online_attention_with_backend_f32, tiled_online_scaled_dot_product_attention_f32,
        },
    },
    memory::{Builder, ContextShape, MemoryLimits, Plan, Pools, Storage},
    profile::{self, Category},
    quantiles::summarize,
};

pub(crate) const COLUMN_DIM: usize = 128;
pub(crate) const ICL_DIM: usize = 512;
pub(crate) const HEADS: usize = 8;
pub(crate) const INDUCING: usize = 128;
/// CLS streams ahead of the features in the row stage.
pub(crate) const RESERVED: usize = 4;
const ATTENTION_QUERY_BLOCK_ROWS: usize = 16;
const ATTENTION_KEY_TILE_ROWS: usize = 128;
const MAX_ACTIVATION_BYTES: usize = 1536 * 1024 * 1024;
const MAX_ACTIVATION_ELEMENTS: usize = MAX_ACTIVATION_BYTES / std::mem::size_of::<f32>();
/// Cached prediction splits larger queries. The GPU keeps the buffers and
/// compiled graph of every query bucket up to this size.
pub(crate) const QUERY_CHUNK_ROWS: usize = 1024;
const SPARSE_INDEX_TENSORS: [&str; 4] = [
    "icl.index.centroids",
    "icl.index.radii",
    "icl.index.offsets",
    "icl.index.members",
];
static EXECUTOR_THREADS: OnceLock<usize> = OnceLock::new();

thread_local! {
    static BF16_AFFINE_INPUT: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
    static BF16_AFFINE_INPUT_RESIDUAL: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
struct Bf16DensePolicy {
    column_q: bool,
    column_out: bool,
    column_ffn1: bool,
    column_ffn2: bool,
    icl_q: bool,
    icl_out: bool,
    icl_ffn1: bool,
    icl_ffn2: bool,
    qass: bool,
    decoder: bool,
}

#[derive(Clone, Copy)]
struct SparseIndex<'a> {
    centroids: &'a [f32],
    radii: &'a [f32],
    offsets: &'a [f32],
    members: &'a [f32],
    clusters: usize,
}

impl Bf16DensePolicy {
    /// Every dense stage in BF16. The `bf16-dense` feature enables this whole
    /// policy; finer selection was a runtime-only tuning knob that is gone.
    const fn full() -> Self {
        Self {
            column_q: true,
            column_out: true,
            column_ffn1: true,
            column_ffn2: true,
            icl_q: true,
            icl_out: true,
            icl_ffn1: true,
            icl_ffn2: true,
            qass: true,
            decoder: true,
        }
    }
}

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

/// Owned FP32 values for a traced classifier stage in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceStage<const D: usize> {
    shape: [usize; D],
    values: Vec<f32>,
}

impl<const D: usize> TraceStage<D> {
    /// Stage dimensions in row-major order.
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// Stage values in row-major order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Coarse intermediate outputs from one classifier prediction.
///
/// This diagnostic result retains all requested ensemble stage outputs. Use it
/// only for small requests because it increases the activation memory needed.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardTrace {
    column_embeddings: TraceStage<4>,
    row_representations: TraceStage<3>,
    decoder_logits: TraceStage<3>,
    logits: DecoderOutputs,
}

impl ForwardTrace {
    /// Column embeddings with shape `[B, T, H + 4, 128]`.
    #[must_use]
    pub fn column_embeddings(&self) -> &TraceStage<4> {
        &self.column_embeddings
    }

    /// Row representations with shape `[B, T, 512]`, before ICL label injection.
    #[must_use]
    pub fn row_representations(&self) -> &TraceStage<3> {
        &self.row_representations
    }

    /// Decoder outputs with shape `[B, T, width]`, before test-row and class slicing.
    #[must_use]
    pub fn decoder_logits(&self) -> &TraceStage<3> {
        &self.decoder_logits
    }

    /// Final prediction logits with shape `[B, test_rows, classes]`.
    #[must_use]
    pub fn logits(&self) -> &DecoderOutputs {
        &self.logits
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
    gpu_weights: Arc<OnceLock<Option<Arc<crate::metal::ModelWeights>>>>,
}

/// One context's Metal buffers and the graph that reads them.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
struct GpuContext {
    predict: crate::metal::PredictGraph,
    /// How queries split into graph runs, `(chunk, small)` from
    /// [`crate::metal::PredictGraph::query_split`], which keeps the buckets the
    /// graph keeps within the device's share of memory.
    split: (usize, usize),
    /// Keeps page-aligned context memory alive while GPU buffers wrap it.
    _storage: Option<Arc<memmap2::Mmap>>,
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl std::fmt::Debug for GpuContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("GpuContext")
    }
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
    affine_pool: Arc<ThreadPool>,
    column_input_weight: Arc<Vec<f32>>,
    qass_base_scales: Arc<Vec<Vec<f32>>>,
    packed_f32_affines: Option<Arc<BTreeMap<String, PackedF32Affine>>>,
    packed_f32_row: bool,
    packed_bf16_affines: Option<Arc<BTreeMap<String, PackedBf16Affine>>>,
    bf16_dense_compensation: Bf16AffineCompensation,
    attention_backend: crate::kernels::native::AttentionBackend,
    sparse_attention_stats: Arc<SparseAttentionStats>,
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    gpu: Option<Arc<GpuContext>>,
}

/// Sparse ICL attention decisions since the last reset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseAttentionStatistics {
    /// Calls that evaluated sparse candidates.
    pub attempted: u64,
    /// Calls whose candidate set passed the omitted-mass certificate.
    pub accepted: u64,
    /// Calls that fell back to dense attention.
    pub fallback: u64,
}

#[derive(Debug, Default)]
struct SparseAttentionStats {
    attempted: AtomicU64,
    accepted: AtomicU64,
}

impl SparseAttentionStats {
    fn record(&self, accepted: bool) {
        self.attempted.fetch_add(1, Ordering::Relaxed);
        if accepted {
            self.accepted.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.attempted.store(0, Ordering::Relaxed);
        self.accepted.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> SparseAttentionStatistics {
        let attempted = self.attempted.load(Ordering::Relaxed);
        let accepted = self.accepted.load(Ordering::Relaxed);
        SparseAttentionStatistics {
            attempted,
            accepted,
            fallback: attempted.saturating_sub(accepted),
        }
    }
}

enum PredictionResult {
    Logits(DecoderOutputs),
    Trace(ForwardTrace),
}

impl Executor {
    /// Take ownership of an exact-manifest validated classifier model.
    #[must_use]
    pub fn new(model: ModelArtifact) -> Self {
        Self {
            model: Arc::new(model),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            gpu_weights: Arc::default(),
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
        threads: Option<usize>,
        limits: MemoryLimits,
    ) -> Result<StaticExecutor> {
        let context = StaticContextArtifact::read_mapped_prefault(path)?;
        StaticExecutor::with_executor(self.clone(), context, threads, limits)
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

    /// The GPU and the model's weights in shared GPU buffers, created on first
    /// use and shared by every context of the model. `None` without a GPU or
    /// with `IRONICL_NO_METAL` set.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn gpu_weights(&self) -> Option<Arc<crate::metal::ModelWeights>> {
        if std::env::var_os("IRONICL_NO_METAL").is_some() {
            return None;
        }
        self.gpu_weights
            .get_or_init(|| {
                let gpu = crate::metal::Gpu::new().ok().flatten()?;
                Some(Arc::new(crate::metal::ModelWeights::new(gpu)))
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
    ) -> Result<Option<crate::static_context::PageAlignedTensors>> {
        use crate::static_context::{PageAlignedTensors, TensorStorage};
        let icl_storage = match optimize {
            Optimize::Safe => TensorStorage::Fp32,
            // BF16 had no native GPU arithmetic on an M1 Max: slower and less
            // accurate than FP16, so `Fast` uses FP16 until it gets more.
            Optimize::Auto | Optimize::Fast => TensorStorage::F16,
        };
        if n < crate::metal::GPU_MIN_BUILD_ROWS {
            return Ok(None);
        }
        let Some(gpu_weights) = self.gpu_weights() else {
            return Ok(None);
        };
        let Some((grouped, column_labels, icl_labels)) = self.gpu_build_inputs((b, n, h), x, y_train)? else {
            return Ok(None);
        };
        // The memory plan sizes the estimator groups: half the host's available
        // RAM (unified memory is the host's RAM), capped by Metal's working set
        // and by the memory limit. When one estimator does not fit the device,
        // the CPU builds the context.
        let pools = shared_gpu_pools(gpu_weights.gpu(), limits);
        let estimator_bytes =
            u64::try_from(crate::metal::gpu_build_estimator_bytes(n, h)).unwrap_or(u64::MAX);
        let Some(group) = crate::memory::gpu_build_group(b as u64, estimator_bytes, pools) else {
            return Ok(None);
        };
        let group = usize::try_from(group).map_err(|_| Error::ArithmeticOverflow("GPU build group"))?;
        let column_scales = (0..3)
            .map(|block| {
                qass_base_scale(
                    self,
                    &format!("col_embedder.tf_col.blocks.{block}.multihead_attn1.attn"),
                    n,
                    COLUMN_DIM,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let icl_scales = (0..12)
            .map(|block| {
                qass_base_scale(
                    self,
                    &format!("icl_predictor.tf_icl.blocks.{block}.attn"),
                    n,
                    ICL_DIM,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Only the ICL caches, almost all of a context, take the reduced type.
        let tensors = |ensembles: usize| {
            PageAlignedTensors::new(
                context_tensors(ensembles, n, h)
                    .into_iter()
                    .map(|(name, dimensions)| {
                        let storage = if name.starts_with("icl.") {
                            icl_storage
                        } else {
                            TensorStorage::Fp32
                        };
                        (name, dimensions, storage)
                    })
                    .collect(),
            )
        };
        let mut context = tensors(b)?;
        let mut scratch = if group == b { None } else { Some(tensors(group)?) };
        for start in (0..b).step_by(group) {
            let output = scratch.as_mut().unwrap_or(&mut context);
            crate::metal::build_context(
                &gpu_weights,
                &|name| self.tensor(name),
                crate::metal::Shape {
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
        }
        Ok(Some(context))
    }

    /// The GPU build's inputs: each feature grouped with its neighbors as
    /// `[B * H, N, 3]`, and the column and ICL label embeddings as
    /// `[B, N, 128]` and `[B, N, 512]`. `None` on the all-missing marker.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[allow(clippy::type_complexity)]
    fn gpu_build_inputs(
        &self,
        (b, n, h): (usize, usize, usize),
        x: &[f32],
        y_train: &[f32],
    ) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
        let mut grouped = Vec::with_capacity(b * h * n * 3);
        let mut column_labels = vec![0.0; b * n * COLUMN_DIM];
        let mut icl_labels = vec![0.0; b * n * ICL_DIM];
        for ensemble in 0..b {
            let Some(group) = crate::metal::group_features(&x[ensemble * n * h..][..n * h], n, h) else {
                return Ok(None);
            };
            grouped.extend(group);
            for (prefix, width, output) in [
                ("col_embedder", COLUMN_DIM, &mut column_labels),
                ("icl_predictor", ICL_DIM, &mut icl_labels),
            ] {
                self.target_embeddings(
                    prefix,
                    &y_train[ensemble * n..][..n],
                    width,
                    &mut output[ensemble * n * width..][..n * width],
                )?;
            }
        }
        Ok(Some((grouped, column_labels, icl_labels)))
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
    /// # Errors
    ///
    /// Returns an error for inconsistent dimensions or labels, invalid
    /// permutations, or when the context does not fit in memory.
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
    ) -> Result<StaticExecutor> {
        let shape = (ensembles, training_rows, features);
        let metadata = self.context_metadata(shape, x, y_train, class_permutations, feature_permutations)?;
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(tensors) = self.build_tensors_gpu(shape, x, y_train, optimize, limits)? {
            // The build graph checked every value on the GPU.
            let context = StaticContextArtifact::from_page_aligned(metadata, tensors)?;
            return StaticExecutor::with_executor(self.clone(), context, threads, limits);
        }
        // On a BF16-capable CPU, `fast` stores the ICL values in BF16 (FP32
        // keys, the retained serving layout); every other case is FP32.
        if optimize == Optimize::Fast && crate::kernels::native::bf16_storage_supported() {
            let tensors = self.build_tensors_cpu(shape, x, y_train, limits)?;
            let context = StaticContextArtifact::from_page_aligned(metadata, bf16v_page_aligned(&tensors)?)?;
            return StaticExecutor::with_executor(self.clone(), context, threads, limits);
        }
        let context =
            StaticContextArtifact::new(metadata, self.build_tensors_cpu(shape, x, y_train, limits)?)?;
        StaticExecutor::with_executor(self.clone(), context, threads, limits)
    }

    /// Validate a context's inputs and describe it: model and content hashes,
    /// shape, class count, and permutations.
    fn context_metadata(
        &self,
        (b, n, h): (usize, usize, usize),
        x: &[f32],
        y_train: &[f32],
        class_permutations: Vec<u64>,
        feature_permutations: Vec<u64>,
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
        for (byte, pair) in model_sha256.iter_mut().zip(source.chunks_exact(2)) {
            *byte = std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
                .ok_or_else(|| Error::Artifact("model source SHA-256 is not hexadecimal".into()))?;
        }
        let mut context_sha256 = Sha256::new();
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
        let mut column = (0..3)
            .map(|_| reserve(b * streams * column_elements))
            .collect::<Result<Vec<_>>>()?;
        let mut icl = (0..12)
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
                    let column = (0..3)
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
    pub fn predict_logits(&self, request: Request<'_>) -> Result<DecoderOutputs> {
        match self.predict_logits_inner(request, false)? {
            PredictionResult::Logits(logits) => Ok(logits),
            PredictionResult::Trace(_) => unreachable!("normal prediction cannot produce a trace"),
        }
    }

    /// Execute the classifier and retain coarse stage outputs for diagnostics.
    ///
    /// The trace is admitted with the same resource limit as normal execution,
    /// but stores all ensemble outputs. Use only small requests.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::predict_logits`], including a resource
    /// error when retained trace outputs exceed the activation limit.
    pub fn predict_logits_with_trace(&self, request: Request<'_>) -> Result<ForwardTrace> {
        match self.predict_logits_inner(request, true)? {
            PredictionResult::Logits(_) => unreachable!("traced prediction must produce a trace"),
            PredictionResult::Trace(trace) => Ok(trace),
        }
    }

    fn predict_logits_inner(&self, request: Request<'_>, trace: bool) -> Result<PredictionResult> {
        let (b, t, h, n) = (
            request.ensemble_count(),
            request.total_rows(),
            request.feature_count(),
            request.training_rows(),
        );
        let width = self.decoder_width()?;
        admit_request(b, t, h, n, width)?;
        if trace {
            admit_trace_request(b, t, h, width)?;
        }
        // A regression fit keeps every decoded quantile.
        let classes = match self.validate_targets(request.y_train(), b, n)? {
            0 => width,
            classes => classes,
        };
        let test = t.checked_sub(n).ok_or(Error::ArithmeticOverflow("test rows"))?;
        let output_len = checked_product(&[b, test, classes], "decoder output")?;
        let mut values = Vec::with_capacity(output_len);
        let mut column_values = trace.then(Vec::new);
        let mut row_values = trace.then(Vec::new);
        let mut decoder_values = trace.then(Vec::new);
        for ensemble in 0..b {
            let x = &request.x()[checked_product(&[ensemble, t, h], "feature offset")?..]
                [..checked_product(&[t, h], "feature rows")?];
            let labels = &request.y_train()[ensemble * n..][..n];
            let (columns, _) = self.column_embeddings(x, labels, t, h, n, false)?;
            let rows = self.row_embeddings(&columns, t, h + RESERVED)?;
            if let Some(trace_values) = &mut column_values {
                trace_values.extend_from_slice(&columns);
            }
            if let Some(trace_values) = &mut row_values {
                trace_values.extend_from_slice(&rows);
            }
            let logits = self.icl_logits(rows, labels, t, n)?;
            if let Some(trace_values) = &mut decoder_values {
                trace_values.extend_from_slice(&logits);
            }
            for row in n..t {
                values.extend_from_slice(&logits[row * width..][..classes]);
            }
        }
        let logits = DecoderOutputs {
            shape: [b, test, classes],
            values,
        };
        match (column_values, row_values, decoder_values) {
            (Some(column_embeddings), Some(row_representations), Some(decoder_logits)) => {
                Ok(PredictionResult::Trace(ForwardTrace {
                    column_embeddings: TraceStage {
                        shape: [b, t, h + RESERVED, COLUMN_DIM],
                        values: column_embeddings,
                    },
                    row_representations: TraceStage {
                        shape: [b, t, ICL_DIM],
                        values: row_representations,
                    },
                    decoder_logits: TraceStage {
                        shape: [b, t, width],
                        values: decoder_logits,
                    },
                    logits,
                }))
            }
            (None, None, None) => Ok(PredictionResult::Logits(logits)),
            _ => Err(Error::InvalidArgument("incomplete trace output".into())),
        }
    }

    /// Embed every feature stream. With `capture`, also return each feature's
    /// three `multihead_attn2` key/value pairs, in feature order.
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
        let workers = executor_threads().min(h);
        let chunk = h.div_ceil(workers);
        let y = y.as_slice();
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for start in (0..h).step_by(chunk) {
                let end = (start + chunk).min(h);
                handles.push(scope.spawn(move || -> Result<Vec<ColumnStream>> {
                    let mut results = Vec::with_capacity(end - start);
                    for feature in start..end {
                        let (embedded, caches) =
                            self.column_stream(x, y, t, h, n, feature, in_weight, in_bias, capture)?;
                        results.push((feature + RESERVED, embedded, caches));
                    }
                    Ok(results)
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("column worker must not panic"))
                .collect::<Result<Vec<_>>>()
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

    #[allow(clippy::too_many_arguments)]
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
        for block in 0..3 {
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
        let workers = executor_threads().min(t);
        let chunk = t.div_ceil(workers);
        let chunks = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for start in (0..t).step_by(chunk) {
                let end = (start + chunk).min(t);
                handles.push(scope.spawn(move || -> Result<Vec<f32>> {
                    let mut rows = Vec::with_capacity((end - start) * ICL_DIM);
                    for row in start..end {
                        rows.extend_from_slice(&self.row_embedding(columns, cls, row, streams)?);
                    }
                    Ok(rows)
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("row worker must not panic"))
                .collect::<Result<Vec<_>>>()
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
        for block in 0..12 {
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

    #[allow(clippy::too_many_arguments)]
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

    /// Attend from `q` to projected sources, then apply the residual feed-forward block.
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
        let heads = HEADS;
        let head_dim = dim / heads;
        let mut q_heads = tokens_to_heads(&q_proj, q_rows, dim, heads);
        if qass {
            let original = q_heads.clone();
            self.qass(
                &attn_prefix,
                &original,
                q_rows,
                source_rows,
                heads,
                head_dim,
                &mut q_heads,
            )?;
        }
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
                false,
                crate::kernels::native::attention_backend(),
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

    #[allow(clippy::cast_precision_loss)]
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
    /// Pair an official classifier model with its matching exact full-K/V context.
    ///
    /// # Errors
    ///
    /// Returns an artifact error if the model identity, architecture, or required
    /// cache tensor names and shapes do not match the serving contract, or when
    /// `IRONICL_MEMORY_LIMIT` or `IRONICL_DEVICE_MEMORY_LIMIT` is invalid.
    pub fn new(model: ModelArtifact, context: StaticContextArtifact) -> Result<Self> {
        Self::with_executor(
            Executor::new(model),
            context,
            None,
            MemoryLimits::resolve(None, None)?,
        )
    }

    /// Write this context to `path` atomically, for reuse by [`Executor::load_context`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Artifact`] if encoding or the atomic write fails.
    pub fn save_context(&self, path: &std::path::Path) -> Result<()> {
        self.context.write(path)
    }

    /// Pair a context with an executor whose model weights it shares.
    fn with_executor(
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
        let qass_base_scales = (0..12)
            .map(|block| {
                qass_base_scale(
                    &model,
                    &format!("icl_predictor.tf_icl.blocks.{block}.attn"),
                    context.metadata().training_rows,
                    ICL_DIM,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // The BF16/packed dense experiment is off unless built with the
        // `bf16-dense` feature; it is never activated by a runtime flag.
        let bf16_dense = cfg!(feature = "bf16-dense");
        let packed_f32_row = bf16_dense && packed_f32_affine_supported();
        let packed_f32_policy = bf16_dense.then(Bf16DensePolicy::full);
        let packed_f32_affines = if packed_f32_affine_supported() {
            let mut packed = BTreeMap::new();
            if let Some(policy) = packed_f32_policy {
                packed.extend(Self::build_packed_f32_affines(&model, policy)?);
            }
            if packed_f32_row {
                packed.extend(Self::build_packed_f32_row_affines(&model)?);
            }
            (!packed.is_empty()).then(|| Arc::new(packed))
        } else {
            None
        };
        let bf16_dense_policy = bf16_dense.then(Bf16DensePolicy::full);
        let bf16_dense_compensation = Bf16AffineCompensation::None;
        let column_bf16_blocks = [bf16_dense; 3];
        let icl_bf16_blocks = [bf16_dense; 12];
        let packed_bf16_affines = if let Some(policy) = bf16_dense_policy
            && packed_bf16_affine_supported()
        {
            Some(Arc::new(Self::build_packed_bf16_affines(
                &model,
                policy,
                column_bf16_blocks,
                icl_bf16_blocks,
            )?))
        } else {
            None
        };
        let affine_pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("ironicl-affine-{index}"))
            .build()
            .map_err(|error| Error::InvalidArgument(format!("cannot create affine worker pool: {error}")))?;
        Self {
            model,
            context,
            affine_pool: Arc::new(affine_pool),
            column_input_weight: Arc::new(column_input_weight),
            qass_base_scales: Arc::new(qass_base_scales),
            packed_f32_affines,
            packed_f32_row,
            packed_bf16_affines,
            bf16_dense_compensation,
            attention_backend: crate::kernels::native::attention_backend(),
            sparse_attention_stats: Arc::new(SparseAttentionStats::default()),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            gpu: None,
        }
        .with_gpu(limits)
    }

    /// Build the context's GPU predict graph when Metal is available, with
    /// queries split to keep its buckets within the device's share of memory
    /// under `limits`, as for builds. Packed contexts and contexts with mixed
    /// ICL storage stay on the CPU.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn with_gpu(mut self, limits: MemoryLimits) -> Result<Self> {
        if self.context.has_packed_icl_keys() {
            return Ok(self);
        }
        let model = &self.model;
        let weights = |name: &str| model.tensor(name);
        let Some(gpu_weights) = model.gpu_weights() else {
            return Ok(self);
        };
        let column = |kind: &str| -> Option<Vec<&[f32]>> {
            (0..3)
                .map(|block| {
                    self.context
                        .tensor(&format!("column.{block}.{kind}"))?
                        .fp32_values()
                })
                .collect()
        };
        // ICL caches in whatever type the context stores, as raw bytes.
        let icl = |kind: &str| -> Option<(Vec<&[u8]>, crate::static_context::TensorStorage)> {
            use crate::static_context::TensorStorage;
            let mut storage = None;
            let blocks = (0..12)
                .map(|block| {
                    let (bytes, block_storage) =
                        match self.context.tensor(&format!("icl.{block}.{kind}"))?.data {
                            StaticContextTensorData::Fp32(values) => {
                                (bytemuck::cast_slice(values), TensorStorage::Fp32)
                            }
                            StaticContextTensorData::F16(values) => {
                                (bytemuck::cast_slice(values), TensorStorage::F16)
                            }
                            StaticContextTensorData::Bf16(values) => {
                                (bytemuck::cast_slice(values), TensorStorage::Bf16)
                            }
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
        let predict = crate::metal::PredictGraph::new(
            &gpu_weights,
            &weights,
            crate::metal::Shape {
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
        self.gpu = Some(Arc::new(GpuContext {
            split: predict.query_split(budget),
            predict,
            _storage: self.context.page_aligned_storage().cloned(),
        }));
        Ok(self)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    #[allow(clippy::unnecessary_wraps)]
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
        gpu: &GpuContext,
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
                    crate::metal::group_features_into(&x[ensemble * q * h..][..q * h], (q, h), stride, input)
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

    /// Reset sparse-attention decisions collected by this executor.
    pub fn reset_sparse_attention_statistics(&self) {
        self.sparse_attention_stats.reset();
    }

    /// Return sparse-attention decisions collected by this executor.
    #[must_use]
    pub fn sparse_attention_statistics(&self) -> SparseAttentionStatistics {
        self.sparse_attention_stats.snapshot()
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
        let (b, q, h, classes) = (
            metadata.ensemble_count,
            request.query_rows(),
            metadata.feature_count,
            self.outputs()?,
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
        }
        Ok(DecoderOutputs {
            shape: [b, q, classes],
            values,
        })
    }

    /// How queries split into runs, `(chunk, small)`: the GPU's split within
    /// its budget, or chunks of [`QUERY_CHUNK_ROWS`] on the CPU.
    #[allow(clippy::unused_self)] // Without Metal.
    fn query_split(&self) -> (usize, usize) {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(gpu) = &self.gpu {
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
        if let Some(gpu) = &self.gpu
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

    /// Validate artifact identities and execute one owned query bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when the query references another model or static
    /// context, or when normal query execution fails.
    pub fn predict_bundle(&self, query: &StaticQueryBundle) -> Result<DecoderOutputs> {
        if query.model_sha256() != &self.context.metadata().model_sha256
            || query.context_sha256() != &self.context.metadata().context_sha256
        {
            return Err(Error::InvalidArgument(
                "query bundle model or context identity does not match the static context".into(),
            ));
        }
        self.predict_logits(query.request()?)
    }

    /// Validate artifact identities and return final aggregated probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error when the query references another model or static
    /// context, or when normal query execution fails.
    pub fn predict_bundle_probabilities(
        &self,
        query: &StaticQueryBundle,
        aggregation: ClassifierAggregation,
    ) -> Result<ClassifierProbabilities> {
        if query.model_sha256() != &self.context.metadata().model_sha256
            || query.context_sha256() != &self.context.metadata().context_sha256
        {
            return Err(Error::InvalidArgument(
                "query bundle model or context identity does not match the static context".into(),
            ));
        }
        self.predict_probabilities(query.request()?, aggregation)
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

    #[allow(clippy::too_many_arguments)]
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
        for block in 0..3 {
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
                None,
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
        for block in 0..12 {
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
        let sparse_index = self.sparse_index(ensemble, block)?;
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
            sparse_index,
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
        self.parallel_affine_named(
            "icl_predictor.decoder.0.weight",
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

    #[allow(clippy::too_many_arguments)]
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
        sparse_index: Option<SparseIndex<'_>>,
        workers: usize,
    ) -> Result<Vec<f32>> {
        let weight = self.tensor(&format!("{attn_prefix}.in_proj_weight"))?;
        let bias = self.tensor(&format!("{attn_prefix}.in_proj_bias"))?;
        let mut q_proj = vec![0.0; q.len()];
        let q_name = format!("{attn_prefix}.in_proj_weight.q");
        let mut q_norm = vec![0.0; q.len()];
        layer_norm_f32(
            q,
            q_rows,
            dim,
            self.tensor(&format!("{prefix}.norm1.weight"))?,
            Some(self.tensor(&format!("{prefix}.norm1.bias"))?),
            &mut q_norm,
        )?;
        self.parallel_affine_named(
            &q_name,
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
                        self.attention_backend,
                        &q_heads,
                        q_rows,
                        key,
                        value,
                        source_rows,
                        head_dim,
                        self.context.has_packed_icl_keys(),
                        &mut attended,
                        workers,
                    )?;
                }
                (StaticContextTensorData::Bf16(key), StaticContextTensorData::Bf16(value)) => {
                    parallel_cached_attention_bf16(
                        self.attention_backend,
                        &q_heads,
                        q_rows,
                        key,
                        value,
                        source_rows,
                        head_dim,
                        self.context.has_packed_icl_keys(),
                        &mut attended,
                        workers,
                    )?;
                }
                (StaticContextTensorData::Fp32(key), StaticContextTensorData::Bf16(value)) => {
                    let used_sparse = if let Some(index) = sparse_index {
                        sparse_cached_attention_f32_key_bf16_value(
                            &q_heads,
                            q_rows,
                            key,
                            value,
                            source_rows,
                            head_dim,
                            self.context.has_packed_icl_keys(),
                            index,
                            &mut attended,
                        )?
                    } else {
                        false
                    };
                    if sparse_index.is_some() {
                        self.sparse_attention_stats.record(used_sparse);
                    }
                    if !used_sparse {
                        parallel_cached_attention_f32_key_bf16_value(
                            self.attention_backend,
                            &q_heads,
                            q_rows,
                            key,
                            value,
                            source_rows,
                            head_dim,
                            self.context.has_packed_icl_keys(),
                            &mut attended,
                            workers,
                        )?;
                    }
                }
                // Only queries the GPU hands back reach this: widen the
                // block's FP16 caches and run the FP32 kernel.
                (StaticContextTensorData::F16(key), StaticContextTensorData::F16(value)) => {
                    let widen =
                        |values: &[u16]| values.iter().map(|&bits| f16_to_f32(bits)).collect::<Vec<_>>();
                    parallel_cached_attention(
                        self.attention_backend,
                        &q_heads,
                        q_rows,
                        &widen(key),
                        &widen(value),
                        source_rows,
                        head_dim,
                        self.context.has_packed_icl_keys(),
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
            if column_source_major_attention_enabled(self.attention_backend) {
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
                    false,
                    self.attention_backend,
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
        self.parallel_affine_named(
            &format!("{attn_prefix}.out_proj.weight"),
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
        self.parallel_affine_named(
            &ff_name,
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
        self.parallel_affine_named(
            &format!("{prefix}.linear2.weight"),
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

    #[allow(clippy::too_many_arguments)]
    fn parallel_affine_named(
        &self,
        name: &str,
        input: &[f32],
        rows: usize,
        input_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output_dim: usize,
        output: &mut [f32],
        workers: usize,
    ) -> Result<()> {
        if let Some(packed) = self
            .packed_bf16_affines
            .as_deref()
            .and_then(|affines| affines.get(name))
        {
            return self.parallel_packed_bf16_affine(
                packed, input, rows, input_dim, weight, bias, output_dim, output, workers,
            );
        }
        if let Some(packed) = self
            .packed_f32_affines
            .as_deref()
            .and_then(|affines| affines.get(name))
        {
            return self.parallel_packed_f32_affine(
                packed, input, rows, input_dim, weight, bias, output_dim, output, workers,
            );
        }
        self.parallel_affine(input, rows, input_dim, weight, bias, output_dim, output, workers)
    }

    #[allow(clippy::too_many_arguments)]
    fn parallel_packed_bf16_affine(
        &self,
        packed: &PackedBf16Affine,
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
        if workers <= 1 || rows < 16 {
            if Self::run_packed_bf16_affine(packed, input, rows, bias, output, self.bf16_dense_compensation) {
                return Ok(());
            }
            return affine_f32(input, rows, input_dim, weight, bias, output_dim, output);
        }
        let rows_per_job = rows.div_ceil(workers);
        self.affine_pool.install(|| {
            input
                .par_chunks(input_dim * rows_per_job)
                .zip(output.par_chunks_mut(output_dim * rows_per_job))
                .try_for_each(|(input, output)| {
                    let chunk_rows = input.len() / input_dim;
                    if Self::run_packed_bf16_affine(
                        packed,
                        input,
                        chunk_rows,
                        bias,
                        output,
                        self.bf16_dense_compensation,
                    ) {
                        Ok(())
                    } else {
                        affine_f32(input, chunk_rows, input_dim, weight, bias, output_dim, output)
                    }
                })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn parallel_packed_f32_affine(
        &self,
        packed: &PackedF32Affine,
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
        if workers <= 1 || rows < 16 {
            if packed.run(input, rows, bias, output) {
                return Ok(());
            }
            return affine_f32(input, rows, input_dim, weight, bias, output_dim, output);
        }
        let rows_per_job = rows.div_ceil(workers);
        self.affine_pool.install(|| {
            input
                .par_chunks(input_dim * rows_per_job)
                .zip(output.par_chunks_mut(output_dim * rows_per_job))
                .try_for_each(|(input, output)| {
                    let chunk_rows = input.len() / input_dim;
                    if packed.run(input, chunk_rows, bias, output) {
                        Ok(())
                    } else {
                        affine_f32(input, chunk_rows, input_dim, weight, bias, output_dim, output)
                    }
                })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_packed_bf16_affine(
        packed: &PackedBf16Affine,
        input: &[f32],
        rows: usize,
        bias: Option<&[f32]>,
        output: &mut [f32],
        compensation: Bf16AffineCompensation,
    ) -> bool {
        BF16_AFFINE_INPUT.with(|input_storage| {
            let mut input_bf16 = input_storage.borrow_mut();
            input_bf16.resize(input.len(), 0);
            if compensation == Bf16AffineCompensation::None {
                return packed.run_f32(input, rows, bias, output, &mut input_bf16);
            }
            if matches!(
                compensation,
                Bf16AffineCompensation::Input | Bf16AffineCompensation::FirstOrder
            ) {
                return BF16_AFFINE_INPUT_RESIDUAL.with(|residual_storage| {
                    let mut input_residual_bf16 = residual_storage.borrow_mut();
                    input_residual_bf16.resize(input.len(), 0);
                    packed.run_f32_compensated(
                        input,
                        rows,
                        bias,
                        output,
                        &mut input_bf16,
                        &mut input_residual_bf16,
                        compensation,
                    )
                });
            }
            packed.run_f32_compensated(input, rows, bias, output, &mut input_bf16, &mut [], compensation)
        })
    }

    #[allow(clippy::too_many_arguments)]
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
        if workers <= 1 || rows < 16 || crate::kernels::native::affine_uses_sgemm(rows) {
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
        if self.packed_f32_row && !qass && rope && dim == COLUMN_DIM {
            return self.packed_row_block(q, source, q_rows, source_rows, prefix);
        }
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

    fn packed_row_block(
        &self,
        q: &[f32],
        source: &[f32],
        q_rows: usize,
        source_rows: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Block);
        let norm_weight = self.tensor(&format!("{prefix}.norm1.weight"))?;
        let norm_bias = self.tensor(&format!("{prefix}.norm1.bias"))?;
        let mut q_norm = vec![0.0; q.len()];
        layer_norm_f32(q, q_rows, COLUMN_DIM, norm_weight, Some(norm_bias), &mut q_norm)?;
        let mut source_norm = vec![0.0; source.len()];
        layer_norm_f32(
            source,
            source_rows,
            COLUMN_DIM,
            norm_weight,
            Some(norm_bias),
            &mut source_norm,
        )?;

        let attention = format!("{prefix}.attn");
        let projection_name = format!("{attention}.in_proj_weight");
        let weight = self.tensor(&projection_name)?;
        let bias = self.tensor(&format!("{attention}.in_proj_bias"))?;
        let matrix_elements = COLUMN_DIM * COLUMN_DIM;
        let mut q_projection = vec![0.0; q.len()];
        self.parallel_affine_named(
            &format!("{projection_name}.q"),
            &q_norm,
            q_rows,
            COLUMN_DIM,
            &weight[..matrix_elements],
            Some(&bias[..COLUMN_DIM]),
            COLUMN_DIM,
            &mut q_projection,
            1,
        )?;
        let mut k_projection = vec![0.0; source.len()];
        self.parallel_affine_named(
            &format!("{projection_name}.k"),
            &source_norm,
            source_rows,
            COLUMN_DIM,
            &weight[matrix_elements..2 * matrix_elements],
            Some(&bias[COLUMN_DIM..2 * COLUMN_DIM]),
            COLUMN_DIM,
            &mut k_projection,
            1,
        )?;
        let mut v_projection = vec![0.0; source.len()];
        self.parallel_affine_named(
            &format!("{projection_name}.v"),
            &source_norm,
            source_rows,
            COLUMN_DIM,
            &weight[2 * matrix_elements..],
            Some(&bias[2 * COLUMN_DIM..]),
            COLUMN_DIM,
            &mut v_projection,
            1,
        )?;

        let mut q_heads = tokens_to_heads(&q_projection, q_rows, COLUMN_DIM, HEADS);
        let mut k_heads = tokens_to_heads(&k_projection, source_rows, COLUMN_DIM, HEADS);
        let v_heads = tokens_to_heads(&v_projection, source_rows, COLUMN_DIM, HEADS);
        let frequencies = self.tensor("row_interactor.tf_row.rope.freqs")?;
        let mut rotated = vec![0.0; q_heads.len()];
        rope_non_interleaved_f32(&q_heads, HEADS, q_rows, 16, frequencies, 0, &mut rotated)?;
        q_heads = rotated;
        let mut rotated = vec![0.0; k_heads.len()];
        rope_non_interleaved_f32(&k_heads, HEADS, source_rows, 16, frequencies, 0, &mut rotated)?;
        k_heads = rotated;

        let scratch_rows = q_rows.min(ATTENTION_QUERY_BLOCK_ROWS);
        let scratch_sources = source_rows.min(ATTENTION_KEY_TILE_ROWS);
        let mut scores = vec![0.0; attention_scratch_scores(scratch_rows, scratch_sources)?];
        let mut maximum = vec![0.0; attention_scratch_state(scratch_rows)?];
        let mut denominator = vec![0.0; attention_scratch_state(scratch_rows)?];
        let mut attended = vec![0.0; q_heads.len()];
        tiled_online_scaled_dot_product_attention_f32(
            &q_heads,
            q_rows,
            &k_heads,
            &v_heads,
            source_rows,
            HEADS,
            16,
            ATTENTION_QUERY_BLOCK_ROWS,
            ATTENTION_KEY_TILE_ROWS,
            &mut OnlineAttentionScratch {
                scores: &mut scores,
                maximum: &mut maximum,
                denominator: &mut denominator,
            },
            &mut attended,
        )?;
        let merged = heads_to_tokens(&attended, q_rows, COLUMN_DIM, HEADS);
        let output_weight_name = format!("{attention}.out_proj.weight");
        let mut projected = vec![0.0; q.len()];
        self.parallel_affine_named(
            &output_weight_name,
            &merged,
            q_rows,
            COLUMN_DIM,
            self.tensor(&output_weight_name)?,
            Some(self.tensor(&format!("{attention}.out_proj.bias"))?),
            COLUMN_DIM,
            &mut projected,
            1,
        )?;
        let mut residual = q
            .iter()
            .zip(projected)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let mut ff_norm = vec![0.0; residual.len()];
        layer_norm_f32(
            &residual,
            q_rows,
            COLUMN_DIM,
            self.tensor(&format!("{prefix}.norm2.weight"))?,
            Some(self.tensor(&format!("{prefix}.norm2.bias"))?),
            &mut ff_norm,
        )?;
        let first_weight_name = format!("{prefix}.linear1.weight");
        let mut hidden = vec![0.0; q_rows * COLUMN_DIM * 2];
        self.parallel_affine_named(
            &first_weight_name,
            &ff_norm,
            q_rows,
            COLUMN_DIM,
            self.tensor(&first_weight_name)?,
            Some(self.tensor(&format!("{prefix}.linear1.bias"))?),
            COLUMN_DIM * 2,
            &mut hidden,
            1,
        )?;
        let mut activated = vec![0.0; hidden.len()];
        self.parallel_gelu(&hidden, &mut activated, 1)?;
        let second_weight_name = format!("{prefix}.linear2.weight");
        let mut addition = vec![0.0; q.len()];
        self.parallel_affine_named(
            &second_weight_name,
            &activated,
            q_rows,
            COLUMN_DIM * 2,
            self.tensor(&second_weight_name)?,
            Some(self.tensor(&format!("{prefix}.linear2.bias"))?),
            COLUMN_DIM,
            &mut addition,
            1,
        )?;
        for (value, addition) in residual.iter_mut().zip(addition) {
            *value += addition;
        }
        Ok(residual)
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

    fn sparse_index(&self, ensemble: usize, layer: usize) -> Result<Option<SparseIndex<'_>>> {
        if !sparse_attention_enabled() {
            return Ok(None);
        }
        let Some(centroids) = self.context.tensor("icl.index.centroids") else {
            return Ok(None);
        };
        let radii = self
            .context
            .tensor("icl.index.radii")
            .ok_or_else(|| Error::Artifact("static context sparse index is incomplete".into()))?;
        let offsets = self
            .context
            .tensor("icl.index.offsets")
            .ok_or_else(|| Error::Artifact("static context sparse index is incomplete".into()))?;
        let members = self
            .context
            .tensor("icl.index.members")
            .ok_or_else(|| Error::Artifact("static context sparse index is incomplete".into()))?;
        let centroids = centroids
            .fp32_values()
            .ok_or_else(|| Error::Artifact("static context sparse index centroids must be FP32".into()))?;
        let radii = radii
            .fp32_values()
            .ok_or_else(|| Error::Artifact("static context sparse index radii must be FP32".into()))?;
        let offsets = offsets
            .fp32_values()
            .ok_or_else(|| Error::Artifact("static context sparse index offsets must be FP32".into()))?;
        let members = members
            .fp32_values()
            .ok_or_else(|| Error::Artifact("static context sparse index members must be FP32".into()))?;
        let clusters = usize::try_from(
            self.context
                .tensor("icl.index.centroids")
                .expect("index exists")
                .dimensions[3],
        )
        .map_err(|_| Error::Artifact("static context sparse index cluster count overflows usize".into()))?;
        let group = ensemble * 12 + layer;
        let head_centroids = HEADS * clusters * 64;
        let head_radii = HEADS * clusters;
        let head_offsets = HEADS * (clusters + 1);
        let head_members = HEADS * self.context.metadata().training_rows;
        Ok(Some(SparseIndex {
            centroids: &centroids[group * head_centroids..][..head_centroids],
            radii: &radii[group * head_radii..][..head_radii],
            offsets: &offsets[group * head_offsets..][..head_offsets],
            members: &members[group * head_members..][..head_members],
            clusters,
        }))
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

    fn build_packed_f32_row_affines(model: &Executor) -> Result<BTreeMap<String, PackedF32Affine>> {
        let mut packed = BTreeMap::new();
        for block in 0..3 {
            let prefix = format!("row_interactor.tf_row.blocks.{block}");
            let attention = format!("{prefix}.attn");
            let projection_name = format!("{attention}.in_proj_weight");
            let projection = model.tensor(&projection_name)?;
            let matrix_elements = COLUMN_DIM * COLUMN_DIM;
            for (part, name) in ["q", "k", "v"].into_iter().enumerate() {
                let start = part * matrix_elements;
                let weight = projection.get(start..start + matrix_elements).ok_or_else(|| {
                    Error::Artifact(format!(
                        "model tensor '{projection_name}' is too short for {name} projection"
                    ))
                })?;
                Self::pack_f32_affine(
                    &mut packed,
                    format!("{projection_name}.{name}"),
                    weight,
                    COLUMN_DIM,
                    COLUMN_DIM,
                )?;
            }
            Self::pack_f32_model_affine(
                &mut packed,
                model,
                &format!("{attention}.out_proj.weight"),
                COLUMN_DIM,
                COLUMN_DIM,
            )?;
            Self::pack_f32_model_affine(
                &mut packed,
                model,
                &format!("{prefix}.linear1.weight"),
                COLUMN_DIM,
                COLUMN_DIM * 2,
            )?;
            Self::pack_f32_model_affine(
                &mut packed,
                model,
                &format!("{prefix}.linear2.weight"),
                COLUMN_DIM * 2,
                COLUMN_DIM,
            )?;
        }
        Ok(packed)
    }

    fn build_packed_f32_affines(
        model: &Executor,
        policy: Bf16DensePolicy,
    ) -> Result<BTreeMap<String, PackedF32Affine>> {
        let mut packed = BTreeMap::new();
        if policy.column_q || policy.column_out || policy.column_ffn1 || policy.column_ffn2 {
            for block in 0..3 {
                let prefix = format!("col_embedder.tf_col.blocks.{block}.multihead_attn2");
                Self::pack_f32_cached_block(
                    &mut packed,
                    model,
                    &prefix,
                    &format!("{prefix}.attn"),
                    COLUMN_DIM,
                    policy.column_q,
                    policy.column_out,
                    policy.column_ffn1,
                    policy.column_ffn2,
                )?;
            }
        }
        for block in 0..12 {
            let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
            let attention = format!("{prefix}.attn");
            Self::pack_f32_cached_block(
                &mut packed,
                model,
                &prefix,
                &attention,
                ICL_DIM,
                policy.icl_q,
                policy.icl_out,
                policy.icl_ffn1,
                policy.icl_ffn2,
            )?;
            if policy.qass {
                Self::pack_f32_model_affine(
                    &mut packed,
                    model,
                    &format!("{attention}.ssmax_layer.query_mlp.0.weight"),
                    64,
                    64,
                )?;
                Self::pack_f32_model_affine(
                    &mut packed,
                    model,
                    &format!("{attention}.ssmax_layer.query_mlp.2.weight"),
                    64,
                    64,
                )?;
            }
        }
        if policy.decoder {
            Self::pack_f32_model_affine(
                &mut packed,
                model,
                "icl_predictor.decoder.0.weight",
                ICL_DIM,
                1024,
            )?;
        }
        Ok(packed)
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn pack_f32_cached_block(
        packed: &mut BTreeMap<String, PackedF32Affine>,
        model: &Executor,
        prefix: &str,
        attention: &str,
        dim: usize,
        pack_q: bool,
        pack_out: bool,
        pack_ffn1: bool,
        pack_ffn2: bool,
    ) -> Result<()> {
        if pack_q {
            let projection_name = format!("{attention}.in_proj_weight");
            let projection = model.tensor(&projection_name)?;
            let q_elements = dim
                .checked_mul(dim)
                .ok_or(Error::ArithmeticOverflow("packed FP32 Q projection"))?;
            let q_weight = projection.get(..q_elements).ok_or_else(|| {
                Error::Artifact(format!(
                    "model tensor '{projection_name}' is too short for Q projection"
                ))
            })?;
            Self::pack_f32_affine(packed, format!("{projection_name}.q"), q_weight, dim, dim)?;
        }
        if pack_out {
            Self::pack_f32_model_affine(packed, model, &format!("{attention}.out_proj.weight"), dim, dim)?;
        }
        if pack_ffn1 {
            Self::pack_f32_model_affine(packed, model, &format!("{prefix}.linear1.weight"), dim, dim * 2)?;
        }
        if pack_ffn2 {
            Self::pack_f32_model_affine(packed, model, &format!("{prefix}.linear2.weight"), dim * 2, dim)?;
        }
        Ok(())
    }

    fn pack_f32_model_affine(
        packed: &mut BTreeMap<String, PackedF32Affine>,
        model: &Executor,
        name: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        Self::pack_f32_affine(
            packed,
            name.to_owned(),
            model.tensor(name)?,
            input_dim,
            output_dim,
        )
    }

    fn pack_f32_affine(
        packed: &mut BTreeMap<String, PackedF32Affine>,
        name: String,
        weight: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        let affine = PackedF32Affine::new(weight, input_dim, output_dim).ok_or_else(|| {
            Error::Artifact(format!(
                "model tensor '{name}' cannot be packed as FP32 [{output_dim}, {input_dim}]"
            ))
        })?;
        packed.insert(name, affine);
        Ok(())
    }

    fn build_packed_bf16_affines(
        model: &Executor,
        policy: Bf16DensePolicy,
        column_blocks: [bool; 3],
        icl_blocks: [bool; 12],
    ) -> Result<BTreeMap<String, PackedBf16Affine>> {
        let mut packed = BTreeMap::new();
        if policy.column_q || policy.column_out || policy.column_ffn1 || policy.column_ffn2 {
            for (block, enabled) in column_blocks.into_iter().enumerate() {
                if !enabled {
                    continue;
                }
                let prefix = format!("col_embedder.tf_col.blocks.{block}.multihead_attn2");
                Self::pack_cached_block_affines(
                    &mut packed,
                    model,
                    &prefix,
                    &format!("{prefix}.attn"),
                    COLUMN_DIM,
                    policy.column_q,
                    policy.column_out,
                    policy.column_ffn1,
                    policy.column_ffn2,
                )?;
            }
        }
        for (block, enabled) in icl_blocks.into_iter().enumerate() {
            if !enabled {
                continue;
            }
            let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
            let attention = format!("{prefix}.attn");
            Self::pack_cached_block_affines(
                &mut packed,
                model,
                &prefix,
                &attention,
                ICL_DIM,
                policy.icl_q,
                policy.icl_out,
                policy.icl_ffn1,
                policy.icl_ffn2,
            )?;
            if policy.qass {
                Self::pack_model_affine(
                    &mut packed,
                    model,
                    &format!("{attention}.ssmax_layer.query_mlp.0.weight"),
                    64,
                    64,
                )?;
                Self::pack_model_affine(
                    &mut packed,
                    model,
                    &format!("{attention}.ssmax_layer.query_mlp.2.weight"),
                    64,
                    64,
                )?;
            }
        }
        if policy.decoder {
            Self::pack_model_affine(
                &mut packed,
                model,
                "icl_predictor.decoder.0.weight",
                ICL_DIM,
                1024,
            )?;
        }
        Ok(packed)
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn pack_cached_block_affines(
        packed: &mut BTreeMap<String, PackedBf16Affine>,
        model: &Executor,
        prefix: &str,
        attention: &str,
        dim: usize,
        pack_q: bool,
        pack_out: bool,
        pack_ffn1: bool,
        pack_ffn2: bool,
    ) -> Result<()> {
        if pack_q {
            let projection_name = format!("{attention}.in_proj_weight");
            let projection = model.tensor(&projection_name)?;
            let q_elements = dim
                .checked_mul(dim)
                .ok_or(Error::ArithmeticOverflow("packed BF16 Q projection"))?;
            let q_weight = projection.get(..q_elements).ok_or_else(|| {
                Error::Artifact(format!(
                    "model tensor '{projection_name}' is too short for Q projection"
                ))
            })?;
            Self::pack_affine(packed, format!("{projection_name}.q"), q_weight, dim, dim)?;
        }
        if pack_out {
            Self::pack_model_affine(packed, model, &format!("{attention}.out_proj.weight"), dim, dim)?;
        }
        if pack_ffn1 {
            Self::pack_model_affine(packed, model, &format!("{prefix}.linear1.weight"), dim, dim * 2)?;
        }
        if pack_ffn2 {
            Self::pack_model_affine(packed, model, &format!("{prefix}.linear2.weight"), dim * 2, dim)?;
        }
        Ok(())
    }

    fn pack_model_affine(
        packed: &mut BTreeMap<String, PackedBf16Affine>,
        model: &Executor,
        name: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        Self::pack_affine(
            packed,
            name.to_owned(),
            model.tensor(name)?,
            input_dim,
            output_dim,
        )
    }

    fn pack_affine(
        packed: &mut BTreeMap<String, PackedBf16Affine>,
        name: String,
        weight: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        let affine = PackedBf16Affine::new(weight, input_dim, output_dim).ok_or_else(|| {
            Error::Artifact(format!(
                "model tensor '{name}' cannot be packed as BF16 [{output_dim}, {input_dim}]"
            ))
        })?;
        packed.insert(name, affine);
        Ok(())
    }
}

/// `QASSMax` base scales for `source` keys: `[heads * head_dim]`, the same for
/// every query, so they are computed once per source length.
#[allow(clippy::cast_precision_loss)] // ln(source) needs no more precision.
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
    #[allow(clippy::cast_precision_loss)]
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
        self.parallel_affine_named(
            &query_first_name,
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
        self.parallel_affine_named(
            &query_second_name,
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
    model_hash == crate::artifact::encode_sha256(context_hash).as_bytes()
}

/// A context's cache tensors in order: each column block's key and value as
/// `[B, H + 4, 8, 128, 16]`, then each ICL block's as `[B, 8, N, 64]`.
/// Pack FP32 context tensors into a page-aligned context whose ICL values are
/// stored in BF16 (FP32 keys), the retained CPU serving layout for `fast`.
fn bf16v_page_aligned(tensors: &[StaticContextTensor]) -> Result<crate::static_context::PageAlignedTensors> {
    use crate::static_context::{PageAlignedTensors, TensorStorage};
    let is_bf16 = |name: &str| name.starts_with("icl.") && name.rsplit('.').next() == Some("value");
    let mut page = PageAlignedTensors::new(
        tensors
            .iter()
            .map(|tensor| {
                let storage = if is_bf16(&tensor.name) {
                    TensorStorage::Bf16
                } else {
                    TensorStorage::Fp32
                };
                (tensor.name.clone(), tensor.dimensions.clone(), storage)
            })
            .collect(),
    )?;
    for (index, tensor) in tensors.iter().enumerate() {
        if tensor.values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "static context tensor has a non-finite value".into(),
            ));
        }
        let bytes: Vec<u8> = if is_bf16(&tensor.name) {
            tensor
                .values
                .iter()
                .flat_map(|value| crate::kernels::native::f32_to_bf16_rne(*value).to_le_bytes())
                .collect()
        } else {
            tensor
                .values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        page.fill(index, &bytes);
    }
    Ok(page)
}

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
    }
    validate_sparse_index_contract(context)
}

fn executor_threads() -> usize {
    *EXECUTOR_THREADS.get_or_init(|| {
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
#[allow(unsafe_code)] // One libSystem call; std has no portable equivalent.
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
    // SAFETY: the name is NUL-terminated and `cores` holds `length` bytes.
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

fn column_source_major_attention_enabled(backend: crate::kernels::native::AttentionBackend) -> bool {
    backend == crate::kernels::native::AttentionBackend::Avx512
}

fn row_source_major_attention_enabled() -> bool {
    crate::kernels::native::attention_backend() == crate::kernels::native::AttentionBackend::Avx512
}

const fn sparse_attention_enabled() -> bool {
    cfg!(feature = "sparse-attention")
}

fn static_execution_plan(ensembles: usize, total_threads: usize) -> ExecutionPlan {
    ExecutionPlan::automatic(ensembles, total_threads, false)
}

#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn sparse_cached_attention_f32_key_bf16_value(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    key_is_packed: bool,
    index: SparseIndex<'_>,
    output: &mut [f32],
) -> Result<bool> {
    const MAX_CANDIDATE_FRACTION: usize = 4;
    if !key_is_packed
        || head_dim != 64
        || source_rows == 0
        || !source_rows.is_multiple_of(ATTENTION_KEY_TILE_ROWS)
    {
        return Ok(false);
    }
    let expected = HEADS * query_rows * head_dim;
    if query.len() != expected || key.len() != HEADS * source_rows * head_dim || value.len() != key.len() {
        return Err(Error::InvalidArgument(
            "sparse cached attention dimensions do not match".into(),
        ));
    }
    let scale = (head_dim as f32).sqrt().recip();
    for head in 0..HEADS {
        let centroids = &index.centroids[head * index.clusters * head_dim..][..index.clusters * head_dim];
        let radii = &index.radii[head * index.clusters..][..index.clusters];
        let offsets =
            &index.offsets[head * (index.clusters + 1)..=head * (index.clusters + 1) + index.clusters];
        let members = &index.members[head * source_rows..][..source_rows];
        for row in 0..query_rows {
            let query_row = &query[(head * query_rows + row) * head_dim..][..head_dim];
            let query_norm = query_row.iter().map(|value| value * value).sum::<f32>().sqrt();
            if !query_norm.is_finite() {
                return Ok(false);
            }
            let mut bounds = (0..index.clusters)
                .map(|cluster| {
                    let centroid = &centroids[cluster * head_dim..][..head_dim];
                    let dot = query_row
                        .iter()
                        .zip(centroid)
                        .fold(0.0_f32, |sum, (left, right)| left.mul_add(*right, sum));
                    ((dot + query_norm * radii[cluster]) * scale, cluster)
                })
                .collect::<Vec<_>>();
            if bounds.iter().any(|(bound, _)| !bound.is_finite()) {
                return Ok(false);
            }
            bounds.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
            let mut selected = vec![false; index.clusters];
            let mut candidate_scores = Vec::new();
            let mut accepted = false;
            for (_, cluster) in bounds.iter().copied() {
                selected[cluster] = true;
                let start = offsets[cluster] as usize;
                let end = offsets[cluster + 1] as usize;
                for member in &members[start..end] {
                    let source = *member as usize;
                    let tile_start = source / ATTENTION_KEY_TILE_ROWS * ATTENTION_KEY_TILE_ROWS;
                    let tile_rows = (source_rows - tile_start).min(ATTENTION_KEY_TILE_ROWS);
                    let local_source = source - tile_start;
                    let key_start = head * source_rows * head_dim + tile_start * head_dim;
                    let score = query_row
                        .iter()
                        .enumerate()
                        .fold(0.0_f32, |sum, (dimension, query)| {
                            let key = key[key_start + dimension * tile_rows + local_source];
                            query.mul_add(key, sum)
                        })
                        * scale;
                    candidate_scores.push((source, score));
                }
                if candidate_scores.len() > source_rows / MAX_CANDIDATE_FRACTION {
                    return Ok(false);
                }
                let maximum = candidate_scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denominator = candidate_scores
                    .iter()
                    .map(|(_, score)| (*score - maximum).exp())
                    .sum::<f32>();
                let omitted = bounds.iter().filter(|(_, cluster)| !selected[*cluster]).try_fold(
                    0.0_f32,
                    |sum, (bound, cluster)| {
                        let rows = offsets[*cluster + 1] - offsets[*cluster];
                        let contribution = rows * (*bound - maximum).exp();
                        (sum + contribution).is_finite().then_some(sum + contribution)
                    },
                );
                let Some(omitted) = omitted else {
                    return Ok(false);
                };
                if denominator.is_finite() && omitted <= denominator * sparse_max_omitted_mass() {
                    // Dense fallback resets the complete buffer, so direct writes avoid a
                    // temporary allocation even if a later head rejects sparse execution.
                    let output_row = &mut output[(head * query_rows + row) * head_dim..][..head_dim];
                    output_row.fill(0.0);
                    for (source, score) in &candidate_scores {
                        let weight = (*score - maximum).exp() / denominator;
                        let value_row = &value[(head * source_rows + source) * head_dim..][..head_dim];
                        for dimension in 0..head_dim {
                            output_row[dimension] +=
                                weight * f32::from_bits(u32::from(value_row[dimension]) << 16);
                        }
                    }
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

const fn sparse_max_omitted_mass() -> f32 {
    // The sparse-attention experiment's omitted-mass certificate (see
    // docs/experiments.md). Tune by editing this constant and rebuilding.
    1e-5
}

fn parallel_cached_attention(
    backend: crate::kernels::native::AttentionBackend,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    head_dim: usize,
    key_is_packed: bool,
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
            key_is_packed,
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
                    key_is_packed,
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

fn parallel_cached_attention_bf16(
    backend: crate::kernels::native::AttentionBackend,
    query: &[f32],
    query_rows: usize,
    key: &[u16],
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    key_is_packed: bool,
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
        return source_major_online_attention_bf16_with_backend_f32(
            query,
            query_rows,
            key,
            value,
            source_rows,
            HEADS,
            head_dim,
            ATTENTION_QUERY_BLOCK_ROWS,
            key_tile_rows,
            key_is_packed,
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
                source_major_online_attention_bf16_with_backend_f32(
                    query,
                    query_rows,
                    key,
                    value,
                    source_rows,
                    heads,
                    head_dim,
                    ATTENTION_QUERY_BLOCK_ROWS,
                    key_tile_rows,
                    key_is_packed,
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
                .expect("cached BF16 attention worker must not panic")?;
        }
        Ok(())
    })
}

fn parallel_cached_attention_f32_key_bf16_value(
    backend: crate::kernels::native::AttentionBackend,
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[u16],
    source_rows: usize,
    head_dim: usize,
    key_is_packed: bool,
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
            key_is_packed,
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
                    key_is_packed,
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
    let bf16 = crate::kernels::native::bf16_storage_supported();
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
fn shared_gpu_pools(gpu: &crate::metal::Gpu, limits: MemoryLimits) -> Pools {
    let working_set = u64::try_from(gpu.working_set_bytes()).unwrap_or(u64::MAX);
    Pools {
        device: Some(crate::memory::DeviceMemory::Shared { limit: working_set }),
        ..Pools::detect().with_limits(limits)
    }
}

/// Metal's recommended working set when the GPU would build `rows` training
/// rows, or `None` when the CPU builds: no GPU, `IRONICL_NO_METAL` set, or too
/// few rows.
#[allow(unused_variables)]
fn gpu_working_set(rows: usize) -> Option<u64> {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if std::env::var_os("IRONICL_NO_METAL").is_none() && rows >= crate::metal::GPU_MIN_BUILD_ROWS {
        return crate::metal::system_working_set();
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

/// Trace buffers retain every ensemble's coarse stage output at once.
fn admit_trace_request(b: usize, t: usize, h: usize, width: usize) -> Result<()> {
    let streams = h
        .checked_add(RESERVED)
        .ok_or(Error::ArithmeticOverflow("column streams"))?;
    let trace_elements = checked_sum(
        &[
            checked_product(&[b, t, streams, COLUMN_DIM], "trace column embeddings")?,
            checked_product(&[b, t, ICL_DIM], "trace row representations")?,
            checked_product(&[b, t, width], "trace decoder logits")?,
        ],
        "trace outputs",
    )?;
    // Reserve half the global limit for the normal forward graph, whose peak
    // overlaps retained trace buffers while the final ensemble is executing.
    if trace_elements > MAX_ACTIVATION_ELEMENTS / 2 {
        return Err(Error::InvalidArgument(format!(
            "trace outputs exceed half the uncached scalar activation limit of {MAX_ACTIVATION_BYTES} bytes"
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

#[allow(clippy::cast_precision_loss)]
fn validate_sparse_index_contract(context: &StaticContextArtifact) -> Result<()> {
    let present = SPARSE_INDEX_TENSORS.map(|name| context.tensor(name));
    if present.iter().all(Option::is_none) {
        return Ok(());
    }
    if present.iter().any(Option::is_none) {
        return Err(Error::Artifact(
            "static context sparse index must contain centroids, radii, offsets, and members".into(),
        ));
    }
    let [Some(centroids), Some(radii), Some(offsets), Some(members)] = present else {
        unreachable!("sparse index presence was checked")
    };
    let metadata = context.metadata();
    let b = metadata.ensemble_count;
    let n = metadata.training_rows;
    let clusters = centroids
        .dimensions
        .get(3)
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|clusters| (2..=n).contains(clusters))
        .ok_or_else(|| Error::Artifact("static context sparse index has invalid cluster count".into()))?;
    let expected_centroids = [b as u64, 12, 8, clusters as u64, 64];
    let expected_radii = [b as u64, 12, 8, clusters as u64];
    let expected_offsets = [b as u64, 12, 8, (clusters + 1) as u64];
    let expected_members = [b as u64, 12, 8, n as u64];
    if centroids.dimensions != expected_centroids
        || radii.dimensions != expected_radii
        || offsets.dimensions != expected_offsets
        || members.dimensions != expected_members
    {
        return Err(Error::Artifact(
            "static context sparse index tensor shapes are invalid".into(),
        ));
    }
    let _centroids = centroids
        .fp32_values()
        .ok_or_else(|| Error::Artifact("static context sparse index centroids must be FP32".into()))?;
    let radii = radii
        .fp32_values()
        .ok_or_else(|| Error::Artifact("static context sparse index radii must be FP32".into()))?;
    let offsets = offsets
        .fp32_values()
        .ok_or_else(|| Error::Artifact("static context sparse index offsets must be FP32".into()))?;
    let members = members
        .fp32_values()
        .ok_or_else(|| Error::Artifact("static context sparse index members must be FP32".into()))?;
    if radii.iter().any(|value| *value < 0.0) {
        return Err(Error::Artifact(
            "static context sparse index radii must be non-negative".into(),
        ));
    }
    for group in 0..b * 12 * 8 {
        let offset = &offsets[group * (clusters + 1)..=group * (clusters + 1) + clusters];
        if offset[0] != 0.0 || offset[clusters] != n as f32 {
            return Err(Error::Artifact(
                "static context sparse index offsets must span all source rows".into(),
            ));
        }
        let mut previous = 0_usize;
        for value in offset {
            if !value.is_finite() || value.fract() != 0.0 || *value < 0.0 || *value > n as f32 {
                return Err(Error::Artifact(
                    "static context sparse index offsets must be source-row integers".into(),
                ));
            }
            let current = *value as usize;
            if current < previous {
                return Err(Error::Artifact(
                    "static context sparse index offsets must be monotonic".into(),
                ));
            }
            previous = current;
        }
        let mut seen = vec![false; n];
        for value in &members[group * n..][..n] {
            if !value.is_finite() || value.fract() != 0.0 || *value < 0.0 || *value >= n as f32 {
                return Err(Error::Artifact(
                    "static context sparse index members must be source-row integers".into(),
                ));
            }
            let member = *value as usize;
            if std::mem::replace(&mut seen[member], true) {
                return Err(Error::Artifact(
                    "static context sparse index members must be unique".into(),
                ));
            }
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{
        ClassifierAggregation, DecoderOutputs, admit_request, admit_static_request,
        aggregate_classifier_probabilities, attention_scratch_scores, attention_scratch_state,
        model_hash_matches, parallel_cached_attention_f32_key_bf16_value, tokens_to_heads,
        validate_class_count, validate_labels, validate_static_cache_contract,
    };
    use crate::kernels::reference::{
        OnlineAttentionScratch, affine_f32, tiled_online_scaled_dot_product_attention_f32,
    };
    use crate::{StaticContextArtifact, StaticContextMetadata, StaticContextTensor};

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
            },
            tensors,
        )
        .expect("valid static context")
    }

    #[test]
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
        #[allow(clippy::cast_precision_loss)]
        let source = (0..source_rows * dim)
            .map(|index| index as f32 * 0.03125 - 0.2)
            .collect::<Vec<_>>();
        #[allow(clippy::cast_precision_loss)]
        let query = (0..rows * dim)
            .map(|index| index as f32 * -0.0625 + 0.4)
            .collect::<Vec<_>>();
        #[allow(clippy::cast_precision_loss)]
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
    #[allow(clippy::cast_precision_loss)]
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
            crate::kernels::native::AttentionBackend::Portable,
            &query,
            query_rows,
            &key,
            &value,
            source_rows,
            head_dim,
            false,
            &mut serial,
            1,
        )
        .expect("serial hybrid attention");
        parallel_cached_attention_f32_key_bf16_value(
            crate::kernels::native::AttentionBackend::Portable,
            &query,
            query_rows,
            &key,
            &value,
            source_rows,
            head_dim,
            false,
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

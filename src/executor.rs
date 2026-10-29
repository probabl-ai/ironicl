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

use crate::{
    ClassifierQueryRequest, ClassifierRequest, Error, ModelArtifact, Result, StaticContextArtifact,
    StaticContextTensorData, StaticQueryBundle,
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
            tiled_online_scaled_dot_product_attention_with_backend_f32,
        },
    },
    profile::{self, Category},
};

pub(crate) const COLUMN_DIM: usize = 128;
pub(crate) const ICL_DIM: usize = 512;
pub(crate) const HEADS: usize = 8;
pub(crate) const INDUCING: usize = 128;
pub(crate) const RESERVED: usize = 4;
const ATTENTION_QUERY_BLOCK_ROWS: usize = 16;
const ATTENTION_KEY_TILE_ROWS: usize = 128;
const CLASSIFIER_SOFTMAX_TEMPERATURE: f32 = 0.9;
const MAX_ACTIVATION_BYTES: usize = 1536 * 1024 * 1024;
const MAX_ACTIVATION_ELEMENTS: usize = MAX_ACTIVATION_BYTES / std::mem::size_of::<f32>();
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
    fn from_environment(name: &'static str) -> Result<Option<Self>> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(None);
        };
        let value = value
            .into_string()
            .map_err(|_| Error::InvalidArgument(format!("{name} must be valid UTF-8")))?;
        if value.is_empty() || matches!(value.as_str(), "1" | "all") {
            return Ok(Some(Self {
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
            }));
        }

        let mut policy = Self::default();
        for part in value.split(',') {
            match part {
                "column" => {
                    policy.column_q = true;
                    policy.column_out = true;
                    policy.column_ffn1 = true;
                    policy.column_ffn2 = true;
                }
                "column-q" => policy.column_q = true,
                "column-out" => policy.column_out = true,
                "column-ffn1" => policy.column_ffn1 = true,
                "column-ffn2" => policy.column_ffn2 = true,
                "icl" => {
                    policy.icl_q = true;
                    policy.icl_out = true;
                    policy.icl_ffn1 = true;
                    policy.icl_ffn2 = true;
                    policy.qass = true;
                }
                "icl-q" => policy.icl_q = true,
                "icl-out" => policy.icl_out = true,
                "icl-ffn" => {
                    policy.icl_ffn1 = true;
                    policy.icl_ffn2 = true;
                }
                "icl-ffn1" => policy.icl_ffn1 = true,
                "icl-ffn2" => policy.icl_ffn2 = true,
                "qass" => policy.qass = true,
                "decoder" => policy.decoder = true,
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "unknown {name} stage '{part}'; expected all, column[-q|-out|-ffn1|-ffn2], icl[-q|-out|-ffn|-ffn1|-ffn2], qass, or decoder"
                    )));
                }
            }
        }
        Ok(Some(policy))
    }
}

/// Owned classifier logits in row-major `[B, test_rows, C]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierLogits {
    shape: [usize; 3],
    values: Vec<f32>,
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
pub struct ClassifierTraceStage<const D: usize> {
    shape: [usize; D],
    values: Vec<f32>,
}

impl<const D: usize> ClassifierTraceStage<D> {
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
pub struct ClassifierLogitsTrace {
    column_embeddings: ClassifierTraceStage<4>,
    row_representations: ClassifierTraceStage<3>,
    decoder_logits: ClassifierTraceStage<3>,
    logits: ClassifierLogits,
}

impl ClassifierLogitsTrace {
    /// Column embeddings with shape `[B, T, H + 4, 128]`.
    #[must_use]
    pub fn column_embeddings(&self) -> &ClassifierTraceStage<4> {
        &self.column_embeddings
    }

    /// Row representations with shape `[B, T, 512]`, before ICL label injection.
    #[must_use]
    pub fn row_representations(&self) -> &ClassifierTraceStage<3> {
        &self.row_representations
    }

    /// Decoder logits with shape `[B, T, 10]`, before test-row and class slicing.
    #[must_use]
    pub fn decoder_logits(&self) -> &ClassifierTraceStage<3> {
        &self.decoder_logits
    }

    /// Final prediction logits with shape `[B, test_rows, classes]`.
    #[must_use]
    pub fn logits(&self) -> &ClassifierLogits {
        &self.logits
    }
}

impl ClassifierLogits {
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
pub struct ClassifierExecutor {
    model: ModelArtifact,
}

/// Immutable query-only executor backed by one exact full-K/V static context.
///
/// It can return raw estimator logits in `[B, Q, C]` order or aggregate them
/// into final class probabilities in `[Q, C]` order.
#[derive(Debug, Clone)]
pub struct StaticClassifierExecutor {
    model: ClassifierExecutor,
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
    Logits(ClassifierLogits),
    Trace(ClassifierLogitsTrace),
}

impl ClassifierExecutor {
    /// Take ownership of an exact-manifest validated classifier model.
    #[must_use]
    pub fn new(model: ModelArtifact) -> Self {
        Self { model }
    }

    /// Execute the complete FP32 classifier forward graph without persistent caches.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid labels, arithmetic overflow, excessive
    /// baseline activation memory or an
    /// unavailable validated model tensor. This uncached scalar executor is
    /// intended for small baseline runs, not 50K-row serving workloads.
    pub fn predict_logits(&self, request: ClassifierRequest<'_>) -> Result<ClassifierLogits> {
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
    pub fn predict_logits_with_trace(&self, request: ClassifierRequest<'_>) -> Result<ClassifierLogitsTrace> {
        match self.predict_logits_inner(request, true)? {
            PredictionResult::Logits(_) => unreachable!("traced prediction must produce a trace"),
            PredictionResult::Trace(trace) => Ok(trace),
        }
    }

    fn predict_logits_inner(&self, request: ClassifierRequest<'_>, trace: bool) -> Result<PredictionResult> {
        let (b, t, h, n) = (
            request.ensemble_count(),
            request.total_rows(),
            request.feature_count(),
            request.training_rows(),
        );
        admit_request(b, t, h, n)?;
        if trace {
            admit_trace_request(b, t, h)?;
        }
        let classes = validate_labels(request.y_train(), b, n)?;
        let test = t.checked_sub(n).ok_or(Error::ArithmeticOverflow("test rows"))?;
        let output_len = checked_product(&[b, test, classes], "classifier output")?;
        let mut values = Vec::with_capacity(output_len);
        let mut column_values = trace.then(Vec::new);
        let mut row_values = trace.then(Vec::new);
        let mut decoder_values = trace.then(Vec::new);
        for ensemble in 0..b {
            let x = &request.x()[checked_product(&[ensemble, t, h], "feature offset")?..]
                [..checked_product(&[t, h], "feature rows")?];
            let labels = &request.y_train()[ensemble * n..][..n];
            let columns = self.column_embeddings(x, labels, t, h, n)?;
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
                values.extend_from_slice(&logits[row * 10..][..classes]);
            }
        }
        let logits = ClassifierLogits {
            shape: [b, test, classes],
            values,
        };
        match (column_values, row_values, decoder_values) {
            (Some(column_embeddings), Some(row_representations), Some(decoder_logits)) => {
                Ok(PredictionResult::Trace(ClassifierLogitsTrace {
                    column_embeddings: ClassifierTraceStage {
                        shape: [b, t, h + RESERVED, COLUMN_DIM],
                        values: column_embeddings,
                    },
                    row_representations: ClassifierTraceStage {
                        shape: [b, t, ICL_DIM],
                        values: row_representations,
                    },
                    decoder_logits: ClassifierTraceStage {
                        shape: [b, t, 10],
                        values: decoder_logits,
                    },
                    logits,
                }))
            }
            (None, None, None) => Ok(PredictionResult::Logits(logits)),
            _ => Err(Error::InvalidArgument("incomplete trace output".into())),
        }
    }

    fn column_embeddings(&self, x: &[f32], labels: &[f32], t: usize, h: usize, n: usize) -> Result<Vec<f32>> {
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
        let y_weight = self.tensor("col_embedder.y_encoder.weight")?;
        let y_bias = self.tensor("col_embedder.y_encoder.bias")?;
        let label_indices = label_indices(labels)?;
        let mut y = vec![0.0; checked_product(&[n, COLUMN_DIM], "column target embeddings")?];
        one_hot_linear_f32(&label_indices, y_weight, y_bias, 10, COLUMN_DIM, &mut y)?;
        // RowInteraction replaces the reserved streams with CLS tokens before
        // its first block, so running the column transformer for them is dead work.
        let workers = executor_threads().min(h);
        let chunk = h.div_ceil(workers);
        let y = y.as_slice();
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for start in (0..h).step_by(chunk) {
                let end = (start + chunk).min(h);
                handles.push(scope.spawn(move || -> Result<Vec<(usize, Vec<f32>)>> {
                    let mut results = Vec::with_capacity(end - start);
                    for feature in start..end {
                        let embedded = self.column_stream(x, y, t, h, n, feature, in_weight, in_bias)?;
                        results.push((feature + RESERVED, embedded));
                    }
                    Ok(results)
                }));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("column worker must not panic"))
                .collect::<Result<Vec<_>>>()
        })?;
        for (stream, embedded) in results.into_iter().flatten() {
            for row in 0..t {
                output[(row * streams + stream) * COLUMN_DIM..][..COLUMN_DIM]
                    .copy_from_slice(&embedded[row * COLUMN_DIM..][..COLUMN_DIM]);
            }
        }
        Ok(output)
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
    ) -> Result<Vec<f32>> {
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
        for block in 0..3 {
            embedded = self.isab(embedded, t, n, block)?;
        }
        Ok(embedded)
    }

    fn isab(&self, source: Vec<f32>, t: usize, n: usize, block: usize) -> Result<Vec<f32>> {
        if source.iter().all(|value| *value == -100.0) {
            return Ok(source);
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
        self.block(
            &source,
            &hidden,
            &hidden,
            t,
            INDUCING,
            COLUMN_DIM,
            &format!("{prefix}.multihead_attn2"),
            false,
            None,
        )
    }

    pub(crate) fn row_embeddings(&self, columns: &[f32], t: usize, streams: usize) -> Result<Vec<f32>> {
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

    fn icl_logits(&self, mut rows: Vec<f32>, labels: &[f32], t: usize, n: usize) -> Result<Vec<f32>> {
        let _profile = profile::scope(Category::Icl);
        let indices = label_indices(labels)?;
        let mut targets = vec![0.0; checked_product(&[n, ICL_DIM], "ICL target embeddings")?];
        one_hot_linear_f32(
            &indices,
            self.tensor("icl_predictor.y_encoder.weight")?,
            self.tensor("icl_predictor.y_encoder.bias")?,
            10,
            ICL_DIM,
            &mut targets,
        )?;
        for row in 0..n {
            for dim in 0..ICL_DIM {
                rows[row * ICL_DIM + dim] += targets[row * ICL_DIM + dim];
            }
        }
        for block in 0..12 {
            let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
            rows = self.block(
                &rows,
                &rows[..n * ICL_DIM],
                &rows[..n * ICL_DIM],
                t,
                n,
                ICL_DIM,
                &prefix,
                true,
                None,
            )?;
        }
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
        let mut logits = vec![0.0; checked_product(&[t, 10], "decoder logits")?];
        affine_f32(
            &activated,
            t,
            1024,
            self.tensor("icl_predictor.decoder.2.weight")?,
            Some(self.tensor("icl_predictor.decoder.2.bias")?),
            10,
            &mut logits,
        )?;
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn block(
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
        let _profile = profile::scope(Category::Block);
        let norm_weight = self.tensor(&format!("{prefix}.norm1.weight"))?;
        let norm_bias = self.tensor(&format!("{prefix}.norm1.bias"))?;
        let mut q_norm = vec![0.0; q.len()];
        layer_norm_f32(q, q_rows, dim, norm_weight, Some(norm_bias), &mut q_norm)?;
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
        let (mut k_heads, v_heads) = if std::ptr::eq(k.as_ptr(), v.as_ptr()) {
            self.project_source_kv(k, source_rows, dim, prefix)?
        } else {
            let mut k_norm = vec![0.0; k.len()];
            layer_norm_f32(k, source_rows, dim, norm_weight, Some(norm_bias), &mut k_norm)?;
            let mut v_norm = vec![0.0; v.len()];
            layer_norm_f32(v, source_rows, dim, norm_weight, Some(norm_bias), &mut v_norm)?;
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
            (
                tokens_to_heads(&k_proj, source_rows, dim, heads),
                tokens_to_heads(&v_proj, source_rows, dim, heads),
            )
        };
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
            let frequencies = self.tensor("row_interactor.tf_row.rope.freqs")?;
            let mut rotated = vec![0.0; q_heads.len()];
            rope_non_interleaved_f32(&q_heads, heads, q_rows, head_dim, frequencies, 0, &mut rotated)?;
            q_heads = rotated;
            let mut rotated = vec![0.0; k_heads.len()];
            rope_non_interleaved_f32(
                &k_heads,
                heads,
                source_rows,
                head_dim,
                frequencies,
                0,
                &mut rotated,
            )?;
            k_heads = rotated;
        }
        let mut attended = vec![0.0; q_heads.len()];
        if rope.is_some() && row_source_major_attention_enabled() {
            let query_block_rows = small_attention_query_block_rows();
            let scratch_rows = q_rows.min(query_block_rows);
            let scratch_sources = source_rows.min(ATTENTION_KEY_TILE_ROWS);
            let mut scores = vec![0.0; scratch_rows * scratch_sources];
            let mut maximum = vec![0.0; heads * q_rows];
            let mut denominator = vec![0.0; heads * q_rows];
            source_major_online_attention_with_backend_f32(
                &q_heads,
                q_rows,
                &k_heads,
                &v_heads,
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
                &k_heads,
                &v_heads,
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

    /// Layer-normalise `source` rows with this block's `norm1` and project them to
    /// head-major `[heads, rows, head_dim]` keys and values, exactly as the block's
    /// attention consumes them and as a static context stores them.
    pub(crate) fn project_source_kv(
        &self,
        source: &[f32],
        rows: usize,
        dim: usize,
        prefix: &str,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let norm_weight = self.tensor(&format!("{prefix}.norm1.weight"))?;
        let norm_bias = self.tensor(&format!("{prefix}.norm1.bias"))?;
        let mut normalized = vec![0.0; source.len()];
        layer_norm_f32(source, rows, dim, norm_weight, Some(norm_bias), &mut normalized)?;
        let attn_prefix = format!("{prefix}.attn");
        let weight = self.tensor(&format!("{attn_prefix}.in_proj_weight"))?;
        let bias = self.tensor(&format!("{attn_prefix}.in_proj_bias"))?;
        let mut k_proj = vec![0.0; source.len()];
        affine_f32(
            &normalized,
            rows,
            dim,
            &weight[dim * dim..2 * dim * dim],
            Some(&bias[dim..2 * dim]),
            dim,
            &mut k_proj,
        )?;
        let mut v_proj = vec![0.0; source.len()];
        affine_f32(
            &normalized,
            rows,
            dim,
            &weight[2 * dim * dim..],
            Some(&bias[2 * dim..]),
            dim,
            &mut v_proj,
        )?;
        Ok((
            tokens_to_heads(&k_proj, rows, dim, HEADS),
            tokens_to_heads(&v_proj, rows, dim, HEADS),
        ))
    }

    /// The imported model this executor runs.
    #[must_use]
    pub fn model(&self) -> &ModelArtifact {
        &self.model
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn qass(
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

    pub(crate) fn tensor(&self, name: &str) -> Result<&[f32]> {
        Ok(self.model.tensor_f32(name)?.values())
    }
}

impl StaticClassifierExecutor {
    /// Pair an official classifier model with its matching exact full-K/V context.
    ///
    /// # Errors
    ///
    /// Returns an artifact error if the model identity, architecture, or required
    /// cache tensor names and shapes do not match the serving contract.
    pub fn new(model: ModelArtifact, context: StaticContextArtifact) -> Result<Self> {
        validate_static_context(&model, &context)?;
        let model = ClassifierExecutor::new(model);
        let input_weight = model.tensor("col_embedder.in_linear.weight")?;
        let mut column_input_weight = vec![0.0; input_weight.len()];
        for output in 0..COLUMN_DIM {
            for input in 0..3 {
                column_input_weight[input * COLUMN_DIM + output] = input_weight[output * 3 + input];
            }
        }
        let qass_base_scales = (0..12)
            .map(|block| {
                Self::build_qass_base_scale(
                    &model,
                    &format!("icl_predictor.tf_icl.blocks.{block}.attn"),
                    context.metadata().training_rows,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let packed_f32_row =
            std::env::var_os("IRONICL_PACKED_F32_ROW").is_some() && packed_f32_affine_supported();
        let packed_f32_policy = Bf16DensePolicy::from_environment("IRONICL_PACKED_F32_DENSE")?;
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
        let bf16_dense_policy = Bf16DensePolicy::from_environment("IRONICL_BF16_DENSE")?;
        let bf16_dense_compensation = if bf16_dense_policy.is_some() {
            Self::bf16_dense_compensation_from_environment()?
        } else {
            Bf16AffineCompensation::None
        };
        let column_bf16_blocks = if bf16_dense_policy.is_some() {
            Self::bf16_dense_blocks::<3>("IRONICL_BF16_DENSE_COLUMN_BLOCKS")?
        } else {
            [false; 3]
        };
        let icl_bf16_blocks = if bf16_dense_policy.is_some() {
            Self::bf16_dense_blocks::<12>("IRONICL_BF16_DENSE_ICL_BLOCKS")?
        } else {
            [false; 12]
        };
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
            .num_threads(executor_threads())
            .thread_name(|index| format!("ironicl-affine-{index}"))
            .build()
            .map_err(|error| Error::InvalidArgument(format!("cannot create affine worker pool: {error}")))?;
        Ok(Self {
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
        })
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
    pub fn predict_logits(&self, request: ClassifierQueryRequest<'_>) -> Result<ClassifierLogits> {
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
        let plan = static_execution_plan(metadata.ensemble_count, executor_threads());
        let layer_major = metadata.ensemble_count > 1 && std::env::var_os("IRONICL_LAYER_MAJOR").is_some();
        admit_static_request(
            metadata.ensemble_count,
            request.query_rows(),
            metadata.feature_count,
            metadata.class_count,
            if layer_major || plan.flat_pool_ensembles {
                metadata.ensemble_count
            } else {
                plan.ensemble_workers
            },
        )?;
        let output_elements = checked_product(
            &[
                metadata.ensemble_count,
                request.query_rows(),
                metadata.class_count,
            ],
            "static classifier output",
        )?;
        let mut values = vec![0.0; output_elements];
        let ensemble_output_elements = checked_product(
            &[request.query_rows(), metadata.class_count],
            "static ensemble output",
        )?;
        if layer_major {
            self.predict_layer_major(request, plan.inner_workers, &mut values)?;
        } else if plan.ensemble_workers == 1 {
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
        Ok(ClassifierLogits {
            shape: [
                metadata.ensemble_count,
                request.query_rows(),
                metadata.class_count,
            ],
            values,
        })
    }

    fn predict_layer_major(
        &self,
        request: ClassifierQueryRequest<'_>,
        workers: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let metadata = self.context.metadata();
        let q = request.query_rows();
        let mut rows = vec![Vec::new(); metadata.ensemble_count];
        self.affine_pool.install(|| {
            rows.par_iter_mut().enumerate().try_for_each(|(ensemble, rows)| {
                *rows = self.cached_frontend(request, ensemble, workers)?;
                Ok(())
            })
        })?;

        let _profile = profile::scope(Category::Icl);
        for block in 0..12 {
            self.affine_pool.install(|| {
                rows.par_iter_mut().enumerate().try_for_each(|(ensemble, rows)| {
                    *rows = self.cached_icl_block(rows, ensemble, q, block, workers)?;
                    Ok(())
                })
            })?;
        }

        let ensemble_output_elements = checked_product(&[q, metadata.class_count], "static ensemble output")?;
        self.affine_pool.install(|| {
            output
                .par_chunks_mut(ensemble_output_elements)
                .zip(rows.par_iter())
                .try_for_each(|(output, rows)| {
                    let logits = self.cached_icl_decoder(rows, q, workers)?;
                    for row in 0..q {
                        output[row * metadata.class_count..][..metadata.class_count]
                            .copy_from_slice(&logits[row * 10..][..metadata.class_count]);
                    }
                    Ok(())
                })
        })
    }

    fn predict_estimator(
        &self,
        request: ClassifierQueryRequest<'_>,
        ensemble: usize,
        workers: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let rows = self.cached_frontend(request, ensemble, workers)?;
        let logits = self.cached_icl_logits(rows, ensemble, request.query_rows(), workers)?;
        let metadata = self.context.metadata();
        for row in 0..request.query_rows() {
            output[row * metadata.class_count..][..metadata.class_count]
                .copy_from_slice(&logits[row * 10..][..metadata.class_count]);
        }
        Ok(())
    }

    fn cached_frontend(
        &self,
        request: ClassifierQueryRequest<'_>,
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

    /// Execute cached inference and apply official class correction, logit
    /// averaging, and temperature-scaled softmax.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::predict_logits`].
    pub fn predict_probabilities(
        &self,
        request: ClassifierQueryRequest<'_>,
    ) -> Result<ClassifierProbabilities> {
        let logits = self.predict_logits(request)?;
        Ok(aggregate_classifier_probabilities(
            &logits,
            &self.context.metadata().class_permutations,
        ))
    }

    /// Execute independent prepared-query requests as one cache-sharing microbatch.
    ///
    /// Requests must use the same `[B, H]` shape as this static context. Results
    /// preserve request order and are equivalent to executing each request alone.
    ///
    /// # Errors
    ///
    /// Returns an error when requests are empty, have incompatible dimensions, or
    /// the combined request exceeds the normal static-request admission limits.
    pub fn predict_probabilities_batch(
        &self,
        requests: &[ClassifierQueryRequest<'_>],
    ) -> Result<Vec<ClassifierProbabilities>> {
        if requests.is_empty() {
            return Err(Error::InvalidArgument(
                "query microbatch must not be empty".into(),
            ));
        }
        let metadata = self.context.metadata();
        let (total_rows, packed) =
            pack_query_microbatch(requests, metadata.ensemble_count, metadata.feature_count)?;
        let packed_request = ClassifierQueryRequest::new(
            metadata.ensemble_count,
            total_rows,
            metadata.feature_count,
            &packed,
        )?;
        let probabilities = self.predict_probabilities(packed_request)?;
        let mut offset = 0_usize;
        requests
            .iter()
            .map(|request| {
                let elements = request
                    .query_rows()
                    .checked_mul(metadata.class_count)
                    .ok_or(Error::ArithmeticOverflow("query microbatch output elements"))?;
                let values = probabilities.values[offset..][..elements].to_vec();
                offset += elements;
                Ok(ClassifierProbabilities {
                    shape: [request.query_rows(), metadata.class_count],
                    values,
                })
            })
            .collect()
    }

    /// Validate artifact identities and execute one owned query bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when the query references another model or static
    /// context, or when normal query execution fails.
    pub fn predict_bundle(&self, query: &StaticQueryBundle) -> Result<ClassifierLogits> {
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
    pub fn predict_bundle_probabilities(&self, query: &StaticQueryBundle) -> Result<ClassifierProbabilities> {
        if query.model_sha256() != &self.context.metadata().model_sha256
            || query.context_sha256() != &self.context.metadata().context_sha256
        {
            return Err(Error::InvalidArgument(
                "query bundle model or context identity does not match the static context".into(),
            ));
        }
        self.predict_probabilities(query.request()?)
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
        let mut logits = vec![0.0; q * 10];
        self.parallel_affine(
            &activated,
            q,
            1024,
            self.tensor("icl_predictor.decoder.2.weight")?,
            Some(self.tensor("icl_predictor.decoder.2.bias")?),
            10,
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
                let query_block_rows = small_attention_query_block_rows();
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
        if workers <= 1 || rows < 16 {
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

    fn build_packed_f32_row_affines(model: &ClassifierExecutor) -> Result<BTreeMap<String, PackedF32Affine>> {
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
        model: &ClassifierExecutor,
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
        model: &ClassifierExecutor,
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
        model: &ClassifierExecutor,
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
        model: &ClassifierExecutor,
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

    fn bf16_dense_blocks<const N: usize>(name: &'static str) -> Result<[bool; N]> {
        let Some(value) = std::env::var_os(name) else {
            return Ok([true; N]);
        };
        let value = value
            .into_string()
            .map_err(|_| Error::InvalidArgument(format!("{name} must be valid UTF-8")))?;
        if value.is_empty() || value == "all" {
            return Ok([true; N]);
        }
        let mut selected = [false; N];
        for part in value.split(',') {
            let block = part.parse::<usize>().map_err(|_| {
                Error::InvalidArgument(format!("{name} block '{part}' is not an unsigned integer"))
            })?;
            if block >= N {
                return Err(Error::InvalidArgument(format!(
                    "{name} block {block} is outside 0..{}",
                    N - 1
                )));
            }
            selected[block] = true;
        }
        Ok(selected)
    }

    fn bf16_dense_compensation_from_environment() -> Result<Bf16AffineCompensation> {
        let Some(value) = std::env::var_os("IRONICL_BF16_DENSE_COMPENSATION") else {
            return Ok(Bf16AffineCompensation::None);
        };
        let value = value.into_string().map_err(|_| {
            Error::InvalidArgument("IRONICL_BF16_DENSE_COMPENSATION must be valid UTF-8".into())
        })?;
        match value.as_str() {
            "" | "none" => Ok(Bf16AffineCompensation::None),
            "input" => Ok(Bf16AffineCompensation::Input),
            "weight" => Ok(Bf16AffineCompensation::Weight),
            "first-order" | "both" => Ok(Bf16AffineCompensation::FirstOrder),
            _ => Err(Error::InvalidArgument(format!(
                "unknown IRONICL_BF16_DENSE_COMPENSATION '{value}'; expected none, input, weight, or first-order"
            ))),
        }
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn pack_cached_block_affines(
        packed: &mut BTreeMap<String, PackedBf16Affine>,
        model: &ClassifierExecutor,
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
        model: &ClassifierExecutor,
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

    #[allow(clippy::cast_precision_loss)]
    fn build_qass_base_scale(model: &ClassifierExecutor, prefix: &str, source: usize) -> Result<Vec<f32>> {
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
        let mut scale = vec![0.0; HEADS * 64];
        affine_f32(
            &hidden,
            1,
            64,
            second_weight,
            Some(second_bias),
            HEADS * 64,
            &mut scale,
        )?;
        Ok(scale)
    }

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

fn aggregate_classifier_probabilities(
    logits: &ClassifierLogits,
    class_permutations: &[u64],
) -> ClassifierProbabilities {
    let [ensembles, rows, classes] = logits.shape;
    let mut values = vec![0.0_f32; rows * classes];
    let ensemble_scale =
        1.0 / f32::from(u16::try_from(ensembles).expect("validated ensemble count fits u16"));
    for ensemble in 0..ensembles {
        for row in 0..rows {
            for class in 0..classes {
                let shuffled_class = class_permutations[ensemble * classes + class] as usize;
                values[row * classes + class] +=
                    logits.values[(ensemble * rows + row) * classes + shuffled_class] * ensemble_scale;
            }
        }
    }
    for row in values.chunks_exact_mut(classes) {
        let maximum = row
            .iter()
            .copied()
            .map(|value| value / CLASSIFIER_SOFTMAX_TEMPERATURE)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0_f32;
        for value in &mut *row {
            *value = (*value / CLASSIFIER_SOFTMAX_TEMPERATURE - maximum).exp();
            denominator += *value;
        }
        for value in row {
            *value /= denominator;
        }
    }
    ClassifierProbabilities {
        shape: [rows, classes],
        values,
    }
}

fn pack_query_microbatch(
    requests: &[ClassifierQueryRequest<'_>],
    ensemble_count: usize,
    feature_count: usize,
) -> Result<(usize, Vec<f32>)> {
    let total_rows = requests.iter().try_fold(0_usize, |sum, request| {
        if request.ensemble_count() != ensemble_count || request.feature_count() != feature_count {
            return Err(Error::InvalidArgument(
                "all query microbatch requests must match the static context B and H".into(),
            ));
        }
        sum.checked_add(request.query_rows())
            .ok_or(Error::ArithmeticOverflow("query microbatch rows"))
    })?;
    let mut packed = Vec::with_capacity(
        ensemble_count
            .checked_mul(total_rows)
            .and_then(|values| values.checked_mul(feature_count))
            .ok_or(Error::ArithmeticOverflow("query microbatch elements"))?,
    );
    for ensemble in 0..ensemble_count {
        for request in requests {
            let elements = request
                .query_rows()
                .checked_mul(feature_count)
                .ok_or(Error::ArithmeticOverflow("query microbatch request elements"))?;
            let offset = ensemble
                .checked_mul(elements)
                .ok_or(Error::ArithmeticOverflow("query microbatch request offset"))?;
            packed.extend_from_slice(&request.x()[offset..][..elements]);
        }
    }
    Ok((total_rows, packed))
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
    validate_static_cache_contract(context)
}

fn model_hash_matches(model_hash: &[u8], context_hash: &[u8; 32]) -> bool {
    model_hash == crate::model_artifact::encode_sha256(context_hash).as_bytes()
}

fn validate_static_cache_contract(context: &StaticContextArtifact) -> Result<()> {
    let metadata = context.metadata();
    if !(2..=10).contains(&metadata.class_count) {
        return Err(Error::Artifact(format!(
            "static context class count {} is unsupported; expected 2..=10",
            metadata.class_count
        )));
    }
    for (stage, layers, sources, head_dim) in
        [("column", 3, 128, 16), ("icl", 12, metadata.training_rows, 64)]
    {
        for layer in 0..layers {
            for kind in ["key", "value"] {
                let name = format!("{stage}.{layer}.{kind}");
                let tensor = context.tensor(&name).ok_or_else(|| {
                    Error::Artifact(format!("static context cache tensor '{name}' is missing"))
                })?;
                let expected_shape = if stage == "column" {
                    vec![
                        metadata.ensemble_count as u64,
                        (metadata.feature_count + RESERVED) as u64,
                        8,
                        sources as u64,
                        head_dim,
                    ]
                } else {
                    vec![metadata.ensemble_count as u64, 8, sources as u64, head_dim]
                };
                if tensor.dimensions != expected_shape {
                    return Err(Error::Artifact(format!(
                        "static context cache tensor '{name}' has shape {:?}; expected {:?}",
                        tensor.dimensions, expected_shape
                    )));
                }
            }
        }
    }
    validate_sparse_index_contract(context)
}

pub(crate) fn executor_threads() -> usize {
    *EXECUTOR_THREADS.get_or_init(|| {
        if let Some(value) = std::env::var_os("IRONICL_THREADS") {
            return value
                .to_string_lossy()
                .parse::<usize>()
                .ok()
                .filter(|threads| *threads > 0)
                .unwrap_or(1);
        }
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        if cfg!(target_arch = "x86_64") {
            logical.div_ceil(2)
        } else {
            logical
        }
    })
}

fn cached_attention_key_tile_rows() -> usize {
    static KEY_TILE_ROWS: OnceLock<usize> = OnceLock::new();
    *KEY_TILE_ROWS.get_or_init(|| {
        std::env::var_os("IRONICL_ATTENTION_KEY_TILE_ROWS")
            .and_then(|value| value.to_string_lossy().parse::<usize>().ok())
            .filter(|rows| matches!(rows, 64 | 128 | 256 | 512 | 1024))
            .unwrap_or(ATTENTION_KEY_TILE_ROWS)
    })
}

fn column_source_major_attention_enabled(backend: crate::kernels::native::AttentionBackend) -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    backend == crate::kernels::native::AttentionBackend::Avx512
        && *ENABLED.get_or_init(|| std::env::var_os("IRONICL_COLUMN_QUERY_MAJOR").is_none())
}

fn row_source_major_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::kernels::native::attention_backend() == crate::kernels::native::AttentionBackend::Avx512
            && std::env::var_os("IRONICL_ROW_QUERY_MAJOR").is_none()
    })
}

fn small_attention_query_block_rows() -> usize {
    static QUERY_BLOCK_ROWS: OnceLock<usize> = OnceLock::new();
    *QUERY_BLOCK_ROWS.get_or_init(|| {
        std::env::var_os("IRONICL_SMALL_ATTENTION_QUERY_ROWS")
            .and_then(|value| value.to_string_lossy().parse::<usize>().ok())
            .filter(|rows| matches!(rows, 8 | 16))
            .unwrap_or(16)
    })
}

fn sparse_attention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("IRONICL_SPARSE_ATTENTION").as_deref(), Ok("1")))
}

fn static_execution_plan(ensembles: usize, total_threads: usize) -> ExecutionPlan {
    let force_serial = std::env::var_os("IRONICL_SERIAL_ENSEMBLES").is_some();
    if !force_serial {
        let requested_inner = std::env::var_os("IRONICL_INNER_WORKERS")
            .and_then(|value| value.to_string_lossy().parse::<usize>().ok())
            .filter(|workers| *workers > 0 && *workers <= total_threads);
        let requested = std::env::var_os("IRONICL_ENSEMBLE_WORKERS")
            .and_then(|value| value.to_string_lossy().parse::<usize>().ok())
            .filter(|workers| {
                if *workers == 0 || *workers > ensembles || *workers > total_threads {
                    return false;
                }
                let inner_workers = total_threads / *workers;
                total_threads >= 16 || inner_workers == 1 || *workers == 1
            });
        if let Some(ensemble_workers) = requested {
            return ExecutionPlan {
                ensemble_workers,
                inner_workers: requested_inner.unwrap_or_else(|| (total_threads / ensemble_workers).max(1)),
                flat_pool_ensembles: false,
            };
        }
    }
    let mut plan = ExecutionPlan::automatic(ensembles, total_threads, force_serial);
    if plan.flat_pool_ensembles
        && let Some(inner_workers) = std::env::var_os("IRONICL_INNER_WORKERS")
            .and_then(|value| value.to_string_lossy().parse::<usize>().ok())
            .filter(|workers| *workers > 0 && *workers <= total_threads)
    {
        plan.inner_workers = inner_workers;
    }
    plan
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

fn sparse_max_omitted_mass() -> f32 {
    static MAXIMUM: OnceLock<f32> = OnceLock::new();
    *MAXIMUM.get_or_init(|| {
        std::env::var("IRONICL_SPARSE_MAX_OMITTED_MASS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
            .unwrap_or(1e-5)
    })
}

pub(crate) fn parallel_cached_attention(
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
    static QUERY_MAJOR: OnceLock<bool> = OnceLock::new();
    let query_major =
        *QUERY_MAJOR.get_or_init(|| std::env::var_os("IRONICL_QUERY_MAJOR_ATTENTION").is_some());
    let query_head_elements = checked_product(&[query_rows, head_dim], "cached query head")?;
    let source_head_elements = checked_product(&[source_rows, head_dim], "cached source head")?;
    let workers = workers.min(HEADS);
    if workers == 1 {
        let scratch_query_rows = query_rows.min(ATTENTION_QUERY_BLOCK_ROWS);
        let scratch_source_rows = source_rows.min(ATTENTION_KEY_TILE_ROWS);
        if query_major && !key_is_packed {
            let mut scores = vec![0.0; HEADS * scratch_query_rows * scratch_source_rows];
            let mut maximum = vec![0.0; HEADS * scratch_query_rows];
            let mut denominator = vec![0.0; HEADS * scratch_query_rows];
            return tiled_online_scaled_dot_product_attention_with_backend_f32(
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
                if query_major && !key_is_packed {
                    let mut scores = vec![0.0; heads * scratch_query_rows * scratch_source_rows];
                    let mut maximum = vec![0.0; heads * scratch_query_rows];
                    let mut denominator = vec![0.0; heads * scratch_query_rows];
                    tiled_online_scaled_dot_product_attention_with_backend_f32(
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
                } else {
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
                }
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
    let key_tile_rows = if key_is_packed {
        ATTENTION_KEY_TILE_ROWS
    } else {
        cached_attention_key_tile_rows()
    };
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
    let key_tile_rows = if key_is_packed {
        ATTENTION_KEY_TILE_ROWS
    } else {
        cached_attention_key_tile_rows()
    };
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

pub(crate) fn validate_labels(labels: &[f32], b: usize, n: usize) -> Result<usize> {
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

pub(crate) fn checked_product(values: &[usize], name: &'static str) -> Result<usize> {
    values.iter().try_fold(1_usize, |result, value| {
        result.checked_mul(*value).ok_or(Error::ArithmeticOverflow(name))
    })
}

/// Estimate the largest persistent and intermediate FP32 activation set for one ensemble.
///
/// The executor processes ensembles sequentially. The estimates intentionally retain
/// all operands that can coexist during each stage rather than assuming allocator reuse.
fn admit_request(b: usize, t: usize, h: usize, n: usize) -> Result<()> {
    let streams = h
        .checked_add(RESERVED)
        .ok_or(Error::ArithmeticOverflow("column streams"))?;
    let test = t.checked_sub(n).ok_or(Error::ArithmeticOverflow("test rows"))?;

    let output = checked_product(&[b, test, 10], "classifier output")?;
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
    let activation_peak = checked_sum(&[output, column_peak, row_peak, icl_peak], "activation peak")?;
    if activation_peak > MAX_ACTIVATION_ELEMENTS {
        return Err(Error::InvalidArgument(format!(
            "request exceeds the uncached scalar activation limit of {MAX_ACTIVATION_BYTES} bytes"
        )));
    }

    Ok(())
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
fn admit_trace_request(b: usize, t: usize, h: usize) -> Result<()> {
    let streams = h
        .checked_add(RESERVED)
        .ok_or(Error::ArithmeticOverflow("column streams"))?;
    let trace_elements = checked_sum(
        &[
            checked_product(&[b, t, streams, COLUMN_DIM], "trace column embeddings")?,
            checked_product(&[b, t, ICL_DIM], "trace row representations")?,
            checked_product(&[b, t, 10], "trace decoder logits")?,
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
    let query_block_rows = small_attention_query_block_rows();
    checked_sum(
        &[
            attention_scratch_scores(query_block_rows, ATTENTION_KEY_TILE_ROWS)?,
            attention_scratch_state(query_block_rows)?,
            attention_scratch_state(query_block_rows)?,
        ],
        "attention scratch",
    )
}

pub(crate) fn label_indices(labels: &[f32]) -> Result<Vec<usize>> {
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

pub(crate) fn tokens_to_heads(input: &[f32], rows: usize, dim: usize, heads: usize) -> Vec<f32> {
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
pub(crate) fn heads_to_tokens(input: &[f32], rows: usize, dim: usize, heads: usize) -> Vec<f32> {
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
        CLASSIFIER_SOFTMAX_TEMPERATURE, ClassifierLogits, admit_request, admit_static_request,
        aggregate_classifier_probabilities, attention_scratch_scores, attention_scratch_state,
        model_hash_matches, pack_query_microbatch, parallel_cached_attention_f32_key_bf16_value,
        tokens_to_heads, validate_labels, validate_static_cache_contract,
    };
    use crate::kernels::reference::{
        OnlineAttentionScratch, affine_f32, tiled_online_scaled_dot_product_attention_f32,
    };
    use crate::{ClassifierQueryRequest, StaticContextArtifact, StaticContextMetadata, StaticContextTensor};

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
    fn cached_request_admission_is_bounded() {
        assert!(admit_static_request(1, 64, 54, 7, 1).is_ok());
        assert!(admit_static_request(8, 64, 54, 7, 8).is_ok());
        assert!(admit_static_request(1, 1_000_000, 1, 2, 1).is_err());
    }

    #[test]
    fn query_microbatch_packs_each_ensemble_contiguously() {
        let first = ClassifierQueryRequest::new(2, 1, 2, &[1.0, 2.0, 3.0, 4.0]).expect("first request");
        let second = ClassifierQueryRequest::new(2, 2, 2, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("second request");
        let (rows, packed) = pack_query_microbatch(&[first, second], 2, 2).expect("packed microbatch");
        assert_eq!(rows, 3);
        assert_eq!(
            packed,
            vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]
        );
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
    fn probabilities_reverse_class_permutations_before_averaging_logits() {
        let probabilities = aggregate_classifier_probabilities(
            &ClassifierLogits {
                shape: [2, 1, 3],
                values: vec![1.0, 2.0, 3.0, 4.0, 8.0, 6.0],
            },
            &[2, 0, 1, 1, 2, 0],
        );
        assert_eq!(probabilities.shape(), [1, 3]);
        let expected_logits = [5.5_f32, 3.5, 3.0];
        let maximum = expected_logits[0] / CLASSIFIER_SOFTMAX_TEMPERATURE;
        let denominator = expected_logits
            .iter()
            .map(|value| (*value / CLASSIFIER_SOFTMAX_TEMPERATURE - maximum).exp())
            .sum::<f32>();
        for (actual, expected) in probabilities.values().iter().zip(expected_logits) {
            let expected = (expected / CLASSIFIER_SOFTMAX_TEMPERATURE - maximum).exp() / denominator;
            assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
        }
    }

    #[test]
    fn admission_allows_a_small_baseline_run() {
        assert!(admit_request(8, 1024, 100, 512).is_ok());
    }

    #[test]
    fn admission_allows_bounded_attention_scratch() {
        // The former [heads, query_rows, source_rows] score allocation here
        // exceeded the attention limit; the remaining activation estimate fits.
        assert!(admit_request(1, 11_000, 1, 3_000).is_ok());
    }

    #[test]
    fn admission_rejects_huge_dimensions_without_input_allocation() {
        assert!(admit_request(1, 1_000_000, 1, 1).is_err());
        assert!(admit_request(1, usize::MAX, usize::MAX, 1).is_err());
    }
}

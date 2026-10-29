//! Native static-context construction.
//!
//! Runs the preprocessed training table through the column, row, and ICL stages
//! once and stores the attention key/value state that [`StaticClassifierExecutor`]
//! serves from, so a fitted context never needs the upstream `PyTorch` model.
//!
//! Training rows never attend to query rows (column attention sources are the
//! inducing vectors' view of the training rows; ICL keys are training rows), so
//! the build runs the uncached stages with no query rows at all. The cache holds
//! each block's `norm1`-normalised, in-projected keys and values in head-major
//! layout; `QASSMax` scaling and the softmax scale are query-side and stay in the
//! cached executor.
//!
//! [`StaticClassifierExecutor`]: crate::StaticClassifierExecutor

use std::path::Path;

use crate::{
    Error, Result, StaticContextArtifact, StaticContextLayout, StaticContextMetadata, StaticContextTensor,
    StaticContextWriter,
    executor::{
        COLUMN_DIM, ClassifierExecutor, HEADS, ICL_DIM, INDUCING, RESERVED, checked_product,
        executor_threads, heads_to_tokens, label_indices, parallel_cached_attention, tokens_to_heads,
        validate_labels,
    },
    kernels::reference::{affine_f32, gelu_f32, layer_norm_f32, one_hot_linear_f32},
    model_artifact::decode_sha256,
    static_context::{KvDtype, validate_estimator_permutations},
};

const COLUMN_LAYERS: usize = 3;
const ICL_LAYERS: usize = 12;
const COLUMN_HEAD_DIM: usize = 16;
const ICL_HEAD_DIM: usize = 64;
const MISSING: f32 = -100.0;
const MAX_ENSEMBLE_COUNT: usize = 4096;
const MAX_TRAINING_ROWS: usize = 1_000_000;
const MAX_FEATURE_COUNT: usize = 65_536;
/// Largest per-estimator column activation buffer the builder will allocate.
const MAX_BUILD_ACTIVATION_BYTES: usize = 16 * 1024 * 1024 * 1024;
const KMEANS_ITERATIONS: usize = 4;

/// The preprocessed training table of every estimator, as the engine consumes it.
///
/// `x` holds `[B, row_stride, H]` FP32 ensemble variants of which the first
/// `training_rows` rows per estimator are used, so an `IRINPUT` bundle's
/// `[B, T, H]` tensor can be passed without copying. `y` holds `[B, N]` class
/// indices as FP32, already class-permuted. Permutations are estimator-major
/// `[B, C]` and `[B, H]` and are stored verbatim in the context.
#[derive(Debug, Clone, Copy)]
pub struct StaticContextRequest<'a> {
    ensemble_count: usize,
    training_rows: usize,
    row_stride: usize,
    feature_count: usize,
    class_count: usize,
    x: &'a [f32],
    y: &'a [f32],
    class_permutations: &'a [u64],
    feature_permutations: &'a [u64],
    context_sha256: [u8; 32],
}

impl<'a> StaticContextRequest<'a> {
    /// Validate shapes, finiteness, labels, and permutations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when any input is inconsistent.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ensemble_count: usize,
        training_rows: usize,
        row_stride: usize,
        feature_count: usize,
        x: &'a [f32],
        y: &'a [f32],
        class_permutations: &'a [u64],
        feature_permutations: &'a [u64],
        context_sha256: [u8; 32],
    ) -> Result<Self> {
        let invalid = |message: &str| Error::InvalidArgument(message.into());
        if ensemble_count == 0
            || ensemble_count > MAX_ENSEMBLE_COUNT
            || training_rows == 0
            || training_rows > MAX_TRAINING_ROWS
            || feature_count == 0
            || feature_count > MAX_FEATURE_COUNT
            || row_stride < training_rows
        {
            return Err(invalid(
                "static context request needs B, N, H > 0 within resource limits and row stride >= N",
            ));
        }
        if x.len() != checked_product(&[ensemble_count, row_stride, feature_count], "X element count")? {
            return Err(invalid("X length does not match [B, T, H]"));
        }
        if y.len() != checked_product(&[ensemble_count, training_rows], "y element count")? {
            return Err(invalid("y length does not match [B, N]"));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(invalid("X contains a non-finite value"));
        }
        let class_count = validate_labels(y, ensemble_count, training_rows)?;
        if context_sha256 == [0; 32] {
            return Err(invalid("context SHA-256 identity is empty"));
        }
        validate_estimator_permutations(class_permutations, ensemble_count, class_count, "class")
            .map_err(|error| invalid(&error.to_string()))?;
        validate_estimator_permutations(feature_permutations, ensemble_count, feature_count, "feature")
            .map_err(|error| invalid(&error.to_string()))?;
        let streams = feature_count + RESERVED;
        let column_elements = checked_product(&[training_rows, streams, COLUMN_DIM], "column activations")?;
        if column_elements.saturating_mul(4) > MAX_BUILD_ACTIVATION_BYTES {
            return Err(invalid(
                "training table is too large for one estimator's column activations",
            ));
        }
        Ok(Self {
            ensemble_count,
            training_rows,
            row_stride,
            feature_count,
            class_count,
            x,
            y,
            class_permutations,
            feature_permutations,
            context_sha256,
        })
    }

    /// Estimator count `B`.
    #[must_use]
    pub const fn ensemble_count(&self) -> usize {
        self.ensemble_count
    }
    /// Training row count `N`.
    #[must_use]
    pub const fn training_rows(&self) -> usize {
        self.training_rows
    }
    /// Feature count `H`.
    #[must_use]
    pub const fn feature_count(&self) -> usize {
        self.feature_count
    }
    /// Class count `C`, derived from the labels.
    #[must_use]
    pub const fn class_count(&self) -> usize {
        self.class_count
    }

    /// The shared class count of `[B, N]` labels, validated as the engine requires
    /// (contiguous `0..C-1` with the same `C` in every estimator).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the labels are invalid.
    pub fn class_count_of(labels: &[f32], ensemble_count: usize, training_rows: usize) -> Result<usize> {
        if labels.len() != checked_product(&[ensemble_count, training_rows], "label count")? {
            return Err(Error::InvalidArgument(
                "labels length does not match [B, N]".into(),
            ));
        }
        validate_labels(labels, ensemble_count, training_rows)
    }

    fn metadata(&self, model_sha256: [u8; 32]) -> StaticContextMetadata {
        StaticContextMetadata {
            model_sha256,
            context_sha256: self.context_sha256,
            ensemble_count: self.ensemble_count,
            training_rows: self.training_rows,
            feature_count: self.feature_count,
            class_count: self.class_count,
            class_permutations: self.class_permutations.to_vec(),
            feature_permutations: self.feature_permutations.to_vec(),
        }
    }

    fn features(&self, ensemble: usize) -> &'a [f32] {
        let offset = ensemble * self.row_stride * self.feature_count;
        &self.x[offset..][..self.training_rows * self.feature_count]
    }

    fn labels(&self, ensemble: usize) -> &'a [f32] {
        &self.y[ensemble * self.training_rows..][..self.training_rows]
    }
}

/// Storage choices for a written static context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StaticContextOptions {
    /// Storage precision of the ICL key/value tensors.
    pub kv_dtype: KvDtype,
    /// Store ICL keys in source-tile, dimension-major packed order (format version 3).
    pub pack_keys: bool,
    /// Build a coarse k-means index over the ICL keys with this many clusters per head.
    pub sparse_index_clusters: Option<usize>,
}

/// What [`build_static_context_file`] wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticContextSummary {
    pub ensemble_count: usize,
    pub training_rows: usize,
    pub feature_count: usize,
    pub class_count: usize,
    pub version: u32,
    pub kv_dtype: KvDtype,
    pub packed_icl_keys: bool,
    pub sparse_index_clusters: Option<usize>,
    pub file_bytes: u64,
    pub file_sha256: [u8; 32],
}

trait ContextSink {
    fn put_column(
        &mut self,
        layer: usize,
        kind: &str,
        ensemble: usize,
        stream: usize,
        values: &[f32],
    ) -> Result<()>;
    fn put_icl(&mut self, layer: usize, kind: &str, ensemble: usize, values: &[f32]) -> Result<()>;
    fn put_index(
        &mut self,
        name: &str,
        ensemble: usize,
        layer: usize,
        head: usize,
        values: &[f32],
    ) -> Result<()>;
}

impl ContextSink for StaticContextWriter {
    fn put_column(
        &mut self,
        layer: usize,
        kind: &str,
        ensemble: usize,
        stream: usize,
        values: &[f32],
    ) -> Result<()> {
        self.write_column(layer, kind, ensemble, stream, values)
    }
    fn put_icl(&mut self, layer: usize, kind: &str, ensemble: usize, values: &[f32]) -> Result<()> {
        self.write_icl(layer, kind, ensemble, values)
    }
    fn put_index(
        &mut self,
        name: &str,
        ensemble: usize,
        layer: usize,
        head: usize,
        values: &[f32],
    ) -> Result<()> {
        self.write_index(name, ensemble, layer, head, values)
    }
}

/// Collects exact FP32 tensors in memory for [`StaticContextArtifact::new`].
struct OwnedSink {
    ensemble_count: usize,
    training_rows: usize,
    streams: usize,
    tensors: std::collections::BTreeMap<String, (Vec<u64>, Vec<f32>)>,
}

impl OwnedSink {
    fn new(request: &StaticContextRequest<'_>) -> Result<Self> {
        let (b, n, streams) = (
            request.ensemble_count,
            request.training_rows,
            request.feature_count + RESERVED,
        );
        let mut tensors = std::collections::BTreeMap::new();
        let column = checked_product(&[b, streams, HEADS, INDUCING, COLUMN_HEAD_DIM], "column cache")?;
        let icl = checked_product(&[b, HEADS, n, ICL_HEAD_DIM], "ICL cache")?;
        for layer in 0..COLUMN_LAYERS {
            for kind in ["key", "value"] {
                tensors.insert(
                    format!("column.{layer}.{kind}"),
                    (
                        vec![
                            b as u64,
                            streams as u64,
                            HEADS as u64,
                            INDUCING as u64,
                            COLUMN_HEAD_DIM as u64,
                        ],
                        vec![0.0; column],
                    ),
                );
            }
        }
        for layer in 0..ICL_LAYERS {
            for kind in ["key", "value"] {
                tensors.insert(
                    format!("icl.{layer}.{kind}"),
                    (
                        vec![b as u64, HEADS as u64, n as u64, ICL_HEAD_DIM as u64],
                        vec![0.0; icl],
                    ),
                );
            }
        }
        Ok(Self {
            ensemble_count: b,
            training_rows: n,
            streams,
            tensors,
        })
    }

    fn store(&mut self, name: &str, offset: usize, values: &[f32]) -> Result<()> {
        let (_, target) = self
            .tensors
            .get_mut(name)
            .ok_or_else(|| Error::Artifact(format!("unexpected cache tensor '{name}'")))?;
        target
            .get_mut(offset..offset + values.len())
            .ok_or_else(|| Error::Artifact(format!("cache tensor '{name}' chunk is out of range")))?
            .copy_from_slice(values);
        Ok(())
    }

    fn into_tensors(self) -> Vec<StaticContextTensor> {
        self.tensors
            .into_iter()
            .map(|(name, (dimensions, values))| StaticContextTensor {
                name,
                dimensions,
                values,
            })
            .collect()
    }
}

impl ContextSink for OwnedSink {
    fn put_column(
        &mut self,
        layer: usize,
        kind: &str,
        ensemble: usize,
        stream: usize,
        values: &[f32],
    ) -> Result<()> {
        let chunk = HEADS * INDUCING * COLUMN_HEAD_DIM;
        self.store(
            &format!("column.{layer}.{kind}"),
            (ensemble * self.streams + stream) * chunk,
            values,
        )
    }
    fn put_icl(&mut self, layer: usize, kind: &str, ensemble: usize, values: &[f32]) -> Result<()> {
        let chunk = HEADS * self.training_rows * ICL_HEAD_DIM;
        debug_assert!(ensemble < self.ensemble_count);
        self.store(&format!("icl.{layer}.{kind}"), ensemble * chunk, values)
    }
    fn put_index(&mut self, _: &str, _: usize, _: usize, _: usize, _: &[f32]) -> Result<()> {
        Err(Error::NotImplemented(
            "in-memory static contexts do not carry a sparse index".into(),
        ))
    }
}

/// Build a static context and stream it to `output` with the storage `options`.
///
/// # Errors
///
/// Returns an error when the request is invalid, a model tensor is missing, or
/// the file cannot be written.
pub fn build_static_context_file(
    executor: &ClassifierExecutor,
    request: &StaticContextRequest<'_>,
    options: &StaticContextOptions,
    output: &Path,
) -> Result<StaticContextSummary> {
    let layout = StaticContextLayout::new(
        request.ensemble_count,
        request.training_rows,
        request.feature_count,
        options.kv_dtype,
        options.pack_keys,
        options.sparse_index_clusters,
    )?;
    let version = layout.version();
    let metadata = request.metadata(model_hash(executor)?);
    let mut writer = StaticContextWriter::create(output, &metadata, layout)?;
    for ensemble in 0..request.ensemble_count {
        build_ensemble(
            executor,
            request,
            ensemble,
            options.sparse_index_clusters,
            &mut writer,
        )?;
    }
    let file = writer.finish()?;
    Ok(StaticContextSummary {
        ensemble_count: request.ensemble_count,
        training_rows: request.training_rows,
        feature_count: request.feature_count,
        class_count: request.class_count,
        version,
        kv_dtype: options.kv_dtype,
        packed_icl_keys: options.pack_keys,
        sparse_index_clusters: options.sparse_index_clusters,
        file_bytes: file.file_bytes,
        file_sha256: file.file_sha256,
    })
}

/// Build an exact FP32 static context in memory (no sparse index).
///
/// # Errors
///
/// Returns an error when the request is invalid or a model tensor is missing.
pub fn build_static_context(
    executor: &ClassifierExecutor,
    request: &StaticContextRequest<'_>,
) -> Result<StaticContextArtifact> {
    let mut sink = OwnedSink::new(request)?;
    for ensemble in 0..request.ensemble_count {
        build_ensemble(executor, request, ensemble, None, &mut sink)?;
    }
    StaticContextArtifact::new(request.metadata(model_hash(executor)?), sink.into_tensors())
}

fn model_hash(executor: &ClassifierExecutor) -> Result<[u8; 32]> {
    decode_sha256(&executor.model().summary().source_sha256)
}

/// One column stream's cached keys and values for every column block.
struct ColumnStream {
    stream: usize,
    embedded: Vec<f32>,
    caches: Vec<(Vec<f32>, Vec<f32>)>,
}

fn build_ensemble(
    executor: &ClassifierExecutor,
    request: &StaticContextRequest<'_>,
    ensemble: usize,
    clusters: Option<usize>,
    sink: &mut dyn ContextSink,
) -> Result<()> {
    let (n, h) = (request.training_rows, request.feature_count);
    let streams = h + RESERVED;
    let x = request.features(ensemble);
    let indices = label_indices(request.labels(ensemble))?;
    let workers = executor_threads();

    // Column stage: every stream independently, as in the uncached executor.
    let mut column_targets = vec![0.0; n * COLUMN_DIM];
    one_hot_linear_f32(
        &indices,
        executor.tensor("col_embedder.y_encoder.weight")?,
        executor.tensor("col_embedder.y_encoder.bias")?,
        10,
        COLUMN_DIM,
        &mut column_targets,
    )?;
    let in_weight = executor.tensor("col_embedder.in_linear.weight")?;
    let in_bias = executor.tensor("col_embedder.in_linear.bias")?;
    let column_targets = column_targets.as_slice();
    let stream_workers = workers.min(streams);
    let chunk = streams.div_ceil(stream_workers);
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(stream_workers);
        for start in (0..streams).step_by(chunk) {
            let end = (start + chunk).min(streams);
            handles.push(scope.spawn(move || -> Result<Vec<ColumnStream>> {
                (start..end)
                    .map(|stream| {
                        column_stream(executor, x, column_targets, n, h, stream, in_weight, in_bias)
                    })
                    .collect()
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("column builder worker must not panic"))
            .collect::<Result<Vec<_>>>()
    })?;
    let mut columns = vec![0.0; checked_product(&[n, streams, COLUMN_DIM], "column activations")?];
    for result in results.into_iter().flatten() {
        for row in 0..n {
            columns[(row * streams + result.stream) * COLUMN_DIM..][..COLUMN_DIM]
                .copy_from_slice(&result.embedded[row * COLUMN_DIM..][..COLUMN_DIM]);
        }
        for (layer, (key, value)) in result.caches.iter().enumerate() {
            sink.put_column(layer, "key", ensemble, result.stream, key)?;
            sink.put_column(layer, "value", ensemble, result.stream, value)?;
        }
    }

    // Row stage.
    let mut rows = executor.row_embeddings(&columns, n, streams)?;
    drop(columns);

    // ICL stage: cache each block's source keys and values, then advance the rows.
    let mut targets = vec![0.0; n * ICL_DIM];
    one_hot_linear_f32(
        &indices,
        executor.tensor("icl_predictor.y_encoder.weight")?,
        executor.tensor("icl_predictor.y_encoder.bias")?,
        10,
        ICL_DIM,
        &mut targets,
    )?;
    for (row, target) in rows.iter_mut().zip(&targets) {
        *row += target;
    }
    for layer in 0..ICL_LAYERS {
        let prefix = format!("icl_predictor.tf_icl.blocks.{layer}");
        let (key, value) = executor.project_source_kv(&rows, n, ICL_DIM, &prefix)?;
        if let Some(clusters) = clusters {
            for head in 0..HEADS {
                let index =
                    coarse_key_index(&key[head * n * ICL_HEAD_DIM..][..n * ICL_HEAD_DIM], n, clusters);
                sink.put_index("icl.index.centroids", ensemble, layer, head, &index.centroids)?;
                sink.put_index("icl.index.radii", ensemble, layer, head, &index.radii)?;
                sink.put_index("icl.index.offsets", ensemble, layer, head, &index.offsets)?;
                sink.put_index("icl.index.members", ensemble, layer, head, &index.members)?;
            }
        }
        sink.put_icl(layer, "key", ensemble, &key)?;
        sink.put_icl(layer, "value", ensemble, &value)?;
        if layer + 1 < ICL_LAYERS {
            rows = icl_block(executor, &rows, n, &prefix, &key, &value, workers)?;
        }
    }
    Ok(())
}

/// Embed one column stream over the training rows and run the column blocks,
/// capturing each block's second-attention keys and values over the inducing
/// vectors. Reserved streams use the all-missing grouped source, as upstream.
#[allow(clippy::too_many_arguments)]
fn column_stream(
    executor: &ClassifierExecutor,
    x: &[f32],
    column_targets: &[f32],
    n: usize,
    h: usize,
    stream: usize,
    in_weight: &[f32],
    in_bias: &[f32],
) -> Result<ColumnStream> {
    let mut source = vec![MISSING; n * 3];
    if stream >= RESERVED {
        let feature = stream - RESERVED;
        for row in 0..n {
            for (part, offset) in [1_usize, 2, 4].iter().enumerate() {
                source[row * 3 + part] = x[row * h + (feature + offset) % h];
            }
        }
    }
    let mut embedded = vec![0.0; n * COLUMN_DIM];
    affine_f32(&source, n, 3, in_weight, Some(in_bias), COLUMN_DIM, &mut embedded)?;
    for row in 0..n {
        if source[row * 3..][..3].iter().all(|value| *value == MISSING) {
            embedded[row * COLUMN_DIM..][..COLUMN_DIM].fill(MISSING);
        }
    }
    for (value, target) in embedded.iter_mut().zip(column_targets) {
        *value += target;
    }
    let mut caches = Vec::with_capacity(COLUMN_LAYERS);
    for block in 0..COLUMN_LAYERS {
        let prefix = format!("col_embedder.tf_col.blocks.{block}");
        let inducing = executor.tensor(&format!("{prefix}.ind_vectors"))?;
        let attn1 = format!("{prefix}.multihead_attn1");
        let attn2 = format!("{prefix}.multihead_attn2");
        let hidden = executor.block(
            inducing, &embedded, &embedded, INDUCING, n, COLUMN_DIM, &attn1, true, None,
        )?;
        caches.push(executor.project_source_kv(&hidden, INDUCING, COLUMN_DIM, &attn2)?);
        embedded = executor.block(
            &embedded, &hidden, &hidden, n, INDUCING, COLUMN_DIM, &attn2, false, None,
        )?;
    }
    Ok(ColumnStream {
        stream,
        embedded,
        caches,
    })
}

/// One ICL block over the training rows, reusing the block's cached keys and
/// values and the parallel kernels the cached executor serves with.
fn icl_block(
    executor: &ClassifierExecutor,
    rows: &[f32],
    n: usize,
    prefix: &str,
    key: &[f32],
    value: &[f32],
    workers: usize,
) -> Result<Vec<f32>> {
    let dim = ICL_DIM;
    let attn_prefix = format!("{prefix}.attn");
    let mut q_norm = vec![0.0; rows.len()];
    layer_norm_f32(
        rows,
        n,
        dim,
        executor.tensor(&format!("{prefix}.norm1.weight"))?,
        Some(executor.tensor(&format!("{prefix}.norm1.bias"))?),
        &mut q_norm,
    )?;
    let weight = executor.tensor(&format!("{attn_prefix}.in_proj_weight"))?;
    let bias = executor.tensor(&format!("{attn_prefix}.in_proj_bias"))?;
    let mut q_proj = vec![0.0; rows.len()];
    parallel_affine(
        &q_norm,
        n,
        dim,
        &weight[..dim * dim],
        Some(&bias[..dim]),
        dim,
        &mut q_proj,
        workers,
    )?;
    let original = tokens_to_heads(&q_proj, n, dim, HEADS);
    let mut q_heads = vec![0.0; original.len()];
    executor.qass(&attn_prefix, &original, n, n, HEADS, ICL_HEAD_DIM, &mut q_heads)?;
    let mut attended = vec![0.0; q_heads.len()];
    parallel_cached_attention(
        crate::kernels::native::attention_backend(),
        &q_heads,
        n,
        key,
        value,
        n,
        ICL_HEAD_DIM,
        false,
        &mut attended,
        workers,
    )?;
    let merged = heads_to_tokens(&attended, n, dim, HEADS);
    let mut projected = vec![0.0; rows.len()];
    parallel_affine(
        &merged,
        n,
        dim,
        executor.tensor(&format!("{attn_prefix}.out_proj.weight"))?,
        Some(executor.tensor(&format!("{attn_prefix}.out_proj.bias"))?),
        dim,
        &mut projected,
        workers,
    )?;
    let mut residual = rows
        .iter()
        .zip(&projected)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let mut ff_norm = vec![0.0; residual.len()];
    layer_norm_f32(
        &residual,
        n,
        dim,
        executor.tensor(&format!("{prefix}.norm2.weight"))?,
        Some(executor.tensor(&format!("{prefix}.norm2.bias"))?),
        &mut ff_norm,
    )?;
    let mut ff = vec![0.0; checked_product(&[n, dim, 2], "feedforward hidden rows")?];
    parallel_affine(
        &ff_norm,
        n,
        dim,
        executor.tensor(&format!("{prefix}.linear1.weight"))?,
        Some(executor.tensor(&format!("{prefix}.linear1.bias"))?),
        dim * 2,
        &mut ff,
        workers,
    )?;
    let mut activated = vec![0.0; ff.len()];
    gelu_f32(&ff, &mut activated)?;
    let mut ff_out = vec![0.0; rows.len()];
    parallel_affine(
        &activated,
        n,
        dim * 2,
        executor.tensor(&format!("{prefix}.linear2.weight"))?,
        Some(executor.tensor(&format!("{prefix}.linear2.bias"))?),
        dim,
        &mut ff_out,
        workers,
    )?;
    for (value, addition) in residual.iter_mut().zip(ff_out) {
        *value += addition;
    }
    Ok(residual)
}

/// Row-chunked `affine_f32` over scoped threads.
#[allow(clippy::too_many_arguments)]
fn parallel_affine(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
    workers: usize,
) -> Result<()> {
    let workers = workers.min(rows);
    if workers <= 1 || rows < 16 {
        return affine_f32(input, rows, input_dim, weight, bias, output_dim, output);
    }
    let rows_per_job = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for (input, output) in input
            .chunks(input_dim * rows_per_job)
            .zip(output.chunks_mut(output_dim * rows_per_job))
        {
            handles.push(scope.spawn(move || {
                affine_f32(
                    input,
                    input.len() / input_dim,
                    input_dim,
                    weight,
                    bias,
                    output_dim,
                    output,
                )
            }));
        }
        handles
            .into_iter()
            .try_for_each(|handle| handle.join().expect("affine worker must not panic"))
    })
}

/// A deterministic coarse index over one attention head's `[N, 64]` keys.
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseKeyIndex {
    /// `[clusters, 64]` cluster centres.
    pub centroids: Vec<f32>,
    /// `[clusters]` largest member distance from each centre.
    pub radii: Vec<f32>,
    /// `[clusters + 1]` member offsets, as integers stored in FP32.
    pub offsets: Vec<f32>,
    /// `[N]` source rows grouped by cluster, as integers stored in FP32.
    pub members: Vec<f32>,
}

/// Seed centroids at evenly spaced rows, run a few Lloyd iterations, and group
/// the rows by cluster in a stable order. The index is an acceleration
/// structure whose decisions the executor certifies at query time, so only its
/// well-formedness matters.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn coarse_key_index(keys: &[f32], rows: usize, clusters: usize) -> CoarseKeyIndex {
    let dim = ICL_HEAD_DIM;
    debug_assert_eq!(keys.len(), rows * dim);
    debug_assert!(1 < clusters && clusters <= rows);
    let mut centroids = vec![0.0; clusters * dim];
    for cluster in 0..clusters {
        let position = if clusters == 1 {
            0.0
        } else {
            (rows - 1) as f64 * cluster as f64 / (clusters - 1) as f64
        };
        let row = (position as usize).min(rows - 1);
        centroids[cluster * dim..][..dim].copy_from_slice(&keys[row * dim..][..dim]);
    }
    let mut assignments = vec![0_usize; rows];
    for _ in 0..KMEANS_ITERATIONS {
        for row in 0..rows {
            let key = &keys[row * dim..][..dim];
            let mut best = (f32::INFINITY, 0);
            for cluster in 0..clusters {
                let centre = &centroids[cluster * dim..][..dim];
                let distance = key
                    .iter()
                    .zip(centre)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>();
                if distance < best.0 {
                    best = (distance, cluster);
                }
            }
            assignments[row] = best.1;
        }
        let mut sums = vec![0.0_f64; clusters * dim];
        let mut counts = vec![0_usize; clusters];
        for row in 0..rows {
            let cluster = assignments[row];
            counts[cluster] += 1;
            for (sum, value) in sums[cluster * dim..][..dim]
                .iter_mut()
                .zip(&keys[row * dim..][..dim])
            {
                *sum += f64::from(*value);
            }
        }
        for cluster in 0..clusters {
            if counts[cluster] > 0 {
                for (centre, sum) in centroids[cluster * dim..][..dim]
                    .iter_mut()
                    .zip(&sums[cluster * dim..][..dim])
                {
                    *centre = (*sum / counts[cluster] as f64) as f32;
                }
            }
        }
    }
    let mut order = (0..rows).collect::<Vec<_>>();
    order.sort_by_key(|row| assignments[*row]);
    let mut counts = vec![0_usize; clusters];
    for cluster in &assignments {
        counts[*cluster] += 1;
    }
    let mut offsets = Vec::with_capacity(clusters + 1);
    offsets.push(0.0);
    let mut total = 0_usize;
    for count in &counts {
        total += count;
        offsets.push(total as f32);
    }
    let mut radii = vec![0.0_f32; clusters];
    for row in 0..rows {
        let cluster = assignments[row];
        let distance = keys[row * dim..][..dim]
            .iter()
            .zip(&centroids[cluster * dim..][..dim])
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        radii[cluster] = radii[cluster].max(distance);
    }
    CoarseKeyIndex {
        centroids,
        radii,
        offsets,
        members: order.iter().map(|row| *row as f32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{StaticContextRequest, coarse_key_index};

    #[test]
    fn request_rejects_inconsistent_shapes_and_labels() {
        let x = vec![0.5; 2 * 3 * 2];
        let y = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let classes = [0, 1, 1, 0];
        let features = [0, 1, 1, 0];
        assert!(StaticContextRequest::new(2, 3, 3, 2, &x, &y, &classes, &features, [1; 32]).is_ok());
        assert!(StaticContextRequest::new(2, 3, 2, 2, &x, &y, &classes, &features, [1; 32]).is_err());
        assert!(StaticContextRequest::new(2, 3, 3, 2, &x[1..], &y, &classes, &features, [1; 32]).is_err());
        assert!(StaticContextRequest::new(2, 3, 3, 2, &x, &y[1..], &classes, &features, [1; 32]).is_err());
        assert!(StaticContextRequest::new(2, 3, 3, 2, &x, &y, &[0, 0, 1, 0], &features, [1; 32]).is_err());
        assert!(StaticContextRequest::new(2, 3, 3, 2, &x, &y, &classes, &features, [0; 32]).is_err());
        let mut bad = x.clone();
        bad[0] = f32::NAN;
        assert!(StaticContextRequest::new(2, 3, 3, 2, &bad, &y, &classes, &features, [1; 32]).is_err());
        let one_class = vec![0.0; 6];
        assert!(StaticContextRequest::new(2, 3, 3, 2, &x, &one_class, &classes, &features, [1; 32]).is_err());
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn coarse_index_groups_every_row_once() {
        let (rows, clusters) = (37, 5);
        let keys = (0..rows * 64)
            .map(|index| ((index * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect::<Vec<_>>();
        let index = coarse_key_index(&keys, rows, clusters);
        assert_eq!(index.centroids.len(), clusters * 64);
        assert_eq!(index.offsets.first(), Some(&0.0));
        assert_eq!(index.offsets.last(), Some(&(rows as f32)));
        assert!(index.offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        let mut members = index
            .members
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        members.sort_unstable();
        assert_eq!(members, (0..rows).collect::<Vec<_>>());
        assert!(index.radii.iter().all(|radius| *radius >= 0.0));
    }
}

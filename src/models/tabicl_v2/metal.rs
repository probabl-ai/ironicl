//! `TabICLv2` on Metal, built with `MPSGraph`.
//!
//! Each context gets two graphs with its ensemble, feature, and training-row
//! counts fixed: a predict graph (column, row, and ICL stages and the decoder
//! for every estimator in one run, with only the query count unknown) and a
//! build graph (the training rows through every stage, capturing each block's
//! keys and values). Model weights live once per model in shared GPU buffers
//! and are fed to the graphs as placeholders, so contexts do not copy them.

#![allow(unsafe_code)]
// Objective-C calls; each `unsafe` block states its invariant.
// Tensor dimensions are model constants or validated sizes far below `isize::MAX`.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Mutex};

use objc2::{AllocAnyThread, rc::Retained};
use objc2_foundation::{NSArray, NSDictionary};
use objc2_metal::MTLBuffer;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphDevice, MPSGraphExecutable, MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
};

use crate::kernels::metal::{
    Binding, Buffer, FLOAT, Gpu, ModelWeights, Tensor, Weights, builder::Builder, dictionary, kind,
    metal_error, shape,
};
use crate::{
    Result,
    format::static_context::TensorStorage,
    models::tabicl_v2::cpu::{
        COLUMN_BLOCKS, COLUMN_DIM, HEADS, ICL_BLOCKS, ICL_DIM, INDUCING, QUERY_CHUNK_ROWS, RESERVED,
        rope_tables,
    },
};

const ICL_HEAD_DIM: usize = ICL_DIM / HEADS;
const COLUMN_HEAD_DIM: usize = COLUMN_DIM / HEADS;

/// A column block's cached keys or values: `[B, streams, 8, 128, 16]`.
fn column_cache_shape(ensembles: isize, streams: isize) -> [isize; 5] {
    [
        ensembles,
        streams,
        HEADS as isize,
        INDUCING as isize,
        COLUMN_HEAD_DIM as isize,
    ]
}

/// An ICL block's cached keys or values: `[B, 8, N, 64]`.
fn icl_cache_shape(ensembles: isize, training_rows: isize) -> [isize; 4] {
    [ensembles, HEADS as isize, training_rows, ICL_HEAD_DIM as isize]
}

/// Smallest training set that builds its context on the GPU. Building costs
/// about 250 ms of graph setup on an M1 Max, so the CPU is faster below about
/// 150 rows. Prediction always uses the GPU when the context is there: one
/// batched run beat the CPU from a single query row.
pub(crate) const GPU_MIN_BUILD_ROWS: usize = 160;

/// An upper bound on one estimator's share of the GPU build's peak memory.
/// Measured on an M1 Max: the peak grows linearly in N, with no N x N term,
/// by about 1.7 GB per estimator at 8192 x 54; this estimates 2.2 GB.
pub(crate) fn gpu_build_estimator_bytes(n: usize, h: usize) -> usize {
    crate::memory::gpu_build_estimator_bytes(n, h)
}

/// Group each feature of `[rows, H]` inputs with the features 1, 2, and 4
/// places after it, as `[H, rows, 3]`. Returns `None` when a triple is the
/// all-missing marker, which only the CPU column stage handles.
pub(crate) fn group_features(x: &[f32], rows: usize, h: usize) -> Option<Vec<f32>> {
    let mut grouped = vec![0.0; h * rows * 3];
    group_features_into(x, (rows, h), rows, &mut grouped).then_some(grouped)
}

/// [`group_features`] into `[H, stride, 3]` output, leaving rows past `rows`
/// untouched. Returns false on the all-missing marker.
#[allow(clippy::float_cmp)] // The marker is an exact sentinel value.
pub(crate) fn group_features_into(
    x: &[f32],
    (rows, h): (usize, usize),
    stride: usize,
    output: &mut [f32],
) -> bool {
    for feature in 0..h {
        for row in 0..rows {
            let triple = &mut output[(feature * stride + row) * 3..][..3];
            for (part, offset) in [1_usize, 2, 4].iter().enumerate() {
                triple[part] = x[row * h + (feature + offset) % h];
            }
            if triple.iter().all(|value| *value == -100.0) {
                return false;
            }
        }
    }
    true
}

/// Attention sources: cached keys and values, or tokens projected in the block.
enum Source<'t> {
    Cached(&'t Tensor, &'t Tensor),
    Tokens(&'t Tensor),
}

/// `RoPE` tables for queries and keys, each `[rows, head_dim]` cos and sin.
struct Rope<'t> {
    query: (&'t Tensor, &'t Tensor),
    key: (&'t Tensor, &'t Tensor),
}

// SAFETY (all methods): plain `MPSGraph` construction on `self.graph`, as in
// the `Builder` impl: every tensor came from `self.graph`, names are `None`, and
// no call touches buffer memory.
impl Builder<'_, '_> {
    /// A `QASSMax` base scale, `[heads, head_dim]` values, shaped to scale
    /// `[lead, heads, rows, head_dim]` queries.
    fn base_scale(&self, values: &[f32], head_dim: usize) -> Tensor {
        self.constant(values, &[1, HEADS as isize, 1, head_dim as isize])
    }

    /// `[lead, rows, heads * head_dim]` -> `[lead, heads, rows, head_dim]`.
    fn split_heads(&self, input: &Tensor, lead: isize, rows: isize, head_dim: usize) -> Tensor {
        self.transpose(
            &self.reshape(input, &[lead, rows, HEADS as isize, head_dim as isize]),
            1,
            2,
        )
    }

    /// One pre-norm transformer block: attention with a residual, then a GELU
    /// feed-forward with a residual. `input` is `[lead, q_rows, dim]`. Returns
    /// the output and the head-major keys and values it attended to.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        input: &Tensor,
        source: &Source<'_>,
        prefix: &str,
        (lead, q_rows, source_rows): (isize, isize, isize),
        dim: usize,
        rope: Option<&Rope<'_>>,
        qass: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let head_dim = dim / HEADS;
        let norm = format!("{prefix}.norm1");
        let attn = format!("{prefix}.attn");
        // The query, key, and value projections are parts 0, 1, and 2 of one fused weight.
        let in_proj = (format!("{attn}.in_proj_weight"), format!("{attn}.in_proj_bias"));
        let project = |part: usize, x: &Tensor| self.affine(x, (&in_proj.0, &in_proj.1), part, (dim, dim));
        let query = project(0, &self.layer_norm(input, &norm, dim)?)?;
        let mut query = self.split_heads(&query, lead, q_rows, head_dim);
        if let Some(base_scale) = qass {
            // QASSMax: q * base_scale * (1 + tanh(mlp(q))), the MLP shared across heads.
            let ssmax = format!("{attn}.ssmax_layer");
            let hidden = self.gelu(&self.linear(&query, &format!("{ssmax}.query_mlp.0"), head_dim, 64)?);
            let modulation = self.linear(&hidden, &format!("{ssmax}.query_mlp.2"), 64, head_dim)?;
            // SAFETY: graph construction on `self.graph` (impl comment).
            let gate = self.add(
                &unsafe { self.graph.tanhWithTensor_name(&modulation, None) },
                &self.scalar(1.0),
            );
            query = self.multiply(&self.multiply(&query, base_scale), &gate);
        }
        let (key, value) = match *source {
            Source::Cached(key, value) => (key.clone(), value.clone()),
            Source::Tokens(tokens) => {
                let normalized = self.layer_norm(tokens, &norm, dim)?;
                (
                    self.split_heads(&project(1, &normalized)?, lead, source_rows, head_dim),
                    self.split_heads(&project(2, &normalized)?, lead, source_rows, head_dim),
                )
            }
        };
        let (query, key) = match rope {
            Some(rope) => (
                self.rope(&query, rope.query, head_dim),
                self.rope(&key, rope.key, head_dim),
            ),
            None => (query, key),
        };
        // Reduced-precision caches: attention runs in their type, so the
        // caches are read as stored instead of converted to an FP32 copy.
        // Stored 16-bit caches set the attention type; otherwise `self.attention` does.
        // SAFETY: a property read of a live tensor.
        let stored = unsafe { key.dataType() };
        let kind = if stored == FLOAT {
            self.attention.get()
        } else {
            stored
        };
        // SAFETY: graph construction on `self.graph` (impl comment).
        let attended = unsafe {
            self.graph
                .scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name(
                    &self.cast(&query, kind),
                    &self.cast(&key, kind),
                    &self.cast(&value, kind),
                    (head_dim as f32).sqrt().recip(),
                    None,
                )
        };
        let attended = self.cast(&attended, FLOAT);
        let merged = self.reshape(&self.transpose(&attended, 1, 2), &[lead, q_rows, dim as isize]);
        let residual = self.add(
            input,
            &self.linear(&merged, &format!("{attn}.out_proj"), dim, dim)?,
        );
        let hidden = self.gelu(&self.linear(
            &self.layer_norm(&residual, &format!("{prefix}.norm2"), dim)?,
            &format!("{prefix}.linear1"),
            dim,
            2 * dim,
        )?);
        let output = self.add(
            &residual,
            &self.linear(&hidden, &format!("{prefix}.linear2"), 2 * dim, dim)?,
        );
        Ok((output, key, value))
    }

    /// The row stage for `[B * H, rows, 128]` column embeddings: the row
    /// transformer over `[B * rows, 4 + H, 128]` sequences with the CLS tokens
    /// first. Returns `[B, rows, 512]`.
    fn row_stage(&self, columns: &Tensor, (ensembles, features): (usize, usize)) -> Result<Tensor> {
        let (b, h) = (ensembles as isize, features as isize);
        let streams = h + RESERVED as isize;
        // [B * H, rows, 128] -> [B, rows, H, 128] -> [B * rows, H, 128]
        let columns = self.transpose(&self.reshape(columns, &[b, h, -1, COLUMN_DIM as isize]), 1, 2);
        let columns = self.reshape(&columns, &[-1, h, COLUMN_DIM as isize]);
        let zeros = self.multiply(&self.slice(&columns, 1, 0, 1), &self.scalar(0.0));
        let cls_tokens = self.weight("row_interactor.cls_tokens")?;
        let cls = self.add(
            &zeros,
            &self.model_weight(
                "row_interactor.cls_tokens",
                || cls_tokens[..RESERVED * COLUMN_DIM].to_vec(),
                &[RESERVED as isize, COLUMN_DIM as isize],
            )?,
        );
        let mut sequence = self.concat(&cls, &columns, 1);
        let frequencies = self.weight("row_interactor.tf_row.rope.freqs")?;
        let table = |rows: usize| {
            let (cos, sin) = rope_tables(rows, frequencies);
            let dims = [rows as isize, COLUMN_HEAD_DIM as isize];
            (self.constant(&cos, &dims), self.constant(&sin, &dims))
        };
        let (all, cls_rows) = (table(features + RESERVED), table(RESERVED));
        for block in 0..2 {
            let prefix = format!("row_interactor.tf_row.blocks.{block}");
            (sequence, _, _) = self.block(
                &sequence,
                &Source::Tokens(&sequence),
                &prefix,
                (-1, streams, streams),
                COLUMN_DIM,
                Some(&Rope {
                    query: (&all.0, &all.1),
                    key: (&all.0, &all.1),
                }),
                None,
            )?;
        }
        // The last block updates only the CLS tokens.
        let prefix = "row_interactor.tf_row.blocks.2";
        let (cls_out, _, _) = self.block(
            &self.slice(&sequence, 1, 0, RESERVED as isize),
            &Source::Tokens(&sequence),
            prefix,
            (-1, RESERVED as isize, streams),
            COLUMN_DIM,
            Some(&Rope {
                query: (&cls_rows.0, &cls_rows.1),
                key: (&all.0, &all.1),
            }),
            None,
        )?;
        Ok(self.reshape(
            &self.layer_norm(&cls_out, "row_interactor.out_ln", COLUMN_DIM)?,
            &[b, -1, ICL_DIM as isize],
        ))
    }
}

/// Context dimensions shared by both graphs.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) ensembles: usize,
    pub(crate) features: usize,
    pub(crate) training_rows: usize,
}

/// One context's Metal buffers and the graph that reads them.
pub(crate) struct GpuContext {
    pub(crate) predict: PredictGraph,
    /// How queries split into graph runs, `(chunk, small)` from
    /// [`PredictGraph::query_split`], which keeps the buckets the
    /// graph keeps within the device's share of memory.
    pub(crate) split: (usize, usize),
    /// Keeps page-aligned context memory alive while GPU buffers wrap it.
    pub(crate) _storage: Option<std::sync::Arc<memmap2::Mmap>>,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("GpuContext")
    }
}

/// One context's prediction graph: every stage for every estimator in one
/// run. The context's keys and values are bound to it.
pub(crate) struct PredictGraph {
    gpu: Gpu,
    graph: Retained<MPSGraph>,
    shape: Shape,
    bindings: Vec<Binding>,
    grouped: Tensor,
    logits: Tensor,
    /// Decoder outputs per row.
    outputs: usize,
    /// Each query bucket's buffers and executable. The lock also serializes
    /// graph runs.
    prepared: Mutex<HashMap<usize, Prepared>>,
}

/// Round a query count up to a bucket that keeps its top four significant
/// bits, so the graph compiles for 8 shapes per doubling of the batch size and
/// pads at most 12.5%. Power-of-two buckets padded up to 2x, which cost more
/// than the compiles they saved on large batches.
fn bucket(rows: usize) -> usize {
    let shift = rows.bit_width().saturating_sub(4);
    rows.next_multiple_of(1 << shift)
}

/// How to split queries so the buckets a predict graph keeps stay within
/// `budget` bytes, at `row_bytes` per bucket row: `(chunk, small)`, powers of
/// two with `small <= chunk <= QUERY_CHUNK_ROWS`. A query of at most `small`
/// rows runs as one bucket; a larger one runs in power-of-two pieces of at most
/// `chunk` rows until at most `small` remain, so the graph keeps only the
/// buckets up to `small` and the powers of two up to `chunk`.
///
/// `MPSGraph` keeps each input shape's intermediate buffers until the graph is
/// freed, so the graph keeps every bucket it has run. The powers of two take
/// at most half the budget, as large a `chunk` as fits, and the small buckets
/// the rest. A row is estimated as building it for every estimator; the graph
/// measured 0.7x this.
fn query_split(row_bytes: usize, budget: u64) -> (usize, usize) {
    let bytes = |rows: usize| u64::try_from(rows.saturating_mul(row_bytes)).unwrap_or(u64::MAX);
    let mut chunk = QUERY_CHUNK_ROWS;
    while chunk > 1 && bytes(2 * chunk) > budget / 2 {
        chunk /= 2;
    }
    let mut small = chunk;
    while small > 1 && bytes(kept_rows(chunk, small)) > budget {
        small /= 2;
    }
    (chunk, small)
}

/// The bucket rows a predict graph keeps once it has run every split of
/// [`query_split`]: each bucket up to `small`, then the powers of two up to
/// `chunk`.
fn kept_rows(chunk: usize, small: usize) -> usize {
    let (mut rows, mut kept) = (1, 0);
    while rows <= small {
        let bucket = bucket(rows);
        kept += bucket;
        rows = bucket + 1;
    }
    kept + 2 * (chunk - small)
}

/// One query bucket's input and output buffers and its compiled executable,
/// with its inputs in feed order. Skipping the per-call executable lookup saved
/// 0.5-1.4 ms per single-row predict.
struct Prepared {
    input: Buffer,
    output: Buffer,
    executable: Retained<MPSGraphExecutable>,
    inputs: Retained<NSArray<MPSGraphTensorData>>,
    results: Retained<NSArray<MPSGraphTensorData>>,
}

// SAFETY: after `new`, `graph`, the tensors, and the bindings are only used by
// `prepare` and `run`, which hold the `prepared` lock, so one thread at a time
// compiles or runs this graph and writes its input buffer. The `Gpu` device and
// queue are thread-safe (`Send + Sync` in objc2-metal); its descriptors are
// shared with other graphs but only read. Retain and release are atomic.
unsafe impl Send for PredictGraph {}
// SAFETY: as above.
unsafe impl Sync for PredictGraph {}

impl PredictGraph {
    /// `column_keys` and `column_values` hold each column block's
    /// `[B, H + 4, 8, 128, 16]` cache, `icl_keys` and `icl_values` each ICL
    /// block's `[B, 8, N, 64]` in `icl`'s storage type, and `base_scales` each
    /// ICL block's `[8, 64]`. With `page_aligned`, the caches are wrapped in
    /// place instead of copied: each must start a page of page-padded memory
    /// that stays alive and unwritten while the graph lives ([`Gpu::wrap`]'s
    /// contract; `GpuContext::_storage` keeps it alive).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: &ModelWeights,
        weights: &Weights<'_, '_>,
        shape: Shape,
        column_keys: &[&[f32]],
        column_values: &[&[f32]],
        icl: (&[&[u8]], &[&[u8]], TensorStorage),
        base_scales: &[Vec<f32>],
        page_aligned: bool,
    ) -> Result<Self> {
        let (icl_keys, icl_values, icl_storage) = icl;
        let gpu = model.gpu();
        let builder = Builder::new(model, weights);
        let (b, h, n) = (
            shape.ensembles as isize,
            shape.features as isize,
            shape.training_rows as isize,
        );
        // Column stage, feature-major: [B * H, Q, 3] grouped inputs -> [B * H, Q, 128].
        let grouped = builder.placeholder(&[b * h, -1, 3]);
        let mut columns = builder.linear(&grouped, "col_embedder.in_linear", 3, COLUMN_DIM)?;
        let column_shape = column_cache_shape(b, h + RESERVED as isize);
        for block in 0..COLUMN_BLOCKS {
            let prefix = format!("col_embedder.tf_col.blocks.{block}.multihead_attn2");
            // The cached path never reads the reserved streams.
            let cache = |values: &[f32]| -> Result<Tensor> {
                let full = builder.bound(
                    gpu.tensor_buffer(bytemuck::cast_slice(values), page_aligned)?,
                    &column_shape,
                );
                Ok(builder.reshape(
                    &builder.slice(&full, 1, RESERVED as isize, h),
                    &[b * h, HEADS as isize, INDUCING as isize, COLUMN_HEAD_DIM as isize],
                ))
            };
            let (key, value) = (cache(column_keys[block])?, cache(column_values[block])?);
            (columns, _, _) = builder.block(
                &columns,
                &Source::Cached(&key, &value),
                &prefix,
                (b * h, -1, INDUCING as isize),
                COLUMN_DIM,
                None,
                None,
            )?;
        }
        let mut hidden = builder.row_stage(&columns, (shape.ensembles, shape.features))?;
        let icl_shape = icl_cache_shape(b, n);
        for block in 0..ICL_BLOCKS {
            let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
            // Attention runs in the caches' stored type.
            let key = builder.bound_typed(
                gpu.tensor_buffer(icl_keys[block], page_aligned)?,
                &icl_shape,
                kind(icl_storage),
            );
            let value = builder.bound_typed(
                gpu.tensor_buffer(icl_values[block], page_aligned)?,
                &icl_shape,
                kind(icl_storage),
            );
            let base_scale = builder.base_scale(&base_scales[block], ICL_HEAD_DIM);
            (hidden, _, _) = builder.block(
                &hidden,
                &Source::Cached(&key, &value),
                &prefix,
                (b, -1, n),
                ICL_DIM,
                None,
                Some(&base_scale),
            )?;
        }
        let hidden = builder.gelu(&builder.linear(
            &builder.layer_norm(&hidden, "icl_predictor.ln", ICL_DIM)?,
            "icl_predictor.decoder.0",
            ICL_DIM,
            1024,
        )?);
        let outputs = weights("icl_predictor.decoder.2.bias")?.len();
        let logits = builder.linear(&hidden, "icl_predictor.decoder.2", 1024, outputs)?;
        Ok(Self {
            gpu: gpu.clone(),
            graph: builder.graph,
            shape,
            bindings: builder.bindings.into_inner(),
            grouped,
            logits,
            outputs,
            prepared: Mutex::default(),
        })
    }

    /// Allocate a bucket's buffers and compile its executable.
    fn prepare(&self, rows: usize) -> Result<Prepared> {
        let (b, h) = (self.shape.ensembles, self.shape.features);
        let input = self.gpu.empty(b * h * rows * 3)?;
        // Padding rows must be finite; they never mix with real rows.
        // SAFETY: `empty` made a shared buffer of `b * h * rows * 3` floats that no
        // one else has yet; `contents` is its CPU-visible, page-aligned start.
        unsafe { std::ptr::write_bytes(input.contents().as_ptr().cast::<f32>(), 0, b * h * rows * 3) };
        let output = self.gpu.empty(b * rows * self.outputs)?;
        let input = (
            self.grouped.clone(),
            input,
            vec![(b * h) as isize, rows as isize, 3],
            FLOAT,
        );
        let output = (
            self.logits.clone(),
            output,
            vec![b as isize, rows as isize, self.outputs as isize],
            FLOAT,
        );
        let mut inputs = self.bindings.iter().collect::<Vec<_>>();
        inputs.push(&input);
        let (feeds, results) = (dictionary(&inputs), dictionary(&[&output]));
        // SAFETY: compiles `self.graph` (under the caller's `prepared` lock) for
        // its own feed tensors, typed with valid shapes; touches no buffer memory.
        let executable = unsafe {
            let tensors: Vec<&MPSGraphTensor> = inputs.iter().map(|item| &*item.0).collect();
            let types: Vec<_> = inputs
                .iter()
                .map(|item| {
                    MPSGraphShapedType::initWithShape_dataType(
                        MPSGraphShapedType::alloc(),
                        Some(&shape(&item.2)),
                        item.3,
                    )
                })
                .collect();
            self.graph
                .compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                    Some(&MPSGraphDevice::deviceWithMTLDevice(&self.gpu.device)),
                    &NSDictionary::from_retained_objects(&tensors, &types),
                    &NSArray::from_slice(&[&*self.logits]),
                    None,
                    Some(&self.gpu.compilation),
                )
        };
        // SAFETY: a property read of the executable compiled above.
        let order =
            unsafe { executable.feedTensors() }.ok_or_else(|| metal_error("compiled graph has no feeds"))?;
        let ordered = order
            .iter()
            .map(|tensor| {
                feeds
                    .objectForKey(&tensor)
                    .ok_or_else(|| metal_error("compiled graph feed is unbound"))
            })
            .collect::<Result<Vec<_>>>()?;
        let result = results
            .objectForKey(&self.logits)
            .ok_or_else(|| metal_error("graph output is unbound"))?;
        Ok(Prepared {
            input: input.1,
            output: output.1,
            executable,
            inputs: NSArray::from_retained_slice(&ordered),
            results: NSArray::from_retained_slice(&[result]),
        })
    }

    /// How to split queries so the buckets the graph keeps stay within
    /// `budget` bytes of device memory; see [`query_split`].
    pub(crate) fn query_split(&self, budget: u64) -> (usize, usize) {
        query_split(
            gpu_build_estimator_bytes(1, self.shape.features).saturating_mul(self.shape.ensembles),
            budget,
        )
    }

    /// Predict `query_rows` rows, at most [`QUERY_CHUNK_ROWS`]; split queries
    /// as [`Self::query_split`] says to stay within the budget. `fill` writes
    /// the `[B * H, rows, 3]` grouped inputs, given the bucket's row count as
    /// the stride, and returns false to cancel. Returns `[B, query_rows,
    /// outputs]` decoder outputs.
    pub(crate) fn run(
        &self,
        query_rows: usize,
        fill: impl FnOnce(&mut [f32], usize) -> bool,
    ) -> Result<Option<Vec<f32>>> {
        let (b, h, rows) = (self.shape.ensembles, self.shape.features, bucket(query_rows));
        if rows > QUERY_CHUNK_ROWS {
            return Err(metal_error("query chunk exceeds the largest bucket"));
        }
        let mut cache = self
            .prepared
            .lock()
            .map_err(|_| metal_error("graph lock poisoned"))?;
        let prepared = match cache.entry(rows) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(self.prepare(rows)?),
        };
        // SAFETY: `prepare(rows)` sized the shared buffer to `b * h * rows * 3`
        // floats; the held lock excludes other writers and the GPU, whose runs
        // on this buffer are synchronous and also take the lock.
        let input = unsafe {
            std::slice::from_raw_parts_mut(prepared.input.contents().as_ptr().cast::<f32>(), b * h * rows * 3)
        };
        if !fill(input, rows) {
            return Ok(None);
        }
        // SAFETY: `executable_execution` waits until completion; the tensor data in
        // `prepared` may not retain their buffers, but `prepared` and `self.bindings`
        // do, and the page-aligned caches outlive `self` (`GpuContext::_storage`).
        unsafe {
            prepared
                .executable
                .runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                    &self.gpu.queue,
                    &prepared.inputs,
                    Some(&prepared.results),
                    Some(&self.gpu.executable_execution),
                );
        }
        // SAFETY: the synchronous run above finished, the shared buffer holds
        // `b * rows * outputs` floats (`prepare`), and the lock is still held.
        let output = unsafe {
            std::slice::from_raw_parts(
                prepared.output.contents().as_ptr().cast::<f32>(),
                b * rows * self.outputs,
            )
        };
        Ok(Some(
            output
                .chunks_exact(rows * self.outputs)
                .flat_map(|estimator| &estimator[..query_rows * self.outputs])
                .copied()
                .collect(),
        ))
    }
}

/// Build a context on the GPU: the training rows of every estimator run
/// through every stage once, and each column and ICL block's keys and values
/// are captured. `column_base_scales` and `icl_base_scales` are the `QASSMax`
/// base scales for `N` sources: 3 of `[8, 16]` and 12 of `[8, 64]`.
/// `grouped` is `[B * H, N, 3]`, `column_labels` `[B, N, 128]`, and
/// `icl_labels` `[B, N, 512]`. The keys and values are written in place into
/// `outputs`, page-aligned regions in context order: each column block's key
/// and value as `[B, H + 4, 8, 128, 16]` with zero reserved streams, then each
/// ICL block's as `[B, 8, N, 64]` in `icl_storage`'s type. Each region must
/// satisfy [`Gpu::wrap`]'s contract and stay untouched until this returns.
/// Returns an error if any written value is not finite, including FP16 overflow.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // One linear graph definition.
pub(crate) fn build_context(
    model: &ModelWeights,
    weights: &Weights<'_, '_>,
    shape: Shape,
    column_base_scales: &[Vec<f32>],
    icl_base_scales: &[Vec<f32>],
    grouped: &[f32],
    column_labels: &[f32],
    icl_labels: &[f32],
    icl_storage: TensorStorage,
    outputs: &[(std::ptr::NonNull<u8>, usize)],
) -> Result<()> {
    let gpu = model.gpu();
    let builder = Builder::new(model, weights);
    let (b, h, n) = (
        shape.ensembles as isize,
        shape.features as isize,
        shape.training_rows as isize,
    );
    let grouped_input = builder.bound(gpu.buffer(grouped)?, &[b * h, n, 3]);
    let column_labels = builder.bound(gpu.buffer(column_labels)?, &[b, 1, n, COLUMN_DIM as isize]);
    let icl_labels = builder.bound(gpu.buffer(icl_labels)?, &[b, n, ICL_DIM as isize]);
    // [B * H, N, 128] embeddings plus each estimator's label embeddings.
    let embedded = builder.reshape(
        &builder.linear(&grouped_input, "col_embedder.in_linear", 3, COLUMN_DIM)?,
        &[b, h, n, COLUMN_DIM as isize],
    );
    let mut source = builder.reshape(
        &builder.add(&embedded, &column_labels),
        &[b * h, n, COLUMN_DIM as isize],
    );
    let zeros = builder.multiply(&builder.slice(&source, 1, 0, 1), &builder.scalar(0.0));
    let mut column = Vec::with_capacity(COLUMN_BLOCKS);
    for (block, base_scale) in column_base_scales.iter().enumerate() {
        // ISAB: the inducing points attend to the training rows, then the rows
        // attend back; the second attention's keys and values are cached.
        let prefix = format!("col_embedder.tf_col.blocks.{block}");
        let inducing = builder.add(
            &zeros,
            &builder.named_weight(
                &format!("{prefix}.ind_vectors"),
                &[INDUCING as isize, COLUMN_DIM as isize],
            )?,
        );
        let base_scale = builder.base_scale(base_scale, COLUMN_HEAD_DIM);
        let first = format!("{prefix}.multihead_attn1");
        let (hidden, _, _) = builder.block(
            &inducing,
            &Source::Tokens(&source),
            &first,
            (b * h, INDUCING as isize, n),
            COLUMN_DIM,
            None,
            Some(&base_scale),
        )?;
        let second = format!("{prefix}.multihead_attn2");
        let (output, key, value) = builder.block(
            &source,
            &Source::Tokens(&hidden),
            &second,
            (b * h, n, INDUCING as isize),
            COLUMN_DIM,
            None,
            None,
        )?;
        source = output;
        column.push((key, value));
    }
    let mut hidden = builder.add(
        &builder.row_stage(&source, (shape.ensembles, shape.features))?,
        &icl_labels,
    );
    // FP16 caches also run the N x N ICL self-attention in FP16. Doing the same
    // for the column and row attention added error without saving time.
    builder.set_attention(kind(icl_storage));
    let mut icl = Vec::with_capacity(ICL_BLOCKS);
    for (block, base_scale) in icl_base_scales.iter().enumerate() {
        let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
        let base_scale = builder.base_scale(base_scale, ICL_HEAD_DIM);
        let (output, key, value) = builder.block(
            &hidden,
            &Source::Tokens(&hidden),
            &prefix,
            (b, n, n),
            ICL_DIM,
            None,
            Some(&base_scale),
        )?;
        hidden = output;
        icl.push((key, value));
    }

    // Column caches in context layout: zero reserved streams, then the features.
    let reserved = builder.zeros(&column_cache_shape(b, RESERVED as isize));
    let full_column =
        |tensor: &Tensor| builder.concat(&reserved, &builder.reshape(tensor, &column_cache_shape(b, h)), 1);
    let column_shape = column_cache_shape(b, h + RESERVED as isize).to_vec();
    let icl_shape = icl_cache_shape(b, n).to_vec();
    let mut targets = Vec::with_capacity(2 * (COLUMN_BLOCKS + ICL_BLOCKS));
    for (key, value) in &column {
        targets.extend([
            (full_column(key), column_shape.clone(), FLOAT),
            (full_column(value), column_shape.clone(), FLOAT),
        ]);
    }
    let icl_kind = kind(icl_storage);
    for (key, value) in &icl {
        targets.extend([
            (builder.cast(key, icl_kind), icl_shape.clone(), icl_kind),
            (builder.cast(value, icl_kind), icl_shape.clone(), icl_kind),
        ]);
    }
    if targets.len() != outputs.len() {
        return Err(metal_error("context output regions do not match the graph"));
    }
    // One NaN-propagating max |value| over every output, as stored, so the
    // CPU checks a single float instead of scanning the whole context.
    let largest = targets
        .iter()
        // SAFETY: graph construction on `builder.graph`, as in `Builder`.
        .map(|(tensor, _, _)| unsafe {
            let largest = builder.graph.reductionMaximumPropagateNaNWithTensor_axis_name(
                &builder.reshape(&builder.graph.absoluteWithTensor_name(tensor, None), &[-1]),
                0,
                None,
            );
            builder.cast(&largest, FLOAT)
        })
        // SAFETY: as above.
        .reduce(|left, right| unsafe {
            builder
                .graph
                .maximumWithNaNPropagationWithPrimaryTensor_secondaryTensor_name(&left, &right, None)
        })
        .ok_or_else(|| metal_error("context has no outputs"))?;
    let mut outputs = targets
        .into_iter()
        .zip(outputs)
        .map(|((tensor, dimensions, kind), &(pointer, bytes))| {
            Ok((tensor, gpu.wrap(pointer, bytes)?, dimensions, kind))
        })
        .collect::<Result<Vec<Binding>>>()?;
    outputs.push((largest, gpu.empty(1)?, vec![1], FLOAT));
    let bindings = builder.bindings.into_inner();
    gpu.run(
        &builder.graph,
        &bindings.iter().collect::<Vec<_>>(),
        &outputs.iter().collect::<Vec<_>>(),
    );
    // SAFETY: `gpu.run` returned after the GPU finished, and the last output is a
    // live shared buffer of one float (`gpu.empty(1)`); any bit pattern is an `f32`.
    let largest = unsafe { *outputs[outputs.len() - 1].1.contents().as_ptr().cast::<f32>() };
    if !largest.is_finite() {
        return Err(metal_error("the built context contains a non-finite value"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_split_keeps_every_bucket_within_the_budget() {
        // 8 estimators of 100 features: 3.7 MB a row.
        let row_bytes = gpu_build_estimator_bytes(1, 100) * 8;
        for (budget, split) in [
            (u64::MAX, (QUERY_CHUNK_ROWS, QUERY_CHUNK_ROWS)),
            (10 << 30, (512, 128)),
            (4 << 30, (256, 64)),
            (2 << 30, (128, 32)),
            (0, (1, 1)),
        ] {
            assert_eq!(query_split(row_bytes, budget), split, "budget {budget}");
            let (chunk, small) = split;
            assert!(budget == 0 || (kept_rows(chunk, small) * row_bytes) as u64 <= budget);
        }
        // Every bucket up to 16 rows, 8 per doubling to 32, then 64.
        assert_eq!(
            kept_rows(64, 32),
            (1..=16).sum::<usize>() + (18..=32).step_by(2).sum::<usize>() + 64
        );
    }
}

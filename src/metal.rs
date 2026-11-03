//! Metal backend for cached prediction on Apple Silicon, built with `MPSGraph`.
//!
//! For large query batches each estimator runs on the GPU in two graphs: a
//! per-context front graph (column and row stages) and a per-model ICL graph
//! (twelve cached ICL blocks and the decoder). The front graph writes its rows
//! into a shared buffer that the ICL graph reads directly. Keys and values are
//! copied once into shared Metal buffers when a context is built.
//!
//! Each graph runs one estimator at a time: `MPSGraph` reshapes allow only one
//! unknown dimension, and the query count must stay unknown.

#![allow(unsafe_code)]
// Objective-C calls; each `unsafe` block states its invariant.
// Tensor dimensions are model constants or validated sizes far below `isize::MAX`.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use std::sync::Mutex;

use objc2::{AllocAnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{MPSGraph, MPSGraphTensor, MPSGraphTensorData};

use crate::{Error, Result};

const ICL_DIM: usize = 512;
const COLUMN_DIM: usize = 128;
const HEADS: usize = 8;
const ICL_HEAD_DIM: usize = ICL_DIM / HEADS;
const COLUMN_HEAD_DIM: usize = COLUMN_DIM / HEADS;
const INDUCING: usize = 128;
const RESERVED: usize = 4;
const ICL_BLOCKS: usize = 12;
const COLUMN_BLOCKS: usize = 3;
const DECODER_OUTPUTS: usize = 10;
const FLOAT: MPSDataType = MPSDataType::Float32;

type Tensor = Retained<MPSGraphTensor>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Weights<'w, 'a> = dyn Fn(&str) -> Result<&'a [f32]> + 'w;

fn shape(dimensions: &[isize]) -> Retained<MPSShape> {
    let numbers = dimensions
        .iter()
        .map(|&value| NSNumber::new_isize(value))
        .collect::<Vec<_>>();
    NSArray::from_retained_slice(&numbers)
}

fn metal_error(message: &str) -> Error {
    Error::InvalidArgument(format!("Metal backend: {message}"))
}

/// The shared device and command queue.
#[derive(Clone)]
pub(crate) struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Gpu {
    /// The system GPU, or `None` without one.
    pub(crate) fn new() -> Result<Option<Self>> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| metal_error("cannot create a command queue"))?;
        Ok(Some(Self { device, queue }))
    }

    fn buffer(&self, values: &[f32]) -> Result<Buffer> {
        let pointer = std::ptr::NonNull::new(values.as_ptr().cast_mut().cast())
            .ok_or_else(|| metal_error("empty buffer"))?;
        // SAFETY: Metal copies the bytes of a live slice into a new shared buffer.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                std::mem::size_of_val(values),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    fn empty(&self, elements: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(elements * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Run `graph` synchronously: feed `inputs`, write `outputs` into buffers.
    fn run(
        &self,
        graph: &MPSGraph,
        inputs: &[(&Tensor, &Buffer, &[isize])],
        outputs: &[(&Tensor, &Buffer, &[isize])],
    ) {
        let data = |buffer: &Buffer, dimensions: &[isize]| {
            // SAFETY: each buffer is live, shared, and at least as long as its shape.
            unsafe {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    buffer,
                    &shape(dimensions),
                    FLOAT,
                )
            }
        };
        let datas = |items: &[(&Tensor, &Buffer, &[isize])]| -> Vec<Retained<MPSGraphTensorData>> {
            items
                .iter()
                .map(|(_, buffer, dimensions)| data(buffer, dimensions))
                .collect()
        };
        let input_tensors: Vec<&MPSGraphTensor> = inputs.iter().map(|item| &**item.0).collect();
        let output_tensors: Vec<&MPSGraphTensor> = outputs.iter().map(|item| &**item.0).collect();
        let feeds = NSDictionary::from_retained_objects(&input_tensors, &datas(inputs));
        let results = NSDictionary::from_retained_objects(&output_tensors, &datas(outputs));
        // SAFETY: synchronous run; the buffers outlive it.
        unsafe {
            graph.runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
                &self.queue,
                &feeds,
                None,
                &results,
            );
        }
    }
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

/// `MPSGraph` construction helpers over the model's weights.
struct Builder<'w, 'a> {
    graph: Retained<MPSGraph>,
    weights: &'w Weights<'w, 'a>,
}

// SAFETY (all methods): plain `MPSGraph` construction. Shapes come from fixed
// TabICLv2 dimensions and weights checked by the model artifact.
impl<'w, 'a> Builder<'w, 'a> {
    fn new(weights: &'w Weights<'w, 'a>) -> Self {
        // SAFETY: see the impl comment.
        Self {
            graph: unsafe { MPSGraph::new() },
            weights,
        }
    }

    fn weight(&self, name: &str) -> Result<&'a [f32]> {
        (self.weights)(name)
    }

    fn placeholder(&self, dimensions: &[isize]) -> Tensor {
        unsafe {
            self.graph
                .placeholderWithShape_dataType_name(Some(&shape(dimensions)), FLOAT, None)
        }
    }

    fn constant(&self, values: &[f32], dimensions: &[isize]) -> Tensor {
        let bytes = NSData::with_bytes(bytemuck::cast_slice(values));
        unsafe {
            self.graph
                .constantWithData_shape_dataType(&bytes, &shape(dimensions), FLOAT)
        }
    }

    fn scalar(&self, value: f64) -> Tensor {
        unsafe { self.graph.constantWithScalar_dataType(value, FLOAT) }
    }

    fn add(&self, left: &Tensor, right: &Tensor) -> Tensor {
        unsafe {
            self.graph
                .additionWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    fn multiply(&self, left: &Tensor, right: &Tensor) -> Tensor {
        unsafe {
            self.graph
                .multiplicationWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    fn matmul(&self, left: &Tensor, right: &Tensor) -> Tensor {
        unsafe {
            self.graph
                .matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    fn reshape(&self, input: &Tensor, dimensions: &[isize]) -> Tensor {
        unsafe {
            self.graph
                .reshapeTensor_withShape_name(input, &shape(dimensions), None)
        }
    }

    fn transpose(&self, input: &Tensor, first: usize, second: usize) -> Tensor {
        unsafe {
            self.graph
                .transposeTensor_dimension_withDimension_name(input, first, second, None)
        }
    }

    fn slice(&self, input: &Tensor, dimension: usize, start: isize, length: isize) -> Tensor {
        unsafe {
            self.graph
                .sliceTensor_dimension_start_length_name(input, dimension, start, length, None)
        }
    }

    /// `input @ weight^T + bias` for a `[outputs, inputs]` weight.
    fn affine(&self, input: &Tensor, weight: &[f32], bias: &[f32], inputs: usize, outputs: usize) -> Tensor {
        let mut transposed = vec![0.0; inputs * outputs];
        for out in 0..outputs {
            for index in 0..inputs {
                transposed[index * outputs + out] = weight[out * inputs + index];
            }
        }
        let product = self.matmul(
            input,
            &self.constant(&transposed, &[inputs as isize, outputs as isize]),
        );
        self.add(&product, &self.constant(bias, &[outputs as isize]))
    }

    fn linear(&self, input: &Tensor, prefix: &str, inputs: usize, outputs: usize) -> Result<Tensor> {
        Ok(self.affine(
            input,
            self.weight(&format!("{prefix}.weight"))?,
            self.weight(&format!("{prefix}.bias"))?,
            inputs,
            outputs,
        ))
    }

    /// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    fn gelu(&self, input: &Tensor) -> Tensor {
        let scaled = self.multiply(input, &self.scalar(std::f64::consts::FRAC_1_SQRT_2));
        let one_plus = self.add(
            &unsafe { self.graph.erfWithTensor_name(&scaled, None) },
            &self.scalar(1.0),
        );
        self.multiply(&self.multiply(input, &self.scalar(0.5)), &one_plus)
    }

    fn layer_norm(&self, input: &Tensor, prefix: &str, width: usize) -> Result<Tensor> {
        let axes = shape(&[-1]);
        let gamma = self.constant(self.weight(&format!("{prefix}.weight"))?, &[width as isize]);
        let beta = self.constant(self.weight(&format!("{prefix}.bias"))?, &[width as isize]);
        unsafe {
            let mean = self.graph.meanOfTensor_axes_name(input, &axes, None);
            let variance = self
                .graph
                .varianceOfTensor_meanTensor_axes_name(input, &mean, &axes, None);
            Ok(self
                .graph
                .normalizationWithTensor_meanTensor_varianceTensor_gammaTensor_betaTensor_epsilon_name(
                    input,
                    &mean,
                    &variance,
                    Some(&gamma),
                    Some(&beta),
                    1e-5,
                    None,
                ))
        }
    }

    /// `[lead, rows, heads * head_dim]` -> `[lead, heads, rows, head_dim]`.
    fn split_heads(&self, input: &Tensor, lead: isize, rows: isize, head_dim: usize) -> Tensor {
        self.transpose(
            &self.reshape(input, &[lead, rows, HEADS as isize, head_dim as isize]),
            1,
            2,
        )
    }

    /// Non-interleaved `RoPE`: `x * cos + rotate_half(x) * sin`, with
    /// `rotate_half([left, right]) = [-right, left]`.
    fn rope(&self, input: &Tensor, (cos, sin): (&Tensor, &Tensor), head_dim: usize) -> Tensor {
        let half = head_dim / 2;
        let mut rotation = vec![0.0; head_dim * head_dim];
        for index in 0..half {
            rotation[(index + half) * head_dim + index] = -1.0;
            rotation[index * head_dim + index + half] = 1.0;
        }
        let rotated = self.matmul(
            input,
            &self.constant(&rotation, &[head_dim as isize, head_dim as isize]),
        );
        self.add(&self.multiply(input, cos), &self.multiply(&rotated, sin))
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
        attn: &str,
        (lead, q_rows, source_rows): (isize, isize, isize),
        dim: usize,
        rope: Option<&Rope<'_>>,
        qass: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let head_dim = dim / HEADS;
        let in_weight = self.weight(&format!("{attn}.in_proj_weight"))?;
        let in_bias = self.weight(&format!("{attn}.in_proj_bias"))?;
        let norm = format!("{prefix}.norm1");
        let query = self.affine(
            &self.layer_norm(input, &norm, dim)?,
            &in_weight[..dim * dim],
            &in_bias[..dim],
            dim,
            dim,
        );
        let mut query = self.split_heads(&query, lead, q_rows, head_dim);
        if let Some(base_scale) = qass {
            // QASSMax: q * base_scale * (1 + tanh(mlp(q))), the MLP shared across heads.
            let ssmax = format!("{attn}.ssmax_layer");
            let hidden = self.gelu(&self.linear(&query, &format!("{ssmax}.query_mlp.0"), head_dim, 64)?);
            let modulation = self.linear(&hidden, &format!("{ssmax}.query_mlp.2"), 64, head_dim)?;
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
                let project = |part: usize| {
                    let projected = self.affine(
                        &normalized,
                        &in_weight[part * dim * dim..][..dim * dim],
                        &in_bias[part * dim..][..dim],
                        dim,
                        dim,
                    );
                    self.split_heads(&projected, lead, source_rows, head_dim)
                };
                (project(1), project(2))
            }
        };
        let (query, key) = match rope {
            Some(rope) => (
                self.rope(&query, rope.query, head_dim),
                self.rope(&key, rope.key, head_dim),
            ),
            None => (query, key),
        };
        let attended = unsafe {
            self.graph
                .scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name(
                    &query,
                    &key,
                    &value,
                    (head_dim as f32).sqrt().recip(),
                    None,
                )
        };
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

    /// The row stage for `[H, Q, 128]` column embeddings: the row transformer
    /// over `[Q, 4 + H, 128]` sequences with the CLS tokens first, returning
    /// `[1, Q, 512]` rows.
    fn row_stage(&self, columns: &Tensor, features: usize) -> Result<Tensor> {
        let streams = (features + RESERVED) as isize;
        // Row stage, row-major: [Q, 4 + H, 128] with the CLS tokens first.
        let columns = self.transpose(columns, 0, 1);
        let zeros = self.multiply(&self.slice(&columns, 1, 0, 1), &self.scalar(0.0));
        let cls = self.add(
            &zeros,
            &self.constant(
                &self.weight("row_interactor.cls_tokens")?[..RESERVED * COLUMN_DIM],
                &[RESERVED as isize, COLUMN_DIM as isize],
            ),
        );
        let mut sequence = unsafe {
            self.graph
                .concatTensor_withTensor_dimension_name(&cls, &columns, 1, None)
        };
        let frequencies = self.weight("row_interactor.tf_row.rope.freqs")?;
        let table = |rows: usize| {
            let (mut cos, mut sin) = (
                vec![0.0; rows * COLUMN_HEAD_DIM],
                vec![0.0; rows * COLUMN_HEAD_DIM],
            );
            let half = COLUMN_HEAD_DIM / 2;
            for row in 0..rows {
                for (index, frequency) in frequencies[..half].iter().enumerate() {
                    let (s, c) = (row as f32 * frequency).sin_cos();
                    for offset in [index, index + half] {
                        cos[row * COLUMN_HEAD_DIM + offset] = c;
                        sin[row * COLUMN_HEAD_DIM + offset] = s;
                    }
                }
            }
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
                &format!("{prefix}.attn"),
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
            &format!("{prefix}.attn"),
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
            &[1, -1, ICL_DIM as isize],
        ))
    }
}

/// The twelve cached ICL blocks and the decoder, shared by every context of a model.
pub(crate) struct IclGraph {
    gpu: Gpu,
    graph: Retained<MPSGraph>,
    rows: Tensor,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    base_scales: Vec<Tensor>,
    logits: Tensor,
    /// `MPSGraph` runs are not documented as thread-safe; this serializes them.
    lock: Mutex<()>,
}

// SAFETY: the Objective-C objects are immutable after construction, and every
// graph run holds `lock`.
unsafe impl Send for IclGraph {}
// SAFETY: as above.
unsafe impl Sync for IclGraph {}

impl IclGraph {
    pub(crate) fn new(gpu: &Gpu, weights: &Weights<'_, '_>) -> Result<Self> {
        let builder = Builder::new(weights);
        let rows = builder.placeholder(&[1, -1, ICL_DIM as isize]);
        let kv_shape = [1, HEADS as isize, -1, ICL_HEAD_DIM as isize];
        let (mut keys, mut values, mut base_scales) = (Vec::new(), Vec::new(), Vec::new());
        let mut hidden = rows.clone();
        for block in 0..ICL_BLOCKS {
            let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
            let (key, value) = (builder.placeholder(&kv_shape), builder.placeholder(&kv_shape));
            let base_scale = builder.placeholder(&[1, HEADS as isize, 1, ICL_HEAD_DIM as isize]);
            (hidden, _, _) = builder.block(
                &hidden,
                &Source::Cached(&key, &value),
                &prefix,
                &format!("{prefix}.attn"),
                (1, -1, -1),
                ICL_DIM,
                None,
                Some(&base_scale),
            )?;
            keys.push(key);
            values.push(value);
            base_scales.push(base_scale);
        }
        let hidden = builder.gelu(&builder.linear(
            &builder.layer_norm(&hidden, "icl_predictor.ln", ICL_DIM)?,
            "icl_predictor.decoder.0",
            ICL_DIM,
            1024,
        )?);
        let logits = builder.linear(&hidden, "icl_predictor.decoder.2", 1024, DECODER_OUTPUTS)?;
        Ok(Self {
            gpu: gpu.clone(),
            graph: builder.graph,
            rows,
            keys,
            values,
            base_scales,
            logits,
            lock: Mutex::new(()),
        })
    }

    /// Copy one context's ICL keys and values, given per block as
    /// `[B, 8, N, 64]`, and its `[8, 64]` base scale per block to the GPU.
    pub(crate) fn context(
        &self,
        ensembles: usize,
        training_rows: usize,
        keys: &[&[f32]],
        values: &[&[f32]],
        base_scales: &[Vec<f32>],
    ) -> Result<IclContext> {
        let per_estimator = HEADS * training_rows * ICL_HEAD_DIM;
        let split = |blocks: &[&[f32]]| -> Result<Vec<Vec<Buffer>>> {
            (0..ensembles)
                .map(|ensemble| {
                    blocks
                        .iter()
                        .map(|values| {
                            self.gpu
                                .buffer(&values[ensemble * per_estimator..][..per_estimator])
                        })
                        .collect()
                })
                .collect()
        };
        Ok(IclContext {
            training_rows,
            keys: split(keys)?,
            values: split(values)?,
            base_scales: base_scales
                .iter()
                .map(|values| self.gpu.buffer(values))
                .collect::<Result<_>>()?,
        })
    }

    /// Run the ICL blocks and decoder for one estimator's `[Q, 512]` rows.
    /// Returns `[Q, 10]` logits.
    pub(crate) fn run(
        &self,
        context: &IclContext,
        ensemble: usize,
        rows: &Buffer,
        query_rows: usize,
    ) -> Result<Vec<f32>> {
        let (q, n) = (query_rows as isize, context.training_rows as isize);
        let output = self.gpu.empty(query_rows * DECODER_OUTPUTS)?;
        let kv = [1, HEADS as isize, n, ICL_HEAD_DIM as isize];
        let scale = [1, HEADS as isize, 1, ICL_HEAD_DIM as isize];
        let rows_shape = [1, q, ICL_DIM as isize];
        let mut inputs = vec![(&self.rows, rows, &rows_shape[..])];
        for block in 0..ICL_BLOCKS {
            inputs.push((&self.keys[block], &context.keys[ensemble][block], &kv[..]));
            inputs.push((&self.values[block], &context.values[ensemble][block], &kv[..]));
            inputs.push((&self.base_scales[block], &context.base_scales[block], &scale[..]));
        }
        let _guard = self.lock.lock().map_err(|_| metal_error("graph lock poisoned"))?;
        self.gpu.run(
            &self.graph,
            &inputs,
            &[(&self.logits, &output, &[1, q, DECODER_OUTPUTS as isize])],
        );
        // SAFETY: the run is complete and wrote `query_rows * 10` floats.
        Ok(unsafe {
            std::slice::from_raw_parts(
                output.contents().as_ptr().cast::<f32>(),
                query_rows * DECODER_OUTPUTS,
            )
        }
        .to_vec())
    }
}

/// One context's ICL keys and values per estimator and block, and `QASSMax`
/// base scales per block, in shared GPU memory.
pub(crate) struct IclContext {
    training_rows: usize,
    keys: Vec<Vec<Buffer>>,
    values: Vec<Vec<Buffer>>,
    base_scales: Vec<Buffer>,
}

// SAFETY: the buffers are written once at construction and only read after.
unsafe impl Send for IclContext {}
// SAFETY: as above.
unsafe impl Sync for IclContext {}

/// The column and row stages for one context's feature count, with the
/// context's column keys and values per estimator and block.
pub(crate) struct FrontGraph {
    gpu: Gpu,
    graph: Retained<MPSGraph>,
    features: usize,
    grouped: Tensor,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    rows: Tensor,
    column_keys: Vec<Vec<Buffer>>,
    column_values: Vec<Vec<Buffer>>,
    lock: Mutex<()>,
}

// SAFETY: as for `IclGraph`.
unsafe impl Send for FrontGraph {}
// SAFETY: as above.
unsafe impl Sync for FrontGraph {}

impl FrontGraph {
    /// Build the front graph for `features` columns. `column_keys` and
    /// `column_values` hold each column block's `[B, H + 4, 8, 128, 16]` cache.
    #[allow(clippy::too_many_lines)] // One linear graph definition reads best in one place.
    pub(crate) fn new(
        gpu: &Gpu,
        weights: &Weights<'_, '_>,
        ensembles: usize,
        features: usize,
        column_keys: &[&[f32]],
        column_values: &[&[f32]],
    ) -> Result<Self> {
        let builder = Builder::new(weights);
        let h = features as isize;
        // Column stage, feature-major: [H, Q, 3] grouped inputs -> [H, Q, 128].
        let grouped = builder.placeholder(&[h, -1, 3]);
        let mut columns = builder.linear(&grouped, "col_embedder.in_linear", 3, COLUMN_DIM)?;
        let kv_shape = [h, HEADS as isize, INDUCING as isize, COLUMN_HEAD_DIM as isize];
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for block in 0..COLUMN_BLOCKS {
            let prefix = format!("col_embedder.tf_col.blocks.{block}.multihead_attn2");
            let (key, value) = (builder.placeholder(&kv_shape), builder.placeholder(&kv_shape));
            (columns, _, _) = builder.block(
                &columns,
                &Source::Cached(&key, &value),
                &prefix,
                &format!("{prefix}.attn"),
                (h, -1, INDUCING as isize),
                COLUMN_DIM,
                None,
                None,
            )?;
            keys.push(key);
            values.push(value);
        }
        let rows = builder.row_stage(&columns, features)?;
        // Each estimator's feature streams, skipping the unused reserved streams.
        let stream_elements = HEADS * INDUCING * COLUMN_HEAD_DIM;
        let split = |blocks: &[&[f32]]| -> Result<Vec<Vec<Buffer>>> {
            (0..ensembles)
                .map(|ensemble| {
                    let start = (ensemble * (features + RESERVED) + RESERVED) * stream_elements;
                    blocks
                        .iter()
                        .map(|values| gpu.buffer(&values[start..][..features * stream_elements]))
                        .collect()
                })
                .collect()
        };
        Ok(Self {
            gpu: gpu.clone(),
            graph: builder.graph,
            features,
            grouped,
            keys,
            values,
            rows,
            column_keys: split(column_keys)?,
            column_values: split(column_values)?,
            lock: Mutex::new(()),
        })
    }

    /// Run the column and row stages for one estimator's `[H, Q, 3]` grouped
    /// inputs. Returns a shared buffer of `[Q, 512]` rows for `IclGraph::run`.
    pub(crate) fn run(&self, ensemble: usize, grouped: &[f32], query_rows: usize) -> Result<Buffer> {
        let (h, q) = (self.features as isize, query_rows as isize);
        let input = self.gpu.buffer(grouped)?;
        let output = self.gpu.empty(query_rows * ICL_DIM)?;
        let kv = [h, HEADS as isize, INDUCING as isize, COLUMN_HEAD_DIM as isize];
        let grouped_shape = [h, q, 3];
        let mut inputs = vec![(&self.grouped, &input, &grouped_shape[..])];
        for block in 0..COLUMN_BLOCKS {
            inputs.push((&self.keys[block], &self.column_keys[ensemble][block], &kv[..]));
            inputs.push((&self.values[block], &self.column_values[ensemble][block], &kv[..]));
        }
        let _guard = self.lock.lock().map_err(|_| metal_error("graph lock poisoned"))?;
        self.gpu.run(
            &self.graph,
            &inputs,
            &[(&self.rows, &output, &[1, q, ICL_DIM as isize])],
        );
        Ok(output)
    }
}

/// One estimator's static context built on the GPU: the training rows run
/// through every stage once, and each column and ICL block's keys and values
/// are captured. All shapes are fixed, so the graph is built per context.
pub(crate) struct ContextGraph {
    gpu: Gpu,
    graph: Retained<MPSGraph>,
    features: usize,
    training_rows: usize,
    grouped: Tensor,
    column_labels: Tensor,
    icl_labels: Tensor,
    column: Vec<(Tensor, Tensor)>,
    icl: Vec<(Tensor, Tensor)>,
}

/// Captured `(keys, values)` per block: column blocks `[H, 8, 128, 16]`, ICL
/// blocks `[8, N, 64]`.
pub(crate) type CapturedContext = (Vec<(Vec<f32>, Vec<f32>)>, Vec<(Vec<f32>, Vec<f32>)>);

impl ContextGraph {
    /// `column_base_scales` and `icl_base_scales` are the `QASSMax` base scales
    /// for `training_rows` sources: 3 of `[8, 16]` and 12 of `[8, 64]`.
    pub(crate) fn new(
        gpu: &Gpu,
        weights: &Weights<'_, '_>,
        features: usize,
        training_rows: usize,
        column_base_scales: &[Vec<f32>],
        icl_base_scales: &[Vec<f32>],
    ) -> Result<Self> {
        let builder = Builder::new(weights);
        let (h, n) = (features as isize, training_rows as isize);
        let grouped = builder.placeholder(&[h, n, 3]);
        let column_labels = builder.placeholder(&[n, COLUMN_DIM as isize]);
        let icl_labels = builder.placeholder(&[1, n, ICL_DIM as isize]);
        let mut source = builder.add(
            &builder.linear(&grouped, "col_embedder.in_linear", 3, COLUMN_DIM)?,
            &column_labels,
        );
        let zeros = builder.multiply(&builder.slice(&source, 1, 0, 1), &builder.scalar(0.0));
        let mut column = Vec::with_capacity(COLUMN_BLOCKS);
        for (block, base_scale) in column_base_scales.iter().enumerate() {
            // ISAB: the inducing points attend to the training rows, then the
            // rows attend back; the second attention's keys and values are cached.
            let prefix = format!("col_embedder.tf_col.blocks.{block}");
            let inducing = builder.add(
                &zeros,
                &builder.constant(
                    builder.weight(&format!("{prefix}.ind_vectors"))?,
                    &[INDUCING as isize, COLUMN_DIM as isize],
                ),
            );
            let base_scale = builder.constant(base_scale, &[1, HEADS as isize, 1, COLUMN_HEAD_DIM as isize]);
            let first = format!("{prefix}.multihead_attn1");
            let (hidden, _, _) = builder.block(
                &inducing,
                &Source::Tokens(&source),
                &first,
                &format!("{first}.attn"),
                (h, INDUCING as isize, n),
                COLUMN_DIM,
                None,
                Some(&base_scale),
            )?;
            let second = format!("{prefix}.multihead_attn2");
            let (output, key, value) = builder.block(
                &source,
                &Source::Tokens(&hidden),
                &second,
                &format!("{second}.attn"),
                (h, n, INDUCING as isize),
                COLUMN_DIM,
                None,
                None,
            )?;
            source = output;
            column.push((key, value));
        }
        let mut rows = builder.add(&builder.row_stage(&source, features)?, &icl_labels);
        let mut icl = Vec::with_capacity(ICL_BLOCKS);
        for (block, base_scale) in icl_base_scales.iter().enumerate() {
            let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
            let base_scale = builder.constant(base_scale, &[1, HEADS as isize, 1, ICL_HEAD_DIM as isize]);
            let (output, key, value) = builder.block(
                &rows,
                &Source::Tokens(&rows),
                &prefix,
                &format!("{prefix}.attn"),
                (1, n, n),
                ICL_DIM,
                None,
                Some(&base_scale),
            )?;
            rows = output;
            icl.push((key, value));
        }
        Ok(Self {
            gpu: gpu.clone(),
            graph: builder.graph,
            features,
            training_rows,
            grouped,
            column_labels,
            icl_labels,
            column,
            icl,
        })
    }

    /// Build one estimator's context from `[H, N, 3]` grouped inputs and its
    /// `[N, 128]` column and `[N, 512]` ICL label embeddings.
    pub(crate) fn run(
        &self,
        grouped: &[f32],
        column_labels: &[f32],
        icl_labels: &[f32],
    ) -> Result<CapturedContext> {
        let (h, n) = (self.features as isize, self.training_rows as isize);
        let inputs = [grouped, column_labels, icl_labels]
            .iter()
            .map(|values| self.gpu.buffer(values))
            .collect::<Result<Vec<_>>>()?;
        let column_shape = [h, HEADS as isize, INDUCING as isize, COLUMN_HEAD_DIM as isize];
        let icl_shape = [1, HEADS as isize, n, ICL_HEAD_DIM as isize];
        let column_elements = self.features * HEADS * INDUCING * COLUMN_HEAD_DIM;
        let icl_elements = HEADS * self.training_rows * ICL_HEAD_DIM;
        let column_buffers = (0..2 * COLUMN_BLOCKS)
            .map(|_| self.gpu.empty(column_elements))
            .collect::<Result<Vec<_>>>()?;
        let icl_buffers = (0..2 * ICL_BLOCKS)
            .map(|_| self.gpu.empty(icl_elements))
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = Vec::with_capacity(2 * (COLUMN_BLOCKS + ICL_BLOCKS));
        for (index, (key, value)) in self.column.iter().enumerate() {
            outputs.push((key, &column_buffers[2 * index], &column_shape[..]));
            outputs.push((value, &column_buffers[2 * index + 1], &column_shape[..]));
        }
        for (index, (key, value)) in self.icl.iter().enumerate() {
            outputs.push((key, &icl_buffers[2 * index], &icl_shape[..]));
            outputs.push((value, &icl_buffers[2 * index + 1], &icl_shape[..]));
        }
        self.gpu.run(
            &self.graph,
            &[
                (&self.grouped, &inputs[0], &[h, n, 3]),
                (&self.column_labels, &inputs[1], &[n, COLUMN_DIM as isize]),
                (&self.icl_labels, &inputs[2], &[1, n, ICL_DIM as isize]),
            ],
            &outputs,
        );
        let read = |buffer: &Buffer, elements: usize| -> Vec<f32> {
            // SAFETY: the run is complete and wrote `elements` floats.
            unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), elements) }.to_vec()
        };
        let pairs = |buffers: &[Buffer], elements: usize| {
            buffers
                .chunks_exact(2)
                .map(|pair| (read(&pair[0], elements), read(&pair[1], elements)))
                .collect()
        };
        Ok((
            pairs(&column_buffers, column_elements),
            pairs(&icl_buffers, icl_elements),
        ))
    }
}

// SAFETY: as for `IclGraph`; a `ContextGraph` is used by one thread at a time.
unsafe impl Send for ContextGraph {}

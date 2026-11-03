//! Metal backend for the ICL stage on Apple Silicon, built with `MPSGraph`.
//!
//! The CPU still runs the column and row stages. For large query batches the
//! twelve cached ICL blocks and the decoder run as one `MPSGraph` on the GPU,
//! once per estimator, against keys and values copied once into shared Metal
//! buffers. The graph has batch size 1 because `MPSGraph` reshapes allow only
//! one unknown dimension, and splitting heads needs the query count unknown.

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
const HEADS: usize = 8;
const HEAD_DIM: usize = ICL_DIM / HEADS;
const BLOCKS: usize = 12;
const DECODER_OUTPUTS: usize = 10;

type Tensor = Retained<MPSGraphTensor>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Model weights as one compiled-on-demand graph, shared by every context.
pub(crate) struct IclGraph {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
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

/// One context's keys and values per estimator and block, and `QASSMax` base
/// scales per block, in shared GPU memory.
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

impl IclGraph {
    /// Build the ICL graph from model weights, or return `None` without a GPU.
    #[allow(clippy::too_many_lines)] // One linear graph definition reads best in one place.
    pub(crate) fn new<'a>(tensor: impl Fn(&str) -> Result<&'a [f32]>) -> Result<Option<Self>> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| metal_error("cannot create a command queue"))?;
        // SAFETY: plain MPSGraph construction; all shapes and data lengths below
        // are derived from the fixed TabICLv2 dimensions and checked weights.
        unsafe {
            let graph = MPSGraph::new();
            let float = MPSDataType::Float32;
            let constant = |values: &[f32], dimensions: &[isize]| -> Tensor {
                let bytes = NSData::with_bytes(bytemuck::cast_slice(values));
                graph.constantWithData_shape_dataType(&bytes, &shape(dimensions), float)
            };
            // Weights are stored `[out, in]`; the graph multiplies by `[in, out]`.
            let linear = |input: &Tensor, prefix: &str, inputs: usize, outputs: usize| -> Result<Tensor> {
                let weight = tensor(&format!("{prefix}.weight"))?;
                let mut transposed = vec![0.0; inputs * outputs];
                for out in 0..outputs {
                    for index in 0..inputs {
                        transposed[index * outputs + out] = weight[out * inputs + index];
                    }
                }
                let weight = constant(&transposed, &[inputs as isize, outputs as isize]);
                let bias = constant(tensor(&format!("{prefix}.bias"))?, &[outputs as isize]);
                let product =
                    graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(input, &weight, None);
                Ok(graph.additionWithPrimaryTensor_secondaryTensor_name(&product, &bias, None))
            };
            let gelu = |input: &Tensor| -> Tensor {
                // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
                let scaled = graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    input,
                    &graph.constantWithScalar_dataType(std::f64::consts::FRAC_1_SQRT_2, float),
                    None,
                );
                let one_plus = graph.additionWithPrimaryTensor_secondaryTensor_name(
                    &graph.erfWithTensor_name(&scaled, None),
                    &graph.constantWithScalar_dataType(1.0, float),
                    None,
                );
                let half = graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    input,
                    &graph.constantWithScalar_dataType(0.5, float),
                    None,
                );
                graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&half, &one_plus, None)
            };
            let layer_norm = |input: &Tensor, prefix: &str| -> Result<Tensor> {
                let axes = shape(&[-1]);
                let mean = graph.meanOfTensor_axes_name(input, &axes, None);
                let variance = graph.varianceOfTensor_meanTensor_axes_name(input, &mean, &axes, None);
                let gamma = constant(tensor(&format!("{prefix}.weight"))?, &[ICL_DIM as isize]);
                let beta = constant(tensor(&format!("{prefix}.bias"))?, &[ICL_DIM as isize]);
                Ok(graph
                    .normalizationWithTensor_meanTensor_varianceTensor_gammaTensor_betaTensor_epsilon_name(
                        input,
                        &mean,
                        &variance,
                        Some(&gamma),
                        Some(&beta),
                        1e-5,
                        None,
                    ))
            };

            let rows = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[1, -1, ICL_DIM as isize])),
                float,
                None,
            );
            let mut keys = Vec::with_capacity(BLOCKS);
            let mut values = Vec::with_capacity(BLOCKS);
            let mut base_scales = Vec::with_capacity(BLOCKS);
            let mut hidden = rows.clone();
            for block in 0..BLOCKS {
                let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
                let attn = format!("{prefix}.attn");
                let key = graph.placeholderWithShape_dataType_name(
                    Some(&shape(&[1, HEADS as isize, -1, HEAD_DIM as isize])),
                    float,
                    None,
                );
                let value = graph.placeholderWithShape_dataType_name(
                    Some(&shape(&[1, HEADS as isize, -1, HEAD_DIM as isize])),
                    float,
                    None,
                );
                let base_scale = graph.placeholderWithShape_dataType_name(
                    Some(&shape(&[1, HEADS as isize, 1, HEAD_DIM as isize])),
                    float,
                    None,
                );

                let normalized = layer_norm(&hidden, &format!("{prefix}.norm1"))?;
                let in_weight = tensor(&format!("{attn}.in_proj_weight"))?;
                let in_bias = tensor(&format!("{attn}.in_proj_bias"))?;
                let mut q_weight = vec![0.0; ICL_DIM * ICL_DIM];
                for out in 0..ICL_DIM {
                    for index in 0..ICL_DIM {
                        q_weight[index * ICL_DIM + out] = in_weight[out * ICL_DIM + index];
                    }
                }
                let query = graph.additionWithPrimaryTensor_secondaryTensor_name(
                    &graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(
                        &normalized,
                        &constant(&q_weight, &[ICL_DIM as isize, ICL_DIM as isize]),
                        None,
                    ),
                    &constant(&in_bias[..ICL_DIM], &[ICL_DIM as isize]),
                    None,
                );
                // [B, Q, 512] -> [B, 8, Q, 64]
                let query = graph.transposeTensor_dimension_withDimension_name(
                    &graph.reshapeTensor_withShape_name(
                        &query,
                        &shape(&[1, -1, HEADS as isize, HEAD_DIM as isize]),
                        None,
                    ),
                    1,
                    2,
                    None,
                );
                // QASSMax: q * base_scale * (1 + tanh(mlp(q))), the MLP shared across heads.
                let ssmax = format!("{attn}.ssmax_layer");
                let modulation = linear(
                    &gelu(&linear(&query, &format!("{ssmax}.query_mlp.0"), HEAD_DIM, 64)?),
                    &format!("{ssmax}.query_mlp.2"),
                    64,
                    HEAD_DIM,
                )?;
                let gate = graph.additionWithPrimaryTensor_secondaryTensor_name(
                    &graph.tanhWithTensor_name(&modulation, None),
                    &graph.constantWithScalar_dataType(1.0, float),
                    None,
                );
                let query = graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&query, &base_scale, None),
                    &gate,
                    None,
                );
                let attended = graph
                    .scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name(
                        &query,
                        &key,
                        &value,
                        (HEAD_DIM as f32).sqrt().recip(),
                        None,
                    );
                // [B, 8, Q, 64] -> [B, Q, 512]
                let merged = graph.reshapeTensor_withShape_name(
                    &graph.transposeTensor_dimension_withDimension_name(&attended, 1, 2, None),
                    &shape(&[1, -1, ICL_DIM as isize]),
                    None,
                );
                let projected = linear(&merged, &format!("{attn}.out_proj"), ICL_DIM, ICL_DIM)?;
                let residual =
                    graph.additionWithPrimaryTensor_secondaryTensor_name(&hidden, &projected, None);
                let feedforward = linear(
                    &gelu(&linear(
                        &layer_norm(&residual, &format!("{prefix}.norm2"))?,
                        &format!("{prefix}.linear1"),
                        ICL_DIM,
                        2 * ICL_DIM,
                    )?),
                    &format!("{prefix}.linear2"),
                    2 * ICL_DIM,
                    ICL_DIM,
                )?;
                hidden = graph.additionWithPrimaryTensor_secondaryTensor_name(&residual, &feedforward, None);
                keys.push(key);
                values.push(value);
                base_scales.push(base_scale);
            }
            let decoded = linear(
                &gelu(&linear(
                    &layer_norm(&hidden, "icl_predictor.ln")?,
                    "icl_predictor.decoder.0",
                    ICL_DIM,
                    1024,
                )?),
                "icl_predictor.decoder.2",
                1024,
                DECODER_OUTPUTS,
            )?;
            Ok(Some(Self {
                device,
                queue,
                graph,
                rows,
                keys,
                values,
                base_scales,
                logits: decoded,
                lock: Mutex::new(()),
            }))
        }
    }

    fn buffer(&self, values: &[f32]) -> Result<Buffer> {
        let bytes = std::mem::size_of_val(values);
        let pointer = std::ptr::NonNull::new(values.as_ptr().cast_mut().cast())
            .ok_or_else(|| metal_error("empty buffer"))?;
        // SAFETY: Metal copies `bytes` bytes from a live slice into a new shared buffer.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Copy one context's keys and values, given per block as `[B, 8, N, 64]`,
    /// and its `[8, 64]` base scale per block into shared GPU buffers.
    pub(crate) fn context(
        &self,
        ensembles: usize,
        training_rows: usize,
        keys: &[&[f32]],
        values: &[&[f32]],
        base_scales: &[Vec<f32>],
    ) -> Result<IclContext> {
        let per_estimator = HEADS * training_rows * HEAD_DIM;
        let split = |blocks: &[&[f32]]| -> Result<Vec<Vec<Buffer>>> {
            (0..ensembles)
                .map(|ensemble| {
                    blocks
                        .iter()
                        .map(|values| self.buffer(&values[ensemble * per_estimator..][..per_estimator]))
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
                .map(|values| self.buffer(values))
                .collect::<Result<_>>()?,
        })
    }

    /// Run the ICL blocks and decoder for one estimator's `[Q, 512]` rows.
    /// Returns `[Q, 10]` logits.
    pub(crate) fn run(
        &self,
        context: &IclContext,
        ensemble: usize,
        rows: &[f32],
        query_rows: usize,
    ) -> Result<Vec<f32>> {
        let (q, n) = (query_rows as isize, context.training_rows as isize);
        let float = MPSDataType::Float32;
        let output = self.buffer(&vec![0.0_f32; query_rows * DECODER_OUTPUTS])?;
        let input = self.buffer(rows)?;
        let _guard = self.lock.lock().map_err(|_| metal_error("graph lock poisoned"))?;
        // SAFETY: every tensor data wraps a live shared buffer whose length
        // matches its shape; the run is synchronous, so the buffers outlive it.
        unsafe {
            let data = |buffer: &Buffer, dimensions: &[isize]| {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    buffer,
                    &shape(dimensions),
                    float,
                )
            };
            let mut tensors: Vec<&MPSGraphTensor> = vec![&self.rows];
            let mut feeds = vec![data(&input, &[1, q, ICL_DIM as isize])];
            for block in 0..BLOCKS {
                let kv = [1, HEADS as isize, n, HEAD_DIM as isize];
                tensors.extend([
                    &*self.keys[block],
                    &*self.values[block],
                    &*self.base_scales[block],
                ]);
                feeds.push(data(&context.keys[ensemble][block], &kv));
                feeds.push(data(&context.values[ensemble][block], &kv));
                feeds.push(data(
                    &context.base_scales[block],
                    &[1, HEADS as isize, 1, HEAD_DIM as isize],
                ));
            }
            let feeds = NSDictionary::from_retained_objects(&tensors, &feeds);
            let results = NSDictionary::from_retained_objects(
                &[&*self.logits],
                &[data(&output, &[1, q, DECODER_OUTPUTS as isize])],
            );
            self.graph
                .runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
                    &self.queue,
                    &feeds,
                    None,
                    &results,
                );
            let length = query_rows * DECODER_OUTPUTS;
            Ok(std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), length).to_vec())
        }
    }
}

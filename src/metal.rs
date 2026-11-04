//! Metal backend for static contexts on Apple Silicon, built with `MPSGraph`.
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

use objc2::{AllocAnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphCompilationDescriptor, MPSGraphDevice, MPSGraphExecutable,
    MPSGraphExecutableExecutionDescriptor, MPSGraphExecutionDescriptor, MPSGraphOptimization,
    MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
};

use crate::{Error, Result, static_context::TensorStorage};

/// The `MPSGraph` element type for a stored tensor type.
fn kind(storage: TensorStorage) -> MPSDataType {
    match storage {
        TensorStorage::Fp32 => FLOAT,
        TensorStorage::F16 => MPSDataType::Float16,
        TensorStorage::Bf16 => MPSDataType::BFloat16,
    }
}

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
/// A graph input or output: its tensor, buffer, shape, and data type.
type Binding = (Tensor, Buffer, Vec<isize>, MPSDataType);

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

/// The shared device and command queue, and how graphs run on them.
#[derive(Clone)]
pub(crate) struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    compilation: Retained<MPSGraphCompilationDescriptor>,
    execution: Retained<MPSGraphExecutionDescriptor>,
    executable_execution: Retained<MPSGraphExecutableExecutionDescriptor>,
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
        // Optimization level 0 measured as fast as the default level 1 on an
        // M1 Max. Level 1's pass that places operations on the Neural Engine
        // or CPU printed "Failed to import MPS module" on every compile.
        // SAFETY: plain descriptor construction.
        let (compilation, execution, executable_execution) = unsafe {
            let compilation = MPSGraphCompilationDescriptor::new();
            compilation.setOptimizationLevel(MPSGraphOptimization::Level0);
            let execution = MPSGraphExecutionDescriptor::new();
            execution.setWaitUntilCompleted(true);
            execution.setCompilationDescriptor(Some(&compilation));
            let executable_execution = MPSGraphExecutableExecutionDescriptor::new();
            executable_execution.setWaitUntilCompleted(true);
            (compilation, execution, executable_execution)
        };
        Ok(Some(Self {
            device,
            queue,
            compilation,
            execution,
            executable_execution,
        }))
    }

    fn buffer(&self, values: &[f32]) -> Result<Buffer> {
        self.buffer_bytes(bytemuck::cast_slice(values))
    }

    fn buffer_bytes(&self, bytes: &[u8]) -> Result<Buffer> {
        let pointer = std::ptr::NonNull::new(bytes.as_ptr().cast_mut().cast())
            .ok_or_else(|| metal_error("empty buffer"))?;
        // SAFETY: Metal copies the bytes of a live slice into a new shared buffer.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Wrap `bytes` of existing page-aligned memory without copying it. The
    /// memory must outlive the buffer.
    fn wrap(&self, pointer: std::ptr::NonNull<u8>, bytes: usize) -> Result<Buffer> {
        // SAFETY: callers pass a page-aligned, page-padded region that outlives
        // the buffer; no deallocator, because the region's owner frees it.
        unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                pointer.cast(),
                bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or_else(|| metal_error("cannot wrap page-aligned memory"))
    }

    /// A buffer over a tensor's `bytes`: wrapped in place when `page_aligned`
    /// (the bytes start on a page inside page-padded memory), otherwise copied.
    fn tensor_buffer(&self, bytes: &[u8], page_aligned: bool) -> Result<Buffer> {
        if !page_aligned {
            return self.buffer_bytes(bytes);
        }
        let pointer = std::ptr::NonNull::from(bytes).cast::<u8>();
        self.wrap(
            pointer,
            bytes
                .len()
                .next_multiple_of(crate::static_context::TENSOR_PAGE_BYTES),
        )
    }

    /// The memory Metal recommends the device use without paging.
    pub(crate) fn working_set_bytes(&self) -> usize {
        usize::try_from(self.device.recommendedMaxWorkingSetSize()).unwrap_or(usize::MAX)
    }

    fn empty(&self, elements: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(elements * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Run `graph` synchronously, feeding `inputs` and writing `outputs`.
    fn run(&self, graph: &MPSGraph, inputs: &[&Binding], outputs: &[&Binding]) {
        self.run_prepared(graph, &dictionary(inputs), &dictionary(outputs));
    }

    fn run_prepared(&self, graph: &MPSGraph, feeds: &Dictionary, results: &Dictionary) {
        // SAFETY: the descriptor waits until the run completes, and the
        // dictionaries keep their buffers alive.
        unsafe {
            graph.runAsyncWithMTLCommandQueue_feeds_targetOperations_resultsDictionary_executionDescriptor(
                &self.queue,
                feeds,
                None,
                results,
                Some(&self.execution),
            );
        }
    }
}

type Dictionary = Retained<NSDictionary<MPSGraphTensor, MPSGraphTensorData>>;

/// Map each binding's tensor to its buffer viewed with the binding's shape.
fn dictionary(items: &[&Binding]) -> Dictionary {
    let tensors: Vec<&MPSGraphTensor> = items.iter().map(|item| &*item.0).collect();
    let data: Vec<_> = items
        .iter()
        .map(|(_, buffer, dimensions, kind)| {
            // SAFETY: each buffer is live, shared, and at least as long as its shape.
            unsafe {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    buffer,
                    &shape(dimensions),
                    *kind,
                )
            }
        })
        .collect();
    NSDictionary::from_retained_objects(&tensors, &data)
}

/// A model's weights in shared GPU buffers, converted once and reused by every
/// context's graphs.
pub(crate) struct ModelWeights {
    gpu: Gpu,
    buffers: Mutex<HashMap<String, Buffer>>,
}

// SAFETY: buffers are written once when created and only read after; the map
// is behind a mutex.
unsafe impl Send for ModelWeights {}
// SAFETY: as above.
unsafe impl Sync for ModelWeights {}

impl ModelWeights {
    pub(crate) fn new(gpu: Gpu) -> Self {
        Self {
            gpu,
            buffers: Mutex::default(),
        }
    }

    pub(crate) fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn get(&self, key: &str, values: impl FnOnce() -> Vec<f32>) -> Result<Buffer> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| metal_error("weight cache lock poisoned"))?;
        if let Some(buffer) = buffers.get(key) {
            return Ok(buffer.clone());
        }
        let buffer = self.gpu.buffer(&values())?;
        buffers.insert(key.to_owned(), buffer.clone());
        Ok(buffer)
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

/// `MPSGraph` construction helpers. Weights become placeholders bound to the
/// model's shared buffers; the bindings are kept for every run.
struct Builder<'w, 'a> {
    graph: Retained<MPSGraph>,
    weights: &'w Weights<'w, 'a>,
    model: &'w ModelWeights,
    bindings: std::cell::RefCell<Vec<Binding>>,
    /// The element type attention runs in when its keys and values are FP32.
    attention: std::cell::Cell<MPSDataType>,
}

// SAFETY (all methods): plain `MPSGraph` construction. Shapes come from fixed
// TabICLv2 dimensions and weights checked by the model artifact.
impl<'w, 'a> Builder<'w, 'a> {
    fn new(model: &'w ModelWeights, weights: &'w Weights<'w, 'a>) -> Self {
        // SAFETY: see the impl comment.
        Self {
            graph: unsafe { MPSGraph::new() },
            weights,
            model,
            bindings: std::cell::RefCell::default(),
            attention: std::cell::Cell::new(FLOAT),
        }
    }

    fn set_attention(&self, kind: MPSDataType) {
        self.attention.set(kind);
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

    /// A placeholder bound to `buffer` for every run.
    fn bound(&self, buffer: Buffer, dimensions: &[isize]) -> Tensor {
        self.bound_typed(buffer, dimensions, FLOAT)
    }

    /// A placeholder of element type `kind` bound to `buffer`, in that type.
    fn bound_typed(&self, buffer: Buffer, dimensions: &[isize], kind: MPSDataType) -> Tensor {
        let tensor = unsafe {
            self.graph
                .placeholderWithShape_dataType_name(Some(&shape(dimensions)), kind, None)
        };
        self.bindings
            .borrow_mut()
            .push((tensor.clone(), buffer, dimensions.to_vec(), kind));
        tensor
    }

    /// `input` converted to element type `kind`; a no-op for the same type.
    fn cast(&self, input: &Tensor, kind: MPSDataType) -> Tensor {
        if unsafe { input.dataType() } == kind {
            return input.clone();
        }
        unsafe { self.graph.castTensor_toType_name(input, kind, None) }
    }

    /// A model weight under `key`, converted by `values` on first use.
    fn model_weight(
        &self,
        key: &str,
        values: impl FnOnce() -> Vec<f32>,
        dimensions: &[isize],
    ) -> Result<Tensor> {
        Ok(self.bound(self.model.get(key, values)?, dimensions))
    }

    fn named_weight(&self, name: &str, dimensions: &[isize]) -> Result<Tensor> {
        let values = self.weight(name)?;
        self.model_weight(name, || values.to_vec(), dimensions)
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

    /// `input @ weight^T + bias` for a `[outputs, inputs]` weight stored under
    /// `key`, or for block `part` of a weight that stacks several of them.
    fn affine(
        &self,
        input: &Tensor,
        key: &str,
        part: usize,
        (inputs, outputs): (usize, usize),
    ) -> Result<Tensor> {
        let (weight, bias) = (
            self.weight(&format!("{key}.weight"))?,
            self.weight(&format!("{key}.bias"))?,
        );
        let weight = &weight[part * inputs * outputs..][..inputs * outputs];
        let transposed = self.model_weight(
            &format!("{key}.weight[{part}]^T"),
            || {
                let mut transposed = vec![0.0; inputs * outputs];
                for out in 0..outputs {
                    for index in 0..inputs {
                        transposed[index * outputs + out] = weight[out * inputs + index];
                    }
                }
                transposed
            },
            &[inputs as isize, outputs as isize],
        )?;
        let bias = &bias[part * outputs..][..outputs];
        let bias = self.model_weight(
            &format!("{key}.bias[{part}]"),
            || bias.to_vec(),
            &[outputs as isize],
        )?;
        Ok(self.add(&self.matmul(input, &transposed), &bias))
    }

    fn linear(&self, input: &Tensor, prefix: &str, inputs: usize, outputs: usize) -> Result<Tensor> {
        self.affine(input, prefix, 0, (inputs, outputs))
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
        let gamma = self.named_weight(&format!("{prefix}.weight"), &[width as isize])?;
        let beta = self.named_weight(&format!("{prefix}.bias"), &[width as isize])?;
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
        // `sin` carries the sign: [-sin, sin], so rotate_half is a half swap.
        let half = (head_dim / 2) as isize;
        let swapped = unsafe {
            self.graph.concatTensor_withTensor_dimension_name(
                &self.slice(input, 3, half, half),
                &self.slice(input, 3, 0, half),
                3,
                None,
            )
        };
        self.add(&self.multiply(input, cos), &self.multiply(&swapped, sin))
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
        let norm = format!("{prefix}.norm1");
        let in_proj = format!("{attn}.in_proj");
        let in_proj_key = |part: usize, x: &Tensor| -> Result<Tensor> {
            // The fused projection is stored as `{attn}.in_proj_weight/_bias`.
            let weight = self.weight(&format!("{attn}.in_proj_weight"))?;
            let bias = self.weight(&format!("{attn}.in_proj_bias"))?;
            let rows = &weight[part * dim * dim..][..dim * dim];
            let transposed = self.model_weight(
                &format!("{in_proj}[{part}]^T"),
                || {
                    let mut transposed = vec![0.0; dim * dim];
                    for out in 0..dim {
                        for index in 0..dim {
                            transposed[index * dim + out] = rows[out * dim + index];
                        }
                    }
                    transposed
                },
                &[dim as isize, dim as isize],
            )?;
            let bias = &bias[part * dim..][..dim];
            let bias = self.model_weight(
                &format!("{in_proj}[{part}].bias"),
                || bias.to_vec(),
                &[dim as isize],
            )?;
            Ok(self.add(&self.matmul(x, &transposed), &bias))
        };
        let query = in_proj_key(0, &self.layer_norm(input, &norm, dim)?)?;
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
                (
                    self.split_heads(&in_proj_key(1, &normalized)?, lead, source_rows, head_dim),
                    self.split_heads(&in_proj_key(2, &normalized)?, lead, source_rows, head_dim),
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
        let stored = unsafe { key.dataType() };
        let kind = if stored == FLOAT {
            self.attention.get()
        } else {
            stored
        };
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
                    }
                    sin[row * COLUMN_HEAD_DIM + index] = -s;
                    sin[row * COLUMN_HEAD_DIM + index + half] = s;
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

/// One context's prediction graph: every stage for every estimator in one
/// run. The context's keys and values are bound to it.
pub(crate) struct PredictGraph {
    gpu: Gpu,
    graph: Retained<MPSGraph>,
    shape: Shape,
    bindings: Vec<Binding>,
    grouped: Tensor,
    logits: Tensor,
    /// Buffers and dictionaries per cached query bucket. The lock also
    /// serializes graph runs.
    prepared: Mutex<HashMap<usize, Prepared>>,
}

/// Query buckets up to this many rows keep their buffers between calls.
const CACHED_BUCKET_ROWS: usize = 1024;

/// Round a query count up to a bucket that keeps its top four significant
/// bits, so the graph compiles for 8 shapes per doubling of the batch size and
/// pads at most 12.5%. Power-of-two buckets padded up to 2x, which cost more
/// than the compiles they saved on large batches.
fn bucket(rows: usize) -> usize {
    let shift = (usize::BITS - rows.leading_zeros()).saturating_sub(4);
    rows.next_multiple_of(1 << shift)
}

/// One query bucket's input and output buffers and how to run the graph on
/// them.
struct Prepared {
    input: Buffer,
    output: Buffer,
    run: PreparedRun,
}

enum PreparedRun {
    /// A graph run that `MPSGraph` matches to a compiled executable per call.
    Graph { feeds: Dictionary, results: Dictionary },
    /// An executable compiled once, with its inputs in feed order. Skipping
    /// the per-call lookup saved 0.5-1.4 ms per single-row predict.
    Executable {
        executable: Retained<MPSGraphExecutable>,
        inputs: Retained<NSArray<MPSGraphTensorData>>,
        results: Retained<NSArray<MPSGraphTensorData>>,
    },
}

// SAFETY: the Objective-C objects are immutable after construction, and every
// graph run holds the `prepared` lock.
unsafe impl Send for PredictGraph {}
// SAFETY: as above.
unsafe impl Sync for PredictGraph {}

impl PredictGraph {
    /// `column_keys` and `column_values` hold each column block's
    /// `[B, H + 4, 8, 128, 16]` cache, `icl_keys` and `icl_values` each ICL
    /// block's `[B, 8, N, 64]` in `icl`'s storage type, and `base_scales` each
    /// ICL block's `[8, 64]`. With `page_aligned`, the caches are wrapped in
    /// place instead of copied.
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
        let column_shape = [
            b,
            h + RESERVED as isize,
            HEADS as isize,
            INDUCING as isize,
            COLUMN_HEAD_DIM as isize,
        ];
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
                &format!("{prefix}.attn"),
                (b * h, -1, INDUCING as isize),
                COLUMN_DIM,
                None,
                None,
            )?;
        }
        let mut hidden = builder.row_stage(&columns, (shape.ensembles, shape.features))?;
        let icl_shape = [b, HEADS as isize, n, ICL_HEAD_DIM as isize];
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
            let base_scale = builder.constant(
                &base_scales[block],
                &[1, HEADS as isize, 1, ICL_HEAD_DIM as isize],
            );
            (hidden, _, _) = builder.block(
                &hidden,
                &Source::Cached(&key, &value),
                &prefix,
                &format!("{prefix}.attn"),
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
        let logits = builder.linear(&hidden, "icl_predictor.decoder.2", 1024, DECODER_OUTPUTS)?;
        Ok(Self {
            gpu: gpu.clone(),
            graph: builder.graph,
            shape,
            bindings: builder.bindings.into_inner(),
            grouped,
            logits,
            prepared: Mutex::default(),
        })
    }

    /// Allocate a bucket's buffers. A `cached` bucket also compiles its
    /// executable once; an uncached one would compile on every call.
    fn prepare(&self, rows: usize, cached: bool) -> Result<Prepared> {
        let (b, h) = (self.shape.ensembles, self.shape.features);
        let input = self.gpu.empty(b * h * rows * 3)?;
        // Padding rows must be finite; they never mix with real rows.
        // SAFETY: the new buffer holds `b * h * rows * 3` floats.
        unsafe { std::ptr::write_bytes(input.contents().as_ptr().cast::<f32>(), 0, b * h * rows * 3) };
        let output = self.gpu.empty(b * rows * DECODER_OUTPUTS)?;
        let input = (
            self.grouped.clone(),
            input,
            vec![(b * h) as isize, rows as isize, 3],
            FLOAT,
        );
        let output = (
            self.logits.clone(),
            output,
            vec![b as isize, rows as isize, DECODER_OUTPUTS as isize],
            FLOAT,
        );
        let mut inputs = self.bindings.iter().collect::<Vec<_>>();
        inputs.push(&input);
        let (feeds, results) = (dictionary(&inputs), dictionary(&[&output]));
        let run = if cached {
            // SAFETY: compiling the graph for these inputs' shapes; every feed
            // tensor has a value in `feeds`.
            unsafe {
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
                let executable = self
                    .graph
                    .compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                        Some(&MPSGraphDevice::deviceWithMTLDevice(&self.gpu.device)),
                        &NSDictionary::from_retained_objects(&tensors, &types),
                        &NSArray::from_slice(&[&*self.logits]),
                        None,
                        Some(&self.gpu.compilation),
                    );
                let order = executable
                    .feedTensors()
                    .ok_or_else(|| metal_error("compiled graph has no feeds"))?;
                let inputs = order
                    .iter()
                    .map(|tensor| {
                        feeds
                            .objectForKey(&tensor)
                            .ok_or_else(|| metal_error("compiled graph feed is unbound"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let output = results
                    .objectForKey(&self.logits)
                    .ok_or_else(|| metal_error("graph output is unbound"))?;
                PreparedRun::Executable {
                    executable,
                    inputs: NSArray::from_retained_slice(&inputs),
                    results: NSArray::from_retained_slice(&[output]),
                }
            }
        } else {
            PreparedRun::Graph { feeds, results }
        };
        Ok(Prepared {
            input: input.1,
            output: output.1,
            run,
        })
    }

    /// Predict `query_rows` rows. `fill` writes the `[B * H, rows, 3]` grouped
    /// inputs, given the bucket's row count as the stride, and returns false
    /// to cancel. Returns `[B, query_rows, 10]` logits.
    pub(crate) fn run(
        &self,
        query_rows: usize,
        fill: impl FnOnce(&mut [f32], usize) -> bool,
    ) -> Result<Option<Vec<f32>>> {
        let (b, h, rows) = (self.shape.ensembles, self.shape.features, bucket(query_rows));
        let mut cache = self
            .prepared
            .lock()
            .map_err(|_| metal_error("graph lock poisoned"))?;
        let uncached;
        let prepared = if rows <= CACHED_BUCKET_ROWS {
            match cache.entry(rows) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => entry.insert(self.prepare(rows, true)?),
            }
        } else {
            uncached = self.prepare(rows, false)?;
            &uncached
        };
        // SAFETY: the lock gives this call sole use of the buffer, which holds
        // `b * h * rows * 3` floats.
        let input = unsafe {
            std::slice::from_raw_parts_mut(prepared.input.contents().as_ptr().cast::<f32>(), b * h * rows * 3)
        };
        if !fill(input, rows) {
            return Ok(None);
        }
        match &prepared.run {
            PreparedRun::Graph { feeds, results } => self.gpu.run_prepared(&self.graph, feeds, results),
            // SAFETY: synchronous run; the arrays keep their buffers alive.
            PreparedRun::Executable {
                executable,
                inputs,
                results,
            } => unsafe {
                executable.runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                    &self.gpu.queue,
                    inputs,
                    Some(results),
                    Some(&self.gpu.executable_execution),
                );
            },
        }
        // SAFETY: the run is complete and wrote `[b, rows, 10]` floats.
        let output = unsafe {
            std::slice::from_raw_parts(
                prepared.output.contents().as_ptr().cast::<f32>(),
                b * rows * DECODER_OUTPUTS,
            )
        };
        Ok(Some(
            output
                .chunks_exact(rows * DECODER_OUTPUTS)
                .flat_map(|estimator| &estimator[..query_rows * DECODER_OUTPUTS])
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
/// ICL block's as `[B, 8, N, 64]` in `icl_storage`'s type. Returns an error
/// if any written value is not finite, including FP16 overflow.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // One linear graph definition.
pub(crate) fn build_context(
    model: &ModelWeights,
    weights: &Weights<'_, '_>,
    size: Shape,
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
        size.ensembles as isize,
        size.features as isize,
        size.training_rows as isize,
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
        let base_scale = builder.constant(base_scale, &[1, HEADS as isize, 1, COLUMN_HEAD_DIM as isize]);
        let first = format!("{prefix}.multihead_attn1");
        let (hidden, _, _) = builder.block(
            &inducing,
            &Source::Tokens(&source),
            &first,
            &format!("{first}.attn"),
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
            &format!("{second}.attn"),
            (b * h, n, INDUCING as isize),
            COLUMN_DIM,
            None,
            None,
        )?;
        source = output;
        column.push((key, value));
    }
    let mut rows = builder.add(
        &builder.row_stage(&source, (size.ensembles, size.features))?,
        &icl_labels,
    );
    // FP16 caches also run the N x N ICL self-attention in FP16. Doing the same
    // for the column and row attention added error without saving time.
    builder.set_attention(kind(icl_storage));
    let mut icl = Vec::with_capacity(ICL_BLOCKS);
    for (block, base_scale) in icl_base_scales.iter().enumerate() {
        let prefix = format!("icl_predictor.tf_icl.blocks.{block}");
        let base_scale = builder.constant(base_scale, &[1, HEADS as isize, 1, ICL_HEAD_DIM as isize]);
        let (output, key, value) = builder.block(
            &rows,
            &Source::Tokens(&rows),
            &prefix,
            &format!("{prefix}.attn"),
            (b, n, n),
            ICL_DIM,
            None,
            Some(&base_scale),
        )?;
        rows = output;
        icl.push((key, value));
    }

    // Column caches in context layout: zero reserved streams, then the features.
    let feature_shape = [b, h, HEADS as isize, INDUCING as isize, COLUMN_HEAD_DIM as isize];
    let reserved = unsafe {
        builder.graph.constantWithScalar_shape_dataType(
            0.0,
            &shape(&[
                b,
                RESERVED as isize,
                HEADS as isize,
                INDUCING as isize,
                COLUMN_HEAD_DIM as isize,
            ]),
            FLOAT,
        )
    };
    let full_column = |tensor: &Tensor| unsafe {
        builder.graph.concatTensor_withTensor_dimension_name(
            &reserved,
            &builder.reshape(tensor, &feature_shape),
            1,
            None,
        )
    };
    let column_shape = vec![
        b,
        h + RESERVED as isize,
        HEADS as isize,
        INDUCING as isize,
        COLUMN_HEAD_DIM as isize,
    ];
    let icl_shape = vec![b, HEADS as isize, n, ICL_HEAD_DIM as isize];
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
        .map(|(tensor, _, _)| unsafe {
            let largest = builder.graph.reductionMaximumPropagateNaNWithTensor_axis_name(
                &builder.reshape(&builder.graph.absoluteWithTensor_name(tensor, None), &[-1]),
                0,
                None,
            );
            builder.cast(&largest, FLOAT)
        })
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
    // SAFETY: the run is complete and wrote one float.
    let largest = unsafe { *outputs[outputs.len() - 1].1.contents().as_ptr().cast::<f32>() };
    if !largest.is_finite() {
        return Err(metal_error("the built context contains a non-finite value"));
    }
    Ok(())
}

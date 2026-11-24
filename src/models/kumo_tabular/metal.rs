//! Kumo-Tabular's transformer blocks on Apple Silicon GPUs, through `MPSGraph`.
//!
//! [`super::cpu`] runs the model; each block's keys and values and its
//! forward pass run here instead when there is a GPU. A graph takes the
//! layer's weights as inputs, so every layer of a kind shares one graph per
//! shape; `MPSGraph` compiles each shape once. The weights stay in shared GPU
//! buffers, converted on first use.

#![expect(
    unsafe_code,
    reason = "Objective-C calls; each unsafe block states its invariant"
)]
#![allow(
    clippy::cast_possible_wrap,
    reason = "tensor dimensions are model constants or validated sizes far below isize::MAX"
)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use objc2::{AllocAnyThread, rc::Retained};
use objc2_foundation::{NSArray, NSDictionary};
use objc2_metal::MTLBuffer;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphDevice, MPSGraphExecutable, MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
};

use super::cpu::Scaling;
use crate::kernels::metal::{
    Binding, Buffer, FLOAT, Gpu, ModelWeights, Tensor, builder::Builder, metal_error, shape,
};
use crate::{Error, Result};

/// The model's weights on the GPU and the graphs built so far.
pub(crate) struct Metal {
    weights: ModelWeights,
    graphs: Mutex<HashMap<Key, Arc<Graph>>>,
}

/// What a block's graph depends on: its kind, widths, and shapes.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Key {
    /// The forward pass, or else the keys and values.
    pub forward: bool,
    pub width: usize,
    pub heads: usize,
    pub groups: usize,
    pub rows: usize,
    pub kv_groups: usize,
    pub kv_heads: usize,
    pub kv_rows: usize,
    pub rope: bool,
    pub scaling: Scaling,
}

/// A graph compiled for its shapes, its placeholders in the order callers
/// feed them, its outputs, and the executable's orders of both. Compiling
/// skips `MPSGraph`'s per-run lookup, which cost milliseconds per block.
struct Graph {
    /// The graph the executable and the tensors come from, kept alive with them.
    #[expect(dead_code, reason = "held only to keep the graph alive")]
    source: Retained<MPSGraph>,
    executable: Retained<MPSGraphExecutable>,
    inputs: Vec<(Tensor, Vec<isize>)>,
    outputs: Vec<(Tensor, Vec<isize>)>,
    feed_order: Vec<usize>,
    target_order: Vec<usize>,
}

// SAFETY: the graph is only built under `Metal::graphs`'s lock and run by
// `Gpu::run`, which waits for completion; `MPSGraph` runs do not mutate the
// graph. Retain and release are atomic.
unsafe impl Send for Graph {}
// SAFETY: as above.
unsafe impl Sync for Graph {}

/// What a graph input is fed from: data for this call, or a model weight,
/// converted once under its name.
pub(crate) enum Feed<'a> {
    Data(&'a [f32]),
    Weight(String, Box<dyn FnOnce() -> Vec<f32> + 'a>),
}

impl Metal {
    /// The system GPU, or `None` without one or with `IRONICL_NO_METAL` set.
    pub(crate) fn new() -> Option<Self> {
        if std::env::var_os("IRONICL_NO_METAL").is_some() {
            return None;
        }
        let gpu = Gpu::new().ok().flatten()?;
        Some(Self {
            weights: ModelWeights::new(gpu),
            graphs: Mutex::default(),
        })
    }

    /// Run the graph of `key` on `feeds`, in its feed order, and return its
    /// outputs.
    pub(crate) fn run(&self, key: Key, feeds: Vec<Feed<'_>>) -> Result<Vec<Vec<f32>>> {
        let graph = {
            let mut graphs = self
                .graphs
                .lock()
                .map_err(|_| metal_error("graph cache lock poisoned"))?;
            if let Some(graph) = graphs.get(&key) {
                Arc::clone(graph)
            } else {
                let graph = Arc::new(self.build(key)?);
                graphs.insert(key, Arc::clone(&graph));
                graph
            }
        };
        if feeds.len() != graph.inputs.len() {
            return Err(Error::InvalidArgument("a GPU block got the wrong inputs".into()));
        }
        let gpu = self.weights.gpu();
        let inputs = graph
            .inputs
            .iter()
            .zip(feeds)
            .map(|((tensor, dimensions), feed)| {
                let buffer = match feed {
                    Feed::Data(values) => gpu.buffer(values)?,
                    Feed::Weight(name, values) => self.weights.get(&name, values)?,
                };
                Ok((tensor.clone(), buffer, dimensions.clone(), FLOAT))
            })
            .collect::<Result<Vec<Binding>>>()?;
        let outputs = graph
            .outputs
            .iter()
            .map(|(tensor, dimensions)| {
                let elements = dimensions.iter().product::<isize>().unsigned_abs();
                Ok((tensor.clone(), gpu.empty(elements)?, dimensions.clone(), FLOAT))
            })
            .collect::<Result<Vec<Binding>>>()?;
        let data = |bindings: &[Binding], order: &[usize]| {
            let items: Vec<Retained<MPSGraphTensorData>> = order
                .iter()
                .map(|&index| {
                    let (_, buffer, dimensions, kind) = &bindings[index];
                    // SAFETY: the binding retains `buffer`, which holds `dimensions`
                    // floats (made by `buffer`, `get`, or `empty` above).
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
            NSArray::from_retained_slice(&items)
        };
        let (feeds, results) = (
            data(&inputs, &graph.feed_order),
            data(&outputs, &graph.target_order),
        );
        // SAFETY: the execution descriptor waits until the GPU finishes; `inputs` and
        // `outputs` retain every buffer the tensor data refer to until then.
        unsafe {
            graph
                .executable
                .runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                    &gpu.queue,
                    &feeds,
                    Some(&results),
                    Some(&Gpu::executable_execution()),
                );
        }
        Ok(outputs
            .iter()
            .map(|(_, buffer, dimensions, _)| read(buffer, dimensions))
            .collect())
    }

    fn build(&self, key: Key) -> Result<Graph> {
        let no_weights = |name: &str| -> Result<&[f32]> { Err(metal_error(&format!("no weight {name}"))) };
        let builder = Builder::new(&self.weights, &no_weights);
        let mut inputs = Vec::new();
        let mut input = |dimensions: &[isize]| {
            let tensor = builder.placeholder(dimensions);
            inputs.push((tensor.clone(), dimensions.to_vec()));
            tensor
        };
        let outputs = if key.forward {
            forward_graph(&builder, key, &mut input)
        } else {
            key_value_graph(&builder, key, &mut input)
        };
        let gpu = self.weights.gpu();
        let tensors: Vec<&MPSGraphTensor> = inputs.iter().map(|(tensor, _)| &**tensor).collect();
        let targets: Vec<&MPSGraphTensor> = outputs.iter().map(|(tensor, _)| &**tensor).collect();
        // SAFETY: compiles the graph for its own placeholders, typed with their
        // shapes, under the cache lock; touches no buffer memory.
        let executable = unsafe {
            let types: Vec<_> = inputs
                .iter()
                .map(|(_, dimensions)| {
                    MPSGraphShapedType::initWithShape_dataType(
                        MPSGraphShapedType::alloc(),
                        Some(&shape(dimensions)),
                        FLOAT,
                    )
                })
                .collect();
            builder
                .graph
                .compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                    Some(&MPSGraphDevice::deviceWithMTLDevice(&gpu.device)),
                    &NSDictionary::from_retained_objects(&tensors, &types),
                    &NSArray::from_slice(&targets),
                    None,
                    Some(&Gpu::compilation()),
                )
        };
        let position = |tensors: &[&MPSGraphTensor], tensor: &MPSGraphTensor| {
            tensors
                .iter()
                .position(|candidate| std::ptr::eq(*candidate, tensor))
                .ok_or_else(|| metal_error("compiled graph has an unknown tensor"))
        };
        // SAFETY: property reads of the executable compiled above.
        let (feed_tensors, target_tensors) =
            unsafe { (executable.feedTensors(), executable.targetTensors()) };
        let feed_order = feed_tensors
            .ok_or_else(|| metal_error("compiled graph has no feeds"))?
            .iter()
            .map(|tensor| position(&tensors, &tensor))
            .collect::<Result<_>>()?;
        let target_order = target_tensors
            .ok_or_else(|| metal_error("compiled graph has no targets"))?
            .iter()
            .map(|tensor| position(&targets, &tensor))
            .collect::<Result<_>>()?;
        Ok(Graph {
            source: builder.graph,
            executable,
            inputs,
            outputs,
            feed_order,
            target_order,
        })
    }
}

type Outputs = Vec<(Tensor, Vec<isize>)>;

/// A block's keys and values. Feeds, in order: the `[groups, rows, width]`
/// rows, the key-value norm, the key and value projections (transposed weight,
/// bias), and with `RoPE` its `[rows, head_dim]` cosines and signed sines.
/// Outputs: the `[groups, heads, rows, head_dim]` keys and values.
fn key_value_graph(
    builder: &Builder<'_, '_>,
    key: Key,
    input: &mut impl FnMut(&[isize]) -> Tensor,
) -> Outputs {
    let (width, heads, head_dim) = (
        key.width as isize,
        key.heads as isize,
        (key.width / key.heads) as isize,
    );
    let (groups, rows) = (key.groups as isize, key.rows as isize);
    let tokens = input(&[groups, rows, width]);
    let norm = input(&[width]);
    let normed = rms(builder, &tokens, Some(&norm), f64::from(f32::EPSILON));
    let (key_weight, key_bias) = (input(&[width, width]), input(&[width]));
    let (value_weight, value_bias) = (input(&[width, width]), input(&[width]));
    let split = |projected: &Tensor| {
        builder.transpose(
            &builder.reshape(projected, &[groups, rows, heads, head_dim]),
            1,
            2,
        )
    };
    let mut keys = split(&builder.add(&builder.matmul(&normed, &key_weight), &key_bias));
    if key.rope {
        let (cos, sin) = (input(&[rows, head_dim]), input(&[rows, head_dim]));
        keys = builder.rope(&keys, (&cos, &sin), key.width / key.heads);
    }
    let keys = rms(builder, &keys, None, 1e-6);
    let values = split(&builder.add(&builder.matmul(&normed, &value_weight), &value_bias));
    let dimensions = vec![groups, heads, rows, head_dim];
    vec![(keys, dimensions.clone()), (values, dimensions)]
}

/// A block's forward pass. Feeds, in order: the `[groups, rows, width]`
/// queries, the keys and values, the query norm, the query projection, with
/// `RoPE` the queries' cosines and signed sines, with query scaling each
/// head's `head_scale · ln n` and, gated, the gate's two layers, the output
/// projection, and the MLP's norm and two layers. Output: `[groups, rows,
/// width]`.
fn forward_graph(builder: &Builder<'_, '_>, key: Key, input: &mut impl FnMut(&[isize]) -> Tensor) -> Outputs {
    let (width, heads) = (key.width as isize, key.heads as isize);
    let head_dim = key.width / key.heads;
    let dim = head_dim as isize;
    let (groups, rows) = (key.groups as isize, key.rows as isize);
    let query = input(&[groups, rows, width]);
    let cached = [
        key.kv_groups as isize,
        key.kv_heads as isize,
        key.kv_rows as isize,
        dim,
    ];
    let (keys, values) = (input(&cached), input(&cached));
    let norm = input(&[width]);
    let (query_weight, query_bias) = (input(&[width, width]), input(&[width]));
    let normed = rms(builder, &query, Some(&norm), f64::from(f32::EPSILON));
    let projected = builder.add(&builder.matmul(&normed, &query_weight), &query_bias);
    let mut queries = builder.transpose(&builder.reshape(&projected, &[groups, rows, heads, dim]), 1, 2);
    if key.rope {
        let (cos, sin) = (input(&[rows, dim]), input(&[rows, dim]));
        queries = builder.rope(&queries, (&cos, &sin), head_dim);
    }
    queries = rms(builder, &queries, None, 1e-6);
    if key.scaling != Scaling::None {
        let scale = input(&[1, heads, 1, 1]);
        let scaled = builder.multiply(&queries, &scale);
        queries = if key.scaling == Scaling::Gated {
            let (first, first_bias) = (input(&[dim, 64]), input(&[64]));
            let (second, second_bias) = (input(&[64, dim]), input(&[dim]));
            let hidden = builder.gelu(&builder.add(&builder.matmul(&queries, &first), &first_bias));
            let modulation = builder.add(&builder.matmul(&hidden, &second), &second_bias);
            // SAFETY: graph construction on `builder.graph`, as in `Builder`.
            let tanh = unsafe { builder.graph.tanhWithTensor_name(&modulation, None) };
            builder.multiply(&scaled, &builder.add(&tanh, &builder.scalar(1.0)))
        } else {
            scaled
        };
    }
    let full = [groups, heads, key.kv_rows as isize, dim];
    let (keys, values) = (
        expand(builder, &keys, key, &full),
        expand(builder, &values, key, &full),
    );
    #[expect(clippy::cast_precision_loss, reason = "Head widths are small")]
    let scale = (head_dim as f32).sqrt().recip();
    // SAFETY: graph construction on `builder.graph`, as in `Builder`.
    let attended = unsafe {
        builder
            .graph
            .scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name(
                &queries, &keys, &values, scale, None,
            )
    };
    let merged = builder.reshape(&builder.transpose(&attended, 1, 2), &[groups, rows, width]);
    let (out_weight, out_bias) = (input(&[width, width]), input(&[width]));
    let residual = builder.add(
        &query,
        &builder.add(&builder.matmul(&merged, &out_weight), &out_bias),
    );
    let norm = input(&[width]);
    let (first, first_bias) = (input(&[width, 2 * width]), input(&[2 * width]));
    let (second, second_bias) = (input(&[2 * width, width]), input(&[width]));
    let hidden = rms(builder, &residual, Some(&norm), f64::from(f32::EPSILON));
    let hidden = builder.gelu(&builder.add(&builder.matmul(&hidden, &first), &first_bias));
    let output = builder.add(
        &residual,
        &builder.add(&builder.matmul(&hidden, &second), &second_bias),
    );
    vec![(output, vec![groups, rows, width])]
}

/// `x / sqrt(mean(x²) + eps)` over the last axis, times `weight` when given.
fn rms(builder: &Builder<'_, '_>, x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Tensor {
    // SAFETY: graph construction on `builder.graph`, as in `Builder`.
    let scale = unsafe {
        let mean = builder
            .graph
            .meanOfTensor_axes_name(&builder.multiply(x, x), &shape(&[-1]), None);
        builder
            .graph
            .reciprocalSquareRootWithTensor_name(&builder.add(&mean, &builder.scalar(eps)), None)
    };
    let normed = builder.multiply(x, &scale);
    weight.map_or(normed.clone(), |weight| builder.multiply(&normed, weight))
}

/// Keys or values broadcast to the queries' groups and heads: query head `h`
/// uses key head `h / (heads / key heads)`.
fn expand(builder: &Builder<'_, '_>, x: &Tensor, key: Key, full: &[isize; 4]) -> Tensor {
    if key.kv_heads == key.heads && key.kv_groups == key.groups {
        return x.clone();
    }
    let [g, heads, n, hd] = *full;
    let (kv_groups, kv_heads) = (key.kv_groups as isize, key.kv_heads as isize);
    let grouped = builder.reshape(x, &[kv_groups, kv_heads, 1, n, hd]);
    // SAFETY: graph construction on `builder.graph`, as in `Builder`.
    let broadcast = unsafe {
        builder.graph.broadcastTensor_toShape_name(
            &grouped,
            &shape(&[g, kv_heads, heads / kv_heads, n, hd]),
            None,
        )
    };
    builder.reshape(&broadcast, full)
}

/// A shared buffer's `dimensions` floats.
fn read(buffer: &Buffer, dimensions: &[isize]) -> Vec<f32> {
    let elements = dimensions.iter().product::<isize>().unsigned_abs();
    // SAFETY: `Metal::run` made the buffer with `empty(elements)` and the
    // synchronous run that wrote it finished.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), elements).to_vec() }
}

//! Kumo-Tabular on Apple Silicon GPUs: a member's whole forward pass as one
//! `MPSGraph` executable, in FP32.
//!
//! Fitting runs one graph per training shape, for every member: the cell
//! embedding, the interleaved column and row stages, and the ICL stage, with
//! the member's preprocessed rows, kinds, and label vectors as inputs, and
//! each column layer's inducing-point keys and values and each ICL layer's
//! keys and values as outputs. Predicting runs one graph per query bucket that
//! reads them. Both graphs build their blocks from the shared block of
//! `kernels::metal::block`, the weights bound to the model's shared buffers,
//! and split their attentions into chunks that bound the scores held at once.

#![expect(
    unsafe_code,
    reason = "Objective-C calls; each unsafe block states its invariant"
)]
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
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

use super::Config;
use crate::Result;
use crate::kernels::metal::{
    Binding, Buffer, FLOAT, Gpu, ModelWeights, Tensor, Weights,
    block::{Norm, Rope, Source, Spec},
    builder::Builder,
    metal_error, shape,
};

/// Graph outputs with their shapes.
type Outputs = Vec<(Tensor, Vec<isize>)>;

/// Attention scores a graph holds at once, in values.
const SCORES: usize = 1 << 27;
/// The largest query bucket; more rows run in several.
const MAX_BUCKET: usize = 1024;

fn dim(value: u64) -> usize {
    usize::try_from(value).expect("model dimensions fit in usize")
}

/// What a graph is compiled for.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Key {
    Fit {
        rows: usize,
        columns: usize,
    },
    Predict {
        rows: usize,
        columns: usize,
        training: usize,
    },
}

/// A compiled graph: its weight bindings, its inputs and outputs with their
/// shapes, and the executable's orders of feeds (weights, then inputs) and
/// of targets.
struct Compiled {
    /// The graph the executable and the tensors come from, kept alive with them.
    #[expect(dead_code, reason = "held only to keep the graph alive")]
    source: Retained<MPSGraph>,
    executable: Retained<MPSGraphExecutable>,
    weights: Vec<Binding>,
    inputs: Vec<(Tensor, Vec<isize>)>,
    outputs: Vec<(Tensor, Vec<isize>)>,
    feed_order: Vec<usize>,
    target_order: Vec<usize>,
}

// SAFETY: a compiled graph is built under `Metal::graphs`'s lock and only run
// after, by synchronous runs that do not mutate it. Retain and release are
// atomic.
unsafe impl Send for Compiled {}
// SAFETY: as above.
unsafe impl Sync for Compiled {}

/// A graph input: a member's data, or a buffer kept on the GPU.
enum Feed<'a> {
    Data(&'a [f32]),
    Buffer(&'a Buffer),
}

/// A member's fitted context: each column layer's keys and values, then each
/// ICL layer's, in GPU buffers.
pub(crate) struct Context {
    caches: Vec<Buffer>,
    training_rows: usize,
}

// SAFETY: the buffers are filled by a finished run and only read after.
unsafe impl Send for Context {}
// SAFETY: as above.
unsafe impl Sync for Context {}

/// Kumo-Tabular's GPU: the model's weights in shared buffers and its graphs.
pub(crate) struct Metal {
    weights: ModelWeights,
    config: Config,
    outputs: usize,
    graphs: Mutex<HashMap<Key, Arc<Compiled>>>,
}

/// A block's weight names.
struct Names {
    query_norm: String,
    key_value_norm: String,
    qkv: (String, String),
    out: String,
    mlp_norm: String,
    mlp_in: String,
    mlp_out: String,
    gate: (String, String),
}

impl Names {
    fn new(prefix: &str) -> Self {
        Self {
            query_norm: format!("{prefix}.query_norm"),
            key_value_norm: format!("{prefix}.key_value_norm"),
            qkv: (
                format!("{prefix}.attn.qkv_lin.weight"),
                format!("{prefix}.attn.qkv_lin.bias"),
            ),
            out: format!("{prefix}.attn.out_lin"),
            mlp_norm: format!("{prefix}.mlp.0"),
            mlp_in: format!("{prefix}.mlp.1"),
            mlp_out: format!("{prefix}.mlp.3"),
            gate: (
                format!("{prefix}.attn.sdpa.query_scaling.gate.0"),
                format!("{prefix}.attn.sdpa.query_scaling.gate.2"),
            ),
        }
    }

    fn spec<'a>(
        &'a self,
        (width, heads): (usize, usize),
        rope: Option<&'a Rope<'a>>,
        scale: Option<&'a Tensor>,
        gated: bool,
    ) -> Spec<'a> {
        Spec {
            width,
            heads,
            norm: Norm::Rms,
            query_norm: &self.query_norm,
            key_value_norm: &self.key_value_norm,
            qkv: (&self.qkv.0, &self.qkv.1),
            out: &self.out,
            mlp_norm: &self.mlp_norm,
            mlp_in: &self.mlp_in,
            mlp_out: &self.mlp_out,
            rope,
            head_norm: true,
            scale,
            gate: gated.then_some((self.gate.0.as_str(), self.gate.1.as_str())),
        }
    }
}

/// The column caches during fitting, or as inputs when predicting.
enum Columns<'t> {
    Fit(&'t Tensor),
    Predict(&'t [(Tensor, Tensor)]),
}

impl Metal {
    /// The system GPU, or `None` without one or with `IRONICL_NO_METAL` set.
    pub(crate) fn new(config: Config, outputs: usize) -> Option<Self> {
        if std::env::var_os("IRONICL_NO_METAL").is_some() {
            return None;
        }
        let gpu = Gpu::new().ok().flatten()?;
        Some(Self {
            weights: ModelWeights::new(gpu),
            config,
            outputs,
            graphs: Mutex::default(),
        })
    }

    /// Fit a member's context on its `[rows, columns]` training rows `x`,
    /// whose `kinds` are 1 for category codes, with their imputation `means`
    /// and label vectors for the cells and for the ICL stage.
    pub(crate) fn fit(
        &self,
        weights: &Weights<'_, '_>,
        (x, means, kinds): (&[f32], &[f32], &[f32]),
        (cell_labels, icl_labels): (&[f32], &[f32]),
    ) -> Result<Context> {
        let columns = kinds.len();
        let rows = x.len() / columns;
        let (imputed, missing) = impute(x, means);
        let feeds = vec![
            Feed::Data(&imputed),
            Feed::Data(&missing),
            Feed::Data(kinds),
            Feed::Data(cell_labels),
            Feed::Data(icl_labels),
        ];
        let caches = self.run(Key::Fit { rows, columns }, weights, &feeds)?;
        Ok(Context {
            caches,
            training_rows: rows,
        })
    }

    /// The outputs of each of the `[rows, columns]` query rows `x`, in
    /// buckets of at most [`MAX_BUCKET`] rows.
    pub(crate) fn predict(
        &self,
        weights: &Weights<'_, '_>,
        context: &Context,
        (x, means, kinds): (&[f32], &[f32], &[f32]),
    ) -> Result<Vec<f32>> {
        let columns = kinds.len();
        let mut out = Vec::with_capacity(x.len() / columns * self.outputs);
        for chunk in x.chunks(MAX_BUCKET * columns) {
            let rows = chunk.len() / columns;
            let bucket = rows.next_power_of_two().clamp(16, MAX_BUCKET);
            let mut padded = chunk.to_vec();
            padded.resize(bucket * columns, 0.0);
            let (imputed, missing) = impute(&padded, means);
            let mut feeds = vec![Feed::Data(&imputed), Feed::Data(&missing), Feed::Data(kinds)];
            feeds.extend(context.caches.iter().map(Feed::Buffer));
            let key = Key::Predict {
                rows: bucket,
                columns,
                training: context.training_rows,
            };
            let outputs = self.run(key, weights, &feeds)?;
            out.extend_from_slice(&read(&outputs[0], rows * self.outputs));
        }
        Ok(out)
    }

    /// Run the graph of `key`, compiling it on first use, on `feeds`, and
    /// return its output buffers.
    fn run(&self, key: Key, weights: &Weights<'_, '_>, feeds: &[Feed<'_>]) -> Result<Vec<Buffer>> {
        let compiled = {
            let mut graphs = self
                .graphs
                .lock()
                .map_err(|_| metal_error("graph cache lock poisoned"))?;
            if let Some(compiled) = graphs.get(&key) {
                Arc::clone(compiled)
            } else {
                let compiled = Arc::new(self.compile(key, weights)?);
                graphs.insert(key, Arc::clone(&compiled));
                compiled
            }
        };
        let gpu = self.weights.gpu();
        let mut bindings = compiled.weights.clone();
        for ((tensor, dimensions), feed) in compiled.inputs.iter().zip(feeds) {
            let buffer = match feed {
                Feed::Data(values) => gpu.buffer(values)?,
                Feed::Buffer(buffer) => (*buffer).clone(),
            };
            bindings.push((tensor.clone(), buffer, dimensions.clone(), FLOAT));
        }
        let outputs = compiled
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
                    // SAFETY: the binding retains `buffer`, which holds `dimensions` values of `kind`.
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
        let (inputs, results) = (
            data(&bindings, &compiled.feed_order),
            data(&outputs, &compiled.target_order),
        );
        // NaN until the run writes it: a run that fails without a word leaves it.
        // SAFETY: the first output holds at least one float and nothing else uses it yet.
        unsafe { *outputs[0].1.contents().as_ptr().cast::<f32>() = f32::NAN };
        // SAFETY: the execution descriptor waits until the GPU finishes; `bindings` and
        // `outputs` retain every buffer the tensor data refer to until then.
        unsafe {
            compiled
                .executable
                .runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                    &gpu.queue,
                    &inputs,
                    Some(&results),
                    Some(&Gpu::executable_execution()),
                );
        }
        // SAFETY: the synchronous run above finished.
        if unsafe { *outputs[0].1.contents().as_ptr().cast::<f32>() }.is_nan() {
            return Err(metal_error("the GPU run failed or wrote NaN"));
        }
        Ok(outputs.into_iter().map(|(_, buffer, _, _)| buffer).collect())
    }

    /// The fit graph for `rows` training rows of `columns` columns. Inputs:
    /// the imputed values and missing flags `[rows, columns]`, the kinds `[1,
    /// columns]`, and the cell and ICL label vectors. Outputs: each column
    /// layer's keys and values, then each ICL layer's.
    fn fit_graph(
        &self,
        builder: &Builder<'_, '_>,
        input: &mut impl FnMut(&[isize]) -> Tensor,
        (rows, columns): (usize, usize),
    ) -> Result<Outputs> {
        let config = self.config;
        let (cell, icl, icl_heads) = (
            dim(config.cell_channels),
            dim(config.icl_channels),
            dim(config.icl_heads),
        );
        let key_heads = config.query_key_value_heads.map_or(icl_heads, dim);
        let (r, c) = (rows as isize, columns as isize);
        let (x, missing, kinds) = (input(&[r, c]), input(&[r, c]), input(&[1, c]));
        let (cell_labels, icl_labels) = (input(&[r, cell as isize]), input(&[r, icl as isize]));
        let (hidden, mut outputs) = self.row_embedding(
            builder,
            (&x, &missing, &kinds),
            (rows, columns),
            &Columns::Fit(&cell_labels),
        )?;
        let mut hidden = builder.reshape(&builder.add(&hidden, &icl_labels), &[1, r, icl as isize]);
        let (layers, log) = (dim(config.icl_layers), (rows.max(1) as f32).ln());
        for layer in 0..layers {
            let names = Names::new(&format!("icl_block.layers.{layer}"));
            let scale = log_scale(builder, &names, log, icl_heads)?;
            let spec = names.spec((icl, icl_heads), None, Some(&scale), false);
            let (key, value) = builder.key_value(&hidden, &spec, (1, r))?;
            if layer + 1 < layers {
                hidden = attend(
                    builder,
                    &hidden,
                    &Source::Cached(&key, &value),
                    &spec,
                    (1, rows, rows),
                )?;
            }
            let dims = vec![1, key_heads as isize, r, (icl / icl_heads) as isize];
            outputs.push((builder.slice(&key, 1, 0, key_heads as isize), dims.clone()));
            outputs.push((builder.slice(&value, 1, 0, key_heads as isize), dims));
        }
        Ok(outputs)
    }

    /// The predict graph for `rows` query rows of `columns` columns against
    /// `training` training rows. Inputs: the imputed values, missing flags,
    /// and kinds, then the context's caches. Output: `[rows, outputs]`.
    fn predict_graph(
        &self,
        builder: &Builder<'_, '_>,
        input: &mut impl FnMut(&[isize]) -> Tensor,
        (rows, columns, training): (usize, usize, usize),
    ) -> Result<Outputs> {
        let config = self.config;
        let (cell, icl, icl_heads) = (
            dim(config.cell_channels),
            dim(config.icl_channels),
            dim(config.icl_heads),
        );
        let key_heads = config.query_key_value_heads.map_or(icl_heads, dim);
        let (r, c) = (rows as isize, columns as isize);
        let (x, missing, kinds) = (input(&[r, c]), input(&[r, c]), input(&[1, c]));
        let (points, heads) = (dim(config.inducing_points), dim(config.embedding_heads));
        let column_dims = [c, heads as isize, points as isize, (cell / heads) as isize];
        let column_caches: Vec<(Tensor, Tensor)> = (0..dim(config.embedding_layers))
            .map(|_| (input(&column_dims), input(&column_dims)))
            .collect();
        let icl_dims = [
            1,
            key_heads as isize,
            training as isize,
            (icl / icl_heads) as isize,
        ];
        let icl_caches: Vec<(Tensor, Tensor)> = (0..dim(config.icl_layers))
            .map(|_| (input(&icl_dims), input(&icl_dims)))
            .collect();
        let (hidden, _) = self.row_embedding(
            builder,
            (&x, &missing, &kinds),
            (rows, columns),
            &Columns::Predict(&column_caches),
        )?;
        let mut hidden = builder.reshape(&hidden, &[1, r, icl as isize]);
        let log = (training.max(1) as f32).ln();
        for (layer, (key, value)) in icl_caches.iter().enumerate() {
            let names = Names::new(&format!("icl_block.layers.{layer}"));
            let scale = log_scale(builder, &names, log, icl_heads)?;
            let spec = names.spec((icl, icl_heads), None, Some(&scale), false);
            hidden = attend(
                builder,
                &hidden,
                &Source::Cached(key, value),
                &spec,
                (1, rows, training),
            )?;
        }
        let hidden = builder.reshape(&hidden, &[r, icl as isize]);
        let norm = builder.named_weight("icl_block.norm.weight", &[icl as isize])?;
        let normed = builder.rms(&hidden, Some(&norm), f64::from(f32::EPSILON));
        let wide = builder.gelu(&builder.linear(&normed, "icl_block.head.0", icl, 2 * icl)?);
        let out = builder.linear(&wide, "icl_block.head.2", 2 * icl, self.outputs)?;
        Ok(vec![(out, vec![r, self.outputs as isize])])
    }

    fn compile(&self, key: Key, weights: &Weights<'_, '_>) -> Result<Compiled> {
        let builder = Builder::new(&self.weights, weights);
        let mut inputs = Vec::new();
        let mut input = |dimensions: &[isize]| {
            let tensor = builder.placeholder(dimensions);
            inputs.push((tensor.clone(), dimensions.to_vec()));
            tensor
        };
        let outputs = match key {
            Key::Fit { rows, columns } => self.fit_graph(&builder, &mut input, (rows, columns))?,
            Key::Predict {
                rows,
                columns,
                training,
            } => self.predict_graph(&builder, &mut input, (rows, columns, training))?,
        };
        let weight_bindings = builder.bindings.borrow().clone();
        let gpu = self.weights.gpu();
        let mut tensors: Vec<&MPSGraphTensor> = weight_bindings.iter().map(|binding| &*binding.0).collect();
        tensors.extend(inputs.iter().map(|(tensor, _)| &**tensor));
        let mut dimensions: Vec<&Vec<isize>> = weight_bindings.iter().map(|binding| &binding.2).collect();
        dimensions.extend(inputs.iter().map(|(_, dims)| dims));
        let targets: Vec<&MPSGraphTensor> = outputs.iter().map(|(tensor, _)| &**tensor).collect();
        // SAFETY: compiles the graph for its own placeholders, typed with their shapes, under the
        // cache lock; touches no buffer memory.
        let executable = unsafe {
            let types: Vec<_> = dimensions
                .iter()
                .map(|dims| {
                    MPSGraphShapedType::initWithShape_dataType(
                        MPSGraphShapedType::alloc(),
                        Some(&shape(dims)),
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
        Ok(Compiled {
            source: builder.graph,
            executable,
            weights: weight_bindings,
            inputs,
            outputs,
            feed_order,
            target_order,
        })
    }

    /// The row embedding of `rows` rows of `columns` columns, `[rows, icl]`,
    /// from the imputed values, their missing flags, and the columns' kinds,
    /// and, fitting, the column keys and values to record as outputs.
    fn row_embedding(
        &self,
        builder: &Builder<'_, '_>,
        (x, missing, kinds): (&Tensor, &Tensor, &Tensor),
        (rows, columns): (usize, usize),
        stage: &Columns<'_>,
    ) -> Result<(Tensor, Outputs)> {
        let config = self.config;
        let (cell, readout, points) = (
            dim(config.cell_channels),
            dim(config.readout_tokens),
            dim(config.inducing_points),
        );
        let heads = dim(config.embedding_heads);
        let head_dim = cell / heads;
        let (c, r, d) = (columns as isize, rows as isize, cell as isize);
        let mut cells = builder.transpose(&self.cells(builder, (x, missing, kinds), (rows, columns))?, 0, 1);
        if let Columns::Fit(labels) = stage {
            cells = builder.add(&cells, &builder.reshape(labels, &[1, r, d]));
        }
        let readout_token = builder.named_weight("row_embedding.readout_token", &[1, readout as isize, d])?;
        let mut readouts = builder.broadcast(&readout_token, &[r, readout as isize, d]);
        let mut recorded = Vec::new();
        let tokens = readout + columns;
        let layers = dim(config.embedding_layers);
        for layer in 0..layers {
            let prefix = format!("row_embedding.col_blocks.{layer}");
            let output = Names::new(&format!("{prefix}.output_block"));
            let output_spec = output.spec((cell, heads), None, None, false);
            let fitted;
            let (key, value) = match stage {
                Columns::Fit(_) => {
                    let inducing = Names::new(&format!("{prefix}.inducing_block"));
                    let scale = log_scale(builder, &inducing, (rows.max(1) as f32).ln(), heads)?;
                    let spec = inducing.spec((cell, heads), None, Some(&scale), false);
                    let point_weights = builder
                        .named_weight(&format!("{prefix}.inducing_points"), &[1, points as isize, d])?;
                    let queries = builder.broadcast(&point_weights, &[c, points as isize, d]);
                    let summaries = attend(
                        builder,
                        &queries,
                        &Source::Tokens(&cells),
                        &spec,
                        (columns, points, rows),
                    )?;
                    fitted = builder.key_value(&summaries, &output_spec, (c, points as isize))?;
                    let dims = vec![c, heads as isize, points as isize, head_dim as isize];
                    recorded.push((fitted.0.clone(), dims.clone()));
                    recorded.push((fitted.1.clone(), dims));
                    (&fitted.0, &fitted.1)
                }
                Columns::Predict(caches) => (&caches[layer].0, &caches[layer].1),
            };
            cells = attend(
                builder,
                &cells,
                &Source::Cached(key, value),
                &output_spec,
                (columns, rows, points),
            )?;
            // The row stage: each row's readouts and cells attend to each other.
            let row = Names::new(&format!("row_embedding.row_blocks.{layer}"));
            let scale = log_scale(builder, &row, (tokens as f32).ln(), heads)?;
            let frequencies = builder.weight(&format!(
                "row_embedding.row_blocks.{layer}.attn.query_transform.0.inv_freq"
            ))?;
            let tables = |len: usize| {
                let (cos, sin) = rope_tables(len, frequencies, head_dim);
                let dims = [len as isize, head_dim as isize];
                (builder.constant(&cos, &dims), builder.constant(&sin, &dims))
            };
            let (all, firsts) = (tables(tokens), tables(readout));
            let sequences = builder.concat(&readouts, &builder.transpose(&cells, 0, 1), 1);
            if layer + 1 == layers {
                let rope = Rope {
                    query: (&firsts.0, &firsts.1),
                    key: (&all.0, &all.1),
                };
                let spec = row.spec((cell, heads), Some(&rope), Some(&scale), true);
                readouts = attend_rows(builder, &readouts, &sequences, &spec, (rows, readout, tokens))?;
            } else {
                let rope = Rope {
                    query: (&all.0, &all.1),
                    key: (&all.0, &all.1),
                };
                let spec = row.spec((cell, heads), Some(&rope), Some(&scale), true);
                let out = attend_rows(builder, &sequences, &sequences, &spec, (rows, tokens, tokens))?;
                readouts = builder.slice(&out, 1, 0, readout as isize);
                cells = builder.transpose(&builder.slice(&out, 1, readout as isize, c), 0, 1);
            }
        }
        let norm = builder.named_weight("row_embedding.norm.weight", &[d])?;
        let normed = builder.reshape(
            &builder.rms(&readouts, Some(&norm), f64::from(f32::EPSILON)),
            &[r, (readout * cell) as isize],
        );
        let icl = dim(config.icl_channels);
        let hidden = if readout * cell == icl {
            normed
        } else {
            builder.linear(&normed, "row_project", readout * cell, icl)?
        };
        Ok((hidden, recorded))
    }

    /// The Fourier cell embedding `[rows, columns, cell]`: each cell embeds
    /// its column with the columns 1 and 3 after it, circularly, by its
    /// kind's frequencies and projection, plus each one's missing term.
    fn cells(
        &self,
        builder: &Builder<'_, '_>,
        (x, missing, kinds): (&Tensor, &Tensor, &Tensor),
        (rows, columns): (usize, usize),
    ) -> Result<Tensor> {
        let cell = dim(self.config.cell_channels);
        let frequencies = dim(self.config.frequencies);
        let (c, r) = (columns as isize, rows as isize);
        let name = "row_embedding.cell_embedding";
        let numeric = builder.named_weight(&format!("{name}.num_freq"), &[3, frequencies as isize])?;
        let categorical = builder.named_weight(&format!("{name}.cat_freq"), &[3, frequencies as isize])?;
        let nan = builder.weight(&format!("{name}.nan_lin.weight"))?;
        let one = builder.scalar(1.0);
        let mut sum: Option<Tensor> = None;
        for g in 0..3 {
            let offset = ((1 << g) - 1) % columns;
            let (value, flag, kind) = (
                roll(builder, x, offset, c),
                roll(builder, missing, offset, c),
                roll(builder, kinds, offset, c),
            );
            let kind = builder.reshape(&kind, &[1, c, 1]);
            // SAFETY: graph construction on `builder.graph`, as in `Builder`.
            let other = unsafe {
                builder
                    .graph
                    .subtractionWithPrimaryTensor_secondaryTensor_name(&one, &kind, None)
            };
            let row = |table: &Tensor| {
                builder.reshape(
                    &builder.slice(table, 0, g as isize, 1),
                    &[1, 1, frequencies as isize],
                )
            };
            let frequency = builder.add(
                &builder.multiply(&kind, &row(&categorical)),
                &builder.multiply(&other, &row(&numeric)),
            );
            let z = builder.multiply(&builder.reshape(&value, &[r, c, 1]), &frequency);
            // SAFETY: graph construction on `builder.graph`, as in `Builder`.
            let features = unsafe {
                builder.concat(
                    &builder.graph.sinWithTensor_name(&z, None),
                    &builder.graph.cosWithTensor_name(&z, None),
                    2,
                )
            };
            let projected = builder.add(
                &builder.multiply(
                    &kind,
                    &builder.linear(&features, &format!("{name}.cat_lin"), 2 * frequencies, cell)?,
                ),
                &builder.multiply(
                    &other,
                    &builder.linear(&features, &format!("{name}.num_lin"), 2 * frequencies, cell)?,
                ),
            );
            let nan_column = builder.model_weight(
                &format!("{name}.nan_lin.weight[:, {g}]"),
                || (0..cell).map(|channel| nan[channel * 3 + g]).collect(),
                &[cell as isize],
            )?;
            let term = builder.add(
                &projected,
                &builder.multiply(&builder.reshape(&flag, &[r, c, 1]), &nan_column),
            );
            sum = Some(match sum {
                Some(sum) => builder.add(&sum, &term),
                None => term,
            });
        }
        Ok(sum.expect("three groups"))
    }
}

/// `head_scale · ln n` of block `names`, shaped to scale `[lead, heads,
/// rows, head_dim]` queries.
fn log_scale(builder: &Builder<'_, '_>, names: &Names, log: f32, heads: usize) -> Result<Tensor> {
    let prefix = names.out.trim_end_matches(".attn.out_lin");
    let scales: Vec<f32> = builder
        .weight(&format!("{prefix}.attn.sdpa.query_scaling.head_scale"))?
        .iter()
        .map(|a| a * log)
        .collect();
    Ok(builder.constant(&scales, &[1, heads as isize, 1, 1]))
}

/// A block on `input` `[lead, rows, width]` attending to `source`, its
/// attention split into chunks of leading groups, or for one group of
/// query rows, so that each holds at most [`SCORES`] scores.
fn attend(
    builder: &Builder<'_, '_>,
    input: &Tensor,
    source: &Source<'_>,
    spec: &Spec<'_>,
    (lead, rows, keys): (usize, usize, usize),
) -> Result<Tensor> {
    let per_query = spec.heads * keys;
    let shape = |lead: usize, rows: usize| (lead as isize, rows as isize, keys as isize);
    if lead > 1 {
        let step = (SCORES / (per_query * rows)).clamp(1, lead);
        let mut parts = Vec::new();
        for start in (0..lead).step_by(step) {
            let count = step.min(lead - start) as isize;
            let part = |t: &Tensor| builder.slice(t, 0, start as isize, count);
            let source = match *source {
                Source::Cached(key, value) => (Some((part(key), part(value))), None),
                Source::Tokens(tokens) => (None, Some(part(tokens))),
            };
            let source = match &source {
                (Some((key, value)), _) => Source::Cached(key, value),
                (_, Some(tokens)) => Source::Tokens(tokens),
                _ => unreachable!("a source is cached or tokens"),
            };
            parts.push(
                builder
                    .transformer_block(&part(input), &source, spec, shape(count.unsigned_abs(), rows))?
                    .0,
            );
        }
        return Ok(concat(builder, &parts, 0));
    }
    let step = (SCORES / per_query).clamp(1, rows);
    let mut parts = Vec::new();
    for start in (0..rows).step_by(step) {
        let count = step.min(rows - start);
        let queries = builder.slice(input, 1, start as isize, count as isize);
        parts.push(
            builder
                .transformer_block(&queries, source, spec, shape(1, count))?
                .0,
        );
    }
    Ok(concat(builder, &parts, 1))
}

/// The row stage's block on `queries` `[rows, q, cell]`, each row
/// attending to its own `sequences` `[rows, tokens, cell]`, in chunks of
/// rows.
fn attend_rows(
    builder: &Builder<'_, '_>,
    queries: &Tensor,
    sequences: &Tensor,
    spec: &Spec<'_>,
    (rows, q, tokens): (usize, usize, usize),
) -> Result<Tensor> {
    let step = (SCORES / (spec.heads * q * tokens)).clamp(1, rows);
    let mut parts = Vec::new();
    for start in (0..rows).step_by(step) {
        let count = step.min(rows - start) as isize;
        let (queries, sequences) = (
            builder.slice(queries, 0, start as isize, count),
            builder.slice(sequences, 0, start as isize, count),
        );
        parts.push(
            builder
                .transformer_block(
                    &queries,
                    &Source::Tokens(&sequences),
                    spec,
                    (count, q as isize, tokens as isize),
                )?
                .0,
        );
    }
    Ok(concat(builder, &parts, 0))
}

/// `x` with its columns (axis 1) rotated left by `offset`: column `c` takes
/// column `(c + offset) % columns`.
fn roll(builder: &Builder<'_, '_>, x: &Tensor, offset: usize, columns: isize) -> Tensor {
    if offset == 0 {
        return x.clone();
    }
    let offset = offset as isize;
    builder.concat(
        &builder.slice(x, 1, offset, columns - offset),
        &builder.slice(x, 1, 0, offset),
        1,
    )
}

fn concat(builder: &Builder<'_, '_>, parts: &[Tensor], axis: isize) -> Tensor {
    let mut joined = parts[0].clone();
    for part in &parts[1..] {
        joined = builder.concat(&joined, part, axis);
    }
    joined
}

/// `RoPE`'s cosines and signed sines `[-sin, sin]` for `len` positions, as
/// `Builder::rope` takes them.
fn rope_tables(len: usize, frequencies: &[f32], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
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

/// `x` with missing values replaced by their column's `means`, and the
/// missing flags.
fn impute(x: &[f32], means: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let columns = means.len();
    let mut imputed = x.to_vec();
    let mut missing = vec![0.0; x.len()];
    for (index, value) in imputed.iter_mut().enumerate() {
        if value.is_nan() {
            *value = means[index % columns];
            missing[index] = 1.0;
        }
    }
    (imputed, missing)
}

/// A shared buffer's first `len` floats.
fn read(buffer: &Buffer, len: usize) -> Vec<f32> {
    // SAFETY: the buffer holds at least `len` floats, written by a finished run.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), len).to_vec() }
}

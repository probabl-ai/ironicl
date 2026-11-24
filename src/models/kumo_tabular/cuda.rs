//! Kumo-Tabular on NVIDIA GPUs: a member's whole forward pass on the device,
//! in FP32, its context kept there.
//!
//! The weights are uploaded once. Fitting runs the training rows through every
//! stage and keeps each column layer's inducing-point keys and values, each
//! ICL layer's keys and values (the 2 heads test rows use, in medium and
//! large), and the imputation means. Predicting uploads only the query rows
//! and downloads only the outputs. Every block chains the shared kernels of
//! `kernels::cuda::launch`: `RMSNorm`, products with their bias, GELU, and
//! residual epilogues, heads packed with `RoPE`, the per-head norm, and query
//! scaling, attention over cached keys and values, and heads back to tokens.
//! Kumo-Tabular's own kernels are the cell embedding's features and the row
//! stage's sequences.

#![expect(unsafe_code, reason = "Launches of kernels that trust their grid and shapes")]

use cuda_core::{DeviceBuffer, DeviceCopy};

use super::Config;
use super::cpu::Scaling;
use crate::kernels::cuda::launch::{
    Activation, Dense, HEAD_DIM, Output, Precision, config, n, padded_query_rows, padded_rows, threads,
};
use crate::kernels::cuda::{Device, gpu_error, out_of_memory};
use crate::{Error, Result};

type Buffer = DeviceBuffer<f32>;
type Bytes = DeviceBuffer<u8>;
/// What fitting records the column keys and values into, and the training
/// rows' label vectors.
type Recording<'a> = (&'a mut Vec<(Bytes, Bytes)>, &'a [f32]);

/// What an allocation that does not fit in device memory fails with; a fit
/// that hits it falls back to the CPU.
pub(crate) const OUT_OF_MEMORY: &str = "CUDA: out of device memory";
/// The block norms' epsilon, `torch.finfo(torch.float32).eps`.
const NORM_EPS: f32 = f32::EPSILON;
/// Cell features per column: 3 grouped columns of 32 sines and 32 cosines,
/// their 3 missing flags, and zeros to a multiple of 16.
const FEATURES: usize = 208;
/// Values a row-stage chunk's sequences hold at most.
const CHUNK_VALUES: usize = 1 << 25;

/// One transformer block's weights on the device.
struct Block {
    width: usize,
    heads: usize,
    query_norm: Buffer,
    key_value_norm: Buffer,
    query: Dense,
    key: Dense,
    value: Dense,
    out: Dense,
    mlp_norm: Buffer,
    mlp_in: Dense,
    mlp_out: Dense,
    head_scale: Vec<f32>,
    scaling: Scaling,
    /// The gate's two layers, the first's input zero-padded to 64.
    gate: Option<(Dense, Dense)>,
    /// `RoPE`'s frequencies, for the row blocks.
    rope: Option<Buffer>,
}

impl Block {
    fn dim(&self) -> usize {
        self.width / self.heads
    }
}

/// The model's weights on the device.
struct Weights {
    numeric: Buffer,
    categorical: Buffer,
    /// The cell embedding's projection for each pattern of the 3 grouped
    /// columns' kinds, bit `g` set when column `g` is categorical: weight
    /// `[cell, FEATURES]`, bias the 3 layers' biases summed.
    patterns: Vec<Dense>,
    readout: Buffer,
    inducing: Vec<Buffer>,
    columns: Vec<(Block, Block)>,
    rows: Vec<Block>,
    norm: Buffer,
    row_project: Option<Dense>,
    icl: Vec<Block>,
    icl_norm: Buffer,
    head: (Dense, Dense),
}

/// A member's fitted context on the device.
pub(crate) struct Context {
    means: Buffer,
    kinds: Buffer,
    categorical: Vec<bool>,
    training_rows: usize,
    /// Each column layer's output-block keys and values, `[columns * heads,
    /// inducing points, 64]`.
    columns: Vec<(Bytes, Bytes)>,
    /// Each ICL layer's keys and values for test rows, `[key heads, padded
    /// training rows, 64]`.
    icl: Vec<(Bytes, Bytes)>,
}

/// Kumo-Tabular's GPU: the device and the model's weights on it.
pub(crate) struct Cuda {
    device: Device,
    config: Config,
    outputs: usize,
    weights: Weights,
}

fn dim(value: u64) -> usize {
    usize::try_from(value).expect("model dimensions fit in usize")
}

fn device_error(error: cuda_core::DriverError) -> Error {
    if out_of_memory(error) {
        Error::UnsupportedBackend(OUT_OF_MEMORY.into())
    } else {
        gpu_error(error)
    }
}

impl Cuda {
    /// The GPU with the model's weights, or `None` without a GPU or with
    /// `IRONICL_NO_CUDA` set.
    pub(crate) fn new<'a>(
        config: Config,
        outputs: usize,
        tensor: &dyn Fn(&str) -> &'a [f32],
    ) -> Option<Self> {
        let device = Device::new().ok().flatten()?;
        let weights = Weights::upload(&device, &config, outputs, tensor).ok()?;
        Some(Self {
            device,
            config,
            outputs,
            weights,
        })
    }

    fn zeros(&self, len: usize) -> Result<Buffer> {
        self.device.zeroed(len).map_err(device_error)
    }

    fn upload<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::from_host(self.device.stream(), values).map_err(device_error)
    }

    fn bytes(buffer: Buffer) -> Result<Bytes> {
        buffer
            .cast_chunks::<u8>()
            .map_err(|_| gpu_error("keys and values do not cast to bytes"))
    }

    /// `act(x wᵀ + b) (+ residual)` for `rows` rows of `depth` values, in FP32.
    fn product(
        &self,
        x: &Buffer,
        dense: &Dense,
        residual: Option<&Buffer>,
        (rows, width, depth): (usize, usize, usize),
        gelu: bool,
    ) -> Result<Buffer> {
        let mut out = self.zeros(rows * width)?;
        // `product` splits `depth` so; its partial sums need room for every split.
        let blocks = rows.div_ceil(64) * width.div_ceil(64);
        let wanted = (2 * self.device.multiprocessors())
            .div_ceil(blocks)
            .min(depth / 64)
            .max(1);
        let splits = depth.div_ceil(depth.div_ceil(wanted).next_multiple_of(16));
        let mut partial = self.zeros(if splits > 1 { splits * rows * width } else { 0 })?;
        self.device.product(
            Activation::F32(x),
            dense,
            residual,
            (Some(&mut out), None),
            &mut partial,
            (rows, width, depth),
            gelu,
        )?;
        Ok(out)
    }

    fn rms(&self, x: &Buffer, gamma: &Buffer, rows: usize) -> Result<Buffer> {
        let width = gamma.len();
        let mut out = self.zeros(rows * width)?;
        self.device
            .rms_norm(x, gamma, Output::F32(&mut out), (rows, width), NORM_EPS)?;
        Ok(out)
    }

    /// `x` `[sequences * len, heads * dim]` packed into `[sequences * heads,
    /// padded, 64]`, with `RoPE` and the per-head norm when asked, times `base`.
    fn pack(
        &self,
        x: &Buffer,
        (base, rope): (Option<&Buffer>, Option<&Buffer>),
        (sequences, len, padded): (usize, usize, usize),
        (heads, dim): (usize, usize),
        norm: bool,
    ) -> Result<Buffer> {
        let mut out = self.zeros(sequences * heads * padded * HEAD_DIM)?;
        let none = self.zeros(0)?;
        self.device.pack_heads(
            (x, &none),
            (base.unwrap_or(&none), &none, rope.unwrap_or(&none)),
            Output::F32(&mut out),
            (sequences, len, padded),
            (heads, dim),
            norm,
        )?;
        Ok(out)
    }

    /// A block's keys and values for `groups` groups of `rows` rows of `x`,
    /// each `[groups * heads, padded rows, 64]`.
    fn key_value(
        &self,
        block: &Block,
        x: &Buffer,
        (groups, rows): (usize, usize),
    ) -> Result<(Buffer, Buffer)> {
        let (width, heads, dim) = (block.width, block.heads, block.dim());
        let normed = self.rms(x, &block.key_value_norm, groups * rows)?;
        let shape = (groups * rows, width, width);
        let keys = self.product(&normed, &block.key, None, shape, false)?;
        let values = self.product(&normed, &block.value, None, shape, false)?;
        let padded = padded_rows(rows);
        let keys = self.pack(
            &keys,
            (None, block.rope.as_ref()),
            (groups, rows, padded),
            (heads, dim),
            true,
        )?;
        let values = self.pack(&values, (None, None), (groups, rows, padded), (heads, dim), false)?;
        Ok((keys, values))
    }

    /// A block on `groups` groups of `rows` query rows, each attending to its
    /// group's `keys` and `values` (`key_heads` heads of `key_rows` rows).
    fn forward(
        &self,
        block: &Block,
        query: &Buffer,
        (groups, rows): (usize, usize),
        (keys, values, key_heads, key_rows): (&Bytes, &Bytes, usize, usize),
    ) -> Result<Buffer> {
        let (width, heads, dim) = (block.width, block.heads, block.dim());
        let (pairs, padded, tokens) = (groups * heads, padded_query_rows(rows), groups * rows);
        let normed = self.rms(query, &block.query_norm, tokens)?;
        let projected = self.product(&normed, &block.query, None, (tokens, width, width), false)?;
        // The attention divides by sqrt(64): queries of narrower heads make up for it.
        #[expect(clippy::cast_precision_loss, reason = "Head widths and key counts are small")]
        let (factor, log) = (
            (HEAD_DIM as f32 / dim as f32).sqrt(),
            (key_rows.max(1) as f32).ln(),
        );
        let base: Vec<f32> = match block.scaling {
            Scaling::None => vec![factor; heads * dim],
            Scaling::Log | Scaling::Gated => block
                .head_scale
                .iter()
                .flat_map(|a| std::iter::repeat_n(a * log * factor, dim))
                .collect(),
        };
        let base = self.upload(&base)?;
        let rope = block.rope.as_ref();
        let packed = if let Some((first, second)) = &block.gate {
            // The gate reads each head's query before scaling.
            let mut packed = self.pack(
                &projected,
                (None, rope),
                (groups, rows, padded),
                (heads, dim),
                true,
            )?;
            let hidden = self.product(&packed, first, None, (pairs * padded, 64, HEAD_DIM), true)?;
            let gates = self.product(&hidden, second, None, (pairs * padded, dim, 64), false)?;
            // SAFETY: one thread per value of `packed` `[pairs, padded, 64]`; `base` holds `[heads,
            // dim]` and `gates` `[pairs * padded, dim]`.
            let launch = unsafe {
                self.device.kernels.scale_heads(
                    self.device.stream(),
                    config(threads(pairs * padded * HEAD_DIM), 256),
                    &mut packed,
                    &base,
                    &gates,
                    n(pairs),
                    n(padded),
                    n(heads),
                    n(dim),
                    1,
                )
            };
            launch.map_err(gpu_error)?;
            packed
        } else {
            self.pack(
                &projected,
                (Some(&base), rope),
                (groups, rows, padded),
                (heads, dim),
                true,
            )?
        };
        let (_, splits) = self.device.icl_split(Precision::Fp32, pairs, padded, key_rows);
        let (mut part, mut lse) = (
            self.zeros(pairs * padded * splits * HEAD_DIM)?,
            self.zeros(pairs * padded * splits)?,
        );
        let mut attended = self.zeros(pairs * padded * HEAD_DIM)?;
        self.device.attend_icl(
            Precision::Fp32,
            Activation::F32(&packed),
            (keys, values),
            (&mut part, &mut lse, &mut attended),
            (pairs, padded, key_rows),
            heads / key_heads,
        )?;
        let ones = self.upload(&vec![1.0_f32; pairs * HEAD_DIM])?;
        let mut merged = self.zeros(tokens * width)?;
        self.device.heads_to_tokens(
            (&attended, &ones),
            Output::F32(&mut merged),
            (groups, rows, padded),
            (heads, dim),
        )?;
        let middle = self.product(&merged, &block.out, Some(query), (tokens, width, width), false)?;
        let normed = self.rms(&middle, &block.mlp_norm, tokens)?;
        let hidden = self.product(&normed, &block.mlp_in, None, (tokens, 2 * width, width), true)?;
        self.product(
            &hidden,
            &block.mlp_out,
            Some(&middle),
            (tokens, width, 2 * width),
            false,
        )
    }

    /// Fit a member's context on its `[rows, columns]` training rows `x` and
    /// their label vectors for the cells and for the ICL stage, `[rows, cell]`
    /// and `[rows, icl]`.
    pub(crate) fn fit(&self, x: &[f32], categorical: &[bool], labels: (&[f32], &[f32])) -> Result<Context> {
        let columns = categorical.len();
        let rows = x.len() / columns;
        let means: Vec<f32> = (0..columns)
            .map(|column| {
                let (sum, count) = x
                    .iter()
                    .skip(column)
                    .step_by(columns)
                    .filter(|v| !v.is_nan())
                    .fold((0.0, 0_u32), |(s, c), v| (s + v, c + 1));
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Row counts are exact enough in f32, as in PyTorch"
                )]
                if count == 0 { 0.0 } else { sum / count as f32 }
            })
            .collect();
        let kinds: Vec<f32> = categorical.iter().map(|&c| f32::from(u8::from(c))).collect();
        let mut context = Context {
            means: self.upload(&means)?,
            kinds: self.upload(&kinds)?,
            categorical: categorical.to_vec(),
            training_rows: rows,
            columns: Vec::new(),
            icl: Vec::new(),
        };
        let mut recorded = Vec::new();
        let mut hidden = self.row_embedding(x, &context, Some((&mut recorded, labels.0)))?;
        context.columns = recorded;
        let icl_width = dim(self.config.icl_channels);
        let label_rows = self.upload(labels.1)?;
        self.add_rows(&mut hidden, &label_rows, (rows, icl_width))?;
        let key_heads = self
            .config
            .query_key_value_heads
            .map_or(dim(self.config.icl_heads), dim);
        for (layer, block) in self.weights.icl.iter().enumerate() {
            let (keys, values) = self.key_value(block, &hidden, (1, rows))?;
            let prefix = key_heads * padded_rows(rows) * HEAD_DIM;
            let (mut kept_keys, mut kept_values) = (self.zeros(prefix)?, self.zeros(prefix)?);
            self.device.copy_into(&keys, &mut kept_keys, (0, prefix))?;
            self.device.copy_into(&values, &mut kept_values, (0, prefix))?;
            if layer + 1 < self.weights.icl.len() {
                let (keys, values) = (Self::bytes(keys)?, Self::bytes(values)?);
                hidden = self.forward(block, &hidden, (1, rows), (&keys, &values, block.heads, rows))?;
            }
            context
                .icl
                .push((Self::bytes(kept_keys)?, Self::bytes(kept_values)?));
        }
        Ok(context)
    }

    /// The outputs of each of the `[rows, columns]` query rows `x`.
    pub(crate) fn predict(&self, context: &Context, x: &[f32]) -> Result<Vec<f32>> {
        let rows = x.len() / context.categorical.len();
        let mut hidden = self.row_embedding(x, context, None)?;
        let key_heads = self
            .config
            .query_key_value_heads
            .map_or(dim(self.config.icl_heads), dim);
        for (block, (keys, values)) in self.weights.icl.iter().zip(&context.icl) {
            hidden = self.forward(
                block,
                &hidden,
                (1, rows),
                (keys, values, key_heads, context.training_rows),
            )?;
        }
        let icl = dim(self.config.icl_channels);
        let normed = self.rms(&hidden, &self.weights.icl_norm, rows)?;
        let wide = self.product(&normed, &self.weights.head.0, None, (rows, 2 * icl, icl), true)?;
        let out = self.product(
            &wide,
            &self.weights.head.1,
            None,
            (rows, self.outputs, 2 * icl),
            false,
        )?;
        self.device.download(&out)
    }

    fn add_rows(&self, x: &mut Buffer, vectors: &Buffer, (rows, width): (usize, usize)) -> Result<()> {
        let len = x.len();
        // SAFETY: one thread per value of `x`, which holds groups of `[rows, width]`; `vectors`
        // holds `[rows, width]`.
        let launch = unsafe {
            self.device.kernels.add_rows(
                self.device.stream(),
                config(threads(len), 256),
                x,
                vectors,
                n(rows),
                n(width),
                n(len),
            )
        };
        launch.map_err(gpu_error)
    }

    /// The row embedding of the `[rows, columns]` rows `x`, `[rows, icl]`:
    /// with `training`, the training rows, with their `[rows, cell]` label
    /// vectors, whose column keys and values it records; else query rows,
    /// which read them from `context`.
    fn row_embedding(
        &self,
        x: &[f32],
        context: &Context,
        mut training: Option<Recording<'_>>,
    ) -> Result<Buffer> {
        let config = self.config;
        let columns = context.categorical.len();
        let rows = x.len() / columns;
        let (cell, readout) = (dim(config.cell_channels), dim(config.readout_tokens));
        let mut cells = self.cells(x, context)?;
        if let Some((_, labels)) = &training {
            let labels = self.upload(labels)?;
            self.add_rows(&mut cells, &labels, (rows, cell))?;
        }
        let mut readouts = self.zeros(rows * readout * cell)?;
        self.device.repeat(
            &self.weights.readout,
            &mut readouts,
            (readout * cell, rows * readout * cell),
        )?;
        for (layer, (inducing, output)) in self.weights.columns.iter().enumerate() {
            if let Some((recorded, _)) = &mut training {
                let points = dim(config.inducing_points);
                let mut queries = self.zeros(columns * points * cell)?;
                self.device.repeat(
                    &self.weights.inducing[layer],
                    &mut queries,
                    (points * cell, columns * points * cell),
                )?;
                let (keys, values) = self.key_value(inducing, &cells, (columns, rows))?;
                let (keys, values) = (Self::bytes(keys)?, Self::bytes(values)?);
                let summaries = self.forward(
                    inducing,
                    &queries,
                    (columns, points),
                    (&keys, &values, inducing.heads, rows),
                )?;
                let (keys, values) = self.key_value(output, &summaries, (columns, points))?;
                recorded.push((Self::bytes(keys)?, Self::bytes(values)?));
            }
            let (keys, values) = match &training {
                Some((recorded, _)) => &recorded[layer],
                None => &context.columns[layer],
            };
            let points = dim(config.inducing_points);
            cells = self.forward(
                output,
                &cells,
                (columns, rows),
                (keys, values, output.heads, points),
            )?;
            self.row_stage(layer, &mut readouts, &mut cells, (rows, columns))?;
        }
        let normed = self.rms(&readouts, &self.weights.norm, rows * readout)?;
        match &self.weights.row_project {
            Some(project) => {
                let (width, icl) = (readout * cell, dim(config.icl_channels));
                self.product(&normed, project, None, (rows, icl, width), false)
            }
            None => Ok(normed),
        }
    }

    /// The cell embedding of the `[rows, columns]` rows `x`, `[columns, rows, cell]`.
    fn cells(&self, x: &[f32], context: &Context) -> Result<Buffer> {
        let columns = context.categorical.len();
        let rows = x.len() / columns;
        let cell = dim(self.config.cell_channels);
        let x = self.upload(x)?;
        let mut cells = self.zeros(columns * rows * cell)?;
        let mut features = self.zeros(rows * FEATURES)?;
        for column in 0..columns {
            // SAFETY: one thread per value of `features` `[rows, FEATURES]`; `x` holds `[rows,
            // columns]`, `means` and `kinds` `columns`, and the frequency tables `3 * 32`.
            let launch = unsafe {
                self.device.kernels.kumo_cell_features(
                    self.device.stream(),
                    config(threads(rows * FEATURES), 256),
                    &x,
                    &context.means,
                    &context.kinds,
                    &self.weights.numeric,
                    &self.weights.categorical,
                    &mut features,
                    n(rows),
                    n(columns),
                    n(column),
                )
            };
            launch.map_err(gpu_error)?;
            let pattern = (0..3).fold(0, |bits, g| {
                bits | usize::from(context.categorical[(column + (1 << g) - 1) % columns]) << g
            });
            let embedded = self.product(
                &features,
                &self.weights.patterns[pattern],
                None,
                (rows, cell, FEATURES),
                false,
            )?;
            self.device
                .copy_into(&embedded, &mut cells, (column * rows * cell, rows * cell))?;
        }
        Ok(cells)
    }

    /// Row stage `layer` on `readouts` `[rows, readout, cell]` and `cells`
    /// `[columns, rows, cell]`, a chunk of rows at a time.
    fn row_stage(
        &self,
        layer: usize,
        readouts: &mut Buffer,
        cells: &mut Buffer,
        (rows, columns): (usize, usize),
    ) -> Result<()> {
        let block = &self.weights.rows[layer];
        let (cell, readout) = (dim(self.config.cell_channels), dim(self.config.readout_tokens));
        let tokens = readout + columns;
        let last = layer + 1 == self.weights.rows.len();
        let step = (CHUNK_VALUES / (tokens * cell)).clamp(1, rows);
        for start in (0..rows).step_by(step) {
            let count = step.min(rows - start);
            let mut sequences = self.zeros(count * tokens * cell)?;
            self.sequences(
                readouts,
                cells,
                &mut sequences,
                (start, count, rows, tokens),
                false,
            )?;
            let (keys, values) = self.key_value(block, &sequences, (count, tokens))?;
            let (keys, values) = (Self::bytes(keys)?, Self::bytes(values)?);
            let kv = (&keys, &values, block.heads, tokens);
            if last {
                let mut queries = self.zeros(count * readout * cell)?;
                self.sequences(
                    readouts,
                    cells,
                    &mut queries,
                    (start, count, rows, readout),
                    false,
                )?;
                let mut out = self.forward(block, &queries, (count, readout), kv)?;
                self.sequences(readouts, cells, &mut out, (start, count, rows, readout), true)?;
            } else {
                let mut out = self.forward(block, &sequences, (count, tokens), kv)?;
                self.sequences(readouts, cells, &mut out, (start, count, rows, tokens), true)?;
            }
        }
        Ok(())
    }

    /// Rows `start..start + count` of the sequences of `tokens` tokens,
    /// gathered from `readouts` and `cells` into `sequences`, or scattered back.
    fn sequences(
        &self,
        readouts: &mut Buffer,
        cells: &mut Buffer,
        sequences: &mut Buffer,
        (start, count, rows, tokens): (usize, usize, usize, usize),
        scatter: bool,
    ) -> Result<()> {
        let (cell, readout) = (dim(self.config.cell_channels), dim(self.config.readout_tokens));
        assert!(start + count <= rows && sequences.len() >= count * tokens * cell);
        // SAFETY: one thread per value of `sequences` `[count, tokens, cell]`; `readouts` holds
        // `[rows, readout, cell]` and `cells` `[tokens - readout, rows, cell]` (`tokens == readout`
        // reads no cell), and `start + count <= rows` (asserted).
        let launch = unsafe {
            self.device.kernels.kumo_sequences(
                self.device.stream(),
                config(threads(count * tokens * cell), 256),
                readouts,
                cells,
                sequences,
                n(start),
                n(count),
                n(rows),
                n(readout),
                n(tokens),
                n(cell),
                u32::from(scatter),
            )
        };
        launch.map_err(gpu_error)
    }
}

impl Weights {
    /// Upload every tensor the forward pass reads, by name through `tensor`.
    fn upload<'a>(
        device: &Device,
        config: &Config,
        outputs: usize,
        tensor: &dyn Fn(&str) -> &'a [f32],
    ) -> Result<Self> {
        let up = |values: &[f32]| device.upload(values);
        let dense = |weight: &[f32], bias: &[f32]| Dense::upload_f32(device, weight, bias);
        let linear = |prefix: &str| {
            dense(
                tensor(&format!("{prefix}.weight")),
                tensor(&format!("{prefix}.bias")),
            )
        };
        let (cell, icl) = (dim(config.cell_channels), dim(config.icl_channels));
        let block = |prefix: &str, width: usize, heads: usize, scaling: Scaling, rope: bool| {
            Block::upload(device, tensor, prefix, (width, heads), (scaling, rope))
        };
        let embedding = "row_embedding.cell_embedding";
        let patterns = (0..8)
            .map(|pattern| pattern_dense(device, tensor, cell, pattern))
            .collect::<Result<_>>()?;
        let (embedding_heads, layers) = (dim(config.embedding_heads), dim(config.embedding_layers));
        let columns = (0..layers)
            .map(|layer| {
                let prefix = format!("row_embedding.col_blocks.{layer}");
                Ok((
                    block(
                        &format!("{prefix}.inducing_block"),
                        cell,
                        embedding_heads,
                        Scaling::Log,
                        false,
                    )?,
                    block(
                        &format!("{prefix}.output_block"),
                        cell,
                        embedding_heads,
                        Scaling::None,
                        false,
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
        let rows = (0..layers)
            .map(|layer| {
                block(
                    &format!("row_embedding.row_blocks.{layer}"),
                    cell,
                    embedding_heads,
                    Scaling::Gated,
                    true,
                )
            })
            .collect::<Result<_>>()?;
        let icl_blocks = (0..dim(config.icl_layers))
            .map(|layer| {
                block(
                    &format!("icl_block.layers.{layer}"),
                    icl,
                    dim(config.icl_heads),
                    Scaling::Log,
                    false,
                )
            })
            .collect::<Result<_>>()?;
        let readout = dim(config.readout_tokens);
        Ok(Self {
            numeric: up(tensor(&format!("{embedding}.num_freq")))?,
            categorical: up(tensor(&format!("{embedding}.cat_freq")))?,
            patterns,
            readout: up(tensor("row_embedding.readout_token"))?,
            inducing: (0..layers)
                .map(|layer| {
                    up(tensor(&format!(
                        "row_embedding.col_blocks.{layer}.inducing_points"
                    )))
                })
                .collect::<Result<_>>()?,
            columns,
            rows,
            norm: up(tensor("row_embedding.norm.weight"))?,
            row_project: if readout * cell == icl {
                None
            } else {
                Some(linear("row_project")?)
            },
            icl: icl_blocks,
            icl_norm: up(tensor("icl_block.norm.weight"))?,
            head: (linear("icl_block.head.0")?, {
                let (w, b) = (tensor("icl_block.head.2.weight"), tensor("icl_block.head.2.bias"));
                debug_assert_eq!(b.len(), outputs);
                dense(w, b)?
            }),
        })
    }
}

/// The cell embedding's projection for kinds `pattern` of the 3 grouped
/// columns, bit `g` set when column `g` is categorical: weight `[cell,
/// FEATURES]` laid out as [`Cuda::cells`]'s features, the bias the 3 layers'
/// biases summed.
fn pattern_dense<'a>(
    device: &Device,
    tensor: &dyn Fn(&str) -> &'a [f32],
    cell: usize,
    pattern: usize,
) -> Result<Dense> {
    let embedding = "row_embedding.cell_embedding";
    let kind = |g: usize| if pattern >> g & 1 == 1 { "cat" } else { "num" };
    let mut weight = vec![0.0; cell * FEATURES];
    let mut bias = vec![0.0; cell];
    let nan = tensor(&format!("{embedding}.nan_lin.weight"));
    for g in 0..3 {
        let w = tensor(&format!("{embedding}.{}_lin.weight", kind(g)));
        for channel in 0..cell {
            weight[channel * FEATURES + g * 64..][..64].copy_from_slice(&w[channel * 64..][..64]);
            weight[channel * FEATURES + 192 + g] = nan[channel * 3 + g];
        }
        for (b, value) in bias
            .iter_mut()
            .zip(tensor(&format!("{embedding}.{}_lin.bias", kind(g))))
        {
            *b += value;
        }
    }
    Dense::upload_f32(device, &weight, &bias)
}

impl Block {
    /// Upload block `prefix` of `width` with `heads` heads, its query scaling,
    /// and whether it has `RoPE`.
    fn upload<'a>(
        device: &Device,
        tensor: &dyn Fn(&str) -> &'a [f32],
        prefix: &str,
        (width, heads): (usize, usize),
        (scaling, rope): (Scaling, bool),
    ) -> Result<Self> {
        let up = |name: &str| device.upload(tensor(&format!("{prefix}.{name}")));
        let linear = |name: &str| {
            Dense::upload_f32(
                device,
                tensor(&format!("{prefix}.{name}.weight")),
                tensor(&format!("{prefix}.{name}.bias")),
            )
        };
        let (qkv_weight, qkv_bias) = (
            tensor(&format!("{prefix}.attn.qkv_lin.weight")),
            tensor(&format!("{prefix}.attn.qkv_lin.bias")),
        );
        let part = |p: usize| {
            Dense::upload_f32(
                device,
                &qkv_weight[p * width * width..][..width * width],
                &qkv_bias[p * width..][..width],
            )
        };
        let dim = width / heads;
        let gate = if scaling == Scaling::Gated {
            // The first layer's input is padded to the packed heads' 64 dims.
            let first = tensor(&format!("{prefix}.attn.sdpa.query_scaling.gate.0.weight"));
            let padded: Vec<f32> = first
                .chunks(dim)
                .flat_map(|row| {
                    row.iter()
                        .copied()
                        .chain(std::iter::repeat_n(0.0, HEAD_DIM - dim))
                })
                .collect();
            let bias = tensor(&format!("{prefix}.attn.sdpa.query_scaling.gate.0.bias"));
            Some((
                Dense::upload_f32(device, &padded, bias)?,
                linear("attn.sdpa.query_scaling.gate.2")?,
            ))
        } else {
            None
        };
        Ok(Self {
            width,
            heads,
            query_norm: up("query_norm.weight")?,
            key_value_norm: up("key_value_norm.weight")?,
            query: part(0)?,
            key: part(1)?,
            value: part(2)?,
            out: linear("attn.out_lin")?,
            mlp_norm: up("mlp.0.weight")?,
            mlp_in: linear("mlp.1")?,
            mlp_out: linear("mlp.3")?,
            head_scale: if scaling == Scaling::None {
                Vec::new()
            } else {
                tensor(&format!("{prefix}.attn.sdpa.query_scaling.head_scale")).to_vec()
            },
            scaling,
            gate,
            rope: if rope {
                Some(up("attn.query_transform.0.inv_freq")?)
            } else {
                None
            },
        })
    }
}

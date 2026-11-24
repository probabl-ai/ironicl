//! Kumo-Tabular's forward pass on the CPU, in FP32: fitting records a
//! member's context, the keys and values that query rows attend to, and
//! predicting runs only the query rows against it.
//!
//! Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/kumo/tabular>:
//! the Fourier cell embedding, then per layer a column stage (inducing points
//! summarize each column's training cells, and every cell attends to them)
//! and a row stage (each row's readout tokens and cells attend to each other
//! with `RoPE`), then the ICL stage, where training rows attend to each other
//! and query rows attend to the training rows.

use rayon::prelude::*;

use super::{Config, Release};
use crate::kernels::reference::{
    OnlineAttentionScratch, affine_f32, gelu_f32, rope_non_interleaved_f32,
    tiled_online_scaled_dot_product_attention_f32,
};
use crate::{Error, ModelArtifact, Result, TaskKind, format::model::ModelProfile};

/// `torch.finfo(torch.float32).eps`, the block norms' epsilon.
const NORM_EPS: f32 = f32::EPSILON;
/// The per-head query and key norms' epsilon.
const HEAD_NORM_EPS: f32 = 1e-6;
/// Rows a stage runs at once, which bounds its buffers.
const CHUNK_ROWS: usize = 256;
/// Query rows per attention job.
const ATTENTION_ROWS: usize = 64;

/// A loaded Kumo-Tabular checkpoint.
pub struct Model {
    pub release: Release,
    config: Config,
    artifact: ModelArtifact,
    /// The GPU that runs the transformer blocks, when there is one.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    metal: Option<super::metal::Metal>,
}

/// Keys and values for `groups` independent attentions, each over `rows`
/// rows, stored `[groups, heads, rows, head_dim]`.
#[derive(Clone, Debug, PartialEq)]
struct Kv {
    key: Vec<f32>,
    value: Vec<f32>,
    groups: usize,
    heads: usize,
    rows: usize,
}

impl Kv {
    /// The first `heads` heads of each group.
    fn first_heads(mut self, heads: usize, head_dim: usize) -> Self {
        if heads < self.heads {
            let (old, new) = (self.heads * self.rows * head_dim, heads * self.rows * head_dim);
            for tensor in [&mut self.key, &mut self.value] {
                *tensor = tensor
                    .chunks(old)
                    .flat_map(|group| &group[..new])
                    .copied()
                    .collect();
            }
            self.heads = heads;
        }
        self
    }
}

/// A member's fitted context: the column imputation means, each layer's
/// column keys and values (one group per column), and each ICL layer's keys
/// and values.
#[derive(Clone, Debug, PartialEq)]
pub struct Context {
    categorical: Vec<bool>,
    means: Vec<f32>,
    columns: Vec<Kv>,
    icl: Vec<Kv>,
}

/// How a block scales its queries by the key count `n`: not, by
/// `head_scale · ln n`, or that gated by `1 + tanh(gate(q))`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Scaling {
    None,
    Log,
    Gated,
}

/// One `KumoTabularTransformerBlock`.
struct Block<'a> {
    model: &'a Model,
    name: String,
    width: usize,
    heads: usize,
    rope: bool,
    scaling: Scaling,
}

fn invalid(message: &str) -> Error {
    Error::InvalidArgument(message.into())
}

/// `[rows, width]` FP32 rows normalized by their root mean square, then
/// scaled by `weight` when given.
fn rms_norm(x: &[f32], width: usize, weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    #[expect(clippy::cast_precision_loss, reason = "Widths are small")]
    out.par_chunks_mut(width).with_min_len(64).for_each(|row| {
        let scale = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + eps)
            .sqrt()
            .recip();
        for (index, value) in row.iter_mut().enumerate() {
            *value *= scale;
            if let Some(weight) = weight {
                *value *= weight[index];
            }
        }
    });
    out
}

/// A `Linear` layer on `[rows, input]` rows: one Accelerate `sgemm` on Apple
/// Silicon, whose matrix unit the cores share, else one chunk per thread.
fn linear(x: &[f32], input: usize, weight: &[f32], bias: Option<&[f32]>, output: usize) -> Result<Vec<f32>> {
    let rows = x.len() / input;
    let mut out = vec![0.0; rows * output];
    if crate::kernels::cpu::affine_uses_sgemm(rows) {
        affine_f32(x, rows, input, weight, bias, output, &mut out)?;
        return Ok(out);
    }
    let chunk = rows.div_ceil(rayon::current_num_threads()).max(16);
    out.par_chunks_mut(chunk * output)
        .zip(x.par_chunks(chunk * input))
        .try_for_each(|(out, x)| affine_f32(x, x.len() / input, input, weight, bias, output, out))?;
    Ok(out)
}

fn gelu(x: &[f32]) -> Result<Vec<f32>> {
    let mut out = vec![0.0; x.len()];
    out.par_chunks_mut(16_384)
        .zip(x.par_chunks(16_384))
        .try_for_each(|(out, x)| gelu_f32(x, out))?;
    Ok(out)
}

/// `[groups * rows, heads * head_dim]` as `[groups, heads, rows, head_dim]`.
fn to_heads(x: &[f32], groups: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let rows = x.len() / (groups * heads * head_dim);
    let mut out = vec![0.0; x.len()];
    out.par_chunks_mut(heads * rows * head_dim)
        .enumerate()
        .for_each(|(group, out)| {
            for head in 0..heads {
                for row in 0..rows {
                    let source = ((group * rows + row) * heads + head) * head_dim;
                    out[(head * rows + row) * head_dim..][..head_dim]
                        .copy_from_slice(&x[source..][..head_dim]);
                }
            }
        });
    out
}

/// The inverse of [`to_heads`].
fn from_heads(x: &[f32], groups: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let rows = x.len() / (groups * heads * head_dim);
    let mut out = vec![0.0; x.len()];
    out.par_chunks_mut(rows * heads * head_dim)
        .enumerate()
        .for_each(|(group, out)| {
            for head in 0..heads {
                for row in 0..rows {
                    let source = ((group * heads + head) * rows + row) * head_dim;
                    out[(row * heads + head) * head_dim..][..head_dim]
                        .copy_from_slice(&x[source..][..head_dim]);
                }
            }
        });
    out
}

impl Model {
    /// The model of a Kumo-Tabular checkpoint's artifact.
    ///
    /// # Errors
    ///
    /// When the artifact is another model.
    pub fn new(artifact: ModelArtifact) -> Result<Self> {
        let ModelProfile::KumoTabular(release) = artifact.summary().profile else {
            return Err(Error::Artifact(
                "the model is not a Kumo-Tabular checkpoint".into(),
            ));
        };
        Ok(Self {
            release,
            config: release.size.config(),
            artifact,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal: super::metal::Metal::new(),
        })
    }

    /// Where the transformer blocks run: `"metal"` or `"cpu"`.
    #[must_use]
    pub fn device(&self) -> &'static str {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.metal.is_some() {
            return "metal";
        }
        "cpu"
    }

    /// The checkpoint's SHA-256.
    pub(crate) fn sha256(&self) -> [u8; 32] {
        crate::models::tabicl_v2::checkpoint::decode_sha256(&self.artifact.summary().source_sha256)
            .expect("a verified model's SHA-256")
    }

    fn tensor(&self, name: &str) -> &[f32] {
        self.artifact
            .tensor_f32(name)
            .map(crate::format::model::ModelTensorF32::values)
            .expect("the manifest has every tensor")
    }

    fn dim(value: u64) -> usize {
        usize::try_from(value).expect("model dimensions fit in usize")
    }

    fn cell(&self) -> usize {
        Self::dim(self.config.cell_channels)
    }

    fn block(&self, name: String, width: u64, heads: u64, rope: bool, scaling: Scaling) -> Block<'_> {
        Block {
            model: self,
            name,
            width: Self::dim(width),
            heads: Self::dim(heads),
            rope,
            scaling,
        }
    }

    fn embedding_block(&self, name: String, rope: bool, scaling: Scaling) -> Block<'_> {
        self.block(
            name,
            self.config.cell_channels,
            self.config.embedding_heads,
            rope,
            scaling,
        )
    }

    fn icl_block(&self, layer: u64) -> Block<'_> {
        let config = &self.config;
        self.block(
            format!("icl_block.layers.{layer}"),
            config.icl_channels,
            config.icl_heads,
            false,
            Scaling::Log,
        )
    }

    /// Fit a member's context on its preprocessed `[rows, columns]` training
    /// rows `x`, whose `categorical` columns hold category codes, and their
    /// `targets`, and return it.
    ///
    /// # Errors
    ///
    /// When the shapes disagree, or the targets do not fit the model's task.
    pub fn fit(&self, x: &[f32], categorical: &[bool], targets: Targets<'_>) -> Result<Context> {
        let columns = categorical.len();
        let rows = match targets {
            Targets::Classes(codes) if self.release.task == TaskKind::Classifier => {
                if codes.iter().any(|&code| code >= 10) {
                    return Err(invalid("a class code is not below 10"));
                }
                codes.len()
            }
            Targets::Values(values) if self.release.task == TaskKind::Regressor => values.len(),
            _ => return Err(invalid("the targets do not fit the model's task")),
        };
        if columns == 0 || rows == 0 || x.len() != rows * columns {
            return Err(invalid("the training rows do not match their targets"));
        }
        let means = (0..columns)
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
        let mut context = Context {
            categorical: categorical.to_vec(),
            means,
            columns: Vec::new(),
            icl: Vec::new(),
        };
        let mut recorded = Vec::new();
        let mut rows = self.row_embedding(x, &context, Columns::Record(&mut recorded, targets))?;
        context.columns = recorded;
        self.add_labels(
            "icl_block",
            &mut rows,
            Self::dim(self.config.icl_channels),
            targets,
        );
        let query_heads = self
            .config
            .query_key_value_heads
            .map_or(Self::dim(self.config.icl_heads), Self::dim);
        for layer in 0..self.config.icl_layers {
            let block = self.icl_block(layer);
            let kv = block.key_value(&rows, 1)?;
            if layer + 1 < self.config.icl_layers {
                rows = block.forward(&rows, 1, &kv)?;
            }
            context.icl.push(kv.first_heads(query_heads, block.head_dim()));
        }
        Ok(context)
    }

    /// The outputs, 10 logits or 999 quantiles, of each of the `[rows, columns]`
    /// preprocessed query rows `x`.
    ///
    /// # Errors
    ///
    /// When `x` does not have the context's columns.
    pub fn predict(&self, context: &Context, x: &[f32]) -> Result<Vec<f32>> {
        let columns = context.categorical.len();
        if x.is_empty() || !x.len().is_multiple_of(columns) {
            return Err(invalid("the query rows do not have the context's columns"));
        }
        let mut out = Vec::new();
        for chunk in x.chunks(CHUNK_ROWS * columns) {
            let mut rows = self.row_embedding(chunk, context, Columns::Read(&context.columns))?;
            let rows_in_chunk = chunk.len() / columns;
            for layer in 0..self.config.icl_layers {
                let block = self.icl_block(layer);
                let kv = &context.icl[Self::dim(layer)];
                rows = block.forward_rows(&rows, rows_in_chunk, 1, kv)?;
            }
            out.extend(self.head(&rows)?);
        }
        Ok(out)
    }

    /// Add each training row's label embedding at `stage`, a class's `y_emb`
    /// row or the value times `y_lin`, to its `width` values in `rows`.
    fn add_labels(&self, stage: &str, rows: &mut [f32], width: usize, targets: Targets<'_>) {
        match targets {
            Targets::Classes(codes) => {
                let embedding = self.tensor(&format!("{stage}.y_emb.weight"));
                for (row, &code) in rows.chunks_mut(width).zip(codes) {
                    for (value, label) in row.iter_mut().zip(&embedding[code * width..][..width]) {
                        *value += label;
                    }
                }
            }
            Targets::Values(values) => {
                let weight = self.tensor(&format!("{stage}.y_lin.weight"));
                for (row, &target) in rows.chunks_mut(width).zip(values) {
                    for (value, w) in row.iter_mut().zip(weight) {
                        *value += w * target;
                    }
                }
            }
        }
    }

    /// `head(norm(x))`: the ICL stage's output layers.
    fn head(&self, x: &[f32]) -> Result<Vec<f32>> {
        let icl = Self::dim(self.config.icl_channels);
        let outputs = Self::dim(self.release.outputs());
        let x = rms_norm(x, icl, Some(self.tensor("icl_block.norm.weight")), NORM_EPS);
        let hidden = linear(
            &x,
            icl,
            self.tensor("icl_block.head.0.weight"),
            Some(self.tensor("icl_block.head.0.bias")),
            2 * icl,
        )?;
        linear(
            &gelu(&hidden)?,
            2 * icl,
            self.tensor("icl_block.head.2.weight"),
            Some(self.tensor("icl_block.head.2.bias")),
            outputs,
        )
    }

    /// Each of the `[rows, columns]` rows `x` as the ICL stage's input:
    /// `[rows, icl_channels]`.
    fn row_embedding(&self, x: &[f32], context: &Context, mut kvs: Columns<'_>) -> Result<Vec<f32>> {
        let targets = match &kvs {
            Columns::Record(_, targets) => Some(*targets),
            Columns::Read(_) => None,
        };
        let columns = context.categorical.len();
        let rows = x.len() / columns;
        let (cell, readout) = (self.cell(), Self::dim(self.config.readout_tokens));
        let mut cells = self.cell_embedding(x, context)?; // [columns, rows, cell]
        if let Some(targets) = targets {
            cells
                .par_chunks_mut(rows * cell)
                .for_each(|column| self.add_labels("row_embedding", column, cell, targets));
        }
        let readout_token = self.tensor("row_embedding.readout_token");
        let mut readouts: Vec<f32> = readout_token.repeat(rows); // [rows, readout, cell]
        let layers = self.config.embedding_layers;
        for layer in 0..layers {
            let column = format!("row_embedding.col_blocks.{layer}");
            let output = self.embedding_block(format!("{column}.output_block"), false, Scaling::None);
            if let Columns::Record(recorded, _) = &mut kvs {
                let inducing = self.embedding_block(format!("{column}.inducing_block"), false, Scaling::Log);
                let points = self.tensor(&format!("{column}.inducing_points"));
                let kv = inducing.key_value(&cells, columns)?;
                let summaries = inducing.forward(&points.repeat(columns), columns, &kv)?;
                recorded.push(output.key_value(&summaries, columns)?);
            }
            let kv = match &kvs {
                Columns::Record(recorded, _) => &recorded[Self::dim(layer)],
                Columns::Read(kvs) => &kvs[Self::dim(layer)],
            };
            cells = output.forward(&cells, columns, kv)?;
            self.row_stage(layer, &mut readouts, &mut cells, rows)?;
        }
        let embedded = rms_norm(
            &readouts,
            cell,
            Some(self.tensor("row_embedding.norm.weight")),
            NORM_EPS,
        );
        let width = readout * cell;
        let icl = Self::dim(self.config.icl_channels);
        if width == icl {
            Ok(embedded)
        } else {
            linear(
                &embedded,
                width,
                self.tensor("row_project.weight"),
                Some(self.tensor("row_project.bias")),
                icl,
            )
        }
    }

    /// The row stage of `layer`: each row's readouts and cells attend to each
    /// other, a chunk of rows at a time. On the last layer only the readouts
    /// are queries, and the cells stay as they are.
    fn row_stage(&self, layer: u64, readouts: &mut [f32], cells: &mut [f32], rows: usize) -> Result<()> {
        let (cell, readout) = (self.cell(), Self::dim(self.config.readout_tokens));
        let columns = cells.len() / (rows * cell);
        let tokens = readout + columns;
        let last = layer + 1 == self.config.embedding_layers;
        let block = self.embedding_block(format!("row_embedding.row_blocks.{layer}"), true, Scaling::Gated);
        for start in (0..rows).step_by(CHUNK_ROWS) {
            let count = CHUNK_ROWS.min(rows - start);
            let mut sequences = vec![0.0; count * tokens * cell];
            sequences
                .par_chunks_mut(tokens * cell)
                .enumerate()
                .for_each(|(index, sequence)| {
                    let row = start + index;
                    sequence[..readout * cell]
                        .copy_from_slice(&readouts[row * readout * cell..][..readout * cell]);
                    for column in 0..columns {
                        sequence[(readout + column) * cell..][..cell]
                            .copy_from_slice(&cells[(column * rows + row) * cell..][..cell]);
                    }
                });
            let kv = block.key_value(&sequences, count)?;
            let chunk_readouts = &mut readouts[start * readout * cell..][..count * readout * cell];
            if last {
                let queries: Vec<f32> = sequences
                    .chunks(tokens * cell)
                    .flat_map(|s| &s[..readout * cell])
                    .copied()
                    .collect();
                chunk_readouts.copy_from_slice(&block.forward(&queries, count, &kv)?);
                continue;
            }
            let out = block.forward(&sequences, count, &kv)?;
            for (index, sequence) in out.chunks(tokens * cell).enumerate() {
                chunk_readouts[index * readout * cell..][..readout * cell]
                    .copy_from_slice(&sequence[..readout * cell]);
                for column in 0..columns {
                    cells[(column * rows + start + index) * cell..][..cell]
                        .copy_from_slice(&sequence[(readout + column) * cell..][..cell]);
                }
            }
        }
        Ok(())
    }

    /// The Fourier cell embedding of `[rows, columns]` rows `x`, as
    /// `[columns, rows, cell]`. A missing value takes its column's training
    /// mean and adds its own term. Each cell embeds its column with the
    /// columns 1 and 3 after it, circularly.
    ///
    /// Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/tabfm/cell_embedding.py#L55-L250>.
    fn cell_embedding(&self, x: &[f32], context: &Context) -> Result<Vec<f32>> {
        let columns = context.categorical.len();
        let rows = x.len() / columns;
        let cell = self.cell();
        let group = Self::dim(self.config.group_size);
        let frequencies = Self::dim(self.config.frequencies);
        let name = "row_embedding.cell_embedding";
        let (numeric, categorical) = (
            self.tensor(&format!("{name}.num_freq")),
            self.tensor(&format!("{name}.cat_freq")),
        );
        let nan = self.tensor(&format!("{name}.nan_lin.weight"));
        let mut out = vec![0.0; columns * rows * cell];
        out.par_chunks_mut(rows * cell)
            .enumerate()
            .try_for_each(|(column, out)| {
                let sources: Vec<usize> = (0..group).map(|g| (column + (1 << g) - 1) % columns).collect();
                let kinds: Vec<&str> = sources
                    .iter()
                    .map(|&s| if context.categorical[s] { "cat" } else { "num" })
                    .collect();
                let width = 2 * frequencies;
                // The projection of each group member's features, side by side.
                let mut weight = vec![0.0; cell * group * width];
                let mut bias = vec![0.0; cell];
                for (g, kind) in kinds.iter().enumerate() {
                    let w = self.tensor(&format!("{name}.{kind}_lin.weight"));
                    for channel in 0..cell {
                        weight[(channel * group + g) * width..][..width]
                            .copy_from_slice(&w[channel * width..][..width]);
                    }
                    for (b, value) in bias
                        .iter_mut()
                        .zip(self.tensor(&format!("{name}.{kind}_lin.bias")))
                    {
                        *b += value;
                    }
                }
                let mut features = vec![0.0; rows * group * width];
                for (row, features) in features.chunks_mut(group * width).enumerate() {
                    for (g, &source) in sources.iter().enumerate() {
                        let value = x[row * columns + source];
                        let value = if value.is_nan() {
                            context.means[source]
                        } else {
                            value
                        };
                        let table = if context.categorical[source] {
                            categorical
                        } else {
                            numeric
                        };
                        for k in 0..frequencies {
                            let (sin, cos) = (value * table[g * frequencies + k]).sin_cos();
                            features[g * width + k] = sin;
                            features[g * width + frequencies + k] = cos;
                        }
                    }
                }
                affine_f32(&features, rows, group * width, &weight, Some(&bias), cell, out)?;
                for (row, out) in out.chunks_mut(cell).enumerate() {
                    for (g, &source) in sources.iter().enumerate() {
                        if x[row * columns + source].is_nan() {
                            for (channel, value) in out.iter_mut().enumerate() {
                                *value += nan[channel * group + g];
                            }
                        }
                    }
                }
                Ok::<_, Error>(())
            })?;
        Ok(out)
    }
}

/// Attention scores a GPU block computes at once, which bounds its memory.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
const GPU_SCORES: usize = 1 << 26;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
impl Block<'_> {
    fn key(&self, forward: bool, (groups, rows): (usize, usize), kv: Option<&Kv>) -> super::metal::Key {
        super::metal::Key {
            forward,
            width: self.width,
            heads: self.heads,
            groups,
            rows,
            kv_groups: kv.map_or(0, |kv| kv.groups),
            kv_heads: kv.map_or(0, |kv| kv.heads),
            kv_rows: kv.map_or(0, |kv| kv.rows),
            rope: self.rope,
            scaling: self.scaling,
        }
    }

    /// Rows `part` of the fused projection's weight, transposed, and its bias.
    fn projection(&self, part: usize) -> [super::metal::Feed<'_>; 2] {
        let name = format!("{}.attn.qkv_lin", self.name);
        let w = self.width;
        let weight = &self.tensor("attn.qkv_lin.weight")[part * w * w..][..w * w];
        [
            super::metal::Feed::Weight(
                format!("{name}.weight[{part}]^T"),
                Box::new(move || transpose(weight, w, w)),
            ),
            self.vector_part("attn.qkv_lin.bias", part * w, w),
        ]
    }

    /// A `[outputs, inputs]` weight, transposed, and its bias.
    fn linear_feeds(&self, prefix: &str, inputs: usize, outputs: usize) -> [super::metal::Feed<'_>; 2] {
        let weight = self.tensor(&format!("{prefix}.weight"));
        [
            super::metal::Feed::Weight(
                format!("{}.{prefix}.weight^T", self.name),
                Box::new(move || transpose(weight, outputs, inputs)),
            ),
            self.vector_part(&format!("{prefix}.bias"), 0, outputs),
        ]
    }

    fn vector_part(&self, suffix: &str, start: usize, length: usize) -> super::metal::Feed<'_> {
        let values = &self.tensor(suffix)[start..][..length];
        super::metal::Feed::Weight(
            format!("{}.{suffix}[{start}]", self.name),
            Box::new(move || values.to_vec()),
        )
    }

    /// `RoPE`'s cosines and signed sines for `rows` positions, as
    /// `Builder::rope` takes them.
    #[expect(clippy::cast_precision_loss, reason = "Positions are small")]
    fn rope_tables(&self, rows: usize, key: bool) -> [Vec<f32>; 2] {
        let name = if key { "key_transform" } else { "query_transform" };
        let frequencies = self.tensor(&format!("attn.{name}.0.inv_freq"));
        let (head_dim, half) = (self.head_dim(), self.head_dim() / 2);
        let (mut cos, mut sin) = (vec![0.0; rows * head_dim], vec![0.0; rows * head_dim]);
        for row in 0..rows {
            for (index, frequency) in frequencies.iter().enumerate() {
                let (s, c) = (row as f32 * frequency).sin_cos();
                let at = row * head_dim + index;
                (cos[at], cos[at + half]) = (c, c);
                (sin[at], sin[at + half]) = (-s, s);
            }
        }
        [cos, sin]
    }

    fn key_value_metal(&self, metal: &super::metal::Metal, x: &[f32], groups: usize) -> Result<Kv> {
        use super::metal::Feed;
        let rows = x.len() / (groups * self.width);
        let [k_weight, k_bias] = self.projection(1);
        let [v_weight, v_bias] = self.projection(2);
        let tables = self.rope.then(|| self.rope_tables(rows, true));
        let mut feeds = vec![
            Feed::Data(x),
            self.vector_part("key_value_norm.weight", 0, self.width),
        ];
        feeds.extend([k_weight, k_bias, v_weight, v_bias]);
        if let Some([cos, sin]) = &tables {
            feeds.extend([Feed::Data(cos), Feed::Data(sin)]);
        }
        let mut outputs = metal.run(self.key(false, (groups, rows), None), feeds)?;
        let value = outputs.pop().expect("two outputs");
        let key = outputs.pop().expect("two outputs");
        Ok(Kv {
            key,
            value,
            groups,
            heads: self.heads,
            rows,
        })
    }

    /// [`Self::forward_rows`] on the GPU, in chunks of groups, or of rows for
    /// one group, whose attention scores fit [`GPU_SCORES`].
    fn forward_metal(
        &self,
        metal: &super::metal::Metal,
        query: &[f32],
        rows: usize,
        groups: usize,
        kv: &Kv,
    ) -> Result<Vec<f32>> {
        use super::metal::Feed;
        let (width, head_dim) = (self.width, self.head_dim());
        let per_row = self.heads * kv.rows;
        let kv_group = kv.heads * kv.rows * head_dim;
        let (group_step, row_step) = if groups == 1 || kv.groups == 1 {
            (groups, (GPU_SCORES / (groups * per_row)).clamp(1, rows))
        } else {
            ((GPU_SCORES / (rows * per_row)).clamp(1, groups), rows)
        };
        #[expect(clippy::cast_precision_loss, reason = "Key counts are exact enough in f32")]
        let log = (kv.rows.max(1) as f32).ln();
        let scale: Vec<f32> = if self.scaling == Scaling::None {
            Vec::new()
        } else {
            self.tensor("attn.sdpa.query_scaling.head_scale")
                .iter()
                .map(|a| a * log)
                .collect()
        };
        let mut out = vec![0.0; query.len()];
        for group in (0..groups).step_by(group_step) {
            let count = group_step.min(groups - group);
            for row in (0..rows).step_by(row_step) {
                let length = row_step.min(rows - row);
                let chunk: Vec<f32> = (group..group + count)
                    .flat_map(|g| &query[(g * rows + row) * width..][..length * width])
                    .copied()
                    .collect();
                let (kv_start, kv_count) = if kv.groups == 1 { (0, 1) } else { (group, count) };
                let keys = &kv.key[kv_start * kv_group..][..kv_count * kv_group];
                let values = &kv.value[kv_start * kv_group..][..kv_count * kv_group];
                let tables = if self.rope {
                    let [cos, sin] = self.rope_tables(row + length, false);
                    Some([cos[row * head_dim..].to_vec(), sin[row * head_dim..].to_vec()])
                } else {
                    None
                };
                let [q_weight, q_bias] = self.projection(0);
                let mut feeds = vec![
                    Feed::Data(&chunk),
                    Feed::Data(keys),
                    Feed::Data(values),
                    self.vector_part("query_norm.weight", 0, width),
                    q_weight,
                    q_bias,
                ];
                if let Some([cos, sin]) = &tables {
                    feeds.extend([Feed::Data(cos), Feed::Data(sin)]);
                }
                if self.scaling != Scaling::None {
                    feeds.push(Feed::Data(&scale));
                }
                if self.scaling == Scaling::Gated {
                    feeds.extend(self.linear_feeds("attn.sdpa.query_scaling.gate.0", head_dim, 64));
                    feeds.extend(self.linear_feeds("attn.sdpa.query_scaling.gate.2", 64, head_dim));
                }
                feeds.extend(self.linear_feeds("attn.out_lin", width, width));
                feeds.push(self.vector_part("mlp.0.weight", 0, width));
                feeds.extend(self.linear_feeds("mlp.1", width, 2 * width));
                feeds.extend(self.linear_feeds("mlp.3", 2 * width, width));
                let kv_shape = Kv {
                    key: Vec::new(),
                    value: Vec::new(),
                    groups: kv_count,
                    heads: kv.heads,
                    rows: kv.rows,
                };
                let result = metal
                    .run(self.key(true, (count, length), Some(&kv_shape)), feeds)?
                    .remove(0);
                for (index, g) in (group..group + count).enumerate() {
                    out[(g * rows + row) * width..][..length * width]
                        .copy_from_slice(&result[index * length * width..][..length * width]);
                }
            }
        }
        Ok(out)
    }
}

/// A `[rows, columns]` matrix transposed.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = values[row * columns + column];
        }
    }
    out
}

/// The column stage's keys and values: recorded from the training rows and
/// their targets, or read from a context.
enum Columns<'a> {
    Record(&'a mut Vec<Kv>, Targets<'a>),
    Read(&'a [Kv]),
}

/// A member's training targets: a classifier's class codes, below 10, or a
/// regressor's standardized values.
#[derive(Clone, Copy, Debug)]
pub enum Targets<'a> {
    Classes(&'a [usize]),
    Values(&'a [f32]),
}

impl Block<'_> {
    fn head_dim(&self) -> usize {
        self.width / self.heads
    }

    fn tensor(&self, suffix: &str) -> &[f32] {
        self.model.tensor(&format!("{}.{suffix}", self.name))
    }

    /// The attention's projection rows `part` (0 queries, 1 keys, 2 values) of `x`.
    fn project(&self, x: &[f32], part: usize) -> Result<Vec<f32>> {
        let w = self.width;
        let weight = &self.tensor("attn.qkv_lin.weight")[part * w * w..][..w * w];
        let bias = &self.tensor("attn.qkv_lin.bias")[part * w..][..w];
        linear(x, w, weight, Some(bias), w)
    }

    /// `RoPE` when this block has it, then the per-head norm, of
    /// `[groups, heads, rows, head_dim]` queries or keys.
    fn transform(&self, x: Vec<f32>, groups: usize, key: bool) -> Result<Vec<f32>> {
        let head_dim = self.head_dim();
        let x = if self.rope {
            let rows = x.len() / (groups * self.heads * head_dim);
            let name = if key { "key_transform" } else { "query_transform" };
            let frequencies = self.tensor(&format!("attn.{name}.0.inv_freq"));
            let mut out = vec![0.0; x.len()];
            out.par_chunks_mut(self.heads * rows * head_dim)
                .zip(x.par_chunks(self.heads * rows * head_dim))
                .try_for_each(|(out, x)| {
                    rope_non_interleaved_f32(x, self.heads, rows, head_dim, frequencies, 0, out)
                })?;
            out
        } else {
            x
        };
        Ok(rms_norm(&x, head_dim, None, HEAD_NORM_EPS))
    }

    /// Keys and values of `groups` equal parts of the `[rows, width]` rows `x`.
    fn key_value(&self, x: &[f32], groups: usize) -> Result<Kv> {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(metal) = &self.model.metal {
            return self.key_value_metal(metal, x, groups);
        }
        let normed = rms_norm(
            x,
            self.width,
            Some(self.tensor("key_value_norm.weight")),
            NORM_EPS,
        );
        let head_dim = self.head_dim();
        let key = to_heads(&self.project(&normed, 1)?, groups, self.heads, head_dim);
        let value = to_heads(&self.project(&normed, 2)?, groups, self.heads, head_dim);
        Ok(Kv {
            key: self.transform(key, groups, true)?,
            value,
            groups,
            heads: self.heads,
            rows: x.len() / (groups * self.width),
        })
    }

    /// The block on `groups` equal parts of the `[rows, width]` queries,
    /// each attending to its group of `kv`.
    fn forward(&self, query: &[f32], groups: usize, kv: &Kv) -> Result<Vec<f32>> {
        self.forward_rows(query, query.len() / (groups * self.width), groups, kv)
    }

    /// [`Self::forward`] for `rows` query rows per group; `kv` may have one
    /// group for all.
    fn forward_rows(&self, query: &[f32], rows: usize, groups: usize, kv: &Kv) -> Result<Vec<f32>> {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(metal) = &self.model.metal {
            return self.forward_metal(metal, query, rows, groups, kv);
        }
        let (width, heads, head_dim) = (self.width, self.heads, self.head_dim());
        let normed = rms_norm(query, width, Some(self.tensor("query_norm.weight")), NORM_EPS);
        let mut q = self.transform(
            to_heads(&self.project(&normed, 0)?, groups, heads, head_dim),
            groups,
            false,
        )?;
        if self.scaling != Scaling::None {
            #[expect(clippy::cast_precision_loss, reason = "Key counts are exact enough in f32")]
            let log = (kv.rows.max(1) as f32).ln();
            let head_scale = self.tensor("attn.sdpa.query_scaling.head_scale");
            let gates = if self.scaling == Scaling::Gated {
                Some(self.gates(&q)?)
            } else {
                None
            };
            q.par_chunks_mut(rows * head_dim)
                .enumerate()
                .for_each(|(index, values)| {
                    let scale = log * head_scale[index % heads];
                    for (offset, value) in values.iter_mut().enumerate() {
                        *value *= scale;
                        if let Some(gates) = &gates {
                            *value *= 1.0 + gates[index * rows * head_dim + offset].tanh();
                        }
                    }
                });
        }
        let attended = from_heads(&self.attend(&q, rows, groups, kv)?, groups, heads, head_dim);
        let out = linear(
            &attended,
            width,
            self.tensor("attn.out_lin.weight"),
            Some(self.tensor("attn.out_lin.bias")),
            width,
        )?;
        let residual: Vec<f32> = query.iter().zip(&out).map(|(q, o)| q + o).collect();
        let normed = rms_norm(&residual, width, Some(self.tensor("mlp.0.weight")), NORM_EPS);
        let hidden = gelu(&linear(
            &normed,
            width,
            self.tensor("mlp.1.weight"),
            Some(self.tensor("mlp.1.bias")),
            2 * width,
        )?)?;
        let out = linear(
            &hidden,
            2 * width,
            self.tensor("mlp.3.weight"),
            Some(self.tensor("mlp.3.bias")),
            width,
        )?;
        Ok(residual.iter().zip(&out).map(|(r, o)| r + o).collect())
    }

    /// `GatedLogScale`'s gate `Linear → GELU → Linear` of each head's query.
    fn gates(&self, q: &[f32]) -> Result<Vec<f32>> {
        let head_dim = self.head_dim();
        let hidden = linear(
            q,
            head_dim,
            self.tensor("attn.sdpa.query_scaling.gate.0.weight"),
            Some(self.tensor("attn.sdpa.query_scaling.gate.0.bias")),
            64,
        )?;
        linear(
            &gelu(&hidden)?,
            64,
            self.tensor("attn.sdpa.query_scaling.gate.2.weight"),
            Some(self.tensor("attn.sdpa.query_scaling.gate.2.bias")),
            head_dim,
        )
    }

    /// Softmax attention of `[groups, heads, rows, head_dim]` queries to their
    /// group's keys and values; with fewer key heads, query head `h` uses key
    /// head `h / (heads / key heads)`.
    fn attend(&self, q: &[f32], rows: usize, groups: usize, kv: &Kv) -> Result<Vec<f32>> {
        let (heads, head_dim) = (self.heads, self.head_dim());
        let share = heads / kv.heads;
        let mut out = vec![0.0; q.len()];
        let jobs: Vec<(usize, usize, usize)> = (0..groups)
            .flat_map(|g| {
                (0..heads).flat_map(move |h| (0..rows).step_by(ATTENTION_ROWS).map(move |r| (g, h, r)))
            })
            .collect();
        let pieces: Vec<Vec<f32>> = jobs
            .par_iter()
            .map(|&(group, head, start)| {
                let count = ATTENTION_ROWS.min(rows - start);
                let kv_group = if kv.groups == 1 { 0 } else { group };
                let source = ((kv_group * kv.heads + head / share) * kv.rows) * head_dim;
                let key = &kv.key[source..][..kv.rows * head_dim];
                let value = &kv.value[source..][..kv.rows * head_dim];
                let query = &q[((group * heads + head) * rows + start) * head_dim..][..count * head_dim];
                let mut output = vec![0.0; count * head_dim];
                let (blocks, tiles) = (16, 128);
                let mut scores = vec![0.0; blocks.min(count) * tiles.min(kv.rows)];
                let (mut maximum, mut denominator) = (vec![0.0; blocks], vec![0.0; blocks]);
                tiled_online_scaled_dot_product_attention_f32(
                    query,
                    count,
                    key,
                    value,
                    kv.rows,
                    1,
                    head_dim,
                    blocks,
                    tiles,
                    &mut OnlineAttentionScratch {
                        scores: &mut scores,
                        maximum: &mut maximum,
                        denominator: &mut denominator,
                    },
                    &mut output,
                )?;
                Ok(output)
            })
            .collect::<Result<_>>()?;
        for ((group, head, start), piece) in jobs.into_iter().zip(pieces) {
            out[((group * heads + head) * rows + start) * head_dim..][..piece.len()].copy_from_slice(&piece);
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;

    use super::Model;
    use crate::models::kumo_tabular::preprocess::fixtures::{case, rows};

    /// The model of `IRONICL_CHECKPOINT_FIXTURES/kumo-tabular-<size>-<task>.pt`.
    pub(crate) fn model(size: &str, task: &str) -> Model {
        let directory = std::env::var_os("IRONICL_CHECKPOINT_FIXTURES").expect("IRONICL_CHECKPOINT_FIXTURES");
        let path = PathBuf::from(directory).join(format!("kumo-tabular-{size}-{task}.pt"));
        Model::new(crate::load_checkpoint_model(&path).expect("checkpoint")).expect("a Kumo-Tabular model")
    }

    fn flat(value: &serde_json::Value) -> Vec<f32> {
        #[expect(clippy::cast_possible_truncation, reason = "The model reads FP32")]
        rows(value).into_iter().flatten().map(|v| v as f32).collect()
    }

    #[test]
    #[ignore = "needs IRONICL_CHECKPOINT_FIXTURES with the Kumo-Tabular checkpoints"]
    fn members_predict_nvidias_outputs() {
        for (name, size) in [
            ("numeric", "small"),
            ("mixed", "small"),
            ("mixed_medium", "medium"),
            ("mixed_large", "large"),
            ("numeric_regression", "small"),
            ("mixed_regression_large", "large"),
        ] {
            let case = case(name);
            let regressor = case.json["task"] == "regression";
            let model = model(size, if regressor { "regressor" } else { "classifier" });
            let outputs = if regressor { 999 } else { 10 };
            let categorical: Vec<&str> = case.json["columns"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["categorical"].as_bool().unwrap())
                .map(|c| c["name"].as_str().unwrap())
                .collect();
            for (index, member) in case.json["members"].as_array().unwrap().iter().enumerate() {
                let mask: Vec<bool> = member["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| categorical.contains(&n.as_str().unwrap()))
                    .collect();
                let y = member["y"].as_array().unwrap();
                let codes: Vec<usize> = y
                    .iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|c| usize::try_from(c).unwrap())
                    .collect();
                #[expect(clippy::cast_possible_truncation, reason = "The model reads FP32")]
                let values: Vec<f32> = y.iter().map(|v| v.as_f64().unwrap() as f32).collect();
                let targets = if regressor {
                    super::Targets::Values(&values)
                } else {
                    super::Targets::Classes(&codes)
                };
                let context = model.fit(&flat(&member["x"]), &mask, targets).unwrap();
                let predicted = model.predict(&context, &flat(&member["query"])).unwrap();
                let mut worst = 0.0_f64;
                for (row, values) in rows(&member["output"]["values"]).iter().enumerate() {
                    for (column, value) in values.iter().enumerate() {
                        worst = worst.max((f64::from(predicted[row * outputs + column]) - value).abs());
                    }
                }
                println!("{name} member {index}: max |output error| {worst:.2e}");
                assert!(worst < 1e-4, "{name} member {index}: {worst}");
            }
        }
    }
}

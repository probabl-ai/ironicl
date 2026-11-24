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
use crate::format::static_context::{StaticContextArtifact, StaticContextTensor, StaticContextTensorData};
use crate::kernels::reference::{
    OnlineAttentionScratch, affine_f32, gelu_f32, rope_non_interleaved_f32,
    tiled_online_scaled_dot_product_attention_f32,
};
use crate::{Error, ModelArtifact, Result, format::model::ModelProfile};

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
#[derive(Clone, Copy, PartialEq)]
enum Scaling {
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

/// The FP32 values and dimensions of the tensor `name` in `artifact`.
fn read_tensor(artifact: &StaticContextArtifact, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let tensor = artifact
        .tensor(name)
        .ok_or_else(|| Error::Artifact(format!("the saved context has no {name}")))?;
    let StaticContextTensorData::Fp32(values) = tensor.data else {
        return Err(Error::Artifact(format!("{name} is not FP32")));
    };
    let dimensions = tensor
        .dimensions
        .iter()
        .map(|&n| usize::try_from(n).map_err(|_| Error::Artifact(format!("{name} is too large"))))
        .collect::<Result<_>>()?;
    Ok((values.to_vec(), dimensions))
}

/// Layers of keys and values of one shape, stacked as `{name}.key` and
/// `{name}.value`, `[layers, groups, heads, rows, head_dim]`.
fn stack(kvs: &[Kv], name: &str) -> [StaticContextTensor; 2] {
    let first = &kvs[0];
    let head_dim = first.key.len() / (first.groups * first.heads * first.rows).max(1);
    let dimensions = [kvs.len(), first.groups, first.heads, first.rows, head_dim]
        .map(|n| n as u64)
        .to_vec();
    let values = |part: fn(&Kv) -> &Vec<f32>| kvs.iter().flat_map(part).copied().collect();
    [
        StaticContextTensor {
            name: format!("{name}.key"),
            dimensions: dimensions.clone(),
            values: values(|kv| &kv.key),
        },
        StaticContextTensor {
            name: format!("{name}.value"),
            dimensions,
            values: values(|kv| &kv.value),
        },
    ]
}

/// The `layers` layers [`stack`] stored as `name`.
fn unstack(artifact: &StaticContextArtifact, name: &str, layers: usize) -> Result<Vec<Kv>> {
    let (keys, dimensions) = read_tensor(artifact, &format!("{name}.key"))?;
    let (values, value_dimensions) = read_tensor(artifact, &format!("{name}.value"))?;
    let [stacked, groups, heads, rows, _] = dimensions[..] else {
        return Err(Error::Artifact(format!("{name}.key is not 5-dimensional")));
    };
    if value_dimensions != dimensions || stacked != layers {
        return Err(Error::Artifact(format!(
            "{name} does not hold {layers} layers of keys and values"
        )));
    }
    let size = keys.len() / layers.max(1);
    Ok(keys
        .chunks(size)
        .zip(values.chunks(size))
        .map(|(key, value)| Kv {
            key: key.to_vec(),
            value: value.to_vec(),
            groups,
            heads,
            rows,
        })
        .collect())
}

impl Context {
    /// The context as named FP32 tensors under `prefix`: the imputation means
    /// and the columns' kinds (1 for category codes), `[columns]`, and the
    /// column and ICL layers' keys and values, each stage stacked.
    pub(crate) fn tensors(&self, prefix: &str) -> Vec<StaticContextTensor> {
        let mut tensors = vec![
            StaticContextTensor {
                name: format!("{prefix}.means"),
                dimensions: vec![self.means.len() as u64],
                values: self.means.clone(),
            },
            StaticContextTensor {
                name: format!("{prefix}.categorical"),
                dimensions: vec![self.categorical.len() as u64],
                values: self.categorical.iter().map(|&c| f32::from(u8::from(c))).collect(),
            },
        ];
        tensors.extend(stack(&self.columns, &format!("{prefix}.columns")));
        tensors.extend(stack(&self.icl, &format!("{prefix}.icl")));
        tensors
    }

    /// The context [`Self::tensors`] stored under `prefix`, with
    /// `column_layers` and `icl_layers` layers.
    pub(crate) fn read(
        artifact: &StaticContextArtifact,
        prefix: &str,
        (column_layers, icl_layers): (usize, usize),
    ) -> Result<Self> {
        let (means, _) = read_tensor(artifact, &format!("{prefix}.means"))?;
        let (categorical, _) = read_tensor(artifact, &format!("{prefix}.categorical"))?;
        if means.len() != categorical.len() {
            return Err(Error::Artifact(format!(
                "{prefix}'s means and kinds differ in length"
            )));
        }
        Ok(Self {
            categorical: categorical.iter().map(|&kind| kind != 0.0).collect(),
            means,
            columns: unstack(artifact, &format!("{prefix}.columns"), column_layers)?,
            icl: unstack(artifact, &format!("{prefix}.icl"), icl_layers)?,
        })
    }
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
        })
    }

    /// The checkpoint's SHA-256.
    pub(crate) fn sha256(&self) -> [u8; 32] {
        crate::models::checkpoint::decode_sha256(&self.artifact.summary().source_sha256)
            .expect("a verified model's SHA-256")
    }

    fn tensor(&self, name: &str) -> &[f32] {
        self.artifact
            .tensor_f32(name)
            .map(crate::format::model::ModelTensorF32::values)
            .expect("the manifest has every tensor")
    }

    /// The context [`Context::tensors`] stored under `prefix` in `artifact`,
    /// after checking that it holds exactly this model's tensors for a member
    /// with columns `categorical` and `training_rows` training rows.
    ///
    /// # Errors
    ///
    /// When a tensor is missing, has other dimensions or kinds, or is not FP32.
    pub(crate) fn read_context(
        &self,
        artifact: &StaticContextArtifact,
        prefix: &str,
        categorical: &[bool],
        training_rows: usize,
    ) -> Result<Context> {
        let config = self.config;
        let columns = categorical.len() as u64;
        let heads = config.embedding_heads;
        let query_heads = config.query_key_value_heads.unwrap_or(config.icl_heads);
        let column = vec![
            config.embedding_layers,
            columns,
            heads,
            config.inducing_points,
            config.cell_channels / heads,
        ];
        let icl = vec![
            config.icl_layers,
            1,
            query_heads,
            training_rows as u64,
            config.icl_channels / config.icl_heads,
        ];
        for (name, dimensions) in [
            ("means", vec![columns]),
            ("categorical", vec![columns]),
            ("columns.key", column.clone()),
            ("columns.value", column),
            ("icl.key", icl.clone()),
            ("icl.value", icl),
        ] {
            artifact.check_tensor(&format!("{prefix}.{name}"), &dimensions)?;
        }
        let context = Context::read(artifact, prefix, self.layers())?;
        if context.categorical != categorical {
            return Err(Error::Artifact(format!(
                "{prefix}'s column kinds are not the model's"
            )));
        }
        Ok(context)
    }

    /// The column and ICL layers a context holds.
    pub(crate) fn layers(&self) -> (usize, usize) {
        (
            Self::dim(self.config.embedding_layers),
            Self::dim(self.config.icl_layers),
        )
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
    /// class `codes`, and return it.
    ///
    /// # Errors
    ///
    /// When the shapes disagree or a code is not below 10.
    pub fn fit(&self, x: &[f32], categorical: &[bool], codes: &[usize]) -> Result<Context> {
        let columns = categorical.len();
        if columns == 0 {
            return Err(invalid(
                "no feature varies in the training data: every column is constant or missing",
            ));
        }
        if x.len() != codes.len() * columns || codes.is_empty() {
            return Err(invalid("the training rows do not match their targets"));
        }
        if codes.iter().any(|&code| code >= 10) {
            return Err(invalid("a class code is not below 10"));
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
        let mut rows = self.row_embedding(x, &context, Columns::Record(&mut recorded, codes))?;
        context.columns = recorded;
        let icl = Self::dim(self.config.icl_channels);
        let embedding = self.tensor("icl_block.y_emb.weight");
        for (row, &code) in rows.chunks_mut(icl).zip(codes) {
            for (value, label) in row.iter_mut().zip(&embedding[code * icl..][..icl]) {
                *value += label;
            }
        }
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

    /// The 10 logits of each of the `[rows, columns]` preprocessed query rows `x`.
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
        let codes = match &kvs {
            Columns::Record(_, codes) => Some(*codes),
            Columns::Read(_) => None,
        };
        let columns = context.categorical.len();
        let rows = x.len() / columns;
        let (cell, readout) = (self.cell(), Self::dim(self.config.readout_tokens));
        let mut cells = self.cell_embedding(x, context)?; // [columns, rows, cell]
        if let Some(codes) = codes {
            let embedding = self.tensor("row_embedding.y_emb.weight");
            cells.par_chunks_mut(rows * cell).for_each(|column| {
                for (value, &code) in column.chunks_mut(cell).zip(codes) {
                    for (v, label) in value.iter_mut().zip(&embedding[code * cell..][..cell]) {
                        *v += label;
                    }
                }
            });
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

/// The column stage's keys and values: recorded from the training rows and
/// their class codes, or read from a context.
enum Columns<'a> {
    Record(&'a mut Vec<Kv>, &'a [usize]),
    Read(&'a [Kv]),
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
        let release = format!("{size}-{task}").parse().expect("a Kumo-Tabular release");
        let kind = crate::CheckpointKind::KumoTabular(Some(release));
        Model::new(crate::load_checkpoint_model(&path, kind).expect("checkpoint"))
            .expect("a Kumo-Tabular model")
    }

    fn flat(value: &serde_json::Value) -> Vec<f32> {
        #[expect(clippy::cast_possible_truncation, reason = "The model reads FP32")]
        rows(value).into_iter().flatten().map(|v| v as f32).collect()
    }

    #[test]
    #[ignore = "needs IRONICL_CHECKPOINT_FIXTURES with the Kumo-Tabular checkpoints"]
    fn members_predict_nvidias_logits() {
        for (name, size) in [
            ("numeric", "small"),
            ("mixed", "small"),
            ("mixed_medium", "medium"),
            ("mixed_large", "large"),
        ] {
            let case = case(name);
            let model = model(size, "classifier");
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
                let codes: Vec<usize> = member["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| usize::try_from(c.as_u64().unwrap()).unwrap())
                    .collect();
                let context = model.fit(&flat(&member["x"]), &mask, &codes).unwrap();
                let logits = model.predict(&context, &flat(&member["query"])).unwrap();
                let expected = rows(&member["output"]["values"]);
                let classes = expected[0].len();
                let mut worst = 0.0_f64;
                for (row, values) in expected.iter().enumerate() {
                    for (class, value) in values.iter().enumerate() {
                        worst = worst.max((f64::from(logits[row * 10 + class]) - value).abs());
                    }
                }
                println!("{name} member {index}: max |logit error| {worst:.2e}");
                assert!(worst < 1e-4, "{name} member {index}: {worst}");
                let _ = classes;
            }
        }
    }
}

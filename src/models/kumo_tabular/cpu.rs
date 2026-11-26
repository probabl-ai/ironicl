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
    source_major_online_attention_f32_key_bf16_value_with_backend_f32,
    source_major_online_attention_with_backend_f32,
};
use crate::{Error, ModelArtifact, Result, TaskKind, format::model::ModelProfile};

/// `torch.finfo(torch.float32).eps`, the block norms' epsilon.
const NORM_EPS: f32 = f32::EPSILON;
/// The per-head query and key norms' epsilon.
const HEAD_NORM_EPS: f32 = 1e-6;
/// Rows a stage runs at once, which bounds its buffers.
const CHUNK_ROWS: usize = 256;
/// Max query rows per attention job; a multiple of 16 for the unrolled AVX-512 tiles.
const ATTENTION_ROWS: usize = 128;

/// A loaded Kumo-Tabular checkpoint.
pub struct Model {
    pub release: Release,
    pub(super) optimize: crate::Optimize,
    config: Config,
    artifact: ModelArtifact,
    /// The NVIDIA GPU that fits and predicts, when there is one.
    #[cfg(feature = "cuda")]
    cuda: Option<super::cuda::Cuda>,
    /// The Apple Silicon GPU that fits and predicts, when there is one.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    metal: Option<super::metal::Metal>,
}

/// Keys and values for `groups` independent attentions, each over `rows`
/// rows, stored `[groups, heads, rows, head_dim]`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Kv {
    pub(super) key: Vec<f32>,
    pub(super) value: Values,
    pub(super) groups: usize,
    pub(super) heads: usize,
    pub(super) rows: usize,
}

/// Only retained CPU ICL values may use BF16; keys and transient training values stay FP32.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Values {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl Values {
    /// The portable representation for saving or uploading a context.
    pub(super) fn floats(&self) -> std::borrow::Cow<'_, [f32]> {
        match self {
            Self::F32(v) => v.as_slice().into(),
            Self::Bf16(v) => v.iter().map(|&w| half::bf16::from_bits(w).to_f32()).collect(),
        }
    }

    fn compress(&mut self) {
        let Self::F32(values) = self else { return };
        // A custom checkpoint can produce finite FP32 beyond the finite BF16 range.
        if let Some(bits) = values
            .iter()
            .copied()
            .map(crate::kernels::cpu::f32_to_bf16_finite)
            .collect()
        {
            *self = Self::Bf16(bits);
        }
    }
}

impl Kv {
    /// The first `heads` heads of each group.
    fn first_heads(mut self, heads: usize, head_dim: usize) -> Self {
        if heads < self.heads {
            let (old, new) = (self.heads * self.rows * head_dim, heads * self.rows * head_dim);
            let keep = |tensor: &[f32]| {
                tensor
                    .chunks(old)
                    .flat_map(|group| &group[..new])
                    .copied()
                    .collect()
            };
            self.key = keep(&self.key);
            self.value = Values::F32(keep(&self.value.floats()));
            self.heads = heads;
        }
        self
    }
}

/// A member's fitted context: the column imputation means, each layer's
/// column keys and values (one group per column), and each ICL layer's keys
/// and values; or, fitted on an NVIDIA GPU, all of these there.
pub struct Context {
    categorical: Vec<bool>,
    means: Vec<f32>,
    columns: Vec<Kv>,
    icl: Vec<Kv>,
    #[cfg(feature = "cuda")]
    cuda: Option<super::cuda::Context>,
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    metal: Option<super::metal::Context>,
}

impl Context {
    /// Each column's kind: 1 for category codes, else 0.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn kinds(&self) -> Vec<f32> {
        self.categorical
            .iter()
            .map(|&categorical| f32::from(u8::from(categorical)))
            .collect()
    }
}

/// How a block scales its queries by the key count `n`: not, by
/// `head_scale · ln n`, or that gated by `1 + tanh(gate(q))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    let mut values = Vec::with_capacity(first.key.len() * kvs.len());
    for kv in kvs {
        values.extend_from_slice(&kv.value.floats());
    }
    [
        StaticContextTensor {
            name: format!("{name}.key"),
            dimensions: dimensions.clone(),
            values: kvs.iter().flat_map(|kv| &kv.key).copied().collect(),
        },
        StaticContextTensor {
            name: format!("{name}.value"),
            dimensions,
            values,
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
            value: Values::F32(value.to_vec()),
            groups,
            heads,
            rows,
        })
        .collect())
}

impl Context {
    /// The column and ICL layers' keys and values on the host.
    #[cfg(any(
        feature = "cuda",
        all(feature = "metal", target_os = "macos", target_arch = "aarch64")
    ))]
    pub(super) fn kvs(&self) -> (&[Kv], &[Kv]) {
        (&self.columns, &self.icl)
    }

    /// Each column's kind, true for category codes, and imputation mean.
    #[cfg(feature = "cuda")]
    pub(super) fn columns_and_means(&self) -> (&[bool], &[f32]) {
        (&self.categorical, &self.means)
    }

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
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal: None,
        })
    }
}

/// Whether every row block's query and key `RoPE` frequencies are equal.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn rope_shared(artifact: &ModelArtifact, config: Config) -> bool {
    let frequencies = |layer: u64, transform: &str| {
        artifact
            .tensor_f32(&format!(
                "row_embedding.row_blocks.{layer}.attn.{transform}.0.inv_freq"
            ))
            .ok()
            .map(|tensor| tensor.values().to_vec())
    };
    (0..config.embedding_layers)
        .all(|layer| frequencies(layer, "query_transform") == frequencies(layer, "key_transform"))
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
    /// # Panics
    ///
    /// Never: a Kumo-Tabular artifact holds every tensor of its release.
    ///
    /// # Errors
    ///
    /// When the artifact is another model.
    pub fn new(artifact: ModelArtifact) -> Result<Self> {
        Self::with_optimize(artifact, crate::Optimize::Safe)
    }

    /// Load a checkpoint at the precision used throughout fit and prediction.
    pub(crate) fn with_optimize(artifact: ModelArtifact, optimize: crate::Optimize) -> Result<Self> {
        let ModelProfile::KumoTabular(release) = artifact.summary().profile else {
            return Err(Error::Artifact(
                "the model is not a Kumo-Tabular checkpoint".into(),
            ));
        };
        let config = release.size.config();
        #[cfg(feature = "cuda")]
        let cuda = super::cuda::Cuda::new(release, optimize, &|name| {
            artifact
                .tensor_f32(name)
                .map(crate::format::model::ModelTensorF32::values)
                .expect("the manifest has every tensor")
        });
        // The Metal graphs rotate queries and keys with one table: the checkpoint's two must
        // be equal, or the model runs on the CPU, which applies each.
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        let metal = rope_shared(&artifact, config)
            .then(|| super::metal::Metal::new(release, optimize))
            .flatten();
        Ok(Self {
            release,
            optimize,
            config,
            artifact,
            #[cfg(feature = "cuda")]
            cuda,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal,
        })
    }

    /// Where the model runs: `"cuda"`, `"metal"`, or `"cpu"`.
    #[must_use]
    pub fn device(&self) -> &'static str {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.metal.is_some() {
            return "metal";
        }
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            return "cuda";
        }
        "cpu"
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

    fn bf16_values(&self) -> bool {
        self.optimize == crate::Optimize::Fast && crate::kernels::cpu::bf16_storage_supported()
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
    /// `targets`, and return it.
    ///
    /// # Errors
    ///
    /// When the shapes disagree, or the targets do not fit the model's task.
    pub fn fit(&self, x: &[f32], categorical: &[bool], targets: Targets<'_>) -> Result<Context> {
        let columns = categorical.len();
        if columns == 0 {
            return Err(invalid(
                "no feature varies in the training data: every column is constant or missing",
            ));
        }
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
        if rows == 0 || x.len() != rows * columns {
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
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal: None,
        };
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(metal) = &self.metal {
            let (cell, icl) = (self.cell(), Self::dim(self.config.icl_channels));
            let (mut cell_labels, mut icl_labels) = (vec![0.0; rows * cell], vec![0.0; rows * icl]);
            self.add_labels("row_embedding", &mut cell_labels, cell, targets);
            self.add_labels("icl_block", &mut icl_labels, icl, targets);
            let weights = |name: &str| -> Result<&[f32]> { Ok(self.tensor(name)) };
            let kinds = context.kinds();
            context.metal =
                Some(metal.fit(&weights, (x, &context.means, &kinds), (&cell_labels, &icl_labels))?);
            return Ok(context);
        }
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            let (cell, icl) = (self.cell(), Self::dim(self.config.icl_channels));
            let (mut cell_labels, mut icl_labels) = (vec![0.0; rows * cell], vec![0.0; rows * icl]);
            self.add_labels("row_embedding", &mut cell_labels, cell, targets);
            self.add_labels("icl_block", &mut icl_labels, icl, targets);
            match cuda.fit(x, categorical, (&cell_labels, &icl_labels)) {
                Ok(fitted) => {
                    context.cuda = Some(fitted);
                    return Ok(context);
                }
                // A context that does not fit in device memory is built on the CPU.
                Err(Error::UnsupportedBackend(message)) if message == super::cuda::OUT_OF_MEMORY => {}
                Err(error) => return Err(error),
            }
        }
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
            let mut kept = kv.first_heads(query_heads, block.head_dim());
            if self.bf16_values() {
                kept.value.compress();
            }
            context.icl.push(kept);
        }
        Ok(context)
    }

    /// `context` as named FP32 tensors under `prefix`, as
    /// [`Context::tensors`] gives them, copied from the GPU when it lives there.
    ///
    /// # Errors
    ///
    /// When the copy from the GPU fails.
    #[cfg_attr(
        not(any(
            feature = "cuda",
            all(feature = "metal", target_os = "macos", target_arch = "aarch64")
        )),
        expect(clippy::unused_self, reason = "The GPU paths read it")
    )]
    #[cfg_attr(
        not(feature = "cuda"),
        expect(clippy::unnecessary_wraps, reason = "The CUDA path can fail")
    )]
    pub(crate) fn context_tensors(
        &self,
        context: &Context,
        prefix: &str,
    ) -> Result<Vec<StaticContextTensor>> {
        #[cfg(feature = "cuda")]
        if let (Some(cuda), Some(fitted)) = (&self.cuda, &context.cuda) {
            let (columns, icl) = cuda.download(fitted)?;
            let host = Context {
                categorical: context.categorical.clone(),
                means: context.means.clone(),
                columns,
                icl,
                cuda: None,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                metal: None,
            };
            return Ok(host.tensors(prefix));
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let (Some(metal), Some(fitted)) = (&self.metal, &context.metal) {
            let (columns, icl) = metal.download(fitted, context.categorical.len());
            let host = Context {
                categorical: context.categorical.clone(),
                means: context.means.clone(),
                columns,
                icl,
                #[cfg(feature = "cuda")]
                cuda: None,
                metal: None,
            };
            return Ok(host.tensors(prefix));
        }
        Ok(context.tensors(prefix))
    }

    /// The loaded `context` where this model predicts: on its GPU when it has
    /// one and the context fits in device memory, else on the CPU.
    ///
    /// # Errors
    ///
    /// When the copy to the GPU fails for another reason.
    #[cfg_attr(
        not(any(
            feature = "cuda",
            all(feature = "metal", target_os = "macos", target_arch = "aarch64")
        )),
        expect(clippy::unnecessary_wraps, reason = "GPU placement can fail")
    )]
    pub(crate) fn place(&self, mut context: Context) -> Result<Context> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            let training_rows = context.icl.first().map_or(0, |kv| kv.rows);
            match cuda.upload_context(&context, training_rows) {
                Ok(fitted) => {
                    context.cuda = Some(fitted);
                    context.columns = Vec::new();
                    context.icl = Vec::new();
                }
                Err(Error::UnsupportedBackend(message)) if message == super::cuda::OUT_OF_MEMORY => {}
                Err(error) => return Err(error),
            }
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(metal) = &self.metal
            && context.metal.is_none()
        {
            let training_rows = context.icl.first().map_or(0, |kv| kv.rows);
            context.metal = Some(metal.upload_context(&context, training_rows)?);
            context.columns = Vec::new();
            context.icl = Vec::new();
        }
        // Successful GPU placement emptied `icl`; a CPU fallback still owns its host values.
        if self.bf16_values() {
            for kv in &mut context.icl {
                kv.value.compress();
            }
        }
        Ok(context)
    }

    /// The members' `contexts` stacked on the GPU so one pass predicts them
    /// all, when every one lives there, they have the same columns, and the
    /// stacking fits in device memory; their device buffers move into the stack.
    ///
    /// # Errors
    ///
    /// When stacking fails on the device.
    #[cfg(feature = "cuda")]
    pub(crate) fn stack(&self, contexts: &mut [Vec<Context>]) -> Result<Option<super::cuda::Ensemble>> {
        let Some(cuda) = &self.cuda else {
            return Ok(None);
        };
        let all = || contexts.iter().flatten();
        let columns = all().next().map_or(0, |context| context.categorical.len());
        if all().any(|context| context.cuda.is_none() || context.categorical.len() != columns) {
            return Ok(None);
        }
        // Without the memory to stack, the members keep predicting one at a time.
        let on_device: Vec<_> = all().filter_map(|context| context.cuda.as_ref()).collect();
        if !cuda.stack_fits(&on_device)? {
            return Ok(None);
        }
        let taken = contexts
            .iter_mut()
            .flatten()
            .map(|context| context.cuda.take().expect("checked above"))
            .collect();
        cuda.stack(taken).map(Some)
    }

    /// Each stacked slot's outputs of its preprocessed query rows `xs[slot]`.
    ///
    /// # Errors
    ///
    /// When the rows do not have the contexts' columns.
    #[cfg(feature = "cuda")]
    pub(crate) fn predict_stacked(
        &self,
        stack: &super::cuda::Ensemble,
        xs: &[&[f32]],
        quantiles: Option<super::cuda::Quantiles>,
    ) -> Result<Vec<Vec<f32>>> {
        self.cuda
            .as_ref()
            .expect("a stack lives on its model's GPU")
            .predict_ensemble(stack, xs, quantiles)
    }

    /// The stacked slots as named FP32 tensors, under the prefix next to each
    /// slot's `contexts`, as [`Self::context_tensors`] gives them.
    ///
    /// # Errors
    ///
    /// When the copy from the GPU fails.
    #[cfg(feature = "cuda")]
    pub(crate) fn stacked_tensors(
        &self,
        stack: &super::cuda::Ensemble,
        contexts: &[(&Context, String)],
    ) -> Result<Vec<StaticContextTensor>> {
        let cuda = self.cuda.as_ref().expect("a stack lives on its model's GPU");
        let mut tensors = Vec::new();
        for ((columns, icl), (context, prefix)) in cuda.download_ensemble(stack)?.into_iter().zip(contexts) {
            let host = Context {
                categorical: context.categorical.clone(),
                means: context.means.clone(),
                columns,
                icl,
                cuda: None,
            };
            tensors.extend(host.tensors(prefix));
        }
        Ok(tensors)
    }

    /// Whether `context` lives on the model's GPU: false on the CPU, or when
    /// it did not fit in device memory.
    #[must_use]
    pub fn on_device(&self, context: &Context) -> bool {
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() && context.cuda.is_some() {
            return true;
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.metal.is_some() && context.metal.is_some() {
            return true;
        }
        let _ = context;
        false
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
        #[cfg(feature = "cuda")]
        if let (Some(cuda), Some(fitted)) = (&self.cuda, &context.cuda) {
            return cuda.predict(fitted, x);
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let (Some(metal), Some(fitted)) = (&self.metal, &context.metal) {
            let weights = |name: &str| -> Result<&[f32]> { Ok(self.tensor(name)) };
            return metal.predict(&weights, fitted, (x, &context.means, &context.kinds()));
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
            value: Values::F32(value),
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
        let attended = from_heads(&self.attend(&q, rows, kv)?, groups, heads, head_dim);
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
    fn attend(&self, q: &[f32], rows: usize, kv: &Kv) -> Result<Vec<f32>> {
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let (heads, head_dim) = (self.heads, self.head_dim());
        let share = heads / kv.heads;
        // All `share` query heads for one `(group, kv_head)` are contiguous in `[groups, heads,
        // rows, head_dim]` and attend independently to the same key/value slice. Shrink chunks on
        // small batches so GQA's 2 KV heads still fan out across all CPU threads.
        let shared_rows = share * rows;
        let total_rows = q.len() / head_dim;
        let target = (total_rows / rayon::current_num_threads().max(1)).clamp(1, ATTENTION_ROWS);
        let chunk_rows = if target >= 16 { target / 16 * 16 } else { target };
        let pair_stride = shared_rows * head_dim;
        let chunk_stride = chunk_rows * head_dim;
        let backend = crate::kernels::cpu::Isa::detect();
        let (blocks, tiles) = (16, 128);
        let attend_chunk = |q_chunk: &[f32], out_chunk: &mut [f32], range: std::ops::Range<usize>| {
            let count = out_chunk.len() / head_dim;
            let key = &kv.key[range.clone()];
            let mut scores = vec![0.0; blocks.min(count) * tiles.min(kv.rows)];
            let (mut maximum, mut denominator) = (vec![0.0; count], vec![0.0; count]);
            let mut scratch = OnlineAttentionScratch {
                scores: &mut scores,
                maximum: &mut maximum,
                denominator: &mut denominator,
            };
            match &kv.value {
                Values::F32(v) => source_major_online_attention_with_backend_f32(
                    q_chunk,
                    count,
                    key,
                    &v[range],
                    kv.rows,
                    1,
                    head_dim,
                    blocks,
                    tiles,
                    backend,
                    &mut scratch,
                    out_chunk,
                ),
                Values::Bf16(v) => source_major_online_attention_f32_key_bf16_value_with_backend_f32(
                    q_chunk,
                    count,
                    key,
                    &v[range],
                    kv.rows,
                    1,
                    head_dim,
                    blocks,
                    tiles,
                    backend,
                    &mut scratch,
                    out_chunk,
                ),
            }
        };
        let kv_range = |pair: usize| {
            let (group, kv_head) = (pair / kv.heads, pair % kv.heads);
            let kv_group = if kv.groups == 1 { 0 } else { group };
            let source = ((kv_group * kv.heads + kv_head) * kv.rows) * head_dim;
            source..source + kv.rows * head_dim
        };
        let mut out = vec![0.0; q.len()];
        if shared_rows <= chunk_rows {
            out.par_chunks_mut(pair_stride)
                .zip(q.par_chunks(pair_stride))
                .enumerate()
                .try_for_each(|(pair, (pair_out, pair_q))| attend_chunk(pair_q, pair_out, kv_range(pair)))?;
        } else {
            out.par_chunks_mut(pair_stride)
                .zip(q.par_chunks(pair_stride))
                .enumerate()
                .try_for_each(|(pair, (pair_out, pair_q))| {
                    let range = kv_range(pair);
                    pair_out
                        .par_chunks_mut(chunk_stride)
                        .zip(pair_q.par_chunks(chunk_stride))
                        .try_for_each(|(out_chunk, q_chunk)| attend_chunk(q_chunk, out_chunk, range.clone()))
                })?;
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;

    use super::Model;
    use crate::models::kumo_tabular::preprocess::fixtures::{case, rows};

    #[test]
    fn bf16_values_round_once_and_keep_unrepresentable_layers_in_fp32() {
        use super::Values;
        let mut values = Values::F32(vec![1.003, -2.0]);
        values.compress();
        assert!(matches!(values, Values::Bf16(_)));
        assert_eq!(&*values.floats(), &[1.0, -2.0]);
        for v in [f32::MAX, f32::from_bits(0xffff_ffff)] {
            let mut layer = Values::F32(vec![v]);
            layer.compress();
            assert!(matches!(layer, Values::F32(_)));
        }
    }

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

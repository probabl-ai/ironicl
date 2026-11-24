//! The pre-norm transformer block both models build their `MPSGraph` graphs
//! from: attention with a residual, then a GELU feed-forward with a residual,
//! with each model's norms, `RoPE`, per-head norm, query scaling, and shared
//! key and value heads.

#![expect(
    unsafe_code,
    reason = "Objective-C calls; each unsafe block states its invariant"
)]
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "tensor dimensions are model constants or validated sizes far below isize::MAX"
)]

use super::{FLOAT, Tensor, builder::Builder, shape};
use crate::Result;

/// Where a block's keys and values come from: cached, already projected and
/// head-major, or projected here from tokens.
pub(crate) enum Source<'t> {
    Cached(&'t Tensor, &'t Tensor),
    Tokens(&'t Tensor),
}

/// `RoPE`'s cosine and signed sine tables for the queries and for the keys.
pub(crate) struct Rope<'t> {
    pub(crate) query: (&'t Tensor, &'t Tensor),
    pub(crate) key: (&'t Tensor, &'t Tensor),
}

/// A block's norms: `LayerNorm` with weight and bias, or `RMSNorm` with weight.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Norm {
    Layer,
    Rms,
}

/// A block's names and options.
pub(crate) struct Spec<'a> {
    pub(crate) width: usize,
    pub(crate) heads: usize,
    pub(crate) norm: Norm,
    /// The norm of the queries, and of the keys and values, by weight prefix.
    pub(crate) query_norm: &'a str,
    pub(crate) key_value_norm: &'a str,
    /// The fused query, key, and value projection's weight and bias.
    pub(crate) qkv: (&'a str, &'a str),
    pub(crate) out: &'a str,
    pub(crate) mlp_norm: &'a str,
    pub(crate) mlp_in: &'a str,
    pub(crate) mlp_out: &'a str,
    pub(crate) rope: Option<&'a Rope<'a>>,
    /// The per-head `RMSNorm` of queries and keys, after `RoPE`.
    pub(crate) head_norm: bool,
    /// The queries' scale, shaped to scale `[lead, heads, rows, head_dim]`.
    pub(crate) scale: Option<&'a Tensor>,
    /// The scale's gate `1 + tanh(mlp(q))`: the MLP's two layers by prefix.
    pub(crate) gate: Option<(&'a str, &'a str)>,
}

impl Spec<'_> {
    fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

// SAFETY (all methods): plain `MPSGraph` construction on `self.graph`, as in
// the `Builder` impl: every tensor came from `self.graph`, names are `None`, and
// no call touches buffer memory.
impl Builder<'_, '_> {
    /// `[lead, rows, heads * head_dim]` -> `[lead, heads, rows, head_dim]`.
    pub(crate) fn split_heads(
        &self,
        input: &Tensor,
        (lead, rows): (isize, isize),
        (heads, head_dim): (usize, usize),
    ) -> Tensor {
        self.transpose(
            &self.reshape(input, &[lead, rows, heads as isize, head_dim as isize]),
            1,
            2,
        )
    }

    /// `x / sqrt(mean(x²) + eps)` over the last axis, times `weight` when given.
    pub(crate) fn rms(&self, x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Tensor {
        // SAFETY: graph construction on `self.graph` (impl comment).
        let scale = unsafe {
            let mean = self
                .graph
                .meanOfTensor_axes_name(&self.multiply(x, x), &shape(&[-1]), None);
            self.graph
                .reciprocalSquareRootWithTensor_name(&self.add(&mean, &self.scalar(eps)), None)
        };
        let normed = self.multiply(x, &scale);
        weight.map_or(normed.clone(), |weight| self.multiply(&normed, weight))
    }

    fn block_norm(&self, input: &Tensor, prefix: &str, spec: &Spec<'_>) -> Result<Tensor> {
        match spec.norm {
            Norm::Layer => self.layer_norm(input, prefix, spec.width),
            Norm::Rms => {
                let weight = self.named_weight(&format!("{prefix}.weight"), &[spec.width as isize])?;
                Ok(self.rms(input, Some(&weight), f64::from(f32::EPSILON)))
            }
        }
    }

    /// Keys or values of `key_heads` heads broadcast to `heads`: query head
    /// `h` uses key head `h / (heads / key_heads)`.
    fn share_heads(&self, x: &Tensor, (key_heads, heads): (usize, usize)) -> Tensor {
        if key_heads == heads {
            return x.clone();
        }
        // SAFETY: a property read of a live tensor.
        let dims: Vec<isize> = unsafe { x.shape() }
            .map(|shape| shape.iter().map(|n| n.integerValue()).collect())
            .unwrap_or_default();
        let [lead, _, rows, head_dim] = dims[..] else {
            return x.clone();
        };
        let grouped = self.reshape(x, &[lead, key_heads as isize, 1, rows, head_dim]);
        // SAFETY: graph construction on `self.graph` (impl comment).
        let broadcast = unsafe {
            self.graph.broadcastTensor_toShape_name(
                &grouped,
                &shape(&[
                    lead,
                    key_heads as isize,
                    (heads / key_heads) as isize,
                    rows,
                    head_dim,
                ]),
                None,
            )
        };
        self.reshape(&broadcast, &[lead, heads as isize, rows, head_dim])
    }

    /// A block's head-major keys and values `[lead, heads, rows, head_dim]` of
    /// `tokens` `[lead, rows, width]`, with `RoPE` and the per-head norm on the
    /// keys when the block has them.
    pub(crate) fn key_value(
        &self,
        tokens: &Tensor,
        spec: &Spec<'_>,
        (lead, rows): (isize, isize),
    ) -> Result<(Tensor, Tensor)> {
        let (width, heads, head_dim) = (spec.width, spec.heads, spec.head_dim());
        let normalized = self.block_norm(tokens, spec.key_value_norm, spec)?;
        let split = |part: usize| -> Result<Tensor> {
            let projected = self.affine(&normalized, spec.qkv, part, (width, width))?;
            Ok(self.split_heads(&projected, (lead, rows), (heads, head_dim)))
        };
        let mut key = split(1)?;
        if let Some(rope) = spec.rope {
            key = self.rope(&key, rope.key, head_dim);
        }
        if spec.head_norm {
            key = self.rms(&key, None, 1e-6);
        }
        Ok((key, split(2)?))
    }

    /// `x` broadcast to `dimensions`.
    pub(crate) fn broadcast(&self, x: &Tensor, dimensions: &[isize]) -> Tensor {
        // SAFETY: graph construction on `self.graph` (impl comment).
        unsafe {
            self.graph
                .broadcastTensor_toShape_name(x, &shape(dimensions), None)
        }
    }

    /// One block on `input` `[lead, q_rows, width]`, attending to `source`.
    /// Returns the output and the head-major keys and values it attended to,
    /// before any sharing of key heads.
    pub(crate) fn transformer_block(
        &self,
        input: &Tensor,
        source: &Source<'_>,
        spec: &Spec<'_>,
        (lead, q_rows, source_rows): (isize, isize, isize),
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (width, heads, head_dim) = (spec.width, spec.heads, spec.head_dim());
        // The query, key, and value projections are parts 0, 1, and 2 of one fused weight.
        let project = |part: usize, x: &Tensor| self.affine(x, spec.qkv, part, (width, width));
        let query = project(0, &self.block_norm(input, spec.query_norm, spec)?)?;
        let mut query = self.split_heads(&query, (lead, q_rows), (heads, head_dim));
        let (key, value) = match *source {
            Source::Cached(key, value) => (key.clone(), value.clone()),
            Source::Tokens(tokens) => self.key_value(tokens, spec, (lead, source_rows))?,
        };
        if let Some(rope) = spec.rope {
            query = self.rope(&query, rope.query, head_dim);
        }
        if spec.head_norm {
            query = self.rms(&query, None, 1e-6);
        }
        if let Some(scale) = spec.scale {
            query = match spec.gate {
                // q * scale * (1 + tanh(mlp(q))), the MLP shared across heads.
                Some((first, second)) => {
                    let hidden = self.gelu(&self.linear(&query, first, head_dim, 64)?);
                    let modulation = self.linear(&hidden, second, 64, head_dim)?;
                    // SAFETY: graph construction on `self.graph` (impl comment).
                    let gate = self.add(
                        &unsafe { self.graph.tanhWithTensor_name(&modulation, None) },
                        &self.scalar(1.0),
                    );
                    self.multiply(&self.multiply(&query, scale), &gate)
                }
                None => self.multiply(&query, scale),
            };
        }
        // SAFETY: a property read of a live tensor.
        let key_heads = unsafe { key.shape() }
            .and_then(|shape| shape.iter().nth(1))
            .map_or(heads, |n| n.integerValue().unsigned_abs());
        let (shared_key, shared_value) = (
            self.share_heads(&key, (key_heads, heads)),
            self.share_heads(&value, (key_heads, heads)),
        );
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
                    &self.cast(&shared_key, kind),
                    &self.cast(&shared_value, kind),
                    (head_dim as f32).sqrt().recip(),
                    None,
                )
        };
        let attended = self.cast(&attended, FLOAT);
        let merged = self.reshape(&self.transpose(&attended, 1, 2), &[lead, q_rows, width as isize]);
        let residual = self.add(input, &self.linear(&merged, spec.out, width, width)?);
        let hidden = self.gelu(&self.linear(
            &self.block_norm(&residual, spec.mlp_norm, spec)?,
            spec.mlp_in,
            width,
            2 * width,
        )?);
        let output = self.add(&residual, &self.linear(&hidden, spec.mlp_out, 2 * width, width)?);
        Ok((output, key, value))
    }
}

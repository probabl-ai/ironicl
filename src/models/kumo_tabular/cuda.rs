//! Kumo-Tabular's transformer blocks on NVIDIA GPUs, in FP32.
//!
//! [`super::cpu`] runs the model; each block's keys and values and its
//! forward pass run here instead when there is a GPU, as with Metal. Products
//! run on `TabICLv2`'s FP32 kernel with its bias, GELU, and residual epilogues,
//! and attention on its split FP32 ICL attention, which takes 64-wide heads:
//! narrower heads are zero-padded, and their queries scaled to keep
//! `1 / sqrt(head_dim)`. The norms, `RoPE`, and query scaling have their own
//! kernels in `kernels::cuda`.

#![expect(unsafe_code, reason = "Launches of kernels that trust their grid and shapes")]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use cuda_core::{DeviceBuffer, LaunchConfig};

use super::cpu::Scaling;
use crate::Result;
use crate::kernels::cuda::{Device, gpu_error};
use crate::models::tabicl_v2::cuda::{Activation, Dense, Precision, padded_query_rows, padded_rows};

/// The GPU and the model's dense layers on it, uploaded on first use.
pub(crate) struct Cuda {
    device: Device,
    dense: Mutex<HashMap<String, Arc<Dense>>>,
}

/// A block as the GPU runs it: its name, shape, `RoPE`, query scaling, and
/// its tensors by suffix.
pub(crate) struct Spec<'a> {
    pub name: &'a str,
    pub width: usize,
    pub heads: usize,
    pub rope: bool,
    pub scaling: Scaling,
    pub tensor: &'a dyn Fn(&str) -> &'a [f32],
}

/// Keys and values on the host, `[groups, heads, rows, head_dim]` each.
pub(crate) struct KeyValues<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub groups: usize,
    pub heads: usize,
    pub rows: usize,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Block shapes stay far below u32::MAX"
)]
const fn n(value: usize) -> u32 {
    value as u32
}

fn config(blocks: usize, threads: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n(blocks), 1, 1),
        block_dim: (n(threads), 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Spec<'_> {
    fn head_dim(&self) -> usize {
        self.width / self.heads
    }

    fn frequencies(&self, key: bool) -> Option<&[f32]> {
        let name = if key { "key_transform" } else { "query_transform" };
        self.rope
            .then(|| (self.tensor)(&format!("attn.{name}.0.inv_freq")))
    }
}

impl Cuda {
    /// The GPU, or `None` without one or with `IRONICL_NO_CUDA` set.
    pub(crate) fn new() -> Option<Self> {
        Some(Self {
            device: Device::new().ok().flatten()?,
            dense: Mutex::default(),
        })
    }

    /// The dense layer named `key`, with weight `w` `[outputs, inputs]` and
    /// bias `b`, uploaded on first use.
    fn dense(&self, key: String, layer: impl FnOnce() -> (Vec<f32>, Vec<f32>)) -> Result<Arc<Dense>> {
        let mut dense = self
            .dense
            .lock()
            .map_err(|_| gpu_error("dense layer cache lock poisoned"))?;
        if let Some(found) = dense.get(&key) {
            return Ok(Arc::clone(found));
        }
        let (w, b) = layer();
        let uploaded = Arc::new(Dense::upload(&self.device, &w, &b)?);
        dense.insert(key, Arc::clone(&uploaded));
        Ok(uploaded)
    }

    /// The `width` → `width` part `part` (0 queries, 1 keys, 2 values) of a
    /// block's fused projection.
    fn projection(&self, spec: &Spec<'_>, part: usize) -> Result<Arc<Dense>> {
        let w = spec.width;
        self.dense(format!("{}.attn.qkv_lin[{part}]", spec.name), || {
            let weight = (spec.tensor)("attn.qkv_lin.weight")[part * w * w..][..w * w].to_vec();
            let bias = (spec.tensor)("attn.qkv_lin.bias")[part * w..][..w].to_vec();
            (weight, bias)
        })
    }

    fn linear(&self, spec: &Spec<'_>, prefix: &str) -> Result<Arc<Dense>> {
        self.dense(format!("{}.{prefix}", spec.name), || {
            let weight = (spec.tensor)(&format!("{prefix}.weight")).to_vec();
            let bias = (spec.tensor)(&format!("{prefix}.bias")).to_vec();
            (weight, bias)
        })
    }

    fn upload(&self, values: &[f32]) -> Result<DeviceBuffer<f32>> {
        self.device.upload(values)
    }

    fn zeroed(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        self.device.zeroed(len).map_err(gpu_error)
    }

    /// `rows` rows of `x` normalized by their root mean square, times `weight`.
    fn rms(&self, x: &DeviceBuffer<f32>, weight: &[f32], rows: usize, eps: f32) -> Result<DeviceBuffer<f32>> {
        let width = weight.len();
        let weight = self.upload(weight)?;
        let mut out = self.zeroed(rows * width)?;
        // SAFETY: one warp per row, 256 threads per block; `x` and `out` hold `rows * width` and
        // `weight` `width` values.
        let launch = unsafe {
            self.device.kernels.rms_rows(
                self.device.stream(),
                config((rows * 32).div_ceil(256), 256),
                x,
                &weight,
                &mut out,
                n(rows),
                n(width),
                eps,
                1,
            )
        };
        launch.map_err(gpu_error)?;
        Ok(out)
    }

    /// `act(x wᵀ + b) (+ residual)` for `rows` rows of `depth` values.
    fn product(
        &self,
        x: &DeviceBuffer<f32>,
        dense: &Arc<Dense>,
        residual: Option<&DeviceBuffer<f32>>,
        (rows, width, depth): (usize, usize, usize),
        gelu: bool,
    ) -> Result<DeviceBuffer<f32>> {
        let mut out = self.zeroed(rows * width)?;
        // `product` splits `depth` this way; the partial sums need room for every split.
        let blocks = rows.div_ceil(64) * width.div_ceil(64);
        let wanted = (2 * self.device.multiprocessors())
            .div_ceil(blocks)
            .min(depth / 64)
            .max(1);
        let splits = depth.div_ceil(depth.div_ceil(wanted).next_multiple_of(16));
        let mut partial = self.zeroed(if splits > 1 { splits * rows * width } else { 0 })?;
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

    /// Token-major `x[groups * rows, width]` as head-major `[groups * heads,
    /// padded, 64]`, with `RoPE` by `frequencies` and the per-head norm when `norm`.
    fn heads(
        &self,
        spec: &Spec<'_>,
        x: &DeviceBuffer<f32>,
        frequencies: Option<&[f32]>,
        (groups, rows, padded): (usize, usize, usize),
        norm: bool,
    ) -> Result<DeviceBuffer<f32>> {
        let frequencies = self.upload(frequencies.unwrap_or_default())?;
        let mut out = self.zeroed(groups * spec.heads * padded * 64)?;
        // SAFETY: one warp per (group, row, head), 256 threads per block; `x` holds `groups *
        // rows * width`, `frequencies` is empty or `head_dim / 2` long, and `out` is `[groups *
        // heads, padded, 64]` with `rows <= padded`.
        let launch = unsafe {
            self.device.kernels.to_heads(
                self.device.stream(),
                config((groups * rows * spec.heads * 32).div_ceil(256), 256),
                x,
                &frequencies,
                &mut out,
                n(groups),
                n(rows),
                n(spec.heads),
                n(spec.head_dim()),
                n(padded),
                u32::from(norm),
            )
        };
        launch.map_err(gpu_error)?;
        Ok(out)
    }

    /// A block's keys and values for `groups` equal parts of the `[rows,
    /// width]` rows `x`, each `[groups, heads, rows, head_dim]`.
    pub(crate) fn key_value(
        &self,
        spec: &Spec<'_>,
        x: &[f32],
        groups: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let rows = x.len() / (groups * spec.width);
        let x = self.upload(x)?;
        let normed = self.rms(
            &x,
            (spec.tensor)("key_value_norm.weight"),
            groups * rows,
            f32::EPSILON,
        )?;
        let shape = (groups * rows, spec.width, spec.width);
        let keys = self.product(&normed, &self.projection(spec, 1)?, None, shape, false)?;
        let values = self.product(&normed, &self.projection(spec, 2)?, None, shape, false)?;
        let keys = self.heads(spec, &keys, spec.frequencies(true), (groups, rows, rows), true)?;
        let values = self.heads(spec, &values, None, (groups, rows, rows), false)?;
        let unpad = |padded: Vec<f32>| -> Vec<f32> {
            padded
                .chunks(64)
                .flat_map(|row| &row[..spec.head_dim()])
                .copied()
                .collect()
        };
        Ok((
            unpad(self.device.download(&keys)?),
            unpad(self.device.download(&values)?),
        ))
    }

    /// A block's forward pass for `groups` parts of `rows` query rows each,
    /// each attending to its group of `kv`, or all to its one group.
    pub(crate) fn forward(
        &self,
        spec: &Spec<'_>,
        query: &[f32],
        (rows, groups): (usize, usize),
        kv: &KeyValues<'_>,
    ) -> Result<Vec<f32>> {
        let (width, heads, dim) = (spec.width, spec.heads, spec.head_dim());
        let (pairs, padded) = (groups * heads, padded_query_rows(rows));
        let query = self.upload(query)?;
        let normed = self.rms(
            &query,
            (spec.tensor)("query_norm.weight"),
            groups * rows,
            f32::EPSILON,
        )?;
        let projected = self.product(
            &normed,
            &self.projection(spec, 0)?,
            None,
            (groups * rows, width, width),
            false,
        )?;
        let mut queries = self.heads(
            spec,
            &projected,
            spec.frequencies(false),
            (groups, rows, padded),
            true,
        )?;
        self.scale(spec, &mut queries, (pairs, padded), kv.rows)?;
        let (keys, values) = (
            self.cache(kv, kv.key, (groups, heads))?,
            self.cache(kv, kv.value, (groups, heads))?,
        );
        let (_, splits) = self.device.icl_split(Precision::Fp32, pairs, padded, kv.rows);
        let (mut part, mut lse) = (
            self.zeroed(pairs * padded * splits * 64)?,
            self.zeroed(pairs * padded * splits)?,
        );
        let mut attended = self.zeroed(pairs * padded * 64)?;
        self.device.attend_icl(
            Precision::Fp32,
            Activation::F32(&queries),
            (&keys, &values),
            (&mut part, &mut lse, &mut attended),
            (pairs, padded, kv.rows),
        )?;
        let mut tokens = self.zeroed(groups * rows * width)?;
        // SAFETY: one thread per token value, 256 per block; `attended` is `[pairs, padded, 64]`
        // and `tokens` `[groups * rows, width]`.
        let launch = unsafe {
            self.device.kernels.from_heads(
                self.device.stream(),
                config((groups * rows * width).div_ceil(256), 256),
                &attended,
                &mut tokens,
                n(groups),
                n(rows),
                n(heads),
                n(dim),
                n(padded),
            )
        };
        launch.map_err(gpu_error)?;
        let shape = (groups * rows, width, width);
        let residual = self.product(
            &tokens,
            &self.linear(spec, "attn.out_lin")?,
            Some(&query),
            shape,
            false,
        )?;
        let normed = self.rms(
            &residual,
            (spec.tensor)("mlp.0.weight"),
            groups * rows,
            f32::EPSILON,
        )?;
        let hidden = self.product(
            &normed,
            &self.linear(spec, "mlp.1")?,
            None,
            (groups * rows, 2 * width, width),
            true,
        )?;
        let out = self.product(
            &hidden,
            &self.linear(spec, "mlp.3")?,
            Some(&residual),
            (groups * rows, width, 2 * width),
            false,
        )?;
        self.device.download(&out)
    }

    /// Query scaling, and `sqrt(64 / head_dim)` for heads narrower than the
    /// attention kernel's 64, so that it divides by `sqrt(head_dim)`.
    fn scale(
        &self,
        spec: &Spec<'_>,
        queries: &mut DeviceBuffer<f32>,
        (pairs, padded): (usize, usize),
        keys: usize,
    ) -> Result<()> {
        let dim = spec.head_dim();
        #[expect(clippy::cast_precision_loss, reason = "Head dims and key counts are small")]
        let (factor, log) = ((64.0 / dim as f32).sqrt(), (keys.max(1) as f32).ln());
        if spec.scaling == Scaling::None && dim == 64 {
            return Ok(());
        }
        let scales: Vec<f32> = if spec.scaling == Scaling::None {
            vec![factor; spec.heads]
        } else {
            (spec.tensor)("attn.sdpa.query_scaling.head_scale")
                .iter()
                .map(|a| a * log * factor)
                .collect()
        };
        let is_gated = spec.scaling == Scaling::Gated;
        let gates = if is_gated {
            // The gate reads each head's unscaled query; its input pads to 64 with zero weights.
            let name = "attn.sdpa.query_scaling.gate";
            let first = self.dense(format!("{}.{name}.0[64]", spec.name), || {
                let weight = (spec.tensor)(&format!("{name}.0.weight"));
                let padded = weight
                    .chunks(dim)
                    .flat_map(|row| row.iter().copied().chain(std::iter::repeat_n(0.0, 64 - dim)))
                    .collect();
                (padded, (spec.tensor)(&format!("{name}.0.bias")).to_vec())
            })?;
            let hidden = self.product(queries, &first, None, (pairs * padded, 64, 64), true)?;
            self.product(
                &hidden,
                &self.linear(spec, &format!("{name}.2"))?,
                None,
                (pairs * padded, dim, 64),
                false,
            )?
        } else {
            self.zeroed(0)?
        };
        let scales = self.upload(&scales)?;
        // SAFETY: one thread per value of `queries` `[pairs, padded, 64]`, 256 per block;
        // `scales` holds one per head and `gates` `[pairs * padded, dim]` when gated.
        let launch = unsafe {
            self.device.kernels.scale_queries(
                self.device.stream(),
                config((pairs * padded * 64).div_ceil(256), 256),
                queries,
                &scales,
                &gates,
                n(pairs),
                n(padded),
                n(spec.heads),
                n(dim),
                u32::from(is_gated),
            )
        };
        launch.map_err(gpu_error)
    }

    /// `[groups, heads, rows, head_dim]` keys or values as the attention
    /// kernel reads them, bytes of `[groups * heads, padded rows, 64]`: query
    /// head `h` of group `g` gets key head `h / (heads / key heads)` of key
    /// group `g`, or of the one key group.
    fn cache(
        &self,
        kv: &KeyValues<'_>,
        values: &[f32],
        (groups, heads): (usize, usize),
    ) -> Result<DeviceBuffer<u8>> {
        let dim = values.len() / (kv.groups * kv.heads * kv.rows);
        let padded = padded_rows(kv.rows);
        let share = heads / kv.heads;
        let mut out = vec![0.0_f32; groups * heads * padded * 64];
        for group in 0..groups {
            let source_group = if kv.groups == 1 { 0 } else { group };
            for head in 0..heads {
                let source = ((source_group * kv.heads + head / share) * kv.rows) * dim;
                let target = (group * heads + head) * padded * 64;
                for row in 0..kv.rows {
                    out[target + row * 64..][..dim].copy_from_slice(&values[source + row * dim..][..dim]);
                }
            }
        }
        self.device.upload(bytemuck::cast_slice::<f32, u8>(&out))
    }
}

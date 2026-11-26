//! Kumo-Tabular on NVIDIA GPUs: a member's whole forward pass on the device,
//! with resident contexts. Large block products and attention to FP16 caches can use tensor cores.
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
//! stage's sequences. [`workspace`] holds the scratch buffers that a fit or
//! prediction reuses across layers, chunks, and passes on the same stream.

#![expect(unsafe_code, reason = "Launches of kernels that trust their grid and shapes")]

use std::collections::HashMap;
mod workspace;
use std::sync::{Arc, Mutex, PoisonError};
use workspace::{Grow, Scratch, Workspace};

use cuda_core::{DeviceBuffer, DeviceCopy};

use super::Config;
use super::cpu::{Kv, Scaling};
use crate::kernels::cuda::launch::{
    Activation, Dense, HEAD_DIM, Output, Precision, config, n, padded_query_rows, padded_rows, threads,
};
use crate::kernels::cuda::{Device, gpu_error, out_of_memory};
use crate::{Error, Result};

type Buffer = DeviceBuffer<f32>;
type Bytes = DeviceBuffer<u8>;

/// Per layer, each slot's keys and values, before they are stacked.
type Layers = Vec<Vec<(Bytes, Bytes)>>;
/// What fitting records the column keys and values with: the training rows'
/// label vectors on the device, and [`Cuda::kept`] buffers for the keys and
/// values, filled in order.
struct Recording<'a> {
    labels: &'a Buffer,
    slots: std::vec::IntoIter<CacheSlot>,
    invalid: &'a mut Buffer,
    recorded: Vec<(Bytes, Bytes)>,
}

impl Recording<'_> {
    /// `keys` and `values` copied into the next two buffers and recorded.
    fn record(&mut self, device: &Device, keys: &Buffer, values: &Buffer) -> Result<()> {
        let mut next = |source: &Buffer| -> Result<Bytes> {
            self.slots
                .next()
                .expect("fit allocates two buffers per column layer")
                .keep(device, source, self.invalid)
        };
        let pair = (next(keys)?, next(values)?);
        self.recorded.push(pair);
        Ok(())
    }
}

/// A persistent cache allocation, reserved before fit launches its first kernel.
enum CacheSlot {
    Full(Buffer),
    Half(DeviceBuffer<u16>),
}

impl CacheSlot {
    /// Keep a prefix of the FP32 training heads. Narrow only the copy used by query rows.
    fn keep(self, device: &Device, source: &Buffer, invalid: &mut Buffer) -> Result<Bytes> {
        match self {
            Self::Full(mut out) => {
                let len = out.len();
                device.copy_into(source, &mut out, (0, len))?;
                Cuda::bytes(out)
            }
            Self::Half(mut out) => {
                assert!(out.len() <= source.len() && invalid.len() == 1);
                // SAFETY: one thread per retained element, bounded by `out.len()` in the kernel;
                // `source` covers that prefix and `invalid` holds the one atomic flag.
                unsafe {
                    device.kernels.keep_half(
                        device.stream(),
                        config(threads(out.len()), 256),
                        source,
                        &mut out,
                        invalid,
                    )
                }
                .map_err(gpu_error)?;
                Cuda::bytes(out)
            }
        }
    }
}

/// What an allocation that does not fit in device memory fails with; a fit
/// that hits it falls back to the CPU.
pub(crate) const OUT_OF_MEMORY: &str = "CUDA: out of device memory";
/// The block norms' epsilon, `torch.finfo(torch.float32).eps`.
const NORM_EPS: f32 = f32::EPSILON;
/// Cell features per column: 3 grouped columns of 32 sines and 32 cosines,
/// their 3 missing flags, and zeros to a multiple of 16.
const FEATURES: usize = 208;
/// Query rows, over all slots, one prediction pass holds at most.
const PASS_ROWS: usize = 4096;
/// Values a row-stage chunk's sequences hold at most.
const CHUNK_VALUES: usize = 1 << 25;
/// Query scales [`Cuda`] keeps before it drops them all.
const MAX_SCALES: usize = 1024;

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

    /// Row self-attention uses temporary FP32 keys, not a persistent tiled cache.
    fn key_stride(&self, rows: usize) -> usize {
        if self.gate.is_some() {
            rows
        } else {
            padded_rows(rows)
        }
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

/// Every member's context stacked, so one pass predicts them all, as
/// `TabICLv2`'s ensemble cache does: per column layer, keys and values
/// `[columns, slots, heads, inducing points, 64]`; per ICL layer, `[slots, key
/// heads, padded training rows, 64]`. A slot is a member's context, or with
/// codebook tasks one task's.
pub(crate) struct Ensemble {
    /// Each slot's imputation means, column kinds, and kinds as flags.
    inputs: Vec<(Buffer, Buffer, Vec<bool>)>,
    training_rows: usize,
    columns: Vec<(Bytes, Bytes)>,
    icl: Vec<(Bytes, Bytes)>,
}

/// Contexts that predict together: each slot's inputs and the stacked keys
/// and values, laid out as [`Ensemble`]'s. One member's [`Context`] is one slot
/// in the same layout.
struct Slots<'a> {
    inputs: Vec<(&'a Buffer, &'a Buffer, &'a [bool])>,
    training_rows: usize,
    columns: &'a [(Bytes, Bytes)],
    icl: &'a [(Bytes, Bytes)],
}

/// Combine a small regression ensemble before downloading its outputs. Larger ensembles
/// use the same trimmed mean on the host rather than spilling a per-thread sort to memory.
#[derive(Clone, Copy)]
pub(crate) struct Quantiles {
    mean: f64,
    scale: f64,
    negative: u32,
    members: usize,
}

impl Quantiles {
    pub(crate) fn new((mean, scale): (f64, f64), signs: &[f64]) -> Option<Self> {
        if signs.is_empty() || signs.len() > 16 {
            return None;
        }
        let negative = signs
            .iter()
            .enumerate()
            .fold(0, |bits, (i, sign)| bits | (u32::from(*sign < 0.0) << i));
        Some(Self {
            mean,
            scale,
            negative,
            members: signs.len(),
        })
    }

    fn reduce(
        self,
        device: &Device,
        mut raw: Buffer,
        rows: usize,
        reduced: &mut Grow<f32>,
    ) -> Result<Buffer> {
        let members = self.members;
        assert_eq!(raw.len(), members * rows * 999);
        // SAFETY: one 256-thread block owns each complete row of 999 values. The reduction
        // reads those sorted rows in stream order and initializes every output. This final
        // buffer is downloaded immediately, so it cannot accumulate as queued workspace.
        unsafe {
            device
                .kernels
                .kumo_sort_quantiles(device.stream(), config((members * rows, 1, 1), 256), &mut raw)
                .map_err(gpu_error)?;
            let mut out = reduced.prefix(device, rows * 999)?;
            device
                .kernels
                .kumo_trim_quantiles(
                    device.stream(),
                    config(threads(rows * 999), 256),
                    &raw,
                    &mut out,
                    n(rows),
                    n(members),
                    self.negative,
                    self.mean,
                    self.scale,
                )
                .map_err(gpu_error)?;
            Ok(out)
        }
    }
}

impl Context {
    fn slots(&self) -> Slots<'_> {
        Slots {
            inputs: vec![(&self.means, &self.kinds, &self.categorical)],
            training_rows: self.training_rows,
            columns: &self.columns,
            icl: &self.icl,
        }
    }
}

impl Ensemble {
    fn slots(&self) -> Slots<'_> {
        Slots {
            inputs: self
                .inputs
                .iter()
                .map(|(means, kinds, categorical)| (means, kinds, categorical.as_slice()))
                .collect(),
            training_rows: self.training_rows,
            columns: &self.columns,
            icl: &self.icl,
        }
    }
}

/// Kumo-Tabular's GPU: the device and the model's weights on it.
pub(crate) struct Cuda {
    device: Device,
    config: Config,
    outputs: usize,
    weights: Weights,
    precision: Precision,
    /// Precision of resident caches, independently of the products. `Split`
    /// selects FP16 attention; `Half` would select FP8, which cannot share heads.
    cache_precision: Precision,
    /// Serializes operations sharing `self.device.stream()` across concurrent callers.
    stream_lock: Mutex<()>,
    /// Each block's query scale per key count, uploaded on first use: an
    /// upload waits for the GPU.
    scales: Mutex<HashMap<(usize, usize), Arc<Buffer>>>,
    /// Empty optional inputs in `pack_heads` and `heads_to_tokens`.
    empty: Buffer,
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
        release: super::Release,
        optimize: crate::Optimize,
        tensor: &dyn Fn(&str) -> &'a [f32],
    ) -> Option<Self> {
        let device = Device::new().ok().flatten()?;
        let (config, outputs) = (release.size.config(), dim(release.outputs()));
        let precision = Precision::for_optimize(optimize);
        let weights = Weights::upload(&device, &config, outputs, tensor, precision).ok()?;
        let empty = device.zeroed(0).ok()?;
        Some(Self {
            device,
            empty,
            config,
            outputs,
            weights,
            precision,
            cache_precision: if release.half_kv(optimize) {
                Precision::Split
            } else {
                Precision::Fp32
            },
            stream_lock: Mutex::new(()),
            scales: Mutex::default(),
        })
    }

    fn upload<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::from_host(self.device.stream(), values).map_err(device_error)
    }

    /// A context owns its allocations independently of the fit workspace.
    fn kept<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        self.device.zeroed(len).map_err(device_error)
    }

    fn bytes<T: DeviceCopy + bytemuck::NoUninit>(buffer: DeviceBuffer<T>) -> Result<Bytes> {
        buffer
            .cast_chunks::<u8>()
            .map_err(|_| gpu_error("keys and values do not cast to bytes"))
    }

    /// Block widths are multiples of 32. Small products and grids too tall
    /// for the tensor-core kernel use the tiled FP32 fallback.
    fn product_precision(&self, rows: usize) -> Precision {
        if rows <= self.precision.tensor_rows()
            || rows.div_ceil(64) > crate::kernels::cuda::launch::MAX_GRID_YZ
        {
            Precision::Fp32
        } else {
            self.precision
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Mirrors the shared product launch with its scratch buffer"
    )]
    fn product_into(
        &self,
        partials: &mut Grow<f32>,
        x: Activation<'_>,
        dense: &Dense,
        residual: Option<&Buffer>,
        out: Output<'_>,
        (rows, width, depth): (usize, usize, usize),
        gelu: bool,
    ) -> Result<()> {
        let partial = if matches!(x, Activation::F32(_)) {
            let (_, splits) = self.device.product_splits((rows, width, depth));
            if splits > 1 { splits * rows * width } else { 0 }
        } else {
            0
        };
        let outputs = match out {
            Output::F32(y) => (Some(y), None),
            next => (None, Some(next)),
        };
        self.device.product(
            x,
            dense,
            residual,
            outputs,
            partials.reserve(&self.device, partial)?,
            (rows, width, depth),
            gelu,
        )
    }

    fn pack_into(
        &self,
        x: &Buffer,
        (base, rope): (Option<&Buffer>, Option<&Buffer>),
        (sequences, len, padded): (usize, usize, usize),
        (heads, dim): (usize, usize),
        norm: bool,
        out: Output<'_>,
    ) -> Result<()> {
        let none = &self.empty;
        self.device.pack_heads(
            (x, none),
            (base.unwrap_or(none), none, rope.unwrap_or(none)),
            out,
            (sequences, len, padded),
            (heads, dim),
            norm,
        )
    }

    /// A block's keys and values for `groups` groups of `rows` rows of `x`,
    /// each `[groups * heads, padded rows, 64]`.
    fn key_value(
        &self,
        work: &mut Scratch,
        block: &Block,
        x: &Buffer,
        (groups, rows): (usize, usize),
    ) -> Result<(Buffer, Buffer)> {
        let (width, heads, dim) = (block.width, block.heads, block.dim());
        let tokens = groups * rows;
        let precision = self.product_precision(tokens);
        let a = work.a.output(&self.device, precision, tokens * width)?;
        self.device
            .rms_norm(x, &block.key_value_norm, a, (tokens, width), NORM_EPS)?;
        let padded = block.key_stride(rows);
        let len = groups * heads * padded * HEAD_DIM;
        let mut keys = work.keys.prefix(&self.device, len)?;
        let mut values = work.values.prefix(&self.device, len)?;
        let projected = work.projected.reserve(&self.device, tokens * width)?;
        for (dense, out, rope, norm) in [
            (&block.key, &mut keys, block.rope.as_ref(), true),
            (&block.value, &mut values, None, false),
        ] {
            self.product_into(
                &mut work.partial,
                work.a.input(precision),
                dense,
                None,
                Output::F32(projected),
                (tokens, width, width),
                false,
            )?;
            self.pack_into(
                projected,
                (None, rope),
                (groups, rows, padded),
                (heads, dim),
                norm,
                Output::F32(out),
            )?;
        }
        Ok((keys, values))
    }

    /// Training attends to every head, not just the prefix kept for prediction. At `fast`,
    /// run those long ungated products on tensor cores too; `keep` checks both conversions.
    fn training_block(
        &self,
        work: &mut Scratch,
        block: &Block,
        query: &mut Buffer,
        shape: (usize, usize),
        (keys, values, key_rows): (Buffer, Buffer, usize),
        invalid: &mut Buffer,
    ) -> Result<()> {
        let pairs = shape.0 * block.heads;
        let (_, splits) =
            self.device
                .icl_split(Precision::Split, pairs, padded_query_rows(shape.1), key_rows);
        let (keys, values, precision) = if self.precision == Precision::Half
            && pairs * splits <= crate::kernels::cuda::launch::MAX_GRID_YZ
        {
            let mut narrow = |full: Buffer, role: &mut Grow<u16>| {
                let half = CacheSlot::Half(role.prefix(&self.device, full.len())?);
                half.keep(&self.device, &full, invalid)
            };
            (
                narrow(keys, &mut work.half_keys)?,
                narrow(values, &mut work.half_values)?,
                Precision::Split,
            )
        } else {
            (Self::bytes(keys)?, Self::bytes(values)?, Precision::Fp32)
        };
        self.forward_at(
            work,
            block,
            query,
            shape,
            (&keys, &values, block.heads, key_rows),
            precision,
        )
    }

    /// Pack projected queries for attention. For the row stage, return the base
    /// scales and gates to apply when attention reads them.
    fn packed_query(
        &self,
        work: &mut Scratch,
        block: &Block,
        query: &Buffer,
        (groups, rows, padded): (usize, usize, usize),
        key_rows: usize,
        kind: Precision,
    ) -> Result<Option<(Arc<Buffer>, Buffer)>> {
        let (width, heads, dim) = (block.width, block.heads, block.dim());
        let (pairs, tokens) = (groups * heads, groups * rows);
        let precision = self.product_precision(tokens);
        let a = work.a.output(&self.device, precision, tokens * width)?;
        self.device
            .rms_norm(query, &block.query_norm, a, (tokens, width), NORM_EPS)?;
        let projected = work.projected.reserve(&self.device, tokens * width)?;
        self.product_into(
            &mut work.partial,
            work.a.input(precision),
            &block.query,
            None,
            Output::F32(projected),
            (tokens, width, width),
            false,
        )?;
        let base = self.query_scale(block, key_rows)?;
        let rope = block.rope.as_ref();
        if let Some((first, second)) = &block.gate {
            // Row self-attention and its gate both read these FP32 queries.
            assert_eq!(kind, Precision::Fp32);
            let packed = work
                .packed
                .full
                .reserve(&self.device, pairs * padded * HEAD_DIM)?;
            self.pack_into(
                projected,
                (None, rope),
                (groups, rows, padded),
                (heads, dim),
                true,
                Output::F32(packed),
            )?;
            let hidden = work.middle.reserve(&self.device, pairs * padded * 64)?;
            self.product_into(
                &mut work.partial,
                Activation::F32(packed),
                first,
                None,
                Output::F32(hidden),
                (pairs * padded, 64, HEAD_DIM),
                true,
            )?;
            let mut gates = work.projected.prefix(&self.device, pairs * padded * dim)?;
            self.product_into(
                &mut work.partial,
                Activation::F32(hidden),
                second,
                None,
                Output::F32(&mut gates),
                (pairs * padded, dim, 64),
                false,
            )?;
            Ok(Some((base, gates)))
        } else {
            let q = work
                .packed
                .output(&self.device, kind, pairs * padded * HEAD_DIM)?;
            self.pack_into(
                projected,
                (Some(&*base), rope),
                (groups, rows, padded),
                (heads, dim),
                true,
                q,
            )?;
            Ok(None)
        }
    }

    /// Update the queries in place. The last projection can overwrite them:
    /// the preceding projection has already consumed their residual.
    fn forward_at(
        &self,
        work: &mut Scratch,
        block: &Block,
        query: &mut Buffer,
        (groups, rows): (usize, usize),
        (keys, values, key_heads, key_rows): (&Bytes, &Bytes, usize, usize),
        cache: Precision,
    ) -> Result<()> {
        let (width, heads, dim) = (block.width, block.heads, block.dim());
        // The FP32 kernel masks the final query tile itself. In particular, the row gate
        // should project real tokens, not 64 rows for each short feature sequence.
        let (padded, packed) = if cache == Precision::Fp32 {
            (rows, Precision::Fp32)
        } else {
            (padded_query_rows(rows), Precision::Half)
        };
        let (pairs, tokens) = (groups * heads, groups * rows);
        let gate = self.packed_query(work, block, query, (groups, rows, padded), key_rows, packed)?;
        let (_, splits) = self.device.icl_split(cache, pairs, padded, key_rows);
        let chunks = pairs * padded * splits;
        let part = work
            .part
            .reserve(&self.device, if splits > 1 { chunks * HEAD_DIM } else { 0 })?;
        let lse = work.lse.reserve(&self.device, chunks)?;
        let attended = work.attended.reserve(&self.device, pairs * padded * HEAD_DIM)?;
        self.device.attend_icl(
            cache,
            work.packed.input(packed),
            gate.as_ref().map(|(base, gates)| (base.as_ref(), gates, dim)),
            (keys, values),
            (part, lse, attended),
            (pairs, padded, key_rows, block.key_stride(key_rows)),
            heads / key_heads,
        )?;
        let precision = self.product_precision(tokens);
        let merged = work.a.output(&self.device, precision, tokens * width)?;
        self.device.heads_to_tokens(
            (attended, &self.empty),
            merged,
            (groups, rows, padded),
            (heads, dim),
        )?;
        let middle = work.middle.reserve(&self.device, tokens * width)?;
        self.product_into(
            &mut work.partial,
            work.a.input(precision),
            &block.out,
            Some(query),
            Output::F32(middle),
            (tokens, width, width),
            false,
        )?;
        let a = work.a.output(&self.device, precision, tokens * width)?;
        self.device
            .rms_norm(middle, &block.mlp_norm, a, (tokens, width), NORM_EPS)?;
        let wide = work.wide.output(&self.device, precision, tokens * 2 * width)?;
        self.product_into(
            &mut work.partial,
            work.a.input(precision),
            &block.mlp_in,
            None,
            wide,
            (tokens, 2 * width, width),
            true,
        )?;
        self.product_into(
            &mut work.partial,
            work.wide.input(precision),
            &block.mlp_out,
            Some(middle),
            Output::F32(query),
            (tokens, width, 2 * width),
            false,
        )
    }

    /// `block`'s query scale for `key_rows` keys, `[heads, dim]`. The attention
    /// divides by sqrt(64): queries of narrower heads make up for it.
    fn query_scale(&self, block: &Block, key_rows: usize) -> Result<Arc<Buffer>> {
        let key_rows = if block.scaling == Scaling::None {
            0
        } else {
            key_rows
        };
        let key = (std::ptr::from_ref(block) as usize, key_rows);
        let mut scales = self.scales.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(scale) = scales.get(&key) {
            return Ok(Arc::clone(scale));
        }
        let (heads, dim) = (block.heads, block.dim());
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
        if scales.len() >= MAX_SCALES {
            scales.clear();
        }
        let scale = Arc::new(self.upload(&base)?);
        scales.insert(key, Arc::clone(&scale));
        Ok(scale)
    }

    /// `context`'s column and ICL keys and values on the host, without their
    /// padding: per layer `[columns, heads, inducing points, head_dim]` and
    /// `[1, key heads, training rows, head_dim]`, as the CPU keeps them.
    pub(super) fn download(&self, context: &Context) -> Result<(Vec<Kv>, Vec<Kv>)> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(self.download_slots(&context.slots())?.remove(0))
    }

    /// Each slot's keys and values of `ensemble`, as [`Self::download`] gives them.
    pub(super) fn download_ensemble(&self, ensemble: &Ensemble) -> Result<Vec<(Vec<Kv>, Vec<Kv>)>> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        self.download_slots(&ensemble.slots())
    }

    fn download_slots(&self, slots: &Slots<'_>) -> Result<Vec<(Vec<Kv>, Vec<Kv>)>> {
        let config = self.config;
        let (heads, points) = (dim(config.embedding_heads), dim(config.inducing_points));
        let count = slots.inputs.len();
        let groups = slots.inputs[0].2.len();
        let icl_heads = dim(config.icl_heads);
        let key_heads = config.query_key_value_heads.map_or(icl_heads, dim);
        let rows = slots.training_rows;
        let floats = |bytes: &Bytes| -> Result<Vec<f32>> {
            let bytes = self.device.download(bytes)?;
            Ok(if self.cache_precision == Precision::Fp32 {
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&w| f32::from_le_bytes(w))
                    .collect()
            } else {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&w| half::f16::from_le_bytes(w).to_f32())
                    .collect()
            })
        };
        // One slot's `[pairs, padded rows, 64]` values without their padding.
        let unpad = |values: &[f32], (pairs, rows, head_dim): (usize, usize, usize)| -> Vec<f32> {
            let padded = values.len() / (pairs * HEAD_DIM);
            values
                .chunks(padded * HEAD_DIM)
                .flat_map(|pair| pair[..rows * HEAD_DIM].chunks(HEAD_DIM))
                .flat_map(|row| &row[..head_dim])
                .copied()
                .collect()
        };
        let mut out: Vec<(Vec<Kv>, Vec<Kv>)> = (0..count).map(|_| (Vec::new(), Vec::new())).collect();
        let cell_dim = dim(config.cell_channels) / heads;
        for (key, value) in slots.columns {
            let (keys, values) = (floats(key)?, floats(value)?);
            // `[groups, slots, chunk]`: slot `s` is every group's chunk `s`.
            let chunk = keys.len() / (groups * count);
            let slot = |all: &[f32], s: usize| -> Vec<f32> {
                let mine: Vec<f32> = all
                    .chunks(chunk)
                    .skip(s)
                    .step_by(count)
                    .flatten()
                    .copied()
                    .collect();
                unpad(&mine, (groups * heads, points, cell_dim))
            };
            for (s, (columns, _)) in out.iter_mut().enumerate() {
                columns.push(Kv {
                    key: slot(&keys, s),
                    value: super::cpu::Values::F32(slot(&values, s)),
                    groups,
                    heads,
                    rows: points,
                });
            }
        }
        let icl_dim = dim(config.icl_channels) / icl_heads;
        for (key, value) in slots.icl {
            let (keys, values) = (floats(key)?, floats(value)?);
            let len = keys.len() / count;
            for (s, (_, icl)) in out.iter_mut().enumerate() {
                icl.push(Kv {
                    key: unpad(&keys[s * len..][..len], (key_heads, rows, icl_dim)),
                    value: super::cpu::Values::F32(unpad(
                        &values[s * len..][..len],
                        (key_heads, rows, icl_dim),
                    )),
                    groups: 1,
                    heads: key_heads,
                    rows,
                });
            }
        }
        Ok(out)
    }

    /// Whether stacking `contexts` fits in free device memory: stacking holds
    /// its largest layer's stacked keys and values on top of the contexts.
    pub(crate) fn stack_fits(&self, contexts: &[&Context]) -> Result<bool> {
        // The FP16 kernel launches one y block per (query head, split). Splits exceed
        // one only for small grids; a larger ensemble can still predict member by member.
        if self.cache_precision != Precision::Fp32
            && let Some(first) = contexts.first()
        {
            let pairs =
                dim(self.config.icl_heads).max(first.categorical.len() * dim(self.config.embedding_heads));
            if contexts.len() > crate::kernels::cuda::launch::MAX_GRID_YZ / pairs {
                return Ok(false);
            }
        }
        let largest = contexts.first().map_or(0, |context| {
            context
                .columns
                .iter()
                .chain(&context.icl)
                .map(|(keys, _)| keys.len())
                .max()
                .unwrap_or(0)
        });
        let needed = 2 * contexts.len() * largest;
        Ok(self.device.free_memory()? >= needed + needed / 4)
    }

    /// `contexts`, which have the same columns and training rows, stacked
    /// into one [`Ensemble`]. Each layer's buffers are freed once stacked, so
    /// the device holds at most one layer twice.
    pub(crate) fn stack(&self, contexts: Vec<Context>) -> Result<Ensemble> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        let count = contexts.len();
        if count == 0 {
            return Err(Error::InvalidArgument("cannot stack zero contexts".into()));
        }
        let (groups, training_rows) = (contexts[0].categorical.len(), contexts[0].training_rows);
        if contexts
            .iter()
            .any(|context| context.categorical.len() != groups || context.training_rows != training_rows)
        {
            return Err(Error::InvalidArgument(
                "stacked contexts must have the same columns and training rows".into(),
            ));
        }
        let mut inputs = Vec::with_capacity(count);
        let (mut columns, mut icl): (Layers, Layers) = (Vec::new(), Vec::new());
        for context in contexts {
            inputs.push((context.means, context.kinds, context.categorical));
            for (layers, kvs) in [(&mut columns, context.columns), (&mut icl, context.icl)] {
                layers.resize_with(kvs.len(), Vec::new);
                for (layer, kv) in layers.iter_mut().zip(kvs) {
                    layer.push(kv);
                }
            }
        }
        let words = |bytes: Bytes| {
            bytes
                .cast_chunks::<u32>()
                .map_err(|_| gpu_error("packed head is not whole words"))
        };
        // Each slot's `[groups, chunk]` into `[groups, slots, chunk]`, or with one
        // group its slot's place in `[slots, ...]`.
        let stack_layer = |parts: Vec<(Bytes, Bytes)>, groups: usize| -> Result<(Bytes, Bytes)> {
            let len = parts[0].0.len() / size_of::<u32>();
            // Every padded head is a multiple of four bytes, including FP16 heads. The
            // interleave kernel only copies these 32-bit words, without doing arithmetic.
            let (mut keys, mut values) = (self.kept::<u32>(count * len)?, self.kept::<u32>(count * len)?);
            for (slot, (key, value)) in parts.into_iter().enumerate() {
                for (source, out) in [(key, &mut keys), (value, &mut values)] {
                    let source = words(source)?;
                    let chunk = len / groups;
                    assert!(source.len() == len && out.len() == count * len && slot < count);
                    // SAFETY: one thread per value of `source` `[groups, chunk]`; `out` holds
                    // `[groups, count, chunk]` and `slot < count`, as asserted.
                    let launch = unsafe {
                        self.device.kernels.kumo_interleave(
                            self.device.stream(),
                            config(threads(len), 256),
                            &source,
                            out,
                            n(groups),
                            n(count),
                            n(slot),
                            n(chunk),
                        )
                    };
                    launch.map_err(gpu_error)?;
                }
            }
            Ok((Self::bytes(keys)?, Self::bytes(values)?))
        };
        Ok(Ensemble {
            inputs,
            training_rows,
            columns: columns
                .into_iter()
                .map(|parts| stack_layer(parts, groups))
                .collect::<Result<_>>()?,
            icl: icl
                .into_iter()
                .map(|parts| stack_layer(parts, 1))
                .collect::<Result<_>>()?,
        })
    }

    /// A device context from a host one with `training_rows` training rows,
    /// its keys and values padded as [`Self::fit`] keeps them.
    pub(crate) fn upload_context(
        &self,
        context: &super::cpu::Context,
        training_rows: usize,
    ) -> Result<Context> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        let (columns, icl) = context.kvs();
        let upload = |kvs: &[Kv]| {
            kvs.iter()
                .map(|kv| {
                    let head_dim = kv.key.len() / (kv.groups * kv.heads * kv.rows).max(1);
                    let padded = padded_rows(kv.rows);
                    let pad = |values: &[f32]| -> Result<Bytes> {
                        let mut out = vec![0.0_f32; kv.groups * kv.heads * padded * HEAD_DIM];
                        for (pair, source) in values.chunks(kv.rows * head_dim).enumerate() {
                            for (row, source) in source.chunks(head_dim).enumerate() {
                                out[(pair * padded + row) * HEAD_DIM..][..head_dim].copy_from_slice(source);
                            }
                        }
                        if self.cache_precision == Precision::Fp32 {
                            Self::bytes(self.upload(&out)?)
                        } else {
                            Self::bytes(self.upload(&super::half_cache(&out)?)?)
                        }
                    };
                    Ok((pad(&kv.key)?, pad(&kv.value.floats())?))
                })
                .collect::<Result<Vec<_>>>()
        };
        let (categorical, means) = context.columns_and_means();
        let kinds: Vec<f32> = categorical.iter().map(|&c| f32::from(u8::from(c))).collect();
        Ok(Context {
            means: self.upload(means)?,
            kinds: self.upload(&kinds)?,
            categorical: categorical.to_vec(),
            training_rows,
            columns: upload(columns)?,
            icl: upload(icl)?,
        })
    }

    /// Fit a member's context on its `[rows, columns]` training rows `x` and
    /// their label vectors for the cells and for the ICL stage, `[rows, cell]`
    /// and `[rows, icl]`.
    pub(crate) fn fit(&self, x: &[f32], categorical: &[bool], labels: (&[f32], &[f32])) -> Result<Context> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        let columns = categorical.len();
        let rows = x.len() / columns;
        let cell_values = columns
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(dim(self.config.cell_channels)));
        let icl_values = rows.checked_mul(dim(self.config.icl_channels));
        if cell_values.is_none_or(|v| v > u32::MAX as usize)
            || icl_values.is_none_or(|v| v > u32::MAX as usize)
        {
            return Err(Error::UnsupportedBackend(OUT_OF_MEMORY.into()));
        }
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
        let config = self.config;
        let key_heads = config.query_key_value_heads.map_or(dim(config.icl_heads), dim);
        // Uploads and the context's buffers wait for the GPU: they come before any work is queued.
        let (cell_labels, label_rows) = (self.upload(labels.0)?, self.upload(labels.1)?);
        let kept = |count: usize, len: usize| {
            (0..count)
                .map(|_| {
                    if self.cache_precision == Precision::Fp32 {
                        self.kept(len).map(CacheSlot::Full)
                    } else {
                        self.kept(len).map(CacheSlot::Half)
                    }
                })
                .collect::<Result<Vec<_>>>()
        };
        let mut invalid = self.kept(usize::from(self.cache_precision != Precision::Fp32))?;
        let column_len =
            columns * dim(config.embedding_heads) * padded_rows(dim(config.inducing_points)) * HEAD_DIM;
        let prefix = key_heads * padded_rows(rows) * HEAD_DIM;
        let mut recording = Recording {
            labels: &cell_labels,
            slots: kept(2 * self.weights.columns.len(), column_len)?.into_iter(),
            invalid: &mut invalid,
            recorded: Vec::new(),
        };
        let mut icl_slots = kept(2 * self.weights.icl.len(), prefix)?.into_iter();
        let mut work = Workspace::default();
        let mut hidden = self.row_embedding(&mut work, x, &context, &mut recording)?;
        context.columns = recording.recorded;
        let icl_width = dim(config.icl_channels);
        self.add_rows(&mut hidden, &label_rows, (rows, icl_width))?;
        for (layer, block) in self.weights.icl.iter().enumerate() {
            let (keys, values) = self.key_value(&mut work.block, block, &hidden, (1, rows))?;
            let mut slot = || icl_slots.next().expect("fit allocates two buffers per ICL layer");
            let kept = (
                slot().keep(&self.device, &keys, &mut invalid)?,
                slot().keep(&self.device, &values, &mut invalid)?,
            );
            if layer + 1 < self.weights.icl.len() {
                self.training_block(
                    &mut work.block,
                    block,
                    &mut hidden,
                    (1, rows),
                    (keys, values, rows),
                    &mut invalid,
                )?;
            }
            context.icl.push(kept);
        }
        if !invalid.is_empty() && self.device.download(&invalid)?[0] != 0.0 {
            return Err(gpu_error("cached keys or values overflow FP16"));
        }
        Ok(context)
    }

    /// The outputs of each of the `[rows, columns]` query rows `x`.
    pub(crate) fn predict(&self, context: &Context, x: &[f32]) -> Result<Vec<f32>> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(self.predict_slots(&context.slots(), &[x], None)?.remove(0))
    }

    /// Each slot's outputs of its `[rows, columns]` query rows `xs[slot]`, all
    /// slots in one pass, a chunk of rows at a time.
    pub(crate) fn predict_ensemble(
        &self,
        ensemble: &Ensemble,
        xs: &[&[f32]],
        quantiles: Option<Quantiles>,
    ) -> Result<Vec<Vec<f32>>> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        self.predict_slots(&ensemble.slots(), xs, quantiles)
    }

    fn predict_slots(
        &self,
        slots: &Slots<'_>,
        xs: &[&[f32]],
        quantiles: Option<Quantiles>,
    ) -> Result<Vec<Vec<f32>>> {
        let columns = slots.inputs[0].2.len();
        let rows = xs[0].len() / columns;
        if xs.len() != slots.inputs.len() || xs.iter().any(|x| x.len() != rows * columns) {
            return Err(Error::InvalidArgument(
                "each slot needs the same number of query rows".into(),
            ));
        }
        let step = (PASS_ROWS / xs.len()).clamp(1, rows.max(1));
        let outputs = if quantiles.is_some() { 1 } else { xs.len() };
        let mut out: Vec<Vec<f32>> = (0..outputs)
            .map(|_| Vec::with_capacity(rows * self.outputs))
            .collect();
        let mut work = Workspace::default();
        for start in (0..rows).step_by(step) {
            let count = step.min(rows - start);
            let chunk: Vec<&[f32]> = xs
                .iter()
                .map(|x| &x[start * columns..][..count * columns])
                .collect();
            let raw = self.pass(&mut work, slots, &chunk)?;
            let result = match quantiles {
                Some(q) => q.reduce(&self.device, raw, count, &mut work.reduced)?,
                None => raw,
            };
            let host = self.device.download(&result)?;
            for (all, part) in out.iter_mut().zip(host.chunks_exact(count * self.outputs)) {
                all.extend_from_slice(part);
            }
        }
        if out.iter().any(|slot| slot.iter().any(|value| !value.is_finite())) {
            return Err(gpu_error("CUDA prediction produced a non-finite value"));
        }
        Ok(out)
    }

    /// One pass over every slot's `[rows, columns]` query rows `xs[slot]`:
    /// products, norms, and attention run on all slots' rows at once, each
    /// slot attending to its own keys and values.
    fn pass(&self, work: &mut Workspace, slots: &Slots<'_>, xs: &[&[f32]]) -> Result<Buffer> {
        let config = self.config;
        let (count, columns) = (xs.len(), slots.inputs[0].2.len());
        let rows = xs[0].len() / columns;
        let total = count * rows;
        let (cell, readout) = (dim(config.cell_channels), dim(config.readout_tokens));
        // `[columns, slots, rows, cell]`, so each (column, slot) group is contiguous.
        let mut cells = work.cells.prefix(&self.device, columns * total * cell)?;
        for (slot, (x, inputs)) in xs.iter().zip(&slots.inputs).enumerate() {
            self.embed_cells(work, x, *inputs, &mut cells, (slot, count))?;
        }
        let mut readouts = work.readouts.prefix(&self.device, total * readout * cell)?;
        self.device.repeat(
            &self.weights.readout,
            &mut readouts,
            (readout * cell, total * readout * cell),
        )?;
        let points = dim(config.inducing_points);
        for (layer, ((_, output), (keys, values))) in
            self.weights.columns.iter().zip(slots.columns).enumerate()
        {
            self.forward_at(
                &mut work.block,
                output,
                &mut cells,
                (columns * count, rows),
                (keys, values, output.heads, points),
                self.cache_precision,
            )?;
            self.row_stage(work, layer, &mut readouts, &mut cells, (total, columns))?;
        }
        let mut hidden = self.row_hidden(work, &readouts, total)?;
        let icl = dim(config.icl_channels);
        let key_heads = config.query_key_value_heads.map_or(dim(config.icl_heads), dim);
        for (block, (keys, values)) in self.weights.icl.iter().zip(slots.icl) {
            self.forward_at(
                &mut work.block,
                block,
                &mut hidden,
                (count, rows),
                (keys, values, key_heads, slots.training_rows),
                self.cache_precision,
            )?;
        }
        let kind = if self.precision == Precision::Half {
            self.product_precision(total)
        } else {
            Precision::Fp32
        };
        let block = &mut work.block;
        let a = block.a.output(&self.device, kind, total * icl)?;
        self.device
            .rms_norm(&hidden, &self.weights.icl_norm, a, (total, icl), NORM_EPS)?;
        let wide = block.wide.output(&self.device, kind, total * 2 * icl)?;
        self.product_into(
            &mut block.partial,
            block.a.input(kind),
            &self.weights.head.0,
            None,
            wide,
            (total, 2 * icl, icl),
            true,
        )?;
        let mut logits = work.logits.prefix(&self.device, total * self.outputs)?;
        self.product_into(
            &mut block.partial,
            block.wide.input(kind),
            &self.weights.head.1,
            None,
            Output::F32(&mut logits),
            (total, self.outputs, 2 * icl),
            false,
        )?;
        Ok(logits)
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

    /// The row embedding of the `[rows, columns]` training rows `x`, `[rows,
    /// icl]`, with their `[rows, cell]` label vectors, recording each column
    /// layer's keys and values.
    fn row_embedding(
        &self,
        work: &mut Workspace,
        x: &[f32],
        context: &Context,
        recording: &mut Recording<'_>,
    ) -> Result<Buffer> {
        let config = self.config;
        let columns = context.categorical.len();
        let rows = x.len() / columns;
        let (cell, readout) = (dim(config.cell_channels), dim(config.readout_tokens));
        let mut cells = work.cells.prefix(&self.device, columns * rows * cell)?;
        let inputs = (&context.means, &context.kinds, context.categorical.as_slice());
        self.embed_cells(work, x, inputs, &mut cells, (0, 1))?;
        self.add_rows(&mut cells, recording.labels, (rows, cell))?;
        let mut readouts = work.readouts.prefix(&self.device, rows * readout * cell)?;
        self.device.repeat(
            &self.weights.readout,
            &mut readouts,
            (readout * cell, rows * readout * cell),
        )?;
        let points = dim(config.inducing_points);
        for (layer, (inducing, output)) in self.weights.columns.iter().enumerate() {
            let mut queries = work.queries.prefix(&self.device, columns * points * cell)?;
            self.device.repeat(
                &self.weights.inducing[layer],
                &mut queries,
                (points * cell, columns * points * cell),
            )?;
            let (keys, values) = self.key_value(&mut work.block, inducing, &cells, (columns, rows))?;
            self.training_block(
                &mut work.block,
                inducing,
                &mut queries,
                (columns, points),
                (keys, values, rows),
                recording.invalid,
            )?;
            let (keys, values) = self.key_value(&mut work.block, output, &queries, (columns, points))?;
            recording.record(&self.device, &keys, &values)?;
            self.training_block(
                &mut work.block,
                output,
                &mut cells,
                (columns, rows),
                (keys, values, points),
                recording.invalid,
            )?;
            self.row_stage(work, layer, &mut readouts, &mut cells, (rows, columns))?;
        }
        self.row_hidden(work, &readouts, rows)
    }

    fn row_hidden(&self, work: &mut Workspace, readouts: &Buffer, rows: usize) -> Result<Buffer> {
        let (cell, readout, icl) = (
            dim(self.config.cell_channels),
            dim(self.config.readout_tokens),
            dim(self.config.icl_channels),
        );
        let width = cell * readout;
        let mut hidden = work.hidden.prefix(&self.device, rows * icl)?;
        let normalize = |out| {
            self.device.rms_norm(
                readouts,
                &self.weights.norm,
                Output::F32(out),
                (rows * readout, cell),
                NORM_EPS,
            )
        };
        if let Some(project) = &self.weights.row_project {
            let normed = work.sequences.reserve(&self.device, rows * width)?;
            normalize(normed)?;
            self.product_into(
                &mut work.block.partial,
                Activation::F32(normed),
                project,
                None,
                Output::F32(&mut hidden),
                (rows, icl, width),
                false,
            )?;
        } else {
            assert_eq!(width, icl);
            normalize(&mut hidden)?;
        }
        Ok(hidden)
    }

    /// The cell embedding of slot `slot` of `slots`, its `[rows, columns]`
    /// rows `x` with its imputation means, column kinds, and kinds as flags,
    /// into `cells` `[columns, slots, rows, cell]`.
    fn embed_cells(
        &self,
        work: &mut Workspace,
        x: &[f32],
        (means, kinds, categorical): (&Buffer, &Buffer, &[bool]),
        cells: &mut Buffer,
        (slot, slots): (usize, usize),
    ) -> Result<()> {
        let columns = categorical.len();
        let rows = x.len() / columns;
        let cell = dim(self.config.cell_channels);
        let mut input = work.input.prefix(&self.device, x.len())?;
        input
            .copy_from_host(self.device.stream(), x)
            .map_err(device_error)?;
        let pattern = |column| {
            (0..3).fold(0, |bits, g| {
                bits | usize::from(categorical[(column + (1 << g) - 1) % columns]) << g
            })
        };
        // Adjacent columns with the same kinds use the same projection. Bound each feature
        // workspace to 4 MiB (or one column for taller training tables).
        let limit = if self.precision == Precision::Half {
            ((1 << 20) / (rows * FEATURES)).max(1)
        } else {
            1
        };
        let mut column = 0;
        while column < columns {
            let kind = pattern(column);
            let group = 1
                + ((column + 1)..columns.min(column + limit))
                    .take_while(|&c| pattern(c) == kind)
                    .count();
            let tokens = group * rows;
            let mut features = work.features.prefix(&self.device, tokens * FEATURES)?;
            // SAFETY: one thread per value of [group, rows, FEATURES]; column + group <= columns;
            // x holds [rows, columns], means and kinds hold columns, and each frequency table 3 * 32.
            let launch = unsafe {
                self.device.kernels.kumo_cell_features(
                    self.device.stream(),
                    config(threads(tokens * FEATURES), 256),
                    &input,
                    means,
                    kinds,
                    &self.weights.numeric,
                    &self.weights.categorical,
                    &mut features,
                    n(rows),
                    n(columns),
                    n(column),
                )
            };
            launch.map_err(gpu_error)?;
            let mut embedded = work.embedded.prefix(&self.device, tokens * cell)?;
            self.product_into(
                &mut work.block.partial,
                Activation::F32(&features),
                &self.weights.patterns[kind],
                None,
                Output::F32(&mut embedded),
                (tokens, cell, FEATURES),
                false,
            )?;
            let chunk = rows * cell;
            let offset = (column * slots + slot) * chunk;
            assert!(offset + ((group - 1) * slots + 1) * chunk <= cells.len());
            // SAFETY: the checked destination covers the final strided column and all prior ones.
            unsafe {
                self.device.kernels.kumo_store_cells(
                    self.device.stream(),
                    config(threads(embedded.len()), 256),
                    &embedded,
                    &mut *cells,
                    n(offset),
                    n(slots),
                    n(chunk),
                )
            }
            .map_err(gpu_error)?;
            column += group;
        }
        Ok(())
    }

    /// Row stage `layer` on `readouts` `[rows, readout, cell]` and `cells`
    /// `[columns, rows, cell]`, a chunk of rows at a time.
    fn row_stage(
        &self,
        work: &mut Workspace,
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
            let mut sequences = work.sequences.prefix(&self.device, count * tokens * cell)?;
            self.sequences(
                readouts,
                cells,
                &mut sequences,
                (start, count, rows, tokens),
                false,
            )?;
            let (keys, values) = self.key_value(&mut work.block, block, &sequences, (count, tokens))?;
            let (keys, values) = (Self::bytes(keys)?, Self::bytes(values)?);
            let kv = (&keys, &values, block.heads, tokens);
            let (mut queries, query_tokens) = if last {
                let mut queries = work.queries.prefix(&self.device, count * readout * cell)?;
                self.sequences(
                    readouts,
                    cells,
                    &mut queries,
                    (start, count, rows, readout),
                    false,
                )?;
                (queries, readout)
            } else {
                (sequences, tokens)
            };
            self.forward_at(
                &mut work.block,
                block,
                &mut queries,
                (count, query_tokens),
                kv,
                Precision::Fp32,
            )?;
            self.sequences(
                readouts,
                cells,
                &mut queries,
                (start, count, rows, query_tokens),
                true,
            )?;
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
        assert!(cell.is_multiple_of(4) && start + count <= rows && sequences.len() >= count * tokens * cell);
        assert!(readouts.len() >= rows * readout * cell && cells.len() >= (tokens - readout) * rows * cell);
        // SAFETY: one thread copies four aligned values of `sequences` `[count, tokens, cell]`; `readouts` holds
        // `[rows, readout, cell]` and `cells` `[tokens - readout, rows, cell]` (`tokens == readout`
        // reads no cell), and `start + count <= rows`, all asserted.
        let launch = unsafe {
            self.device.kernels.kumo_sequences(
                self.device.stream(),
                config(threads(count * tokens * (cell / 4)), 256),
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
        precision: Precision,
    ) -> Result<Self> {
        let up = |values: &[f32]| device.upload(values);
        let dense = |weight: &[f32], bias: &[f32]| {
            Dense::for_precision(
                device,
                weight,
                bias,
                if precision == Precision::Half {
                    precision
                } else {
                    Precision::Fp32
                },
            )
        };
        let linear = |prefix: &str| {
            dense(
                tensor(&format!("{prefix}.weight")),
                tensor(&format!("{prefix}.bias")),
            )
        };
        let (cell, icl) = (dim(config.cell_channels), dim(config.icl_channels));
        let block = |prefix: &str, width: usize, heads: usize, scaling: Scaling, rope: bool| {
            Block::upload(device, tensor, prefix, (width, heads), (scaling, rope), precision)
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
        precision: Precision,
    ) -> Result<Self> {
        let up = |name: &str| device.upload(tensor(&format!("{prefix}.{name}")));
        let linear = |name: &str, precision| {
            Dense::for_precision(
                device,
                tensor(&format!("{prefix}.{name}.weight")),
                tensor(&format!("{prefix}.{name}.bias")),
                precision,
            )
        };
        let (qkv_weight, qkv_bias) = (
            tensor(&format!("{prefix}.attn.qkv_lin.weight")),
            tensor(&format!("{prefix}.attn.qkv_lin.bias")),
        );
        let part = |p: usize| {
            Dense::for_precision(
                device,
                &qkv_weight[p * width * width..][..width * width],
                &qkv_bias[p * width..][..width],
                precision,
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
                linear("attn.sdpa.query_scaling.gate.2", Precision::Fp32)?,
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
            out: linear("attn.out_lin", precision)?,
            mlp_norm: up("mlp.0.weight")?,
            mlp_in: linear("mlp.1", precision)?,
            mlp_out: linear("mlp.3", precision)?,
            head_scale: if scaling == Scaling::None {
                Vec::new()
            } else {
                tensor(&format!("{prefix}.attn.sdpa.query_scaling.head_scale")).to_vec()
            },
            scaling,
            gate,
            rope: if rope {
                // The kernels rotate queries and keys with one table: the checkpoint's two
                // must be equal, or the model runs on the CPU, which applies each.
                let (query, key) = (
                    tensor(&format!("{prefix}.attn.query_transform.0.inv_freq")),
                    tensor(&format!("{prefix}.attn.key_transform.0.inv_freq")),
                );
                if query != key {
                    return Err(Error::InvalidArgument(format!(
                        "{prefix}'s query and key RoPE frequencies differ"
                    )));
                }
                Some(up("attn.query_transform.0.inv_freq")?)
            } else {
                None
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheSlot, Device};

    #[test]
    fn device_quantiles_sort_flip_trim_and_reject_invalid_members() {
        let Some(device) = Device::new().unwrap() else {
            return;
        };
        let signs: Vec<_> = (0..16).map(|m| if m % 2 == 0 { -1.0 } else { 1.0 }).collect();
        assert!(super::Quantiles::new((0.0, 1.0), &[1.0; 17]).is_none());
        for members in [1, 4, 5, 16] {
            let (mean, scale) = (9.0, 2.5);
            let q = super::Quantiles::new((mean, scale), &signs[..members]).unwrap();
            let mut raw = Vec::new();
            for (m, sign) in signs[..members].iter().enumerate() {
                for row in 0..2 {
                    for j in 0..999 {
                        // Multiplication by 17 permutes the 999 ranks. After flipping, the
                        // members' rows differ only by a known increasing member offset.
                        let v = i16::try_from((j * 17) % 999 + row * 1000 + m * m).unwrap();
                        raw.push(if *sign < 0.0 { -f32::from(v) } else { f32::from(v) });
                    }
                }
            }
            let reduce = |raw: &[f32]| {
                device
                    .download(
                        &q.reduce(
                            &device,
                            device.upload(raw).unwrap(),
                            2,
                            &mut super::Grow::default(),
                        )
                        .unwrap(),
                    )
                    .unwrap()
            };
            let values = reduce(&raw);
            let cut = members / 5;
            let kept = cut..members - cut;
            let offset = f64::from(u32::try_from(kept.clone().map(|m| m * m).sum::<usize>()).unwrap())
                / f64::from(u32::try_from(kept.len()).unwrap());
            for (i, &v) in values.iter().enumerate() {
                let rank = u16::try_from(i / 999 * 1000 + i % 999).unwrap();
                #[expect(clippy::cast_possible_truncation, reason = "The head returns FP32")]
                let expected = ((f64::from(rank) + offset) * scale + mean) as f32;
                assert_eq!(v, expected);
            }
            raw[42] = f32::NAN;
            assert!(reduce(&raw).iter().any(|v| v.is_nan()));
        }
    }

    #[test]
    fn narrowing_checks_only_retained_heads_and_reports_overflow() {
        let Some(device) = Device::new().unwrap() else {
            return;
        };
        let source = device.upload(&[1.25_f32, -65_504.0, 70_000.0, f32::NAN]).unwrap();
        for (len, valid) in [(2, true), (3, false), (4, false)] {
            let slot = CacheSlot::Half(device.zeroed(len).unwrap());
            let mut flag = device.zeroed(1).unwrap();
            let kept = slot.keep(&device, &source, &mut flag).unwrap();
            assert_eq!(device.download(&flag).unwrap()[0] == 0.0, valid);
            if valid {
                let host = super::super::half_cache(&[1.25, -65_504.0]).unwrap();
                assert_eq!(
                    device.download(&kept).unwrap(),
                    bytemuck::cast_slice::<u16, u8>(&host)
                );
            }
        }
    }
}

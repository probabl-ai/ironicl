//! Kumo-Tabular on NVIDIA GPUs: a member's whole forward pass on the device,
//! with FP32 attention and resident contexts. Large block products can use tensor cores.
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

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

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

/// A block product's operand. Its producer writes the required representation
/// directly: no separate rounding kernel or temporary FP32 copy.
enum Operand {
    F32(Buffer),
    Half(DeviceBuffer<u16>),
    Split(DeviceBuffer<u16>, DeviceBuffer<u16>),
}

impl Operand {
    fn input(&self) -> Activation<'_> {
        match self {
            Self::F32(x) => Activation::F32(x),
            Self::Half(hi) => Activation::Half(hi),
            Self::Split(hi, lo) => Activation::Split { hi, lo },
        }
    }

    fn output(&mut self) -> Output<'_> {
        match self {
            Self::F32(x) => Output::F32(x),
            Self::Half(hi) => Output::Half(hi),
            Self::Split(hi, lo) => Output::Split { hi, lo },
        }
    }
}

/// Per layer, each slot's keys and values, before they are stacked.
type Layers = Vec<Vec<(Bytes, Bytes)>>;
/// What fitting records the column keys and values with: the training rows'
/// label vectors on the device, and [`Cuda::kept`] buffers for the keys and
/// values, filled in order.
struct Recording<'a> {
    labels: &'a Buffer,
    slots: std::vec::IntoIter<Buffer>,
    recorded: Vec<(Bytes, Bytes)>,
}

impl Recording<'_> {
    /// `keys` and `values` copied into the next two buffers and recorded.
    fn record(&mut self, device: &Device, keys: &Buffer, values: &Buffer) -> Result<()> {
        let mut next = |source: &Buffer| -> Result<Bytes> {
            let mut kept = self
                .slots
                .next()
                .expect("fit allocates two buffers per column layer");
            device.copy_into(source, &mut kept, (0, source.len()))?;
            Cuda::bytes(kept)
        };
        let pair = (next(keys)?, next(values)?);
        self.recorded.push(pair);
        Ok(())
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
/// Bytes of buffers allocated ahead of the GPU before [`Cuda`] waits for it.
/// A buffer is freed only once the work using it completes, so this bounds
/// the memory of queued work.
const QUEUED_BYTES: usize = 1 << 30;
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
    /// Serializes operations sharing `self.device.stream()` across concurrent callers.
    stream_lock: Mutex<()>,
    /// Each block's query scale per key count, uploaded on first use: an
    /// upload waits for the GPU.
    scales: Mutex<HashMap<(usize, usize), Arc<Buffer>>>,
    /// Bytes allocated since the last wait for the GPU.
    queued: AtomicUsize,
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
        optimize: crate::Optimize,
        tensor: &dyn Fn(&str) -> &'a [f32],
    ) -> Option<Self> {
        let device = Device::new().ok().flatten()?;
        let precision = Precision::for_optimize(optimize);
        let weights = Weights::upload(&device, &config, outputs, tensor, precision).ok()?;
        Some(Self {
            device,
            config,
            outputs,
            weights,
            precision,
            stream_lock: Mutex::new(()),
            scales: Mutex::default(),
            queued: AtomicUsize::new(0),
        })
    }

    /// `len` zeros, allocated in stream order without waiting for the GPU,
    /// unless [`QUEUED_BYTES`] have been allocated since it last did.
    fn zeros<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(Error::ArithmeticOverflow("CUDA buffer"))?;
        if self.queued.fetch_add(bytes, Ordering::Relaxed) + bytes > QUEUED_BYTES {
            self.queued.store(0, Ordering::Relaxed);
            self.device.stream().synchronize().map_err(gpu_error)?;
        }
        self.device.zeroed_async(len).map_err(device_error)
    }

    fn upload<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::from_host(self.device.stream(), values).map_err(device_error)
    }

    /// `len` zeros for a context, outside the stream-ordered pool: a long-lived
    /// buffer there would keep the pool from reusing its memory. Unlike
    /// [`Self::zeros`], it waits for the GPU.
    fn kept(&self, len: usize) -> Result<Buffer> {
        self.device.zeroed(len).map_err(device_error)
    }

    fn bytes(buffer: Buffer) -> Result<Bytes> {
        buffer
            .cast_chunks::<u8>()
            .map_err(|_| gpu_error("keys and values do not cast to bytes"))
    }

    /// Block widths are multiples of 32. Small products and grids too tall
    /// for the tensor-core kernel use the tiled FP32 fallback.
    fn operand(&self, rows: usize, width: usize) -> Result<Operand> {
        let len = rows * width;
        if rows <= self.precision.tensor_rows()
            || rows.div_ceil(64) > crate::kernels::cuda::launch::MAX_GRID_YZ
        {
            return Ok(Operand::F32(self.zeros(len)?));
        }
        match self.precision {
            Precision::Split => Ok(Operand::Split(self.zeros(len)?, self.zeros(len)?)),
            Precision::Half => Ok(Operand::Half(self.zeros(len)?)),
            Precision::Fp32 => unreachable!("FP32 has no tensor-core row threshold"),
        }
    }

    /// `act(x wᵀ + b) (+ residual)`, with an FP32 result.
    fn product(
        &self,
        x: Activation<'_>,
        dense: &Dense,
        residual: Option<&Buffer>,
        (rows, width, depth): (usize, usize, usize),
        gelu: bool,
    ) -> Result<Buffer> {
        let mut out = self.zeros(rows * width)?;
        self.product_into(
            x,
            dense,
            residual,
            Output::F32(&mut out),
            (rows, width, depth),
            gelu,
        )?;
        Ok(out)
    }

    fn product_into(
        &self,
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
            &mut self.zeros(partial)?,
            (rows, width, depth),
            gelu,
        )
    }

    fn block_norm(&self, x: &Buffer, gamma: &Buffer, rows: usize) -> Result<Operand> {
        let width = gamma.len();
        let mut out = self.operand(rows, width)?;
        self.device
            .rms_norm(x, gamma, out.output(), (rows, width), NORM_EPS)?;
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
        let normed = self.block_norm(x, &block.key_value_norm, groups * rows)?;
        let shape = (groups * rows, width, width);
        let keys = self.product(normed.input(), &block.key, None, shape, false)?;
        let values = self.product(normed.input(), &block.value, None, shape, false)?;
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
        let normed = self.block_norm(query, &block.query_norm, tokens)?;
        let projected = self.product(normed.input(), &block.query, None, (tokens, width, width), false)?;
        let base = self.query_scale(block, key_rows)?;
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
            let hidden = self.product(
                Activation::F32(&packed),
                first,
                None,
                (pairs * padded, 64, HEAD_DIM),
                true,
            )?;
            let gates = self.product(
                Activation::F32(&hidden),
                second,
                None,
                (pairs * padded, dim, 64),
                false,
            )?;
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
                )
            };
            launch.map_err(gpu_error)?;
            packed
        } else {
            self.pack(
                &projected,
                (Some(&*base), rope),
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
        let mut merged = self.operand(tokens, width)?;
        self.device.heads_to_tokens(
            (&attended, &self.zeros(0)?),
            merged.output(),
            (groups, rows, padded),
            (heads, dim),
        )?;
        let middle = self.product(
            merged.input(),
            &block.out,
            Some(query),
            (tokens, width, width),
            false,
        )?;
        let normed = self.block_norm(&middle, &block.mlp_norm, tokens)?;
        let mut hidden = self.operand(tokens, 2 * width)?;
        self.product_into(
            normed.input(),
            &block.mlp_in,
            None,
            hidden.output(),
            (tokens, 2 * width, width),
            true,
        )?;
        self.product(
            hidden.input(),
            &block.mlp_out,
            Some(&middle),
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
            Ok(self
                .device
                .download(bytes)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect())
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
                    value: slot(&values, s),
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
                    value: unpad(&values[s * len..][..len], (key_heads, rows, icl_dim)),
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
        let to_floats = |bytes: Bytes| -> Result<Buffer> {
            bytes
                .cast_chunks::<f32>()
                .map_err(|_| gpu_error("keys and values do not cast to floats"))
        };
        // Each slot's `[groups, chunk]` into `[groups, slots, chunk]`, or with one
        // group its slot's place in `[slots, ...]`.
        let stack_layer = |parts: Vec<(Bytes, Bytes)>, groups: usize| -> Result<(Bytes, Bytes)> {
            let len = parts[0].0.len() / size_of::<f32>();
            let (mut keys, mut values) = (self.kept(count * len)?, self.kept(count * len)?);
            for (slot, (key, value)) in parts.into_iter().enumerate() {
                for (source, out) in [(key, &mut keys), (value, &mut values)] {
                    let source = to_floats(source)?;
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
                        Self::bytes(self.upload(&out)?)
                    };
                    Ok((pad(&kv.key)?, pad(&kv.value)?))
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
        let kept = |count: usize, len: usize| (0..count).map(|_| self.kept(len)).collect::<Result<Vec<_>>>();
        let column_len =
            columns * dim(config.embedding_heads) * padded_rows(dim(config.inducing_points)) * HEAD_DIM;
        let prefix = key_heads * padded_rows(rows) * HEAD_DIM;
        let mut recording = Recording {
            labels: &cell_labels,
            slots: kept(2 * self.weights.columns.len(), column_len)?.into_iter(),
            recorded: Vec::new(),
        };
        let mut icl_slots = kept(2 * self.weights.icl.len(), prefix)?.into_iter();
        let mut hidden = self.row_embedding(x, &context, &mut recording)?;
        context.columns = recording.recorded;
        let icl_width = dim(config.icl_channels);
        self.add_rows(&mut hidden, &label_rows, (rows, icl_width))?;
        for (layer, block) in self.weights.icl.iter().enumerate() {
            let (keys, values) = self.key_value(block, &hidden, (1, rows))?;
            let mut slot = || icl_slots.next().expect("fit allocates two buffers per ICL layer");
            let (mut kept_keys, mut kept_values) = (slot(), slot());
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
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(self.predict_slots(&context.slots(), &[x])?.remove(0))
    }

    /// Each slot's outputs of its `[rows, columns]` query rows `xs[slot]`, all
    /// slots in one pass, a chunk of rows at a time.
    pub(crate) fn predict_ensemble(&self, ensemble: &Ensemble, xs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let _guard = self.stream_lock.lock().unwrap_or_else(PoisonError::into_inner);
        self.predict_slots(&ensemble.slots(), xs)
    }

    fn predict_slots(&self, slots: &Slots<'_>, xs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let columns = slots.inputs[0].2.len();
        let rows = xs[0].len() / columns;
        if xs.len() != slots.inputs.len() || xs.iter().any(|x| x.len() != rows * columns) {
            return Err(Error::InvalidArgument(
                "each slot needs the same number of query rows".into(),
            ));
        }
        let step = (PASS_ROWS / xs.len()).clamp(1, rows.max(1));
        let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(rows * self.outputs); xs.len()];
        for start in (0..rows).step_by(step) {
            let count = step.min(rows - start);
            let chunk: Vec<&[f32]> = xs
                .iter()
                .map(|x| &x[start * columns..][..count * columns])
                .collect();
            for (all, part) in out.iter_mut().zip(self.pass(slots, &chunk)?) {
                all.extend(part);
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
    fn pass(&self, slots: &Slots<'_>, xs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let config = self.config;
        let (count, columns) = (xs.len(), slots.inputs[0].2.len());
        let rows = xs[0].len() / columns;
        let total = count * rows;
        let (cell, readout) = (dim(config.cell_channels), dim(config.readout_tokens));
        // `[columns, slots, rows, cell]`, so each (column, slot) group is contiguous.
        let mut cells = self.zeros(columns * total * cell)?;
        for (slot, (x, inputs)) in xs.iter().zip(&slots.inputs).enumerate() {
            self.embed_cells(x, *inputs, &mut cells, (slot, count))?;
        }
        let mut readouts = self.zeros(total * readout * cell)?;
        self.device.repeat(
            &self.weights.readout,
            &mut readouts,
            (readout * cell, total * readout * cell),
        )?;
        let points = dim(config.inducing_points);
        for (layer, ((_, output), (keys, values))) in
            self.weights.columns.iter().zip(slots.columns).enumerate()
        {
            cells = self.forward(
                output,
                &cells,
                (columns * count, rows),
                (keys, values, output.heads, points),
            )?;
            self.row_stage(layer, &mut readouts, &mut cells, (total, columns))?;
        }
        let mut hidden = self.rms(&readouts, &self.weights.norm, total * readout)?;
        let icl = dim(config.icl_channels);
        if let Some(project) = &self.weights.row_project {
            hidden = self.product(
                Activation::F32(&hidden),
                project,
                None,
                (total, icl, readout * cell),
                false,
            )?;
        }
        let key_heads = config.query_key_value_heads.map_or(dim(config.icl_heads), dim);
        for (block, (keys, values)) in self.weights.icl.iter().zip(slots.icl) {
            hidden = self.forward(
                block,
                &hidden,
                (count, rows),
                (keys, values, key_heads, slots.training_rows),
            )?;
        }
        let normed = self.rms(&hidden, &self.weights.icl_norm, total)?;
        let wide = self.product(
            Activation::F32(&normed),
            &self.weights.head.0,
            None,
            (total, 2 * icl, icl),
            true,
        )?;
        let out = self.product(
            Activation::F32(&wide),
            &self.weights.head.1,
            None,
            (total, self.outputs, 2 * icl),
            false,
        )?;
        Ok(self
            .device
            .download(&out)?
            .chunks(rows * self.outputs)
            .map(<[f32]>::to_vec)
            .collect())
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
    fn row_embedding(&self, x: &[f32], context: &Context, recording: &mut Recording<'_>) -> Result<Buffer> {
        let config = self.config;
        let columns = context.categorical.len();
        let rows = x.len() / columns;
        let (cell, readout) = (dim(config.cell_channels), dim(config.readout_tokens));
        let mut cells = self.zeros(columns * rows * cell)?;
        let inputs = (&context.means, &context.kinds, context.categorical.as_slice());
        self.embed_cells(x, inputs, &mut cells, (0, 1))?;
        self.add_rows(&mut cells, recording.labels, (rows, cell))?;
        let mut readouts = self.zeros(rows * readout * cell)?;
        self.device.repeat(
            &self.weights.readout,
            &mut readouts,
            (readout * cell, rows * readout * cell),
        )?;
        let points = dim(config.inducing_points);
        for (layer, (inducing, output)) in self.weights.columns.iter().enumerate() {
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
            recording.record(&self.device, &keys, &values)?;
            let (keys, values) = &recording.recorded[layer];
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
                self.product(Activation::F32(&normed), project, None, (rows, icl, width), false)
            }
            None => Ok(normed),
        }
    }

    /// The cell embedding of slot `slot` of `slots`, its `[rows, columns]`
    /// rows `x` with its imputation means, column kinds, and kinds as flags,
    /// into `cells` `[columns, slots, rows, cell]`.
    fn embed_cells(
        &self,
        x: &[f32],
        (means, kinds, categorical): (&Buffer, &Buffer, &[bool]),
        cells: &mut Buffer,
        (slot, slots): (usize, usize),
    ) -> Result<()> {
        let columns = categorical.len();
        let rows = x.len() / columns;
        let cell = dim(self.config.cell_channels);
        let x = self.upload(x)?;
        let mut features = self.zeros(rows * FEATURES)?;
        for column in 0..columns {
            // SAFETY: one thread per value of `features` `[rows, FEATURES]`; `x` holds `[rows,
            // columns]`, `means` and `kinds` `columns`, and the frequency tables `3 * 32`.
            let launch = unsafe {
                self.device.kernels.kumo_cell_features(
                    self.device.stream(),
                    config(threads(rows * FEATURES), 256),
                    &x,
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
            let pattern = (0..3).fold(0, |bits, g| {
                bits | usize::from(categorical[(column + (1 << g) - 1) % columns]) << g
            });
            let embedded = self.product(
                Activation::F32(&features),
                &self.weights.patterns[pattern],
                None,
                (rows, cell, FEATURES),
                false,
            )?;
            self.device.copy_into(
                &embedded,
                cells,
                ((column * slots + slot) * rows * cell, rows * cell),
            )?;
        }
        Ok(())
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
        assert!(readouts.len() >= rows * readout * cell && cells.len() >= (tokens - readout) * rows * cell);
        // SAFETY: one thread per value of `sequences` `[count, tokens, cell]`; `readouts` holds
        // `[rows, readout, cell]` and `cells` `[tokens - readout, rows, cell]` (`tokens == readout`
        // reads no cell), and `start + count <= rows`, all asserted.
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
        precision: Precision,
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

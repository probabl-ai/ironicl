//! Launchers of the kernels any model's blocks chain: products, norms,
//! attention over cached keys and values with the queries packed for it, and
//! copies, with the activation forms and dense layers they take. Models keep
//! the launchers of their own fused kernels.

#![expect(unsafe_code, reason = "Launches of kernels that trust their grid and shapes")]

use cuda_core::{DeviceBuffer, DeviceCopy, LaunchConfig};

use super::{Device, gpu_error, kernels::device::LoadedModule};
use crate::{Optimize, error::Result};

/// Attention head width the attention kernels are built for; narrower heads
/// are zero-padded to it.
pub const HEAD_DIM: usize = 64;
/// Keys per tile of the attention kernels.
pub const KEY_TILE: usize = 64;
/// Scale of a value's low FP16 part; matches `LO_SCALE` in the kernels.
pub(crate) const LO_SCALE: f32 = 2048.0;

/// A value's low part, `(v - f16(v)) * LO_SCALE`.
pub(crate) fn low_part(value: f32) -> f32 {
    (value - half::f16::from_f32(value).to_f32()) * LO_SCALE
}

/// A dense layer `[n, k]`: FP32 for the CUDA cores, FP16 high and scaled low
/// parts for the tensor cores, and its bias.
pub(crate) struct Dense {
    /// High part in 64-output by 32-input tiles, in the tensor-core kernel's shared-memory order.
    tiled: bool,
    pub(crate) w: DeviceBuffer<f32>,
    pub(crate) hi: DeviceBuffer<u16>,
    pub(crate) lo: DeviceBuffer<u16>,
    pub(crate) b: DeviceBuffer<f32>,
}

impl Dense {
    /// A dense layer from its FP32 weight `[n, k]` and bias.
    pub(crate) fn upload(device: &Device, w: &[f32], b: &[f32]) -> Result<Self> {
        Self::for_precision(device, w, b, Precision::Split)
    }

    /// Keep the FP32 weights for narrow products, plus only the parts read at `precision`.
    pub(crate) fn for_precision(device: &Device, w: &[f32], b: &[f32], precision: Precision) -> Result<Self> {
        let mut dense = Self::upload_f32(device, w, b)?;
        if precision != Precision::Fp32 {
            let half = |v| half::f16::from_f32(v).to_bits();
            assert!(!b.is_empty() && !w.is_empty() && w.len().is_multiple_of(b.len()));
            let (width, depth) = (b.len(), w.len() / b.len());
            dense.tiled =
                precision == Precision::Half && width.is_multiple_of(64) && depth.is_multiple_of(32);
            let hi = if dense.tiled {
                let mut hi = Vec::with_capacity(w.len());
                for n in (0..width).step_by(64) {
                    for k in (0..depth).step_by(32) {
                        for row in 0..64 {
                            // swizzled32 in the kernel permutes four 16-byte packets per row.
                            // XOR is its own inverse: write the packets in destination order.
                            for chunk in 0..4 {
                                let at = (n + row) * depth + k + (chunk ^ ((row >> 1) & 3)) * 8;
                                hi.extend(w[at..at + 8].iter().copied().map(half));
                            }
                        }
                    }
                }
                hi
            } else {
                w.iter().copied().map(half).collect()
            };
            dense.hi = device.upload(&hi)?;
            if precision == Precision::Split {
                dense.lo = device.upload(&w.iter().copied().map(low_part).map(half).collect::<Vec<_>>())?;
            }
        }
        Ok(dense)
    }

    /// A dense layer for FP32 products only, without the FP16 parts the
    /// tensor cores read.
    pub(crate) fn upload_f32(device: &Device, w: &[f32], b: &[f32]) -> Result<Self> {
        Ok(Self {
            tiled: false,
            w: device.upload(w)?,
            hi: empty(device)?,
            lo: empty(device)?,
            b: device.upload(b)?,
        })
    }

    /// Device memory the layer takes.
    pub(crate) fn bytes(&self) -> usize {
        4 * (self.w.len() + self.b.len()) + 2 * (self.hi.len() + self.lo.len())
    }
}

/// A layer norm's scale and shift.
pub(crate) struct Norm {
    pub(crate) gamma: DeviceBuffer<f32>,
    pub(crate) beta: DeviceBuffer<f32>,
}

/// How the GPU's products run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    /// FP32 on the CUDA cores, no tensor cores (`optimize="safe"`).
    Fp32,
    /// On tensor cores, every input split into a high FP16 part and a scaled
    /// low one, three products each: close to FP32 (`optimize="auto"`).
    Split,
    /// On tensor cores, FP16 inputs, one product each (`optimize="fast"`).
    Half,
}

impl Precision {
    /// The precision of `optimize`'s GPU products.
    #[must_use]
    pub const fn for_optimize(optimize: Optimize) -> Self {
        match optimize {
            Optimize::Safe => Self::Fp32,
            Optimize::Auto => Self::Split,
            Optimize::Fast => Self::Half,
        }
    }

    /// Token rows from which the ICL and decoder products run on tensor cores
    /// rather than in FP32 on the CUDA cores: measured on an RTX 5080, a split
    /// product's three tensor-core products only pay off on larger products.
    /// Never at [`Self::Fp32`].
    #[must_use]
    pub const fn tensor_rows(self) -> usize {
        match self {
            Self::Fp32 => usize::MAX,
            Self::Split => 256,
            Self::Half => 64,
        }
    }
}

/// Bytes per cached key or value at `precision`: FP32, FP16, or FP8.
pub(crate) const fn value_bytes(precision: Precision) -> usize {
    match precision {
        Precision::Fp32 => 4,
        Precision::Split => 2,
        Precision::Half => 1,
    }
}

/// An activation in the form the product reading it takes: FP32 for the CUDA
/// cores, FP16 for `fast`'s tensor cores, or FP16 high and scaled low parts for
/// `auto`'s.
#[derive(Clone, Copy)]
pub(crate) enum Activation<'a> {
    F32(&'a DeviceBuffer<f32>),
    Half(&'a DeviceBuffer<u16>),
    Split {
        hi: &'a DeviceBuffer<u16>,
        lo: &'a DeviceBuffer<u16>,
    },
}

/// Where a kernel writes an activation, in one of the forms of [`Activation`].
pub(crate) enum Output<'a> {
    F32(&'a mut DeviceBuffer<f32>),
    Half(&'a mut DeviceBuffer<u16>),
    Split {
        hi: &'a mut DeviceBuffer<u16>,
        lo: &'a mut DeviceBuffer<u16>,
    },
}

/// Empty buffers for the operands a kernel does not read or write; they
/// take no device memory.
pub(crate) struct Unused {
    f32: DeviceBuffer<f32>,
    hi: DeviceBuffer<u16>,
    lo: DeviceBuffer<u16>,
}

impl Unused {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            f32: empty(device)?,
            hi: empty(device)?,
            lo: empty(device)?,
        })
    }

    /// The FP32, FP16 high, and low outputs of a kernel writing `out`, empty
    /// where it writes none, and the format code its `store_activation` takes.
    pub(crate) fn outputs<'a>(
        &'a mut self,
        out: Option<Output<'a>>,
    ) -> (
        &'a mut DeviceBuffer<f32>,
        &'a mut DeviceBuffer<u16>,
        &'a mut DeviceBuffer<u16>,
        u32,
    ) {
        match out {
            None => (&mut self.f32, &mut self.hi, &mut self.lo, 0),
            Some(Output::F32(y)) => (y, &mut self.hi, &mut self.lo, 0),
            Some(Output::Half(hi)) => (&mut self.f32, hi, &mut self.lo, 1),
            Some(Output::Split { hi, lo }) => (&mut self.f32, hi, lo, 2),
        }
    }
}

fn empty<T: DeviceCopy>(device: &Device) -> Result<DeviceBuffer<T>> {
    device.zeroed(0).map_err(gpu_error)
}

/// Blocks a launch grid holds at most in y and z.
pub(crate) const MAX_GRID_YZ: usize = 65_535;

/// A size as the kernels take it, in `u32`. `TabICLv2`'s build caps its
/// buffers at `u32::MAX` values (see `build::plan`) and its predictions run at
/// most 1024 rows at a time; Kumo-Tabular's sizes are its tables'. A size that
/// does not fit panics rather than wrapping.
#[expect(clippy::cast_possible_truncation, reason = "checked against u32::MAX first")]
pub(crate) const fn n(value: usize) -> u32 {
    assert!(value <= u32::MAX as usize, "a size past what the kernels index");
    value as u32
}

/// A launch of `grid` blocks of `block` threads.
pub(crate) fn config(grid: (usize, usize, usize), block: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n(grid.0), n(grid.1), n(grid.2)),
        block_dim: (n(block), 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `total` threads in blocks of 256.
pub(crate) const fn threads(total: usize) -> (usize, usize, usize) {
    (total.div_ceil(256), 1, 1)
}

/// Training rows padded to whole key tiles, as the ICL caches hold them.
pub(crate) const fn padded_rows(rows: usize) -> usize {
    rows.next_multiple_of(KEY_TILE)
}

/// Query rows padded for the ICL attention kernels: 16 for up to 16 queries,
/// else a multiple of 64.
pub(crate) const fn padded_query_rows(rows: usize) -> usize {
    if rows <= 16 { 16 } else { rows.div_ceil(64) * 64 }
}

/// Query rows per block of the ICL attention kernel at `precision` for
/// `padded_queries` rows: 32 in FP32; on tensor cores, 16 for the decode
/// kernels, else 64.
const fn icl_block_rows(precision: Precision, padded_queries: usize) -> usize {
    match precision {
        Precision::Fp32 => 32,
        Precision::Split | Precision::Half if padded_queries <= 16 => 16,
        Precision::Split | Precision::Half => 64,
    }
}

impl Device {
    /// How [`Self::attend_icl`] at `precision` splits `rows` keys: `(tiles
    /// per chunk, chunks)`, enough chunks for about three blocks per
    /// multiprocessor. FP32's 32-row blocks never take more chunks than the
    /// tensor cores' 64-row ones.
    pub(crate) fn icl_split(
        &self,
        precision: Precision,
        pairs: usize,
        padded_queries: usize,
        rows: usize,
    ) -> (usize, usize) {
        let blocks = padded_queries.div_ceil(icl_block_rows(precision, padded_queries));
        let tiles = rows.div_ceil(KEY_TILE);
        let target = (3 * self.multiprocessors()).div_ceil(pairs * blocks).max(1);
        let chunk_tiles = tiles.div_ceil(target).max(2.min(tiles));
        (chunk_tiles, tiles.div_ceil(chunk_tiles))
    }

    /// `out[..len]` filled with `source[..period]` over and over.
    pub(crate) fn repeat(
        &self,
        source: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        (period, len): (usize, usize),
    ) -> Result<()> {
        assert!(period <= source.len() && len <= out.len());
        // SAFETY: one thread per `out[i]`, `i < len`, and `len <= out.len()` is asserted above.
        let launch = unsafe {
            self.kernels.repeat(
                self.stream(),
                config(threads(len), 256),
                source,
                out,
                n(period),
                n(len),
            )
        };
        launch.map_err(gpu_error)
    }

    /// `source[..len]` into `out` from `offset` on.
    pub(crate) fn copy_into(
        &self,
        source: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        (offset, len): (usize, usize),
    ) -> Result<()> {
        assert!(
            len <= source.len() && offset + len <= out.len(),
            "copy past its buffers"
        );
        // SAFETY: one thread per `out[offset + i]`, `i < len`, and the assert keeps
        // `offset + len <= out.len()`.
        let launch = unsafe {
            self.kernels.copy_into(
                self.stream(),
                config(threads(len), 256),
                source,
                out,
                n(offset),
                n(len),
            )
        };
        launch.map_err(gpu_error)
    }

    /// Layer norm of `rows` rows of `width` (at most 512) into `out`.
    pub(crate) fn layer_norm(
        &self,
        x: &DeviceBuffer<f32>,
        norm: &Norm,
        out: Output<'_>,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        assert!(width <= 512);
        let mut unused = Unused::new(self)?;
        let (y, hi, lo, format) = unused.outputs(Some(out));
        let config = config(threads(rows * 32), 256);
        // SAFETY: one warp per row (`rows * 32` threads), writing `row * width + i < rows * width`
        // only to the buffers `format` names, which `Unused::outputs` matches to `out`'s; every
        // caller's `out` holds `rows * width` values.
        let launch = unsafe {
            self.kernels.layer_norm(
                self.stream(),
                config,
                x,
                &norm.gamma,
                &norm.beta,
                y,
                hi,
                lo,
                n(rows),
                n(width),
                n(width),
                n(width),
                format,
            )
        };
        launch.map_err(gpu_error)
    }

    /// `act(input wᵀ + bias) (+ residual)` over `rows` rows of `depth`
    /// values into `width` outputs each, written to `out` (FP32), `next` (the
    /// next product's operand), or both: on tensor cores from an FP16 or split
    /// `input`, else on the CUDA cores, which write FP32 only into `out`,
    /// through split-K partial sums in `partial`.
    #[expect(
        clippy::too_many_arguments,
        reason = "a launcher takes its kernel's buffers and shape as plain arguments"
    )]
    pub(crate) fn product(
        &self,
        input: Activation<'_>,
        dense: &Dense,
        residual: Option<&DeviceBuffer<f32>>,
        (out, next): (Option<&mut DeviceBuffer<f32>>, Option<Output<'_>>),
        partial: &mut DeviceBuffer<f32>,
        (rows, width, depth): (usize, usize, usize),
        gelu: bool,
    ) -> Result<()> {
        let (blank, mut unused) = (Unused::new(self)?, Unused::new(self)?);
        let mode = u32::from(gelu) | if residual.is_some() { 2 } else { 0 };
        let residual = residual.unwrap_or(&blank.f32);
        let (input_hi, input_lo, weight_lo) = match input {
            Activation::F32(input) => {
                let (Some(out), None) = (out, next) else {
                    return Err(gpu_error("a CUDA-core product writes FP32 only"));
                };
                return self.product_f32(input, dense, residual, (out, partial), (rows, width, depth), mode);
            }
            Activation::Half(hi) => (hi, &blank.hi, &blank.lo),
            Activation::Split { hi, lo } => (hi, lo, &dense.lo),
        };
        // `linear16` writes `next` in its input's form, so another form would write into an empty
        // `Unused` buffer.
        if !matches!(
            (&input, &next),
            (_, None)
                | (Activation::Half(_), Some(Output::Half(_)))
                | (Activation::Split { .. }, Some(Output::Split { .. }))
        ) {
            return Err(gpu_error(
                "a tensor-core product writes its next activation in its input's form",
            ));
        }
        assert!(depth.is_multiple_of(32));
        let mode = mode | if out.is_some() { 4 } else { 0 } | if next.is_some() { 8 } else { 0 };
        let (unused_y, next_hi, next_lo, _) = unused.outputs(next);
        let y = out.unwrap_or(unused_y);
        let config = config((width.div_ceil(64), rows.div_ceil(64), 1), 128);
        if dense.tiled {
            assert!(
                matches!(input, Activation::Half(_))
                    && width == dense.b.len()
                    && dense.hi.len() == width * depth
            );
        }
        let (rows, width, depth) = (n(rows), n(width), n(depth));
        let kernel = match input {
            Activation::Split { .. } => LoadedModule::linear16_split,
            _ if dense.tiled => LoadedModule::linear16_tiled,
            _ => LoadedModule::linear16,
        };
        // SAFETY: grid (ceil(width / 64), ceil(rows / 64)) of 128 threads, as `linear16_body`
        // asks. Callers' `depth` is a multiple of 32 (16-byte rows); `input`
        // holds `rows * depth`, `dense` is `[width, depth]`, `residual` and `y` hold `rows *
        // width`, and `next`'s form matches `input`'s, so no write reaches an empty `Unused`.
        let launch = unsafe {
            kernel(
                &self.kernels,
                self.stream(),
                config,
                input_hi,
                input_lo,
                &dense.hi,
                weight_lo,
                &dense.b,
                residual,
                y,
                next_hi,
                next_lo,
                rows,
                width,
                depth,
                depth,
                width,
                width,
                mode,
            )
        };
        launch.map_err(gpu_error)
    }

    /// Depth per split and split count. The caller sizes the partial sums from the same plan.
    pub(crate) fn product_splits(&self, (rows, width, depth): (usize, usize, usize)) -> (usize, usize) {
        let blocks = rows.div_ceil(64) * width.div_ceil(64);
        let wanted = (2 * self.multiprocessors())
            .div_ceil(blocks)
            .min(depth / 64)
            .max(1);
        let chunk = depth.div_ceil(wanted).next_multiple_of(16);
        (chunk, depth.div_ceil(chunk))
    }

    /// A product in FP32 on the CUDA cores: 64 x 64 tiles, `depth` split over
    /// extra blocks until about two waves run, their partial sums added with
    /// the epilogue by `linear_reduce`.
    fn product_f32(
        &self,
        input: &DeviceBuffer<f32>,
        dense: &Dense,
        residual: &DeviceBuffer<f32>,
        (out, partial): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        (rows, width, depth): (usize, usize, usize),
        mode: u32,
    ) -> Result<()> {
        let (chunk, splits) = self.product_splits((rows, width, depth));
        assert!(
            splits == 1 || splits * rows * width <= partial.len(),
            "split-K partials past their buffer"
        );
        let (m, w, k) = (n(rows), n(width), n(depth));
        let tiles = rows.div_ceil(64);
        for first in (0..tiles).step_by(MAX_GRID_YZ) {
            let grid = config((width.div_ceil(64), MAX_GRID_YZ.min(tiles - first), splits), 256);
            // SAFETY: grid (ceil(width / 64), row tiles `first..` of ceil(rows / 64), splits) of
            // 256 threads, as `linear64` asks; `depth` (a multiple of 16) and `chunk` (a multiple
            // of 16) meet its multiples, and `cuMemAlloc` its 16-byte alignment. `input` holds
            // `rows * depth`, `dense.w` is `[width, depth]`, `out` holds `rows * width`, and the
            // assert keeps the partials in `partial`.
            let launch = unsafe {
                self.kernels.linear64(
                    self.stream(),
                    grid,
                    input,
                    &dense.w,
                    &dense.b,
                    residual,
                    &mut *out,
                    &mut *partial,
                    m,
                    w,
                    k,
                    k,
                    w,
                    w,
                    mode,
                    n(chunk),
                    n(first),
                )
            };
            launch.map_err(gpu_error)?;
        }
        if splits == 1 {
            return Ok(());
        }
        let config = config(threads(rows * width), 256);
        // SAFETY: one thread per output, writing `row * width + col < rows * width`, within `out`;
        // reads of `partial` are checked.
        let launch = unsafe {
            self.kernels.linear_reduce(
                self.stream(),
                config,
                partial,
                &dense.b,
                residual,
                out,
                m,
                w,
                n(splits),
                w,
                w,
                mode,
            )
        };
        launch.map_err(gpu_error)
    }

    /// Token rows `x` `[sequences * len, heads * dim]` packed into heads
    /// `[sequences * heads, padded, 64]` for attention, as [`pack_heads_body`]
    /// says: `RoPE` by `frequencies` and the per-head norm when given, then
    /// times `base`, the gate, and `key_scale` when they are not empty. FP32
    /// or FP16.
    ///
    /// [`pack_heads_body`]: crate::kernels::cuda::ops::pack_heads_body
    pub(crate) fn pack_heads(
        &self,
        (x, gate): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (base, key_scale, frequencies): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        packed: Output<'_>,
        (sequences, len, padded): (usize, usize, usize),
        (heads, dim): (usize, usize),
        norm: bool,
    ) -> Result<()> {
        assert!(!matches!(packed, Output::Split { .. }) && dim <= HEAD_DIM && len <= padded);
        let mut unused = Unused::new(self)?;
        let (y, hi, _, format) = unused.outputs(Some(packed));
        let config = config(threads(sequences * padded * heads * 32), 256);
        // SAFETY: one warp per (row, head) of `sequences * padded * heads`, writing row `row <
        // padded` of pair `sequence * heads + head` (zeros from `len`) to the buffer `format` names,
        // which callers size to `[sequences * heads, padded, 64]`; `x` holds `[sequences * len, heads * dim]`,
        // `gate` is empty or `[sequences * len * heads, dim]`, `base` empty or `[heads, dim]`,
        // `key_scale` empty or `[sequences * heads, 64]`, and `frequencies` empty or `dim / 2`.
        let launch = unsafe {
            self.kernels.pack_heads(
                self.stream(),
                config,
                x,
                gate,
                base,
                key_scale,
                frequencies,
                y,
                hi,
                n(sequences),
                n(len),
                n(padded),
                n(heads),
                n(dim),
                u32::from(norm),
                format,
            )
        };
        launch.map_err(gpu_error)
    }

    /// `RMSNorm` of `rows` rows of `width` (at most 1024) into `out`, with
    /// weight `gamma` and epsilon `eps`.
    pub(crate) fn rms_norm(
        &self,
        x: &DeviceBuffer<f32>,
        gamma: &DeviceBuffer<f32>,
        out: Output<'_>,
        (rows, width): (usize, usize),
        eps: f32,
    ) -> Result<()> {
        assert!(width <= 1024);
        let mut unused = Unused::new(self)?;
        let (y, hi, lo, format) = unused.outputs(Some(out));
        // SAFETY: one warp per row (`rows * 32` threads), writing `row * width + i < rows * width`
        // only to the buffers `format` names, which `Unused::outputs` matches to `out`'s; callers'
        // `x` and `out` hold `rows * width` values and `gamma` `width`.
        let launch = unsafe {
            self.kernels.rms_norm(
                self.stream(),
                config(threads(rows * 32), 256),
                x,
                gamma,
                y,
                hi,
                lo,
                n(rows),
                n(width),
                format,
                eps,
            )
        };
        launch.map_err(gpu_error)
    }

    /// Softmax attention of `queries` `[pairs, padded_queries, 64]`, FP32 at
    /// [`Precision::Fp32`] and FP16 otherwise, against the cache's `keys` and
    /// `values` `[pairs, padded_rows, 64]` (FP32 for `safe`, FP16 for `auto`,
    /// FP8 for `fast`), into `out` in FP32, through the chunks' partial
    /// outputs `part` and log-sum-exps `lse`.
    pub(crate) fn attend_icl(
        &self,
        precision: Precision,
        queries: Activation<'_>,
        (keys, values): (&DeviceBuffer<u8>, &DeviceBuffer<u8>),
        (part, lse, out): (
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
        ),
        (pairs, padded_queries, rows, stride): (usize, usize, usize, usize),
        kv_share: usize,
    ) -> Result<()> {
        let block_rows = icl_block_rows(precision, padded_queries);
        let (chunk_tiles, splits) = self.icl_split(precision, pairs, padded_queries, rows);
        // The kernels index the caches and the chunk outputs unchecked.
        assert!(kv_share > 0 && pairs.is_multiple_of(kv_share));
        // The FP8 values' scales are per query pair: a shared key head would need its own.
        assert!(
            kv_share == 1 || precision != Precision::Half,
            "FP8 caches are not shared"
        );
        assert!(stride == padded_rows(rows) || (precision == Precision::Fp32 && stride == rows));
        let cache = pairs / kv_share * stride * HEAD_DIM * value_bytes(precision);
        let chunks = pairs * padded_queries * splits;
        assert!(
            keys.len() >= cache
                && values.len() >= cache
                && (splits == 1 || part.len() >= chunks * HEAD_DIM)
                && lse.len() >= chunks
                && out.len() >= pairs * padded_queries * HEAD_DIM
        );
        let blocks = pairs * splits;
        let (qp, np, splits32) = (n(padded_queries), n(stride), n(splits));
        let scale = 0.125 / std::f32::consts::LN_2; // 1 / sqrt(64), in the exp2 domain
        let stream = self.stream();
        let tiles = padded_queries.div_ceil(block_rows);
        let dest = if splits == 1 { &mut *out } else { &mut *part };
        // SAFETY: each 128-thread block covers one query tile and key chunk. FP32 guards the
        // last tile; tensor-core callers pad queries and keys to whole tiles. The stride and
        // buffer checks above cover every cache read and every (pair, chunk, query) output.
        let launch = unsafe {
            match (precision, queries) {
                (Precision::Fp32, Activation::F32(queries)) => {
                    // Blocks past the grid's y limit run in further launches.
                    for first in (0..blocks).step_by(MAX_GRID_YZ) {
                        let grid = config((tiles, MAX_GRID_YZ.min(blocks - first), 1), 128);
                        self.kernels
                            .icl_attention_f32(
                                stream,
                                grid,
                                queries,
                                keys,
                                values,
                                &mut *dest,
                                &mut *lse,
                                qp,
                                np,
                                n(rows),
                                n(chunk_tiles),
                                splits32,
                                scale,
                                n(kv_share),
                                n(first),
                            )
                            .map_err(gpu_error)?;
                    }
                    Ok(())
                }
                (Precision::Split | Precision::Half, Activation::Half(queries)) => {
                    assert!(
                        blocks <= MAX_GRID_YZ,
                        "tensor-core attention past the grid's y limit"
                    );
                    let grid = config((tiles, blocks, 1), 128);
                    let kernel = match (precision, block_rows == 16) {
                        (Precision::Split, true) => LoadedModule::icl_attention_decode,
                        (Precision::Split, false) => LoadedModule::icl_attention_wide,
                        (_, true) => LoadedModule::icl_attention_decode_fp8,
                        (_, false) => LoadedModule::icl_attention_wide_fp8,
                    };
                    kernel(
                        &self.kernels,
                        stream,
                        grid,
                        queries,
                        keys,
                        values,
                        dest,
                        lse,
                        qp,
                        np,
                        n(rows),
                        n(chunk_tiles),
                        splits32,
                        scale,
                        n(kv_share),
                    )
                    .map_err(gpu_error)
                }
                _ => {
                    return Err(gpu_error(
                        "ICL queries in a form the cache's precision does not read",
                    ));
                }
            }
        };
        launch?;
        // A single chunk is already normalized and in [pairs, queries, 64] order.
        if splits == 1 {
            return Ok(());
        }
        let config = config(threads(pairs * padded_queries * HEAD_DIM), 256);
        // SAFETY: every access of `icl_attention_merge` is bounds-checked (its write goes through
        // `get_mut`); one thread per output value of `[pairs, qp, 64]`.
        let launch = unsafe {
            self.kernels
                .icl_attention_merge(self.stream(), config, part, lse, out, n(pairs), qp, splits32)
        };
        launch.map_err(gpu_error)
    }

    /// Attention outputs `[ensembles * heads, qp, 64]` back to tokens
    /// `[ensembles * q, heads * dim]`, times the values' scales unless
    /// `value_scale` is empty, into `out`.
    pub(crate) fn heads_to_tokens(
        &self,
        (attended, value_scale): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        out: Output<'_>,
        (ensembles, q, qp): (usize, usize, usize),
        (heads, dim): (usize, usize),
    ) -> Result<()> {
        let mut unused = Unused::new(self)?;
        let (y, hi, lo, format) = unused.outputs(Some(out));
        assert!(dim != 0 && dim <= HEAD_DIM && dim.is_multiple_of(4));
        assert!(q <= qp && attended.len() >= ensembles * heads * qp * HEAD_DIM);
        assert!(value_scale.is_empty() || value_scale.len() >= ensembles * heads * HEAD_DIM);
        let len = ensembles * q * heads * dim;
        assert!(match format {
            0 => y.len() >= len,
            1 => hi.len() >= len,
            _ => hi.len() >= len && lo.len() >= len,
        });
        let config = config(threads(len / 4), 256);
        // SAFETY: one thread per four aligned token values. The shape checks cover all source,
        // optional scale, and destination vectors; unused representations are not touched.
        let launch = unsafe {
            self.kernels.heads_to_tokens(
                self.stream(),
                config,
                attended,
                value_scale,
                y,
                hi,
                lo,
                n(ensembles),
                n(q),
                n(qp),
                n(heads),
                n(dim),
                format,
            )
        };
        launch.map_err(gpu_error)
    }
}

#[cfg(test)]
mod tests {
    use super::{Activation, Dense, HEAD_DIM, MAX_GRID_YZ, Precision, padded_query_rows, padded_rows};
    use crate::kernels::cuda::Device;

    #[expect(clippy::cast_precision_loss, reason = "Small test values")]
    fn value(i: usize, period: usize) -> f32 {
        (i % period) as f32 / period as f32 - 0.5
    }

    #[test]
    fn tiled_half_projections_match_row_major_with_a_partial_row_tile() {
        let Some(device) = Device::new().unwrap() else {
            return;
        };
        let (rows, width, depth) = (67, 128, 96);
        let weights = (0..width * depth).map(|i| value(i, 251)).collect::<Vec<_>>();
        let bias = (0..width).map(|i| value(i, 37)).collect::<Vec<_>>();
        let x = (0..rows * depth)
            .map(|i| half::f16::from_f32(value(i, 31)).to_bits())
            .collect::<Vec<_>>();
        let x = device.upload(&x).unwrap();
        let residual = device
            .upload(&(0..rows * width).map(|i| value(i, 11)).collect::<Vec<_>>())
            .unwrap();
        let plain = Dense::upload(&device, &weights, &bias).unwrap();
        let tiled = Dense::for_precision(&device, &weights, &bias, Precision::Half).unwrap();
        assert!(!plain.tiled && tiled.tiled);
        for gelu in [false, true] {
            let product = |dense: &Dense| {
                let mut out = device.zeroed(rows * width).unwrap();
                let mut next = device.zeroed(rows * width).unwrap();
                device
                    .product(
                        Activation::Half(&x),
                        dense,
                        (!gelu).then_some(&residual),
                        (Some(&mut out), Some(super::Output::Half(&mut next))),
                        &mut device.zeroed(0).unwrap(),
                        (rows, width, depth),
                        gelu,
                    )
                    .unwrap();
                (device.download(&out).unwrap(), device.download(&next).unwrap())
            };
            assert_eq!(product(&plain), product(&tiled));
        }
    }

    /// Products and FP32 attention past the grid's 65,535 blocks in y run in
    /// several launches and match the host.
    #[test]
    fn launches_past_the_grid_limit() {
        let Some(device) = Device::new().expect("load CUDA kernels") else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        // A product one row tile past the limit.
        let (rows, width, depth) = (MAX_GRID_YZ * 64 + 64, 4, 16);
        let inputs: Vec<f32> = (0..rows * depth).map(|i| value(i, 97)).collect();
        let weights: Vec<f32> = (0..width * depth).map(|i| value(i, 13)).collect();
        let bias = vec![0.25_f32; width];
        let dense = Dense::upload_f32(&device, &weights, &bias).unwrap();
        let input = device.upload(&inputs).unwrap();
        let mut out = device.zeroed::<f32>(rows * width).unwrap();
        let mut partial = device.zeroed::<f32>(0).unwrap();
        let shape = (rows, width, depth);
        device
            .product(
                Activation::F32(&input),
                &dense,
                None,
                (Some(&mut out), None),
                &mut partial,
                shape,
                false,
            )
            .unwrap();
        let out = device.download(&out).unwrap();
        for row in [0, rows / 2, rows - 1] {
            for col in 0..width {
                let expected = bias[col]
                    + (0..depth)
                        .map(|k| inputs[row * depth + k] * weights[col * depth + k])
                        .sum::<f32>();
                assert!(
                    (out[row * width + col] - expected).abs() < 1e-4,
                    "row {row}, column {col}"
                );
            }
        }
        // Attention with more pairs than a grid holds in y.
        let (pairs, keys) = (MAX_GRID_YZ + 7, 64);
        let (qp, np) = (padded_query_rows(1), padded_rows(keys));
        let queries: Vec<f32> = (0..pairs * qp * HEAD_DIM).map(|i| value(i, 31)).collect();
        let (key_data, value_data): (Vec<f32>, Vec<f32>) = (
            (0..pairs * np * HEAD_DIM).map(|i| value(i, 37)).collect(),
            (0..pairs * np * HEAD_DIM).map(|i| value(i, 41)).collect(),
        );
        let bytes = |data: &[f32]| device.upload(data).unwrap().cast_chunks::<u8>().ok().unwrap();
        let (k_bytes, v_bytes) = (bytes(&key_data), bytes(&value_data));
        let (_, splits) = device.icl_split(Precision::Fp32, pairs, qp, keys);
        let mut part = device.zeroed::<f32>(pairs * qp * splits * HEAD_DIM).unwrap();
        let mut lse = device.zeroed::<f32>(pairs * qp * splits).unwrap();
        let mut attended = device.zeroed::<f32>(pairs * qp * HEAD_DIM).unwrap();
        let query_buffer = device.upload(&queries).unwrap();
        device
            .attend_icl(
                Precision::Fp32,
                Activation::F32(&query_buffer),
                (&k_bytes, &v_bytes),
                (&mut part, &mut lse, &mut attended),
                (pairs, qp, keys, np),
                1,
            )
            .unwrap();
        let attended = device.download(&attended).unwrap();
        for pair in [0, MAX_GRID_YZ / 2, pairs - 1] {
            let query = &queries[pair * qp * HEAD_DIM..][..HEAD_DIM];
            let scores: Vec<f32> = (0..keys)
                .map(|key| {
                    let at = (pair * np + key) * HEAD_DIM;
                    query
                        .iter()
                        .zip(&key_data[at..at + HEAD_DIM])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / 8.0
                })
                .collect();
            let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
            let total: f32 = weights.iter().sum();
            for d in 0..HEAD_DIM {
                let expected = (0..keys)
                    .map(|key| weights[key] * value_data[(pair * np + key) * HEAD_DIM + d])
                    .sum::<f32>()
                    / total;
                let got = attended[pair * qp * HEAD_DIM + d];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "pair {pair}, dim {d}: {got} vs {expected}"
                );
            }
        }
    }
}

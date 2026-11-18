//! One launcher per kernel family: it takes the kernel's real operands, picks
//! the kernel for the precision, and sizes the grid to the kernel's indexing.
//!
//! The kernels index their buffers without bounds checks, so a launch is
//! `unsafe`: each launcher passes buffers that hold the shapes it is given
//! and the grid its kernel's docs in `kernels.rs` ask for, and nothing else
//! launches a kernel.
//!
//! The kernels' slice indexing (`x[i]`) is bounds-checked and traps, so the
//! SAFETY comments cover the rest: unchecked writes and reads, raw 16-byte
//! vector reads, and `cp.async` copies. Launches take no completion lease on
//! their buffers: a buffer dropped while a queued kernel still uses it (such
//! as `peaks` in `Device::encode_fp8`) relies on `cuMemFree` synchronizing
//! with the device.
#![expect(unsafe_code, reason = "Launches of kernels that trust their grid and shapes")]

use cuda_core::{DeviceBuffer, DeviceCopy, LaunchConfig};

use super::{
    COLUMN_DIM, HEAD_DIM, HEADS, KEY_TILE, Precision,
    device::{Device, gpu_error},
    kernels::device::LoadedModule,
    weights::{ColumnInput, Dense, Norm, TokenBlock},
};
use crate::error::Result;

/// An activation in the form the product reading it takes: FP32 for the CUDA
/// cores, FP16 for `fast`'s tensor cores, or FP16 high and scaled low parts for
/// `auto`'s.
#[derive(Clone, Copy)]
pub(super) enum Activation<'a> {
    F32(&'a DeviceBuffer<f32>),
    Half(&'a DeviceBuffer<u16>),
    Split {
        hi: &'a DeviceBuffer<u16>,
        lo: &'a DeviceBuffer<u16>,
    },
}

/// Where [`Device::icl_kv`] writes an ICL block's keys and values.
pub(super) enum CacheOutput<'a> {
    F32(&'a mut DeviceBuffer<f32>, &'a mut DeviceBuffer<f32>),
    F16(&'a mut DeviceBuffer<u16>, &'a mut DeviceBuffer<u16>),
}

impl CacheOutput<'_> {
    /// The same buffers for one more write.
    pub(super) fn reborrow(&mut self) -> CacheOutput<'_> {
        match self {
            Self::F32(keys, values) => CacheOutput::F32(keys, values),
            Self::F16(keys, values) => CacheOutput::F16(keys, values),
        }
    }
}

/// Where a kernel writes an activation, in one of the forms of [`Activation`].
pub(super) enum Output<'a> {
    F32(&'a mut DeviceBuffer<f32>),
    Half(&'a mut DeviceBuffer<u16>),
    Split {
        hi: &'a mut DeviceBuffer<u16>,
        lo: &'a mut DeviceBuffer<u16>,
    },
}

/// Empty buffers for the operands a kernel does not read or write; they
/// take no device memory.
struct Unused {
    f32: DeviceBuffer<f32>,
    hi: DeviceBuffer<u16>,
    lo: DeviceBuffer<u16>,
}

impl Unused {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            f32: empty(device)?,
            hi: empty(device)?,
            lo: empty(device)?,
        })
    }

    /// The FP32, FP16 high, and low outputs of a kernel writing `out`, empty
    /// where it writes none, and the format code its `store_activation` takes.
    fn outputs<'a>(
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

/// The strides, in FP32 elements, of a 16-wide attention: sequences of
/// `lq` queries attend to `lk` keys and values, head by head, the keys and
/// values from `k_base` and `v_base` on.
pub(super) struct SmallAttention {
    pub(super) seqs: usize,
    pub(super) lq: usize,
    pub(super) lk: usize,
    pub(super) inner: usize,
    pub(super) q_seq: usize,
    pub(super) q_tok: usize,
    pub(super) k_base: usize,
    pub(super) v_base: usize,
    pub(super) kv_outer: usize,
    pub(super) kv_inner: usize,
    pub(super) kv_tok: usize,
    pub(super) kv_head: usize,
    pub(super) o_seq: usize,
    pub(super) o_tok: usize,
}

/// A size as the kernels take it, in `u32`: a build caps its buffers at
/// `u32::MAX` values (see `build::plan`), and a prediction's counts are per
/// run of at most 1024 rows, except the query rows `total` and `start`, which
/// feed only bounds-checked reads and the `missing` offset.
#[expect(
    clippy::cast_possible_truncation,
    reason = "callers pass sizes the build plan caps below u32::MAX"
)]
const fn n(value: usize) -> u32 {
    value as u32
}

/// A launch of `grid` blocks of `block` threads.
fn config(grid: (usize, usize, usize), block: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n(grid.0), n(grid.1), n(grid.2)),
        block_dim: (n(block), 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `total` threads in blocks of 256.
const fn threads(total: usize) -> (usize, usize, usize) {
    (total.div_ceil(256), 1, 1)
}

/// Training rows padded to whole key tiles, as the ICL caches hold them.
pub(super) const fn padded_rows(rows: usize) -> usize {
    rows.next_multiple_of(KEY_TILE)
}

/// Query rows padded for the ICL attention kernels: 16 for up to 16 queries,
/// else a multiple of 64.
pub(super) const fn padded_query_rows(rows: usize) -> usize {
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
    pub(super) fn icl_split(
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

    /// The column stage's input layer from raw queries `[B, total, H]`, rows
    /// `[start, start + q)`, into `out` `[B * H, q, 128]`; `missing[row]`
    /// becomes 1 for a row with the all-missing marker.
    pub(super) fn column_input(
        &self,
        queries: &DeviceBuffer<f32>,
        input: &ColumnInput,
        (out, missing): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        (ensembles, features, total, start, q): (usize, usize, usize, usize, usize),
    ) -> Result<()> {
        let config = config(threads(ensembles * features * q * COLUMN_DIM), 256);
        // SAFETY: one thread per `out[i]`, `i < B * H * q * 128`; `out` is `x0`, which
        // `Workspace::reserve` sizes for runs of `q` rows. `missing[start + row]`, `row < q`,
        // is within the caller's `missing`, one per query row; a row's columns all store 1.0.
        let launch = unsafe {
            self.kernels.column_input(
                self.stream(),
                config,
                queries,
                &input.weight,
                &input.bias,
                out,
                missing,
                n(ensembles),
                n(features),
                n(total),
                n(start),
                n(q),
            )
        };
        launch.map_err(gpu_error)
    }

    /// Row-stage sequences `[B * q, 4 + H, 128]` from the CLS tokens and rows
    /// `start..start + q` of the column embeddings `[B * H, total, 128]`.
    pub(super) fn build_rows(
        &self,
        columns: &DeviceBuffer<f32>,
        cls: &DeviceBuffer<f32>,
        rows: &mut DeviceBuffer<f32>,
        (ensembles, features, q): (usize, usize, usize),
        (total, start): (usize, usize),
    ) -> Result<()> {
        let config = config(threads(ensembles * q * (features + 4) * COLUMN_DIM), 256);
        // SAFETY: one thread per `rows[i]`, `i < B * q * (H + 4) * 128`; `rows` is `x1`, which
        // `Workspace::reserve` sizes for the run's tokens. Reads of `columns` and `cls` are
        // checked.
        let launch = unsafe {
            self.kernels.build_rows(
                self.stream(),
                config,
                columns,
                cls,
                rows,
                n(ensembles),
                n(features),
                n(q),
                n(total),
                n(start),
            )
        };
        launch.map_err(gpu_error)
    }

    /// The column stage's input layer plus label embeddings for training
    /// columns `first..first + columns` of `[B * H]`, over all `rows` rows of
    /// `x[B, rows, H]`, into `out[columns, rows, 128]`.
    pub(super) fn column_train(
        &self,
        (x, labels): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        input: &ColumnInput,
        out: &mut DeviceBuffer<f32>,
        (features, rows): (usize, usize),
        (first, columns): (usize, usize),
    ) -> Result<()> {
        let config = config(threads(columns * rows * COLUMN_DIM), 256);
        // SAFETY: one thread per `out[i]`, `i < columns * rows * 128`; `out` is a chunk's state,
        // `[chunk.columns, rows, 128]` with `columns <= chunk.columns`. Reads are checked.
        let launch = unsafe {
            self.kernels.column_train(
                self.stream(),
                config,
                x,
                &input.weight,
                &input.bias,
                labels,
                out,
                n(features),
                n(rows),
                n(first),
                n(columns),
            )
        };
        launch.map_err(gpu_error)
    }

    /// `out[..len]` filled with `source[..period]` over and over.
    pub(super) fn repeat(
        &self,
        source: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        (period, len): (usize, usize),
    ) -> Result<()> {
        debug_assert!(period <= source.len() && len <= out.len());
        // SAFETY: one thread per `out[i]`, `i < len`; `len <= out.len()` is asserted in debug
        // builds only: the one caller writes `count * 128 * 128` into `[chunk.columns, 128, 128]`.
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
    pub(super) fn copy_into(
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

    /// A column block's inducing-point keys and values from its in-projection
    /// `kv[columns * 128, 384]` into `[columns, 8, 128, 16]`.
    pub(super) fn column_kv(
        &self,
        kv: &DeviceBuffer<f32>,
        (keys, values): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        columns: usize,
    ) -> Result<()> {
        assert!(keys.len() >= columns * 128 * 128 && values.len() == keys.len());
        // SAFETY: one thread per key value, writing index `(column * 8 + head) * 2048 + point * 16
        // + d < columns * 128 * 128`, which the assert keeps within `keys` and `values`.
        let launch = unsafe {
            self.kernels.column_kv(
                self.stream(),
                config(threads(columns * 128 * 128), 256),
                kv,
                keys,
                values,
                n(columns),
            )
        };
        launch.map_err(gpu_error)
    }

    /// Rows `[B, q, width]` of `x` plus rows `start..start + q` of `labels[B,
    /// total, width]`, in place.
    pub(super) fn add_labels(
        &self,
        x: &mut DeviceBuffer<f32>,
        labels: &DeviceBuffer<f32>,
        (ensembles, q, width): (usize, usize, usize),
        (total, start): (usize, usize),
    ) -> Result<()> {
        let len = ensembles * q * width;
        assert!(len <= x.len() && ensembles * total * width <= labels.len());
        // SAFETY: one thread per `x[i]`, `i < len`, which the assert keeps within `x`; reads of
        // `labels` are checked.
        let launch = unsafe {
            self.kernels.add_labels(
                self.stream(),
                config(threads(len), 256),
                x,
                labels,
                n(q),
                n(total),
                n(start),
                n(width),
                n(len),
            )
        };
        launch.map_err(gpu_error)
    }

    /// An ICL block's keys and values for training rows `start..start + q`
    /// from the in-projection `kv[B * q, 1024]` into the caches `[B * 8, np,
    /// 64]`, FP32 or FP16.
    pub(super) fn icl_kv(
        &self,
        kv: &DeviceBuffer<f32>,
        cache: CacheOutput<'_>,
        (ensembles, q, start, np): (usize, usize, usize, usize),
    ) -> Result<()> {
        let (len, other) = match &cache {
            CacheOutput::F32(keys, values) => (keys.len(), values.len()),
            CacheOutput::F16(keys, values) => (keys.len(), values.len()),
        };
        assert!(start + q <= np && len >= ensembles * HEADS * np * HEAD_DIM && other == len);
        let config = config(threads(ensembles * q * 512), 256);
        let (ensembles, q, start, np) = (n(ensembles), n(q), n(start), n(np));
        let stream = self.stream();
        // SAFETY: one thread per key value of `B * q * 512`, writing row `start + row < np` of
        // pair `b * 8 + head < B * 8`, below `B * 8 * np * 64`: the assert keeps it in both caches.
        let launch = unsafe {
            match cache {
                CacheOutput::F32(keys, values) => self
                    .kernels
                    .icl_kv_f32(stream, config, kv, keys, values, ensembles, q, start, np),
                CacheOutput::F16(keys, values) => self
                    .kernels
                    .icl_kv(stream, config, kv, keys, values, ensembles, q, start, np),
            }
        };
        launch.map_err(gpu_error)
    }

    /// The first `rows` rows of FP16 keys or values `[pairs, stride, 64]` as
    /// FP8 E4M3 bytes `[pairs, np, 64]`, each divided by a scale per (pair,
    /// dim) that maps its largest magnitude to 448; returns the bytes and the
    /// scales `[pairs, 64]`, as the host's `cache::encode` does.
    pub(super) fn encode_fp8(
        &self,
        values: &DeviceBuffer<u16>,
        (pairs, rows): (usize, usize),
        (stride, np): (usize, usize),
    ) -> Result<(DeviceBuffer<u8>, DeviceBuffer<f32>)> {
        assert!(rows <= stride && rows <= np && values.len() >= pairs * stride * HEAD_DIM);
        let chunks = rows.div_ceil(256);
        let mut peaks = self.zeroed::<f32>(pairs * chunks * HEAD_DIM).map_err(gpu_error)?;
        let mut scales = self.zeroed::<f32>(pairs * HEAD_DIM).map_err(gpu_error)?;
        let mut packed = self.zeroed::<u16>(pairs * np * HEAD_DIM / 2).map_err(gpu_error)?;
        let (stride32, rows32, chunks32) = (n(stride), n(rows), n(chunks));
        let stream = self.stream();
        // SAFETY: each kernel runs one thread per value of its output (`peaks`, `scales`,
        // `packed`) and writes only index `i < len`, that buffer's length; its reads are checked.
        unsafe {
            let len = peaks.len();
            self.kernels
                .fp8_peaks(
                    stream,
                    config(threads(len), 256),
                    values,
                    &mut peaks,
                    stride32,
                    rows32,
                    chunks32,
                    n(len),
                )
                .map_err(gpu_error)?;
            let len = scales.len();
            self.kernels
                .fp8_scales(
                    stream,
                    config(threads(len), 256),
                    &peaks,
                    &mut scales,
                    chunks32,
                    n(len),
                )
                .map_err(gpu_error)?;
            let len = packed.len();
            self.kernels
                .fp8_encode(
                    stream,
                    config(threads(len), 256),
                    values,
                    &scales,
                    &mut packed,
                    stride32,
                    n(np),
                    rows32,
                    n(len),
                )
                .map_err(gpu_error)?;
        }
        let packed = packed
            .cast_chunks::<u8>()
            .map_err(|_| gpu_error("FP8 cache as bytes"))?;
        Ok((packed, scales))
    }

    /// The four CLS tokens of each `[len, 128]` sequence, as `[seqs * 4, 128]`.
    pub(super) fn gather_cls(
        &self,
        rows: &DeviceBuffer<f32>,
        cls: &mut DeviceBuffer<f32>,
        seqs: usize,
        len: usize,
    ) -> Result<()> {
        let config = config(threads(seqs * 4 * COLUMN_DIM), 256);
        // SAFETY: one thread per `cls[i]`, `i < seqs * 512`; `cls` is the workspace's
        // `[B q, 4, 128]`. Reads of `rows` are checked.
        let launch = unsafe {
            self.kernels
                .gather_cls(self.stream(), config, rows, cls, n(seqs), n(len))
        };
        launch.map_err(gpu_error)
    }

    /// Layer norm of `rows` rows of `width` (at most 512) into `out`.
    pub(super) fn layer_norm(
        &self,
        x: &DeviceBuffer<f32>,
        norm: &Norm,
        out: Output<'_>,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        debug_assert!(width <= 512);
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

    /// A 128-wide block's head: `out = norm1(x) Wᵀ + b` for the first `width`
    /// of its in-projection outputs, with `RoPE` on the queries and keys of
    /// `rope_len`-token sequences (none at 0).
    pub(super) fn token_head(
        &self,
        precision: Precision,
        (x, out): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        block: &TokenBlock,
        (cos, sin): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (rows, width, rope_len): (usize, usize, usize),
    ) -> Result<()> {
        let (norm, w) = (&block.norm1, &block.in_proj);
        let config = config((rows.div_ceil(32), 1, 1), 256);
        let kernel = match precision {
            Precision::Fp32 => {
                // SAFETY: one block of 256 threads per 32 rows, as the kernel asks. It reads rows
                // `< rows` of `x` as unchecked 16-byte vectors and writes `out` below `rows *
                // width`; callers' buffers hold `rows * 128` and `rows * width`, with `width`
                // 128 or 384.
                let launch = unsafe {
                    self.kernels.block_head_f32(
                        self.stream(),
                        config,
                        x,
                        &norm.gamma,
                        &norm.beta,
                        &w.w,
                        &w.b,
                        cos,
                        sin,
                        out,
                        n(rows),
                        n(width),
                        n(rope_len),
                    )
                };
                return launch.map_err(gpu_error);
            }
            Precision::Half => LoadedModule::block_head_half,
            Precision::Split => LoadedModule::block_head_split,
        };
        // SAFETY: as above; `width <= 384`, the in-projection's rows, also keeps the unchecked
        // fragment-order reads of `w.hi` and `w.lo` (`[width / 8, 8, 32, 2]` u32) in bounds.
        let launch = unsafe {
            kernel(
                &self.kernels,
                self.stream(),
                config,
                x,
                &norm.gamma,
                &norm.beta,
                &w.hi,
                &w.lo,
                &w.b,
                cos,
                sin,
                out,
                n(rows),
                n(width),
                n(rope_len),
            )
        };
        launch.map_err(gpu_error)
    }

    /// The rest of a 128-wide block once `attended` holds its attention
    /// output: `mid = state + out_proj(attended)`, then
    /// `state = mid + linear2(gelu(linear1(norm2(mid))))`, in place.
    pub(super) fn token_tail(
        &self,
        precision: Precision,
        (attended, state): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        block: &TokenBlock,
        rows: usize,
    ) -> Result<()> {
        let TokenBlock {
            out_proj: o,
            norm2,
            linear1: l1,
            linear2: l2,
            ..
        } = block;
        let config = config((rows.div_ceil(32), 1, 1), 256);
        let kernel = match precision {
            Precision::Fp32 => {
                // SAFETY: one block of 256 threads per 32 rows, as the kernel asks. It reads
                // `attended` and reads and writes `state` unchecked at rows `< rows`, 128 values
                // each, and every caller's buffers hold `rows * 128`.
                let launch = unsafe {
                    self.kernels.block_tail_f32(
                        self.stream(),
                        config,
                        attended,
                        state,
                        &o.w,
                        &o.b,
                        &norm2.gamma,
                        &norm2.beta,
                        &l1.w,
                        &l1.b,
                        &l2.w,
                        &l2.b,
                        n(rows),
                    )
                };
                return launch.map_err(gpu_error);
            }
            Precision::Half => LoadedModule::block_tail_half,
            Precision::Split => LoadedModule::block_tail_split,
        };
        // SAFETY: as above; the fragment-order weights hold the fixed 128- and 256-wide layers
        // whose unchecked reads the kernel makes.
        let launch = unsafe {
            kernel(
                &self.kernels,
                self.stream(),
                config,
                attended,
                state,
                &o.hi,
                &o.lo,
                &o.b,
                &norm2.gamma,
                &norm2.beta,
                &l1.hi,
                &l1.lo,
                &l1.b,
                &l2.hi,
                &l2.lo,
                &l2.b,
                n(rows),
            )
        };
        launch.map_err(gpu_error)
    }

    /// A whole column block in place on `state` `[B * H, q, 128]`, attending
    /// to the block's cached inducing points `(keys, values)` `[B, H +
    /// reserved, 8, 128, 16]`.
    pub(super) fn column_block(
        &self,
        precision: Precision,
        state: &mut DeviceBuffer<f32>,
        block: &TokenBlock,
        (keys, values): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (ensembles, features, q, reserved): (usize, usize, usize, usize),
    ) -> Result<()> {
        debug_assert!(keys.len() >= ensembles * (features + reserved) * HEADS * 128 * 16);
        let TokenBlock {
            norm1,
            in_proj: w,
            out_proj: o,
            norm2,
            linear1: l1,
            linear2: l2,
        } = block;
        let config = config((q.div_ceil(32), ensembles * features, 1), 256);
        let kernel = match precision {
            Precision::Fp32 => {
                // SAFETY: grid (ceil(q / 32), B * H) of 256 threads, as the kernel asks. It reads
                // and writes `state` rows `< B * H * q` (`x0` or a chunk's state holds them) and
                // reads `keys` and `values` unchecked below `B * (H + reserved) * 16384`, which
                // callers' caches hold (debug-asserted; a context's checked at load).
                let launch = unsafe {
                    self.kernels.column_block_f32(
                        self.stream(),
                        config,
                        state,
                        &norm1.gamma,
                        &norm1.beta,
                        &w.w,
                        &w.b,
                        keys,
                        values,
                        &o.w,
                        &o.b,
                        &norm2.gamma,
                        &norm2.beta,
                        &l1.w,
                        &l1.b,
                        &l2.w,
                        &l2.b,
                        n(features),
                        n(q),
                        n(reserved),
                    )
                };
                return launch.map_err(gpu_error);
            }
            Precision::Half => LoadedModule::column_block_half,
            Precision::Split => LoadedModule::column_block_split,
        };
        // SAFETY: as above, with the weights in fragment order for the block's fixed shapes.
        let launch = unsafe {
            kernel(
                &self.kernels,
                self.stream(),
                config,
                state,
                &norm1.gamma,
                &norm1.beta,
                &w.hi,
                &w.lo,
                &w.b,
                keys,
                values,
                &o.hi,
                &o.lo,
                &o.b,
                &norm2.gamma,
                &norm2.beta,
                &l1.hi,
                &l1.lo,
                &l1.b,
                &l2.hi,
                &l2.lo,
                &l2.b,
                n(features),
                n(q),
                n(reserved),
            )
        };
        launch.map_err(gpu_error)
    }

    /// 16-wide softmax attention of `q` to `k` and `v` into `out`.
    pub(super) fn attention16(
        &self,
        precision: Precision,
        (q, k, v): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        out: &mut DeviceBuffer<f32>,
        layout: &SmallAttention,
    ) -> Result<()> {
        let l = layout;
        let kernel = match precision {
            Precision::Fp32 => {
                // SAFETY: grid (ceil(lq / 64), 8, seqs) of 64 threads, one per query, as the
                // kernel asks. Each caller's layout keeps the unchecked reads `k_base + kv0 + key *
                // kv_tok + 16 <= k.len()` (`key < lk`, likewise `v`) and writes `seq * o_seq +
                // query * o_tok + head * 16 + 16 <= out.len()`; its strides are multiples of 4
                // values, aligning the 16-byte key and value reads.
                let launch = unsafe {
                    self.kernels.attention16_f32(
                        self.stream(),
                        config((l.lq.div_ceil(64), HEADS, l.seqs), 64),
                        q,
                        k,
                        v,
                        out,
                        n(l.lq),
                        n(l.lk),
                        n(l.inner),
                        n(l.q_seq),
                        n(l.q_tok),
                        n(l.k_base),
                        n(l.v_base),
                        n(l.kv_outer),
                        n(l.kv_inner),
                        n(l.kv_tok),
                        n(l.kv_head),
                        n(l.o_seq),
                        n(l.o_tok),
                    )
                };
                return launch.map_err(gpu_error);
            }
            Precision::Half => LoadedModule::attention16_half,
            Precision::Split => LoadedModule::attention16_split,
        };
        let config = config((l.lq.div_ceil(64), HEADS, l.seqs), 128);
        let scale = 0.25; // 1 / sqrt(16)
        // SAFETY: the same grid of 128 threads, four warps of 16 queries, as the kernel asks;
        // bounds as above, and its unchecked query reads stay below `seq * q_seq + query * q_tok +
        // head * 16 + 16 <= q.len()` for `query < lq`.
        let launch = unsafe {
            kernel(
                &self.kernels,
                self.stream(),
                config,
                q,
                k,
                v,
                out,
                n(l.lq),
                n(l.lk),
                n(l.inner),
                n(l.q_seq),
                n(l.q_tok),
                n(l.k_base),
                n(l.v_base),
                n(l.kv_outer),
                n(l.kv_inner),
                n(l.kv_tok),
                n(l.kv_head),
                n(l.o_seq),
                n(l.o_tok),
                scale,
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
    pub(super) fn product(
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
        debug_assert!(depth.is_multiple_of(32));
        let mode = mode | if out.is_some() { 4 } else { 0 } | if next.is_some() { 8 } else { 0 };
        let (unused_y, next_hi, next_lo, _) = unused.outputs(next);
        let y = out.unwrap_or(unused_y);
        let config = config((width.div_ceil(64), rows.div_ceil(64), 1), 128);
        let (rows, width, depth) = (n(rows), n(width), n(depth));
        let kernel = match input {
            Activation::Split { .. } => LoadedModule::linear16_split,
            _ => LoadedModule::linear16,
        };
        // SAFETY: grid (ceil(width / 64), ceil(rows / 64)) of 128 threads, as `linear16_body`
        // asks. Callers' `depth` is 64, 512, or 1024 (a multiple of 32: 16-byte rows); `input`
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
        let blocks = rows.div_ceil(64) * width.div_ceil(64);
        let wanted = (2 * self.multiprocessors())
            .div_ceil(blocks)
            .min(depth / 64)
            .max(1);
        let chunk = depth.div_ceil(wanted).next_multiple_of(16);
        let splits = depth.div_ceil(chunk);
        assert!(
            splits == 1 || splits * rows * width <= partial.len(),
            "split-K partials past their buffer"
        );
        let grid = config((width.div_ceil(64), rows.div_ceil(64), splits), 256);
        let (m, w, k) = (n(rows), n(width), n(depth));
        // SAFETY: grid (ceil(width / 64), ceil(rows / 64), splits) of 256 threads, as `linear64`
        // asks; `depth` (64, 512, or 1024) and `chunk` (a multiple of 16) meet its multiples, and
        // `cuMemAlloc` its 16-byte alignment. `input` holds `rows * depth`, `dense.w` is `[width,
        // depth]`, `out` holds `rows * width`, and the assert keeps the partials in `partial`.
        let launch = unsafe {
            self.kernels.linear64(
                self.stream(),
                grid,
                input,
                &dense.w,
                &dense.b,
                residual,
                out,
                partial,
                m,
                w,
                k,
                k,
                w,
                w,
                mode,
                n(chunk),
            )
        };
        launch.map_err(gpu_error)?;
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

    /// `QASSMax` and packing for ICL attention: each head of `query`
    /// `[B * q, 512]`, scaled by its base scale, gate, and the keys' scales,
    /// at `packed` `[B * 8, qp, 64]`, FP32 or FP16.
    pub(super) fn qass_pack(
        &self,
        (query, gate): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (base, key_scale): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        packed: Output<'_>,
        (ensembles, q, qp): (usize, usize, usize),
    ) -> Result<()> {
        debug_assert!(!matches!(packed, Output::Split { .. }));
        let mut unused = Unused::new(self)?;
        let (y, hi, _, format) = unused.outputs(Some(packed));
        let config = config(threads(ensembles * q * 512), 256);
        // SAFETY: one thread per query value of `B * q * 512`, writing row `row < q <= qp` of pair
        // `b * 8 + head` to the buffer `format` names; `packed` and `packed32` hold
        // `B * 8 * padded_query_rows(run) * 64` (`Workspace::reserve`), and `qp` is at most that.
        let launch = unsafe {
            self.kernels.qass_pack(
                self.stream(),
                config,
                query,
                gate,
                base,
                key_scale,
                y,
                hi,
                n(ensembles),
                n(q),
                n(qp),
                format,
            )
        };
        launch.map_err(gpu_error)
    }

    /// Softmax attention of `queries` `[pairs, padded_queries, 64]`, FP32 at
    /// [`Precision::Fp32`] and FP16 otherwise, against the cache's `keys` and
    /// `values` `[pairs, padded_rows, 64]` (FP32 for `safe`, FP16 for `auto`,
    /// FP8 for `fast`), into `out` in FP32, through the chunks' partial
    /// outputs `part` and log-sum-exps `lse`.
    pub(super) fn attend_icl(
        &self,
        precision: Precision,
        queries: Activation<'_>,
        (keys, values): (&DeviceBuffer<u8>, &DeviceBuffer<u8>),
        (part, lse, out): (
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
        ),
        (pairs, padded_queries, rows): (usize, usize, usize),
    ) -> Result<()> {
        let block_rows = icl_block_rows(precision, padded_queries);
        let (chunk_tiles, splits) = self.icl_split(precision, pairs, padded_queries, rows);
        let grid = config((padded_queries.div_ceil(block_rows), pairs * splits, 1), 128);
        let (qp, np, splits32) = (n(padded_queries), n(padded_rows(rows)), n(splits));
        let scale = 0.125 / std::f32::consts::LN_2; // 1 / sqrt(64), in the exp2 domain
        let stream = self.stream();
        // SAFETY: grid (qp / block_rows, pairs * splits) of 128 threads, `block_rows` being the
        // chosen kernel's rows per block; `qp` (`padded_query_rows`) is a multiple of it, so the
        // unchecked query copies stay in `[pairs, qp, 64]`. The caches hold `pairs * np * 64`
        // values of the precision's width (`BlockCache`); `part` and `lse` hold every chunk, as
        // `Workspace::reserve` sizes them by this `icl_split`. Chunks start below `np / 64` tiles.
        let launch = unsafe {
            match (precision, queries) {
                (Precision::Fp32, Activation::F32(queries)) => self.kernels.icl_attention_f32(
                    stream,
                    grid,
                    queries,
                    keys,
                    values,
                    part,
                    lse,
                    qp,
                    np,
                    n(rows),
                    n(chunk_tiles),
                    splits32,
                    scale,
                ),
                (Precision::Split | Precision::Half, Activation::Half(queries)) => {
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
                        part,
                        lse,
                        qp,
                        np,
                        n(rows),
                        n(chunk_tiles),
                        splits32,
                        scale,
                    )
                }
                _ => {
                    return Err(gpu_error(
                        "ICL queries in a form the cache's precision does not read",
                    ));
                }
            }
        };
        launch.map_err(gpu_error)?;
        let config = config(threads(pairs * padded_queries * HEAD_DIM), 256);
        // SAFETY: every access of `icl_attention_merge` is bounds-checked (its write goes through
        // `get_mut`); one thread per output value of `[pairs, qp, 64]`.
        let launch = unsafe {
            self.kernels
                .icl_attention_merge(self.stream(), config, part, lse, out, n(pairs), qp, splits32)
        };
        launch.map_err(gpu_error)
    }

    /// Attention outputs `[B * 8, qp, 64]` back to tokens `[B * q, 512]`,
    /// times the values' scales, into `out`.
    pub(super) fn heads_to_tokens(
        &self,
        (attended, value_scale): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        out: Output<'_>,
        (ensembles, q, qp): (usize, usize, usize),
    ) -> Result<()> {
        let mut unused = Unused::new(self)?;
        let (y, hi, lo, format) = unused.outputs(Some(out));
        let config = config(threads(ensembles * q * 512), 256);
        // SAFETY: one thread per token value, writing `i < B * q * 512` to the buffers `format`
        // names; `out` is the workspace's `a` halves or `attended`, both at least `[B q, 512]`.
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
                format,
            )
        };
        launch.map_err(gpu_error)
    }
}

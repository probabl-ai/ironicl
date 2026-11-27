//! A context's ICL keys and values on the device: FP32 for `safe`, FP16 for
//! `auto`, or FP8 E4M3 with a scale per (estimator, head, dim) for `fast`, kept
//! on the device or streamed from the host one block at a time on every pass.
//!
//! Streaming goes through two device slots, each with a pinned staging
//! buffer: [`IclCache::begin`] stages the first two blocks, [`IclCache::acquire`]
//! makes the kernels wait for a block's copy, and [`IclCache::release`] marks
//! the slot free once the block's kernels are queued and stages the block two
//! ahead into it.
#![expect(unsafe_code, reason = "Asynchronous copies from pinned staging buffers")]

use std::sync::Arc;

use cuda_core::{DeviceBuffer, Event, PinnedHostBuffer};
use rayon::prelude::*;

use super::{
    HEAD_DIM, ICL_BLOCKS, Precision,
    device::{Device, gpu_error},
    launch::padded_rows,
};
use crate::error::{Error, Result};
pub(crate) use crate::kernels::cuda::launch::value_bytes;

/// One ICL block's keys or values on the host, `[B, 8, N, 64]`.
#[derive(Clone, Copy)]
pub enum IclValues<'a> {
    /// FP32, for [`Precision::Fp32`].
    F32(&'a [f32]),
    /// FP16 bits, for the tensor-core precisions.
    F16(&'a [u16]),
    /// Already encoded E4M3 and its per-(estimator, head, channel) scales.
    F8(&'a [u8], &'a [f32]),
}

impl<'a> IclValues<'a> {
    fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::F8(bytes, _) => bytes.len(),
        }
    }

    fn bytes(self) -> &'a [u8] {
        match self {
            Self::F32(values) => bytemuck::cast_slice(values),
            Self::F16(values) => bytemuck::cast_slice(values),
            Self::F8(bytes, _) => bytes,
        }
    }

    /// Whether a cache at `precision` stores these values: FP32 at
    /// [`Precision::Fp32`], FP16 otherwise.
    #[must_use]
    pub const fn stored_at(self, precision: Precision) -> bool {
        matches!(
            (self, precision),
            (Self::F32(_), Precision::Fp32)
                | (Self::F16(_), Precision::Split | Precision::Half)
                | (Self::F8(..), Precision::Half)
        )
    }
}

/// A context's ICL keys and values on the host, which the cache uploads once
/// or streams on every pass.
pub trait IclSource: Send + Sync {
    /// ICL block `block`'s keys and values.
    fn block(&self, block: usize) -> (IclValues<'_>, IclValues<'_>);
}

/// 16-bit units of one block's keys (or values) for `pairs` pairs and `rows`
/// rows at `precision`.
pub(super) const fn units(precision: Precision, pairs: usize, rows: usize) -> usize {
    pairs * padded_rows(rows) * HEAD_DIM * value_bytes(precision) / 2
}

/// Keys or values `[pairs, rows, 64]` of `size` bytes each, as bytes, into
/// `[pairs, padded_rows(rows), 64]`, pair by pair in parallel, leaving the
/// padding as it is.
fn pad(padded: &mut [u8], values: &[u8], rows: usize, size: usize) {
    padded
        .par_chunks_mut(padded_rows(rows) * HEAD_DIM * size)
        .zip(values.par_chunks(rows * HEAD_DIM * size))
        .for_each(|(to, from)| to[..from.len()].copy_from_slice(from));
}

/// One block's keys and values on the device as the attention kernels read
/// them: bytes of FP32, FP16, or FP8 values, rows padded to whole key tiles.
pub(super) struct KeyValues {
    pub(super) keys: DeviceBuffer<u8>,
    pub(super) values: DeviceBuffer<u8>,
}

impl KeyValues {
    /// Keys and values of any type as the bytes the attention kernels read.
    pub(super) fn from_typed<T: cuda_core::DeviceCopy + bytemuck::NoUninit>(
        keys: DeviceBuffer<T>,
        values: DeviceBuffer<T>,
    ) -> Result<Self> {
        let bytes = |buffer: DeviceBuffer<T>| {
            buffer
                .cast_chunks::<u8>()
                .map_err(|_| gpu_error("ICL cache as bytes"))
        };
        Ok(Self {
            keys: bytes(keys)?,
            values: bytes(values)?,
        })
    }
}

/// An FP8 block's key and value scales, `[B * 8, 64]` each: folded into the
/// queries for keys, applied to the attention outputs for values.
pub(super) struct Scales {
    pub(super) keys: DeviceBuffer<f32>,
    pub(super) values: DeviceBuffer<f32>,
}

/// One block of a cache as its kernels read it.
#[derive(Clone, Copy)]
pub(super) struct BlockCache<'a> {
    /// FP32 at [`Precision::Fp32`], FP16 at [`Precision::Split`], FP8 at
    /// [`Precision::Half`].
    pub(super) precision: Precision,
    pub(super) keys: &'a DeviceBuffer<u8>,
    pub(super) values: &'a DeviceBuffer<u8>,
    pub(super) key_scales: &'a DeviceBuffer<f32>,
    pub(super) value_scales: &'a DeviceBuffer<f32>,
}

/// A context's ICL keys and values that its build left on the device, as
/// predictions at `precision` read them; FP8 blocks with their scales.
pub struct BuiltCache {
    precision: Precision,
    blocks: Vec<KeyValues>,
    scales: Vec<Scales>,
}

impl BuiltCache {
    pub(super) fn new(precision: Precision) -> Self {
        Self {
            precision,
            blocks: Vec::with_capacity(ICL_BLOCKS),
            scales: Vec::new(),
        }
    }

    /// Add the next block, with its scales when FP8.
    pub(super) fn push(&mut self, block: KeyValues, scales: Option<Scales>) {
        self.blocks.push(block);
        self.scales.extend(scales);
    }

    /// Copy a kept FP8 tensor to its final host representation.
    pub(crate) fn write_fp8(
        &self,
        weights: &super::ModelWeights,
        rows: usize,
        block: usize,
        key: bool,
        out: &mut [u8],
        scales: &mut [u8],
    ) -> Result<()> {
        let (b, s) = (&self.blocks[block], &self.scales[block]);
        copy_fp8(
            rows,
            &weights.device.download(if key { &b.keys } else { &b.values })?,
            out,
            &weights.device.download(if key { &s.keys } else { &s.values })?,
            scales,
        )
    }

    /// Device bytes the keys and values take.
    pub(super) fn bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.keys.len() + block.values.len())
            .sum()
    }

    pub(super) const fn precision(&self) -> Precision {
        self.precision
    }
}

/// Where a cache's blocks are.
enum Storage {
    /// Every block on the device.
    Resident(Vec<KeyValues>),
    /// On the host, copied to the device block by block on every pass.
    Streamed(Host),
}

enum Host {
    /// FP16 straight from the context.
    Context(Arc<dyn IclSource>),
    /// FP8 encoded once on the device, `[pairs, padded rows, 64]` bytes two
    /// to a unit.
    Encoded(Vec<(Vec<u16>, Vec<u16>)>),
}

pub(super) struct IclCache {
    precision: Precision,
    pairs: usize,
    rows: usize,
    /// Each block's scales when FP8, none when FP16.
    scales: Vec<Scales>,
    /// The scales of FP16 blocks, `[pairs, 64]` ones.
    ones: DeviceBuffer<f32>,
    storage: Storage,
}

/// A device slot a streamed block is copied into, with its pinned staging:
/// keys and values as on the device, the FP16 padding zero.
pub(super) struct Slot {
    block: KeyValues,
    staged_keys: PinnedHostBuffer<u8>,
    staged_values: PinnedHostBuffer<u8>,
    /// Recorded on the copy stream once the staging is on the device.
    copied: Option<Event>,
    /// Recorded on the kernels' stream after the last kernel reading the slot.
    released: Option<Event>,
}

impl Drop for Slot {
    /// Wait for a copy still reading the staging buffers, as a pass that
    /// failed midway can leave one.
    fn drop(&mut self) {
        if let Some(copied) = &self.copied {
            // Nothing to do on failure: the context is gone with the copy.
            let _ = copied.synchronize();
        }
    }
}

impl IclCache {
    /// The cache of `source` for `pairs` (estimator, head) pairs and `rows`
    /// training rows at `precision`, on the device unless `streamed`. FP8
    /// blocks are encoded on the device.
    pub(super) fn new(
        device: &Device,
        precision: Precision,
        source: &Arc<dyn IclSource>,
        (pairs, rows): (usize, usize),
        streamed: bool,
    ) -> Result<Self> {
        let (mut blocks, mut encoded, mut scales) = (Vec::new(), Vec::new(), Vec::new());
        for block in 0..ICL_BLOCKS {
            let (keys, values) = source.block(block);
            if keys.len() != pairs * rows * HEAD_DIM
                || values.len() != keys.len()
                || !keys.stored_at(precision)
                || !values.stored_at(precision)
            {
                return Err(Error::InvalidArgument("ICL cache shape does not match".into()));
            }
            match (precision, keys, values) {
                // A streamed FP32 or FP16 cache copies straight from the context.
                (Precision::Fp32 | Precision::Split, ..) if streamed => {}
                (Precision::Fp32 | Precision::Split, ..) => blocks.push(KeyValues {
                    keys: upload_padded(device, precision, keys, (pairs, rows))?,
                    values: upload_padded(device, precision, values, (pairs, rows))?,
                }),
                (Precision::Half, IclValues::F8(_, ks), IclValues::F8(_, vs)) => {
                    if ks.len() != pairs * HEAD_DIM || vs.len() != ks.len() {
                        return Err(Error::Artifact("FP8 scale shape mismatch".into()));
                    }
                    scales.push(Scales {
                        keys: device.upload(ks)?,
                        values: device.upload(vs)?,
                    });
                    if !streamed {
                        blocks.push(KeyValues {
                            keys: upload_padded(device, precision, keys, (pairs, rows))?,
                            values: upload_padded(device, precision, values, (pairs, rows))?,
                        });
                    }
                }
                (Precision::Half, IclValues::F16(keys), IclValues::F16(values)) => {
                    let ((keys, key_scales), (values, value_scales)) = (
                        encode_fp8(device, keys, (pairs, rows))?,
                        encode_fp8(device, values, (pairs, rows))?,
                    );
                    scales.push(Scales {
                        keys: key_scales,
                        values: value_scales,
                    });
                    if streamed {
                        encoded.push((download_units(device, keys)?, download_units(device, values)?));
                    } else {
                        blocks.push(KeyValues { keys, values });
                    }
                }
                (Precision::Half, ..) => return Err(Error::Artifact("mixed ICL cache dtypes".into())),
            }
        }
        let storage = if !streamed {
            Storage::Resident(blocks)
        } else if encoded.is_empty() {
            Storage::Streamed(Host::Context(Arc::clone(source)))
        } else if encoded.len() == ICL_BLOCKS {
            Storage::Streamed(Host::Encoded(encoded))
        } else {
            return Err(Error::Artifact("mixed ICL block dtypes".into()));
        };
        Ok(Self {
            precision,
            pairs,
            rows,
            scales,
            ones: ones(device, pairs)?,
            storage,
        })
    }

    /// Copy the exact cache used for inference, removing per-head row padding.
    /// No quantization here: the bytes come from the resident or host-streamed cache.
    pub(super) fn write_fp8(
        &self,
        device: &Device,
        block: usize,
        key: bool,
        out: &mut [u8],
        scales: &mut [u8],
    ) -> Result<()> {
        let downloaded;
        let bytes: &[u8] = match &self.storage {
            Storage::Resident(blocks) => {
                let b = &blocks[block];
                downloaded = device.download(if key { &b.keys } else { &b.values })?;
                &downloaded
            }
            Storage::Streamed(Host::Encoded(blocks)) => {
                let (k, v) = &blocks[block];
                bytemuck::cast_slice(if key { k } else { v })
            }
            Storage::Streamed(Host::Context(_)) => {
                return Err(Error::Artifact("context is already encoded".into()));
            }
        };
        let s = &self.scales[block];
        copy_fp8(
            self.rows,
            bytes,
            out,
            &device.download(if key { &s.keys } else { &s.values })?,
            scales,
        )
    }

    pub(super) fn use_packed_source(&mut self, source: Arc<dyn IclSource>) {
        if self.streamed() {
            self.storage = Storage::Streamed(Host::Context(source));
        }
    }

    /// The cache a build left on the device for `pairs` pairs and `rows`
    /// training rows.
    pub(super) fn built(device: &Device, built: BuiltCache, (pairs, rows): (usize, usize)) -> Result<Self> {
        let expected = 2 * units(built.precision, pairs, rows);
        if built.blocks.len() != ICL_BLOCKS
            || built
                .blocks
                .iter()
                .any(|block| block.keys.len() != expected || block.values.len() != expected)
        {
            return Err(Error::InvalidArgument(
                "built ICL cache shape does not match".into(),
            ));
        }
        Ok(Self {
            precision: built.precision,
            pairs,
            rows,
            scales: built.scales,
            ones: ones(device, pairs)?,
            storage: Storage::Resident(built.blocks),
        })
    }

    pub(super) const fn streamed(&self) -> bool {
        matches!(self.storage, Storage::Streamed(_))
    }

    /// The two device slots a streamed cache copies through; none for a
    /// resident one.
    pub(super) fn slots(&self, device: &Device) -> Result<Vec<Slot>> {
        if !self.streamed() {
            return Ok(Vec::new());
        }
        let bytes = 2 * units(self.precision, self.pairs, self.rows);
        (0..2)
            .map(|_| {
                Ok(Slot {
                    block: KeyValues {
                        keys: device.zeroed(bytes).map_err(gpu_error)?,
                        values: device.zeroed(bytes).map_err(gpu_error)?,
                    },
                    staged_keys: device.pinned(bytes)?,
                    staged_values: device.pinned(bytes)?,
                    copied: None,
                    released: None,
                })
            })
            .collect()
    }

    /// Stage the first two blocks of a pass.
    pub(super) fn begin(&self, device: &Device, slots: &mut [Slot]) -> Result<()> {
        let Storage::Streamed(host) = &self.storage else {
            return Ok(());
        };
        for (block, slot) in slots.iter_mut().enumerate() {
            self.stage(device, host, slot, block)?;
        }
        Ok(())
    }

    /// Block `block` as its kernels read it; a streamed block's kernels first
    /// wait for its copy.
    pub(super) fn acquire<'a>(
        &'a self,
        device: &Device,
        slots: &'a [Slot],
        block: usize,
    ) -> Result<BlockCache<'a>> {
        let (key_scales, value_scales) = self
            .scales
            .get(block)
            .map_or((&self.ones, &self.ones), |scales| (&scales.keys, &scales.values));
        let stored = match &self.storage {
            Storage::Resident(blocks) => &blocks[block],
            Storage::Streamed(_) => {
                let slot = &slots[block % 2];
                if let Some(copied) = &slot.copied {
                    device.stream().wait(copied).map_err(gpu_error)?;
                }
                &slot.block
            }
        };
        Ok(BlockCache {
            precision: self.precision,
            keys: &stored.keys,
            values: &stored.values,
            key_scales,
            value_scales,
        })
    }

    /// After block `block`'s kernels are queued: mark its slot free once they
    /// finish, and stage the block two ahead into it.
    pub(super) fn release(&self, device: &Device, slots: &mut [Slot], block: usize) -> Result<()> {
        let Storage::Streamed(host) = &self.storage else {
            return Ok(());
        };
        let slot = &mut slots[block % 2];
        slot.released = Some(device.stream().record_event(None).map_err(gpu_error)?);
        if block + 2 < ICL_BLOCKS {
            self.stage(device, host, slot, block + 2)?;
        }
        Ok(())
    }

    /// Copy block `block` into `slot` on the copy stream, once the slot's last
    /// copy left its staging buffers and the kernels it fed finished.
    fn stage(&self, device: &Device, host: &Host, slot: &mut Slot, block: usize) -> Result<()> {
        if let Some(copied) = slot.copied.take() {
            copied.synchronize().map_err(gpu_error)?;
        }
        let staged_keys = slot.staged_keys.as_mut_slice();
        let staged_values = slot.staged_values.as_mut_slice();
        match host {
            Host::Context(source) => {
                let (keys, values) = source.block(block);
                let size = value_bytes(self.precision);
                pad(staged_keys, keys.bytes(), self.rows, size);
                pad(staged_values, values.bytes(), self.rows, size);
            }
            Host::Encoded(blocks) => {
                let (keys, values) = &blocks[block];
                let (keys, values) = (
                    bytemuck::cast_slice::<u16, u8>(keys),
                    bytemuck::cast_slice::<u16, u8>(values),
                );
                for (to, from) in [(staged_keys, keys), (staged_values, values)] {
                    to.par_chunks_mut(1 << 20)
                        .zip(from.par_chunks(1 << 20))
                        .for_each(|(to, from)| to.copy_from_slice(from));
                }
            }
        }
        let stream = device.copy_stream();
        if let Some(released) = slot.released.take() {
            stream.wait(&released).map_err(gpu_error)?;
        }
        let mut enqueue = || {
            for (to, from) in [
                (&mut slot.block.keys, &slot.staged_keys),
                (&mut slot.block.values, &slot.staged_values),
            ] {
                // SAFETY: `from` stays alive and unwritten until `copied`, recorded below,
                // completes: the next `stage` into this slot and `Slot::drop` synchronize it
                // first, and on failure the stream is synchronized below before returning. The
                // copy stream waited above for `released`, so no kernel still reads `to`, and
                // kernels read it only after `acquire` makes their stream wait for `copied`. Both
                // hold `2 * units`.
                unsafe { to.copy_from_pinned_host_async(stream, from) }.map_err(gpu_error)?;
            }
            stream.record_event(None).map_err(gpu_error)
        };
        match enqueue() {
            Ok(copied) => {
                slot.copied = Some(copied);
                Ok(())
            }
            Err(error) => {
                // A copy that did start may still read the staging buffers, which the next
                // `stage` rewrites and `Slot::drop` frees: wait for it. Nothing more to do if
                // this fails too.
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }
}

/// `[pairs, 64]` ones on the device: the scales of FP16 keys and values.
fn ones(device: &Device, pairs: usize) -> Result<DeviceBuffer<f32>> {
    device.upload(&vec![1.0; pairs * HEAD_DIM])
}

/// FP32 or FP16 keys or values `[pairs, rows, 64]`, as a cache at `precision`
/// stores them, uploaded with their rows padded.
fn upload_padded(
    device: &Device,
    precision: Precision,
    values: IclValues<'_>,
    (pairs, rows): (usize, usize),
) -> Result<DeviceBuffer<u8>> {
    let mut padded = vec![0u8; 2 * units(precision, pairs, rows)];
    pad(&mut padded, values.bytes(), rows, value_bytes(precision));
    device.upload(&padded)
}

/// Unpad native E4M3 and check the per-channel scales, including the overflow
/// marker produced by `fp8_peaks` for a non-finite FP16 input.
fn copy_fp8(rows: usize, bytes: &[u8], out: &mut [u8], scales: &[f32], scale_out: &mut [u8]) -> Result<()> {
    if !scales.iter().all(|&s| s.is_finite() && s > 0.0) {
        return Err(Error::Artifact("invalid native FP8 scales".into()));
    }
    for (dst, src) in out
        .chunks_exact_mut(rows * HEAD_DIM)
        .zip(bytes.chunks_exact(padded_rows(rows) * HEAD_DIM))
    {
        dst.copy_from_slice(&src[..dst.len()]);
    }
    scale_out.copy_from_slice(bytemuck::cast_slice(scales));
    Ok(())
}

/// FP16 keys or values `[pairs, rows, 64]` encoded as FP8 on the device, with
/// their scales.
fn encode_fp8(
    device: &Device,
    values: &[u16],
    (pairs, rows): (usize, usize),
) -> Result<(DeviceBuffer<u8>, DeviceBuffer<f32>)> {
    let values = device.upload(values)?;
    device.encode_fp8(&values, (pairs, rows), (rows, padded_rows(rows)))
}

/// FP8 bytes on the device as the host's units, two bytes to a unit.
fn download_units(device: &Device, bytes: DeviceBuffer<u8>) -> Result<Vec<u16>> {
    let units = bytes
        .cast_chunks::<u16>()
        .map_err(|_| gpu_error("FP8 cache from bytes"))?;
    device.download(&units)
}

/// One block uploaded for the kernel tests: its keys and values as a
/// predictor at `precision` stores them, and their scales.
pub(super) struct UploadedBlock {
    pub(super) cache: KeyValues,
    pub(super) key_scales: Vec<f32>,
    pub(super) value_scales: Vec<f32>,
}

impl UploadedBlock {
    /// FP16 keys and values `[pairs, rows, 64]` stored as a predictor at
    /// `precision` stores them: in FP32 at [`Precision::Fp32`], which holds
    /// them exactly.
    pub(super) fn new(
        device: &Device,
        precision: Precision,
        (keys, values): (&[u16], &[u16]),
        (pairs, rows): (usize, usize),
    ) -> Result<Self> {
        let widened = |values: &[u16]| -> Vec<f32> {
            values
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect()
        };
        let upload = |values: IclValues<'_>| upload_padded(device, precision, values, (pairs, rows));
        let ones = vec![1.0; pairs * HEAD_DIM];
        let cache = match precision {
            Precision::Fp32 => KeyValues {
                keys: upload(IclValues::F32(&widened(keys)))?,
                values: upload(IclValues::F32(&widened(values)))?,
            },
            Precision::Split => KeyValues {
                keys: upload(IclValues::F16(keys))?,
                values: upload(IclValues::F16(values))?,
            },
            Precision::Half => {
                let ((keys, key_scales), (values, value_scales)) = (
                    encode_fp8(device, keys, (pairs, rows))?,
                    encode_fp8(device, values, (pairs, rows))?,
                );
                return Ok(Self {
                    cache: KeyValues { keys, values },
                    key_scales: device.download(&key_scales)?,
                    value_scales: device.download(&value_scales)?,
                });
            }
        };
        Ok(Self {
            cache,
            key_scales: ones.clone(),
            value_scales: ones,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_fp8_context_streams_from_ram_without_reencoding() {
        const ROWS: usize = 65;
        const PAIRS: usize = 2;
        struct Source {
            blocks: Vec<Vec<u8>>,
            scales: Vec<f32>,
        }
        impl IclSource for Source {
            fn block(&self, block: usize) -> (IclValues<'_>, IclValues<'_>) {
                (
                    IclValues::F8(&self.blocks[block], &self.scales),
                    IclValues::F8(&self.blocks[ICL_BLOCKS - 1 - block], &self.scales),
                )
            }
        }
        let Some(device) = Device::new().unwrap() else {
            return;
        };
        let source: Arc<dyn IclSource> = Arc::new(Source {
            blocks: (0..ICL_BLOCKS)
                .map(|i| vec![u8::try_from(i + 1).unwrap(); PAIRS * ROWS * HEAD_DIM])
                .collect(),
            scales: vec![0.25; PAIRS * HEAD_DIM],
        });
        let cache = IclCache::new(&device, Precision::Half, &source, (PAIRS, ROWS), true).unwrap();
        let mut slots = cache.slots(&device).unwrap();
        for _ in 0..2 {
            cache.begin(&device, &mut slots).unwrap();
            for block in 0..ICL_BLOCKS {
                let loaded = cache.acquire(&device, &slots, block).unwrap();
                let (k, v) = source.block(block);
                for (buffer, expected) in [(loaded.keys, k), (loaded.values, v)] {
                    let bytes = device.download(buffer).unwrap();
                    for pair in 0..PAIRS {
                        let start = pair * padded_rows(ROWS) * HEAD_DIM;
                        let len = ROWS * HEAD_DIM;
                        assert_eq!(&bytes[start..][..len], &expected.bytes()[pair * len..][..len]);
                    }
                }
                assert_eq!(
                    device.download(loaded.key_scales).unwrap(),
                    vec![0.25; PAIRS * HEAD_DIM]
                );
                cache.release(&device, &mut slots, block).unwrap();
            }
        }
    }

    #[test]
    fn native_fp8_export_rejects_nonfinite_sources_before_saturation() {
        let Some(device) = Device::new().unwrap() else {
            return;
        };
        for (bits, valid) in [(0x7bff, true), (0x7c00, false), (0xfc00, false), (0x7e00, false)] {
            let mut source = vec![0; 257 * HEAD_DIM];
            source[256 * HEAD_DIM] = bits; // check the reduction across peak chunks
            let (bytes, scales) = encode_fp8(&device, &source, (1, 257)).unwrap();
            let result = copy_fp8(
                257,
                &device.download(&bytes).unwrap(),
                &mut vec![0; source.len()],
                &device.download(&scales).unwrap(),
                &mut [0; 4 * HEAD_DIM],
            );
            assert_eq!(result.is_ok(), valid);
        }
    }
}

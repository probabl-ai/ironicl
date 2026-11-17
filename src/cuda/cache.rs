//! A context's ICL keys and values on the device: FP16 for `auto`, or FP8
//! E4M3 with a scale per (estimator, head, dim) for `fast`, kept on the device
//! or streamed from the host one block at a time on every pass.
//!
//! Streaming goes through two device slots, each with a pinned staging
//! buffer: [`IclCache::begin`] stages the first two blocks, [`IclCache::acquire`]
//! makes the kernels wait for a block's copy, and [`IclCache::release`] marks
//! the slot free once the block's kernels are queued and stages the block two
//! ahead into it.
#![allow(unsafe_code)] // Asynchronous copies from pinned staging buffers.

use std::sync::Arc;

use cuda_core::{DeviceBuffer, Event, PinnedHostBuffer};
use rayon::prelude::*;

use super::{
    HEAD_DIM, ICL_BLOCKS, Precision,
    device::{Device, gpu_error},
    launch::padded_rows,
};
use crate::error::{Error, Result};

/// A context's ICL keys and values on the host, which the cache uploads once
/// or streams on every pass.
pub trait IclSource: Send + Sync {
    /// ICL block `block`'s keys and values as FP16 bits, `[B, 8, N, 64]`.
    fn block(&self, block: usize) -> (&[u16], &[u16]);
}

/// 16-bit units of one block's keys (or values) for `pairs` pairs and `rows`
/// rows at `precision`: FP16, or FP8 bytes two to a unit.
pub(super) const fn units(precision: Precision, pairs: usize, rows: usize) -> usize {
    let padded = pairs * padded_rows(rows) * HEAD_DIM;
    match precision {
        Precision::Split => padded,
        Precision::Half => padded / 2,
    }
}

/// Keys or values `[pairs, rows, 64]` into `[pairs, padded_rows(rows), 64]`,
/// pair by pair in parallel, leaving the padding as it is.
fn pad<T: Copy + Send + Sync>(padded: &mut [T], values: &[T], rows: usize) {
    padded
        .par_chunks_mut(padded_rows(rows) * HEAD_DIM)
        .zip(values.par_chunks(rows * HEAD_DIM))
        .for_each(|(to, from)| to[..from.len()].copy_from_slice(from));
}

/// One block's keys and values on the device as the attention kernels read
/// them: bytes of FP16 or FP8 values, rows padded to whole key tiles.
pub(super) struct KeyValues {
    pub(super) keys: DeviceBuffer<u8>,
    pub(super) values: DeviceBuffer<u8>,
}

impl KeyValues {
    /// FP16 keys and values as the bytes the attention kernels read.
    pub(super) fn from_halves(keys: DeviceBuffer<u16>, values: DeviceBuffer<u16>) -> Result<Self> {
        let bytes = |buffer: DeviceBuffer<u16>| {
            buffer
                .cast_chunks::<u8>()
                .map_err(|_| gpu_error("FP16 cache as bytes"))
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
    /// FP16 at [`Precision::Split`], FP8 at [`Precision::Half`].
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
            if keys.len() != pairs * rows * HEAD_DIM || values.len() != keys.len() {
                return Err(Error::InvalidArgument("ICL cache shape does not match".into()));
            }
            match (precision, streamed) {
                // A streamed FP16 cache copies straight from the context.
                (Precision::Split, true) => {}
                (Precision::Split, false) => blocks.push(KeyValues {
                    keys: upload_padded(device, keys, (pairs, rows))?,
                    values: upload_padded(device, values, (pairs, rows))?,
                }),
                (Precision::Half, _) => {
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
            }
        }
        let storage = match (precision, streamed) {
            (_, false) => Storage::Resident(blocks),
            (Precision::Split, true) => Storage::Streamed(Host::Context(Arc::clone(source))),
            (Precision::Half, true) => Storage::Streamed(Host::Encoded(encoded)),
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
        // FP16 units of the staging buffers, which hold whole pages.
        let staged_keys = bytemuck::cast_slice_mut::<u8, u16>(slot.staged_keys.as_mut_slice());
        let staged_values = bytemuck::cast_slice_mut::<u8, u16>(slot.staged_values.as_mut_slice());
        match host {
            Host::Context(source) => {
                let (keys, values) = source.block(block);
                pad(staged_keys, keys, self.rows);
                pad(staged_values, values, self.rows);
            }
            Host::Encoded(blocks) => {
                let (keys, values) = &blocks[block];
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
        for (to, from) in [
            (&mut slot.block.keys, &slot.staged_keys),
            (&mut slot.block.values, &slot.staged_values),
        ] {
            // SAFETY: the staging buffers stay untouched until `copied`
            // completes: the next `stage` into this slot and the slot's drop
            // wait for it.
            unsafe { to.copy_from_pinned_host_async(stream, from) }.map_err(gpu_error)?;
        }
        slot.copied = Some(stream.record_event(None).map_err(gpu_error)?);
        Ok(())
    }
}

/// `[pairs, 64]` ones on the device: the scales of FP16 keys and values.
fn ones(device: &Device, pairs: usize) -> Result<DeviceBuffer<f32>> {
    device.upload(&vec![1.0; pairs * HEAD_DIM])
}

/// FP16 keys or values `[pairs, rows, 64]` uploaded with their rows padded.
fn upload_padded(device: &Device, values: &[u16], (pairs, rows): (usize, usize)) -> Result<DeviceBuffer<u8>> {
    let mut padded = vec![0u16; units(Precision::Split, pairs, rows)];
    pad(&mut padded, values, rows);
    device.upload(bytemuck::cast_slice(&padded))
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
    /// `precision` stores them.
    pub(super) fn new(
        device: &Device,
        precision: Precision,
        (keys, values): (&[u16], &[u16]),
        (pairs, rows): (usize, usize),
    ) -> Result<Self> {
        if precision == Precision::Split {
            let ones = vec![1.0; pairs * HEAD_DIM];
            return Ok(Self {
                cache: KeyValues {
                    keys: upload_padded(device, keys, (pairs, rows))?,
                    values: upload_padded(device, values, (pairs, rows))?,
                },
                key_scales: ones.clone(),
                value_scales: ones,
            });
        }
        let ((keys, key_scales), (values, value_scales)) = (
            encode_fp8(device, keys, (pairs, rows))?,
            encode_fp8(device, values, (pairs, rows))?,
        );
        Ok(Self {
            cache: KeyValues { keys, values },
            key_scales: device.download(&key_scales)?,
            value_scales: device.download(&value_scales)?,
        })
    }
}

//! The device buffers a prediction or a build reuses, one per role, sized together
//! before a pass so no buffer grows while kernels may still read it.
//!
//! Several roles share a buffer in turn, in stream order:
//!
//! | Buffer | Holds |
//! |---|---|
//! | `x0` | the column embeddings |
//! | `x1` | the row-stage sequences; the ICL queries in FP32, then the block's `mid` |
//! | `a` | the 16-bit operands of up to `[B q, 512]`: norm1, the gate's hidden layer, the attention output, norm2, the decoder's norm |
//! | `w` | the 16-bit operands of up to `[B q, 1024]`: the gate's input, linear1's output, the decoder's hidden layer |
//!
//! A 16-bit operand is its values in FP16, and for split precision their low
//! parts in a buffer of their own.

use cuda_core::{DeviceBuffer, DeviceCopy, DriverError};

use super::{COLUMN_DIM, HEAD_DIM, HEADS, ICL_DIM, RESERVED, device::Device, launch::padded_query_rows};

/// A context's shape: `B` estimators, `H` features, `N` training rows, and
/// the decoder's outputs per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub ensembles: usize,
    pub features: usize,
    pub rows: usize,
    pub outputs: usize,
}

/// A run of rows `start..start + len` and their state between ICL blocks,
/// `[B, len, 512]`.
pub(super) struct Run<H> {
    pub(super) start: usize,
    pub(super) len: usize,
    pub(super) hidden: H,
}

/// A 16-bit operand's FP16 values and their low parts.
#[derive(Default)]
pub(super) struct Halves {
    pub(super) hi: Buffer<u16>,
    pub(super) lo: Buffer<u16>,
}

#[derive(Default)]
pub(super) struct Workspace {
    pub(super) x0: Buffer<f32>,
    pub(super) x1: Buffer<f32>,
    /// FP32 operands of the CUDA-core products.
    pub(super) normed: Buffer<f32>,
    pub(super) attended: Buffer<f32>,
    pub(super) wide: Buffer<f32>,
    pub(super) gate_hidden: Buffer<f32>,
    /// The `QASSMax` gate, `[B q 8, 64]`.
    pub(super) gate: Buffer<f32>,
    /// The CLS tokens, `[B q, 4, 128]`.
    pub(super) cls: Buffer<f32>,
    pub(super) a: Halves,
    pub(super) w: Halves,
    /// FP16 queries packed for ICL attention, `[B * 8, qp, 64]`.
    pub(super) packed: Buffer<u16>,
    /// ICL attention's chunk outputs and log-sum-exps, and their merge.
    pub(super) part: Buffer<f32>,
    pub(super) lse: Buffer<f32>,
    pub(super) attention: Buffer<f32>,
    /// Split-K partial sums of the CUDA-core products, `[splits, m, n]`.
    pub(super) partial: Buffer<f32>,
    pub(super) logits: Buffer<f32>,
}

/// A pass's buffers: the workspace its stages share, and each run's rows'
/// state between ICL blocks, `[B, run, 512]`.
#[derive(Default)]
pub(super) struct Buffers {
    pub(super) work: Workspace,
    pub(super) hidden: Vec<Buffer<f32>>,
}

/// A device buffer that only grows.
pub(super) struct Buffer<T: DeviceCopy>(Option<DeviceBuffer<T>>);

impl<T: DeviceCopy> Default for Buffer<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: DeviceCopy> Buffer<T> {
    /// Grow to at least `len` elements; the old contents are not kept.
    fn reserve(&mut self, device: &Device, len: usize) -> Result<(), DriverError> {
        if self.0.as_ref().is_none_or(|buffer| buffer.len() < len) {
            self.0 = Some(device.zeroed(len.max(1))?);
        }
        Ok(())
    }

    pub(super) fn get(&self) -> &DeviceBuffer<T> {
        self.0
            .as_ref()
            .expect("the workspace reserves every buffer before a pass")
    }

    pub(super) fn get_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.0
            .as_mut()
            .expect("the workspace reserves every buffer before a pass")
    }
}

impl Buffers {
    /// Reserve every buffer for a pass of `pass` rows in runs of `run` rows.
    pub(super) fn reserve(
        &mut self,
        device: &Device,
        shape: Shape,
        (run, pass): (usize, usize),
    ) -> Result<(), DriverError> {
        let seqs = shape.ensembles * run;
        self.hidden.resize_with(pass.div_ceil(run), Buffer::default);
        for buffer in &mut self.hidden {
            buffer.reserve(device, seqs * ICL_DIM)?;
        }
        self.work.reserve(device, shape, run)
    }
}

impl Workspace {
    /// Reserve every buffer for runs of `run` rows.
    pub(super) fn reserve(&mut self, device: &Device, shape: Shape, run: usize) -> Result<(), DriverError> {
        let (b, h) = (shape.ensembles, shape.features);
        let seqs = b * run;
        let columns = b * h * run;
        let tokens = seqs * (h + RESERVED);
        let pairs = b * HEADS;
        let padded = padded_query_rows(run);
        // A shorter run can split the keys into more chunks: reserve for the
        // most query rows times chunks of any run up to `run` rows.
        let chunks = std::iter::once(16)
            .chain((64..=padded).step_by(64))
            .map(|queries| queries * device.icl_split(pairs, queries, shape.rows).1)
            .max()
            .unwrap_or(padded);
        let state = (columns * COLUMN_DIM)
            .max(tokens * COLUMN_DIM)
            .max(seqs * ICL_DIM);
        let wide = (columns * 2 * COLUMN_DIM)
            .max(tokens * 3 * COLUMN_DIM)
            .max(seqs * 2 * ICL_DIM);
        for buffer in [&mut self.x0, &mut self.x1, &mut self.normed, &mut self.attended] {
            buffer.reserve(device, state)?;
        }
        self.wide.reserve(device, wide)?;
        for buffer in [&mut self.gate_hidden, &mut self.gate, &mut self.cls] {
            buffer.reserve(device, seqs * ICL_DIM)?;
        }
        for (halves, len) in [(&mut self.a, seqs * ICL_DIM), (&mut self.w, 2 * seqs * ICL_DIM)] {
            halves.hi.reserve(device, len)?;
            halves.lo.reserve(device, len)?;
        }
        self.packed.reserve(device, pairs * padded * HEAD_DIM)?;
        self.part.reserve(device, pairs * chunks * HEAD_DIM)?;
        self.lse.reserve(device, pairs * chunks)?;
        self.attention.reserve(device, pairs * padded * HEAD_DIM)?;
        // Four 64 x 64 tiles per multiprocessor.
        self.partial
            .reserve(device, 4 * device.multiprocessors() * 64 * 64)?;
        self.logits.reserve(device, seqs * shape.outputs)
    }
}

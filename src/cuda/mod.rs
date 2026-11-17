//! NVIDIA GPU prediction: Ironicl's kernels, compiled to PTX by cuda-oxide's
//! codegen backend and embedded in the library, run through `cuda-core`.
//!
//! [`build`] runs a context's build on the GPU: the training rows' pass
//! through every stage, keeping each block's keys and values; the ICL ones in
//! FP32 for `safe`, and in FP16 for `auto` and `fast`.
//!
//! A [`Predictor`] runs a cached context's whole prediction on the GPU: the
//! column stage against the cached inducing points, the row transformer, the
//! twelve ICL blocks against the cached keys and values, and the decoder.
//! Products run at a [`Precision`]: in FP32 on the CUDA cores for
//! `optimize="safe"`, which keeps the ICL keys and values in FP32; on tensor
//! cores for `"auto"`, in split FP16 parts close to FP32, and for `"fast"`, in
//! FP16, both accumulating in FP32. Norms and softmax run in FP32. The ICL
//! keys and values are FP16 under `"auto"`, and FP8 E4M3 with a scale per
//! (estimator, head, dim) under `"fast"`, whose wide ICL attention also
//! accumulates in FP16; under both, small products run in FP32 on the CUDA
//! cores. When the ICL cache does not fit the device with room to predict, it
//! streams from host memory one block at a time.
//!
//! The kernels build twice: at `sm_80` for Ampere, embedded by the build, and
//! at `sm_89` for Ada and later, which converts FP8 in hardware and leaves room
//! for the instructions those GPUs add (see `build.rs` and the Makefile).
//!
//! - [`kernels`]: the kernels.
//! - [`build`]: a context's build.
//! - [`device`]: the device and the kernels for its architecture.
//! - [`launch`]: one launcher per kernel family, the only launches.
//! - [`weights`]: the model's weights, resolved once into typed blocks.
//! - [`plan`]: how a context uses device memory.
//! - [`cache`]: the ICL keys and values, resident or streamed.
//! - [`workspace`]: the buffers a prediction reuses.
//! - [`predictor`]: a context's prediction.

mod build;
mod cache;
mod device;
mod kernels;
mod launch;
mod plan;
mod predictor;
mod testing;
mod weights;
mod workspace;

pub use build::{BuildInputs, BuildOutputs, build, context_bytes, fits};
pub use cache::{BuiltCache, IclSource, IclValues};
pub use device::{Arch, Device};
pub use predictor::{ContextTensors, Prediction, Predictor};
pub use weights::ModelWeights;
pub use workspace::Shape;

use crate::executor::{COLUMN_BLOCKS, COLUMN_DIM, HEADS, ICL_DIM, INDUCING, Optimize, RESERVED};

/// Attention head width the ICL kernels are built for.
pub const HEAD_DIM: usize = 64;
/// Keys per tile of the ICL attention kernels.
pub const KEY_TILE: usize = 64;
/// ICL transformer blocks.
pub const ICL_BLOCKS: usize = crate::executor::ICL_BLOCKS;
/// Values of one column's cached inducing-point keys (or values): 8 heads x
/// 128 inducing points x 16 dims.
const COLUMN_CACHE: usize = HEADS * INDUCING * 16;
const ROW_BLOCKS: usize = 3;
/// Scale of a value's low FP16 part; matches `LO_SCALE` in the kernels.
const LO_SCALE: f32 = 2048.0;

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

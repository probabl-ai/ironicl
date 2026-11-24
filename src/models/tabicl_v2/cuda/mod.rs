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
//! The kernels build once per target in `IRONICL_CUDA_ARCH`: by default at
//! `sm_80` for Ampere and at `sm_89` for Ada and later, which converts FP8 in
//! hardware. Each device loads the most specific build for it (see `build.rs`).
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
mod launch;
mod plan;
mod predictor;
mod testing;
mod weights;
mod workspace;

// The device and the kernels live in crate::kernels::cuda.
use crate::kernels::cuda::{self as device, kernels};

pub use build::{BuildInputs, BuildOutputs, build, context_bytes, fits};
pub use cache::{BuiltCache, IclSource, IclValues};
pub use device::{Arch, Device};
pub use predictor::{ContextTensors, Prediction, Predictor};
pub use weights::ModelWeights;
pub use workspace::Shape;

use crate::models::tabicl_v2::cpu::{COLUMN_BLOCKS, COLUMN_DIM, HEADS, ICL_DIM, INDUCING, RESERVED};

pub use crate::kernels::cuda::launch::{HEAD_DIM, KEY_TILE, Precision};
/// ICL transformer blocks.
pub const ICL_BLOCKS: usize = crate::models::tabicl_v2::cpu::ICL_BLOCKS;
/// Values of one column's cached inducing-point keys (or values): 8 heads x
/// 128 inducing points x 16 dims.
const COLUMN_CACHE: usize = HEADS * INDUCING * 16;
const ROW_BLOCKS: usize = 3;

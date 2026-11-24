//! `TabICLv2`.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub(crate) mod ensemble;
pub(crate) mod execution_plan;
pub mod fitted;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod metal;
pub mod preprocess;
pub mod quantiles;
pub mod request;

/// Where a context predicts.
#[derive(Debug, Clone)]
pub(crate) enum Device {
    Cpu,
    /// The context's Metal predict graph.
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    Metal(std::sync::Arc<metal::GpuContext>),
    /// The context's CUDA predictor, its ICL cache resident or streamed.
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<cuda::Predictor>),
}

impl Device {
    /// `"cpu"`, `"metal"`, `"cuda"`, or `"cuda-streamed"` when the CUDA
    /// predictor streams the ICL cache from host memory on every pass.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(_) => "metal",
            #[cfg(feature = "cuda")]
            Self::Cuda(predictor) if predictor.streamed() => "cuda-streamed",
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
        }
    }
}

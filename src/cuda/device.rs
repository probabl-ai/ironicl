//! The CUDA device: its streams and the kernels for its architecture.

use std::{fmt::Display, sync::Arc};

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, DriverError, PinnedHostBuffer, Stream};

use super::kernels::device::{self as kernels, LoadedModule};
use crate::error::{Error, Result};

/// The kernels built at `sm_89`, or nothing for a build without them.
const PTX_SM89: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels_sm89.ptx"));

pub(super) fn gpu_error(error: impl Display) -> Error {
    Error::UnsupportedBackend(format!("CUDA: {error}"))
}

/// Whether a driver error is the device running out of memory.
pub(super) fn out_of_memory(error: DriverError) -> bool {
    error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY
}

/// Which build of the kernels a device runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// `sm_80`, for Ampere: FP8 decodes in software.
    Ampere,
    /// `sm_89`, for Ada and later: FP8 converts in hardware.
    Ada,
}

/// Device 0, with the kernels for its architecture loaded.
pub struct Device {
    context: Arc<CudaContext>,
    /// The kernels' stream.
    stream: Arc<Stream>,
    /// Copies of streamed ICL caches, overlapping the kernels.
    copy: Arc<Stream>,
    multiprocessors: usize,
    arch: Arch,
    pub(super) kernels: LoadedModule,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Device")
            .field("arch", &self.arch)
            .field("multiprocessors", &self.multiprocessors)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Device 0 with the kernels loaded, or `None` without a CUDA driver or
    /// an Ampere or later device, or with `IRONICL_NO_CUDA` set. Ada and later
    /// GPUs run the `sm_89` kernels when the build has them, unless the driver
    /// is too old for their PTX or `IRONICL_CUDA_MODULE=sm80` asks for the
    /// Ampere ones.
    ///
    /// # Errors
    ///
    /// Returns an error if a device exists but the kernels fail to load.
    pub fn new() -> Result<Option<Self>> {
        if std::env::var_os("IRONICL_NO_CUDA").is_some() {
            return Ok(None);
        }
        let Ok(context) = CudaContext::new(0) else {
            return Ok(None);
        };
        let capability = context.compute_capability().map_err(gpu_error)?;
        if capability < (8, 0) {
            return Ok(None);
        }
        // IRONICL_CUDA_MODULE=sm80 loads the Ampere kernels on any GPU, to test them.
        let ada = !PTX_SM89.is_empty()
            && capability >= (8, 9)
            && std::env::var_os("IRONICL_CUDA_MODULE").is_none_or(|module| module != "sm80");
        let ada = if ada {
            match context.load_module_from_ptx_src(PTX_SM89) {
                Ok(module) => Some(kernels::from_module(module).map_err(gpu_error)?),
                Err(error) if error.is_unsupported_ptx_version() => None,
                Err(error) => return Err(gpu_error(error)),
            }
        } else {
            None
        };
        let (kernels, arch) = match ada {
            Some(kernels) => (kernels, Arch::Ada),
            None => (kernels::load(&context).map_err(gpu_error)?, Arch::Ampere),
        };
        let stream = context.new_stream().map_err(gpu_error)?;
        let copy = context.new_stream().map_err(gpu_error)?;
        let multiprocessors =
            usize::try_from(context.multiprocessor_count().map_err(gpu_error)?).unwrap_or(1);
        Ok(Some(Self {
            context,
            stream,
            copy,
            multiprocessors,
            arch,
            kernels,
        }))
    }

    /// The kernels this device runs.
    #[must_use]
    pub const fn arch(&self) -> Arch {
        self.arch
    }

    pub(super) const fn multiprocessors(&self) -> usize {
        self.multiprocessors
    }

    /// The kernels' stream.
    pub(super) fn stream(&self) -> &Stream {
        &self.stream
    }

    pub(super) fn copy_stream(&self) -> &Stream {
        &self.copy
    }

    /// Device memory free now, in bytes.
    pub(super) fn free_memory(&self) -> Result<usize> {
        self.context.bind_to_thread().map_err(gpu_error)?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two valid out-pointers; the context is current on this thread.
        #[allow(unsafe_code)]
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&raw mut free, &raw mut total) };
        if status == 0 {
            Ok(free)
        } else {
            Err(gpu_error(format!("cuMemGetInfo failed with {status}")))
        }
    }

    /// `len` zeroed elements on the device.
    pub(super) fn zeroed<T: DeviceCopy>(
        &self,
        len: usize,
    ) -> std::result::Result<DeviceBuffer<T>, DriverError> {
        DeviceBuffer::zeroed(&self.stream, len)
    }

    /// `values` copied to the device.
    pub(super) fn upload<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::from_host(&self.stream, values).map_err(gpu_error)
    }

    /// `buffer` copied to the host, once all earlier work is done.
    pub(super) fn download<T: DeviceCopy>(&self, buffer: &DeviceBuffer<T>) -> Result<Vec<T>> {
        buffer.to_host_vec(&self.stream).map_err(gpu_error)
    }

    /// `len` zeroed elements of pinned host memory, which copies to the device
    /// asynchronously.
    pub(super) fn pinned<T: DeviceCopy>(&self, len: usize) -> Result<PinnedHostBuffer<T>> {
        PinnedHostBuffer::zeroed(&self.context, len).map_err(gpu_error)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use cuda_core::embedded::{ArtifactPayloadKind, embedded_modules_from_current_exe};

    use super::PTX_SM89;

    /// A PTX module's `.version`, `.target`, and entries.
    fn module(ptx: &str) -> (&str, &str, BTreeSet<&str>) {
        let directive = |key: &str| {
            ptx.lines()
                .find_map(|line| line.trim().strip_prefix(key).map(str::trim))
                .expect("PTX directive")
        };
        let entries = ptx
            .lines()
            .filter_map(|line| line.trim().strip_prefix(".visible .entry "))
            .map(|line| line.split('(').next().unwrap_or(line))
            .collect();
        (directive(".version"), directive(".target"), entries)
    }

    /// Each build of the kernels targets the GPUs it loads on, with the same
    /// kernels: an instruction newer GPUs add raises a module's target
    /// silently, and older GPUs could no longer load it.
    #[test]
    fn kernels_target_their_architectures() {
        let modules = embedded_modules_from_current_exe().expect("read the embedded kernels");
        let ptx = modules
            .iter()
            .find(|module| module.name() == env!("CARGO_PKG_NAME"))
            .and_then(|module| module.payload(ArtifactPayloadKind::Ptx))
            .expect("the embedded sm_80 PTX");
        let (version, target, ampere) = module(std::str::from_utf8(ptx).expect("PTX is text"));
        assert_eq!(target, "sm_80");
        assert!(version.starts_with("7."), "sm_80 kernels at PTX ISA {version}");
        if !PTX_SM89.is_empty() {
            let (_, target, ada) = module(PTX_SM89);
            assert_eq!(target, "sm_89");
            assert_eq!(ada, ampere);
        }
    }
}

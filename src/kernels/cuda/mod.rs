//! The CUDA device: its streams and the kernels for its architecture.

pub(crate) mod kernels;
pub(crate) mod launch;
mod ops;

use std::{fmt::Display, sync::Arc};

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, DriverError, PinnedHostBuffer, Stream};

use self::kernels::device::{self as module, LoadedModule};
use crate::error::{Error, Result};

/// Which GPUs a module's PTX loads on (see `build.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    /// `sm_XY`: compute capability X.Y and later.
    Plain,
    /// `sm_XYf`: X.Y and later X.* GPUs.
    Family,
    /// `sm_XYa`: X.Y only.
    Specific,
}

/// One build of the kernels.
struct Module {
    /// Its target: `sm_<major><minor>`, then `f` or `a` for those kinds.
    name: &'static str,
    /// The PTX, or `None` for the build cuda-oxide embeds itself.
    ptx: Option<&'static str>,
}

impl Module {
    /// The target's kind, major version, and minor version.
    fn target(&self) -> (Kind, i32, i32) {
        let digits = self.name.trim_start_matches("sm_");
        let (digits, kind) = if let Some(digits) = digits.strip_suffix('a') {
            (digits, Kind::Specific)
        } else if let Some(digits) = digits.strip_suffix('f') {
            (digits, Kind::Family)
        } else {
            (digits, Kind::Plain)
        };
        let version: i32 = digits.parse().unwrap_or(0);
        (kind, version / 10, version % 10)
    }

    fn loads_on(&self, device: (i32, i32)) -> bool {
        let (kind, major, minor) = self.target();
        match kind {
            Kind::Plain => (major, minor) <= device,
            Kind::Family => major == device.0 && minor <= device.1,
            Kind::Specific => (major, minor) == device,
        }
    }
}

/// The builds of the kernels, from `IRONICL_CUDA_ARCH`; the first is
/// cuda-oxide's own.
const MODULES: &[Module] = include!(concat!(env!("OUT_DIR"), "/cuda_modules.rs"));
/// Freed stream-ordered memory the pool keeps across synchronizations.
const POOL_KEPT_BYTES: u64 = 1 << 30;

pub(crate) fn gpu_error(error: impl Display) -> Error {
    Error::UnsupportedBackend(format!("CUDA: {error}"))
}

/// Whether a driver error is the device running out of memory.
pub(crate) fn out_of_memory(error: DriverError) -> bool {
    error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY
}

/// Which build of the kernels a device runs: its target, such as `sm_89`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch(&'static str);

impl Arch {
    /// The target, such as `sm_89`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.0
    }
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
    pub(crate) kernels: LoadedModule,
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
    /// an Ampere or later device, or with `IRONICL_NO_CUDA` set. The device
    /// runs the build that loads on it with the most specific target: an `a`
    /// target, then an `f` one, then the newest plain one; a build whose PTX
    /// the driver is too old for gives way to the next. `IRONICL_CUDA_MODULE`
    /// names the one build to consider, such as `sm_80`, to test it.
    ///
    /// # Errors
    ///
    /// Returns an error if a device exists but no build loads on it, or the
    /// kernels fail to load.
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
        let only = std::env::var("IRONICL_CUDA_MODULE")
            .ok()
            .filter(|name| !name.is_empty());
        let mut candidates: Vec<&Module> = MODULES
            .iter()
            .filter(|module| module.loads_on(capability))
            .filter(|module| only.as_deref().is_none_or(|name| name == module.name))
            .collect();
        candidates.sort_by_key(|module| std::cmp::Reverse(module.target()));
        let mut loaded = None;
        for module in candidates {
            let kernels = match module.ptx {
                None => module::load(&context).map_err(gpu_error)?,
                Some(ptx) => match context.load_module_from_ptx_src(ptx) {
                    Ok(ptx) => module::from_module(ptx).map_err(gpu_error)?,
                    Err(error) if error.is_unsupported_ptx_version() => continue,
                    Err(error) => return Err(gpu_error(error)),
                },
            };
            loaded = Some((kernels, Arch(module.name)));
            break;
        }
        let Some((kernels, arch)) = loaded else {
            let built: Vec<&str> = MODULES.iter().map(|module| module.name).collect();
            return Err(gpu_error(format!(
                "no build of the kernels loads on this sm_{}{} GPU{}: built for {}; rebuild with \
                 IRONICL_CUDA_ARCH listing a target for it",
                capability.0,
                capability.1,
                only.map_or_else(String::new, |name| format!(" with IRONICL_CUDA_MODULE={name}")),
                built.join(", "),
            )));
        };
        // Stream-ordered frees return memory to the pool, and by default a synchronization
        // releases it to the driver, so the next pass maps it again. Keep a pass's worth.
        context
            .default_mem_pool()
            .and_then(|pool| pool.set_release_threshold(POOL_KEPT_BYTES))
            .map_err(gpu_error)?;
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

    pub(crate) const fn multiprocessors(&self) -> usize {
        self.multiprocessors
    }

    /// The kernels' stream.
    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    pub(crate) fn copy_stream(&self) -> &Stream {
        &self.copy
    }

    /// Device memory free now, in bytes.
    pub(crate) fn free_memory(&self) -> Result<usize> {
        self.context.bind_to_thread().map_err(gpu_error)?;
        let (mut free, mut total) = (0usize, 0usize);
        #[expect(
            unsafe_code,
            reason = "one driver call for free memory; cuda-core has no wrapper"
        )]
        // SAFETY: `cuMemGetInfo_v2` writes one `size_t` through each of two valid, distinct
        // pointers to locals; `bind_to_thread` above made the context it reports on current.
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&raw mut free, &raw mut total) };
        if status == 0 {
            Ok(free)
        } else {
            Err(gpu_error(format!("cuMemGetInfo failed with {status}")))
        }
    }

    /// `len` zeroed elements on the device.
    pub(crate) fn zeroed<T: DeviceCopy>(
        &self,
        len: usize,
    ) -> std::result::Result<DeviceBuffer<T>, DriverError> {
        DeviceBuffer::zeroed(&self.stream, len)
    }

    /// Allocate on the kernel stream without waiting for the GPU. Drop frees in stream order.
    ///
    /// # Safety
    /// Initialize every element on the same stream before reading it, including padding.
    #[expect(unsafe_code, reason = "the caller initializes this stream-ordered allocation")]
    pub(crate) unsafe fn scratch<T: DeviceCopy>(
        &self,
        len: usize,
    ) -> std::result::Result<DeviceBuffer<T>, DriverError> {
        // SAFETY: the caller is responsible for initialization.
        unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }
    }

    /// `values` copied to the device.
    pub(crate) fn upload<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::from_host(&self.stream, values).map_err(gpu_error)
    }

    /// `buffer` copied to the host, once all earlier work is done.
    pub(crate) fn download<T: DeviceCopy>(&self, buffer: &DeviceBuffer<T>) -> Result<Vec<T>> {
        buffer.to_host_vec(&self.stream).map_err(gpu_error)
    }

    /// `len` zeroed elements of pinned host memory, which copies to the device
    /// asynchronously.
    pub(crate) fn pinned<T: DeviceCopy>(&self, len: usize) -> Result<PinnedHostBuffer<T>> {
        PinnedHostBuffer::zeroed(&self.context, len).map_err(gpu_error)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use cuda_core::embedded::{ArtifactPayloadKind, embedded_modules_from_current_exe};

    use super::MODULES;

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
            .expect("the embedded PTX");
        let (version, target, entries) = module(std::str::from_utf8(ptx).expect("PTX is text"));
        assert_eq!(target, MODULES[0].name);
        if target == "sm_80" {
            assert!(version.starts_with("7."), "sm_80 kernels at PTX ISA {version}");
        }
        for build in &MODULES[1..] {
            let (_, target, other) = module(build.ptx.expect("nested builds carry their PTX"));
            assert_eq!(target, build.name);
            assert_eq!(other, entries, "{} kernels differ", build.name);
        }
    }
}

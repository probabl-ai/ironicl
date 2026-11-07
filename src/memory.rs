//! Memory planning: how much memory a context needs, how much the machine can
//! give, and whether it fits.
//!
//! Memory is modeled as pools by whether they share the host's RAM, not by
//! vendor: a shared device (an Apple GPU today) draws on the host's memory up to
//! its own working-set limit, and a separate device (a discrete GPU) has its own.
//! The plan is advisory here; callers decide what to do when it does not fit.

use std::fmt;

use crate::{
    TaskKind,
    error::{Error, Result},
};

/// Memory the operating system can give this process now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    /// Physical memory available without paging.
    pub available: u64,
    /// A cap that fails instead of paging: a cgroup or job limit, or the
    /// remaining Windows commit charge. `None` when there is none.
    pub hard_limit: Option<u64>,
}

impl HostMemory {
    /// The bytes this process can use: `available`, capped by `hard_limit`.
    #[must_use]
    pub fn usable(self) -> u64 {
        self.hard_limit
            .map_or(self.available, |limit| self.available.min(limit))
    }
}

/// A compute device's memory, described by whether it shares the host's RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemory {
    /// Unified memory, such as an Apple GPU: the host's RAM, capped by the
    /// device's working-set limit.
    Shared { limit: u64 },
    /// The device's own memory, such as a discrete GPU.
    Separate { available: u64 },
}

/// The memory pools a plan draws on. Unknown pools make the verdict unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pools {
    pub host: Option<HostMemory>,
    pub device: Option<DeviceMemory>,
}

impl Pools {
    /// The host's memory from the operating system, with no device.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            host: host_memory(),
            device: None,
        }
    }

    /// Bytes available where the context is built, or `None` if unknown.
    #[must_use]
    pub fn available(self) -> Option<u64> {
        let host = self.host.map(HostMemory::usable);
        match self.device {
            None => host,
            Some(DeviceMemory::Shared { limit }) => host.map(|host| host.min(limit)),
            Some(DeviceMemory::Separate { available }) => Some(available),
        }
    }
}

/// The dataset shape a context is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextShape {
    pub task: TaskKind,
    pub estimators: u64,
    pub rows: u64,
    /// Features after constant columns are dropped.
    pub features: u64,
}

/// Bytes per stored ICL key and value element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Storage {
    pub name: &'static str,
    pub key_bytes: u64,
    pub value_bytes: u64,
}

impl Storage {
    pub const FP32: Self = Self {
        name: "FP32",
        key_bytes: 4,
        value_bytes: 4,
    };
    /// FP32 keys with BF16 values: `optimize="fast"` on a BF16-capable CPU.
    pub const FP32_KEY_BF16_VALUE: Self = Self {
        name: "FP32 keys, BF16 values",
        key_bytes: 4,
        value_bytes: 2,
    };
    /// FP16 keys and values: `optimize="auto"`/`"fast"` on an Apple GPU.
    pub const FP16: Self = Self {
        name: "FP16",
        key_bytes: 2,
        value_bytes: 2,
    };
}

/// Where the context is built, which sets the transient build memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builder {
    Cpu,
    Gpu,
}

/// GPU build groups stop growing past this size: one wide estimator already
/// keeps an M1 Max busy (100 features at 10,000 rows built as fast in groups
/// of 1, 2, or 4), while narrow ones need several per run.
const GPU_BUILD_TARGET: u64 = 6 << 30;
/// Reserve for Python, the allocator, preprocessing, and query buffers.
const HEADROOM: u64 = 512 << 20;
/// A build may use at most this share of the available memory.
const BUILD_SHARE_DIVISOR: u64 = 2;
/// The CPU build's own estimate is 2-3x its measured peak (see
/// `Executor::build_tensors_cpu`), so the planner counts half of it.
const CPU_BUILD_ESTIMATE_DIVISOR: u64 = 2;

const ICL_LAYERS: u64 = 12;
const ICL_DIM: u64 = 512;
/// Column K/V per estimator per feature stream: 2 (K, V) x 3 blocks x 128
/// inducing points x 128 dimensions x 4 bytes.
const COLUMN_BYTES_PER_STREAM: u64 = 2 * 3 * 128 * 128 * 4;
const RESERVED_STREAMS: u64 = 4;

/// A memory plan for one context on one machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub shape: ContextShape,
    pub storage: Storage,
    pub builder: Builder,
    /// Model weights in FP32.
    pub weight_bytes: u64,
    /// Column and ICL caches.
    pub context_bytes: u64,
    /// Transient build memory for one estimator.
    pub build_estimator_bytes: u64,
    /// Estimators built together.
    pub build_group: u64,
    /// Weights, context, one build group, and headroom.
    pub peak_bytes: u64,
    /// Memory available to the builder, or `None` if unknown.
    pub available_bytes: Option<u64>,
    /// Ways to fit, empty when it already fits or memory is unknown.
    pub suggestions: Vec<Suggestion>,
}

/// One change that makes the plan fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Suggestion {
    /// Subsample to at most this many training rows.
    Rows(u64),
    /// Use this many estimators.
    Estimators(u64),
    /// Store the ICL caches in a smaller type; `option` is the caller's name for it.
    Storage { option: String, peak_bytes: u64 },
}

impl Plan {
    /// Whether the peak fits in the available memory, or `None` if unknown.
    #[must_use]
    pub fn fits(&self) -> Option<bool> {
        self.available_bytes.map(|available| self.peak_bytes <= available)
    }
}

/// Plan a context. `alternatives` are smaller storage types the caller can
/// offer, by the name users know them by (such as `optimize="fast"`); only those
/// that fit are suggested.
///
/// # Errors
///
/// Returns [`Error::ArithmeticOverflow`] when a size overflows `u64`, and
/// [`Error::InvalidArgument`] for an empty shape.
pub fn plan(
    shape: ContextShape,
    storage: Storage,
    builder: Builder,
    pools: Pools,
    alternatives: &[(String, Storage)],
) -> Result<Plan> {
    if shape.estimators == 0 || shape.rows == 0 || shape.features == 0 {
        return Err(Error::InvalidArgument(
            "memory plan needs estimators, rows, and features".into(),
        ));
    }
    let mut plan = sized(shape, storage, builder, pools)?;
    if plan.fits() == Some(false) {
        let available = pools.available().unwrap_or(0);
        let fits = |shape: ContextShape, storage: Storage| -> Result<Option<u64>> {
            let peak = sized(shape, storage, builder, pools)?.peak_bytes;
            Ok((peak <= available).then_some(peak))
        };
        let (mut low, mut high) = (0, shape.rows);
        while low < high {
            let rows = (low + high).div_ceil(2);
            if fits(ContextShape { rows, ..shape }, storage)?.is_some() {
                low = rows;
            } else {
                high = rows - 1;
            }
        }
        if low > 0 {
            plan.suggestions.push(Suggestion::Rows(low));
        }
        if let Some(estimators) = (1..shape.estimators)
            .rev()
            .find(|&estimators| matches!(fits(ContextShape { estimators, ..shape }, storage), Ok(Some(_))))
        {
            plan.suggestions.push(Suggestion::Estimators(estimators));
        }
        for (option, alternative) in alternatives {
            if let Some(peak_bytes) = fits(shape, *alternative)? {
                plan.suggestions.push(Suggestion::Storage {
                    option: option.clone(),
                    peak_bytes,
                });
            }
        }
    }
    Ok(plan)
}

/// The plan's sizes, without suggestions. A GPU build that cannot fit one
/// estimator on the device falls back to the CPU, as the executor does.
fn sized(shape: ContextShape, storage: Storage, builder: Builder, pools: Pools) -> Result<Plan> {
    let available = pools.available();
    let streams = shape.features + RESERVED_STREAMS;
    let column = mul(&[shape.estimators, streams, COLUMN_BYTES_PER_STREAM])?;
    let icl = mul(&[
        shape.estimators,
        ICL_LAYERS,
        shape.rows,
        ICL_DIM,
        storage.key_bytes + storage.value_bytes,
    ])?;
    let context_bytes = add(column, icl)?;
    let weight_bytes = shape.task.parameter_count() * 4;
    let device_limit = match pools.device {
        Some(DeviceMemory::Shared { limit }) => Some(limit),
        Some(DeviceMemory::Separate { available }) => Some(available),
        None => None,
    };
    let gpu_group = (builder == Builder::Gpu)
        .then(|| {
            let per_estimator = build_estimator_bytes(shape, Builder::Gpu)?;
            Ok::<_, Error>(gpu_build_group(
                shape.estimators,
                per_estimator,
                available,
                device_limit,
                None,
            ))
        })
        .transpose()?
        .flatten();
    let builder = if gpu_group.is_some() {
        Builder::Gpu
    } else {
        Builder::Cpu
    };
    let build_estimator_bytes = build_estimator_bytes(shape, builder)?;
    let build_group = gpu_group.unwrap_or_else(|| {
        cpu_build_group(
            shape.estimators,
            build_estimator_bytes.saturating_mul(CPU_BUILD_ESTIMATE_DIVISOR),
            available,
        )
    });
    let peak_bytes = add(
        add(weight_bytes, context_bytes)?,
        add(mul(&[build_group, build_estimator_bytes])?, HEADROOM)?,
    )?;
    Ok(Plan {
        shape,
        storage,
        builder,
        weight_bytes,
        context_bytes,
        build_estimator_bytes,
        build_group,
        peak_bytes,
        available_bytes: available,
        suggestions: Vec::new(),
    })
}

/// Transient memory to build one estimator.
fn build_estimator_bytes(shape: ContextShape, builder: Builder) -> Result<u64> {
    let (n, h) = (to_usize(shape.rows)?, to_usize(shape.features)?);
    let bytes = match builder {
        Builder::Gpu => gpu_build_estimator_bytes(n, h),
        Builder::Cpu => {
            let width = match shape.task {
                TaskKind::Classifier => 10,
                TaskKind::Regressor => 999,
            };
            crate::executor::cpu_build_estimator_bytes(n, h, width)? / to_usize(CPU_BUILD_ESTIMATE_DIVISOR)?
        }
    };
    u64::try_from(bytes).map_err(|_| Error::ArithmeticOverflow("build bytes"))
}

/// The GPU build graph's transient working set for one estimator: the column
/// stage's `[N, H + 4, 128]` FP32 activations about 8 times over, and the ICL
/// stage's `[N, 512]` FP32 activations about 16 times over.
pub(crate) fn gpu_build_estimator_bytes(n: usize, h: usize) -> usize {
    n.saturating_mul((h + 4) * 128 * 4 * 8 + 512 * 4 * 16)
}

/// Estimators the GPU builds together, or `None` when the CPU should build.
///
/// `budget` is `IRONICL_GPU_BUILD_BUDGET`: an exact cap, with a CPU build when
/// one estimator exceeds it. Otherwise the group is the largest that divides
/// the estimator count (every run has the same graph shape) and stays under the
/// smallest of a size that keeps the GPU busy, half the available memory, and
/// the device's limit; the CPU builds only when one estimator exceeds the device.
pub(crate) fn gpu_build_group(
    estimators: u64,
    per_estimator: u64,
    available: Option<u64>,
    device_limit: Option<u64>,
    budget: Option<u64>,
) -> Option<u64> {
    let per_estimator = per_estimator.max(1);
    let largest_under = |cap: u64| {
        (1..=estimators)
            .rev()
            .find(|&group| estimators.is_multiple_of(group) && group.saturating_mul(per_estimator) <= cap)
    };
    if let Some(budget) = budget {
        return largest_under(budget);
    }
    if device_limit.is_some_and(|limit| per_estimator > limit) {
        return None;
    }
    let mut target = GPU_BUILD_TARGET;
    if let Some(available) = available {
        target = target.min(available / BUILD_SHARE_DIVISOR);
    }
    if let Some(limit) = device_limit {
        target = target.min(limit);
    }
    Some(largest_under(target).unwrap_or(1))
}

/// Estimators the CPU builds together: as many as fit four activation limits
/// by the build's own estimate (2-3x its measured peak), and, when known, half
/// the available memory by the planner's halved estimate; in equal groups.
pub(crate) fn cpu_build_group(estimators: u64, estimate_bytes: u64, available: Option<u64>) -> u64 {
    let estimate = estimate_bytes.max(1);
    let mut fit = crate::executor::cpu_build_budget_bytes() / estimate;
    if let Some(available) = available {
        let counted = (estimate / CPU_BUILD_ESTIMATE_DIVISOR).max(1);
        fit = fit.min(available / BUILD_SHARE_DIVISOR / counted);
    }
    let fit = fit.clamp(1, estimators);
    estimators.div_ceil(estimators.div_ceil(fit))
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape;
        writeln!(
            f,
            "Context: {} ({} estimators x {} rows x {} features, ICL caches in {}).",
            gib(self.context_bytes),
            shape.estimators,
            shape.rows,
            shape.features,
            self.storage.name
        )?;
        write!(
            f,
            "Peak while building on the {}: {}, including {} for {} estimator(s) at a time",
            match self.builder {
                Builder::Cpu => "CPU",
                Builder::Gpu => "GPU",
            },
            gib(self.peak_bytes),
            gib(self.build_group * self.build_estimator_bytes),
            self.build_group
        )?;
        match self.available_bytes {
            None => write!(f, "; available memory is unknown.")?,
            Some(available) => write!(f, "; available: about {}.", gib(available))?,
        }
        if !self.suggestions.is_empty() {
            write!(f, "\nAny one of these fits:")?;
            for suggestion in &self.suggestions {
                match suggestion {
                    Suggestion::Rows(rows) => write!(
                        f,
                        "\n  - at most {rows} training rows (subsample X, y before fit)"
                    )?,
                    Suggestion::Estimators(n) => write!(f, "\n  - n_estimators={n}")?,
                    Suggestion::Storage { option, peak_bytes } => {
                        write!(f, "\n  - {option} (peak {})", gib(*peak_bytes))?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::cast_precision_loss)] // Display only.
fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / f64::from(1 << 30))
}

fn mul(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(1_u64, |product, value| product.checked_mul(*value))
        .ok_or(Error::ArithmeticOverflow("memory plan"))
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or(Error::ArithmeticOverflow("memory plan"))
}

fn to_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::ArithmeticOverflow("memory plan"))
}

/// The host's memory, or `None` on platforms without a probe.
#[must_use]
pub fn host_memory() -> Option<HostMemory> {
    platform::host_memory()
}

#[cfg(target_os = "linux")]
mod platform {
    use super::HostMemory;

    pub(super) fn host_memory() -> Option<HostMemory> {
        let available = mem_available(&std::fs::read_to_string("/proc/meminfo").ok()?)?;
        Some(HostMemory {
            available,
            hard_limit: cgroup_headroom(),
        })
    }

    /// `MemAvailable` from `/proc/meminfo`, in bytes.
    pub(super) fn mem_available(meminfo: &str) -> Option<u64> {
        let kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))?
            .trim()
            .strip_suffix("kB")?
            .trim()
            .parse::<u64>()
            .ok()?;
        kib.checked_mul(1024)
    }

    /// The cgroup memory limit minus current usage. Only this process's own
    /// cgroup is read; limits set on its ancestors are ignored.
    fn cgroup_headroom() -> Option<u64> {
        let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
        let read = |path: std::path::PathBuf| std::fs::read_to_string(path).ok();
        if let Some(path) = cgroup_v2_path(&cgroup) {
            let directory = std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
            let limit = limit_bytes(&read(directory.join("memory.max"))?)?;
            let usage = limit_bytes(&read(directory.join("memory.current"))?)?;
            return Some(limit.saturating_sub(usage));
        }
        let directory = std::path::Path::new("/sys/fs/cgroup/memory");
        let limit = limit_bytes(&read(directory.join("memory.limit_in_bytes"))?)?;
        let usage = limit_bytes(&read(directory.join("memory.usage_in_bytes"))?)?;
        Some(limit.saturating_sub(usage))
    }

    /// The unified (v2) hierarchy's path from `/proc/self/cgroup`: `0::/path`.
    pub(super) fn cgroup_v2_path(cgroup: &str) -> Option<&str> {
        cgroup.lines().find_map(|line| line.strip_prefix("0::"))
    }

    /// A cgroup byte count; `max` and v1's near-`u64::MAX` sentinel mean no limit.
    pub(super) fn limit_bytes(value: &str) -> Option<u64> {
        let value = value.trim();
        if value == "max" {
            return None;
        }
        value.parse::<u64>().ok().filter(|bytes| *bytes < 1 << 60)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::HostMemory;

    /// `struct vm_statistics64` from `<mach/vm_statistics.h>`.
    #[repr(C)]
    #[derive(Default)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u32,
        throttled_count: u32,
        external_page_count: u32,
        internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    }

    /// Total memory minus app memory (anonymous pages that are not
    /// purgeable), wired memory, and the compressor's pages: the complement of
    /// Activity Monitor's "Memory Used". File cache counts as available, since
    /// the kernel drops it without writing to swap.
    #[allow(unsafe_code)] // Mach and sysctl calls; std has no equivalent.
    pub(super) fn host_memory() -> Option<HostMemory> {
        const HOST_VM_INFO64: i32 = 4;
        unsafe extern "C" {
            fn mach_host_self() -> u32;
            fn host_page_size(host: u32, page_size: *mut usize) -> i32;
            fn host_statistics64(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                value: *mut std::ffi::c_void,
                length: *mut usize,
                new_value: *const std::ffi::c_void,
                new_length: usize,
            ) -> i32;
        }
        let mut stats = VmStatistics64::default();
        let mut count = u32::try_from(size_of::<VmStatistics64>() / size_of::<i32>()).ok()?;
        let (mut page_size, mut total, mut length) = (0_usize, 0_u64, size_of::<u64>());
        // SAFETY: `stats` holds `count` 32-bit words, the name is NUL-terminated,
        // `total` holds `length` bytes, and every out-pointer is valid.
        let ok = unsafe {
            let host = mach_host_self();
            host_page_size(host, &raw mut page_size) == 0
                && host_statistics64(host, HOST_VM_INFO64, (&raw mut stats).cast(), &raw mut count) == 0
                && sysctlbyname(
                    c"hw.memsize".as_ptr(),
                    (&raw mut total).cast(),
                    &raw mut length,
                    std::ptr::null(),
                    0,
                ) == 0
        };
        if !ok || length != size_of::<u64>() {
            return None;
        }
        let used_pages = u64::from(stats.internal_page_count.saturating_sub(stats.purgeable_count))
            + u64::from(stats.wire_count)
            + u64::from(stats.compressor_page_count);
        let used = used_pages.checked_mul(u64::try_from(page_size).ok()?)?;
        Some(HostMemory {
            available: total.saturating_sub(used),
            hard_limit: None,
        })
    }
}

#[cfg(windows)]
mod platform {
    use super::HostMemory;
    use windows_sys::Win32::System::{
        JobObjects::{
            JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            QueryInformationJobObject,
        },
        ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX},
        SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
        Threading::GetCurrentProcess,
    };

    /// Available physical memory, capped by any job memory limit and by the
    /// remaining commit charge: Windows does not overcommit, so past the
    /// commit limit allocations fail instead of paging.
    #[allow(unsafe_code)] // Win32 calls; std has no equivalent.
    pub(super) fn host_memory() -> Option<HostMemory> {
        // SAFETY: MEMORYSTATUSEX is plain data; all zeros is a valid value.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = u32::try_from(size_of::<MEMORYSTATUSEX>()).ok()?;
        // SAFETY: `status` is a valid MEMORYSTATUSEX with dwLength set.
        if unsafe { GlobalMemoryStatusEx(&raw mut status) } == 0 {
            return None;
        }
        let mut hard_limit = status.ullAvailPageFile;
        if let Some(job_headroom) = job_headroom() {
            hard_limit = hard_limit.min(job_headroom);
        }
        Some(HostMemory {
            available: status.ullAvailPhys,
            hard_limit: Some(hard_limit),
        })
    }

    /// A job's process or job memory limit minus this process's private bytes.
    /// For a job limit this assumes the process is alone in the job.
    #[allow(unsafe_code)]
    fn job_headroom() -> Option<u64> {
        // SAFETY: plain data; all zeros is a valid value.
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a null handle queries the calling process's job; the buffer
        // and its length match.
        let ok = unsafe {
            QueryInformationJobObject(
                std::ptr::null_mut(),
                JobObjectExtendedLimitInformation,
                (&raw mut limits).cast(),
                u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>()).ok()?,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return None;
        }
        let flags = limits.BasicLimitInformation.LimitFlags;
        let limit = if flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY != 0 {
            limits.ProcessMemoryLimit
        } else if flags & JOB_OBJECT_LIMIT_JOB_MEMORY != 0 {
            limits.JobMemoryLimit
        } else {
            return None;
        };
        // SAFETY: plain data; all zeros is a valid value.
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        // SAFETY: the counters buffer and its length match.
        let ok = unsafe {
            K32GetProcessMemoryInfo(
                GetCurrentProcess(),
                (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
                u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS_EX>()).ok()?,
            )
        };
        let used = if ok == 0 { 0 } else { counters.PrivateUsage };
        u64::try_from(limit.saturating_sub(used)).ok()
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod platform {
    pub(super) fn host_memory() -> Option<super::HostMemory> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{Builder, ContextShape, DeviceMemory, HostMemory, Pools, Storage, Suggestion, plan};
    use crate::TaskKind;

    const GIB: u64 = 1 << 30;

    fn shape(rows: u64, features: u64) -> ContextShape {
        ContextShape {
            task: TaskKind::Classifier,
            estimators: 8,
            rows,
            features,
        }
    }

    fn host(available: u64) -> Pools {
        Pools {
            host: Some(HostMemory {
                available,
                hard_limit: None,
            }),
            device: None,
        }
    }

    #[test]
    fn context_bytes_match_the_reference_table() {
        // docs/memory-and-io.md: H=100, B=8, 10,000 rows is 3.97 GiB in FP32.
        // In FP16 only the ICL caches shrink (1.83 GiB); the column caches stay
        // FP32 (0.31 GiB), unlike that table's uniform-storage FP16 column.
        for (storage, expected) in [(Storage::FP32, 3.97), (Storage::FP16, 2.14)] {
            let bytes = plan(shape(10_000, 100), storage, Builder::Cpu, host(64 * GIB), &[])
                .expect("plan")
                .context_bytes;
            #[allow(clippy::cast_precision_loss)]
            let gib = bytes as f64 / GIB as f64;
            assert!((gib - expected).abs() < 0.01, "{gib} GiB");
        }
    }

    #[test]
    fn wide_gpu_builds_run_one_estimator_and_narrow_ones_group() {
        let wide = plan(
            shape(10_000, 100),
            Storage::FP16,
            Builder::Gpu,
            host(28 * GIB),
            &[],
        )
        .expect("plan");
        assert_eq!(wide.build_group, 1);
        let narrow = plan(
            shape(10_000, 20),
            Storage::FP16,
            Builder::Gpu,
            host(28 * GIB),
            &[],
        )
        .expect("plan");
        assert_eq!(narrow.build_group, 4);
        // Less memory shrinks the group: half of 4 GiB holds one narrow estimator.
        let tight = plan(shape(10_000, 20), Storage::FP16, Builder::Gpu, host(4 * GIB), &[]).expect("plan");
        assert_eq!(tight.build_group, 1);
    }

    #[test]
    fn gpu_build_groups_follow_the_budget_override_and_the_device_limit() {
        use super::gpu_build_group;
        const MB: u64 = 1 << 20;
        // IRONICL_GPU_BUILD_BUDGET is an exact cap: one 17 MB estimator in
        // 20 MB, and a CPU build when none fits.
        assert_eq!(gpu_build_group(4, 17 * MB, None, None, Some(20 * MB)), Some(1));
        assert_eq!(gpu_build_group(4, 17 * MB, None, None, Some(10 * MB)), None);
        // Without it, one estimator larger than the device builds on the CPU,
        // and one larger than the memory share still builds alone on the GPU.
        assert_eq!(
            gpu_build_group(8, 5 * GIB, Some(28 * GIB), Some(4 * GIB), None),
            None
        );
        assert_eq!(
            gpu_build_group(8, 5 * GIB, Some(4 * GIB), Some(21 * GIB), None),
            Some(1)
        );
        // Groups divide the estimator count: five of 1 GiB under 6 GiB is 5.
        assert_eq!(
            gpu_build_group(5, GIB, Some(28 * GIB), Some(21 * GIB), None),
            Some(5)
        );
        assert_eq!(
            gpu_build_group(6, GIB, Some(28 * GIB), Some(21 * GIB), None),
            Some(6)
        );
        assert_eq!(
            gpu_build_group(7, GIB, Some(28 * GIB), Some(21 * GIB), None),
            Some(1)
        );
    }

    #[test]
    fn a_plan_falls_back_to_the_cpu_when_one_estimator_exceeds_the_device() {
        let pools = Pools {
            host: Some(HostMemory {
                available: 28 * GIB,
                hard_limit: None,
            }),
            device: Some(DeviceMemory::Shared { limit: 2 * GIB }),
        };
        let plan = plan(shape(10_000, 100), Storage::FP16, Builder::Gpu, pools, &[]).expect("plan");
        assert_eq!(plan.builder, Builder::Cpu);
        assert!(plan.to_string().contains("on the CPU"), "{plan}");
    }

    #[test]
    fn cpu_build_groups_shrink_with_available_memory() {
        use super::cpu_build_group;
        // A 1 GiB estimate counts as 0.5 GiB: 4 GiB available allows 4.
        assert_eq!(cpu_build_group(8, GIB, Some(4 * GIB)), 4);
        assert_eq!(cpu_build_group(8, GIB, Some(GIB)), 1);
        assert_eq!(cpu_build_group(8, GIB, Some(0)), 1);
    }

    #[test]
    fn a_shared_device_is_capped_by_its_limit_and_a_separate_one_is_not() {
        let mut pools = host(28 * GIB);
        pools.device = Some(DeviceMemory::Shared { limit: 21 * GIB });
        assert_eq!(pools.available(), Some(21 * GIB));
        pools.device = Some(DeviceMemory::Separate { available: 80 * GIB });
        assert_eq!(pools.available(), Some(80 * GIB));
        pools.host = None;
        pools.device = Some(DeviceMemory::Shared { limit: 21 * GIB });
        assert_eq!(pools.available(), None);
    }

    #[test]
    fn hard_limits_cap_available_memory() {
        let host = HostMemory {
            available: 30 * GIB,
            hard_limit: Some(8 * GIB),
        };
        assert_eq!(host.usable(), 8 * GIB);
    }

    #[test]
    fn overflow_suggests_rows_estimators_and_storage_that_fit() {
        let alternatives = [("optimize=\"fast\"".to_string(), Storage::FP16)];
        let large = shape(50_000, 100);
        // 18 GiB holds five FP32 estimators or all eight in FP16, not eight in FP32.
        let plan = plan(large, Storage::FP32, Builder::Cpu, host(18 * GIB), &alternatives).expect("plan");
        assert_eq!(plan.fits(), Some(false));
        let rows = plan
            .suggestions
            .iter()
            .find_map(|s| match s {
                Suggestion::Rows(rows) => Some(*rows),
                _ => None,
            })
            .expect("a row suggestion");
        let at = |rows| {
            super::plan(
                ContextShape { rows, ..large },
                Storage::FP32,
                Builder::Cpu,
                host(18 * GIB),
                &[],
            )
            .expect("plan")
            .fits()
        };
        assert_eq!(at(rows), Some(true));
        assert_eq!(at(rows + 1), Some(false));
        assert!(
            plan.suggestions
                .iter()
                .any(|s| matches!(s, Suggestion::Estimators(_)))
        );
        assert!(
            plan.suggestions
                .iter()
                .any(|s| matches!(s, Suggestion::Storage { .. }))
        );
        let report = plan.to_string();
        assert!(report.contains("Any one of these fits"), "{report}");
        assert!(report.contains("optimize=\"fast\""), "{report}");
    }

    #[test]
    fn unknown_memory_has_no_verdict_or_suggestions() {
        let unknown = Pools {
            host: None,
            device: None,
        };
        let plan = plan(shape(50_000, 100), Storage::FP32, Builder::Cpu, unknown, &[]).expect("plan");
        assert_eq!(plan.fits(), None);
        assert!(plan.suggestions.is_empty());
        assert!(plan.to_string().contains("unknown"));
    }

    #[test]
    fn empty_shapes_are_rejected() {
        assert!(plan(shape(0, 100), Storage::FP32, Builder::Cpu, host(GIB), &[]).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn this_host_reports_its_memory() {
        let host = super::host_memory().expect("a memory probe for this platform");
        assert!(host.available > 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_parsers_read_meminfo_and_cgroups() {
        use super::platform::{cgroup_v2_path, limit_bytes, mem_available};
        assert_eq!(
            mem_available("MemTotal: 100 kB\nMemAvailable:    2048 kB\n"),
            Some(2048 * 1024)
        );
        assert_eq!(cgroup_v2_path("0::/user.slice/app\n"), Some("/user.slice/app"));
        assert_eq!(limit_bytes("max\n"), None);
        assert_eq!(limit_bytes("8589934592\n"), Some(8 << 30));
        assert_eq!(limit_bytes("9223372036854771712"), None);
    }
}

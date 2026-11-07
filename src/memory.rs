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
    /// Physical memory installed.
    pub total: u64,
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

/// The environment variable that limits the host's RAM, which shared devices use too.
pub const MEMORY_LIMIT_ENV: &str = "IRONICL_MEMORY_LIMIT";
/// The environment variable that limits a separate device's own memory.
pub const DEVICE_MEMORY_LIMIT_ENV: &str = "IRONICL_DEVICE_MEMORY_LIMIT";

/// A user's cap on one memory pool, and where it was set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limit {
    pub bytes: u64,
    /// The argument or environment variable that set it.
    pub source: &'static str,
}

/// User caps on the host's RAM (which a shared device uses too) and on a
/// separate device's own memory. Limits only lower what is detected; they are
/// a planning budget, not enforced by the operating system.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryLimits {
    pub host: Option<Limit>,
    pub device: Option<Limit>,
}

impl MemoryLimits {
    /// Limits from explicit values, such as the Python `memory_limit` and
    /// `device_memory_limit` arguments, falling back to `IRONICL_MEMORY_LIMIT`
    /// and `IRONICL_DEVICE_MEMORY_LIMIT`. Values are parsed by [`parse_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] naming the argument or variable when a
    /// value is not a positive size.
    pub fn resolve(host: Option<&str>, device: Option<&str>) -> Result<Self> {
        let limit = |value: Option<&str>, argument: &'static str, variable: &'static str| {
            let (text, source) = match value {
                Some(text) => (text.to_owned(), argument),
                None => match std::env::var(variable) {
                    Ok(text) if !text.trim().is_empty() => (text, variable),
                    _ => return Ok(None),
                },
            };
            parse_bytes(&text)
                .map(|bytes| Some(Limit { bytes, source }))
                .map_err(|error| Error::InvalidArgument(format!("{source}={text:?}: {error}")))
        };
        Ok(Self {
            host: limit(host, "memory_limit", MEMORY_LIMIT_ENV)?,
            device: limit(device, "device_memory_limit", DEVICE_MEMORY_LIMIT_ENV)?,
        })
    }
}

/// Parse a byte size: a number of bytes, or a number with a binary suffix `K`,
/// `M`, `G`, or `T`, optionally followed by `B` or `iB`, in any case: `512M`,
/// `16G`, `1.5GiB`, `2t`. `16G` and `16GiB` are both 16 x 2^30 bytes.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for anything else, zero, or a size that
/// overflows `u64`.
pub fn parse_bytes(text: &str) -> Result<u64> {
    let invalid = || {
        Error::InvalidArgument(
            "expected a positive size in bytes, or with a K, M, G, or T suffix such as 512M or 16G".into(),
        )
    };
    let lower = text.trim().to_ascii_lowercase();
    let number = lower
        .strip_suffix("ib")
        .or_else(|| lower.strip_suffix('b'))
        .unwrap_or(&lower);
    let shift = number
        .chars()
        .last()
        .and_then(|unit| "kmgt".find(unit))
        .map_or(0, |index| 10 * (index + 1));
    let digits = if shift == 0 {
        number
    } else {
        &number[..number.len() - 1]
    };
    // "16iB" names no unit; digits and one decimal point only, so no signs,
    // exponents, or "inf".
    if (shift == 0 && lower.ends_with("ib"))
        || digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit() || byte == b'.')
    {
        return Err(invalid());
    }
    let scale = 1_u64 << shift;
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let bytes = if digits.contains('.') {
        digits
            .parse::<f64>()
            .ok()
            .map(|value| value * scale as f64)
            .filter(|value| *value < u64::MAX as f64)
            .map(|value| value as u64)
    } else {
        digits
            .parse::<u64>()
            .ok()
            .and_then(|value| value.checked_mul(scale))
    };
    bytes.filter(|bytes| *bytes > 0).ok_or_else(invalid)
}

/// The memory pools a plan draws on, with the user's limits. Unknown pools
/// make the verdict unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pools {
    pub host: Option<HostMemory>,
    pub device: Option<DeviceMemory>,
    pub limits: MemoryLimits,
}

impl Pools {
    /// The host's memory from the operating system, with no device and no limits.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            host: host_memory(),
            ..Self::default()
        }
    }

    /// These pools with `limits` applied.
    #[must_use]
    pub fn with_limits(self, limits: MemoryLimits) -> Self {
        Self { limits, ..self }
    }

    /// Bytes of host RAM this process can use, capped by the memory limit, or
    /// `None` if unknown. A limit alone is enough to know it.
    #[must_use]
    pub fn host_available(self) -> Option<u64> {
        let limit = self.limits.host.map(|limit| limit.bytes);
        match (self.host.map(HostMemory::usable), limit) {
            (Some(host), Some(limit)) => Some(host.min(limit)),
            (host, limit) => host.or(limit),
        }
    }

    /// Bytes available where the context is built, or `None` if unknown: the
    /// host's for a shared device, capped by its limit; a separate device's own,
    /// capped by the device memory limit.
    #[must_use]
    pub fn available(self) -> Option<u64> {
        match self.device {
            None => self.host_available(),
            Some(DeviceMemory::Shared { limit }) => self.host_available().map(|host| host.min(limit)),
            Some(DeviceMemory::Separate { available }) => Some(
                self.limits
                    .device
                    .map_or(available, |limit| available.min(limit.bytes)),
            ),
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

/// Reserve for Python, the allocator, preprocessing, and query buffers.
const HEADROOM: u64 = 512 << 20;
/// A build may use at most this share of memory it shares with everything else
/// on the machine: the host's RAM, for the CPU and for shared devices.
const BUILD_SHARE_DIVISOR: u64 = 2;
/// A separate device's memory is the build's own, less a reserve for the
/// allocator and fragmentation: 1/20 of it, and at least this much.
const SEPARATE_DEVICE_RESERVE: u64 = 256 << 20;
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
    /// The pools and limits the plan was made for.
    pub pools: Pools,
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

/// What a plan means for the fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Fits,
    /// The peak exceeds available memory (or the memory limit), so the system
    /// would page; it can still complete.
    Overflow,
    /// The peak exceeds a hard limit (a cgroup or job limit, or the Windows
    /// commit limit), or the context alone exceeds physical memory: the fit
    /// would fail or page without end.
    CannotComplete,
    /// Available memory is unknown.
    Unknown,
}

impl Plan {
    /// Whether the peak fits in the available memory, or `None` if unknown.
    #[must_use]
    pub fn fits(&self) -> Option<bool> {
        self.available_bytes.map(|available| self.peak_bytes <= available)
    }

    /// Whether the fit fits, overflows, or cannot complete on this machine.
    #[must_use]
    pub fn verdict(&self) -> Verdict {
        let host = self.pools.host;
        let persistent = self.weight_bytes.saturating_add(self.context_bytes);
        if host.is_some_and(|host| {
            host.hard_limit.is_some_and(|limit| self.peak_bytes > limit) || persistent > host.total
        }) {
            return Verdict::CannotComplete;
        }
        match self.fits() {
            Some(true) => Verdict::Fits,
            Some(false) => Verdict::Overflow,
            None => Verdict::Unknown,
        }
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
        let available = plan.available_bytes.unwrap_or(0);
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
    let gpu_group = (builder == Builder::Gpu)
        .then(|| {
            let per_estimator = build_estimator_bytes(shape, Builder::Gpu)?;
            Ok::<_, Error>(gpu_build_group(shape.estimators, per_estimator, pools))
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
            pools.host_available(),
        )
    });
    let available = match builder {
        Builder::Gpu => pools.available(),
        Builder::Cpu => pools.host_available(),
    };
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
        pools,
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
/// The group is the largest that divides the estimator count (every run has the
/// same graph shape) and stays under the build's share of the device's memory:
/// on a shared device, half the host's available RAM, capped by the device's
/// limit; on a separate device, all of its available memory less a reserve. The
/// CPU builds only when one estimator exceeds the device.
pub(crate) fn gpu_build_group(estimators: u64, per_estimator: u64, pools: Pools) -> Option<u64> {
    let per_estimator = per_estimator.max(1);
    let host_share = pools.host_available().map(|host| host / BUILD_SHARE_DIVISOR);
    let cap = match pools.device {
        Some(DeviceMemory::Shared { limit }) => {
            if per_estimator > limit {
                return None;
            }
            host_share.map_or(limit / BUILD_SHARE_DIVISOR, |share| share.min(limit))
        }
        Some(DeviceMemory::Separate { .. }) => {
            let available = pools.available().unwrap_or(0);
            let usable = available.saturating_sub(SEPARATE_DEVICE_RESERVE.max(available / 20));
            if per_estimator > usable {
                return None;
            }
            usable
        }
        None => host_share.unwrap_or(0),
    };
    Some(
        (1..=estimators)
            .rev()
            .find(|&group| estimators.is_multiple_of(group) && group.saturating_mul(per_estimator) <= cap)
            .unwrap_or(1),
    )
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
        if let Some(limit) = self.pools.limits.host {
            write!(
                f,
                "\nMemory limit: {} (set by {}).",
                gib(limit.bytes),
                limit.source
            )?;
        }
        if let Some(limit) = self.pools.limits.device {
            if matches!(self.pools.device, Some(DeviceMemory::Separate { .. })) {
                write!(
                    f,
                    "\nDevice memory limit: {} (set by {}).",
                    gib(limit.bytes),
                    limit.source
                )?;
            } else {
                write!(
                    f,
                    "\nDevice memory limit ({}): not applicable; there is no separate device memory, \
                     so memory_limit applies.",
                    limit.source
                )?;
            }
        }
        if self.fits() == Some(false) && self.suggestions.is_empty() {
            write!(
                f,
                "\nNo fewer rows, fewer estimators, or smaller storage fits this machine."
            )?;
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
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        Some(HostMemory {
            total: meminfo_bytes(&meminfo, "MemTotal:")?,
            available: meminfo_bytes(&meminfo, "MemAvailable:")?,
            hard_limit: cgroup_headroom(),
        })
    }

    /// A `/proc/meminfo` field such as `MemAvailable:`, in bytes.
    pub(super) fn meminfo_bytes(meminfo: &str, field: &str) -> Option<u64> {
        let kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix(field))?
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
            total,
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
            total: status.ullTotalPhys,
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
    use super::{
        Builder, ContextShape, DeviceMemory, HostMemory, Limit, MemoryLimits, Pools, Storage, Suggestion,
        Verdict, cpu_build_group, gpu_build_group, parse_bytes, plan,
    };
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
                total: 64 * GIB,
                available,
                hard_limit: None,
            }),
            ..Pools::default()
        }
    }

    fn shared(available: u64, limit: u64) -> Pools {
        Pools {
            device: Some(DeviceMemory::Shared { limit }),
            ..host(available)
        }
    }

    fn separate(host_available: u64, device_available: u64) -> Pools {
        Pools {
            device: Some(DeviceMemory::Separate {
                available: device_available,
            }),
            ..host(host_available)
        }
    }

    fn limits(host: Option<u64>, device: Option<u64>) -> MemoryLimits {
        let limit = |bytes: Option<u64>| {
            bytes.map(|bytes| Limit {
                bytes,
                source: "test",
            })
        };
        MemoryLimits {
            host: limit(host),
            device: limit(device),
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
    fn shared_device_builds_use_half_the_hosts_memory() {
        let group = |features, pools| {
            plan(shape(10_000, features), Storage::FP16, Builder::Gpu, pools, &[])
                .expect("plan")
                .build_group
        };
        // 28 GiB available: a 14 GiB share holds two wide (4.3 GiB) estimators
        // and all eight narrow (1.2 GiB) ones.
        assert_eq!(group(100, shared(28 * GIB, 21 * GIB)), 2);
        assert_eq!(group(20, shared(28 * GIB, 21 * GIB)), 8);
        // The device's limit caps the share, and less memory shrinks it.
        assert_eq!(group(20, shared(28 * GIB, 5 * GIB)), 4);
        assert_eq!(group(20, shared(4 * GIB, 21 * GIB)), 1);
    }

    #[test]
    fn gpu_build_groups_divide_the_estimators_and_fall_back_to_the_cpu() {
        // One estimator larger than a shared device builds on the CPU; one
        // larger than the memory share still builds alone on the GPU.
        assert_eq!(gpu_build_group(8, 5 * GIB, shared(28 * GIB, 4 * GIB)), None);
        assert_eq!(gpu_build_group(8, 5 * GIB, shared(4 * GIB, 21 * GIB)), Some(1));
        // Groups divide the estimator count: 14 GiB holds five or six of 1 GiB,
        // but seven only one at a time.
        assert_eq!(gpu_build_group(5, GIB, shared(28 * GIB, 21 * GIB)), Some(5));
        assert_eq!(gpu_build_group(6, GIB, shared(28 * GIB, 21 * GIB)), Some(6));
        assert_eq!(gpu_build_group(7, GIB, shared(12 * GIB, 21 * GIB)), Some(1));
        // Unknown host memory: half the device's limit.
        let unknown_host = Pools {
            host: None,
            ..shared(0, 8 * GIB)
        };
        assert_eq!(gpu_build_group(8, GIB, unknown_host), Some(4));
        // Nothing known: one at a time.
        assert_eq!(gpu_build_group(8, GIB, Pools::default()), Some(1));
    }

    #[test]
    fn separate_devices_use_their_own_memory_less_a_reserve() {
        let wide = 4_400 * (1 << 20);
        // 24 GiB less 1.2 GiB holds four 4.3 GiB estimators, not eight; the
        // host's memory does not matter.
        assert_eq!(gpu_build_group(8, wide, separate(2 * GIB, 24 * GIB)), Some(4));
        // The device memory limit caps it: 10 GiB less 512 MiB holds two.
        let limited = separate(2 * GIB, 24 * GIB).with_limits(limits(None, Some(10 * GIB)));
        assert_eq!(gpu_build_group(8, wide, limited), Some(2));
        // The reserve is at least 256 MiB, and one estimator over what is left
        // builds on the CPU.
        assert_eq!(
            gpu_build_group(8, GIB, separate(64 * GIB, GIB + 128 * (1 << 20))),
            None
        );
    }

    #[test]
    fn a_plan_falls_back_to_the_cpu_when_one_estimator_exceeds_the_device() {
        let plan = plan(
            shape(10_000, 100),
            Storage::FP16,
            Builder::Gpu,
            shared(28 * GIB, 2 * GIB),
            &[],
        )
        .expect("plan");
        assert_eq!(plan.builder, Builder::Cpu);
        assert!(plan.to_string().contains("on the CPU"), "{plan}");
    }

    #[test]
    fn cpu_build_groups_shrink_with_available_memory() {
        // A 1 GiB estimate counts as 0.5 GiB: 4 GiB available allows 4.
        assert_eq!(cpu_build_group(8, GIB, Some(4 * GIB)), 4);
        assert_eq!(cpu_build_group(8, GIB, Some(GIB)), 1);
        assert_eq!(cpu_build_group(8, GIB, Some(0)), 1);
    }

    #[test]
    fn available_memory_follows_the_pool_and_its_limit() {
        assert_eq!(shared(28 * GIB, 21 * GIB).available(), Some(21 * GIB));
        assert_eq!(separate(28 * GIB, 80 * GIB).available(), Some(80 * GIB));
        let unknown_host = Pools {
            host: None,
            ..shared(0, 21 * GIB)
        };
        assert_eq!(unknown_host.available(), None);
        // The memory limit caps the host and shared devices, and only lowers.
        let capped = shared(28 * GIB, 21 * GIB).with_limits(limits(Some(8 * GIB), None));
        assert_eq!(
            (capped.host_available(), capped.available()),
            (Some(8 * GIB), Some(8 * GIB))
        );
        assert_eq!(
            host(28 * GIB)
                .with_limits(limits(Some(64 * GIB), None))
                .host_available(),
            Some(28 * GIB)
        );
        // A limit alone is enough to know the host's memory.
        let limit_only = Pools::default().with_limits(limits(Some(8 * GIB), None));
        assert_eq!(limit_only.host_available(), Some(8 * GIB));
        // The device memory limit applies only to a separate device.
        let device_limit = limits(None, Some(10 * GIB));
        assert_eq!(
            separate(28 * GIB, 80 * GIB).with_limits(device_limit).available(),
            Some(10 * GIB)
        );
        assert_eq!(
            shared(28 * GIB, 21 * GIB).with_limits(device_limit).available(),
            Some(21 * GIB)
        );
        assert_eq!(
            host(28 * GIB).with_limits(device_limit).available(),
            Some(28 * GIB)
        );
    }

    #[test]
    fn hard_limits_cap_available_memory() {
        let host = HostMemory {
            total: 32 * GIB,
            available: 30 * GIB,
            hard_limit: Some(8 * GIB),
        };
        assert_eq!(host.usable(), 8 * GIB);
    }

    #[test]
    fn sizes_parse_with_binary_suffixes() {
        for (text, bytes) in [
            ("1048576", 1 << 20),
            ("512M", 512 << 20),
            ("16G", 16 * GIB),
            ("16g", 16 * GIB),
            ("16GB", 16 * GIB),
            ("16GiB", 16 * GIB),
            (" 1.5G ", 3 * GIB / 2),
            ("2T", 2 << 40),
            ("64K", 64 << 10),
            ("100b", 100),
        ] {
            assert_eq!(parse_bytes(text).expect(text), bytes, "{text}");
        }
        for text in [
            "",
            "0",
            "0G",
            "-1G",
            "1e9",
            "inf",
            "G",
            "16iB",
            "16X",
            "1.2.3G",
            "+5G",
            "99999999999T",
        ] {
            assert!(parse_bytes(text).is_err(), "{text:?} should be rejected");
        }
    }

    #[test]
    fn explicit_limits_win_and_errors_name_their_source() {
        let resolved = MemoryLimits::resolve(Some("16G"), Some("8G")).expect("resolve");
        assert_eq!(
            resolved.host,
            Some(Limit {
                bytes: 16 * GIB,
                source: "memory_limit"
            })
        );
        assert_eq!(resolved.device.map(|limit| limit.bytes), Some(8 * GIB));
        let error = MemoryLimits::resolve(Some("lots"), None).expect_err("invalid");
        assert!(error.to_string().contains("memory_limit=\"lots\""), "{error}");
    }

    #[test]
    fn verdicts_separate_overflow_from_cannot_complete() {
        let wide = shape(10_000, 100);
        let verdict = |pools| {
            plan(wide, Storage::FP32, Builder::Cpu, pools, &[])
                .expect("plan")
                .verdict()
        };
        assert_eq!(verdict(host(28 * GIB)), Verdict::Fits);
        // The peak exceeds what is available, or the memory limit: overflow.
        assert_eq!(verdict(host(4 * GIB)), Verdict::Overflow);
        assert_eq!(
            verdict(host(28 * GIB).with_limits(limits(Some(2 * GIB), None))),
            Verdict::Overflow
        );
        // A hard limit, or a context larger than physical memory: cannot complete.
        let mut capped = host(28 * GIB);
        capped.host.as_mut().expect("host").hard_limit = Some(2 * GIB);
        assert_eq!(verdict(capped), Verdict::CannotComplete);
        let mut small = host(2 * GIB);
        small.host.as_mut().expect("host").total = 3 * GIB;
        assert_eq!(verdict(small), Verdict::CannotComplete);
        assert_eq!(verdict(Pools::default()), Verdict::Unknown);
    }

    #[test]
    fn reports_name_the_limits_that_applied() {
        let pools = shared(28 * GIB, 21 * GIB).with_limits(MemoryLimits {
            host: Some(Limit {
                bytes: 16 * GIB,
                source: "IRONICL_MEMORY_LIMIT",
            }),
            device: Some(Limit {
                bytes: 8 * GIB,
                source: "device_memory_limit",
            }),
        });
        let report = plan(shape(1_000, 10), Storage::FP16, Builder::Gpu, pools, &[])
            .expect("plan")
            .to_string();
        assert!(
            report.contains("Memory limit: 16.0 GiB (set by IRONICL_MEMORY_LIMIT)"),
            "{report}"
        );
        assert!(
            report.contains("device_memory_limit): not applicable"),
            "{report}"
        );
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
        let plan = plan(
            shape(50_000, 100),
            Storage::FP32,
            Builder::Cpu,
            Pools::default(),
            &[],
        )
        .expect("plan");
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
        use super::platform::{cgroup_v2_path, limit_bytes, meminfo_bytes};
        let meminfo = "MemTotal: 100 kB\nMemAvailable:    2048 kB\n";
        assert_eq!(meminfo_bytes(meminfo, "MemAvailable:"), Some(2048 * 1024));
        assert_eq!(meminfo_bytes(meminfo, "MemTotal:"), Some(100 * 1024));
        assert_eq!(cgroup_v2_path("0::/user.slice/app\n"), Some("/user.slice/app"));
        assert_eq!(limit_bytes("max\n"), None);
        assert_eq!(limit_bytes("8589934592\n"), Some(8 << 30));
        assert_eq!(limit_bytes("9223372036854771712"), None);
    }
}

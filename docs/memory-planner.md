# Memory Planner

Ironicl decides how much memory a fitted context needs, how much the machine
can give, and what to do when it does not fit. Before the planner, these were
separate fixed rules: the GPU build used Metal's whole working set, the CPU
build a fixed activation limit, and prediction a fixed query chunk. On an
M1 Max (32 GB) that let a 100-feature context at 10,000 rows peak at 21.4 GiB
while building, for a 2 GiB context.

## Design

### Pools, by sharing rather than vendor

| Pool | Shares the host's RAM | Available memory |
|---|---|---|
| Host | (is the host) | Per platform, below |
| Shared device (an Apple GPU) | Yes | The host's figure, capped by the device's limit (Metal's working set) |
| Separate device (a discrete GPU, future) | No | The device's own query |

A plan asks each pool only how much is available and whether it shares the
host's memory. Any device that fits those two questions fits the planner.

### Host memory per platform

| Platform | Available | Hard limit (fails instead of paging) |
|---|---|---|
| macOS | Total minus app memory (anonymous pages that are not purgeable), wired, and compressed memory: the complement of Activity Monitor's "Memory Used" | None |
| Linux | `MemAvailable` from `/proc/meminfo` | The cgroup limit minus usage (v2, or v1) |
| Windows | `ullAvailPhys` from `GlobalMemoryStatusEx` | The remaining commit charge (`ullAvailPageFile`), capped by any job memory limit |
| Other | Unknown: no verdict, today's behavior | None |

Windows does not overcommit: past the commit limit, allocations fail instead of
paging. Read-only memory-mapped files do not count against it, which makes a
file-backed context the only way to serve one larger than the commit limit.
The Windows probe compiles and passes Clippy but has not run on Windows yet.

### Demand

- **Persistent:** FP32 weights, FP32 column caches, and the ICL caches at
  their storage type (FP32, FP32 keys with BF16 values, or FP16).
- **Build, per estimator:** the GPU build graph's working set, or half the CPU
  build's own estimate (which is 2-3x its measured peak).
- **Headroom:** 512 MiB for Python, the allocator, and query buffers.

The peak is weights + context + one build group + headroom.

### Build groups

The GPU builds the largest group of estimators that divides the estimator
count and stays under the smallest of 6 GiB, half the available memory, and the
device's limit. Measured on an M1 Max at 10,000 rows:

| Features | Estimators per group | Fit | Peak |
|---|---|---|---|
| 100 | 1 / 2 / 4 | 11.5 / 11.2 / 11.3 s | 7.0 / 12.0 / 21.4 GiB |
| 20 | 1 / 2 / 8 | 6.9 / 6.0 / 5.3 s | 4.1 / 5.9 / 9.7 GiB |

One wide estimator already keeps the GPU busy, so wide tables build one at a
time; narrow ones group.

### Plan

`memory::plan` returns the context size, build group, peak, available memory,
whether it fits, and, when it does not, the changes that would: at most this
many training rows, fewer estimators, or a smaller storage type the caller
offers. Every suggestion is checked to fit. `ironicl plan` prints the report.

## Python API (PR C)

Two trailing Ironicl-only parameters on both estimators, after `cache_dir`:

- `memory_limit=None`: a byte cap, or automatic. Default from
  `IRONICL_MEMORY_LIMIT`.
- `memory_overflow=None`: `"warn"`, `"error"`, or `"ignore"`, default from
  `IRONICL_MEMORY_OVERFLOW`, then `"warn"`.

`fit` plans after preprocessing, when the real shape is known, and before any
build work. `"warn"` emits `IroniclMemoryWarning` (a `UserWarning`, so it is
shown by default) and continues; `"error"` raises `IroniclMemoryError` (a
`MemoryError`) and leaves the estimator unfitted. The plan is kept as
`memory_plan_`, and `ironicl.plan_memory(...)` plans without data.

## Phases

| PR | Scope |
|---|---|
| A | Pools, platform probes, demand, plan, and the `ironicl plan` report. No behavior change |
| B | The GPU and CPU builds use the plan's group sizes; a peak-memory fixture test at 10,000 rows and 100 features |
| C | The Python API above, with warnings and errors |
| D | Prediction buffers bounded by the plan, and large query buckets released after use |

Later, as separate plans: building straight into the final storage, out-of-core
prediction from a memory-mapped file, per-layer normalized states for streamed
contexts, checkpointed layers, and Windows CI.

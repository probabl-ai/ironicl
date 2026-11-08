# Memory Planner

Ironicl decides how much memory a fitted context needs, how much the machine
can give, and what to do when it does not fit. Before the planner, these were
separate fixed rules: the GPU build used Metal's whole working set, the CPU
build a fixed activation limit, and prediction a fixed query chunk. On an
M1 Max (32 GB) that let a 100-feature context at 10,000 rows peak at 21.4 GiB
while building, for a 2 GiB context.

## Design

### Pools, by sharing rather than vendor

| Pool | Shares the host's RAM | Available memory | A build may use | Limited by |
|---|---|---|---|---|
| Host (CPU build) | (is the host) | Per platform, below | Half | `memory_limit` |
| Shared device (an Apple GPU) | Yes | The host's figure, capped by the device's limit (Metal's working set) | Half the host's, up to the device's limit | `memory_limit` |
| Separate device (a discrete GPU, future) | No | The device's own query | All of it, less 5% (at least 256 MiB) | `device_memory_limit` |

A plan asks each pool only how much is available and whether it shares the
host's memory. Any device that fits those two questions fits the planner.
Memory shared with the rest of the machine is shared with the operating system
and other programs, so a build takes half of it; a separate device's memory is
the build's own.

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

The GPU builds the largest group of estimators that divides the estimator count
and stays under the pool's share above. The CPU builds equal groups under half
the host's available memory. Measured on an M1 Max (32 GB) with
`optimize="auto"`, about 19 GiB free:

| Rows x features | Limit | Estimators per group | Fit | Peak (planned) |
|---|---|---|---|---|
| 10,000 x 100 | None | 2 | 11.0 s | 12.0 GiB (11.3) |
| 10,000 x 100 | 8 GiB | 1 | 11.7 s | 7.1 GiB (7.0) |
| 10,000 x 20 | None | 8 | 5.3 s | 9.7 GiB (12.3) |
| 10,000 x 20 | 8 GiB | 2 | 6.1 s | 5.9 GiB (5.0) |
| 2,000 x 20 | Either | 8 | 1.2 s | 3.1 GiB (3.0) |

Narrow tables gain most from grouping; one wide estimator already keeps the GPU
busy. The planned peak is within about 1 GiB of the measured one.

### Limits

| Python | Environment | Limits |
|---|---|---|
| `memory_limit` | `IRONICL_MEMORY_LIMIT` | The host's RAM, and any device that shares it |
| `device_memory_limit` | `IRONICL_DEVICE_MEMORY_LIMIT` | A separate device's own memory; reported as not applicable where memory is shared |

A value is a number of bytes or a size with a binary suffix: `512M`, `16G`,
`1.5GiB`, `2T` (`K`, `M`, `G`, `T`, any case, optionally followed by `B` or
`iB`). The argument wins over the environment, which wins over detection.
Limits only lower the memory detected, and they plan memory rather than enforce
it: a fit past its limit warns or raises as `memory_overflow` says. Changing a
limit changes build groups, not results, so it is not part of the `cache_dir`
key.

### Plan

`memory::plan` returns the context size, build group, peak, available memory,
the limits that applied, and a verdict:

- **Fits.**
- **Overflow:** the peak exceeds available memory or the memory limit, but the
  system can page.
- **Cannot complete:** the peak exceeds a hard limit (a cgroup or job limit, or
  the Windows commit limit), or the context alone exceeds physical memory.
- **Unknown:** no probe for this platform and no limit.

When it does not fit, the plan lists the changes that would: at most this many
training rows, fewer estimators, or a faster `optimize` level with smaller
storage. Every suggestion is checked to fit. `ironicl::plan_context` plans with
the storage, builder, and pools the build itself would use; `ironicl plan`
prints the report.

## Python API

Three trailing Ironicl-only parameters on both estimators, after `cache_dir`:

- `memory_limit=None` and `device_memory_limit=None`: an int number of bytes or
  a size string, as above.
- `memory_overflow=None`: `"warn"`, `"error"`, or `"ignore"`, default from
  `IRONICL_MEMORY_OVERFLOW`, then `"warn"`.

`fit` plans after preprocessing, when the real shape is known, and before
loading the model or building. On overflow, `"warn"` emits
`IroniclMemoryWarning` (a `UserWarning`, so it is shown by default) and
continues; `"error"` raises `IroniclMemoryError` (a `MemoryError`) and leaves
the estimator unfitted; `"ignore"` continues silently. A fit that cannot
complete raises under `"warn"` too. The plan is kept as `memory_plan_`, and
`ironicl.plan_memory(n_samples, n_features, ...)` plans without data:

```python
>>> print(ironicl.plan_memory(30_000, 100, memory_limit="8G"))
Context: 11.3 GiB (8 estimators x 30000 rows x 100 features, ICL caches in FP32).
Peak while building on the GPU: 24.7 GiB, including 12.8 GiB for 1 estimator(s) at a time; available: about 8.0 GiB.
Memory limit: 8.0 GiB (set by memory_limit).
Any one of these fits:
  - at most 8938 training rows (subsample X, y before fit)
```

Loading a saved or pickled estimator plans again on the loading machine and
follows `memory_overflow` the same way. Its peak is the weights, the context,
and headroom, without a build, unless the save left out the context and the
load rebuilds it.

## Prediction

`MPSGraph` keeps the intermediate buffers of every query shape a predict graph
has run until the graph is freed: about 2.3 MB per padded row at 8 estimators
and 100 features. A single shape cannot be freed. Every bucket up to 1024 rows
used to stay, so memory grew with each new predict size: ten predicts of 1 to
50,000 rows against a 2,000-row context kept 11.6 GiB.

### Budget

The predict graph's budget is the device's share, as for builds: half the
host's available memory, capped by Metal's working set and `memory_limit`,
measured when the context is built or loaded. Instead of freeing shapes, the
graph runs only a fixed set of shapes whose total fits that budget, chosen from
two powers of two, `chunk` and `small`:

- A query of at most `small` rows runs once, padded to a bucket that keeps its
  top four significant bits (at most 12.5% padding).
- A larger query runs in pieces: whole chunks of `chunk` rows, then the largest
  power of two that fits, until at most `small` rows remain, which run as one
  bucket. Pieces are not padded; a query takes at most log2(`chunk`/`small`)
  extra runs, about 10 ms each.

The powers of two up to `chunk` take about `2 x chunk` rows, so `chunk` is the
largest, up to 1024, that takes at most half the budget. `small` is the largest
whose buckets fit the rest. A kept row is estimated as building that row for
every estimator, 3.7 MB at 8 estimators and 100 features, about 1.4 times what
the graph keeps. At that width:

| Budget | `chunk` | `small` | Kept at most (estimated) |
|---|---|---|---|
| No limit, about 19 GiB free (9.5 GiB) | 512 | 128 | 7.9 GiB |
| 4 GiB | 256 | 64 | 3.8 GiB |
| 2 GiB | 128 | 32 | 1.8 GiB |

With no limit, 700 rows run as 512 + 128 + 60, and 10,000 rows as
19 chunks of 512, then 256, then 16.

The same ten predicts, and 40 predicts of mixed sizes from 1 to 2,000 rows run a
second time, on an M1 Max with about 19 GiB free:

| Limit | Kept after the ten predicts | Mixed sizes, second time |
|---|---|---|
| None, before | 11.6 GiB | 0.93 s |
| None | 2.2 GiB | 0.92 s |
| 4 GiB | 1.4 GiB | 1.01 s |
| 2 GiB | 1.0 GiB | 1.12 s |

Without a limit, predicts take as long as before. Under a 4 GiB limit, 10,000
rows take 3% longer, and 100 rows 76 ms instead of 60 ms (64 + 36 rows);
predicts of 1 and 10 rows take 11 ms and 15 ms either way.
`tests/build_memory.rs` checks the kept memory under a limit.

Python's `predict` and `predict_proba` preprocess 8,192 rows at a time, so the
preprocessing copies and the `[B, rows, H]` query array stay bounded for any
number of rows. The CPU path keeps its fixed 1.5 GiB activation limit: it
measured at most 0.35 GiB above the context.

GPU results vary with the batch shape by about 1e-6: a single row differs from
the same row in a batch of 1,000 by up to 5.6e-6 in probability. A memory limit
changes which shapes run, so it can change results that much.

## Phases

| PR | Scope |
|---|---|
| A (#27) | Pools, platform probes, demand, plan, and the `ironicl plan` report. No behavior change |
| B (#28) | The builds use the plan's group sizes by pool; memory limits; the Python API with warnings, errors, and the "cannot complete" verdict; a peak-memory fixture test against the plan |
| D (#29) | GPU queries split into a fixed set of shapes that fits the device's share; Python predicts in row chunks |

Later, as separate plans:

| Phase | Capability |
|---|---|
| E | Build straight into a file-backed context, one estimator at a time, with feature and row chunks, so the build never holds the whole context in RAM |
| F | Out-of-core prediction from the mapped file: one ICL layer at a time with prefetch, per-layer normalized states (half the bytes of K and V), and request batching |
| G | A representation cache, and checkpointed layers between it and full K/V |

Also later: Windows CI, and measuring whether the prefetch hints
(`madvise(WILLNEED)`, `PrefetchVirtualMemory`) speed up cold loads; keep both
or drop both.

## When nothing fits

Some fits cannot complete on a machine at all. 100,000 training rows with 100
features and 8 estimators need a 36.9 GiB context in FP32, 18.5 GiB in FP16, and
9.2 GiB in INT8; one estimator alone needs 4.6 GiB plus its build. On an 8 GB
machine no setting fits, and proceeding does not just slow down: macOS swaps for
hours, Linux's out-of-memory killer stops the process, and Windows fails an
allocation at the commit limit.

The planner therefore separates overflow, which `memory_overflow` governs, from
"cannot complete", which raises even under `"warn"`, before any work starts.
Until phases E and F exist, a context larger than physical memory cannot
complete. The error says what is possible on this machine (32 GB, about 20 GiB
free):

```
IroniclMemoryError: This fit cannot complete on this machine.
Context: 36.9 GiB (8 estimators x 100000 rows x 100 features, ICL caches in FP32).
Peak while building on the CPU: 49.0 GiB, including 11.5 GiB for 1 estimator(s) at a time; available: about 20.1 GiB.
Any one of these fits:
  - at most 24145 training rows (subsample X, y before fit)
  - n_estimators=1
Pass memory_overflow='ignore' to try anyway.
```

Once phases E-G exist, the planner offers streaming from a file and the
representation cache instead, and "cannot complete" becomes rare.

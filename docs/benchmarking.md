# Benchmarking

Ironicl has no benchmark command today. Time it through the Python estimators, which start from
raw rows as TabICL's do, and compare it with TabICL through the harness below. A cross-engine
harness that runs Ironicl, TabICL, and other models from the same raw rows is planned; see the
[roadmap](roadmap.md).

Run competing thread counts or kernel variants one after another. Concurrent runs on the same
physical cores invalidate latency comparisons. Record whether the machine was otherwise idle, and
keep model loading outside timed calls.

## Environment Variables

These runtime flags apply to any run:

| Variable | Effect |
|---|---|
| `IRONICL_OPTIMIZE` | precision/speed level `safe\|auto\|fast` (also set through the Python API) |
| `IRONICL_NO_METAL` | set to disable the Apple Silicon Metal GPU path |
| `IRONICL_NO_CUDA` | set to disable the NVIDIA GPU path in a `cuda` build |
| `IRONICL_THREADS` | executor thread count; without it, half the logical CPUs on x86-64, the performance-core count on Apple Silicon, and all logical CPUs elsewhere |
| `IRONICL_MEMORY_LIMIT` | the most host RAM to plan for, such as `16G` or `512M`; also covers GPUs that share RAM (also set through the Python API) |
| `IRONICL_DEVICE_MEMORY_LIMIT` | the same for a GPU's own dedicated memory; no effect where memory is shared |
| `IRONICL_MEMORY_OVERFLOW` | `warn\|error\|ignore` when a fit's plan exceeds memory (also set through the Python API) |
| `IRONICL_PROFILE` | set to collect phase timings in a `profile` build; see [Phase Timings](#phase-timings) |

Exact-math control runs (scalar GELU, softmax, and LayerNorm for parity debugging) are a
build-time choice: build with `--features exact-math`.

The dormant experiment features (`bf16-dense`, `bf16-value-dot`, `sparse-attention`,
`attention-stats`, and `experimental`) are removed. See
[removed-experiments.md](removed-experiments.md#dormant-experiment-features).

The test and benchmark harnesses also read `IRONICL_CHECKPOINT_FIXTURES`,
`IRONICL_TABICL_SOURCE`, and the `IRONICL_BENCH_*` variables. They are not engine runtime flags.

## Phase Timings

The `profile` Cargo feature times the engine's phases: the column, row, and ICL stages, the
transformer blocks, and the kernels inside them (affine, LayerNorm, GELU, QASSMax, RoPE, and the
attention steps). Without the feature, the timers compile to nothing.

Build the extension with the feature, set `IRONICL_PROFILE` before the first fit, and read the
totals from Python:

```bash
maturin develop --release --uv --features python,metal,profile
```

```python
import os
os.environ["IRONICL_PROFILE"] = "1"

from ironicl import TabICLClassifier, _core

model = TabICLClassifier().fit(x_train, y_train)
_core.profile_reset()
model.predict_proba(x_test)
for category, calls, seconds in _core.profile_report():
    print(f"{category:>18} {calls:>8} {seconds:10.4f} s")
```

`_core.profile_report()` returns one `(category, calls, total seconds)` tuple per category,
accumulated since the start or the last `_core.profile_reset()`. It returns an empty list in a
build without the feature or when `IRONICL_PROFILE` was not set. The timers cover CPU code only:
work that runs on Metal or CUDA goes untimed, so set `IRONICL_NO_METAL` or `IRONICL_NO_CUDA` for a
full breakdown of the CPU path.

## Memory Traces

`tools/trace_memory.py` fits and predicts with the Python estimator on a synthetic dataset and
prints each stage's current and peak physical footprint: model load, fit, and predictions of 1,
64, and 1,000 rows. The footprint includes GPU buffers on Apple Silicon. It runs on macOS only.

```bash
python tools/trace_memory.py --checkpoint PATH --rows 10000 --features 100 --optimize auto
```

Set `IRONICL_NO_METAL=1` to trace the CPU path and `IRONICL_MEMORY_LIMIT` to plan for less memory.

## TabICL Baseline

`tools/benchmark_tabicl_sklearn_cache.py` benchmarks TabICL's public `TabICLClassifier.fit()` and
`predict_proba()`. It uses a deterministic, stratified Covertype subset, an official local
checkpoint, and either `kv_cache="kv"` or `kv_cache="repr"`. The `fit_cache_build_ms` field
includes model loading, preprocessing, and cache construction. Timed latency includes only cached
`predict_proba()` calls after warmups.

The JSON result records context and query sizes, cache mode, precision, output shape and
checksums, warmup and iteration counts, and peak process RSS and CUDA allocated and reserved VRAM
for the fit and predict phases. It also records every prediction sample, the CPU affinity mask,
and the fixed inter-op thread count of `1`. CUDA samples synchronize before and after each
prediction.

`predict_proba()` timing includes query validation, query preprocessing, class correction, and
probability aggregation, as Ironicl's `predict_proba()` does. Compare the two only at that
boundary.

On the Ryzen benchmark host, run all required CPU and CUDA FP32 points with:

```bash
tools/run_ryzen_tabicl_sklearn_cache_benchmarks.sh "$BENCH_ROOT"
```

`tools/run_large_cached_benchmarks.sh` runs the same harness at larger context sizes.

See [real-dataset-benchmarks.md](real-dataset-benchmarks.md) for earlier Ryzen, RTX 5080, and
real-dataset measurements, taken with tools that have since been removed.

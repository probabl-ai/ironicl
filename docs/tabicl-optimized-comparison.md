# Ironicl vs. an Optimized TabICL on Apple Silicon

This report compares Ironicl with the best TabICL we could build on an Apple
Silicon GPU, using only what PyTorch offers. It documents what we changed in
TabICL, what we tried and rejected, the head-to-head results, and what the
results imply for Ironicl. Measurements were taken on 2026-11-06.

## Summary

We did not compare against stock TabICL alone. We optimized TabICL first, in a
fork, so that the comparison holds up against a competent maintainer's best
effort: six code changes, one open upstream PR, tuned settings, `torch.compile`,
and `aot_compile` artifacts. Custom kernels were out of scope, because they are
what Ironicl is.

| Result (classifier, M1 Max GPU) | Stock TabICL | Optimized TabICL (compiled) | Ironicl `auto` |
|---|---|---|---|
| One-row predict, N = 2,000 | 106.5 ms | 25.9 ms | **6.7 ms** |
| 64-row predict, N = 2,000 | 112.8 ms | 28.5 ms | **16.6 ms** |
| 1000-row predict, N = 2,000 | 556.5 ms | 379.3 ms | **193.4 ms** |
| 1000-row predict, N = 10,000 | 882.3 ms (swaps) | 629.4 ms | **497.2 ms** |
| Fit, N = 10,000 | 35.7 s (swaps) | 9.9 s | **5.0 s** |
| First predict in a fresh process | under 1 s (eager) | 36 s compile, or a 44 s prebuilt artifact | **0.1 s** |

- **Optimizing TabICL roughly quadrupled its single-row speed on MPS** (106.5 →
  25.9 ms). Most of the old gap to Ironicl was avoidable overhead in TabICL.
- **Ironicl still wins every steady-state case in both precision tiers:** 2–4×
  at one row, 1.4–1.7× at 64 rows, 1.1–2.0× at 1000 rows, and about 2× on fit.
- **With 100 features the lead grows** to 1.9–2.6× across all batch sizes,
  because the column stage gains most from Ironicl's caching.
- **Cold start is Ironicl's largest structural advantage.** Compiled TabICL needs
  a compile per dataset shape: 36–61 s with `torch.compile`, or a 44–46 s
  prebuilt `aot_compile` artifact per dataset and query size. Ironicl's first
  predict takes about 0.1 s with no build step.
- **Ironicl's weakest case is FP32 attention at large N with large batches**:
  1.14–1.19× at N = 10,000 and 1000 rows. Fitted cost lines cross near
  N ≈ 30,000.
- **Ironicl has a memory problem with wide tables**: 21.5 GiB peak at
  N = 10,000 with 100 features, against TabICL's 12.4 GiB.

## Setup

| Item | Value |
|---|---|
| Machine | Apple M1 Max (8 performance + 2 efficiency cores), 32 GB unified memory, macOS 27.2 |
| PyTorch | 2.15.0, MPS backend; Python 3.12 |
| TabICL | `main` at `d83091b`, plus the `perf/baseline` branch described below |
| Ironicl | `feat/regression` at `27f5c0c` (#26) |
| Checkpoints | `tabicl-classifier-v2-20260212.ckpt`, `tabicl-regressor-v2-20260212.ckpt` |
| Data | `make_classification` (3 classes, 10 informative features) and `make_regression`, 20 features unless stated |
| Ensemble | 8 estimators, `norm_methods` default (`none` and `power`), `random_state=42` |

### Protocol

- **One process per case** (engine × task × N), so peak memory belongs to that
  case alone.
- **Warm-up before timing:** one small fit to load the model and initialize the
  device, then two predicts per query size.
- **Medians:** fit is the median of 2 runs; predict is the median of 15 (1 row),
  7 (64 rows), or 3 (1000 rows) runs, with device synchronization around each.
- **Peak memory:** the process's lifetime maximum physical footprint
  (`proc_pid_rusage`, `ri_lifetime_max_phys_footprint`), which includes GPU
  buffers on unified memory.
- **Swap:** recorded before and after every case. A case whose swap grows is
  flagged and excluded from speed claims.
- **Clean machine:** after rebooting with Docker Desktop (8.2 GB VM) and other
  heavy apps closed, runs started only after the load average fell below 2.5.
  Earlier runs with the VM active were discarded.
- **Precision tiers paired by error:** TabICL FP32 against Ironicl `safe`; TabICL
  with an FP16 ICL cache against Ironicl `auto`.

| Tier accuracy (max / mean error against its own FP32) | Classifier, probability | Regressor, target std |
|---|---|---|
| TabICL with FP16 ICL cache | 3.2–4.4e-4 / 7.4–7.5e-6 | 1.6–1.8e-4 / 3.8e-5 |
| Ironicl `auto` | 3.7–5.3e-4 / 8.9–9.0e-6 | 1.9–2.3e-4 / 3.9–4.0e-5 |

## The optimized TabICL

All changes live on `perf/baseline` in the fork `Aljutor/tabicl`, 15 commits on
top of `d83091b`. Predictions stay within stock TabICL's own run-to-run
variation (up to 7.2e-7 relative; two stock runs agree bit for bit in only 6 of
18 test cases), and TabICL's test suite passes (267 passed, 236 skipped).

### Changes kept

| Commit | Change | Effect on MPS (classifier, N = 2,000) |
|---|---|---|
| `5619ea5` | **One forward pass for all estimators.** Fit joins the per-normalization-group KV caches into one merged cache (group entries become views); predict runs every estimator in one pass instead of one per group | One row −22%, 64 rows −36% |
| `90de90f`…`6fab9d2` | **PR #179** (Anne Beyer, cherry-picked): caps automatic batching to real free memory on unified-memory GPUs | N = 10,000 peak 29.1 → 9.7 GiB, fit 30.3 → 10.1 s, no swapping |
| `2cc9827` | **Cache `log(n)` in QASSMax.** Every QASSMax forward created a scalar tensor on the device, about 100 host-to-device copies per predict | One row 86.2 → 54.4 ms (−37%) |
| `8245801` | **Attend at the precision of an FP16 ICL cache** (opt-in `_cast_icl_cache`). Only the ICL keys and values are FP16; queries are cast for the attention call | 1000 rows −6 to −13%; error as in the tier table |
| `d1a0770` | **Branch-free skip masks.** `if mask.any()` checks forced a GPU sync in the column input layer and every cached column block | One row −4% (A/B: 54.1 → 51.9 ms) |
| `338cf3c`, `8d413a5` | **Batching decided once per input shape.** Every stage called `InferenceManager.configure()` on every forward, rebuilding the pinned-buffer pool; `__call__` re-queried GPU, CPU (`psutil`), and disk memory on every call. Decisions are now cached per input signature, and an OOM still halves the cached batch size | One row 51.7 → 35.4 ms (−32%); `torch.compile` graph breaks 16 → 0 |

Settings for every TabICL run: `device="mps"`, `kv_cache=True`,
`use_amp=False`, `batch_size=8`. A sweep at N = 10,000 confirmed `batch_size=8`
as fastest once #179 is in (values 4, 2, and 1 were up to 2.7× slower at one
row, at the same memory).

### Candidates rejected

| Candidate | Measurement | Why rejected |
|---|---|---|
| AMP (FP16 autocast) on MPS | 1000 rows +8–26% slower, one row +38–62% slower | Per-op casting costs more than FP16 saves |
| Whole model and cache in FP16 (`model.half()`) | Logit error 0.12–0.21; 1000 rows only 8–16% faster | LayerNorm, softmax, and QASSMax lose too much precision |
| Attention as `matmul` → softmax → `matmul` (CPU) | 3–4× slower than SDPA | PyTorch's CPU flash attention already tiles through Accelerate (about 785 GFLOP/s measured) |
| `torch.compile` before the batching fix | Dynamic shapes crash in Dynamo (`f"{output_mb:.0f}"` on a symbolic value, `inference.py:1108`); fixed shapes give 17 graphs and a 5× slower single row | Fixed by `338cf3c`; see compiled results |
| Regional compilation, `nested_compile_region` | Dynamo `AssertionError: Freevar has no source` (an `.item()` call in the block) | Experimental feature fails on TabICL's blocks |
| Regional compilation, per block | Cold first calls 9.1 s, warm 3.5 s, new N 4.3 s; one row 32.9 ms, 1000 rows 431 ms | Fixes cold start but keeps only about 9% (one row) and 1% (1000 rows) of the speedup over eager |
| `torch.compile` on CPU | One row slower than eager (32.1 vs 26.9 ms with `aot_compile`); 1000 rows 4% faster | Not worth the compile time on CPU |

## Results on the Apple GPU

### Classifier, 20 features: steady state

All runs from one session (`clf_compiled.csv`); compile time excluded.

**Exact tier: Ironicl `safe` vs. TabICL**

| N | Engine | Fit | 1 row | 64 rows | 1000 rows | Peak memory |
|---|---|---|---|---|---|---|
| 2,000 | Ironicl `safe` | **0.80 s** | **8.6 ms** | **18.8 ms** | **218 ms** | **3.0 GiB** |
| | TabICL eager FP32 | 1.67 s | 36.0 ms | 34.5 ms | 431 ms | 6.2 GiB |
| | TabICL compiled FP32 | 1.69 s | 26.7 ms | 30.0 ms | 399 ms | 6.5 GiB |
| 5,000 | Ironicl `safe` | **2.10 s** | **11.4 ms** | **26.8 ms** | **336 ms** | **5.2 GiB** |
| | TabICL eager FP32 | 4.00 s | 39.4 ms | 43.2 ms | 596 ms | 8.1 GiB |
| | TabICL compiled FP32 | 4.08 s | 30.0 ms | 39.3 ms | 518 ms | 8.5 GiB |
| 10,000 | Ironicl `safe` | **5.99 s** | **17.3 ms** | **40.8 ms** | **627 ms** | **9.8 GiB** |
| | TabICL eager FP32 | 10.26 s | 45.2 ms | 59.8 ms | 829 ms | 11.6 GiB |
| | TabICL compiled FP32 | 10.23 s | 35.4 ms | 55.4 ms | 713 ms | 11.6 GiB |

**FP16 tier: Ironicl `auto` vs. TabICL with FP16 ICL cache**

| N | Engine | Fit | 1 row | 64 rows | 1000 rows |
|---|---|---|---|---|---|
| 2,000 | Ironicl `auto` | **0.75 s** | **6.7 ms** | **16.6 ms** | **193 ms** |
| | TabICL compiled | 1.69 s | 25.9 ms | 28.5 ms | 379 ms |
| 5,000 | Ironicl `auto` | **1.87 s** | **9.1 ms** | **22.5 ms** | **282 ms** |
| | TabICL compiled | 4.05 s | 27.4 ms | 36.0 ms | 472 ms |
| 10,000 | Ironicl `auto` | **5.01 s** | **13.4 ms** | **31.8 ms** | **497 ms** |
| | TabICL compiled | 9.92 s | 29.9 ms | 48.4 ms | 629 ms |

**Ironicl's lead over compiled TabICL**

| N | Fit | 1 row | 64 rows | 1000 rows |
|---|---|---|---|---|
| 2,000 | 2.1× / 2.3× | 3.1× / 3.9× | 1.6× / 1.7× | 1.8× / 2.0× |
| 5,000 | 1.9× / 2.2× | 2.6× / 3.0× | 1.5× / 1.6× | 1.5× / 1.7× |
| 10,000 | 1.7× / 2.0× | 2.0× / 2.2× | 1.4× / 1.5× | 1.14× / 1.3× |

Each cell is exact tier / FP16 tier.

### Stock TabICL, for reference

| N | Fit | 1 row | 64 rows | 1000 rows | Peak memory | Notes |
|---|---|---|---|---|---|---|
| 2,000 | 1.91 s | 106.5 ms | 112.8 ms | 556.5 ms | 8.6 GiB | |
| 5,000 | 4.76 s | 110.6 ms | 123.4 ms | 680.5 ms | 17.6 GiB | |
| 10,000 | 35.7 s | 117.2 ms | 140.7 ms | 882.3 ms | 29.4 GiB | Swapped (+473 MB): excluded from speed claims |

### Regressor, 20 features

From an earlier session, before the skip-mask and batching-decision commits, so
TabICL's single-row numbers here are about 20 ms slower than its current eager
state. Ironicl `auto` vs. optimized eager TabICL with FP16 ICL cache:

| N | Fit | 1 row | 64 rows | 1000 rows |
|---|---|---|---|---|
| 2,000 | 0.75 vs 1.83 s | 6.8 vs 57.5 ms | 19.2 vs 69.4 ms | 228 vs 501 ms |
| 5,000 | 1.85 vs 4.01 s | 9.1 vs 58.8 ms | 25.5 vs 78.1 ms | 314 vs 599 ms |
| 10,000 | 5.02 vs 10.08 s | 13.6 vs 61.5 ms | 34.4 vs 90.9 ms | 478 vs 754 ms |

The regressor tracks the classifier closely in both engines; a rerun against
the final TabICL state is pending.

### Wide table: 100 features, N = 10,000 (classifier)

| Engine | Fit | 1 row | 64 rows | 1000 rows | Peak memory | First predicts (1 / 64 / 1000 rows) |
|---|---|---|---|---|---|---|
| Ironicl `auto` | **10.7 s** | **17.6 ms** | **52.6 ms** | **852 ms** | 21.5 GiB | 0.26 / 0.16 / 0.94 s |
| Ironicl `fast` | 10.6 s | 17.6 ms | 52.8 ms | 875 ms | 21.5 GiB | 0.26 / 0.16 / 1.43 s |
| TabICL eager, FP16 ICL | 27.9 s | 41.5 ms | 132.5 ms | 2006 ms | 13.4 GiB | 0.16 / 0.18 / 1.97 s |
| TabICL compiled, FP16 ICL | 27.7 s | 32.5 ms | 125.5 ms | 1662 ms | **12.4 GiB** | 37.8 / 13.1 / 42.4 s |

Against compiled TabICL, Ironicl is 2.6× faster on fit and 1.85×, 2.4×, and
1.95× faster at 1, 64, and 1000 rows. With 20 features at the same N the
ratios were 2.0×, 1.5×, and 1.3×. `fast` matches `auto` on the Apple GPU, where
both store FP16 ICL caches.

## Cold start

### First predict and one-time costs

| Engine | One-time build | First predict, 1 row | 64 rows | 1000 rows |
|---|---|---|---|---|
| Ironicl | none | 0.12–0.19 s | 0.12–0.14 s | 0.31–0.66 s |
| TabICL eager | none | 0.08–0.10 s | 0.07–0.10 s | 0.5–0.8 s |
| TabICL `torch.compile`, cold cache | in the first call | 36–37 s | 10.8 s | 14.6–15.2 s |
| TabICL `torch.compile`, warm cache, same N | in the first call | 12.5 s | 4.1 s | 2.9 s |
| TabICL `torch.compile`, warm cache, new N | in the first call | about 36 s (no cache reuse across N) | | |
| TabICL `aot_compile`, prebuilt | 44–46 s per dataset (3 artifacts) | 1.44–1.51 s (load + call) | 0.24–0.28 s | 0.68–1.01 s |

`aot_compile` artifacts load in a fresh process (0.7 s load, 0.1 s first
one-row call) and run at compiled speed, but each is fixed to one shape:
training size, feature count, estimator count, and query rows. A new dataset
shape needs a new build of 12–25 s per query size. The artifacts must be
compiled under `torch.no_grad()`, or the grad-mode guard rejects them at load.

### Import time

| Import in a fresh process | Time |
|---|---|
| `numpy` | 0.05 s |
| `scipy.stats` | 0.51 s |
| `sklearn.base` (imports SciPy) | 0.84 s |
| `torch` | 0.69 s |
| `ironicl` | 0.88 s |
| `tabicl` | 1.84 s |

Almost all of Ironicl's import time is scikit-learn, which the drop-in API
subclasses.

## Analysis

### Why TabICL was slower, and still is

Fitting each engine's timings to *time ≈ fixed cost + cost per 1,000 training
rows* (classifier, before compilation) separates overhead from compute:

| Predict | TabICL fixed | Ironicl fixed | TabICL per 1k rows | Ironicl per 1k rows |
|---|---|---|---|---|
| 1 row, FP32 vs `safe` | about 52 ms | about 5 ms | about 1.2 ms | about 1.2 ms |
| 1000 rows, FP32 vs `safe` | about 480 ms | about 120 ms | about 38 ms | **about 50 ms** |
| 1000 rows, FP16 vs `auto` | about 440 ms | about 110 ms | about 32 ms | about 33 ms |

- **The model math costs the same in both engines.** Both use Apple's attention
  and matmul kernels; Ironicl's Metal path uses MPSGraph's
  `scaledDotProductAttention`.
- **The gap was fixed cost per call.** Eager TabICL launches about 810 linear
  layers, 102 attention calls, and hundreds of small operations one at a time
  from Python for a single row. The profile showed about 14 ms of GPU work in a
  54 ms predict. Ironicl submits one compiled MPSGraph executable per batch-size
  bucket, with LayerNorm, GELU, and residual adds fused.
- **Compilation closes part of the fixed-cost gap,** at the cost of the compile
  step described above.
- **FP32 attention at large N is Ironicl's weakness:** its per-row cost (about
  50 ms per 1,000 training rows at 1000 query rows) exceeds TabICL's (about
  38 ms). The lines cross near N ≈ 30,000 (an estimate, not measured).

### Preprocessing is not a factor

Both engines run the same preprocessing algorithm (Ironicl's is a port of
TabICL's), at the same cost:

| Case | TabICL | Ironicl |
|---|---|---|
| N = 2,000, 20 features, 1 row | 0.75 ms (2.1% of predict) | 0.65 ms (7.4%) |
| N = 2,000, 20 features, 1000 rows | 3.1 ms (0.7%) | 2.0 ms (0.9%) |
| N = 2,000, 100 features, 1 row | 2.1 ms (5.3%) | 1.9 ms (15.6%) |
| Fit, N = 2,000, 100 features | 340 ms (6.7%) | 329 ms (16.8%) |

Porting the transform to Rust would save Ironicl 5–15% at one row and widen its
lead slightly, since TabICL keeps the cost.

### CPU note

On this Mac's CPU (ARM NEON), optimized eager TabICL predicts one row in
26.9 ms. A comparable Ironicl CPU run in an earlier session took 24.8 ms
(regressor, same N and features). The single-row CPU lead may now be small. No
paired CPU benchmark has been run yet, and x86 CPUs (AVX2, AVX-512, AMX) are
untested.

## Issues found in TabICL

These are candidates for upstream PRs or issues; none has been filed from this
work yet.

| Issue | Where | Status |
|---|---|---|
| Regressor variance converted with `σ·v + μ` instead of `σ²·v` | `regressor.py` | [soda-inria/tabicl#180](https://github.com/soda-inria/tabicl/issues/180) |
| `configure()` runs on every forward and rebuilds the pinned-buffer pool | `inference.py`, all stages | Fixed on `perf/baseline` (`338cf3c`) |
| Memory and offload decisions re-run on every forward pass | `InferenceManager.__call__` | Fixed on `perf/baseline` (`338cf3c`) |
| `log(n)` tensor created on the device on every QASSMax forward | `ssmax.py` | Fixed on `perf/baseline` (`2cc9827`) |
| Data-dependent skip-mask branches force GPU syncs | `layers.py` | Fixed on `perf/baseline` (`d1a0770`) |
| One forward pass per normalization group | `classifier.py`, `regressor.py` | Fixed on `perf/baseline` (`5619ea5`) |
| Auto-batching overestimates free memory on unified-memory GPUs | `_torch_devices.py` | Open upstream as #179 |
| Predictions are not reproducible run to run in multi-group ensembles | Inference path | Not investigated |
| Dynamo crashes formatting a symbolic value under dynamic shapes | PyTorch, triggered at `inference.py:1108` | Avoided by `338cf3c` |

## What this means for Ironicl

- **Defensible claims on Apple GPUs:** small-batch latency (2–4× at one row),
  fit (about 2×), wide tables (about 2× across batch sizes), and cold start
  (0.1 s against tens of seconds for compiled TabICL). Against stock TabICL the
  ratios are much larger, but the optimized comparison is the honest one.
- **To fix before publishing:**
  1. Peak memory with wide tables (21.5 GiB at N = 10,000, 100 features).
  2. FP32 attention on Metal at large N, where compiled TabICL is close and
     likely overtakes near N ≈ 30,000.
- **To measure next:** a paired CPU benchmark on Apple Silicon, x86 CPUs
  (AVX2, AVX-512, AMX), N of 20,000–50,000, and the regressor against the final
  TabICL state.
- **Cheap wins:** a transform-only Rust preprocessing port (5–15% at one row)
  and a lean serving entry point that skips scikit-learn's 0.84 s import.

## Limitations

- One machine (M1 Max), synthetic data, mostly the classifier.
- Each number is a median from a single process; run-to-run variation between
  sessions was up to about 7% for large batches.
- Swap was not zero at the start of most sessions (0.6–1.4 GB left from earlier
  work); only cases whose swap grew are flagged.
- The benchmark scripts live in the fork's working tree
  (`~/Developer/probabl/tabicl-perf/bench/`, untracked) and should move into
  `tools/` before this report is cited externally.

## Reproduction

1. Clone `Aljutor/tabicl` and check out `perf/baseline` (local only for now).
2. Create a Python 3.12 environment with `torch==2.15.0`, the fork installed
   with `pip install -e ".[test,shap]"`, and Ironicl built with
   `maturin develop --release`.
3. Run each case in its own process:

   ```bash
   # Ironicl on Metal
   python bench/mps_vs_metal.py ironicl-auto ironicl-auto clf 2000
   # TabICL on MPS, FP16 ICL cache, compiled
   TABICL_ICL_FP16=1 TABICL_COMPILE=1 python bench/mps_vs_metal.py tabicl tabicl-compiled clf 2000
   # TabICL aot_compile artifacts (build in a subprocess, then serve)
   python bench/aot_grid.py tabicl-iclfp16 2000
   ```

   Set `FEATURES=100` for the wide case, `TABICL_KW='{"batch_size": 2}'` to
   override TabICL settings, and `TORCHINDUCTOR_CACHE_DIR` to a persistent
   directory for the compile cache.

# Benchmarks

*The benchmark record for IronICL: how measurements are taken, the recorded
results, the optimisations that were kept and those that were rejected, and
what the numbers support.*

The design of the engine is in [design.md](design.md); the validation policy is
in [development.md](development.md#5-validation).

**Contents**

1. [Measurement setup](#1-measurement-setup)
2. [Methodology](#2-methodology)
3. [Results](#3-results)
4. [Retained optimisations](#4-retained-optimisations)
5. [Rejected experiments](#5-rejected-experiments)
6. [Summary of findings](#6-summary-of-findings)
7. [Reporting new results](#7-reporting-new-results)

## 1. Measurement setup

All recorded numbers come from one host:

| Component | Value |
|---|---|
| CPU | AMD Ryzen 7 9800X3D, 8 physical / 16 logical cores |
| Memory | 60 GiB usable RAM |
| GPU (reference rows only) | NVIDIA RTX 5080, 16 GiB |
| Operating system | Arch Linux |
| PyTorch (reference rows only) | 2.13.0+cu130 |
| IronICL build | release, `-C target-cpu=znver5` |

The Apple M1 Max was a build-and-correctness machine only; no latency was
recorded on it.

> [!NOTE]
> The source documents record neither run dates nor commit hashes (the
> history starts at one initial commit), so these appear as "not recorded"
> below. `benchmarks/bench_cached_serving.py` writes the short commit, the
> host and the IronICL version into its JSON output for future runs.

## 2. Methodology

### 2.1. What is timed

| Boundary | Included | Excluded |
|---|---|---|
| Python raw-X cached serving (`benchmarks/bench_cached_serving.py`) | `predict_proba` on raw numeric X: validation, native preprocessing and ensemble variant generation, cached inference, class-permutation correction, logit averaging, temperature softmax, host FP32 probabilities | `fit` (model load, preprocessing fit, native K/V context build); reported separately as `fit_seconds` |
| Engine-only CLI (`ironicl benchmark`) | One prediction call on a prepared `IRINPUT` or `IRQUERY` request; with `--context --probabilities`, also correction, averaging and softmax | Preprocessing, artifact and bundle loading, executor construction, checkpoint import and conversion, context load |
| Historical uncached cross-engine rows | Inference on an officially preprocessed request; CUDA samples synchronise before and after each call | Preprocessing, model loading, input loading |

Only the Python boundary may be used for a cross-engine serving ratio, because
both engines start from the same raw rows. Engine-only CLI timings start from a
prepared query and must not be compared with public `predict_proba` timings.

### 2.2. Warm-up, iterations and statistics

- Python benchmark: `--warmup 3 --iterations 20` by default; the recorded runs
  used 5 warm-ups and 50 timed samples (30 at `Q=500`).
- CLI benchmark: `--warmup` defaults to 2 (0 to 1,000) and `--iterations` to
  10 (1 to 10,000). Each call is timed separately, and the run is rejected if
  an output changes shape or contains a non-finite value.
- Both tools use nearest-rank percentiles: sorted sample `ceil(p * N)`.
- Thread-count and kernel variants must run sequentially on an idle host;
  parallel benchmark processes on the same cores produced contaminated data.

### 2.3. Threads and pinning

IronICL reads `IRONICL_THREADS`; when it is unset, it uses the physical cores
on x86_64 (logical / 2) and the logical cores elsewhere. The Python benchmark
sets `IRONICL_THREADS` to `--threads` (default `os.cpu_count()`). The
historical PyTorch comparisons used `torch.set_num_threads` with the same
value, except that the oldest runs pinned PyTorch to CPUs 0-7 and tuned
IronICL to CPUs 0-15. Record the affinity and SMT policy with every published
number.

### 2.4. Running the Python benchmark

```bash
pip install ironicl
python benchmarks/bench_cached_serving.py --dataset covertype \
    --context-rows 4096 --query-rows 128 --estimators 8 --threads 16 \
    --warmup 5 --iterations 50 --kv-dtype fp32 bf16 --check-parity --output results.json
```

The script fits one `IronICLClassifier` per `--kv-dtype` (`fp32`, `bf16`,
`bf16-v`) on the same table, times `predict_proba` on the same raw query rows,
and reports p50, p95 and mean latency, rows per second, and, for every
precision after the first, its maximum absolute probability difference and
label agreement with the first. `--check-parity` fails the run above `1e-2`
difference or on any label mismatch.

> [!NOTE]
> The PyTorch rows in the tables below were measured with
> `TabICLClassifier(kv_cache="kv", device="cpu")` on the same rows when the
> benchmark still drove both engines. This repository no longer depends on
> tabicl or PyTorch; reproduce those rows from a separate tabicl environment.

### 2.5. Running the CLI benchmark

Uncached (training table inside the request) and cached (query-only request
against a static context):

```bash
ironicl benchmark --model model.ironicl --input request.irinput --warmup 2 --iterations 10
IRONICL_THREADS=16 ironicl benchmark --model model.ironicl \
    --context context.ironiclctx --input request.irquery \
    --probabilities --warmup 5 --iterations 50
```

The output is `key: value` lines: paths, `input_load_ms`, `model_load_ms`,
`warmup`, `iterations`, `output`, `shape`, `min/p50/p95/p99/max/mean_ms`,
`inferences_per_second`, `query_rows_per_second`, `output_checksum` (FP64
sum), `logits_checksum`, and `samples_ns`.

### 2.6. Environment variables

The engine reads the following variables (in `src/executor.rs` unless noted);
unset means the serving default.

| Variable | Effect |
|---|---|
| `IRONICL_THREADS=N` | Total worker budget (default: physical cores on x86_64). |
| `IRONICL_SERIAL_ENSEMBLES=1` | Disable estimator-level parallelism; control runs only. |
| `IRONICL_ENSEMBLE_WORKERS=N` | Explicit estimator concurrency; each gets `THREADS / N` inner workers. Ignored when it could exhaust the pool. |
| `IRONICL_INNER_WORKERS=N` | Override the per-estimator inner budget (oversubscription experiments). |
| `IRONICL_LAYER_MAJOR=1` | Layer-major schedule across estimators; benchmarked slower, kept for experiments. |
| `IRONICL_EXACT_GELU=1` | Scalar `erf` GELU instead of the bounded AVX-512 approximation (`kernels/reference.rs`). |
| `IRONICL_EXACT_SOFTMAX=1` | Scalar online-softmax exponential instead of the native path (`kernels/native.rs`). |
| `IRONICL_EXACT_LAYER_NORM=1` | Scalar LayerNorm control path (`kernels/reference.rs`). |
| `IRONICL_ATTENTION_KEY_TILE_ROWS=64..1024` | Source-row tile for cached attention (default 128; powers of two only). |
| `IRONICL_SMALL_ATTENTION_QUERY_ROWS=8\|16` | Query rows per attention block (default 16). |
| `IRONICL_COLUMN_QUERY_MAJOR=1`, `IRONICL_ROW_QUERY_MAJOR=1`, `IRONICL_QUERY_MAJOR_ATTENTION=1` | Fall back to query-major traversal for column, row or cached ICL attention instead of source-major AVX-512. |
| `IRONICL_BF16_VALUE_DOT=1` | Runtime BF16 V dot product (rejected; `kernels/reference.rs`). |
| `IRONICL_BF16_DENSE=1\|all\|list` | BF16 weight storage for selected dense stages (`column`, `icl`, `qass`, `decoder`, ...). Rejected for serving. |
| `IRONICL_BF16_DENSE_COLUMN_BLOCKS`, `IRONICL_BF16_DENSE_ICL_BLOCKS`, `IRONICL_BF16_DENSE_COMPENSATION=none\|input\|weight\|first-order` | Restrict BF16 dense to the listed block indices; choose its error compensation. |
| `IRONICL_PACKED_F32_DENSE`, `IRONICL_PACKED_F32_ROW` | Packed FP32 dense and row affine kernels (rejected at `Q=500`). |
| `IRONICL_SPARSE_ATTENTION=1` | Opt-in k-means sparse index with dense fallback; neutral, not for serving. |
| `IRONICL_SPARSE_MAX_OMITTED_MASS=x` | Sparse acceptance certificate (default `1e-5`). |
| `IRONICL_ATTENTION_STATS=1` | Sample softmax-mass statistics per head (`kernels/reference.rs`). |
| `IRONICL_PROFILE=1` | Phase timing to stderr when built with `--features profile` (`profile.rs`). |

`IRONICL_CACHE_DIR` (Python package) only chooses where imported models and
contexts are stored; it does not change engine behaviour.

## 3. Results

The dataset is Covertype unless stated otherwise. "Not recorded" means the
source documents did not give the value.

### 3.1. Uncached, Covertype scaling

Date and commit: not recorded. Host: Ryzen 7 9800X3D, Arch Linux. `B=1`,
`Q=64` (`Q=32` at `N=512`), 8 cores for both CPU engines, one timed iteration
at the large points; CUDA uses AMP.

| N | Features | IronICL p50 | PyTorch CPU p50 | CUDA AMP p50 | IronICL peak RSS | IronICL / PyTorch CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 46 | 1.838 s | 284 ms | 12.1 ms | 206 MiB | 6.5x slower |
| 1,024 | 49 | 6.038 s | 535 ms | 15.9 ms | 255 MiB | 11.3x slower |
| 1,800 | 51 | 16.721 s | 1.021 s | 24.7 ms | 380 MiB | 16.4x slower |
| 2,048 | 52 | rejected | 1.169 s | 27.6 ms | admission only | |

Parity at `N=1024`: the FP32 output checksums were 1073.207335 (IronICL),
1073.210327 (PyTorch CPU) and 1073.208862 (CUDA FP32); the roadmap records a
maximum absolute logit error below `1e-5` on the official uncached fixture.
IronICL rejects `N=2048` before allocation because the uncached score buffer
would exceed `8 * total_rows * training_rows <= 32 * 1024 * 1024` FP32
scores. The uncached path is 6.5x to 16x slower than PyTorch CPU and does not
scale.

### 3.2. Uncached, wide and other datasets

Date and commit: not recorded. Host: Ryzen 7 9800X3D. `B=1`, 8 cores.

| Dataset | N | Q | Features | IronICL p50 | PyTorch CPU p50 | CUDA AMP p50 | IronICL peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST (top-variance pixels) | 1,024 | 64 | 128 | 8.067 s | not recorded | 33.6 ms | 345 MiB |
| Fashion-MNIST | 512 | 32 | 256 | 4.834 s | not recorded | 34.3 ms | 299 MiB |
| MiniBooNE | 1,024 | 64 | 50 | 6.048 s | 538 ms (1.25 GiB RSS) | 16.1 ms (336 MiB VRAM) | 267 MiB |

Row attention scales quadratically with the feature count. Memory is where
IronICL is clearly ahead: 267 MiB against 1.25 GiB RSS on MiniBooNE.

For GPU context only: uncached PyTorch on the RTX 5080 with `B=8`, `Q=128`
ran in 440 ms at `N=4096` (6.18 GiB VRAM), 5.03 s at `N=32768` (allocation
retries and batching), and 41.0 s at `N=120000` with 37 GiB host RSS. The
16 GiB card loses residency between 8,192 and 32,768 training rows and shifts
pressure to host memory.

### 3.3. Cached, single estimator (engine-only)

Date and commit: not recorded. Host: Ryzen 7 9800X3D. `B=1`, `N=1024`,
`Q=64`, `H=54`, `C=7`, FP32 K/V, prepared query, 5 warm-ups / 50 iterations.

| Engine | Threads | p50 | p95 | Range | Rows/s | Parity vs cached PyTorch logits |
|---|---:|---:|---:|---:|---:|---|
| IronICL cached FP32, final path | 8 | 34.57 ms | 36.30 ms | 31.49-36.50 ms | 1,854.62 | max abs logit error `3.05e-5` (gate `1e-4`) |
| Cached PyTorch CPU FP32 | 1 | 92 ms | | | | baseline |
| Cached PyTorch CUDA FP32 | GPU | 5.84 ms | | | | |

The PyTorch CPU reference ran on one thread, so the 2.7x ratio is not a fair
engine comparison. The internal progression from 338 ms scalar to this path
(about 10x) is itemised in the [retained optimisations](#4-retained-optimisations).

### 3.4. Cached, 8 estimators, prepared query (engine-only)

Date and commit: not recorded. Host: Ryzen 7 9800X3D, 16 logical workers.
`B=8`, `N=4096`, `H=53`, `C=7`, packed FP32 K / BF16 V, final probabilities.

| Q | Samples | p50 | p95 | Rows/s | Max probability error | Labels | PyTorch comparison |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 50 | 199.64 ms | 202.39 ms | 642.76 | `2.458e-4` | 100% | none valid at this boundary |
| 500 | 30 | 752.07 ms | 756.22 ms | 665.14 | `2.575e-4` | 100% | none valid at this boundary |

The `Q=128` progression from the 350.72 ms paired default to 199.64 ms is also
itemised in the retained optimisations. `Q=500` scales nearly linearly but
misses the 200 ms target.

### 3.5. Cached, fair raw-X serving, FP32 exact controls

Date and commit: not recorded. Host: Ryzen 7 9800X3D (CPU) and RTX 5080
(GPU). `B=8`, `N=4096`, 54 raw / 53 prepared features, `C=7`, FP32 K/V on
both engines, IronICL with `IRONICL_EXACT_GELU=1 IRONICL_EXACT_SOFTMAX=1`,
batch size 8, 5 warm-ups / 50 samples. This is the only fair cross-engine
comparison.

| Q | Engine | Threads / affinity | p50 | p95 | Rows/s | Max prob. diff | Labels |
|---:|---|---|---:|---:|---:|---:|---:|
| 128 | PyTorch CPU FP32 | 16 / CPUs 0-7 | 367.88 ms | 382.45 ms | 344.68 | baseline | |
| 128 | IronICL FP32 exact | 16 / CPUs 0-15 | 349.98 ms | 353.47 ms | 365.77 | `1.47e-5` | 100% |
| 128 | PyTorch RTX 5080 FP32 | GPU | 31.78 ms | 32.20 ms | 4,016.81 | `1.52e-5` | 100% |
| 128 | PyTorch RTX 5080 AMP | GPU | 17.79 ms | 18.11 ms | 7,156.75 | `4.13e-2` (rejected) | 99.22% |
| 500 | PyTorch CPU FP32 | 16 / CPUs 0-7 | 1,643.12 ms | 1,702.17 ms | 303.48 | baseline | |
| 500 | IronICL FP32 exact | 16 / CPUs 0-15 | 1,322.27 ms | 1,333.37 ms | 378.02 | `1.47e-5` | 100% |
| 500 | PyTorch RTX 5080 FP32 | GPU | 104.49 ms | 104.65 ms | 4,782.73 | `2.00e-5` | 100% |

IronICL over PyTorch CPU: 1.051x at `Q=128` and 1.243x at `Q=500`. RTX 5080
FP32 over IronICL: 11.01x and 12.65x. The affinity mismatch (8 against 16
CPUs) favours IronICL, so even these ratios are upper bounds.

### 3.6. Cached, fair raw-X serving, accepted BF16-V fast lane

Date and commit: not recorded. Host: Ryzen 7 9800X3D, 16 threads on CPUs
0-15. `B=8`, `N=4096`, `Q=128`, packed FP32 K, source-major BF16 V, FP32
column K/V, FP32 accumulation, native GELU and softmax.

| Metric | PyTorch CPU FP32 (CPUs 0-7) | IronICL fast lane |
|---|---:|---:|
| p50 | 367.88 ms | 193.13 ms |
| Rows/s | 344.68 | 662.68 |
| Observed ratio | baseline | 1.90x |
| Max probability difference | baseline | `2.21e-4` |
| Label agreement | baseline | 100% |
| Context size | about 1.79 GB | about 1.39 GB (BF16 V saves 384 MiB, 22.5%) |

Two `Q=128` figures exist for this configuration: 193.13 ms from the raw-X
serving harness (the later run, recorded as the accepted result) and
199.64 ms from the engine-only prepared-query CLI benchmark that closed the
optimisation progression. The same configuration also produced 192.50 ms
(1.66 ms preprocessing + 190.82 ms inference) and 194.33 ms (1.74 + 192.68
ms), so a run-to-run spread of about 1% is normal. No fast-lane `Q=500` raw-X
run was recorded. Two engine-only variants of this lane were neutral: an
8 x `Q=16` microbatch stream reached 656.35 rows/s against 658.31 for the
isolated control with identical output, and the opt-in k-means sparse index
(M=64) ran 195.00 ms against 195.17 ms indexed dense while accepting 0 of
1,920 attention calls at every omitted-mass threshold from `1e-5` to `1e-1`.

### 3.7. Cached, large context (engine-only)

Date and commit: not recorded. Host: Ryzen 7 9800X3D. `B=1`, `N=100000`,
`Q=128`, `H=54`, `C=7`. RTX 5080 AMP on the same request: 10.04 ms p50; the
GPU cannot hold a `B=8` cache at 50,000 rows or more in 16 GiB.

| Context policy | Artifact | CPU p50 | Max logit error | Max prob. error | Labels |
|---|---:|---:|---:|---:|---:|
| Packed FP32 K, BF16 V | 3.71 GB | 1,005 ms | `4.43e-3` | `7.13e-4` | 100% |
| Packed BF16 K/V | 2.48 GB | 1,018 ms | `4.46e-2` | `6.15e-3` | 100% |

## 4. Retained optimisations

Acceptance gates: at least 5% p50 improvement for invasive changes and 3% for
tile-only tuning; at most 5% p95 regression; cached-logit error at or below
`1e-4` for exact work, or probability error at or below `1e-3` with 100%
labels for reduced-precision work.

| Optimisation | Measured gain | Where |
|---|---|---|
| Resident full K/V static context, checksummed and memory-mapped | Removes the training graph from every request; 6.0 s uncached -> 34.6 ms cached at `N=1024` | `src/static_context.rs` (`memmap2`), `StaticClassifierExecutor` in `src/executor.rs` |
| Head-parallel ICL attention on scoped workers (not Rayon) | 364 -> 111 ms (`B=1`); 70.94 vs 76.53 ms against a Rayon pool | `parallel_cached_attention_*` in `src/executor.rs` |
| AVX-512 packed-K score tiles and weighted-V kernels | 111 -> 55.95 ms | `attention_score_packed_key_*_avx512` in `src/kernels/native.rs` |
| Fused 16 x 128 online-attention tile (score, softmax, V in one pass) | 55.95 -> 48.44 ms; no full score matrix | `src/executor.rs`, `src/kernels/native.rs` |
| Persistent executor-owned Rayon pool for cached dense rows (Q/out/FFN projections, decoder, GELU, QASS) | 48.44 -> 38.51 -> 34.57 ms | `affine_pool` in `src/executor.rs`, `parallel_gelu` |
| Native vector softmax/max, source-major row and column attention, RoPE changes | 350.72 -> 273.07 ms (`B=8`, `Q=128`) | `src/kernels/native.rs` |
| Bounded AVX-512 vector GELU (erf/exp polynomial) | 273.07 -> 247.21 ms | `gelu_avx512` in `src/kernels/native.rs` |
| 16-row ICL query blocks | 247.21 -> 231.72 ms | `small_attention_query_block_rows` in `src/executor.rs` |
| Const-specialised 16-row packed score kernel (accumulators stay in ZMM) | 231.72 -> 203.69 ms | `attention_score_packed_key_tile_avx512_rows::<16>` in `src/kernels/native.rs` |
| Persisted source-major BF16 ICL V with FP32 accumulation and const-specialised kernel | 203.69 -> 199.64 ms; context 1.79 -> 1.39 GB | `attention_score_packed_key_bf16_tile_avx512`, `kv_dtype="bf16-v"` |
| Precomputed context-only QASS base scales; input-major copy of the 3 -> 128 column weight; six-row Zen 5 affine block | Part of the 350 -> 273 ms step; a seven-row block regressed (357.57 vs 355.40 ms) | `build_qass_base_scale`, `affine_input3_transposed_avx512`, `affine_packed_*_avx512` |
| Flat estimator tasks with four bounded inner jobs at 16 logical CPUs | Beat five inner workers (350.95 vs 356.24 ms) and three at `Q=500` | `ExecutionPlan::automatic` in `src/executor.rs` |

## 5. Rejected experiments

| Experiment | Result (candidate vs control) | Reason for rejection |
|---|---:|---|
| One Rayon pool for column, row and head attention (`B=1`, no control recorded); persistent Rayon attention pool (`B=1`, `N=4096`) | 71.74 ms p50; 76.53 vs 70.94 ms | Slower than scoped head workers |
| Shared-pool hybrid attention tasks; flat scheduler with 5 inner workers; 3 inner workers at `Q=500` | 357.74 vs 355.40; 356.24 vs 350.95; 763.37 vs 752.07 ms | Every scheduling variant regressed |
| Layer-major schedule (`Q=500`); layer-major batched ICL dense stages | 779.32 vs 752.07; 236.05 vs 192.50 ms | Both scheduling and batching regress |
| Request microbatching (8 x `Q=16`) | 656.35 vs 658.31 rows/s | No throughput gain; the API is kept for a future scheduler |
| Approximate cached GELU and QASS | 63-67 ms vs 55.95 ms | Latency and accuracy both worse |
| Full request-local ICL scratch reuse; in-place GELU buffers; in-place attention residual; four dense workers; seven-row AVX-512 affine block | 39.51 vs 38.51; 351.12 vs 350.22; 572.80 vs 569.15; 39.27 vs 38.51; 357.57 vs 355.40 ms | No gain |
| Key tile 256; unspecialised / dynamic 16-row query tiles; lane-spilling native approximate GELU | 48.42 vs 48.44; 48.24 vs 48.44; 388.44 vs 373.85; 397.78 vs 373.85 ms | Below the 3% gate, or superseded by const-specialised / ZMM kernels |
| Packed FP32 V | 372.69 vs 373.85 ms, +403 MB | 0.3% gain is not worth the memory |
| Runtime BF16 V dot product (pack and convert per call) | 203.57 vs 192.50 ms | Conversion dominates; only the persisted layout works |
| Packed FP32 dense at `Q=500` | 885.84 vs 752.07 ms | Large regression |
| Fused LayerNorm with ICL Q and FFN1 projections | 194.02 vs 193.13 ms, error `1.22e-6` | Accurate but no latency gain |
| BF16 ICL FFN1; BF16 FFN1 + FFN2 | 188.96 ms, error `1.978e-3`; 182.90 ms, error `2.379e-3` | Probability gate (`1e-3`) failed |
| BF16 ICL FFN2 (with conversion workspace); BF16 FFN1 in blocks 6-11; BF16 decoder | 189.92 vs 193.13 (1.69%); 193.40 vs 193.13; 192.82 vs 192.50 ms | Below the 5% gate or no gain; FFN2 used most of the error budget (`9.744e-4`) |
| Full BF16 K/V at `N=100000` | 1,018 vs 1,005 ms, error `6.15e-3` | Smaller but slower and far less accurate |
| Fixed top-K attention | Median retained softmax mass 17.0% (top 64), 39.1% (top 256), 72.9% (top 1024) | Attention is diffuse; unsafe |
| K-means sparse index (M=16, M=64), omitted mass `1e-5` to `1e-1` | 195.00 vs 195.17 ms; 0 of 1,920 calls accepted | Neutral; disabled by default |
| Parallel benchmark processes on one host | Contaminated scaling data | Invalid evidence, not an optimisation |

## 6. Summary of findings

- **Correctness.** Logit error is below `1e-5` uncached and `3.05e-5` cached,
  probabilities are within `1.47e-5` on the fair raw-X fixture, and label
  agreement is 100% on every accepted configuration.
- **Cached CPU serving.** The performance goal was not met. Fair raw-X cached
  serving with FP32 K/V and equal thread counts is about 5% faster than
  PyTorch CPU at `Q=128` (349.98 vs 367.88 ms) and 24% faster at `Q=500`
  (1,322.27 vs 1,643.12 ms), and even those runs gave IronICL twice as many
  CPUs.
- **The 1.90x figure.** The headline ratio (193.13 vs 367.88 ms) used BF16
  value storage and unequal affinity (16 against 8 CPUs). It is a valid
  reduced-storage observation, not a general speedup, and should not be
  quoted as one.
- **Latency target.** The 200 ms per-request target is met only by the
  `Q=128` BF16-V lane; the FP32-exact lane is 350 ms, and `Q=500` is 752 ms
  or more.
- **Uncached path.** 6 to 16x slower than PyTorch CPU, and contexts of 2,048
  rows or more are rejected.
- **GPU.** A consumer GPU (RTX 5080, FP32) is 11 to 12.7x faster than the best
  IronICL CPU result. The documented conclusion is that microkernel tuning
  alone cannot close the large-context gap.
- **What IronICL delivers.** Lower memory (267 MiB vs 1.25 GiB RSS on
  MiniBooNE; 22.5% smaller contexts with BF16 V) and restart-safe
  memory-mapped contexts.

## 7. Reporting new results

Any new claim must come from `benchmarks/bench_cached_serving.py` with equal
`--threads`, FP32 `--kv-dtype`, `--check-parity`, and the date, commit and
host recorded with the numbers.

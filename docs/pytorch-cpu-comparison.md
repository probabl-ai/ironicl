# Ironicl vs. TabICL PyTorch CPU

This document summarizes how Ironicl's static-context CPU serving path differs
from the official TabICLv2 PyTorch implementation, what improved performance,
and which optimization experiments failed.

## Summary

Ironicl keeps the released TabICLv2 classifier checkpoint and implements the
same supported classifier graph in Rust. It changes the serving system around
that graph:

- training K/V state is exported once into a checked, memory-mapped artifact;
- cached attention and dense operations use explicit CPU kernels and layouts;
- work is scheduled across estimators and attention heads without Python or
  ATen dispatch between stages;
- an optional serving lane stores ICL values in BF16 while accumulating in
  FP32.

On the tested Ryzen 7 9800X3D workload, the fastest accepted `Q=128` path took
`193.13 ms` for 128 rows, or `662.68 rows/s`, versus `367.88 ms` and
`344.68 rows/s` for the recorded PyTorch CPU FP32 run. This is an observed
`1.90x` p50 speedup. The maximum probability difference was `2.21e-4`, with
identical predicted labels.

The historical PyTorch and Ironicl CPU runs used different process affinities:
PyTorch ran on CPUs 0-7, while tuned Ironicl ran on CPUs 0-15. The latency
ratios below are therefore observations from the recorded runs, not a controlled
claim that isolates the engine implementation. A publishable like-for-like
speedup requires rerunning both engines with the same affinity and SMT policy.

## Compared Workload

The reported measurements used:

| Property | Value |
|---|---:|
| CPU | AMD Ryzen 7 9800X3D |
| Dataset | Covertype |
| Estimators `B` | 8 |
| Context rows `N` | 4096 |
| Raw features | 54 |
| Prepared features `H` | 53 |
| Classes `C` | 7 |
| Query rows `Q` | 128 or 500 |
| Warmups / samples | 5 / 50 |
| Output | Host FP32 probabilities |

The measurements used the released classifier checkpoint with SHA-256:

```text
bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0
```

The `raw_x_cached_serving_v1` harness times:

- finite numeric fixture shape and value checks;
- the fitted official numerical encoder;
- the fitted official ensemble generator;
- resident cached inference;
- class-permutation correction and estimator averaging;
- temperature softmax and host FP32 output.

It excludes model loading, fitting, cache construction, artifact export, and
executor construction.

This is not the complete public `TabICLClassifier.predict_proba()` boundary.
The harness directly invokes the fitted encoder, ensemble generator, and
private cached forward method. It does not exercise the public API's DataFrame
handling, categorical input path, fitted-state checks, or all-NaN feature
masking and uncached fallback. The comparison applies to finite numeric arrays.

The benchmark comparator checks fixture, query, checkpoint, context, storage,
shape, and timing-boundary fields. It does not currently enforce CPU affinity,
runtime versions, thread count, or the Ironicl math mode.

## Measured Results

### FP32 Exact-Control Runs

Both engines used resident FP32 K/V. Ironicl enabled its exact GELU and softmax
controls.

| Q | Engine | Configured threads / affinity | p50 | p95 | Rows/s | Max probability difference |
|---:|---|---|---:|---:|---:|---:|
| 128 | PyTorch CPU FP32 | 16 / CPUs 0-7 | 367.88 ms | 382.45 ms | 344.68 | Baseline |
| 128 | Ironicl FP32 controls | 16 / CPUs 0-15 | 349.98 ms | 353.47 ms | 365.77 | `1.47e-5` |
| 500 | PyTorch CPU FP32 | 16 / CPUs 0-7 | 1,643.12 ms | 1,702.17 ms | 303.48 | Baseline |
| 500 | Ironicl FP32 controls | 16 / CPUs 0-15 | 1,322.27 ms | 1,333.37 ms | 378.02 | `1.47e-5` |

The observed p50 ratios were `1.051x` at `Q=128` and `1.243x` at `Q=500`.
They must be reported with the affinity caveat above.

"Exact controls" does not mean bitwise PyTorch equivalence. Ironicl still uses
different SIMD FMA, online-attention, and reduction orders. The result is
tolerance-equivalent for this fixture.

### Accepted Fast Lane

The accepted `Q=128` serving lane uses packed FP32 ICL K, source-major BF16 ICL
V, FP32 column K/V, FP32 accumulation, and bounded native GELU and softmax.

| Metric | PyTorch recorded FP32 run | Ironicl fast lane |
|---|---:|---:|
| p50 | 367.88 ms | 193.13 ms |
| Rows/s | 344.68 | 662.68 |
| Observed p50 ratio | Baseline | `1.90x` |
| Maximum probability difference | Baseline | `2.21e-4` |
| Label agreement | Baseline | 100% |
| Context size | About 1.79 GB | About 1.39 GB |

BF16 V saves about 384 MiB, or 22.5% of the full FP32 context. This is a
separately validated reduced-storage result, not an FP32-equivalence result.
The `1.90x` observation also carries the CPU-affinity caveat.

A comparable fast-lane `Q=500` raw-X run has not been recorded. The older
`752.07 ms` Ironicl result starts from a prepared query and must not be compared
with a public PyTorch `predict_proba()` timing.

## Main Implementation Differences

| Area | Official PyTorch CPU | Ironicl CPU |
|---|---|---|
| Interface | Public sklearn-compatible `fit()` and `predict_proba()` | Rust executor and C ABI consume prepared `[B,Q,H]` FP32 arrays |
| Supported scope | General released classifier configuration and input handling | Validated TabICLv2 static classifier path, with 2 through 10 classes |
| Cache lifecycle | `fit()` creates Python `TabICLCache` objects | Offline export creates a restart-safe static-context artifact |
| Cache storage | Torch tensors in their normal operator layouts | Checked tensor manifest with explicit dtype and packed layouts |
| Loading | Python/Torch allocations reconstruct model and cache state | Model and context payloads are checksummed and memory-mapped |
| Attention | `torch.nn.functional.scaled_dot_product_attention` | Tiled online attention with explicit score, softmax, value, and finalize kernels |
| ICL K layout | Determined by Torch tensor strides and operators | Source tiles are packed dimension-major so source rows become SIMD lanes |
| Dense operations | ATen linear, LayerNorm, GELU, and tensor operators | Portable and AVX-512 affine, LayerNorm, GELU, QASS, RoPE, and residual kernels |
| Scheduling | ATen intra-op scheduling plus Python estimator batching | Fixed estimator partitioning, scoped attention-head workers, and a persistent Rayon pool for selected dense work |
| Precision | CPU path normally retains FP32 tensors | Per-tensor FP32/BF16 storage policy with FP32 accumulation at sensitive boundaries |
| Aggregation | Public API supports configurable `average_logits` and softmax temperature | Current executor implements the benchmark defaults: logit averaging and temperature `0.9` |
| Query identity | Provenance follows the fitted classifier object | Artifact and `IRQUERY` execution validate identities; raw C-ABI arrays carry shape and finite-value checks but no query identity metadata |
| Hardware | PyTorch dispatch selects kernels for its build | Ironicl selects portable, NEON, or AVX-512 implementations; INT8/VNNI execution is not implemented |
| Device scope | CPU, CUDA, and MPS | CPU only |

## What Worked

The useful gains came from changing data movement, physical layout, and the
execution schedule rather than changing the model:

- a full resident K/V cache removed the training graph from every request;
- memory-mapped static context avoided rebuilding or copying the cache at
  restart;
- packed FP32 ICL K made score computation contiguous for SIMD;
- source-major BF16 ICL V reduced memory traffic while retaining FP32 weighted
  accumulation;
- 16-query by 128-source online-attention tiles bounded scratch space and
  avoided a full score matrix;
- const-specialized AVX-512 score and BF16-V kernels kept accumulators in
  vector registers;
- source-major column and row attention improved traversal;
- scoped attention-head workers beat routing attention through Rayon;
- a persistent Rayon pool helped selected projections, FFNs, decoder work,
  GELU chunks, and QASS rows;
- precomputed context-only QASS scales and a transposed 3-to-128 column input
  weight removed repeated or poorly laid-out work.

On an earlier prepared-query optimization sequence, representative p50 latency
fell from `350.72 ms` to `273.07 ms` after native attention/softmax/RoPE work,
then to `247.21 ms` with vector GELU, `231.72 ms` with 16-row query blocks,
`203.69 ms` with a specialized score kernel, and `199.64 ms` with the retained
BF16-V kernel. These are internal prepared-query measurements, not cross-engine
end-to-end speedups.

Python preprocessing is small for this fixture. A current isolated run measured
`1.74 ms` preprocessing and `192.68 ms` resident inference within a `194.33 ms`
request.

## What Failed

### Scheduling and Batching

- A persistent Rayon attention pool took `76.53 ms` against `70.94 ms` for
  scoped head workers.
- A shared-pool hybrid schedule took `357.74 ms` against `355.40 ms`.
- A flat scheduler with five inner workers took `356.24 ms` against `350.95 ms`.
- Layer-major scheduling at `Q=500` took `779.32 ms` against `752.07 ms`.
- Layer-major batched dense stages took `236.05 ms` against `192.50 ms`.
- Request microbatching did not improve throughput: isolated `Q=128` reached
  `658.31 rows/s`, while batched `8 x Q=16` reached `656.35 rows/s`.

The microbatch API remains useful serving infrastructure, but it is not a
performance optimization by itself.

### Attention, Layout, and Sparsity

- A 256-row key tile was effectively neutral.
- Packed FP32 V improved latency by only 0.3% and added about 403 MB.
- An earlier BF16-V implementation that packed and converted values at runtime
  took `203.57 ms` against `192.50 ms`; BF16 V only became useful after changing
  the persisted layout and kernel together.
- Fixed top-K attention failed because attention was diffuse. Median retained
  softmax mass was only 17.0% for top 64, 39.1% for top 256, and 72.9% for top
  1024 source rows.
- The K-means sparse index accepted zero complete sparse calls at omitted-mass
  thresholds from `1e-5` through `1e-1`. Its best M=64 result, `195.00 ms`, was
  effectively tied with the `195.17 ms` indexed dense control.

Sparse work is disabled by default. A different retrieval method must first
show that it can safely skip meaningful work before a cluster-specific K/V
layout is justified.

### Dense Fusion and Reduced Precision

- Fusing LayerNorm with ICL Q and FFN1 projections took `194.02 ms` against
  `193.13 ms`; accuracy passed, but latency did not improve.
- A packed FP32 dense path at `Q=500` took `885.84 ms` against `752.07 ms`.
- BF16 ICL FFN2 reached `189.92 ms` against `193.13 ms`, only a 1.69% gain, and
  used nearly the full probability-error budget (`9.744e-4`).
- BF16 ICL FFN1 was faster but exceeded the error gate (`1.978e-3`).
- BF16 ICL FFN1 plus FFN2 reached `182.90 ms` but increased error to
  `2.379e-3`.
- BF16 decoder storage was neutral (`192.82 ms` against `192.50 ms`).
- Full BF16 K/V at `N=100000` was smaller but slower and raised maximum
  probability error to `6.15e-3`.

There is no implemented or accepted INT8/VNNI serving path. Existing dtype and
capability declarations are scaffolding only.

## Takeaways

For this workload, the strongest CPU improvements came from persistent cache
state, cache layout, attention traversal, and specialized kernels. More generic
parallelism, request packing, isolated fusion, and broader BF16 conversion did
not help enough or consumed too much accuracy margin.

The current evidence supports these narrow conclusions:

- the accepted Ironicl fast lane serves the tested finite-numeric Covertype
  fixture at `662.68 rows/s` with `2.21e-4` maximum probability difference and
  identical labels;
- BF16 ICL V saves about 384 MiB without changing the checkpoint;
- current microbatch and sparse-index paths provide no throughput gain;
- no general CPU speedup claim should be made until both engines are rerun with
  identical CPU affinity and SMT policy;
- results from the Ryzen 7 9800X3D do not predict Threadripper or EPYC behavior
  without separate NUMA-aware measurements.

## Sources

- Benchmark contract: `tools/benchmark_cached_end_to_end.py`
- Result comparator: `tools/compare_cached_end_to_end.py`
- Ironicl executor: `src/executor.rs`
- Static context: `src/static_context.rs`
- CPU kernels: `src/kernels/native.rs` and `src/kernels/reference.rs`
- Official classifier: `../tabicl/src/tabicl/_sklearn/classifier.py`
- Official preprocessing: `../tabicl/src/tabicl/_sklearn/preprocessing.py`
- Official attention: `../tabicl/src/tabicl/_model/attention.py`
- Full experiment log: [`cached-fp32-optimization.md`](cached-fp32-optimization.md)
- Benchmark tables: [`real-dataset-benchmarks.md`](real-dataset-benchmarks.md)

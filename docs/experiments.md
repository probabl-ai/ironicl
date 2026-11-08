# Experiments and Tests Log

This is the authoritative record of what Ironicl tried, what it kept, and what
it rejected, together with the tests and gates that verify each retained claim.
It covers two backends: the Ryzen 7 9800X3D CPU path and the Apple Silicon
(Accelerate, NEON, Metal GPU) path. For the project's goals and non-goals, see
[goals.md](goals.md); this log records evidence, not scope.

Companion documents: [pytorch-cpu-comparison.md](pytorch-cpu-comparison.md) is
the cross-engine CPU comparison, and [real-dataset-benchmarks.md](real-dataset-benchmarks.md)
holds the raw scaling measurements this log cites. Do not restate their numbers
here; link to them.

## Scope

The CPU sections below record the accepted query-only full-K/V path for the
Ryzen 7 9800X3D. They separate retained changes from experiments that failed the
latency or accuracy gates. The [Apple Silicon experiments](#apple-silicon-experiments)
section records the Accelerate, NEON, and Metal GPU work and its own rejected
experiments. [Acceptance gates](#acceptance-gates) lists every retention and
accuracy gate, including the FP16 gate for the `optimize` levels, and
[Claim to check](#claim-to-check) maps each retained claim to the test or
benchmark that backs it.

The current CPU reference workload is:

```text
B=8, N=4096, Q=128 or 500, H=53, C=7
dataset: Covertype
cache: packed FP32 ICL K, BF16 ICL V, FP32 column K/V
CPU: Ryzen 7 9800X3D, eight physical cores, 16 logical CPUs
build: release, -C target-cpu=znver5
```

The earlier `B=1,N=1024,Q=64` result remains below as a historical
single-estimator calibration.

## Historical Single-Estimator Result

The final uncontended benchmark used five warmups and 50 timed iterations:

| Metric | Result |
|---|---:|
| Minimum | 31.49 ms |
| Median | 34.57 ms |
| p95 | 36.30 ms |
| p99 | 36.50 ms |
| Maximum | 36.50 ms |
| Mean | 34.51 ms |
| Query rows per second | 1,854.62 |

The previous accepted implementation had a 55.95 ms median. The current path
is 38.2% faster. The cached PyTorch CPU reference is 92 ms and the CUDA FP32
reference is 5.84 ms.

The maximum absolute difference from official cached PyTorch logits is
`3.05e-5`. The cached serving gate is `1e-4`. A small `B=2,N=16,Q=2,H=3`
fixture has `4.77e-6` maximum absolute error.

## Current Result

All current results use final `predict_proba` output and 16 logical workers.

| Q | Timed samples | p50 | p95 | Query rows/s | Max probability error | Label agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 50 | 199.64 ms | 202.39 ms | 642.76 | `2.458e-4` | 100% |
| 500 | 30 | 752.07 ms | 756.22 ms | 665.14 | `2.575e-4` | 100% |

The earlier public PyTorch `Q=128` result has a different query boundary and
must not be used for a cross-engine speedup claim. The `Q=500` result shows
nearly linear prepared-query scaling, but it does not meet the 200 ms latency
target. Strict prepared-query PyTorch parity remains outstanding.

## Retained Execution

### Attention

- Select `Portable`, `NEON`, or `AVX-512` once when the static executor is
  constructed.
- Keep packed FP32 K and source-major BF16 V in head-major storage.
- Use a `16 x 128` query/source online-attention tile for ICL.
- Compute one complete row-major score tile with AVX-512.
- Convert scores to stable online-softmax weights in FP32.
- Use const-specialized 16-row AVX-512 score and BF16-V kernels. This lets the
  compiler retain accumulators in ZMM registers and reuses each V conversion.
- Use source-major AVX-512 attention for column and row stages, with portable
  query-major fallbacks.
- Keep independent ICL heads on scoped workers. This was faster than Rayon for
  attention on the target host.
- Keep scalar and portable fallbacks for unsupported hosts.

The fused tile changes dot-product reduction order but preserves source order
for each weighted-V accumulator. Therefore, parity is tolerance-based rather
than bitwise.

### Dense Stages

One executor-owned Rayon pool persists for the executor lifetime. It handles
only cached dense row work:

- ICL Q projection;
- attention output projection;
- both ICL feed-forward projections;
- decoder projections;
- native GELU chunks;
- exact QASS query MLP rows.

The Zen 5 affine kernel reuses each weight block across six input rows. The
column `3 -> 128` projection uses a construction-time input-major weight copy so
AVX-512 lanes cover output channels. Static ICL QASS base scales are computed
once when the executor is constructed because they depend only on context
length and layer weights.

The pool does not run column feature streams, row interaction, or head
attention. This avoids nested scheduling and retains the faster scoped-worker
paths for those stages.

### Precision Policy

- LayerNorm remains FP32.
- AVX-512 GELU uses a bounded vector erf/exp approximation. Build with
  `--features exact-math` to restore the scalar `libm::erff` path.
- The cached executor keeps reference QASS GELU and `tanh` arithmetic.
- AVX-512 online softmax uses the native bounded exponential path. The
  `exact-math` feature restores the scalar control path.
- Online-softmax maximum, denominator, and weighted output remain FP32.
- Decoder and residual operations remain FP32.

## Performance Progression

| Implementation | Threads | Median |
|---|---:|---:|
| Cached scalar path | 1 | 338 ms |
| Parallel column and row work before ICL parallelization | 8 | 364 ms |
| Head-parallel ICL | 8 | 111 ms |
| AVX-512 score dot and V row kernels | 8 | 55.95 ms |
| Fused AVX-512 score and V tiles | 8 | 48.44 ms |
| Persistent parallel cached dense stages | 8 | 38.51 ms |
| Parallel exact GELU | 8 | 36.33 ms |
| Parallel exact QASS | 8 | 34.57 ms |

These measurements were taken during iterative development. Only the final
result is the current performance contract.

### Estimator-Parallel Calibration

The `B=8,N=4096,Q=128` packed-FP32-K/BF16-V workload uses flat estimator tasks
with four bounded inner jobs when 16 logical CPUs are available. The path also
precomputes context-dependent QASS base scales and transposes the shared
`3 -> 128` column input weight for output-channel SIMD.

Both engines used CPUs 0-7 for the PyTorch physical-core baseline or CPUs 0-15
for the tuned Ironicl SMT path, five warmups, 50 timed iterations, and final
probability output:

| Engine | p50 | p95 |
|---|---:|---:|
| Ironicl prepared-query packed FP32 K/BF16 V | 199.64 ms | 202.39 ms |

The retained `Q=128` progression on the same host was:

| Change | p50 |
|---|---:|
| Paired pre-optimization default | 350.72 ms |
| Native softmax/max, source-major row/column attention, and RoPE changes | 273.07 ms |
| Vector GELU | 247.21 ms |
| 16-row ICL query blocks | 231.72 ms |
| Const-specialized packed score kernel | 203.69 ms |
| Const-specialized BF16-V kernel | 199.64 ms |

The final probability maximum absolute error is `2.458e-4`, with 100% label
agreement. These are prepared-query kernel results, not end-to-end serving
speedups.

## Rejected Experiments

The following experiments were removed because they did not meet the retention
gate:

| Experiment | Observed result | Decision |
|---|---:|---|
| One Rayon pool for column, row, and head attention | 71.74 ms p50 | Removed |
| Approximate cached GELU and QASS | 63-67 ms p50 | Removed |
| Full request-local ICL scratch reuse | 39.51 ms versus 38.51 ms | Removed |
| Four dense workers | 39.27 ms versus 38.51 ms | Removed |
| Unspecialized query tile 16 on the older path | 48.24 ms versus 48.44 ms | Superseded by the source-major const-specialized kernel |
| Key tile 256 | 48.42 ms versus 48.44 ms | Removed; below 3% gate |
| Persistent Rayon attention pool, `B=1,N=4096,Q=128` | 76.53 ms versus 70.94 ms | Removed |
| In-place attention residual, `B=8,N=4096,Q=128` | 572.80 ms versus 569.15 ms | Removed |
| Packed FP32 V at `B=8,N=4096,Q=128` | 372.69 ms versus 373.85 ms, 403 MB larger | Removed; insufficient gain |
| Dynamic native 16-row attention block | 388.44 ms versus 373.85 ms | Superseded by const specialization |
| Lane-spilling native approximate GELU | 397.78 ms versus 373.85 ms | Superseded by the ZMM vector polynomial |
| Shared-pool hybrid attention tasks | 357.74 ms versus 355.40 ms | Removed |
| In-place GELU buffers | 351.12 ms versus 350.22 ms | Removed |
| Seven-row AVX-512 affine block | 357.57 ms versus 355.40 ms | Removed |
| Flat scheduler inner budget 5 | 356.24 ms versus 350.95 ms | Removed |
| Layer-major `Q=500` schedule | 779.32 ms versus 752.07 ms | Rejected |
| Three inner workers at `Q=500` | 763.37 ms versus 752.07 ms | Rejected |
| Packed FP32 dense at `Q=500` | 885.84 ms versus 752.07 ms | Rejected |
| Fused LayerNorm with ICL Q and FFN1 projections, fair raw-X `Q=128` | 194.02 ms versus 193.13 ms; `1.22e-6` max probability error, 100% labels | Removed; no latency gain |
| BF16 V dot product, fair raw-X `Q=128` | 203.57 ms versus 192.50 ms; exact final output | Removed; runtime packing and conversion dominate |
| Layer-major batched ICL dense stages, fair raw-X `Q=128` | 236.05 ms versus 192.50 ms; exact final output | Removed; layer-major scheduling and batching both regress |
| BF16 ICL FFN1, fair raw-X `Q=128` | 188.96 ms; `1.978e-3` max probability error, 100% labels | Rejected; probability gate failed |
| BF16 ICL FFN1 and FFN2, fair raw-X `Q=128` | 182.90 ms; `2.379e-3` max probability error, 100% labels | Rejected; probability gate failed |
| BF16 ICL FFN2 with reusable conversion workspace, fair raw-X `Q=128` | 189.92 ms versus 193.13 ms; `9.744e-4` max probability error, 100% labels | Rejected; 1.69% gain is below the 5% retention gate |
| BF16 ICL FFN1 in blocks 6-11, fair raw-X `Q=128` | 193.40 ms versus 193.13 ms; `5.589e-4` max probability error, 100% labels | Rejected; no latency gain |
| BF16 decoder, fair raw-X `Q=128` | 192.82 ms versus 192.50 ms; `1.701e-4` max probability error, 100% labels | Rejected; no latency gain |

Do not reintroduce these configurations without a new workload, a changed
kernel, or stronger measurements. Parallel benchmark processes on the same
eight-core host produced contaminated scaling data and are not valid evidence.

## Raw-X Phase Split and Sparse Oracle

On the fair `B=8,N=4096,Q=128` fast BF16-V path, 20 samples measured a
`192.50 ms` p50 request. TabICL raw-X validation, numerical encoding, and
ensemble generation used `1.66 ms` p50. Resident Ironicl inference used
`190.82 ms` p50. Python preprocessing is not a material part of the remaining
latency gap.

An exact-score oracle sampled 384 ICL attention calls. The median softmax mass
in top-64, top-256, and top-1024 source rows was 17.0%, 39.1%, and 72.9%,
respectively. Only 24 calls retained at least 99.9% mass in top-1024. A fixed
top-K attention policy is therefore not eligible. Any sparse path needs an
indexed candidate search and a dense fallback for diffuse heads.

## Throughput Serving Foundation

`StaticExecutor` now has a microbatch API and matching C ABI. It
accepts equal prepared `[B,Q,H]` requests, packs them ensemble-major, executes
one cache-sharing forward pass, and returns request-ordered `[Q,C]` outputs.
The API is the boundary for a future bounded native request scheduler; it does
not itself increase dense-kernel throughput.

The raw-X benchmark can model collected requests with `--request-rows`. Its
default `--stream-preprocess-mode batched` validates and transforms the
collected raw rows once, then sends equal prepared requests through the batch
ABI. `per-request` is retained only to measure preprocessing overhead.

On an equal 50-sample Ryzen run with the fair `B=8,N=4096,Q=128` fast BF16-V
context:

| Mode | p50 | Inference p50 | Preprocessing p50 | Rows/s |
|---|---:|---:|---:|---:|
| Isolated Q=128 control | 194.33 ms | 192.68 ms | 1.74 ms | 658.31 |
| Stream, 8 x Q=16, batched preprocessing | 195.03 ms | 193.29 ms | 1.71 ms | 656.35 |
| Stream, 8 x Q=16, per-request preprocessing | 199.27 ms | 190.12 ms | 9.26 ms | 643.34 |

Both stream modes produced output SHA-256
`2303a8f3ce69058fad88532c68a2fc3bdb285e1742fab47275cd93b3e1078281`, equal
to the isolated control. Batched preprocessing removes the artificial 7.5 ms
Python overhead. The remaining 0.70 ms stream cost is query packing and output
splitting, and is below the retention threshold. Keep this path for the native
scheduler rather than claiming it as an optimization.

### Sparse Index Quality

The v3 sparse index remains opt-in through the `sparse-attention` cargo feature
(build with `--features sparse-attention`; it is never enabled at runtime). It
uses exact candidate scores and a maximum omitted-mass certificate of `1e-5`;
it falls back to dense attention when the certificate fails. The checkpoint and
dense K/V tensors remain unchanged.

Increasing the deterministic per-head K-means index from 16 to 64 clusters
increased the artifact from 1,403,102,496 to 1,412,834,592 bytes. Before the
temporary output removal, M=64 sparse had `197.85 ms` p50 versus `195.17 ms`
for indexed dense. Sparse attention now writes directly to its caller output;
dense fallback overwrites partial output if a later head rejects sparse
execution. This removed the temporary allocation and copy: M=64 sparse now has
`195.00 ms` p50 with the same final output hash. It is neutral, not an accepted
speedup, so do not enable it for serving yet.

Executor-owned telemetry, reset after warmup, measured 1,920 M=64 sparse ICL
calls over 20 timed requests: 0 accepted and 1,920 dense fallbacks under the
default `1e-5` omitted-mass certificate. Candidate K/V layout is therefore not
the current blocker. Raising the omitted-mass certificate (then a runtime knob,
now the `1e-5` constant under the feature) to `1e-3`, `1e-2`, and `1e-1` also
accepted zero calls. The `1e-1` 50-sample run had
`196.35 ms` p50 versus `195.17 ms` indexed dense, with bit-identical output.
Threshold tuning is closed for this index. The next sparse work is stronger
candidate bounds or a different retrieval index, not v4 cluster-contiguous
K/V storage. The threshold is recorded in the benchmark contract. Any future
threshold must meet the final `1e-3` probability and 100% label-agreement gates
against the indexed dense control before it is eligible for serving.

Keep M=16 and M=64 artifacts as quality controls. Do not build a
cluster-contiguous K/V variant until a stronger index accepts enough sparse
calls to make candidate execution measurable.

## Large Context Result

The large-context workload uses `B=1,N=100000,Q=128,H=54,C=7`. The matching
RTX 5080 AMP result is `10.04 ms` p50. The GPU cannot retain a full `B=8` K/V
cache at 50,000 rows or more in 16 GiB.

| Context policy | Artifact size | CPU p50 | Max logit error | Max probability error | Label agreement |
|---|---:|---:|---:|---:|---:|
| Packed FP32 K, BF16 V | 3.71 GB | 1,005 ms | `4.43e-3` | `7.13e-4` | 100% |
| Packed BF16 K/V | 2.48 GB | 1,018 ms | `4.46e-2` | `6.15e-3` | 100% |

The hybrid policy is the current large-context choice. Full BF16 reduces storage
but does not improve latency and has substantially higher error.

An opt-in profile (the `attention-stats` cargo feature) sampled the first query from all
96 ICL head/layer calls. Top-4096 mass ranged from 4.1% to 100%, with a 62.0%
median. Top-16384 mass ranged from 16.4% to 100%, with an 89.2% median. A fixed
small top-K policy is therefore not safe. Some heads are close to uniform and
need a dense fallback or a different approximation.

## Apple Silicon experiments

These cover M1 Max and later. The CPU path adds Accelerate and NEON kernels; the
GPU path runs the model with Metal. All share unified memory, so a context needs
no copy to reach the coprocessor or GPU. The GPU path is behind the `metal` cargo
feature and is disabled by `IRONICL_NO_METAL=1`.

### Accelerate and NEON kernels (retained)

- The affine kernel dispatches to custom NEON on AArch64. Affine layers with at
  least 16 rows and attention with at least 4096 scores per head call Accelerate's
  `cblas_sgemm` instead, which runs on the matrix coprocessor.
- NEON LayerNorm, NEON attention for short sources, and a vDSP-vectorized softmax.
- macOS schedules all estimators in parallel when threads outnumber them, and
  prefers performance cores for executor threads.

Rejected: batching the row stage across rows measured slower on an M1 Max, so it
stays per row.

### Metal GPU build and predict graphs (retained)

Each context gets two `MPSGraph` graphs with its ensemble, feature, and
training-row counts fixed.

- The predict graph runs the column, row, and ICL stages and the decoder for
  every estimator in one run, with only the query count unknown. One batched run
  beat the CPU on an M1 Max from a single query row, so prediction always uses it.
- The build graph runs the training rows through every stage once and captures
  each column and ICL block's keys and values. Its setup costs about 250 ms, so
  contexts with fewer than 160 training rows build on the CPU.
- The build graph runs as many estimators at once as fit half the host's
  available memory and Metal's recommended working set (about 2.2 GB per
  estimator at 8192 x 54), in equal groups; `IRONICL_MEMORY_LIMIT` lowers the
  memory planned for. A single estimator that exceeds the working set builds on
  the CPU.
- Model weights live once per model in shared GPU buffers and are fed as graph
  placeholders. A GPU-built context writes its keys and values once into
  page-aligned memory that the CPU reads as tensors and the GPU wraps without
  copying; a CPU-built context is copied to the GPU once.
- Graphs compile at `MPSGraph` optimization level 0.

### optimize levels and FP16 caches (retained)

`optimize=safe|auto|fast` (or `IRONICL_OPTIMIZE`) selects the precision ladder.
Under `auto` or `fast`, the build graph stores the ICL keys and values in FP16,
and both graphs run ICL attention in FP16, converting only the queries. The CPU
fallback reads the same FP16 caches (a query row whose grouped features are all
the missing marker runs on the CPU). The accuracy gate is in
[Acceptance gates](#acceptance-gates); measured on an M1 Max across 12 datasets,
`auto` stayed within `4.9e-4` max and `2.3e-5` mean probability error of `safe`.

### Query bucketing and chunking (retained)

- Query counts are padded to a bucket that keeps their top four significant bits,
  at most 12.5% padding, so new batch sizes rarely trigger a 150-400 ms graph
  compile. Buckets keep their buffers between calls; queries split so the
  kept buckets fit the [memory budget](memory-planner.md#prediction).
- Large cached queries are split into chunks; each estimator's rows land in place
  when the chunks are joined.

### Metal rejected experiments

| Experiment | Observed result | Decision |
|---|---|---|
| `MPSGraph` optimization level 1 | No faster; its pass that places operations on the Neural Engine or CPU printed "Failed to import MPS module" on every compile | Rejected; stay at level 0 |
| Convert stored FP16 caches to FP32 inside the graph | Wrote a full FP32 copy on every run and was slower | Rejected |
| BF16 ICL keys and values on the GPU | Slower and less accurate than FP16 on an M1 Max | Rejected; FP16 is the reduced-precision choice |
| FP16 column and row attention | Added error without saving time | Rejected; FP16 limited to ICL attention |
| Neural Engine (Core ML) placement | The only path to it was the level-1 NE placement pass, which failed to import | Not implemented; a Core ML lane remains a goals.md experiment, not shipped code |

These Metal results come from manual M1 Max measurements in code comments and
`kernels.md`; no tracked benchmark harness exercises the Metal path the way the
`tools/` scripts exercise the Ryzen path. The Rust tests gate Metal *correctness*
(`tests/gpu_build_groups.rs`, `tests/checkpoint_fixtures.rs`) but emit no latency
numbers. Treat the Metal latencies above as indicative, not a contract. Closing
this gap is open work.

## Acceptance Gates

Retain an optimization only when all conditions hold:

- at least 5% p50 improvement for invasive changes;
- at least 3% p50 improvement for tile-only tuning;
- no more than 5% p95 regression;
- maximum absolute cached-logit error at or below `1e-4` for exact-preserving
  work, or final probability error at or below `1e-3` for explicitly approved
  reduced-precision or native-approximation work;
- 100% final label agreement for reduced-precision or native-approximation
  work;
- unchanged output shape and finite logits;
- no regression in the official uncached PyTorch oracle;
- all-target tests, strict Clippy, and Linux target checks pass.

Evaluate closely coupled score and value specializations as one attention
microkernel change.

### FP16 gate for the optimize levels

The Apple Silicon `optimize` levels store reduced-precision caches, so they have
their own accuracy gate against the `safe` (FP32) output, not the `1e-4` PyTorch
tolerance. A reduced-precision level is retained only when, measured against the
`safe` result on the same context:

- `auto` stays within `1e-2` maximum absolute logit error (`1e-3` maximum
  probability error) and `fast` within `5e-2`;
- labels flip only on near-ties;
- the GPU rejects any non-finite or overflowing value (FP16 saturates above
  65504, which the gate must cover for attention scores and activations).

These bounds are enforced in `tests/checkpoint_fixtures.rs`
(`built_context_matches_the_uncached_forward_pass`) and
`tests/parity/test_tabicl_classifier.py`
(`test_optimize_levels_stay_within_their_bounds`), and the probability-error
budget is checked in `python/ironicl/classifier.py`.

Run latency comparisons sequentially on an otherwise idle host. Do not run
thread-count variants in parallel.

## Benchmark Commands

Build on the Ryzen host:

```bash
RUSTFLAGS="-C target-cpu=znver5" cargo build \
  --manifest-path $BENCH_ROOT/src/Ironicl/Cargo.toml \
  --release
```

Run the exact target:

```bash
IRONICL_THREADS=16 \
$BENCH_ROOT/src/Ironicl/target/release/ironicl benchmark \
  --model $BENCH_ROOT/fixtures/tabicl-classifier-v2.irmodel \
  --context $BENCH_ROOT/data/covertype-h54-n4096-b8-packed-bf16v.irctx \
  --input $BENCH_ROOT/data/covertype-h54-n4096-b8-packed-bf16v-q128.irquery \
  --probabilities --warmup 5 --iterations 50
```

Compare final probabilities:

```bash
IRONICL_THREADS=16 \
$BENCH_ROOT/src/Ironicl/target/release/ironicl run \
  --model $BENCH_ROOT/fixtures/tabicl-classifier-v2.irmodel \
  --context $BENCH_ROOT/data/covertype-h54-n4096-b8-packed-bf16v.irctx \
  --input $BENCH_ROOT/data/covertype-h54-n4096-b8-packed-bf16v-q128.irquery \
  --probabilities > rust-probabilities.txt

python tools/compare_cached_logits.py \
  --oracle covertype-h54-n4096-b8-packed-bf16v-q128.irlogits \
  --probabilities-actual rust-probabilities.txt \
  --context covertype-h54-n4096-b8-packed-bf16v.irctx \
  --atol 1e-3
```

## Claim to check

Each retained claim maps to a test or benchmark that would fail if it broke. The
fixture-gated Rust tests run with `cargo test -- --ignored` once
`IRONICL_CHECKPOINT_FIXTURES` and `IRONICL_ORACLE_FIXTURES` point at the official
checkpoints (see [validation.md](validation.md)); the parity tests need the
separate upstream environment.

| Claim | Backing check |
|---|---|
| Cached FP32 logits match the PyTorch oracle within `1e-4` | `checkpoint_fixtures.rs::matches_python_classifier_oracle_when_fixtures_are_available`; oracle built by `tools/export_tabicl_oracle.py` |
| `optimize=auto`/`fast` FP16 stays within its gate, labels flip only on near-ties | `checkpoint_fixtures.rs::built_context_matches_the_uncached_forward_pass`; `test_tabicl_classifier.py::test_optimize_levels_stay_within_their_bounds` |
| Grouped/budgeted GPU build matches a single-graph build | `gpu_build_groups.rs::grouped_gpu_build_matches_one_graph_run` (passes a 40 MiB memory limit) |
| Large cached queries chunk to identical logits | `checkpoint_fixtures.rs::large_cached_query_runs_in_chunks` |
| All-missing query rows fall back to CPU and still read the FP16 `auto` caches | `checkpoint_fixtures.rs::reduced_precision_context_predicts_all_missing_rows` |
| Public `predict_proba` matches upstream TabICL within `atol=1e-4`, identical labels | `test_tabicl_classifier.py::test_predictions_match` (live parity CI job) |
| Preprocessing matches upstream (frozen oracle and live) | `tests/test_preprocessing.py` (15 tests) and `tests/parity/test_tabicl_preprocessing.py` (11 tests) |
| Accepted fast lane serves Covertype at the recorded rate and error | `tools/benchmark_cached_end_to_end.py` + `tools/compare_cached_end_to_end.py`; see [Benchmark commands](#benchmark-commands) |

## Next Work

Prioritize these changes for `Q=500` and larger batches:

1. Evaluate INT8 or other lower-precision dense and attention kernels only with
   the final `1e-3` probability and 100% label gates.
2. Fuse attention output projection and feed-forward projection with their
   residual additions.
3. Build strict prepared-query PyTorch parity for kernel-only comparisons.

Prioritize these changes for larger contexts:

1. Test an adaptive approximation that keeps dense handling for diffuse heads.
2. Evaluate compressed K/V storage with a measured dense fallback.
3. Build `B=8,N=100000` artifacts with bounded exporter memory.
4. Add explicit context prefaulting and report page-fault behavior.

# Implementation Roadmap

## Phase 0: Executable Skeleton

Status: complete.

- portable Rust crate;
- memory planner and CLI;
- pure-Rust checkpoint inspection;
- runtime CPU capability model;
- artifact descriptor contract;
- scalar dot-product and softmax references;
- reference design documents.

Exit gate: formatting, tests, and lint pass on M1 Max.

## Phase 1: Checkpoint Import and Oracle Fixtures

- maintain compatibility tests for Candle's parser against both released
  checkpoints;
- import checkpoint tensors and primitive model configuration in pure Rust;
- write and verify the portable Ironicl model artifact;
- create or reuse content-addressed model imports automatically on checkpoint
  runs;
- export deterministic preprocessing and layer fixtures;
- record checkpoint and fixture hashes;
- add importer shape and count tests.

Python is used only to produce comparison fixtures from the official model. It
is not part of checkpoint import, artifact creation, or runtime execution.

Exit gate: all classifier tensors and configuration import exactly. Status:
complete for the official classifier, including exact tensor manifests. The
Python oracle fixtures were superseded by the live parity suite and removed
([removed-experiments.md](removed-experiments.md#external-pytorch-oracle-test)).

## Phase 2: Portable FP32 Model

Status: baseline complete.

- tensor views and aligned storage;
- linear, LayerNorm, GELU, QASSMax, RoPE, and attention;
- column, row, ICL, and decoder stages;
- uncached classifier inference;
- full layer-level PyTorch comparisons.

Exit gate: approved FP32 stage and end-to-end parity. The current official
classifier fixture passes at less than `1e-5` maximum absolute logit error.

## Phase 3: Static Full-KV Context

Status: executable full-KV path and native preprocessing complete;
representation-cache execution and zero-allocation serving remain.

Completed:

- Python full-KV context builder and preprocessed query exporter: superseded by
  the Rust context builder and removed
  ([removed-experiments.md](removed-experiments.md#python-static-context-exporter-irquery-irlogits-and-irtrace));
- Rust context builder on the CPU, Metal, and CUDA, with a content-addressed
  context cache;
- full-KV artifact reader/writer and whole-file checksum;
- cached inference for one query row or many;
- online stable attention;
- mmap-backed static-context loading;
- native preprocessing and a fitted-model format
  (see [native-preprocessing.md](native-preprocessing.md)).

Remaining:

- representation-cache execution;
- no steady-state allocations.

Exit gate: cached parity and restart-safe artifacts.

## Phase 4: 9800X3D FP32 Backend

Status: active. Runtime-gated AVX-512 affine and fused attention-tile kernels are
validated on the target host. The exact cached target reaches 34.57 ms median
for `B=1,N=1024,Q=64,H=54`, with `3.05e-5` maximum absolute logit error.

- run CPUID probe on the Arch machine;
- establish memory, L3, and kernel baselines;
- fuse LayerNorm/projection and projection/residual boundaries;
- add affinity and warmup controls;
- benchmark physical cores and SMT.

Exit gate: optimized FP32 is faster than portable FP32 without accuracy change.

## Phase 4b: Apple Silicon Backend

Status: shipped. Apple Silicon has its own performance backend, not just the
portable correctness path.

- Accelerate `cblas_sgemm` for dense layers and attention on the matrix
  coprocessor, with NEON LayerNorm and short-source attention kernels;
- a Metal GPU path (behind the `metal` feature) that builds static contexts and
  runs the full predict graph per estimator in one run, with query bucketing,
  large-query chunking, and memory-bounded estimator grouping;
- the `optimize=safe|auto|fast` levels, with FP16 ICL caches and FP16 attention
  under `auto` and `fast` behind the FP16 accuracy gate;
- performance-core executor threads and parallel estimators on macOS.

Exit gate: GPU and reduced-precision paths match the portable FP32 output within
their gates. See [experiments.md](experiments.md#apple-silicon-experiments).

## Phase 5: BF16 Storage

Status: partially shipped. Under `optimize="fast"`, a BF16-capable CPU stores
ICL values in BF16 with FP32 accumulation. The packed BF16 affine experiment failed its gates and is
removed (see [removed-experiments.md](removed-experiments.md#packed-fp32-and-bf16-dense-layers-bf16-dense));
the full accuracy suite, artifact variants, and migration policy remain.

- BF16 weight conversion and packed kernels;
- BF16 column and ICL cache storage;
- FP32 accumulation and sensitive-operation boundaries;
- full accuracy suite;
- artifact variants and migration policy.

Exit gate: measured speed or memory benefit with accepted accuracy.

## Phase 6: INT8 Experiments

- per-channel INT8 weights;
- BF16 K plus INT8 V;
- INT8 K experiments;
- quantization calibration reports;
- VNNI kernels and metadata accounting.

Exit gate: no default change without dataset-level accuracy approval.

## Phase 7: Regression and Packaging

Status: regression and the Python API shipped; packaging remains.

- 999-quantile decoder and postprocessing;
- regressor parity and accuracy suite;
- Python API bridge: delivered as the sklearn-compatible `TabICLClassifier` and
  `TabICLRegressor` over the PyO3 bindings;
- reproducible release artifacts;
- deployment and operations documentation.

The next three phases run in this order.

## Phase 8: Profiling

- a cross-engine benchmark that starts every engine from the same raw rows:
  Ironicl, TabICL, and Kumo-Tabular as baselines;
- per-phase timings from the `profile` feature through `_core.profile_report()`
  ([benchmarking.md](benchmarking.md#phase-timings));
- fit and predict latency, throughput, and peak memory per device and level.

Exit gate: one harness reports every engine on the same datasets and hosts.

## Phase 9: Raw-Row CLI

- `ironicl` commands that fit and predict from raw rows, as the estimators do;
- the fitted model's three files (`IRMODEL`, `IRPREP`, `IRCNTXT`) as their
  inputs and outputs.

Exit gate: CLI predictions match the Python estimators byte for byte.

## Phase 10: Server

Step E of [source-layout.md](source-layout.md#steps): a prediction server behind
the `server` feature, with `/health`, `/v1/models`, and `predict` and
`predict_proba` per model, in JSON, for fitted models loaded read-only at
startup. It runs without Python.

Exit gate: server predictions match the Python estimators.

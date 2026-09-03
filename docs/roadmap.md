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
complete for the official classifier, including exact tensor manifests and
deterministic Python oracle fixtures.

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

Status: executable FP32 full-KV path complete; native preprocessing,
representation-cache execution, mmap loading, and zero-allocation serving
remain.

Completed:

- Python full-KV context builder and preprocessed query exporter;
- full-KV artifact reader/writer and whole-file checksum;
- one-row and microbatch cached inference;
- online stable attention.

Remaining:

- fixed native preprocessing schema and implementation;
- representation-cache execution;
- mmap-backed loading;
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

## Phase 5: BF16 Storage

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

- 999-quantile decoder and postprocessing;
- regressor parity and accuracy suite;
- Python API bridge if required;
- reproducible release artifacts;
- deployment and operations documentation.

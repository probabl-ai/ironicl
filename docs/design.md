# Design

*The engine's scope, the path a prediction takes, the model, the artifacts,
and the precision, kernel, memory and security policies.*

Measured results are in [benchmarks.md](benchmarks.md); building, testing and
validation are in [development.md](development.md).

**Contents**

1. [Overview](#1-overview)
2. [From training table to prediction](#2-from-training-table-to-prediction)
3. [The model](#3-the-model)
4. [The engine](#4-the-engine)
5. [Artifacts](#5-artifacts)
6. [Precision](#6-precision)
7. [Kernels](#7-kernels)
8. [Memory](#8-memory)
9. [Security](#9-security)
10. [Upstream and licenses](#10-upstream-and-licenses)

Everything on this page is implemented unless marked **Candidate** (intended,
not yet measured), **Experiment** (never the default without accuracy and
performance evidence) or **Out of scope**. When the code and this page
disagree, tests and measurements win.

## 1. Overview

IronICL is a Rust engine for cached CPU inference of TabICLv2, packaged as the
scikit-learn compatible Python package `ironicl`. Its goal is to minimise
latency and maximise throughput when the training table is static and test
rows arrive repeatedly. To that end the engine:

- imports a released FP32 checkpoint in pure Rust;
- builds a deterministic static-context artifact from the fitted
  preprocessing, the training data and the ensemble configuration;
- passes the validation policy in [development.md](development.md#5-validation);
- dispatches optimised kernels only after runtime CPU checks, with a portable
  FP32 fallback;
- reports memory and precision choices before allocating (`ironicl plan`);
- parses no files and makes no hidden allocations on the request path.

The first optimised target is an AMD Ryzen 7 9800X3D (Zen 5, AVX-512) on
Linux, for classification with at most 10 classes, a fixed feature schema,
one-row latency and 8 to 16-row microbatches. Correctness tests must keep
passing on Apple Silicon through portable code.

**Out of scope** for the first engine: training or fine-tuning; GPU and NPU
execution; more than 10 classes or regression in the cached path; online
insertion or deletion of training rows (rebuild instead); bitwise parity
across PyTorch, operating systems or CPU architectures; per-query NVMe reads
while the cache fits in RAM; `unsafe` code outside `src/kernels/native.rs`.

**Preprocessing.** `ironicl/_preprocessing.py` is a numpy and scikit-learn
port of tabicl's preprocessing and ensemble generation: constant-feature
filtering, standardisation, Yeo-Johnson, quantile and robust normalisation,
outlier clipping, and Latin-square feature and class permutations, with the
same random streams and floating-point order as upstream. The engine receives
`[B, N, H]` training variants to build a context and `[B, Q, H]` query
variants to serve. Strings, dates and missing values are handled upstream of
the estimator, typically by a skrub `TableVectorizer`. Neither tabicl nor
PyTorch is a dependency; tabicl is the external oracle the frozen test goldens
were generated from.

## 2. From training table to prediction

`IronICLClassifier` (`ironicl/_classifier.py`) mirrors the parameters of
`tabicl.TabICLClassifier`, so the two can be swapped in a pipeline.

**`fit(X, y)`**

1. Validate the table (finite float32, 2 to 10 classes); label-encode `y`.
2. Resolve the checkpoint (local path, or the Hugging Face cache through
   `huggingface_hub`) and fit the `EnsembleGenerator`: constant-feature
   filter, one `PreprocessingPipeline` per normalisation method, feature and
   class permutations.
3. Import the checkpoint into a content-addressed `IRMODEL` artifact under
   `<cache_dir>/models/` (`prepare_model`, `src/python.rs`).
4. Hash the training table and every cache-affecting setting into the context
   identity (`ironicl/_context.py`).
5. Build the static context (`build_context`, `src/context_builder.rs`): run
   the training variants through the column, row and ICL stages once,
   estimator by estimator, and stream each block's `norm1`-normalised,
   in-projected keys and values (head-major; no QASS scale, no RoPE) into
   `<cache_dir>/contexts/<hash>.ironiclctx` in the requested `kv_dtype`.
   `sparse_index=True` adds a deterministic k-means index over the ICL keys.
   Training rows never attend to query rows, so no queries are needed.
6. Load model and context into a resident `StaticExecutor`
   (`ironicl._ironicl`).

**`predict_proba(X)`** validates the rows against the fitted schema, applies
the fitted `ensemble_generator_` in test mode to get `[B, Q, H]` variants, and
runs them in chunks of 1,024 rows against the context. The engine reverses the
class permutations, averages the estimator logits and applies the official
softmax temperature 0.9 (`aggregate_classifier_probabilities`,
`src/executor.rs`), returning `[Q, C]` probabilities. Pickling keeps the
artifact paths and rebuilds the executor on unpickle.

The command-line tool reaches the same path:
`ironicl build-context` builds a context from an `IRINPUT` bundle,
`ironicl run --context` serves an `IRQUERY` bundle against it, and the
uncached `ironicl run --input request.irinput` recomputes the training rows
every call for parity work (`--trace-stages`). See
[development.md](development.md#4-the-command-line-tool).

## 3. The model

Checkpoints are published on Hugging Face under `jingang/TabICL`.

| Model | Parameters | FP32 tensor bytes |
|---|---:|---:|
| Classifier v2 | 27,552,258 | 110,209,032 |
| Regressor v2 | 28,544,991 | 114,179,964 |

The importer checks these constants against the checkpoint configuration and
rejects other architectures:

| Constant | Value |
|---|---:|
| Input feature group size | 3 |
| Column embedding width `E` | 128 |
| Column induced-attention blocks; heads; head dim | 3; 8; 16 |
| Inducing points per column block `M` | 128 |
| Row CLS tokens `C` | 4 |
| Row-transformer blocks; heads | 3; 8 |
| ICL width `D = C * E` | 512 |
| ICL blocks; heads; head dim | 12; 8; 64 |
| ICL feed-forward width | 1,024 |
| Native classifier outputs; regressor quantiles | 10; 999 |

Activation is GELU, normalisation is pre-norm LayerNorm, attention uses the
query-aware elementwise QASSMax, and rows use non-interleaved RoPE with base
100,000.

**Why caching is exact.** With `embed_with_test=false`, test rows never
attend to other test rows: inducing points see only training rows, the row
stage is per-row, ICL queries see only training keys and values, and
LayerNorm and the nonlinearities are token-local. Cached test rows can
therefore be served one at a time or in any microbatch, equal up to
floating-point reduction order. Conversely, any change to a training feature
or label invalidates every later layer, and appending a training row is not
local, because training tokens attend to the full context at every ICL layer.

## 4. The engine

The crate lives in `src/` (package `ironicl-engine`; library and binary are
both named `ironicl`). Feature `checkpoint` enables the checkpoint reader,
feature `python` the pyo3 module.

| Module | Role |
|---|---|
| `checkpoint.rs`, `checkpoint_cache.rs` | Pure-Rust reader for PyTorch `.ckpt` files (ZIP + pickle, via Candle's pickle reader after a bounded opcode preflight) and the content-addressed import cache `<cache>/models/v1/<sha256>.ironicl`. |
| `model_artifact.rs` | `IRMODEL` reader and atomic writer; loaded into RAM. |
| `static_context.rs` | `IRONICL` reader (mmap-backed FP32 and BF16 views), layout, and the streaming memory-mapped writer (versions 1 to 3, BF16 rounding, packed keys). |
| `context_builder.rs` | Native context build: column, row and ICL stages over the training variants, key/value capture per block, sparse k-means index, per-estimator streaming into the writer. |
| `input_bundle.rs`, `query_bundle.rs` | `IRINPUT` and `IRQUERY` request bundles. |
| `executor.rs`, `execution_plan.rs` | `ClassifierExecutor` (uncached) and `StaticClassifierExecutor` (cached): tiling, worker partitioning, QASS base scales, sparse-index gating, aggregation. |
| `kernels/reference.rs`, `kernels/native.rs` | Portable FP32 kernel contracts, and the AVX-512 and NEON implementations they dispatch to. |
| `platform.rs`, `config.rs`, `cache.rs`, `backend.rs` | CPU detection and the planning types behind `ironicl plan` (not read by the executor). |
| `cli.rs`, `main.rs`, `python.rs` | The `ironicl` binary and the pyo3 bindings (`StaticExecutor`, `prepare_model`, `inspect_model`, `build_context`). |
| `benchmark.rs`, `profile.rs`, `error.rs` | Latency statistics, profiling scopes, the error type. |

Artifacts are immutable after validation and shared read-only. Each cached
executor owns one persistent Rayon pool (`IRONICL_THREADS`) and each request
owns its scratch; there is no global mutable kernel state. `ExecutionPlan`
splits a request into ensemble workers times inner workers: ICL attention
assigns independent heads to scoped workers, while affine, GELU, decoder and
QASS row work runs on the pool. Nested Rayon jobs from attention workers are
not permitted.

## 5. Artifacts

Every format is little-endian with fixed-width metadata, an explicit magic,
version and kind, checked `u64` offsets and lengths, 64-byte tensor
alignment, and one trailing whole-file SHA-256 verified before any tensor is
exposed. No native struct is ever serialised.

| Format | Magic | Versions | Header | Payload |
|---|---|---:|---:|---|
| Model artifact | `IRMODEL\0` | 1 | 144 B | checkpoint tensors |
| Static context | `IRONICL\0` | 1, 2, 3 | 164 B | permutations, K/V tensors |
| Query bundle | `IRQUERY\0` | 1 | 112 B | FP32 `[B,Q,H]` |
| Input bundle | `IRINPUT\0` | 1 | 80 B | FP32 `X [B,T,H]`, `y_train [B,N]` |
| Oracle logits | `IRLOGITS` | 1 | 48 B | FP32 `[B,Q,C]` |
| Oracle trace | `IRTRACE` | 1 | length-prefixed | named FP32 stage tensors |

`IRMODEL` and `IRONICL` share a tensor directory: entries sorted by byte-wise
name and unique, each with name length (`u32`), rank (`u16`), dtype
(`IRMODEL`: `u8` plus a zero byte; `IRONICL`: `u16`, zero in version 1,
1 = FP32 or 2 = BF16 later), payload offset and byte length (`u64`),
dimensions (`u64`), then the UTF-8 name. Offsets are 64-byte aligned and
non-overlapping.

### 5.1. Model artifact

The header carries the source-checkpoint SHA-256, tensor count, declared
parameter count and tensor bytes, seven `u64` configuration values
(`max_classes`, `num_quantiles`, `embed_dim`, `column_blocks`, `row_blocks`,
`row_cls_tokens`, `icl_blocks`), and the directory and payload lengths.
Tensors keep their checkpoint dtype in the portable logical layout; released
checkpoints are recognised by pinned SHA-256 and expected counts.

| Limit | Value |
|---|---:|
| File | 512 MiB |
| Per tensor; total | 64 MiB; 256 MiB |
| Tensors; rank | 4,096; 16 |

Writes are atomic: sibling temporary file, `fsync`, reopen and re-verify with
the normal reader, rename, parent-directory `fsync`, temporary removed on
error.

### 5.2. Static context

| Version | Tensor storage | ICL K layout |
|---:|---|---|
| 1 | FP32 only | logical row-major |
| 2 | FP32 or BF16 per tensor | logical row-major |
| 3 | FP32 or BF16 per tensor | 128-row source tiles, dimension-major |

The header carries the model SHA-256 (the checkpoint hash stored in
`IRMODEL`), the context SHA-256, `B`, `N`, `H`, the class count, the
architecture tuple `[3, 12, 8, 16, 64]`, and the counts and lengths of what
follows: estimator-major class `[B,C]` and feature `[B,H]` permutations as
`u64`, the directory, the aligned payloads and the checksum.

The tensors are exactly `column.{0..2}.{key,value}`, shape `[B,H+4,8,128,16]`,
always FP32, and `icl.{0..11}.{key,value}`, logical shape `[B,8,N,64]`, in
FP32 or BF16 according to `kv_dtype` (`fp32`, `bf16`, `bf16-v`). An optional
sparse index adds `icl.index.{centroids,radii,offsets,members}` (FP32,
estimator-, layer- and head-major); it is all-or-nothing. Version 3 packs
each ICL K tile dimension-major so that one AVX-512 vector holds 16 source
lanes while a query dimension is broadcast; V stays source-major.

The loader verifies the checksum, then every header field, permutation,
directory entry, alignment, overlap and value (no non-finite FP32 or BF16).
`read` and `read_mapped` return read-only mmap-backed views; `from_bytes`
accepts version 1 up to 2 GiB.

| Limit | Value |
|---|---:|
| File; per tensor | 48 GiB; 4 GiB |
| Tensors | 4,096 |
| Rows; features | 1,000,000; 65,536 |
| Estimators; classes | 4,096; 4,096 (the executor requires 2 to 10 classes) |

The context identity hashes a domain prefix, the JSON of every
cache-affecting setting (checkpoint SHA-256, `n_estimators`, normalisation and
shuffle methods, outlier threshold, seed, `kv_dtype`, sparse-index settings)
and the canonical float64 features and int64 labels. Any change rebuilds the
context; incremental updates are **Out of scope**. Writes stream into a
sibling temporary file while hashing, then `fsync` and rename; reopen
validation and a directory sync are **Candidate**.

### 5.3. Request and oracle bundles

| Bundle | Header fields | Payload |
|---|---|---|
| `IRQUERY` | model and context SHA-256 (both must match the context), `B`, `Q`, `H`, element count | FP32 `[B,Q,H]` |
| `IRINPUT` | `B`, total rows `T`, `H`, training rows `N`, element counts and byte lengths | FP32 `X` and `y_train` (integer labels 0 to 9) |
| `IRLOGITS` | `B`, test rows, classes, element count | FP32 `[B,Q,C]`; decoded by `tests/checkpoint_fixtures.rs` |
| `IRTRACE` | named tensors | row-major FP32 stages (`col_embedder`, `row_interactor`, raw decoder output, logits); the engine prints the equivalent with `run --trace-stages` and does not read the file |

Per-section checksums, explicit arithmetic-policy metadata and a combined
artifact with packed weights are **Candidate**.

## 6. Precision

| Policy | Status | Description |
|---|---|---|
| FP32 | baseline | FP32 arithmetic and K/V; the correctness reference below PyTorch. Released checkpoints are FP32; reduced formats are derived. |
| BF16 storage, FP32 accumulation | implemented | K/V may be stored as BF16; dense weights may be packed as BF16 (`PackedBf16Affine`, opt-in through `IRONICL_BF16_DENSE`, optional first-order residual compensation). Every dot product accumulates in FP32 (`VDPBF16PS` on AVX512_BF16, conversion elsewhere). Residual stream, LayerNorm, QASSMax, RoPE, score scaling, softmax, GELU, decoder, temperature and ensemble reduction stay FP32. BF16 K/V is not lossless and is validated separately from BF16 weights. |
| INT8 | **Experiment** | Per-output-channel weight scales, INT32 accumulation, FP32 dequantisation; INT8 K/V needs per-row or per-tile scales. Evaluation order: BF16 K + BF16 V, BF16 K + INT8 V, INT8 K + BF16 V, INT8 K + INT8 V. Only the planner knows the INT8 element size. |
| FP16 | **Out of scope** | |

Fast forms are the default; exact forms exist for parity work: GELU uses the
native approximation unless `IRONICL_EXACT_GELU` is set (`gelu_f32`,
`src/kernels/reference.rs`), and LayerNorm and softmax have
`IRONICL_EXACT_LAYER_NORM` and `IRONICL_EXACT_SOFTMAX`.

## 7. Kernels

`src/kernels/reference.rs` defines the kernel contracts (affine, LayerNorm,
GELU, QASSMax, RoPE, online attention, softmax, residual, decoder) in portable
FP32 and is the oracle for every native kernel through scalar differential
tests. `src/kernels/native.rs` holds the specialised versions behind
`#[target_feature]` functions, called only after a runtime check:

| Host | Check | Kernels |
|---|---|---|
| x86-64 | `is_x86_feature_detected!` for AVX512F and AVX512BW | FP32 path |
| x86-64 | additionally AVX512_BF16 | packed BF16 affine, BF16 attention tiles |
| AArch64 (Apple Silicon) | compile-time | NEON attention dot and value FMA (`AttentionBackend::Neon`) |
| anything else | | portable scalar path |

Features are never inferred from a CPU model name; the backend is chosen once
per executor, and `ironicl platform` prints it.

**Online attention.** The engine never materialises a `[queries, heads, N]`
score matrix. Keys and values stream in source tiles while per-query, per-head
running maximum, denominator and weighted-value accumulators stay in FP32.
The cached tile is 16 query rows (`ATTENTION_QUERY_BLOCK_ROWS`,
`MAX_PACKED_ATTENTION_QUERY_ROWS`) by 128 source rows
(`ATTENTION_KEY_TILE_ROWS`, overridable with
`IRONICL_ATTENTION_KEY_TILE_ROWS`). The native score kernel produces one
`[16,128]` tile per head from row-major, packed FP32 or packed BF16 keys; the
native V kernel loads each V row once and updates all query accumulators, so a
microbatch reads each K/V tile once.

**Sparse ICL attention** (**Experiment**). With `IRONICL_SPARSE_ATTENTION=1`
and an indexed context, candidate clusters are used only when the omitted-mass
certificate is at most `1e-5` (`IRONICL_SPARSE_MAX_OMITTED_MASS`); otherwise
the dense kernel runs. `sparse_attention_statistics` reports the decisions.

**Packing and fusion.** Weights are packed at executor construction, never
per request, and artifact weights stay portable. Kernel fusion is
**Candidate** and must reduce measured traffic. On the Zen 5 target (8 cores
on one CCD, 96 MiB L3, dual-channel DDR5) the complete context does not fit in
L3, so the design relies on access order, compact storage and bounded tiles
rather than on residency; SMT, huge pages, affinity and allocator effects are
measured, not assumed.

## 8. Memory

With `N` training rows, `H` retained features, `B` estimators, `s` bytes per
element and the model constants of section 3 (`E = 128`, `C = 4`, `D = 512`,
`M = 128`, `L_col = 3`, `L_icl = 12`):

```text
column bytes         = 2 * s * B * L_col * (H + C) * M * E = s * B * 98,304 * (H + 4)
ICL K/V bytes        = 2 * s * B * L_icl * N * D           = s * B * 12,288 * N
representation bytes = s * B * N * D                       = s * B * 512 * N
weight bytes         = parameters * weight element size
ideal scan seconds   = cache bytes / sustained bytes per second
```

`ironicl plan` (`estimate_memory`, `src/cache.rs`) reports weight, column and
context bytes for cache mode `kv` or `repr` and FP32, BF16 or INT8 element
sizes, resolves the backend, and with `--bandwidth-gbps` prints the ideal
full-scan floor. Scales, alignment, scratch, page cache and OS memory are
excluded. The representation cache is 24 times smaller than ICL K/V but
recomputes all 12 ICL blocks per request; it exists only in the planner.

Static contexts are mmap-backed read-only after whole-file verification;
model artifacts are read into RAM. An mmap is not a residency promise, so
prefault and major-fault reporting are **Candidate**. Direct I/O and per-query
NVMe reads are **Out of scope** while the context fits in RAM.

## 9. Security

Checkpoints and artifacts are untrusted until their hash is verified. Parsers
validate magic, version and kind before any section metadata; use checked
arithmetic for every offset, length, count and alignment; require sections to
lie inside the file without overlap; bound rank, name length, tensor count and
dimensions; match payload length to dtype and shape; verify the checksum
before exposing data; require sorted unique names and finite values; execute
no embedded code; and serve from read-only mappings. Checkpoint conversion is
offline and pure Rust, with archive, pickle, opcode, memo, object-depth,
tensor-count, tensor-size, rank and layout limits; pickle globals are kept as
data and no Python callable runs. The serving process reads only IronICL
formats.

> [!IMPORTANT]
> A static context encodes the complete training set and targets and is as
> sensitive as the training data: apply the same access controls and never
> log K/V payloads.

Against denial of service: enforce the limits above, compute memory before
allocating, reject non-finite request values, bound executors and scratch, and
apply request timeouts outside the kernel layer.

## 10. Upstream and licenses

- TabICL (BSD-3-Clause) is the behavioural authority for preprocessing,
  ensembles, model stages and the cached path. NanoTabICL
  (<https://github.com/soda-inria/nanotabicl>) is a concise reference that
  yields to the full checkout and checkpoint configuration when they differ.
- Checkpoints, Hugging Face `jingang/TabICL`, pinned by SHA-256:
  `tabicl-classifier-v2-20260212.ckpt`
  `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`;
  `tabicl-regressor-v2-20260212.ckpt`
  `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`.
- Checkpoint parser: `candle-core` 0.9.1 (MIT or Apache-2.0), CPU-only.
- Papers: Qu et al., "TabICL: A Tabular Foundation Model for In-Context
  Learning on Large Data", ICML 2025, arXiv:2502.05564; Qu et al., "TabICLv2:
  A better, faster, scalable, and open tabular foundation model",
  arXiv:2602.11139, 2026.
- IronICL is BSD-3-Clause (`LICENSE`, Copyright 2026 Probabl): a fresh
  implementation informed by TabICL, using no forecast-derived upstream code.
  `ironicl/_preprocessing.py` is a port of tabicl 2.2.0's
  `_sklearn/preprocessing.py` and carries its notice. Before copying further
  upstream code, retain its notices, mark modified files and ship the
  license; verify the checkpoint license and deployment terms independently.

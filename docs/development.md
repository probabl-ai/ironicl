# Development

*How to build, test and extend IronICL: a Rust engine wrapped by pyo3 as the
`ironicl` scikit-learn package.*

The engine's design is in [design.md](design.md); measured results are in
[benchmarks.md](benchmarks.md).

**Contents**

1. [Setting up](#1-setting-up)
2. [Repository layout](#2-repository-layout)
3. [Developing on Apple Silicon](#3-developing-on-apple-silicon)
4. [The command-line tool](#4-the-command-line-tool)
5. [Validation](#5-validation)
6. [Roadmap](#6-roadmap)
7. [Known cleanup opportunities](#7-known-cleanup-opportunities)

## 1. Setting up

Read [CONTRIBUTING.md](../CONTRIBUTING.md) first: it covers the toolchain,
`make install`, `make lint`, `make test`, pull-request titles and code
conventions.

In short: a stable Rust toolchain plus Python 3.10 or newer, then
`make install` in a virtual environment. That builds the extension in editable
mode through maturin and installs the pre-commit hooks. Nothing else is
needed: the package depends on numpy, scikit-learn and `huggingface_hub` only,
and never on tabicl or PyTorch.

`make test` runs `cargo test` from the root, then `pytest`. CI also runs
`cargo test --no-default-features`, without the `checkpoint` feature and its
Candle-based parser.

> [!NOTE]
> Tests marked `slow` download the TabICLv2 checkpoint from Hugging Face on
> first use, or read it from `IRONICL_CHECKPOINT_DIR`. Set
> `IRONICL_SKIP_SLOW=1`, or run `pytest -m "not slow"`, to skip them.

## 2. Repository layout

```text
ironicl/
├── ironicl/        Python package: the scikit-learn estimator, preprocessing, context identity
├── src/            Rust crate: engine, artifacts, kernels, CLI, pyo3 bindings
├── tests/          pytest suite, frozen tabicl goldens (data/), cargo integration tests
├── benchmarks/     cached-serving benchmark
├── examples/       skrub + skore pipeline example
├── docs/           design, benchmarks, development; images/ for the README logo
├── pyproject.toml  Python build (maturin), ruff, mypy, pytest
├── Cargo.toml      Rust crate `ironicl-engine`; features: checkpoint, python, profile
└── Makefile        install, lint, test, bench
```

The Rust modules and their roles are listed in
[design.md](design.md#4-the-engine); Python modules carry their role in
their module docstring.

## 3. Developing on Apple Silicon

Apple Silicon is a full development platform for everything except the Zen 5
performance work. On `aarch64-apple-darwin` you can build, lint and run every
Rust and Python test, import checkpoints, write and verify artifacts, execute
the portable FP32 path, and check parity against PyTorch on CPU.

### 3.1. What runs

The portable kernels in `src/kernels/reference.rs`; the NEON attention dot
products, value accumulation and GELU in `src/kernels/native.rs`, selected at
compile time for `aarch64`; and the pyo3 extension and estimator, for which CI
builds the wheel on `macos-14`.

### 3.2. What cannot be measured

- AVX-512 code paths: the fused attention tiles, the packed BF16 affine
  kernels (`VDPBF16PS`) and the BF16 to FP32 conversion are compiled only for
  `x86_64` and dispatched after `is_x86_feature_detected!` checks.
  `ironicl platform` reports every AVX-512 flag as `no` on Apple Silicon.
- INT8 VNNI, 96 MiB L3 behaviour, DDR5 bandwidth, Linux affinity and
  huge-page APIs, `perf` and uProf counters, and any Ryzen performance number.

### 3.3. Portability rules

- `cargo test` and `cargo clippy -D warnings` must pass on
  `aarch64-apple-darwin`.
- x86 intrinsics stay behind `#[cfg(target_arch = "x86_64")]` plus a runtime
  feature check, with `unsafe` confined to `src/kernels/native.rs`.
- Artifacts use explicit little-endian byte layouts and never serialise
  `usize`, native enums or Rust structs.
- Fixtures are small enough to commit, or are generated deterministically.
  There are no fuzzing or property-based test suites.
- Cross-compiling for `x86_64` catches target-gating mistakes but proves
  nothing about whether an instruction is supported or fast. Native runs on
  the target host are required before an optimised kernel becomes the
  default.

## 4. The command-line tool

`ironicl` is the engine's development tool. `cargo build --release` places it
at `target/release/ironicl`; `ironicl help COMMAND` prints a command's page.

| Command | Purpose |
|---|---|
| `platform` | Print the detected CPU features. |
| `plan` | Size memory and bandwidth for a workload, without a model. |
| `inspect` | Describe a checkpoint or a model artifact. |
| `import` | Convert a checkpoint into a model artifact. |
| `build-context` | Build a static context from the training rows of an input bundle. |
| `run` | Compute logits or probabilities for one request. |
| `benchmark` | Time repeated predictions on a resident model. |

All commands take named options only. Two options recur: `--checkpoint FILE`
is a PyTorch checkpoint (`.ckpt`, `.pt`, `.pth`) and `--model FILE` is a
verified IronICL model artifact. A command that accepts both requires exactly
one. `--output FILE` always names a new artifact.

### 4.1. `platform`

```console
$ ironicl platform
```

Prints the detected CPU features (AVX2 and the AVX-512 subsets) and the
attention backend the engine would select.

### 4.2. `plan`

```console
$ ironicl plan --rows N --features N --estimators N [options]
```

Reports weight, column and context bytes and the ideal full-scan time for a
workload. Each option is accepted at most once.

| Option | Values |
|---|---|
| `--task` | `classifier`, `regressor` |
| `--rows`, `--features`, `--estimators`, `--microbatch` | integers |
| `--cache` | `kv`, `repr` |
| `--weights`, `--kv-dtype` | `fp32`, `bf16`, `int8` |
| `--backend` | `auto`, `portable`, `zen5` |
| `--bandwidth-gbps` | sustained memory bandwidth used for the scan floor |

### 4.3. `inspect`

```console
$ ironicl inspect --checkpoint FILE
$ ironicl inspect --model FILE
```

On a checkpoint, reports the source SHA-256, profile, tensor and parameter
counts, payload bytes, dtypes and configuration without executing pickle code.
On a model artifact, runs the full checksum validation.

### 4.4. `import`

```console
$ ironicl import --checkpoint FILE --output FILE
```

Copies the checkpoint tensors into a 64-byte-aligned portable artifact,
writes it atomically and verifies it. The artifact holds no preprocessing
state and no key/value caches.

### 4.5. `build-context`

```console
$ ironicl build-context --model FILE --input FILE --output FILE [options]
```

Builds a full-K/V static context natively from the training rows of an
`IRINPUT` bundle. Query rows are ignored, permutations are the identity, and
the context identity is the bundle's SHA-256. The resulting file serves
`run --context` and `benchmark --context`.

| Option | Effect |
|---|---|
| `--kv-dtype fp32\|bf16\|bf16-v` | Storage precision of the ICL keys and values (default `fp32`). |
| `--pack-keys` | Write format version 3, with ICL keys in packed source tiles. |
| `--index-clusters N` | Add the sparse key index with `N` clusters per head. |

### 4.6. `run`

```console
$ ironicl run (--checkpoint FILE | --model FILE) --input FILE [options]
```

Computes classifier logits for one request. With `--input` naming an
`IRINPUT` bundle the training table travels with the request (uncached path);
with `--context FILE` the input is an `IRQUERY` bundle served against a static
context, whose model and context identities, ensemble count and feature count
must match.

| Option | Effect |
|---|---|
| `--context FILE` | Serve the query bundle against this static context; output is per-estimator `[B, Q, C]` logits. |
| `--probabilities` | With `--context`: apply class permutations and ensemble aggregation, print `[Q, C]` `predict_proba` rows. |
| `--trace-stages` | Without `--context`: print shape, minimum, maximum, sum and first eight values of `column_embeddings`, `row_representations` and `decoder_logits` before the logits. |
| `--rebuild-cache`, `--cache-dir DIR`, `--no-cache` | With `--checkpoint`: the checkpoint is imported into a content-addressed cache (`$XDG_CACHE_HOME/ironicl` or `$HOME/.cache/ironicl`) and reused on later runs; `--rebuild-cache` replaces it, `--cache-dir` relocates it, `--no-cache` is rejected because execution needs an imported artifact. |

The output is text: `shape: B test_rows C` followed by one
`logits[ensemble,row]:` line per row in scientific notation, or `shape: Q C`
and `probabilities[row]:` lines with `--probabilities`.

### 4.7. `benchmark`

```console
$ ironicl benchmark --model FILE --input FILE [--context FILE] [--probabilities] [--warmup N] [--iterations N]
```

Times repeated predictions on a resident model after loading and executor
construction; `--context` and `--probabilities` select the cached lane as for
`run`. Prints load times, nearest-rank `p50`/`p95`/`p99` latencies,
throughput, an output checksum and the raw `samples_ns`; the `profile`
feature appends stage timers.

| Option | Range | Default |
|---|---|---|
| `--warmup N` | 0 to 1,000 | 2 |
| `--iterations N` | 1 to 10,000 | 10 |

## 5. Validation

### 5.1. Oracle hierarchy

The levels, in order, are: released TabICLv2 PyTorch FP32 CPU inference, then
the portable FP32 engine, then optimised FP32, then BF16 storage. Each level
must pass before the next is used for performance conclusions.

### 5.2. Parity gates

| Comparison | Tolerance | Note |
|---|---|---|
| Uncached classifier logits against the Python oracle | `rtol=1e-4`, `atol=1e-4` | The official fixture passes below `1e-5` maximum absolute error; the cached AVX-512 path at `3.05e-5`. |
| Python `predict_proba` against tabicl goldens | `rtol=1e-3`, `atol=1e-4`, identical argmax labels | The looser tolerance reflects a different FP32 operation order, the same reason upstream's cache tests use `1e-4`. |
| Reduced precision (BF16, any future INT8) | Dataset-level accuracy, ROC AUC and log loss per dataset | No concentrated regression may hide behind an acceptable mean. |

### 5.3. Rust fixture tests

`tests/checkpoint_fixtures.rs` reads the two official checkpoints from
`IRONICL_CHECKPOINT_DIR` and the oracle fixtures `input.irinput` and
`logits.irlogits` from `IRONICL_CHECKPOINT_DIR/oracle/`. The tests return
early, and pass, when the variable is unset or the oracle files are absent, so
a green run on a fresh clone exercises none of the import, artifact or parity
paths. Set the variable to make them real:

```sh
IRONICL_CHECKPOINT_DIR=path/to/checkpoints cargo test -- --nocapture
```

The oracle fixtures are little-endian binary formats with SHA-256 trailers:
`IRINPUT` v1 (`src/input_bundle.rs`: 80-byte header, row-major FP32 `X`, FP32
`y_train`), `IRLOGITS` v1 (48-byte header with `B`, test rows, classes and
element count, then `[B, test_rows, C]` FP32 logits), and `IRTRACE` v1 (named
FP32 tensors for `col_embedder`, `row_interactor`, the raw 10-wide decoder
output and the logits, compared by hand against `run --trace-stages`). The
exporter that produced them from a sibling tabicl checkout went away with
`tools/`; the hashes pinned in the Rust test still identify the exact files.

### 5.4. Python fixtures

For the Python package the same role is played by the frozen fixtures under
`tests/data/<dataset>/`, produced once by `tests/generate_goldens.py` with
tabicl 2.2.0 on CPU (one thread, `kv_cache="kv"`, `random_state=42`,
`make_classification` tables; four datasets covering binary and multiclass
targets, a constant feature, and every normalisation method). Each holds the
raw table, tabicl's `probabilities`, and under `preprocessing/` the ensemble
variants, labels and permutations of tabicl's fitted `EnsembleGenerator`, per
normalisation method. The script is kept as the record of how they were made
and needs a separate environment with tabicl installed; this repository does
not depend on tabicl or PyTorch.

### 5.5. Cross-checks

- `tests/test_preprocessing.py` (always on) requires `ironicl._preprocessing`
  to reproduce the frozen variants to `1e-6` with identical permutations.
- `tests/test_consistency.py` (marked `slow`) fits `IronICLClassifier` with
  the recorded parameters and compares `predict_proba` with tabicl's at
  `rtol=1e-3`, `atol=1e-4` and identical argmax. It reads the checkpoint from
  `IRONICL_CHECKPOINT_DIR` when set, otherwise from the Hugging Face cache or
  download.
- The Rust test `native_static_context_matches_uncached_executor` checks that
  a natively built context served by `StaticClassifierExecutor` reproduces
  `ClassifierExecutor`'s uncached logits to `1e-5` (FP32 and packed keys),
  with matching labels for BF16 storage.
- Against a context built with tabicl's PyTorch `forward_with_cache`, the
  native build matched every ICL tensor within `2e-4` and every feature
  column stream within `3e-5`; only the four reserved column streams, which
  the executor never reads, differ.

## 6. Roadmap

| Phase | Status (checked against the code) |
|---|---|
| 0 Executable skeleton | Done. |
| 1 Checkpoint import and oracle fixtures | Done for the classifier; exact tensor manifests, pinned fixtures. |
| 2 Portable FP32 model | Done; parity below `1e-5`. |
| 3 Static full-K/V context | Done except representation-cache execution and zero steady-state allocations: full-K/V artifact, native context build and preprocessing, one-row and microbatch cached inference, online attention, mmap-backed loading. |
| 4 9800X3D FP32 backend | Active; runtime-gated AVX-512 affine and attention-tile kernels, 34.57 ms median for `B=1, N=1024, Q=64, H=54`. |
| 5 BF16 storage | BF16 K/V storage (`kv_dtype="bf16"`, `bf16-v`) and packed-BF16 dense kernels exist; the full accuracy suite and migration policy do not. |
| 6 INT8 experiments | Not started; `plan` accepts `int8` but the executor ignores it. |
| 7 Regression and packaging | The Python API bridge is done as the pyo3 package; regressor parity, the 999-quantile decoder and release artifacts are not started. |

Remaining work, in order:

1. A faster ICL stage in the native context build for tables above about
   10k rows (it uses the head-parallel cached-attention kernel and
   row-chunked affines, not the Zen 5 tile kernels).
2. Representation-cache execution and the no-allocation request path.
3. Regressor support.
4. INT8 K/V as an experiment behind the dataset-level accuracy gate.

## 7. Known cleanup opportunities

Factual observations from a code read; none is scheduled.

- `src/executor.rs` carries four variants of the transformer block (`block`,
  `cached_block`, `uncached_block`, `packed_row_block`) and two executors with
  their own `row_embeddings`, `column_stream` and `qass`.
- QASSMax exists three times (`kernels/reference.rs`, inline in
  `executor.rs`, test-only in `kernels/native.rs`); the engine uses inline
  softmax and GELU rather than `reference::softmax_f32` and the kernel GELU.
- Three near-identical online-attention kernels in `kernels/reference.rs` sit
  under four executor drivers; weight-packing helpers come in parallel FP32
  and BF16 pairs.
- `tokens_to_heads` and `heads_to_tokens` are defined in both `executor.rs`
  and `kernels/reference.rs`; the reference copies serve only their own tests.
- `ByteReader` and `sha256` are pasted into `input_bundle.rs`,
  `model_artifact.rs`, `query_bundle.rs` and `static_context.rs`;
  `checked_product`, `checked_elements`, `write_u16/32/64`, `align_up` and
  `pad_to_alignment` are duplicated between the two artifact modules.
- `config.rs`, `backend.rs` and `cache.rs` serve only `ironicl plan`; the
  `int8` dtypes and the `zen5` backend they model are never consulted by the
  executor.
- Runtime behaviour is tuned by 25 `IRONICL_*` environment variables read in
  `executor.rs`, `kernels/`, `profile.rs` and the Python package
  (`IRONICL_THREADS`, `IRONICL_CACHE_DIR`, `IRONICL_BF16_DENSE`,
  `IRONICL_SPARSE_ATTENTION`, `IRONICL_LAYER_MAJOR`, `IRONICL_EXACT_GELU` and
  others) rather than by `ExecutionConfig`; they are documented in
  [benchmarks.md](benchmarks.md#26-environment-variables).
- `native::layer_norm_statistics_f32` and `reference::{dot_f32,
  embedding_lookup_f32, residual_add_f32, packed_qkv_projection_f32}` have no
  non-test callers, and the `profile` feature is enabled by no script or
  target.
- The fixture tests pass silently without `IRONICL_CHECKPOINT_DIR`,
  `tests/cli.rs` only checks help text, and `executor.rs` has few unit tests
  for its size and none end-to-end.
- The module docs at the top of `lib.rs` and `executor.rs` still describe
  scalar, uncached execution and predate the cached path, BF16 and Rayon.

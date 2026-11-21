# Engine Architecture

## Components

### Checkpoint Importer

Reads the trusted released checkpoint, validates tensor names and shapes, and
converts tensors into an internal layer-major format. Serving must not depend on
Python or PyTorch. The pure-Rust checkpoint front end uses Candle's pinned
PyTorch pickle reader.

### Preprocessing

Rust fits and applies TabICL's preprocessing: feature filtering,
standardization, Yeo-Johnson power transforms, outlier handling, and feature
and class permutations, at parity with TabICL. The fitted model predicts from
raw features ([native-preprocessing.md](native-preprocessing.md)). Category
encoding and imputation come next.

### Context Builder

`Executor::build_context` runs the preprocessed training rows through the column, row, and ICL
stages once, on the CPU, Metal, or CUDA, and keeps each stage's attention keys and values as a
static context (`IRCNTXT`). `optimize` selects their storage precision. With a cache directory, it
reuses a context with the same identity or writes the new one atomically
([artifact-format.md](artifact-format.md#static-context)).

### Artifact Loader

The loader validates format version, dimensions, identities, tensor names, dtypes, offsets,
alignment, finite values, and the trailing whole-file checksum. It keeps read-only, memory-mapped
tensor views and prefaults the mapping. Per-section checksums are not implemented.

### Portable Model Cache

Loading a checkpoint hashes it and resolves a versioned, content-addressed portable model
(`IRMODEL`). A cache miss performs the same checked conversion as `ironicl import`; a hit verifies
the model checksum and source hash without reparsing pickle. The model cache is separate from the
dataset-specific context cache.

### Command Interface

The single `ironicl` binary exposes platform reporting, checkpoint and model inspection, and
checkpoint import ([cli.md](cli.md)). A CLI that fits and predicts from raw rows is planned.

### Planner

`memory::plan` sizes a context and its build against the host and device
memory and says whether it fits ([memory-planner.md](memory-planner.md)).
`ironicl platform` reports the CPU instruction set the kernels detect (`Isa`:
portable, NEON, or AVX-512).

### Portable Backend

Provides deterministic FP32 kernels and a correctness path on Apple Silicon and
x86-64. It is the implementation oracle below PyTorch.

### AMD Zen 5 Backend

Provides AVX-512 FP32 affine and cached-attention kernels (`Isa::Avx512`),
selected through runtime CPUID checks. Current model weights remain in the
portable canonical layout. Native dispatch has no effect on artifact semantics.

### Apple Silicon Backend

Provides Accelerate `cblas_sgemm` dense and attention kernels on the matrix
coprocessor, NEON LayerNorm and short-source attention, and a Metal GPU path
(behind the `metal` feature) that builds static contexts and runs the predict
graph per estimator. Under `optimize=auto`/`fast` the GPU stores ICL caches and
runs ICL attention in FP16. The CPU kernels run as `Isa::Neon`; the GPU path is
selected at execution time rather than by the planner.

### NVIDIA Backend

Behind the `cuda` feature (Linux), kernels written with cuda-oxide build contexts and predict on
NVIDIA GPUs. The ICL cache stays on the GPU, or streams from host memory when it does not fit.

### Request Executor

`StaticExecutor` owns one persistent dense-work pool and request-local activation buffers. It runs
preprocessed query rows against an immutable static context, reverses each estimator's class
permutation, and aggregates the estimators into probabilities, or summarizes the regressor's
quantiles. `FittedModel` pairs it with the fitted preprocessing, so callers pass raw rows.

## Fit Lifecycle

```text
checkpoint -> content-addressed IRMODEL -> Executor
                                             |
raw X_train, y_train -> Rust preprocessing (IRPREP) -> transformed rows per estimator
                                             |
                                             v
                      build_context (CPU, Metal, or CUDA) -> static context (IRCNTXT)
                                             |
                                             v
                            optional atomic write to the context cache
```

## Prediction Lifecycle

```text
raw X_test -> fitted preprocessing -> transformed rows per estimator
                                             |
                                             v
           column stage -> row transformer -> ICL attention over the cached keys and values
                                             |
                                             v
           decoder -> class-order reversal and estimator aggregation -> probabilities or outputs
```

The Python estimators drive both lifecycles today. A server that loads the same three files and
predicts from raw rows is planned ([roadmap.md](roadmap.md)).

## Parallelism

- One worker group owns a request or query chunk.
- Cached ICL attention partitions independent heads across scoped workers.
- Cached ICL affine, exact GELU, decoder, and QASS row work uses one persistent
  executor-owned Rayon pool.
- Column feature streams and query row interaction use scoped workers.
- Do not submit nested Rayon jobs from attention workers.
- Do not permit independent threads to reread the same K/V when a query chunk
  can share one streamed tile.
- Pinning and NUMA placement are platform policies, not model semantics.
- The 9800X3D is one CCD, so no cross-CCD policy is needed on the first target.

## Memory Ownership

- Artifacts are immutable after validation.
- Weights and caches are shared read-only across executors.
- Each executor owns its dense-work pool. Request execution owns activation and
  online-attention scratch buffers.
- No global mutable kernel state is permitted.
- The model loader owns decoded weights. Static-context tensor storage is read-only and
  memory-mapped, or page-aligned memory a build produced. Request execution allocates
  intermediate activations.

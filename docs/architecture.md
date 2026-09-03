# Engine Architecture

## Components

### Checkpoint Importer

Reads the trusted released checkpoint, validates tensor names and shapes, and
converts tensors into an internal layer-major format. Serving must not depend on
Python or PyTorch. The pure-Rust checkpoint front end uses Candle's pinned
PyTorch pickle reader and is compiled through the `checkpoint` feature.

### Preprocessing Bridge

The current Python exporter runs official fitted preprocessing and writes
preprocessed query variants. A later native implementation can replace this
only after parity tests cover category encoding, imputation, feature filtering,
standardization, Yeo-Johnson power transforms, outlier handling, feature
permutations, and class permutations.

### Context Builder

The current Python exporter runs all training-dependent stages through official
TabICL. It streams FP32 or BF16 K/V tensors into a temporary artifact and then
atomically replaces the destination. Version 3 can pack ICL K for the source-lane
score kernel during export.

### Artifact Loader

The loader validates format version, dimensions, identities, tensor names,
dtypes, offsets, alignment, finite values, and the trailing whole-file checksum.
It retains read-only mmap-backed FP32 or BF16 tensor views. Per-section checksums
and prefault controls are not implemented.

### Portable Model Cache

`run --checkpoint` hashes the source and resolves a versioned, content-addressed
portable model. A cache miss performs the same checked conversion as explicit
`import`; a hit verifies the model checksum and source hash without reparsing
pickle. The portable cache is separate from dataset-specific static context.

### Command Interface

The single `ironicl` binary exposes checkpoint inspection, artifact import, and
execution. The parser is independent of the checkpoint dependency so builds can
disable direct checkpoint support while retaining artifact execution.

### Planner

Selects a backend and reports model/cache payload sizes. With a bandwidth input,
it also reports ideal full-cache scan floors. Scratch-memory and detailed K/V
traffic reports are not yet exposed.

### Portable Backend

Provides deterministic FP32 kernels and a correctness path on Apple Silicon and
x86-64. It is the implementation oracle below PyTorch.

### AMD Zen 5 Backend

Provides AVX-512 FP32 affine and cached-attention kernels selected through
runtime CPUID checks. Current model weights remain in the portable canonical
layout. Native dispatch has no effect on artifact semantics.

### Request Executor

Owns one persistent dense-work pool and request-local activation buffers. It
executes preprocessed query variants against an immutable static context and
returns per-estimator FP32 logits. Class permutation and ensemble aggregation
remain outside the current cached executor.

## Offline Lifecycle

The current exporter implements the temporary-file, file-sync, and atomic-rename
steps. It does not reopen and validate the completed artifact before rename.

```text
checkpoint + fitted preprocessing + X_train + y_train
                         |
                         v
                  validate and hash
                         |
                         v
                build and stream K/V
                         |
                         v
              write immutable artifact.tmp
                         |
                         v
              fsync, verify, atomic rename
```

## Serving Lifecycle

```text
read model, context, and IRQUERY -> validate -> construct executor
                                              |
preprocessed variants -> column cache -> row transformer -> ICL K/V scan
                                              |
                                              v
                              decoder -> per-estimator logits
```

Current Rust serving does not preprocess raw rows, reverse class permutations,
or aggregate estimators. Static contexts are mmap-backed. Model mmap and explicit
prefaulting remain future loading modes.

## Parallelism

- One worker group owns a request or microbatch.
- Cached ICL attention partitions independent heads across scoped workers.
- Cached ICL affine, exact GELU, decoder, and QASS row work uses one persistent
  executor-owned Rayon pool.
- Column feature streams and query row interaction use scoped workers.
- Do not submit nested Rayon jobs from attention workers.
- Do not permit independent threads to reread the same K/V when a microbatch
  can share one streamed tile.
- Pinning and NUMA placement are platform policies, not model semantics.
- The 9800X3D is one CCD, so no cross-CCD policy is needed on the first target.

## Memory Ownership

- Artifacts are immutable after validation.
- Weights and caches are shared read-only across executors.
- Each executor owns its dense-work pool. Request execution owns activation and
  online-attention scratch buffers.
- No global mutable kernel state is permitted.
- The model and query loaders own decoded values. Static-context tensor storage
  is read-only mmap-backed. Request execution allocates intermediate activations.

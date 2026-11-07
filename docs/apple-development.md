# Apple M1 Max Development

## Purpose

The M1 Max is the primary development machine. It validates architecture,
artifact parsing, portable kernels, tests, CLI behavior, and PyTorch parity, and
it runs its own performance backend (Accelerate, NEON, and the Metal GPU). It
cannot validate the AMD performance backend.

## Work Supported on macOS

- Rust compilation, formatting, linting, and unit tests.
- Checkpoint metadata conversion tools.
- Static artifact writer and reader tests.
- FP32 portable model execution.
- Layer-level comparison against PyTorch on CPU or MPS, with backend-specific
  tolerances recorded separately.
- Property tests for layout transforms and online softmax.
- Fuzzing of artifact metadata and bounds checks.
- Small functional benchmarks that detect gross regressions.

## Apple Silicon Performance Backend

macOS is a performance target, not only a development host. On Apple Silicon:

- the affine kernel dispatches to NEON; affine layers with at least 16 rows and
  attention with at least 4096 scores per head call Accelerate's `cblas_sgemm`
  on the matrix coprocessor. NEON also provides LayerNorm and short-source
  attention, and softmax is vectorized with vDSP;
- the `metal` cargo feature enables a Metal GPU path that builds static contexts
  and runs the predict graph. `IRONICL_NO_METAL=1` disables it; contexts with
  fewer than 160 training rows build on the CPU because the build graph's setup
  costs about 250 ms; the [memory planner](memory-planner.md) sizes the groups
  of estimators built together, and `IRONICL_MEMORY_LIMIT` lowers the memory it
  plans for. Query counts pad to buckets that keep their top four significant
  bits.

The full experiment record and gates are in
[experiments.md](experiments.md#apple-silicon-experiments). The GPU latencies
there are manual measurements; no tracked harness exercises the Metal path yet.

## Work Deferred to the Arch Linux Host

- CPUID verification for all AVX-512 subsets.
- Zen 5 code generation and disassembly review.
- BF16 `VDPBF16PS` kernel validation.
- INT8 VNNI kernel validation.
- 96 MiB L3 behavior.
- dual-channel DDR5 bandwidth and scaling.
- Linux affinity, huge-page, and NUMA APIs.
- AMD uProf and Linux `perf` counters.
- Gen5 NVMe startup and mapping benchmarks.

## Portability Contract

- `cargo test` and `cargo clippy` must pass on `aarch64-apple-darwin`.
- architecture-specific intrinsics (x86 AVX-512 and AArch64 NEON) must remain
  behind target and runtime gates.
- Artifacts use explicit byte order and layout metadata.
- Do not serialize `usize`, native enums, or native Rust structs.
- Golden fixtures must be small enough to keep in the repository or generated
  deterministically from documented seeds.

## Cross-Compilation

Cross-compiling can catch target-gating errors, but it cannot prove that an
instruction is supported or fast. Native tests on the 9800X3D remain required
before merging an optimized backend as the default.

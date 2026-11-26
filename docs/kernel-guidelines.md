# Kernel and Performance Engineering Guidelines

This document defines the engineering standards for writing, porting, and tuning
kernels across Ironicl's backends. It complements [kernels.md](kernels.md)
(inventory and layouts), [benchmarking.md](benchmarking.md) (measurement
harnesses), and [experiments.md](experiments.md) (acceptance gates and
historical results).

## 1. General Guidelines (All Backends)

### Profile Before Editing

- **Do not optimize blind.** Collect hardware profiles before proposing fusions,
  overlap buffers, or new tile sizes (`ncu`/`nsys` on CUDA, Instruments and
  Metal counters on macOS, `perf` on Linux CPU).
- **Establish a reference ceiling first.** Measure target shapes (batch sizes
  1, 64, 1,000, 10,000, and large contexts) against the reference engine
  (PyTorch, cuBLAS, FlashAttention, or Accelerate) so each pass knows the exact
  ms, TFLOPS, and GB/s headroom.
- **Account for total wall time.** Separate compute kernels, device memsets,
  H2D/D2H transfers, inter-kernel idle gaps, and host overhead. Remove wasted
  work (redundant memsets, per-op allocations, unnecessary transposes) before
  tuning microkernels.

### Equal-Footing Benchmarks

- **Keep methodology symmetric.** Use identical warmup counts, timed iterations,
  process isolation, device-synchronized wall time, and release build flags
  (`lto = "fat"`, `codegen-units = 1` where supported). Never compare a warmed
  Ironicl run against a cold PyTorch, MPS, or CUDA compile.
- **Verify the actual execution path.** Confirm the active backend, thread
  count, and zero silent CPU fallbacks, swap thrashing, or OOM fallbacks before
  reporting numbers.
- **Enforce retention gates.** Retain an optimization only when it clears a
  measurable end-to-end speedup threshold and passes parity gates (`safe` exact
  or bitwise; `auto`/`fast` within probability error and 100% label-agreement
  bounds; see [experiments.md](experiments.md#acceptance-gates)). Revert dead
  or sub-threshold experiments immediately.

### Complete Performance Parity Across Models

- **Do not treat a correctness-only port as finished.** Any new model must wire
  into Ironicl's existing performance primitives across its supported backends:
  - reused per-pass workspaces rather than per-op scratch allocations,
  - zero redundant buffer clears,
  - all three `optimize` levels (`safe`, `auto`, `fast`) wired end to end,
  - shared SIMD and GPU dispatchers parameterized by alignment or multiples
    rather than hardcoded TabICL-only shapes (such as `head_dim == 16 | 64`).
- **No silent backend downgrades.** Do not map an `optimize` level to CPU inside
  a GPU build (for example, routing `Optimize::Safe` to CPU on CUDA). Fall back
  to CPU only on genuine hardware or memory limits, and surface the fallback
  explicitly.
- **Keep `safe` simple; push throughput in `auto` and `fast`.** `safe` stays
  clean, readable, hardware-accelerated FP32. Put tensor-core, reduced-precision,
  and aggressive fusion complexity in `auto` and `fast`.

### Threading and Scaling

- **Parallelize inside the estimator, not only across estimators.** When memory
  limits force `group == 1` (such as 50k+ row CPU builds) or a model uses few KV
  heads (GQA), inner stages must still fan out across available cores. Never
  allow large fits or small KV-head models to collapse to one thread.

### Safety, Numerics, and Scope

- **Validate at every safe-to-`unsafe` boundary.** Every safe `pub(crate)`
  dispatcher must verify slice lengths and checked index arithmetic
  (`checked_mul`, `u32::MAX` kernel and grid bounds) before calling `unsafe`
  SIMD or GPU code.
- **Harden reduced precision and NaN handling.** Every FP16, BF16, and FP8
  conversion must handle NaNs and check post-rounding overflow to `±Inf`. Every
  backend must check full output finiteness (not only `output[0]`) and match
  reference error semantics on NaN scores.
- **Treat intermittent OOMs and sanitizer errors as blockers.** Isolate and fix
  non-deterministic GPU OOMs or sanitizer hits rather than moving past them as
  flakes.
- **Keep PR scope tight.** Do not modify another model's hot path during a
  single-model performance PR without explicit agreement. While Ironicl is
  pre-release, update internal kernel layouts and artifact formats in place
  instead of adding compatibility shims or bumping format versions.

## 2. Platform-Specific Guidelines

### NVIDIA CUDA (`cuda-oxide`)

- **Codegen rules:**
  - Keep hot inner loops directly inside `#[kernel]` (or verified-inlined
    helpers). Avoid `#[device]` helpers that take raw pointers on hot paths:
    `cuda-oxide` lowers them to real `CALL` instructions with generic `LD`
    instead of `LDG`/`LDS`.
  - Inspect generated PTX and `ncu` after every structural edit. Check local
    memory spills (`lmem`), generic loads, `#[unroll]` hitting the ~65K op
    limit, warp stall reasons (`long_scoreboard`, `short_scoreboard`, `wait`),
    L2 hit rate, and achieved DRAM bandwidth.
- **Memory and launch hygiene:**
  - Use grow-only workspaces reserved once per pass shape, not per-op
    stream-ordered allocations with retirement fences. When using device memory
    pools, raise the pool release threshold so synchronizations do not unmap
    memory every pass.
  - Have producer and packing kernels zero-pad tail rows instead of issuing
    `cudaMemset`. Verify every removed memset with `compute-sanitizer`
    (`initcheck` and `memcheck`).
  - Serialize shared CUDA streams across concurrent callers with a `Mutex`.
  - Pre-pack eligible FP16 weights into tile layout at upload time.
- **Tensor cores and precision tiers:**
  - Drive 2–3 stage `cp.async` pipelines, XOR-swizzled shared memory,
    `ldmatrix`, and `mma.sync` (`m16n8k16`) with hoisted per-thread offsets and
    cross-stage fragment prefetching.
  - Use split-FP16 (`hi + lo / LO_SCALE`) with `mma_m16n8k16_f16_f16`
    corrections for `auto`, and FP16 or E4M3 FP8 KV caches (folding scales into
    packing and unpacking) plus FP16 wide-attention accumulators for `fast`.
  - Beat PyTorch and cuBLAS through fusion (LayerNorm/RMSNorm, row gating, GELU,
    residual adds, and next-stage packing) and low launch overhead. Do not burn
    engineering effort trying to out-tune cuBLAS on bare FP32 CUDA-core SGEMM
    (`linear64`).
- **Multi-arch targets and verification:**
  - Gate `sm_89+` instructions (such as hardware FP8 `cvt.rn.f16x2.e4m3x2`)
    cleanly behind multi-arch PTX builds so `sm_80` continues to compile and
    pass tests (see [source-layout.md](source-layout.md#cuda-build-targets)).
  - Test both `IRONICL_CUDA_MODULE=sm_80` and native arch (`sm_89`/`sm_120`),
    `u32::MAX` and grid-limit launches, and `compute-sanitizer`.

### Apple Silicon (Metal `MPSGraph`, Accelerate/AMX, and NEON)

- **Metal (`MPSGraph`):**
  - Keep an entire stage pipeline in one graph per bucket. Splitting graphs
    breaks MPSGraph fusion and introduces extra memory round trips.
  - Bucket query counts (top four significant bits, at most 12.5% padding) and
    retain bucket buffers within the memory budget to avoid 150–400 ms graph
    recompiles.
  - Compile at optimization level 0 (level 1 gave no speedup and emitted MPS
    import errors).
  - Use zero-copy page-aligned unified memory (`PageAlignedTensors`) between CPU
    and GPU. Tie backing storage lifetime directly inside the graph struct
    (dropped last) with pointer-range assertions.
  - Wire FP16 ICL caches and attention under `auto` and `fast` for every model
    where accuracy holds; keep column and row attention in FP32 when FP16 adds
    error without saving time. Verify full output finiteness for FP16 saturation
    (>65504).
- **Apple Silicon CPU (Accelerate/AMX and NEON):**
  - Verify that every model's dense layers and large attention stages dispatch
    to Accelerate `cblas_sgemm` (AMX) above measured thresholds, rather than
    falling into unblocked NEON or scalar loops. Benchmark new models against
    PyTorch CPU on Apple Silicon.
  - Prefer performance cores (`hw.perflevel0.physicalcpu`) for executor threads.
  - Match portable reference error handling on `NaN` and zero-sum attention
    scores rather than propagating `NaN` silently.
- **Neural Engine:**
  - Do not rely on `MPSGraph` optimization level 1 for ANE placement. Pursue ANE
    only as an isolated Core ML experiment gated on end-to-end latency and FP16
    overflow bounds.

### x86-64 CPU (Zen 5 AVX-512 / AVX-512 BF16 and Portable)

- **Dispatch and tiling:**
  - Predicate SIMD dispatch on general alignment and multiples
    (`head_dim != 0 && head_dim % 16 == 0`), not model-specific constants.
  - Use padded packed-key tiles so non-multiple-of-16 tail lengths stay on
    AVX-512 instead of falling back to transpose or scalar paths.
  - Skip key packing and transposing when `query_rows < 8` to avoid wasted
    transpose work on small batches and single-row serving.
  - Prefer const-specialized row-unrolled microkernels (such as 16-row tiles) so
    accumulators stay in ZMM registers.
- **Precision and threading:**
  - Store ICL values in BF16 with FP32 keys and FP32 accumulation under `fast`
    on `AVX512_BF16` hosts. Avoid runtime BF16 weight packing unless
    measurements show a net end-to-end gain.
  - Keep scoped workers for independent attention heads and one persistent Rayon
    pool for dense chunks. Avoid nested Rayon pools, and scale chunk sizes
    dynamically when outer parallelism is small.
  - Gate every `#[target_feature]` `unsafe` kernel behind a length-checked safe
    wrapper.

### WebAssembly (`wasm32` / Pyodide)

- Build with `+simd128` (and evaluate relaxed SIMD where target runtimes support
  it).
- Use direct `v128` loads instead of assembling vectors from bounds-checked
  scalar reads.
- Verify that SIMD kernel symbols are present in the emitted `.wasm` binary
  (via symbol table or `wasm-objdump`); do not assume `cfg` dispatch fired.
- Propagate dispatcher `bool` return values cleanly, keep pool abstractions
  compatible with `wasm32` without atomics, and benchmark inside real Pyodide or
  Node runtimes rather than extrapolating from native ratios.

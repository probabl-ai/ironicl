# Kernel Design

## Kernel Inventory

### Required Portable Kernels

- FP32 matrix-vector and small matrix-matrix multiplication.
- Packed Q, K, and V projection.
- Output projection.
- LayerNorm.
- GELU.
- QASSMax base and query MLPs.
- RoPE.
- online stable attention.
- residual addition.
- classifier and regression decoders.
- FP32 ensemble reduction.

The initial scalar `dot_f32` and `softmax_f32` functions are contracts, not
performance implementations.

The current affine kernel dispatches to custom NEON on AArch64 and AVX-512F on
supported x86-64 hosts, with scalar fallback. On Apple Silicon, affine layers with
at least 16 rows and attention with at least 4096 scores per head call
Accelerate's `cblas_sgemm` instead; Accelerate runs them on the matrix
coprocessor. Batching the row stage across rows measured slower on an M1 Max, so
it stays per row.

With the `metal` cargo feature on Apple Silicon, each context gets two `MPSGraph`
graphs with its ensemble, feature, and training-row counts fixed. The predict
graph runs the column, row, and ICL stages and the decoder for every estimator in
one run, with only the query count unknown; one batched run beat the CPU on an M1
Max from a single query row, so prediction always uses it. Query counts are padded
to a bucket that keeps their top four significant bits, at most 12.5% padding, so
new batch sizes rarely trigger a 150–400 ms graph compile; buckets up to 1024 rows
keep their buffers between calls. The build graph runs the training rows through
every stage once and captures each column and ICL block's keys and values; its
graph setup costs about 250 ms, so contexts with fewer than 160 training rows
build on the CPU. The build graph runs as many estimators at once as fit Metal's
recommended working set, at most 2.2 GB per estimator at 8192 x 54, in equal
groups; `IRONICL_GPU_BUILD_BUDGET` sets the budget in bytes instead. Grouped
contexts take one graph run per group and a copy into the context, and a
context whose single estimator exceeds the budget builds on the CPU. Model
weights live once per model in shared GPU buffers and are fed to the graphs as
placeholders. The CPU only groups each feature with its neighbors and embeds
labels, and keeps any request whose grouped inputs contain the all-missing
marker. A GPU-built context writes its keys and values once into page-aligned
memory that the CPU reads as tensors and the GPU wraps without copying; a
CPU-built context is copied to the GPU once. `IRONICL_NO_METAL=1` disables the
GPU. Graphs compile at `MPSGraph` optimization level 0: level 1 was no faster,
and its pass that places operations on the Neural Engine or CPU printed "Failed
to import MPS module" on every compile. Under `optimize="auto"` or `"fast"`, the
build graph stores the ICL keys and values in FP16, and both graphs run ICL
attention in FP16, converting only the queries. Converting the stored FP16
caches to FP32 inside the graph instead wrote a full FP32 copy on every run and
was slower. BF16 measured slower and less accurate than FP16 on an M1 Max, and
FP16 column and row attention added error without saving time.

Cached attention selects its backend once per executor. The AVX-512 path
implements score tiles and softmax-weighted V tiles; portable execution retains
the FP32 reference path.

### Current Native FP32 Kernels

- FP32 AVX-512 FMA.
- cache-tiled online attention specialized for width 512, 8 heads, and head
  dimension 64.
- width-128 row and column kernels with 8 heads and head dimension 16.

The native module also contains approximate GELU and QASS modulation kernels,
but the accepted cached executor does not use them. It keeps the reference FP32
GELU and QASS arithmetic because the approximate path failed the latency gate.

### Planned Reduced-Precision Kernels

- BF16 dot product with FP32 accumulation through `VDPBF16PS`.
- INT8 VNNI projection kernels with INT32 accumulation.
- conversion and packing kernels at artifact-build boundaries.

## Online Attention

Do not allocate a complete `[queries, heads, training_rows]` score matrix.
Process K/V in training-row tiles while maintaining, per query and head:

- running maximum `m`;
- running exponential denominator `l`;
- running weighted-value accumulator `o`.

For each tile, update these values using the online-softmax rescaling rule. This
keeps scratch memory bounded and reads each K/V tile once per query microbatch.

All running values remain FP32.

The accepted Zen 5 tile is eight query rows by 128 source rows. For each head,
the native score kernel produces one row-major `[8,128]` tile. The online
softmax updates FP32 state, then the native V kernel loads each V row once and
updates all query accumulators. Query tile 16 and source tile 256 did not meet
the performance retention gate on the 1,024-row target.

## Query Microbatch

One K/V tile should serve multiple test queries before eviction. Candidate
microbatches are `1, 2, 4, 8, 16, 32`. The best value depends on register
pressure, cache size, memory bandwidth, and service latency goals.

The engine must report both:

- whole-microbatch latency;
- amortized latency per query.

## Weight Scheduling

The 12 ICL blocks dominate model parameters. A classifier ICL block is about:

- 8.18 MiB FP32;
- 4.09 MiB BF16;
- 2.05 MiB INT8.

Run layer-major so the current block's weights remain hot while all ensemble and
query work for that block completes. Do not attempt to keep all model weights
and K/V in L3 simultaneously.

## Packing

- Pack only during artifact build or first-load conversion.
- Use 64-byte aligned starts and padded dimensions where needed.
- Preserve a portable canonical representation when packed layouts depend on
  backend instructions.
- Record packing version independently from artifact format version.
- Benchmark K/V as separate streams versus interleaved blocks.

## Fusion Candidates

- LayerNorm plus Q projection.
- Q projection plus QASSMax query modulation.
- online softmax plus V accumulation. The current cached AVX-512 path fuses V
  tile traversal and multi-query accumulation, but softmax exponent calculation
  remains outside the native kernel.
- output projection plus residual addition.
- FFN second projection plus residual addition.

Do not fuse before the unfused FP32 path has layer-level parity fixtures. Fusion
must reduce memory traffic or launch overhead measurably.

## Threading

- Begin with one physical core and no SMT for kernel baselines.
- Scale across physical cores while measuring bandwidth saturation.
- Avoid a parallel reduction over the same output when independent head or
  ensemble partitions are available.
- Use scoped workers for independent attention heads.
- Use the persistent dense-work pool for cached affine, exact GELU, decoder, and
  QASS row chunks only.
- Keep request scratch buffers private to one job or disjoint output partition.
- Linux affinity is an optional policy and must not affect correctness.

# AMD Zen 5 Target

## Development Target

- CPU: Ryzen 7 9800X3D.
- Cores: 8 physical, 16 logical.
- L3: 96 MiB shared on one CCD.
- RAM: 64 GB nominal as two 32 GB DDR5 DIMMs, dual-channel; benchmarks report
  about 60 GiB available to the operating system.
- Storage: Gen5 NVMe.
- OS: Arch Linux.

Record the exact motherboard, BIOS/AGESA, kernel, compiler, DIMM part number,
configured transfer rate, timings, and power policy with every published
benchmark.

## ISA Policy

The optimized backend checks these capabilities at runtime:

- AVX2;
- AVX512F;
- AVX512BW;
- AVX512VL;
- AVX512_BF16 for BF16 policies;
- AVX512_VNNI for INT8 policies.

The public AMD product page lists AVX-512 but does not define every subset.
CPUID on the actual machine is authoritative. Ironicl must print detected
features before benchmarks.

## Compiler Policy

- Keep the portable binary runnable on non-Zen hosts.
- Use function-level runtime dispatch rather than compiling the whole binary
  with a host-only target.
- Benchmark stable Rust code generation against explicit kernels only after the
  safe portable implementation is correct.
- Record `rustc -Vv` and enabled target features.

Architecture intrinsics are isolated behind safe kernel contracts and runtime
feature checks. Each native kernel requires documented preconditions and scalar
differential tests.

The cached FP32 path dispatches AVX-512 kernels for affine operations, attention
score tiles, and softmax-weighted V tiles. Cached ICL attention assigns
independent heads to up to eight scoped workers. A dedicated persistent Rayon
pool parallelizes cached ICL affine, exact GELU, decoder, and QASS rows. Column
feature streams and query row interaction use the bounded scoped-worker path.

## L3 Strategy

The complete context does not fit in L3. The intended hot working set is:

- one layer's packed weights;
- one or more K/V tiles;
- query microbatch state;
- online-softmax accumulators;
- code and ordinary system activity.

The hardware controls cache residency. Ironicl cannot reserve the entire L3.
Use access order, compact storage, affinity, and bounded tiles rather than
claims that a buffer is pinned in cache.

## DDR5 Setup

- Install DIMMs in the motherboard's recommended two-DIMM slots.
- Enable the validated EXPO profile if stable.
- Confirm dual-channel operation and configured MT/s in Linux.
- Run a memory stability test before trusting model validation.
- Measure sustained read bandwidth instead of using peak transfer rate.

DDR5-6000 dual-channel has a 96 GB/s theoretical payload rate. A measured
streaming result is expected to be lower. Do not hard-code 80 GB/s as a machine
property.

## Linux Tuning Experiments

Measure, do not assume, the effects of:

- physical cores only versus SMT;
- `performance` versus default CPU-frequency policy;
- transparent huge pages versus explicit 2 MiB pages;
- prefaulted mmap versus buffered read into anonymous RAM;
- thread affinity;
- allocator choice;
- page-cache cold and warm starts;
- different K/V tile and query-microbatch sizes.

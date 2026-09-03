# Real-Dataset Scaling Benchmarks

## Hardware

- CPU: AMD Ryzen 7 9800X3D, 8 physical cores.
- RAM: 60 GiB.
- GPU: NVIDIA GeForce RTX 5080, 16 GiB.
- PyTorch: 2.13.0+cu130.
- Ironicl: release build with `-C target-cpu=znver5`.

The benchmark source and results are stored on the Ryzen host at:

```text
/home/semyon/Projects/ironicl-bench
```

Each historical request uses official TabICL preprocessing before it is encoded
as an `IRINPUT` bundle. Timed inference excludes preprocessing, model loading,
and input loading. CUDA samples synchronize before and after each inference.
Do not use the historical cross-engine rows below for end-to-end speedups.

## Matched Covertype Scaling

These requests use one ensemble member and 64 test rows except at `N=512`,
which uses 32 test rows. PyTorch CPU and Ironicl use eight Ryzen cores. CUDA
uses AMP. Large points use one timed iteration for CPU and Ironicl.

| Training rows | Features | Ironicl | PyTorch CPU | CUDA AMP | Ironicl peak RSS |
|---:|---:|---:|---:|---:|---:|
| 512 | 46 | 1.838 s | 284 ms | 12.1 ms | 206 MiB |
| 1,024 | 49 | 6.038 s | 535 ms | 15.9 ms | 255 MiB |
| 1,800 | 51 | 16.721 s | 1.021 s | 24.7 ms | 380 MiB |
| 2,048 | 52 | rejected | 1.169 s | 27.6 ms | admission only |

Ironicl rejects the 2,048-row request before allocation because its uncached
attention buffer would exceed 32 million FP32 scores. The exact constraint is:

```text
8 * total_rows * training_rows <= 32 * 1024 * 1024
```

The 60 GiB host cannot change this software limit.

## Wide Fashion-MNIST

The exporter keeps the highest-variance pixels before official TabICL
preprocessing. Each request uses one ensemble member.

| Training rows | Test rows | Features | Ironicl | CUDA AMP | Ironicl peak RSS | CUDA peak allocated |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 64 | 128 | 8.067 s | 33.6 ms | 345 MiB | 616 MiB |
| 512 | 32 | 256 | 4.834 s | 34.3 ms | 299 MiB | 627 MiB |

These points show the cost of row attention, which scales quadratically with
the feature count.

## MiniBooNE

For one ensemble member, 1,024 training rows, 64 test rows, and 50 features:

| Engine | Latency | Peak memory |
|---|---:|---:|
| Ironicl, 8 cores | 6.048 s | 267 MiB RSS |
| PyTorch CPU, 8 cores | 538 ms | 1.25 GiB RSS |
| CUDA AMP | 16.1 ms | 336 MiB allocated VRAM |

## CUDA Memory Saturation

These Covertype requests use eight official ensemble variants, 128 test rows,
and 53 or 54 retained features. PyTorch automatically reduces internal batch
sizes when allocations fail.

| Training rows | Latency | Peak RSS | Peak VRAM allocated | Peak VRAM reserved | Behavior |
|---:|---:|---:|---:|---:|---|
| 4,096 | 440 ms | 1.64 GiB | 6.18 GiB | 9.20 GiB | resident |
| 8,192 | 906 ms | 1.64 GiB | 8.84 GiB | 12.19 GiB | near VRAM pressure |
| 16,384 | 1.984 s | 1.64 GiB | 10.41 GiB | 12.95 GiB | allocation retries |
| 32,768 | 5.030 s | 1.72 GiB | 13.04 GiB | 13.31 GiB | allocation retries and batching |
| 65,536 | 16.324 s | 23.09 GiB | 7.77 GiB | 11.40 GiB | host staging dominates |
| 100,000 | 31.082 s | 37.12 GiB | 8.32 GiB | 10.08 GiB | host-memory pressure |
| 120,000 | 41.006 s | 37.42 GiB | 8.64 GiB | 11.78 GiB | safe benchmark endpoint |

The RTX 5080 reaches its practical resident limit between 8,192 and 32,768
training rows for `B=8`. TabICL continues by batching, which shifts pressure to
host RAM. Tests stop at 120,000 rows to keep adequate headroom on the 60 GiB
host.

## Numerical Checks

FP32 outputs remain close across engines on real requests. For Covertype at
`N=1024`, the output checksums are:

```text
Ironicl:       1073.207335
PyTorch CPU:  1073.210327
CUDA FP32:    1073.208862
```

CUDA AMP uses lower precision and is a throughput comparison, not the strict
FP32 parity mode.

## Cached FP32 Serving

The full-K/V executor was measured after the uncached results above. The exact
target uses one estimator, 1,024 training rows, 64 query rows, and 54 retained
Covertype features. The 1,024-row subset includes feature extrema so official
unique-feature filtering retains all 54 columns.

| Engine | Threads | Median latency |
|---|---:|---:|
| Ironicl cached FP32, scalar attention | 1 | 338 ms |
| Ironicl cached FP32, parallel features and rows | 8 | 364 ms before ICL parallelization |
| Ironicl cached FP32, head-parallel ICL | 8 | 111 ms |
| Ironicl cached FP32, AVX-512 score and V kernels | 8 | 55.9 ms |
| Ironicl cached FP32, fused attention tiles and parallel dense stages | 8 | 34.6 ms |
| Cached PyTorch FP32 reference | 1 | 92 ms |
| Cached PyTorch CUDA FP32 reference | GPU | 5.84 ms |

The final Ironicl result uses 50 timed iterations after five warmups. Its range
was 31.5 to 36.5 ms, with p95 at 36.3 ms. The maximum absolute difference from
official cached PyTorch logits was `3.05e-5`. The native score reduction changes
summation order, so cached parity uses an absolute tolerance of `1e-4`.

The final path selects its attention backend once, fuses score and weighted-V
work over an `8 x 128` query/source tile, and uses a dedicated persistent Rayon
pool for cached ICL affine, exact GELU, decoder, and QASS work. Scoped head
workers remain faster than Rayon for attention on this target. Full scratch
reuse, approximate activations, 16-row query tiles, 256-row key tiles, and a
single Rayon pool for all stages failed their latency gates and were not kept.

## Fair Raw-X Cached Serving

This is the only cross-engine cached-serving comparison in this document. The
training K/V cache is resident for both engines. Each sample starts with the
same raw numeric Covertype query and includes validation, TabICL numerical
encoding, ensemble variant generation, cached inference, class correction,
logit averaging, temperature softmax, and host FP32 probability output.

| Q | Engine | Precision | p50 | p95 | Query rows/s | Accuracy gate |
|---:|---|---|---:|---:|---:|---|
| 128 | PyTorch CPU, 16 threads | FP32 | 367.88 ms | 382.45 ms | 344.68 | baseline |
| 128 | Ironicl CPU, 16 threads | FP32 K/V, exact controls | 349.98 ms | 353.47 ms | 365.77 | max error `1.47e-5`, 100% labels |
| 128 | PyTorch RTX 5080 | FP32 | 31.78 ms | 32.20 ms | 4,016.81 | max error `1.52e-5`, 100% labels |
| 128 | PyTorch RTX 5080 | AMP | 17.79 ms | 18.11 ms | 7,156.75 | rejected: max error `4.13e-2`, 99.22% labels |
| 500 | PyTorch CPU, 16 threads | FP32 | 1,643.12 ms | 1,702.17 ms | 303.48 | baseline |
| 500 | Ironicl CPU, 16 threads | FP32 K/V, exact controls | 1,322.27 ms | 1,333.37 ms | 378.02 | max error `1.47e-5`, 100% labels |
| 500 | PyTorch RTX 5080 | FP32 | 104.49 ms | 104.65 ms | 4,782.73 | max error `2.00e-5`, 100% labels |

The shared fixture has `B=8`, `N=4096`, `Q=128`, 54 raw features, and 53
prepared features. It uses the official checkpoint, FP32 K/V caches, batch size
8, five warmups, and 50 timed samples. At `Q=128`, Ironicl is 1.051x faster
than PyTorch CPU and RTX 5080 FP32 is 11.01x faster than Ironicl. At `Q=500`,
Ironicl is 1.243x faster than PyTorch CPU and RTX 5080 FP32 is 12.65x faster
than Ironicl. AMP is not eligible for a serving ratio on this fixture.

## Conclusion

The uncached executor does not scale to large contexts. Full-K/V serving removes
the repeated training graph. Fair raw-X cached serving is only 5.1% faster than
PyTorch CPU at `N=4096,Q=128`; GPU FP32 remains much faster. Remaining
architecture work is:

1. Replace scoped head workers only when a lower-overhead persistent scheduler is available.
2. Fuse LayerNorm with Q projection and projection outputs with residual additions.
3. Add mmap-backed context loading for artifacts above 2 GiB.
4. Add BF16 K/V storage with FP32 accumulation for memory-bandwidth-bound contexts.

Microkernel tuning alone cannot close the large-context gap.

# Memory and I/O

## Symbols

- `N`: training rows.
- `H`: retained features.
- `B`: ensemble estimators.
- `s`: bytes per cache element.
- `E = 128`: column embedding width.
- `C = 4`: row CLS tokens.
- `D = C * E = 512`: ICL width.
- `L_col = 3`: column blocks.
- `M = 128`: inducing points.
- `L_icl = 12`: ICL blocks.

With `col_feature_group="same"`, the number of feature streams is `H + C`.

## Full-KV Cache

```text
column bytes = 2 * s * B * L_col * (H + C) * M * E
ICL bytes    = 2 * s * B * L_icl * N * D
total bytes  = s * B * [98,304 * (H + 4) + 12,288 * N]
```

The factor of two stores both K and V.

## Representation Cache

```text
column bytes = 2 * s * B * L_col * (H + C) * M * E
row bytes    = s * B * N * D
total bytes  = s * B * [98,304 * (H + 4) + 512 * N]
```

The row-dependent representation payload is 24 times smaller than ICL K/V, but
the engine must rerun all 12 ICL blocks for each request.

## Reference Sizes

For `H=100`, `B=8`:

| Rows | FP32 full K/V | BF16 full K/V | FP16 full K/V | INT8 payload |
|---:|---:|---:|---:|---:|
| 1,000 | 0.67 GiB | 0.34 GiB | 0.34 GiB | 0.17 GiB |
| 10,000 | 3.97 GiB | 1.98 GiB | 1.98 GiB | 0.99 GiB |
| 48,000 | 17.88 GiB | 8.94 GiB | 8.94 GiB | 4.47 GiB |
| 50,000 | 18.62 GiB | 9.31 GiB | 9.31 GiB | 4.65 GiB |
| 100,000 | 36.93 GiB | 18.46 GiB | 18.46 GiB | 9.23 GiB |

FP16 uses the same two bytes per element as BF16; it is the Apple Silicon GPU
cache policy under `optimize=auto`/`fast`. The table assumes one storage type for the whole
context; in practice only the ICL caches take the reduced type and the column
caches stay FP32, which adds `0.31 GiB` at `H=100`, `B=8`. INT8 values exclude scale metadata and
alignment. The planner currently reports payload bytes only.

## RAM Budget

Add these allocations to cache payload:

- packed weights;
- preprocessing and ensemble state;
- query and output buffers;
- online-attention scratch;
- artifact metadata and quantization scales;
- allocator and alignment overhead;
- page cache if using mmap;
- operating system and other processes.

On the nominal 64 GB host, which exposes about 60 GiB to the operating system,
50K-row FP32 and BF16 contexts fit comfortably. A 100K-row FP32 context fits by
payload but leaves less room for duplicated build buffers. The context builder
must stream into preallocated final storage.

## GPU Build Memory

On Apple Silicon with the `metal` feature, the Metal build graph needs a transient
working set beyond the final context payload: about 2.2 GB per estimator at
8192 x 54, and about 4 GB at 10,000 x 100. The [memory planner](memory-planner.md)
sizes the groups of estimators built per graph: the largest group that divides
the estimator count and stays under the smallest of 6 GiB, half the available
memory, and Metal's recommended working set. Each group is copied into the
context. Only an estimator larger than the working set builds on the CPU.
`IRONICL_GPU_BUILD_BUDGET` overrides the plan with an exact byte cap, building
on the CPU when one estimator exceeds it.

Before the planner, groups filled the whole working set, which on unified memory
is most of the machine's RAM: at 10,000 x 100 on an M1 Max (32 GB) the build
peaked at 21.4 GiB for a 2 GiB context. It now peaks at 7.1 GiB, with fit 8%
slower; `tests/build_memory.rs` guards it. Predict-time GPU query buffers are kept
between calls for buckets up to 1024 rows, and large cached queries are split into
chunks so GPU scratch stays bounded.

## Bandwidth Lower Bound

For strict single-query full-KV inference:

```text
ideal scan seconds = cache bytes / sustained bytes per second
```

At 50K rows, BF16, 100 features, and eight estimators, the cache is
9,993,977,856 bytes. At 80 GB/s, one ideal scan takes 124.9 ms. This is not a
latency prediction: it excludes computation, imperfect bandwidth, cache
conflicts, synchronization, and other traffic.

A microbatch can reuse one K/V stream for multiple queries. Divide the scan
cost by microbatch size only as an ideal amortized floor; compute eventually
becomes dominant.

## Gen5 NVMe

Use NVMe for:

- durable static artifacts;
- quick cold loading;
- artifacts larger than available RAM as an explicit degraded mode;
- benchmark fixtures and profiling output.

Do not use NVMe as the normal per-query K/V source when a context fits in RAM.
Even fast Gen5 NVMe is much slower and has higher latency than dual-channel RAM.

Measure these startup modes:

1. buffered read into preallocated RAM;
2. mmap plus sequential prefault;
3. warm page-cache mmap;
4. cold page-cache mmap.

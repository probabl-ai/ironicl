# Ironicl

Ironicl is an experimental, cache-aware CPU inference engine for TabICLv2. It
targets AMD Zen 5 systems, starting with the Ryzen 7 9800X3D, while retaining a
portable reference path that builds on Apple Silicon.

The project currently provides:

- exact TabICLv2 classifier and regressor parameter counts;
- safe, pure-Rust inspection of PyTorch ZIP checkpoints;
- full-KV and representation-cache memory planning;
- runtime CPU capability reporting;
- backend and precision-policy validation;
- portable reference kernels and classifier execution from model artifacts;
- exact FP32 full-K/V static-context export and cached query execution;
- AVX-512 cached attention and dense-stage parallelism on the Ryzen target;
- reference design documents for the engine and static-context artifacts.

It can write and verify portable Ironicl model artifacts. `run` accepts a
validated binary classifier input bundle and prints classifier logits.

## Quick Start

```bash
cargo test
cargo run -- platform
cargo run -- inspect --checkpoint model.ckpt
cargo run -- plan --rows 50000 --features 100 --estimators 8 \
  --weights bf16 --kv-dtype bf16 --cache kv --bandwidth-gbps 80
```

The planned model commands use the same explicit source convention:

```bash
ironicl inspect --checkpoint model.ckpt
ironicl import --checkpoint model.ckpt --output model.ironicl
ironicl run --checkpoint model.ckpt --input request.irinput
ironicl run --model model.ironicl --input request.irinput
ironicl run --model model.ironicl --context context.ironiclctx --input request.irquery
ironicl benchmark --model model.ironicl --input request.irinput --warmup 2 --iterations 10
```

Checkpoint and model inspection work now. `import` creates a verified portable
model. `run --checkpoint` creates or reuses a content-addressed model cache,
loads the bundle, and executes the portable FP32 classifier. `--no-cache` is
not available for execution because it does not create an artifact.

## Benchmarking

Build the release binary and benchmark a resident imported model:

```bash
cargo run --release -- benchmark --model model.ironicl --input request.irinput
```

`benchmark` times only prediction calls after bundle and artifact loading and
executor construction complete. It excludes preprocessing, checkpoint import,
conversion, and load time from inference latency. See
[`docs/benchmarking.md`](docs/benchmarking.md) for the output contract and
[`docs/pytorch-cpu-comparison.md`](docs/pytorch-cpu-comparison.md) for the
point-by-point comparison with official PyTorch CPU and the failed optimization
record.

On the M1 Max, `auto` selects the portable backend. On the target Arch Linux
machine, it selects the Zen 5 backend only when all required AVX-512 features
are detected for the chosen storage formats.

## Design

Start with [`docs/README.md`](docs/README.md). The key decisions are:

- precompute all training-dependent state for a static training set;
- persist model and context artifacts on NVMe;
- load the current owned artifact into RAM before serving, with mmap as future
  work;
- process exact test queries against fixed training K/V;
- evaluate BF16 cache storage with FP32 accumulation as the next
  large-context path;
- keep FP32 preprocessing, LayerNorm, QASSMax, RoPE, softmax, GELU, residuals,
  decoder output, and ensemble reduction;
- treat INT8 K/V as an accuracy experiment, not as the baseline.

## Upstream

Ironicl is a new implementation informed by the BSD-3-Clause TabICL project.
See [`docs/references.md`](docs/references.md) before copying or distributing
upstream code or model artifacts.

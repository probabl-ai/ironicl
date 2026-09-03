# Goals and Scope

## Goal

Build a CPU inference engine for released TabICLv2 checkpoints that minimizes
latency and maximizes throughput when the training table is static and test rows
arrive repeatedly.

The first optimized target is:

- AMD Ryzen 7 9800X3D;
- Arch Linux;
- 64 GB dual-channel DDR5;
- Gen5 NVMe;
- TabICLv2 classification with at most 10 classes;
- a fixed feature schema and training context;
- one-row latency and 8-16-row microbatch throughput.

The project must continue to build and run correctness tests on macOS and Apple
M1 Max through portable Rust code.

## Required Outcomes

1. Inspect and import a released FP32 TabICLv2 checkpoint in pure Rust.
2. Build a deterministic static-context artifact from fitted preprocessing,
   training features, training targets, and ensemble configuration.
3. Produce predictions that pass the validation policy in `validation.md`.
4. Dispatch optimized kernels only after runtime CPU capability checks.
5. Fall back to a portable FP32 implementation on unsupported hosts.
6. Report memory, bandwidth, and precision choices before allocating a context.
7. Avoid hidden allocations in the steady-state request path.

## Initial Non-Goals

- Pretraining or fine-tuning TabICLv2.
- CUDA, ROCm, Metal, or NPU execution.
- More than 10 classifier classes in the cached path.
- Exact online insertion or deletion of training rows.
- Bitwise parity across PyTorch, operating systems, or CPU architectures.
- Direct NVMe reads for every query when the cache fits in RAM.
- An independent replacement for all pandas and scikit-learn input semantics in
  the first milestone.
- A Python package or Python runtime dependency.
- Broad unsafe model logic. Unsafe code is isolated in native kernel modules
  where architecture intrinsics require it.

## Success Metrics

Correctness precedes speed. The first performance targets are provisional and
must be replaced by measurements on the 9800X3D:

- FP32 portable output within the agreed PyTorch tolerance.
- BF16-storage output with no material benchmark accuracy regression.
- No request-time training-table transformation.
- No request-time model or context file parsing.
- Sustained RAM bandwidth close to a purpose-built streaming benchmark.
- Query microbatching that demonstrates K/V reuse rather than rereading K/V for
  each query.

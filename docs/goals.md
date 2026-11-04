# Goals and Scope

## Goal

Build an inference engine for released TabICLv2 checkpoints that minimizes
latency and maximizes throughput when the training table is static and test rows
arrive repeatedly. It runs on the compute built into the host, with no PyTorch:

- CPU vector units on every host;
- on Apple Silicon, also the matrix coprocessor (through Accelerate) and the
  integrated GPU (through Metal). These share the CPU's memory, so a training
  context needs no copy to reach them;
- on Apple Silicon, the Neural Engine (through Core ML) as an opt-in experiment.
  It computes in FP16, so it needs its own accuracy gate, like the BF16 variants:
  full label agreement and a bounded probability error, not the FP32 `1e-4`
  tolerance. FP16 overflows above 65504, which BF16 does not, so the gate must
  cover attention scores and activations.

The first optimized target is:

- AMD Ryzen 7 9800X3D;
- Arch Linux;
- 64 GB dual-channel DDR5;
- Gen5 NVMe;
- TabICLv2 classification with at most 10 classes;
- a fixed feature schema and training context;
- one-row latency and 8-16-row microbatch throughput.

Apple Silicon (M1 Max and later) is the second target and is implemented: it uses
the matrix coprocessor through Accelerate, the integrated GPU through Metal, and
NEON CPU kernels, with FP16 under the `optimize` levels. Intel Macs are not a
target and use the portable path. The project must also build and run correctness
tests on other hosts through portable Rust code.

## Required Outcomes

1. Inspect and import a released FP32 TabICLv2 checkpoint in pure Rust.
2. Build a deterministic static-context artifact from fitted preprocessing,
   training features, training targets, and ensemble configuration.
3. Produce predictions that pass the validation policy in `validation.md`.
4. Dispatch optimized kernels only after runtime CPU capability checks, and the
   Metal GPU path only when the `metal` feature is built and `IRONICL_NO_METAL`
   is unset.
5. Fall back to a portable FP32 implementation on unsupported hosts.
6. Report memory, bandwidth, and precision choices before allocating a context.
7. Avoid hidden allocations in the steady-state request path.

## Initial Non-Goals

- Pretraining or fine-tuning TabICLv2.
- Discrete-GPU runtimes such as CUDA and ROCm.
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

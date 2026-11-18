# Source Layout

How `src/` is organized, what is still planned, and how CUDA builds choose
their GPU targets.

## Rule

- Kernels and operations live under `kernels/<device>/`: `cpu`, `metal`, `cuda`.
  They know nothing about a model.
- Each model's graph lives under `models/<model>/`, one file or directory per
  device: `models/tabicl_v2/{cpu.rs, metal.rs, cuda/}`.
- Artifact formats live under `format/`. Shared infrastructure (configuration,
  errors, memory planning, progress) stays at the crate root.

The split leaves room for specialized kernels per instruction set and per GPU
architecture, and for more models.

## Tree

Entries marked *planned* do not exist yet.

```text
src/
├── lib.rs, main.rs, cli.rs        crate root and the `ironicl` CLI
├── config.rs, error.rs            ExecutionConfig and friends; Error
├── memory.rs                      memory planning and the context cache estimate
├── progress.rs, profile.rs        progress and cancellation; `profile` timers
├── benchmark.rs                   benchmark statistics
├── python.rs                      PyO3 bindings (`python` feature)
├── kernels/
│   ├── reference.rs               portable reference kernels
│   ├── cpu/
│   │   ├── mod.rs                 Isa: the instruction set, detected once
│   │   ├── ops/                   affine, attention, norm, pointwise
│   │   ├── arch/                  x86_64 (AVX-512), aarch64 (NEON, Accelerate),
│   │   │                          wasm32 (SIMD128)
│   │   └── tests.rs
│   ├── metal/mod.rs               device, shared buffers, MPSGraph plumbing
│   └── cuda/
│       ├── mod.rs                 Device, Arch, the embedded modules and their choice
│       └── kernels.rs             the #[cuda_module] kernels
├── models/
│   └── tabicl_v2/
│       ├── cpu.rs                 Executor, StaticExecutor
│       ├── execution_plan.rs      worker partitioning
│       ├── metal.rs               weights, graph builder, predict graph, build
│       ├── cuda/                  build, predictor, launchers, weights, cache, …
│       ├── checkpoint.rs          checkpoint import
│       ├── checkpoint_cache.rs    prepared checkpoints
│       ├── quantiles.rs           regression outputs
│       └── preprocess.rs          planned: TabICL's preprocessing pipeline
├── format/
│   ├── mod.rs                     container primitives
│   ├── model.rs                   model artifacts
│   ├── static_context.rs          static contexts
│   └── input_bundle.rs, query_bundle.rs
├── preprocess/                    planned: pure-Rust fit + transform
│                                  (power, scale, encode, random)
├── server/                        planned: `server` feature, off by default
└── bin/ironicl-server.rs          planned: [[bin]] with required-features = ["server"]
```

## Steps

Each step is one pull request.

| Step | Change | Status |
|---|---|---|
| A | Move files into this layout: the CPU kernel split, `cache.rs` into `memory.rs`, the Metal and CUDA splits by file, `models/tabicl_v2/`, `format/`. No logic changes. | Done |
| B | Consolidate: fold `platform.rs` and `backend.rs` into `kernels/cpu` (`Isa`); one `Device` enum for TabICLv2; separate the Metal graph builder's generic operations from the TabICLv2 stages. | Planned |
| C | Split the CUDA kernels: confirm that cuda-oxide compiles device helpers outside `#[cuda_module]`, then split `kernels/cuda/kernels.rs` into `ops/` and `arch/`, with TabICLv2's fused kernels in `ops/tabicl_v2.rs`. Move the launchers into `kernels/cuda` once they no longer take TabICLv2 weight types. Test on an NVIDIA GPU. | Planned |
| D | Rust preprocessing: `preprocess/` and `models/tabicl_v2/preprocess.rs`, fit and transform, at parity with TabICL. | Planned |
| E | Prediction server behind the `server` feature: `/health`, `/v1/models`, and `predict` and `predict_proba` per model, in JSON, for models loaded read-only at startup. Needs D, so it runs without Python. | Planned |

## CUDA build targets

cuda-oxide's backend builds one GPU target per crate build and writes PTX.
`IRONICL_CUDA_ARCH` lists the targets a build embeds, `sm_80,sm_89` by default:

```bash
. tools/cuda-env.sh
cargo build --release --features cuda                         # sm_80 and sm_89
IRONICL_CUDA_ARCH=sm_120 cargo build --release --features cuda # one GPU
```

- `build.rs` builds the first target as this build's, through
  `CUDA_OXIDE_TARGET`, and each further target as a nested build of the crate
  in its own directory under `OUT_DIR`. It embeds the nested builds' PTX and
  writes the module list to `OUT_DIR/cuda_modules.rs`.
- One table in `build.rs` lists the accepted targets and their capabilities.
  Each build sets an `ironicl_cap_*` cfg per capability; kernels test
  capabilities, never architecture names, because newer GPUs do not always have
  older ones' instructions (`wgmma` exists only on `sm_90a`). Today the only
  capability is `ironicl_cap_fp8_cvt`, FP8 conversion in hardware, from
  `sm_89` on.
- A device loads the most specific module for it: an `a` target (that compute
  capability only), then an `f` target (later GPUs of the same major version),
  then the newest plain target at or below its capability. A module whose PTX
  the driver is too old for gives way to the next. `IRONICL_CUDA_MODULE=<target>`
  restricts the choice to one module, for testing.
- Builds embed PTX only; the driver compiles it for the GPU when it loads. A
  target joins the default only when a capability changes its kernels:
  `sm_89` speeds up `optimize="fast"` prediction by 2–9% on an RTX 5080, more
  with more training rows.

## Possible future

Not planned or ready; recorded so the layout can take them.

- **Vulkan** (AMD, Intel Arc): `kernels/vulkan/{ops,arch}`, keyed by subgroup
  size, cooperative matrices, and FP16; `models/tabicl_v2/vulkan/`. The shader
  language is open: rust-gpu, GLSL, or WGSL through wgpu.
- **OpenVINO** (Intel CPU, integrated GPU, NPU): a graph runtime like MPSGraph,
  with op builders in `kernels/openvino/` and `models/tabicl_v2/openvino.rs`.
  Risks: the Rust bindings may only load IR or ONNX models, and cached
  attention may not map well.
- **ROCm/HIP** as an alternative to Vulkan on AMD: `kernels/rocm/arch/` by
  `gfx` target.
- More models under `models/`, such as Kumo-Tabular.

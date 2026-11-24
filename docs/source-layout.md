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
- Preprocessing that several models can share lives under `preprocess/`; each
  model's own pipeline lives with the model.

The split leaves room for specialized kernels per instruction set and per GPU
architecture, and for more models.

## Tree

Entries marked *planned* do not exist yet.

```text
src/
├── lib.rs, main.rs, cli.rs        crate root and the `ironicl` CLI, `serve` included
├── config.rs, error.rs            TaskKind and StorageDType; Error
├── memory.rs                      memory planning
├── progress.rs, profile.rs        progress and cancellation; `profile` timers
├── python.rs                      PyO3 bindings (`python` feature)
├── kernels/
│   ├── reference.rs               portable reference kernels and test oracles
│   ├── cpu/
│   │   ├── mod.rs                 Isa (the instruction set), CpuCapabilities
│   │   ├── ops/                   affine, attention, norm, pointwise
│   │   ├── arch/                  x86_64 (AVX-512), aarch64 (NEON, Accelerate),
│   │   │                          wasm32 (SIMD128)
│   │   └── tests.rs
│   ├── metal/
│   │   ├── mod.rs                 device, shared buffers, model weight buffers
│   │   ├── builder.rs             MPSGraph builder and generic operations
│   │   └── block.rs               the transformer block both models build
│   └── cuda/
│       ├── mod.rs                 Device, Arch, the embedded modules and their choice
│       ├── kernels.rs             the #[kernel] entry points, in one #[cuda_module]
│       ├── launch.rs              the launchers any model's blocks chain
│       └── ops/                   their device code: attention, block (norms,
│                                  packing heads), fp8, linear, tabicl_v2 (the
│                                  fused block and column kernels)
├── models/
│   ├── kumo_tabular/
│   │   ├── mod.rs                 sizes, released checkpoints, tensor manifests
│   │   ├── preprocess.rs          NVIDIA's recipe and the random draws
│   │   ├── cpu.rs                 the forward pass and a member's context
│   │   ├── cuda.rs                both on an NVIDIA GPU, the context resident
│   │   ├── metal.rs               both on an Apple Silicon GPU, as MPSGraph graphs
│   │   └── fitted.rs              the fitted ensemble
│   └── tabicl_v2/
│       ├── mod.rs                 Device: where a context predicts
│       ├── cpu.rs                 Executor, StaticExecutor
│       ├── execution_plan.rs      worker partitioning
│       ├── metal.rs               TabICLv2 stages, predict graph, build
│       ├── cuda/                  build, predictor, launchers, weights, cache, …
│       ├── checkpoint.rs          checkpoint import
│       ├── checkpoint_cache.rs    prepared checkpoints
│       ├── quantiles.rs           regression outputs
│       ├── request.rs             QueryRequest and Request: rows for a prediction
│       ├── preprocess.rs          TabICL's preprocessing pipeline
│       ├── ensemble.rs            its ensemble: groups, feature and class orders
│       ├── fitted.rs              the fitted model
│       └── directory.rs           model directories, of either model: export,
│                                  prepare, inspect
├── format/
│   ├── mod.rs                     container primitives
│   ├── model.rs                   model artifacts (IRMODEL)
│   ├── static_context.rs          static contexts (IRCNTXT)
│   └── preprocessor.rs            fitted preprocessing (IRPREP)
├── preprocess/                    what must reproduce Python exactly; no `unsafe`:
│                                  python_random, scale, yeo_johnson, labels, columns
└── server/                        `ironicl serve` (`server` feature, off by default):
                                   mod.rs startup, api.rs handlers, request.rs,
                                   response.rs, error.rs, worker.rs batching
```

## Steps

Each step is one pull request.

| Step | Change | Status |
|---|---|---|
| A | Move files into this layout: the CPU kernel split, `cache.rs` into `memory.rs`, the Metal and CUDA splits by file, `models/tabicl_v2/`, `format/`. No logic changes. | Done |
| B | Consolidate: fold `platform.rs` and `backend.rs` into `kernels/cpu` (`Isa`); one `Device` enum for TabICLv2; separate the Metal graph builder's generic operations from the TabICLv2 stages. | Done |
| C | Split the CUDA kernels: their device code moves into `kernels/cuda/ops/`, with TabICLv2's fused kernels in `ops/tabicl_v2.rs`. The `#[kernel]` entry points stay in `kernels.rs`, because cuda-oxide generates launchers only for kernels written inline in the `#[cuda_module]` block; it compiles every function reachable from a kernel, wherever it lives. The PTX is unchanged apart from symbol names. | Done |
| C2 | Move the generic launchers into `kernels/cuda/launch.rs` (dense products, norms, attention over cached keys and values, packing queries into heads, copies), with the types they take: `Precision`, the activation forms, the dense and norm weights, and the kernel constants. Their kernels take any width and head count, `RoPE` and a per-head norm when packing heads, and key and value heads shared by groups of query heads, so other models' blocks chain them. The launchers of TabICLv2's fused kernels stay with the model. | Done |
| D | Rust preprocessing: `preprocess/` and `models/tabicl_v2/preprocess.rs`, fit and transform, at parity with TabICL, and `models/tabicl_v2/fitted.rs`, the fitted model. | Done |
| E | Prediction server behind the `server` feature: `ironicl serve DIR` for one model directory, `POST /v1/predict` and `GET /v1/health` in JSON, with batching; see [server.md](server.md). Needs D, so it runs without Python. | Done |

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
  `sm_89` on. Capability-gated code sits next to its portable version, as the FP8
  conversion does in `kernels/cuda/ops/fp8.rs`; a `kernels/cuda/arch/`
  directory waits for kernels that exist only on some targets.
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
- More models under `models/`. Kumo-Tabular is in progress ([kumo-tabular.md](kumo-tabular.md)).

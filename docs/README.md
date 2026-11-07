# Reference Design Index

These documents define the Ironicl design. They separate facts derived from the
current TabICLv2 implementation from hypotheses that require measurement. The two
measured backends are the Ryzen 7 9800X3D CPU path and the Apple Silicon
(Accelerate, NEON, and Metal GPU) path; the [experiments log](experiments.md)
records what each has verified.

| Document | Purpose |
|---|---|
| [goals.md](goals.md) | Scope, non-goals, and success criteria |
| [model.md](model.md) | TabICLv2 architecture and exact row-streaming boundary |
| [architecture.md](architecture.md) | Engine components and request lifecycle |
| [cli.md](cli.md) | Stable command and model-source conventions |
| [python.md](python.md) | Python preprocessing package |
| [native-preprocessing.md](native-preprocessing.md) | Pure-Rust preprocessing plan and parity surfaces |
| [static-context.md](static-context.md) | Training-time precomputation and invalidation |
| [artifact-format.md](artifact-format.md) | Persistent model and context artifact design |
| [precision.md](precision.md) | FP32, BF16, and INT8 numerical policy |
| [kernels.md](kernels.md) | Kernel inventory, layouts, and fusion boundaries |
| [amd-zen5.md](amd-zen5.md) | Ryzen 7 9800X3D target and runtime dispatch |
| [apple-development.md](apple-development.md) | M1 Max development workflow and limitations |
| [memory-and-io.md](memory-and-io.md) | RAM sizing, bandwidth model, and NVMe use |
| [validation.md](validation.md) | PyTorch parity and accuracy gates |
| [benchmarking.md](benchmarking.md) | Reproducible performance methodology |
| [experiments.md](experiments.md) | Authoritative experiments-and-tests log: retained and rejected work (CPU and Apple Silicon), acceptance gates, and claim-to-test mapping |
| [real-dataset-benchmarks.md](real-dataset-benchmarks.md) | Raw real-dataset scaling measurements the experiments log cites |
| [pytorch-cpu-comparison.md](pytorch-cpu-comparison.md) | Cross-engine PyTorch CPU comparison and failed optimization results |
| [memory-planner.md](memory-planner.md) | How Ironicl plans memory across host and device pools, and the planner's phases |
| [security.md](security.md) | Artifact parsing and untrusted-input requirements |
| [roadmap.md](roadmap.md) | Ordered implementation milestones |
| [references.md](references.md) | Upstream sources, checkpoints, and licenses |

## Decision Language

- **Required**: part of the accepted design.
- **Candidate**: intended for implementation but still needs measurement.
- **Experiment**: must not become the default without accuracy and performance
  evidence.
- **Out of scope**: deliberately excluded from the first engine.

When code and these documents disagree, tests and measured behavior take
priority. Update the document in the same change that alters a design contract.

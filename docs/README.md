# Reference Design Index

These documents define the initial Ironicl design. They separate facts derived
from the current TabICLv2 implementation from hypotheses that require
measurement on the Ryzen 7 9800X3D system.

| Document | Purpose |
|---|---|
| [goals.md](goals.md) | Scope, non-goals, and success criteria |
| [model.md](model.md) | TabICLv2 architecture and exact row-streaming boundary |
| [architecture.md](architecture.md) | Engine components and request lifecycle |
| [cli.md](cli.md) | Stable command and model-source conventions |
| [static-context.md](static-context.md) | Training-time precomputation and invalidation |
| [artifact-format.md](artifact-format.md) | Persistent model and context artifact design |
| [precision.md](precision.md) | FP32, BF16, and INT8 numerical policy |
| [kernels.md](kernels.md) | Kernel inventory, layouts, and fusion boundaries |
| [amd-zen5.md](amd-zen5.md) | Ryzen 7 9800X3D target and runtime dispatch |
| [apple-development.md](apple-development.md) | M1 Max development workflow and limitations |
| [memory-and-io.md](memory-and-io.md) | RAM sizing, bandwidth model, and NVMe use |
| [validation.md](validation.md) | PyTorch parity and accuracy gates |
| [benchmarking.md](benchmarking.md) | Reproducible performance methodology |
| [pytorch-cpu-comparison.md](pytorch-cpu-comparison.md) | Point-by-point PyTorch CPU comparison and failed optimization results |
| [cached-fp32-optimization.md](cached-fp32-optimization.md) | Retained cached FP32 optimizations and rejected experiments |
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

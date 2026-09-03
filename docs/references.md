# References and Provenance

## Local Upstream Checkout

The adjacent project is expected at `../tabicl` during development.

Important source references:

- `../tabicl/src/tabicl/_model/tabicl.py`: top-level architecture and cache path.
- `../tabicl/src/tabicl/_model/embedding.py`: feature grouping and column stage.
- `../tabicl/src/tabicl/_model/interaction.py`: row-transformer stage.
- `../tabicl/src/tabicl/_model/learning.py`: ICL and decoder stages.
- `../tabicl/src/tabicl/_model/layers.py`: attention blocks and cache behavior.
- `../tabicl/src/tabicl/_model/attention.py`: Q/K/V projection and SDPA.
- `../tabicl/src/tabicl/_model/ssmax.py`: QASSMax.
- `../tabicl/src/tabicl/_model/rope.py`: rotary encoding.
- `../tabicl/src/tabicl/_model/kv_cache.py`: cache shape and accounting.
- `../tabicl/src/tabicl/_sklearn/preprocessing.py`: preprocessing and ensembles.
- `../tabicl/tests/test_sklearn.py`: existing cached-inference tolerances.

## Published Checkpoints

Hugging Face repository: `jingang/TabICL`.

- `tabicl-classifier-v2-20260212.ckpt`
- `tabicl-regressor-v2-20260212.ckpt`

Pin and verify the complete SHA-256 digest before conversion. Do not identify an
artifact only by a mutable repository branch or filename.

- Classifier SHA-256:
  `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`
- Regressor SHA-256:
  `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`

## Minimal Architecture Reference

NanoTabICL provides a concise educational implementation:

- <https://github.com/soda-inria/nanotabicl>

Use the full TabICL checkout and released checkpoint configuration as the
behavioral authority when they differ.

## Checkpoint Parser

The offline pure-Rust checkpoint path uses Candle's PyTorch pickle reader:

- <https://github.com/huggingface/candle>
- `candle-core` version `0.9.1`;
- MIT or Apache-2.0 license;
- CPU-only default feature set.

Candle's parser exposes the parsed pickle object tree without executing Python
code. Ironicl applies a bounded opcode preflight, reads the primitive `config`
mapping from that tree, and decodes tensor metadata with checked patterns. It
treats released-checkpoint tensor counts, element counts, payload bytes, model
configuration, and independent hashes as mandatory compatibility gates. Both
official v2 checkpoints pass these gates with Candle 0.9.1.

## Papers

- Qu et al., "TabICL: A Tabular Foundation Model for In-Context Learning on
  Large Data," ICML 2025, arXiv:2502.05564.
- Qu et al., "TabICLv2: A better, faster, scalable, and open tabular foundation
  model," arXiv:2602.11139, 2026.

## Hardware

- AMD Ryzen 7 9800X3D product specification:
  <https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-7-9800x3d.html>
- Runtime CPUID from the development target is authoritative for instruction
  subsets.

## License

The local TabICL code uses the BSD 3-Clause License, with separate terms noted
for its forecast-derived code. Ironicl does not need forecast code.

Before copying upstream code:

1. retain the required copyright and license notices;
2. identify modified source files;
3. include the applicable license in distributions;
4. verify the model checkpoint license and deployment terms independently.

The current Ironicl skeleton contains a fresh implementation and references
upstream behavior. Add a project license before public distribution.

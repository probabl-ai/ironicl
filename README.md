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

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, local checks, and
CI coverage limits.

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

## Python Preprocessing

The standalone preprocessing package requires Python 3.12+, NumPy, scikit-learn,
and its SciPy dependency. It does not depend on TabICL or PyTorch:

```bash
python -m pip install .
```

```python
from ironicl.preprocessing import ClassifierPreprocessor

preprocessor = ClassifierPreprocessor(n_estimators=8, random_state=42)
training_groups = preprocessor.fit_transform(X_train, y_train)
training_labels = preprocessor.encode_labels(y_train)
query_groups = preprocessor.transform(X_query)
```

Each returned group is one contiguous FP32 matrix `[rows, filtered_features]`.
`group_names_` records normalization order. `ensemble_groups_` maps ensemble
members to those matrices; `feature_permutations_` and `class_permutations_`
provide their feature and encoded-label mappings. The API does not expand
features or labels per ensemble. `n_estimators_` records the actual ensemble
count, which can be smaller than requested.

This initial implementation supports finite dense numeric inputs, 2–10 classes,
and the default `none`/`power` transforms with Latin feature and cyclic class
permutations. It removes constant features and rejects all-constant training
data. Categorical inputs, missing values, and sparse matrices are not supported.
Numeric DataFrames must have unique string column names in the same order at
prediction time. Preprocessing is fitted on training data only, retains no
training rows, and does not require `PYTHONHASHSEED` configuration. It preserves
upstream's power-transform fallback, including its batch-wide clipping when a
transform raises a domain error.

**This is a preprocessing-only Python package, not yet a Python classifier.**
PyO3 bindings and Rust training-context construction remain to be implemented.
`tools/prepare_real_dataset.py` now uses this package without `--tabicl-root`;
it expands ensembles only when writing the existing IRINPUT file format.
Other upstream oracle and context-export tools still require TabICL/PyTorch.
The current pure-Python build backend will be replaced by maturin when the
native extension is added.

Run the Python checks after installation:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The checked-in preprocessing oracle runs without upstream dependencies. The
supported scikit-learn 1.9.x / SciPy 1.18.x combination preserves its power-fit
behavior; older float32 solvers produce different results. NumPy 2.x is required.
Run live comparisons separately, in an environment with both projects installed
and an unmodified TabICL checkout at commit
`bd030d47ff7d9f987f0f33431d743a3003bfe0d3`:

```bash
IRONICL_TABICL_SOURCE=/path/to/tabicl/src \
  python -m unittest discover -s tests/parity -p 'test_*.py' -v
```

This dedicated suite compares independently fitted preprocessing chains, using
synthetic inputs and bundled scikit-learn datasets. Missing prerequisites fail
rather than skip. It tests preprocessing, not classifier predictions; see
[`docs/validation.md`](docs/validation.md#python-preprocessing-parity) for setup,
coverage, and numerical requirements.

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

The Rust engine is a new implementation informed by the BSD-3-Clause TabICL
project. Python preprocessing adapts the upstream algorithms; its source records
provenance and distributions include the applicable license in `LICENSES/`.
See [`docs/references.md`](docs/references.md) before copying or distributing
upstream code or model artifacts. The project-wide license is still undecided;
the third-party notice does not license the rest of Ironicl.

# Ironicl

[![Rust checks](https://github.com/probabl-ai/ironicl/actions/workflows/rust.yml/badge.svg)](https://github.com/probabl-ai/ironicl/actions/workflows/rust.yml)
[![Python preprocessing](https://github.com/probabl-ai/ironicl/actions/workflows/python.yml/badge.svg)](https://github.com/probabl-ai/ironicl/actions/workflows/python.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Ironicl is an inference engine for [TabICLv2](https://github.com/soda-inria/tabicl),
written in Rust. It runs without PyTorch on the compute built into the host. It
is built for a training table that does not change: Ironicl computes the
training context once, stores it, and then answers new test rows against it.

> [!WARNING]
> Ironicl is experimental. Formats, commands, and APIs can change without notice.
> The optimized paths target AMD Zen 5 (Ryzen 7 9800X3D) and Apple Silicon, where
> dense layers and attention use Apple Accelerate. Other CPUs, including Intel
> Macs, use a slower portable path.

## Status

| Works now | Not yet |
|---|---|
| Safe checkpoint import in pure Rust (no pickle execution) | Fast context building on non-Zen 5 CPUs |
| FP32 classifier inference on any CPU | Regression |
| Static training context with AVX-512 kernels on Zen 5 | Categorical features and missing values |
| Python `TabICLClassifier`, a drop-in for TabICL's | Published packages on crates.io or PyPI |

## Quickstart

You need Rust (via [rustup](https://rustup.rs)) and Python 3.12 or newer.

1. Build Ironicl:

   ```bash
   git clone https://github.com/probabl-ai/ironicl.git
   cd ironicl
   cargo build --release
   ```

2. Download the released TabICLv2 classifier checkpoint (110 MB):

   ```bash
   curl -LO https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt
   ```

3. Convert the checkpoint into an Ironicl model:

   ```bash
   ./target/release/ironicl import \
     --checkpoint tabicl-classifier-v2-20260212.ckpt \
     --output classifier.ironicl
   ```

4. Prepare a small request from the Covertype dataset:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install .
   .venv/bin/python tools/prepare_real_dataset.py \
     --dataset covertype --train-rows 1024 --test-rows 8 \
     --output covertype.irinput
   ```

5. Run inference:

   ```bash
   ./target/release/ironicl run --model classifier.ironicl --input covertype.irinput
   ```

   Ironicl prints one row of class logits for each test row.

See the [command-line reference](docs/cli.md) for `inspect`, `plan`, `platform`,
and `benchmark`.

## Python Package

The `ironicl` Python package replaces TabICL's classifier without PyTorch.
Change only the import:

```python
from ironicl import TabICLClassifier  # was: from tabicl import TabICLClassifier

clf = TabICLClassifier(n_estimators=8, random_state=42).fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

Building the package needs Rust. `fit` builds the training context in Rust,
and `predict_proba` runs only the query rows against it. See
[Python package](docs/python.md) for supported inputs.

## Documentation

- [Documentation index](docs/README.md): design, formats, and validation.
- [Benchmarking](docs/benchmarking.md): how Ironicl measures latency.
- [Contributing](CONTRIBUTING.md): development setup and checks.

## License

Ironicl is licensed under the [BSD 3-Clause License](LICENSE). The Python
preprocessing adapts code from TabICL under its own
[BSD 3-Clause notice](LICENSES/TabICL-BSD-3-Clause.txt). The TabICLv2
checkpoints are published separately by their authors under BSD-3-Clause.

If you use Ironicl, please cite the TabICL papers listed in
[references](docs/references.md#papers).

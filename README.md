# Ironicl

[![Rust checks](https://github.com/probabl-ai/ironicl/actions/workflows/rust.yml/badge.svg)](https://github.com/probabl-ai/ironicl/actions/workflows/rust.yml)
[![Python](https://github.com/probabl-ai/ironicl/actions/workflows/python.yml/badge.svg)](https://github.com/probabl-ai/ironicl/actions/workflows/python.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Ironicl is an inference engine for [TabICLv2](https://github.com/soda-inria/tabicl),
written in Rust. It runs without PyTorch on the compute built into the host: the
CPU, Apple's Metal GPU, or an NVIDIA GPU. It computes a training table's context
once and then answers new rows against it.

Use it in two ways:

- **In Python**, as a drop-in for TabICL's `TabICLClassifier` and `TabICLRegressor`.
- **As a server**, which serves a fitted model over HTTP without Python.

> [!WARNING]
> Ironicl is experimental. Formats, commands, and APIs can change without notice.
> The optimized paths target AMD Zen 5 (Ryzen 7 9800X3D) and Apple Silicon. Other
> CPUs, including Intel Macs, use a slower portable path.

## Quick start

### 1. Use it from Python

Install the package; building it needs Rust ([rustup](https://rustup.rs)) and
Python 3.12 or newer:

```bash
pip install git+https://github.com/probabl-ai/ironicl.git
```

Then change only the import:

```python
from sklearn.datasets import load_iris
from ironicl import TabICLClassifier  # was: from tabicl import TabICLClassifier

X, y = load_iris(return_X_y=True)
clf = TabICLClassifier().fit(X, y)
clf.predict_proba(X[:3])
```

The first fit downloads the TabICLv2 checkpoint (110 MB). `TabICLRegressor`
works the same way, with TabICL's `output_type` and `alphas`. See
[Python package](docs/python.md) for options and supported inputs.

### 2. Export the fitted model

```python
clf.export("my-model")
```

This writes the directory `my-model/`: the fitted preprocessing, the training
data, and the KV cache. The weights stay in a shared folder on this machine.

### 3. Serve it

Install the `ironicl` command with the server, then serve the directory:

```bash
cargo install --locked --git https://github.com/probabl-ai/ironicl.git \
    --features server,metal   # drop metal off Apple Silicon
ironicl serve my-model
```

```bash
curl -s localhost:8080/v1/predict -d '{"rows": [[5.1, 3.5, 1.4, 0.2]]}'
```

```json
{"labels": [0], "classes": [0, 1, 2], "probabilities": [[0.9999957, 0.0000031, 0.0000012]]}
```

To serve on another machine, copy `my-model/` there and run `ironicl import` on
the checkpoint once; the server builds the KV cache for that machine on first
start. NVIDIA GPUs need a CUDA build ([CUDA build targets](docs/source-layout.md#cuda-build-targets)).
See [Server](docs/server.md) for the API, batching, and measured throughput.

## Status

| Works now | Not yet |
|---|---|
| Classifier and regressor, at parity with TabICL | Categorical features and missing values |
| TabICL's preprocessing, in Rust | More than 10 classes |
| CPU (AVX-512 on Zen 5, Accelerate on Apple Silicon), Metal, CUDA | Packages on PyPI or crates.io |
| Python drop-ins for TabICL's estimators | |
| A prediction server that batches requests, without Python | |
| Checkpoint import without executing pickle code | |

## Documentation

- [Documentation index](docs/README.md): design, formats, and validation.
- [Command line](docs/cli.md): `inspect`, `import`, `prepare`, and `serve`, model directories, and weights.
- [Benchmarking](docs/benchmarking.md): how Ironicl measures latency.
- [Contributing](CONTRIBUTING.md): development setup and checks.

## License

Ironicl is licensed under the [BSD 3-Clause License](LICENSE). Its Rust
preprocessing adapts code from TabICL under its own
[BSD 3-Clause notice](LICENSES/TabICL-BSD-3-Clause.txt), and ports parts of
SciPy, scikit-learn, and CPython under theirs, in [LICENSES](LICENSES).
The TabICLv2 checkpoints are published separately by their authors under
BSD-3-Clause.

If you use Ironicl, please cite the TabICL papers listed in
[references](docs/references.md#papers).

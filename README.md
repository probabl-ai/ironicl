<div align="center">

  ![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)
  ![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?style=flat&logo=python)
  [![CI](https://github.com/probabl-ai/ironicl/actions/workflows/ci.yml/badge.svg)](https://github.com/probabl-ai/ironicl/actions/workflows/ci.yml)

</div>

<div align="center">

  <picture>
    <source srcset="docs/images/ironicl_logo_dark.svg" media="(prefers-color-scheme: dark)">
    <img width="120" src="docs/images/ironicl_logo.svg" alt="IronICL logo">
  </picture>
  <h3>TabICL on CPU, without PyTorch</h3>

Cached TabICLv2 inference in Rust, wrapped as a scikit-learn classifier \
[Documentation](docs/design.md) — [TabICL](https://github.com/soda-inria/tabicl) — [Changelog](CHANGES.md)

</div>

<br />

## 🧩 What is IronICL?

IronICL is a CPU inference engine for the [TabICLv2](https://github.com/soda-inria/tabicl)
tabular foundation model, packaged as a [scikit-learn](https://scikit-learn.org/stable/) classifier.

Just like TabICL, IronICL can cache the key/value states used for inference
rather than recomputing them at every call (see tabicl's
[`kv_cache` option](https://tabicl.readthedocs.io/en/stable/api.html)).
IronICL, however, is designed as a CPU acceleration experiment: it
reimplements TabICL's cached inference path in Rust, without PyTorch, so that
the training table's state becomes a persistent, checksummed artifact and the
query path runs on kernels chosen for the host CPU. The current scope is
classification with the released TabICLv2 checkpoint, at parity with tabicl's
predictions. The engine is the foundation for what comes next: faster context
construction on large tables, regression, and serving outside Python.

`IronICLClassifier` keeps the parameter names of `tabicl.TabICLClassifier`
where the meaning is the same, so the two can be swapped inside a pipeline
and produce the same predictions.

## 📦 Installation

### Dependencies

IronICL requires:

- Python (>= 3.10)
- NumPy (>= 1.26)
- scikit-learn (>= 1.6)
- huggingface_hub (>= 0.23), to download the TabICLv2 checkpoint

The first `fit` downloads the classifier checkpoint
`tabicl-classifier-v2-20260212.ckpt` (about 110 MB) from Hugging Face into
the same cache tabicl uses, and imports it once into IronICL's model format
under `~/.cache/ironicl` (or `$IRONICL_CACHE_DIR`). Pass `model_path=` to use
a local checkpoint and `allow_auto_download=False` to forbid downloads.

### User installation

```bash
pip install ironicl
```

Until a release is on PyPI, install from a checkout. That additionally needs
a stable Rust toolchain from [rustup](https://rustup.rs):

```bash
pip install .
```

## ⚡️ Usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ironicl import IronICLClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = IronICLClassifier(n_estimators=8)
clf.fit(X_train, y_train)        # builds the static context for this table
clf.predict_proba(X_test)        # cached inference on the query rows
```

The estimator takes finite numeric input. For real tables with strings, dates
or missing values, let [skrub](https://skrub-data.org) do the encoding and
[skore](https://skore.probabl.ai) the evaluation:

```python
from skore import EstimatorReport
from skrub import tabular_pipeline

pipeline = tabular_pipeline(IronICLClassifier())
report = EstimatorReport(pipeline, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
report.metrics.accuracy()
```

See [examples/plot_tabular_pipeline.py](examples/plot_tabular_pipeline.py).

Useful options:

```python
IronICLClassifier(
    n_estimators=8,          # ensemble members, as in tabicl
    kv_dtype="bf16",         # store the key/value state in BF16 (FP32 compute); or "bf16-v"
    sparse_index=True,       # coarse key index so far-away training rows can be skipped
    cache_dir="/fast/disk",  # where models and contexts are written
)
```

`IRONICL_THREADS` sets the engine's thread count for the process; it is read
once, so set it before the first prediction. Classification supports at most
10 classes (the model's native decoder); regression is not supported.

## 📝 Changelog

See the [changelog](CHANGES.md) for a history of notable changes to IronICL.

## 🛠️ Development

### Source code

The Python package lives in `ironicl/`, the Rust crate in `src/`, with one
`Cargo.toml` and one `pyproject.toml` at the root:

```bash
git clone https://github.com/probabl-ai/ironicl.git
```

### Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the setup, the checks and the
pull-request conventions. [docs/design.md](docs/design.md) describes the
engine, its artifacts and its validation policy; [docs/development.md](docs/development.md)
describes the layout, the `ironicl` command-line tool and the roadmap.

### Testing

After `make install` in a virtual environment:

```bash
make test        # cargo test, then pytest
```

Tests marked `slow` download the checkpoint on first use; `pytest -m "not slow"`
skips them.

## 👋 Help and support

### Documentation

- Design and artifact formats: [docs/design.md](docs/design.md)
- Benchmarks and methodology: [docs/benchmarks.md](docs/benchmarks.md)
- Building, testing and the CLI: [docs/development.md](docs/development.md)

### Citation

IronICL runs the TabICL models. If you use it in a scientific publication,
please cite the TabICL papers: Qu et al., *TabICL: A Tabular Foundation Model
for In-Context Learning on Large Data*, ICML 2025 (arXiv:2502.05564), and Qu
et al., *TabICLv2: A better, faster, scalable, and open tabular foundation
model*, 2026 (arXiv:2602.11139).

## ⚖️ License

BSD-3-Clause. IronICL is a new implementation informed by the BSD-3-Clause
[TabICL](https://github.com/soda-inria/tabicl) project; its preprocessing
module is a port of tabicl's and carries its notice. See the TabICL repository
for the model license.

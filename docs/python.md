# Python Package

The `ironicl` Python package replaces TabICL's `TabICLClassifier`. It requires
Python 3.12+, NumPy, scikit-learn, and SciPy. It does not depend on TabICL or
PyTorch. Building it from source requires Rust, because maturin compiles the
private `ironicl._core` extension module.

```bash
python -m pip install .
```

## Classifier

```python
from ironicl import TabICLClassifier

clf = TabICLClassifier(n_estimators=8, random_state=42).fit(X_train, y_train)
proba = clf.predict_proba(X_test)
labels = clf.predict(X_test)
```

The constructor, `fit`, `predict_proba`, and `predict` have TabICL 2.1.1's
parameter names and defaults. A live parity test checks this.

- Without `model_path`, `fit` downloads the released checkpoint from Hugging Face
  to `$XDG_CACHE_HOME/ironicl/checkpoints` (default `~/.cache`). The Rust
  importer verifies its SHA-256 and caches the converted model.
- These values raise `NotImplementedError`: a `feat_shuffle_method` other than
  `latin`, a `class_shuffle_method` other than `shift`, a `device` other than
  `cpu`, any `inference_config`, and `random_state=None`.
- `batch_size`, `kv_cache`, `use_amp`, `use_fa3`, `offload_mode`,
  `disk_offload_dir`, `n_jobs`, and `verbose` control PyTorch execution and have
  no effect.
- Prediction runs the uncached forward pass over training and test rows. It is
  meant for small tables until Rust builds the static context.

## Preprocessing

`ironicl.preprocessing.ClassifierPreprocessor` exposes the preprocessing step
on its own:

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

## Supported Inputs

This initial implementation supports finite dense numeric inputs, 2–10 classes,
and the default `none`/`power` transforms with Latin feature and cyclic class
permutations. It removes constant features and rejects all-constant training
data. Categorical inputs, missing values, and sparse matrices are not supported.
Numeric DataFrames must have unique string column names in the same order at
prediction time. Preprocessing is fitted on training data only, retains no
training rows, and does not require `PYTHONHASHSEED` configuration. It preserves
upstream's power-transform fallback, including its batch-wide clipping when a
transform raises a domain error.

`tools/prepare_real_dataset.py` uses this package without `--tabicl-root`; it
expands ensembles only when writing the existing IRINPUT file format. Other
upstream oracle and context-export tools still require TabICL/PyTorch.

## Version Requirements

The package requires NumPy 2.0+, scikit-learn 1.7+, and SciPy 1.17+. These are
the oldest releases that pass the frozen upstream oracle. With SciPy 1.16 or
older, the float32 power-transform fit gives different results. There are no
upper bounds; the CI parity job installs the latest releases.

## Tests

Run the Python checks after installation:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The checked-in preprocessing oracle runs without upstream dependencies. Run live
comparisons separately, in an environment with both projects installed and an
unmodified TabICL checkout at commit `bd030d47ff7d9f987f0f33431d743a3003bfe0d3`:

```bash
IRONICL_TABICL_SOURCE=/path/to/tabicl/src \
  python -m unittest discover -s tests/parity -p 'test_*.py' -v
```

This dedicated suite compares independently fitted preprocessing chains, using
synthetic inputs and bundled scikit-learn datasets. It also compares
`TabICLClassifier` signatures and `predict_proba` within `1e-4`, and downloads the
released checkpoint for that. Missing prerequisites fail rather than skip. See
[validation](validation.md#python-preprocessing-parity) for setup, coverage, and
numerical requirements.

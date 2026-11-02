# Python Preprocessing

The `ironicl` Python package reproduces TabICLv2 classifier preprocessing. It
requires Python 3.12+, NumPy, scikit-learn, and its SciPy dependency. It does
not depend on TabICL or PyTorch.

```bash
python -m pip install .
```

**This is a preprocessing-only Python package, not yet a Python classifier.**
PyO3 bindings and Rust training-context construction remain to be implemented.

## Usage

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
upstream oracle and context-export tools still require TabICL/PyTorch. The
current pure-Python build backend will be replaced by maturin when the native
extension is added.

## Version Requirements

The supported scikit-learn 1.9.x / SciPy 1.18.x combination preserves the
upstream power-fit behavior; older float32 solvers produce different results.
NumPy 2.x is required.

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
synthetic inputs and bundled scikit-learn datasets. Missing prerequisites fail
rather than skip. It tests preprocessing, not classifier predictions; see
[validation](validation.md#python-preprocessing-parity) for setup, coverage, and
numerical requirements.

# Python Package

The `ironicl` Python package replaces TabICL's `TabICLClassifier` and
`TabICLRegressor`. It requires
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
- `n_jobs` sets the executor thread count (negative values follow TabICL, so
  `-1` is all logical cores). `batch_size`, `verbose`, and `support_many_classes`
  have no effect and warn if changed; `use_amp=True`, `use_fa3=True`, a
  non-default `offload_mode`/`disk_offload_dir`, and `kv_cache="repr"` raise
  `NotImplementedError`.
- `fit` runs the training rows through the model once and keeps their attention
  keys and values in memory, like upstream's `kv_cache=True`. `predict_proba`
  then runs only the query rows. The context takes about 49 KB per training row
  and estimator. A fitted estimator cannot be pickled yet.

### Optimization levels

`TabICLClassifier` takes one parameter TabICL does not have, after all of
TabICL's: `optimize`. It sets how far Ironicl may trade exactness for speed on
Apple Silicon GPUs:

| `optimize` | Precision | Accuracy against `"safe"` |
|---|---|---|
| `"safe"` | FP32 throughout | matches TabICL within `1e-4` |
| `"auto"` | context keys and values stored in FP16; ICL attention in FP16 when building and when predicting | max probability error below `1e-3`, labels unchanged except near-ties |
| `"fast"` | currently the same as `"auto"`; further reduced precision lands here first | probabilities within about `1e-2` |

`None`, the default, reads the `IRONICL_OPTIMIZE` environment variable and
falls back to `"safe"`. An explicit argument wins over the variable. Without a
Metal GPU, every level behaves like `"safe"`. On an M1 Max, `"auto"` made
7071x22 fitting 16% faster, batch prediction 14% faster, and single-row
prediction 25% faster, with a maximum probability error of `4.9e-4` across 12
datasets.

### Context cache

`cache_dir` is Ironicl's second extra parameter. `fit` builds the training context
once; with a cache it is written to disk and reused when the same data, model, and
`optimize` level are fit again. Each context is large, so caching is opt-in:

| `cache_dir` | Behavior |
|---|---|
| `False` (default) | caching disabled |
| `None` | cache under `IRONICL_CACHE_DIR` if set, else `<XDG_CACHE_HOME or ~/.cache>/ironicl/contexts` |
| str or `Path` | cache in that directory |

Entries are content-addressed by the model identity, the preprocessed inputs, and
`optimize`, with a `.irctx` name. A load prefaults the mapping so prediction runs
at RAM speed from the first query. A corrupt or mismatched entry is silently
rebuilt, and a write failure never fails `fit`. There is no automatic eviction;
remove files to reclaim space.

### Pickling

A fitted estimator pickles with `pickle` and `joblib`, and copies with
`copy.deepcopy`. The pickle holds the whole fitted context, so it is as large as
the context in `memory_plan_` and needs no cache or training data to load.
Loading needs the checkpoint: the pickled path if it exists, otherwise the
default checkpoint location, downloaded when `allow_auto_download` allows. The
loaded estimator uses the loading machine's threads and memory limits; on the
same machine it predicts exactly as the pickled one did. Pickling briefly holds
up to two more copies of the context in memory, and loading holds the pickled
bytes alongside the loaded context until the load finishes.

## Regressor

```python
from ironicl import TabICLRegressor

reg = TabICLRegressor(n_estimators=8, random_state=42).fit(X_train, y_train)
mean = reg.predict(X_test)
stats = reg.predict(X_test, output_type=["median", "variance", "quantiles"], alphas=[0.05, 0.95])
```

The constructor, `fit`, and `predict` have TabICL 2.1.1's parameter names and
defaults, plus the trailing `optimize` and `cache_dir` described above. The
unsupported and no-op parameters are the classifier's, without the class ones.

The model predicts 999 quantiles of the standardized target per row. `predict`
turns them into `mean`, `median`, `variance`, `quantiles` (at `alphas`, default
0.1 to 0.9), or `raw_quantiles` in Rust, as TabICL's `QuantileToDistribution`
does: sorted quantiles, linear interpolation, and exponential tails beyond the
levels 0.001 and 0.999. Results match TabICL within `1e-4` target standard
deviations, with one deliberate difference: `variance` is scaled back by the
target variance `σ²`. TabICL applies `σ·v + μ` to it, as it does to the mean,
which is wrong unless the target is already standardized
([soda-inria/tabicl#180](https://github.com/soda-inria/tabicl/issues/180)).

## Preprocessing

`ironicl.preprocessing.Preprocessor` exposes the preprocessing step
on its own:

```python
from ironicl.preprocessing import Preprocessor

preprocessor = Preprocessor(n_estimators=8, random_state=42)
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
`TabICLClassifier` and `TabICLRegressor` signatures and predictions within `1e-4`,
and downloads the released checkpoints for that. Missing prerequisites fail rather than skip. See
[validation](validation.md#python-preprocessing-parity) for setup, coverage, and
numerical requirements.

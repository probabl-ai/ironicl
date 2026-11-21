# Python Package

The `ironicl` Python package replaces TabICL's `TabICLClassifier` and
`TabICLRegressor`. It requires
Python 3.12+, NumPy, and scikit-learn. It does not depend on TabICL or
PyTorch. Building it from source requires Rust, because maturin compiles the
private `ironicl._core` extension module, which preprocesses, builds the
training context, and predicts. Python converts inputs and keeps
scikit-learn's interface.

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
  importer verifies its SHA-256 and caches the converted model under
  `$XDG_CACHE_HOME/ironicl/models`. When that directory is not writable, as in a
  read-only container, it converts the checkpoint in memory on every load.
- These values raise `NotImplementedError`: a `feat_shuffle_method` other than
  `latin`, a `class_shuffle_method` other than `shift`, a `device` other than
  `cpu`, any `inference_config`, and `random_state=None`.
- `n_jobs` sets the executor thread count (negative values follow TabICL, so
  `-1` is all logical cores). `verbose=True` shows progress (see
  [Progress and interrupts](#progress-and-interrupts)). `batch_size` and
  `support_many_classes` have no effect and warn if changed; `use_amp=True`, `use_fa3=True`, a
  non-default `offload_mode`/`disk_offload_dir`, and `kv_cache="repr"` raise
  `NotImplementedError`.
- `fit` runs the training rows through the model once and keeps their attention
  keys and values in memory, like TabICL's `kv_cache=True`. `predict_proba`
  then runs only the query rows. The context takes about 49 KB per training row
  and estimator. A fitted estimator can be saved and pickled; see
  [Saving and pickling](#saving-and-pickling).

### Optimization levels

`TabICLClassifier` takes one parameter TabICL does not have, after all of
TabICL's: `optimize`. It sets how far Ironicl may trade exactness for speed on
Apple Silicon and NVIDIA GPUs:

| `optimize` | Precision | Accuracy against `"safe"` |
|---|---|---|
| `"safe"` | FP32 throughout | matches TabICL within `1e-4` |
| `"auto"` | context keys and values stored in FP16; ICL attention in FP16 when building and when predicting | max probability error below `1e-3`, labels unchanged except near-ties |
| `"fast"` | currently the same as `"auto"`; further reduced precision lands here first | probabilities within about `1e-2` |

`None`, the default, reads the `IRONICL_OPTIMIZE` environment variable and
falls back to `"safe"`. An explicit argument wins over the variable. Without a
Metal or CUDA GPU, every level behaves like `"safe"`. Every level runs on the
GPU when there is one; on an NVIDIA GPU, `"safe"` runs in FP32 on the CUDA
cores, without tensor cores or TF32. On an M1 Max, `"auto"` made
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

Entries are content-addressed by the Ironicl version, the model identity, the
preprocessed inputs, and `optimize`, with a `.irctx` name. Each entry records that
identity, and a load uses an entry only when it matches; a corrupt or mismatched
entry is silently rebuilt, and a write failure never fails `fit`. A load
prefaults the mapping so prediction runs at RAM speed from the first query. On
Unix, entries are readable only by their owner, because they encode the training
data. Entries from other Ironicl versions are not reused; `clear_cache` removes
them. Each entry records its last use, when `fit` built or loaded it.

Nothing is evicted automatically. `ironicl.cache_info()` lists the entries,
most recently used first, and `ironicl.clear_cache()` removes them: all by
default, those last used before `older_than` (a `timedelta` or `datetime`), or
the least recently used until the rest fit `max_size` (bytes or a size such as
`"10G"`). Both take a directory, and default to where `cache_dir=None` caches.
`clear_cache` returns what it removed, and also removes temporary files that
interrupted writes left behind:

```python
>>> print(ironicl.cache_info())
~/.cache/ironicl/contexts: 2 contexts, 1.73 GiB
       1.04 GiB  last used 2026-11-08 21:40  3f2a....irctx
       0.69 GiB  last used 2026-11-07 10:12  9c1e....irctx
>>> ironicl.clear_cache(max_size="1.5G").size_bytes
740800512
```

### Saving and pickling

As in TabICL, `save` and `load` write and read a fitted estimator, and `pickle`,
`joblib`, and `copy.deepcopy` work as `save` does with its defaults:

```python
clf.save("model.pkl", save_model_weights=False, save_training_data=True, save_kv_cache=True)
clf = TabICLClassifier.load("model.pkl", device=None)
```

- `save_kv_cache` keeps the fitted context. Without it, `load` rebuilds the
  context from the training data, which takes as long as the build in `fit`;
  with `cache_dir` set, a context cached on this machine loads from the cache
  instead.
- `save_training_data` keeps the training data, `X` and `y` as `fit` received
  them (`fit` keeps a copy for this). Leaving out both the training data and
  the context raises `ValueError`.
- `save_model_weights` includes the checkpoint, so loading needs no checkpoint
  file or download. Without it, loading uses the saved checkpoint path if it
  exists, otherwise `model_path` or the default location, downloaded when
  `allow_auto_download` allows. An included checkpoint is written there when it
  is missing.

The fitted model is saved in Rust's formats
([artifact-format.md](artifact-format.md#fitted-models)): its preprocessing,
and the context and the training data as the options say. Loading refuses a
different checkpoint.

A 2,000-row, 100-feature classifier with 8 estimators, on an M1 Max:

| `save` options | `optimize="safe"` | `optimize="auto"` |
|---|---|---|
| Defaults | 1,064 MiB | 689 MiB |
| `save_kv_cache=False` | 1.6 MiB | 1.6 MiB |
| `save_model_weights=True` | 1,169 MiB | 794 MiB |

The context compresses to only 94% of its size, so compressing the file is not
worth it. Decoding a 1 GiB context takes about 0.7 s and encoding it 0.5 s.
Saving encodes the context straight into the pickled bytes, one more copy in
memory; loading holds those bytes alongside the loaded context until the load
finishes. Writing a `cache_dir` entry streams to the file without a copy.

Loading plans its memory on the loading machine as `fit` does: the weights,
the context, and headroom, or the whole build when it rebuilds the context. It
warns or raises as `memory_overflow` says, and `memory_plan_` becomes the
loading machine's plan. The loaded estimator uses the loading machine's threads
and memory limits; on the same machine and with the same limits, it predicts
exactly as the saved one did. A GPU-built context also loads without the GPU:
on the CPU, `optimize="auto"` predictions moved by at most `6.7e-4` in
probability. `device` accepts what the constructor does. Loading warns when
the file was saved with another Ironicl version. Load only files you trust:
like any pickle, loading can run code.

### Progress and interrupts

With `verbose=True`, `fit` shows the estimators built, `predict` and
`predict_proba` the rows predicted, and a load that rebuilds the context the
build: as a tqdm bar when tqdm is installed (`pip install ironicl[progress]`),
and otherwise as a line on stderr for each tenth of the way. Updates come
between build groups and between query pieces of up to 1,024 rows, so a build
that runs as one group shows only its end.

Ctrl-C stops a build or prediction at the next of those steps with
`KeyboardInterrupt`, instead of when the work ends, so a build that runs as one
group stops only at its end. Only the main thread receives Ctrl-C; work started
on another thread runs to its end.

### Configuration

`ironicl.show_config()` prints what Ironicl detects and how it would run: its
version and dependencies, the GPU and its Metal working set, CPU threads and BF16
support, memory, the ICL cache storage each `optimize` level gives, the
downloaded checkpoints, the context cache, and the `IRONICL_*` environment
variables that are set. Include it in bug reports.

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

Preprocessing runs in Rust, at parity with TabICL's. There is no separate
preprocessing class: `fit` keeps the fitted model as `fitted_`
([native-preprocessing.md](native-preprocessing.md)). `fit` also sets
scikit-learn's `n_features_in_`, `feature_names_in_` for named data, `classes_`
and `n_classes_` for the classifier, `memory_plan_`, and `checkpoint_path_`. TabICL's other fitted
attributes, such as `X_encoder_`, `ensemble_generator_`, and the regressor's
`y_scaler_`, do not exist.

## Supported Inputs

This initial implementation supports finite dense numeric inputs, 2–10 classes
whose labels are all integers, all floats, all booleans, or all strings, and the
default `none`/`power` transforms with Latin feature and cyclic class
permutations. It removes constant features and rejects all-constant training
data. Categorical inputs, missing values, and sparse matrices are not supported.
Numeric DataFrames must have unique string column names in the same order at
prediction time. Preprocessing is fitted on training data only and does not
require `PYTHONHASHSEED` configuration. It preserves TabICL's power-transform
fallback, including its batch-wide clipping when a transform raises a domain
error.

## Version Requirements

The package requires NumPy 2.0+ and scikit-learn 1.7+, the oldest releases that
pass its tests. Preprocessing does not use the installed SciPy. There are no
upper bounds; the CI parity job installs the latest releases.

## Tests

Run the Python checks after installation:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Rust's tests (`cargo test`) check preprocessing against a frozen TabICL fixture
without TabICL. Run live comparisons separately, in an environment with both
projects installed and an unmodified TabICL checkout at commit
`bd030d47ff7d9f987f0f33431d743a3003bfe0d3`:

```bash
IRONICL_TABICL_SOURCE=/path/to/tabicl/src \
  python -m unittest discover -s tests/parity -p 'test_*.py' -v
```

This dedicated suite compares independently fitted preprocessing chains, using
synthetic inputs and bundled scikit-learn datasets. It also compares
`TabICLClassifier` and `TabICLRegressor` signatures and predictions within `1e-4`,
and downloads the released checkpoints for that. Missing prerequisites fail rather than skip. See
[validation](validation.md#preprocessing-parity) for setup, coverage, and
numerical requirements.

# Ironicl

Ironicl runs [TabICLv2](https://github.com/soda-inria/tabicl) inference without
PyTorch. The model runs in a native engine written in Rust, on the compute built
into your machine. It is built for a training table that does not change:
Ironicl computes the training context once in `fit`, keeps it, and then
predicts new rows against it.

`ironicl.TabICLClassifier` and `ironicl.TabICLRegressor` are drop-in
replacements for TabICL's estimators.

> **Warning**
> Ironicl is experimental. Formats and APIs can change without notice.

## Installation

```bash
pip install ironicl
```

Ironicl needs Python 3.12 or newer, NumPy, scikit-learn, and SciPy. It does not
depend on TabICL or PyTorch. For a progress bar, install the `progress` extra:
`pip install "ironicl[progress]"`.

Prebuilt wheels are currently published for **macOS on Apple Silicon** only.
There, dense layers and attention use Apple Accelerate, and Metal runs context
building and prediction on the GPU.

## Quickstart

Change only the import:

```python
from ironicl import TabICLClassifier, TabICLRegressor  # was: from tabicl import ...

clf = TabICLClassifier(n_estimators=8, random_state=42).fit(X_train, y_train)
proba = clf.predict_proba(X_test)
labels = clf.predict(X_test)

reg = TabICLRegressor(n_estimators=8, random_state=42).fit(X_train, y_train)
mean = reg.predict(X_test)
q = reg.predict(X_test, output_type="quantiles", alphas=[0.05, 0.95])
```

On first use, `fit` downloads the released TabICLv2 checkpoint from
[Hugging Face](https://huggingface.co/jingang/TabICL) to
`~/.cache/ironicl/checkpoints` (or `$XDG_CACHE_HOME/ironicl/checkpoints`).
Ironicl verifies its SHA-256 and imports it safely, without executing pickle
code. Pass `model_path=` to use a local checkpoint instead.

The constructors, `fit`, `predict`, and `predict_proba` keep TabICL 2.1.1's
parameter names and defaults. `fit` runs the training rows through the model
once and keeps their attention keys and values, like TabICL's `kv_cache=True`;
prediction then runs only the query rows. The context takes about 49 KB per
training row and estimator.

## Optimization levels

`optimize`, a parameter TabICL does not have, sets how far Ironicl may trade
exactness for speed on the Apple Silicon GPU:

| `optimize` | Precision | Accuracy against `"safe"` |
|---|---|---|
| `"safe"` | FP32 throughout | matches TabICL within `1e-4` |
| `"auto"` | context and ICL attention in FP16 | max probability error below `1e-3` |
| `"fast"` | currently the same as `"auto"` | probabilities within about `1e-2` |

The default, `None`, reads the `IRONICL_OPTIMIZE` environment variable and falls
back to `"safe"`. On an M1 Max, `"auto"` made fitting a 7071x22 table 16% faster,
batch prediction 14% faster, and single-row prediction 25% faster.

## Context cache

With `cache_dir`, `fit` writes the training context to disk and reuses it when
the same data, model, and `optimize` level are fit again:

| `cache_dir` | Behavior |
|---|---|
| `False` (default) | no caching |
| `None` | `$IRONICL_CACHE_DIR`, else `~/.cache/ironicl/contexts` |
| str or `Path` | that directory |

Nothing is evicted automatically. `ironicl.cache_info()` lists the entries and
`ironicl.clear_cache()` removes them, all by default, or by age
(`older_than=timedelta(days=7)`) or total size (`max_size="10G"`).

## Saving and pickling

As in TabICL, `save` and `load` write and read a fitted estimator, and `pickle`,
`joblib`, and `copy.deepcopy` work too:

```python
clf.save("model.pkl", save_model_weights=False, save_training_data=True, save_kv_cache=True)
clf = TabICLClassifier.load("model.pkl")
```

Without `save_kv_cache`, the file is small and `load` rebuilds the context from
the training data. Load only files you trust: like any pickle, loading can run
code.

## Progress, interrupts, and configuration

- `verbose=True` shows build and prediction progress, as a tqdm bar when tqdm is
  installed and as plain lines on stderr otherwise.
- Ctrl-C stops a build or prediction early with `KeyboardInterrupt`.
- `n_jobs` sets the thread count (`-1` uses all logical cores).
- `ironicl.show_config()` prints the version, GPU, threads, memory, cached
  checkpoints and contexts, and the `IRONICL_*` variables that are set. Include
  it in bug reports.

## Supported inputs

- Finite, dense, numeric features, as NumPy arrays or numeric DataFrames.
- Classification with 2 to 10 classes, and regression.
- Not yet: categorical features, missing values, sparse matrices, more than 10
  classes.

## License

Ironicl is licensed under the BSD 3-Clause License. Its Python preprocessing
adapts code from TabICL, under TabICL's own BSD 3-Clause notice; both license
files ship with the package. The TabICLv2 checkpoints are published separately
by their authors under BSD-3-Clause.

## Citation

If you use Ironicl, please cite the TabICL papers:

- Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan.
  *TabICLv2: A better, faster, scalable, and open tabular foundation model.*
  2026. [arXiv:2602.11139](https://arxiv.org/abs/2602.11139)
- Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan.
  *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.*
  ICML 2025. [arXiv:2502.05564](https://arxiv.org/abs/2502.05564)

Ironicl is developed by [Probabl](https://probabl.ai).

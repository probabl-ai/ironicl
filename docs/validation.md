# Validation Strategy

## Oracle Hierarchy

1. Released TabICLv2 PyTorch FP32 CPU inference.
2. Ironicl at `optimize="safe"`: FP32 on every device.
3. Ironicl at `optimize="auto"` and `"fast"`: reduced precision, bounded against `"safe"`.

Each level must pass before the next level is used for performance conclusions.

## Test Layers

### Checkpoint Import

- exact tensor count, names, shapes, dtypes, and element counts;
- model-configuration equality;
- checksums before and after conversion;
- rejection of unknown or missing tensors.

Set `IRONICL_CHECKPOINT_FIXTURES` to a directory containing both official v2
checkpoint files and the imported classifier `tabicl-classifier-v2.irmodel` to
enable the fixture-backed Rust integration tests; see
[CONTRIBUTING.md](../CONTRIBUTING.md). They are ignored by default.

### Preprocessing Parity

Preprocessing runs in Rust ([native-preprocessing.md](native-preprocessing.md))
and has three test layers. Rust unit tests check each piece against golden
fixtures from CPython, SciPy, and scikit-learn
(`tests/fixtures/python_random.json`, `tests/fixtures/yeo_johnson.json`, which
`tools/export_preprocess_fixtures.py` regenerates) and the whole pipeline
against the frozen TabICL fixture `tests/fixtures/preprocessing.json`. The
standalone Python checks require neither TabICL nor PyTorch. The [Python workflow](../.github/workflows/python.yml)
runs on pull requests and pushes to `main`, using Linux and Python 3.12.

Its `standalone` job builds an sdist and then builds a wheel from that sdist,
verifies TabICL's license notice in both distributions, and installs the wheel
in a clean environment. It checks that imports resolve to the installed package
and that neither TabICL nor PyTorch is available before running the standalone suite.

To reproduce the build and standalone tests, run these commands from the repository
root. The temporary directories keep build tools and test dependencies separate:

```sh
checks=$(mktemp -d)
python3.12 -m venv "$checks/build"
"$checks/build/bin/python" -m pip install build
"$checks/build/bin/python" -m build --outdir "$checks/dist"
python3.12 -m venv "$checks/standalone"
"$checks/standalone/bin/python" -m pip install "$checks"/dist/*.whl
env -u PYTHONPATH "$checks/standalone/bin/python" \
  -m unittest discover -s tests -p 'test_*.py' -v
```

`tests/test_preprocessing.py` covers validation, shapes, label types, batching,
and dependency isolation, through the Rust extension.

The `tabicl-parity` CI job downloads the wheel tested by `standalone` and installs
it in a separate environment with CPU-only PyTorch and the pinned TabICL checkout.
It runs the live suite against that wheel, not an editable Ironicl checkout.
TabICL and PyTorch are installed only for this job, never as Ironicl dependencies.

To reproduce the live tests, reuse `$checks` and the wheel built above. From the
repository root, create a TabICL checkout at commit
`bd030d47ff7d9f987f0f33431d743a3003bfe0d3` (version 2.1.1):

```sh
git clone https://github.com/soda-inria/tabicl.git "$checks/tabicl"
git -C "$checks/tabicl" checkout --detach bd030d47ff7d9f987f0f33431d743a3003bfe0d3
python3.12 -m venv "$checks/parity"
# On Linux, install CPU-only PyTorch before installing TabICL:
# "$checks/parity/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$checks/parity/bin/python" -m pip install "$checks"/dist/*.whl "$checks/tabicl" pandas
env -u PYTHONPATH IRONICL_TABICL_SOURCE="$checks/tabicl/src" \
  "$checks/parity/bin/python" -m unittest discover -s tests/parity -p 'test_*.py' -v
```

The suite verifies the checkout revision, rejects changes under `src/tabicl`,
and checks the imported preprocessing module's source path.
An absent checkout, wrong revision, or missing TabICL dependency fails the run with setup guidance;
it never silently skips. It reports Python, platform, NumPy, SciPy, scikit-learn,
Torch, and TabICL versions. TabICL/PyTorch remain test-only dependencies, with no
model or dataset downloads during the suite. Use `PYTHONPATH=python` to test the
working source instead of an installed Ironicl package.

`tests/parity/test_tabicl_preprocessing.py` fits an independent TabICL chain:
`LabelEncoder → TransformToNumerical → EnsembleGenerator`. Both chains start
from the same raw training data and labels. The oracle does not use Ironicl's
encoded labels, fitted statistics, masks, or permutations. Tests compare each
member's rows and align TabICL's normalization groups by name, not by
hash-dependent iteration order.

The cases include NumPy arrays and pandas DataFrames with every column type
TabICL encodes: numbers with missing values, `object`, `string`, `category`, and
nullable `boolean` and `Int64` columns, with categories training never saw. The
classifier suite needs pandas too, for a DataFrame prediction case.

Live comparisons require exact class ordering, encoded labels, feature masks,
actual ensemble counts, group membership, feature/class permutations, and
permuted training labels. Preprocessed training and query matrices, converted to
FP32 on TabICL's side, must agree within the
[parity contract](native-preprocessing.md#parity-contract)'s tolerance. Failures
identify the case, seed, dtype, stage, group, ensemble, and first differing
element where applicable, with maximum absolute and relative errors. The summary
reports completed fit cases, array comparisons, and observed maximum errors in
the main process.

Coverage includes:

- float32, float64, integer, and noncontiguous inputs, with TabICL fed float32
  input as `X.astype(float64)`;
- a skewed, heavy-tailed synthetic set;
- binary, multiclass, and 10-class labels, including strings and nonconsecutive integers;
- constant and near-constant features, finite outliers, one useful feature, and two training rows;
- multiple seeds, ensemble counts of 1/8/32, reduced actual counts, normalization selection/order,
  and a nondefault outlier threshold;
- fixed stratified splits of bundled Iris, Breast Cancer, and Digits datasets;
- training retransform, single-row/batched queries, and contiguous copies;
- separate Python processes with `PYTHONHASHSEED=0`, `1`, and `42`.

Power-transform domain errors have an important exception to batch invariance:
TabICL clips the **entire query batch** to fitted training bounds before retrying.
A row can therefore change when another row triggers the fallback. A Rust unit
test (`an_overflowing_power_transform_clips_the_whole_batch`) checks this batch
dependence with a lambda that overflows. Pickle round-trips are tested on the
estimators, in `tests/test_pickle.py`.

These checks establish preprocessing parity only. They do not validate neural
outputs, probabilities, predictions, native Python bindings, or Rust training-context
construction; those require separate end-to-end tests.

### End-to-End Parity

The live suite also compares the estimators with TabICL's, from the same raw rows and the same
released checkpoints. `tests/parity/test_tabicl_classifier.py` and
`tests/parity/test_tabicl_regressor.py` check that:

- the public signatures match TabICL's;
- at `optimize="safe"`, classifier probabilities match TabICL's within `atol=1e-4` on Iris, Wine,
  and Breast Cancer, and labels match exactly; regressor outputs (mean, median, variance,
  quantiles, and raw quantiles) match within `1e-4` of the target's standard deviation on
  Diabetes and synthetic sets. Each suite includes a case large enough for the Metal ICL path;
- float32 input matches TabICL on its float64 upcast;
- `"auto"` and `"fast"` stay within their error bounds of `"safe"`, and labels differ only on
  near-ties;
- a checkpoint for the wrong task is rejected.

Run it with the setup in [Preprocessing Parity](#preprocessing-parity).

An earlier oracle compared Rust logits with a direct PyTorch forward pass through exported
fixture files. It was removed with its file formats; see
[removed-experiments.md](removed-experiments.md#external-pytorch-oracle-test).

### Primitive Kernels

- projection against generated FP32 fixtures;
- LayerNorm with constant and near-constant inputs;
- GELU over tails and values near zero;
- QASSMax over representative source lengths;
- RoPE against TabICL's outputs;
- stable softmax with large, small, equal, and masked logits;
- online attention against materialized attention.

### Cached Prediction

The fixture-backed Rust tests in `tests/checkpoint_fixtures.rs` check the cached path against the
model itself:

- `built_context_matches_the_uncached_forward_pass` builds a context from training rows and checks
  that its predictions match one uncached forward pass over the training and query rows together,
  with per-estimator class shifts;
- `large_cached_query_runs_in_chunks` checks that a query larger than one pass matches the same
  rows predicted piece by piece;
- `reduced_precision_context_predicts_all_missing_rows` checks that query rows holding the
  all-missing marker, which run on the CPU, read an `"auto"` context's FP16 caches and match
  `"safe"`.

The GPU tests, also fixture-backed, check that grouped Metal builds match one-graph builds
(`tests/gpu_build_groups.rs`), that rows with the all-missing marker match between `"safe"` and
`"auto"` contexts on Metal or CUDA (`tests/gpu_all_missing.rs`), and that Metal builds and
predictions stay within the memory plan (`tests/build_memory.rs`). `tests/test_cuda.py` checks
CUDA predictions against the CPU, reduced levels against `"safe"`, and uploaded and streamed ICL
caches against resident ones.

TabICL's cache tests use `rtol=1e-4` and `atol=1e-4` even for FP32 because the operation order
changes. Treat this as context, not an automatic acceptance threshold for every Ironicl layer.

## Numerical Metrics

- maximum and percentile absolute error;
- maximum and percentile relative error;
- cosine similarity for internal representations;
- classification probability and logit differences;
- label agreement and changed decision margins;
- regression quantile monotonicity and prediction error;
- NaN, infinity, denormal, and overflow counts.

## Accuracy Gates

Reduced precision requires dataset-level evaluation, not only tensor error.
Track at minimum:

- classification accuracy, ROC AUC where applicable, and log loss;
- regression RMSE, MAE, and R-squared;
- per-dataset differences, not only aggregate mean;
- confidence intervals across fixed seeds and ensemble policies.

A reduced-precision level is rejected as a default if it causes a material or concentrated
regression, even when the global mean appears acceptable.

## Platform Matrix

| Platform | Required validation |
|---|---|
| M1 Max macOS | CPU and Metal at every level, artifact handling, test fixtures |
| 9800X3D Arch Linux | CPU at every level, CUDA on the NVIDIA rig |
| generic x86-64 CI | compilation and portable fallback |

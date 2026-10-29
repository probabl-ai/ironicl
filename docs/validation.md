# Validation Strategy

## Oracle Hierarchy

1. Released TabICLv2 PyTorch FP32 CPU inference.
2. Ironicl portable FP32 implementation.
3. Ironicl optimized FP32 implementation.
4. Ironicl mixed BF16-storage implementation.
5. Experimental INT8 policies.

Each level must pass before the next level is used for performance conclusions.

## Test Layers

### Checkpoint Import

- exact tensor count, names, shapes, dtypes, and element counts;
- model-configuration equality;
- checksums before and after conversion;
- rejection of unknown or missing tensors.

Set `IRONICL_CHECKPOINT_FIXTURES` to a directory containing both official v2
checkpoint files to enable the fixture-backed Rust integration tests. The local
OpenCode fixture directory is detected automatically during development.

### Python Preprocessing Parity

Python preprocessing has two test layers. Run the standalone checks after
installing Ironicl; they require neither TabICL nor PyTorch:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

`tests/test_preprocessing.py` covers validation, grouped storage, labels,
serialization, dependency isolation, and the checked-in upstream oracle in
`tests/fixtures/preprocessing.json`. The frozen numeric oracle uses
`rtol=2e-6, atol=2e-6`; metadata comparisons are exact.

Run live comparisons explicitly from the repository root. Use a separate
Python 3.12+ environment and a TabICL checkout at commit
`bd030d47ff7d9f987f0f33431d743a3003bfe0d3` (version 2.1.1):

```sh
python -m venv /tmp/ironicl-parity
/tmp/ironicl-parity/bin/python -m pip install . /path/to/tabicl
IRONICL_TABICL_SOURCE=/path/to/tabicl/src \
  /tmp/ironicl-parity/bin/python -m unittest discover -s tests/parity -p 'test_*.py' -v
```

The suite verifies the checkout revision, rejects changes under `src/tabicl`,
and checks the imported preprocessing module's source path. An absent checkout,
wrong revision, or missing upstream dependency fails the run with setup guidance;
it never silently skips. It reports Python, platform, NumPy, SciPy, scikit-learn,
Torch, and TabICL versions. TabICL/PyTorch remain test-only dependencies, with no
model or dataset downloads during the suite. Use `PYTHONPATH=python` to test the
working source instead of an installed Ironicl package.

`tests/parity/test_tabicl_preprocessing.py` fits an independent upstream chain:
`LabelEncoder → TransformToNumerical → EnsembleGenerator`. Both chains start
from the same raw training data and labels. The oracle does not use Ironicl's
encoded labels, fitted statistics, masks, or permutations. Tests expand
Ironicl's grouped matrices only at the comparison boundary and align upstream
normalization groups by name, not by hash-dependent iteration order.

Live comparisons require exact class ordering, encoded labels, feature masks,
actual ensemble counts, group membership, feature/class permutations, permuted
training labels, and prepared training/query matrices. Upstream feature matrices
are converted to FP32 before comparison, matching Ironicl's output contract.
There is no numerical tolerance in the live suite. Failures identify the case,
seed, dtype, stage, group, ensemble, and first differing element where applicable,
with maximum absolute and relative errors. The summary reports completed fit and
shuffle cases, array comparisons, and observed maximum errors in the main process.

Coverage includes:

- float32, float64, integer, and noncontiguous inputs;
- binary, multiclass, and 10-class labels, including strings and nonconsecutive integers;
- constant and near-constant features, finite outliers, one useful feature, and two training rows;
- multiple seeds, ensemble counts of 1/8/32, reduced actual counts, normalization selection/order,
  and a nondefault outlier threshold;
- fixed stratified splits of bundled Iris, Breast Cancer, and Digits datasets;
- training retransform, single-row/batched queries, contiguous copies, and pickle round-trips;
- separate Python processes with `PYTHONHASHSEED=0`, `1`, and `42`;
- Latin shuffling and the large-feature random-shuffle fallback.

Power-transform domain errors have an important exception to batch invariance:
upstream clips the **entire query batch** to fitted training bounds before retrying.
A row can therefore change when another row triggers the fallback. The suite
injects a deterministic domain error into each independently fitted power
transformer, verifies identical fallback outputs, and checks this batch dependence.
It does not assume that naturally occurring domain failures are reproducible
across numerical-library versions.

These checks establish preprocessing parity only. They do not validate neural
outputs, probabilities, predictions, native Python bindings, or Rust training-context
construction; those require separate end-to-end tests.

### External PyTorch Oracle

`tools/export_tabicl_oracle.py` runs the sibling TabICL checkout directly. It
does not add a Python dependency to the Rust crate. The exporter accepts only
the released classifier checkpoint with SHA-256
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.

From the Ironicl repository root, run the exporter with `uv` after placing the
official checkpoint locally:

```sh
uv run --project ../tabicl python tools/export_tabicl_oracle.py \
  --checkpoint /path/to/tabicl-classifier-v2-20260212.ckpt
```

Use `--tabicl-root /path/to/tabicl/src` for a checkout outside the default
`../tabicl/src`. The script uses CPU FP32, one PyTorch intra-op and inter-op
thread, no AMP, no Flash Attention 3, evaluation mode, and `torch.no_grad()`.
It writes a direct-neural fixture with `X` shape `[1, 5, 3]`, four training
labels `[0, 1, 0, 1]`, and one test row. It does not use sklearn preprocessing.

Add `--trace-output fixtures/tabicl-oracle.irtrace` to capture coarse parity
traces. IRTRACE v1 contains named row-major FP32 tensors for `col_embedder`,
`row_interactor`, the raw 10-wide decoder output, and returned logits. It uses
little-endian lengths, dimensions, and values, followed by a SHA-256 trailer.
The exporter also prints each trace tensor's shape, minimum, maximum, sum, and
first eight values.

Set `IRONICL_ORACLE_FIXTURES` to the directory that contains `input.irinput`
and `logits.irlogits` to require the Rust end-to-end classifier parity test.
The test loads the official artifact, validates both checked fixtures, and
compares every logit at `rtol=1e-4`, `atol=1e-4`. Without this environment
variable, the local OpenCode oracle fixture directory is used when available;
the test skips when it is absent. For example:

```sh
IRONICL_ORACLE_FIXTURES=/path/to/oracle cargo test --test checkpoint_fixtures \
  matches_python_classifier_oracle_when_fixtures_are_available -- --nocapture
```

The `.irinput` output is exact IRINPUT v1: `IRINPUT\0`, a 80-byte header with
little-endian `u32` version and kind and eight little-endian `u64`
dimension/count/byte fields, row-major FP32 `X`, FP32 `y_train`, and a SHA-256
trailer over all preceding bytes. This is the format decoded by
`src/input_bundle.rs`.

The `.irlogits` output is IRLOGITS v1. It has a 48-byte header of `IRLOGITS`
(8 bytes), little-endian `u32` version (1), little-endian `u32` kind (1), and
little-endian `u64` values for `B`, test rows, classes, and FP32 element count.
The header is followed by row-major little-endian FP32 logits with shape
`[B, test_rows, classes]`, then a 32-byte SHA-256 trailer over the header and
logits. The exporter prints each file hash, the logits shape, and values.

### Primitive Kernels

- projection against generated FP32 fixtures;
- LayerNorm with constant and near-constant inputs;
- GELU over tails and values near zero;
- QASSMax over representative source lengths;
- RoPE against upstream outputs;
- stable softmax with large, small, equal, and masked logits;
- online attention against materialized attention.

### Stage Parity

Capture outputs after:

- preprocessing;
- input projection;
- every column block;
- every row block;
- row representation;
- target embedding;
- every ICL block;
- decoder;
- ensemble aggregation.

Use deterministic small fixtures first, then real datasets.

### Cache Parity

Compare:

- uncached batch inference;
- PyTorch full-KV inference;
- Ironicl full-KV batch inference;
- Ironicl one-row-at-a-time inference;
- multiple query microbatch sizes.

The upstream cache tests use `rtol=1e-4` and `atol=1e-4` even for FP32 because
the operation order changes. Treat this as context, not an automatic acceptance
threshold for every Ironicl layer.

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

INT8 K/V is rejected as a default if it causes a material or concentrated
regression, even when the global mean appears acceptable.

## Platform Matrix

| Platform | Required validation |
|---|---|
| M1 Max macOS | portable FP32, artifact handling, test fixtures |
| 9800X3D Arch Linux | portable FP32, optimized FP32, BF16, INT8 experiments |
| generic x86-64 CI | compilation and portable fallback |

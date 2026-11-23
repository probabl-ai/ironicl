# Native Preprocessing

Preprocessing runs in Rust. The fitted model, `models/tabicl_v2/fitted.rs`, holds
the fitted preprocessing, with the class labels or the regressor's target scale,
and the training context, so it predicts from raw features without Python. The
Python estimators convert inputs and keep scikit-learn's interface; the planned
prediction server uses the same fitted model, saved as `IRPREP` and `IRCNTXT`
([artifact-format.md](artifact-format.md#fitted-models)).

| Part | Where |
|---|---|
| `CPython`'s `random.Random`: seeding, `choice`, `shuffle`, `sample` | `src/preprocess/python_random.rs`, on `rand_mt` |
| Standardize and clip; two-pass outlier bounds; `StandardScaler` | `src/preprocess/scale.rs` |
| Missing values: `SimpleImputer`'s mean | `src/models/tabicl_v2/preprocess.rs` |
| Typed columns; categories and codes: `OrdinalEncoder` | `src/preprocess/columns.rs` |
| Yeo-Johnson fit and transform | `src/preprocess/yeo_johnson.rs` |
| Class labels | `src/preprocess/labels.rs` |
| TabICL's pipeline: constant-feature mask, ensembles, groups | `src/models/tabicl_v2/preprocess.rs` |
| Fitting, context building, chunked prediction, saving | `src/models/tabicl_v2/fitted.rs` |
| Fitted preprocessing as `IRPREP` bytes | `src/format/preprocessor.rs` ([artifact-format.md](artifact-format.md#fitted-models)) |
| Input conversion, label types, scikit-learn's interface | `python/ironicl/base.py` |

`src/preprocess/` forbids `unsafe`: it handles request data and, in the server,
fitted preprocessing from outside.

## Parity Contract

Ironicl fits and transforms in float64. On float32 input it matches TabICL fed
`X.astype(float64)`; TabICL's own float32 path loses precision when
standardizing heavy-tailed columns before Yeo-Johnson. On a skewed synthetic
set, TabICL's float32 probabilities differed from its float64 ones by up to
0.23. Ironicl's results on float32 `X` are exactly its results on
`X.astype(float64)`.

Against TabICL 2.1.1 (commit `bd030d47ff7d9f987f0f33431d743a3003bfe0d3`), in the
live suite and the frozen fixture `tests/fixtures/preprocessing.json`:

- class order, encoded labels, feature masks, ensemble counts, group
  membership, and feature and class permutations are exact;
- preprocessed FP32 values are within `rtol=2e-6, atol=2e-6`;
- predictions are within `1e-4`.

Values are not bit-exact: summation order and the platform's math library change
the last bits of the float64 statistics.

Fitting copies the kept features once, column by column, as float64, and fits
each feature in parallel, keeping only its statistics. Transforming writes each
group's FP32 rows directly from the input.

## Missing Values

A missing numeric value is NaN, in Python and in Rust; the server reads `null`
as NaN.
Fitting takes each feature's mean over its values that are not missing, as
`SimpleImputer(strategy="mean")` does, and preprocessing replaces every missing
value with it before anything else. A feature missing in every training row is
constant once imputed, so the feature mask drops it; `TabICL` 2.1.1's imputer drops
it too, but then fails on every `predict` call, because its per-call mask of
missing features still counts that feature. Infinity is rejected, as `TabICL`
rejects it.

`TabICL`'s estimators also drop, for one `predict` call, any feature that is
missing in every row of that call, and skip the KV cache to do so. Ironicl does
not: it imputes the training mean, so a row preprocesses alike alone and in any
batch, and the cached context stays valid.

## Categorical Features

`TabICL`'s `TransformToNumerical` encodes a DataFrame's categorical columns with
scikit-learn's `OrdinalEncoder` and puts them first, ahead of the numeric
columns, which its `SimpleImputer` fills. Ironicl does the same in Rust:

- Python decides each column's type from its dtype, as `TabICL` does, and sends
  a categorical column as its distinct values and each row's code
  (`pandas.factorize`). The server reads strings, numbers, and booleans.
- Fitting keeps each categorical feature's categories: the sorted distinct
  values its training rows use, of one type (integers, floats, booleans, or
  text, sorted as `NumPy` sorts them). Categories that a pandas `Categorical`
  declares but no row uses are not kept, and the declared order is ignored, as
  in `TabICL`.
- A value becomes its category's index, or -1 when it is missing or training
  never saw it. Integers and floats compare as numbers, so `2.0` finds the
  category `2`.
- The preprocessor stores the categories per input feature in `IRPREP`, and
  moves categorical features first; from there, codes are numbers like any
  other: the constant-feature mask, normalization, and the ensemble treat them
  alike.

The categories are model-independent (`src/preprocess/columns.rs`), so another
model's preprocessing can use them, in its own column order.

## The Ensemble

TabICL draws its feature and class orders from `random.Random(random_state)`.
`rand_mt` provides the Mersenne Twister with `CPython`'s seeding
(`init_by_array` over the seed's 32-bit words); `python_random.rs` adds
`_randbelow`, `choice`, `shuffle`, and `sample` from `Lib/random.py`. Golden
draws from `CPython` 3.12, `tests/fixtures/python_random.json`, pin them.
`rand_mt` reproduces `torch.Generator`'s CPU stream too, for Kumo later.

TabICL builds a Latin square by inserting each drawn feature into every row, in
`O(H³)` time and `O(H²)` memory: 5 seconds at 4,000 features. The square has a
closed form: its first row lists the features in the order they are drawn, and
row `r` is that row rotated by `r`. Both shuffles draw only by the square's size,
so Rust computes each member's order from two shuffled index lists, in `O(H)`.
A test checks the closed form against the literal construction.

## The Power Transform

scikit-learn's `PowerTransformer` fits one Yeo-Johnson lambda per feature with
`SciPy`'s `yeojohnson_normmax`: `fminbound` within bounds that keep the
transformed values and their squares finite. A constant feature gets lambda 1.
For one-signed data, `SciPy` computes the log-likelihood's variance in log space;
Rust shifts the powers by their largest exponent instead, which avoids the same
overflow and loss of precision. `tests/fixtures/yeo_johnson.json` pins the
log-likelihood, the lambda, and the transformer.

## Next

- Fitting from Parquet in `ironicl fit`, with Arrow's dictionary columns as
  categorical features.
- Kumo's preprocessing, on the same pieces where it uses them.

## References

- [python.md](python.md): the estimators and their fitted attributes.
- [validation.md](validation.md#preprocessing-parity): the parity tests.
- `CPython`'s "Notes on Reproducibility": only `random()` is promised across
  versions, so golden draws pin the rest.

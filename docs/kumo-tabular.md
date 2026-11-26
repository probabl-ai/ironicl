# Kumo-Tabular

[Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular) is NVIDIA's tabular foundation model.
At size `large` it is second on [TabArena](https://huggingface.co/spaces/TabArena/leaderboard),
and first among models whose weights allow commercial use. Ironicl is adding it as its second
model, next to TabICLv2.

Status: the classifier and the regressor fit and predict on the CPU, on NVIDIA GPUs, and on Apple
Silicon GPUs, from Python and through `ironicl serve`.

```python
from ironicl import KumoTabularClassifier, KumoTabularRegressor

clf = KumoTabularClassifier(size="large").fit(X_train, y_train)
probabilities = clf.predict_proba(X_test)

reg = KumoTabularRegressor(size="large").fit(X_train, y_train)
mean = reg.predict(X_test)
```

`X` is a NumPy array or a pandas DataFrame with numeric and categorical columns, as for
`TabICLClassifier` ([python.md](python.md#supported-inputs)); NaN marks a missing value.
`n_estimators` defaults to NVIDIA's benchmark settings, 16 for `large` and 8 otherwise. The
checkpoint is downloaded from Hugging Face at the pinned revision, unless `model_path` names it.
Any number of classes works: above 10, members combine tasks of error-correcting output codes, as
NVIDIA's code does.

The regressor's members each predict 999 quantiles. As in NVIDIA's code, each member's are
sorted, then averaged over the members after dropping the lowest and highest fifth of them per
quantile. `predict` summarizes those quantiles as `TabICLRegressor.predict` does, with the same
`output_type` and `alphas` ([python.md](python.md#regressor)): `"mean"`, their mean, is what
NVIDIA's benchmark predicts, and `"raw_quantiles"` are NVIDIA's output.

## Precision

`optimize` selects the precision for fitting and predicting, not the size, preprocessing, or
ensemble. `"safe"` is the FP32 reference; `"auto"` and `"fast"` permit progressively less precise
arithmetic on backends that support it. Otherwise they run in FP32. An explicit value overrides
`IRONICL_OPTIMIZE`; with neither set the level is `"safe"`. The estimator records it in `optimize_`.
Saving or exporting a model preserves its level, even without the KV cache. As with TabICL,
`ironicl prepare --optimize` and `ironicl serve --optimize` rebuild a cache at a different level.

On CUDA, large transformer projections and MLPs use TabICL's tensor-core kernels: `auto` uses a
high and a low FP16 part per operand; `fast` uses just the high part. Small products, the cell
embedding, query gates, output heads, residuals, and training attention stay FP32. For classifiers
at `auto` or `fast` and regressors at `fast`, the GPU stores the column and ICL caches in FP16.
Predictions attend to these caches with FP16 queries. This halves resident KV storage, not the
entire model. Regression `auto` uses split products but retains FP32 caches. Both levels keep
medium and large's shared KV heads; neither uses FP8. The checkpoint, seed, and members do not change.

On Metal, the classifier's `auto` and `fast`, and the regressor's `fast`, store the in-context
learning (ICL) keys and values in FP16 on the GPU and use FP16 attention in that stage. Column
caches and all other stages stay FP32. Regression `auto` stays FP32: FP16 missed the `1e-3`
training-target standard deviation budget at the quantile tails. Exported caches use the same
portable FP32 tensors as before, with their build level recorded; loading on the CPU requires no GPU.

On CPUs with `AVX512_BF16`, `fast` stores retained ICL values in BF16 and keeps keys in FP32.
Training still uses FP32 attention; all CPU levels share TabICL's source-major online attention.
Other CPUs and CPU `auto` keep values in FP32. On a Ryzen 7 9800X3D (96 MiB L3), with 1,500
training rows, 256 query rows, 10 columns, and two small members, `fast` halves retained ICL value
storage (peak process RSS fell by about 37 MiB for classification) and cut single-row prediction
from 8.3 to 7.4 ms for classification and 8.5 to 7.5 ms for regression. Because both caches fit in
L3, 256-row batch prediction matched FP32 (`76.5` vs. `77.8` ms classification; `82.8` vs. `81.3` ms
regression), and compressing values added about 12 ms to the ~493 ms fit.

## Checkpoints

NVIDIA releases a classifier and a regressor in three sizes, as FP32 PyTorch files, at revision
`c5893d418e162b9f841ae179dc767afefff73831` of `nvidia/Kumo-Tabular`. A file is a bare state dict
with no configuration, so its caller says which size and task it is, and the import checks that it
holds exactly those tensors and shapes (`src/models/kumo_tabular/mod.rs`): NVIDIA's weights, or
fine-tuned ones of the same architecture. The architecture values come from
[`model.py`](https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/kumo/tabular/model.py#L24-L64),
not from the checkpoint. Python downloads them to `checkpoints/kumo-tabular/<size>-<task>.pt` in
Ironicl's cache folder.

| Size | Cell channels | Column and row layers | Inducing points | ICL channels | ICL layers | ICL heads | Classifier | Regressor |
|---|---|---|---|---|---|---|---|---|
| small | 128 | 4 | 128 | 512 | 12 | 8 | 110 MB | 114 MB |
| medium | 256 | 6 | 256 | 512 | 24 | 8 | 246 MB | 250 MB |
| large | 256 | 6 | 256 | 1024 | 24 | 16 | 855 MB | 863 MB |

The classifier outputs 10 logits and the regressor 999 quantiles. In medium and large, test rows
attend to the training rows through 2 of the ICL key and value heads.

```bash
ironicl inspect large-classifier.pt
ironicl import large-classifier.pt --model kumo-tabular
```

Import checks that a checkpoint holds exactly one release's tensors, by name and shape, takes that
release's size and task, and stores
them as `IRMODEL` with architecture 2 ([artifact-format.md](artifact-format.md#model)). The label
embeddings are saved transposed; import copies them into row-major order.

## How it runs

`src/models/kumo_tabular/`:

- `preprocess.rs`: NVIDIA's default recipe, fitted on the training rows one column at a time in
  `f64`. Each member standardizes, clips, and transforms every column by one of four choices in
  turn (none, Yeo-Johnson, robust scaling, rank to Gaussian), clips it at four sigmas, flips numeric
  columns' signs at random, shuffles the columns, and keeps 500. Categorical features become
  category codes, with a count column above 50 categories; classes shift by a random offset. A
  regressor's target is standardized and flips sign at random.
- `cpu.rs`: the forward pass in FP32 on the CPU kernels TabICLv2 uses. Fitting runs the training
  rows once and keeps each member's context: the column means that replace missing values, each
  column layer's keys and values of its inducing points, and each ICL layer's keys and values,
  in medium and large only the 2 heads query rows use. Predicting runs only the query rows.
- `cuda.rs`: on an NVIDIA GPU, with the `cuda` feature, a member's whole forward pass runs on
  the device and its context stays there: the imputation means and each layer's column
  and ICL keys and values. Predicting uploads only the query rows and downloads only the outputs.
  The blocks chain the kernels TabICLv2's ICL stage uses (`kernels/cuda/launch.rs`): `RMSNorm`,
  products with their bias, GELU, and residual epilogues, heads packed with `RoPE`, the per-head
  norm, and query scaling, and split attention over the cached keys and values, which test rows
  of medium and large read through 2 heads. The small model's 32-wide heads are zero-padded to
  the kernels' 64. Kumo-Tabular's own kernels build the cell embedding's features and the row
  stage's sequences. The weights upload once per process. Buffers are allocated and freed in
  stream order, so the CPU queues work without waiting for the GPU; it waits once 1 GiB has
  been allocated since it last did, which bounds the memory of queued work. A context that does not fit in device
  memory is built on the CPU, and `IRONICL_NO_CUDA` keeps everything there. When every member's
  context is on the GPU, the contexts are stacked and one pass predicts all members, as
  TabICLv2's ensemble does: products and norms run on every member's rows at once, reading each
  weight once, and each member attends to its own keys and values. Saving copies a
  context to the host; loading copies it to the GPU, so a context saved on one device loads on
  the other.
- `metal.rs`: on an Apple Silicon GPU, with the `metal` feature, a member's whole forward pass
  runs as one `MPSGraph` executable: one per training shape to fit, which outputs the
  member's column and ICL keys and values, and one per power-of-two query bucket to predict,
  which reads them. The blocks are the transformer block TabICLv2's Metal graphs use
  (`kernels/metal/block.rs`), with `RMSNorm`, `RoPE`, the per-head norm, query scaling, and shared
  key heads. The weights are bound to buffers shared with the CPU, and attentions run in chunks
  that bound the scores held at once. `IRONICL_NO_METAL` keeps everything on the CPU.
  Saving and loading copy contexts to and from the GPU, as for CUDA.
- `fitted.rs`: the ensemble. It averages the members' logits, or above 10 classes their codebook
  scores, and applies softmax; or it combines the members' quantiles.

Random draws: Ironicl draws the class offsets or target signs, the feature signs, the Latin column
orders, and the codebooks from CPython's `random.Random(random_state)`, in that order, so a seed gives the same
model on every machine. `random_state=None` is not supported.

Saving: `save`, pickling, and `export` work as for TabICL. They store the training data and the
seed as `IRPREP` with architecture 2 ([artifact-format.md](artifact-format.md#fitted-models)),
from which loading fits the preprocessing again, and by default the members' contexts as
`IRCNTXT` of kind 2 ([artifact-format.md](artifact-format.md#layout)). Without the contexts
(`save_kv_cache=False`, `export(kv_cache=False)`), loading fits them again, one forward pass over
the training rows, which gives the same model. The contexts are large: about 3 MB per column
and 25 KB per training row per member for size large, against a few bytes per value of
training data. Loading always needs the training data, so `save_training_data=False` raises.

## Parity

`tools/export_kumo_fixtures.py` runs NVIDIA's code, pinned to commit `3fdf108`, on the CPU in
FP32 and records each member's draws, preprocessed rows, and outputs, and the prediction, in
`tests/fixtures/kumo_tabular/`. Given NVIDIA's draws, Ironicl matches it:

| Case | Size | Members | Classes | Preprocessed rows | Outputs | Prediction |
|---|---|---|---|---|---|---|
| numeric, NaN, ±inf, constant and empty columns | small | 4 | 3 | < 1e-5 | 5e-6 | 6e-7 |
| numeric and categorical, a count column | small | 5 | 4 | < 1e-5 | 3e-6 | 2e-7 |
| 12 classes, error-correcting output codes | small | 2 | 12 | < 1e-5 | | 8e-8 |
| numeric and categorical | medium | 2 | 4 | | 4e-6 | |
| numeric and categorical | large | 2 | 4 | | 4e-6 | |
| numeric, regression | small | 5 | | < 1e-5 | 4e-6 | 3e-7 |
| numeric and categorical, regression | large | 2 | | < 1e-5 | 1e-5 | 2e-6 |

Outputs are the members' logits or standardized quantiles; the prediction is the probabilities,
or the quantiles relative to their size. The errors are the largest absolute ones, relative for
the preprocessed rows, on an Apple M1 Max and on an AMD Ryzen 7 9800X3D. The CUDA path, on an
RTX 5080, and the Metal path, on the M1 Max, match within the same bounds. The tests need the checkpoints in `IRONICL_CHECKPOINT_FIXTURES`:

```bash
IRONICL_CHECKPOINT_FIXTURES=checkpoints cargo test --release --lib kumo_tabular -- --include-ignored
```

On a Ryzen 7 9800X3D, with 1,000 training and 500 query rows of 20 features and 4 members, Ironicl
fits small in 1.3 s and predicts in 0.5 s, where NVIDIA's PyTorch code takes 0.8 s and 0.4 s;
large takes 6.2 s and 2.0 s, against 4.4 s and 1.7 s. Closing that gap is a follow-up.

With its RTX 5080, large with 4 members on 20 features and 500 query rows, after a first fit that
uploads the weights (1.8 s):

| Training rows | CPU fit | CUDA fit | CPU predict | CUDA predict |
|---|---|---|---|---|
| 1,000 | 5.3 s | 0.28 s | 2.1 s | 0.13 s |
| 5,000 | 39.3 s | 1.80 s | 3.3 s | 0.18 s |

NVIDIA's code on the same GPU, in FP32 and in FP16, fits in 0.24 s and 0.09 s and predicts in 0.10 s
and 0.04 s at 1,000 rows; at 5,000 rows it fits in 1.51 s and 0.42 s and predicts in 0.17 s and
0.05 s. In FP32 the remaining gap is the matrix product, about 1.4 times slower than cuBLAS's;
tensor-core products reduce that gap. To isolate their effect, the following measurements use
FP32 caches at **all** levels. Times are medians of three warm runs on the RTX 5080 with 1,000
training rows, 20 features, and four members:

| Model | Level | Fit | Predict 1 row | Predict 500 rows | Max error vs. `safe` |
|---|---|---:|---:|---:|---:|
| Large classifier | `safe` | 306 ms | 7.4 ms | 109 ms | — |
| | `auto` | 247 ms | 7.6 ms | 89 ms | 1.1e-6 |
| | `fast` | 200 ms | 6.8 ms | 70 ms | 4.3e-4 |
| Small regressor | `safe` | 111 ms | 4.0 ms | 55 ms | — |
| | `auto` | 104 ms | 4.0 ms | 54 ms | 3.4e-5 |
| | `fast` | 97 ms | 3.9 ms | 50 ms | 3.9e-3 |

Errors are absolute probability differences (no label changes here), or differences across all
999 quantiles divided by the training target's standard deviation. These timings exclude weight upload.

With FP16 ICL caches alone, the large classifier's `auto`/`fast` batch prediction took 79/62 ms;
with FP16 column caches too, 76/59 ms. The final maximum probability errors were `1.1e-4` and
`3.2e-4`. The small regressor at `fast` took 46 ms with a maximum error of `7.5e-3` training-target
standard deviations. Using FP16 caches for regression `auto` exceeded its `1e-3` error budget,
so it keeps the FP32-cache result in the table. On a wider table (100 features, 200 queries),
sampled peak process GPU memory was 4,638/4,594/4,128 MiB for `safe`/`auto`/`fast`. The weights
and temporary buffers also consume memory, so halving the KV cache does not halve the peak.

To reproduce the warm comparison with the installed package and cached checkpoints:

```bash
python tools/benchmark_kumo_precision.py classifier --size large --device cuda
python tools/benchmark_kumo_precision.py regressor --size small --device cuda
```

On an Apple M1 Max, medium with 4 members on 20 features and 500 query rows, after a first fit
that compiles the graphs (about 1.4 s), against NVIDIA's code on the GPU through PyTorch's MPS
backend:

| Training rows | CPU fit | Metal fit | MPS fit | CPU predict | Metal predict | MPS predict |
|---|---|---|---|---|---|---|
| 1,000 | 3.77 s | 0.48 s | 0.68 s | 1.73 s | 0.19 s | 0.30 s |
| 5,000 | 21.95 s | 2.88 s | 3.84 s | 2.31 s | 0.27 s | 0.42 s |

With 2,000 training rows, 20 columns, four members and 500 queries, the medium classifier on the
M1 Max fit in 1,032 ms at `safe` and 980 ms at `auto` (warm medians). Batch prediction took 213 ms
and 197 ms, single-row prediction 41 ms and 40 ms. Peak process RSS fell from 1.80 to 1.26 GiB.
The largest probability difference was `4.9e-5`, with no label changes. On the small regressor
at 1,000 rows, FP16 ICL attention's largest quantile difference was `1.9e-3` training-target
standard deviations, within the `fast` budget (`1e-2`) but not the `auto` budget.

## License

The weights are under OpenMDW-1.1, which allows commercial use, modification, conversion, and
redistribution. Distributing the weights, converted or not, requires the license text and NVIDIA's
notices. Ironicl downloads the weights from Hugging Face and never ships them. NVIDIA's code,
`structured-data-models`, is Apache-2.0; Ironicl reimplements it and refers to it by pinned links.

## Plan

1. Import the weights. Done.
2. The classifier on the CPU, as `KumoTabularClassifier`, with pickling, export, and the server.
   Done.
3. The regressor on the CPU, as `KumoTabularRegressor`. Done.
4. Metal. Done, with FP32 and FP16 ICL attention.
5. CUDA. Done, with FP32 and tensor-core products.

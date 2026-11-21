# Prediction Server

`ironicl serve` serves one exported model over HTTP, without Python. It is built with the `server`
feature:

```bash
cargo build --release --features server          # add metal on Apple Silicon, cuda on NVIDIA
```

## Workflow

1. Fit in Python and export the model ([python.md](python.md#exporting-for-the-server)):

   ```python
   clf = TabICLClassifier().fit(X, y)
   clf.export("churn")
   ```

2. Optionally, rebuild the KV cache on the serving machine or at another `optimize` level
   ([cli.md](cli.md#prepare)):

   ```bash
   ironicl prepare churn --optimize auto
   ```

3. Serve it:

   ```bash
   ironicl serve churn --bind 127.0.0.1:8080
   ```

The server loads the weights, the preprocessing, and the KV cache, checks that they belong
together, and predicts once before it opens the port, so any answer means it is ready. It predicts
at the level the cache records. It refuses a cache without a level, or one that does not match
the preprocessing, and says to run `ironicl prepare`.

| Option | Default |
|---|---|
| `--bind ADDR` | `127.0.0.1:8080` |
| `--max-rows N` | `10000` rows per request |
| `--max-body SIZE` | `16M` |

Predictions run one at a time. The server has no authentication or TLS: keep it on localhost or
put it behind a proxy that has them.

## `POST /predict`

The body is JSON. Rows are arrays of numbers, each as long as the training data's feature count.
Options are named after TabICL's predict arguments; the response has one key per output.

| Field | Type | Default |
|---|---|---|
| `rows` | `number[][]`, at least one row | required |
| `output_type` | list of output names, none twice | classifier: `["labels", "probabilities"]`; regressor: `["mean"]` |
| `alphas` | regressor only: quantile levels in (0, 1) | `[0.1, 0.2, …, 0.9]` |

| Task | Output | Response keys |
|---|---|---|
| classifier | `labels` | `labels`: one per row, of the training labels' type |
| classifier | `probabilities` | `probabilities`: `[rows][classes]`, and `classes`, their order |
| regressor | `mean`, `median`, `variance` | the same name: one number per row |
| regressor | `quantiles` | `quantiles`: `[rows][alphas]`, and `alphas`, their order |
| regressor | `raw_quantiles` | `raw_quantiles`: `[rows][999]`, TabICL's sorted model quantiles |

A classifier uses the `softmax_temperature` and `average_logits` it was fitted with.

```json
→ {"rows": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}
← {"labels": ["setosa", "virginica"],
   "classes": ["setosa", "versicolor", "virginica"],
   "probabilities": [[0.99999, 0.00001, 0.0], [0.00001, 0.00132, 0.99867]]}

→ {"rows": [[5.1, 3.5, 1.4, 0.2]], "output_type": ["labels"]}
← {"labels": ["setosa"]}

→ {"rows": [[0.03, 0.05, ...]], "output_type": ["mean", "quantiles"], "alphas": [0.1, 0.9]}
← {"mean": [195.58], "alphas": [0.1, 0.9], "quantiles": [[103.75, 285.18]]}
```

Missing values (`null`) and categorical features (strings) are not supported yet and return 400.

## `GET /health`

`{"status": "ok"}`.

## Errors

Errors are `{"error": {"code": ..., "message": ...}}`.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Malformed JSON, a wrong type, an unknown field, an empty or ragged `rows`, an unknown or repeated output, or `alphas` outside (0, 1). The message says which. |
| 400 | `wrong_feature_count` | Rows of another length than the training data's. |
| 400 | `too_many_rows` | More rows than `--max-rows`. |
| 413 | `body_too_large` | A body larger than `--max-body`. |
| 500 | `internal` | A failed prediction. |

# Prediction Server

`ironicl serve` serves one model directory over HTTP, without Python. It is built with the `server`
feature:

```bash
cargo build --release --features server          # add metal on Apple Silicon, cuda on NVIDIA
```

## Workflow

1. Fit in Python and export the model directory
   ([python.md](python.md#exporting-for-the-server)):

   ```python
   clf = TabICLClassifier().fit(X, y)
   clf.export("my-model")
   ```

2. On the serving machine, make sure the weights are in the weights folder (`export` already put
   them there on its own machine) and, optionally, build the KV cache for this device and level
   ([cli.md](cli.md#prepare)):

   ```bash
   ironicl import tabicl-classifier-v2-20260212.ckpt
   ironicl prepare my-model --optimize auto
   ```

3. Serve it:

   ```bash
   ironicl serve my-model --bind 127.0.0.1:8080
   ```

The server finds the model's weights by its checkpoint's SHA-256, loads the preprocessing, and
prepares the KV cache as `ironicl prepare` does: it reuses `context.ircntxt` when it is exactly
the cache this model builds at the level, and otherwise builds it and replaces the file. It then
predicts a full batch before it opens the port, so any answer means it is ready.

| Option | Default |
|---|---|
| `--weights DIR` | the shared weights folder ([cli.md](cli.md#model-directories-and-weights)) |
| `--optimize safe\|auto\|fast` | the saved cache's level, else `safe` |
| `--bind ADDR` | `127.0.0.1:8080` |
| `--max-request-rows N` | `10000` rows per request |
| `--max-body SIZE` | `16M` |
| `--max-batch-rows N` | `1024` on a GPU, `32` on the CPU |
| `--max-queue MS` | `1000` |

The server has no authentication or TLS: keep it on localhost or put it behind a proxy that has
them. Each connection takes a file descriptor: for thousands of connections, raise the open-file
limit (`ulimit -n`, or `LimitNOFILE` under systemd).

## Batching

One worker thread owns the model. Requests queue for it, and it predicts the queued ones
together: the oldest request, plus the requests queued after it while they fit in
`--max-batch-rows` rows. It never waits for more requests, so at light load a batch is a single
request, and under load batches fill up while the previous one runs. A request's rows always stay
in one batch, even when they alone exceed the limit. Requests whose client has disconnected are
dropped before predicting.

- `--max-batch-rows` trades latency for throughput. On a GPU, up to 1,024 rows run as one pass;
  a larger batch runs as several passes in a row, which adds latency but little throughput.
- `--max-queue` bounds how long a request may wait. The server measures the time per row of
  full batches and refuses a request with `503 overloaded` and `Retry-After: 1` when the rows
  queued ahead of it, plus its own, would take longer than this to predict. An empty queue
  accepts any request. A request's latency is its wait, plus up to one batch that is already
  running, plus its own batch.

The `Server-Timing` response header reports each request's `queue` and `predict` times in
milliseconds.

## Throughput and latency

Measured with [`tools/bench_server.py`](../tools/bench_server.py) and
[oha](https://github.com/hatoo/oha) on an AMD Ryzen 7 9800X3D with an NVIDIA RTX 5080 (16 GB)
and 64 GB of RAM, the load generator on the same machine, for a classifier with 4,000 training
rows and 50 features. Requests arrive at a fixed rate (open loop); latencies are of successful
requests.

The GPU is the limit: at saturation it runs at 98 to 99%, at about 22,000 rows/s at `auto` and
33,000 at `fast`. Batches above about 2,000 rows add nothing. HTTP and JSON are not the limit:
the server answered 31,000 one-row requests a second at `fast`.

For one-row requests, with `--max-batch-rows 256 --max-queue 10`:

| Requests/s offered | `auto` p50 / p99 | `fast` p50 / p99 |
|---|---|---|
| 2,000 | 3.9 / 5.1 ms | 2.7 / 3.8 ms |
| 10,000 | 6.0 / 8.1 ms | 3.9 / 5.1 ms |
| 15,000 | 10.1 / 13.8 ms | 4.5 / 5.8 ms |
| 25,000 | over capacity: 17,300/s served, p99 20.2 ms | 7.9 / 11.0 ms |
| 30,000 | over capacity: 17,300/s served, p99 20.2 ms | over capacity: 27,800/s served, p99 19.9 ms |

Over capacity, throughput holds at the maximum and the excess gets 503. For five minutes at about
80% of capacity (20,000 requests/s at `fast`), every request was answered, p99.9 was 7.5 ms, and
server and GPU memory stayed flat after the first 30 seconds.

Two settings cover most uses:

| Use | Settings | Result on the machine above |
|---|---|---|
| Scoring single records within a deadline | `--max-batch-rows 256 --max-queue 10` | 25,000 requests/s at 11 ms p99 at `fast`; at most about 20 ms over capacity |
| Bulk scoring | the defaults | 21,600 rows/s at `auto`, 32,300 at `fast`; requests wait at most about 1 s |

The cost of a prediction grows with the training rows. With 50,000 training rows and 100
features, the KV cache is 9.5 GiB, one row takes 7 ms at `fast`, and the server scores 4,000
one-row requests a second at 18.7 ms p99. If the cache does not fit the GPU, it streams from host
memory on every batch, and throughput depends on the batch size: 23% of the in-memory rate at
1,024 rows, 89% at 8,192.

## Several models on one GPU

One process serves one model. To serve several on one GPU, run one process per model on its own
port, each with an `IRONICL_DEVICE_MEMORY_LIMIT` that covers its KV cache. Two servers of the
model above shared the RTX 5080 at 24,400 requests/s together, against 27,700 for one alone. On
a machine with several GPUs, select one per process with `CUDA_VISIBLE_DEVICES`.

## `POST /v1/predict`

The body is JSON. Rows are arrays of numbers, each as long as the training data's feature count.
Options are named after TabICL's predict arguments; the response has one key per output.

| Field | Type | Default |
|---|---|---|
| `rows` | `number[][]`, at least one row | required |
| `output_type` | list of output names, none twice | classifier: `["labels", "probabilities"]`; regressor: `["mean"]` |
| `alphas` | regressor only: up to 999 quantile levels in (0, 1) | `[0.1, 0.2, …, 0.9]` |

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

## `GET /v1/health`

`{"status": "ok"}`.

## Errors

Errors are `{"error": {"code": ..., "message": ...}}`.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Malformed JSON, a wrong type, or an unknown field; `rows` empty, ragged, of another length than the training data's, or longer than `--max-request-rows`; an output the model does not have, or one named twice; `alphas` for a classifier, outside (0, 1), or more than 999 of them. The message names the field. |
| 413 | `body_too_large` | A body larger than `--max-body`. |
| 503 | `overloaded` | The request would wait longer than `--max-queue`. Retry after the `Retry-After` seconds. |
| 500 | `internal` | A failed prediction. It fails every request in the same batch. |

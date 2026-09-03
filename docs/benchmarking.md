# Benchmarking

Use an imported Ironicl model artifact. The command does not accept
checkpoints, because checkpoint import and conversion must stay outside
comparable inference timing.

```bash
cargo run --release -- benchmark --model model.ironicl --input request.irinput \
  --warmup 2 --iterations 10
```

For cached query-only inference:

```bash
IRONICL_THREADS=8 cargo run --release --no-default-features -- benchmark \
  --model model.ironicl \
  --context context.ironiclctx \
  --input request.irquery \
  --warmup 5 --iterations 50
```

Add `--probabilities` to cached mode to time class-permutation correction,
estimator averaging, and final probability production. Set
`IRONICL_SERIAL_ENSEMBLES=1` only for a control run that disables automatic
estimator-level parallelism. `IRONICL_ENSEMBLE_WORKERS=N` selects an explicit
estimator concurrency for scheduler tuning; each estimator receives
`IRONICL_THREADS / N` inner workers. The executor ignores override combinations
that can exhaust the shared pool while nested work is pending.
`IRONICL_INNER_WORKERS=N` overrides the per-estimator budget for controlled
oversubscription experiments.
`IRONICL_EXACT_GELU=1` and `IRONICL_EXACT_SOFTMAX=1` restore scalar control
paths for accuracy and latency comparisons on native targets.

The command reads and validates the input bundle and model artifact. Cached mode
also reads and validates the context. It constructs the executor, runs untimed
warmups, then times each prediction call separately. Latency excludes
preprocessing, all artifact and bundle loading, executor construction,
checkpoint import, and conversion. Context load time is excluded but is not
reported as a separate field.

`warmup` defaults to `2` and accepts `0..=1000`. `iterations` defaults to `10`
and accepts `1..=10000`.

## Output Contract

Standard output contains these `key: value` lines in this order:

```text
input: PATH
model: PATH
context: PATH  # present only for cached execution
input_load_ms: MILLISECONDS
model_load_ms: MILLISECONDS
warmup: COUNT
iterations: COUNT
output: estimator_logits|probabilities
shape: ENSEMBLES QUERY_ROWS CLASSES
min_ms: MILLISECONDS
p50_ms: MILLISECONDS
p95_ms: MILLISECONDS
p99_ms: MILLISECONDS
max_ms: MILLISECONDS
mean_ms: MILLISECONDS
inferences_per_second: RATE
query_rows_per_second: RATE
output_checksum: CHECKSUM
logits_checksum: CHECKSUM
samples_ns: COMMA-SEPARATED_NANOSECONDS
```

All duration and rate values are decimal values. `output_checksum` is the FP64
sum of the final timed output. `logits_checksum` is also present for the default
estimator-logit mode to preserve the previous parser contract. Probability
shape is reported as `1 QUERY_ROWS CLASSES`. The command rejects a run when a
timed output changes shape or contains a non-finite value. Percentiles use the
nearest-rank method: for sorted samples, percentile `p` is sample `ceil(p * N)`,
where ranks start at one.

When internal profiling is compiled and `IRONICL_PROFILE` is set, phase totals
are written to standard error after the standard benchmark fields.

Run competing thread-count or kernel variants sequentially. Concurrent runs on
the same physical cores invalidate latency comparisons. Record whether the
machine was otherwise idle and keep model/context loading outside timed calls.

See `real-dataset-benchmarks.md` for the Ryzen, RTX 5080, and real-dataset
memory-saturation protocol and results.

## Official sklearn Cached Serving

`tools/benchmark_tabicl_sklearn_cache.py` benchmarks the public
`TabICLClassifier.fit()` and `predict_proba()` APIs. It uses a deterministic,
stratified Covertype subset, an official local checkpoint, and either
`kv_cache="kv"` or `kv_cache="repr"`. The `fit_cache_build_ms` field includes
model loading, preprocessing, and cache construction. Timed latency includes
only cached `predict_proba()` calls after warmups.

The JSON result records context and query sizes, cache mode, precision, output
shape and checksums, warmup and iteration counts, plus peak process RSS and
CUDA allocated/reserved VRAM for the fit/cache-build and predict phases. It also
records every prediction sample, the CPU affinity mask, and fixed inter-op
thread count `1`. CUDA samples synchronize before and after each prediction.

Public `predict_proba()` timing includes query validation, query preprocessing,
class correction, and probability aggregation. Compare it only with Ironicl
`--probabilities` when both sides start at an equivalent query boundary. Use a
separate prepared-query benchmark for kernel-only estimator-logit comparisons.

## Raw-X Cached Serving

`tools/benchmark_cached_end_to_end.py` is the required cross-engine latency
contract. It keeps the training K/V cache resident but times, for every sample:

- raw numeric query validation;
- TabICL numerical encoding and ensemble variant generation;
- cached model execution;
- class correction, logit averaging, temperature softmax, and host FP32 output.

It excludes model/context loading and cache construction. Run each engine in a
separate process. Then use `tools/compare_cached_end_to_end.py` to reject
results with different fixture, prepared-query, cache, model, output, or timing
contract identities before reporting a ratio. The strict lane requires FP32 K/V
and exact Ironicl GELU and softmax controls. AMP is a separate lane and must
pass its own probability and label gates before it can be compared.

On the Ryzen benchmark host, run all required CPU and CUDA FP32 points with:

```bash
/home/semyon/Projects/ironicl-bench/src/Ironicl/tools/run_ryzen_tabicl_sklearn_cache_benchmarks.sh
```

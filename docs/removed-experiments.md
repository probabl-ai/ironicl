# Removed Experiments

This log records design work that Ironicl has removed. Each section says what
the work was, how it worked, what it was used for, and why it was removed. Links
point at the last commit that has the code,
[`e91a75d`](https://github.com/probabl-ai/ironicl/tree/e91a75d3995c52456b282a47138764fa064d81e1),
so the code stays readable after the removal. To restore a piece, start from
those files.

Every section uses the same headings: **What it was**, **How it worked**,
**What it showed**, **Why it was removed**, and **Code**.

## Python static-context exporter, IRQUERY, IRLOGITS, and IRTRACE

**What it was.** A Python tool, `tools/export_static_context.py`, that ran
official TabICL on a training table, captured its full key/value cache, and
wrote it as an `IRCNTXT` static context. It also wrote the preprocessed query
rows as an `IRQUERY` bundle and, with `--raw-logits-output`, TabICL's own
per-estimator logits as an `IRLOGITS` oracle. `tools/compare_cached_logits.py`
compared Ironicl's output with that oracle.

**How it worked.** As [artifact-format.md](artifact-format.md) described,
`IRQUERY` v1 had a 112-byte `<8sII32s32sQQQQ>` header with the model and context
SHA-256 values, `B`, `Q`, `H`, and an element count, then FP32 `[B,Q,H]` query
rows and a SHA-256 trailer. The context and query hashes had to match before
`StaticExecutor::predict_bundle` ran. `IRLOGITS` v1 had a 48-byte `<8sIIQQQQ>`
header, FP32 `[B,Q,C]` logits, and a SHA-256 trailer. TabICL preprocessed the
queries and applied the ensemble permutations in Python, so the Rust cached
executor started from those variants ([static-context.md](static-context.md)).

**What it showed.** It was the first end-to-end path for cached Rust execution:
a context built by TabICL, served by Ironicl. The Ryzen benchmark commands in
[experiments.md](experiments.md) used it to time `--probabilities` output and
to compare final probabilities with the TabICL oracle at `atol=1e-3`.

**Why it was removed.** Ironicl now preprocesses in Rust, builds the context in
Rust, and saves it from a fitted model, so nothing needs a context or query
bundle written by Python. The parity tests compare the Python estimators with
TabICL directly.

**Code.**

- Exporter: [`tools/export_static_context.py`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/export_static_context.py#L1-L510)
  and its tests, [`tests/test_export_static_context.py`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tests/test_export_static_context.py#L1-L129)
- Comparison: [`tools/compare_cached_logits.py`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/compare_cached_logits.py#L1-L142)
- `IRQUERY` reader: [`src/format/query_bundle.rs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/format/query_bundle.rs#L1-L207)
- Identity-checked execution: [`StaticExecutor::predict_bundle` and `predict_bundle_probabilities`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L2350-L2386)

## External PyTorch oracle test

**What it was.** A fixture-gated Rust test,
`matches_python_classifier_oracle_when_fixtures_are_available`, that compared
Ironicl's uncached classifier logits with logits from TabICL's PyTorch model.
`tools/export_tabicl_oracle.py` wrote the fixtures: an `IRINPUT` request and an
`IRLOGITS` result.

**How it worked.** As [validation.md](validation.md) described, the exporter ran
a sibling TabICL checkout on CPU in FP32 with one thread, no AMP, and
`torch.no_grad()`, and accepted only the released classifier checkpoint. It
wrote a direct-neural fixture with `X` of shape `[1, 5, 3]`, four training
labels, and one test row, without sklearn preprocessing. `IRINPUT` v1 had an
80-byte header, FP32 `X` and `y_train`, and a SHA-256 trailer. With
`--trace-output`, it also wrote an `IRTRACE` file of coarse stage outputs. The
Rust test read the fixtures from `IRONICL_ORACLE_FIXTURES` and compared every
logit at `rtol=1e-4`, `atol=1e-4`.

**What it showed.** It backed the claim that FP32 logits match the PyTorch
oracle within `1e-4` in the claim-to-check table of
[experiments.md](experiments.md).

**Why it was removed.** It tested preprocessed tensors written by Python, a path
no user takes now that Ironicl preprocesses in Rust. The live parity tests in
`tests/parity` compare the Python estimators with TabICL from raw rows, and
`built_context_matches_the_uncached_forward_pass` still checks the built context
against the uncached forward pass.

**Code.**

- Exporter: [`tools/export_tabicl_oracle.py`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/export_tabicl_oracle.py#L1-L228)
- `IRINPUT` reader: [`src/format/input_bundle.rs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/format/input_bundle.rs#L1-L496)
- Test: [`matches_python_classifier_oracle_when_fixtures_are_available`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tests/checkpoint_fixtures.rs#L151-L200)
  and its fixture readers, [`OracleFixtures` and `read_irlogits`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tests/checkpoint_fixtures.rs#L442-L570)

## `ironicl run`, `ironicl benchmark`, and `--trace-stages`

**What it was.** Two CLI commands that ran a model on preprocessed tensor
bundles. `run` printed per-estimator logits, or final probabilities for a cached
query. `benchmark` timed repeated predictions and reported latency percentiles.
`run --trace-stages` printed summaries of intermediate stage outputs.
`tools/prepare_real_dataset.py` wrote `IRINPUT` requests for them, and
`tools/benchmark_tabicl.py`, `tools/benchmark_scale.py`, and
`tools/run_ryzen_benchmarks.sh` timed them against TabICL.

**How it worked.** As [cli.md](cli.md) described, `run` took a checkpoint or
model with an `IRINPUT` request, or a model, a `--context`, and an `IRQUERY`
request. A checkpoint was imported into a content-addressed cache first;
`--no-cache` only validated it, so `run` rejected it. `--trace-stages` kept the
column embeddings, row representations, and raw decoder output of every
estimator, and was limited to half the activation budget. As
[benchmarking.md](benchmarking.md) described, `benchmark` accepted only an
imported model, ran untimed warmups, and timed each prediction alone, so
latency excluded preprocessing, artifact loading, and executor construction.

**What it showed.** The historical scaling tables in
[real-dataset-benchmarks.md](real-dataset-benchmarks.md) came from `IRINPUT`
requests preprocessed by TabICL and timed with these tools. The stage traces
supported early parity debugging against `IRTRACE`.

**Why it was removed.** Both commands read only the removed tensor bundles.
Ironicl now preprocesses in Rust, so a CLI that serves predictions should take
raw rows and a fitted model; that comes later. The Python estimators are the
supported way to fit and predict meanwhile.

**Code.**

- Command handlers and output: [`src/main.rs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/main.rs#L197-L446)
  and their help text, [`src/main.rs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/main.rs#L470-L495)
- Argument parsing: [`RunArgs`, `BenchmarkArgs`, `CheckpointCacheOptions`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/cli.rs#L67-L95)
  and [`parse_run`, `parse_benchmark`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/cli.rs#L263-L401)
- Latency statistics: [`src/benchmark.rs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/benchmark.rs#L1-L70)
- Stage traces: [`TraceStage` and `ForwardTrace`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L201-L258),
  [`predict_logits_with_trace`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L977-L1065),
  and [`admit_trace_request`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L4938-L4959)
- Tools: [`tools/prepare_real_dataset.py`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/prepare_real_dataset.py#L1-L162),
  [`tools/benchmark_tabicl.py`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/benchmark_tabicl.py#L1-L356),
  [`tools/benchmark_scale.py`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/benchmark_scale.py#L1-L200),
  and [`tools/run_ryzen_benchmarks.sh`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/run_ryzen_benchmarks.sh#L1-L34)

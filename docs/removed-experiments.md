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

## Dormant experiment features

The four sections below cover experiments that stayed in the tree after they
failed their gates. Each was first switched on by an `IRONICL_*` environment
variable. Commit `3c30317` (2026-11-04) replaced those variables with Cargo
features that defaulted to off, so the code compiled in every build but never
ran. The measurements below come from the runtime-flag era on the Ryzen 7
9800X3D (Zen 5) reference host. They were already recorded in
[experiments.md](experiments.md) and [pytorch-cpu-comparison.md](pytorch-cpu-comparison.md)
when the history begins (commit `20929b2`, 2026-09-03); neither file gives a
measurement date.

## Sparse ICL attention (`sparse-attention`)

**What it was.** Candidate retrieval for ICL attention. A per-head index over
the training-row keys let a query score only the rows that could hold softmax
mass, then fall back to dense attention when it could not prove that the
skipped rows were negligible.

**How it worked.** `tools/export_static_context.py --sparse-index` built a
deterministic K-means index for each estimator, ICL layer, and head and stored
it in the context as four FP32 tensors: `icl.index.centroids` `[B,12,8,M,64]`,
`icl.index.radii` `[B,12,8,M]`, `icl.index.offsets` `[B,12,8,M+1]`, and
`icl.index.members` `[B,12,8,N]` (cluster-sorted row numbers). The index was
all-or-nothing: on load, `validate_sparse_index_contract` rejected a partial or
malformed index. For each query row and head, the kernel bounded every
cluster's best score by `q·c + |q|·r`, visited clusters in bound order, scored
their members exactly, and stopped once the bounded mass of the unvisited
clusters was at most `1e-5` of the visited mass. It gave up and ran the dense
kernel when the candidates passed a quarter of the rows, when any value was not
finite, or when any head failed. It ran only for FP32-key/BF16-value contexts
with packed (v3) keys and head dimension 64. `StaticExecutor` counted attempts
and acceptances, exposed through `sparse_attention_statistics()` and the
re-exported `SparseAttentionStatistics`.

**What it showed.** On `B=8,N=4096,Q=128` Covertype, an exact-score oracle
over 384 ICL attention calls found the median softmax mass in the top 64, 256,
and 1024 rows was 17.0%, 39.1%, and 72.9%, so a fixed top-K policy was ruled
out. Moving the index from 16 to 64 clusters grew the artifact from
1,403,102,496 to 1,412,834,592 bytes. Across 1,920 sparse ICL calls in 20 timed
requests, the M=64 index accepted none: every call fell back to dense
attention. Raising the certificate to `1e-3`, `1e-2`, and `1e-1` still accepted
none. The best M=64 time, `195.00 ms` p50, tied the `195.17 ms` indexed-dense
control ([experiments.md](experiments.md#sparse-index-quality),
[pytorch-cpu-comparison.md](pytorch-cpu-comparison.md#attention-layout-and-sparsity)).

**Why it was removed.** The index never accepted a call, so it never skipped
any work. It also needed packed (v3) keys, which only the Python
exporter wrote, and that exporter is gone. Contexts that still carry
`icl.index.*` tensors now load with the tensors ignored, instead of having
the index validated.

**Code.**

- Index tensors and lookup: [`SPARSE_INDEX_TENSORS`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L55-L60),
  [`SparseIndex`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L86-L93), and
  [`StaticExecutor::sparse_index`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L3359-L3409)
- Kernel and certificate: [`sparse_cached_attention_f32_key_bf16_value` and `sparse_max_omitted_mass`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L4296-L4433)
- Load-time validation: [`validate_sparse_index_contract`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L5021-L5122),
  called from [`validate_static_cache_contract`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L4125-L4143)
- Statistics: [`SparseAttentionStatistics` and `SparseAttentionStats`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L328-L367),
  [the executor methods](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L2040-L2049), and
  [the `lib.rs` re-export](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/lib.rs#L48-L51)
- Index builder: [`sparse_index_tensors` and `deterministic_kmeans_index`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/export_static_context.py#L104-L177)
  in the removed exporter
- Restore: add `sparse-attention = []` under `[features]` in
  [`Cargo.toml`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/Cargo.toml#L30-L36), restore the code above with
  [`sparse_attention_enabled`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L4288-L4290), and hook
  it in where [`cached_icl_block`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L2518-L2541)
  fetches the index and where [`cached_block`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L2671-L2704)
  tries it before the dense FP32-key/BF16-value kernel. A context also needs
  packed keys and an index writer again.

## Packed FP32 and BF16 dense layers (`bf16-dense`)

**What it was.** Prepacked weights for the cached executor's affine layers:
an AVX-512 FP32 layout and a BF16 layout run with `VDPBF16PS` and FP32
accumulation. It covered the column and ICL Q projections, attention outputs,
both feed-forward layers, QASS, the decoder, and a packed row-stage block.

**How it worked.** `PackedF32Affine` packed `[output, input]` weights as
`[output_block, input, 16 lanes]` so one ZMM load fed 16 output channels.
`PackedBf16Affine` rounded weights to BF16 and packed them as
`[output_block, input_pair, 16 lanes, 2]`, so one `VDPBF16PS` updated a whole
output block. It also kept a BF16 residual of each weight. `Bf16DensePolicy`
chose the stages. At construction, `StaticExecutor` packed the chosen
weights into maps keyed by tensor name. `parallel_affine_named` then sent a
layer to the BF16 kernel, else to the packed FP32 kernel, else to the ordinary
affine path. `Bf16AffineCompensation` could add the input residual, the weight
residual, or both as first-order correction terms. The Cargo feature turned on
the full policy with no compensation; the runtime flags it replaced also chose
stages, blocks, and compensation. The kernels reported `false` without
AVX-512 (and `AVX512_BF16` for BF16), so the ordinary path ran instead.

**What it showed.** All on `B=8,N=4096` Covertype, against the
[rejected-experiments table](experiments.md#rejected-experiments):

- Packed FP32 dense at `Q=500`: `885.84 ms` against `752.07 ms`. Rejected.
- BF16 ICL FFN1 at `Q=128`: `188.96 ms`, but `1.978e-3` maximum probability
  error, over the `1e-3` gate.
- BF16 ICL FFN1 and FFN2: `182.90 ms` with `2.379e-3` error. Rejected.
- BF16 ICL FFN2 with a reused conversion workspace: `189.92 ms` against
  `193.13 ms` with `9.744e-4` error; the 1.69% gain was below the 5% gate.
- BF16 ICL FFN1 in blocks 6-11 only: `193.40 ms` against `193.13 ms`.
- BF16 decoder: `192.82 ms` against `192.50 ms`.

All BF16 runs kept 100% label agreement.
[pytorch-cpu-comparison.md](pytorch-cpu-comparison.md#dense-fusion-and-reduced-precision)
summarizes the same results. No result was recorded for the compensation terms
or the packed row-stage block. The ignored Zen 5 kernel benchmarks in
`src/kernels/cpu/tests.rs` printed timings but recorded none in the repo.

**Why it was removed.** Every configuration either missed the latency gate or
spent the error budget. The Cargo feature enabled the full policy, which
includes the FFN1 configuration that failed the error gate.

**Code.**

- Executor wiring: [`Bf16DensePolicy`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L68-L85) and
  [its `full` policy](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L95-L112),
  [the BF16 input thread-locals](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L63-L66),
  [the `StaticExecutor` fields](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L319-L322),
  [construction](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L1777-L1809), and
  [weight packing](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L3428-L3762)
- Dispatch: [`parallel_affine_named`, `parallel_packed_bf16_affine`, `parallel_packed_f32_affine`, and `run_packed_bf16_affine`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L2852-L3008)
- Packed row block: [the `uncached_block` hook](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L3161-L3163)
  and [`packed_row_block`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L3177-L3337)
- Packing types: [`PackedF32Affine`, `Bf16AffineCompensation`, and `PackedBf16Affine`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/affine.rs#L7-L306)
- AVX-512 kernels: [block sizes](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L26-L30),
  [BF16 support check](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L52-L57),
  [FP32-to-BF16 conversion](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L66-L108),
  [packed FP32 and BF16 affine kernels](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L110-L328),
  and [the portable stubs](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/pointwise.rs#L26-L40)
- Tests and ignored Zen 5 benchmarks: [`src/kernels/cpu/tests.rs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/tests.rs#L193-L604)
- Restore: add `bf16-dense = []` under `[features]` in
  [`Cargo.toml`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/Cargo.toml#L30-L36), restore the code above, and replace
  the `parallel_affine` calls in `cached_block`, `cached_icl_decoder`, and
  `qass` with `parallel_affine_named` and their weight names.

## BF16 attention value dot product (`bf16-value-dot`)

**What it was.** A `VDPBF16PS` kernel for the value step of FP32-key/BF16-value
ICL attention: it multiplied BF16 softmax weights by BF16 values in pairs, with
FP32 accumulation.

**How it worked.** For each 128-row key tile and head,
`pack_attention_value_bf16` repacked the source-major BF16 values as
`[dimension_block, source_pair, 16 lanes, 2]`. `attention_value_packed_bf16_dot_f32`
rounded the FP32 softmax weights to BF16 and ran
`attention_value_packed_bf16_dot_avx512` for up to eight query rows. When it
declined (no `AVX512_BF16`, more than eight rows, or an odd tile), the retained
`attention_value_tile_bf16_fma_f32` kernel ran, which widens the values to
FP32 and keeps the weights in FP32.

**What it showed.** On the fair raw-X `B=8,N=4096,Q=128` Covertype run it took
`203.57 ms` against `192.50 ms`, with the same final output. Runtime packing
and conversion cost more than the faster dot product saved
([experiments.md](experiments.md#rejected-experiments),
[pytorch-cpu-comparison.md](pytorch-cpu-comparison.md#attention-layout-and-sparsity)).

**Why it was removed.** It was slower, and the retained FMA kernel already
serves `optimize="fast"`. A faster version would need values stored in the
packed layout, not repacked for every tile.

**Code.**

- Packing and dispatch: [`pack_attention_value_bf16` and `attention_value_packed_bf16_dot_f32`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/attention.rs#L252-L322)
  and [the portable stub](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/attention.rs#L472-L482)
- AVX-512 kernel: [`attention_value_packed_bf16_dot_avx512`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L1150-L1257)
- Hook: [scratch buffers](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1301-L1303),
  [per-tile packing](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1313-L1320), and
  [the value step](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1427-L1449) in
  `source_major_online_attention_f32_key_bf16_value_with_backend_f32`
- Tests and an ignored Zen 5 benchmark: [`src/kernels/cpu/tests.rs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/tests.rs#L35-L191)
- Restore: add `bf16-value-dot = []` under `[features]` in
  [`Cargo.toml`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/Cargo.toml#L30-L36), restore the code above with
  [`bf16_attention_value_dot_enabled`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1506-L1508).
  The kernel also needs `convert_f32_to_bf16` and the BF16 support check from
  the `bf16-dense` code.

## Attention mass profile (`attention-stats`)

**What it was.** A diagnostic that showed how concentrated ICL attention is,
to judge whether sparse attention could pay off.

**How it worked.** In the FP32-key/BF16-value attention kernel, for sources of
at least 4096 rows, it copied the scores of the first query row. After the
call, `report_attention_stats` sorted them, computed the softmax
mass in the top 64, 256, 1024, 4096, and 16,384 rows, and printed one
`attention_stats` line to stderr.

**What it showed.** On `B=1,N=100000,Q=128`, the first query of all 96 ICL
head/layer calls kept 4.1% to 100% of its mass in the top 4096 rows (median
62.0%) and 16.4% to 100% in the top 16,384 (median 89.2%). Some heads were close
to uniform, so a small fixed top-K was unsafe
([experiments.md](experiments.md#large-context-result)).

**Why it was removed.** It answered its question, and the sparse-attention
work it served is removed as well. The `profile` feature still times the
attention stages.

**Code.**

- Profile: [`attention_stats_enabled` and `report_attention_stats`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1510-L1537)
- Hook: [score buffer](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1304-L1305),
  [score copy](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1382-L1386), and
  [report](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1483-L1485)
- Restore: add `attention-stats = []` under `[features]` in
  [`Cargo.toml`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/Cargo.toml#L30-L36) and restore the code above.

## The `experimental` umbrella feature

**What it was.** A Cargo feature that turned on all four features above.

**How it worked.** `experimental = ["bf16-dense", "bf16-value-dot",
"sparse-attention", "attention-stats"]` in
[`Cargo.toml`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/Cargo.toml#L30-L36).

**What it showed.** No recorded measurement.

**Why it was removed.** Nothing is left to turn on.

**Code.** [`Cargo.toml`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/Cargo.toml#L30-L36). To restore it, restore the
features it lists first.

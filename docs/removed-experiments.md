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

## IRCNTXT v3 pre-packed keys

**What it was.** Version 3 of the `IRCNTXT` static-context format, in which
each ICL key tensor arrived already in the layout of the CPU score kernel. Only
`tools/export_static_context.py --pack-keys` wrote it.

**How it worked.** The logical shape of `icl.{0..11}.key` stayed `[B,8,N,64]`,
but within each estimator and head the exporter stored every 128-row source
tile dimension-major: dimension 0 of all its rows, then dimension 1, and so on.
One AVX-512 vector then held one dimension of 16 source rows
([artifact-format.md](artifact-format.md#layout-strategy)). On load,
`has_packed_icl_keys()` reported version 3, and the cached ICL stage passed
`key_is_packed` to the attention kernels. With it set, the kernels skipped
their per-tile transpose and the Accelerate path, scored straight from the file
with `attention_score_packed_key_tile_f32`, and used a scalar dimension-major
fallback. The Metal and CUDA loaders kept v3 contexts on the CPU.

**What it showed.** The Ryzen 7 9800X3D reference workload in
[experiments.md](experiments.md#current-result) used packed FP32 ICL keys with
BF16 values: `199.64 ms` p50 at `B=8,N=4096,Q=128` and `752.07 ms` at `Q=500`.
The const-specialized packed score kernel took that workload from `231.72 ms`
to `203.69 ms`. At `B=1,N=100000,Q=128`, the packed FP32-key/BF16-value
context was 3.71 GB and ran in `1,005 ms`
([experiments.md](experiments.md#large-context-result)). No recorded
measurement compares keys packed in the file with keys packed per tile at run
time.

**Why it was removed.** The Rust context builder writes v1 and v2 contexts
with row-major keys, and the kernels pack each key tile into scratch before
scoring, so the packed score kernels still run for every context Ironicl
builds. Only the removed Python exporter wrote v3, and the GPU backends never
read it. The reader now rejects version 3 with "static context version 3
(pre-packed ICL keys) is no longer supported; rebuild the context".

**Code.**

- Format: [the version constants](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/format/static_context.rs#L25-L30),
  [the version check](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/format/static_context.rs#L571-L579), and
  [`has_packed_icl_keys`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/format/static_context.rs#L1041-L1045)
- GPU fallbacks: [`with_cuda`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L1862-L1864)
  and [`with_gpu`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L1918-L1920)
- Dispatch: [the cached ICL attention match](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L2641-L2728),
  which passed `has_packed_icl_keys()` to [`parallel_cached_attention`, `parallel_cached_attention_bf16`, and `parallel_cached_attention_f32_key_bf16_value`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L4435-L4715)
- Kernels: the `key_is_packed` branches in
  [`source_major_online_attention_with_backend_f32`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L785-L1024)
  and [`source_major_online_attention_f32_key_bf16_value_with_backend_f32`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1234-L1487)
- Test: [the pre-packed case](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1948-L1982)
  of `tiled_online_attention_matches_materialized_across_tiles_and_scores`
- Writer: [`--pack-keys` in `write_static_context`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/export_static_context.py#L299-L308)
  in the removed exporter
- Restore: accept version 3 in the reader again, restore `has_packed_icl_keys`,
  thread `key_is_packed` from the dispatch to the kernels, and have the
  context builder write packed key tiles. Measure first against the per-tile
  packing the kernels do now.

## BF16 attention keys

**What it was.** Contexts that stored the ICL keys in BF16, and the CPU
attention kernels that scored from them. The exporter wrote them with
`--kv-dtype bf16` (BF16 keys and values) or `--kv-dtype bf16-k` (BF16 keys, FP32
values, which no kernel served).

**How it worked.** For BF16 keys and values, `parallel_cached_attention_bf16`
split heads across workers and ran
`source_major_online_attention_bf16_with_backend_f32`. It scored each tile
from BF16 keys with FP32 accumulation, widening each key by a 16-bit shift:
`attention_score_packed_key_bf16_tile_f32` on a per-tile packed copy or a v3
packed file, else `attention_score_tile_bf16_f32` on row-major keys. On x86_64
both ran AVX-512F/BW kernels, with scalar fallbacks elsewhere. Softmax stayed
FP32, and the values went through the retained
`attention_value_tile_bf16_fma_f32`. The Metal loader also uploaded BF16 keys
and values to the GPU graph.

**What it showed.** At `B=1,N=100000,Q=128` on the CPU, packed BF16 keys and
values took the context from 3.71 GB to 2.48 GB, but ran in
`1,018 ms` against `1,005 ms` for FP32 keys with BF16 values. Maximum logit
error rose from `4.43e-3` to `4.46e-2` and maximum probability error from
`7.13e-4` to `6.15e-3`, over the `1e-3` probability gate; labels agreed 100%
([experiments.md](experiments.md#large-context-result),
[pytorch-cpu-comparison.md](pytorch-cpu-comparison.md#dense-fusion-and-reduced-precision)).
On an M1 Max GPU, BF16 ICL keys and values were slower and less accurate than
FP16 ([experiments.md](experiments.md#metal-rejected-experiments)).

**Why it was removed.** BF16 keys failed the probability gate without saving
time, which is why `optimize="fast"` keeps FP32 keys with BF16 values. The
Rust builder never writes BF16 keys, so only the removed exporter produced
them. Loading such a context now fails with "static context cache tensor
'icl.N.key' stores BF16 keys, which are no longer supported; rebuild the
context". The FP32-key/BF16-value kernel and `attention_value_tile_bf16_fma_f32`
stay.

**Code.**

- Dispatch: [the BF16 key and value arm](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L2657-L2670)
  and [`parallel_cached_attention_bf16`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L4527-L4620)
- Kernel: [`source_major_online_attention_bf16_with_backend_f32`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1026-L1232)
  and its test, [`bf16_source_major_attention_matches_dequantized_fp32`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/reference.rs#L1987-L2059)
- Score wrappers: [`attention_score_tile_bf16_f32`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/attention.rs#L105-L130),
  [`attention_score_packed_key_bf16_tile_f32`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/attention.rs#L188-L213),
  and the portable stubs, [`attention_score_packed_key_bf16_tile_for_target`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/attention.rs#L339-L355)
  and [`attention_score_tile_bf16_for_target`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/ops/attention.rs#L394-L410)
- AVX-512 kernels: [`attention_score_tile_bf16_for_target`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L358-L386),
  [`attention_score_packed_key_bf16_tile_for_target`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L473-L512),
  [`attention_score_tile_bf16_avx512`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L698-L747), and
  [`attention_score_packed_key_bf16_tile_avx512`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/arch/x86_64.rs#L916-L973)
- Metal: [the BF16 arm of the ICL cache upload](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/models/tabicl_v2/cpu.rs#L1949-L1951)
- Writer: [`--kv-dtype` selection](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/tools/export_static_context.py#L260-L265)
  in the removed exporter
- Restore: restore the code above, drop the BF16-key check in
  `validate_static_cache_contract`, and add a BF16-key storage choice to the
  context builder. It would need a new error budget: at `N=100000` it was six
  times over the probability gate.

## Legacy CLI planner

**What it was.** The `ironicl plan` command and the configuration types behind
it: `ExecutionConfig`, `TabIclV2Model`, `Workload` (with
`query_microbatch`), `CacheMode` (`kv` or `repr`), `IsaPreference`,
`Isa::resolve`, and `estimate_memory`, which returned a `MemoryPlan` with a
`CacheEstimate`. `StorageDType` also had an `Int8` variant that only this
planner could select.

**How it worked.** `ironicl plan` took `--task`, `--rows`, `--features`,
`--estimators`, `--microbatch`, `--cache kv|repr`, `--weights` and
`--kv-dtype` (`fp32|bf16|int8`), `--isa auto|portable|neon|avx512`, and
`--bandwidth-gbps`. The defaults were 50,000 rows, 100 features, 8 estimators,
microbatch 1, full K/V, and BF16 weights and caches. `Isa::resolve` checked an
explicit ISA against the host: AVX-512 needed `AVX512F`, plus `AVX512_BF16`
for BF16 storage and `AVX512_VNNI` for INT8. `estimate_memory` applied the
payload formulas in [memory-and-io.md](memory-and-io.md): column K/V
`2 * s * B * 3 * (H + 4) * 128 * 128`, and either ICL K/V
`2 * s * B * 12 * N * 512` or, for `repr`, row representations
`s * B * N * 512`. It excluded quantization scales, alignment, and scratch
memory. The command then printed the `memory::plan` report for a CPU build,
and with a bandwidth it printed the ideal time to scan the cache once and that
time divided by the microbatch size.

**What it showed.** It reproduced the reference sizes in
[memory-and-io.md](memory-and-io.md#reference-sizes): at 50,000 rows, 100
features, 8 estimators, and BF16, a 9,993,977,856-byte cache, which at 80 GB/s
is an ideal 124.9 ms scan ([memory-and-io.md](memory-and-io.md#bandwidth-lower-bound)).
Its tests pinned that figure and the 24-fold smaller representation cache per
context row. [Architecture](architecture.md) described it as the report of ISA
and payload sizes; [memory-planner.md](memory-planner.md#phases) records that
phase A (#27) added the `memory::plan` report to the command.

**Why it was removed.** Nothing outside the command used it. Python plans with
`plan_context`, which calls `memory::plan`, through `_core.plan_memory`; that
backs `ironicl.plan_memory` and the fitted estimators' `memory_plan_`. The
context builder sizes its CPU and GPU build groups with the same `memory`
module. The kernels pick their ISA with
`Isa::detect`, so `--isa` changed only the report. The representation cache
and INT8 storage were never implemented, and the microbatch API went with the C
ABI in `9e96273`. The CLI now has `help`, `platform`, `inspect`, and
`import`. `StorageDType` tag `4` in the `IRCNTXT` tensor directory was
`int8`; no context ever stored it, and it stays reserved so the number is not
reused.

**Code.**

- Types: [`CacheMode`, `TabIclV2Model`, `Workload`, and `ExecutionConfig`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/config.rs#L131-L298)
  and [`StorageDType` with `Int8`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/config.rs#L48-L129)
- ISA choice: [`Isa::resolve`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/mod.rs#L86-L119),
  [`IsaPreference`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/mod.rs#L122-L156), and
  [their tests](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/kernels/cpu/mod.rs#L219-L259)
- Estimate: [`CacheEstimate`, `MemoryPlan`, `cache_scan_seconds`, and `estimate_memory`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/memory.rs#L905-L1017)
  and [their tests](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/memory.rs#L1451-L1498)
- CLI: [`PlanArgs`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/cli.rs#L47-L52),
  [`parse_plan`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/cli.rs#L154-L225),
  [`print_plan`](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/main.rs#L40-L103), and
  [its help](https://github.com/probabl-ai/ironicl/blob/e91a75d3995c52456b282a47138764fa064d81e1/src/main.rs#L496-L509)
- Restore: for a memory report, add a `plan` command that prints
  `memory::plan` for a shape, as `_core.plan_memory` does; that is the planner
  the builds use. Restore `estimate_memory` and the configuration types only
  to size a cache mode or storage type that the builder implements. Never give
  tag `4` to a new dtype.

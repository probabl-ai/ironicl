Unreleased
==========

New features
------------

- Add the `ironicl` Python package with `IronICLClassifier`, a scikit-learn
  compatible classifier that runs cached TabICLv2 inference on CPU through the
  Rust engine. Checkpoints are fetched from Hugging Face like tabicl does.
- Build static contexts natively: the Rust engine runs the training table
  through the column, row and ICL stages once and streams the key/value state
  to the context file (`ironicl build-context` on the CLI,
  `ironicl._ironicl.build_context` in Python). Contexts match the PyTorch-built
  ones to about `1e-4` per cached value and reproduce uncached logits to
  `3e-6`.
- Port tabicl's preprocessing and ensemble generation (standardisation,
  Yeo-Johnson, quantile and robust normalisation, outlier clipping, Latin
  square feature and class permutations) to numpy and scikit-learn in
  `ironicl._preprocessing`; it reproduces tabicl 2.2.0's ensemble variants
  exactly on the frozen fixtures.

Changes
-------

- Lay the repository out as a Python package with a Rust core (maturin's mixed
  layout): the `ironicl` package at the root, the crate's sources in `src/`,
  one `Cargo.toml` and one `pyproject.toml`; the `ironicl` CLI is unchanged.
- Remove the C ABI (`ffi.rs`) and the `ArtifactDescriptor` draft, both unused
  by the engine.
- Replace the `tools/` scripts with `benchmarks/bench_cached_serving.py` and
  `tests/generate_goldens.py`.
- Remove the `tabicl` extra: `IronICLClassifier` no longer imports tabicl or
  PyTorch for `fit` or `predict`; the checkpoint is fetched with
  `huggingface_hub`. The goldens under `tests/data/` are frozen fixtures
  produced once with tabicl 2.2.0, and the benchmark times IronICL's K/V
  storage precisions against each other instead of against tabicl.
- Drop the `batch_size` parameter of `IronICLClassifier`: the native context
  build processes estimators one at a time.
- Drop the `bf16-k` K/V storage option (BF16 keys with FP32 values), which the
  cached executor never had an attention kernel for and rejected at load time;
  `fp32`, `bf16` and `bf16-v` remain.

Documentation
-------------

- Collapse the design notes into `docs/design.md`, `docs/benchmarks.md` and
  `docs/development.md`, and fix statements that no longer matched the code
  (16-row attention query tile, approximate GELU by default, BF16 K/V and
  mmap loading implemented).

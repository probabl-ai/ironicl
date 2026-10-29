# Contributing to IronICL

IronICL is a scikit-learn compatible classifier whose compiled core is a Rust
crate. The Python package lives in `ironicl/` and the Rust crate in `src/`, maturin's mixed
layout: one `Cargo.toml` and one `pyproject.toml` at the root.

## Setting up

You need a Rust toolchain (`rustup`, stable channel) and Python 3.10 or newer.

```bash
git clone <your fork>
cd ironicl
python -m venv .venv && source .venv/bin/activate
make install          # builds the extension in editable mode, installs pre-commit hooks
```

Nothing else is needed: preprocessing and the static-context build are native,
so neither `tabicl` nor PyTorch is installed. Tests marked `slow` download the
TabICLv2 checkpoint from Hugging Face (about 110 MB) on first use.

## Checking your work

```bash
make lint             # pre-commit: ruff, cargo fmt, cargo clippy
make test             # cargo test, then pytest
```

Tests that need the official TabICLv2 checkpoint read it from the directory in
`IRONICL_CHECKPOINT_DIR` when set and download it otherwise. Set
`IRONICL_SKIP_SLOW=1` to skip them, or run `pytest -m "not slow"`. The Rust
fixture tests in `tests/checkpoint_fixtures.rs` need
`IRONICL_CHECKPOINT_DIR` to contain both checkpoints and the imported
`tabicl-classifier-v2.ironicl`.

## Pull requests

- Open or comment on an issue first so the change is agreed before you write it.
- Branch from `main`, keep the change focused, and link the PR to the issue.
- Title the PR as a Conventional Commit, `type(scope): Imperative summary`, for
  example `feat(classifier): Add bf16 K/V storage` or `docs: Fix tile size`.
  Types: `feat`, `fix`, `docs`, `build`, `ci`, `chore`, `refactor`, `test`, `perf`.
- Add a line to `CHANGES.md` under "Unreleased" for user-facing changes.
- Benchmark results belong in `docs/benchmarks.md` with the date, commit and
  host they were measured on, never in code comments or the README.

## Code conventions

- Python: ruff (configured in `pyproject.toml`, skore's rule set), numpy-style
  docstrings, scikit-learn's estimator conventions (`validate_data`,
  `check_is_fitted`, fitted attributes end in `_`).
- Rust: `cargo fmt`, `cargo clippy` with the pedantic group enabled in
  `Cargo.toml`. `unsafe` is confined to `src/kernels/native.rs`.
- Design contracts in `docs/design.md` change in the same PR as the code they
  describe. When code and docs disagree, tests and measured behaviour win.

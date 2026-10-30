# Contributing

Keep changes focused and include tests for behavior changes. Run the relevant
checks below before requesting review. Run all commands from the repository root.

## Setup

Install rustup, Make, and Python 3.12 or newer. On macOS, install the Xcode Command
Line Tools; on Linux, install a C compiler and linker.

Rust commands use the version and components in `rust-toolchain.toml`. Install them
with `rustup toolchain install`. This is the tested toolchain, not a declared minimum
Rust version. Update the pin deliberately and validate both CI platforms.

For Python development, create a virtual environment and install the package:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --editable .
```

This environment does not need TabICL or PyTorch. Use a separate environment for
live upstream comparisons.

## Local checks

| Command | Scope |
|---|---|
| `make test-rust` | Rust tests with default features and with `--no-default-features`, using `Cargo.lock`. |
| `make lint-rust` | Formatting and Clippy for both feature configurations; warnings are errors. |
| `make test-python` | Standalone preprocessing and exporter tests, including the frozen upstream oracle. |
| `make test-parity` | Direct comparisons with the pinned TabICL checkout; requires a separate environment. |

The Python targets use `python3` from `PATH`. Override the interpreter when needed:

```sh
make test-python PYTHON=/path/to/venv/bin/python
```

The Python targets do not install dependencies. None of the targets downloads model
checkpoints. Rust checks may download the pinned toolchain and Cargo dependencies
on first use. To fix Rust formatting, run `cargo fmt --all`; `make lint-rust` only
checks it.

### Live preprocessing parity

Follow [the parity setup instructions](docs/validation.md#python-preprocessing-parity)
to install the Ironicl wheel and pinned TabICL checkout in a separate environment.
Then run:

```sh
IRONICL_TABICL_SOURCE=/path/to/tabicl/src \
  make test-parity PYTHON=/path/to/parity/bin/python
```

The suite verifies the upstream revision and imported source path. Missing
prerequisites fail rather than skip. These tests compare preprocessing, not neural
outputs or classifier predictions.

## CI coverage and limits

The Rust workflow runs `make lint-rust` and `make test-rust` on Linux x86-64 and
Apple Silicon. The Python workflow tests an installed wheel built from the source
distribution, rather than the editable checkout used for local development. It
runs standalone and live parity tests in separate Linux/Python 3.12 environments.

Rust CI does not provision official model or oracle fixtures. Tests in
`tests/checkpoint_fixtures.rs` return early when those files are absent, even though
Cargo reports them as passed. To exercise them locally, provide:

- `IRONICL_CHECKPOINT_FIXTURES`: a directory containing both official v2 checkpoints
  and the imported classifier artifact `tabicl-classifier-v2.ironicl`.
- `IRONICL_ORACLE_FIXTURES`: a directory containing `input.irinput` and
  `logits.irlogits` from the upstream exporter.

See [the validation guide](docs/validation.md) for fixture generation and numerical
requirements. Hardware-gated kernel tests may also return early when CPU features
are unavailable, and benchmarks marked `#[ignore]` do not run by default. Green CI
is not proof of upstream model parity or Zen 5 performance.

PYTHON ?= python3

.PHONY: test-rust lint-rust audit lint-python test-python test-parity test-cuda lint-cuda develop-cuda

test-rust:
	cargo test --locked

lint-rust:
	cargo fmt --all -- --check
	cargo clippy --locked --all-targets -- -D warnings
	cargo clippy --locked --all-targets --features python,metal -- -D warnings

# RUSTSEC-2024-0436: paste is unmaintained; it comes through candle-core.
audit:
	cargo audit --deny warnings --ignore RUSTSEC-2024-0436

lint-python:
	ruff check python/ironicl tools

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-parity:
	$(PYTHON) -m unittest discover -s tests/parity -p 'test_*.py' -v

# CUDA (Linux): builds through cuda-oxide's codegen backend; see tools/cuda-env.sh.
# build.rs builds the kernels for each target in IRONICL_CUDA_ARCH.
# Cargo builds a library and its unit tests at once, and the backend names its
# files after the crate, so the library builds before the tests.
CUDA_ENV = . tools/cuda-env.sh

test-cuda:
	$(CUDA_ENV) && cargo build --release --locked --lib --features cuda \
		&& cargo test --release --locked --features cuda \
		&& IRONICL_CUDA_MODULE=sm_80 cargo test --release --locked --features cuda --test 'cuda_*'

lint-cuda:
	$(CUDA_ENV) && cargo clippy --release --locked --all-targets --features cuda,python -- -D warnings

develop-cuda:
	$(CUDA_ENV) && maturin develop --release --uv --features python,cuda

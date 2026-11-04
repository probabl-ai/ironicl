PYTHON ?= python3

.PHONY: test-rust lint-rust audit lint-python test-python test-parity

test-rust:
	cargo test --locked
	cargo test --locked --no-default-features

lint-rust:
	cargo fmt --all -- --check
	cargo clippy --locked --all-targets -- -D warnings
	cargo clippy --locked --all-targets --no-default-features -- -D warnings
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

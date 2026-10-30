PYTHON ?= python3

.PHONY: test-rust lint-rust test-python test-parity

test-rust:
	cargo test --locked
	cargo test --locked --no-default-features

lint-rust:
	cargo fmt --all -- --check
	cargo clippy --locked --all-targets -- -D warnings
	cargo clippy --locked --all-targets --no-default-features -- -D warnings

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-parity:
	$(PYTHON) -m unittest discover -s tests/parity -p 'test_*.py' -v

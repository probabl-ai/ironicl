.PHONY: install lint test test-rust test-python bench

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	pre-commit install

lint:
	pre-commit run --all-files

test-rust:
	cargo test

test-python:
	python -m pytest

test: test-rust test-python

bench:
	python benchmarks/bench_cached_serving.py --help

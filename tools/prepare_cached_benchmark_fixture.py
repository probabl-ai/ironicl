#!/usr/bin/env python3
"""Write one immutable raw-data fixture for cached serving benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_array(values: np.ndarray, dtype: str) -> bytes:
    values = np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    return np.asarray(values.ndim, dtype="<u8").tobytes() + np.asarray(values.shape, dtype="<u8").tobytes() + values.tobytes()


def checksum(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(canonical_array(values, dtype)).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("covertype",), default="covertype")
    parser.add_argument("--data-home", type=Path)
    parser.add_argument("--context-rows", type=int, required=True)
    parser.add_argument("--query-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.context_rows < 2 or args.query_rows < 1:
        parser.error("--context-rows must be at least 2 and --query-rows must be positive")
    return args


def main() -> None:
    args = parse_args()
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    dataset = fetch_covtype(data_home=str(args.data_home) if args.data_home else None)
    x = np.asarray(dataset.data)
    y = np.asarray(dataset.target)
    context_indices, query_indices = train_test_split(
        np.arange(y.size),
        train_size=args.context_rows,
        test_size=args.query_rows,
        random_state=args.seed,
        stratify=y,
    )
    x_context = np.ascontiguousarray(x[context_indices], dtype="<f4")
    y_context = np.ascontiguousarray(y[context_indices], dtype="<i8")
    x_query = np.ascontiguousarray(x[query_indices], dtype="<f4")
    metadata = {
        "benchmark_contract": "raw_x_cached_serving_v1",
        "dataset": "covertype",
        "dataset_shape": [int(value) for value in x.shape],
        "context_rows": int(args.context_rows),
        "query_rows": int(args.query_rows),
        "seed": int(args.seed),
        "context_indices_sha256": checksum(context_indices, "<i8"),
        "query_indices_sha256": checksum(query_indices, "<i8"),
        "x_context_sha256": checksum(x_context, "<f4"),
        "y_context_sha256": checksum(y_context, "<i8"),
        "x_query_sha256": checksum(x_query, "<f4"),
    }
    fixture_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    metadata["fixture_sha256"] = fixture_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, x_context=x_context, y_context=y_context, x_query=x_query)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()

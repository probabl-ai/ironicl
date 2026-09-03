#!/usr/bin/env python3
"""Validate and compare two raw-X cached-serving benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_MATCHES = (
    "benchmark_contract",
    "timed_boundary",
    "fixture_sha256",
    "fixture_x_query_sha256",
    "prepared_query_sha256_f32",
    "checkpoint_sha256",
    "context_sha256",
    "cache_storage_dtype",
    "compute_precision",
    "context_rows",
    "query_rows",
    "raw_feature_count",
    "prepared_feature_count",
    "estimators",
    "ensemble_batch_size",
    "stream_request_rows",
    "stream_preprocess_mode",
    "sparse_max_omitted_mass",
    "output_shape",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-probabilities", type=Path, required=True)
    parser.add_argument("--candidate-probabilities", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    for field in REQUIRED_MATCHES:
        if baseline.get(field) != candidate.get(field):
            raise SystemExit(f"benchmark mismatch for {field}: {baseline.get(field)!r} != {candidate.get(field)!r}")
    expected = np.load(args.baseline_probabilities, allow_pickle=False)
    actual = np.load(args.candidate_probabilities, allow_pickle=False)
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise SystemExit("probability output shapes or values are invalid")
    maximum = float(np.max(np.abs(actual - expected)))
    agreement = float(np.mean(np.argmax(actual, axis=-1) == np.argmax(expected, axis=-1)))
    if maximum > args.atol or agreement != 1.0:
        raise SystemExit(f"accuracy gate failed: max_abs={maximum:.9e} label_agreement={agreement:.9f}")
    baseline_p50 = float(baseline["latency_ms"]["p50"])
    candidate_p50 = float(candidate["latency_ms"]["p50"])
    print(
        json.dumps(
            {
                "baseline_engine": baseline["engine"],
                "candidate_engine": candidate["engine"],
                "max_abs": maximum,
                "label_agreement": agreement,
                "baseline_p50_ms": baseline_p50,
                "candidate_p50_ms": candidate_p50,
                "speedup_candidate_vs_baseline": baseline_p50 / candidate_p50,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

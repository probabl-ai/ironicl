"""Benchmark cached serving: raw numeric X in, FP32 probabilities out.

Times ``IronICLClassifier.predict_proba`` on raw X, which includes the
preprocessing and the engine call and excludes ``fit`` (reported separately as
``fit_seconds``). Several K/V storage precisions can be compared in one run.

    python benchmarks/bench_cached_serving.py --context-rows 1024 \
        --query-rows 128 --iterations 20 --kv-dtype fp32 bf16 --output results.json

Needs only ``pip install ironicl``. Record results in ``docs/benchmarks.md`` with
the date, commit and host. To compare with tabicl's own KV-cached CPU path, run
``TabICLClassifier(kv_cache="kv", device="cpu")`` on the same rows in a separate
environment; this repository does not depend on tabicl or PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT = "raw_x_cached_serving_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset", choices=("synthetic", "covertype"), default="synthetic"
    )
    parser.add_argument("--context-rows", type=int, default=1024)
    parser.add_argument("--query-rows", type=int, default=128)
    parser.add_argument("--features", type=int, default=54, help="synthetic only")
    parser.add_argument("--classes", type=int, default=7, help="synthetic only")
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--kv-dtype",
        nargs="+",
        default=["fp32"],
        choices=("fp32", "bf16", "bf16-v"),
        help="K/V storage precisions to time; the first one is the parity reference",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-parity",
        action="store_true",
        help="assert every K/V precision agrees with the first one",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    if args.dataset == "covertype":
        from sklearn.datasets import fetch_covtype

        data = fetch_covtype()
        X, y = np.asarray(data.data, np.float32), np.asarray(data.target)
        index = rng.permutation(len(y))
        train, query = (
            index[: args.context_rows],
            index[args.context_rows :][: args.query_rows],
        )
        return X[train], y[train], X[query]
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=args.context_rows + args.query_rows,
        n_features=args.features,
        n_informative=max(2, args.features // 2),
        n_classes=args.classes,
        n_clusters_per_class=1,
        random_state=args.seed,
    )
    X = X.astype(np.float32)
    return X[: args.context_rows], y[: args.context_rows], X[args.context_rows :]


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[max(0, math.ceil(fraction * len(samples)) - 1)]


def time_predict(
    estimator: Any, X_query: np.ndarray, warmup: int, iterations: int
) -> dict:
    for _ in range(warmup):
        estimator.predict_proba(X_query)
    samples = []
    probabilities = None
    for _ in range(iterations):
        start = time.perf_counter()
        probabilities = estimator.predict_proba(X_query)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_ms": {
            "min": min(samples),
            "p50": percentile(samples, 0.5),
            "p95": percentile(samples, 0.95),
            "mean": sum(samples) / len(samples),
        },
        "query_rows_per_second": len(X_query) * 1000.0 / (sum(samples) / len(samples)),
        "probabilities": np.asarray(probabilities, np.float32),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    os.environ.setdefault("IRONICL_THREADS", str(args.threads))
    import ironicl
    from ironicl import IronICLClassifier

    X_train, y_train, X_query = load_dataset(args)
    engines: dict[str, Any] = {
        f"ironicl-{kv_dtype}": IronICLClassifier(
            n_estimators=args.estimators,
            kv_dtype=kv_dtype,
            random_state=args.seed,
        )
        for kv_dtype in args.kv_dtype
    }
    results: dict[str, Any] = {
        "benchmark_contract": CONTRACT,
        "dataset": args.dataset,
        "context_rows": int(len(X_train)),
        "query_rows": int(len(X_query)),
        "features": int(X_train.shape[1]),
        "estimators": args.estimators,
        "threads": args.threads,
        "kv_dtypes": args.kv_dtype,
        "iterations": args.iterations,
        "host": platform.node(),
        "machine": platform.machine(),
        "ironicl_version": ironicl.__version__,
        "commit": git_commit(),
        "engines": {},
    }
    outputs: dict[str, np.ndarray] = {}
    for name, estimator in engines.items():
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        timing = time_predict(estimator, X_query, args.warmup, args.iterations)
        outputs[name] = timing.pop("probabilities")
        results["engines"][name] = {"fit_seconds": fit_seconds, **timing}
        p50 = timing["latency_ms"]["p50"]
        print(f"{name:8s} fit {fit_seconds:7.2f} s   predict p50 {p50:8.2f} ms")
    if len(outputs) > 1:
        reference_name = next(iter(outputs))
        reference = outputs[reference_name]
        results["parity"] = {}
        for name, output in outputs.items():
            if name == reference_name:
                continue
            difference = float(np.abs(reference - output).max())
            agreement = float((reference.argmax(1) == output.argmax(1)).mean())
            results["parity"][name] = {
                "reference": reference_name,
                "max_abs_difference": difference,
                "label_agreement": agreement,
            }
            print(
                f"parity   {name} vs {reference_name}: max |diff| {difference:.2e}"
                f"   label agreement {agreement:.4f}"
            )
            if args.check_parity and (difference > 1e-2 or agreement < 1.0):
                raise SystemExit(
                    f"{name} disagrees with {reference_name} beyond tolerance"
                )
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

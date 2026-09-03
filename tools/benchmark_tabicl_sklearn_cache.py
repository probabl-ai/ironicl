#!/usr/bin/env python3
"""Benchmark official fitted TabICL sklearn cached serving on a fixed real-data subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Official TabICL classifier checkpoint.")
    parser.add_argument("--tabicl-root", type=Path, default=root.parent / "tabicl" / "src")
    parser.add_argument("--dataset", choices=("covertype",), default="covertype")
    parser.add_argument("--data-home", type=Path, help="scikit-learn dataset cache directory.")
    parser.add_argument("--context-rows", type=int, default=1024)
    parser.add_argument("--query-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1, help="Official TabICL ensemble batch size.")
    parser.add_argument("--estimators", type=int, default=1)
    parser.add_argument("--kv-cache", choices=("kv", "repr"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--use-amp", action="store_true", help="Enable the official CUDA AMP path.")
    parser.add_argument(
        "--offload-mode",
        choices=("gpu", "cpu", "disk", "auto"),
        default="gpu",
        help="Official TabICL column-output storage policy (default: gpu).",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Write the JSON result to this path too.")
    parser.add_argument("--probabilities-output", type=Path, help="Write final predict_proba FP32 values as .npy.")
    args = parser.parse_args()
    if args.context_rows < 2 or args.query_rows < 1:
        parser.error("--context-rows must be at least 2 and --query-rows must be at least 1")
    if args.batch_size < 1 or args.estimators < 1 or args.threads < 1:
        parser.error("--batch-size, --estimators, and --threads must be positive")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.use_amp and args.device != "cuda":
        parser.error("--use-amp requires --device cuda")
    return args


class PeakRss:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.start = self.process.memory_info().rss
        self.peak = self.start
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            self.stop.wait(0.002)


def cuda_synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def phase(torch: Any, device: str, action: Any) -> tuple[Any, dict[str, int]]:
    """Time one synchronized phase and retain its absolute process and CUDA peaks."""
    cuda_synchronize(torch, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    monitor = PeakRss()
    thread = threading.Thread(target=monitor.run, daemon=True)
    thread.start()
    start = time.perf_counter_ns()
    try:
        result = action()
        cuda_synchronize(torch, device)
        elapsed_ns = time.perf_counter_ns() - start
    finally:
        monitor.stop.set()
        thread.join()
    metrics = {
        "elapsed_ns": elapsed_ns,
        "peak_rss_bytes": monitor.peak,
        "peak_rss_delta_bytes": max(0, monitor.peak - monitor.start),
    }
    if device == "cuda":
        metrics["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result, metrics


def percentile_ns(samples: list[int], fraction: float) -> int:
    return sorted(samples)[max(0, int(np.ceil(fraction * len(samples))) - 1)]


def array_checksum(values: np.ndarray, dtype: str | None = None) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def load_subset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    data = fetch_covtype(data_home=str(args.data_home) if args.data_home else None)
    x = np.asarray(data.data)
    y = np.asarray(data.target)
    indices = np.arange(y.shape[0])
    context_indices, query_indices = train_test_split(
        indices,
        train_size=args.context_rows,
        test_size=args.query_rows,
        random_state=args.seed,
        stratify=y,
    )
    return x[context_indices], y[context_indices], x[query_indices], {
        "dataset": args.dataset,
        "dataset_source": "sklearn.datasets.fetch_covtype",
        "dataset_shape": [int(x.shape[0]), int(x.shape[1])],
        "context_indices_sha256": array_checksum(context_indices, "<i8"),
        "query_indices_sha256": array_checksum(query_indices, "<i8"),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    tabicl_root = args.tabicl_root.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    if not (tabicl_root / "tabicl" / "__init__.py").is_file():
        raise SystemExit(f"--tabicl-root must contain tabicl/__init__.py: {tabicl_root}")
    if args.device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    sys.path.insert(0, str(tabicl_root))
    import torch
    from tabicl import TabICLClassifier

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot access a CUDA device")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    x_context, y_context, x_query, source = load_subset(args)
    classifier = TabICLClassifier(
        n_estimators=args.estimators,
        batch_size=args.batch_size,
        kv_cache=args.kv_cache,
        model_path=checkpoint,
        allow_auto_download=False,
        device=args.device,
        use_amp=args.use_amp,
        use_fa3=False,
        offload_mode=False if args.offload_mode == "gpu" else args.offload_mode,
        random_state=args.seed,
        n_jobs=None,
        verbose=False,
    )
    _, fit_metrics = phase(torch, args.device, lambda: classifier.fit(x_context, y_context))

    # Warmups exercise the same public sklearn serving call, outside timed latency.
    for _ in range(args.warmup):
        classifier.predict_proba(x_query)
    cuda_synchronize(torch, args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    monitor = PeakRss()
    thread = threading.Thread(target=monitor.run, daemon=True)
    thread.start()
    samples: list[int] = []
    output = None
    try:
        for _ in range(args.iterations):
            cuda_synchronize(torch, args.device)
            start = time.perf_counter_ns()
            output = classifier.predict_proba(x_query)
            cuda_synchronize(torch, args.device)
            samples.append(time.perf_counter_ns() - start)
    finally:
        monitor.stop.set()
        thread.join()
    assert output is not None
    predict_metrics: dict[str, Any] = {
        "peak_rss_bytes": monitor.peak,
        "peak_rss_delta_bytes": max(0, monitor.peak - monitor.start),
    }
    if args.device == "cuda":
        predict_metrics["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
        predict_metrics["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result = {
        "benchmark": "official_tabicl_sklearn_fitted_cache",
        "cache_mode": args.kv_cache,
        "device": args.device,
        "precision": "amp" if args.use_amp else "fp32",
        "offload_mode": args.offload_mode,
        "context_rows": args.context_rows,
        "query_batch_size": args.query_rows,
        "feature_count": int(classifier.ensemble_generator_.n_features_in_),
        "ensemble_estimators": args.estimators,
        "ensemble_batch_size": args.batch_size,
        "threads": args.threads,
        "interop_threads": 1,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,  # type: ignore[attr-defined]
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "fit_cache_build_ms": fit_metrics["elapsed_ns"] / 1_000_000,
        "fit_cache_build_memory": {key: value for key, value in fit_metrics.items() if key != "elapsed_ns"},
        "predict_latency_ms": {
            "min": min(samples) / 1_000_000,
            "p50": percentile_ns(samples, 0.50) / 1_000_000,
            "p95": percentile_ns(samples, 0.95) / 1_000_000,
            "mean": sum(samples) / len(samples) / 1_000_000,
        },
        "predict_samples_ns": samples,
        "predict_memory": predict_metrics,
        "output_shape": [int(value) for value in output.shape],
        "output_checksum_sha256_f32": array_checksum(output, "<f4"),
        "output_checksum_sum_f64": float(np.asarray(output, dtype=np.float64).sum()),
        **source,
    }
    if args.device == "cuda":
        result["cuda_device"] = torch.cuda.get_device_name()
        result["cuda_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.probabilities_output:
        args.probabilities_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.probabilities_output, np.ascontiguousarray(output, dtype="<f4"), allow_pickle=False)


if __name__ == "__main__":
    main()

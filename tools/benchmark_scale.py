#!/usr/bin/env python3
"""Measure latency and peak process/GPU memory for one preprocessed request."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("ironicl", "tabicl-cpu", "tabicl-cuda-fp32", "tabicl-cuda-amp"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ironicl", type=Path)
    parser.add_argument("--tabicl-root", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_dimensions(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as source:
        header = source.read(80)
    fields = struct.unpack("<8sIIQQQQQQQQ", header)
    if fields[0] != b"IRINPUT\0":
        raise ValueError("input magic does not match")
    return tuple(int(value) for value in fields[3:7])  # type: ignore[return-value]


class PeakRss:
    def __init__(self, process: psutil.Process):
        self.process = process
        self.peak = 0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                rss = self.process.memory_info().rss
                for child in self.process.children(recursive=True):
                    rss += child.memory_info().rss
                self.peak = max(self.peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self.stop.wait(0.002)


def percentile(samples: list[int], p: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)] / 1_000_000


def benchmark_ironicl(args: argparse.Namespace) -> dict[str, Any]:
    if args.ironicl is None or args.model is None:
        raise ValueError("Ironicl requires --ironicl and --model")
    command = [
        str(args.ironicl), "benchmark", "--model", str(args.model), "--input", str(args.input),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
    ]
    environment = os.environ.copy()
    environment["IRONICL_THREADS"] = str(args.threads)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    monitor = PeakRss(psutil.Process(process.pid))
    thread = threading.Thread(target=monitor.run, daemon=True)
    thread.start()
    stdout, stderr = process.communicate()
    monitor.stop.set()
    thread.join()
    metrics = {
        key.strip(): value.strip()
        for line in stdout.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    result: dict[str, Any] = {
        "status": "ok" if process.returncode == 0 else "error",
        "returncode": process.returncode,
        "peak_rss_bytes": monitor.peak,
        "stdout": stdout,
        "stderr": stderr,
    }
    if process.returncode == 0:
        result.update({
            "mean_ms": float(metrics["mean_ms"]),
            "p50_ms": float(metrics["p50_ms"]),
            "p95_ms": float(metrics["p95_ms"]),
            "checksum": float(metrics["logits_checksum"]),
        })
    return result


def benchmark_tabicl(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.checkpoint is None or args.tabicl_root is None:
        raise ValueError("TabICL requires --checkpoint and --tabicl-root")
    import sys
    sys.path.insert(0, str(args.tabicl_root))
    from tabicl._model.inference_config import InferenceConfig
    from tabicl._model.tabicl import TabICL
    from benchmark_tabicl import load_input

    device = "cpu" if args.engine == "tabicl-cpu" else "cuda"
    amp = args.engine == "tabicl-cuda-amp"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    _, x, y_train = load_input(args.input)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TabICL(**checkpoint["config"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = x.to(device)
    y_train = y_train.to(device)
    config = {"device": device, "use_amp": amp, "use_fa3": False, "offload": False, "verbose": False}
    inference_config = InferenceConfig(COL_CONFIG=config, ROW_CONFIG=config, ICL_CONFIG=config)

    def infer() -> Any:
        return model(x, y_train, embed_with_test=False, return_logits=True, inference_config=inference_config)

    process = psutil.Process()
    monitor = PeakRss(process)
    thread = threading.Thread(target=monitor.run, daemon=True)
    thread.start()
    output = None
    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                output = infer()
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(args.iterations):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter_ns()
                output = infer()
                if device == "cuda":
                    torch.cuda.synchronize()
                samples.append(time.perf_counter_ns() - start)
    finally:
        monitor.stop.set()
        thread.join()
    assert output is not None
    mean_ms = sum(samples) / len(samples) / 1_000_000
    result = {
        "status": "ok",
        "mean_ms": mean_ms,
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "peak_rss_bytes": monitor.peak,
        "checksum": float(output.float().sum().item()),
        "output_shape": list(output.shape),
    }
    if device == "cuda":
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result


def main() -> None:
    args = parse_args()
    dimensions = read_dimensions(args.input)
    started = time.time()
    result: dict[str, Any]
    try:
        result = benchmark_ironicl(args) if args.engine == "ironicl" else benchmark_tabicl(args)
    except Exception as error:
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    result.update({
        "engine": args.engine,
        "dimensions": dimensions,
        "threads": args.threads,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "wall_seconds": time.time() - started,
    })
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()

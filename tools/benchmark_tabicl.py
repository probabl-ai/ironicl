#!/usr/bin/env python3
"""Compare official TabICL CPU inference with a prebuilt Ironicl benchmark."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict


CLASSIFIER_CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
INPUT_MAGIC = b"IRINPUT\0"
HEADER_BYTES = 80
CHECKSUM_BYTES = 32
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_ENSEMBLE_COUNT = 4096
MAX_TOTAL_ROWS = 1_000_000
MAX_FEATURE_COUNT = 65_536
MAX_TRAINING_ROWS = 1_000_000


class BenchmarkSummary(TypedDict):
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    inferences_per_s: float
    query_rows_per_s: float
    checksum: float
    shape: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Official TabICLv2 classifier checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Imported Ironicl model artifact.")
    parser.add_argument("--input", type=Path, required=True, help="Checked IRINPUT v1 tensor bundle.")
    parser.add_argument(
        "--tabicl-root",
        type=Path,
        default=root.parent / "tabicl" / "src",
        help="Sibling tabicl/src directory.",
    )
    parser.add_argument(
        "--ironicl",
        type=Path,
        default=root / "target" / "release" / "ironicl",
        help="Prebuilt Ironicl binary.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Untimed runs for each implementation.")
    parser.add_argument("--iterations", type=int, default=10, help="Timed runs for each implementation.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch intra-op and inter-op CPU threads.")
    parser.add_argument("--ironicl-threads", type=int, default=1, help="Ironicl executor threads.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="TabICL execution device.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be zero or greater")
    if args.iterations < 1:
        parser.error("--iterations must be at least one")
    if args.threads < 1:
        parser.error("--threads must be at least one")
    if args.ironicl_threads < 1:
        parser.error("--ironicl-threads must be at least one")
    if args.amp and args.device != "cuda":
        parser.error("--amp requires --device cuda")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def checked_product(values: tuple[int, ...], name: str) -> int:
    result = 1
    for value in values:
        result *= value
        if result > sys.maxsize:
            raise ValueError(f"{name} exceeds platform size")
    return result


def load_input(path: Path) -> tuple[tuple[int, int, int, int], Any, Any]:
    """Strictly decode the IRINPUT v1 format used by Ironicl."""
    import torch

    data = bytearray(path.read_bytes())
    if len(data) < HEADER_BYTES + CHECKSUM_BYTES:
        raise ValueError("IRINPUT bundle is truncated")
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("IRINPUT bundle exceeds the resource limit")
    payload, checksum = data[:-CHECKSUM_BYTES], data[-CHECKSUM_BYTES:]
    if hashlib.sha256(payload).digest() != checksum:
        raise ValueError("IRINPUT bundle checksum does not match")

    fields = struct.unpack("<8sIIQQQQQQQQ", data[:HEADER_BYTES])
    magic, version, kind, b, t, h, n, x_count, y_count, x_bytes, y_bytes = fields
    if magic != INPUT_MAGIC:
        raise ValueError("IRINPUT bundle magic does not match")
    if version != 1:
        raise ValueError(f"IRINPUT version {version} is unsupported; expected 1")
    if kind != 1:
        raise ValueError("IRINPUT bundle kind is not a classifier input")
    if not (0 < b <= MAX_ENSEMBLE_COUNT and 0 < t <= MAX_TOTAL_ROWS and 0 < h <= MAX_FEATURE_COUNT):
        raise ValueError(f"invalid IRINPUT dimensions B={b}, T={t}, H={h}")
    if not (0 < n <= MAX_TRAINING_ROWS and t > n):
        raise ValueError(f"invalid IRINPUT training rows N={n}; require 0 < N < T")
    expected_x = checked_product((b, t, h), "IRINPUT X element count")
    expected_y = checked_product((b, n), "IRINPUT y_train element count")
    if (x_count, y_count) != (expected_x, expected_y):
        raise ValueError("IRINPUT tensor element counts do not match dimensions")
    if (x_bytes, y_bytes) != (x_count * 4, y_count * 4):
        raise ValueError("IRINPUT tensor byte lengths do not match element counts")
    if len(data) != HEADER_BYTES + x_bytes + y_bytes + CHECKSUM_BYTES:
        raise ValueError("IRINPUT sections do not match whole-file length")

    x_end = HEADER_BYTES + x_bytes
    # Clone detaches tensors from the immutable byte buffer before model execution.
    x = torch.frombuffer(memoryview(data)[HEADER_BYTES:x_end], dtype=torch.float32).clone().reshape(b, t, h)
    y_train = torch.frombuffer(memoryview(data)[x_end:-CHECKSUM_BYTES], dtype=torch.float32).clone().reshape(b, n)
    if not torch.isfinite(x).all().item():
        raise ValueError("IRINPUT X contains a non-finite FP32 value")
    if not torch.isfinite(y_train).all().item() or not ((y_train >= 0) & (y_train <= 9)).all().item():
        raise ValueError("IRINPUT labels must be finite integers in 0..=9")
    if not torch.equal(y_train, torch.floor(y_train)):
        raise ValueError("IRINPUT labels must be finite integers in 0..=9")
    return (b, t, h, n), x, y_train


def percentile_nearest_rank(samples_ns: list[int], percentile: float) -> int:
    ordered = sorted(samples_ns)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize(samples_ns: list[int], query_rows: int, output: Any) -> BenchmarkSummary:
    mean_ns = sum(samples_ns) / len(samples_ns)
    return {
        "min_ms": min(samples_ns) / 1_000_000,
        "p50_ms": percentile_nearest_rank(samples_ns, 0.50) / 1_000_000,
        "p95_ms": percentile_nearest_rank(samples_ns, 0.95) / 1_000_000,
        "p99_ms": percentile_nearest_rank(samples_ns, 0.99) / 1_000_000,
        "max_ms": max(samples_ns) / 1_000_000,
        "mean_ms": mean_ns / 1_000_000,
        "inferences_per_s": 1_000_000_000 / mean_ns,
        "query_rows_per_s": query_rows * 1_000_000_000 / mean_ns,
        # This numeric checksum permits a tolerance for valid FP32 operation-order differences.
        "checksum": float(output.detach().to(dtype=output.dtype, device="cpu").sum().item()),
        "shape": tuple(int(value) for value in output.shape),
    }


def benchmark_tabicl(
    checkpoint: Path,
    tabicl_root: Path,
    x: Any,
    y_train: Any,
    warmup: int,
    iterations: int,
    threads: int,
    device: str,
    amp: bool,
) -> BenchmarkSummary:
    import torch

    if not (tabicl_root / "tabicl" / "__init__.py").is_file():
        raise ValueError(f"--tabicl-root must contain tabicl/__init__.py: {tabicl_root}")
    sys.path.insert(0, str(tabicl_root))
    from tabicl._model.inference_config import InferenceConfig
    from tabicl._model.tabicl import TabICL

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    torch.set_default_dtype(torch.float32)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access a CUDA device")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TabICL(**checkpoint_data["config"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint_data["state_dict"])
    model.eval()
    x = x.to(device=device)
    y_train = y_train.to(device=device)
    manager_config = {"device": device, "use_amp": amp, "use_fa3": False, "offload": False, "verbose": False}
    inference_config = InferenceConfig(
        COL_CONFIG=manager_config,
        ROW_CONFIG=manager_config,
        ICL_CONFIG=manager_config,
    )

    def infer() -> Any:
        return model(x, y_train, embed_with_test=False, return_logits=True, inference_config=inference_config)

    with torch.no_grad():
        for _ in range(warmup):
            infer()
        if device == "cuda":
            torch.cuda.synchronize()
        samples_ns = []
        output = None
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            output = infer()
            if device == "cuda":
                torch.cuda.synchronize()
            samples_ns.append(time.perf_counter_ns() - start)
    assert output is not None
    return summarize(samples_ns, x.shape[0] * (x.shape[1] - y_train.shape[1]), output)


def normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def metric_value(metrics: dict[str, str], names: set[str]) -> float:
    for key, value in metrics.items():
        if key in names:
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
            if match:
                return float(match.group())
    raise ValueError(f"Ironicl benchmark output is missing metric: {sorted(names)[0]}")


def parse_ironicl_output(output: str) -> BenchmarkSummary:
    metrics = {
        normalized_key(key): value.strip()
        for line in output.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    result: BenchmarkSummary = {
        "min_ms": metric_value(metrics, {"min", "minms", "minimumms", "minlatency", "minlatencyms"}),
        "p50_ms": metric_value(metrics, {"p50", "p50ms", "p50latency", "p50latencyms"}),
        "p95_ms": metric_value(metrics, {"p95", "p95ms", "p95latency", "p95latencyms"}),
        "p99_ms": metric_value(metrics, {"p99", "p99ms", "p99latency", "p99latencyms"}),
        "max_ms": metric_value(metrics, {"max", "maxms", "maximumms", "maxlatency", "maxlatencyms"}),
        "mean_ms": metric_value(metrics, {"mean", "meanms", "meanlatency", "meanlatencyms"}),
        "inferences_per_s": metric_value(metrics, {"inferencesps", "inferencespersecond", "inferencesperseconds", "inferencespersec"}),
        "query_rows_per_s": metric_value(metrics, {"queryrowsps", "queryrowspersecond", "queryrowsperseconds", "queryrowspersec"}),
        "checksum": metric_value(metrics, {"checksum", "outputchecksum", "logitschecksum"}),
        "shape": (),
    }
    shape_text = next((value for key, value in metrics.items() if key in {"shape", "outputshape", "logitsshape"}), None)
    if shape_text is None:
        raise ValueError("Ironicl benchmark output is missing output shape")
    shape = tuple(int(value) for value in re.findall(r"\d+", shape_text))
    if len(shape) != 3:
        raise ValueError(f"Ironicl benchmark output has invalid output shape: {shape_text}")
    result["shape"] = shape
    return result


def benchmark_ironicl(
    binary: Path,
    model: Path,
    input_path: Path,
    warmup: int,
    iterations: int,
    threads: int,
) -> BenchmarkSummary:
    command = [str(binary), "benchmark", "--model", str(model), "--input", str(input_path), "--warmup", str(warmup), "--iterations", str(iterations)]
    environment = os.environ.copy()
    environment["IRONICL_THREADS"] = str(threads)
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    if completed.returncode:
        raise RuntimeError(f"Ironicl benchmark failed ({completed.returncode}):\n{completed.stderr}{completed.stdout}")
    try:
        return parse_ironicl_output(completed.stdout)
    except ValueError as error:
        raise RuntimeError(f"{error}\nIronicl benchmark output:\n{completed.stdout}") from error


def print_summary(name: str, metrics: BenchmarkSummary) -> None:
    print(
        f"{name}: min={metrics['min_ms']:.3f} p50={metrics['p50_ms']:.3f} "
        f"p95={metrics['p95_ms']:.3f} p99={metrics['p99_ms']:.3f} "
        f"max={metrics['max_ms']:.3f} mean={metrics['mean_ms']:.3f} ms "
        f"inferences/s={metrics['inferences_per_s']:.3f} query_rows/s={metrics['query_rows_per_s']:.3f} "
        f"shape={metrics['shape']} checksum={metrics['checksum']:.9g}"
    )


def main() -> None:
    args = parse_args()
    checkpoint = require_file(args.checkpoint, "checkpoint")
    model = require_file(args.model, "model")
    input_path = require_file(args.input, "input")
    binary = require_file(args.ironicl, "Ironicl binary")
    tabicl_root = args.tabicl_root.resolve()
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != CLASSIFIER_CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 does not match the official TabICLv2 classifier: {checkpoint_hash}")

    dimensions, x, y_train = load_input(input_path)
    tabicl = benchmark_tabicl(
        checkpoint,
        tabicl_root,
        x,
        y_train,
        args.warmup,
        args.iterations,
        args.threads,
        args.device,
        args.amp,
    )
    ironicl = benchmark_ironicl(binary, model, input_path, args.warmup, args.iterations, args.ironicl_threads)
    labels = sorted({int(value) for value in y_train.reshape(-1).tolist()})
    if labels != list(range(len(labels))) or not 2 <= len(labels) <= 10:
        raise SystemExit(f"input labels must be contiguous 0..C-1 with C in 2..=10: {labels}")
    expected_shape = (dimensions[0], dimensions[1] - dimensions[3], len(labels))
    if tabicl["shape"] != ironicl["shape"] or tabicl["shape"] != expected_shape:
        raise SystemExit(f"output shape mismatch: TabICL={tabicl['shape']} Ironicl={ironicl['shape']} expected={expected_shape}")
    count = math.prod(expected_shape)
    checksum_tolerance = 1e-3 if args.amp else 1e-4
    if not math.isclose(
        float(tabicl["checksum"]),
        float(ironicl["checksum"]),
        rel_tol=checksum_tolerance,
        abs_tol=checksum_tolerance * count,
    ):
        raise SystemExit(f"output checksum mismatch: TabICL={tabicl['checksum']:.9g} Ironicl={ironicl['checksum']:.9g}")

    print(f"checkpoint_sha256={checkpoint_hash}")
    print(f"input_shape={dimensions} input_sha256={sha256_file(input_path)}")
    print(f"tabicl_device={args.device} amp={args.amp} threads={args.threads}")
    print(f"ironicl_threads={args.ironicl_threads}")
    print(f"checksum_tolerance={checksum_tolerance}")
    print_summary("tabicl", tabicl)
    print_summary("ironicl", ironicl)
    print(f"speedup_ironicl_vs_tabicl = {float(tabicl['mean_ms']) / float(ironicl['mean_ms']):.3f}x")


if __name__ == "__main__":
    main()

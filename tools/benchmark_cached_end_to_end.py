#!/usr/bin/env python3
"""Benchmark one cache-resident engine from raw numeric X to FP32 probabilities.

The timed boundary includes shared raw-query validation, TabICL numerical
encoding, ensemble variant generation, cached model execution, class correction,
logit averaging, temperature softmax, and host-resident FP32 probabilities.
It excludes model/context loading and training-cache construction.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT = "raw_x_cached_serving_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_f32(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def percentile(samples: list[int], fraction: float) -> float:
    return sorted(samples)[max(0, math.ceil(fraction * len(samples)) - 1)] / 1_000_000


def latency_summary(samples: list[int]) -> dict[str, float]:
    mean_ns = sum(samples) / len(samples)
    return {
        "min": min(samples) / 1_000_000,
        "p50": percentile(samples, 0.5),
        "p95": percentile(samples, 0.95),
        "mean": mean_ns / 1_000_000,
    }


def sparse_max_omitted_mass() -> float:
    try:
        value = float(os.environ.get("IRONICL_SPARSE_MAX_OMITTED_MASS", "1e-5"))
    except ValueError:
        return 1e-5
    return value if math.isfinite(value) and 0.0 < value <= 1.0 else 1e-5


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("pytorch", "ironicl"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--context-metadata", type=Path, required=True)
    parser.add_argument("--context-kv-dtype", choices=("fp32", "bf16-v"), default="fp32")
    parser.add_argument("--tabicl-root", type=Path, default=root.parent / "tabicl" / "src")
    parser.add_argument("--library", type=Path, help="Ironicl cdylib; required for --engine ironicl.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--request-rows", type=int, help="Split the raw query into equal Ironicl serving requests.")
    parser.add_argument(
        "--stream-preprocess-mode",
        choices=("batched", "per-request"),
        default="batched",
        help="Prepare collected stream requests together (default) or separately for diagnostics.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--fast-ironicl", action="store_true", help="Allow native approximate GELU and softmax.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probabilities-output", type=Path, required=True)
    args = parser.parse_args()
    if args.engine == "ironicl" and args.library is None:
        parser.error("--engine ironicl requires --library")
    if args.use_amp and args.device != "cuda":
        parser.error("--use-amp requires --device cuda")
    if args.engine == "ironicl" and (args.device != "cpu" or args.use_amp):
        parser.error("Ironicl cached execution is CPU-only")
    if args.request_rows is not None and args.engine != "ironicl":
        parser.error("--request-rows currently supports only --engine ironicl")
    if min(args.estimators, args.batch_size, args.threads, args.iterations) < 1 or args.warmup < 0:
        parser.error("estimator, batch, thread, and iteration counts must be positive; warmup must be non-negative")
    return args


def read_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("benchmark_contract") != CONTRACT:
        raise ValueError("fixture does not use the raw-X cached-serving contract")
    with np.load(path, allow_pickle=False) as fixture:
        x_context = np.ascontiguousarray(fixture["x_context"], dtype=np.float32)
        y_context = np.ascontiguousarray(fixture["y_context"], dtype=np.int64)
        x_query = np.ascontiguousarray(fixture["x_query"], dtype=np.float32)
    if x_context.ndim != 2 or y_context.shape != (x_context.shape[0],) or x_query.ndim != 2 or x_query.shape[1] != x_context.shape[1]:
        raise ValueError("fixture arrays have invalid shapes")
    if not np.isfinite(x_context).all() or not np.isfinite(x_query).all():
        raise ValueError("fixture contains non-finite values")
    return x_context, y_context, x_query, metadata


def validate_context(
    context: Path, metadata_path: Path, fixture: dict[str, Any], expected_kv_dtype: str
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("context_file_sha256") != sha256_file(context):
        raise ValueError("context metadata checksum does not match --context")
    if metadata.get("kv_dtype") != expected_kv_dtype:
        raise ValueError(f"context K/V dtype does not match --context-kv-dtype={expected_kv_dtype}")
    source = metadata.get("source", {})
    if source.get("fixture_sha256") != fixture["fixture_sha256"]:
        raise ValueError("context does not originate from the benchmark fixture")
    if metadata.get("N") != fixture["context_rows"] or metadata.get("B") != 8:
        raise ValueError("context dimensions do not match the B=8 fixture contract")
    return metadata


def prepare_query_variants(classifier: Any, raw_x: np.ndarray) -> np.ndarray:
    """Shared raw-X preprocessing used before both engine calls."""
    raw_x = np.asarray(raw_x, dtype=np.float32, order="C")
    if raw_x.ndim != 2 or raw_x.shape[1] != classifier.n_features_in_:
        raise ValueError("raw query shape does not match the fitted classifier")
    if not np.isfinite(raw_x).all():
        raise ValueError("raw query contains non-finite values")
    encoded = classifier.X_encoder_.transform(raw_x)
    groups = classifier.ensemble_generator_.transform(encoded, mode="test")
    variants = np.ascontiguousarray(np.concatenate([group[0] for group in groups.values()]), dtype=np.float32)
    if variants.shape[0] != classifier.n_estimators or variants.shape[1] != raw_x.shape[0]:
        raise ValueError("TabICL preprocessing produced unexpected ensemble variants")
    return variants


def pytorch_probabilities(classifier: Any, variants: np.ndarray) -> np.ndarray:
    outputs = []
    offset = 0
    for cache in classifier.model_kv_cache_.values():
        count = next(iter(cache.icl_cache.kv.values())).key.shape[0]
        outputs.append(classifier._batch_forward_with_cache(variants[offset:offset + count], cache))
        offset += count
    if offset != variants.shape[0]:
        raise ValueError("cached PyTorch batches do not cover prepared variants")
    logits = np.concatenate(outputs, axis=0)
    permutations = [shuffle for shuffles in classifier.ensemble_generator_.class_shuffles_.values() for shuffle in shuffles]
    if len(permutations) != logits.shape[0]:
        raise ValueError("class permutations do not match cached estimator output")
    average = np.zeros_like(logits[0])
    for output, permutation in zip(logits, permutations):
        average += output[..., permutation]
    average /= len(permutations)
    shifted = average / classifier.softmax_temperature
    shifted -= shifted.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.ascontiguousarray(probabilities, dtype=np.float32)


class IroniclLibrary:
    def __init__(self, path: Path, model: Path, context: Path, class_count: int) -> None:
        self.library = ctypes.CDLL(str(path))
        self.library.ironicl_static_executor_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.library.ironicl_static_executor_create.restype = ctypes.c_void_p
        self.library.ironicl_static_executor_predict_probabilities.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.library.ironicl_static_executor_predict_probabilities.restype = ctypes.c_int
        self.library.ironicl_static_executor_predict_probabilities_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.library.ironicl_static_executor_predict_probabilities_batch.restype = ctypes.c_int
        self.library.ironicl_static_executor_reset_sparse_attention_statistics.argtypes = [ctypes.c_void_p]
        self.library.ironicl_static_executor_reset_sparse_attention_statistics.restype = ctypes.c_int
        self.library.ironicl_static_executor_sparse_attention_statistics.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        self.library.ironicl_static_executor_sparse_attention_statistics.restype = ctypes.c_int
        self.library.ironicl_static_executor_destroy.argtypes = [ctypes.c_void_p]
        self.library.ironicl_static_executor_destroy.restype = None
        error = ctypes.create_string_buffer(4096)
        self.handle = self.library.ironicl_static_executor_create(
            os.fsencode(model), os.fsencode(context), error, len(error)
        )
        if not self.handle:
            raise RuntimeError(f"Ironicl executor creation failed: {error.value.decode(errors='replace')}")
        self.class_count = class_count

    def predict(self, variants: np.ndarray) -> np.ndarray:
        variants = np.ascontiguousarray(variants, dtype=np.float32)
        output = np.empty((variants.shape[1], self.class_count), dtype=np.float32)
        error = ctypes.create_string_buffer(4096)
        status = self.library.ironicl_static_executor_predict_probabilities(
            self.handle,
            variants.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            variants.shape[0], variants.shape[1], variants.shape[2],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
            error, len(error),
        )
        if status:
            raise RuntimeError(f"Ironicl cached prediction failed ({status}): {error.value.decode(errors='replace')}")
        return output

    def predict_batch(self, requests: list[np.ndarray]) -> np.ndarray:
        if not requests:
            raise ValueError("Ironicl microbatch must contain at least one request")
        request_rows = requests[0].shape[1]
        if any(values.shape != requests[0].shape for values in requests):
            raise ValueError("Ironicl microbatch requests must have equal [B,Q,H] shapes")
        values = np.ascontiguousarray(np.stack(requests), dtype=np.float32)
        output = np.empty((len(requests), request_rows, self.class_count), dtype=np.float32)
        error = ctypes.create_string_buffer(4096)
        status = self.library.ironicl_static_executor_predict_probabilities_batch(
            self.handle,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(requests), values.shape[1], values.shape[2],
            values.shape[3], output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size,
            error, len(error),
        )
        if status:
            raise RuntimeError(f"Ironicl cached microbatch prediction failed ({status}): {error.value.decode(errors='replace')}")
        return output.reshape(request_rows * len(requests), self.class_count)

    def reset_sparse_attention_statistics(self) -> None:
        if self.library.ironicl_static_executor_reset_sparse_attention_statistics(self.handle):
            raise RuntimeError("Ironicl sparse attention statistics reset failed")

    def sparse_attention_statistics(self) -> dict[str, int]:
        values = (ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong())
        if self.library.ironicl_static_executor_sparse_attention_statistics(
            self.handle, *(ctypes.byref(value) for value in values)
        ):
            raise RuntimeError("Ironicl sparse attention statistics read failed")
        return {"attempted": values[0].value, "accepted": values[1].value, "fallback": values[2].value}

    def close(self) -> None:
        if self.handle:
            self.library.ironicl_static_executor_destroy(self.handle)
            self.handle = None


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required for deterministic TabICL ensemble ordering")
    if args.engine == "ironicl":
        os.environ["IRONICL_THREADS"] = str(args.threads)
        if not args.fast_ironicl:
            os.environ["IRONICL_EXACT_GELU"] = "1"
            os.environ["IRONICL_EXACT_SOFTMAX"] = "1"
    if args.device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import sys
    sys.path.insert(0, str(args.tabicl_root.resolve()))
    import torch
    from tabicl import TabICLClassifier

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    x_context, y_context, raw_query, fixture = read_fixture(args.fixture)
    context_metadata = validate_context(args.context, args.context_metadata, fixture, args.context_kv_dtype)
    classifier = TabICLClassifier(
        n_estimators=args.estimators,
        batch_size=args.batch_size,
        kv_cache="kv",
        model_path=args.checkpoint,
        allow_auto_download=False,
        device=args.device,
        use_amp=args.use_amp,
        use_fa3=False,
        offload_mode=False,
        random_state=fixture["seed"],
        n_jobs=None,
        verbose=False,
    )
    classifier.fit(x_context, y_context)
    prepared = prepare_query_variants(classifier, raw_query)
    prepared_hash = canonical_f32(prepared)
    if args.engine == "ironicl" and prepared.shape[0] != args.estimators:
        raise ValueError("Ironicl context ensemble count must equal the prepared query ensemble count")
    if args.request_rows is not None and (args.request_rows < 1 or raw_query.shape[0] % args.request_rows):
        raise ValueError("--request-rows must divide the fixture query-row count")
    bridge = (
        IroniclLibrary(args.library, args.model, args.context, len(classifier.classes_))
        if args.engine == "ironicl"
        else None
    )

    def infer() -> tuple[np.ndarray, int, int]:
        preprocessing_start = time.perf_counter_ns()
        if args.request_rows is None:
            variants = prepare_query_variants(classifier, raw_query)
            microbatch = None
        elif args.stream_preprocess_mode == "batched":
            variants = prepare_query_variants(classifier, raw_query)
            microbatch = [
                np.ascontiguousarray(variants[:, start:start + args.request_rows])
                for start in range(0, raw_query.shape[0], args.request_rows)
            ]
        else:
            microbatch = [
                prepare_query_variants(classifier, raw_query[start:start + args.request_rows])
                for start in range(0, raw_query.shape[0], args.request_rows)
            ]
            variants = None
        preprocessing_ns = time.perf_counter_ns() - preprocessing_start
        inference_start = time.perf_counter_ns()
        probabilities = bridge.predict_batch(microbatch) if microbatch is not None else (bridge.predict(variants) if bridge else pytorch_probabilities(classifier, variants))
        if args.device == "cuda":
            torch.cuda.synchronize()
        return probabilities, preprocessing_ns, time.perf_counter_ns() - inference_start

    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                infer()
            if args.device == "cuda":
                torch.cuda.synchronize()
            if bridge:
                bridge.reset_sparse_attention_statistics()
            samples: list[int] = []
            preprocessing_samples: list[int] = []
            inference_samples: list[int] = []
            probabilities = None
            for _ in range(args.iterations):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter_ns()
                probabilities, preprocessing_ns, inference_ns = infer()
                samples.append(time.perf_counter_ns() - start)
                preprocessing_samples.append(preprocessing_ns)
                inference_samples.append(inference_ns)
            sparse_attention_statistics = bridge.sparse_attention_statistics() if bridge else None
    finally:
        if bridge:
            bridge.close()
    assert probabilities is not None
    if not np.isfinite(probabilities).all() or probabilities.shape != (raw_query.shape[0], len(classifier.classes_)):
        raise RuntimeError("engine returned invalid probability output")
    mean_ns = sum(samples) / len(samples)
    result: dict[str, Any] = {
        "benchmark_contract": CONTRACT,
        "timed_boundary": "raw_numeric_validation+numerical_encoding+ensemble_variants+resident_kv_inference+class_correction+logit_average+temperature_softmax+host_f32_output",
        "engine": args.engine,
        "device": args.device,
        "precision": "amp" if args.use_amp else "fp32",
        "ironicl_math": "fast" if args.fast_ironicl else "exact_controls",
        "fixture_sha256": fixture["fixture_sha256"],
        "fixture_x_query_sha256": fixture["x_query_sha256"],
        "prepared_query_sha256_f32": prepared_hash,
        "model_sha256": sha256_file(args.model),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "context_sha256": sha256_file(args.context),
        "context_metadata_sha256": sha256_file(args.context_metadata),
        "cache_storage_dtype": args.context_kv_dtype,
        "compute_precision": "amp" if args.use_amp else "fp32",
        "context_rows": int(x_context.shape[0]),
        "query_rows": int(raw_query.shape[0]),
        "raw_feature_count": int(raw_query.shape[1]),
        "prepared_feature_count": int(prepared.shape[2]),
        "estimators": args.estimators,
        "ensemble_batch_size": args.batch_size,
        "stream_request_rows": args.request_rows,
        "stream_preprocess_mode": args.stream_preprocess_mode if args.request_rows is not None else None,
        "sparse_max_omitted_mass": sparse_max_omitted_mass() if args.engine == "ironicl" else None,
        "threads": args.threads,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms": latency_summary(samples),
        "preprocessing_latency_ms": latency_summary(preprocessing_samples),
        "inference_latency_ms": latency_summary(inference_samples),
        "samples_ns": samples,
        "query_rows_per_second": raw_query.shape[0] * 1_000_000_000 / mean_ns,
        "output_shape": [int(value) for value in probabilities.shape],
        "output_sha256_f32": canonical_f32(probabilities),
        "output_sum_f64": float(probabilities.astype(np.float64).sum()),
        "torch_version": torch.__version__,
        "context_format": context_metadata["format"],
    }
    if sparse_attention_statistics is not None:
        result["sparse_attention_statistics"] = sparse_attention_statistics
    if args.device == "cuda":
        result["cuda_device"] = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.probabilities_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    np.save(args.probabilities_output, probabilities, allow_pickle=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export official TabICL full-KV caches into IRONICL and IRQUERY artifacts.

Run with PYTHONHASHSEED=0. Inputs are either Covertype subsets or three .npy
arrays: X context [N,H], y context [N], and X query [Q,H].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
STATIC_MAGIC, QUERY_MAGIC, LOGITS_MAGIC = b"IRONICL\0", b"IRQUERY\0", b"IRLOGITS"
VERSION, STATIC_BF16_VERSION, STATIC_PACKED_VERSION, ALIGNMENT = 1, 2, 3, 64
DTYPE_FP32, DTYPE_BF16 = 1, 2
MAX_BYTES, MAX_TENSOR_BYTES = 48 * 1024**3, 4 * 1024**3
STATIC_FORMAT = "<8sII32s32sQQQQ5I3IQQI"
QUERY_FORMAT = "<8sII32s32sQQQQ"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_source_sha256(path: Path) -> str:
    """Read the source-checkpoint identity from a verified IRMODEL v1 artifact."""
    data = path.read_bytes()
    if len(data) < 176 or data[:8] != b"IRMODEL\0" or struct.unpack_from("<II", data, 8) != (1, 1):
        raise ValueError("model is not an IRMODEL v1 artifact")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("model artifact checksum does not match")
    source_hash = data[16:48]
    if source_hash == bytes(32):
        raise ValueError("model artifact source checkpoint hash is empty")
    return source_hash.hex()


def canonical_array(array: np.ndarray, dtype: str) -> bytes:
    array = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
    return struct.pack("<Q", array.ndim) + struct.pack(f"<{array.ndim}Q", *array.shape) + array.tobytes()


def context_identity(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> str:
    """Hash all source data and cache-affecting configuration canonically."""
    digest = hashlib.sha256(b"ironicl-static-context-v1\0")
    values = ((b"config", json.dumps(config, sort_keys=True, separators=(",", ":")).encode("ascii")), (b"x", canonical_array(x, "<f8")), (b"y", canonical_array(y, "<i8")))
    for label, value in values:
        digest.update(struct.pack("<Q", len(label)) + label + struct.pack("<Q", len(value)) + value)
    return digest.hexdigest()


def align(data: bytearray) -> None:
    data.extend(b"\0" * (-len(data) % ALIGNMENT))


def tensor_bytes(tensor: Any, name: str) -> tuple[tuple[int, ...], bytes]:
    if hasattr(tensor, "detach"):
        array = tensor.detach().to(device="cpu").float().contiguous().numpy()
    else:
        array = np.asarray(tensor)
    array = np.ascontiguousarray(array, dtype="<f4")
    if array.ndim == 0 or 0 in array.shape or not np.isfinite(array).all() or array.nbytes > MAX_TENSOR_BYTES:
        raise ValueError(f"invalid FP32 tensor {name}")
    return tuple(map(int, array.shape)), array.tobytes()


def tensor_chunks(value: Any, name: str) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    chunks = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if not chunks:
        raise ValueError(f"empty tensor chunks for {name}")
    shapes = [tuple(map(int, chunk.shape)) for chunk in chunks]
    if any(len(shape) == 0 or shape[0] == 0 or shape[1:] != shapes[0][1:] for shape in shapes):
        raise ValueError(f"inconsistent tensor chunks for {name}")
    return chunks, (sum(shape[0] for shape in shapes), *shapes[0][1:])


SPARSE_INDEX_NAMES = (
    "icl.index.centroids",
    "icl.index.members",
    "icl.index.offsets",
    "icl.index.radii",
)


def expected_cache_names() -> list[str]:
    return sorted(
        [f"column.{layer}.{kind}" for layer in range(3) for kind in ("key", "value")]
        + [f"icl.{layer}.{kind}" for layer in range(12) for kind in ("key", "value")]
    )


def sparse_index_shapes(batches: int, rows: int, clusters: int) -> dict[str, tuple[int, ...]]:
    return {
        "icl.index.centroids": (batches, 12, 8, clusters, 64),
        "icl.index.radii": (batches, 12, 8, clusters),
        "icl.index.offsets": (batches, 12, 8, clusters + 1),
        "icl.index.members": (batches, 12, 8, rows),
    }


def deterministic_kmeans_index(keys: np.ndarray, clusters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic coarse index for one logical [N, 64] key matrix."""
    rows, dimensions = keys.shape
    if dimensions != 64 or not 1 < clusters <= rows:
        raise ValueError("sparse index requires 2..=N clusters for width-64 ICL keys")
    keys = np.ascontiguousarray(keys, dtype=np.float32)
    centroids = keys[np.linspace(0, rows - 1, clusters, dtype=np.int64)].copy()
    assignments = np.zeros(rows, dtype=np.int64)
    key_norms = np.einsum("nd,nd->n", keys, keys)
    for _ in range(4):
        centroid_norms = np.einsum("cd,cd->c", centroids, centroids)
        distances = key_norms[:, None] + centroid_norms[None, :] - 2.0 * keys @ centroids.T
        assignments = distances.argmin(axis=1)
        for cluster in range(clusters):
            members = keys[assignments == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0, dtype=np.float32)
    order = np.lexsort((np.arange(rows, dtype=np.int64), assignments))
    counts = np.bincount(assignments, minlength=clusters)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    radii = np.zeros(clusters, dtype=np.float32)
    for cluster in range(clusters):
        members = keys[assignments == cluster]
        if len(members):
            radii[cluster] = np.linalg.norm(members - centroids[cluster], axis=1).max()
    return centroids, radii, offsets, order


def sparse_index_tensors(caches: list[Any], clusters: int) -> list[tuple[str, np.ndarray]]:
    """Build per-estimator, per-layer, per-head coarse ICL key indices."""
    first_layers = getattr(caches[0], "icl_cache").kv
    layers = len(first_layers)
    if layers != 12:
        raise ValueError("sparse index requires 12 ICL layers")
    first_key = np.concatenate(
        [cache.icl_cache.kv[0].key.detach().to(device="cpu").float().contiguous().numpy() for cache in caches],
        axis=0,
    )
    batches, heads, rows, dimensions = map(int, first_key.shape)
    if heads != 8 or dimensions != 64:
        raise ValueError("sparse index requires [8, N, 64] ICL keys")
    centroids = np.empty((batches, layers, heads, clusters, dimensions), dtype=np.float32)
    radii = np.empty((batches, layers, heads, clusters), dtype=np.float32)
    offsets = np.empty((batches, layers, heads, clusters + 1), dtype=np.float32)
    members = np.empty((batches, layers, heads, rows), dtype=np.float32)
    for layer in range(layers):
        keys = np.concatenate(
            [cache.icl_cache.kv[layer].key.detach().to(device="cpu").float().contiguous().numpy() for cache in caches],
            axis=0,
        )
        if tuple(map(int, keys.shape)) != (batches, heads, rows, dimensions):
            raise ValueError("ICL cache key shapes changed while building sparse index")
        for batch in range(batches):
            for head in range(heads):
                center, radius, offset, member = deterministic_kmeans_index(keys[batch, head], clusters)
                centroids[batch, layer, head] = center
                radii[batch, layer, head] = radius
                offsets[batch, layer, head] = offset
                members[batch, layer, head] = member
    return [
        ("icl.index.centroids", centroids),
        ("icl.index.radii", radii),
        ("icl.index.offsets", offsets),
        ("icl.index.members", members),
    ]


def encode_static_context(model_hash: str, context_hash: str, classes: np.ndarray, features: np.ndarray, tensors: list[tuple[str, Any]]) -> bytes:
    """Encode exactly the layout read by src/static_context.rs."""
    classes, features = np.ascontiguousarray(classes, dtype="<u8"), np.ascontiguousarray(features, dtype="<u8")
    if classes.ndim != 2 or features.ndim != 2 or classes.shape[0] != features.shape[0]:
        raise ValueError("permutations must be estimator-major rank-two arrays")
    b, c, h = classes.shape[0], classes.shape[1], features.shape[1]
    if not b or not c or not h:
        raise ValueError("empty permutation")
    for name, values, width in (("class", classes, c), ("feature", features, h)):
        if not np.all(np.sort(values, axis=1) == np.arange(width, dtype=np.uint64)):
            raise ValueError(f"invalid {name} permutations")
    entries = sorted((name, *tensor_bytes(value, name)) for name, value in tensors)
    expected = expected_cache_names()
    names = [entry[0] for entry in entries]
    if names != expected and names != sorted([*expected, *SPARSE_INDEX_NAMES]):
        raise ValueError("expected cache tensors with an optional complete sparse index")
    n = next(shape[2] for name, shape, _ in entries if name == "icl.0.key")
    expected_column, expected_icl = (b, h + 4, 8, 128, 16), (b, 8, n, 64)
    for name, shape, _ in entries:
        if name in SPARSE_INDEX_NAMES:
            clusters = next(index_shape[3] for index_name, index_shape, _ in entries if index_name == "icl.index.centroids")
            if shape != sparse_index_shapes(b, n, clusters)[name]:
                raise ValueError(f"invalid sparse index tensor shape for {name}: {shape}")
            continue
        required = expected_column if name.startswith("column.") else expected_icl
        if shape != required:
            raise ValueError(f"unexpected official full-KV shape for {name}: {shape}; expected {required}")
    directory, payload = bytearray(), bytearray()
    for name, shape, values in entries:
        if shape[0] != b:
            raise ValueError(f"batch mismatch for {name}")
        align(payload)
        encoded_name = name.encode("utf-8")
        directory.extend(struct.pack("<IHHQQ", len(encoded_name), len(shape), 0, len(payload), len(values)))
        directory.extend(struct.pack(f"<{len(shape)}Q", *shape) + encoded_name)
        payload.extend(values)
    header = struct.pack(STATIC_FORMAT, STATIC_MAGIC, VERSION, 1, bytes.fromhex(model_hash), bytes.fromhex(context_hash), b, n, h, c, 3, 12, 8, 16, 64, len(entries), classes.size, features.size, len(directory), len(payload), 0)
    if len(header) != 164 or len(payload) > MAX_BYTES:
        raise ValueError("invalid static context size")
    data = bytearray(header)
    data.extend(classes.tobytes() + features.tobytes() + directory)
    align(data)
    data.extend(payload)
    data.extend(hashlib.sha256(data).digest())
    if len(data) > MAX_BYTES:
        raise ValueError("static context exceeds resource limit")
    return bytes(data)


def write_static_context(path: Path, model_hash: str, context_hash: str, classes: np.ndarray, features: np.ndarray, tensors: list[tuple[str, Any]], kv_dtype: str = "fp32", pack_keys: bool = False) -> dict[str, Any]:
    """Stream an IRONICL artifact without materializing a second full cache."""
    classes, features = np.ascontiguousarray(classes, dtype="<u8"), np.ascontiguousarray(features, dtype="<u8")
    if classes.ndim != 2 or features.ndim != 2 or classes.shape[0] != features.shape[0]:
        raise ValueError("permutations must be estimator-major rank-two arrays")
    b, c, h = classes.shape[0], classes.shape[1], features.shape[1]
    if not b or not c or not h:
        raise ValueError("empty permutation")
    for name, values, width in (("class", classes, c), ("feature", features, h)):
        if not np.all(np.sort(values, axis=1) == np.arange(width, dtype=np.uint64)):
            raise ValueError(f"invalid {name} permutations")
    if kv_dtype not in ("fp32", "bf16", "bf16-k", "bf16-v"):
        raise ValueError(f"unsupported K/V dtype: {kv_dtype}")
    context_version = STATIC_PACKED_VERSION if pack_keys else (VERSION if kv_dtype == "fp32" else STATIC_BF16_VERSION)
    entries = sorted((name, *tensor_chunks(value, name)) for name, value in tensors)
    expected = expected_cache_names()
    names = [entry[0] for entry in entries]
    if names != expected and names != sorted([*expected, *SPARSE_INDEX_NAMES]):
        raise ValueError("expected cache tensors with an optional complete sparse index")
    n = next(shape[2] for name, _, shape in entries if name == "icl.0.key")
    expected_column, expected_icl = (b, h + 4, 8, 128, 16), (b, 8, n, 64)
    directory, payload_length, descriptors = bytearray(), 0, []
    index_shapes = None
    if any(name in SPARSE_INDEX_NAMES for name, _, _ in entries):
        clusters = next(shape[3] for name, _, shape in entries if name == "icl.index.centroids")
        index_shapes = sparse_index_shapes(b, n, clusters)
    for name, chunks, shape in entries:
        required = index_shapes[name] if index_shapes and name in SPARSE_INDEX_NAMES else (expected_column if name.startswith("column.") else expected_icl)
        if shape != required:
            raise ValueError(f"unexpected static-context shape for {name}: {shape}; expected {required}")
        payload_length = (payload_length + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        use_bf16 = name.startswith("icl.") and name not in SPARSE_INDEX_NAMES and (
            kv_dtype == "bf16"
            or (kv_dtype == "bf16-k" and name.endswith(".key"))
            or (kv_dtype == "bf16-v" and name.endswith(".value"))
        )
        dtype = DTYPE_BF16 if use_bf16 else DTYPE_FP32
        byte_length = int(np.prod(shape, dtype=np.uint64)) * (2 if dtype == DTYPE_BF16 else 4)
        if byte_length > MAX_TENSOR_BYTES:
            raise ValueError(f"FP32 tensor exceeds resource limit: {name}")
        encoded_name = name.encode("utf-8")
        directory_dtype = 0 if context_version == VERSION else dtype
        directory.extend(struct.pack("<IHHQQ", len(encoded_name), len(shape), directory_dtype, payload_length, byte_length))
        directory.extend(struct.pack(f"<{len(shape)}Q", *shape) + encoded_name)
        descriptors.append((name, chunks, shape, dtype, payload_length, byte_length))
        payload_length += byte_length
    header = struct.pack(STATIC_FORMAT, STATIC_MAGIC, context_version, 1, bytes.fromhex(model_hash), bytes.fromhex(context_hash), b, n, h, c, 3, 12, 8, 16, 64, len(entries), classes.size, features.size, len(directory), payload_length, 0)
    prefix = bytearray(header)
    prefix.extend(classes.tobytes() + features.tobytes() + directory)
    align(prefix)
    final_length = len(prefix) + payload_length + 32
    if len(header) != 164 or final_length > MAX_BYTES:
        raise ValueError("static context exceeds resource limit")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as file:
            file.write(prefix); digest.update(prefix)
            position = 0
            for name, chunks, expected_shape, dtype, offset, expected_bytes in descriptors:
                padding = offset - position
                if padding:
                    zeros = bytes(padding); file.write(zeros); digest.update(zeros)
                written = 0
                for chunk in chunks:
                    shape, encoded = tensor_bytes(chunk, name)
                    if shape[1:] != expected_shape[1:]:
                        raise ValueError(f"tensor changed while exporting: {name}")
                    if pack_keys and name.startswith("icl.") and name.endswith(".key"):
                        source = np.frombuffer(encoded, dtype="<f4").reshape(shape)
                        packed = np.empty(source.size, dtype="<f4")
                        for ensemble in range(shape[0]):
                            for head in range(shape[1]):
                                base = (ensemble * shape[1] + head) * shape[2] * shape[3]
                                for start in range(0, shape[2], 128):
                                    rows = min(128, shape[2] - start)
                                    packed[base + start * shape[3]:base + (start + rows) * shape[3]] = source[ensemble, head, start:start + rows].T.reshape(-1)
                        encoded = packed.tobytes()
                    if dtype == DTYPE_BF16:
                        fp32 = np.frombuffer(encoded, dtype="<f4")
                        bits = fp32.view("<u4")
                        rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
                        encoded = np.ascontiguousarray(rounded >> np.uint32(16), dtype="<u2").tobytes()
                    file.write(encoded); digest.update(encoded); written += len(encoded)
                if written != expected_bytes:
                    raise ValueError(f"tensor byte length changed while exporting: {name}")
                position = offset + expected_bytes
            if position != payload_length:
                raise ValueError("static context payload length changed while exporting")
            file.write(digest.digest())
            file.flush(); os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "B": b,
        "N": n,
        "H": h,
        "C": c,
        "tensor_shapes": {name: list(shape) for name, _, shape, _, _, _ in descriptors},
        "kv_dtype": kv_dtype,
        "static_context_version": context_version,
        "packed_icl_keys": pack_keys,
        "context_file_sha256": sha256_file(path),
        "context_file_bytes": final_length,
    }


def parse_static_context(data: bytes) -> dict[str, Any]:
    """Dependency-free IRONICL v1 parser for exporter tests and verification."""
    if len(data) < 196 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("invalid IRONICL checksum")
    fields = struct.unpack_from(STATIC_FORMAT, data)
    if fields[0] != STATIC_MAGIC or fields[1] not in (VERSION, STATIC_BF16_VERSION, STATIC_PACKED_VERSION) or fields[2] != 1:
        raise ValueError("unsupported IRONICL format")
    model, context, b, n, h, c = fields[3:9]
    tensors, class_count, feature_count, directory_length, payload_length, reserved = fields[14:20]
    if reserved or class_count != b * c or feature_count != b * h:
        raise ValueError("invalid IRONICL metadata")
    directory_start = 164 + 8 * (class_count + feature_count)
    directory_end = directory_start + directory_length
    payload_start = (directory_end + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    if payload_start + payload_length != len(data) - 32:
        raise ValueError("invalid IRONICL sections")
    offset, result = directory_start, []
    for _ in range(tensors):
        name_len, rank, dtype, payload_offset, byte_len = struct.unpack_from("<IHHQQ", data, offset)
        offset += 24
        shape = struct.unpack_from(f"<{rank}Q", data, offset)
        offset += rank * 8
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len
        element_bytes = 4 if fields[1] == VERSION or dtype == DTYPE_FP32 else 2
        valid_dtype = dtype == 0 if fields[1] == VERSION else dtype in (DTYPE_FP32, DTYPE_BF16)
        if not name_len or not rank or not valid_dtype or np.prod(shape, dtype=np.uint64) * element_bytes != byte_len or payload_offset + byte_len > payload_length:
            raise ValueError("invalid IRONICL tensor")
        result.append({
            "name": name,
            "shape": list(shape),
            "dtype": "bf16" if element_bytes == 2 else "fp32",
        })
    if offset != directory_end:
        raise ValueError("invalid IRONICL directory")
    return {"version": fields[1], "model_sha256": model.hex(), "context_sha256": context.hex(), "B": b, "N": n, "H": h, "C": c, "tensors": result}


def encode_query(model_hash: str, context_hash: str, variants: np.ndarray) -> bytes:
    variants = np.ascontiguousarray(variants, dtype="<f4")
    if variants.ndim != 3 or 0 in variants.shape or not np.isfinite(variants).all():
        raise ValueError("queries must be finite [B,Q,H] FP32")
    b, q, h = variants.shape
    data = struct.pack(QUERY_FORMAT, QUERY_MAGIC, VERSION, 1, bytes.fromhex(model_hash), bytes.fromhex(context_hash), b, q, h, variants.size) + variants.tobytes()
    return data + hashlib.sha256(data).digest()


def parse_query(data: bytes) -> dict[str, Any]:
    if len(data) < struct.calcsize(QUERY_FORMAT) + 32 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("invalid IRQUERY checksum")
    magic, version, kind, model, context, b, q, h, count = struct.unpack_from(QUERY_FORMAT, data)
    if (magic, version, kind) != (QUERY_MAGIC, VERSION, 1) or count != b * q * h or len(data) != struct.calcsize(QUERY_FORMAT) + count * 4 + 32:
        raise ValueError("invalid IRQUERY metadata")
    return {"model_sha256": model.hex(), "context_sha256": context.hex(), "B": b, "Q": q, "H": h}


def arguments() -> argparse.Namespace:
    root, parser = Path(__file__).resolve().parents[1], argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context-output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--raw-logits-output", type=Path)
    parser.add_argument("--tabicl-root", type=Path, default=root.parent / "tabicl" / "src")
    parser.add_argument("--fixture", type=Path, help="Raw-X .npz fixture from prepare_cached_benchmark_fixture.py")
    parser.add_argument("--x-context", type=Path); parser.add_argument("--y-context", type=Path); parser.add_argument("--x-query", type=Path)
    parser.add_argument("--dataset", choices=("covertype",)); parser.add_argument("--data-home", type=Path)
    parser.add_argument("--context-rows", type=int, default=128); parser.add_argument("--query-rows", type=int, default=8)
    parser.add_argument("--estimators", type=int, default=2); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--threads", type=int, default=1); parser.add_argument("--kv-dtype", choices=("fp32", "bf16", "bf16-k", "bf16-v"), default="fp32"); parser.add_argument("--pack-keys", action="store_true")
    parser.add_argument("--sparse-index", action="store_true", help="Export deterministic ICL key cluster index tensors.")
    parser.add_argument("--index-clusters", type=int, default=64, help="Clusters per ICL attention head when --sparse-index is set.")
    return parser.parse_args()


def source_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if args.fixture:
        metadata_path = args.fixture.with_suffix(args.fixture.suffix + ".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("benchmark_contract") != "raw_x_cached_serving_v1":
            raise ValueError("fixture does not use the raw-X cached-serving contract")
        with np.load(args.fixture, allow_pickle=False) as fixture:
            x_context = np.ascontiguousarray(fixture["x_context"])
            y_context = np.ascontiguousarray(fixture["y_context"])
            x_query = np.ascontiguousarray(fixture["x_query"])
        return x_context, y_context, x_query, {
            "dataset": "fixture",
            "fixture_sha256": metadata["fixture_sha256"],
            "context_indices_sha256": metadata["context_indices_sha256"],
            "query_indices_sha256": metadata["query_indices_sha256"],
            "x_context_sha256": metadata["x_context_sha256"],
            "y_context_sha256": metadata["y_context_sha256"],
            "x_query_sha256": metadata["x_query_sha256"],
        }
    if args.dataset:
        from sklearn.datasets import fetch_covtype
        from sklearn.model_selection import train_test_split
        dataset = fetch_covtype(data_home=str(args.data_home) if args.data_home else None)
        train, query = train_test_split(np.arange(len(dataset.target)), train_size=args.context_rows, test_size=args.query_rows, random_state=args.seed, stratify=dataset.target)
        return np.asarray(dataset.data)[train], np.asarray(dataset.target)[train], np.asarray(dataset.data)[query], {"dataset": "covertype", "context_indices_sha256": hashlib.sha256(canonical_array(train, "<i8")).hexdigest(), "query_indices_sha256": hashlib.sha256(canonical_array(query, "<i8")).hexdigest()}
    if not all((args.x_context, args.y_context, args.x_query)):
        raise ValueError("provide a fixture, three .npy arrays, or --dataset")
    return np.load(args.x_context, allow_pickle=False), np.load(args.y_context, allow_pickle=False), np.load(args.x_query, allow_pickle=False), {"dataset": "explicit-npy"}


def build_kv_cache_batches(classifier: Any, torch: Any) -> dict[str, list[Any]]:
    """Build cache batches without TabICLCache.concat duplicating all K/V."""
    train_data = classifier.ensemble_generator_.transform(X=None, mode="train")
    result = {}
    for norm_method, (features, labels) in train_data.items():
        batch_size = classifier.batch_size or features.shape[0]
        batches = int(np.ceil(features.shape[0] / batch_size))
        feature_batches = np.array_split(features, batches)
        label_batches = np.array_split(labels, batches)
        caches = []
        for feature_batch, label_batch in zip(feature_batches, label_batches):
            feature_batch = torch.from_numpy(feature_batch).float().to(classifier.device_)
            label_batch = torch.from_numpy(label_batch).float().to(classifier.device_)
            with torch.no_grad():
                classifier.model_.forward_with_cache(
                    X_train=feature_batch,
                    y_train=label_batch,
                    use_cache=False,
                    store_cache=True,
                    cache_mode="kv",
                    inference_config=classifier.inference_config_,
                )
            caches.append(classifier.model_._cache)
            classifier.model_.clear_cache()
        result[norm_method] = caches
    return result


def main() -> None:
    args = arguments()
    if os.environ.get("PYTHONHASHSEED") != "0": raise SystemExit("PYTHONHASHSEED=0 is required")
    if not args.checkpoint.is_file() or not args.model.is_file() or not (args.tabicl_root / "tabicl" / "__init__.py").is_file(): raise SystemExit("invalid checkpoint, model artifact, or TabICL root")
    checkpoint_hash = sha256_file(args.checkpoint)
    model_file_hash = sha256_file(args.model)
    try:
        model_hash = model_source_sha256(args.model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if checkpoint_hash != OFFICIAL_CHECKPOINT_SHA256: raise SystemExit(f"checkpoint is not official: {checkpoint_hash}")
    if model_hash != checkpoint_hash: raise SystemExit(f"model artifact source checkpoint does not match --checkpoint: {model_hash}")
    x_train, y_train, x_query, source = source_arrays(args)
    x_train, y_train, x_query = np.asarray(x_train), np.asarray(y_train), np.asarray(x_query)
    if x_train.ndim != 2 or y_train.ndim != 1 or x_query.ndim != 2 or len(x_train) != len(y_train) or x_train.shape[1] != x_query.shape[1] or not np.issubdtype(x_train.dtype, np.number) or not np.issubdtype(x_query.dtype, np.number): raise SystemExit("inputs must be numeric X[N,H], y[N], X[Q,H]")
    if not (2 <= len(x_train) <= 1_000_000 and 1 <= len(x_query) <= 1_000_000 and 1 <= x_train.shape[1] <= 65_536): raise SystemExit("input dimensions exceed resource limits")
    if args.sparse_index and not (2 <= args.index_clusters <= len(x_train)):
        raise SystemExit("--index-clusters must be in 2..=context rows when --sparse-index is set")
    config = {"checkpoint_sha256": checkpoint_hash, "model_artifact_sha256": model_file_hash, "n_estimators": args.estimators, "batch_size": args.batch_size, "random_state": args.seed, "kv_cache": "kv", "kv_dtype": args.kv_dtype, "pack_keys": args.pack_keys, "sparse_index": args.sparse_index, "index_clusters": args.index_clusters if args.sparse_index else None, "norm_methods": ["none", "power"], "feature_shuffle": "latin", "class_shuffle": "shift", "outlier_threshold": 4.0}
    context_hash = context_identity(x_train, y_train, config)
    sys.path.insert(0, str(args.tabicl_root.resolve()))
    import torch
    from tabicl import TabICLClassifier
    if args.threads < 1: raise SystemExit("--threads must be positive")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    classifier = TabICLClassifier(n_estimators=args.estimators, batch_size=args.batch_size, kv_cache=False, model_path=args.checkpoint, allow_auto_download=False, device="cpu", use_amp=False, use_fa3=False, offload_mode=False, random_state=args.seed, n_jobs=None, verbose=False)
    classifier.fit(x_train, y_train)
    cache_batches = build_kv_cache_batches(classifier, torch)
    query_data = classifier.ensemble_generator_.transform(classifier.X_encoder_.transform(x_query), mode="test")
    queries, classes, features, logits = [], [], [], []
    for norm, caches in cache_batches.items():
        group = query_data[norm][0]; queries.append(group); classes.extend(classifier.ensemble_generator_.class_shuffles_[norm]); features.extend(classifier.ensemble_generator_.feature_shuffles_[norm])
        offset = 0
        for cache in caches:
            count = next(iter(cache.icl_cache.kv.values())).key.shape[0]
            logits.append(classifier._batch_forward_with_cache(group[offset:offset + count], cache)); offset += count
        if offset != group.shape[0]: raise RuntimeError(f"cache batch count changed for {norm}")
    all_caches = [cache for caches in cache_batches.values() for cache in caches]
    tensors: list[tuple[str, Any]] = []
    for prefix, cache_name in (("column", "col_cache"), ("icl", "icl_cache")):
        layers = getattr(all_caches[0], cache_name).kv
        for layer in sorted(layers):
            for kind in ("key", "value"):
                tensors.append((f"{prefix}.{layer}.{kind}", [getattr(getattr(cache, cache_name).kv[layer], kind) for cache in all_caches]))
    if args.sparse_index:
        tensors.extend(sparse_index_tensors(all_caches, args.index_clusters))
    query_variants = np.ascontiguousarray(np.concatenate(queries), dtype="<f4")
    context_info = write_static_context(args.context_output, model_hash, context_hash, np.asarray(classes, dtype="<u8"), np.asarray(features, dtype="<u8"), tensors, args.kv_dtype, args.pack_keys)
    query = encode_query(model_hash, context_hash, query_variants)
    args.query_output.parent.mkdir(parents=True, exist_ok=True); args.query_output.write_bytes(query)
    if args.raw_logits_output:
        values = np.ascontiguousarray(np.concatenate(logits), dtype="<f4"); raw = struct.pack("<8sIIQQQQ", LOGITS_MAGIC, VERSION, 1, *values.shape, values.size) + values.tobytes(); args.raw_logits_output.parent.mkdir(parents=True, exist_ok=True); args.raw_logits_output.write_bytes(raw + hashlib.sha256(raw).digest())
    metadata = {"format": f"IRONICL static-context v{context_info['static_context_version']}", "checkpoint_sha256": checkpoint_hash, "model_source_sha256": model_hash, "model_artifact_sha256": model_file_hash, "context_sha256": context_hash, "source": source, **context_info, "sparse_index": args.sparse_index, "index_clusters": args.index_clusters if args.sparse_index else None, "query_shape": list(query_variants.shape), "query_file_sha256": hashlib.sha256(query).hexdigest()}
    path = args.metadata_output or args.context_output.with_suffix(args.context_output.suffix + ".json"); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__": main()

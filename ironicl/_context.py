"""Static-context identity and a pure-Python reference of the ``IRONICL`` format.

The estimator builds contexts natively through ``ironicl._ironicl.build_context``
(``src/context_builder.rs``). This module keeps the canonical context
identity hash the estimator uses to name contexts, and a numpy reference
writer/parser of the format read by ``src/static_context.rs`` for
tests and tooling.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

STATIC_MAGIC = b"IRONICL\0"
VERSION, BF16_VERSION, PACKED_VERSION = 1, 2, 3
ALIGNMENT = 64
DTYPE_FP32, DTYPE_BF16 = 1, 2
MAX_BYTES, MAX_TENSOR_BYTES = 48 * 1024**3, 4 * 1024**3
HEADER_FORMAT = "<8sII32s32sQQQQ5I3IQQI"
HEADER_BYTES = 164
KV_DTYPES = ("fp32", "bf16", "bf16-v")
COLUMN_LAYERS, ICL_LAYERS, HEADS = 3, 12, 8
SPARSE_INDEX_NAMES = (
    "icl.index.centroids",
    "icl.index.members",
    "icl.index.offsets",
    "icl.index.radii",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array(array: np.ndarray, dtype: str) -> bytes:
    array = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
    header = struct.pack("<Q", array.ndim) + struct.pack(
        f"<{array.ndim}Q", *array.shape
    )
    return header + array.tobytes()


def context_identity(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> str:
    """Hash the training table and every cache-affecting setting canonically."""
    digest = hashlib.sha256(b"ironicl-static-context-v1\0")
    encoded_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
    values = (
        (b"config", encoded_config.encode("ascii")),
        (b"x", canonical_array(x, "<f8")),
        (b"y", canonical_array(y, "<i8")),
    )
    for label, value in values:
        digest.update(struct.pack("<Q", len(label)) + label)
        digest.update(struct.pack("<Q", len(value)) + value)
    return digest.hexdigest()


def _align(data: bytearray) -> None:
    data.extend(b"\0" * (-len(data) % ALIGNMENT))


def tensor_bytes(tensor: Any, name: str) -> tuple[tuple[int, ...], bytes]:
    """Return the shape and little-endian FP32 bytes of an array-like tensor."""
    array = np.ascontiguousarray(np.asarray(tensor), dtype="<f4")
    if (
        array.ndim == 0
        or 0 in array.shape
        or not np.isfinite(array).all()
        or array.nbytes > MAX_TENSOR_BYTES
    ):
        raise ValueError(f"invalid FP32 tensor {name}")
    return tuple(map(int, array.shape)), array.tobytes()


def _tensor_chunks(value: Any, name: str) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    chunks = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if not chunks:
        raise ValueError(f"empty tensor chunks for {name}")
    shapes = [tuple(map(int, chunk.shape)) for chunk in chunks]
    if any(len(s) == 0 or s[0] == 0 or s[1:] != shapes[0][1:] for s in shapes):
        raise ValueError(f"inconsistent tensor chunks for {name}")
    return chunks, (sum(shape[0] for shape in shapes), *shapes[0][1:])


def expected_cache_names() -> list[str]:
    return sorted(
        [
            f"column.{layer}.{kind}"
            for layer in range(COLUMN_LAYERS)
            for kind in ("key", "value")
        ]
        + [
            f"icl.{layer}.{kind}"
            for layer in range(ICL_LAYERS)
            for kind in ("key", "value")
        ]
    )


def sparse_index_shapes(
    batches: int, rows: int, clusters: int
) -> dict[str, tuple[int, ...]]:
    return {
        "icl.index.centroids": (batches, ICL_LAYERS, HEADS, clusters, 64),
        "icl.index.radii": (batches, ICL_LAYERS, HEADS, clusters),
        "icl.index.offsets": (batches, ICL_LAYERS, HEADS, clusters + 1),
        "icl.index.members": (batches, ICL_LAYERS, HEADS, rows),
    }


def deterministic_kmeans_index(
    keys: np.ndarray, clusters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic coarse index for one ``[N, 64]`` key matrix."""
    rows, dimensions = keys.shape
    if dimensions != 64 or not 1 < clusters <= rows:
        raise ValueError("sparse index requires 2..=N clusters for width-64 ICL keys")
    keys = np.ascontiguousarray(keys, dtype=np.float32)
    centroids = keys[np.linspace(0, rows - 1, clusters, dtype=np.int64)].copy()
    assignments = np.zeros(rows, dtype=np.int64)
    key_norms = np.einsum("nd,nd->n", keys, keys)
    for _ in range(4):
        centroid_norms = np.einsum("cd,cd->c", centroids, centroids)
        distances = (
            key_norms[:, None] + centroid_norms[None, :] - 2.0 * keys @ centroids.T
        )
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


def sparse_index_tensors(
    icl_keys: list[np.ndarray], clusters: int
) -> list[tuple[str, np.ndarray]]:
    """Build per-estimator, per-layer, per-head coarse ICL key indices.

    ``icl_keys[layer]`` is the full ``[B, 8, N, 64]`` key tensor of that layer.
    """
    if len(icl_keys) != ICL_LAYERS:
        raise ValueError(f"sparse index requires {ICL_LAYERS} ICL layers")
    batches, heads, rows, dimensions = map(int, icl_keys[0].shape)
    if heads != HEADS or dimensions != 64:
        raise ValueError("sparse index requires [B, 8, N, 64] ICL keys")
    centroids = np.empty(
        (batches, ICL_LAYERS, heads, clusters, dimensions), dtype=np.float32
    )
    radii = np.empty((batches, ICL_LAYERS, heads, clusters), dtype=np.float32)
    offsets = np.empty((batches, ICL_LAYERS, heads, clusters + 1), dtype=np.float32)
    members = np.empty((batches, ICL_LAYERS, heads, rows), dtype=np.float32)
    for layer, keys in enumerate(icl_keys):
        if tuple(map(int, keys.shape)) != (batches, heads, rows, dimensions):
            raise ValueError("ICL cache key shapes differ between layers")
        for batch in range(batches):
            for head in range(heads):
                center, radius, offset, member = deterministic_kmeans_index(
                    keys[batch, head], clusters
                )
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


def _check_permutations(
    classes: np.ndarray, features: np.ndarray
) -> tuple[int, int, int]:
    if classes.ndim != 2 or features.ndim != 2 or classes.shape[0] != features.shape[0]:
        raise ValueError("permutations must be estimator-major rank-two arrays")
    b, c, h = classes.shape[0], classes.shape[1], features.shape[1]
    if not b or not c or not h:
        raise ValueError("empty permutation")
    for name, values, width in (("class", classes, c), ("feature", features, h)):
        if not np.all(np.sort(values, axis=1) == np.arange(width, dtype=np.uint64)):
            raise ValueError(f"invalid {name} permutations")
    return b, c, h


def _pack_keys(encoded: bytes, shape: tuple[int, ...]) -> bytes:
    source = np.frombuffer(encoded, dtype="<f4").reshape(shape)
    packed = np.empty(source.size, dtype="<f4")
    rows_per_tile = 128
    for ensemble in range(shape[0]):
        for head in range(shape[1]):
            base = (ensemble * shape[1] + head) * shape[2] * shape[3]
            for start in range(0, shape[2], rows_per_tile):
                rows = min(rows_per_tile, shape[2] - start)
                lo, hi = base + start * shape[3], base + (start + rows) * shape[3]
                packed[lo:hi] = source[ensemble, head, start : start + rows].T.reshape(
                    -1
                )
    return packed.tobytes()


def _to_bf16(encoded: bytes) -> bytes:
    bits = np.frombuffer(encoded, dtype="<f4").view("<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray(rounded >> np.uint32(16), dtype="<u2").tobytes()


def write_static_context(
    path: Path,
    model_hash: str,
    context_hash: str,
    classes: np.ndarray,
    features: np.ndarray,
    tensors: list[tuple[str, Any]],
    kv_dtype: str = "fp32",
    pack_keys: bool = False,
) -> dict[str, Any]:
    """Stream an ``IRONICL`` artifact to ``path`` without a second in-memory copy.

    ``tensors`` maps each cache name to one tensor or a list of batch chunks
    along the estimator axis. Returns a metadata dictionary describing the file.
    """
    classes = np.ascontiguousarray(classes, dtype="<u8")
    features = np.ascontiguousarray(features, dtype="<u8")
    b, c, h = _check_permutations(classes, features)
    if kv_dtype not in KV_DTYPES:
        raise ValueError(f"unsupported K/V dtype: {kv_dtype}")
    if pack_keys:
        context_version = PACKED_VERSION
    else:
        context_version = VERSION if kv_dtype == "fp32" else BF16_VERSION
    entries = sorted((name, *_tensor_chunks(value, name)) for name, value in tensors)
    names = [entry[0] for entry in entries]
    expected = expected_cache_names()
    if names != expected and names != sorted([*expected, *SPARSE_INDEX_NAMES]):
        raise ValueError(
            "expected cache tensors with an optional complete sparse index"
        )
    n = next(shape[2] for name, _, shape in entries if name == "icl.0.key")
    expected_column, expected_icl = (b, h + 4, HEADS, 128, 16), (b, HEADS, n, 64)
    index_shapes = None
    if any(name in SPARSE_INDEX_NAMES for name, _, _ in entries):
        clusters = next(
            shape[3] for name, _, shape in entries if name == "icl.index.centroids"
        )
        index_shapes = sparse_index_shapes(b, n, clusters)
    directory, payload_length, descriptors = bytearray(), 0, []
    for name, chunks, shape in entries:
        if index_shapes and name in SPARSE_INDEX_NAMES:
            required = index_shapes[name]
        elif name.startswith("column."):
            required = expected_column
        else:
            required = expected_icl
        if shape != required:
            raise ValueError(
                f"unexpected shape for {name}: {shape}; expected {required}"
            )
        payload_length = (payload_length + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        use_bf16 = (
            name.startswith("icl.")
            and name not in SPARSE_INDEX_NAMES
            and (
                kv_dtype == "bf16" or (kv_dtype == "bf16-v" and name.endswith(".value"))
            )
        )
        dtype = DTYPE_BF16 if use_bf16 else DTYPE_FP32
        byte_length = int(np.prod(shape, dtype=np.uint64)) * (2 if use_bf16 else 4)
        if byte_length > MAX_TENSOR_BYTES:
            raise ValueError(f"tensor exceeds resource limit: {name}")
        encoded_name = name.encode("utf-8")
        directory_dtype = 0 if context_version == VERSION else dtype
        directory.extend(
            struct.pack(
                "<IHHQQ",
                len(encoded_name),
                len(shape),
                directory_dtype,
                payload_length,
                byte_length,
            )
        )
        directory.extend(struct.pack(f"<{len(shape)}Q", *shape) + encoded_name)
        descriptors.append((name, chunks, shape, dtype, payload_length, byte_length))
        payload_length += byte_length
    header = struct.pack(
        HEADER_FORMAT,
        STATIC_MAGIC,
        context_version,
        1,
        bytes.fromhex(model_hash),
        bytes.fromhex(context_hash),
        b,
        n,
        h,
        c,
        COLUMN_LAYERS,
        ICL_LAYERS,
        HEADS,
        16,
        64,
        len(entries),
        classes.size,
        features.size,
        len(directory),
        payload_length,
        0,
    )
    prefix = bytearray(header)
    prefix.extend(classes.tobytes() + features.tobytes() + directory)
    _align(prefix)
    final_length = len(prefix) + payload_length + 32
    if len(header) != HEADER_BYTES or final_length > MAX_BYTES:
        raise ValueError("static context exceeds resource limit")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as file:
            file.write(prefix)
            digest.update(prefix)
            position = 0
            for (
                name,
                chunks,
                expected_shape,
                dtype,
                offset,
                expected_bytes,
            ) in descriptors:
                padding = offset - position
                if padding:
                    zeros = bytes(padding)
                    file.write(zeros)
                    digest.update(zeros)
                written = 0
                for chunk in chunks:
                    shape, encoded = tensor_bytes(chunk, name)
                    if shape[1:] != expected_shape[1:]:
                        raise ValueError(f"tensor changed while exporting: {name}")
                    if pack_keys and name.startswith("icl.") and name.endswith(".key"):
                        encoded = _pack_keys(encoded, shape)
                    if dtype == DTYPE_BF16:
                        encoded = _to_bf16(encoded)
                    file.write(encoded)
                    digest.update(encoded)
                    written += len(encoded)
                if written != expected_bytes:
                    raise ValueError(
                        f"tensor byte length changed while exporting: {name}"
                    )
                position = offset + expected_bytes
            if position != payload_length:
                raise ValueError(
                    "static context payload length changed while exporting"
                )
            file.write(digest.digest())
            file.flush()
            os.fsync(file.fileno())
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
    """Parse an ``IRONICL`` header and tensor directory without the Rust reader."""
    if (
        len(data) < HEADER_BYTES + 32
        or hashlib.sha256(data[:-32]).digest() != data[-32:]
    ):
        raise ValueError("invalid IRONICL checksum")
    fields = struct.unpack_from(HEADER_FORMAT, data)
    versions = (VERSION, BF16_VERSION, PACKED_VERSION)
    if fields[0] != STATIC_MAGIC or fields[1] not in versions or fields[2] != 1:
        raise ValueError("unsupported IRONICL format")
    model, context, b, n, h, c = fields[3:9]
    tensors, class_count, feature_count, directory_length, payload_length, reserved = (
        fields[14:20]
    )
    if reserved or class_count != b * c or feature_count != b * h:
        raise ValueError("invalid IRONICL metadata")
    directory_start = HEADER_BYTES + 8 * (class_count + feature_count)
    directory_end = directory_start + directory_length
    payload_start = (directory_end + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    if payload_start + payload_length != len(data) - 32:
        raise ValueError("invalid IRONICL sections")
    offset, result = directory_start, []
    for _ in range(tensors):
        name_len, rank, dtype, payload_offset, byte_len = struct.unpack_from(
            "<IHHQQ", data, offset
        )
        offset += 24
        shape = struct.unpack_from(f"<{rank}Q", data, offset)
        offset += rank * 8
        name = data[offset : offset + name_len].decode("utf-8")
        offset += name_len
        element_bytes = 4 if fields[1] == VERSION or dtype == DTYPE_FP32 else 2
        valid_dtype = (
            dtype == 0 if fields[1] == VERSION else dtype in (DTYPE_FP32, DTYPE_BF16)
        )
        if (
            not name_len
            or not rank
            or not valid_dtype
            or np.prod(shape, dtype=np.uint64) * element_bytes != byte_len
            or payload_offset + byte_len > payload_length
        ):
            raise ValueError("invalid IRONICL tensor")
        result.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": "bf16" if element_bytes == 2 else "fp32",
            }
        )
    if offset != directory_end:
        raise ValueError("invalid IRONICL directory")
    return {
        "version": fields[1],
        "model_sha256": model.hex(),
        "context_sha256": context.hex(),
        "B": b,
        "N": n,
        "H": h,
        "C": c,
        "tensors": result,
    }

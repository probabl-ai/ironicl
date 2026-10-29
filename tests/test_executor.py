"""Always-on tests of the compiled module and the static-context encoder."""

from __future__ import annotations

import numpy as np
import pytest

import ironicl
from ironicl import IronICLError, StaticExecutor, _context
from ironicl._ironicl import build_context, inspect_model, prepare_model

MODEL_HASH = "11" * 32
CONTEXT_HASH = "22" * 32


def test_version_is_shared_with_the_extension():
    from ironicl import _ironicl

    assert isinstance(ironicl.__version__, str)
    assert ironicl.__version__ == _ironicl.__version__


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(IronICLError):
        StaticExecutor(tmp_path / "missing.ironicl", tmp_path / "missing.ironiclctx")
    with pytest.raises(IronICLError):
        StaticExecutor.from_bytes(tmp_path / "missing.ironicl", b"not a context")
    with pytest.raises(IronICLError):
        prepare_model(tmp_path / "missing.ckpt", tmp_path / "cache")
    with pytest.raises(IronICLError):
        inspect_model(tmp_path / "missing.ironicl")


def test_build_context_validates_inputs_before_loading_the_model(tmp_path):
    x = np.zeros((2, 4, 3), np.float32)
    y = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], np.float32)
    classes = np.array([[0, 1], [1, 0]], np.uint64)
    features = np.array([[0, 1, 2], [2, 1, 0]], np.uint64)
    missing = tmp_path / "missing.ironicl"
    output = tmp_path / "context.ironiclctx"
    with pytest.raises(ValueError, match=r"\[B, N\]"):
        build_context(missing, x, y[:1], classes, features, b"\x01" * 32, output)
    with pytest.raises(ValueError, match="32 bytes"):
        build_context(missing, x, y, classes, features, b"\x01" * 4, output)
    with pytest.raises(ValueError, match="dtype"):
        build_context(
            missing, x, y, classes, features, b"\x01" * 32, output, kv_dtype="int8"
        )
    with pytest.raises(ValueError, match="permutation"):
        build_context(
            missing, x, y, np.zeros((2, 2), np.uint64), features, b"\x01" * 32, output
        )
    with pytest.raises(IronICLError):
        build_context(missing, x, y, classes, features, b"\x01" * 32, output)
    assert not output.exists()


def _cache_tensors(b: int = 1, h: int = 5, n: int = 4) -> list[tuple[str, np.ndarray]]:
    tensors = []
    for layer in range(_context.COLUMN_LAYERS):
        tensors.append(
            (f"column.{layer}.key", np.zeros((b, h + 4, 8, 128, 16), np.float32))
        )
        tensors.append(
            (f"column.{layer}.value", np.ones((b, h + 4, 8, 128, 16), np.float32))
        )
    for layer in range(_context.ICL_LAYERS):
        tensors.append((f"icl.{layer}.key", np.zeros((b, 8, n, 64), np.float32)))
        tensors.append((f"icl.{layer}.value", np.ones((b, 8, n, 64), np.float32)))
    return tensors


def _permutations(b: int = 1, c: int = 3, h: int = 5) -> tuple[np.ndarray, np.ndarray]:
    classes = np.tile(np.arange(c, dtype=np.uint64), (b, 1))
    features = np.tile(np.arange(h, dtype=np.uint64), (b, 1))
    return classes, features


@pytest.mark.parametrize(
    ("kv_dtype", "pack_keys", "version"),
    [("fp32", False, 1), ("bf16", False, 2), ("bf16-v", False, 2), ("fp32", True, 3)],
)
def test_static_context_roundtrip(tmp_path, kv_dtype, pack_keys, version):
    path = tmp_path / "context.ironiclctx"
    classes, features = _permutations()
    info = _context.write_static_context(
        path,
        MODEL_HASH,
        CONTEXT_HASH,
        classes,
        features,
        _cache_tensors(),
        kv_dtype=kv_dtype,
        pack_keys=pack_keys,
    )
    parsed = _context.parse_static_context(path.read_bytes())
    assert info["static_context_version"] == version == parsed["version"]
    assert (parsed["B"], parsed["N"], parsed["H"], parsed["C"]) == (1, 4, 5, 3)
    assert parsed["model_sha256"] == MODEL_HASH
    assert parsed["context_sha256"] == CONTEXT_HASH
    assert len(parsed["tensors"]) == 2 * (_context.COLUMN_LAYERS + _context.ICL_LAYERS)
    dtypes = {tensor["name"]: tensor["dtype"] for tensor in parsed["tensors"]}
    assert dtypes["column.0.key"] == "fp32"
    expected_value = "bf16" if kv_dtype in ("bf16", "bf16-v") else "fp32"
    assert dtypes["icl.0.value"] == expected_value
    assert info["context_file_bytes"] == path.stat().st_size


def test_static_context_with_sparse_index(tmp_path):
    tensors = _cache_tensors()
    icl_keys = [
        np.random.default_rng(layer).standard_normal((1, 8, 4, 64), np.float32)
        for layer in range(_context.ICL_LAYERS)
    ]
    tensors.extend(_context.sparse_index_tensors(icl_keys, clusters=2))
    classes, features = _permutations()
    path = tmp_path / "context.ironiclctx"
    _context.write_static_context(
        path, MODEL_HASH, CONTEXT_HASH, classes, features, tensors
    )
    parsed = _context.parse_static_context(path.read_bytes())
    names = {tensor["name"] for tensor in parsed["tensors"]}
    assert set(_context.SPARSE_INDEX_NAMES) <= names


def test_static_context_rejects_bad_permutations(tmp_path):
    classes, features = _permutations()
    classes[0, 0] = classes[0, 1]
    with pytest.raises(ValueError, match="class permutations"):
        _context.write_static_context(
            tmp_path / "x",
            MODEL_HASH,
            CONTEXT_HASH,
            classes,
            features,
            _cache_tensors(),
        )


def test_context_identity_is_canonical():
    x = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([0, 1, 0])
    first = _context.context_identity(x, y, {"b": 1, "a": 2})
    second = _context.context_identity(
        x.astype(np.float32), y.astype(np.int32), {"a": 2, "b": 1}
    )
    assert first == second
    assert first != _context.context_identity(x, y, {"a": 3, "b": 1})

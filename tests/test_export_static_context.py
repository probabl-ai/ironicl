"""Exporter container tests that run without torch or TabICL."""
import importlib.util
import hashlib
import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np


path = pathlib.Path(__file__).parents[1] / "tools" / "export_static_context.py"
if not path.is_file():
    path = pathlib.Path("/home/semyon/Projects/ironicl-bench/export_static_context.py")
spec = importlib.util.spec_from_file_location("export_static_context", path)
assert spec and spec.loader
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportFormatTest(unittest.TestCase):
    def test_static_context_and_query_roundtrip(self):
        original = exporter.tensor_bytes
        exporter.tensor_bytes = lambda values, _name: (values.shape, np.ascontiguousarray(values, dtype="<f4").tobytes())
        try:
            tensors = []
            for prefix, layers, shape in (("column", range(3), (1, 5, 8, 128, 16)), ("icl", range(12), (1, 8, 2, 64))):
                for layer in layers:
                    tensors.extend(((f"{prefix}.{layer}.key", np.zeros(shape)), (f"{prefix}.{layer}.value", np.ones(shape))))
            context = exporter.encode_static_context("01" * 32, "02" * 32, np.array([[0, 1]], dtype=np.uint64), np.array([[0]], dtype=np.uint64), tensors)
        finally:
            exporter.tensor_bytes = original
        parsed = exporter.parse_static_context(context)
        self.assertEqual((parsed["B"], parsed["N"], parsed["H"], parsed["C"]), (1, 2, 1, 2))
        self.assertEqual(len(parsed["tensors"]), 30)
        query = exporter.encode_query("01" * 32, "02" * 32, np.zeros((1, 3, 1), dtype=np.float32))
        self.assertEqual(exporter.parse_query(query)["Q"], 3)
        with self.assertRaises(ValueError):
            exporter.parse_static_context(context[:-1] + b"x")

    def test_context_identity_is_stable(self):
        x, y = np.array([[1.0], [2.0]]), np.array([0, 1])
        self.assertEqual(exporter.context_identity(x, y, {"a": 1}), exporter.context_identity(x, y, {"a": 1}))

    def test_fixture_source_arrays_retain_raw_values_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = pathlib.Path(directory) / "fixture.npz"
            np.savez(
                fixture,
                x_context=np.array([[1.0], [2.0]], dtype=np.float32),
                y_context=np.array([0, 1], dtype=np.int64),
                x_query=np.array([[3.0]], dtype=np.float32),
            )
            fixture.with_suffix(".npz.json").write_text(
                json.dumps(
                    {
                        "benchmark_contract": "raw_x_cached_serving_v1",
                        "fixture_sha256": "fixture",
                        "context_indices_sha256": "context-indices",
                        "query_indices_sha256": "query-indices",
                        "x_context_sha256": "x-context",
                        "y_context_sha256": "y-context",
                        "x_query_sha256": "x-query",
                    }
                )
            )
            args = type("Arguments", (), {"fixture": fixture, "dataset": None, "x_context": None, "y_context": None, "x_query": None})()
            x_context, y_context, x_query, source = exporter.source_arrays(args)
            self.assertEqual(x_context.dtype, np.float32)
            self.assertEqual(y_context.dtype, np.int64)
            self.assertEqual(x_query.dtype, np.float32)
            self.assertEqual(source["fixture_sha256"], "fixture")

    def test_streamed_context_matches_in_memory_encoding(self):
        original = exporter.tensor_bytes
        exporter.tensor_bytes = lambda values, _name: (values.shape, np.ascontiguousarray(values, dtype="<f4").tobytes())
        try:
            tensors = []
            for prefix, layers, shape in (("column", range(3), (1, 5, 8, 128, 16)), ("icl", range(12), (1, 8, 2, 64))):
                for layer in layers:
                    tensors.extend(((f"{prefix}.{layer}.key", np.zeros(shape)), (f"{prefix}.{layer}.value", np.ones(shape))))
            classes = np.array([[0, 1]], dtype=np.uint64)
            features = np.array([[0]], dtype=np.uint64)
            expected = exporter.encode_static_context("01" * 32, "02" * 32, classes, features, tensors)
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "context.ironiclctx"
                info = exporter.write_static_context(path, "01" * 32, "02" * 32, classes, features, tensors)
                self.assertEqual(path.read_bytes(), expected)
                self.assertEqual(info["context_file_bytes"], len(expected))
                bf16_path = pathlib.Path(directory) / "context-bf16.ironiclctx"
                bf16_info = exporter.write_static_context(
                    bf16_path, "01" * 32, "02" * 32, classes, features, tensors, "bf16"
                )
                parsed = exporter.parse_static_context(bf16_path.read_bytes())
                self.assertEqual(parsed["version"], 2)
                self.assertEqual({item["dtype"] for item in parsed["tensors"] if item["name"].startswith("icl.")}, {"bf16"})
                self.assertLess(bf16_info["context_file_bytes"], len(expected))

                combined = []
                chunked = []
                for name, values in tensors:
                    values = np.concatenate([values, values + 2.0])
                    combined.append((name, values))
                    chunked.append((name, [values[:1], values[1:]]))
                classes = np.array([[0, 1], [1, 0]], dtype=np.uint64)
                features = np.array([[0], [0]], dtype=np.uint64)
                expected = exporter.encode_static_context(
                    "01" * 32, "02" * 32, classes, features, combined
                )
                chunked_path = pathlib.Path(directory) / "context-chunked.ironiclctx"
                exporter.write_static_context(
                    chunked_path, "01" * 32, "02" * 32, classes, features, chunked
                )
                self.assertEqual(chunked_path.read_bytes(), expected)
        finally:
            exporter.tensor_bytes = original

    def test_streamed_context_preserves_optional_sparse_index(self):
        tensors = []
        for prefix, layers, shape in (("column", range(3), (1, 5, 8, 128, 16)), ("icl", range(12), (1, 8, 2, 64))):
            for layer in layers:
                tensors.extend(((f"{prefix}.{layer}.key", np.zeros(shape)), (f"{prefix}.{layer}.value", np.ones(shape))))
        tensors.extend(
            (
                ("icl.index.centroids", np.zeros((1, 12, 8, 2, 64), dtype=np.float32)),
                ("icl.index.radii", np.zeros((1, 12, 8, 2), dtype=np.float32)),
                ("icl.index.offsets", np.tile(np.array([0, 1, 2], dtype=np.float32), (1, 12, 8, 1))),
                ("icl.index.members", np.tile(np.array([0, 1], dtype=np.float32), (1, 12, 8, 1))),
            )
        )
        classes = np.array([[0, 1]], dtype=np.uint64)
        features = np.array([[0]], dtype=np.uint64)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "indexed.ironiclctx"
            exporter.write_static_context(path, "01" * 32, "02" * 32, classes, features, tensors, "bf16-v", True)
            parsed = exporter.parse_static_context(path.read_bytes())
        tensors = {item["name"]: item for item in parsed["tensors"]}
        self.assertEqual(len(tensors), 34)
        self.assertEqual(tensors["icl.index.centroids"]["dtype"], "fp32")
        self.assertEqual(tensors["icl.index.members"]["shape"], [1, 12, 8, 2])

    def test_model_source_identity(self):
        body = bytearray(144)
        body[:8] = b"IRMODEL\0"
        struct.pack_into("<II", body, 8, 1, 1)
        body[16:48] = bytes.fromhex("03" * 32)
        body.extend(bytes(32))
        body.extend(hashlib.sha256(body).digest())
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / "model.ironicl"
            model.write_bytes(body)
            self.assertEqual(exporter.model_source_sha256(model), "03" * 32)
            body[-1] ^= 1
            model.write_bytes(body)
            with self.assertRaises(ValueError):
                exporter.model_source_sha256(model)


if __name__ == "__main__":
    unittest.main()

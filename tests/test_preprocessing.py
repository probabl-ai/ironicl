"""Rust preprocessing through ``_core.preprocess``; TabICL and PyTorch are not required.

Rust's own tests check preprocessing against TabICL's frozen fixture. Explicit
live comparisons belong in tests/parity; see docs/validation.md for setup.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from ironicl import _core
from ironicl.base import _array, _features, _label_values, _labels

DEFAULTS = {"n_estimators": 8, "random_state": 0, "norm_methods": ["none", "power"], "outlier_threshold": 4.0}


def preprocess(x, y, **options):
    """Fit preprocessing as a classifier's ``fit`` does, on converted inputs."""
    return _core.preprocess(_features(x)[0], labels=_label_values(_labels(y)), **{**DEFAULTS, **options})


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(37)
        self.x = rng.normal(size=(40, 5))
        self.x[:, 1] = 7
        self.x[-1, 2] = 50
        self.y = np.array(["zebra", "ant", "yak"])[np.arange(40) % 3]
        self.q = rng.normal(size=(9, 5))

    def test_shapes_and_metadata(self):
        preprocessed = preprocess(self.x, self.y)
        self.assertEqual(preprocessed.shape, [8, 40, 4])
        self.assertEqual(preprocessed.feature_mask, [True, False, True, True, True])
        self.assertEqual(preprocessed.group_names, ["none", "power"])
        self.assertEqual(_array(*preprocessed.query_rows(self.q)).shape, (8, 9, 4))
        for permutations, width in ((preprocessed.feature_permutations, 4), (preprocessed.class_permutations, 3)):
            orders = _array(*permutations, np.uint64)
            assert_array_equal(np.sort(orders, axis=1), np.tile(np.arange(width), (8, 1)))

    def test_label_types_keep_their_order(self):
        for labels, expected in (
            (np.array([3, -1, 3, 8] * 5), [1, 0, 1, 2]),
            (np.array([2.0, -2.0, 2.0, 1e9] * 5), [1, 0, 1, 2]),
            (np.array([True, False] * 10), [1, 0]),
            (np.array(["é", "b", "a", "b"] * 5, dtype=object), [2, 1, 0, 1]),
        ):
            with self.subTest(dtype=labels.dtype):
                x = np.random.default_rng(1).normal(size=(len(labels), 3))
                # One member keeps the class order, so its targets are the class indices.
                _, targets, _, _ = preprocess(x, labels, n_estimators=1).training_inputs()
                assert_array_equal(_array(*targets)[0, : len(expected)], expected)
        with self.assertRaises(TypeError):
            _label_values(np.array(["a", 1] * 10, dtype=object))

    def test_requested_and_actual_ensemble_counts(self):
        x = np.arange(20).reshape(-1, 1)
        y = np.arange(20) % 2
        self.assertEqual(preprocess(x, y, n_estimators=32).shape, [4, 20, 1])
        single = preprocess(self.x, self.y, n_estimators=1)
        self.assertEqual(single.group_names, ["none"])
        assert_array_equal(_array(*single.feature_permutations, np.uint64), [np.arange(4)])
        assert_array_equal(_array(*single.class_permutations, np.uint64), [np.arange(3)])

    def test_dtypes_and_noncontiguous_inputs(self):
        for dtype in (np.float32, np.float64, np.int64):
            x = (self.x * 10).astype(dtype)
            with self.subTest(dtype=dtype):
                preprocessed = preprocess(x, self.y)
                view = x[::2]
                self.assertFalse(view.flags.c_contiguous)
                expected = _array(*preprocessed.query_rows(_features(view.copy())[0]))
                assert_array_equal(_array(*preprocessed.query_rows(_features(view)[0])), expected)

    def test_float32_input_is_its_float64_upcast(self):
        # Heavy tails, where computing in float32 would lose precision.
        rng = np.random.default_rng(5)
        x = np.exp(rng.normal(size=(60, 4)) * [0.5, 1.0, 2.0, 3.0]).astype(np.float32)
        q = np.exp(rng.normal(size=(9, 4)) * 2).astype(np.float32)
        y = np.arange(60) % 3
        for name, fitted in (
            ("labels", lambda x: preprocess(x, y)),
            ("values", lambda x: _core.preprocess(x, values=np.arange(60, dtype=np.float32), **DEFAULTS)),
        ):
            with self.subTest(name):
                narrow, wide = fitted(x), fitted(x.astype(np.float64))
                self.assertEqual(narrow.training_inputs(), wide.training_inputs())
                for query in (q, q.astype(np.float64)):
                    self.assertEqual(narrow.query_rows(query), wide.query_rows(q))

    def test_rows_preprocess_alike_alone_and_in_batches(self):
        preprocessed = preprocess(self.x, self.y)
        batched = _array(*preprocessed.query_rows(self.q))
        alone = np.concatenate([_array(*preprocessed.query_rows(row[None, :])) for row in self.q], axis=1)
        assert_array_equal(batched, alone)
        preprocessed.query_rows(self.q * 1000)  # Outside the training range, by the fallback.

    def test_invalid_data(self):
        for x in (self.x.astype(complex), np.full((40, 5), "word")):
            with self.subTest(dtype=x.dtype), self.assertRaises(TypeError):
                _features(x)
        for x in (np.ones((40, 5)), self.x[:1], np.full((40, 5), np.nan), np.full((40, 5), np.inf), np.empty((0, 5))):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                preprocess(x, self.y[: len(x)])
        for y in (self.y[:-1], np.zeros(40), np.arange(40) % 11, np.arange(40) * 0.1, np.full(40, np.nan)):
            with self.subTest(y=y), self.assertRaises(ValueError):
                preprocess(self.x, y)
        preprocessed = preprocess(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "features"):
            preprocessed.query_rows(self.q[:, :3])
        with self.assertRaises(ValueError):
            preprocessed.query_rows(np.full_like(self.q, np.nan))

    def test_sparse_inputs_are_rejected(self):
        from scipy.sparse import csr_matrix

        with self.assertRaises(TypeError):
            _features(csr_matrix(self.x))

    def test_parameter_validation(self):
        for options in (
            {"n_estimators": 0},
            {"n_estimators": 4097},
            {"norm_methods": []},
            {"norm_methods": ["none", "none"]},
            {"norm_methods": ["quantile"]},
            {"outlier_threshold": 0.0},
            {"outlier_threshold": float("nan")},
            {"outlier_threshold": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                preprocess(self.x, self.y, **options)

    def test_does_not_import_torch_or_tabicl(self):
        script = """
import importlib.abc, sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "tabicl"}:
            raise AssertionError("Unexpected dependency: " + fullname)
sys.meta_path.insert(0, BlockImports())
import numpy as np
from ironicl import _core
x = np.arange(120, dtype=np.float64).reshape(40, 3)
_core.preprocess(x, labels=("int", np.arange(40) % 3), n_estimators=8, random_state=0,
                 norm_methods=["none", "power"], outlier_threshold=4.0).query_rows(x[:2])
"""
        subprocess.run([sys.executable, "-B", "-c", script], check=True)

    def test_real_dataset_export_matches_rust_preprocessing(self):
        path = Path(__file__).parents[1] / "tools" / "prepare_real_dataset.py"
        spec = importlib.util.spec_from_file_location("prepare_real_dataset", path)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.irinput"
            argv = [str(path), "--dataset", "covertype", "--train-rows", "25", "--test-rows", "9",
                    "--estimators", "8", "--output", str(output)]  # fmt: skip
            with (
                patch.object(sys, "argv", argv),
                patch.object(exporter, "load_dataset", return_value=(self.x, self.y, "fixture")),
                patch("builtins.print"),
            ):
                exporter.main()
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["normalization_methods"], ["none", "power"])
            self.assertEqual(metadata["shape"], [8, 34, 4])
            encoded = output.read_bytes()
        header = struct.unpack_from("<8sIIQQQQQQQQ", encoded)
        self.assertEqual(header[:3], (b"IRINPUT\0", 1, 1))
        x_train, x_test, y_train, _ = exporter.stratified_rows(self.x, self.y, 25, 9, 42)
        preprocessed = preprocess(x_train, y_train, random_state=42)
        x_values = np.frombuffer(encoded, dtype="<f4", count=header[7], offset=80).reshape(8, 34, 4)
        y_values = np.frombuffer(encoded, dtype="<f4", count=header[8], offset=80 + header[9]).reshape(8, 25)
        rows, targets, _, _ = preprocessed.training_inputs()
        assert_array_equal(x_values[:, :25], _array(*rows))
        assert_array_equal(x_values[:, 25:], _array(*preprocessed.query_rows(x_test)))
        assert_array_equal(y_values, _array(*targets))


if __name__ == "__main__":
    unittest.main()

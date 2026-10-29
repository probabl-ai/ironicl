"""Standalone preprocessing tests; TabICL/PyTorch are not required.

The checked-in oracle always runs without upstream dependencies. Explicit live
comparisons belong in tests/parity; see docs/validation.md for setup.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ironicl.preprocessing import ClassifierPreprocessor, _feature_shuffles

FIXTURE = Path(__file__).parent / "fixtures" / "preprocessing.json"


class NamedArray:
    """Exercise the named-array contract without a pandas test dependency."""

    def __init__(self, values, columns):
        self.values = np.asarray(values)
        self.columns = columns

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.values, dtype=dtype)
        return result.copy() if copy else result


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(37)
        self.x = rng.normal(size=(40, 5))
        self.x[:, 1] = 7
        self.x[-1, 2] = 50
        self.y = np.array(["zebra", "ant", "yak"])[np.arange(40) % 3]
        self.q = rng.normal(size=(9, 5))

    def test_checked_in_upstream_oracle(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture["upstream_commit"], "bd030d47ff7d9f987f0f33431d743a3003bfe0d3")
        for case in fixture["cases"]:
            with self.subTest(dtype=case["dtype"]):
                x = np.asarray(fixture["training"], dtype=case["dtype"])
                q = np.asarray(fixture["query"], dtype=case["dtype"])
                preprocessor = ClassifierPreprocessor(random_state=42).fit(x, fixture["labels"])
                self.assertEqual(list(preprocessor.group_names_), case["group_names"])
                for attribute in (
                    "classes_",
                    "feature_mask_",
                    "ensemble_groups_",
                    "feature_permutations_",
                    "class_permutations_",
                ):
                    assert_array_equal(getattr(preprocessor, attribute), case[attribute])
                assert_array_equal(preprocessor.encode_labels(fixture["labels"]), case["encoded_labels"])
                for actual, expected in zip(preprocessor.transform(x), case["training_groups"], strict=True):
                    assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                for actual, expected in zip(preprocessor.transform(q), case["query_groups"], strict=True):
                    assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_grouped_layout_and_label_metadata(self):
        preprocessor = ClassifierPreprocessor().fit(self.x, self.y)
        groups = preprocessor.transform(self.q)
        self.assertEqual(preprocessor.n_features_in_, 5)
        self.assertEqual(preprocessor.n_features_out_, 4)
        self.assertEqual(preprocessor.n_estimators_, 8)
        self.assertEqual(preprocessor.group_names_, ("none", "power"))
        self.assertEqual(sum(group.nbytes for group in groups), 2 * 9 * 4 * 4)
        for group in groups:
            self.assertEqual(group.shape, (9, 4))
            self.assertEqual(group.dtype, np.float32)
            self.assertTrue(group.flags.c_contiguous)
            self.assertFalse(np.shares_memory(group, self.q))
        for permutations, width in (
            (preprocessor.feature_permutations_, 4),
            (preprocessor.class_permutations_, 3),
        ):
            self.assertEqual(permutations.shape, (8, width))
            self.assertEqual(permutations.dtype, np.uint64)
            self.assertFalse(permutations.flags.writeable)
            assert_array_equal(np.sort(permutations, axis=1), np.tile(np.arange(width), (8, 1)))
        encoded = preprocessor.encode_labels(self.y)
        self.assertEqual(encoded.dtype, np.int64)
        self.assertEqual(encoded.shape, (len(self.y),))
        assert_array_equal(preprocessor.decode_labels(encoded), self.y)
        with self.assertRaises(ValueError):
            preprocessor.encode_labels(["unknown"])
        for invalid in ([0.5], [-1], [3], [[0]]):
            with self.subTest(indices=invalid), self.assertRaises(ValueError):
                preprocessor.decode_labels(invalid)

    def test_requested_and_actual_ensemble_counts(self):
        x = np.arange(20).reshape(-1, 1)
        y = np.arange(20) % 2
        preprocessor = ClassifierPreprocessor(n_estimators=32).fit(x, y)
        self.assertEqual(preprocessor.n_estimators, 32)
        self.assertEqual(preprocessor.n_estimators_, 4)
        self.assertEqual(len(preprocessor.transform(x)), 2)
        single = ClassifierPreprocessor(n_estimators=1).fit(self.x, self.y)
        self.assertEqual(single.group_names_, ("none",))
        assert_array_equal(single.feature_permutations_, [np.arange(4)])
        assert_array_equal(single.class_permutations_, [np.arange(3)])

    def test_finite_numeric_dtypes_and_noncontiguous_inputs(self):
        for dtype in (np.float32, np.float64, np.int64):
            x = (self.x * 10).astype(dtype)
            with self.subTest(dtype=dtype):
                preprocessor = ClassifierPreprocessor().fit(x, self.y)
                view = x[::2]
                self.assertFalse(view.flags.c_contiguous)
                for actual, expected in zip(
                    preprocessor.transform(view), preprocessor.transform(view.copy()), strict=True
                ):
                    assert_array_equal(actual, expected)

    def test_training_only_statistics_and_batching(self):
        preprocessor = ClassifierPreprocessor().fit(self.x, self.y)
        original_x, original_q = self.x.copy(), self.q.copy()
        before = pickle.dumps(preprocessor)
        batched = preprocessor.transform(self.q)
        individual = [preprocessor.transform(row[None, :]) for row in self.q]
        for index, group in enumerate(batched):
            assert_allclose(group, np.concatenate([row[index] for row in individual]), rtol=2e-6, atol=2e-6)
        preprocessor.transform(self.q * 1000)
        self.assertEqual(pickle.dumps(preprocessor), before)
        assert_array_equal(self.x, original_x)
        assert_array_equal(self.q, original_q)
        restored = pickle.loads(before)
        for actual, expected in zip(restored.transform(self.q), batched, strict=True):
            assert_array_equal(actual, expected)

    def test_no_training_arrays_retained(self):
        preprocessor = ClassifierPreprocessor().fit(self.x, self.y)
        self.assertFalse(hasattr(preprocessor, "X_"))
        self.assertFalse(hasattr(preprocessor, "y_"))
        for normalizer in preprocessor.normalizers_:
            for value in vars(normalizer).values():
                if isinstance(value, np.ndarray):
                    self.assertEqual(value.shape, (4,))
                    self.assertFalse(np.shares_memory(value, self.x))
            if normalizer.power is not None:
                self.assertFalse(hasattr(normalizer.power, "X_transformed_"))

    def test_named_features_and_refitting(self):
        names = ["a", "b", "c", "d", "e"]
        preprocessor = ClassifierPreprocessor().fit(NamedArray(self.x, names), self.y)
        preprocessor.transform(NamedArray(self.q, names))
        for invalid in (self.q, NamedArray(self.q, names[::-1])):
            with self.assertRaisesRegex(ValueError, "feature names"):
                preprocessor.transform(invalid)
        for invalid_names in (["a"] * 5, list(range(5))):
            with self.assertRaises(ValueError):
                ClassifierPreprocessor().fit(NamedArray(self.x, invalid_names), self.y)
        preprocessor.fit(self.x, self.y)
        self.assertFalse(hasattr(preprocessor, "feature_names_in_"))
        preprocessor.transform(self.q)

    def test_invalid_data_and_unfitted_use(self):
        with self.assertRaises(NotFittedError):
            ClassifierPreprocessor().transform(self.q)
        with self.assertRaises(NotFittedError):
            ClassifierPreprocessor().encode_labels(self.y)
        invalid_x = (
            np.ones((40, 5)),
            self.x[:, 0],
            self.x[:1],
            np.empty((0, 5)),
            np.empty((40, 0)),
            self.x.astype(complex),
            np.full((40, 5), "word"),
            np.full((40, 5), np.nan),
            np.full((40, 5), np.inf),
        )
        for x in invalid_x:
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises((ValueError, TypeError)):
                ClassifierPreprocessor().fit(x, self.y)
        for y in (
            self.y[:-1],
            self.y[:, None],
            np.zeros(40),
            np.arange(40) % 11,
            np.arange(40) * 0.1,
            np.full(40, np.nan),
        ):
            with self.subTest(y=y), self.assertRaises(ValueError):
                ClassifierPreprocessor().fit(self.x, y)
        preprocessor = ClassifierPreprocessor().fit(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "features"):
            preprocessor.transform(self.q[:, :3])
        for invalid in (np.full_like(self.q, np.nan), np.full_like(self.q, np.inf)):
            with self.assertRaises(ValueError):
                preprocessor.transform(invalid)

    def test_sparse_inputs_are_rejected(self):
        from scipy.sparse import csr_matrix

        with self.assertRaises(TypeError):
            ClassifierPreprocessor().fit(csr_matrix(self.x), self.y)

    def test_parameter_validation_and_sklearn_clone(self):
        invalid = (
            {"n_estimators": 0},
            {"n_estimators": 4097},
            {"n_estimators": 1.5},
            {"n_estimators": True},
            {"random_state": None},
            {"random_state": True},
            {"random_state": 0.5},
            {"norm_methods": ()},
            {"norm_methods": "none"},
            {"norm_methods": ("none", "none")},
            {"norm_methods": ("quantile",)},
            {"outlier_threshold": 0},
            {"outlier_threshold": np.nan},
            {"outlier_threshold": np.inf},
            {"outlier_threshold": "four"},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                ClassifierPreprocessor(**parameters).fit(self.x, self.y)
        fitted = ClassifierPreprocessor(n_estimators=3, norm_methods=("power",)).fit(self.x, self.y)
        copied = clone(fitted)
        self.assertEqual(copied.get_params(), fitted.get_params())
        self.assertFalse(hasattr(copied, "normalizers_"))

    def test_failed_fit_preserves_previous_state(self):
        preprocessor = ClassifierPreprocessor().fit(self.x, self.y)
        before = pickle.dumps(preprocessor)
        with self.assertRaises(ValueError):
            preprocessor.fit(np.ones_like(self.x), self.y)
        self.assertEqual(pickle.dumps(preprocessor), before)

    def test_iterative_latin_square_does_not_change_recursion_limit(self):
        limit = sys.getrecursionlimit()
        permutations = _feature_shuffles(limit + 1, 8, 0) if limit < 2000 else _feature_shuffles(1100, 8, 0)
        self.assertEqual(sys.getrecursionlimit(), limit)
        count = len(permutations)
        self.assertEqual(len(permutations[0]), count)
        assert_array_equal(np.sort(permutations, axis=0), np.tile(np.arange(count)[:, None], (1, count)))

    def test_large_feature_shuffle_fallback(self):
        permutations = _feature_shuffles(4001, 3, 0)
        self.assertEqual(len(permutations), 3)
        for permutation in permutations:
            self.assertEqual(sorted(permutation), list(range(4001)))

    def test_hash_seed_independence_and_no_torch_import(self):
        script = """
import importlib.abc, json, sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "tabicl"}:
            raise AssertionError("Unexpected dependency: " + fullname)
sys.meta_path.insert(0, BlockImports())
import numpy as np
from ironicl.preprocessing import ClassifierPreprocessor
x = np.arange(120, dtype=np.float64).reshape(40, 3)
p = ClassifierPreprocessor().fit(x, np.arange(40) % 3)
print(json.dumps({"groups": p.group_names_, "indices": p.ensemble_groups_.tolist(),
                  "features": p.feature_permutations_.tolist(), "classes": p.class_permutations_.tolist(),
                  "data": [group.tolist() for group in p.transform(x[:2])]}))
assert not any(name.split(".")[0] in {"torch", "tabicl"} for name in sys.modules)
"""
        results = []
        for seed in ("0", "1", "42"):
            results.append(
                subprocess.check_output(
                    [sys.executable, "-B", "-c", script],
                    text=True,
                    env={**os.environ, "PYTHONHASHSEED": seed},
                )
            )
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_real_dataset_export_uses_standalone_preprocessing(self):
        path = Path(__file__).parents[1] / "tools" / "prepare_real_dataset.py"
        spec = importlib.util.spec_from_file_location("prepare_real_dataset", path)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.irinput"
            argv = [
                str(path),
                "--dataset",
                "covertype",
                "--train-rows",
                "25",
                "--test-rows",
                "9",
                "--estimators",
                "8",
                "--output",
                str(output),
            ]
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
        p = ClassifierPreprocessor(random_state=42).fit(x_train, y_train)
        train, query = p.transform(x_train), p.transform(x_test)
        x_values = np.frombuffer(encoded, dtype="<f4", count=header[7], offset=80).reshape(8, 34, 4)
        y_values = np.frombuffer(encoded, dtype="<f4", count=header[8], offset=80 + header[9]).reshape(8, 25)
        for ensemble, group in enumerate(p.ensemble_groups_):
            assert_array_equal(x_values[ensemble, :25], train[group][:, p.feature_permutations_[ensemble]])
            assert_array_equal(x_values[ensemble, 25:], query[group][:, p.feature_permutations_[ensemble]])
            assert_array_equal(y_values[ensemble], p.class_permutations_[ensemble][p.encode_labels(y_train)])


if __name__ == "__main__":
    unittest.main()

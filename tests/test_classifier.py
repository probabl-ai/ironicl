"""Parameter validation, n_jobs resolution, and fitted attributes for TabICLClassifier.

None of these needs a checkpoint: fit runs against a stand-in for the Rust model.
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from ironicl import TabICLClassifier
from ironicl.cache import _cache_dir


class ValidateParamsTest(unittest.TestCase):
    def test_unsupported_values_raise(self):
        cases = {
            "feat_shuffle_method": "random",
            "class_shuffle_method": "latin",
            "device": "cuda",
            "inference_config": {"COL_CONFIG": {}},
            "use_amp": True,
            "use_fa3": True,
            "offload_mode": "cpu",
            "disk_offload_dir": "/tmp/offload",
            "kv_cache": "repr",
            "checkpoint_version": "tabicl-classifier-v1-20250208.ckpt",
            "norm_methods": ["quantile"],
            "random_state": None,
        }
        for param, value in cases.items():
            with self.subTest(param=param), self.assertRaises(NotImplementedError):
                TabICLClassifier(**{param: value})._validate_params()

    def test_supported_values_pass(self):
        TabICLClassifier()._validate_params()
        TabICLClassifier(use_amp=False, use_fa3=False, kv_cache="kv", norm_methods="power")._validate_params()

    def test_noop_values_warn(self):
        for param, value in (("batch_size", 16), ("support_many_classes", False)):
            with self.subTest(param=param), self.assertWarns(UserWarning):
                TabICLClassifier(**{param: value})._validate_params()

    def test_defaults_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TabICLClassifier()._validate_params()


class ResolveThreadsTest(unittest.TestCase):
    def test_none_keeps_default(self):
        self.assertIsNone(TabICLClassifier(n_jobs=None)._resolve_threads())

    def test_positive_is_literal(self):
        self.assertEqual(TabICLClassifier(n_jobs=3)._resolve_threads(), 3)

    def test_negative_follows_tabicl(self):
        cores = os.cpu_count() or 1
        self.assertEqual(TabICLClassifier(n_jobs=-1)._resolve_threads(), max(1, cores))
        self.assertEqual(TabICLClassifier(n_jobs=-(cores * 2))._resolve_threads(), 1)


class CacheDirTest(unittest.TestCase):
    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {"IRONICL_CACHE_DIR": "/data/ctx"}):
            self.assertIsNone(_cache_dir(TabICLClassifier().cache_dir))  # default False ignores the env

    def test_false_disables(self):
        self.assertIsNone(_cache_dir(TabICLClassifier(cache_dir=False).cache_dir))

    def test_explicit_path(self):
        self.assertEqual(_cache_dir(TabICLClassifier(cache_dir="/tmp/ctx").cache_dir), Path("/tmp/ctx"))

    def test_none_uses_ironicl_cache_dir(self):
        with mock.patch.dict(os.environ, {"IRONICL_CACHE_DIR": "/data/ctx"}):
            self.assertEqual(_cache_dir(TabICLClassifier(cache_dir=None).cache_dir), Path("/data/ctx"))

    def test_none_falls_back_to_xdg(self):
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/xdg"}):
            os.environ.pop("IRONICL_CACHE_DIR", None)
            self.assertEqual(_cache_dir(TabICLClassifier(cache_dir=None).cache_dir), Path("/xdg/ironicl/contexts"))


class NamedArray:
    """Exercise the named-array contract without a pandas test dependency."""

    def __init__(self, values, columns):
        self.values = np.asarray(values)
        self.columns = columns

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.values, dtype=dtype)
        return result.copy() if copy else result


class FakeModel:
    """Stands in for the Rust model, so fit runs preprocessing without a checkpoint."""

    task = "classifier"

    def build(self, *args, **kwargs):
        return object()


class FittedAttributesTest(unittest.TestCase):
    def test_a_refit_without_names_drops_the_previous_names(self):
        x = np.random.default_rng(0).normal(size=(40, 3))
        y = np.arange(40) % 2
        with tempfile.NamedTemporaryFile() as checkpoint, mock.patch("ironicl.base._model", return_value=FakeModel()):
            model = TabICLClassifier(model_path=checkpoint.name)
            model.fit(NamedArray(x, ["a", "b", "c"]), y)
            self.assertEqual(list(model.feature_names_in_), ["a", "b", "c"])
            model.fit(x[:, :2], y)
            self.assertFalse(hasattr(model, "feature_names_in_"))
            self.assertEqual(model.n_features_in_, 2)

    def test_queries_must_match_the_fitted_features(self):
        x = np.random.default_rng(0).normal(size=(40, 3))
        y = np.array(["b", "a"])[np.arange(40) % 2]
        with tempfile.NamedTemporaryFile() as checkpoint, mock.patch("ironicl.base._model", return_value=FakeModel()):
            model = TabICLClassifier(model_path=checkpoint.name).fit(NamedArray(x, ["a", "b", "c"]), y)
        self.assertEqual(list(model.classes_), ["a", "b"])
        self.assertEqual(model._query(NamedArray(x[:2], ["a", "b", "c"])).shape, (2, 3))
        for invalid, message in (
            (x[:2], "feature names"),
            (NamedArray(x[:2], ["c", "b", "a"]), "feature names"),
            (x[:2, :2], "features"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                model._query(invalid)
        with self.assertRaises(TypeError):
            model._query(np.full((2, 3), "word"))

    def test_parameter_types_are_checked_before_rust(self):
        for name, value in (
            ("n_estimators", 1.5),
            ("n_estimators", True),
            ("random_state", 0.5),
            ("random_state", True),
            ("outlier_threshold", "four"),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                TabICLClassifier(**{name: value})._validate_params()


if __name__ == "__main__":
    unittest.main()

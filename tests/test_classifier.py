"""Parameter validation and n_jobs resolution for TabICLClassifier.

These exercise ``_validate_params`` and ``_resolve_threads`` directly, so they
need neither a checkpoint nor a fitted model.
"""

import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()

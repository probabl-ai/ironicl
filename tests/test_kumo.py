"""KumoTabularClassifier and KumoTabularRegressor: validation, and with the
small checkpoints in the default cache, which ``allow_auto_download`` fills,
fitting, predicting, pickling, and exporting. Parity with NVIDIA's code is
tested in Rust."""

import os
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ironicl import KumoTabularClassifier, KumoTabularRegressor

try:
    import pandas as pd
except ImportError:
    pd = None

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ironicl" / "checkpoints"
CHECKPOINT = CACHE / "kumo-tabular" / "small-classifier.pt"
NEEDS_CHECKPOINT = f"needs {CHECKPOINT}"
REGRESSOR = CACHE / "kumo-tabular" / "small-regressor.pt"


class ValidateTest(unittest.TestCase):
    def test_invalid_values_raise(self):
        for params in ({"size": "xl"}, {"n_estimators": 0}, {"n_estimators": 2.0}, {"random_state": None}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                KumoTabularClassifier(**params).fit(np.zeros((2, 1)), [0, 1])

    def test_default_estimators_follow_the_size(self):
        self.assertEqual(KumoTabularClassifier()._estimators(), 16)
        self.assertEqual(KumoTabularClassifier(size="small")._estimators(), 8)
        self.assertEqual(KumoTabularClassifier(n_estimators=3)._estimators(), 3)


@unittest.skipUnless(CHECKPOINT.exists(), NEEDS_CHECKPOINT)
@unittest.skipIf(pd is None, "needs pandas")
class FitTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = pd.DataFrame(
            {
                "a": np.where(rng.random(90) < 0.1, np.nan, rng.standard_normal(90)),
                "color": pd.Series(rng.choice(["red", "blue", None], 90), dtype=object),
            }
        )
        self.y = np.where(self.x["a"].fillna(0) > 0, "yes", "no")
        self.model = KumoTabularClassifier(size="small", n_estimators=3).fit(self.x[:70], self.y[:70])

    def test_predicts_probabilities_of_the_classes(self):
        probabilities = self.model.predict_proba(self.x[70:])
        self.assertEqual(probabilities.shape, (20, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5)
        self.assertEqual(list(self.model.classes_), ["no", "yes"])
        self.assertGreater((self.model.predict(self.x[70:]) == self.y[70:]).mean(), 0.8)

    def test_a_seed_gives_the_same_model(self):
        again = KumoTabularClassifier(size="small", n_estimators=3).fit(self.x[:70], self.y[:70])
        np.testing.assert_array_equal(again.predict_proba(self.x[70:]), self.model.predict_proba(self.x[70:]))
        other = KumoTabularClassifier(size="small", n_estimators=3, random_state=7).fit(self.x[:70], self.y[:70])
        self.assertFalse(np.array_equal(other.predict_proba(self.x[70:]), self.model.predict_proba(self.x[70:])))

    def test_pickling_fits_the_same_model(self):
        loaded = pickle.loads(pickle.dumps(self.model))
        np.testing.assert_array_equal(loaded.predict_proba(self.x[70:]), self.model.predict_proba(self.x[70:]))

    def test_save_and_load_give_the_same_model(self):
        expected = self.model.predict_proba(self.x[70:])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            for options in ({}, {"save_kv_cache": False}, {"save_model_weights": True}):
                with self.subTest(options=options):
                    self.model.save(path, **options)
                    loaded = KumoTabularClassifier.load(path)
                    np.testing.assert_array_equal(loaded.predict_proba(self.x[70:]), expected)
            with self.assertRaises(ValueError):
                self.model.save(path, save_training_data=False)

    def test_saved_weights_restore_a_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "small-classifier.pt"
            checkpoint.write_bytes(CHECKPOINT.read_bytes())
            model = KumoTabularClassifier(size="small", n_estimators=2, model_path=checkpoint, allow_auto_download=False)
            model.fit(self.x[:70], self.y[:70])
            data = pickle.dumps(model)
            saved = Path(directory) / "with-weights.pkl"
            model.save(saved, save_model_weights=True)
            checkpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                pickle.loads(data)
            loaded = KumoTabularClassifier.load(saved)
            np.testing.assert_array_equal(loaded.predict_proba(self.x[70:]), model.predict_proba(self.x[70:]))
            self.assertTrue(checkpoint.exists())

    def test_training_data_without_a_varying_feature_raises(self):
        with self.assertRaisesRegex(ValueError, "no feature varies"):
            KumoTabularClassifier(size="small", n_estimators=2).fit(np.ones((10, 2)), [0, 1] * 5)

    @unittest.skipUnless(os.environ.get("IRONICL_BIN"), "needs IRONICL_BIN built with --features server")
    def test_prepare_and_serve_reject_unsupported_memory_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.export(directory)
            limits = ("IRONICL_MEMORY_LIMIT", "IRONICL_DEVICE_MEMORY_LIMIT")
            env = {key: value for key, value in os.environ.items() if key not in limits}
            for command in ("prepare", "serve"):
                for variable in limits:
                    with self.subTest(command=command, variable=variable):
                        result = subprocess.run(
                            [os.environ["IRONICL_BIN"], command, directory],
                            env={**env, variable: "1MiB"},
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Kumo-Tabular does not support memory limits", result.stderr)
                        self.assertIn(variable, result.stderr)

    def test_columns_must_match_training(self):
        with self.assertRaises(ValueError):
            self.model.predict_proba(self.x[70:][["color", "a"]])

    @unittest.skipUnless(os.environ.get("IRONICL_BIN"), "needs IRONICL_BIN, an ironicl built with --features server")
    def test_export_writes_a_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.export(Path(directory) / "model")
            command = [os.environ["IRONICL_BIN"], "inspect", str(Path(directory) / "model")]
            output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
            self.assertIn("KV cache:      safe", output)
            self.model.export(Path(directory) / "refits", kv_cache=False)
            self.assertFalse((Path(directory) / "refits" / "context.ircntxt").exists())


@unittest.skipUnless(REGRESSOR.exists(), f"needs {REGRESSOR}")
class RegressorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.x = rng.standard_normal((90, 3))
        self.y = 10 + 3 * self.x[:, 0] - self.x[:, 1] + 0.1 * rng.standard_normal(90)
        self.model = KumoTabularRegressor(size="small", n_estimators=3).fit(self.x[:70], self.y[:70])

    def test_predicts_in_the_targets_units(self):
        mean = self.model.predict(self.x[70:])
        self.assertEqual(mean.shape, (20,))
        self.assertLess(np.abs(mean - self.y[70:]).mean(), 1.0)
        outputs = self.model.predict(self.x[70:], output_type=["median", "quantiles", "raw_quantiles"], alphas=[0.1, 0.9])
        self.assertEqual(outputs["quantiles"].shape, (20, 2))
        self.assertTrue((outputs["quantiles"][:, 0] <= outputs["quantiles"][:, 1]).all())
        raw = outputs["raw_quantiles"]
        self.assertEqual(raw.shape, (20, 999))
        np.testing.assert_allclose(raw.mean(axis=1), mean, rtol=1e-5)

    def test_pickling_fits_the_same_model(self):
        loaded = pickle.loads(pickle.dumps(self.model))
        np.testing.assert_array_equal(loaded.predict(self.x[70:]), self.model.predict(self.x[70:]))

    def test_a_column_vector_y_fits_the_same_model(self):
        with self.assertWarns(Warning):
            column = KumoTabularRegressor(size="small", n_estimators=3).fit(self.x[:70], self.y[:70, None])
        np.testing.assert_array_equal(column.predict(self.x[70:]), self.model.predict(self.x[70:]))


if __name__ == "__main__":
    unittest.main()

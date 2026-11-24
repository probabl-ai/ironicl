"""KumoTabularClassifier: validation, and with the small checkpoint in the
default cache, which ``allow_auto_download`` fills, fitting, predicting,
pickling, and exporting. Parity with NVIDIA's code is tested in Rust."""

import os
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ironicl import KumoTabularClassifier
from ironicl.kumo import REVISION

try:
    import pandas as pd
except ImportError:
    pd = None

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ironicl" / "checkpoints"
CHECKPOINT = CACHE / "kumo-tabular" / REVISION / "small-classifier.pt"
NEEDS_CHECKPOINT = f"needs {CHECKPOINT}"


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

    def test_columns_must_match_training(self):
        with self.assertRaises(ValueError):
            self.model.predict_proba(self.x[70:][["color", "a"]])

    @unittest.skipUnless(os.environ.get("IRONICL_BIN"), "needs IRONICL_BIN, an ironicl built with --features server")
    def test_export_writes_a_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.export(Path(directory) / "model")
            command = [os.environ["IRONICL_BIN"], "inspect", str(Path(directory) / "model")]
            output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
            self.assertIn("Kumo-Tabular fits again when loaded", output)


if __name__ == "__main__":
    unittest.main()

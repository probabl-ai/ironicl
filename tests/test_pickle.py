"""Pickling fitted estimators. Needs the checkpoints in the default cache, which
``allow_auto_download`` fills; each estimator is skipped without its own."""

import copy
import os
import pickle
import unittest
from pathlib import Path

import numpy as np

from ironicl import TabICLClassifier, TabICLRegressor

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ironicl" / "checkpoints"


def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6))
    return x, x[:, 0] + 0.5 * x[:, 1]


class PickleTest(unittest.TestCase):
    def check(self, estimator, y, predict):
        model = estimator(allow_auto_download=False, cache_dir=False)
        if not (CACHE / model.checkpoint_version).exists():
            self.skipTest(f"{model.checkpoint_version} is not in {CACHE}")
        x, _ = data()
        model.fit(x, y)
        expected = predict(model, x[:50])
        for name, restored in (
            ("pickle", pickle.loads(pickle.dumps(model))),
            ("deepcopy", copy.deepcopy(model)),
        ):
            with self.subTest(name):
                np.testing.assert_array_equal(predict(restored, x[:50]), expected)
                self.assertIsNot(restored.context_, model.context_)

    def test_classifier_round_trips(self):
        _, y = data()
        self.check(TabICLClassifier, (y > 0).astype(int), TabICLClassifier.predict_proba)

    def test_regressor_round_trips(self):
        _, y = data()
        self.check(TabICLRegressor, y, TabICLRegressor.predict)

    def test_unfitted_estimator_pickles(self):
        model = pickle.loads(pickle.dumps(TabICLClassifier(n_estimators=4)))
        self.assertEqual(model.n_estimators, 4)
        self.assertFalse(hasattr(model, "context_"))


if __name__ == "__main__":
    unittest.main()

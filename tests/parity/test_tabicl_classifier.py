"""Live classifier parity against a pinned, unmodified TabICL checkout.

Setup matches test_tabicl_preprocessing.py. The released classifier checkpoint
is downloaded once through upstream's Hugging Face cache and shared by both.
"""

from __future__ import annotations

import importlib
import inspect
import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from test_tabicl_preprocessing import load_upstream

from ironicl import TabICLClassifier

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


class TabICLClassifierParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_upstream()  # Verifies the pinned checkout and imports the tabicl package.
        cls.upstream = importlib.import_module("tabicl._sklearn.classifier").TabICLClassifier
        hub = importlib.import_module("huggingface_hub")
        cls.checkpoint = hub.hf_hub_download("jingang/TabICL", CHECKPOINT)

    def test_public_signatures_match(self):
        # TabICL's parameters in order, then Ironicl's own trailing keywords.
        extras = {"__init__": [("optimize", None)]}
        for name in ("__init__", "fit", "predict_proba", "predict"):
            ours = inspect.signature(getattr(TabICLClassifier, name)).parameters.values()
            theirs = inspect.signature(getattr(self.upstream, name)).parameters.values()
            self.assertEqual(
                [(p.name, p.default) for p in ours],
                [(p.name, p.default) for p in theirs] + extras.get(name, []),
                name,
            )

    def test_predictions_match(self):
        cases = [
            (load_iris, 96, 32, {}),
            (load_wine, 96, 32, {"n_estimators": 3, "average_logits": False, "softmax_temperature": 0.5}),
            (load_breast_cancer, 96, 32, {"norm_methods": "power", "random_state": 7}),
            # Large enough for Ironicl's Metal ICL path on Apple Silicon.
            (load_breast_cancer, 400, 169, {}),
        ]
        for load, train_size, test_size, params in cases:
            with self.subTest(dataset=load.__name__, train_size=train_size, **params):
                x, y = load(return_X_y=True)
                x_train, x_test, y_train, _ = train_test_split(
                    x, y, train_size=train_size, test_size=test_size, random_state=0, stratify=y
                )
                ours = TabICLClassifier(model_path=self.checkpoint, **params).fit(x_train, y_train)
                theirs = self.upstream(model_path=self.checkpoint, device="cpu", **params)
                theirs.fit(x_train, y_train)
                np.testing.assert_allclose(
                    ours.predict_proba(x_test), theirs.predict_proba(x_test), rtol=0, atol=1e-4
                )
                np.testing.assert_array_equal(ours.predict(x_test), theirs.predict(x_test))
                np.testing.assert_array_equal(ours.classes_, theirs.classes_)

    def test_failed_fit_is_not_fitted(self):
        clf = TabICLClassifier(model_path="/nonexistent/model.ckpt", allow_auto_download=False)
        with self.assertRaises(FileNotFoundError):
            clf.fit([[0.0], [1.0]], [0, 1])
        with self.assertRaises(NotFittedError):
            clf.predict([[0.0]])


    def test_optimize_levels_stay_within_their_bounds(self):
        # Reduced levels against "safe" (M1 Max measurements on 12 datasets:
        # max 4.9e-4, mean 2.3e-5). Labels may differ only on near-ties. Without
        # a Metal GPU every level is "safe" and the bounds hold trivially.
        bounds = {"auto": (1e-3, 1e-4), "fast": (1e-2, 1e-3)}
        x, y = load_breast_cancer(return_X_y=True)
        x_train, x_test, y_train, _ = train_test_split(
            x, y, train_size=400, test_size=169, random_state=0, stratify=y
        )
        safe = TabICLClassifier(model_path=self.checkpoint, optimize="safe").fit(x_train, y_train)
        expected = safe.predict_proba(x_test)
        for optimize, (max_error, mean_error) in bounds.items():
            with self.subTest(optimize=optimize):
                clf = TabICLClassifier(model_path=self.checkpoint, optimize=optimize)
                actual = clf.fit(x_train, y_train).predict_proba(x_test)
                error = np.abs(actual - expected)
                self.assertLessEqual(error.max(), max_error)
                self.assertLessEqual(error.mean(), mean_error)
                top = np.sort(expected, axis=1)
                near_tie = top[:, -1] - top[:, -2] < 2 * max_error
                flipped = actual.argmax(1) != expected.argmax(1)
                self.assertFalse((flipped & ~near_tie).any())

    def test_optimize_rejects_unknown_levels(self):
        with self.assertRaisesRegex(ValueError, "optimize"):
            TabICLClassifier(model_path=self.checkpoint, optimize="turbo").fit([[0.0], [1.0]], [0, 1])


if __name__ == "__main__":
    unittest.main()

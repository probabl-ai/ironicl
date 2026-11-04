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
        for name in ("__init__", "fit", "predict_proba", "predict"):
            ours = inspect.signature(getattr(TabICLClassifier, name)).parameters.values()
            theirs = inspect.signature(getattr(self.upstream, name)).parameters.values()
            self.assertEqual(
                [(p.name, p.default) for p in ours], [(p.name, p.default) for p in theirs], name
            )

    def test_predictions_match(self):
        cases = [
            (load_iris, {}),
            (load_wine, {"n_estimators": 3, "average_logits": False, "softmax_temperature": 0.5}),
            (load_breast_cancer, {"norm_methods": "power", "random_state": 7}),
        ]
        for load, params in cases:
            with self.subTest(dataset=load.__name__, **params):
                x, y = load(return_X_y=True)
                x_train, x_test, y_train, _ = train_test_split(
                    x, y, train_size=96, test_size=32, random_state=0, stratify=y
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


if __name__ == "__main__":
    unittest.main()

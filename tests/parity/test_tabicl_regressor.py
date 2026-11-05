"""Live regressor parity against a pinned, unmodified TabICL checkout.

Setup matches test_tabicl_preprocessing.py. The released regressor checkpoint
is downloaded once through TabICL's Hugging Face cache and shared by both.
"""

from __future__ import annotations

import importlib
import inspect
import unittest

import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from test_tabicl_preprocessing import load_upstream

from ironicl import TabICLRegressor

CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
OUTPUT_TYPES = ["mean", "median", "variance", "quantiles", "raw_quantiles"]
# Inside and outside the outermost decoded levels, 0.001 and 0.999.
ALPHAS = [0.0005, 0.01, 0.1, 0.5, 0.9, 0.99, 0.9995]


class TabICLRegressorParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_upstream()  # Verifies the pinned checkout and imports the tabicl package.
        cls.tabicl = importlib.import_module("tabicl._sklearn.regressor").TabICLRegressor
        hub = importlib.import_module("huggingface_hub")
        cls.checkpoint = hub.hf_hub_download("jingang/TabICL", CHECKPOINT)

    def test_public_signatures_match(self):
        # TabICL's parameters in order, then Ironicl's own trailing keywords.
        extras = {"__init__": [("optimize", None), ("cache_dir", False)]}
        for name in ("__init__", "fit", "predict"):
            ours = inspect.signature(getattr(TabICLRegressor, name)).parameters.values()
            theirs = inspect.signature(getattr(self.tabicl, name)).parameters.values()
            self.assertEqual(
                [(p.name, p.default) for p in ours],
                [(p.name, p.default) for p in theirs] + extras.get(name, []),
                name,
            )

    def test_predictions_match(self):
        x, y = make_regression(n_samples=700, n_features=12, noise=10.0, random_state=0)
        cases = [
            (*load_diabetes(return_X_y=True), 200, 100, {}),
            # A shifted target: a location error shows up as a large difference.
            (x, y + 1000.0, 200, 100, {"n_estimators": 3, "norm_methods": "power", "random_state": 7}),
            # Large enough for Ironicl's Metal ICL path on Apple Silicon.
            (x, y, 400, 200, {}),
        ]
        for x, y, train_size, test_size, params in cases:
            with self.subTest(train_size=train_size, **params):
                x_train, x_test, y_train, _ = train_test_split(
                    x, y, train_size=train_size, test_size=test_size, random_state=0
                )
                ours = TabICLRegressor(model_path=self.checkpoint, **params).fit(x_train, y_train)
                theirs = self.tabicl(model_path=self.checkpoint, device="cpu", **params).fit(x_train, y_train)
                actual = ours.predict(x_test, output_type=OUTPUT_TYPES, alphas=ALPHAS)
                expected = theirs.predict(x_test, output_type=OUTPUT_TYPES, alphas=ALPHAS)
                # TabICL scales the variance back like a location, σ·v + μ;
                # Ironicl uses σ²·v (soda-inria/tabicl#180). Undo TabICL's.
                mean, scale = theirs.y_scaler_.mean_[0], theirs.y_scaler_.scale_[0]
                expected["variance"] = (expected["variance"] - mean) * scale
                for name in OUTPUT_TYPES:
                    # Within 1e-4 of the target's standard deviation (its variance for "variance").
                    unit = scale**2 if name == "variance" else scale
                    self.assertEqual(actual[name].shape, expected[name].shape, name)
                    np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-4 * unit, err_msg=name)
                np.testing.assert_allclose(
                    ours.predict(x_test), theirs.predict(x_test), rtol=0, atol=1e-4 * scale
                )

    def test_optimize_levels_stay_within_their_bounds(self):
        # Reduced levels against "safe", in target standard deviations (M1 Max:
        # max 2.8e-4, mean 5.2e-5 on diabetes and a synthetic set). Without a
        # Metal GPU or AVX512-BF16 every level is "safe" and the bounds hold trivially.
        bounds = {"auto": (1e-3, 1e-4), "fast": (1e-2, 1e-3)}
        x, y = make_regression(n_samples=600, n_features=12, noise=10.0, random_state=1)
        x_train, x_test, y_train, _ = train_test_split(x, y, train_size=400, random_state=0)
        types = ["mean", "quantiles"]
        safe = TabICLRegressor(model_path=self.checkpoint, optimize="safe").fit(x_train, y_train)
        expected = safe.predict(x_test, output_type=types)
        for optimize, (max_error, mean_error) in bounds.items():
            with self.subTest(optimize=optimize):
                reg = TabICLRegressor(model_path=self.checkpoint, optimize=optimize).fit(x_train, y_train)
                actual = reg.predict(x_test, output_type=types)
                for name in types:
                    error = np.abs(actual[name] - expected[name]) / y_train.std()
                    self.assertLessEqual(error.max(), max_error, name)
                    self.assertLessEqual(error.mean(), mean_error, name)

    def test_wrong_checkpoint_is_rejected(self):
        classifier = importlib.import_module("huggingface_hub").hf_hub_download(
            "jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt"
        )
        reg = TabICLRegressor(model_path=classifier)
        with self.assertRaisesRegex(ValueError, "classifier checkpoint"):
            reg.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 3.0])
        with self.assertRaises(NotFittedError):
            reg.predict([[0.0]])


if __name__ == "__main__":
    unittest.main()

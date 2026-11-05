"""Parameter validation for TabICLRegressor, without a checkpoint or fitted model."""

import unittest
import warnings

from sklearn.exceptions import NotFittedError

from ironicl import TabICLRegressor


class ValidateParamsTest(unittest.TestCase):
    def test_unsupported_values_raise(self):
        cases = {
            "feat_shuffle_method": "random",
            "device": "cuda",
            "use_amp": True,
            "kv_cache": "repr",
            "checkpoint_version": "tabicl-regressor-v1.ckpt",
            "norm_methods": ["quantile"],
            "random_state": None,
        }
        for param, value in cases.items():
            with self.subTest(param=param), self.assertRaises(NotImplementedError):
                TabICLRegressor(**{param: value})._validate_params()

    def test_defaults_pass_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TabICLRegressor()._validate_params()

    def test_noop_values_warn(self):
        for param, value in (("batch_size", 16), ("verbose", True)):
            with self.subTest(param=param), self.assertWarns(UserWarning):
                TabICLRegressor(**{param: value})._validate_params()

    def test_unfitted_predict_raises(self):
        with self.assertRaises(NotFittedError):
            TabICLRegressor().predict([[0.0]])


if __name__ == "__main__":
    unittest.main()

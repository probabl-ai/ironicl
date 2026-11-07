"""Memory limits, the overflow policy, and planning in fit.

Planning needs no checkpoint, and fit plans before loading one, so these run
without model files: a missing checkpoint shows that fit got past the plan.
"""

import os
import unittest
import warnings
from dataclasses import replace
from unittest import mock

import numpy as np
from sklearn.base import clone

import ironicl
from ironicl import IroniclMemoryError, IroniclMemoryWarning, MemoryPlan, TabICLClassifier, TabICLRegressor
from ironicl.memory import _enforce, _overflow_policy

GIB = 1 << 30
CLEAN_ENV = {"IRONICL_MEMORY_LIMIT": "", "IRONICL_DEVICE_MEMORY_LIMIT": "", "IRONICL_MEMORY_OVERFLOW": ""}


def unfittable(estimator=TabICLClassifier, **params):
    """An estimator whose checkpoint is missing: fit fails right after planning."""
    return estimator(model_path="/nonexistent/ironicl.ckpt", allow_auto_download=False, **params)


def data(rows=200):
    rng = np.random.default_rng(0)
    return rng.normal(size=(rows, 4)), np.arange(rows) % 2


class PlanMemoryTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, CLEAN_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_small_fit_fits(self):
        plan = ironicl.plan_memory(1_000, 10)
        self.assertIsInstance(plan, MemoryPlan)
        self.assertIn(plan.verdict, ("fits", "unknown"))
        self.assertGreater(plan.context_bytes, 0)
        self.assertGreater(plan.peak_bytes, plan.context_bytes + plan.weight_bytes)
        self.assertEqual(str(plan), plan.report)

    def test_limits_accept_bytes_and_sizes(self):
        self.assertEqual(ironicl.plan_memory(1_000, 10, memory_limit="16G").memory_limit, 16 * GIB)
        self.assertEqual(ironicl.plan_memory(1_000, 10, memory_limit="1.5GiB").memory_limit, 3 * GIB // 2)
        self.assertEqual(ironicl.plan_memory(1_000, 10, memory_limit=GIB).memory_limit, GIB)

    def test_a_limit_below_the_peak_overflows(self):
        plan = ironicl.plan_memory(1_000, 10, memory_limit="64M")
        self.assertEqual(plan.verdict, "overflow")
        self.assertLessEqual(plan.available_bytes, 64 << 20)
        self.assertIn("Memory limit: 0.1 GiB (set by memory_limit)", plan.report)

    def test_invalid_limits_raise(self):
        for value, error in (("lots", ValueError), ("0", ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(value=value), self.assertRaises(error):
                ironicl.plan_memory(1_000, 10, memory_limit=value)

    def test_the_environment_applies_unless_an_argument_is_given(self):
        with mock.patch.dict(os.environ, {"IRONICL_MEMORY_LIMIT": "2G"}):
            plan = ironicl.plan_memory(1_000, 10)
            self.assertEqual(plan.memory_limit, 2 * GIB)
            self.assertIn("IRONICL_MEMORY_LIMIT", plan.report)
            self.assertEqual(ironicl.plan_memory(1_000, 10, memory_limit="4G").memory_limit, 4 * GIB)
        with mock.patch.dict(os.environ, {"IRONICL_MEMORY_LIMIT": "lots"}):
            with self.assertRaisesRegex(ValueError, "IRONICL_MEMORY_LIMIT"):
                ironicl.plan_memory(1_000, 10)

    def test_device_memory_limit_without_device_memory_is_not_applicable(self):
        # Neither Apple Silicon nor a CPU-only machine has separate device memory.
        plan = ironicl.plan_memory(1_000, 10, device_memory_limit="8G")
        self.assertEqual(plan.device_memory_limit, 8 * GIB)
        self.assertIn("not applicable", plan.report)


class OverflowPolicyTest(unittest.TestCase):
    def test_precedence_and_default(self):
        with mock.patch.dict(os.environ, CLEAN_ENV):
            self.assertEqual(_overflow_policy(None), "warn")
        with mock.patch.dict(os.environ, {"IRONICL_MEMORY_OVERFLOW": "error"}):
            self.assertEqual(_overflow_policy(None), "error")
            self.assertEqual(_overflow_policy("ignore"), "ignore")
        with mock.patch.dict(os.environ, {"IRONICL_MEMORY_OVERFLOW": "crash"}):
            with self.assertRaisesRegex(ValueError, "IRONICL_MEMORY_OVERFLOW"):
                _overflow_policy(None)
        with self.assertRaisesRegex(ValueError, "memory_overflow"):
            _overflow_policy("raise")

    def test_each_verdict_under_each_policy(self):
        with mock.patch.dict(os.environ, CLEAN_ENV):
            plan = ironicl.plan_memory(1_000, 10)
        for verdict in ("fits", "unknown"):
            for policy in ("warn", "error"):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    _enforce(replace(plan, verdict=verdict), policy, stacklevel=1)
        overflow = replace(plan, verdict="overflow")
        with self.assertWarns(IroniclMemoryWarning):
            _enforce(overflow, "warn", stacklevel=1)
        with self.assertRaises(IroniclMemoryError):
            _enforce(overflow, "error", stacklevel=1)
        # Cannot complete raises under "warn" too; only "ignore" proceeds.
        cannot = replace(plan, verdict="cannot_complete")
        for policy in ("warn", "error"):
            with self.subTest(policy=policy), self.assertRaisesRegex(IroniclMemoryError, "cannot complete"):
                _enforce(cannot, policy, stacklevel=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _enforce(overflow, "ignore", stacklevel=1)
            _enforce(cannot, "ignore", stacklevel=1)

    def test_memory_error_is_a_memory_error(self):
        self.assertTrue(issubclass(IroniclMemoryError, MemoryError))
        self.assertTrue(issubclass(IroniclMemoryWarning, UserWarning))


class FitPlansMemoryTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, CLEAN_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_error_raises_before_loading_the_model_and_leaves_it_unfitted(self):
        for estimator in (TabICLClassifier, TabICLRegressor):
            model = unfittable(estimator, memory_limit="1M", memory_overflow="error")
            with self.subTest(estimator=estimator.__name__), self.assertRaises(IroniclMemoryError):
                model.fit(*data())
            self.assertFalse(hasattr(model, "context_"))
            self.assertFalse(hasattr(model, "memory_plan_"))

    def test_warn_continues_to_the_build(self):
        model = unfittable(memory_limit="1M")
        with self.assertWarns(IroniclMemoryWarning) as caught, self.assertRaises(FileNotFoundError):
            model.fit(*data())
        self.assertEqual(caught.filename, __file__)  # Points at the caller's fit.

    def test_ignore_and_fitting_plans_are_silent(self):
        for params in ({"memory_limit": "1M", "memory_overflow": "ignore"}, {}):
            with self.subTest(params=params), warnings.catch_warnings():
                warnings.simplefilter("error", IroniclMemoryWarning)
                with self.assertRaises(FileNotFoundError):
                    unfittable(**params).fit(*data())

    def test_the_environment_sets_the_policy(self):
        with mock.patch.dict(os.environ, {"IRONICL_MEMORY_LIMIT": "1M", "IRONICL_MEMORY_OVERFLOW": "error"}):
            with self.assertRaises(IroniclMemoryError):
                unfittable().fit(*data())

    def test_invalid_values_fail_fit(self):
        for params, error in (
            ({"memory_overflow": "crash"}, ValueError),
            ({"memory_limit": "lots"}, ValueError),
            ({"device_memory_limit": -1}, ValueError),
            ({"memory_limit": 2.5}, TypeError),
        ):
            with self.subTest(params=params), self.assertRaises(error):
                unfittable(**params).fit(*data())

    def test_parameters_clone(self):
        model = TabICLClassifier(memory_limit="16G", device_memory_limit=8 * GIB, memory_overflow="error")
        params = clone(model).get_params()
        self.assertEqual(
            (params["memory_limit"], params["device_memory_limit"], params["memory_overflow"]),
            ("16G", 8 * GIB, "error"),
        )


if __name__ == "__main__":
    unittest.main()

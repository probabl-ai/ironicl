"""CUDA builds and predictions: the context built on the GPU, the ICL cache
streamed from host memory, and rows only the CPU path predicts. Skipped
without a CUDA GPU."""

import pickle
import unittest

import numpy as np

from ironicl import TabICLClassifier, TabICLRegressor
from ironicl.base import _ensemble_rows, _model
from tests.test_pickle import checkpoint


def classification(rows):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(rows, 8))
    return x, (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)


class CudaTest(unittest.TestCase):
    def setUp(self):
        checkpoint(TabICLClassifier)
        self.x, self.y = classification(3000)

    def fit(self, **params):
        params = {"optimize": "auto", **params}
        model = TabICLClassifier(cache_dir=False, **params).fit(self.x[:2000], self.y[:2000])
        if not model.context_.device.startswith("cuda"):
            self.skipTest("no CUDA GPU")
        return model

    def test_a_gpu_built_context_predicts_like_a_cpu_built_one(self):
        expected = TabICLClassifier(optimize="safe", cache_dir=False).fit(self.x[:2000], self.y[:2000])
        expected = expected.predict_proba(self.x[2000:])
        actual = self.fit().predict_proba(self.x[2000:])
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_all_missing_training_rows_build_on_the_cpu(self):
        model = self.fit()
        preprocessor = model.preprocessor_
        rows = _ensemble_rows(preprocessor, model._training["groups"])
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        # Training rows whose every feature is the all-missing marker; only
        # the CPU build handles it.
        rows[:, [3, 70], :] = -100.0
        targets = model._expand_targets(preprocessor, model._training["targets"])
        arguments = (
            rows,
            np.ascontiguousarray(targets, dtype=np.float32),
            np.ascontiguousarray(preprocessor.class_permutations_, dtype=np.uint64),
            np.ascontiguousarray(preprocessor.feature_permutations_, dtype=np.uint64),
        )
        cpu, gpu = (
            _model(model.checkpoint_path_).build_context(*arguments, optimize, None) for optimize in ("safe", "auto")
        )
        self.assertEqual(gpu.device, "cuda")
        queries = _ensemble_rows(preprocessor, preprocessor.transform(self.x[2000:]))
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        def proba(context):
            values, shape = context.predict_proba(queries, 0.9, True)
            return np.frombuffer(values, dtype=np.float32).reshape(shape)

        np.testing.assert_allclose(proba(gpu), proba(cpu), atol=1e-3)

    def test_a_gpu_built_regressor_predicts_like_a_cpu_built_one(self):
        checkpoint(TabICLRegressor)
        target = self.x[:, 0] + 0.5 * self.x[:, 1]
        cpu, gpu = (
            TabICLRegressor(optimize=optimize, cache_dir=False).fit(self.x[:2000], target[:2000])
            for optimize in ("safe", "auto")
        )
        if not gpu.context_.device.startswith("cuda"):
            self.skipTest("no CUDA GPU")
        np.testing.assert_allclose(gpu.predict(self.x[2000:]), cpu.predict(self.x[2000:]), atol=1e-3)

    def test_a_kept_cache_predicts_like_an_uploaded_one(self):
        # The build leaves the cache on the device, encoded there; a pickled
        # copy uploads the cache the build wrote to the host, encoded there.
        for optimize in ("auto", "fast"):
            with self.subTest(optimize=optimize):
                model = self.fit(optimize=optimize)
                copy = pickle.loads(pickle.dumps(model))
                self.assertEqual(copy.context_.device, "cuda")
                np.testing.assert_allclose(
                    copy.predict_proba(self.x[2000:]), model.predict_proba(self.x[2000:]), atol=1e-6
                )

    def test_a_streamed_cache_predicts_like_a_resident_one(self):
        # The 2,000-row cache takes about 400 MB on the device in FP16, one
        # block of it 34 MB, and half that in FP8: these limits leave room to
        # stream it but not to keep it. The same context loads under each.
        # Shorter runs change which products run in which precision, which
        # moves `fast` by up to its own error.
        cases = (("auto", ("350M", "450M", "550M"), 1e-3), ("fast", ("300M", "350M", "400M"), 3e-2))
        for optimize, limits, tolerance in cases:
            with self.subTest(optimize=optimize):
                resident = self.fit(optimize=optimize)
                self.assertEqual(resident.context_.device, "cuda")
                self.assertIsNone(resident.context_.batch_rows)
                expected = resident.predict_proba(self.x[2000:])
                streamed = 0
                for limit in limits:
                    model = pickle.loads(pickle.dumps(resident.set_params(device_memory_limit=limit)))
                    if model.context_.device != "cuda-streamed":
                        continue
                    streamed += 1
                    self.assertGreater(model.context_.batch_rows, 0)
                    actual = model.predict_proba(self.x[2000:])
                    np.testing.assert_allclose(actual, expected, atol=tolerance)
                    self.assertGreaterEqual(np.mean(actual.argmax(1) == expected.argmax(1)), 0.99)
                self.assertGreater(streamed, 0)

    def test_all_missing_rows_match_the_cpu_path(self):
        model = self.fit()
        cpu = TabICLClassifier(optimize="safe", cache_dir=False).fit(self.x[:2000], self.y[:2000])
        rows = _ensemble_rows(model.preprocessor_, model.preprocessor_.transform(self.x[2000:2100]))
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        # A row whose every feature is the all-missing marker; only the CPU
        # column stage handles it.
        rows[:, [3, 70], :] = -100.0

        def proba(context):
            values, shape = context.predict_proba(rows, 0.9, True)
            return np.frombuffer(values, dtype=np.float32).reshape(shape)

        np.testing.assert_allclose(proba(model.context_), proba(cpu.context_), atol=1e-3)


if __name__ == "__main__":
    unittest.main()

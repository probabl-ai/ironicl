"""Pickling, saving, and loading fitted estimators. Needs the checkpoints in the
default cache, which ``allow_auto_download`` fills; each estimator is skipped
without its own."""

import copy
import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ironicl import IroniclMemoryError, IroniclMemoryWarning, TabICLClassifier, TabICLRegressor

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ironicl" / "checkpoints"


def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6))
    return x, x[:, 0] + 0.5 * x[:, 1]


def checkpoint(estimator) -> Path:
    path = CACHE / estimator().checkpoint_version
    if not path.exists():
        raise unittest.SkipTest(f"{path.name} is not in {CACHE}")
    return path


def fitted(estimator, **params):
    model = estimator(allow_auto_download=False, cache_dir=False, **params)
    checkpoint(estimator)
    x, y = data()
    return model.fit(x, (y > 0).astype(int) if estimator is TabICLClassifier else y)


def predict(model):
    x, _ = data()
    return model.predict_proba(x[:50]) if isinstance(model, TabICLClassifier) else model.predict(x[:50])


class PickleTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

    def test_pickle_and_deepcopy_round_trip(self):
        for estimator in (TabICLClassifier, TabICLRegressor):
            model = fitted(estimator)
            for name, restored in (
                ("pickle", pickle.loads(pickle.dumps(model))),
                ("deepcopy", copy.deepcopy(model)),
            ):
                with self.subTest(estimator=estimator.__name__, copy=name):
                    np.testing.assert_array_equal(predict(restored), predict(model))
                    self.assertIsNot(restored.context_, model.context_)

    def test_save_without_the_context_rebuilds_it(self):
        model = fitted(TabICLClassifier)
        full, small = self.directory / "full.pkl", self.directory / "small.pkl"
        model.save(full)
        model.save(small, save_kv_cache=False)
        self.assertLess(small.stat().st_size, full.stat().st_size / 100)
        for path in (full, small):
            with self.subTest(path.name):
                np.testing.assert_array_equal(predict(TabICLClassifier.load(path)), predict(model))

    def test_save_without_training_data_keeps_the_context(self):
        model = fitted(TabICLClassifier)
        path = self.directory / "model.pkl"
        model.save(path, save_training_data=False)
        loaded = TabICLClassifier.load(path)
        self.assertFalse(hasattr(loaded, "_training"))
        np.testing.assert_array_equal(predict(loaded), predict(model))
        with self.assertRaisesRegex(ValueError, "both"):
            model.save(path, save_training_data=False, save_kv_cache=False)

    def test_saved_model_weights_restore_a_missing_checkpoint(self):
        copy = self.directory / "checkpoint.ckpt"
        shutil.copyfile(checkpoint(TabICLClassifier), copy)
        model = fitted(TabICLClassifier, model_path=str(copy))
        path = self.directory / "model.pkl"
        model.save(path, save_model_weights=True)
        copy.unlink()
        loaded = TabICLClassifier.load(path)
        self.assertTrue(copy.exists())
        np.testing.assert_array_equal(predict(loaded), predict(model))

    def test_load_checks_the_class_device_and_version(self):
        model = fitted(TabICLClassifier)
        path = self.directory / "model.pkl"
        model.save(path)
        with self.assertRaises(TypeError):
            TabICLRegressor.load(path)
        with self.assertRaises(NotImplementedError):
            TabICLClassifier.load(path, device="cuda")
        self.assertEqual(TabICLClassifier.load(path, device="cpu").device, "cpu")
        state = model.__getstate__()
        state["_ironicl_version"] = "0.0.1"
        with self.assertWarnsRegex(UserWarning, "saved with Ironicl 0.0.1"):
            TabICLClassifier().__setstate__(state)

    def test_load_plans_its_memory(self):
        model = fitted(TabICLClassifier)
        self.assertIn("Peak while loading", str(pickle.loads(pickle.dumps(model)).memory_plan_))
        rebuilt = self.directory / "small.pkl"
        model.save(rebuilt, save_kv_cache=False)
        self.assertIn("Peak while building", str(TabICLClassifier.load(rebuilt).memory_plan_))
        model.memory_limit = "1M"
        with self.assertWarns(IroniclMemoryWarning):
            loaded = pickle.loads(pickle.dumps(model))
        self.assertEqual(loaded.memory_plan_.memory_limit, 1 << 20)
        # The limit shrinks the GPU's query pieces, and batch shape moves results by about 1e-6.
        np.testing.assert_allclose(predict(loaded), predict(model), atol=1e-5)
        model.memory_overflow = "error"
        with self.assertRaisesRegex(IroniclMemoryError, "load"):
            pickle.loads(pickle.dumps(model))

    def test_unfitted_estimator_pickles(self):
        model = pickle.loads(pickle.dumps(TabICLClassifier(n_estimators=4)))
        self.assertEqual(model.n_estimators, 4)
        self.assertFalse(hasattr(model, "context_"))


if __name__ == "__main__":
    unittest.main()

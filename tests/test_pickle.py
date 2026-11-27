"""Pickling, saving, and loading fitted estimators. Needs the checkpoints in the
default cache, which ``allow_auto_download`` fills; each estimator is skipped
without its own."""

import copy
import json
import os
import pickle
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ironicl import _core
from ironicl import IroniclMemoryError, IroniclMemoryWarning, TabICLClassifier, TabICLRegressor

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ironicl" / "checkpoints"


def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6))
    return x, x[:, 0] + 0.5 * x[:, 1]


def checkpoint(estimator) -> Path:
    path = CACHE / "tabicl" / estimator().checkpoint_version
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
                    self.assertIsNot(restored.fitted_, model.fitted_)

    def test_save_with_and_without_the_context_and_training_data(self):
        for estimator in (TabICLClassifier, TabICLRegressor):
            model = fitted(estimator)
            sizes = {}
            for training, context in ((True, True), (True, False), (False, True)):
                with self.subTest(estimator=estimator.__name__, training=training, context=context):
                    path = self.directory / f"{training}-{context}.pkl"
                    model.save(path, save_training_data=training, save_kv_cache=context)
                    if os.name == "posix":
                        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                    sizes[training, context] = path.stat().st_size
                    loaded = estimator.load(path)
                    self.assertEqual(loaded.fitted_.preprocessed.has_training, training)
                    # Without the context, loading rebuilds the same one.
                    np.testing.assert_array_equal(predict(loaded), predict(model))
                    np.testing.assert_array_equal(predict(pickle.loads(pickle.dumps(loaded))), predict(model))
            self.assertLess(sizes[True, False], sizes[True, True] / 100)
            with self.assertRaisesRegex(ValueError, "both"):
                model.save(self.directory / "neither.pkl", save_training_data=False, save_kv_cache=False)

    def test_context_from_another_fit_is_rejected_with_or_without_rows(self):
        ckpt = checkpoint(TabICLClassifier)
        x, y = data()

        def fit(labels):
            return TabICLClassifier(model_path=ckpt, n_estimators=1, cache_dir=False).fit(x, labels)

        a, b = fit(y > 0), fit(y <= 0)
        rust = _core.Model(ckpt)
        context = b.fitted_.context_bytes()
        for training in (False, True):
            with self.subTest(training=training), self.assertRaisesRegex(RuntimeError, "does not match its preprocessing"):
                rust.load_fitted(a.fitted_.preprocessor_bytes(training), "safe", None, context=context)

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
        self.assertFalse(hasattr(model, "fitted_"))

    def test_export_writes_a_model_directory(self):
        for estimator in (TabICLClassifier, TabICLRegressor):
            with self.subTest(estimator=estimator.__name__):
                model = fitted(estimator)
                path = self.directory / estimator.__name__
                model.export(path)
                files = ["context.ir", "ironicl.json", "preprocessing.ir"]
                self.assertEqual(sorted(file.name for file in path.iterdir()), files)
                metadata = json.loads((path / "ironicl.json").read_text())
                self.assertEqual(
                    (metadata["format"], metadata["version"], metadata["task"], metadata["n_features"]),
                    ("ironicl-model", 1, model._task, model.n_features_in_),
                )
                if os.name == "posix":  # The preprocessing and the cache hold the training data.
                    for name in files:
                        self.assertEqual((path / name).stat().st_mode & 0o777, 0o600, name)
                # Rust loads it, with the cache or, without it, from the training rows.
                rust = _core.Model(Path(model.checkpoint_path_))
                preprocessor = (path / "preprocessing.ir").read_bytes()
                self.assertEqual(preprocessor, model.fitted_.preprocessor_bytes())
                for context in ((path / "context.ir").read_bytes(), None):
                    loaded = rust.load_fitted(preprocessor, "safe", None, context=context)
                    self.assertEqual(loaded.preprocessor_bytes(), preprocessor)
                # Again, over the first export, without the cache.
                model.export(path, kv_cache=False)
                self.assertEqual(sorted(file.name for file in path.iterdir()), files[1:])

    def test_export_needs_the_training_data(self):
        saved = self.directory / "no-training.pkl"
        fitted(TabICLClassifier).save(saved, save_training_data=False)
        with self.assertRaisesRegex(ValueError, "save_training_data=True"):
            TabICLClassifier.load(saved).export(self.directory / "no-training")


if __name__ == "__main__":
    unittest.main()

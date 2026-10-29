"""The native preprocessing must reproduce tabicl's ensemble variants exactly.

``tests/data/<dataset>/preprocessing/`` holds, per normalisation method, the
training and query variants, labels and permutations that tabicl 2.2.0's fitted
``EnsembleGenerator`` produced for the same raw table and parameters (see
``tests/generate_goldens.py``). These tests need neither the checkpoint nor
tabicl.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ironicl import _preprocessing

DATA = Path(__file__).parent / "data"
DATASETS = sorted(
    path.name for path in DATA.iterdir() if (path / "preprocessing").is_dir()
)
PARAMS = (
    "n_estimators",
    "random_state",
    "norm_methods",
    "feat_shuffle_method",
    "class_shuffle_method",
    "outlier_threshold",
)


def _generator(folder: Path) -> tuple[_preprocessing.EnsembleGenerator, dict]:
    metadata = json.loads((folder / "metadata.json").read_text())
    params = {key: metadata[key] for key in PARAMS if key in metadata}
    params.setdefault("norm_methods", ["none", "power"])
    X_train = np.load(folder / "X_train.npy")
    y_train = np.load(folder / "y_train.npy")
    generator = _preprocessing.EnsembleGenerator(classification=True, **params)
    generator.fit(X_train, y_train.astype(np.int64))
    return generator, metadata


@pytest.mark.parametrize("dataset", DATASETS)
def test_training_variants_match_tabicl(dataset):
    folder = DATA / dataset
    generator, metadata = _generator(folder)
    assert sorted(generator.ensemble_configs_) == metadata["norm_methods_used"]
    np.testing.assert_array_equal(
        generator.unique_filter_.features_to_keep_,
        np.load(folder / "preprocessing" / "features_to_keep.npy"),
    )
    train = generator.transform(X=None, mode="train")
    for method, (X_variants, y_variants) in train.items():
        prefix = folder / "preprocessing"
        np.testing.assert_array_equal(
            np.asarray(generator.feature_shuffles_[method]),
            np.load(prefix / f"feature_shuffles.{method}.npy"),
        )
        np.testing.assert_array_equal(
            np.asarray(generator.class_shuffles_[method]),
            np.load(prefix / f"class_shuffles.{method}.npy"),
        )
        np.testing.assert_array_equal(
            y_variants, np.load(prefix / f"train_y.{method}.npy")
        )
        expected = np.load(prefix / f"train_X.{method}.npy")
        assert X_variants.dtype == expected.dtype
        np.testing.assert_allclose(X_variants, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("dataset", DATASETS)
def test_query_variants_match_tabicl(dataset):
    folder = DATA / dataset
    generator, _ = _generator(folder)
    X_test = np.load(folder / "X_test.npy")
    test = generator.transform(X_test, mode="test")
    for method, (X_variants,) in test.items():
        expected = np.load(folder / "preprocessing" / f"test_X.{method}.npy")
        np.testing.assert_allclose(X_variants, expected, rtol=0, atol=1e-6)
    variants = _preprocessing.query_variants(generator, X_test)
    assert variants.shape == (
        generator.n_estimators,
        len(X_test),
        generator.n_features_in_,
    )
    assert variants.dtype == np.float32


def test_training_variants_are_engine_ready():
    folder = DATA / "all_norms"
    generator, _ = _generator(folder)
    x, y, classes, features = _preprocessing.training_variants(generator)
    b, n, h = x.shape
    assert (b, n, h) == (
        generator.n_estimators,
        len(generator.y_),
        generator.n_features_in_,
    )
    assert x.dtype == np.float32 and x.flags.c_contiguous
    assert y.shape == (b, n) and y.dtype == np.float32
    assert classes.shape == (b, generator.n_classes_) and classes.dtype == np.uint64
    assert features.shape == (b, h) and features.dtype == np.uint64
    for permutation in classes:
        assert sorted(permutation) == list(range(generator.n_classes_))
    for permutation in features:
        assert sorted(permutation) == list(range(h))


def test_shuffler_methods_are_deterministic():
    for method, expected_count in (
        ("none", 1),
        ("shift", 4),
        ("random", 3),
        ("latin", 4),
    ):
        first = _preprocessing.Shuffler(4, method=method, random_state=7).shuffle(3)
        second = _preprocessing.Shuffler(4, method=method, random_state=7).shuffle(3)
        assert first == second
        assert len(first) == expected_count
        for permutation in first:
            assert sorted(permutation) == [0, 1, 2, 3]


def test_unique_feature_filter_keeps_at_least_one_feature():
    X = np.ones((10, 3), dtype=np.float32)
    kept = _preprocessing.UniqueFeatureFilter().fit(X).features_to_keep_
    np.testing.assert_array_equal(kept, [True, False, False])


def test_resolve_checkpoint_rejects_unknown_versions(tmp_path):
    with pytest.raises(ValueError, match="Invalid checkpoint version"):
        _preprocessing.resolve_checkpoint(None, "nope.ckpt", True)
    existing = tmp_path / "model.ckpt"
    existing.write_bytes(b"x")
    assert (
        _preprocessing.resolve_checkpoint(
            existing, "tabicl-classifier-v2-20260212.ckpt", False
        )
        == existing
    )

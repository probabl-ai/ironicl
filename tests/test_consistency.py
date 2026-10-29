"""Predictions must match the frozen goldens of the upstream tabicl classifier.

The goldens were produced once by ``tests/generate_goldens.py`` with tabicl 2.2.0
on CPU and live in ``tests/data/<dataset>/``. They are compared with a tolerance
because the engine evaluates the same FP32 arithmetic in a different order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ironicl import IronICLClassifier

DATA = Path(__file__).parent / "data"
DATASETS = sorted(
    path.name for path in DATA.iterdir() if (path / "metadata.json").is_file()
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("dataset", DATASETS)
def test_matches_tabicl_goldens(dataset, cache_dir, checkpoint_kwargs):
    folder = DATA / dataset
    metadata = json.loads((folder / "metadata.json").read_text())
    X_train = np.load(folder / "X_train.npy")
    y_train = np.load(folder / "y_train.npy")
    X_test = np.load(folder / "X_test.npy")
    expected = np.load(folder / "probabilities.npy")
    params = {
        key: metadata[key]
        for key in (
            "n_estimators",
            "random_state",
            "norm_methods",
            "feat_shuffle_method",
            "class_shuffle_method",
            "outlier_threshold",
        )
        if key in metadata
    }
    clf = IronICLClassifier(cache_dir=cache_dir, **params, **checkpoint_kwargs).fit(
        X_train, y_train
    )
    probabilities = clf.predict_proba(X_test)
    np.testing.assert_allclose(probabilities, expected, rtol=1e-3, atol=1e-4)
    assert (probabilities.argmax(axis=1) == expected.argmax(axis=1)).all()

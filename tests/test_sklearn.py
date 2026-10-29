"""scikit-learn compatibility of IronICLClassifier."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import parametrize_with_checks

from ironicl import IronICLClassifier


def test_init_and_get_params_are_cheap():
    estimator = IronICLClassifier(n_estimators=2, kv_dtype="bf16")
    params = estimator.get_params()
    assert params["n_estimators"] == 2
    assert params["kv_dtype"] == "bf16"
    assert params["checkpoint_version"].startswith("tabicl-classifier")


def test_invalid_parameters_fail_before_touching_the_model(tmp_path):
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    with pytest.raises(ValueError, match="kv_dtype"):
        IronICLClassifier(kv_dtype="int8", cache_dir=tmp_path).fit(X, y)
    with pytest.raises(ValueError, match="n_estimators"):
        IronICLClassifier(n_estimators=0, cache_dir=tmp_path).fit(X, y)
    with pytest.raises(ValueError, match="checkpoint version"):
        IronICLClassifier(
            checkpoint_version="tabicl-classifier-v9.ckpt", cache_dir=tmp_path
        ).fit(X, y)
    with pytest.raises(ValueError, match="automatic download is disabled"):
        IronICLClassifier(
            model_path=tmp_path / "missing.ckpt",
            allow_auto_download=False,
            cache_dir=tmp_path,
        ).fit(X, y)


def _checkpoint_kwargs() -> dict[str, str]:
    # Needed at collection time by parametrize_with_checks, so not a fixture.
    directory = os.environ.get("IRONICL_CHECKPOINT_DIR")
    if directory:
        return {
            "model_path": str(Path(directory) / "tabicl-classifier-v2-20260212.ckpt")
        }
    return {}


def _small_estimator(**kwargs) -> IronICLClassifier:
    return IronICLClassifier(n_estimators=2, **_checkpoint_kwargs(), **kwargs)


@pytest.mark.slow
@parametrize_with_checks([_small_estimator()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.slow
def test_predict_proba_rows_sum_to_one(cache_dir):
    X, y = make_classification(
        n_samples=96, n_features=6, n_informative=4, n_classes=3, random_state=0
    )
    clf = _small_estimator(cache_dir=cache_dir).fit(X[:64], y[:64])
    probabilities = clf.predict_proba(X[64:])
    assert probabilities.shape == (32, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert (clf.predict(X[64:]) == clf.classes_[probabilities.argmax(axis=1)]).all()


@pytest.mark.slow
def test_pickle_roundtrip_keeps_predictions(cache_dir):
    X, y = make_classification(n_samples=80, n_features=5, random_state=1)
    clf = _small_estimator(cache_dir=cache_dir).fit(X[:64], y[:64])
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(
        restored.predict_proba(X[64:]), clf.predict_proba(X[64:])
    )


@pytest.mark.slow
@pytest.mark.parametrize("kv_dtype", ["bf16", "bf16-v"])
def test_bf16_storage_keeps_labels(cache_dir, kv_dtype):
    X, y = make_classification(
        n_samples=96, n_features=6, n_informative=4, n_classes=3, random_state=2
    )
    exact = _small_estimator(cache_dir=cache_dir).fit(X[:64], y[:64])
    reduced = _small_estimator(cache_dir=cache_dir, kv_dtype=kv_dtype).fit(
        X[:64], y[:64]
    )
    assert reduced.context_info_["kv_dtype"] == kv_dtype
    np.testing.assert_allclose(
        reduced.predict_proba(X[64:]), exact.predict_proba(X[64:]), atol=2e-2
    )


@pytest.mark.slow
def test_sparse_index_context_loads_and_predicts(cache_dir):
    X, y = make_classification(n_samples=96, n_features=6, random_state=3)
    exact = _small_estimator(cache_dir=cache_dir).fit(X[:64], y[:64])
    indexed = _small_estimator(
        cache_dir=cache_dir, sparse_index=True, index_clusters=4
    ).fit(X[:64], y[:64])
    assert indexed.context_info_["index_clusters"] == 4
    np.testing.assert_allclose(
        indexed.predict_proba(X[64:]), exact.predict_proba(X[64:]), atol=1e-5
    )


@pytest.mark.slow
def test_more_than_ten_classes_raises(cache_dir):
    X = np.random.default_rng(0).standard_normal((44, 4))
    y = np.repeat(np.arange(11), 4)
    with pytest.raises(ValueError, match="at most 10 classes"):
        _small_estimator(cache_dir=cache_dir).fit(X, y)

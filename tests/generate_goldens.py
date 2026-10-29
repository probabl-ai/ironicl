"""Generate golden fixtures with the upstream tabicl package on CPU.

This script was run once, with tabicl 2.2.0 and PyTorch installed, to freeze
the fixtures under ``tests/data/<dataset>/``. IronICL itself no longer depends
on tabicl or torch, so the script is kept only as the record of how the
fixtures were produced; running it needs a separate environment with
``pip install "tabicl==2.2.0"``.

Each dataset folder holds:

- ``X_train.npy``, ``y_train.npy``, ``X_test.npy``: the raw table;
- ``probabilities.npy``: ``TabICLClassifier.predict_proba`` on CPU, used by
  ``tests/test_consistency.py`` for end-to-end parity;
- ``preprocessing/``: the fitted upstream preprocessing applied to the same
  rows, used by ``tests/test_preprocessing.py`` to check IronICL's native port
  of the tabicl preprocessing exactly. tabicl groups ensemble members by
  normalisation method in ``set`` order, which depends on ``PYTHONHASHSEED``,
  so every array is stored per method: ``train_X.<method>.npy`` (``[B_m, N, H']``),
  ``train_y.<method>.npy`` (``[B_m, N]``), ``test_X.<method>.npy``
  (``[B_m, Q, H']``), ``class_shuffles.<method>.npy`` (``[B_m, C]``),
  ``feature_shuffles.<method>.npy`` (``[B_m, H']``) and ``features_to_keep.npy``;
- ``metadata.json``: the estimator parameters and library versions used.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

GOLDEN_TABICL_VERSION = "2.2.0"
CHECKPOINT_NAME = "tabicl-classifier-v2-20260212.ckpt"
RANDOM_STATE = 42
DATASETS = {
    "binary": {
        "data": {"n_classes": 2, "n_informative": 5},
        "params": {"n_estimators": 2},
    },
    "multiclass": {
        "data": {"n_classes": 4, "n_informative": 6, "n_clusters_per_class": 1},
        "params": {"n_estimators": 2},
    },
    "constant_feature": {
        "data": {"n_classes": 3, "n_informative": 5, "n_clusters_per_class": 1},
        "constant_columns": 1,
        "params": {"n_estimators": 4},
    },
    "all_norms": {
        "data": {"n_classes": 3, "n_informative": 6, "n_clusters_per_class": 1},
        "params": {
            "n_estimators": 8,
            "norm_methods": ["none", "power", "quantile", "quantile_rtdl", "robust"],
            "feat_shuffle_method": "latin",
            "class_shuffle_method": "shift",
            "outlier_threshold": 3.0,
        },
    },
}


def checkpoint_kwargs() -> dict[str, str]:
    """Estimator kwargs selecting a local checkpoint when one is configured."""
    directory = os.environ.get("IRONICL_CHECKPOINT_DIR")
    return {"model_path": str(Path(directory) / CHECKPOINT_NAME)} if directory else {}


def main() -> None:
    installed = version("tabicl")
    if installed != GOLDEN_TABICL_VERSION:
        sys.exit(
            f"tabicl {GOLDEN_TABICL_VERSION} is pinned for goldens; found {installed}"
        )
    import torch
    from tabicl import TabICLClassifier

    torch.set_num_threads(1)
    root = Path(__file__).parent / "data"
    for name, settings in DATASETS.items():
        X, y = make_classification(
            n_samples=160, n_features=8, random_state=0, **settings["data"]
        )
        X = X.astype(np.float32)
        for _ in range(settings.get("constant_columns", 0)):
            X = np.concatenate([X, np.full((len(X), 1), 1.5, np.float32)], axis=1)
        X_train, X_test, y_train = X[:128], X[128:], y[:128]
        params = dict(settings["params"])
        classifier = TabICLClassifier(
            kv_cache="kv",
            device="cpu",
            random_state=RANDOM_STATE,
            **params,
            **checkpoint_kwargs(),
        ).fit(X_train, y_train)
        probabilities = classifier.predict_proba(X_test).astype(np.float32)
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "X_train.npy", X_train)
        np.save(folder / "y_train.npy", y_train)
        np.save(folder / "X_test.npy", X_test)
        np.save(folder / "probabilities.npy", probabilities)

        generator = classifier.ensemble_generator_
        encoded_test = classifier.X_encoder_.transform(X_test)
        train = generator.transform(X=None, mode="train")
        test = generator.transform(encoded_test, mode="test")
        preprocessing = folder / "preprocessing"
        preprocessing.mkdir(exist_ok=True)
        for method in train:
            np.save(preprocessing / f"train_X.{method}.npy", train[method][0])
            np.save(preprocessing / f"train_y.{method}.npy", train[method][1])
            np.save(preprocessing / f"test_X.{method}.npy", test[method][0])
            np.save(
                preprocessing / f"class_shuffles.{method}.npy",
                np.asarray(generator.class_shuffles_[method], dtype=np.int64),
            )
            np.save(
                preprocessing / f"feature_shuffles.{method}.npy",
                np.asarray(generator.feature_shuffles_[method], dtype=np.int64),
            )
        np.save(
            preprocessing / "features_to_keep.npy",
            np.asarray(generator.unique_filter_.features_to_keep_, dtype=bool),
        )
        metadata = {
            "tabicl_version": installed,
            "sklearn_version": version("scikit-learn"),
            "checkpoint": Path(classifier.model_path_).name,
            "random_state": RANDOM_STATE,
            "n_classes": int(settings["data"]["n_classes"]),
            "norm_methods_used": sorted(train),
            **params,
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"{name}: wrote goldens for {len(X_test)} query rows, "
            f"methods {sorted(train)}"
        )


if __name__ == "__main__":
    main()

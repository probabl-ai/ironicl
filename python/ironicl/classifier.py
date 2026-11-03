"""A drop-in replacement for TabICL's ``TabICLClassifier``, without PyTorch.

The constructor and public methods match TabICL 2.1.1 (commit
bd030d47ff7d9f987f0f33431d743a3003bfe0d3). Values that would change results
but are not implemented yet raise ``NotImplementedError``.
"""

from __future__ import annotations

import os
import urllib.request
from functools import cache
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ironicl._core import Model
from ironicl.preprocessing import ClassifierPreprocessor

CHECKPOINT_URL = "https://huggingface.co/jingang/TabICL/resolve/main/{}"


@cache  # ponytail: one resident model per checkpoint for the process lifetime.
def _model(checkpoint: Path) -> Model:
    return Model(checkpoint)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x / np.float32(temperature)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TabICLClassifier(ClassifierMixin, BaseEstimator):
    """TabICLv2 classifier with Ironicl's Rust inference engine.

    Parameters keep TabICL's names and defaults. ``batch_size``, ``kv_cache``,
    ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``,
    ``n_jobs``, and ``verbose`` control PyTorch execution and have no effect.
    ``support_many_classes`` has no effect because more than 10 classes are not
    supported yet.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: int | None = 8,
        kv_cache: bool | str = False,
        model_path: str | Path | None = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
        device: str | None = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: str | None = None,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        verbose: bool = False,
        inference_config: dict | None = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.inference_config = inference_config

    def fit(self, X, y) -> TabICLClassifier:
        """Fit preprocessing, load the model, and keep the training context."""
        for name, supported in (
            ("feat_shuffle_method", ("latin",)),
            ("class_shuffle_method", ("shift",)),
            ("device", (None, "cpu")),
            ("inference_config", (None,)),
        ):
            if getattr(self, name) not in supported:
                raise NotImplementedError(f"{name}={getattr(self, name)!r} is not supported")
        if self.random_state is None:
            raise NotImplementedError("random_state=None is not supported")
        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")

        methods = self.norm_methods or ["none", "power"]
        preprocessor = ClassifierPreprocessor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            norm_methods=(methods,) if isinstance(methods, str) else tuple(methods),
            outlier_threshold=self.outlier_threshold,
        )
        self.train_groups_ = preprocessor.fit_transform(X, y)
        self.y_train_ = preprocessor.encode_labels(y)
        self.preprocessor_ = preprocessor
        self.checkpoint_path_ = self._checkpoint()
        _model(self.checkpoint_path_)
        self.classes_ = preprocessor.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = preprocessor.n_features_in_
        if hasattr(preprocessor, "feature_names_in_"):
            self.feature_names_in_ = preprocessor.feature_names_in_
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities with shape ``(n_samples, n_classes)``."""
        check_is_fitted(self)
        pre = self.preprocessor_
        test_groups = pre.transform(X)
        train_rows, test_rows = len(self.y_train_), len(test_groups[0])
        x = np.empty((pre.n_estimators_, train_rows + test_rows, pre.n_features_out_), dtype=np.float32)
        y = np.empty((pre.n_estimators_, train_rows), dtype=np.float32)
        for estimator, group in enumerate(pre.ensemble_groups_):
            features = pre.feature_permutations_[estimator]
            x[estimator, :train_rows] = self.train_groups_[group][:, features]
            x[estimator, train_rows:] = test_groups[group][:, features]
            y[estimator] = pre.class_permutations_[estimator][self.y_train_]

        values, shape = _model(self.checkpoint_path_).predict_logits(x, y)
        logits = np.frombuffer(values, dtype=np.float32).reshape(shape)
        # Undo each estimator's class shift: logits[e, :, c] belongs to class permutation[e][c].
        logits = np.take_along_axis(logits, pre.class_permutations_[:, None, :].astype(np.intp), axis=2)
        if self.average_logits:
            proba = _softmax(logits.mean(axis=0), self.softmax_temperature)
        else:
            proba = _softmax(logits, self.softmax_temperature).mean(axis=0)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Return the most probable class label for each sample."""
        return self.preprocessor_.decode_labels(np.argmax(self.predict_proba(X), axis=1))

    def _checkpoint(self) -> Path:
        if self.model_path is not None and Path(self.model_path).exists():
            return Path(self.model_path)
        if self.model_path is not None:
            path = Path(self.model_path)
        else:
            cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            path = cache_root / "ironicl" / "checkpoints" / self.checkpoint_version
            if path.exists():
                return path
        if not self.allow_auto_download:
            raise FileNotFoundError(f"Checkpoint {path} not found and allow_auto_download=False")
        # The Rust importer verifies the checkpoint's SHA-256 before use.
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        urllib.request.urlretrieve(CHECKPOINT_URL.format(self.checkpoint_version), partial)
        partial.replace(path)
        return path

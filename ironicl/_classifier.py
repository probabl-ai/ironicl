"""The scikit-learn estimator served by the Rust engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ironicl import _context, _preprocessing
from ironicl._ironicl import StaticExecutor, build_context, inspect_model, prepare_model

MAX_CLASSES = 10
QUERY_CHUNK_ROWS = 1024
MAX_PICKLED_CONTEXT_BYTES = 1024**3
DEFAULT_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def default_cache_dir() -> Path:
    """Directory for imported models and static contexts (``IRONICL_CACHE_DIR``)."""
    configured = os.environ.get("IRONICL_CACHE_DIR")
    return Path(configured) if configured else Path.home() / ".cache" / "ironicl"


class IronICLClassifier(ClassifierMixin, BaseEstimator):
    """TabICLv2 classifier served from a precomputed training context on CPU.

    ``fit`` preprocesses the training table exactly as ``tabicl.TabICLClassifier``
    does (standardisation, Yeo-Johnson or quantile normalisation, outlier
    clipping, feature and class permutations), runs it through the TabICLv2
    transformer once in the Rust engine, and stores the resulting attention
    key/value state as a static context. ``predict`` then only processes the
    query rows against that context, which is the cached serving mode TabICL
    offers through ``kv_cache=True``. Neither step needs PyTorch or the
    ``tabicl`` package. The parameters below keep the meaning they have in
    ``tabicl.TabICLClassifier``, so the two estimators can be swapped inside a
    pipeline.

    The estimator takes finite numeric input. For tables with strings, dates or
    missing values put a ``skrub.TableVectorizer`` (or ``skrub.tabular_pipeline``)
    in front of it.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members. Each member sees a different feature
        permutation, class permutation and normalisation.
    norm_methods : str or list of str, default=None
        Normalisation methods spread across the ensemble, from ``"none"``,
        ``"power"``, ``"quantile"``, ``"quantile_rtdl"`` and ``"robust"``;
        ``None`` uses tabicl's default ``["none", "power"]``.
    feat_shuffle_method : {"none", "shift", "random", "latin"}, default="latin"
        How feature permutations are generated.
    class_shuffle_method : {"none", "shift", "random", "latin"}, default="shift"
        How class permutations are generated.
    outlier_threshold : float, default=4.0
        Z-score bound used by the outlier clipping step.
    model_path : str, default=None
        Local TabICLv2 checkpoint. ``None`` downloads ``checkpoint_version``
        from Hugging Face on first use, exactly as tabicl does.
    allow_auto_download : bool, default=True
        Whether a missing checkpoint may be downloaded.
    checkpoint_version : str, default="tabicl-classifier-v2-20260212.ckpt"
        Checkpoint file name to download when ``model_path`` is ``None``.
    random_state : int, default=42
        Seed for the feature and class permutations and the quantile noise.
    kv_dtype : {"fp32", "bf16", "bf16-v"}, default="fp32"
        Storage precision of the in-context-learning key/value tensors
        (``"bf16-v"`` keeps keys in FP32). Computation stays in FP32.
    sparse_index : bool, default=False
        Store a coarse cluster index over the ICL keys so the engine can skip
        far-away training rows when the omitted softmax mass is negligible.
    index_clusters : int, default=64
        Clusters per attention head when ``sparse_index`` is set.
    cache_dir : str, default=None
        Where imported models and static contexts are written. Defaults to
        ``$IRONICL_CACHE_DIR`` or ``~/.cache/ironicl``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_classes_ : int
        Number of classes, at most 10.
    n_features_in_ : int
        Number of features seen during ``fit``.
    ensemble_generator_ : EnsembleGenerator
        The fitted preprocessing and ensemble configuration reused at
        prediction time.
    checkpoint_path_ : str
        The TabICLv2 checkpoint the model was imported from.
    model_path_ : str
        Imported model artifact, content-addressed by the checkpoint hash.
    context_path_ : str
        Static-context artifact built for the training table.

    Notes
    -----
    Threads are a per-process setting of the engine, read once from the
    ``IRONICL_THREADS`` environment variable. The official softmax temperature
    (0.9) and logit averaging are applied inside the engine.

    Examples
    --------
    >>> from ironicl import IronICLClassifier
    >>> clf = IronICLClassifier(n_estimators=4).fit(X_train, y_train)  # doctest: +SKIP
    >>> clf.predict_proba(X_test)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        n_estimators: int = 8,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        model_path: str | None = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = DEFAULT_CHECKPOINT,
        random_state: int | None = 42,
        kv_dtype: str = "fp32",
        sparse_index: bool = False,
        index_clusters: int = 64,
        cache_dir: str | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.random_state = random_state
        self.kv_dtype = kv_dtype
        self.sparse_index = sparse_index
        self.index_clusters = index_clusters
        self.cache_dir = cache_dir

    def __sklearn_tags__(self):  # noqa: D105
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.classifier_tags.multi_class = True
        return tags

    def _resolved_cache_dir(self) -> Path:
        return (
            Path(self.cache_dir) if self.cache_dir is not None else default_cache_dir()
        )

    def fit(self, X, y):
        """Build the static context for the training table.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric training table without missing values.
        y : array-like of shape (n_samples,)
            Class labels, at most 10 distinct values.

        Returns
        -------
        self : IronICLClassifier
            The fitted estimator.
        """
        X, y = validate_data(
            self, X, y, dtype=np.float32, order="C", ensure_all_finite=True
        )
        check_classification_targets(y)
        if self.kv_dtype not in _context.KV_DTYPES:
            raise ValueError(
                f"kv_dtype must be one of {_context.KV_DTYPES}; got {self.kv_dtype!r}"
            )
        if self.n_estimators < 1:
            raise ValueError(
                f"n_estimators must be at least 1; got {self.n_estimators}"
            )
        if self.sparse_index and self.index_clusters < 2:
            raise ValueError(
                f"index_clusters must be at least 2; got {self.index_clusters}"
            )
        self.y_encoder_ = LabelEncoder().fit(y)
        y_encoded = self.y_encoder_.transform(y).astype(np.int64)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ > MAX_CLASSES:
            raise ValueError(
                f"IronICLClassifier supports at most {MAX_CLASSES} classes; "
                f"got {self.n_classes_}"
            )
        if self.n_classes_ < 2:
            raise ValueError(
                "IronICLClassifier needs at least two classes in y; got 1 class"
            )

        checkpoint = _preprocessing.resolve_checkpoint(
            self.model_path, self.checkpoint_version, self.allow_auto_download
        )
        cache_dir = self._resolved_cache_dir()
        self.checkpoint_path_ = str(Path(checkpoint).resolve())
        self.model_path_ = prepare_model(self.checkpoint_path_, cache_dir / "models")

        self.ensemble_generator_ = _preprocessing.EnsembleGenerator(
            classification=True,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=self.class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y_encoded)
        variants, labels, class_permutations, feature_permutations = (
            _preprocessing.training_variants(self.ensemble_generator_)
        )

        model_hash = str(inspect_model(self.model_path_)["source_sha256"])
        config: dict[str, Any] = {
            "checkpoint_sha256": model_hash,
            "n_estimators": self.n_estimators,
            "norm_methods": self.norm_methods,
            "feat_shuffle_method": self.feat_shuffle_method,
            "class_shuffle_method": self.class_shuffle_method,
            "outlier_threshold": self.outlier_threshold,
            "random_state": self.random_state,
            "kv_dtype": self.kv_dtype,
            "sparse_index": self.sparse_index,
            "index_clusters": self.index_clusters if self.sparse_index else None,
        }
        context_hash = _context.context_identity(X, y_encoded, config)
        context_path = cache_dir / "contexts" / f"{context_hash}.ironiclctx"
        self.context_info_ = build_context(
            self.model_path_,
            variants,
            labels,
            class_permutations,
            feature_permutations,
            bytes.fromhex(context_hash),
            str(context_path),
            kv_dtype=self.kv_dtype,
            index_clusters=self.index_clusters if self.sparse_index else None,
        )
        self.context_path_ = str(context_path)
        small = (
            self.kv_dtype == "fp32"
            and context_path.stat().st_size <= MAX_PICKLED_CONTEXT_BYTES
        )
        self.context_bytes_ = context_path.read_bytes() if small else None
        self._executor = self._build_executor()
        return self

    def _build_executor(self) -> StaticExecutor:
        if not Path(self.model_path_).is_file():
            self.model_path_ = prepare_model(
                self.checkpoint_path_, self._resolved_cache_dir() / "models"
            )
        if self.context_bytes_ is not None:
            return StaticExecutor.from_bytes(self.model_path_, self.context_bytes_)
        return StaticExecutor(self.model_path_, self.context_path_)

    def _get_executor(self) -> StaticExecutor:
        executor = self.__dict__.get("_executor")
        return executor if executor is not None else self._build_executor()

    def predict_proba(self, X):
        """Class probabilities for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric query rows without missing values.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Rows sum to one; columns follow ``classes_``.
        """
        check_is_fitted(self)
        X = validate_data(
            self, X, reset=False, dtype=np.float32, order="C", ensure_all_finite=True
        )
        executor = self._get_executor()
        variants = _preprocessing.query_variants(self.ensemble_generator_, X)
        outputs = [
            executor.predict_proba(
                np.ascontiguousarray(variants[:, start : start + QUERY_CHUNK_ROWS])
            )
            for start in range(0, X.shape[0], QUERY_CHUNK_ROWS)
        ]
        probabilities = np.concatenate(outputs, axis=0).astype(np.float64)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X):
        """Predicted class labels for ``X``."""
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __getstate__(self):  # noqa: D105
        state = super().__getstate__()
        state.pop("_executor", None)
        return state

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        if "model_path_" not in state:
            return
        try:
            self._executor = self._build_executor()
        except Exception:  # noqa: BLE001 - rebuilt with a clear error at predict time
            self._executor = None

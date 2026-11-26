"""NVIDIA's Kumo-Tabular as a scikit-learn estimator, without PyTorch.

Rust fits and predicts (``_core.KumoTabular``); this module downloads the
checkpoint, converts inputs as ``TabICLClassifier`` does, and keeps
scikit-learn's interface. See docs/kumo-tabular.md.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from numbers import Integral
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from ironicl import _core
from ironicl.base import SavedEstimator, _array, _features, _label_values, _labels, _progress, _threads
from ironicl.checkpoints import _checkpoints_dir, fetch

# NVIDIA's checkpoints at a pinned revision of `nvidia/Kumo-Tabular`, with their SHA-256.
REVISION = "c5893d418e162b9f841ae179dc767afefff73831"
CHECKPOINT_URL = "https://huggingface.co/nvidia/Kumo-Tabular/resolve/{revision}/{size}/{task}.pt"
SHA256 = {
    "small-classifier": "1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617",
    "small-regressor": "fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423",
    "medium-classifier": "70d6aee8fd13416300afead71a5145a40edb790952e2417d87f27b3f7e57793d",
    "medium-regressor": "daa6a17420b5aed231007e5d2f9c712abe093b4f94b3e1189b178947a95fb725",
    "large-classifier": "cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613",
    "large-regressor": "745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421",
}
SIZES = ("small", "medium", "large")
# NVIDIA's benchmark defaults.
DEFAULT_ESTIMATORS = {"small": 8, "medium": 8, "large": 16}


@lru_cache(maxsize=1)  # Keep repeated fits warm without pinning every precision's GPU weights.
def _model(checkpoint: Path, release: str, optimize: str) -> _core.KumoTabular:
    model = _core.KumoTabular(checkpoint, release, optimize)
    expected = SHA256.get(release)
    if expected is not None and model.sha256 != expected:
        warnings.warn(
            f"{checkpoint} has SHA-256 {model.sha256}, not {expected} as Ironicl expects; "
            "using custom weights",
            UserWarning,
            stacklevel=3,
        )
    return model


class _KumoTabular(SavedEstimator, BaseEstimator):
    """What the classifier and the regressor share; subclasses set ``_task``."""

    def __init__(
        self,
        size: str = "large",
        n_estimators: int | None = None,
        random_state: int = 42,
        n_jobs: int | None = None,
        verbose: bool = False,
        model_path: str | Path | None = None,
        allow_auto_download: bool = True,
        optimize: str | None = None,
    ):
        self.size = size
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.optimize = optimize

    _task: str
    _needs_training_data = True

    def _reset_fit(self) -> None:
        # Release the previous context before validation or allocating another one.
        for name in tuple(vars(self)):
            if name.endswith("_") and not name.startswith("_"):
                delattr(self, name)

    def _fit(self, X, targets: dict) -> None:
        """Fit the members' preprocessing and contexts in Rust on ``X`` and
        ``targets``, as ``_core.KumoTabular.fit`` takes them."""
        self._validate()
        optimize = self._optimize()
        x, names = _features(X)
        checkpoint = self._checkpoint()
        with self._progress("Building context", "estimator") as progress:
            fitted = _model(checkpoint, self._release(), optimize).fit(
                x,
                **targets,
                n_estimators=self._estimators(),
                random_state=self.random_state,
                threads=self._threads(),
                feature_names=None if names is None else list(names),
                progress=progress,
            )
        _warn_cpu_members(fitted)
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(x) if isinstance(x, list) else x.shape[1]
        self.optimize_ = optimize
        self.checkpoint_path_ = checkpoint
        self.fitted_ = fitted

    def _query(self, X):
        """``X`` as Rust predicts from it, checked against the training data."""
        check_is_fitted(self)
        expected = tuple(self.feature_names_in_) if hasattr(self, "feature_names_in_") else None
        if hasattr(X, "columns") and len(X.columns) == self.n_features_in_ and tuple(X.columns) != expected:
            raise ValueError("X must have the same feature names and order as the training data")
        x, names = _features(X, self.fitted_.categorical)
        width = len(x) if isinstance(x, list) else x.shape[1]
        if width != self.n_features_in_:
            raise ValueError(f"X has {width} features; expected {self.n_features_in_}")
        if names != expected:
            raise ValueError("X must have the same feature names and order as the training data")
        return x

    def export(self, path: str | Path, kv_cache: bool = True) -> None:
        """Export the fitted model to the directory ``path`` for ``ironicl serve``,
        as ``TabICLClassifier.export`` does.

        It holds the training data (``preprocessing.irprep``), the members'
        contexts when ``kv_cache`` is set (``context.ircntxt``; without them the
        server fits them again), and the metadata (``ironicl.json``). The weights
        go to the shared weights folder.
        """
        check_is_fitted(self)
        self.fitted_.export(Path(path), Path(self.checkpoint_path_), kv_cache)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _fitted_state(self, training: bool, context: bool) -> tuple[bytes, bytes | None]:
        return self.fitted_.to_bytes(), self.fitted_.context_bytes() if context else None

    def _load_fitted(self, checkpoint: bytes | None, context: bytes | None) -> None:
        """Load with this machine's threads: the saved contexts, else fitted again."""
        self._restore_checkpoint(checkpoint)
        # Pickles written before Kumo supported optimize always ran at safe.
        self.optimize_ = getattr(self, "optimize_", "safe")
        with self._progress("Building context", "estimator") as progress:
            self.fitted_ = _model(self.checkpoint_path_, self._release(), self.optimize_).load(
                self.fitted_, self._threads(), progress, context
            )
        _warn_cpu_members(self.fitted_)

    def _validate(self) -> None:
        if self.size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}, not {self.size!r}")
        for name, minimum in (("n_estimators", 1), ("random_state", None)):
            value = getattr(self, name)
            if name == "n_estimators" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or (minimum and value < minimum):
                raise ValueError(f"{name} must be an integer{' of at least 1' if minimum else ''}, not {value!r}")
        _threads(self.n_jobs)

    def _estimators(self) -> int:
        return DEFAULT_ESTIMATORS[self.size] if self.n_estimators is None else self.n_estimators

    def _threads(self) -> int | None:
        return _threads(self.n_jobs)

    def _progress(self, description: str, unit: str):
        return _progress(self.verbose, description, unit)

    def _release(self) -> str:
        """The checkpoint's size and task, such as ``large-classifier``."""
        return f"{self.size}-{self._task}"

    def _checkpoint(self, data: bytes | None = None) -> Path:
        """The checkpoint path, downloaded at the pinned revision if missing, or
        written from ``data`` when given; Rust checks its tensors."""
        name = self._release()
        path = Path(self.model_path) if self.model_path is not None else _checkpoints_dir("kumo-tabular") / f"{name}.pt"
        url = CHECKPOINT_URL.format(revision=REVISION, size=self.size, task=self._task)
        return fetch(path, url, SHA256.get(name), allow_download=self.allow_auto_download, data=data)


def _warn_cpu_members(fitted) -> None:
    if fitted.cpu_members:
        warnings.warn(
            f"{fitted.cpu_members} ensemble members did not fit in GPU memory and run on the CPU, "
            "which is much slower; use fewer n_estimators or training rows",
            UserWarning,
            stacklevel=3,
        )


class KumoTabularClassifier(ClassifierMixin, _KumoTabular):
    """Kumo-Tabular classifier with Ironicl's Rust inference engine, on the CPU or a GPU.

    ``size`` is ``"small"``, ``"medium"``, or ``"large"``. ``n_estimators``
    ensemble members each preprocess the data differently; ``None`` uses
    NVIDIA's benchmark defaults, 16 for ``"large"`` and 8 otherwise.
    ``random_state`` seeds the members' random draws: Ironicl draws them
    itself, so a seed gives the same model on every machine. ``n_jobs`` sets
    the worker threads (``None`` uses every core; negative values count back
    from all cores). ``verbose=True`` shows progress. ``optimize`` selects
    ``"safe"``, ``"auto"``, or ``"fast"``; ``None`` reads ``IRONICL_OPTIMIZE``
    and defaults to ``"safe"``. The fitted level is ``optimize_``.

    The checkpoint is downloaded from Hugging Face's ``nvidia/Kumo-Tabular``
    at a pinned revision into Ironicl's cache folder, unless ``model_path``
    names it. Its weights are under NVIDIA's OpenMDW-1.1 license. ``save``,
    ``load``, and pickling work as for ``TabICLClassifier``; the training data
    is always saved, since loading fits the preprocessing again from it.
    """

    _task = "classifier"

    def fit(self, X, y) -> KumoTabularClassifier:
        """Discard any previous fit and fit the classifier. A failed fit leaves it unfitted."""
        self._reset_fit()
        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")
        labels = _labels(y)
        targets = _label_values(labels)
        classes = np.unique(labels)  # Rust sorts the classes as NumPy does.
        self._fit(X, {"labels": targets})
        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities with shape ``(n_samples, n_classes)``."""
        x = self._query(X)
        with self._progress("Predicting", "row") as progress:
            values, shape = self.fitted_.predict_proba(x, progress)
        return _array(values, shape).copy()

    def predict(self, X) -> np.ndarray:
        """Return the most probable class label for each sample."""
        probabilities = self.predict_proba(X)  # Checks first that the estimator is fitted.
        return self.classes_[np.argmax(probabilities, axis=1)]


class KumoTabularRegressor(RegressorMixin, _KumoTabular):
    """Kumo-Tabular regressor with Ironicl's Rust inference engine, on the CPU or a GPU.

    Each member predicts 999 quantiles; their sorted values are averaged over
    the members after dropping the lowest and highest fifth, as NVIDIA's code
    does. ``predict`` summarizes them as ``TabICLRegressor.predict`` does.
    The parameters are ``KumoTabularClassifier``'s.
    """

    _task = "regressor"

    def fit(self, X, y) -> KumoTabularRegressor:
        """Discard any previous fit and fit the regressor. A failed fit leaves it unfitted."""
        self._reset_fit()
        if y is None:
            raise ValueError("This regressor requires y to be passed, but the target y is None.")
        # A column vector becomes one-dimensional, with scikit-learn's warning.
        y = column_or_1d(check_array(y, ensure_2d=False, dtype=np.float32), warn=True)
        if not np.isfinite(y).all():
            raise ValueError("y must be a one-dimensional array of finite numbers")
        self._fit(X, {"values": np.ascontiguousarray(y)})
        return self

    def predict(self, X, output_type: str | list[str] = "mean", alphas: list[float] | None = None):
        """Predict target statistics for each sample.

        ``output_type`` is ``"mean"`` (the mean of the 999 quantiles, as
        NVIDIA's benchmark predicts), ``"median"``, or ``"variance"`` (shape
        ``(n_samples,)``), ``"quantiles"`` at ``alphas`` (default 0.1 to 0.9),
        ``"raw_quantiles"`` (the 999 combined quantiles), or a list of these,
        which returns a dict.
        """
        x = self._query(X)
        types = [output_type] if isinstance(output_type, str) else list(output_type)
        alphas = None if alphas is None else list(alphas)
        with self._progress("Predicting", "row") as progress:
            results = self.fitted_.predict_regression(x, types, alphas, progress)
        final = {}
        for name, (values, shape) in zip(types, results):
            values = _array(values, shape).copy()
            final[name] = values.ravel() if name in ("mean", "variance", "median") else values
        return final[types[0]] if len(types) == 1 else final

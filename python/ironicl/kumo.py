"""NVIDIA's Kumo-Tabular as a scikit-learn estimator, without PyTorch.

Rust fits and predicts (``_core.KumoTabular``); this module downloads the
checkpoint, converts inputs as ``TabICLClassifier`` does, and keeps
scikit-learn's interface. See docs/kumo-tabular.md.
"""

from __future__ import annotations

import urllib.request
import warnings
from functools import cache
from numbers import Integral
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ironicl import _core
from ironicl.base import _array, _features, _label_values, _labels, _progress, _threads, _version
from ironicl.cache import _checkpoints_dir

REVISION = "c5893d418e162b9f841ae179dc767afefff73831"
CHECKPOINT_URL = "https://huggingface.co/nvidia/Kumo-Tabular/resolve/{revision}/{size}/{task}.pt"
SIZES = ("small", "medium", "large")
# NVIDIA's benchmark defaults.
DEFAULT_ESTIMATORS = {"small": 8, "medium": 8, "large": 16}


@cache  # One resident model per checkpoint for the process lifetime.
def _model(checkpoint: Path) -> _core.KumoTabular:
    return _core.KumoTabular(checkpoint)


class KumoTabularClassifier(ClassifierMixin, BaseEstimator):
    """Kumo-Tabular classifier with Ironicl's Rust inference engine, on the CPU.

    ``size`` is ``"small"``, ``"medium"``, or ``"large"``. ``n_estimators``
    ensemble members each preprocess the data differently; ``None`` uses
    NVIDIA's benchmark defaults, 16 for ``"large"`` and 8 otherwise.
    ``random_state`` seeds the members' random draws: Ironicl draws them
    itself, so a seed gives the same model on every machine. ``n_jobs`` sets
    the worker threads (``None`` uses every core; negative values count back
    from all cores). ``verbose=True`` shows progress.

    The checkpoint is downloaded from Hugging Face's ``nvidia/Kumo-Tabular``
    at a pinned revision into Ironicl's cache folder, unless ``model_path``
    names it. Its weights are under NVIDIA's OpenMDW-1.1 license. Pickling
    saves the training data; loading fits again.
    """

    def __init__(
        self,
        size: str = "large",
        n_estimators: int | None = None,
        random_state: int = 42,
        n_jobs: int | None = None,
        verbose: bool = False,
        model_path: str | Path | None = None,
        allow_auto_download: bool = True,
    ):
        self.size = size
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download

    _task = "classifier"

    def fit(self, X, y) -> KumoTabularClassifier:
        """Fit the members' preprocessing and contexts in Rust."""
        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")
        self._validate()
        labels = _labels(y)
        x, names = _features(X)
        checkpoint = self._checkpoint()
        with self._progress("Building context", "estimator") as progress:
            self.fitted_ = _model(checkpoint).fit(
                x,
                _label_values(labels),
                self._estimators(),
                self.random_state,
                threads=self._threads(),
                feature_names=None if names is None else list(names),
                progress=progress,
            )
        self.checkpoint_path_ = checkpoint
        self.classes_ = np.unique(labels)  # Rust sorts the classes as NumPy does.
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = len(x) if isinstance(x, list) else x.shape[1]
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_  # Left by an earlier fit on named data.
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities with shape ``(n_samples, n_classes)``."""
        check_is_fitted(self)
        expected = tuple(self.feature_names_in_) if hasattr(self, "feature_names_in_") else None
        if hasattr(X, "columns") and len(X.columns) == self.n_features_in_ and tuple(X.columns) != expected:
            raise ValueError("X must have the same feature names and order as the training data")
        x, names = _features(X, self.fitted_.categorical)
        if names != expected:
            raise ValueError("X must have the same feature names and order as the training data")
        with self._progress("Predicting", "row") as progress:
            values, shape = self.fitted_.predict_proba(x, progress)
        return _array(values, shape).copy()

    def predict(self, X) -> np.ndarray:
        """Return the most probable class label for each sample."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def export(self, path: str | Path) -> None:
        """Export the fitted model to the directory ``path`` for ``ironicl serve``.

        It holds the training data (``preprocessing.irprep``), from which the
        server fits the model again, and the metadata (``ironicl.json``). The
        weights go to the shared weights folder, as for ``TabICLClassifier.export``.
        """
        check_is_fitted(self)
        self.fitted_.export(Path(path), Path(self.checkpoint_path_))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def __getstate__(self):
        """The pickled state, with the fitted model as its training data."""
        state = dict(super().__getstate__())
        state["_ironicl_version"] = _version()
        if "fitted_" in state:
            state["fitted_"] = self.fitted_.to_bytes()
        return state

    def __setstate__(self, state):
        """Load the fitted model by fitting it again, with this machine's threads."""
        state = dict(state)
        version = state.pop("_ironicl_version", None)
        if version != _version():
            warnings.warn(
                f"This estimator was saved with Ironicl {version} and is loaded with {_version()}",
                UserWarning,
                stacklevel=2,
            )
        super().__setstate__(state)
        if "fitted_" in state:
            if not Path(self.checkpoint_path_).exists():
                self.checkpoint_path_ = self._checkpoint()
            with self._progress("Building context", "estimator") as progress:
                self.fitted_ = _model(self.checkpoint_path_).load(self.fitted_, self._threads(), progress)

    def _validate(self) -> None:
        if self.size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}, not {self.size!r}")
        for name, minimum in (("n_estimators", 1), ("random_state", None)):
            value = getattr(self, name)
            if name == "n_estimators" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or (minimum and value < minimum):
                raise ValueError(f"{name} must be an integer{' of at least 1' if minimum else ''}, not {value!r}")

    def _estimators(self) -> int:
        return DEFAULT_ESTIMATORS[self.size] if self.n_estimators is None else self.n_estimators

    def _threads(self) -> int | None:
        return _threads(self.n_jobs)

    def _progress(self, description: str, unit: str):
        return _progress(self.verbose, description, unit)

    def _checkpoint(self) -> Path:
        """The checkpoint, downloaded at the pinned revision if missing; Rust checks its SHA-256."""
        if self.model_path is not None:
            return Path(self.model_path)
        path = _checkpoints_dir() / "kumo-tabular" / REVISION / f"{self.size}-{self._task}.pt"
        if path.exists():
            return path
        if not self.allow_auto_download:
            raise FileNotFoundError(f"Checkpoint {path} not found and allow_auto_download=False")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        url = CHECKPOINT_URL.format(revision=REVISION, size=self.size, task=self._task)
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
        return path


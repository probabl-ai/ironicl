"""NVIDIA's Kumo-Tabular as a scikit-learn estimator, without PyTorch.

Rust fits and predicts (``_core.KumoTabular``); this module downloads the
checkpoint, converts inputs as ``TabICLClassifier`` does, and keeps
scikit-learn's interface. See docs/kumo-tabular.md.
"""

from __future__ import annotations

from functools import cache
from numbers import Integral
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

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


@cache  # One resident model per checkpoint for the process lifetime.
def _model(checkpoint: Path, release: str) -> _core.KumoTabular:
    return _core.KumoTabular(checkpoint, release)


class KumoTabularClassifier(ClassifierMixin, SavedEstimator, BaseEstimator):
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
    names it. Its weights are under NVIDIA's OpenMDW-1.1 license. ``save``,
    ``load``, and pickling work as for ``TabICLClassifier``; the training data
    is always saved, since loading fits the preprocessing again from it.
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
    _needs_training_data = True

    def fit(self, X, y) -> KumoTabularClassifier:
        """Fit the members' preprocessing and contexts in Rust."""
        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")
        self._validate()
        labels = _labels(y)
        x, names = _features(X)
        checkpoint = self._checkpoint()
        with self._progress("Building context", "estimator") as progress:
            self.fitted_ = _model(checkpoint, self._release()).fit(
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
        with self._progress("Building context", "estimator") as progress:
            self.fitted_ = _model(self.checkpoint_path_, self._release()).load(
                self.fitted_, self._threads(), progress, context
            )

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

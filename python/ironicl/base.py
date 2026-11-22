"""What ``TabICLClassifier`` and ``TabICLRegressor`` share: input conversion, the
checkpoint, fitting and its disk cache, threads, ``optimize``, and memory planning.

Rust does the work: ``_core.preprocess`` fits preprocessing and ``Model.build``
builds the context, which gives the fitted model (docs/native-preprocessing.md).
Python converts inputs and keeps scikit-learn's interface."""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import urllib.request
import warnings
from functools import cache
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

from ironicl import _core
from ironicl._core import Model
from ironicl.cache import _cache_dir
from ironicl.memory import _enforce, _limit_text, _overflow_policy, _plan
from ironicl.progress import Progress

CHECKPOINT_URL = "https://huggingface.co/jingang/TabICL/resolve/main/{}"


def _version() -> str:
    return version("ironicl")


@cache  # ponytail: one resident model per checkpoint for the process lifetime.
def _model(checkpoint: Path) -> Model:
    return Model(checkpoint)


class TabICLBase(BaseEstimator):
    """Shared fit machinery. Subclasses set ``_task`` and ``_checkpoint_version``."""

    _task: str
    _checkpoint_version: str

    def _fit(self, X, y):
        """Fit preprocessing in Rust on ``X`` and the checked targets ``y``, plan
        the memory, then build the context.

        Fitted attributes are set only once the context exists, so a failed fit
        never looks fitted. The memory plan is checked after preprocessing, when
        the context's shape is known, and before any download or build work.
        """
        self._validate_params()
        optimize = self._optimize()
        policy = _overflow_policy(self.memory_overflow)
        limits = self._limits()
        x, names = _features(X)
        preprocessed = _core.preprocess(
            x,
            **self._targets(y),
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            norm_methods=self._norm_methods(),
            outlier_threshold=float(self.outlier_threshold),
            feature_names=None if names is None else list(names),
            **self._aggregation(),
        )
        plan = _plan(self._task, *preprocessed.shape, optimize, *limits)
        _enforce(plan, policy, stacklevel=3)
        checkpoint = self._checkpoint()
        self.fitted_ = self._build(checkpoint, preprocessed, optimize, limits)
        self._training = (x.copy(), y.copy())  # Copies: the caller may change X and y.
        self.memory_plan_ = plan
        self.checkpoint_path_ = checkpoint
        self.n_features_in_ = x.shape[1]
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_  # Left by an earlier fit on named data.

    def _targets(self, y) -> dict:
        """The checked targets ``y`` as ``_core.preprocess`` takes them."""
        raise NotImplementedError

    def _aggregation(self) -> dict:
        """The classifier's prediction settings, which a saved model keeps."""
        return {}

    def _build(self, checkpoint: Path, preprocessed, optimize, limits):
        """Build the context of ``preprocessed``, through ``cache_dir`` when set.
        Rust checks and writes the cache; a bad entry is rebuilt and a failed
        write is ignored."""
        model = self._task_model(checkpoint)
        with self._progress("Building context", "estimator") as progress:
            arguments = (optimize, self._resolve_threads(), *limits)
            return model.build(preprocessed, *arguments, progress=progress, cache_dir=_cache_dir(self.cache_dir))

    def _load(self, checkpoint: Path, preprocessor: bytes, context: bytes | None, optimize, limits):
        """Load the fitted model from its ``IRPREP`` and ``IRCNTXT`` bytes, or
        without the context, build it from the training data as ``_build`` does."""
        model = self._task_model(checkpoint)
        x, y = getattr(self, "_training", (None, None))
        training = {} if x is None else {"x": x, **self._targets(y)}
        with self._progress("Building context", "estimator") as progress:
            arguments = (optimize, self._resolve_threads(), *limits)
            return model.load_fitted(
                preprocessor,
                *arguments,
                progress=progress,
                cache_dir=_cache_dir(self.cache_dir),
                context=context,
                **training,
            )

    def _task_model(self, checkpoint: Path) -> Model:
        """The checkpoint's model, which must be for this estimator's task."""
        model = _model(checkpoint)
        if model.task != self._task:
            raise ValueError(f"{self.checkpoint_version} is a {model.task} checkpoint, not a {self._task}")
        return model

    def _query(self, X) -> np.ndarray:
        """``X`` as the float matrix Rust predicts from, checked against the training data."""
        x, names = _features(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features; expected {self.n_features_in_}")
        expected = tuple(self.feature_names_in_) if hasattr(self, "feature_names_in_") else None
        if names != expected:
            raise ValueError("X must have the same feature names and order as the training data")
        return x

    def save(
        self,
        path: str | Path,
        save_model_weights: bool = False,
        save_training_data: bool = True,
        save_kv_cache: bool = True,
    ) -> None:
        """Save the fitted estimator to ``path``, for ``load``, ``pickle.load``, or ``joblib.load``.

        ``save_model_weights`` includes the checkpoint, so loading needs no
        file or download. ``save_kv_cache`` includes the fitted context; without
        it, loading rebuilds the context from the training data, through
        ``cache_dir`` when set. ``save_training_data`` includes the training
        data, which only a load without the context needs. The fitted model is
        saved in Rust's formats, not as Python objects.
        """
        check_is_fitted(self)
        if not (save_training_data or save_kv_cache):
            raise ValueError(
                "Cannot exclude both the training data and the KV cache: loading needs one of them."
            )
        self._save_options = (save_model_weights, save_training_data, save_kv_cache)
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as file:
                pickle.dump(self, file, protocol=5)
        finally:
            del self._save_options

    def export(self, path: str | Path) -> None:
        """Export the fitted model to the directory ``path`` for ``ironicl serve``.

        It holds the model weights (``model.irmodel``), the fitted preprocessing
        (``preprocessing.irprep``), the KV cache built at this fit's ``optimize``
        level (``context.ircntxt``), the training data (``train_x.npy``,
        ``train_y.npy``), from which ``ironicl prepare`` rebuilds the cache for
        another machine or level, and last a manifest (``ironicl.json``) that
        marks the directory complete and describes the model.

        Each file replaces the old one whole, so a server that maps it keeps a
        valid file. The data files are readable only by their owner on Unix,
        since they hold the training data.
        """
        check_is_fitted(self)
        if not hasattr(self, "_training"):
            raise ValueError(
                "export needs the training data, which this estimator was saved without; "
                "save it with save_training_data=True"
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Until the new manifest, the directory is incomplete.
        (path / "ironicl.json").unlink(missing_ok=True)
        _core.import_checkpoint(Path(self.checkpoint_path_), path / "model.irmodel")
        _write_private(path / "preprocessing.irprep", lambda file: file.write(self.fitted_.preprocessor_bytes()))
        _write_private(path / "context.ircntxt", lambda file: file.write(self.fitted_.context_bytes()))
        x, y = self._training
        codes = np.searchsorted(self.classes_, y) if self._task == "classifier" else y
        _write_private(path / "train_x.npy", lambda file: np.save(file, x))
        _write_private(path / "train_y.npy", lambda file: np.save(file, codes))
        names = getattr(self, "feature_names_in_", None)
        manifest = {
            "format": "ironicl-export",
            "version": 1,
            "ironicl": _version(),
            "task": self._task,
            "n_features": int(self.n_features_in_),
            "feature_names": None if names is None else [str(name) for name in names],
            "classes": self.classes_.tolist() if self._task == "classifier" else None,
        }
        _write_private(path / "ironicl.json", lambda file: file.write((json.dumps(manifest, indent=2) + "\n").encode()))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None):
        """Load an estimator that ``save`` wrote. ``device`` accepts what the constructor does."""
        with open(path, "rb") as file:
            estimator = pickle.load(file)
        if not isinstance(estimator, cls):
            raise TypeError(f"{path} holds a {type(estimator).__name__}, not a {cls.__name__}")
        if device is not None:
            if device != "cpu":
                raise NotImplementedError(f"device={device!r} is not supported")
            estimator.device = device
        return estimator

    def __getstate__(self):
        """The pickled state: by default self-contained, with the fitted model as bytes."""
        state = dict(super().__getstate__())  # May be this estimator's own __dict__.
        weights, training, context = state.pop("_save_options", (False, True, True))
        state["_ironicl_version"] = _version()
        if "fitted_" in state:
            state["fitted_"] = self.fitted_.preprocessor_bytes()
            state["_context"] = self.fitted_.context_bytes() if context else None
            if not training:
                state.pop("_training", None)
            if weights:
                state["_checkpoint_bytes"] = Path(self.checkpoint_path_).read_bytes()
        return state

    def __setstate__(self, state):
        """Load the fitted model, building its context if it was saved without one,
        with this machine's threads and memory limits.

        Loading plans its memory as ``fit`` does and follows ``memory_overflow``;
        ``memory_plan_`` becomes this machine's plan."""
        state = dict(state)
        version = state.pop("_ironicl_version", None)
        if version != _version():
            warnings.warn(
                f"This estimator was saved with Ironicl {version} and is loaded with {_version()}",
                UserWarning,
                stacklevel=2,
            )
        checkpoint = state.pop("_checkpoint_bytes", None)
        context = state.pop("_context", None)
        super().__setstate__(state)
        if "fitted_" not in state:
            return  # Not fitted.
        saved_plan, limits = self.memory_plan_, self._limits()
        shape = (saved_plan.estimators, saved_plan.rows, saved_plan.features)
        plan = _plan(self._task, *shape, saved_plan.optimize, *limits, context is not None)
        _enforce(plan, _overflow_policy(self.memory_overflow), stacklevel=2, action="load")
        self.memory_plan_ = plan
        if not Path(self.checkpoint_path_).exists():
            self.checkpoint_path_ = self._checkpoint(checkpoint)
        self.fitted_ = self._load(self.checkpoint_path_, self.fitted_, context, plan.optimize, limits)

    def _limits(self) -> tuple[str | None, str | None]:
        return (
            _limit_text("memory_limit", self.memory_limit),
            _limit_text("device_memory_limit", self.device_memory_limit),
        )

    def _norm_methods(self) -> list[str]:
        """``norm_methods`` as a list, TabICL's default when unset or empty."""
        methods = self.norm_methods or ["none", "power"]
        return [methods] if isinstance(methods, str) else list(methods)

    @contextlib.contextmanager
    def _progress(self, description: str, unit: str, total: int | None = None):
        """Yield the callback Rust reports ``(stage, done, total)`` to, which
        drives a ``Progress`` for ``verbose=True``, else ``None``."""
        if not self.verbose:
            yield None
            return
        with Progress(description, unit, total) as progress:
            yield lambda _stage, done, total: progress.update(done, total)

    def _validate_params(self, unsupported=(), noops=()) -> None:
        # Values that would change results or request an unimplemented feature raise.
        for name, supported in (
            ("feat_shuffle_method", ("latin",)),
            ("device", (None, "cpu")),
            ("inference_config", (None,)),
            ("use_amp", ("auto", False)),
            ("use_fa3", ("auto", False)),
            ("offload_mode", ("auto", False)),
            ("disk_offload_dir", (None,)),
            ("kv_cache", (False, True, "kv")),
            ("checkpoint_version", (self._checkpoint_version,)),
            *unsupported,
        ):
            if getattr(self, name) not in supported:
                raise NotImplementedError(f"{name}={getattr(self, name)!r} is not supported")
        if self.random_state is None:
            raise NotImplementedError("random_state=None is not supported")
        # Rust checks the ranges; Python checks the types it would convert.
        for name, kind in (("n_estimators", Integral), ("random_state", Integral), ("outlier_threshold", Real)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, kind):
                raise ValueError(f"{name} must be a{'n integer' if kind is Integral else ' number'}, not {value!r}")
        unsupported = sorted(set(self._norm_methods()) - {"none", "power"})
        if unsupported:
            raise NotImplementedError(f"norm_methods {unsupported} are not supported; only 'none' and 'power'")
        # Genuine no-ops: warn when changed so the caller knows they are ignored.
        for name, default in (("batch_size", 8), *noops):
            if getattr(self, name) != default:
                warnings.warn(
                    f"{name}={getattr(self, name)!r} has no effect in Ironicl",
                    UserWarning,
                    stacklevel=4,
                )

    def _resolve_threads(self) -> int | None:
        """Map ``n_jobs`` to an executor thread count (``None`` keeps the default)."""
        if self.n_jobs is None or self.n_jobs > 0:
            return self.n_jobs
        return max(1, (os.cpu_count() or 1) + 1 + self.n_jobs)

    def _optimize(self) -> str:
        optimize = self.optimize if self.optimize is not None else os.environ.get("IRONICL_OPTIMIZE", "safe")
        if optimize not in ("safe", "auto", "fast"):
            raise ValueError(f"optimize must be 'safe', 'auto', or 'fast', not {optimize!r}")
        return optimize

    def _checkpoint(self, data: bytes | None = None) -> Path:
        """The checkpoint path, downloaded if missing, or written from ``data`` when given."""
        if self.model_path is not None:
            path = Path(self.model_path)
        else:
            cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            path = cache_root / "ironicl" / "checkpoints" / self.checkpoint_version
        if path.exists():
            return path
        if data is None and not self.allow_auto_download:
            raise FileNotFoundError(f"Checkpoint {path} not found and allow_auto_download=False")
        # The Rust importer verifies the checkpoint's SHA-256 before use.
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        if data is None:
            urllib.request.urlretrieve(CHECKPOINT_URL.format(self.checkpoint_version), partial)
        else:
            partial.write_bytes(data)
        partial.replace(path)
        return path


def _write_private(path: Path, write) -> None:
    """Write ``path`` through ``write(file)``, owner-only on Unix, to a temporary
    file that then replaces ``path`` whole."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)  # One a failed export left, maybe with wider permissions.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)  # No newline translation on Windows.
    try:
        with os.fdopen(os.open(temporary, flags, 0o600), "wb") as file:
            write(file)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _features(X) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """``X`` as a C-ordered float32 or float64 matrix, and its feature names if it has any."""
    names = None
    if hasattr(X, "columns"):
        names = tuple(X.columns)
        if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("DataFrame feature names must be unique strings")
    if np.asarray(X).dtype.kind not in "biuf":
        raise TypeError(
            "X must contain dense, real numeric features; categorical and sparse inputs are unsupported"
        )
    return check_array(X, dtype=[np.float64, np.float32], order="C"), names


def _array(values: bytes, shape, dtype=np.float32) -> np.ndarray:
    """A read-only array over bytes Rust returned."""
    return np.frombuffer(values, dtype=dtype).reshape(shape)


def _labels(y) -> np.ndarray:
    """``y`` as a one-dimensional array of class labels."""
    labels = check_array(y, ensure_2d=False, dtype=None)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional array of class labels")
    check_classification_targets(labels)
    return labels


def _label_values(labels: np.ndarray) -> tuple[str, object]:
    """``labels`` as the ``(kind, values)`` pair Rust reads: int64, float64, or
    uint8 arrays for integers, floats, and booleans, or a list of str."""
    kind = labels.dtype.kind
    if kind == "O":
        items = labels.tolist()
        if all(isinstance(item, str) for item in items):
            return "text", items
        if all(isinstance(item, (bool, np.bool_)) for item in items):
            kind, labels = "b", np.asarray(items, dtype=bool)
        elif all(isinstance(item, Integral) and not isinstance(item, (bool, np.bool_)) for item in items):
            kind, labels = "i", np.asarray(items, dtype=np.int64)
        elif all(isinstance(item, Real) and not isinstance(item, (bool, np.bool_)) for item in items):
            kind, labels = "f", np.asarray(items, dtype=np.float64)
        else:
            raise TypeError("Class labels must be all strings, all booleans, or all numbers")
    if kind == "U":
        return "text", labels.tolist()
    if kind == "b":
        return "bool", np.ascontiguousarray(labels, dtype=np.uint8)
    if kind in "iu":
        if kind == "u" and labels.size and labels.max() > np.iinfo(np.int64).max:
            raise ValueError("Integer class labels must fit in int64")
        return "int", np.ascontiguousarray(labels, dtype=np.int64)
    if kind == "f":
        return "float", np.ascontiguousarray(labels, dtype=np.float64)
    raise TypeError(f"Unsupported class label dtype {labels.dtype}")

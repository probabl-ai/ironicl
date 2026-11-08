"""What ``TabICLClassifier`` and ``TabICLRegressor`` share: the checkpoint, the
fitted context and its disk cache, threads, ``optimize``, and memory planning."""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import urllib.request
import warnings
from functools import cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples, check_is_fitted

from ironicl._core import Model
from ironicl.cache import _cache_dir
from ironicl.memory import _enforce, _limit_text, _overflow_policy, _plan
from ironicl.preprocessing import Preprocessor
from ironicl.progress import Progress

CHECKPOINT_URL = "https://huggingface.co/jingang/TabICL/resolve/main/{}"
# Prediction preprocesses this many rows at a time, which bounds the [B, rows, H]
# copies it makes. Rust splits each chunk further to fit its own buffers.
PREDICT_CHUNK_ROWS = 8192


def _version() -> str:
    return version("ironicl")


@cache  # ponytail: one resident model per checkpoint for the process lifetime.
def _model(checkpoint: Path) -> Model:
    return Model(checkpoint)


class TabICLBase(BaseEstimator):
    """Shared fit machinery. Subclasses set ``_task`` and ``_checkpoint_version``."""

    _task: str
    _checkpoint_version: str

    def _fit_context(self, X, y, targets, classification: bool) -> Preprocessor:
        """Fit preprocessing and build the Rust context from the per-row ``targets(preprocessor)``.

        Fitted attributes are set only once the context exists, so a failed fit
        never looks fitted. The memory plan is checked after preprocessing, when
        the context's shape is known, and before any model or build work.
        """
        self._validate_params()
        optimize = self._optimize()
        policy = _overflow_policy(self.memory_overflow)
        limits = self._limits()
        methods = self.norm_methods or ["none", "power"]
        preprocessor = Preprocessor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            norm_methods=(methods,) if isinstance(methods, str) else tuple(methods),
            outlier_threshold=self.outlier_threshold,
            classification=classification,
        )
        groups = tuple(group.astype(np.float32) for group in preprocessor.fit_transform(X, y))
        rows, features = groups[0].shape
        plan = _plan(self._task, preprocessor.n_estimators_, rows, features, optimize, *limits)
        _enforce(plan, policy, stacklevel=3)
        checkpoint = self._checkpoint()
        training = {"groups": groups, "targets": targets(preprocessor)}
        self.context_ = self._build(_model(checkpoint), optimize, limits, preprocessor, training)
        self._training = training
        self.memory_plan_ = plan
        self.preprocessor_ = preprocessor
        self.checkpoint_path_ = checkpoint
        self.n_features_in_ = preprocessor.n_features_in_
        if hasattr(preprocessor, "feature_names_in_"):
            self.feature_names_in_ = preprocessor.feature_names_in_
        return preprocessor

    def _build(self, model, optimize, limits, preprocessor, training):
        """Build the context from ``training``: the grouped training rows and
        the per-row targets."""
        if model.task != self._task:
            raise ValueError(f"{self.checkpoint_version} is a {model.task} checkpoint, not a {self._task}")
        return self._context(
            model,
            optimize,
            limits,
            _ensemble_rows(preprocessor, training["groups"]),
            np.ascontiguousarray(self._expand_targets(preprocessor, training["targets"]), dtype=np.float32),
            np.ascontiguousarray(preprocessor.class_permutations_, dtype=np.uint64),
            np.ascontiguousarray(preprocessor.feature_permutations_, dtype=np.uint64),
        )

    def _expand_targets(self, preprocessor: Preprocessor, targets: np.ndarray) -> np.ndarray:
        """Each estimator's ``[B, N]`` targets from the per-row ``targets``."""
        raise NotImplementedError

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
        ``cache_dir`` when set. ``save_training_data`` includes the preprocessed
        training rows, which only a load without the context needs.
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
        """The pickled state: by default self-contained, with the context as bytes."""
        state = dict(super().__getstate__())  # May be this estimator's own __dict__.
        weights, training, context = state.pop("_save_options", (False, True, True))
        state["_ironicl_version"] = _version()
        if "context_" in state:
            state["context_"] = state["context_"].to_bytes() if context else None
            if not training:
                del state["_training"]
            if weights:
                state["_checkpoint_bytes"] = Path(self.checkpoint_path_).read_bytes()
        return state

    def __setstate__(self, state):
        """Load the context, or rebuild it, with this machine's threads and memory limits.

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
        super().__setstate__(state)
        if "context_" not in state:
            return  # Not fitted.
        fitted, limits = self.memory_plan_, self._limits()
        load = self.context_ is not None
        plan = _plan(self._task, fitted.estimators, fitted.rows, fitted.features, fitted.optimize, *limits, load)
        _enforce(plan, _overflow_policy(self.memory_overflow), stacklevel=2, action="load")
        self.memory_plan_ = plan
        if not Path(self.checkpoint_path_).exists():
            self.checkpoint_path_ = self._checkpoint(checkpoint)
        model = _model(self.checkpoint_path_)
        if load:
            self.context_ = model.context_from_bytes(self.context_, self._resolve_threads(), *limits)
        else:
            self.context_ = self._build(model, plan.optimize, limits, self.preprocessor_, self._training)

    def _limits(self) -> tuple[str | None, str | None]:
        return (
            _limit_text("memory_limit", self.memory_limit),
            _limit_text("device_memory_limit", self.device_memory_limit),
        )

    def _query_chunks(self, X):
        """Yield the preprocessed ``[B, rows, H]`` query rows of ``X``, ``PREDICT_CHUNK_ROWS`` at a
        time, each with the progress callback for its prediction (``None`` unless ``verbose``)."""
        rows = _num_samples(X)
        with self._progress("Predicting", "row", rows) as progress:
            for start in range(0, max(rows, 1), PREDICT_CHUNK_ROWS):
                chunk = _safe_indexing(X, slice(start, start + PREDICT_CHUNK_ROWS))
                report = progress and (lambda _stage, done, _total, start=start: progress.update(start + done))
                yield _ensemble_rows(self.preprocessor_, self.preprocessor_.transform(chunk)), report
                if progress:
                    progress.update(min(start + PREDICT_CHUNK_ROWS, rows))

    def _progress(self, description: str, unit: str, total: int | None = None):
        """A ``Progress`` for ``verbose=True``, else a context that yields ``None``."""
        return Progress(description, unit, total) if self.verbose else contextlib.nullcontext()

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
        methods = self.norm_methods
        if methods is not None:
            methods = (methods,) if isinstance(methods, str) else methods
            unsupported = sorted(set(methods) - {"none", "power"})
            if unsupported:
                raise NotImplementedError(
                    f"norm_methods {unsupported} are not supported; only 'none' and 'power'"
                )
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

    def _context(self, model, optimize, limits, train_rows, targets, class_perms, feature_perms):
        """Build the Rust context, or load it from ``cache_dir`` when available."""
        threads = self._resolve_threads()

        def build():
            with self._progress("Building context", "estimator") as progress:
                return model.build_context(
                    train_rows,
                    targets,
                    class_perms,
                    feature_perms,
                    optimize,
                    threads,
                    *limits,
                    progress=progress and (lambda _stage, done, total: progress.update(done, total)),
                )

        cache_dir = _cache_dir(self.cache_dir)
        if cache_dir is None:
            return build()
        key = self._cache_key(model, optimize, train_rows, targets, class_perms, feature_perms)
        path = cache_dir / f"{key}.irctx"
        if path.exists():
            try:
                context = model.load_context(str(path), threads, *limits)
            except (OSError, ValueError, RuntimeError):
                pass  # Corrupt, truncated, or mismatched cache entry: rebuild it.
            else:
                with contextlib.suppress(OSError):
                    os.utime(path)  # The entry's last use, for clear_cache.
                return context
        context = build()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            context.save(str(path))
        except OSError:
            pass  # Caching is best-effort; a write failure must not fail fit.
        return context

    def _cache_key(self, model, optimize, train_rows, targets, class_perms, feature_perms) -> str:
        """Content hash of everything the built context depends on."""
        digest = hashlib.sha256(b"ironicl-context-cache-v1\0")
        digest.update(model.source_sha256.encode("ascii"))
        digest.update(optimize.encode("ascii"))
        for array in (train_rows, targets, class_perms, feature_perms):
            array = np.ascontiguousarray(array)
            digest.update(f"{array.shape}{array.dtype.str}".encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

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


def _ensemble_rows(preprocessor: Preprocessor, groups: tuple[np.ndarray, ...]) -> np.ndarray:
    """Stack each estimator's feature-permuted group into ``[B, rows, H]``."""
    rows = [
        groups[group][:, preprocessor.feature_permutations_[e]]
        for e, group in enumerate(preprocessor.ensemble_groups_)
    ]
    return np.stack(rows).astype(np.float32, copy=False)

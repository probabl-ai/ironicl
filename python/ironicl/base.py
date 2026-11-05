"""What ``TabICLClassifier`` and ``TabICLRegressor`` share: the checkpoint, the
fitted context and its disk cache, threads, and ``optimize``."""

from __future__ import annotations

import hashlib
import os
import urllib.request
import warnings
from functools import cache
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator

from ironicl._core import Model
from ironicl.preprocessing import Preprocessor

CHECKPOINT_URL = "https://huggingface.co/jingang/TabICL/resolve/main/{}"


@cache  # ponytail: one resident model per checkpoint for the process lifetime.
def _model(checkpoint: Path) -> Model:
    return Model(checkpoint)


class TabICLBase(BaseEstimator):
    """Shared fit machinery. Subclasses set ``_task`` and ``_checkpoint_version``."""

    _task: str
    _checkpoint_version: str

    def _fit_context(self, X, y, targets, classification: bool) -> Preprocessor:
        """Fit preprocessing and build the Rust context from per-estimator ``targets(preprocessor)``.

        Fitted attributes are set only once the context exists, so a failed fit
        never looks fitted.
        """
        self._validate_params()
        optimize = self._optimize()
        methods = self.norm_methods or ["none", "power"]
        preprocessor = Preprocessor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            norm_methods=(methods,) if isinstance(methods, str) else tuple(methods),
            outlier_threshold=self.outlier_threshold,
            classification=classification,
        )
        train_rows = _ensemble_rows(preprocessor, preprocessor.fit_transform(X, y))
        checkpoint = self._checkpoint()
        model = _model(checkpoint)
        if model.task != self._task:
            raise ValueError(f"{self.checkpoint_version} is a {model.task} checkpoint, not a {self._task}")
        self.context_ = self._context(
            model,
            optimize,
            train_rows,
            np.ascontiguousarray(targets(preprocessor), dtype=np.float32),
            np.ascontiguousarray(preprocessor.class_permutations_, dtype=np.uint64),
            np.ascontiguousarray(preprocessor.feature_permutations_, dtype=np.uint64),
        )
        self.preprocessor_ = preprocessor
        self.checkpoint_path_ = checkpoint
        self.n_features_in_ = preprocessor.n_features_in_
        if hasattr(preprocessor, "feature_names_in_"):
            self.feature_names_in_ = preprocessor.feature_names_in_
        return preprocessor

    def _query_rows(self, X) -> np.ndarray:
        return _ensemble_rows(self.preprocessor_, self.preprocessor_.transform(X))

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
        for name, default in (("batch_size", 8), ("verbose", False), *noops):
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

    def _context(self, model, optimize, train_rows, targets, class_perms, feature_perms):
        """Build the Rust context, or load it from ``cache_dir`` when available."""
        threads = self._resolve_threads()

        def build():
            return model.build_context(train_rows, targets, class_perms, feature_perms, optimize, threads)

        cache_dir = self._cache_dir()
        if cache_dir is None:
            return build()
        key = self._cache_key(model, optimize, train_rows, targets, class_perms, feature_perms)
        path = cache_dir / f"{key}.irctx"
        if path.exists():
            try:
                return model.load_context(str(path), threads)
            except (OSError, ValueError, RuntimeError):
                pass  # Corrupt, truncated, or mismatched cache entry: rebuild it.
        context = build()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            context.save(str(path))
        except OSError:
            pass  # Caching is best-effort; a write failure must not fail fit.
        return context

    def _cache_dir(self) -> Path | None:
        """Resolve ``cache_dir`` HuggingFace-style.

        ``None``/``True`` use ``IRONICL_CACHE_DIR`` or ``<cache root>/ironicl/contexts``;
        ``False`` disables caching; a string or path is used directly.
        """
        cache_dir = self.cache_dir
        if cache_dir is False:
            return None
        if cache_dir is None or cache_dir is True:
            override = os.environ.get("IRONICL_CACHE_DIR")
            if override:
                return Path(override)
            root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            return root / "ironicl" / "contexts"
        return Path(cache_dir)

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

    def _checkpoint(self) -> Path:
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


def _ensemble_rows(preprocessor: Preprocessor, groups: list[np.ndarray]) -> np.ndarray:
    """Stack each estimator's feature-permuted group into ``[B, rows, H]``."""
    rows = [
        groups[group][:, preprocessor.feature_permutations_[e]]
        for e, group in enumerate(preprocessor.ensemble_groups_)
    ]
    return np.stack(rows).astype(np.float32, copy=False)

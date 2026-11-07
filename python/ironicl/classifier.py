"""A drop-in replacement for TabICL's ``TabICLClassifier``, without PyTorch.

The constructor and public methods match TabICL 2.1.1 (commit
bd030d47ff7d9f987f0f33431d743a3003bfe0d3). Values that would change results
but are not implemented yet raise ``NotImplementedError``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ironicl.base import TabICLBase


class TabICLClassifier(ClassifierMixin, TabICLBase):
    """TabICLv2 classifier with Ironicl's Rust inference engine.

    Parameters keep TabICL's names and defaults. ``n_jobs`` sets the executor
    thread count (``None`` leaves Ironicl's default; negative values follow
    TabICL, so ``-1`` is all logical cores). ``batch_size``, ``verbose``, and
    ``support_many_classes`` have no effect and warn if changed (the last because
    more than 10 classes are not supported yet). Parameters that request an
    unimplemented feature raise ``NotImplementedError``: ``use_amp=True``,
    ``use_fa3=True``, a non-default ``offload_mode``/``disk_offload_dir``,
    ``kv_cache="repr"``, ``norm_methods`` other than ``"none"``/``"power"``, and a
    non-default ``checkpoint_version``.

    ``optimize`` is Ironicl's own parameter, after TabICL's. It sets how far
    Ironicl may trade exactness for speed on Apple Silicon GPUs:

    - ``"safe"``: FP32 throughout; matches TabICL within 1e-4.
    - ``"auto"``: context keys and values stored in FP16 and attention against
      them computed in FP16; probabilities stay within about 1e-3 of ``"safe"``.
    - ``"fast"``: everything ``"auto"`` does, and later further reduced
      precision with probabilities within about 1e-2.

    ``None`` (default) reads the ``IRONICL_OPTIMIZE`` environment variable and
    falls back to ``"safe"``. Without a Metal GPU, every level behaves like
    ``"safe"``.

    ``cache_dir`` persists the built context so refitting the same data reuses it
    instead of rerunning the forward pass. ``False`` (default) disables it;
    ``None`` caches under ``IRONICL_CACHE_DIR`` if set, else
    ``<XDG_CACHE_HOME or ~/.cache>/ironicl/contexts``; a path caches there (each
    context is large, so caching is opt-in). Entries are keyed by the model,
    inputs, and ``optimize``, and a corrupt entry is silently rebuilt.

    ``memory_limit``, ``device_memory_limit``, and ``memory_overflow`` are
    Ironicl's own, after ``cache_dir``. ``fit`` plans the memory it needs after
    preprocessing and before building:

    - ``memory_limit``: the most host RAM to plan for, as an int number of bytes
      or a size such as ``"16G"`` or ``"512M"`` (binary units). It also covers a
      GPU that shares RAM, such as an Apple GPU, and only lowers what is
      detected. ``None`` (default) reads ``IRONICL_MEMORY_LIMIT``, else uses the
      memory available.
    - ``device_memory_limit``: the same for a GPU's own dedicated memory, from
      ``IRONICL_DEVICE_MEMORY_LIMIT``. It has no effect where memory is shared.
    - ``memory_overflow``: what to do when the plan exceeds available memory or
      ``memory_limit``: ``"warn"`` emits ``IroniclMemoryWarning`` and continues,
      ``"error"`` raises ``IroniclMemoryError`` and leaves the estimator
      unfitted, ``"ignore"`` continues silently. ``None`` (default) reads
      ``IRONICL_MEMORY_OVERFLOW``, else ``"warn"``. A fit that cannot complete
      on this machine at all (past a container's hard limit, or a context larger
      than physical memory) raises under ``"warn"`` too.

    The plan is kept as ``memory_plan_``; ``ironicl.plan_memory`` plans without
    data. Limits plan memory; the operating system does not enforce them.
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
        optimize: str | None = None,
        cache_dir: str | Path | bool | None = False,
        memory_limit: int | str | None = None,
        device_memory_limit: int | str | None = None,
        memory_overflow: str | None = None,
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
        self.optimize = optimize
        self.cache_dir = cache_dir
        self.memory_limit = memory_limit
        self.device_memory_limit = device_memory_limit
        self.memory_overflow = memory_overflow

    _task = "classifier"
    _checkpoint_version = "tabicl-classifier-v2-20260212.ckpt"

    def fit(self, X, y) -> TabICLClassifier:
        """Fit preprocessing and build the training context in Rust."""
        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")
        self._fit_context(
            X,
            y,
            lambda preprocessor: preprocessor.class_permutations_[:, preprocessor.encode_labels(y)],
            classification=True,
        )
        self.classes_ = self.preprocessor_.classes_
        self.n_classes_ = len(self.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities with shape ``(n_samples, n_classes)``."""
        check_is_fitted(self)
        values, shape = self.context_.predict_proba(
            self._query_rows(X),
            self.softmax_temperature,
            self.average_logits,
        )
        return np.frombuffer(values, dtype=np.float32).reshape(shape).copy()  # Writable, like TabICL.

    def predict(self, X) -> np.ndarray:
        """Return the most probable class label for each sample."""
        probabilities = self.predict_proba(X)  # Checks that the classifier is fitted.
        return self.preprocessor_.decode_labels(np.argmax(probabilities, axis=1))

    def _validate_params(self) -> None:
        super()._validate_params(
            unsupported=[("class_shuffle_method", ("shift",))],
            noops=[("support_many_classes", True)],
        )

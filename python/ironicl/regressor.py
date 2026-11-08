"""A drop-in replacement for TabICL's ``TabICLRegressor``, without PyTorch.

The constructor and public methods match TabICL 2.1.1 (commit
bd030d47ff7d9f987f0f33431d743a3003bfe0d3). Values that would change results
but are not implemented yet raise ``NotImplementedError``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ironicl.base import TabICLBase

OUTPUT_TYPES = ("mean", "variance", "median", "quantiles", "raw_quantiles")
DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TabICLRegressor(RegressorMixin, TabICLBase):
    """TabICLv2 regressor with Ironicl's Rust inference engine.

    Parameters keep TabICL's names and defaults, and behave as in
    ``TabICLClassifier``: ``n_jobs`` sets the thread count, ``batch_size`` and
    ``verbose`` have no effect and warn if changed, and parameters that request
    an unimplemented feature raise ``NotImplementedError``. ``optimize``,
    ``cache_dir``, ``memory_limit``, ``device_memory_limit``, and ``memory_overflow``
    are Ironicl's own trailing parameters; see ``TabICLClassifier``.

    The model predicts 999 quantiles of the standardized target per row.
    ``predict`` turns them into the requested statistics exactly as TabICL does,
    in Rust, with one difference: ``"variance"`` is scaled back by the target's
    variance, ``σ²``. TabICL applies ``σ·v + μ`` to it, which is only right for
    locations such as the mean, so its variance is wrong unless the target is
    already standardized (https://github.com/soda-inria/tabicl/issues/180).
    """

    _task = "regressor"
    _checkpoint_version = "tabicl-regressor-v2-20260212.ckpt"

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: int | None = 8,
        kv_cache: bool | str = False,
        model_path: str | Path | None = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt",
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
        self.outlier_threshold = outlier_threshold
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

    def fit(self, X, y) -> TabICLRegressor:
        """Standardize the target, fit preprocessing, and build the training context in Rust."""
        if y is None:
            raise ValueError("This regressor requires y to be passed, but the target y is None.")
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. Please change "
                "the shape of y to (n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
            y = y.ravel()
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError("y must be a one-dimensional array of finite numbers")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()
        self._fit_context(
            X,
            y,
            lambda _: scaled,
            classification=False,
        )
        self.y_scaler_ = scaler
        return self

    def _expand_targets(self, preprocessor, targets):
        return np.tile(targets, (preprocessor.n_estimators_, 1))

    def predict(self, X, output_type: str | list[str] = "mean", alphas: list[float] | None = None):
        """Predict target statistics for each sample.

        ``output_type`` is ``"mean"``, ``"median"``, or ``"variance"`` (shape
        ``(n_samples,)``), ``"quantiles"`` at ``alphas`` (default 0.1 to 0.9;
        shape ``(n_samples, len(alphas))``), ``"raw_quantiles"`` (the 999 sorted
        model quantiles), or a list of these, which returns a dict.
        """
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and X.ndim == 1:
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")
        types = [output_type] if isinstance(output_type, str) else list(output_type)
        alphas = DEFAULT_ALPHAS if alphas is None else list(alphas)
        chunks = [self.context_.predict_regression(rows, types, alphas) for rows in self._query_chunks(X)]
        mean, scale = self.y_scaler_.mean_[0], self.y_scaler_.scale_[0]
        final = {}
        for name, *parts in zip(types, *chunks):
            values = np.concatenate([np.frombuffer(values, dtype=np.float32).reshape(shape) for values, shape in parts])
            # Standardized units back to the target's: locations by σ·v + μ,
            # the variance, a squared spread, by σ².
            values = values * scale**2 if name == "variance" else values * scale + mean
            values = values.astype(np.float32)
            final[name] = values.ravel() if name in ("mean", "variance", "median") else values
        return final[types[0]] if len(types) == 1 else final

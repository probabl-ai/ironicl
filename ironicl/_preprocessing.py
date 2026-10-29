"""Preprocessing and ensemble generation for TabICLv2, in numpy and scikit-learn.

This module is a port of ``tabicl/_sklearn/preprocessing.py`` from TabICL 2.2.0
(BSD-3-Clause, Copyright (c) the TabICL authors). It keeps the upstream
transformers, parameter names, fitted attributes, random-number streams and
floating-point order, so ``IronICLClassifier`` produces the same ensemble
variants as ``tabicl.TabICLClassifier`` without importing tabicl or torch.

The estimator validates its input to a finite float32 array before anything here
runs, so upstream's ``TransformToNumerical`` (which ordinal-encodes DataFrame
columns) is not needed: strings, dates and missing values belong to a skrub
``TableVectorizer`` or ``tabular_pipeline`` step in front of the estimator.
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted, validate_data

HF_REPO_ID = "jingang/TabICL"
CHECKPOINT_VERSIONS = (
    "tabicl-classifier-v1-20250208.ckpt",
    "tabicl-classifier-v1.1-20250506.ckpt",
    "tabicl-classifier-v2-20260212.ckpt",
)
NORM_METHODS = ("none", "power", "quantile", "quantile_rtdl", "robust")
SHUFFLE_METHODS = ("none", "shift", "random", "latin")


class RecursionLimitManager:
    """Temporarily raise the recursion limit (Latin squares recurse per element)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.original_limit: int | None = None

    def __enter__(self) -> RecursionLimitManager:  # noqa: D105
        self.original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: D105
        if self.original_limit is not None:
            sys.setrecursionlimit(self.original_limit)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
    """Drop features with at most ``threshold`` distinct training values.

    With ``n_samples <= threshold`` every feature is kept, and if every feature
    would be dropped the first one is kept so the output is never empty.
    """

    def __init__(self, threshold: int = 1) -> None:
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn which features to keep."""
        X = validate_data(self, X)
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            self.features_to_keep_ = np.array(
                [
                    len(np.unique(X[:, i])) > self.threshold
                    for i in range(self.n_features_in_)
                ]
            )
            if not self.features_to_keep_.any():
                self.features_to_keep_[0] = True
        self.n_features_out_ = np.sum(self.features_to_keep_)
        return self

    def transform(self, X):
        """Keep the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
    """Soft-clip values beyond ``threshold`` standard deviations.

    Bounds come from a two-pass z-score: values outside the first-pass bounds
    are ignored when the statistics are recomputed. ``transform`` clips with
    ``log1p`` so extreme values keep their ordering.
    """

    def __init__(self, threshold: float = 4.0) -> None:
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn the clipping bounds."""
        X = validate_data(self, X)
        ddof = 1 if X.shape[0] > 1 else 0
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0, ddof=ddof)
        self.stds_ = np.maximum(self.stds_, 1e-6)
        X_clean = X.copy()
        lower_bounds = self.means_ - self.threshold * self.stds_
        upper_bounds = self.means_ + self.threshold * self.stds_
        lower_mask = lower_bounds[np.newaxis, :] > X
        upper_mask = upper_bounds[np.newaxis, :] < X
        X_clean[np.logical_or(lower_mask, upper_mask)] = np.nan
        self.means_ = np.nanmean(X_clean, axis=0)
        self.stds_ = np.nanstd(X_clean, axis=0, ddof=ddof)
        self.stds_ = np.maximum(self.stds_, 1e-6)
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_
        return self

    def transform(self, X):
        """Clip to the learned bounds."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)
        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Z-score with an epsilon on the scale and clipping to ``[clip_min, clip_max]``."""

    def __init__(
        self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6
    ) -> None:
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the per-feature mean and scale."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = validate_data(self, X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon
        return self

    def transform(self, X):
        """Standardise and clip."""
        is_vector = len(X.shape) == 1
        if is_vector:
            X = X.reshape(-1, 1)
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_clipped = np.clip(
            (X - self.mean_) / self.scale_, self.clip_min, self.clip_max
        )
        return X_clipped.reshape(-1) if is_vector else X_clipped

    def inverse_transform(self, X):
        """Undo the standardisation."""
        is_vector = len(X.shape) == 1
        if is_vector:
            X = X.reshape(-1, 1)
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_out = X * self.scale_ + self.mean_
        return X_out.reshape(-1) if is_vector else X_out


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    """Quantile transform with training noise, after the RTDL tabular DL recipe.

    The number of quantiles is ``max(min(n_samples // 30, n_quantiles), 10)``
    and Gaussian noise scaled by ``noise / max(std, noise)`` is added before
    fitting.
    """

    def __init__(
        self,
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
        random_state: int | None = None,
    ) -> None:
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the quantile transformer on (noisy) training data."""
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        X_modified = self._add_noise(X) if self.noise > 0 else X
        normalizer.fit(X_modified)
        self.normalizer_ = normalizer
        return self

    def transform(self, X, y=None):
        """Apply the fitted quantile transform."""
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        return X + noise_std * rng.standard_normal(X.shape)


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Standard scaling, one normalisation method, then outlier clipping.

    ``normalization_method`` is one of ``'none'``, ``'power'`` (Yeo-Johnson),
    ``'quantile'``, ``'quantile_rtdl'`` or ``'robust'``. The transformed
    training data is kept in ``X_transformed_``.
    """

    def __init__(
        self,
        normalization_method: str = "power",
        outlier_threshold: float = 4.0,
        random_state: int | None = None,
    ) -> None:
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit every stage on the training data."""
        X = validate_data(self, X)
        if hasattr(X, "dtype") and X.dtype == np.float16:
            X = X.astype(np.float32)
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(
                    method="yeo-johnson", standardize=True
                )
            elif self.normalization_method == "quantile":
                # sklearn clamps n_quantiles to n_samples (with a warning); pass
                # the clamped value so the result is identical and silent.
                self.normalizer_ = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, X_scaled.shape[0]),
                    random_state=self.random_state,
                )
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        (
                            "quantile_rtdl",
                            RTDLQuantileTransformer(
                                output_distribution="normal",
                                random_state=self.random_state,
                            ),
                        ),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.normalization_method}"
                )
            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)
        return self

    def transform(self, X):
        """Apply the fitted stages."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=True)
        if hasattr(X, "dtype") and X.dtype == np.float16:
            X = X.astype(np.float32)
        X = self.standard_scaler_.transform(X)
        if self.normalizer_ is not None:
            try:
                X = self.normalizer_.transform(X)
            except ValueError:
                # An unseen extreme value can break the normaliser: clip to the
                # training range first, as upstream does.
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)
        return self.outlier_remover_.transform(X)


class Shuffler:
    """Permutations for ensemble members.

    ``method`` is ``'none'``, ``'shift'``, ``'random'`` or ``'latin'``.

    Latin squares fall back to random permutations above
    ``max_elements_for_latin`` elements. The ``random.Random`` stream and call
    order match upstream so the same seed gives the same permutations.
    """

    def __init__(
        self,
        n_elements: int,
        method: str = "latin",
        max_elements_for_latin: int = 4000,
        random_state: int | None = None,
    ) -> None:
        self.n_elements = n_elements
        self.method = method
        self.max_elements_for_latin = max_elements_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> list[list[int]]:
        """Return ``n_estimators`` permutations.

        ``'shift'`` returns one per element and ``'none'`` a single identity.
        """
        self.rng_ = random.Random(self.random_state)
        indices = list(range(self.n_elements))
        if self.n_elements > self.max_elements_for_latin and self.method == "latin":
            method = "random"
        else:
            method = self.method
        if method == "none" or n_estimators == 1:
            return [indices]
        if method == "shift":
            return [indices[-i:] + indices[:-i] for i in range(self.n_elements)]
        if method == "random":
            if self.n_elements <= 5:
                all_perms = [list(perm) for perm in itertools.permutations(indices)]
                return self.rng_.sample(all_perms, min(n_estimators, len(all_perms)))
            return [
                self.rng_.sample(indices, self.n_elements) for _ in range(n_estimators)
            ]
        if method == "latin":
            with RecursionLimitManager(100000):
                return self._latin_squares()
        raise ValueError(
            f"Unknown method: {method}. Use 'shift', 'random', 'latin', or 'none'."
        )

    def _latin_squares(self) -> list[list[int]]:
        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix)
            self.rng_.shuffle(square)
            trans = list(zip(*square, strict=True))
            self.rng_.shuffle(trans)
            return trans

        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols]
            sym = self.rng_.choice(symbols)
            symbols.remove(sym)
            square = _rls(symbols)
            square.append(square[0].copy())
            for i in range(n):
                square[i].insert(i, sym)
            return square

        square = _rls(list(range(self.n_elements)))
        return [list(shuffle) for shuffle in _shuffle_transpose_shuffle(square)]


class EnsembleGenerator(TransformerMixin, BaseEstimator):
    """Ensemble variants of a training table.

    Members combine a normalisation method, a feature permutation and, for
    classification, a class permutation.

    Fitted attributes follow upstream: ``unique_filter_``, ``preprocessors_``
    (one :class:`PreprocessingPipeline` per normalisation method),
    ``ensemble_configs_``, ``feature_shuffles_`` and ``class_shuffles_`` keyed by
    normalisation method, ``X_``, ``y_``, ``n_features_in_`` (after filtering)
    and ``n_classes_``.
    """

    def __init__(
        self,
        classification: bool,
        n_estimators: int,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        random_state: int | None = None,
    ) -> None:
        self.classification = classification
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        if class_shuffle_method not in SHUFFLE_METHODS:
            raise ValueError(
                f"class_shuffle_method must be one of {SHUFFLE_METHODS}; "
                f"got {class_shuffle_method!r}"
            )
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Filter constant features, draw the ensemble and fit the preprocessors."""
        validate_data(self, X, y)
        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        elif isinstance(self.norm_methods, str):
            self.norm_methods_ = [self.norm_methods]
        else:
            self.norm_methods_ = list(self.norm_methods)
        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        if self.classification:
            self.n_classes_ = len(np.unique(y))
        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_, self.feature_shuffles_, y_patterns = (
            self._generate_ensemble()
        )
        if self.classification:
            self.class_shuffles_ = y_patterns
        self.preprocessors_ = {}
        for norm_method in self.ensemble_configs_:
            if norm_method not in self.preprocessors_:
                preprocessor = PreprocessingPipeline(
                    normalization_method=norm_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=self.random_state,
                )
                preprocessor.fit(X)
                self.preprocessors_[norm_method] = preprocessor
        return self

    def _generate_ensemble(self):
        feat_shuffler = Shuffler(
            n_elements=self.n_features_in_,
            method=self.feat_shuffle_method,
            random_state=self.random_state,
        )
        X_shuffles = feat_shuffler.shuffle(self.n_estimators)
        if self.classification:
            class_shuffler = Shuffler(
                n_elements=self.n_classes_,
                method=self.class_shuffle_method,
                random_state=self.random_state,
            )
            y_patterns = class_shuffler.shuffle(self.n_estimators)
        else:
            y_patterns = [None]
        shuffle_configs = list(itertools.product(X_shuffles, y_patterns))
        self.rng_.shuffle(shuffle_configs)
        shuffle_norm_configs = list(
            itertools.product(shuffle_configs, self.norm_methods_)
        )
        shuffle_norm_configs = shuffle_norm_configs[: self.n_estimators]
        # Upstream groups members with ``list(set(...))``, whose order depends on
        # the string hash seed. Members are averaged, so the order never affects
        # predictions; the context stores whatever order this process draws.
        used_methods = list({config[1] for config in shuffle_norm_configs})
        ensemble_configs = OrderedDict()
        X_shuffle_dict = OrderedDict()
        y_pattern_dict = OrderedDict()
        for method in used_methods:
            configs = [
                config[0] for config in shuffle_norm_configs if config[1] == method
            ]
            X_shuffle_dict[method] = [config[0] for config in configs]
            y_pattern_dict[method] = [config[1] for config in configs]
            ensemble_configs[method] = configs
        return ensemble_configs, X_shuffle_dict, y_pattern_dict

    def transform(self, X=None, mode="both"):
        """Return ensemble variants per normalisation method.

        ``mode='train'`` returns ``(X_train[B_m, N, H'], y[B_m, N])`` and needs no
        ``X``; ``mode='test'`` returns ``(X_test[B_m, Q, H'],)``; ``mode='both'``
        returns the training and test rows concatenated along the row axis.
        """
        check_is_fitted(self, ["ensemble_configs_"])
        if mode not in ("both", "train", "test"):
            raise ValueError(f"Invalid mode: {mode}")
        if mode == "train":
            y = self.y_
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                X_preprocessed = self.preprocessors_[norm_method].X_transformed_
                X_ensemble, y_ensemble = [], []
                for feat_shuffle, y_pattern in shuffle_configs:
                    X_ensemble.append(X_preprocessed[:, feat_shuffle])
                    if self.classification:
                        y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                    else:
                        y_ensemble.append(y)
                data[norm_method] = (
                    np.stack(X_ensemble, axis=0),
                    np.stack(y_ensemble, axis=0),
                )
            return data
        if X is None:
            raise ValueError("X is required when mode is 'test' or 'both'")
        X = self.unique_filter_.transform(X)
        if mode == "test":
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                X_test_preprocessed = self.preprocessors_[norm_method].transform(X)
                data[norm_method] = (
                    np.stack(
                        [X_test_preprocessed[:, s] for s, _ in shuffle_configs], axis=0
                    ),
                )
            return data
        y = self.y_
        data = OrderedDict()
        for norm_method, shuffle_configs in self.ensemble_configs_.items():
            preprocessor = self.preprocessors_[norm_method]
            X_variant = np.concatenate(
                [preprocessor.X_transformed_, preprocessor.transform(X)], axis=0
            )
            X_ensemble, y_ensemble = [], []
            for feat_shuffle, y_pattern in shuffle_configs:
                X_ensemble.append(X_variant[:, feat_shuffle])
                if self.classification:
                    y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                else:
                    y_ensemble.append(y)
            data[norm_method] = (
                np.stack(X_ensemble, axis=0),
                np.stack(y_ensemble, axis=0),
            )
        return data


def training_variants(
    generator: EnsembleGenerator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the engine's inputs for the training table.

    ``(x[B, N, H'] float32, y[B, N] float32, class_permutations[B, C] uint64,
    feature_permutations[B, H'] uint64)``, concatenated over normalisation
    methods in ``ensemble_configs_`` order.
    """
    train = generator.transform(X=None, mode="train")
    x = np.ascontiguousarray(
        np.concatenate([features for features, _ in train.values()]), dtype=np.float32
    )
    y = np.ascontiguousarray(
        np.concatenate([labels for _, labels in train.values()]), dtype=np.float32
    )
    classes = np.concatenate(
        [np.asarray(generator.class_shuffles_[method]) for method in train]
    )
    features = np.concatenate(
        [np.asarray(generator.feature_shuffles_[method]) for method in train]
    )
    return (
        x,
        y,
        np.ascontiguousarray(classes, dtype=np.uint64),
        np.ascontiguousarray(features, dtype=np.uint64),
    )


def query_variants(generator: EnsembleGenerator, X: np.ndarray) -> np.ndarray:
    """Encode raw query rows into the ``[B, Q, H']`` float32 engine variants."""
    X = np.asarray(X, dtype=np.float32, order="C")
    groups = generator.transform(X, mode="test")
    variants = np.ascontiguousarray(
        np.concatenate([group[0] for group in groups.values()]), dtype=np.float32
    )
    if variants.shape[1] != X.shape[0]:
        raise ValueError("preprocessing produced unexpected ensemble variants")
    return variants


def resolve_checkpoint(
    model_path: str | Path | None, checkpoint_version: str, allow_auto_download: bool
) -> Path:
    """Locate or download the TabICLv2 checkpoint, as ``tabicl.TabICLClassifier`` does.

    With ``model_path=None`` the file named ``checkpoint_version`` is read from,
    or downloaded to, the Hugging Face cache. With a path that does not exist,
    the checkpoint is downloaded next to it and renamed. Downloads require
    ``allow_auto_download``.
    """
    if checkpoint_version not in CHECKPOINT_VERSIONS:
        raise ValueError(
            f"Invalid checkpoint version {checkpoint_version!r}. "
            f"Available ones are: {', '.join(CHECKPOINT_VERSIONS)}."
        )
    if model_path is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        try:
            return Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=checkpoint_version,
                    local_files_only=True,
                )
            )
        except LocalEntryNotFoundError:
            if not allow_auto_download:
                raise ValueError(
                    f"Checkpoint {checkpoint_version!r} is not cached and automatic "
                    "download is disabled. Set allow_auto_download=True to download it "
                    f"from Hugging Face Hub ({HF_REPO_ID})."
                ) from None
            return Path(
                hf_hub_download(repo_id=HF_REPO_ID, filename=checkpoint_version)
            )
    path = Path(model_path)
    if path.exists():
        return path
    if not allow_auto_download:
        raise ValueError(
            f"Checkpoint not found at {str(path)!r} and automatic download is "
            "disabled. Provide a valid checkpoint path, or set "
            "allow_auto_download=True to download "
            f"{checkpoint_version!r} from Hugging Face Hub ({HF_REPO_ID})."
        )
    from huggingface_hub import hf_hub_download

    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID, filename=checkpoint_version, local_dir=path.parent
    )
    Path(downloaded).rename(path)
    return path

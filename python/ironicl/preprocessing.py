"""PyTorch-free classifier preprocessing with grouped, unpermuted FP32 output.

The scaling, outlier handling, and ensemble algorithms are adapted from TabICL
2.1.1, commit bd030d47ff7d9f987f0f33431d743a3003bfe0d3.
Copyright (c) 2025, Soda team @ Inria. See LICENSES/TabICL-BSD-3-Clause.txt.

Ironicl limits this implementation to finite numeric features, classification,
and the ``none``/``power`` normalization policies. Unlike upstream, it retains
no training arrays, uses deterministic normalization-group order, and returns
one matrix per group instead of expanding every ensemble member. Latin-square
construction is iterative and does not change the process recursion limit.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted


@dataclass
class _Normalization:
    mean: np.ndarray
    scale: np.ndarray
    power: PowerTransformer | None
    minimum: np.ndarray
    maximum: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, method: str, threshold: float) -> _Normalization:
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0) + 1e-6
        scaled = np.clip((x - mean) / scale, -100, 100)
        minimum, maximum = scaled.min(axis=0), scaled.max(axis=0)
        power = PowerTransformer(method="yeo-johnson", standardize=True) if method == "power" else None
        normalized = power.fit_transform(scaled) if power is not None else scaled

        # Preserve upstream's sample-std, two-pass outlier bounds.
        center = np.nanmean(normalized, axis=0)
        std = np.maximum(np.nanstd(normalized, axis=0, ddof=1), 1e-6)
        cleaned = normalized.copy()
        cleaned[(normalized < center - threshold * std) | (normalized > center + threshold * std)] = np.nan
        center = np.nanmean(cleaned, axis=0)
        std = np.maximum(np.nanstd(cleaned, axis=0, ddof=1), 1e-6)
        lower, upper = center - threshold * std, center + threshold * std
        if not all(np.isfinite(values).all() for values in (mean, scale, minimum, maximum, lower, upper)):
            raise ValueError("Training data produced non-finite preprocessing statistics")
        return cls(mean, scale, power, minimum, maximum, lower, upper)

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.clip((x - self.mean) / self.scale, -100, 100)
        if self.power is not None:
            try:
                values = self.power.transform(values)
            except ValueError:
                # Match upstream's fallback for values outside the fitted domain.
                values = self.power.transform(np.clip(values, self.minimum, self.maximum))
        values = np.maximum(-np.log1p(np.abs(values)) + self.lower, values)
        values = np.minimum(np.log1p(np.abs(values)) + self.upper, values)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessing produced non-finite FP32 values")
        return values


def _feature_shuffles(features: int, estimators: int, seed: int) -> list[list[int]]:
    """Match upstream's default Latin shuffle, including its large-H fallback."""
    indices = list(range(features))
    if estimators == 1:
        return [indices]
    rng = random.Random(seed)
    if features > 4000:
        return [rng.sample(indices, features) for _ in range(estimators)]

    removed = []
    while len(indices) > 1:
        symbol = rng.choice(indices)
        indices.remove(symbol)
        removed.append(symbol)
    square = [indices]
    for symbol in reversed(removed):
        square.append(square[0].copy())
        for index, row in enumerate(square):
            row.insert(index, symbol)
    rng.shuffle(square)
    transposed = list(zip(*square))
    rng.shuffle(transposed)
    return [list(row) for row in transposed]


def _indices(values) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.uint64)
    result.flags.writeable = False
    return result


def _features(x) -> tuple[np.ndarray, tuple[str, ...] | None]:
    names = None
    if hasattr(x, "columns"):
        names = tuple(x.columns)
        if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("DataFrame feature names must be unique strings")
    if np.asarray(x).dtype.kind not in "biuf":
        raise TypeError(
            "X must contain dense, real numeric features; categorical and sparse inputs are unsupported"
        )
    return check_array(x, dtype=[np.float64, np.float32]), names


def _labels(y) -> np.ndarray:
    values = check_array(y, ensure_2d=False, dtype=None)
    if values.ndim != 1:
        raise ValueError("y must be a one-dimensional array of class labels")
    check_classification_targets(values)
    return values


class ClassifierPreprocessor(BaseEstimator):
    """Fit TabICLv2's default numeric classification preprocessing without torch.

    Inputs must be finite, dense numeric matrices with at least two training
    rows and 2–10 classes. Categorical values, missing values, and all-constant
    training features are rejected. Named inputs require the same feature names
    and order at transform time.

    ``transform`` returns a tuple of C-contiguous FP32 matrices ``[rows, H]``,
    ordered by ``group_names_``. ``ensemble_groups_`` maps each ensemble member
    to one matrix; ``feature_permutations_`` and ``class_permutations_`` describe
    its feature order and encoded-label mapping. Permutations refer to features
    *after* constant-feature filtering. These uint64 metadata arrays are read-only.

    No ensemble feature matrices or repeated label arrays are materialized.
    Neither raw nor transformed training rows are retained after fitting.
    ``n_estimators_`` records the actual ensemble count, which may be smaller
    than the requested ``n_estimators``.

    Parameters
    ----------
    n_estimators : int, default=8
        Maximum ensemble count, in 1..4096.
    random_state : int, default=0
        Seed for deterministic feature and class ensemble generation.
    norm_methods : tuple of str, default=("none", "power")
        Unique normalization methods in canonical group order. Only ``none``
        and ``power`` are supported. Unused groups are not fitted or returned.
    outlier_threshold : float, default=4.0
        Positive, finite threshold for upstream's two-pass outlier transform.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        random_state: int = 0,
        norm_methods: tuple[str, ...] = ("none", "power"),
        outlier_threshold: float = 4.0,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.norm_methods = norm_methods
        self.outlier_threshold = outlier_threshold

    def fit(self, X, y):
        """Fit training-only state. Failed fitting leaves previous state intact."""
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, Integral)
            or not 1 <= self.n_estimators <= 4096
        ):
            raise ValueError("n_estimators must be an integer in 1..4096")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral):
            raise ValueError("random_state must be an integer")
        methods = self.norm_methods
        if (
            not isinstance(methods, (tuple, list))
            or not methods
            or any(method not in ("none", "power") for method in methods)
            or len(set(methods)) != len(methods)
        ):
            raise ValueError("norm_methods must be a nonempty sequence of unique 'none'/'power' methods")
        if (
            isinstance(self.outlier_threshold, bool)
            or not isinstance(self.outlier_threshold, Real)
            or not np.isfinite(self.outlier_threshold)
            or self.outlier_threshold <= 0
        ):
            raise ValueError("outlier_threshold must be positive and finite")

        x, names = _features(X)
        labels = _labels(y)
        if len(x) != len(labels) or len(x) < 2:
            raise ValueError("X and y must have the same length and at least two training rows")
        encoder = LabelEncoder().fit(labels)
        classes = len(encoder.classes_)
        if not 2 <= classes <= 10:
            raise ValueError("The classifier requires 2..10 classes")
        # Inputs are finite: comparing to the first row is equivalent to counting
        # unique values, without sorting every feature.
        mask = np.any(x != x[0], axis=0)
        if not mask.any():
            raise ValueError("Training data must contain at least one nonconstant feature")
        filtered = x[:, mask]
        seed = int(self.random_state)
        features = _feature_shuffles(filtered.shape[1], int(self.n_estimators), seed)
        indices = list(range(classes))
        permutations = (
            [indices] if self.n_estimators == 1 else [indices[-i:] + indices[:-i] for i in range(classes)]
        )
        combinations = list(itertools.product(features, permutations))
        random.Random(seed).shuffle(combinations)
        selected = list(itertools.islice(itertools.product(combinations, methods), int(self.n_estimators)))
        group_names = tuple(method for method in methods if any(name == method for _, name in selected))
        groups, feature_orders, class_orders = [], [], []
        for group, method in enumerate(group_names):
            for (feature_order, class_order), name in selected:
                if name == method:
                    groups.append(group)
                    feature_orders.append(feature_order)
                    class_orders.append(class_order)
        normalizers = tuple(
            _Normalization.fit(filtered, method, self.outlier_threshold) for method in group_names
        )
        mask.flags.writeable = False
        state = {
            "n_features_in_": x.shape[1],
            "n_features_out_": filtered.shape[1],
            "feature_mask_": mask,
            "classes_": encoder.classes_,
            "label_encoder_": encoder,
            "group_names_": group_names,
            "ensemble_groups_": _indices(groups),
            "feature_permutations_": _indices(feature_orders),
            "class_permutations_": _indices(class_orders),
            "n_estimators_": len(groups),
            "normalizers_": normalizers,
        }
        if names is not None:
            state["feature_names_in_"] = np.asarray(names, dtype=object)
        for key in tuple(self.__dict__):
            if key.endswith("_"):
                del self.__dict__[key]
        self.__dict__.update(state)
        return self

    def transform(self, X) -> tuple[np.ndarray, ...]:
        """Prepare each normalization group once, without ensemble expansion."""
        check_is_fitted(self, "normalizers_")
        x, names = _features(X)
        expected_names = tuple(self.feature_names_in_) if hasattr(self, "feature_names_in_") else None
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features; expected {self.n_features_in_}")
        if names != expected_names:
            raise ValueError("X must have the same feature names and order as the training data")
        filtered = x[:, self.feature_mask_]
        return tuple(normalizer.transform(filtered) for normalizer in self.normalizers_)

    def fit_transform(self, X, y) -> tuple[np.ndarray, ...]:
        """Fit and return grouped training features without retaining them."""
        return self.fit(X, y).transform(X)

    def encode_labels(self, y) -> np.ndarray:
        """Encode known class labels once as a contiguous int64 vector."""
        check_is_fitted(self, "label_encoder_")
        return np.ascontiguousarray(self.label_encoder_.transform(_labels(y)), dtype=np.int64)

    def decode_labels(self, indices) -> np.ndarray:
        """Map a one-dimensional vector of class indices to original labels."""
        check_is_fitted(self, "label_encoder_")
        values = np.asarray(indices)
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError("Class indices must be a one-dimensional integer array")
        return self.label_encoder_.inverse_transform(values)

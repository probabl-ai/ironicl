//! Column statistics and scaling, in `f64`, one column at a time. From
//! `TabICL` 2.1.1: `CustomStandardScaler` (standardize, then clip to ±100) and
//! `OutlierRemover` (two-pass sigma bounds, then a log-tail clamp). From
//! scikit-learn 1.9: `StandardScaler`'s statistics (`sklearn/preprocessing/_data.py`,
//! `sklearn/utils/extmath.py`).
//! Copyright (c) 2025, Soda team @ Inria; see `LICENSES/TabICL-BSD-3-Clause.txt`.
//! Copyright (c) 2007-2026 the scikit-learn developers; see
//! `LICENSES/scikit-learn-BSD-3-Clause.txt`.

#![expect(clippy::cast_precision_loss, reason = "Counts stay far below 2^53")]

/// The mean and the variance with `ddof` delta degrees of freedom, in two passes.
pub(crate) fn mean_variance(values: &[f64], ddof: usize) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let squares: f64 = values.iter().map(|value| (value - mean) * (value - mean)).sum();
    (mean, squares / (values.len() - ddof) as f64)
}

/// scikit-learn's `_is_constant_feature`: a variance within the error bound of
/// the two-pass algorithm.
///
/// Port of <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_data.py#L83-L98>.
pub(crate) fn is_constant(variance: f64, mean: f64, rows: usize) -> bool {
    let n = rows as f64;
    let bound = n * mean * f64::EPSILON;
    variance <= n * f64::EPSILON * variance + bound * bound
}

/// `StandardScaler().fit`'s mean and scale: the corrected two-pass variance,
/// with unit scale for a constant column.
///
/// Port of
/// <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_data.py#L931-L1090>,
/// <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/utils/extmath.py#L1118-L1264>,
/// and <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_data.py#L101-L133>.
pub(crate) fn standard_scaler(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let (correction, squares) = values.iter().fold((0.0, 0.0), |(sum, squares), value| {
        let deviation = value - mean;
        (sum + deviation, squares + deviation * deviation)
    });
    let variance = (squares - correction * correction / n) / n;
    let scale = if is_constant(variance, mean, values.len()) {
        1.0
    } else {
        variance.sqrt()
    };
    (mean, scale)
}

/// `CustomStandardScaler`'s mean and scale: the population standard deviation
/// plus 1e-6.
///
/// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L421-L447>.
pub(crate) fn standard_clip(values: &[f64]) -> (f64, f64) {
    let (mean, variance) = mean_variance(values, 0);
    (mean, variance.sqrt() + 1e-6)
}

/// `CustomStandardScaler.transform` of one value: standardized, then clipped to ±100.
///
/// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L449-L474>.
pub(crate) fn clip_standardized(value: f64, mean: f64, scale: f64) -> f64 {
    ((value - mean) / scale).clamp(-100.0, 100.0)
}

/// `OutlierRemover`'s bounds, `center ± threshold·std` from the sample standard
/// deviation, refitted once on the values within the first bounds. NaN when
/// fewer than two values remain, as `NumPy`'s `nanvar(ddof=1)` gives.
///
/// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L306-L355>.
pub(crate) fn outlier_bounds(values: &[f64], threshold: f64) -> (f64, f64) {
    let bounds = |values: &[f64]| {
        if values.len() < 2 {
            return (f64::NAN, f64::NAN);
        }
        let (center, variance) = mean_variance(values, 1);
        let spread = threshold * variance.sqrt().max(1e-6);
        (center - spread, center + spread)
    };
    let (lower, upper) = bounds(values);
    let inside: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| (lower..=upper).contains(value))
        .collect();
    bounds(&inside)
}

/// `OutlierRemover.transform` of one value: pulled back to within
/// `log(1 + |x|)` of the bounds.
///
/// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L357-L385>.
pub(crate) fn clamp_outlier(value: f64, lower: f64, upper: f64) -> f64 {
    let low = value.max(lower - value.abs().ln_1p());
    low.min(upper + low.abs().ln_1p())
}

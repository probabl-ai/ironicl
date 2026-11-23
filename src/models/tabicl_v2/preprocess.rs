//! `TabICLv2`'s preprocessing, fitted and applied in Rust.
//!
//! It reproduces `TabICL` 2.1.1 (commit bd030d47ff7d9f987f0f33431d743a3003bfe0d3)
//! for numeric and categorical features and the `none` and `power`
//! normalizations: categorical features come first, as their category codes
//! (see `crate::preprocess::columns`), missing numeric values (NaN) are
//! replaced by their feature's training mean, constant features are dropped,
//! each normalization method is fitted once as a group, and each ensemble
//! member is a group, a feature order, and a class order. The ensemble comes from `CPython`'s `random.Random(random_state)`, so
//! it matches `TabICL` member for member.
//!
//! Statistics are fitted and applied in `f64`, whatever the input's type, one
//! feature at a time; see `docs/native-preprocessing.md`.
//! Copyright (c) 2025, Soda team @ Inria; see `LICENSES/TabICL-BSD-3-Clause.txt`.

use rayon::prelude::*;

use crate::preprocess::{
    columns::MISSING,
    labels::Labels,
    python_random::PythonRandom,
    scale::{clamp_outlier, clip_standardized, outlier_bounds, standard_clip, standard_scaler},
    yeo_johnson::{fit_power, yeo_johnson},
};
use crate::{ClassifierAggregation, Error, Result, TaskKind};

/// The most ensemble members fitting may request.
pub const MAX_MEMBERS: usize = 4096;
/// A classifier's fewest and most classes.
const CLASSES: std::ops::RangeInclusive<usize> = 2..=10;
/// Above this many features, feature orders are random samples instead of a
/// Latin square's rows.
const LATIN_FEATURES: usize = 4000;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidArgument(message.into())
}

/// Whether no method appears twice.
fn are_unique(methods: &[NormMethod]) -> bool {
    (1..methods.len()).all(|i| !methods[..i].contains(&methods[i]))
}

/// Each member's order as `u64` indices, member after member.
pub(crate) fn flatten_orders(orders: &[Vec<usize>]) -> Vec<u64> {
    orders.iter().flatten().map(|&index| index as u64).collect()
}

/// A normalization method: one group of ensemble members.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormMethod {
    None,
    Power,
}

impl NormMethod {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Power => "power",
        }
    }

    /// # Errors
    ///
    /// When `name` is not `"none"` or `"power"`.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "power" => Ok(Self::Power),
            _ => Err(invalid(format!("unsupported normalization method '{name}'"))),
        }
    }
}

/// `TabICL`'s preprocessing parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessOptions {
    /// The most ensemble members, in `1..=4096`; fitting may make fewer.
    pub n_estimators: usize,
    pub random_state: i128,
    /// Unique methods, in the order their groups are made.
    pub norm_methods: Vec<NormMethod>,
    /// Positive and finite: outliers lie this many standard deviations out.
    pub outlier_threshold: f64,
}

impl PreprocessOptions {
    fn validate(&self) -> Result<()> {
        if !(1..=MAX_MEMBERS).contains(&self.n_estimators) {
            return Err(invalid(format!("n_estimators must be in 1..={MAX_MEMBERS}")));
        }
        if self.norm_methods.is_empty() || !are_unique(&self.norm_methods) {
            return Err(invalid("norm_methods must be a nonempty list of unique methods"));
        }
        if !(self.outlier_threshold.is_finite() && self.outlier_threshold > 0.0) {
            return Err(invalid("outlier_threshold must be positive and finite"));
        }
        Ok(())
    }
}

/// What fitting learns from, one per training row: class labels, or the
/// regressor's target values.
#[derive(Clone, Debug, PartialEq)]
pub enum Targets {
    Labels(Labels),
    Values(Vec<f32>),
}

impl Targets {
    fn len(&self) -> usize {
        match self {
            Self::Labels(labels) => labels.len(),
            Self::Values(values) => values.len(),
        }
    }
}

/// A C-ordered `[rows, features]` matrix of `f32` or `f64` values, read as `f64`.
#[derive(Clone, Copy, Debug)]
pub enum Features<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl Features<'_> {
    pub(crate) fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// The value at `index`, exactly.
    fn get(self, index: usize) -> f64 {
        match self {
            Self::F32(values) => f64::from(values[index]),
            Self::F64(values) => values[index],
        }
    }

    /// The value at `index`, or `fill` when it is missing (NaN).
    fn get_or(self, index: usize, fill: f64) -> f64 {
        let value = self.get(index);
        if value.is_nan() { fill } else { value }
    }

    /// Rows `start..end` of a matrix with `columns` columns.
    pub(crate) fn rows(self, start: usize, end: usize, columns: usize) -> Self {
        match self {
            Self::F32(values) => Self::F32(&values[start * columns..end * columns]),
            Self::F64(values) => Self::F64(&values[start * columns..end * columns]),
        }
    }
}

/// A fitted `PowerTransformer`: each feature's lambda, then the
/// `StandardScaler` mean and scale of the transformed values.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PowerTransform {
    pub(crate) lambdas: Vec<f64>,
    pub(crate) mean: Vec<f64>,
    pub(crate) scale: Vec<f64>,
}

/// One group's fitted normalization, one value per kept feature in each
/// statistic. Its method is `power` exactly when it has a power transform.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Normalization {
    /// `CustomStandardScaler`'s mean and scale.
    pub(crate) mean: Vec<f64>,
    pub(crate) scale: Vec<f64>,
    /// Each feature's range after scaling: the power transform's fallback domain.
    pub(crate) minimum: Vec<f64>,
    pub(crate) maximum: Vec<f64>,
    /// `OutlierRemover`'s bounds.
    pub(crate) lower: Vec<f64>,
    pub(crate) upper: Vec<f64>,
    pub(crate) power: Option<PowerTransform>,
}

impl Normalization {
    pub(crate) fn method(&self) -> NormMethod {
        if self.power.is_some() {
            NormMethod::Power
        } else {
            NormMethod::None
        }
    }

    /// Every statistic: mean, scale, minimum, maximum, lower and upper bounds,
    /// then the power transform's lambdas, mean, and scale.
    pub(crate) fn statistics(&self) -> Vec<&Vec<f64>> {
        let mut statistics = vec![
            &self.mean,
            &self.scale,
            &self.minimum,
            &self.maximum,
            &self.lower,
            &self.upper,
        ];
        if let Some(power) = &self.power {
            statistics.extend([&power.lambdas, &power.mean, &power.scale]);
        }
        statistics
    }

    /// A `method` normalization from `statistics`, in `statistics()`'s order.
    pub(crate) fn from_statistics(method: NormMethod, statistics: Vec<Vec<f64>>) -> Self {
        let mut statistics = statistics.into_iter();
        let mut next = || statistics.next().expect("a statistic");
        let [mean, scale, minimum, maximum, lower, upper] = std::array::from_fn(|_| next());
        let power = (method == NormMethod::Power).then(|| PowerTransform {
            lambdas: next(),
            mean: next(),
            scale: next(),
        });
        Self {
            mean,
            scale,
            minimum,
            maximum,
            lower,
            upper,
            power,
        }
    }

    /// One value of feature `k`, preprocessed; `None` when the power transform
    /// overflows. `clip` first clips it to the training range, as `TabICL`'s
    /// fallback does.
    fn value(&self, value: f64, k: usize, clip: bool) -> Option<f64> {
        let mut value = clip_standardized(value, self.mean[k], self.scale[k]);
        if let Some(power) = &self.power {
            if clip {
                value = value.max(self.minimum[k]).min(self.maximum[k]);
            }
            value = yeo_johnson(value, power.lambdas[k]);
            if !value.is_finite() {
                return None;
            }
            value = (value - power.mean[k]) / power.scale[k];
        }
        Some(clamp_outlier(value, self.lower[k], self.upper[k]))
    }

    /// The FP32 `[rows, features]` matrix of `x`'s `kept` columns, with a
    /// missing value of kept feature `k` replaced by `means[k]`.
    ///
    /// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L740-L769>.
    fn transform(&self, x: Features<'_>, columns: usize, kept: &[usize], means: &[f64]) -> Result<Vec<f32>> {
        let mut out = vec![0.0; x.len() / columns * kept.len()];
        #[expect(clippy::cast_possible_truncation, reason = "The model reads FP32")]
        let fill = |out: &mut [f32], clip: bool| {
            out.par_chunks_mut(kept.len()).enumerate().all(|(row, out)| {
                out.iter_mut().zip(kept).enumerate().all(|(k, (out, &column))| {
                    self.value(x.get_or(row * columns + column, means[k]), k, clip)
                        .map(|value| *out = value as f32)
                        .is_some()
                })
            })
        };
        // TabICL's fallback when the transform overflows: clip the whole batch
        // to the training range and transform it again.
        if !fill(&mut out, false) && !fill(&mut out, true) {
            return Err(invalid(
                "the power transform overflows even within the training range",
            ));
        }
        if out.iter().any(|value| !value.is_finite()) {
            return Err(invalid("Preprocessing produced non-finite FP32 values"));
        }
        Ok(out)
    }
}

/// One feature's statistics for each method, in `Normalization::statistics`'s
/// order, from its training values.
///
/// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L684-L738>.
fn fit_feature(values: &[f64], methods: &[NormMethod], threshold: f64) -> Result<Vec<Vec<f64>>> {
    let (mean, scale) = standard_clip(values);
    let standardized: Vec<f64> = values
        .iter()
        .map(|&value| clip_standardized(value, mean, scale))
        .collect();
    let (minimum, maximum) = standardized
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    methods
        .iter()
        .map(|&method| {
            let ((lower, upper), power) = match method {
                NormMethod::None => (outlier_bounds(&standardized, threshold), None),
                NormMethod::Power => {
                    let mut normalized = standardized.clone();
                    let power = fit_power(&mut normalized)?;
                    (outlier_bounds(&normalized, threshold), Some(power))
                }
            };
            let statistics: Vec<f64> = [mean, scale, minimum, maximum, lower, upper]
                .into_iter()
                .chain(power.into_iter().flatten())
                .collect();
            if statistics.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "Training data produced non-finite preprocessing statistics",
                ));
            }
            Ok(statistics)
        })
        .collect()
}

/// The feature orders of `TabICL`'s `Shuffler(method="latin")`.
///
/// For up to 4000 features they are the rows of a Latin square whose rows and
/// columns are shuffled. The square `TabICL` builds has a closed form: its first
/// row lists the features in the order they are drawn, and row `r` is that row
/// rotated right by `r`. Both shuffles draw only by the square's size, so the
/// orders come from two shuffled index lists without building the square.
///
/// Equivalent of
/// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L811-L868>
/// and <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L870-L903>.
pub(crate) enum FeatureShuffles {
    Identity(usize),
    Latin {
        first_row: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
    },
    Sampled(Vec<Vec<usize>>),
}

impl FeatureShuffles {
    pub(crate) fn new(features: usize, members: usize, seed: i128) -> Self {
        if members == 1 {
            return Self::Identity(features);
        }
        let mut rng = PythonRandom::new(seed);
        let mut indices: Vec<usize> = (0..features).collect();
        if features > LATIN_FEATURES {
            return Self::Sampled((0..members).map(|_| rng.sample(&indices, features)).collect());
        }
        let mut first_row = Vec::with_capacity(features);
        while indices.len() > 1 {
            first_row.push(indices.remove(rng.index_below(indices.len())));
        }
        first_row.extend(indices);
        let mut rows: Vec<usize> = (0..features).collect();
        rng.shuffle(&mut rows);
        let mut columns: Vec<usize> = (0..features).collect();
        rng.shuffle(&mut columns);
        Self::Latin {
            first_row,
            rows,
            columns,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Identity(_) => 1,
            Self::Latin { rows, .. } => rows.len(),
            Self::Sampled(orders) => orders.len(),
        }
    }

    pub(crate) fn get(&self, index: usize) -> Vec<usize> {
        match self {
            Self::Identity(features) => (0..*features).collect(),
            Self::Latin {
                first_row,
                rows,
                columns,
            } => {
                let n = first_row.len();
                rows.iter()
                    .map(|&row| first_row[(columns[index] + n - row) % n])
                    .collect()
            }
            Self::Sampled(orders) => orders[index].clone(),
        }
    }
}

/// `TabICL`'s `EnsembleGenerator`: every (feature order, class order) pair,
/// shuffled, then paired with each method in turn up to `n_estimators`. Returns
/// the methods that make groups, in option order, and each member's group,
/// feature order, and class order (empty for a regressor, `classes` `None`).
///
/// Equivalent of
/// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L1077-L1128>
/// and <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L811-L868>.
#[expect(clippy::type_complexity, reason = "the ensemble's four parts")]
fn ensemble(
    options: &PreprocessOptions,
    features: usize,
    classes: Option<usize>,
) -> (Vec<NormMethod>, Vec<usize>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let members = options.n_estimators;
    let shuffles = FeatureShuffles::new(features, members, options.random_state);
    let class_orders: Vec<Vec<usize>> = match classes {
        None => vec![Vec::new()],
        Some(classes) if members == 1 => vec![(0..classes).collect()],
        Some(classes) => (0..classes)
            .map(|shift| {
                let mut order: Vec<usize> = (0..classes).collect();
                order.rotate_right(shift);
                order
            })
            .collect(),
    };
    let mut combinations: Vec<usize> = (0..shuffles.len() * class_orders.len()).collect();
    PythonRandom::new(options.random_state).shuffle(&mut combinations);
    let methods = &options.norm_methods;
    let selected: Vec<(usize, NormMethod)> = (0..members.min(combinations.len() * methods.len()))
        .map(|k| (combinations[k / methods.len()], methods[k % methods.len()]))
        .collect();
    let group_methods: Vec<NormMethod> = methods
        .iter()
        .copied()
        .filter(|method| selected.iter().any(|(_, name)| name == method))
        .collect();
    let (mut member_groups, mut feature_permutations, mut class_permutations) = (vec![], vec![], vec![]);
    for (group, &method) in group_methods.iter().enumerate() {
        for &(combination, _) in selected.iter().filter(|(_, name)| *name == method) {
            member_groups.push(group);
            feature_permutations.push(shuffles.get(combination / class_orders.len()));
            class_permutations.push(class_orders[combination % class_orders.len()].clone());
        }
    }
    (
        group_methods,
        member_groups,
        feature_permutations,
        class_permutations,
    )
}

/// Fitted `TabICL` preprocessing. It keeps no training rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Preprocessor {
    /// Each input feature's sorted categories, or `None` for a numeric
    /// feature. A categorical feature's values are category indices or -1.
    pub(crate) categories: Vec<Option<Labels>>,
    /// Which features vary in training, categorical features first; the
    /// others are dropped.
    pub(crate) feature_mask: Vec<bool>,
    /// Each kept feature's training mean, which replaces its missing values:
    /// `SimpleImputer`'s `statistics_`.
    pub(crate) imputer_statistics: Vec<f64>,
    /// One per group, in group order.
    pub(crate) normalizations: Vec<Normalization>,
    /// Each member's group, feature order, and class order (empty for regression).
    pub(crate) member_groups: Vec<usize>,
    pub(crate) feature_permutations: Vec<Vec<usize>>,
    pub(crate) class_permutations: Vec<Vec<usize>>,
    /// The sorted classes, for a classifier.
    pub(crate) classes: Option<Labels>,
    /// The regressor's target mean and scale, as `StandardScaler` fits them.
    pub(crate) target_scale: Option<(f64, f64)>,
    /// The training data's feature names, when it had them.
    pub(crate) feature_names: Option<Vec<String>>,
    /// The classifier's `softmax_temperature` and `average_logits`.
    pub(crate) aggregation: Option<ClassifierAggregation>,
}

/// Check a C-ordered matrix without infinities and return its rows. NaN marks
/// a missing value.
fn check_matrix(x: Features<'_>, columns: usize) -> Result<usize> {
    let rows = matrix_rows(x, columns)?;
    if (0..x.len()).any(|index| x.get(index).is_infinite()) {
        return Err(invalid(
            "X must not contain infinity; mark missing values with NaN",
        ));
    }
    Ok(rows)
}

/// The rows of a C-ordered matrix with `columns` columns.
pub(crate) fn matrix_rows(x: Features<'_>, columns: usize) -> Result<usize> {
    if columns == 0 || !x.len().is_multiple_of(columns) {
        return Err(invalid("X must be a nonempty matrix of features"));
    }
    Ok(x.len() / columns)
}

/// `SimpleImputer(strategy="mean")`'s statistics: each column's mean over its
/// values that are not missing, row after row, or 0 for a column with none.
/// `TabICL` 2.1.1's imputer drops such a column; it is constant here, so the
/// feature mask drops it instead.
///
/// Port of <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/impute/_base.py#L529-L543>.
fn column_means(x: Features<'_>, columns: usize) -> Vec<f64> {
    let mut sums = vec![0.0; columns];
    let mut counts = vec![0_usize; columns];
    for index in 0..x.len() {
        let value = x.get(index);
        if !value.is_nan() {
            sums[index % columns] += value;
            counts[index % columns] += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "Counts stay far below 2^53")]
    sums.iter()
        .zip(counts)
        .map(|(&sum, count)| if count == 0 { 0.0 } else { sum / count as f64 })
        .collect()
}

/// `x`, `[rows, columns]`, with its categorical features first, as `TabICL`'s
/// `ColumnTransformer` orders them, or `None` when it has none. `categories`
/// has one entry per column.
///
/// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L131-L149>.
fn in_tabicl_order(
    categories: &[Option<Labels>],
    x: Features<'_>,
    columns: usize,
) -> Result<Option<Vec<f64>>> {
    debug_assert_eq!(categories.len(), columns, "one entry per column");
    if categories.iter().all(Option::is_none) {
        return Ok(None);
    }
    let order: Vec<usize> = (0..columns)
        .filter(|&j| categories[j].is_some())
        .chain((0..columns).filter(|&j| categories[j].is_none()))
        .collect();
    let mut ordered = Vec::with_capacity(x.len());
    for row in 0..x.len() / columns {
        for &j in &order {
            let value = x.get(row * columns + j);
            #[expect(clippy::cast_precision_loss, reason = "Category counts stay far below 2^53")]
            if let Some(categories) = &categories[j]
                && value != MISSING
                && !(value.fract() == 0.0 && (0.0..categories.len() as f64).contains(&value))
            {
                return Err(invalid(format!(
                    "categorical feature {j} has a value that is not a category index"
                )));
            }
            ordered.push(value);
        }
    }
    Ok(Some(ordered))
}

impl Preprocessor {
    /// Fit on `x`, `[rows, columns]`, whose features have `categories`, as
    /// [`Self::categories`] holds them, and one target per row: a classifier
    /// for labels, else a regressor, which shuffles no classes.
    ///
    /// Port of
    /// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L1013-L1075>,
    /// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L216-L245>,
    /// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/classifier.py#L417-L522>,
    /// and <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/regressor.py#L351-L440>.
    ///
    /// # Errors
    ///
    /// When an option is out of range, `x` is empty, holds infinity or a
    /// value that is not a category index, or is constant once imputed, or
    /// the targets do not fit the task.
    pub fn fit(
        options: &PreprocessOptions,
        x: Features<'_>,
        columns: usize,
        categories: Vec<Option<Labels>>,
        targets: &Targets,
    ) -> Result<Self> {
        options.validate()?;
        let rows = check_matrix(x, columns)?;
        if categories.len() != columns || categories.iter().flatten().any(|c| !c.are_classes()) {
            return Err(invalid(
                "categories must be one per feature, each sorted and distinct",
            ));
        }
        let ordered = in_tabicl_order(&categories, x, columns)?;
        let x = ordered.as_deref().map_or(x, Features::F64);
        if targets.len() != rows {
            return Err(invalid("X and y must have the same length"));
        }
        if rows < 2 {
            return Err(invalid("X and y must have at least two training rows"));
        }
        let (classes, target_scale) = match targets {
            Targets::Labels(labels) => {
                let classes = labels.classes()?;
                if !CLASSES.contains(&classes.len()) {
                    return Err(invalid("The classifier requires 2..10 classes"));
                }
                (Some(classes), None)
            }
            Targets::Values(values) => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("y must contain only finite values"));
                }
                let values: Vec<f64> = values.iter().copied().map(f64::from).collect();
                (None, Some(standard_scaler(&values)))
            }
        };
        let means = column_means(x, columns);
        let value = |row: usize, j: usize| x.get_or(row * columns + j, means[j]);
        #[expect(clippy::float_cmp, reason = "A feature is kept when any value differs")]
        let feature_mask: Vec<bool> = (0..columns)
            .map(|j| (1..rows).any(|row| value(row, j) != value(0, j)))
            .collect();
        let kept: Vec<usize> = (0..columns).filter(|&j| feature_mask[j]).collect();
        if kept.is_empty() {
            return Err(invalid(
                "Training data must contain at least one nonconstant feature",
            ));
        }
        let (group_methods, member_groups, feature_permutations, class_permutations) =
            ensemble(options, kept.len(), classes.as_ref().map(Labels::len));
        // One feature at a time, from a column-major copy of the kept features.
        let mut by_feature = vec![0.0; rows * kept.len()];
        by_feature
            .par_chunks_mut(rows)
            .zip(&kept)
            .for_each(|(values, &column)| {
                for (row, slot) in values.iter_mut().enumerate() {
                    *slot = value(row, column);
                }
            });
        let features: Vec<Vec<Vec<f64>>> = by_feature
            .par_chunks(rows)
            .map(|values| fit_feature(values, &group_methods, options.outlier_threshold))
            .collect::<Result<_>>()?;
        let normalizations = group_methods
            .iter()
            .enumerate()
            .map(|(group, &method)| {
                let statistics = (0..features[0][group].len())
                    .map(|statistic| features.iter().map(|feature| feature[group][statistic]).collect())
                    .collect();
                Normalization::from_statistics(method, statistics)
            })
            .collect();
        Ok(Self {
            categories,
            feature_mask,
            imputer_statistics: kept.iter().map(|&j| means[j]).collect(),
            normalizations,
            member_groups,
            feature_permutations,
            class_permutations,
            aggregation: classes.as_ref().map(|_| ClassifierAggregation::default()),
            classes,
            target_scale,
            feature_names: None,
        })
    }

    /// Each group's FP32 `[rows, n_features_out]` matrix for `x`, `[rows, columns]`.
    ///
    /// Port of
    /// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L247-L263>
    /// and <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L1130-L1259>.
    ///
    /// # Errors
    ///
    /// When `x` has the wrong number of features, holds infinity or a value
    /// that is not a category index, or a result is not finite in FP32.
    pub fn transform(&self, x: Features<'_>, columns: usize) -> Result<Vec<Vec<f32>>> {
        if columns != self.n_features_in() {
            return Err(invalid(format!(
                "X has {columns} features; expected {}",
                self.n_features_in()
            )));
        }
        check_matrix(x, columns)?;
        let ordered = in_tabicl_order(&self.categories, x, columns)?;
        let x = ordered.as_deref().map_or(x, Features::F64);
        let kept: Vec<usize> = (0..columns).filter(|&j| self.feature_mask[j]).collect();
        self.normalizations
            .iter()
            .map(|normalization| normalization.transform(x, columns, &kept, &self.imputer_statistics))
            .collect()
    }

    /// Each member's feature-permuted group from `group_rows`, `transform`'s
    /// output, stacked as FP32 `[B, rows, H]`, and that shape.
    ///
    /// Port of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L1130-L1259>.
    pub(crate) fn ensemble_rows(&self, group_rows: &[Vec<f32>]) -> (Vec<f32>, [usize; 3]) {
        let features = self.n_features_out();
        let rows = group_rows[0].len() / features;
        debug_assert!(
            group_rows.len() == self.normalizations.len()
                && group_rows.iter().all(|group| group.len() == rows * features),
            "group rows are not transform's shapes"
        );
        let shape = [self.member_count(), rows, features];
        let mut stacked = vec![0.0; shape.iter().product()];
        if rows > 0 {
            stacked
                .par_chunks_mut(rows * features)
                .enumerate()
                .for_each(|(member, block)| {
                    let group = &group_rows[self.member_groups[member]];
                    let order = &self.feature_permutations[member];
                    for (target, source) in block.chunks_exact_mut(features).zip(group.chunks_exact(features))
                    {
                        for (value, &feature) in target.iter_mut().zip(order) {
                            *value = source[feature];
                        }
                    }
                });
        }
        (stacked, shape)
    }

    /// Each label's class index.
    ///
    /// # Errors
    ///
    /// For a regressor, or a label that is not a class.
    pub fn encode_labels(&self, labels: &Labels) -> Result<Vec<usize>> {
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| invalid("a regressor has no classes"))?;
        labels.encode(classes)
    }

    /// A classifier when fitted on labels, else a regressor.
    #[must_use]
    pub fn task(&self) -> TaskKind {
        if self.classes.is_some() {
            TaskKind::Classifier
        } else {
            TaskKind::Regressor
        }
    }

    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.feature_mask.len()
    }
    #[must_use]
    pub fn n_features_out(&self) -> usize {
        self.feature_mask.iter().filter(|kept| **kept).count()
    }
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.member_groups.len()
    }
    #[must_use]
    pub fn feature_mask(&self) -> &[bool] {
        &self.feature_mask
    }
    /// Each group's normalization method, in group order.
    #[must_use]
    pub fn group_methods(&self) -> Vec<NormMethod> {
        self.normalizations.iter().map(Normalization::method).collect()
    }
    #[must_use]
    pub fn member_groups(&self) -> &[usize] {
        &self.member_groups
    }
    #[must_use]
    pub fn feature_permutations(&self) -> &[Vec<usize>] {
        &self.feature_permutations
    }
    #[must_use]
    pub fn class_permutations(&self) -> &[Vec<usize>] {
        &self.class_permutations
    }
    #[must_use]
    pub fn classes(&self) -> Option<&Labels> {
        self.classes.as_ref()
    }
    /// Each input feature's sorted categories, or `None` for a numeric feature.
    #[must_use]
    pub fn categories(&self) -> &[Option<Labels>] {
        &self.categories
    }

    /// Check what fitting guarantees, for a preprocessor read from bytes.
    pub(crate) fn validate(&self) -> Result<()> {
        let fail = |message: &str| Err(Error::Artifact(format!("preprocessor: {message}")));
        let features = self.n_features_out();
        let is_permutation = |order: &[usize], n: usize| {
            let mut seen = vec![false; n];
            order.len() == n
                && order
                    .iter()
                    .all(|&i| i < n && !std::mem::replace(&mut seen[i], true))
        };
        let classes = self.classes.as_ref().map_or(0, Labels::len);
        let members = self.member_count();
        let groups = self.normalizations.len();
        if features == 0 {
            return fail("no features are kept");
        }
        if groups == 0 || !are_unique(&self.group_methods()) {
            return fail("the normalization groups are invalid");
        }
        if !(1..=MAX_MEMBERS).contains(&members)
            || self.feature_permutations.len() != members
            || self.class_permutations.len() != members
            || self.member_groups.iter().any(|&group| group >= groups)
            || (0..groups).any(|group| !self.member_groups.contains(&group))
        {
            return fail("the ensemble members are invalid");
        }
        if !self
            .feature_permutations
            .iter()
            .all(|order| is_permutation(order, features))
            || !self
                .class_permutations
                .iter()
                .all(|order| is_permutation(order, classes))
        {
            return fail("a member's feature or class order is not a permutation");
        }
        if let Some(labels) = &self.classes
            && !(CLASSES.contains(&labels.len()) && labels.are_classes())
        {
            return fail("the classes are invalid");
        }
        match (&self.classes, self.target_scale, self.aggregation) {
            (Some(_), None, Some(aggregation))
                if aggregation.softmax_temperature.is_finite() && aggregation.softmax_temperature > 0.0 => {}
            (None, Some((mean, scale)), None) if mean.is_finite() && scale.is_finite() && scale > 0.0 => {}
            _ => return fail("the target scale or aggregation does not match the task"),
        }
        if self
            .feature_names
            .as_ref()
            .is_some_and(|names| names.len() != self.feature_mask.len())
        {
            return fail("the feature names do not match the features");
        }
        if self.categories.len() != self.feature_mask.len()
            || !self.categories.iter().flatten().all(Labels::are_classes)
        {
            return fail("the categories are invalid");
        }
        if self.imputer_statistics.len() != features
            || !self.imputer_statistics.iter().all(|value| value.is_finite())
        {
            return fail("the imputer statistics are invalid");
        }
        if !self.normalizations.iter().all(|normalization| {
            normalization
                .statistics()
                .iter()
                .all(|values| values.len() == features && values.iter().all(|value| value.is_finite()))
        }) {
            return fail("a group's statistics are invalid");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{FeatureShuffles, Features, MISSING, NormMethod, PreprocessOptions, Preprocessor, Targets};
    use crate::format::preprocessor::{decode, encode};
    use crate::preprocess::{fixture, integers, labels::Labels, numbers, python_random::PythonRandom};
    use serde_json::Value;

    /// Every feature order of `FeatureShuffles`.
    fn feature_shuffles(features: usize, members: usize, seed: i128) -> Vec<Vec<usize>> {
        let shuffles = FeatureShuffles::new(features, members, seed);
        (0..shuffles.len()).map(|index| shuffles.get(index)).collect()
    }

    fn count(value: &Value) -> usize {
        usize::try_from(value.as_u64().unwrap()).unwrap()
    }

    fn indices(value: &Value) -> Vec<usize> {
        integers(value)
            .into_iter()
            .map(|index| usize::try_from(index).unwrap())
            .collect()
    }

    fn rows(value: &Value) -> Vec<Vec<usize>> {
        value.as_array().unwrap().iter().map(indices).collect()
    }

    fn matrix(value: &Value) -> Vec<f64> {
        value.as_array().unwrap().iter().flat_map(numbers).collect()
    }

    fn options() -> PreprocessOptions {
        PreprocessOptions {
            n_estimators: 8,
            random_state: 42,
            norm_methods: vec![NormMethod::None, NormMethod::Power],
            outlier_threshold: 4.0,
        }
    }

    /// `assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)`, the Python oracle test's bound.
    fn assert_close(actual: &[f32], expected: &[f64], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: lengths");
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (f64::from(a) - e).abs() <= 2e-6 + 2e-6 * e.abs(),
                "{what}[{index}]: {a} vs {e}"
            );
        }
    }

    #[test]
    fn matches_tabicl() {
        let fixture = fixture("preprocessing.json");
        let columns = fixture["training"][0].as_array().unwrap().len();
        let (training, query) = (matrix(&fixture["training"]), matrix(&fixture["query"]));
        let text = Labels::Text(
            fixture["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|label| label.as_str().unwrap().to_owned())
                .collect(),
        );
        let labels = Targets::Labels(text.clone());
        for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            let (x, q) = (Features::F64(&training), Features::F64(&query));
            let preprocessor =
                Preprocessor::fit(&options(), x, columns, vec![None; columns], &labels).unwrap();
            let names: Vec<&str> = preprocessor
                .group_methods()
                .iter()
                .map(|method| method.name())
                .collect();
            let expected: Vec<&str> = case["group_names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| name.as_str().unwrap())
                .collect();
            assert_eq!(names, expected, "case {index}");
            let classes: Vec<String> = case["classes_"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| name.as_str().unwrap().to_owned())
                .collect();
            assert_eq!(
                preprocessor.classes(),
                Some(&Labels::Text(classes)),
                "case {index}"
            );
            assert_eq!(
                preprocessor.encode_labels(&text).unwrap(),
                indices(&case["encoded_labels"]),
                "case {index}"
            );
            let mask: Vec<bool> = case["feature_mask_"]
                .as_array()
                .unwrap()
                .iter()
                .map(|kept| kept.as_bool().unwrap())
                .collect();
            assert_eq!(preprocessor.feature_mask(), mask, "case {index}");
            assert_eq!(
                preprocessor.member_groups(),
                indices(&case["ensemble_groups_"]),
                "case {index}"
            );
            assert_eq!(
                preprocessor.feature_permutations(),
                rows(&case["feature_permutations_"]),
                "case {index}"
            );
            assert_eq!(
                preprocessor.class_permutations(),
                rows(&case["class_permutations_"]),
                "case {index}"
            );
            for (data, key) in [(x, "training_groups"), (q, "query_groups")] {
                let groups = preprocessor.transform(data, columns).unwrap();
                for (group, expected) in groups.iter().zip(case[key].as_array().unwrap()) {
                    assert_close(group, &matrix(expected), &format!("case {index} {key}"));
                }
            }
            let sha = [7; 32];
            assert_eq!(
                decode(&encode(&preprocessor, None, &sha).unwrap()).unwrap(),
                (sha, preprocessor, None),
                "case {index}"
            );
        }
    }

    #[test]
    fn missing_values_are_the_training_mean() {
        let columns = 4;
        // Column 2 is all missing, so it is constant once imputed and dropped.
        let mut x: Vec<f64> = (0..120).map(|i| f64::from((i * 37 % 29) - 14) / 3.0).collect();
        for row in 0..30 {
            x[row * columns + 2] = f64::NAN;
        }
        for index in [1, 6, 13, 50, 97] {
            x[index] = f64::NAN;
        }
        let labels = Targets::Labels(Labels::Int((0..30).map(|i| i % 3).collect()));
        let preprocessor = Preprocessor::fit(
            &options(),
            Features::F64(&x),
            columns,
            vec![None; columns],
            &labels,
        )
        .unwrap();
        assert_eq!(preprocessor.feature_mask(), [true, true, false, true]);
        // The same data with each missing value replaced by its column's mean.
        let means: Vec<f64> = (0..columns)
            .map(|j| {
                let (sum, count) = (0..30)
                    .map(|row| x[row * columns + j])
                    .filter(|v| !v.is_nan())
                    .fold((0.0, 0.0), |(sum, count), v| (sum + v, count + 1.0));
                if count == 0.0 { 0.0 } else { sum / count }
            })
            .collect();
        let filled: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if value.is_nan() {
                    means[index % columns]
                } else {
                    value
                }
            })
            .collect();
        assert_eq!(preprocessor.imputer_statistics, [means[0], means[1], means[3]]);
        let reference = Preprocessor::fit(
            &options(),
            Features::F64(&filled),
            columns,
            vec![None; columns],
            &labels,
        )
        .unwrap();
        for (data, other) in [(&x, &filled), (&filled, &filled)] {
            assert_eq!(
                preprocessor.transform(Features::F64(data), columns).unwrap(),
                reference.transform(Features::F64(other), columns).unwrap()
            );
        }
        // A query row preprocesses alike alone and in a batch, whatever else is missing.
        let query = [f64::NAN, 1.0, f64::NAN, 2.0, 0.5, f64::NAN, 3.0, f64::NAN];
        let batch = preprocessor.transform(Features::F64(&query), columns).unwrap();
        let alone = preprocessor
            .transform(Features::F64(&query[..columns]), columns)
            .unwrap();
        assert_eq!(batch[0][..3], alone[0][..]);
        let sha = [1; 32];
        assert_eq!(
            decode(&encode(&preprocessor, None, &sha).unwrap()).unwrap(),
            (sha, preprocessor, None)
        );
    }

    #[test]
    fn categorical_features_come_first() {
        // Numeric, categorical, numeric, categorical: as numbers, with every
        // categorical feature moved to the front, the data fits alike.
        let rows = 30;
        let text = Labels::Text(["a", "b", "c"].map(String::from).to_vec());
        let column = |row: usize, j: usize| match j {
            0 => f64::from(u32::try_from(row * 7 % 11).unwrap()) / 2.0,
            1 => [0.0, 1.0, 2.0, MISSING][row % 4],
            2 => {
                if row.is_multiple_of(5) {
                    f64::NAN
                } else {
                    f64::from(u32::try_from(row).unwrap()).sqrt()
                }
            }
            _ => [1.0, 0.0][row % 2],
        };
        let x: Vec<f64> = (0..rows)
            .flat_map(|row| (0..4).map(move |j| column(row, j)))
            .collect();
        let front: Vec<f64> = (0..rows)
            .flat_map(|row| [1, 3, 0, 2].map(|j| column(row, j)))
            .collect();
        let categories = vec![
            None,
            Some(text.clone()),
            None,
            Some(Labels::Bool(vec![false, true])),
        ];
        let labels = Targets::Labels(Labels::Int((0..30).map(|i| i % 3).collect()));
        let preprocessor =
            Preprocessor::fit(&options(), Features::F64(&x), 4, categories.clone(), &labels).unwrap();
        let numeric =
            Preprocessor::fit(&options(), Features::F64(&front), 4, vec![None; 4], &labels).unwrap();
        assert_eq!(preprocessor.categories(), categories);
        assert_eq!(preprocessor.feature_mask, numeric.feature_mask);
        assert_eq!(preprocessor.imputer_statistics, numeric.imputer_statistics);
        assert_eq!(
            preprocessor.transform(Features::F64(&x), 4).unwrap(),
            numeric.transform(Features::F64(&front), 4).unwrap()
        );
        // A categorical feature's values are category indices or -1.
        for value in [3.0, 0.5, -2.0, f64::NAN] {
            let mut bad = x.clone();
            bad[1] = value;
            assert!(preprocessor.transform(Features::F64(&bad), 4).is_err(), "{value}");
            assert!(
                Preprocessor::fit(&options(), Features::F64(&bad), 4, categories.clone(), &labels).is_err(),
                "{value}"
            );
        }
        let unsorted = vec![None, Some(Labels::Int(vec![2, 1])), None, None];
        assert!(Preprocessor::fit(&options(), Features::F64(&x), 4, unsorted, &labels).is_err());
        let sha = [1; 32];
        assert_eq!(
            decode(&encode(&preprocessor, None, &sha).unwrap()).unwrap(),
            (sha, preprocessor, None)
        );
    }

    #[test]
    fn infinity_and_all_missing_data_are_rejected() {
        let labels = Targets::Labels(Labels::Int(vec![0, 1, 0, 1]));
        let fit = |x: &[f64]| Preprocessor::fit(&options(), Features::F64(x), 2, vec![None; 2], &labels);
        assert!(fit(&[1.0, f64::NAN, 2.0, 3.0, f64::INFINITY, 4.0, 0.0, 5.0]).is_err());
        assert!(fit(&[f64::NAN; 8]).is_err());
        let preprocessor = fit(&[1.0, f64::NAN, 2.0, 3.0, 5.0, 4.0, 0.0, 5.0]).unwrap();
        assert!(
            preprocessor
                .transform(Features::F64(&[f64::NEG_INFINITY, 1.0]), 2)
                .is_err()
        );
        assert!(
            preprocessor
                .transform(Features::F64(&[f64::NAN, f64::NAN]), 2)
                .is_ok()
        );
    }

    #[test]
    fn an_overflowing_power_transform_clips_the_whole_batch() {
        use super::{NormMethod, Normalization};
        let statistics = [0.0, 1.0, -1.0, 0.25, -1e30, 1e30, 200.0, 0.0, 1.0];
        let normalization = Normalization::from_statistics(
            NormMethod::Power,
            statistics.iter().map(|&value| vec![value]).collect(),
        );
        let transform = |x: &[f64]| normalization.transform(Features::F64(x), 1, &[0], &[0.0]);
        // 51^200 overflows f64, so the batch falls back: every row is clipped
        // to the training range, including 0.5, which alone would not be.
        let batch = transform(&[50.0, 0.5]).unwrap();
        assert_eq!(batch, transform(&[0.25, 0.25]).unwrap());
        assert_ne!(transform(&[0.5]).unwrap()[0], batch[1]);
        assert!(transform(&[1e6]).is_ok(), "clipped, so it no longer overflows");
    }

    #[test]
    fn feature_shuffles_match_cpython() {
        for case in fixture("python_random.json")["feature_shuffles"]
            .as_array()
            .unwrap()
        {
            let (features, members) = (count(&case["features"]), count(&case["estimators"]));
            let seed: i128 = case["seed"].as_str().unwrap().parse().unwrap();
            let orders = feature_shuffles(features, members, seed);
            assert_eq!(orders.len(), count(&case["count"]), "{features} features");
            if let Some(expected) = case.get("orders") {
                assert_eq!(orders, rows(expected), "{features} features");
            } else {
                let digest = orders
                    .iter()
                    .flatten()
                    .fold(0xcbf2_9ce4_8422_2325_u64, |state, &value| {
                        (state ^ value as u64).wrapping_mul(0x0100_0000_01b3)
                    });
                assert_eq!(
                    format!("{digest:016x}"),
                    case["digest"].as_str().unwrap(),
                    "{features} features"
                );
            }
        }
    }

    /// `TabICL`'s Latin square, built as it builds it.
    fn literal_latin_square(features: usize, seed: i128) -> Vec<Vec<usize>> {
        let mut rng = PythonRandom::new(seed);
        let mut indices: Vec<usize> = (0..features).collect();
        let mut removed = Vec::new();
        while indices.len() > 1 {
            removed.push(indices.remove(rng.index_below(indices.len())));
        }
        let mut square = vec![indices];
        for &symbol in removed.iter().rev() {
            square.push(square[0].clone());
            for (index, row) in square.iter_mut().enumerate() {
                row.insert(index, symbol);
            }
        }
        rng.shuffle(&mut square);
        let mut transposed: Vec<Vec<usize>> = (0..features)
            .map(|column| square.iter().map(|row| row[column]).collect())
            .collect();
        rng.shuffle(&mut transposed);
        transposed
    }

    #[test]
    fn the_closed_form_is_the_square_tabicl_builds() {
        for features in 1..40 {
            for seed in [0, 1, 42, -3, 1 << 40] {
                assert_eq!(
                    feature_shuffles(features, 8, seed),
                    literal_latin_square(features, seed)
                );
            }
        }
    }

    #[test]
    fn one_member_keeps_the_feature_order() {
        assert_eq!(feature_shuffles(5, 1, 7), vec![vec![0, 1, 2, 3, 4]]);
        let sampled = FeatureShuffles::new(4001, 3, 0);
        assert_eq!(sampled.len(), 3);
        let mut order = sampled.get(2);
        order.sort_unstable();
        assert_eq!(order, (0..4001).collect::<Vec<_>>());
    }
}

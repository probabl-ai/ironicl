//! Kumo-Tabular's preprocessing: NVIDIA's default recipe, fitted on the
//! training rows, one column at a time, in `f64`.
//!
//! Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/kumo/tabular/recipe.py>.
//! A numeric feature is a column of the numeric route; a categorical feature,
//! encoded as [`crate::preprocess::columns`] encodes it, is a code column of the
//! categorical route, after its count column when it has more than 50
//! categories. Every route column drops when constant, then is standardized,
//! clipped, transformed by the member's choice of four, and clipped at four
//! sigmas; numeric route columns also flip sign at random. Each member then
//! shuffles the columns and keeps the first 500.
//!
//! The random draws are Ironicl's own ([`Draws::new`]), so a `random_state`
//! gives the same ensemble everywhere; tests feed NVIDIA's draws instead.

use crate::preprocess::labels::Labels;
use crate::preprocess::python_random::PythonRandom;
use crate::{Error, Result};

/// Columns each member keeps after shuffling.
const MAX_COLUMNS: usize = 500;
/// Categorical features with more categories than this get a count column.
const COUNT_CARDINALITY: usize = 50;
/// The rank-Gaussian transform's most knots per column.
const MAX_KNOTS: usize = 8192;
/// Classes the model separates; more go through error-correcting output codes.
pub(crate) const MAX_CLASSES: usize = 10;
/// The `numerical_processor` options, chosen round-robin by member.
const CHOICES: usize = 4;

fn invalid(message: &str) -> Error {
    Error::InvalidArgument(message.into())
}

/// The random draws of an ensemble: each member's class order (code `c`
/// becomes `class_orders[m][c]`), numeric route signs, column order, and for
/// more than [`MAX_CLASSES`] classes its codebook, one row of symbols per
/// task.
#[derive(Clone, Debug, PartialEq)]
pub struct Draws {
    pub class_orders: Vec<Vec<usize>>,
    pub signs: Vec<Vec<f64>>,
    pub column_orders: Vec<Vec<usize>>,
    pub codebooks: Vec<Vec<Vec<usize>>>,
}

impl Draws {
    /// Draw from `CPython`'s `random.Random(seed)`, in this order: the class
    /// offsets, from shuffled pools of every offset; each member's signs,
    /// negative when `random() < 0.5`; the two shuffles of the Latin column
    /// orders; and each member's codebook.
    #[must_use]
    pub fn new(seed: i128, members: usize, classes: usize, numeric: usize, columns: usize) -> Self {
        let mut rng = PythonRandom::new(seed);
        let mut pool = Vec::new();
        let class_orders = (0..members)
            .map(|_| {
                if pool.is_empty() {
                    pool = (0..classes).collect();
                    rng.shuffle(&mut pool);
                    pool.reverse();
                }
                let offset = pool.pop().unwrap_or(0);
                (0..classes)
                    .map(|code| (code + classes - offset) % classes)
                    .collect()
            })
            .collect();
        let signs = (0..members)
            .map(|_| {
                (0..numeric)
                    .map(|_| if rng.random() < 0.5 { -1.0 } else { 1.0 })
                    .collect()
            })
            .collect();
        let mut base: Vec<usize> = (0..columns).collect();
        let mut rows = base.clone();
        rng.shuffle(&mut base);
        rng.shuffle(&mut rows);
        let column_orders = (0..members)
            .map(|member| {
                if columns <= 1 {
                    return (0..columns).collect();
                }
                let pattern = member % columns;
                rows.iter()
                    .map(|&row| base[(pattern + columns - row) % columns])
                    .collect()
            })
            .collect();
        let codebooks = (0..members)
            .map(|_| {
                if classes > MAX_CLASSES {
                    codebook(classes, &mut rng)
                } else {
                    Vec::new()
                }
            })
            .collect();
        Self {
            class_orders,
            signs,
            column_orders,
            codebooks,
        }
    }
}

/// An error-correcting codebook for `classes` classes: tasks that each
/// separate nine classes as symbols 0 to 8 and merge the rest as symbol 9,
/// the best of 50 draws by minimum, then total, pairwise distance.
///
/// Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/base/ecoc.py#L141-L197>.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Class counts are small"
)]
fn codebook(classes: usize, rng: &mut PythonRandom) -> Vec<Vec<usize>> {
    let rest = MAX_CLASSES - 1;
    let tasks = classes
        .div_ceil(rest)
        .max(4 * (classes as f64).log(MAX_CLASSES as f64).ceil() as usize);
    let draws = if classes <= 200 { 50 } else { 1 };
    let mut best = (Vec::new(), 0);
    for _ in 0..draws {
        let mut code = vec![vec![rest; classes]; tasks];
        let mut coverage = vec![0.0; classes];
        for task in &mut code {
            let priority: Vec<f64> = coverage.iter().map(|c| c + 0.1 * rng.random()).collect();
            let chosen = &argsort(&priority)[..rest];
            let symbols = argsort(&(0..rest).map(|_| rng.random()).collect::<Vec<_>>());
            for (&class, symbol) in chosen.iter().zip(symbols) {
                task[class] = symbol;
                coverage[class] += 1.0;
            }
        }
        let (mut minimum, mut total) = (usize::MAX, 0);
        for a in 0..classes {
            for b in a + 1..classes {
                let distance = code.iter().filter(|task| task[a] != task[b]).count();
                minimum = minimum.min(distance);
                total += distance;
            }
        }
        let pairs = classes * (classes - 1) / 2;
        let score = minimum * (pairs * tasks + 1) + total;
        if best.0.is_empty() || score > best.1 {
            best = (code, score);
        }
    }
    best.0
}

fn argsort(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    order
}

/// Where a route column's values come from.
#[derive(Clone, Debug, PartialEq)]
enum Source {
    /// A feature's values: numbers, or category codes, -1 when missing.
    Feature(usize),
    /// `log1p` of how many training rows share the feature's code, indexed
    /// by code, the last entry for missing and unknown values.
    Count { feature: usize, log_counts: Vec<f64> },
}

/// One of the `numerical_processor` choices, fitted.
#[derive(Clone, Debug, PartialEq)]
enum Choice {
    Identity,
    Power {
        lambda: f64,
        mean: f64,
        scale: f64,
    },
    Robust {
        median: f64,
        scale: f64,
    },
    RankGaussian {
        values: Vec<f64>,
        probabilities: Vec<f64>,
    },
}

/// A kept route column: its source, its standardization, and per choice the
/// fitted transform and four-sigma bounds, for the choices members use.
#[derive(Clone, Debug, PartialEq)]
struct RouteColumn {
    source: Source,
    categorical: bool,
    numeric: bool,
    mean: f64,
    scale: f64,
    choices: Vec<Option<(Choice, f64, f64)>>,
}

impl RouteColumn {
    fn value(&self, x: &[f64], features: usize, row: usize) -> f64 {
        match &self.source {
            Source::Feature(feature) => x[row * features + feature],
            Source::Count { feature, log_counts } => {
                let code = x[row * features + feature];
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "Codes are indices"
                )]
                let slot = if code >= 0.0 {
                    code as usize
                } else {
                    log_counts.len() - 1
                };
                log_counts[slot.min(log_counts.len() - 1)]
            }
        }
    }

    /// Standardized and clipped to ±100.
    fn standardized(&self, value: f64) -> f64 {
        ((value - self.mean) / self.scale).clamp(-100.0, 100.0)
    }

    fn transform(&self, choice: usize, value: f64) -> f64 {
        let (choice, lower, upper) = self.choices[choice].as_ref().expect("a fitted choice");
        let value = choice.transform(self.standardized(value));
        let log = value.abs().ln_1p();
        // `ClipSigma(method="log")`; NaN stays NaN.
        if value.is_nan() {
            value
        } else {
            value.max(lower - log).min(upper + log)
        }
    }
}

impl Choice {
    fn fit(choice: usize, values: &[f64]) -> Self {
        match choice {
            0 => Self::Identity,
            1 => power(values),
            2 => robust(values),
            _ => rank_gaussian(values),
        }
    }

    fn transform(&self, value: f64) -> f64 {
        match self {
            Self::Identity => value,
            Self::Power { lambda, mean, scale } => {
                ((yeo_johnson(value, *lambda) - mean) / scale).clamp(-f64::MAX, f64::MAX)
            }
            Self::Robust { median, scale } => saturate((value - median) / scale),
            Self::RankGaussian {
                values,
                probabilities,
            } => {
                if value.is_nan() {
                    f64::NAN
                } else {
                    ndtri(interpolate(value, values, probabilities))
                }
            }
        }
    }
}

/// A fitted preprocessing: the kept route columns and each member's columns,
/// as route column indices and signs.
#[derive(Clone, Debug, PartialEq)]
pub struct Preprocessor {
    features: usize,
    columns: Vec<RouteColumn>,
    members: Vec<Vec<(usize, f64)>>,
    pub(crate) classes: Labels,
    pub(crate) draws: Draws,
}

/// One member's preprocessed rows, `[rows, columns]` in FP32, NaN marking a
/// missing number, and which columns are category codes.
#[derive(Clone, Debug, PartialEq)]
pub struct MemberRows {
    pub x: Vec<f32>,
    pub columns: usize,
    pub categorical: Vec<bool>,
}

impl Preprocessor {
    /// Fit on the encoded `[rows, features]` matrix `x`, whose `categorical`
    /// features hold category codes, and the class `labels`, for `members`
    /// members drawn from `seed`.
    ///
    /// # Errors
    ///
    /// When the shapes disagree or a label is NaN.
    pub fn fit(x: &[f64], categorical: &[bool], labels: &Labels, members: usize, seed: i128) -> Result<Self> {
        Self::fit_with(x, categorical, labels, members, |classes, numeric, columns| {
            Draws::new(seed, members, classes, numeric, columns)
        })
    }

    /// [`Self::fit`] with the draws `draw(classes, numeric route columns, columns)` gives.
    pub(crate) fn fit_with(
        x: &[f64],
        categorical: &[bool],
        labels: &Labels,
        members: usize,
        draw: impl FnOnce(usize, usize, usize) -> Draws,
    ) -> Result<Self> {
        let features = categorical.len();
        if features == 0 || x.is_empty() || !x.len().is_multiple_of(features) {
            return Err(invalid("X must be a nonempty matrix of features"));
        }
        let rows = x.len() / features;
        if labels.len() != rows {
            return Err(invalid("X and y must have the same length"));
        }
        if members == 0 {
            return Err(invalid("n_estimators must be at least 1"));
        }
        let classes = labels.classes()?;
        let sources = sources(x, categorical);
        let used: Vec<bool> = (0..CHOICES).map(|choice| choice < members).collect();
        let mut columns = Vec::new();
        for (source, is_categorical, numeric) in sources {
            let mut route = RouteColumn {
                source,
                categorical: is_categorical,
                numeric,
                mean: 0.0,
                scale: 1.0,
                choices: vec![None; CHOICES],
            };
            let values: Vec<f64> = (0..rows).map(|row| route.value(x, features, row)).collect();
            if !varies(&values) {
                continue;
            }
            (route.mean, route.scale) = standardize(&values);
            let values: Vec<f64> = values.iter().map(|&value| route.standardized(value)).collect();
            for (choice, slot) in route
                .choices
                .iter_mut()
                .enumerate()
                .filter(|(choice, _)| used[*choice])
            {
                let fitted = Choice::fit(choice, &values);
                let transformed: Vec<f64> = values.iter().map(|&value| fitted.transform(value)).collect();
                let (lower, upper) = sigma_bounds(&transformed);
                *slot = Some((fitted, lower, upper));
            }
            columns.push(route);
        }
        let numeric = columns.iter().filter(|column| column.numeric).count();
        let draws = draw(classes.len(), numeric, columns.len());
        if draws.class_orders.len() != members
            || draws.codebooks.len() != members
            || draws.signs.iter().any(|signs| signs.len() != numeric)
            || draws
                .column_orders
                .iter()
                .any(|order| order.len() != columns.len())
        {
            return Err(invalid("draws do not fit the ensemble"));
        }
        let members = draws
            .column_orders
            .iter()
            .zip(&draws.signs)
            .map(|(order, signs)| {
                order
                    .iter()
                    .take(MAX_COLUMNS)
                    .map(|&index| (index, if columns[index].numeric { signs[index] } else { 1.0 }))
                    .collect()
            })
            .collect();
        Ok(Self {
            features,
            columns,
            members,
            classes,
            draws,
        })
    }

    #[must_use]
    pub fn members(&self) -> usize {
        self.members.len()
    }

    /// Member `member`'s preprocessed columns: true for category codes.
    #[must_use]
    pub fn categorical(&self, member: usize) -> Vec<bool> {
        self.members[member]
            .iter()
            .map(|&(index, _)| self.columns[index].categorical)
            .collect()
    }

    /// Member `member`'s preprocessed rows of the encoded `[rows, features]` matrix `x`.
    ///
    /// # Errors
    ///
    /// When `x` does not have this preprocessing's features.
    pub fn transform(&self, x: &[f64], member: usize) -> Result<MemberRows> {
        if x.is_empty() || !x.len().is_multiple_of(self.features) {
            return Err(invalid("X has a different number of features than in training"));
        }
        let rows = x.len() / self.features;
        let columns = &self.members[member];
        let choice = member % CHOICES;
        let mut out = Vec::with_capacity(rows * columns.len());
        for row in 0..rows {
            for &(index, sign) in columns {
                let column = &self.columns[index];
                #[expect(clippy::cast_possible_truncation, reason = "The model reads FP32")]
                out.push((sign * column.transform(choice, column.value(x, self.features, row))) as f32);
            }
        }
        Ok(MemberRows {
            x: out,
            columns: columns.len(),
            categorical: self.categorical(member),
        })
    }

    /// Each training label's class code, as member `member` sees it.
    ///
    /// # Errors
    ///
    /// When a label is not one of the classes.
    pub fn class_codes(&self, labels: &Labels, member: usize) -> Result<Vec<usize>> {
        let order = &self.draws.class_orders[member];
        Ok(labels
            .encode(&self.classes)?
            .into_iter()
            .map(|code| order[code])
            .collect())
    }
}

/// The route columns' sources, each with whether it is a category code
/// and whether it is on the numeric route: numeric features, then count
/// columns, then codes.
fn sources(x: &[f64], categorical: &[bool]) -> Vec<(Source, bool, bool)> {
    let features = categorical.len();
    let rows = x.len() / features;
    let column = |feature: usize| (0..rows).map(move |row| x[row * features + feature]);
    let mut sources: Vec<(Source, bool, bool)> = (0..features)
        .filter(|&feature| !categorical[feature])
        .map(|feature| (Source::Feature(feature), false, true))
        .collect();
    for feature in (0..features).filter(|&feature| categorical[feature]) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "Codes are indices"
        )]
        let codes: Vec<Option<usize>> = column(feature)
            .map(|code| (code >= 0.0).then_some(code as usize))
            .collect();
        let cardinality = codes.iter().flatten().max().map_or(0, |code| code + 1);
        if cardinality > COUNT_CARDINALITY {
            let mut counts = vec![0.0_f64; cardinality + 1];
            for code in &codes {
                counts[code.unwrap_or(cardinality)] += 1.0;
            }
            let log_counts = counts.into_iter().map(f64::ln_1p).collect();
            sources.push((Source::Count { feature, log_counts }, false, false));
        }
    }
    sources.extend(
        (0..features)
            .filter(|&feature| categorical[feature])
            .map(|feature| (Source::Feature(feature), true, false)),
    );
    sources
}

/// `DropConstantColumns`: whether a column differs from its first row
/// somewhere, NaN differing from everything, and is not all NaN. One row is
/// kept.
#[expect(
    clippy::float_cmp,
    reason = "PyTorch compares exactly, NaN differing from everything"
)]
fn varies(values: &[f64]) -> bool {
    values.len() <= 1
        || (values.iter().any(|value| *value != values[0]) && values.iter().any(|value| !value.is_nan()))
}

fn finite(values: &[f64]) -> impl Iterator<Item = f64> + '_ {
    values.iter().copied().filter(|value| value.is_finite())
}

/// The population mean and variance of the finite values, both 0 without any.
#[expect(clippy::cast_precision_loss, reason = "Row counts stay far below 2^53")]
fn moments(values: &[f64], ddof: usize) -> (f64, f64, usize) {
    let count = finite(values).count();
    let mean = finite(values).sum::<f64>() / count as f64;
    let mean = if mean.is_nan() { 0.0 } else { mean };
    let squares: f64 = finite(values).map(|value| (value - mean) * (value - mean)).sum();
    (mean, squares / count.saturating_sub(ddof).max(1) as f64, count)
}

/// `Standardize(eps=1e-6)`'s mean and scale.
fn standardize(values: &[f64]) -> (f64, f64) {
    let (mean, variance, count) = moments(values, 0);
    let variance = if count == 0 { 0.0 } else { variance };
    (mean, variance.sqrt() + 1e-6)
}

/// `_constant_feature_mask`.
#[expect(clippy::cast_precision_loss, reason = "Row counts stay far below 2^53")]
fn is_constant(variance: f64, mean: f64, count: usize) -> bool {
    let n = count as f64;
    variance <= n * f64::EPSILON * variance + (n * mean * f64::EPSILON).powi(2)
}

/// `ClipSigma(threshold=4)`'s bounds: four sample deviations around the
/// mean, refitted on the values within the first bounds.
#[expect(clippy::cast_precision_loss, reason = "Row counts stay far below 2^53")]
fn sigma_bounds(values: &[f64]) -> (f64, f64) {
    let (mean, variance, _) = moments(values, 1);
    let std = variance.sqrt().max(1e-6);
    let (lower, upper) = (mean - 4.0 * std, mean + 4.0 * std);
    let kept: Vec<f64> = finite(values)
        .filter(|value| (lower..=upper).contains(value))
        .collect();
    if kept.is_empty() {
        return (lower, upper);
    }
    let mean = kept.iter().sum::<f64>() / kept.len() as f64;
    let squares: f64 = kept.iter().map(|value| (value - mean) * (value - mean)).sum();
    let std = (squares / (kept.len() - 1).max(1) as f64).sqrt().max(1e-6);
    (mean - 4.0 * std, mean + 4.0 * std)
}

/// `Clip(-3, 3, method="saturate")`.
fn saturate(value: f64) -> f64 {
    if value.is_nan() {
        return value;
    }
    let unit = (value / 3.0).abs();
    let root = (unit * unit + 1.0).sqrt();
    let unit = if root.is_finite() { unit / root } else { 1.0 };
    unit * 3.0 * if value == 0.0 { 0.0 } else { value.signum() }
}

/// The Yeo-Johnson transform as NVIDIA computes it.
///
/// Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/processing/numerical/power.py#L21-L43>.
fn yeo_johnson(value: f64, lambda: f64) -> f64 {
    let log = value.abs().ln_1p();
    let exponent = if value >= 0.0 { lambda } else { 2.0 - lambda };
    let out = if exponent.abs() < f64::EPSILON {
        log
    } else {
        (exponent * log).exp_m1() / exponent
    };
    out.copysign(value)
}

/// `PowerTransform(standardize=True)`: the Yeo-Johnson lambda that maximizes
/// the log-likelihood, by 44 golden-section steps, then the transformed
/// values' mean and scale.
///
/// Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/processing/numerical/power.py#L72-L280>.
#[expect(clippy::cast_precision_loss, reason = "Row counts stay far below 2^53")]
fn power(values: &[f64]) -> Choice {
    let observed: Vec<f64> = finite(values).collect();
    let count = observed.len().max(1) as f64;
    let (mean, variance, _) = moments(values, 0);
    let constant = is_constant(variance, mean, observed.len().max(1));
    let moments_of = |lambda: f64| {
        let transformed: Vec<f64> = observed.iter().map(|&value| yeo_johnson(value, lambda)).collect();
        let mean = transformed.iter().sum::<f64>() / count;
        let variance = transformed
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count;
        (mean, variance)
    };
    let lambda = if constant {
        1.0
    } else {
        let jacobian: f64 = observed
            .iter()
            .map(|value| value.abs().ln_1p().copysign(*value))
            .sum();
        let likelihood = |lambda: f64| {
            let (_, variance) = moments_of(lambda);
            if variance.is_finite() && variance >= f64::MIN_POSITIVE {
                -count / 2.0 * variance.ln() + (lambda - 1.0) * jacobian
            } else {
                f64::NEG_INFINITY
            }
        };
        let (mut left, mut right) = yeo_johnson_bounds(&observed);
        let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
        let span = (right - left) * ratio;
        let (mut c, mut d) = (right - span, left + span);
        let (mut fc, mut fd) = (likelihood(c), likelihood(d));
        for _ in 0..44 {
            if fc < fd {
                left = c;
                let span = (right - left) * ratio;
                (c, d) = (d, left + span);
                (fc, fd) = (fd, likelihood(d));
            } else {
                right = d;
                let span = (right - left) * ratio;
                (c, d) = (right - span, c);
                (fc, fd) = (likelihood(c), fc);
            }
        }
        f64::midpoint(left, right)
    };
    let (mean, variance) = moments_of(lambda);
    let scale = if is_constant(variance, mean, observed.len().max(1)) {
        1.0
    } else {
        variance.sqrt()
    };
    Choice::Power { lambda, mean, scale }
}

/// `_yeojohnson_bounds`: lambdas that keep every transformed value finite.
fn yeo_johnson_bounds(values: &[f64]) -> (f64, f64) {
    let max_abs = values.iter().fold(0.0_f64, |max, value| max.max(value.abs()));
    if max_abs == 0.0 {
        return (1.0, 1.0);
    }
    let log = (20.0 * max_abs).ln_1p();
    let log_eps = f64::EPSILON.ln();
    let lower = (f64::MIN_POSITIVE.ln() - log_eps) / 2.0 / log;
    let upper = f64::midpoint(f64::MAX.ln(), log_eps) / log;
    if values.iter().all(|value| *value < 0.0) {
        (2.0 - upper, 2.0 - lower)
    } else if values.iter().any(|value| *value < 0.0) {
        let mixed_lower = (2.0 - upper).max(lower);
        (mixed_lower, (2.0 - mixed_lower).min(upper))
    } else {
        (lower, upper)
    }
}

/// `torch.quantile`'s linear interpolation of the sorted `values` at `q`.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Row counts stay far below 2^53"
)]
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let rank = q * (sorted.len() - 1) as f64;
    let below = rank.floor() as usize;
    let above = rank.ceil() as usize;
    lerp(sorted[below], sorted[above], rank - below as f64)
}

/// `torch.lerp`.
fn lerp(start: f64, end: f64, weight: f64) -> f64 {
    if weight.abs() < 0.5 {
        start + weight * (end - start)
    } else {
        end - (end - start) * (1.0 - weight)
    }
}

/// `RobustScale()`: the median and interquartile range.
#[expect(clippy::float_cmp, reason = "PyTorch compares exactly")]
fn robust(values: &[f64]) -> Choice {
    let mut sorted: Vec<f64> = finite(values).collect();
    sorted.sort_by(f64::total_cmp);
    let (lower, median, upper) = (
        quantile(&sorted, 0.25),
        quantile(&sorted, 0.5),
        quantile(&sorted, 0.75),
    );
    let scale = if lower == upper { 1.0 } else { upper - lower };
    Choice::Robust { median, scale }
}

/// `RankGaussian(max_knots=8192)`'s knots: values and their mid-rank
/// probabilities, at most [`MAX_KNOTS`] of them spaced evenly in normal
/// quantiles.
///
/// Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/processing/numerical/rank_gaussian.py#L49-L140>.
#[expect(clippy::cast_precision_loss, reason = "Row counts stay far below 2^53")]
fn rank_gaussian(values: &[f64]) -> Choice {
    let rows = values.len();
    let mut sorted: Vec<f64> = values
        .iter()
        .map(|&value| if value.is_finite() { value } else { f64::INFINITY })
        .collect();
    sorted.sort_by(f64::total_cmp);
    let count = finite(values).count();
    let below: Vec<usize> = sorted
        .iter()
        .map(|value| sorted.partition_point(|other| other < value))
        .collect();
    let probabilities: Vec<f64> = sorted
        .iter()
        .zip(&below)
        .map(|(value, &left)| {
            let right = sorted.partition_point(|other| other <= value);
            ((left + right) as f64 / (2 * count.max(1)) as f64).min(1.0 - f64::EPSILON)
        })
        .collect();
    let last = count.saturating_sub(1);
    let positions: Vec<usize> = if rows <= MAX_KNOTS {
        (0..rows).map(|row| row.min(last)).collect()
    } else {
        // Each row's distinct-value index, and the first row of each.
        let mut distinct = Vec::with_capacity(rows);
        let mut index = 0_usize;
        for (row, &left) in below.iter().enumerate() {
            index += usize::from(left == row);
            distinct.push(index - 1);
        }
        let distincts = distinct[last] + 1;
        if distincts <= MAX_KNOTS {
            (0..MAX_KNOTS)
                .map(|step| distinct.partition_point(|&d| d < step.min(distincts - 1)))
                .collect()
        } else {
            let (lower, upper) = (ndtri(probabilities[0]), ndtri(probabilities[last]));
            let mut positions: Vec<usize> = (0..MAX_KNOTS)
                .map(|step| {
                    let q = ndtr(lerp(lower, upper, step as f64 / (MAX_KNOTS - 1) as f64));
                    let above = probabilities.partition_point(|&p| p < q).min(last);
                    let below = above.saturating_sub(1);
                    if q - probabilities[below] < probabilities[above] - q {
                        below
                    } else {
                        above
                    }
                })
                .collect();
            positions[0] = 0;
            positions[MAX_KNOTS - 1] = last;
            positions
        }
    };
    let pick = |source: &[f64]| {
        positions
            .iter()
            .map(|&p| if count == 0 { f64::NAN } else { source[p] })
            .collect()
    };
    Choice::RankGaussian {
        values: pick(&sorted),
        probabilities: pick(&probabilities),
    }
}

/// `_batched_interp`: `value` interpolated linearly between the knots, and
/// clamped to the end probabilities.
fn interpolate(value: f64, knots: &[f64], probabilities: &[f64]) -> f64 {
    let n = knots.len();
    if n == 1 {
        return probabilities[0];
    }
    let index = knots.partition_point(|&knot| knot <= value).clamp(1, n - 1);
    let (x0, x1) = (knots[index - 1], knots[index]);
    let denominator = x1 - x0;
    let weight = if denominator == 0.0 {
        0.0
    } else {
        (value - x0) / denominator
    };
    let result = lerp(probabilities[index - 1], probabilities[index], weight);
    if value <= knots[0] {
        probabilities[0]
    } else if value >= knots[n - 1] {
        probabilities[n - 1]
    } else {
        result
    }
}

/// `torch.special.ndtr`: the standard normal distribution function.
fn ndtr(x: f64) -> f64 {
    let y = x * std::f64::consts::FRAC_1_SQRT_2;
    let z = y.abs();
    if z < std::f64::consts::FRAC_1_SQRT_2 {
        0.5 + 0.5 * libm::erf(y)
    } else {
        let r = 0.5 * libm::erfc(z);
        if y > 0.0 { 1.0 - r } else { r }
    }
}

fn polynomial(x: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().fold(0.0, |sum, &c| sum.mul_add(x, c))
}

/// `polynomial` with a leading coefficient of 1.
fn monic(x: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().fold(1.0, |sum, &c| sum.mul_add(x, c))
}

/// `torch.special.ndtri`, Cephes' inverse of [`ndtr`].
///
/// Port of <https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Math.h#L2232-L2330>.
#[expect(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    clippy::float_cmp,
    reason = "Cephes' coefficients, as published, and its exact end points"
)]
fn ndtri(y0: f64) -> f64 {
    const P0: [f64; 5] = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ];
    const Q0: [f64; 8] = [
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ];
    const P1: [f64; 9] = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ];
    const Q1: [f64; 8] = [
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ];
    const P2: [f64; 9] = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ];
    const Q2: [f64; 8] = [
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ];
    const EXP_MINUS_2: f64 = 0.13533528323661269189;
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&y0) {
        return f64::NAN;
    }
    let (mut y, negate) = if y0 > 1.0 - EXP_MINUS_2 {
        (1.0 - y0, false)
    } else {
        (y0, true)
    };
    if y > EXP_MINUS_2 {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polynomial(y2, &P0) / monic(y2, &Q0));
        return x * (2.0 * std::f64::consts::PI).sqrt();
    }
    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polynomial(z, &P1) / monic(z, &Q1)
    } else {
        z * polynomial(z, &P2) / monic(z, &Q2)
    };
    if negate { x1 - x0 } else { x0 - x1 }
}

#[cfg(test)]
mod tests {
    use super::{ndtr, ndtri};

    #[test]
    fn ndtri_inverts_ndtr() {
        assert!((ndtri(0.975) - 1.959_963_984_540_054).abs() < 1e-14);
        assert!((ndtri(0.5)).abs() < 1e-15);
        for x in [-9.0, -3.0, -0.5, 0.1, 2.0, 5.0] {
            assert!((ndtri(ndtr(x)) - x).abs() < 1e-9 * (1.0 + x.abs()), "{x}");
        }
    }
}

/// NVIDIA's recorded cases from `tools/export_kumo_fixtures.py`.
#[cfg(test)]
pub(crate) mod fixtures {
    use serde_json::Value;

    use super::Draws;
    use crate::preprocess::columns::{self, Column};
    use crate::preprocess::labels::Labels;

    pub(crate) struct Case {
        pub json: Value,
        /// The encoded `[rows, features]` matrix and which features are categorical.
        pub x: Vec<f64>,
        pub categorical: Vec<bool>,
        pub features: usize,
        pub context: usize,
        pub labels: Labels,
        pub draws: Draws,
    }

    fn number(value: &Value) -> f64 {
        value
            .as_f64()
            .unwrap_or_else(|| value.as_str().expect("a number").parse().expect("a float"))
    }

    fn matrix(value: &Value) -> Vec<Vec<f64>> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(number).collect())
            .collect()
    }

    fn orders(value: &Value) -> Vec<Vec<usize>> {
        value[0]
            .as_array()
            .unwrap()
            .iter()
            .map(|order| {
                order
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| usize::try_from(i.as_u64().unwrap()).unwrap())
                    .collect()
            })
            .collect()
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "The fixture's integer labels are exact"
    )]
    pub(crate) fn case(name: &str) -> Case {
        let json = crate::preprocess::fixture(&format!("kumo_tabular/{name}.json"));
        let context = usize::try_from(json["context_rows"].as_u64().unwrap()).unwrap();
        let mut table = Vec::new();
        let mut categorical = Vec::new();
        for column in json["columns"].as_array().unwrap() {
            let values = column["values"].as_array().unwrap();
            if column["categorical"].as_bool().unwrap() {
                // pandas turned the missing values into NaN.
                let values: Vec<Value> = values
                    .iter()
                    .map(|v| if v == "nan" { Value::Null } else { v.clone() })
                    .collect();
                let mut names: Vec<String> = values
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                names.sort();
                names.dedup();
                let codes = values
                    .iter()
                    .map(|v| {
                        v.as_str().map_or(-1, |s| {
                            i64::try_from(names.binary_search(&s.to_string()).unwrap()).unwrap()
                        })
                    })
                    .collect();
                table.push(Column::Categorical {
                    categories: Labels::Text(names),
                    codes,
                });
            } else {
                table.push(Column::Numeric(values.iter().map(number).collect()));
            }
            categorical.push(column["categorical"].as_bool().unwrap());
        }
        let training: Vec<Column> = table
            .iter()
            .map(|column| match column {
                Column::Numeric(values) => Column::Numeric(values[..context].to_vec()),
                Column::Categorical { categories, codes } => Column::Categorical {
                    categories: categories.clone(),
                    codes: codes[..context].to_vec(),
                },
            })
            .collect();
        let categories = columns::categories(&training).unwrap();
        let x = columns::encode(&table, &categories).unwrap();
        let y = json["y"].as_array().unwrap();
        let labels = if y[0].is_string() {
            Labels::Text(y[..context].iter().map(|v| v.as_str().unwrap().into()).collect())
        } else {
            Labels::Int(y[..context].iter().map(|v| v.as_f64().unwrap() as i64).collect())
        };
        let draws = &json["draws"];
        let draws = Draws {
            class_orders: draws.get("class_orders").map(orders).unwrap_or_default(),
            signs: matrix(&draws["flip_signs"][0]),
            column_orders: orders(&draws["column_orders"]),
            codebooks: draws.get("codebooks").map_or_else(
                || vec![Vec::new(); json["members"].as_array().unwrap().len()],
                |books| {
                    let books = books.as_array().unwrap();
                    books
                        .iter()
                        .map(|book| orders(&Value::Array(vec![book.clone()])))
                        .collect()
                },
            ),
        };
        Case {
            features: categorical.len(),
            json,
            x,
            categorical,
            context,
            labels,
            draws,
        }
    }

    pub(crate) fn rows(value: &Value) -> Vec<Vec<f64>> {
        matrix(value)
    }
}

#[cfg(test)]
mod parity {
    use super::Preprocessor;
    use super::fixtures::{case, rows};

    #[test]
    fn members_see_nvidias_preprocessed_rows() {
        for name in ["numeric", "mixed", "many_classes"] {
            let case = case(name);
            let members = case.json["members"].as_array().unwrap();
            let draws = case.draws.clone();
            let context = &case.x[..case.context * case.features];
            let preprocessor = Preprocessor::fit_with(
                context,
                &case.categorical,
                &case.labels,
                members.len(),
                |_, _, _| draws,
            )
            .unwrap();
            for (index, member) in members.iter().enumerate() {
                let names = member["columns"].as_array().unwrap();
                let codes: Vec<usize> = member["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| usize::try_from(c.as_u64().unwrap()).unwrap())
                    .collect();
                assert_eq!(
                    preprocessor.class_codes(&case.labels, index).unwrap(),
                    codes,
                    "{name} member {index} classes"
                );
                for (rows_name, x, expected) in [
                    ("context", context, rows(&member["x"])),
                    (
                        "query",
                        &case.x[case.context * case.features..],
                        rows(&member["query"]),
                    ),
                ] {
                    let ours = preprocessor.transform(x, index).unwrap();
                    assert_eq!(ours.columns, names.len(), "{name} member {index}");
                    let mask: Vec<bool> = names
                        .iter()
                        .map(|n| {
                            let n = n.as_str().unwrap();
                            !n.ends_with("__count")
                                && case.json["columns"]
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .any(|c| c["name"] == n && c["categorical"].as_bool().unwrap())
                        })
                        .collect();
                    assert_eq!(ours.categorical, mask, "{name} member {index}");
                    let mut worst = (0.0_f64, 0, 0);
                    for (r, row) in expected.iter().enumerate() {
                        for (c, &value) in row.iter().enumerate() {
                            let got = f64::from(ours.x[r * ours.columns + c]);
                            let error = if value.is_nan() && got.is_nan() {
                                0.0
                            } else {
                                (got - value).abs() / (1.0 + value.abs())
                            };
                            if error > worst.0 || error.is_nan() {
                                worst = (error, r, c);
                            }
                        }
                    }
                    assert!(
                        worst.0 < 1e-5,
                        "{name} member {index} {rows_name}: {worst:?} {}",
                        names[worst.2]
                    );
                }
            }
        }
    }
}

//! Regression outputs from the regressor's decoded quantiles.
//!
//! A port of the default path of `TabICL`'s `QuantileToDistribution`
//! (`tabicl/_model/quantile_dist.py`): sort the quantiles, interpolate linearly
//! between them, and extrapolate exponential tails fitted to the outermost
//! quantiles. Each estimator is summarized on its own and the results are
//! averaged, as `TabICLRegressor.predict` does. Values stay in the standardized
//! target units the model works in.
// Counts here are at most a few thousand, so `as f64` is exact, and the
// results are FP32 like TabICL's.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::{
    DecoderOutputs,
    error::{Error, Result},
};

/// `QuantileDistributionConfig.TOL`.
const TOL: f64 = 1e-6;
/// `QuantileDistributionConfig.TAIL_QUANTILES_FOR_ESTIMATION`.
const TAIL_QUANTILES: usize = 20;
/// `QuantileDistributionConfig.MIN_BETA` and `MAX_BETA`.
const BETA_RANGE: (f64, f64) = (0.01, 100.0);

/// A regression statistic, named as `TabICLRegressor.predict`'s `output_type`.
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionOutput {
    /// The mean of the quantiles, `[Q]`.
    Mean,
    /// The unbiased variance of the quantiles, `[Q]`.
    Variance,
    /// The inverse CDF at 0.5, `[Q]`.
    Median,
    /// The inverse CDF at each probability level, `[Q, levels]`.
    Quantiles(Vec<f32>),
    /// The sorted decoded quantiles, `[Q, K]`.
    RawQuantiles,
}

impl RegressionOutput {
    /// Parse an `output_type` name. `alphas` are the levels for `"quantiles"`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] for an unknown name or a non-finite level.
    pub fn parse(name: &str, alphas: &[f32]) -> Result<Self> {
        match name {
            "mean" => Ok(Self::Mean),
            "variance" => Ok(Self::Variance),
            "median" => Ok(Self::Median),
            "quantiles" if alphas.iter().all(|alpha| alpha.is_finite()) => {
                Ok(Self::Quantiles(alphas.to_vec()))
            }
            "quantiles" => Err(Error::InvalidArgument("quantile levels must be finite".into())),
            "raw_quantiles" => Ok(Self::RawQuantiles),
            _ => Err(Error::InvalidArgument(format!(
                "unknown output_type {name:?}; expected mean, variance, median, quantiles, or raw_quantiles"
            ))),
        }
    }

    fn columns(&self, quantiles: usize) -> usize {
        match self {
            Self::Mean | Self::Variance | Self::Median => 1,
            Self::Quantiles(alphas) => alphas.len(),
            Self::RawQuantiles => quantiles,
        }
    }
}

/// Summarize `[B, Q, K]` decoded quantiles into each requested output, averaged
/// over the `B` estimators. Each result is row-major `[Q, columns]`; the second
/// value is `columns`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when there are fewer than two quantiles.
pub fn summarize(decoded: &DecoderOutputs, outputs: &[RegressionOutput]) -> Result<Vec<(Vec<f32>, usize)>> {
    let [b, q, k] = decoded.shape();
    if k < 2 {
        return Err(Error::InvalidArgument(
            "regression needs at least two quantiles".into(),
        ));
    }
    let levels = levels(k);
    let mut sums: Vec<Vec<f64>> = outputs
        .iter()
        .map(|output| vec![0.0; q * output.columns(k)])
        .collect();
    let mut sorted = vec![0.0; k];
    for (row, raw) in decoded.values().chunks_exact(k).enumerate() {
        let row = row % q;
        sorted.copy_from_slice(raw);
        sorted.sort_unstable_by(f32::total_cmp);
        let distribution = Distribution::new(&sorted, &levels);
        for (output, sums) in outputs.iter().zip(&mut sums) {
            let columns = output.columns(k);
            let sums = &mut sums[row * columns..][..columns];
            match output {
                RegressionOutput::Mean => sums[0] += mean(&sorted),
                RegressionOutput::Variance => {
                    let mean = mean(&sorted);
                    let squares: f64 = sorted
                        .iter()
                        .map(|&value| (f64::from(value) - mean).powi(2))
                        .sum();
                    sums[0] += squares / (k - 1) as f64;
                }
                RegressionOutput::Median => sums[0] += distribution.icdf(0.5),
                RegressionOutput::Quantiles(alphas) => {
                    for (sum, &alpha) in sums.iter_mut().zip(alphas) {
                        *sum += distribution.icdf(f64::from(alpha));
                    }
                }
                RegressionOutput::RawQuantiles => {
                    for (sum, &value) in sums.iter_mut().zip(&sorted) {
                        *sum += f64::from(value);
                    }
                }
            }
        }
    }
    Ok(outputs
        .iter()
        .zip(sums)
        .map(|(output, sums)| {
            let values = sums.into_iter().map(|sum| (sum / b as f64) as f32).collect();
            (values, output.columns(k))
        })
        .collect())
}

/// The probability levels of `K` decoded quantiles: `linspace(0, 1, K + 2)[1:-1]`.
fn levels(k: usize) -> Vec<f64> {
    (1..=k)
        .map(|i| f64::from((i as f64 / (k + 1) as f64) as f32))
        .collect()
}

fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64
}

/// One row's sorted quantiles with linear interpolation and exponential tails.
struct Distribution<'a> {
    quantiles: &'a [f32],
    levels: &'a [f64],
    /// Left and right tail scales, `β_L` and `β_R`.
    beta: (f64, f64),
}

impl<'a> Distribution<'a> {
    fn new(quantiles: &'a [f32], levels: &'a [f64]) -> Self {
        let k = TAIL_QUANTILES.min(quantiles.len() / 4).max(1);
        let n = quantiles.len();
        let left = tail_slope(
            &quantiles[..k],
            levels[..k].iter().map(|&alpha| alpha.max(TOL).ln()),
        );
        let right = tail_slope(
            &quantiles[n - k..],
            levels[n - k..].iter().map(|&alpha| (1.0 - alpha).max(TOL).ln()),
        );
        let clamp = |beta: f64| beta.abs().clamp(BETA_RANGE.0, BETA_RANGE.1);
        Self {
            quantiles,
            levels,
            beta: (clamp(left), clamp(-right)),
        }
    }

    /// The quantile function `Q(α)`.
    fn icdf(&self, alpha: f64) -> f64 {
        let (q, levels) = (self.quantiles, self.levels);
        let (alpha_l, alpha_r) = (levels[0], levels[levels.len() - 1]);
        let (q_l, q_r) = (f64::from(q[0]), f64::from(q[q.len() - 1]));
        if alpha < alpha_l {
            // Q(α) = β_L·ln(α) + c_L, through (α_L, q_L).
            return q_l + self.beta.0 * (alpha.max(TOL).ln() - alpha_l.max(TOL).ln());
        }
        if alpha > alpha_r {
            // Q(α) = -β_R·ln(1 - α) + c_R, through (α_R, q_R).
            let one_minus_r = 1.0 - alpha_r.min(1.0 - TOL);
            return q_r - self.beta.1 * ((1.0 - alpha).max(TOL).ln() - one_minus_r.ln());
        }
        if alpha >= alpha_r {
            return q_r;
        }
        // The last segment whose lower level is at most α.
        let segment = levels[..levels.len() - 1]
            .partition_point(|&level| level <= alpha)
            .saturating_sub(1);
        let (lo, hi) = (levels[segment], levels[segment + 1]);
        let t = ((alpha - lo) / (hi - lo).max(TOL)).clamp(0.0, 1.0);
        let (q_lo, q_hi) = (f64::from(q[segment]), f64::from(q[segment + 1]));
        q_lo + t * (q_hi - q_lo)
    }
}

/// The least-squares slope of `quantiles` against `x`.
fn tail_slope(quantiles: &[f32], x: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = quantiles.len() as f64;
    let x_mean = x.clone().sum::<f64>() / n;
    let q_mean = mean(quantiles);
    let (mut covariance, mut variance) = (0.0, 0.0);
    for (x, &q) in x.zip(quantiles) {
        covariance += (f64::from(q) - q_mean) * (x - x_mean);
        variance += (x - x_mean).powi(2);
    }
    (covariance / n) / (variance / n).max(TOL)
}

#[cfg(test)]
mod tests {
    use super::{RegressionOutput, levels, summarize};
    use crate::DecoderOutputs;

    fn decoded(shape: [usize; 3], values: Vec<f32>) -> DecoderOutputs {
        DecoderOutputs::new(shape, values).expect("valid shape")
    }

    #[test]
    fn levels_match_tabicl_linspace() {
        let levels = levels(999);
        assert_eq!(levels.len(), 999);
        assert!((levels[0] - 0.001).abs() < 1e-7);
        assert!((levels[499] - 0.5).abs() < 1e-7);
        assert!((levels[998] - 0.999).abs() < 1e-7);
    }

    #[test]
    fn linear_quantiles_give_exact_statistics() {
        // Q(α) = 10α - 5 at every level, shuffled: the sort restores it.
        let k = 999;
        let levels = levels(k);
        let mut values: Vec<f32> = levels.iter().map(|&alpha| (10.0 * alpha - 5.0) as f32).collect();
        values.reverse();
        let outputs = [
            RegressionOutput::Mean,
            RegressionOutput::Median,
            RegressionOutput::Quantiles(vec![0.25, 0.75]),
            RegressionOutput::Variance,
        ];
        let results = summarize(&decoded([1, 1, k], values), &outputs).expect("summarize");
        assert!(results[0].0[0].abs() < 1e-5);
        assert!(results[1].0[0].abs() < 1e-5);
        assert!((results[2].0[0] + 2.5).abs() < 1e-5 && (results[2].0[1] - 2.5).abs() < 1e-5);
        // The unbiased variance of 10α over the levels.
        let mean = levels.iter().sum::<f64>() / k as f64;
        let expected = levels
            .iter()
            .map(|alpha| 100.0 * (alpha - mean).powi(2))
            .sum::<f64>()
            / (k - 1) as f64;
        assert!((f64::from(results[3].0[0]) - expected).abs() < 1e-4);
    }

    #[test]
    fn tails_extrapolate_exponentially_and_estimators_average() {
        // Exponential quantiles: Q(α) = ln(α) on the left; both estimators
        // are equal, so the average is the single-estimator value.
        let k = 999;
        let row: Vec<f32> = levels(k).iter().map(|&alpha| alpha.ln() as f32).collect();
        let values = [row.clone(), row].concat();
        let alpha = 0.0001_f32;
        let results = summarize(
            &decoded([2, 1, k], values),
            &[RegressionOutput::Quantiles(vec![alpha])],
        )
        .expect("summarize");
        // The fitted left slope is 1, so the tail continues ln(α).
        assert!((f64::from(results[0].0[0]) - f64::from(alpha).ln()).abs() < 1e-3);
    }

    #[test]
    fn unknown_output_is_rejected() {
        assert!(RegressionOutput::parse("mode", &[]).is_err());
        assert!(RegressionOutput::parse("quantiles", &[f32::NAN]).is_err());
        assert_eq!(
            RegressionOutput::parse("quantiles", &[0.5]).expect("parse"),
            RegressionOutput::Quantiles(vec![0.5])
        );
    }
}

//! scikit-learn's `PowerTransformer(method="yeo-johnson", standardize=True)`,
//! in `f64`, one column at a time.
//!
//! Each column's lambda maximizes `SciPy`'s Yeo-Johnson log-likelihood, found
//! by `SciPy`'s `fminbound` within the bounds `yeojohnson_normmax` sets so the
//! transform cannot overflow. A `StandardScaler` then standardizes the result.
//!
//! The fit, log-likelihood, and `fminbound` follow `SciPy` 1.18
//! (`scipy/stats/_morestats.py`, `scipy/optimize/_optimize.py`), and the
//! transformer follows scikit-learn 1.9 (`sklearn/preprocessing/_data.py`).
//! Copyright (c) 2001-2002 Enthought, Inc. 2003, `SciPy` Developers, and
//! (c) 2007-2026 the scikit-learn developers; see `LICENSES/SciPy-BSD-3-Clause.txt`
//! and `LICENSES/scikit-learn-BSD-3-Clause.txt`.

use super::scale::{is_constant, mean_variance, standard_scaler};
use crate::{Error, Result};

/// `SciPy`'s `_yeojohnson_transform` of one value.
///
/// Port of <https://github.com/scipy/scipy/blob/v1.18.1/scipy/stats/_morestats.py#L1728-L1759>.
pub(crate) fn yeo_johnson(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if lambda.abs() < f64::EPSILON {
            x.ln_1p()
        } else {
            (lambda * x.ln_1p()).exp_m1() / lambda
        }
    } else if (lambda - 2.0).abs() > f64::EPSILON {
        -((2.0 - lambda) * (-x).ln_1p()).exp_m1() / (2.0 - lambda)
    } else {
        -(-x).ln_1p()
    }
}

/// The log of the variance of `((1 + y)^power - 1) / power` over the `log1p(y)`
/// values `logs`. Like `SciPy`'s `_log_var`, it never forms the powers
/// themselves, which can overflow or lose precision near -1/power: it takes
/// the variance of `exp(power·log1p(y) − largest)` and adds the shift back.
///
/// Equivalent of <https://github.com/scipy/scipy/blob/v1.18.1/scipy/stats/_morestats.py#L903-L909>.
fn log_variance_of_powers(logs: &[f64], power: f64) -> f64 {
    if power.abs() < f64::EPSILON {
        return mean_variance(logs, 0).1.ln();
    }
    let largest = logs
        .iter()
        .fold(f64::NEG_INFINITY, |high, log| high.max(power * log));
    let shifted: Vec<f64> = logs.iter().map(|log| (power * log - largest).exp()).collect();
    mean_variance(&shifted, 0).1.ln() + 2.0 * largest - 2.0 * power.abs().ln()
}

/// `SciPy`'s `yeojohnson_llf(lambda, data)`.
///
/// Port of <https://github.com/scipy/scipy/blob/v1.18.1/scipy/stats/_morestats.py#L1881-L1920>.
#[expect(clippy::cast_precision_loss, reason = "Counts stay far below 2^53")]
fn log_likelihood(data: &[f64], lambda: f64) -> f64 {
    let log_variance = if data.iter().all(|&x| x >= 0.0) {
        let logs: Vec<f64> = data.iter().map(|x| x.ln_1p()).collect();
        log_variance_of_powers(&logs, lambda)
    } else if data.iter().all(|&x| x < 0.0) {
        let logs: Vec<f64> = data.iter().map(|x| (-x).ln_1p()).collect();
        log_variance_of_powers(&logs, 2.0 - lambda)
    } else {
        let transformed: Vec<f64> = data.iter().map(|&x| yeo_johnson(x, lambda)).collect();
        mean_variance(&transformed, 0).1.ln()
    };
    let jacobian: f64 = data.iter().map(|x| x.signum() * x.abs().ln_1p()).sum();
    -(data.len() as f64) / 2.0 * log_variance + (lambda - 1.0) * jacobian
}

/// `SciPy`'s `_minimize_scalar_bounded` (`fminbound`): golden-section search
/// with parabolic steps for a minimum of `f` in `[a, b]`.
///
/// Port of <https://github.com/scipy/scipy/blob/v1.18.1/scipy/optimize/_optimize.py#L2289-L2436>.
#[expect(
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "SciPy's algorithm, names, and exact comparisons"
)]
fn fminbound(mut f: impl FnMut(f64) -> f64, mut a: f64, mut b: f64, xatol: f64, maxfun: usize) -> f64 {
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let step_sign = |value: f64| if value < 0.0 { -1.0 } else { 1.0 };

    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0_f64, 0.0_f64);
    let mut fx = f(xf);
    let mut calls = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    rat = tol1 * step_sign(xm - xf);
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let x = xf + step_sign(rat) * rat.abs().max(tol1);
        let fu = f(x);
        calls += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            (fulc, ffulc) = (nfc, fnfc);
            (nfc, fnfc) = (xf, fx);
            (xf, fx) = (x, fu);
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                (fulc, ffulc) = (nfc, fnfc);
                (nfc, fnfc) = (x, fu);
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                (fulc, ffulc) = (x, fu);
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if calls >= maxfun {
            break;
        }
    }
    xf
}

/// `SciPy`'s `yeojohnson_normmax(x)`: the lambda that maximizes the
/// log-likelihood, searched within bounds that keep the transformed values and
/// their squares finite.
///
/// Port of <https://github.com/scipy/scipy/blob/v1.18.1/scipy/stats/_morestats.py#L1924-L2024>.
#[expect(clippy::manual_midpoint, reason = "SciPy's formula")]
fn normmax(x: &[f64]) -> Result<f64> {
    if x.iter().all(|&value| value == 0.0) {
        return Ok(1.0);
    }
    let largest = x.iter().fold(0.0, |high: f64, value| high.max(value.abs()));
    let log1p_max_x = (20.0 * largest).ln_1p();
    let log_eps = f64::EPSILON.ln();
    let log_tiny = (f64::MIN_POSITIVE.ln() - log_eps) / 2.0;
    let log_max = (f64::MAX.ln() + log_eps) / 2.0;
    let (mut lower, mut upper) = (log_tiny / log1p_max_x, log_max / log1p_max_x);
    if x.iter().all(|&value| value < 0.0) {
        (lower, upper) = (2.0 - upper, 2.0 - lower);
    } else if x.iter().any(|&value| value < 0.0) {
        (lower, upper) = (lower.max(2.0 - upper), upper.min(2.0 - lower));
    }
    if !(lower.is_finite() && upper.is_finite()) || lower > upper {
        return Err(Error::InvalidArgument("Yeo-Johnson bounds are not finite".into()));
    }
    let negative_log_likelihood = |lambda: f64| {
        let value = log_likelihood(x, lambda);
        if value.is_infinite() {
            f64::INFINITY
        } else {
            -value
        }
    };
    Ok(fminbound(negative_log_likelihood, lower, upper, 1.48e-8, 500))
}

/// `PowerTransformer.fit_transform` of one column, in place: its lambda, then
/// the `StandardScaler` mean and scale of the transformed values. A constant
/// column gets lambda 1 and, in the fit only, no transform at all:
/// `expm1(log1p(x))` is not exactly `x`.
///
/// Port of
/// <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_data.py#L3397-L3414>
/// and <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_data.py#L3416-L3458>.
pub(crate) fn fit_power(values: &mut [f64]) -> Result<[f64; 3]> {
    let (mean, variance) = mean_variance(values, 0);
    let lambda = if is_constant(variance, mean, values.len()) {
        1.0
    } else {
        let lambda = normmax(values)?;
        for value in values.iter_mut() {
            *value = yeo_johnson(*value, lambda);
        }
        lambda
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument(
            "the power transform of the training data is not finite".into(),
        ));
    }
    let (mean, scale) = standard_scaler(values);
    for value in values.iter_mut() {
        *value = (*value - mean) / scale;
    }
    Ok([lambda, mean, scale])
}

#[cfg(test)]
mod tests {
    use super::{fit_power, log_likelihood, normmax, yeo_johnson};
    use crate::preprocess::{fixture, numbers};

    fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{what}: {actual} vs {expected}"
        );
    }

    #[test]
    fn the_fit_matches_scipy() {
        for case in fixture("yeo_johnson.json")["normmax"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let x = numbers(&case["x"]);
            for pair in case["llf"].as_array().unwrap() {
                let (at, expected) = (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap());
                close(
                    log_likelihood(&x, at),
                    expected,
                    1e-13,
                    &format!("{name} llf({at})"),
                );
            }
            // fminbound stops within xatol (1.48e-8) along its own path, and
            // last-ULP differences in the log-likelihood can change that path.
            close(
                normmax(&x).unwrap(),
                case["lambda"].as_f64().unwrap(),
                1e-7,
                &format!("{name} lambda"),
            );
        }
    }

    #[test]
    fn the_power_transformer_matches_scikit_learn() {
        for case in fixture("yeo_johnson.json")["power_transformer"]
            .as_array()
            .unwrap()
        {
            let name = case["name"].as_str().unwrap();
            let columns = usize::try_from(case["columns"].as_u64().unwrap()).unwrap();
            let (x, query) = (numbers(&case["x"]), numbers(&case["query"]));
            let (fitted, transformed) = (numbers(&case["fit_transform"]), numbers(&case["transform"]));
            for column in 0..columns {
                let what = format!("{name} column {column}");
                let mut values: Vec<f64> = x.iter().skip(column).step_by(columns).copied().collect();
                let [lambda, mean, scale] = fit_power(&mut values).unwrap();
                close(
                    lambda,
                    numbers(&case["lambdas"])[column],
                    1e-7,
                    &format!("{what} lambda"),
                );
                close(
                    mean,
                    numbers(&case["mean"])[column],
                    1e-7,
                    &format!("{what} mean"),
                );
                close(
                    scale,
                    numbers(&case["scale"])[column],
                    1e-7,
                    &format!("{what} scale"),
                );
                for (row, value) in values.iter().enumerate() {
                    close(
                        *value,
                        fitted[row * columns + column],
                        1e-7,
                        &format!("{what} fit_transform"),
                    );
                }
                for (row, value) in query.iter().skip(column).step_by(columns).enumerate() {
                    let actual = (yeo_johnson(*value, lambda) - mean) / scale;
                    close(
                        actual,
                        transformed[row * columns + column],
                        1e-7,
                        &format!("{what} transform"),
                    );
                }
            }
        }
    }
}

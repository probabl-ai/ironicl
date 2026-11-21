//! Borrowed, validated preprocessed tensors for one forward pass.

use crate::error::{Error, Result};

const MAX_ENSEMBLE_COUNT: usize = 4096;
const MAX_TOTAL_ROWS: usize = 1_000_000;
const MAX_FEATURE_COUNT: usize = 65_536;
const MAX_TRAINING_ROWS: usize = 1_000_000;

/// Borrowed classifier tensors `X` with shape `[B, T, H]` and `y_train` with
/// shape `[B, N]`, for the uncached [`crate::Executor::predict_logits`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Request<'a> {
    ensemble_count: usize,
    total_rows: usize,
    feature_count: usize,
    training_rows: usize,
    x: &'a [f32],
    y_train: &'a [f32],
}

/// Borrowed preprocessed test queries with shape `[B, Q, H]`.
///
/// This request is for a [`crate::StaticExecutor`]. The static
/// context supplies the training rows and labels, so query execution accepts
/// neither of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryRequest<'a> {
    ensemble_count: usize,
    query_rows: usize,
    feature_count: usize,
    x: &'a [f32],
}

impl<'a> QueryRequest<'a> {
    /// Validate borrowed FP32 query values with shape `[B, Q, H]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when dimensions, length, or values
    /// are invalid.
    pub fn new(ensemble_count: usize, query_rows: usize, feature_count: usize, x: &'a [f32]) -> Result<Self> {
        if ensemble_count == 0
            || ensemble_count > MAX_ENSEMBLE_COUNT
            || query_rows == 0
            || query_rows > MAX_TOTAL_ROWS
            || feature_count == 0
            || feature_count > MAX_FEATURE_COUNT
        {
            return Err(Error::InvalidArgument(format!(
                "invalid query dimensions B={ensemble_count}, Q={query_rows}, H={feature_count}"
            )));
        }
        if x.len()
            != checked_product(
                &[ensemble_count, query_rows, feature_count],
                "query X element count",
            )?
        {
            return Err(Error::InvalidArgument(
                "query X length does not match dimensions".into(),
            ));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(
                "query X contains a non-finite value".into(),
            ));
        }
        Ok(Self {
            ensemble_count,
            query_rows,
            feature_count,
            x,
        })
    }

    /// Ensemble count `B`.
    #[must_use]
    pub fn ensemble_count(self) -> usize {
        self.ensemble_count
    }

    /// Query row count `Q`.
    #[must_use]
    pub fn query_rows(self) -> usize {
        self.query_rows
    }

    /// Feature count `H`.
    #[must_use]
    pub fn feature_count(self) -> usize {
        self.feature_count
    }

    /// FP32 query values in `[B, Q, H]` row-major order.
    #[must_use]
    pub fn x(self) -> &'a [f32] {
        self.x
    }
}

impl<'a> Request<'a> {
    /// Validate borrowed classifier tensors with shapes `[B, T, H]` and `[B, N]`.
    ///
    /// Labels must be finite integer-valued FP32 values from 0 through 9.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] for invalid dimensions, lengths,
    /// tensor values, or labels.
    #[expect(
        clippy::many_single_char_names,
        reason = "b, t, h, n name the model's dimensions"
    )]
    pub fn new(
        ensemble_count: usize,
        total_rows: usize,
        feature_count: usize,
        training_rows: usize,
        x: &'a [f32],
        y_train: &'a [f32],
    ) -> Result<Self> {
        let (b, t, h, n) = (ensemble_count, total_rows, feature_count, training_rows);
        if b == 0
            || b > MAX_ENSEMBLE_COUNT
            || t == 0
            || t > MAX_TOTAL_ROWS
            || h == 0
            || h > MAX_FEATURE_COUNT
            || n == 0
            || n > MAX_TRAINING_ROWS
            || t <= n
        {
            return Err(Error::InvalidArgument(format!(
                "invalid dimensions B={b}, T={t}, H={h}, N={n}; require B, H, N > 0, T > N, and resource limits"
            )));
        }
        if x.len() != checked_product(&[b, t, h], "X element count")?
            || y_train.len() != checked_product(&[b, n], "y_train element count")?
        {
            return Err(Error::InvalidArgument(
                "tensor lengths do not match dimensions".into(),
            ));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument("X contains a non-finite value".into()));
        }
        if y_train
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 9.0 || value.fract() != 0.0)
        {
            return Err(Error::InvalidArgument(
                "y_train labels must be finite integer-valued FP32 values in 0..=9".into(),
            ));
        }
        Ok(Self {
            ensemble_count,
            total_rows,
            feature_count,
            training_rows,
            x,
            y_train,
        })
    }

    /// Ensemble count `B`.
    #[must_use]
    pub fn ensemble_count(self) -> usize {
        self.ensemble_count
    }
    /// Total row count `T`.
    #[must_use]
    pub fn total_rows(self) -> usize {
        self.total_rows
    }
    /// Feature count `H`.
    #[must_use]
    pub fn feature_count(self) -> usize {
        self.feature_count
    }
    /// Training row count `N`.
    #[must_use]
    pub fn training_rows(self) -> usize {
        self.training_rows
    }
    /// FP32 feature values in `[B, T, H]` row-major order.
    #[must_use]
    pub fn x(self) -> &'a [f32] {
        self.x
    }
    /// FP32 class labels in `[B, N]` row-major order.
    #[must_use]
    pub fn y_train(self) -> &'a [f32] {
        self.y_train
    }
}

fn checked_product(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| Error::InvalidArgument(format!("{name} overflows usize")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_rejects_invalid_shapes_and_noninteger_labels() {
        assert!(Request::new(2, 3, 2, 2, &[0.0; 12], &[0.0, 9.0, 1.0, 8.0]).is_ok());
        assert!(Request::new(1, 2, 1, 2, &[0.0; 2], &[0.0; 2]).is_err());
        assert!(Request::new(1, 2, 1, 1, &[0.0; 2], &[1.5]).is_err());
        assert!(Request::new(1, 2, 1, 1, &[0.0; 3], &[1.0]).is_err());
    }

    #[test]
    fn query_request_rejects_shape_and_nonfinite_values() {
        assert!(QueryRequest::new(1, 2, 3, &[0.0; 6]).is_ok());
        assert!(QueryRequest::new(1, 2, 3, &[0.0; 5]).is_err());
        assert!(QueryRequest::new(1, 2, 3, &[0.0, 0.0, 0.0, 0.0, 0.0, f32::NAN]).is_err());
    }
}

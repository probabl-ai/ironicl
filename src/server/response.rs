//! Response bodies: a classifier's labels and probabilities, a regressor's
//! statistics.

use serde::Serialize;

use super::error::ApiError;
use crate::{Labels, RegressionOutput};

#[derive(Serialize)]
struct ClassifierResponse<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<Labels>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    probabilities: Option<ClassProbabilities<'a>>,
}

#[derive(Serialize)]
struct ClassProbabilities<'a> {
    /// The columns of `probabilities`.
    classes: &'a Labels,
    probabilities: Vec<&'a [f32]>,
}

/// A classifier's response body for one request's rows of probabilities.
pub(super) fn classifier_response(
    labels: bool,
    probabilities: bool,
    classes: &Labels,
    rows: Vec<&[f32]>,
) -> Result<Vec<u8>, ApiError> {
    if rows.iter().any(|row| row.iter().any(|value| !value.is_finite())) {
        return Err(ApiError::internal(
            "prediction produced non-finite class probabilities",
        ));
    }
    let labels = labels.then(|| {
        // The first most probable class, as NumPy's argmax picks it.
        let best: Vec<usize> = rows
            .iter()
            .map(|row| (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }))
            .collect();
        classes.select(&best)
    });
    let probabilities = probabilities.then_some(ClassProbabilities {
        classes,
        probabilities: rows,
    });
    serde_json::to_vec(&ClassifierResponse {
        labels,
        probabilities,
    })
    .map_err(ApiError::internal)
}

#[derive(Serialize, Default)]
struct RegressorResponse<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    mean: Option<&'a [f32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    median: Option<&'a [f32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    variance: Option<&'a [f32]>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    quantiles: Option<Quantiles<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_quantiles: Option<Vec<&'a [f32]>>,
}

#[derive(Serialize)]
struct Quantiles<'a> {
    /// The columns of `quantiles`.
    alphas: &'a [f32],
    quantiles: Vec<&'a [f32]>,
}

/// A regressor's response body. `parts` are each of `outputs` for one
/// request's rows, with their width.
pub(super) fn regressor_response(
    outputs: &[RegressionOutput],
    alphas: &[f32],
    parts: Vec<(&[f32], usize)>,
) -> Result<Vec<u8>, ApiError> {
    if parts
        .iter()
        .any(|(values, _)| values.iter().any(|value| !value.is_finite()))
    {
        return Err(ApiError::internal(
            "prediction produced non-finite regression values",
        ));
    }
    let mut response = RegressorResponse::default();
    for (output, (values, width)) in outputs.iter().zip(parts) {
        let matrix = || values.chunks(width.max(1)).collect();
        match output {
            RegressionOutput::Mean => response.mean = Some(values),
            RegressionOutput::Median => response.median = Some(values),
            RegressionOutput::Variance => response.variance = Some(values),
            RegressionOutput::Quantiles(_) => {
                response.quantiles = Some(Quantiles {
                    alphas,
                    quantiles: matrix(),
                });
            }
            RegressionOutput::RawQuantiles => response.raw_quantiles = Some(matrix()),
        }
    }
    serde_json::to_vec(&response).map_err(ApiError::internal)
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::{classifier_response, regressor_response};
    use crate::{Labels, RegressionOutput};

    #[test]
    fn non_finite_predictions_return_internal_error_instead_of_null() {
        let classes = Labels::Int(vec![0, 1]);
        let error = classifier_response(true, true, &classes, vec![&[0.5, f32::NAN]]).unwrap_err();
        assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let error =
            regressor_response(&[RegressionOutput::Mean], &[], vec![(&[f32::INFINITY], 1)]).unwrap_err();
        assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}

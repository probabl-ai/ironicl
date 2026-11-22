//! The HTTP API: the request, its validation, the responses, and the errors.

use std::{fmt, sync::Arc};

use axum::{
    body::Bytes,
    extract::{State, rejection::BytesRejection},
    http::{HeaderName, StatusCode, header},
    response::{IntoResponse, Response},
};
use garde::Validate;
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{DeserializeSeed, Error as _, SeqAccess, Visitor},
};

use super::{
    Server,
    worker::{Answer, Outputs, Query},
};
use crate::{Error, Labels, RegressionOutput, TaskKind};

/// `TabICLRegressor.predict`'s default quantile levels.
const DEFAULT_ALPHAS: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// The most quantile levels a request may ask for: as many as the model's raw
/// quantiles. Each level costs a value per row of the whole batch.
const MAX_ALPHAS: usize = 999;
const CLASSIFIER_OUTPUTS: [&str; 2] = ["labels", "probabilities"];
const REGRESSOR_OUTPUTS: [&str; 5] = ["mean", "median", "variance", "quantiles", "raw_quantiles"];

/// What a request is checked against.
pub(super) struct Limits {
    pub task: TaskKind,
    pub features: usize,
    pub max_request_rows: usize,
}

/// `POST /v1/predict`'s body. Options are named after `TabICL`'s predict
/// arguments.
#[derive(Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[garde(context(Limits))]
struct PredictRequest {
    #[garde(custom(fit_the_model))]
    rows: Rows,
    #[garde(length(min = 1), custom(known_and_unique))]
    output_type: Option<Vec<String>>,
    #[garde(custom(for_a_regressor), length(min = 1, max = MAX_ALPHAS), inner(inner(custom(between_0_and_1))))]
    alphas: Option<Vec<f64>>,
}

fn fit_the_model(rows: &Rows, limits: &Limits) -> garde::Result {
    if rows.features != limits.features {
        return Err(garde::Error::new(format!(
            "rows have {} values; the model expects {}",
            rows.features, limits.features
        )));
    }
    if rows.count > limits.max_request_rows {
        return Err(garde::Error::new(format!(
            "{} rows; the limit is {}",
            rows.count, limits.max_request_rows
        )));
    }
    Ok(())
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn known_and_unique(names: &Option<Vec<String>>, limits: &Limits) -> garde::Result {
    let Some(names) = names else { return Ok(()) };
    let known = match limits.task {
        TaskKind::Classifier => &CLASSIFIER_OUTPUTS[..],
        TaskKind::Regressor => &REGRESSOR_OUTPUTS[..],
    };
    if let Some(name) = names.iter().find(|name| !known.contains(&name.as_str())) {
        return Err(garde::Error::new(format!(
            "unknown output {name:?}; a {} has {}",
            limits.task,
            known.join(", ")
        )));
    }
    if (1..names.len()).any(|i| names[..i].contains(&names[i])) {
        return Err(garde::Error::new("names an output twice"));
    }
    Ok(())
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn for_a_regressor(alphas: &Option<Vec<f64>>, limits: &Limits) -> garde::Result {
    match (alphas, limits.task) {
        (Some(_), TaskKind::Classifier) => Err(garde::Error::new("only a regressor takes alphas")),
        _ => Ok(()),
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "garde passes each value by reference"
)]
fn between_0_and_1(alpha: &f64, _: &Limits) -> garde::Result {
    if *alpha > 0.0 && *alpha < 1.0 {
        Ok(())
    } else {
        Err(garde::Error::new(format!("must be between 0 and 1, got {alpha}")))
    }
}

impl PredictRequest {
    /// The checked request as the worker's query for a `task` model.
    fn into_query(self, task: TaskKind) -> Result<Query, ApiError> {
        let names: Vec<&str> = match (&self.output_type, task) {
            (Some(names), _) => names.iter().map(String::as_str).collect(),
            (None, TaskKind::Classifier) => CLASSIFIER_OUTPUTS.to_vec(),
            (None, TaskKind::Regressor) => vec!["mean"],
        };
        let outputs = match task {
            TaskKind::Classifier => Outputs::Classifier {
                labels: names.contains(&"labels"),
                probabilities: names.contains(&"probabilities"),
            },
            TaskKind::Regressor => {
                let alphas = self.alphas.unwrap_or_else(|| DEFAULT_ALPHAS.to_vec());
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "The model computes quantiles in FP32"
                )]
                let levels: Vec<f32> = alphas.iter().map(|&alpha| alpha as f32).collect();
                let outputs = names
                    .iter()
                    .map(|name| RegressionOutput::parse(name, &levels))
                    .collect::<crate::Result<_>>()?;
                Outputs::Regressor { outputs, alphas }
            }
        };
        Ok(Query {
            rows: self.rows,
            outputs,
        })
    }
}

/// `count` rows of zeros with the default outputs, to warm a `task` model up
/// with.
pub(super) fn zeros(task: TaskKind, count: usize, features: usize) -> Result<Query, ApiError> {
    let rows = Rows {
        values: vec![0.0; count * features],
        count,
        features,
    };
    let request = PredictRequest {
        rows,
        output_type: None,
        alphas: None,
    };
    request.into_query(task)
}

/// At least one row, all of the same length, flattened in row order.
pub(super) struct Rows {
    pub values: Vec<f64>,
    pub count: usize,
    pub features: usize,
}

/// Rows parse straight into one flat buffer, without an allocation per row,
/// and a ragged row fails as soon as it ends.
impl<'de> Deserialize<'de> for Rows {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(RowsVisitor)
    }
}

struct RowsVisitor;

impl<'de> Visitor<'de> for RowsVisitor {
    type Value = Rows;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an array of rows of numbers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut rows: A) -> Result<Rows, A::Error> {
        let mut values = Vec::new();
        let (mut count, mut features) = (0, None);
        while let Some(length) = rows.next_element_seed(Row(&mut values))? {
            let features = *features.get_or_insert(length);
            if length != features {
                return Err(A::Error::custom(format!(
                    "row {count} has {length} values; row 0 has {features}"
                )));
            }
            count += 1;
        }
        let features = features.ok_or_else(|| A::Error::custom("rows is empty"))?;
        Ok(Rows {
            values,
            count,
            features,
        })
    }
}

/// Appends one row's values and returns how many there were.
struct Row<'a>(&'a mut Vec<f64>);

impl<'de> DeserializeSeed<'de> for Row<'_> {
    type Value = usize;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<usize, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for Row<'_> {
    type Value = usize;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a row of numbers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut row: A) -> Result<usize, A::Error> {
        let start = self.0.len();
        while let Some(value) = row.next_element()? {
            self.0.push(value);
        }
        Ok(self.0.len() - start)
    }
}

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
) -> Vec<u8> {
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
    .expect("the response serializes")
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
    alphas: &'a [f64],
    quantiles: Vec<&'a [f32]>,
}

/// A regressor's response body. `parts` are each of `outputs` for one
/// request's rows, with their width.
pub(super) fn regressor_response(
    outputs: &[RegressionOutput],
    alphas: &[f64],
    parts: Vec<(&[f32], usize)>,
) -> Vec<u8> {
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
    serde_json::to_vec(&response).expect("the response serializes")
}

/// An error response: `{"error": {"code": ..., "message": ...}}`.
#[derive(Clone, Debug)]
pub(super) struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    pub(super) fn overloaded() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded",
            "the queue is full; retry later",
        )
    }

    pub(super) fn worker_stopped() -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "the prediction worker stopped",
        )
    }

    pub(super) fn message(&self) -> &str {
        &self.message
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        match error {
            Error::InvalidArgument(message) => Self::invalid(message),
            error => Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", error.to_string()),
        }
    }
}

impl From<BytesRejection> for ApiError {
    fn from(rejection: BytesRejection) -> Self {
        match rejection.status() {
            StatusCode::PAYLOAD_TOO_LARGE => Self::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "body_too_large",
                rejection.body_text(),
            ),
            _ => Self::invalid(rejection.body_text()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = axum::Json(serde_json::json!({"error": {"code": self.code, "message": self.message}}));
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            return (self.status, [(header::RETRY_AFTER, "1")], body).into_response();
        }
        (self.status, body).into_response()
    }
}

impl IntoResponse for Answer {
    /// The body, with the request's queue and prediction times in milliseconds
    /// as `Server-Timing`.
    fn into_response(self) -> Response {
        let timing = format!(
            "queue;dur={:.3}, predict;dur={:.3}",
            self.queue.as_secs_f64() * 1e3,
            self.predict.as_secs_f64() * 1e3
        );
        let headers = [
            (header::CONTENT_TYPE, "application/json".to_owned()),
            (HeaderName::from_static("server-timing"), timing),
        ];
        (headers, self.body).into_response()
    }
}

/// `POST /v1/predict`: check the request, then queue it for the next batch.
pub(super) async fn predict(
    State(server): State<Arc<Server>>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Answer, ApiError> {
    let body = body?;
    // A full queue refuses the request before the work of parsing it.
    if !server.queue.has_room() {
        return Err(ApiError::overloaded());
    }
    let request: PredictRequest =
        serde_json::from_slice(&body).map_err(|error| ApiError::invalid(error.to_string()))?;
    request
        .validate_with(&server.limits)
        .map_err(|report| ApiError::invalid(report.to_string().trim_end()))?;
    server.submit(request.into_query(server.limits.task)?).await
}

/// `GET /v1/health`: the model is loaded and warmed up.
pub(super) async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({"status": "ok"}))
}

#[cfg(test)]
mod tests {
    use garde::Validate;

    use super::{Limits, MAX_ALPHAS, PredictRequest};
    use crate::TaskKind;

    /// The error that parsing and checking `json` for a `task` model with two
    /// features gives, or `""`.
    fn error(task: TaskKind, json: &str) -> String {
        let limits = Limits {
            task,
            features: 2,
            max_request_rows: 3,
        };
        match serde_json::from_str::<PredictRequest>(json) {
            Err(error) => error.to_string(),
            Ok(request) => request
                .validate_with(&limits)
                .err()
                .map(|report| report.to_string())
                .unwrap_or_default(),
        }
    }

    #[test]
    fn requests_are_checked_against_the_model() {
        use TaskKind::{Classifier, Regressor};
        let request: PredictRequest = serde_json::from_str(r#"{"rows": [[1, 2], [3, 4]]}"#).unwrap();
        assert_eq!(
            (request.rows.values, request.rows.count, request.rows.features),
            (vec![1.0, 2.0, 3.0, 4.0], 2, 2)
        );
        let alphas = vec!["0.5"; MAX_ALPHAS + 1].join(", ");
        let too_many_alphas = format!(r#"{{"rows": [[1, 2]], "alphas": [{alphas}]}}"#);
        for (task, json, message) in [
            (Classifier, r#"{"rows": [[1, 2]], "output_type": ["labels"]}"#, ""),
            (
                Regressor,
                r#"{"rows": [[1, 2]], "output_type": ["quantiles"], "alphas": [0.5]}"#,
                "",
            ),
            (Classifier, r#"{"rows": []}"#, "rows is empty"),
            (
                Classifier,
                r#"{"rows": [[1, 2], [3]]}"#,
                "row 1 has 1 values; row 0 has 2",
            ),
            (
                Classifier,
                r#"{"rows": [[1, 2], [3, 4, 5], "never read"]}"#,
                "row 1 has 3 values",
            ),
            (Classifier, r#"{"rows": [[1, null]]}"#, "invalid type: null"),
            (Classifier, r#"{"rows": [1]}"#, "expected a row of numbers"),
            (
                Classifier,
                r#"{"rows": [[1]]}"#,
                "rows: rows have 1 values; the model expects 2",
            ),
            (
                Classifier,
                r#"{"rows": [[1, 2], [1, 2], [1, 2], [1, 2]]}"#,
                "rows: 4 rows; the limit is 3",
            ),
            (
                Classifier,
                r#"{"rows": [[1, 2]], "output_type": ["mean"]}"#,
                "unknown output \"mean\"",
            ),
            (
                Classifier,
                r#"{"rows": [[1, 2]], "output_type": []}"#,
                "output_type: length is lower than 1",
            ),
            (
                Classifier,
                r#"{"rows": [[1, 2]], "output_type": ["labels", "labels"]}"#,
                "names an output twice",
            ),
            (
                Classifier,
                r#"{"rows": [[1, 2]], "alphas": [0.5]}"#,
                "only a regressor takes alphas",
            ),
            (
                Classifier,
                r#"{"rows": [[1, 2]], "extra": 1}"#,
                "unknown field `extra`",
            ),
            (
                Regressor,
                r#"{"rows": [[1, 2]], "alphas": [0.5, 1]}"#,
                "between 0 and 1, got 1",
            ),
            (Regressor, &too_many_alphas, "alphas: length is greater than 999"),
        ] {
            let got = error(task, json);
            if message.is_empty() {
                assert_eq!(got, "", "{json}");
            } else {
                assert!(got.contains(message), "{json}: {got}");
            }
        }
    }
}

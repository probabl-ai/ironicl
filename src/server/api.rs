//! The HTTP API: the request, its validation, the responses, and the errors.

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{State, rejection::BytesRejection},
    http::{HeaderName, StatusCode, header},
    response::{IntoResponse, Response},
};
use garde::Validate;
use serde::{Deserialize, Serialize};

use super::{
    Server,
    worker::{Answer, Outputs, Query},
};
use crate::{Error, Labels, RegressionOutput, TaskKind};

/// `TabICLRegressor.predict`'s default quantile levels.
const DEFAULT_ALPHAS: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// The most quantile levels a request may ask for: as many as the model's raw
/// quantiles. Each level costs a value per row of the whole batch.
const MAX_ALPHAS: usize = 999;

/// What a request is checked against.
pub(super) struct Limits {
    pub task: TaskKind,
    pub features: usize,
    /// The training data's column names, when it had them.
    pub feature_names: Option<Vec<String>>,
    pub max_request_rows: usize,
}

/// `POST /v1/predict`'s body: a table as pandas'
/// `to_json(orient="split", index=False)` writes it, and options named after
/// `TabICL`'s predict arguments.
#[derive(Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[garde(context(Limits))]
struct PredictRequest {
    /// The column names, in the order of each row's values. Without them,
    /// the values are in the training data's order.
    #[garde(custom(match_the_model))]
    columns: Option<Vec<String>>,
    /// The rows. `null`, pandas' missing value, parses but is refused.
    #[garde(custom(fit_the_model))]
    data: Vec<Vec<Option<f64>>>,
    #[garde(length(min = 1), custom(for_the_task_once))]
    output_type: Option<Vec<Output>>,
    #[garde(custom(for_a_regressor), length(min = 1, max = MAX_ALPHAS), inner(inner(custom(between_0_and_1))))]
    alphas: Option<Vec<f32>>,
}

/// An `output_type` name.
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Output {
    Labels,
    Probabilities,
    Mean,
    Median,
    Variance,
    Quantiles,
    RawQuantiles,
}

impl Output {
    fn task(self) -> TaskKind {
        match self {
            Self::Labels | Self::Probabilities => TaskKind::Classifier,
            _ => TaskKind::Regressor,
        }
    }
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn match_the_model(columns: &Option<Vec<String>>, limits: &Limits) -> garde::Result {
    let Some(columns) = columns else { return Ok(()) };
    let Some(names) = &limits.feature_names else {
        return Err(garde::Error::new(
            "the model was fit without column names; leave columns out",
        ));
    };
    if let Some(column) = columns.iter().find(|column| !names.contains(column)) {
        return Err(garde::Error::new(format!("unknown column {column:?}")));
    }
    if let Some(name) = names.iter().find(|name| !columns.contains(name)) {
        return Err(garde::Error::new(format!("missing column {name:?}")));
    }
    if columns.len() != names.len() {
        return Err(garde::Error::new("names a column twice"));
    }
    Ok(())
}

fn fit_the_model(data: &[Vec<Option<f64>>], limits: &Limits) -> garde::Result {
    if data.is_empty() {
        return Err(garde::Error::new("has no rows"));
    }
    if data.len() > limits.max_request_rows {
        return Err(garde::Error::new(format!(
            "{} rows; the limit is {}",
            data.len(),
            limits.max_request_rows
        )));
    }
    for (i, row) in data.iter().enumerate() {
        if row.len() != limits.features {
            return Err(garde::Error::new(format!(
                "row {i} has {} values; the model expects {}",
                row.len(),
                limits.features
            )));
        }
        if row.contains(&None) {
            return Err(garde::Error::new(format!(
                "row {i} has a missing value; missing values are not supported yet"
            )));
        }
    }
    Ok(())
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn for_the_task_once(outputs: &Option<Vec<Output>>, limits: &Limits) -> garde::Result {
    let Some(outputs) = outputs else { return Ok(()) };
    if let Some(output) = outputs.iter().find(|output| output.task() != limits.task) {
        return Err(garde::Error::new(format!(
            "a {} has no output {}",
            limits.task,
            serde_json::json!(output)
        )));
    }
    if (1..outputs.len()).any(|i| outputs[..i].contains(&outputs[i])) {
        return Err(garde::Error::new("names an output twice"));
    }
    Ok(())
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn for_a_regressor(alphas: &Option<Vec<f32>>, limits: &Limits) -> garde::Result {
    match (alphas, limits.task) {
        (Some(_), TaskKind::Classifier) => Err(garde::Error::new("only a regressor takes alphas")),
        _ => Ok(()),
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "garde passes each value by reference"
)]
fn between_0_and_1(alpha: &f32, _: &Limits) -> garde::Result {
    if *alpha > 0.0 && *alpha < 1.0 {
        Ok(())
    } else {
        Err(garde::Error::new(format!("must be between 0 and 1, got {alpha}")))
    }
}

impl PredictRequest {
    /// The checked request as the worker's query.
    fn into_query(self, limits: &Limits) -> Query {
        // Where each of the model's features is in a row.
        let order: Vec<usize> = match (&self.columns, &limits.feature_names) {
            (Some(columns), Some(names)) => names
                .iter()
                .map(|name| {
                    columns
                        .iter()
                        .position(|column| column == name)
                        .expect("the columns were checked against the model")
                })
                .collect(),
            _ => (0..limits.features).collect(),
        };
        let rows = Rows {
            values: self
                .data
                .iter()
                .flat_map(|row| order.iter().map(|&i| row[i].unwrap_or(f64::NAN)))
                .collect(),
            count: self.data.len(),
        };
        let outputs = match limits.task {
            TaskKind::Classifier => {
                let names = self
                    .output_type
                    .unwrap_or_else(|| vec![Output::Labels, Output::Probabilities]);
                Outputs::Classifier {
                    labels: names.contains(&Output::Labels),
                    probabilities: names.contains(&Output::Probabilities),
                }
            }
            TaskKind::Regressor => {
                let alphas = self.alphas.unwrap_or_else(|| DEFAULT_ALPHAS.to_vec());
                let outputs = self
                    .output_type
                    .unwrap_or_else(|| vec![Output::Mean])
                    .into_iter()
                    .map(|output| match output {
                        Output::Median => RegressionOutput::Median,
                        Output::Variance => RegressionOutput::Variance,
                        Output::Quantiles => RegressionOutput::Quantiles(alphas.clone()),
                        Output::RawQuantiles => RegressionOutput::RawQuantiles,
                        Output::Mean | Output::Labels | Output::Probabilities => RegressionOutput::Mean,
                    })
                    .collect();
                Outputs::Regressor { outputs, alphas }
            }
        };
        Query { rows, outputs }
    }
}

/// `count` rows of zeros with the default outputs, to warm a model up with.
pub(super) fn zeros(limits: &Limits, count: usize) -> Query {
    let request = PredictRequest {
        columns: None,
        data: vec![vec![Some(0.0); limits.features]; count],
        output_type: None,
        alphas: None,
    };
    request.into_query(limits)
}

/// A query's rows, flattened in the training data's column order.
pub(super) struct Rows {
    pub values: Vec<f64>,
    pub count: usize,
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
    alphas: &'a [f32],
    quantiles: Vec<&'a [f32]>,
}

/// A regressor's response body. `parts` are each of `outputs` for one
/// request's rows, with their width.
pub(super) fn regressor_response(
    outputs: &[RegressionOutput],
    alphas: &[f32],
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
    server.submit(request.into_query(&server.limits)).await
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

    /// Limits for a `task` model with the columns `a` and `b`.
    fn limits(task: TaskKind) -> Limits {
        Limits {
            task,
            features: 2,
            feature_names: Some(vec!["a".into(), "b".into()]),
            max_request_rows: 3,
        }
    }

    /// The error that parsing and checking `json` against `limits` gives, or
    /// `""`.
    fn error(limits: &Limits, json: &str) -> String {
        match serde_json::from_str::<PredictRequest>(json) {
            Err(error) => error.to_string(),
            Ok(request) => request
                .validate_with(limits)
                .err()
                .map(|report| report.to_string())
                .unwrap_or_default(),
        }
    }

    #[test]
    fn columns_put_the_values_in_the_training_order() {
        let limits = limits(TaskKind::Classifier);
        for (json, values) in [
            (r#"{"data": [[1, 2], [3, 4]]}"#, [1.0, 2.0, 3.0, 4.0]),
            (
                r#"{"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}"#,
                [1.0, 2.0, 3.0, 4.0],
            ),
            (
                r#"{"columns": ["b", "a"], "data": [[1, 2], [3, 4]]}"#,
                [2.0, 1.0, 4.0, 3.0],
            ),
        ] {
            let request: PredictRequest = serde_json::from_str(json).unwrap();
            request.validate_with(&limits).unwrap();
            let rows = request.into_query(&limits).rows;
            assert_eq!((rows.values, rows.count), (values.to_vec(), 2), "{json}");
        }
    }

    #[test]
    fn requests_are_checked_against_the_model() {
        use TaskKind::{Classifier, Regressor};
        let alphas = vec!["0.5"; MAX_ALPHAS + 1].join(", ");
        let too_many_alphas = format!(r#"{{"data": [[1, 2]], "alphas": [{alphas}]}}"#);
        for (task, json, message) in [
            (Classifier, r#"{"data": [[1, 2]], "output_type": ["labels"]}"#, ""),
            (
                Regressor,
                r#"{"data": [[1, 2]], "output_type": ["quantiles"], "alphas": [0.5]}"#,
                "",
            ),
            (Classifier, r#"{"data": []}"#, "data: has no rows"),
            (
                Classifier,
                r#"{"data": [[1, 2], [3]]}"#,
                "data: row 1 has 1 values; the model expects 2",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2], [1, 2], [1, 2], [1, 2]]}"#,
                "data: 4 rows; the limit is 3",
            ),
            (
                Classifier,
                r#"{"data": [[1, null]]}"#,
                "row 0 has a missing value; missing values are not supported yet",
            ),
            (Classifier, r#"{"data": [[1, "x"]]}"#, "invalid type: string"),
            (Classifier, r#"{"data": [1]}"#, "invalid type: integer"),
            (Classifier, r#"{"rows": [[1, 2]]}"#, "unknown field `rows`"),
            (
                Classifier,
                r#"{"columns": ["a", "c"], "data": [[1, 2]]}"#,
                "columns: unknown column \"c\"",
            ),
            (
                Classifier,
                r#"{"columns": ["a"], "data": [[1, 2]]}"#,
                "columns: missing column \"b\"",
            ),
            (
                Classifier,
                r#"{"columns": ["a", "b", "a"], "data": [[1, 2]]}"#,
                "columns: names a column twice",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": ["mean"]}"#,
                "a classifier has no output \"mean\"",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": ["average"]}"#,
                "unknown variant `average`",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": []}"#,
                "output_type: length is lower than 1",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": ["labels", "labels"]}"#,
                "names an output twice",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "alphas": [0.5]}"#,
                "only a regressor takes alphas",
            ),
            (
                Regressor,
                r#"{"data": [[1, 2]], "alphas": [0.5, 1]}"#,
                "between 0 and 1, got 1",
            ),
            (Regressor, &too_many_alphas, "alphas: length is greater than 999"),
        ] {
            let got = error(&limits(task), json);
            if message.is_empty() {
                assert_eq!(got, "", "{json}");
            } else {
                assert!(got.contains(message), "{json}: {got}");
            }
        }
        let unnamed = Limits {
            feature_names: None,
            ..limits(Classifier)
        };
        assert!(
            error(&unnamed, r#"{"columns": ["a", "b"], "data": [[1, 2]]}"#)
                .contains("fit without column names")
        );
    }
}

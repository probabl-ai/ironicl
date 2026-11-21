//! `ironicl serve`: an HTTP server for one exported model.
//!
//! `POST /predict` takes `{"rows": [[...], ...]}` and options named after
//! `TabICL`'s predict arguments, and answers with one key per output.
//! `GET /health` answers once the model is loaded. Predictions run one at a
//! time. See `docs/server.md`.

use std::sync::{Arc, Mutex, PoisonError};

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use crate::{
    ClassifierAggregation, Error, Features, FittedModel, Labels, RegressionOutput, Result, ServeArgs,
    TaskKind, memory::MemoryLimits, models::tabicl_v2::saved,
};

/// `TabICLRegressor.predict`'s default quantile levels.
const DEFAULT_ALPHAS: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassifierRequest {
    rows: Rows,
    output_type: Option<Outputs<ClassifierOutput>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegressorRequest {
    rows: Rows,
    output_type: Option<Outputs<RegressorOutput>>,
    alphas: Option<Alphas>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ClassifierOutput {
    Labels,
    Probabilities,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegressorOutput {
    Mean,
    Median,
    Variance,
    Quantiles,
    RawQuantiles,
}

/// At least one row, all of the same length, flattened in row order.
#[derive(Deserialize)]
#[serde(try_from = "Vec<Vec<f64>>")]
struct Rows {
    values: Vec<f64>,
    count: usize,
    features: usize,
}

impl TryFrom<Vec<Vec<f64>>> for Rows {
    type Error = String;

    fn try_from(rows: Vec<Vec<f64>>) -> std::result::Result<Self, String> {
        let features = rows.first().ok_or("rows is empty")?.len();
        if let Some(index) = rows.iter().position(|row| row.len() != features) {
            return Err(format!(
                "row {index} has {} values; row 0 has {features}",
                rows[index].len()
            ));
        }
        Ok(Self {
            count: rows.len(),
            values: rows.concat(),
            features,
        })
    }
}

/// Requested outputs: at least one, none twice.
#[derive(Deserialize)]
#[serde(try_from = "Vec<T>", bound(deserialize = "T: Deserialize<'de> + PartialEq"))]
struct Outputs<T>(Vec<T>);

impl<T: PartialEq> TryFrom<Vec<T>> for Outputs<T> {
    type Error = &'static str;

    fn try_from(outputs: Vec<T>) -> std::result::Result<Self, &'static str> {
        if outputs.is_empty() {
            return Err("output_type is empty");
        }
        if (1..outputs.len()).any(|i| outputs[..i].contains(&outputs[i])) {
            return Err("output_type names an output twice");
        }
        Ok(Self(outputs))
    }
}

/// Quantile levels, each strictly between 0 and 1.
#[derive(Deserialize)]
#[serde(try_from = "Vec<f64>")]
struct Alphas(Vec<f64>);

impl TryFrom<Vec<f64>> for Alphas {
    type Error = String;

    fn try_from(alphas: Vec<f64>) -> std::result::Result<Self, String> {
        if alphas.is_empty() {
            return Err("alphas is empty".into());
        }
        if let Some(alpha) = alphas.iter().find(|&&alpha| !(alpha > 0.0 && alpha < 1.0)) {
            return Err(format!("alphas must be between 0 and 1, got {alpha}"));
        }
        Ok(Self(alphas))
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

/// An error response: `{"error": {"code": ..., "message": ...}}`.
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn invalid(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        match error {
            Error::InvalidArgument(message) => Self::invalid("invalid_request", message),
            error => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "internal",
                message: error.to_string(),
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({"error": {"code": self.code, "message": self.message}});
        (self.status, axum::Json(body)).into_response()
    }
}

/// The loaded model and what requests are checked against.
struct Server {
    /// Locked for each prediction, so they run one at a time.
    model: Mutex<FittedModel>,
    task: TaskKind,
    features: usize,
    classes: Option<Labels>,
    aggregation: ClassifierAggregation,
    max_rows: usize,
}

impl Server {
    fn respond(&self, body: &[u8]) -> std::result::Result<Vec<u8>, ApiError> {
        let parse_error = |error: serde_json::Error| ApiError::invalid("invalid_request", error.to_string());
        let json = match self.task {
            TaskKind::Classifier => {
                let request: ClassifierRequest = serde_json::from_slice(body).map_err(parse_error)?;
                self.classify(&request)?
            }
            TaskKind::Regressor => {
                let request: RegressorRequest = serde_json::from_slice(body).map_err(parse_error)?;
                self.regress(&request)?
            }
        };
        Ok(json)
    }

    fn check(&self, rows: &Rows) -> std::result::Result<(), ApiError> {
        if rows.features != self.features {
            return Err(ApiError::invalid(
                "wrong_feature_count",
                format!(
                    "rows have {} values; the model expects {}",
                    rows.features, self.features
                ),
            ));
        }
        if rows.count > self.max_rows {
            return Err(ApiError::invalid(
                "too_many_rows",
                format!("{} rows; the limit is {}", rows.count, self.max_rows),
            ));
        }
        Ok(())
    }

    fn model(&self) -> std::sync::MutexGuard<'_, FittedModel> {
        self.model.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn classify(&self, request: &ClassifierRequest) -> std::result::Result<Vec<u8>, ApiError> {
        let rows = &request.rows;
        self.check(rows)?;
        let outputs = request.output_type.as_ref().map_or(
            &[ClassifierOutput::Labels, ClassifierOutput::Probabilities][..],
            |outputs| &outputs.0,
        );
        let probabilities = self.model().predict_probabilities(
            Features::F64(&rows.values),
            rows.features,
            self.aggregation,
        )?;
        let classes = self.classes.as_ref().expect("a classifier has classes");
        let rows: Vec<&[f32]> = probabilities.values().chunks(classes.len()).collect();
        let labels = outputs.contains(&ClassifierOutput::Labels).then(|| {
            // The first most probable class, as NumPy's argmax picks it.
            let best: Vec<usize> = rows
                .iter()
                .map(|row| (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }))
                .collect();
            classes.select(&best)
        });
        let probabilities =
            outputs
                .contains(&ClassifierOutput::Probabilities)
                .then_some(ClassProbabilities {
                    classes,
                    probabilities: rows,
                });
        Ok(serde_json::to_vec(&ClassifierResponse {
            labels,
            probabilities,
        })
        .expect("the response serializes"))
    }

    fn regress(&self, request: &RegressorRequest) -> std::result::Result<Vec<u8>, ApiError> {
        let rows = &request.rows;
        self.check(rows)?;
        let outputs = request
            .output_type
            .as_ref()
            .map_or(&[RegressorOutput::Mean][..], |outputs| &outputs.0);
        let alphas = request
            .alphas
            .as_ref()
            .map_or(&DEFAULT_ALPHAS[..], |alphas| &alphas.0);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The model computes quantiles in FP32"
        )]
        let requested: Vec<RegressionOutput> = outputs
            .iter()
            .map(|output| match output {
                RegressorOutput::Mean => RegressionOutput::Mean,
                RegressorOutput::Median => RegressionOutput::Median,
                RegressorOutput::Variance => RegressionOutput::Variance,
                RegressorOutput::Quantiles => {
                    RegressionOutput::Quantiles(alphas.iter().map(|&alpha| alpha as f32).collect())
                }
                RegressorOutput::RawQuantiles => RegressionOutput::RawQuantiles,
            })
            .collect();
        let results =
            self.model()
                .predict_regression(Features::F64(&rows.values), rows.features, &requested)?;
        let mut response = RegressorResponse::default();
        for (output, (values, width)) in outputs.iter().zip(&results) {
            let matrix = || values.chunks((*width).max(1)).collect();
            match output {
                RegressorOutput::Mean => response.mean = Some(values),
                RegressorOutput::Median => response.median = Some(values),
                RegressorOutput::Variance => response.variance = Some(values),
                RegressorOutput::Quantiles => {
                    response.quantiles = Some(Quantiles {
                        alphas,
                        quantiles: matrix(),
                    });
                }
                RegressorOutput::RawQuantiles => response.raw_quantiles = Some(matrix()),
            }
        }
        Ok(serde_json::to_vec(&response).expect("the response serializes"))
    }
}

async fn predict(
    State(server): State<Arc<Server>>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            let mut error = ApiError::invalid("body_too_large", rejection.body_text());
            error.status = StatusCode::PAYLOAD_TOO_LARGE;
            return error.into_response();
        }
        Err(rejection) => return ApiError::invalid("invalid_request", rejection.body_text()).into_response(),
    };
    match tokio::task::spawn_blocking(move || server.respond(&body)).await {
        Ok(Ok(json)) => ([(header::CONTENT_TYPE, "application/json")], json).into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(error) => {
            ApiError::from(Error::InvalidConfig(format!("prediction failed: {error}"))).into_response()
        }
    }
}

async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({"status": "ok"}))
}

/// Load the exported model, predict once to warm it up, and serve it until
/// Ctrl-C.
///
/// # Errors
///
/// When the model does not load, as [`saved::load`], or the address cannot
/// be bound.
pub fn serve(args: &ServeArgs) -> Result<()> {
    let model = saved::load(&args.dir, None, MemoryLimits::resolve(None, None)?)?;
    let preprocessor = model.preprocessed().preprocessor();
    let server = Server {
        task: model.task(),
        features: preprocessor.n_features_in(),
        classes: preprocessor.classes().cloned(),
        aggregation: preprocessor.aggregation.unwrap_or_default(),
        max_rows: args.max_rows,
        model: Mutex::new(model),
    };
    let warm_up = format!("{{\"rows\": [[{}]]}}", vec!["0"; server.features].join(","));
    server
        .respond(warm_up.as_bytes())
        .map_err(|error| Error::InvalidConfig(format!("warm-up prediction failed: {}", error.message)))?;
    let device = server.model().device();
    let app = Router::new()
        .route("/predict", post(predict))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(args.max_body))
        .with_state(Arc::new(server));
    let failed = |error: std::io::Error| Error::InvalidConfig(format!("server: {error}"));
    tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .map_err(failed)?
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(args.bind).await.map_err(failed)?;
            println!(
                "Serving {} on http://{} ({device})",
                args.dir.display(),
                args.bind
            );
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
                .map_err(failed)
        })
}

#[cfg(test)]
mod tests {
    use super::{ClassifierRequest, RegressorRequest};

    fn error<T: for<'de> serde::Deserialize<'de>>(json: &str) -> String {
        serde_json::from_str::<T>(json)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn requests_are_checked_as_they_parse() {
        let request: RegressorRequest = serde_json::from_str(
            r#"{"rows": [[1, 2], [3, 4]], "output_type": ["quantiles"], "alphas": [0.5]}"#,
        )
        .unwrap();
        assert_eq!(
            (request.rows.values, request.rows.count, request.rows.features),
            (vec![1.0, 2.0, 3.0, 4.0], 2, 2)
        );
        for (json, message) in [
            (r#"{"rows": []}"#, "rows is empty"),
            (r#"{"rows": [[1, 2], [3]]}"#, "row 1 has 1 values; row 0 has 2"),
            (r#"{"rows": [[1, null]]}"#, "invalid type: null"),
            (
                r#"{"rows": [[1]], "output_type": ["mean"]}"#,
                "unknown variant `mean`",
            ),
            (
                r#"{"rows": [[1]], "output_type": ["labels", "labels"]}"#,
                "names an output twice",
            ),
            (r#"{"rows": [[1]], "alphas": [0.5]}"#, "unknown field `alphas`"),
        ] {
            assert!(
                error::<ClassifierRequest>(json).contains(message),
                "{json}: {}",
                error::<ClassifierRequest>(json)
            );
        }
        assert!(
            error::<RegressorRequest>(r#"{"rows": [[1]], "alphas": [0.5, 1]}"#)
                .contains("between 0 and 1, got 1")
        );
    }
}

//! The HTTP handlers. The request, the responses, and the errors are in
//! `request`, `response`, and `error`.

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{State, rejection::BytesRejection},
    http::{HeaderName, header},
    response::{IntoResponse, Response},
};

use super::{Server, error::ApiError, request::PredictRequest, worker::Answer};

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
/// Logs the outcome at debug level.
pub(super) async fn predict(
    State(server): State<Arc<Server>>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Answer, ApiError> {
    let result = answer(server, body).await;
    match &result {
        Ok(answer) => tracing::debug!(
            status = 200,
            rows = answer.rows,
            queue_ms = answer.queue.as_secs_f64() * 1e3,
            predict_ms = answer.predict.as_secs_f64() * 1e3,
            "predict"
        ),
        Err(error) => tracing::debug!(
            status = error.status().as_u16(),
            code = error.code(),
            message = error.message(),
            "predict"
        ),
    }
    result
}

async fn answer(server: Arc<Server>, body: Result<Bytes, BytesRejection>) -> Result<Answer, ApiError> {
    let body = body?;
    // A full queue refuses the request before the work of parsing it.
    if !server.queue.has_room() {
        return Err(ApiError::overloaded());
    }
    // Bound concurrent JSON parsing tasks so bursts cannot starve the inference worker.
    let permit = server
        .parse_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(ApiError::internal)?;
    if !server.queue.has_room() {
        return Err(ApiError::overloaded());
    }
    // Parsing a large body takes CPU time; keep it off the threads serving connections.
    let parsing = Arc::clone(&server);
    let query = tokio::task::spawn_blocking(move || PredictRequest::parse(&body, &parsing.schema))
        .await
        .map_err(ApiError::internal)??;
    drop(permit);
    server.submit(query).await
}

/// `GET /v1/health`: the model is loaded and warmed up.
pub(super) async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({"status": "ok"}))
}

//! Error responses and what they come from.

use axum::{
    extract::rejection::BytesRejection,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::Error;

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

    pub(super) fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    pub(super) fn overloaded() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded",
            "the queue is full; retry later",
        )
    }

    /// A 500 that hides `detail` from the client and logs it instead.
    pub(super) fn internal(detail: impl std::fmt::Display) -> Self {
        tracing::error!("internal error: {detail}");
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", "internal error")
    }

    pub(super) fn status(&self) -> StatusCode {
        self.status
    }

    pub(super) fn code(&self) -> &'static str {
        self.code
    }

    pub(super) fn message(&self) -> &str {
        &self.message
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        match error {
            Error::InvalidArgument(message) => Self::invalid(message),
            error => Self::internal(error),
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

#[cfg(test)]
mod tests {
    use super::ApiError;
    use crate::Error;

    #[test]
    fn only_invalid_requests_say_why() {
        assert_eq!(
            ApiError::from(Error::InvalidArgument("bad row".into())).message(),
            "bad row"
        );
        assert_eq!(
            ApiError::from(Error::Cache("/secret/path".into())).message(),
            "internal error"
        );
    }
}

use axum::{
    Json,
    body::Body,
    http::{Response, StatusCode},
    response::IntoResponse,
};
use serde_json::json;

use crate::logging::append_error_log;

#[derive(Debug)]
pub(crate) struct ProxyError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

impl ProxyError {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub(crate) fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
        }
    }

    pub(crate) fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
        }
    }

    pub(crate) fn server_error(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

pub(crate) fn is_context_window_error_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("context_length_exceeded")
        || lower.contains("context window")
        || lower.contains("max context")
        || lower.contains("maximum context")
        || lower.contains("too many tokens")
}

pub(crate) fn normalize_upstream_error_message(message: &str) -> String {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return "upstream request failed".to_string();
    }

    if trimmed
        .to_ascii_lowercase()
        .contains("prompt is too long")
    {
        return trimmed.to_string();
    }

    if is_context_window_error_message(trimmed) {
        return format!("Prompt is too long: {trimmed}");
    }

    trimmed.to_string()
}

pub(crate) fn error_response(error: ProxyError) -> Response<Body> {
    append_error_log(
        "proxy error",
        &format!("status: {}\nmessage: {}", error.status, error.message),
    );
    let body = Json(json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": error.message,
        }
    }));
    let mut response = (error.status, body).into_response();
    if error.status.is_client_error() && error.status != StatusCode::TOO_MANY_REQUESTS {
        response.headers_mut().insert(
            "x-should-retry",
            axum::http::HeaderValue::from_static("false"),
        );
    }
    response
}

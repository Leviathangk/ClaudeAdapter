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
    (error.status, body).into_response()
}

use std::{env, net::SocketAddr, sync::Arc};

use anyhow::{Context, Result, anyhow};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, Response},
    response::IntoResponse,
    routing::{get, post},
};
use futures_util::TryStreamExt;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use crate::{
    config::{
        Config, ProviderConfig, RunOptions, log_startup_config, mapped_model, start_watchers,
        validate_bind,
    },
    error::{ProxyError, error_response},
    logging::{append_error_log, preview_text},
    protocol::{
        AnthropicMessagesRequest, anthropic_error_response, anthropic_text_response,
        anthropic_to_provider_request, extract_anthropic_message_preview, extract_message_preview,
        provider_response_to_anthropic,
    },
    streaming::{anthropic_single_response_sse, anthropic_sse_stream, collect_text_from_sse},
};

const DEFAULT_CHAT_PATH: &str = "/chat/completions";
const DEFAULT_RESPONSES_PATH: &str = "/responses";

#[derive(Clone)]
pub(crate) struct AppState {
    config: Arc<RwLock<Config>>,
    client: Client,
}

pub fn build_router(config: Config) -> Router {
    build_router_with_shared_config(Arc::new(RwLock::new(config)))
}

fn build_router_with_shared_config(config: Arc<RwLock<Config>>) -> Router {
    let state = AppState {
        config,
        client: Client::new(),
    };

    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/messages", post(anthropic_messages_handler))
        .route("/chat/completions", post(chat_handler))
        .route("/responses", post(responses_handler))
        .with_state(state)
}

pub async fn run(config: Config, options: RunOptions) -> Result<()> {
    let bind = config.bind.clone();
    log_startup_config(&config);
    let shared_config = Arc::new(RwLock::new(config));
    start_watchers(shared_config.clone(), options)?;
    let app = build_router_with_shared_config(shared_config);
    let addr: SocketAddr = validate_bind(&bind)?;

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind: {bind}"))?;
    tracing::info!(%bind, "claude adapter listening");
    axum::serve(listener, app).await.context("server error")?;
    Ok(())
}

async fn current_config(state: &AppState) -> Config {
    state.config.read().await.clone()
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "ok": true }))
}

async fn anthropic_messages_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let config = current_config(&state).await;
    if let Err(err) = authorize_incoming("v1_messages", &config, &headers) {
        return error_response(err);
    }

    let raw_body = String::from_utf8_lossy(&body).to_string();
    let body_preview = preview_text(&raw_body, 400);
    tracing::info!(body_preview = %body_preview, "incoming anthropic raw body");

    let payload: AnthropicMessagesRequest = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(error) => {
            tracing::error!(error = %error, body_preview = %body_preview, "failed to parse anthropic request body");
            append_error_log(
                "failed to parse anthropic request body",
                &format!("error: {error}\nbody_preview: {body_preview}"),
            );
            return error_response(ProxyError::bad_request(format!(
                "invalid anthropic request body: {error}"
            )));
        }
    };

    match anthropic_messages_inner(state, payload).await {
        Ok(response) => response,
        Err(err) => error_response(err),
    }
}

async fn chat_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response<Body> {
    proxy_request(state, headers, payload, "chat_completions").await
}

async fn responses_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response<Body> {
    proxy_request(state, headers, payload, "responses").await
}

async fn anthropic_messages_inner(
    state: AppState,
    payload: AnthropicMessagesRequest,
) -> Result<Response<Body>, ProxyError> {
    let config = current_config(&state).await;

    let provider = active_provider(&config)?;
    let target_model = mapped_model(provider, &payload.model);
    let message_preview = extract_anthropic_message_preview(&payload);

    tracing::info!(
        endpoint = "v1_messages",
        provider = %config.activate_provider,
        requested_model = %payload.model,
        target_model = %target_model,
        message_preview = %message_preview,
        "incoming request"
    );

    let upstream_payload = anthropic_to_provider_request(&payload, provider, &target_model)?;
    let upstream = send_upstream_request(&state, provider, upstream_payload).await?;

    if payload.stream {
        return stream_anthropic_response(provider.api_mode, &payload, upstream).await;
    }

    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let body = upstream.bytes().await.map_err(|e| {
        tracing::error!(error = %e, "failed to read anthropic upstream body");
        ProxyError::bad_gateway(format!("failed to read upstream body: {e}"))
    })?;
    let body_preview = preview_text(&String::from_utf8_lossy(&body), 400);

    tracing::info!(status = %status, content_type = %content_type, body_preview = %body_preview, "anthropic upstream response");

    if !status.is_success() {
        return anthropic_error_response(status, &content_type, &body);
    }

    let response = if content_type.starts_with("text/event-stream") {
        let text = collect_text_from_sse(provider.api_mode, &body)?;
        anthropic_text_response(&payload.model, &payload, text, None, None)
    } else {
        let upstream_json: Value = serde_json::from_slice(&body).map_err(|e| {
            append_error_log(
                "invalid upstream json",
                &format!(
                    "error: {e}\ncontent_type: {content_type}\nbody_preview: {}",
                    preview_text(&String::from_utf8_lossy(&body), 400)
                ),
            );
            ProxyError::bad_gateway(format!("invalid upstream json: {e}"))
        })?;
        provider_response_to_anthropic(provider.api_mode, &payload.model, &payload, upstream_json)?
    };

    Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&response).map_err(|e| {
            ProxyError::server_error(format!("failed to encode response: {e}"))
        })?))
        .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")))
}

async fn stream_anthropic_response(
    api_mode: crate::config::ApiMode,
    request: &AnthropicMessagesRequest,
    upstream: reqwest::Response,
) -> Result<Response<Body>, ProxyError> {
    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    tracing::info!(status = %status, content_type = %content_type, "anthropic upstream stream opened");
    if !status.is_success() {
        let body = upstream.bytes().await.map_err(|e| {
            tracing::error!(error = %e, "failed to read anthropic error upstream body");
            ProxyError::bad_gateway(format!("failed to read upstream body: {e}"))
        })?;
        return anthropic_error_response(status, &content_type, &body);
    }

    if !content_type.starts_with("text/event-stream") {
        let body = upstream.bytes().await.map_err(|e| {
            tracing::error!(error = %e, "failed to read anthropic non-sse upstream body");
            ProxyError::bad_gateway(format!("failed to read upstream body: {e}"))
        })?;
        let body_preview = preview_text(&String::from_utf8_lossy(&body), 400);
        tracing::info!(status = %status, content_type = %content_type, body_preview = %body_preview, "anthropic upstream json response");

        let upstream_json: Value = serde_json::from_slice(&body).map_err(|e| {
            append_error_log(
                "invalid upstream json",
                &format!("error: {e}\ncontent_type: {content_type}\nbody_preview: {body_preview}"),
            );
            ProxyError::bad_gateway(format!("invalid upstream json: {e}"))
        })?;
        let response =
            provider_response_to_anthropic(api_mode, &request.model, request, upstream_json)?;
        let stream = anthropic_single_response_sse(response)?;

        return Response::builder()
            .status(axum::http::StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
            .header(axum::http::header::CACHE_CONTROL, "no-cache")
            .body(Body::from(stream))
            .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")));
    }

    let message_id = format!("msg_{}", simple_id());
    let model = request.model.clone();
    let input_text = request
        .messages
        .iter()
        .filter_map(|message| crate::protocol::anthropic_content_to_text(&message.content))
        .collect::<Vec<_>>()
        .join("\n");
    let input_tokens = crate::protocol::estimate_token_count(&input_text);
    let stream = anthropic_sse_stream(api_mode, upstream, message_id, model, input_tokens);

    Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")))
}

async fn proxy_request(
    state: AppState,
    headers: HeaderMap,
    mut payload: Value,
    local_endpoint: &'static str,
) -> Response<Body> {
    match proxy_request_inner(state, headers, &mut payload, local_endpoint).await {
        Ok(response) => response,
        Err(err) => error_response(err),
    }
}

async fn proxy_request_inner(
    state: AppState,
    headers: HeaderMap,
    payload: &mut Value,
    local_endpoint: &'static str,
) -> Result<Response<Body>, ProxyError> {
    let config = current_config(&state).await;
    authorize_incoming(local_endpoint, &config, &headers)?;

    let requested_model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::bad_request("missing or invalid 'model' field"))?;

    let provider = active_provider(&config)?;
    let target_model = mapped_model(provider, requested_model);
    let message_preview = extract_message_preview(payload);

    tracing::info!(
        endpoint = local_endpoint,
        provider = %config.activate_provider,
        requested_model = requested_model,
        target_model = %target_model,
        message_preview = %message_preview,
        "incoming request"
    );

    payload["model"] = Value::String(target_model);
    let streaming = is_streaming_request(payload);

    let upstream = send_upstream_request(&state, provider, payload.clone()).await?;

    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .cloned();
    let content_type_text = content_type
        .as_ref()
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    tracing::info!(endpoint = local_endpoint, status = %status, content_type = %content_type_text, "openai upstream response opened");
    let mut response = Response::builder().status(status);
    if let Some(content_type) = content_type {
        response = response.header(axum::http::header::CONTENT_TYPE, content_type.clone());
    }

    if let Some(cache_control) = upstream
        .headers()
        .get(axum::http::header::CACHE_CONTROL)
        .cloned()
    {
        response = response.header(axum::http::header::CACHE_CONTROL, cache_control);
    }

    if streaming {
        let stream = upstream.bytes_stream().map_err(move |e| {
            tracing::error!(endpoint = local_endpoint, error = %e, "openai upstream stream error");
            std::io::Error::other(format!("upstream stream error: {e}"))
        });
        return response
            .body(Body::from_stream(stream))
            .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")));
    }

    let body = upstream.bytes().await.map_err(|e| {
        tracing::error!(endpoint = local_endpoint, error = %e, "failed to read openai upstream body");
        ProxyError::bad_gateway(format!("failed to read upstream body: {e}"))
    })?;
    let body_preview = preview_text(&String::from_utf8_lossy(&body), 400);
    tracing::info!(endpoint = local_endpoint, status = %status, content_type = %content_type_text, body_preview = %body_preview, "openai upstream response");

    response
        .body(Body::from(body))
        .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")))
}

fn is_streaming_request(payload: &Value) -> bool {
    payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn active_provider(config: &Config) -> Result<&ProviderConfig, ProxyError> {
    config
        .providers
        .get(&config.activate_provider)
        .ok_or_else(|| {
            ProxyError::server_error(format!(
                "active provider not found: {}",
                config.activate_provider
            ))
        })
}

async fn send_upstream_request(
    state: &AppState,
    provider: &ProviderConfig,
    payload: Value,
) -> Result<reqwest::Response, ProxyError> {
    let url = provider_url(provider)?;
    tracing::info!(url = %url, api_mode = ?provider.api_mode, "sending upstream request");
    let mut request = state.client.post(&url).json(&payload);

    let mut outbound_headers = HeaderMap::new();
    for (name, value) in &provider.headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|e| ProxyError::server_error(format!("invalid header name '{name}': {e}")))?;
        let header_value = HeaderValue::from_str(value).map_err(|e| {
            ProxyError::server_error(format!("invalid header value for '{name}': {e}"))
        })?;
        outbound_headers.insert(header_name, header_value);
    }

    let api_key = resolve_secret(&provider.api_key)
        .map_err(|e| ProxyError::server_error(format!("provider api_key error: {e}")))?;
    outbound_headers.insert(
        axum::http::header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| ProxyError::server_error(format!("invalid authorization header: {e}")))?,
    );

    request = request.headers(outbound_headers);
    request.send().await.map_err(|e| {
        tracing::error!(url = %url, error = %e, "upstream request failed");
        let message = if e.is_timeout() {
            "upstream request timed out".to_string()
        } else if e.is_connect() {
            format!("failed to connect to upstream: {e}")
        } else {
            format!("upstream request failed: {e}")
        };
        ProxyError::bad_gateway(message)
    })
}

fn authorize_incoming(
    endpoint: &'static str,
    config: &Config,
    headers: &HeaderMap,
) -> Result<(), ProxyError> {
    let Some(expected) = &config.incoming_api_key else {
        return Ok(());
    };

    let expected = resolve_secret(expected)
        .map_err(|e| ProxyError::server_error(format!("incoming_api_key error: {e}")))?;
    let actual = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
                .map(|s| s.to_string())
        });

    match actual.as_deref() {
        Some(token) if token == expected => Ok(()),
        _ => {
            tracing::warn!(
                endpoint,
                auth_source = incoming_auth_source(headers),
                has_bearer = headers
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok())
                    .map(|value| value.starts_with("Bearer "))
                    .unwrap_or(false),
                "incoming authorization rejected"
            );
            Err(ProxyError::unauthorized("invalid bearer token"))
        }
    }
}

fn incoming_auth_source(headers: &HeaderMap) -> &'static str {
    if headers.contains_key(axum::http::header::AUTHORIZATION) {
        "authorization"
    } else if headers.contains_key("x-api-key") {
        "x-api-key"
    } else {
        "none"
    }
}

fn provider_url(provider: &ProviderConfig) -> Result<String, ProxyError> {
    let base = provider.base_url.trim_end_matches('/');
    let path = match provider.api_mode {
        crate::config::ApiMode::ChatCompletions => DEFAULT_CHAT_PATH,
        crate::config::ApiMode::Responses => DEFAULT_RESPONSES_PATH,
    };

    Ok(format!("{base}{path}"))
}

fn resolve_secret(raw: &str) -> Result<String> {
    if let Some(key) = raw.strip_prefix("env:") {
        let value =
            env::var(key).with_context(|| format!("missing environment variable: {key}"))?;
        if value.is_empty() {
            return Err(anyhow!("environment variable is empty: {key}"));
        }
        return Ok(value);
    }

    if raw.is_empty() {
        return Err(anyhow!("secret is empty"));
    }

    Ok(raw.to_string())
}

fn simple_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}")
}

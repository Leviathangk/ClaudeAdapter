use std::{collections::HashMap, env, net::SocketAddr, sync::Arc};

use anyhow::{Context, Result, anyhow};
use async_stream::try_stream;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use futures_util::TryStreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_CHAT_PATH: &str = "/chat/completions";
const DEFAULT_RESPONSES_PATH: &str = "/responses";

#[derive(Clone)]
pub struct AppState {
    config: Arc<Config>,
    client: Client,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default)]
    pub incoming_api_key: Option<String>,
    pub activate_provider: String,
    pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_mode: ApiMode,
    pub api_key: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub model_default: String,
    #[serde(default)]
    pub model_map: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiMode {
    ChatCompletions,
    Responses,
}

fn default_bind() -> String {
    "127.0.0.1:8787".to_string()
}

pub fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {path}"))?;
    let config: Config = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse yaml config: {path}"))?;
    Ok(config)
}

pub fn build_router(config: Config) -> Router {
    let state = AppState {
        config: Arc::new(config),
        client: Client::new(),
    };

    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/messages", post(anthropic_messages_handler))
        .route("/chat/completions", post(chat_handler))
        .route("/responses", post(responses_handler))
        .with_state(state)
}

pub async fn run(config: Config) -> Result<()> {
    let bind = config.bind.clone();
    log_startup_config(&config);
    let app = build_router(config);
    let addr: SocketAddr = bind
        .parse()
        .with_context(|| format!("invalid bind address: {bind}"))?;

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind: {bind}"))?;
    tracing::info!(%bind, "claude adapter listening");
    axum::serve(listener, app).await.context("server error")?;
    Ok(())
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "ok": true }))
}

async fn anthropic_messages_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let raw_body = String::from_utf8_lossy(&body).to_string();
    let body_preview = preview_text(&raw_body, 400);
    tracing::info!(body_preview = %body_preview, "incoming anthropic raw body");

    let payload: AnthropicMessagesRequest = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(error) => {
            tracing::error!(
                error = %error,
                body_preview = %body_preview,
                "failed to parse anthropic request body"
            );
            return error_response(ProxyError::bad_request(format!(
                "invalid anthropic request body: {error}"
            )));
        }
    };

    match anthropic_messages_inner(state, headers, payload).await {
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
    headers: HeaderMap,
    payload: AnthropicMessagesRequest,
) -> Result<Response<Body>, ProxyError> {
    authorize_incoming(&state.config, &headers)?;

    let provider = active_provider(&state.config)?;
    let target_model = mapped_model(provider, &payload.model);
    let message_preview = extract_anthropic_message_preview(&payload);

    tracing::info!(
        endpoint = "v1_messages",
        provider = %state.config.activate_provider,
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

    tracing::info!(
        status = %status,
        content_type = %content_type,
        body_preview = %body_preview,
        "anthropic upstream response"
    );

    if !status.is_success() {
        return Response::builder()
            .status(status)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")));
    }

    let response = if content_type.starts_with("text/event-stream") {
        let text = collect_text_from_sse(provider.api_mode, &body)?;
        anthropic_text_response(&payload.model, &payload, text)
    } else {
        let upstream_json: Value = serde_json::from_slice(&body)
            .map_err(|e| ProxyError::bad_gateway(format!("invalid upstream json: {e}")))?;
        provider_response_to_anthropic(provider.api_mode, &payload.model, &payload, upstream_json)?
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&response).map_err(|e| {
            ProxyError::server_error(format!("failed to encode response: {e}"))
        })?))
        .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")))
}

async fn stream_anthropic_response(
    api_mode: ApiMode,
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
        let content_type = upstream
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("application/json"));
        let stream = upstream
            .bytes_stream()
            .map_err(|e| std::io::Error::other(format!("upstream stream error: {e}")));
        return Response::builder()
            .status(status)
            .header(axum::http::header::CONTENT_TYPE, content_type)
            .body(Body::from_stream(stream))
            .map_err(|e| ProxyError::server_error(format!("failed to build response: {e}")));
    }

    if !content_type.starts_with("text/event-stream") {
        let body = upstream.bytes().await.map_err(|e| {
            tracing::error!(error = %e, "failed to read anthropic non-sse upstream body");
            ProxyError::bad_gateway(format!("failed to read upstream body: {e}"))
        })?;
        let body_preview = preview_text(&String::from_utf8_lossy(&body), 400);
        tracing::info!(status = %status, content_type = %content_type, body_preview = %body_preview, "anthropic upstream json response");

        let upstream_json: Value = serde_json::from_slice(&body)
            .map_err(|e| ProxyError::bad_gateway(format!("invalid upstream json: {e}")))?;
        let response =
            provider_response_to_anthropic(api_mode, &request.model, request, upstream_json)?;
        let stream = anthropic_single_response_sse(response)?;

        return Response::builder()
            .status(StatusCode::OK)
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
        .filter_map(|message| anthropic_content_to_text(&message.content))
        .collect::<Vec<_>>()
        .join("\n");
    let input_tokens = estimate_token_count(&input_text);
    let stream = anthropic_sse_stream(api_mode, upstream, message_id, model, input_tokens);

    Response::builder()
        .status(StatusCode::OK)
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
    authorize_incoming(&state.config, &headers)?;

    let requested_model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::bad_request("missing or invalid 'model' field"))?;

    let provider = active_provider(&state.config)?;
    let target_model = mapped_model(provider, requested_model);
    let message_preview = extract_message_preview(payload);

    tracing::info!(
        endpoint = local_endpoint,
        provider = %state.config.activate_provider,
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

    let body = upstream
        .bytes()
        .await
        .map_err(|e| {
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

fn log_startup_config(config: &Config) {
    match config.providers.get(&config.activate_provider) {
        Some(provider) => {
            let mut mappings: Vec<String> = provider
                .model_map
                .iter()
                .map(|(from, to)| format!("{from} -> {to}"))
                .collect();
            mappings.sort();

            tracing::info!(
                provider = %config.activate_provider,
                model_default = %provider.model_default,
                model_map = %mappings.join(", "),
                "active provider loaded"
            );
        }
        None => {
            tracing::error!(
                provider = %config.activate_provider,
                "active provider missing from config"
            );
        }
    }
}

fn mapped_model(provider: &ProviderConfig, requested_model: &str) -> String {
    provider
        .model_map
        .get(requested_model)
        .cloned()
        .unwrap_or_else(|| provider.model_default.clone())
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
        ProxyError::bad_gateway(format!("upstream request failed: {e}"))
    })
}

fn extract_message_preview(payload: &Value) -> String {
    extract_text_from_messages(payload)
        .or_else(|| extract_text_from_input(payload))
        .unwrap_or_default()
        .chars()
        .take(10)
        .collect()
}

fn extract_text_from_messages(payload: &Value) -> Option<String> {
    let messages = payload.get("messages")?.as_array()?;
    for message in messages {
        let content = message.get("content")?;
        if let Some(text) = value_as_text(content) {
            if !text.is_empty() {
                return Some(text);
            }
        }
    }
    None
}

fn extract_text_from_input(payload: &Value) -> Option<String> {
    value_as_text(payload.get("input")?)
}

fn extract_anthropic_message_preview(payload: &AnthropicMessagesRequest) -> String {
    payload
        .messages
        .iter()
        .find_map(|message| anthropic_content_to_text(&message.content))
        .unwrap_or_default()
        .chars()
        .take(10)
        .collect()
}

fn anthropic_content_to_text(content: &AnthropicContent) -> Option<String> {
    match content {
        AnthropicContent::Text(text) => Some(text.clone()),
        AnthropicContent::Blocks(blocks) => blocks.iter().find_map(|block| match block {
            AnthropicContentBlock::Text { text, .. } => Some(text.clone()),
            AnthropicContentBlock::Other => None,
        }),
    }
}

fn anthropic_to_provider_request(
    payload: &AnthropicMessagesRequest,
    provider: &ProviderConfig,
    target_model: &str,
) -> Result<Value, ProxyError> {
    match provider.api_mode {
        ApiMode::ChatCompletions => Ok(anthropic_to_chat_completions(payload, target_model)),
        ApiMode::Responses => anthropic_to_responses(payload, target_model),
    }
}

fn anthropic_to_chat_completions(payload: &AnthropicMessagesRequest, target_model: &str) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = payload.system_text() {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for message in &payload.messages {
        messages.push(json!({
            "role": message.role,
            "content": anthropic_content_to_text(&message.content).unwrap_or_default()
        }));
    }

    let mut body = json!({
        "model": target_model,
        "messages": messages,
    });
    if let Some(max_tokens) = payload.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = payload.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = payload.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(stop_sequences) = &payload.stop_sequences {
        body["stop"] = json!(stop_sequences);
    }
    body
}

fn anthropic_to_responses(
    payload: &AnthropicMessagesRequest,
    target_model: &str,
) -> Result<Value, ProxyError> {
    let mut lines = Vec::new();
    if let Some(system) = payload.system_text() {
        lines.push(format!("System: {system}"));
    }
    for message in &payload.messages {
        let role = match message.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            _ => "User",
        };
        let text = anthropic_content_to_text(&message.content).unwrap_or_default();
        lines.push(format!("{role}: {text}"));
    }
    let input = lines.join("\n\n");
    if input.is_empty() {
        return Err(ProxyError::bad_request(
            "anthropic messages content is empty",
        ));
    }

    let mut body = json!({
        "model": target_model,
        "input": input,
    });
    if let Some(max_tokens) = payload.max_tokens {
        body["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = payload.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = payload.top_p {
        body["top_p"] = json!(top_p);
    }
    Ok(body)
}

fn provider_response_to_anthropic(
    api_mode: ApiMode,
    requested_model: &str,
    request: &AnthropicMessagesRequest,
    upstream_json: Value,
) -> Result<AnthropicMessagesResponse, ProxyError> {
    let text = match api_mode {
        ApiMode::ChatCompletions => extract_chat_completion_text(&upstream_json),
        ApiMode::Responses => extract_responses_text(&upstream_json),
    }
    .ok_or_else(|| ProxyError::bad_gateway("failed to extract text from upstream response"))?;

    Ok(anthropic_text_response(requested_model, request, text))
}

fn anthropic_text_response(
    requested_model: &str,
    request: &AnthropicMessagesRequest,
    text: String,
) -> AnthropicMessagesResponse {
    let output_tokens = estimate_token_count(&text);
    let input_text = request
        .messages
        .iter()
        .filter_map(|message| anthropic_content_to_text(&message.content))
        .collect::<Vec<_>>()
        .join("\n");

    AnthropicMessagesResponse {
        id: format!("msg_{}", simple_id()),
        message_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![AnthropicTextBlock {
            block_type: "text".to_string(),
            text,
        }],
        model: requested_model.to_string(),
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: estimate_token_count(&input_text),
            output_tokens,
        },
    }
}

fn extract_chat_completion_text(value: &Value) -> Option<String> {
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")
        .and_then(value_as_text)
}

fn extract_responses_text(value: &Value) -> Option<String> {
    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }

    let output = value.get("output")?.as_array()?;
    for item in output {
        if let Some(content) = item.get("content").and_then(Value::as_array) {
            for part in content {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }
        }
    }
    None
}

fn collect_text_from_sse(api_mode: ApiMode, body: &[u8]) -> Result<String, ProxyError> {
    let text = String::from_utf8_lossy(body);
    let mut parser = SseParser::default();
    let events = parser.push(&text);
    let mut output = String::new();

    for event in events {
        if event == "[DONE]" {
            continue;
        }
        if let Some(delta) = extract_stream_text(api_mode, &event) {
            output.push_str(&delta);
        }
    }

    if output.is_empty() {
        return Err(ProxyError::bad_gateway(
            "failed to extract text from upstream sse response",
        ));
    }

    Ok(output)
}

fn anthropic_sse_stream(
    api_mode: ApiMode,
    upstream: reqwest::Response,
    message_id: String,
    model: String,
    input_tokens: i64,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    try_stream! {
        yield sse_event_bytes(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0
                    }
                }
            }),
        )?;
        yield sse_event_bytes(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }),
        )?;

        let mut parser = SseParser::default();
        let mut output_text = String::new();
        let mut stop_reason = "end_turn".to_string();
        let mut upstream_stream = upstream.bytes_stream();
        while let Some(chunk) = upstream_stream.try_next().await.map_err(|e| {
            tracing::error!(error = %e, "anthropic upstream stream read error");
            io_error(e)
        })? {
            let events = parser.push(&String::from_utf8_lossy(&chunk));
            for event in events {
                match event.as_str() {
                    "[DONE]" => {}
                    _ => {
                        if let Some(text) = extract_stream_text(api_mode, &event) {
                            if !text.is_empty() {
                                output_text.push_str(&text);
                                yield sse_event_bytes(
                                    "content_block_delta",
                                    json!({
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": text
                                        }
                                    }),
                                )?;
                            }
                        }

                        if let Some(reason) = extract_stream_stop_reason(api_mode, &event) {
                            stop_reason = reason;
                        }
                    }
                }
            }
        }

        yield sse_event_bytes(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": 0
            }),
        )?;

        yield sse_event_bytes(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": Value::Null
                },
                "usage": {
                    "output_tokens": estimate_token_count(&output_text)
                }
            }),
        )?;
        yield sse_event_bytes("message_stop", json!({"type": "message_stop"}))?;
        tracing::info!(output_preview = %preview_text(&output_text, 120), "anthropic stream completed");
    }
}

fn anthropic_single_response_sse(response: AnthropicMessagesResponse) -> Result<Bytes, ProxyError> {
    let text = response
        .content
        .first()
        .map(|block| block.text.clone())
        .unwrap_or_default();

    let mut chunks = Vec::new();
    chunks.push(
        sse_event_bytes(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": response.id,
                    "type": response.message_type,
                    "role": response.role,
                    "content": [],
                    "model": response.model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": 0
                    }
                }
            }),
        )
        .map_err(|e| ProxyError::server_error(format!("failed to encode sse event: {e}")))?,
    );
    chunks.push(
        sse_event_bytes(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }),
        )
        .map_err(|e| ProxyError::server_error(format!("failed to encode sse event: {e}")))?,
    );
    if !text.is_empty() {
        chunks.push(
            sse_event_bytes(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": text
                    }
                }),
            )
            .map_err(|e| ProxyError::server_error(format!("failed to encode sse event: {e}")))?,
        );
    }
    chunks.push(
        sse_event_bytes(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": 0
            }),
        )
        .map_err(|e| ProxyError::server_error(format!("failed to encode sse event: {e}")))?,
    );
    chunks.push(
        sse_event_bytes(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence
                },
                "usage": {
                    "output_tokens": response.usage.output_tokens
                }
            }),
        )
        .map_err(|e| ProxyError::server_error(format!("failed to encode sse event: {e}")))?,
    );
    chunks.push(
        sse_event_bytes("message_stop", json!({"type": "message_stop"}))
            .map_err(|e| ProxyError::server_error(format!("failed to encode sse event: {e}")))?,
    );

    let mut bytes = Vec::new();
    for chunk in chunks {
        bytes.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(bytes))
}

fn sse_event_bytes(event: &str, data: Value) -> Result<Bytes, std::io::Error> {
    let payload = serde_json::to_string(&data)
        .map_err(|e| std::io::Error::other(format!("failed to encode sse event: {e}")))?;
    Ok(Bytes::from(format!("event: {event}\ndata: {payload}\n\n")))
}

fn extract_stream_text(api_mode: ApiMode, raw: &str) -> Option<String> {
    let value: Value = serde_json::from_str(raw).ok()?;
    match api_mode {
        ApiMode::ChatCompletions => value
            .get("choices")?
            .as_array()?
            .first()?
            .get("delta")?
            .get("content")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        ApiMode::Responses => extract_responses_stream_text(&value),
    }
}

fn extract_responses_stream_text(value: &Value) -> Option<String> {
    value
        .get("delta")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("output_text")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .or_else(|| {
            value
                .get("output")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("content"))
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|part| part.get("text"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
}

fn extract_stream_stop_reason(api_mode: ApiMode, raw: &str) -> Option<String> {
    let value: Value = serde_json::from_str(raw).ok()?;
    let reason = match api_mode {
        ApiMode::ChatCompletions => value
            .get("choices")?
            .as_array()?
            .first()?
            .get("finish_reason")
            .and_then(Value::as_str),
        ApiMode::Responses => value
            .get("stop_reason")
            .and_then(Value::as_str)
            .or_else(|| value.get("finish_reason").and_then(Value::as_str)),
    }?;
    Some(map_stop_reason(reason))
}

fn map_stop_reason(reason: &str) -> String {
    match reason {
        "length" | "max_tokens" => "max_tokens",
        "stop" | "end_turn" => "end_turn",
        other => other,
    }
    .to_string()
}

fn io_error<E: std::fmt::Display>(error: E) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

#[derive(Default)]
struct SseParser {
    buffer: String,
}

impl SseParser {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        while let Some(index) = self.buffer.find("\n\n") {
            let frame = self.buffer[..index].to_string();
            self.buffer.drain(..index + 2);
            if let Some(data) = parse_sse_frame(&frame) {
                events.push(data);
            }
        }
        events
    }
}

fn parse_sse_frame(frame: &str) -> Option<String> {
    let normalized = frame.replace("\r", "");
    let mut data_lines = Vec::new();
    for line in normalized.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }
    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
    }
}

fn preview_text(text: &str, limit: usize) -> String {
    let mut preview: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        preview.push_str("...");
    }
    preview
}

fn estimate_token_count(text: &str) -> i64 {
    ((text.chars().count() as i64) / 4).max(1)
}

fn simple_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}")
}

fn value_as_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            for item in items {
                if let Some(text) = value_as_text(item) {
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
            None
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                return Some(text.to_string());
            }
            if let Some(content) = map.get("content") {
                return value_as_text(content);
            }
            None
        }
        _ => None,
    }
}

fn authorize_incoming(config: &Config, headers: &HeaderMap) -> Result<(), ProxyError> {
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
        _ => Err(ProxyError::unauthorized("invalid bearer token")),
    }
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessagesRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    max_tokens: Option<i64>,
    #[serde(default)]
    system: Option<AnthropicContent>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    stream: bool,
}

impl AnthropicMessagesRequest {
    fn system_text(&self) -> Option<String> {
        self.system.as_ref().and_then(anthropic_content_to_text)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Serialize)]
struct AnthropicMessagesResponse {
    id: String,
    #[serde(rename = "type")]
    message_type: String,
    role: String,
    content: Vec<AnthropicTextBlock>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct AnthropicTextBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: i64,
    output_tokens: i64,
}

fn provider_url(provider: &ProviderConfig) -> Result<String, ProxyError> {
    let base = provider.base_url.trim_end_matches('/');
    let path = match provider.api_mode {
        ApiMode::ChatCompletions => DEFAULT_CHAT_PATH,
        ApiMode::Responses => DEFAULT_RESPONSES_PATH,
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

#[derive(Debug)]
struct ProxyError {
    status: StatusCode,
    message: String,
}

impl ProxyError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
        }
    }

    fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
        }
    }

    fn server_error(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

fn error_response(error: ProxyError) -> Response<Body> {
    let body = Json(json!({
        "error": {
            "message": error.message,
        }
    }));
    (error.status, body).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Bytes, to_bytes},
        http::Request,
    };
    use httpmock::{Method::POST, MockServer};
    use std::{
        convert::Infallible,
        sync::{Arc, Mutex},
    };
    use tower::ServiceExt;

    fn test_config(base_url: String, api_mode: ApiMode) -> Config {
        Config {
            bind: default_bind(),
            incoming_api_key: Some("incoming-secret".to_string()),
            activate_provider: "active-provider".to_string(),
            providers: HashMap::from([(
                "active-provider".to_string(),
                ProviderConfig {
                    base_url,
                    api_mode,
                    api_key: "provider-secret".to_string(),
                    headers: HashMap::from([("x-extra-header".to_string(), "adapter".to_string())]),
                    model_default: "fallback-model".to_string(),
                    model_map: HashMap::from([
                        ("claude-sonnet-4.6".to_string(), "gpt-4.1-mini".to_string()),
                        ("claude-opus-4-6".to_string(), "o3".to_string()),
                    ]),
                },
            )]),
        }
    }

    #[tokio::test]
    async fn forwards_chat_requests_with_model_mapping() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("authorization", "Bearer provider-secret")
                .header("x-extra-header", "adapter")
                .json_body(json!({
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "hi"}]
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "chatcmpl_123"}));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
        let response = app
            .oneshot(
                Request::post("/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "messages": [{"role": "user", "content": "hi"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["id"],
            "chatcmpl_123"
        );
        upstream.assert();
    }

    #[tokio::test]
    async fn forwards_responses_requests_to_custom_path() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/responses")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "o3",
                    "input": "hello"
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "resp_123"}));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::Responses));
        let response = app
            .oneshot(
                Request::post("/responses")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-opus-4-6",
                            "input": "hello"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["id"],
            "resp_123"
        );
        upstream.assert();
    }

    #[tokio::test]
    async fn rejects_requests_without_expected_token() {
        let app = build_router(test_config(
            "http://127.0.0.1:1".to_string(),
            ApiMode::ChatCompletions,
        ));
        let response = app
            .oneshot(
                Request::post("/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "messages": []
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn streams_chat_completions_sse_without_buffering() {
        let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
        let captured_for_route = Arc::clone(&captured);

        let upstream_app = Router::new().route(
            "/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let captured_for_route = Arc::clone(&captured_for_route);
                async move {
                    captured_for_route.lock().unwrap().push(payload);
                    let stream = futures_util::stream::iter(vec![
                        Ok::<Bytes, Infallible>(Bytes::from("data: {\"id\":\"chunk_1\"}\n\n")),
                        Ok::<Bytes, Infallible>(Bytes::from("data: [DONE]\n\n")),
                    ]);

                    Response::builder()
                        .status(StatusCode::OK)
                        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
                        .header(axum::http::header::CACHE_CONTROL, "no-cache")
                        .body(Body::from_stream(stream))
                        .unwrap()
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let upstream_handle = tokio::spawn(async move {
            axum::serve(listener, upstream_app).await.unwrap();
        });

        let app = build_router(test_config(
            format!("http://{address}"),
            ApiMode::ChatCompletions,
        ));
        let response = app
            .oneshot(
                Request::post("/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/event-stream"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"data: {\"id\":\"chunk_1\"}\n\ndata: [DONE]\n\n");

        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["model"], "gpt-4.1-mini");
        assert_eq!(requests[0]["stream"], true);

        upstream_handle.abort();
    }

    #[tokio::test]
    async fn falls_back_to_model_default_when_mapping_is_missing() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/responses")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "fallback-model",
                    "input": "hello"
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "resp_default"}));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::Responses));
        let response = app
            .oneshot(
                Request::post("/responses")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-unknown-model",
                            "input": "hello"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["id"],
            "resp_default"
        );
        upstream.assert();
    }

    #[tokio::test]
    async fn converts_anthropic_messages_to_chat_completions() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "system", "content": "Be brief"},
                        {"role": "user", "content": "hello"}
                    ],
                    "max_tokens": 32,
                    "temperature": 0.2
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "choices": [
                        {
                            "message": {"content": "hi"}
                        }
                    ]
                }));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "max_tokens": 32,
                            "temperature": 0.2,
                            "system": "Be brief",
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json = serde_json::from_slice::<Value>(&body).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["text"], "hi");
        assert_eq!(json["model"], "claude-sonnet-4.6");
        upstream.assert();
    }

    #[tokio::test]
    async fn accepts_anthropic_system_as_content_blocks() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "system", "content": "Be brief"},
                        {"role": "user", "content": "hello"}
                    ]
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "choices": [
                        {
                            "message": {"content": "hi"}
                        }
                    ]
                }));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "system": [{"type": "text", "text": "Be brief"}],
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        upstream.assert();
    }

    #[tokio::test]
    async fn converts_anthropic_messages_to_responses() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/responses")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "o3",
                    "input": "System: Answer politely\n\nUser: hello"
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "output": [
                        {
                            "content": [
                                {"text": "hello there"}
                            ]
                        }
                    ]
                }));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::Responses));
        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-opus-4-6",
                            "system": "Answer politely",
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json = serde_json::from_slice::<Value>(&body).unwrap();
        assert_eq!(json["content"][0]["text"], "hello there");
        assert_eq!(json["model"], "claude-opus-4-6");
        upstream.assert();
    }

    #[tokio::test]
    async fn streams_anthropic_messages_from_chat_completions() {
        let upstream_app = Router::new().route(
            "/chat/completions",
            post(|Json(_payload): Json<Value>| async move {
                let stream = futures_util::stream::iter(vec![
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from("data: [DONE]\n\n")),
                ]);

                Response::builder()
                    .status(StatusCode::OK)
                    .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let upstream_handle = tokio::spawn(async move {
            axum::serve(listener, upstream_app).await.unwrap();
        });

        let app = build_router(test_config(
            format!("http://{address}"),
            ApiMode::ChatCompletions,
        ));
        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "stream": true,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/event-stream"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("event: message_start"));
        assert!(text.contains("event: content_block_start"));
        assert!(text.contains("event: content_block_delta"));
        assert!(text.contains("\"text\":\"hel\""));
        assert!(text.contains("\"text\":\"lo\""));
        assert!(text.contains("event: message_delta"));
        assert!(text.contains("event: message_stop"));

        upstream_handle.abort();
    }

    #[tokio::test]
    async fn streams_anthropic_messages_from_json_upstream_response() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "choices": [
                        {
                            "message": {"content": "hello json"}
                        }
                    ]
                }));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "stream": true,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/event-stream"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("event: message_start"));
        assert!(text.contains("event: content_block_delta"));
        assert!(text.contains("\"text\":\"hello json\""));
        assert!(text.contains("event: message_stop"));
        upstream.assert();
    }

    #[tokio::test]
    async fn converts_non_stream_anthropic_messages_from_sse_upstream_response() {
        let upstream_app = Router::new().route(
            "/responses",
            post(|Json(_payload): Json<Value>| async move {
                let stream = futures_util::stream::iter(vec![
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "event: response.created\ndata: {\"type\":\"response.created\"}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "event: response.output_text.delta\ndata: {\"delta\":\"hello \"}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "event: response.output_text.delta\ndata: {\"delta\":\"world\"}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n",
                    )),
                ]);

                Response::builder()
                    .status(StatusCode::OK)
                    .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let upstream_handle = tokio::spawn(async move {
            axum::serve(listener, upstream_app).await.unwrap();
        });

        let app = build_router(test_config(format!("http://{address}"), ApiMode::Responses));
        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-opus-4-6",
                            "stream": false,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "application/json"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json = serde_json::from_slice::<Value>(&body).unwrap();
        assert_eq!(json["content"][0]["text"], "hello world");

        upstream_handle.abort();
    }
}

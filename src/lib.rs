use std::{
    collections::HashMap,
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

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
use notify::{Config as NotifyConfig, Event, RecommendedWatcher, RecursiveMode, Watcher};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;

const DEFAULT_CHAT_PATH: &str = "/chat/completions";
const DEFAULT_RESPONSES_PATH: &str = "/responses";

#[derive(Clone)]
pub struct AppState {
    config: Arc<RwLock<Config>>,
    client: Client,
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub config_path: PathBuf,
    pub env_path: Option<PathBuf>,
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

pub fn load_config(path: &Path) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;
    let config: Config = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse yaml config: {}", path.display()))?;
    Ok(config)
}

pub fn load_env_file(path: &Path) -> Result<Option<Vec<String>>> {
    if !path.exists() {
        return Ok(None);
    }

    let entries = dotenvy::from_path_iter(path)
        .with_context(|| format!("failed to read env file: {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to parse env file: {}", path.display()))?;

    let mut keys = Vec::new();
    for (key, value) in entries {
        unsafe {
            env::set_var(&key, value);
        }
        keys.push(key);
    }

    keys.sort();
    keys.dedup();
    tracing::info!(path = %path.display(), env_keys = %keys.join(", "), ".env file loaded");
    Ok(Some(keys))
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

async fn current_config(state: &AppState) -> Config {
    state.config.read().await.clone()
}

fn start_watchers(config: Arc<RwLock<Config>>, options: RunOptions) -> Result<()> {
    let config_path = options.config_path;
    let env_path = options.env_path;
    let watch_paths = collect_watch_paths(&config_path, env_path.as_deref())?;

    std::thread::Builder::new()
        .name("claude-adapter-watch".to_string())
        .spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut watcher = RecommendedWatcher::new(
                move |result| {
                    let _ = tx.send(result);
                },
                NotifyConfig::default(),
            )
            .expect("failed to create file watcher");

            for watch_path in &watch_paths {
                watcher
                    .watch(watch_path, RecursiveMode::NonRecursive)
                    .expect("failed to watch path");
            }

            while let Ok(event) = rx.recv() {
                match event {
                    Ok(event) => {
                        handle_watch_event(&config, &config_path, env_path.as_deref(), event)
                    }
                    Err(error) => tracing::error!(error = %error, "file watcher error"),
                }
            }
        })
        .context("failed to start file watcher thread")?;

    Ok(())
}

fn collect_watch_paths(config_path: &Path, env_path: Option<&Path>) -> Result<Vec<PathBuf>> {
    let mut paths = vec![parent_dir(config_path)?];
    if let Some(env_path) = env_path {
        let env_parent = parent_dir(env_path)?;
        if !paths.iter().any(|path| path == &env_parent) {
            paths.push(env_parent);
        }
    }
    Ok(paths)
}

fn parent_dir(path: &Path) -> Result<PathBuf> {
    path.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))
}

fn handle_watch_event(
    shared_config: &Arc<RwLock<Config>>,
    config_path: &Path,
    env_path: Option<&Path>,
    event: Event,
) {
    if event
        .paths
        .iter()
        .any(|path| matches_target(path, config_path))
    {
        match load_config(config_path) {
            Ok(config) => {
                let previous = shared_config.blocking_read().clone();
                *shared_config.blocking_write() = config.clone();
                log_startup_config(&config);
                log_config_change(&previous, &config, config_path);
            }
            Err(error) => {
                tracing::error!(path = %config_path.display(), error = %error, "failed to reload config");
            }
        }
    }

    if let Some(env_path) =
        env_path.filter(|target| event.paths.iter().any(|path| matches_target(path, target)))
    {
        match load_env_file(env_path) {
            Ok(Some(keys)) => {
                tracing::info!(path = %env_path.display(), refreshed_keys = %keys.join(", "), ".env reloaded")
            }
            Ok(None) => {
                tracing::info!(path = %env_path.display(), ".env not found, skipping reload")
            }
            Err(error) => {
                tracing::error!(path = %env_path.display(), error = %error, "failed to reload .env")
            }
        }
    }
}

fn matches_target(path: &Path, target: &Path) -> bool {
    path == target || (path.file_name() == target.file_name() && path.parent() == target.parent())
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
    let config = current_config(&state).await;
    authorize_incoming(&config, &headers)?;

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

    tracing::info!(
        status = %status,
        content_type = %content_type,
        body_preview = %body_preview,
        "anthropic upstream response"
    );

    if !status.is_success() {
        return anthropic_error_response(status, &content_type, &body);
    }

    let response = if content_type.starts_with("text/event-stream") {
        let text = collect_text_from_sse(provider.api_mode, &body)?;
        anthropic_text_response(&payload.model, &payload, text, None, None)
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
    let config = current_config(&state).await;
    authorize_incoming(&config, &headers)?;

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

fn log_config_change(previous: &Config, current: &Config, path: &Path) {
    let provider_change = format!(
        "{} -> {}",
        previous.activate_provider, current.activate_provider
    );
    let old_default = previous
        .providers
        .get(&previous.activate_provider)
        .map(|provider| provider.model_default.as_str())
        .unwrap_or("<missing>");
    let new_default = current
        .providers
        .get(&current.activate_provider)
        .map(|provider| provider.model_default.as_str())
        .unwrap_or("<missing>");

    tracing::info!(
        path = %path.display(),
        provider_change = %provider_change,
        model_default_change = %format!("{old_default} -> {new_default}"),
        "config reloaded"
    );
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
            AnthropicContentBlock::ToolUse { .. } => None,
            AnthropicContentBlock::ToolResult { .. } => None,
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
        messages.extend(anthropic_message_to_openai_messages(message));
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
    apply_tools_to_openai_body(&mut body, payload);
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
        lines.extend(anthropic_message_to_response_lines(message));
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
    apply_tools_to_openai_body(&mut body, payload);
    Ok(body)
}

fn apply_tools_to_openai_body(body: &mut Value, payload: &AnthropicMessagesRequest) {
    if !payload.tools.is_empty() {
        body["tools"] = json!(
            payload
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                    })
                })
                .collect::<Vec<_>>()
        );
    }

    if let Some(tool_choice) = &payload.tool_choice {
        body["tool_choice"] = anthropic_tool_choice_to_openai(tool_choice);
    }
}

fn anthropic_tool_choice_to_openai(tool_choice: &AnthropicToolChoice) -> Value {
    match tool_choice {
        AnthropicToolChoice::Auto {} => json!("auto"),
        AnthropicToolChoice::Any {} => json!("required"),
        AnthropicToolChoice::Tool { name } => json!({
            "type": "function",
            "function": {
                "name": name,
            }
        }),
    }
}

fn anthropic_message_to_openai_messages(message: &AnthropicMessage) -> Vec<Value> {
    match &message.content {
        AnthropicContent::Text(text) => vec![json!({
            "role": message.role,
            "content": text,
        })],
        AnthropicContent::Blocks(blocks) => {
            let mut messages = Vec::new();
            let text = blocks
                .iter()
                .filter_map(|block| match block {
                    AnthropicContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            let tool_calls = blocks
                .iter()
                .filter_map(|block| match block {
                    AnthropicContentBlock::ToolUse { id, name, input } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string()),
                        }
                    })),
                    _ => None,
                })
                .collect::<Vec<_>>();

            log_ignored_content_blocks(blocks);

            if !text.is_empty() || !tool_calls.is_empty() {
                let mut assistant = json!({
                    "role": message.role,
                    "content": text,
                });
                if !tool_calls.is_empty() {
                    assistant["tool_calls"] = json!(tool_calls);
                }
                messages.push(assistant);
            }

            for block in blocks {
                if let AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } = block
                {
                    let tool_content = anthropic_tool_result_text(content);
                    let mut tool_message = json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": tool_content,
                    });
                    if is_error.unwrap_or(false) {
                        tool_message["content"] = json!(format!(
                            "ERROR: {}",
                            tool_message["content"].as_str().unwrap_or_default()
                        ));
                    }
                    messages.push(tool_message);
                }
            }

            messages
        }
    }
}

fn anthropic_message_to_response_lines(message: &AnthropicMessage) -> Vec<String> {
    let mut lines = Vec::new();
    let role = match message.role.as_str() {
        "user" => "User",
        "assistant" => "Assistant",
        _ => "User",
    };

    if let Some(text) = anthropic_content_to_text(&message.content) {
        if !text.is_empty() {
            lines.push(format!("{role}: {text}"));
        }
    }

    if let AnthropicContent::Blocks(blocks) = &message.content {
        log_ignored_content_blocks(blocks);
        for block in blocks {
            match block {
                AnthropicContentBlock::ToolUse { name, input, .. } => {
                    lines.push(format!("Assistant tool_use {name}: {input}"));
                }
                AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let prefix = if is_error.unwrap_or(false) {
                        "Tool error"
                    } else {
                        "Tool result"
                    };
                    lines.push(format!(
                        "{prefix} {tool_use_id}: {}",
                        anthropic_tool_result_text(content)
                    ));
                }
                AnthropicContentBlock::Text { .. } | AnthropicContentBlock::Other => {}
            }
        }
    }

    lines
}

fn anthropic_tool_result_text(content: &Value) -> String {
    value_as_text(content).unwrap_or_else(|| content.to_string())
}

fn log_ignored_content_blocks(blocks: &[AnthropicContentBlock]) {
    for block in blocks {
        if matches!(block, AnthropicContentBlock::Other) {
            tracing::info!("ignoring unsupported anthropic content block");
        }
    }
}

fn provider_response_to_anthropic(
    api_mode: ApiMode,
    requested_model: &str,
    request: &AnthropicMessagesRequest,
    upstream_json: Value,
) -> Result<AnthropicMessagesResponse, ProxyError> {
    if let Some(response) =
        extract_tool_use_response(api_mode, requested_model, request, &upstream_json)
    {
        return Ok(response);
    }

    let text = match api_mode {
        ApiMode::ChatCompletions => extract_chat_completion_text(&upstream_json),
        ApiMode::Responses => extract_responses_text(&upstream_json),
    }
    .ok_or_else(|| ProxyError::bad_gateway("failed to extract text from upstream response"))?;

    Ok(anthropic_text_response(
        requested_model,
        request,
        text,
        extract_upstream_usage(&upstream_json),
        extract_upstream_stop_reason(api_mode, &upstream_json),
    ))
}

fn anthropic_text_response(
    requested_model: &str,
    request: &AnthropicMessagesRequest,
    text: String,
    usage: Option<AnthropicUsage>,
    stop_reason: Option<String>,
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
        content: vec![AnthropicContentResponseBlock::Text { text }],
        model: requested_model.to_string(),
        stop_reason: stop_reason.or_else(|| Some("end_turn".to_string())),
        stop_sequence: None,
        usage: usage.unwrap_or(AnthropicUsage {
            input_tokens: estimate_token_count(&input_text),
            output_tokens,
        }),
    }
}

fn extract_tool_use_response(
    api_mode: ApiMode,
    requested_model: &str,
    request: &AnthropicMessagesRequest,
    upstream_json: &Value,
) -> Option<AnthropicMessagesResponse> {
    let (text, tool_calls) = match api_mode {
        ApiMode::ChatCompletions => {
            let message = upstream_json
                .get("choices")?
                .as_array()?
                .first()?
                .get("message")?;
            let text = message.get("content").and_then(value_as_text);
            let tool_calls = message.get("tool_calls")?.as_array()?;
            (text, tool_calls)
        }
        ApiMode::Responses => (
            upstream_json
                .get("output_text")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            upstream_json.get("tool_calls")?.as_array()?,
        ),
    };

    if tool_calls.is_empty() {
        return None;
    }

    let mut content = Vec::new();
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        content.push(AnthropicContentResponseBlock::Text { text });
    }
    for call in tool_calls {
        let id = call.get("id")?.as_str()?.to_string();
        let function = call.get("function")?;
        let name = function.get("name")?.as_str()?.to_string();
        let input = function
            .get("arguments")
            .and_then(Value::as_str)
            .and_then(|args| serde_json::from_str::<Value>(args).ok())
            .unwrap_or_else(|| json!({}));
        content.push(AnthropicContentResponseBlock::ToolUse { id, name, input });
    }

    Some(AnthropicMessagesResponse {
        id: format!("msg_{}", simple_id()),
        message_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: requested_model.to_string(),
        stop_reason: Some("tool_use".to_string()),
        stop_sequence: None,
        usage: extract_upstream_usage(upstream_json).unwrap_or_else(|| AnthropicUsage {
            input_tokens: estimate_token_count(
                &request
                    .messages
                    .iter()
                    .filter_map(|message| anthropic_content_to_text(&message.content))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            output_tokens: 1,
        }),
    })
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

fn extract_upstream_usage(value: &Value) -> Option<AnthropicUsage> {
    let usage = value.get("usage")?;
    let input_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_i64)?;
    let output_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_i64)?;

    Some(AnthropicUsage {
        input_tokens,
        output_tokens,
    })
}

fn extract_upstream_stop_reason(api_mode: ApiMode, value: &Value) -> Option<String> {
    let raw = match api_mode {
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

    Some(map_stop_reason(raw))
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

fn anthropic_error_response(
    status: reqwest::StatusCode,
    content_type: &str,
    body: &[u8],
) -> Result<Response<Body>, ProxyError> {
    let message = extract_upstream_error_message(content_type, body)
        .unwrap_or_else(|| format!("upstream request failed with status {status}"));
    let payload = json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message,
        }
    });

    Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY))
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).map_err(|e| {
            ProxyError::server_error(format!("failed to encode error response: {e}"))
        })?))
        .map_err(|e| ProxyError::server_error(format!("failed to build error response: {e}")))
}

fn extract_upstream_error_message(content_type: &str, body: &[u8]) -> Option<String> {
    if content_type.starts_with("application/json") || content_type.starts_with("text/event-stream")
    {
        if let Ok(value) = serde_json::from_slice::<Value>(body) {
            if let Some(message) = value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
            {
                return Some(message.to_string());
            }
            if let Some(message) = value.get("message").and_then(Value::as_str) {
                return Some(message.to_string());
            }
        }
    }

    let preview = preview_text(&String::from_utf8_lossy(body), 400);
    if preview.is_empty() {
        None
    } else {
        Some(preview)
    }
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

        let mut parser = SseParser::default();
        let mut output_text = String::new();
        let mut stop_reason = "end_turn".to_string();
        let mut active_block: Option<StreamBlockState> = None;
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
                        if let Some(tool_delta) = extract_stream_tool_delta(api_mode, &event) {
                            let should_start = match &active_block {
                                Some(StreamBlockState::ToolUse { id, name, .. }) => {
                                    let incoming_id = if tool_delta.id.is_empty() { id } else { &tool_delta.id };
                                    let incoming_name = if tool_delta.name.is_empty() { name } else { &tool_delta.name };
                                    id != incoming_id || name != incoming_name
                                }
                                Some(StreamBlockState::Text) => true,
                                None => true,
                            };

                            if should_start {
                                if let Some(previous) = active_block.take() {
                                    yield content_block_stop_event(previous.index())?;
                                }
                                yield sse_event_bytes(
                                    "content_block_start",
                                    json!({
                                        "type": "content_block_start",
                                        "index": 0,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tool_delta.id,
                                            "name": tool_delta.name,
                                            "input": {}
                                        }
                                    }),
                                )?;
                                active_block = Some(StreamBlockState::ToolUse {
                                    index: 0,
                                    id: tool_delta.id.clone(),
                                    name: tool_delta.name.clone(),
                                });
                            } else if let Some(StreamBlockState::ToolUse { id, name, .. }) = &mut active_block {
                                if !tool_delta.id.is_empty() {
                                    *id = tool_delta.id.clone();
                                }
                                if !tool_delta.name.is_empty() {
                                    *name = tool_delta.name.clone();
                                }
                            }

                            if !tool_delta.arguments.is_empty() {
                                yield sse_event_bytes(
                                    "content_block_delta",
                                    json!({
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": tool_delta.arguments
                                        }
                                    }),
                                )?;
                            }
                        }

                        if let Some(text) = extract_stream_text(api_mode, &event) {
                            if !text.is_empty() {
                                let start_text_block = !matches!(active_block, Some(StreamBlockState::Text));
                                if start_text_block {
                                    if let Some(previous) = active_block.take() {
                                        yield content_block_stop_event(previous.index())?;
                                    }
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
                                    active_block = Some(StreamBlockState::Text);
                                }
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

        if let Some(previous) = active_block.take() {
            yield content_block_stop_event(previous.index())?;
            if matches!(previous, StreamBlockState::ToolUse { .. }) && stop_reason == "end_turn" {
                stop_reason = "tool_use".to_string();
            }
        }

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

fn content_block_stop_event(index: usize) -> Result<Bytes, std::io::Error> {
    sse_event_bytes(
        "content_block_stop",
        json!({
            "type": "content_block_stop",
            "index": index
        }),
    )
}

fn anthropic_single_response_sse(response: AnthropicMessagesResponse) -> Result<Bytes, ProxyError> {
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
    for (index, block) in response.content.iter().enumerate() {
        let block_start = match block {
            AnthropicContentResponseBlock::Text { .. } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }),
            AnthropicContentResponseBlock::ToolUse { id, name, .. } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": {}
                }
            }),
        };
        chunks.push(
            sse_event_bytes("content_block_start", block_start).map_err(|e| {
                ProxyError::server_error(format!("failed to encode sse event: {e}"))
            })?,
        );

        match block {
            AnthropicContentResponseBlock::Text { text } => {
                if !text.is_empty() {
                    chunks.push(
                        sse_event_bytes(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": text
                                }
                            }),
                        )
                        .map_err(|e| {
                            ProxyError::server_error(format!("failed to encode sse event: {e}"))
                        })?,
                    );
                }
            }
            AnthropicContentResponseBlock::ToolUse { input, .. } => {
                let partial_json = serde_json::to_string(input).map_err(|e| {
                    ProxyError::server_error(format!("failed to encode tool input: {e}"))
                })?;
                chunks.push(
                    sse_event_bytes(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": partial_json
                            }
                        }),
                    )
                    .map_err(|e| {
                        ProxyError::server_error(format!("failed to encode sse event: {e}"))
                    })?,
                );
            }
        }

        chunks.push(
            sse_event_bytes(
                "content_block_stop",
                json!({
                    "type": "content_block_stop",
                    "index": index
                }),
            )
            .map_err(|e| ProxyError::server_error(format!("failed to encode sse event: {e}")))?,
        );
    }
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

fn extract_stream_tool_delta(api_mode: ApiMode, raw: &str) -> Option<StreamToolDelta> {
    let value: Value = serde_json::from_str(raw).ok()?;
    match api_mode {
        ApiMode::ChatCompletions => {
            let call = value
                .get("choices")?
                .as_array()?
                .first()?
                .get("delta")?
                .get("tool_calls")?
                .as_array()?
                .first()?;
            let function = call.get("function")?;
            Some(StreamToolDelta {
                id: call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                arguments: function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            })
        }
        ApiMode::Responses => {
            let event_type = value.get("type")?.as_str()?;
            match event_type {
                "response.output_item.added" => {
                    let item = value.get("item")?;
                    if item.get("type")?.as_str()? != "function_call" {
                        return None;
                    }
                    Some(StreamToolDelta {
                        id: item
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        name: item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        arguments: String::new(),
                    })
                }
                "response.function_call_arguments.delta" => Some(StreamToolDelta {
                    id: value
                        .get("item_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: String::new(),
                    arguments: value
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }),
                _ => None,
            }
        }
    }
}

fn extract_responses_stream_text(value: &Value) -> Option<String> {
    let event_type = value.get("type").and_then(Value::as_str);
    match event_type {
        Some("response.output_text.delta") => value
            .get("delta")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        Some("response.output_item.added") => None,
        _ => value
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
            }),
    }
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
            .get("response")
            .and_then(|response| response.get("stop_reason"))
            .and_then(Value::as_str)
            .or_else(|| value.get("stop_reason").and_then(Value::as_str))
            .or_else(|| value.get("finish_reason").and_then(Value::as_str)),
    }?;
    Some(map_stop_reason(reason))
}

fn map_stop_reason(reason: &str) -> String {
    match reason {
        "length" | "max_tokens" => "max_tokens",
        "stop" | "end_turn" => "end_turn",
        "tool_calls" | "function_call" => "tool_use",
        other => other,
    }
    .to_string()
}

struct StreamToolDelta {
    id: String,
    name: String,
    arguments: String,
}

enum StreamBlockState {
    Text,
    ToolUse {
        index: usize,
        id: String,
        name: String,
    },
}

impl StreamBlockState {
    fn index(&self) -> usize {
        match self {
            StreamBlockState::Text => 0,
            StreamBlockState::ToolUse { index, .. } => *index,
        }
    }
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
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut preview: String = normalized.chars().take(limit).collect();
    if normalized.chars().count() > limit {
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
    tools: Vec<AnthropicTool>,
    #[serde(default)]
    tool_choice: Option<AnthropicToolChoice>,
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
struct AnthropicTool {
    name: String,
    #[serde(default)]
    description: String,
    input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto {},
    #[serde(rename = "any")]
    Any {},
    #[serde(rename = "tool")]
    Tool { name: String },
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
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(default)]
        is_error: Option<bool>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Serialize)]
struct AnthropicMessagesResponse {
    id: String,
    #[serde(rename = "type")]
    message_type: String,
    role: String,
    content: Vec<AnthropicContentResponseBlock>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
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
        "type": "error",
        "error": {
            "type": "api_error",
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
    async fn normalizes_upstream_json_error_for_anthropic() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(429)
                .header("content-type", "application/json")
                .json_body(json!({
                    "error": {
                        "message": "rate limit exceeded"
                    }
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
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json = serde_json::from_slice::<Value>(&body).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "api_error");
        assert_eq!(json["error"]["message"], "rate limit exceeded");
        upstream.assert();
    }

    #[tokio::test]
    async fn reports_connect_error_for_anthropic() {
        let app = build_router(test_config(
            "http://127.0.0.1:1".to_string(),
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
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json = serde_json::from_slice::<Value>(&body).unwrap();
        assert_eq!(json["type"], "error");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("failed to connect to upstream")
        );
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
        assert_eq!(json["stop_reason"], "end_turn");
        upstream.assert();
    }

    #[tokio::test]
    async fn preserves_usage_and_stop_reason_from_chat_completion_response() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "hi"}
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7
                    }
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
        assert_eq!(json["stop_reason"], "max_tokens");
        assert_eq!(json["usage"]["input_tokens"], 12);
        assert_eq!(json["usage"]["output_tokens"], 7);
        upstream.assert();
    }

    #[tokio::test]
    async fn forwards_anthropic_tools_to_chat_completions() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {"role": "user", "content": "hello"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "description": "Read a file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"}
                                    },
                                    "required": ["path"]
                                }
                            }
                        }
                    ],
                    "tool_choice": "required"
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
                            "messages": [{"role": "user", "content": "hello"}],
                            "tools": [
                                {
                                    "name": "Read",
                                    "description": "Read a file",
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {
                                            "path": {"type": "string"}
                                        },
                                        "required": ["path"]
                                    }
                                }
                            ],
                            "tool_choice": {"type": "any"}
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
    async fn converts_openai_tool_calls_to_anthropic_tool_use() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "Read",
                                            "arguments": "{\"path\":\"src/main.rs\"}"
                                        }
                                    }
                                ]
                            }
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
        assert_eq!(json["stop_reason"], "tool_use");
        assert_eq!(json["content"][0]["type"], "tool_use");
        assert_eq!(json["content"][0]["id"], "call_123");
        assert_eq!(json["content"][0]["name"], "Read");
        assert_eq!(json["content"][0]["input"]["path"], "src/main.rs");
        upstream.assert();
    }

    #[tokio::test]
    async fn preserves_text_alongside_tool_use_in_non_stream_response() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "choices": [
                        {
                            "message": {
                                "content": "I need to inspect a file first.",
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "Read",
                                            "arguments": "{\"path\":\"src/main.rs\"}"
                                        }
                                    }
                                ]
                            }
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
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(
            json["content"][0]["text"],
            "I need to inspect a file first."
        );
        assert_eq!(json["content"][1]["type"], "tool_use");
        assert_eq!(json["content"][1]["name"], "Read");
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
    async fn forwards_anthropic_tools_to_responses() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/responses")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "o3",
                    "input": "User: hello",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "Glob",
                                "description": "Find files",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "pattern": {"type": "string"}
                                    },
                                    "required": ["pattern"]
                                }
                            }
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {
                            "name": "Glob"
                        }
                    }
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "output": [
                        {
                            "content": [
                                {"text": "done"}
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
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-opus-4-6",
                            "messages": [{"role": "user", "content": "hello"}],
                            "tools": [
                                {
                                    "name": "Glob",
                                    "description": "Find files",
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {
                                            "pattern": {"type": "string"}
                                        },
                                        "required": ["pattern"]
                                    }
                                }
                            ],
                            "tool_choice": {"type": "tool", "name": "Glob"}
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
    async fn forwards_tool_result_to_chat_completions() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": "{\"path\":\"src/main.rs\"}"
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_123",
                            "content": "file content"
                        }
                    ]
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "choices": [
                        {"message": {"content": "done"}}
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
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "tool_use",
                                            "id": "call_123",
                                            "name": "Read",
                                            "input": {"path": "src/main.rs"}
                                        }
                                    ]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": "call_123",
                                            "content": "file content"
                                        }
                                    ]
                                }
                            ]
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
    async fn preserves_claude_code_like_multiturn_context_for_chat_completions() {
        let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
        let captured_for_route = Arc::clone(&captured);

        let upstream_app = Router::new().route(
            "/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let captured_for_route = Arc::clone(&captured_for_route);
                async move {
                    captured_for_route.lock().unwrap().push(payload);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            json!({
                                "choices": [
                                    {"message": {"content": "done"}}
                                ]
                            })
                            .to_string(),
                        ))
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
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "system": [
                                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.107"},
                                {"type": "text", "text": "You are Claude Code."}
                            ],
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": "read src/main.rs"}]
                                },
                                {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "text", "text": "I will inspect the file."},
                                        {"type": "tool_use", "id": "call_123", "name": "Read", "input": {"path": "src/main.rs"}}
                                    ]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "tool_result", "tool_use_id": "call_123", "content": "fn main() {}"},
                                        {"type": "text", "text": "continue"}
                                    ]
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let messages = requests[0]["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("x-anthropic-billing-header")
        );
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["tool_calls"][0]["function"]["name"], "Read");
        assert_eq!(messages[3]["role"], "user");
        assert_eq!(messages[4]["role"], "tool");
        assert_eq!(messages[4]["tool_call_id"], "call_123");

        upstream_handle.abort();
    }

    #[tokio::test]
    async fn ignores_unknown_anthropic_content_blocks_without_breaking_request() {
        let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
        let captured_for_route = Arc::clone(&captured);

        let upstream_app = Router::new().route(
            "/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let captured_for_route = Arc::clone(&captured_for_route);
                async move {
                    captured_for_route.lock().unwrap().push(payload);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .body(Body::from(
                            json!({
                                "choices": [
                                    {"message": {"content": "done"}}
                                ]
                            })
                            .to_string(),
                        ))
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
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-sonnet-4.6",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "hello"},
                                        {"type": "thinking", "thinking": "internal"},
                                        {"type": "tool_result", "tool_use_id": "call_123", "content": "file content"}
                                    ]
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let messages = requests[0]["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_123");

        upstream_handle.abort();
    }

    #[tokio::test]
    async fn forwards_tool_result_to_responses() {
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(POST)
                .path("/responses")
                .header("authorization", "Bearer provider-secret")
                .json_body(json!({
                    "model": "o3",
                    "input": "Assistant tool_use Read: {\"path\":\"src/main.rs\"}\n\nTool result call_123: file content"
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "output": [
                        {"content": [{"text": "done"}]}
                    ]
                }));
        });

        let app = build_router(test_config(server.base_url(), ApiMode::Responses));
        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer incoming-secret")
                    .body(Body::from(
                        json!({
                            "model": "claude-opus-4-6",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "tool_use",
                                            "id": "call_123",
                                            "name": "Read",
                                            "input": {"path": "src/main.rs"}
                                        }
                                    ]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": "call_123",
                                            "content": "file content"
                                        }
                                    ]
                                }
                            ]
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
    async fn streams_openai_tool_calls_to_anthropic_tool_use_events() {
        let upstream_app = Router::new().route(
            "/chat/completions",
            post(|Json(_payload): Json<Value>| async move {
                let stream = futures_util::stream::iter(vec![
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_123\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"path\\\":\\\"src/main.rs\\\"}\"}}]}}]}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
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
                            "messages": [{"role": "user", "content": "use a tool"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("event: content_block_start"));
        assert!(text.contains("\"type\":\"tool_use\""));
        assert!(text.contains("\"name\":\"Read\""));
        assert!(text.contains("\"type\":\"input_json_delta\""));
        assert!(text.contains("\"stop_reason\":\"tool_use\""));

        upstream_handle.abort();
    }

    #[tokio::test]
    async fn streams_openai_responses_tool_events_to_anthropic_tool_use() {
        let upstream_app = Router::new().route(
            "/responses",
            post(|Json(_payload): Json<Value>| async move {
                let stream = futures_util::stream::iter(vec![
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_123\",\"name\":\"Read\"}}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_123\",\"delta\":\"{\\\"path\\\":\\\"src/main.rs\\\"}\"}\n\n",
                    )),
                    Ok::<Bytes, Infallible>(Bytes::from(
                        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"stop_reason\":\"function_call\"}}\n\n",
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
                            "stream": true,
                            "messages": [{"role": "user", "content": "use a tool"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"type\":\"tool_use\""));
        assert!(text.contains("\"name\":\"Read\""));
        assert!(text.contains("\"type\":\"input_json_delta\""));
        assert!(text.contains("\"stop_reason\":\"tool_use\""));

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

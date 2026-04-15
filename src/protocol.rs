use axum::{
    body::Body,
    http::{Response, StatusCode},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    config::{ApiMode, ProviderConfig},
    error::ProxyError,
    logging::{append_error_log, preview_text},
};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AnthropicMessagesRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub(crate) max_tokens: Option<i64>,
    #[serde(default)]
    pub(crate) system: Option<AnthropicContent>,
    #[serde(default)]
    pub(crate) temperature: Option<f64>,
    #[serde(default)]
    pub(crate) top_p: Option<f64>,
    #[serde(default)]
    pub(crate) stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) tools: Vec<AnthropicTool>,
    #[serde(default)]
    pub(crate) tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub(crate) stream: bool,
}

impl AnthropicMessagesRequest {
    pub(crate) fn system_text(&self) -> Option<String> {
        self.system.as_ref().and_then(anthropic_content_to_text)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AnthropicMessage {
    pub(crate) role: String,
    pub(crate) content: AnthropicContent,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AnthropicTool {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) description: String,
    pub(crate) input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto {},
    #[serde(rename = "any")]
    Any {},
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicContentBlock {
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
pub(crate) struct AnthropicMessagesResponse {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) message_type: String,
    pub(crate) role: String,
    pub(crate) content: Vec<AnthropicContentResponseBlock>,
    pub(crate) model: String,
    pub(crate) stop_reason: Option<String>,
    pub(crate) stop_sequence: Option<String>,
    pub(crate) usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicContentResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct AnthropicUsage {
    pub(crate) input_tokens: i64,
    pub(crate) output_tokens: i64,
}

pub(crate) fn extract_message_preview(payload: &Value) -> String {
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

pub(crate) fn extract_anthropic_message_preview(payload: &AnthropicMessagesRequest) -> String {
    payload
        .messages
        .iter()
        .find_map(|message| anthropic_content_to_text(&message.content))
        .unwrap_or_default()
        .chars()
        .take(10)
        .collect()
}

pub(crate) fn anthropic_content_to_text(content: &AnthropicContent) -> Option<String> {
    match content {
        AnthropicContent::Text(text) => Some(text.clone()),
        AnthropicContent::Blocks(blocks) => blocks.iter().find_map(|block| match block {
            AnthropicContentBlock::Text { text } => Some(text.clone()),
            AnthropicContentBlock::ToolUse { .. } => None,
            AnthropicContentBlock::ToolResult { .. } => None,
            AnthropicContentBlock::Other => None,
        }),
    }
}

pub(crate) fn anthropic_to_provider_request(
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
    apply_tools_to_chat_completions_body(&mut body, payload);
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
    apply_tools_to_responses_body(&mut body, payload);
    Ok(body)
}

fn apply_tools_to_chat_completions_body(body: &mut Value, payload: &AnthropicMessagesRequest) {
    if !payload.tools.is_empty() {
        body["tools"] = json!(payload
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
            .collect::<Vec<_>>());
    }

    if let Some(tool_choice) = &payload.tool_choice {
        body["tool_choice"] = anthropic_tool_choice_to_chat_completions(tool_choice);
    }
}

fn apply_tools_to_responses_body(body: &mut Value, payload: &AnthropicMessagesRequest) {
    if !payload.tools.is_empty() {
        body["tools"] = json!(payload
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                })
            })
            .collect::<Vec<_>>());
    }

    if let Some(tool_choice) = &payload.tool_choice {
        body["tool_choice"] = anthropic_tool_choice_to_responses(tool_choice);
    }
}

fn anthropic_tool_choice_to_chat_completions(tool_choice: &AnthropicToolChoice) -> Value {
    match tool_choice {
        AnthropicToolChoice::Auto {} => json!("auto"),
        AnthropicToolChoice::Any {} => json!("required"),
        AnthropicToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

fn anthropic_tool_choice_to_responses(tool_choice: &AnthropicToolChoice) -> Value {
    match tool_choice {
        AnthropicToolChoice::Auto {} => json!("auto"),
        AnthropicToolChoice::Any {} => json!("required"),
        AnthropicToolChoice::Tool { name } => json!({
            "type": "function",
            "name": name,
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
            append_error_log(
                "ignoring unsupported anthropic content block",
                &format!("block_debug: {block:?}"),
            );
        }
    }
}

pub(crate) fn provider_response_to_anthropic(
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
    .ok_or_else(|| {
        append_error_log(
            "failed to extract text from upstream response",
            &format!("api_mode: {:?}\nupstream_json: {}", api_mode, upstream_json),
        );
        ProxyError::bad_gateway("failed to extract text from upstream response")
    })?;

    Ok(anthropic_text_response(
        requested_model,
        request,
        text,
        extract_upstream_usage(&upstream_json),
        extract_upstream_stop_reason(api_mode, &upstream_json),
    ))
}

pub(crate) fn anthropic_text_response(
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

pub(crate) fn anthropic_error_response(
    status: StatusCode,
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

pub(crate) fn estimate_token_count(text: &str) -> i64 {
    ((text.chars().count() as i64) / 4).max(1)
}

pub(crate) fn map_stop_reason(reason: &str) -> String {
    match reason {
        "length" | "max_tokens" => "max_tokens",
        "stop" | "end_turn" => "end_turn",
        "tool_calls" | "function_call" => "tool_use",
        other => other,
    }
    .to_string()
}

pub(crate) fn value_as_text(value: &Value) -> Option<String> {
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

fn simple_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}")
}

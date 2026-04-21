use axum::{
    body::Body,
    http::{Response, StatusCode},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    config::{ApiMode, ProviderConfig, ResponsesMetadataMode},
    error::ProxyError,
    logging::preview_text,
    normalized::{
        anthropic_response_from_normalized, normalized_messages_from_anthropic,
        normalized_messages_to_chat_completions, normalized_messages_to_responses_input,
        normalized_response_from_openai,
    },
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
    #[allow(dead_code)]
    pub(crate) betas: Vec<String>,
    #[serde(default)]
    pub(crate) metadata: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) thinking: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) context_management: Option<Value>,
    #[serde(default)]
    pub(crate) output_config: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) speed: Option<String>,
    #[serde(default)]
    pub(crate) stream: bool,
}

impl AnthropicMessagesRequest {
    pub(crate) fn system_text(&self) -> Option<String> {
        self.system.as_ref().and_then(anthropic_content_to_text)
    }

    fn metadata_object(&self) -> Option<&Map<String, Value>> {
        self.metadata.as_ref()?.as_object()
    }

    fn output_config_object(&self) -> Option<&Map<String, Value>> {
        self.output_config.as_ref()?.as_object()
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
    #[serde(default)]
    pub(crate) strict: Option<bool>,
    #[serde(default)]
    pub(crate) defer_loading: Option<bool>,
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
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "image")]
    Image {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "document")]
    Document {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "mcp_tool_use")]
    McpToolUse {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "mcp_tool_result")]
    McpToolResult {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "code_execution_tool_result")]
    CodeExecutionToolResult {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "container_upload")]
    ContainerUpload {
        #[serde(flatten)]
        data: Map<String, Value>,
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
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "image")]
    Image {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "document")]
    Document {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "mcp_tool_use")]
    McpToolUse {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "mcp_tool_result")]
    McpToolResult {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "code_execution_tool_result")]
    CodeExecutionToolResult {
        #[serde(flatten)]
        data: Map<String, Value>,
    },
    #[serde(rename = "container_upload")]
    ContainerUpload {
        #[serde(flatten)]
        data: Map<String, Value>,
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
        AnthropicContent::Blocks(blocks) => {
            let text = blocks
                .iter()
                .filter_map(|block| match block {
                    AnthropicContentBlock::Text { text } => Some(text.as_str()),
                    AnthropicContentBlock::ToolUse { .. }
                    | AnthropicContentBlock::ToolResult { .. }
                    | AnthropicContentBlock::Thinking { .. }
                    | AnthropicContentBlock::RedactedThinking { .. }
                    | AnthropicContentBlock::Image { .. }
                    | AnthropicContentBlock::Document { .. }
                    | AnthropicContentBlock::ServerToolUse { .. }
                    | AnthropicContentBlock::McpToolUse { .. }
                    | AnthropicContentBlock::McpToolResult { .. }
                    | AnthropicContentBlock::CodeExecutionToolResult { .. }
                    | AnthropicContentBlock::ContainerUpload { .. }
                    | AnthropicContentBlock::Other => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            if text.is_empty() { None } else { Some(text) }
        }
    }
}

pub(crate) fn anthropic_to_provider_request(
    payload: &AnthropicMessagesRequest,
    provider: &ProviderConfig,
    target_model: &str,
) -> Result<Value, ProxyError> {
    match provider.api_mode {
        ApiMode::ChatCompletions => Ok(anthropic_to_chat_completions(payload, target_model)),
        ApiMode::Responses => anthropic_to_responses(payload, provider, target_model),
    }
}

fn anthropic_to_chat_completions(payload: &AnthropicMessagesRequest, target_model: &str) -> Value {
    let messages =
        normalized_messages_to_chat_completions(&normalized_messages_from_anthropic(payload));

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
    apply_request_options_to_chat_completions_body(&mut body, payload);
    body
}

fn anthropic_to_responses(
    payload: &AnthropicMessagesRequest,
    provider: &ProviderConfig,
    target_model: &str,
) -> Result<Value, ProxyError> {
    let input =
        normalized_messages_to_responses_input(&normalized_messages_from_anthropic(payload));
    if input.is_empty() {
        return Err(ProxyError::bad_request(
            "anthropic messages content is empty",
        ));
    }

    let mut body = json!({
        "model": target_model,
        "input": input,
        "parallel_tool_calls": false,
        "store": false,
        "include": [],
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
    body["stream"] = json!(payload.stream);
    if let Some(system) = payload.system_text() {
        body["instructions"] = json!(system);
    }
    apply_tools_to_responses_body(&mut body, payload);
    apply_request_options_to_responses_body(&mut body, payload, provider);
    Ok(body)
}

fn apply_tools_to_chat_completions_body(body: &mut Value, payload: &AnthropicMessagesRequest) {
    if !payload.tools.is_empty() {
        body["tools"] = json!(
            payload
                .tools
                .iter()
                .map(|tool| {
                    let mut function = json!({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    });
                    if tool.strict == Some(true) {
                        function["strict"] = json!(true);
                    }
                    json!({
                        "type": "function",
                        "function": function,
                    })
                })
                .collect::<Vec<_>>()
        );
    }

    if let Some(tool_choice) = &payload.tool_choice {
        body["tool_choice"] = anthropic_tool_choice_to_chat_completions(tool_choice);
    }
}

fn apply_tools_to_responses_body(body: &mut Value, payload: &AnthropicMessagesRequest) {
    if !payload.tools.is_empty() {
        body["tools"] = json!(
            payload
                .tools
                .iter()
                .map(|tool| {
                    let mut mapped = json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "strict": tool.strict.unwrap_or(false),
                        "parameters": tool.input_schema,
                    });
                    if tool.defer_loading == Some(true) {
                        mapped["defer_loading"] = json!(true);
                    }
                    mapped
                })
                .collect::<Vec<_>>()
        );
    }

    if let Some(tool_choice) = &payload.tool_choice {
        body["tool_choice"] = anthropic_tool_choice_to_responses(tool_choice);
    } else if !payload.tools.is_empty() {
        body["tool_choice"] = json!("auto");
    }
}

fn apply_request_options_to_chat_completions_body(
    body: &mut Value,
    payload: &AnthropicMessagesRequest,
) {
    if let Some(metadata) = payload.metadata_object() {
        body["metadata"] = Value::Object(metadata.clone());
    }

    if let Some(effort) = payload
        .output_config_object()
        .and_then(|config| config.get("effort"))
        .and_then(Value::as_str)
        .and_then(openai_reasoning_effort)
    {
        body["reasoning_effort"] = json!(effort);
    }

    if let Some(format) = payload
        .output_config_object()
        .and_then(|config| config.get("format"))
        .and_then(anthropic_output_format_to_chat_completions)
    {
        body["response_format"] = format;
    }
}

fn apply_request_options_to_responses_body(
    body: &mut Value,
    payload: &AnthropicMessagesRequest,
    provider: &ProviderConfig,
) {
    if let Some(metadata) = payload.metadata_object() {
        if provider.responses_metadata_mode == ResponsesMetadataMode::ClientMetadata {
            let client_metadata = anthropic_metadata_to_client_metadata(metadata);
            if !client_metadata.is_empty() {
                body["client_metadata"] = Value::Object(client_metadata);
            }
        }
    }

    if let Some(effort) = payload
        .output_config_object()
        .and_then(|config| config.get("effort"))
        .and_then(Value::as_str)
        .and_then(openai_reasoning_effort)
    {
        body["reasoning"] = json!({ "effort": effort });
    }

    if let Some(format) = payload
        .output_config_object()
        .and_then(|config| config.get("format"))
        .and_then(anthropic_output_format_to_responses)
    {
        body["text"] = json!({ "format": format });
    }
}

fn anthropic_metadata_to_client_metadata(metadata: &Map<String, Value>) -> Map<String, Value> {
    metadata
        .iter()
        .map(|(key, value)| {
            let string_value = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            (key.clone(), Value::String(string_value))
        })
        .collect()
}

fn openai_reasoning_effort(effort: &str) -> Option<&'static str> {
    match effort {
        "minimal" => Some("minimal"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" | "max" => Some("xhigh"),
        _ => None,
    }
}

fn anthropic_output_format_to_chat_completions(format: &Value) -> Option<Value> {
    match format.get("type").and_then(Value::as_str) {
        Some("json_object") => Some(json!({ "type": "json_object" })),
        Some("json_schema") => {
            let schema = format.get("schema")?.clone();
            let mut json_schema = json!({
                "name": output_format_name(format),
                "schema": schema,
            });

            if let Some(description) = format.get("description").and_then(Value::as_str) {
                json_schema["description"] = json!(description);
            }
            if let Some(strict) = format.get("strict").and_then(Value::as_bool) {
                json_schema["strict"] = json!(strict);
            }

            Some(json!({
                "type": "json_schema",
                "json_schema": json_schema,
            }))
        }
        _ => None,
    }
}

fn anthropic_output_format_to_responses(format: &Value) -> Option<Value> {
    match format.get("type").and_then(Value::as_str) {
        Some("json_object") => Some(json!({ "type": "json_object" })),
        Some("json_schema") => {
            let schema = format.get("schema")?.clone();
            let mut mapped = json!({
                "type": "json_schema",
                "name": responses_output_format_name(format),
                "schema": schema,
                "strict": format.get("strict").and_then(Value::as_bool).unwrap_or(true),
            });

            if let Some(description) = format.get("description").and_then(Value::as_str) {
                mapped["description"] = json!(description);
            }

            Some(mapped)
        }
        _ => None,
    }
}

fn output_format_name(format: &Value) -> &str {
    format
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("claude_code_output")
}

fn responses_output_format_name(format: &Value) -> &str {
    format
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("codex_output_schema")
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

pub(crate) fn provider_response_to_anthropic(
    api_mode: ApiMode,
    requested_model: &str,
    request: &AnthropicMessagesRequest,
    upstream_json: Value,
) -> Result<AnthropicMessagesResponse, ProxyError> {
    let normalized = normalized_response_from_openai(api_mode, &upstream_json)?;
    Ok(anthropic_response_from_normalized(
        requested_model,
        request,
        normalized,
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

pub(crate) fn extract_upstream_usage(value: &Value) -> Option<AnthropicUsage> {
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

pub(crate) fn extract_upstream_stop_reason(api_mode: ApiMode, value: &Value) -> Option<String> {
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
    fn collect(value: &Value, fragments: &mut Vec<String>) {
        match value {
            Value::String(text) => {
                if !text.is_empty() {
                    fragments.push(text.clone());
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect(item, fragments);
                }
            }
            Value::Object(map) => {
                if let Some(text) = map.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        fragments.push(text.to_string());
                    }
                    return;
                }
                if let Some(content) = map.get("content") {
                    collect(content, fragments);
                }
            }
            _ => {}
        }
    }

    let mut fragments = Vec::new();
    collect(value, &mut fragments);
    if fragments.is_empty() {
        None
    } else {
        Some(fragments.join("\n"))
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

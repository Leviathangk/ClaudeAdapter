use serde_json::{json, Value};

use crate::{
    config::ApiMode,
    error::ProxyError,
    logging::append_error_log,
    protocol::{
        anthropic_content_to_text, estimate_token_count, extract_upstream_stop_reason,
        extract_upstream_usage, map_stop_reason, value_as_text, AnthropicContent,
        AnthropicContentBlock, AnthropicContentResponseBlock, AnthropicMessage,
        AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicUsage,
    },
    rules::normalize_incoming_messages,
};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum NormalizedRole {
    SystemPrompt,
    User,
    Assistant,
    Progress,
    GroupedToolUse,
    SystemEvent,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum NormalizedBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Value,
        is_error: bool,
    },
    Thinking(Value),
    RedactedThinking(Value),
    Image(Value),
    Document(Value),
    ServerToolUse(Value),
    McpToolUse(Value),
    McpToolResult(Value),
    CodeExecutionToolResult(Value),
    ContainerUpload(Value),
    Unknown(Value),
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedMessage {
    pub(crate) role: NormalizedRole,
    pub(crate) blocks: Vec<NormalizedBlock>,
    pub(crate) subtype: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedResponse {
    pub(crate) blocks: Vec<NormalizedBlock>,
    pub(crate) stop_reason: Option<String>,
    pub(crate) usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone)]
pub(crate) enum NormalizedStreamEvent {
    TextDelta(String),
    ToolUseStart { id: String, name: String },
    ToolInputDelta { partial_json: String },
    StopReason(String),
}

pub(crate) fn normalized_messages_from_anthropic(
    payload: &AnthropicMessagesRequest,
) -> Vec<NormalizedMessage> {
    let mut messages = Vec::new();
    if let Some(system) = payload.system_text() {
        messages.push(NormalizedMessage {
            role: NormalizedRole::SystemPrompt,
            blocks: vec![NormalizedBlock::Text(system)],
            subtype: None,
        });
    }

    for message in &payload.messages {
        messages.push(normalized_message_from_anthropic(message));
    }

    normalize_incoming_messages(messages)
}

fn normalized_message_from_anthropic(message: &AnthropicMessage) -> NormalizedMessage {
    let role = match message.role.as_str() {
        "assistant" => NormalizedRole::Assistant,
        _ => NormalizedRole::User,
    };

    let blocks = match &message.content {
        AnthropicContent::Text(text) => vec![NormalizedBlock::Text(text.clone())],
        AnthropicContent::Blocks(blocks) => {
            blocks.iter().map(normalized_block_from_anthropic).collect()
        }
    };

    NormalizedMessage {
        role,
        blocks,
        subtype: None,
    }
}

fn normalized_block_from_anthropic(block: &AnthropicContentBlock) -> NormalizedBlock {
    match block {
        AnthropicContentBlock::Text { text } => NormalizedBlock::Text(text.clone()),
        AnthropicContentBlock::ToolUse { id, name, input } => NormalizedBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        AnthropicContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => NormalizedBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: is_error.unwrap_or(false),
        },
        AnthropicContentBlock::Thinking { data } => {
            NormalizedBlock::Thinking(Value::Object(data.clone()))
        }
        AnthropicContentBlock::RedactedThinking { data } => {
            NormalizedBlock::RedactedThinking(Value::Object(data.clone()))
        }
        AnthropicContentBlock::Image { data } => {
            NormalizedBlock::Image(Value::Object(data.clone()))
        }
        AnthropicContentBlock::Document { data } => {
            NormalizedBlock::Document(Value::Object(data.clone()))
        }
        AnthropicContentBlock::ServerToolUse { data } => {
            NormalizedBlock::ServerToolUse(Value::Object(data.clone()))
        }
        AnthropicContentBlock::McpToolUse { data } => {
            NormalizedBlock::McpToolUse(Value::Object(data.clone()))
        }
        AnthropicContentBlock::McpToolResult { data } => {
            NormalizedBlock::McpToolResult(Value::Object(data.clone()))
        }
        AnthropicContentBlock::CodeExecutionToolResult { data } => {
            NormalizedBlock::CodeExecutionToolResult(Value::Object(data.clone()))
        }
        AnthropicContentBlock::ContainerUpload { data } => {
            NormalizedBlock::ContainerUpload(Value::Object(data.clone()))
        }
        AnthropicContentBlock::Other => {
            NormalizedBlock::Unknown(Value::String("unsupported_anthropic_block".to_string()))
        }
    }
}

pub(crate) fn normalized_messages_to_chat_completions(
    messages: &[NormalizedMessage],
) -> Vec<Value> {
    let mut result = Vec::new();

    for message in messages {
        match message.role {
            NormalizedRole::SystemPrompt => {
                let text = strip_system_reminders(&collect_text_blocks(&message.blocks));
                if !text.is_empty() {
                    result.push(json!({ "role": "system", "content": text }));
                }
            }
            NormalizedRole::User => {
                let mut pending_text = Vec::new();
                for block in &message.blocks {
                    match block {
                        NormalizedBlock::Text(text) => pending_text.push(text.clone()),
                        NormalizedBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            if !pending_text.is_empty() {
                                result.push(json!({
                                    "role": "user",
                                    "content": pending_text.join("\n"),
                                }));
                                pending_text.clear();
                            }

                            let mut tool_message = json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": anthropic_tool_result_text(content),
                            });
                            if *is_error {
                                tool_message["content"] = json!(format!(
                                    "ERROR: {}",
                                    tool_message["content"].as_str().unwrap_or_default()
                                ));
                            }
                            result.push(tool_message);
                        }
                        _ => {}
                    }
                }

                if !pending_text.is_empty() {
                    result.push(json!({
                        "role": "user",
                        "content": strip_system_reminders(&pending_text.join("\n")),
                    }));
                }
            }
            NormalizedRole::Assistant => {
                let text = strip_system_reminders(&collect_text_blocks(&message.blocks));
                let tool_calls = message
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        NormalizedBlock::ToolUse { id, name, input } => Some(json!({
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

                if !text.is_empty() || !tool_calls.is_empty() {
                    let mut assistant = json!({
                        "role": "assistant",
                        "content": text,
                    });
                    if !tool_calls.is_empty() {
                        assistant["tool_calls"] = json!(tool_calls);
                    }
                    result.push(assistant);
                }
            }
            NormalizedRole::Progress
            | NormalizedRole::GroupedToolUse
            | NormalizedRole::SystemEvent => {}
        }
    }

    result
}

pub(crate) fn normalized_messages_to_responses_input(messages: &[NormalizedMessage]) -> String {
    let mut lines = Vec::new();

    for message in messages {
        match message.role {
            NormalizedRole::SystemPrompt => {
                let text = strip_system_reminders(&collect_text_blocks(&message.blocks));
                if !text.is_empty() {
                    lines.push(format!("System: {text}"));
                }
            }
            NormalizedRole::User => {
                let text = strip_system_reminders(&collect_text_blocks(&message.blocks));
                if !text.is_empty() {
                    lines.push(format!("User: {text}"));
                }
                for block in &message.blocks {
                    if let NormalizedBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = block
                    {
                        let prefix = if *is_error {
                            "Tool error"
                        } else {
                            "Tool result"
                        };
                        lines.push(format!(
                            "{prefix} {tool_use_id}: {}",
                            anthropic_tool_result_text(content)
                        ));
                    }
                }
            }
            NormalizedRole::Assistant => {
                let text = strip_system_reminders(&collect_text_blocks(&message.blocks));
                if !text.is_empty() {
                    lines.push(format!("Assistant: {text}"));
                }
                for block in &message.blocks {
                    if let NormalizedBlock::ToolUse { name, input, .. } = block {
                        lines.push(format!("Assistant tool_use {name}: {input}"));
                    }
                }
            }
            NormalizedRole::Progress
            | NormalizedRole::GroupedToolUse
            | NormalizedRole::SystemEvent => {}
        }
    }

    lines.join("\n\n")
}

fn collect_text_blocks(blocks: &[NormalizedBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            NormalizedBlock::Text(text) => Some(text.clone()),
            _ => special_block_to_context_text(block),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn normalized_response_from_openai(
    api_mode: ApiMode,
    upstream_json: &Value,
) -> Result<NormalizedResponse, ProxyError> {
    match api_mode {
        ApiMode::ChatCompletions => normalized_response_from_chat_completions(upstream_json),
        ApiMode::Responses => normalized_response_from_responses(upstream_json),
    }
}

fn normalized_response_from_chat_completions(
    upstream_json: &Value,
) -> Result<NormalizedResponse, ProxyError> {
    let message = upstream_json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .ok_or_else(|| {
            append_error_log(
                "invalid chat.completions response",
                &format!("upstream_json: {}", upstream_json),
            );
            ProxyError::bad_gateway("missing message in chat.completions response")
        })?;

    let mut blocks = Vec::new();
    if let Some(text) = message.get("content").and_then(value_as_text) {
        if !text.is_empty() {
            blocks.push(NormalizedBlock::Text(text));
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let Some(id) = call.get("id").and_then(Value::as_str) else {
                continue;
            };
            let Some(function) = call.get("function") else {
                continue;
            };
            let Some(name) = function.get("name").and_then(Value::as_str) else {
                continue;
            };
            let input = function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|args| serde_json::from_str::<Value>(args).ok())
                .unwrap_or_else(|| json!({}));
            blocks.push(NormalizedBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input,
            });
        }
    }

    Ok(NormalizedResponse {
        blocks,
        stop_reason: extract_upstream_stop_reason(ApiMode::ChatCompletions, upstream_json),
        usage: extract_upstream_usage(upstream_json),
    })
}

fn normalized_response_from_responses(
    upstream_json: &Value,
) -> Result<NormalizedResponse, ProxyError> {
    let mut blocks = Vec::new();

    if let Some(text) = upstream_json.get("output_text").and_then(Value::as_str) {
        if !text.is_empty() {
            blocks.push(NormalizedBlock::Text(text.to_string()));
        }
    }

    if let Some(tool_calls) = upstream_json.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let Some(id) = call.get("id").and_then(Value::as_str) else {
                continue;
            };
            let Some(function) = call.get("function") else {
                continue;
            };
            let Some(name) = function.get("name").and_then(Value::as_str) else {
                continue;
            };
            let input = function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|args| serde_json::from_str::<Value>(args).ok())
                .unwrap_or_else(|| json!({}));
            blocks.push(NormalizedBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input,
            });
        }
    }

    if let Some(output) = upstream_json.get("output").and_then(Value::as_array) {
        for item in output {
            let Some(item_type) = item.get("type").and_then(Value::as_str) else {
                continue;
            };
            match item_type {
                "message" => {
                    if let Some(content) = item.get("content").and_then(Value::as_array) {
                        for part in content {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                if !text.is_empty() {
                                    blocks.push(NormalizedBlock::Text(text.to_string()));
                                }
                            }
                        }
                    }
                }
                "function_call" => {
                    let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
                    let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                    let input = item.get("arguments").cloned().unwrap_or_else(|| json!({}));
                    blocks.push(NormalizedBlock::ToolUse {
                        id: id.to_string(),
                        name: name.to_string(),
                        input: if input.is_string() {
                            input
                                .as_str()
                                .and_then(|args| serde_json::from_str::<Value>(args).ok())
                                .unwrap_or_else(|| json!({}))
                        } else {
                            input
                        },
                    });
                }
                "thinking" => blocks.push(NormalizedBlock::Thinking(item.clone())),
                "redacted_thinking" => blocks.push(NormalizedBlock::RedactedThinking(item.clone())),
                "image" => blocks.push(NormalizedBlock::Image(item.clone())),
                "document" => blocks.push(NormalizedBlock::Document(item.clone())),
                "server_tool_use" => blocks.push(NormalizedBlock::ServerToolUse(item.clone())),
                "mcp_tool_use" => blocks.push(NormalizedBlock::McpToolUse(item.clone())),
                "mcp_tool_result" => blocks.push(NormalizedBlock::McpToolResult(item.clone())),
                "code_execution_tool_result" => {
                    blocks.push(NormalizedBlock::CodeExecutionToolResult(item.clone()))
                }
                "container_upload" => blocks.push(NormalizedBlock::ContainerUpload(item.clone())),
                _ => blocks.push(NormalizedBlock::Unknown(item.clone())),
            }
        }
    }

    Ok(NormalizedResponse {
        blocks,
        stop_reason: extract_upstream_stop_reason(ApiMode::Responses, upstream_json),
        usage: extract_upstream_usage(upstream_json),
    })
}

pub(crate) fn anthropic_response_from_normalized(
    requested_model: &str,
    request: &AnthropicMessagesRequest,
    normalized: NormalizedResponse,
) -> AnthropicMessagesResponse {
    let input_text = request
        .messages
        .iter()
        .filter_map(|message| anthropic_content_to_text(&message.content))
        .collect::<Vec<_>>()
        .join("\n");

    let output_text = normalized
        .blocks
        .iter()
        .filter_map(|block| match block {
            NormalizedBlock::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    let content = normalized
        .blocks
        .into_iter()
        .filter_map(|block| match block {
            NormalizedBlock::Text(text) => Some(AnthropicContentResponseBlock::Text {
                text: strip_system_reminders(&text),
            }),
            NormalizedBlock::ToolUse { id, name, input } => {
                Some(AnthropicContentResponseBlock::ToolUse { id, name, input })
            }
            NormalizedBlock::Thinking(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::Thinking { data })
            }
            NormalizedBlock::RedactedThinking(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::RedactedThinking { data })
            }
            NormalizedBlock::Image(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::Image { data })
            }
            NormalizedBlock::Document(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::Document { data })
            }
            NormalizedBlock::ServerToolUse(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::ServerToolUse { data })
            }
            NormalizedBlock::McpToolUse(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::McpToolUse { data })
            }
            NormalizedBlock::McpToolResult(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::McpToolResult { data })
            }
            NormalizedBlock::CodeExecutionToolResult(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::CodeExecutionToolResult { data })
            }
            NormalizedBlock::ContainerUpload(Value::Object(data)) => {
                Some(AnthropicContentResponseBlock::ContainerUpload { data })
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let inferred_stop_reason = if content
        .iter()
        .any(|block| matches!(block, AnthropicContentResponseBlock::ToolUse { .. }))
    {
        Some("tool_use".to_string())
    } else {
        Some("end_turn".to_string())
    };

    AnthropicMessagesResponse {
        id: format!("msg_{}", simple_id()),
        message_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: requested_model.to_string(),
        stop_reason: normalized.stop_reason.or(inferred_stop_reason),
        stop_sequence: None,
        usage: normalized.usage.unwrap_or(AnthropicUsage {
            input_tokens: estimate_token_count(&input_text),
            output_tokens: estimate_token_count(&output_text),
        }),
    }
}

fn anthropic_tool_result_text(content: &Value) -> String {
    strip_system_reminders(&value_as_text(content).unwrap_or_else(|| content.to_string()))
}

fn special_block_to_context_text(block: &NormalizedBlock) -> Option<String> {
    match block {
        NormalizedBlock::Thinking(value) => Some(format!("[thinking] {value}")),
        NormalizedBlock::RedactedThinking(value) => Some(format!("[redacted_thinking] {value}")),
        NormalizedBlock::Image(value) => Some(format!("[image] {value}")),
        NormalizedBlock::Document(value) => Some(format!("[document] {value}")),
        NormalizedBlock::ServerToolUse(value) => Some(format!("[server_tool_use] {value}")),
        NormalizedBlock::McpToolUse(value) => Some(format!("[mcp_tool_use] {value}")),
        NormalizedBlock::McpToolResult(value) => Some(format!("[mcp_tool_result] {value}")),
        NormalizedBlock::CodeExecutionToolResult(value) => {
            Some(format!("[code_execution_tool_result] {value}"))
        }
        NormalizedBlock::ContainerUpload(value) => Some(format!("[container_upload] {value}")),
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

fn strip_system_reminders(raw: &str) -> String {
    const SYSTEM_REMINDER_OPEN: &str = "<system-reminder>";
    const SYSTEM_REMINDER_CLOSE: &str = "</system-reminder>";

    let mut text = raw.to_string();
    let mut open = text.find(SYSTEM_REMINDER_OPEN);
    while let Some(start) = open {
        let Some(rel_end) = text[start..].find(SYSTEM_REMINDER_CLOSE) else {
            break;
        };
        let end = start + rel_end + SYSTEM_REMINDER_CLOSE.len();
        text.replace_range(start..end, "");
        open = text.find(SYSTEM_REMINDER_OPEN);
    }

    text.trim().to_string()
}

pub(crate) fn normalized_stream_events_from_openai(
    api_mode: ApiMode,
    raw: &str,
) -> Vec<NormalizedStreamEvent> {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return Vec::new();
    };
    match api_mode {
        ApiMode::ChatCompletions => normalized_stream_events_from_chat_chunk(&value),
        ApiMode::Responses => normalized_stream_events_from_responses_event(&value),
    }
    .unwrap_or_default()
}

fn normalized_stream_events_from_chat_chunk(value: &Value) -> Option<Vec<NormalizedStreamEvent>> {
    let choice = value.get("choices")?.as_array()?.first()?;
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        return Some(vec![NormalizedStreamEvent::StopReason(map_stop_reason(
            reason,
        ))]);
    }

    let delta = choice.get("delta")?;
    if let Some(text) = delta.get("content").and_then(Value::as_str) {
        return Some(vec![NormalizedStreamEvent::TextDelta(text.to_string())]);
    }

    let tool_call = delta.get("tool_calls")?.as_array()?.first()?;
    let function = tool_call.get("function")?;
    let mut events = Vec::new();
    if let Some(name) = function.get("name").and_then(Value::as_str) {
        events.push(NormalizedStreamEvent::ToolUseStart {
            id: tool_call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: name.to_string(),
        });
    }
    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
        events.push(NormalizedStreamEvent::ToolInputDelta {
            partial_json: arguments.to_string(),
        });
    }

    if events.is_empty() {
        None
    } else {
        Some(events)
    }
}

fn normalized_stream_events_from_responses_event(
    value: &Value,
) -> Option<Vec<NormalizedStreamEvent>> {
    let event_type = value.get("type")?.as_str()?;
    match event_type {
        "response.output_text.delta" => Some(vec![NormalizedStreamEvent::TextDelta(
            value
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        )]),
        "response.output_item.added" => {
            let item = value.get("item")?;
            if item.get("type")?.as_str()? != "function_call" {
                return None;
            }
            Some(vec![NormalizedStreamEvent::ToolUseStart {
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
            }])
        }
        "response.function_call_arguments.delta" => {
            Some(vec![NormalizedStreamEvent::ToolInputDelta {
                partial_json: value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            }])
        }
        "response.completed" => value
            .get("response")
            .and_then(|response| response.get("stop_reason"))
            .and_then(Value::as_str)
            .map(|reason| vec![NormalizedStreamEvent::StopReason(map_stop_reason(reason))]),
        _ => None,
    }
}

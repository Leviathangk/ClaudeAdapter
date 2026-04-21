use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    config::ApiMode,
    error::ProxyError,
    logging::{append_error_log, preview_text},
    protocol::{
        AnthropicContent, AnthropicContentBlock, AnthropicContentResponseBlock, AnthropicMessage,
        AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicUsage,
        anthropic_content_to_text, estimate_token_count, extract_upstream_stop_reason,
        extract_upstream_usage, map_stop_reason, value_as_text,
    },
    rules::normalize_incoming_messages,
};

static UNHANDLED_RESPONSES_EVENT_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
const MAX_UNHANDLED_RESPONSES_EVENT_LOGS: usize = 20;

pub(crate) fn reset_responses_stream_debug_state() {
    UNHANDLED_RESPONSES_EVENT_LOG_COUNT.store(0, Ordering::Relaxed);
}

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
    TextSnapshot(String),
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolInputDelta {
        partial_json: String,
    },
    ToolUseSnapshot {
        id: String,
        name: String,
        input_json: String,
    },
    UpstreamError(String),
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
            content: sanitize_tool_result_content(content),
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
                let mut pending_parts = Vec::new();
                for block in &message.blocks {
                    match block {
                        NormalizedBlock::Text(_)
                        | NormalizedBlock::Image(_)
                        | NormalizedBlock::Document(_) => {
                            pending_parts.extend(chat_content_parts_from_block(block));
                        }
                        NormalizedBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            if let Some(content) =
                                chat_message_content_from_parts(&std::mem::take(&mut pending_parts))
                            {
                                result.push(json!({
                                    "role": "user",
                                    "content": content,
                                }));
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

                if let Some(content) = chat_message_content_from_parts(&pending_parts) {
                    result.push(json!({
                        "role": "user",
                        "content": content,
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

pub(crate) fn normalized_messages_to_responses_input(messages: &[NormalizedMessage]) -> Vec<Value> {
    let mut items = Vec::new();

    for message in messages {
        match message.role {
            NormalizedRole::SystemPrompt => {}
            NormalizedRole::User => {
                let mut pending_content = Vec::new();

                for block in &message.blocks {
                    match block {
                        NormalizedBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            if !pending_content.is_empty() {
                                items.push(responses_message(
                                    "user",
                                    std::mem::take(&mut pending_content),
                                ));
                            }
                            items.push(json!({
                                "type": "function_call_output",
                                "id": format!("fco_{}", simple_id()),
                                "call_id": tool_use_id,
                                "output": responses_output_from_tool_result(content),
                            }));
                        }
                        _ => {
                            if let Some(content_item) =
                                responses_content_item_from_block(block, false)
                            {
                                pending_content.push(content_item);
                            }
                        }
                    }
                }

                if !pending_content.is_empty() {
                    items.push(responses_message("user", pending_content));
                }
            }
            NormalizedRole::Assistant => {
                let mut pending_content = Vec::new();

                for block in &message.blocks {
                    match block {
                        NormalizedBlock::ToolUse { id, name, input } => {
                            if !pending_content.is_empty() {
                                items.push(responses_message(
                                    "assistant",
                                    std::mem::take(&mut pending_content),
                                ));
                            }
                            items.push(json!({
                                "type": "function_call",
                                "id": responses_function_call_item_id(id),
                                "call_id": id,
                                "name": name,
                                "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string()),
                            }));
                        }
                        _ => {
                            if let Some(content_item) =
                                responses_content_item_from_block(block, true)
                            {
                                pending_content.push(content_item);
                            }
                        }
                    }
                }

                if !pending_content.is_empty() {
                    items.push(responses_message("assistant", pending_content));
                }
            }
            NormalizedRole::Progress
            | NormalizedRole::GroupedToolUse
            | NormalizedRole::SystemEvent => {}
        }
    }

    items
}

fn collect_text_blocks(blocks: &[NormalizedBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            NormalizedBlock::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn sanitize_tool_result_content(content: &Value) -> Value {
    let Some(entries) = content.as_array() else {
        return content.clone();
    };

    let filtered = entries
        .iter()
        .filter(|entry| !is_tool_reference_block(entry))
        .cloned()
        .collect::<Vec<_>>();

    if filtered.len() == entries.len() {
        return content.clone();
    }

    if filtered.is_empty() {
        return json!([{
            "type": "text",
            "text": "[Tool references removed - unsupported by upstream]",
        }]);
    }

    Value::Array(filtered)
}

fn is_tool_reference_block(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("tool_reference")
}

fn chat_content_parts_from_block(block: &NormalizedBlock) -> Vec<Value> {
    match block {
        NormalizedBlock::Text(text) => chat_text_part(text).into_iter().collect(),
        NormalizedBlock::Image(value) => chat_image_part_from_value(value)
            .or_else(|| chat_attachment_fallback_part("image", value))
            .into_iter()
            .collect(),
        NormalizedBlock::Document(value) => chat_document_parts_from_value(value),
        _ => Vec::new(),
    }
}

fn chat_message_content_from_parts(parts: &[Value]) -> Option<Value> {
    if parts.is_empty() {
        return None;
    }

    let all_text = parts
        .iter()
        .all(|part| part.get("type").and_then(Value::as_str) == Some("text"));
    if all_text {
        let text = parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if text.is_empty() {
            None
        } else {
            Some(json!(text))
        }
    } else {
        Some(Value::Array(parts.to_vec()))
    }
}

fn chat_text_part(text: &str) -> Option<Value> {
    let text = strip_system_reminders(text);
    if text.is_empty() {
        None
    } else {
        Some(json!({
            "type": "text",
            "text": text,
        }))
    }
}

fn chat_image_part_from_value(value: &Value) -> Option<Value> {
    let source = value.get("source")?;
    let mut image_url = json!({});

    if let Some(detail) = value.get("detail").and_then(Value::as_str) {
        image_url["detail"] = json!(detail);
    }

    if let Some(url) = source
        .get("image_url")
        .or_else(|| source.get("url"))
        .and_then(Value::as_str)
    {
        image_url["url"] = json!(url);
        return Some(json!({
            "type": "image_url",
            "image_url": image_url,
        }));
    }

    if source.get("type").and_then(Value::as_str) == Some("base64") {
        let media_type = source
            .get("media_type")
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream");
        let data = source.get("data").and_then(Value::as_str)?;
        image_url["url"] = json!(format!("data:{media_type};base64,{data}"));
        return Some(json!({
            "type": "image_url",
            "image_url": image_url,
        }));
    }

    None
}

fn chat_document_parts_from_value(value: &Value) -> Vec<Value> {
    let Some(source) = value.get("source") else {
        return Vec::new();
    };

    if source.get("type").and_then(Value::as_str) == Some("text") {
        if let Some(text) = source.get("data").and_then(Value::as_str) {
            return chat_text_part(text).into_iter().collect();
        }
    }

    chat_attachment_fallback_part("document", value)
        .into_iter()
        .collect()
}

fn chat_attachment_fallback_part(kind: &str, value: &Value) -> Option<Value> {
    let source = value.get("source");
    let label = value
        .get("filename")
        .or_else(|| value.get("title"))
        .or_else(|| source.and_then(|source| source.get("filename")))
        .or_else(|| source.and_then(|source| source.get("file_id")))
        .or_else(|| source.and_then(|source| source.get("url")))
        .or_else(|| source.and_then(|source| source.get("file_url")))
        .and_then(Value::as_str)
        .map(|value| format!(": {value}"))
        .unwrap_or_default();

    chat_text_part(&format!("[{kind} attachment omitted{label}]"))
}

fn responses_message(role: &str, content: Vec<Value>) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": content,
    })
}

fn responses_content_item_from_block(
    block: &NormalizedBlock,
    assistant_message: bool,
) -> Option<Value> {
    match block {
        NormalizedBlock::Text(text) => {
            let text = strip_system_reminders(text);
            if text.is_empty() {
                None
            } else {
                Some(json!({
                    "type": if assistant_message { "output_text" } else { "input_text" },
                    "text": text,
                }))
            }
        }
        NormalizedBlock::Image(value) => responses_input_image_from_value(value),
        NormalizedBlock::Document(value) => responses_input_file_or_text_from_value(value),
        _ => None,
    }
}

fn responses_output_from_tool_result(content: &Value) -> Value {
    if let Some(items) = responses_output_items_from_value(content) {
        return Value::Array(items);
    }

    json!(anthropic_tool_result_text(content))
}

fn responses_output_items_from_value(content: &Value) -> Option<Vec<Value>> {
    let items = match content {
        Value::Array(entries) => entries
            .iter()
            .filter_map(responses_content_item_from_value)
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };

    if items.is_empty() { None } else { Some(items) }
}

fn responses_content_item_from_value(value: &Value) -> Option<Value> {
    match value.get("type").and_then(Value::as_str) {
        Some("text") => value
            .get("text")
            .and_then(Value::as_str)
            .map(|text| json!({"type": "input_text", "text": text})),
        Some("image") => responses_input_image_from_value(value),
        Some("document") => responses_input_file_or_text_from_value(value),
        _ => None,
    }
}

fn responses_input_image_from_value(value: &Value) -> Option<Value> {
    let source = value.get("source")?;
    let mut item = json!({ "type": "input_image" });

    if let Some(detail) = value.get("detail").and_then(Value::as_str) {
        item["detail"] = json!(detail);
    }

    if let Some(file_id) = source.get("file_id").and_then(Value::as_str) {
        item["file_id"] = json!(file_id);
        return Some(item);
    }

    if let Some(image_url) = source
        .get("image_url")
        .or_else(|| source.get("url"))
        .and_then(Value::as_str)
    {
        item["image_url"] = json!(image_url);
        return Some(item);
    }

    if source.get("type").and_then(Value::as_str) == Some("base64") {
        let media_type = source
            .get("media_type")
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream");
        let data = source.get("data").and_then(Value::as_str)?;
        item["image_url"] = json!(format!("data:{media_type};base64,{data}"));
        return Some(item);
    }

    None
}

fn responses_input_file_or_text_from_value(value: &Value) -> Option<Value> {
    let source = value.get("source")?;

    if source.get("type").and_then(Value::as_str) == Some("text") {
        let text = source.get("data").and_then(Value::as_str)?;
        if text.is_empty() {
            return None;
        }
        return Some(json!({
            "type": "input_text",
            "text": text,
        }));
    }

    let mut item = json!({ "type": "input_file" });

    if let Some(detail) = value.get("detail").and_then(Value::as_str) {
        item["detail"] = json!(detail);
    }

    if let Some(filename) = value
        .get("filename")
        .or_else(|| value.get("title"))
        .or_else(|| source.get("filename"))
        .and_then(Value::as_str)
    {
        item["filename"] = json!(filename);
    }

    if let Some(file_id) = source.get("file_id").and_then(Value::as_str) {
        item["file_id"] = json!(file_id);
        return Some(item);
    }

    if let Some(file_url) = source
        .get("file_url")
        .or_else(|| source.get("url"))
        .and_then(Value::as_str)
    {
        item["file_url"] = json!(file_url);
        return Some(item);
    }

    if let Some(file_data) = source.get("data").and_then(Value::as_str) {
        item["file_data"] = json!(file_data);
        return Some(item);
    }

    None
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
            let Some(id) = responses_function_call_id(call) else {
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
                id,
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
                    let id = responses_function_call_id(item).unwrap_or_default();
                    let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                    let input = item.get("arguments").cloned().unwrap_or_else(|| json!({}));
                    blocks.push(NormalizedBlock::ToolUse {
                        id,
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

fn responses_function_call_item_id(call_id: &str) -> String {
    let sanitized = call_id
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();

    if sanitized.is_empty() {
        format!("fc_{}", simple_id())
    } else if sanitized.starts_with("fc") {
        sanitized
    } else {
        format!("fc_{sanitized}")
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
        "response.output_text.done" => {
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if text.is_empty() {
                None
            } else {
                Some(vec![NormalizedStreamEvent::TextSnapshot(text)])
            }
        }
        "response.output_text.delta" => Some(vec![NormalizedStreamEvent::TextDelta(
            value
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        )]),
        "response.content_part.added" | "response.content_part.delta" => {
            let text = responses_output_text_from_content_part(value)?;
            if text.is_empty() {
                None
            } else {
                Some(vec![NormalizedStreamEvent::TextDelta(text)])
            }
        }
        "response.content_part.done" => {
            let text = responses_output_text_from_content_part(value)?;
            if text.is_empty() {
                None
            } else {
                Some(vec![NormalizedStreamEvent::TextSnapshot(text)])
            }
        }
        "response.output_item.added" => {
            let item = value.get("item")?;
            if item.get("type")?.as_str()? != "function_call" {
                return None;
            }
            let call_id = responses_function_call_id(item).unwrap_or_default();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let arguments_raw = item.get("arguments");
            tracing::info!(
                call_id = %call_id,
                name = %name,
                arguments_shape = responses_value_shape(arguments_raw),
                arguments_preview = %preview_text(&responses_json_like_text(arguments_raw), 180),
                "responses stream function_call added"
            );
            Some(vec![NormalizedStreamEvent::ToolUseStart {
                id: call_id,
                name,
            }])
        }
        "response.output_item.done" => {
            normalized_stream_events_from_responses_output_item(value.get("item")?)
        }
        "response.function_call_arguments.delta" => {
            let delta_text = value
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            tracing::info!(
                delta_preview = %preview_text(&delta_text, 180),
                "responses stream function_call arguments delta"
            );
            Some(vec![NormalizedStreamEvent::ToolInputDelta {
                partial_json: delta_text,
            }])
        }
        "response.function_call_arguments.done" => {
            let arguments_text = responses_json_like_text(value.get("arguments"));
            tracing::info!(
                arguments_shape = responses_value_shape(value.get("arguments")),
                arguments_preview = %preview_text(&arguments_text, 180),
                "responses stream function_call arguments done (ignored; output_item.done will finalize args)"
            );
            None
        }
        "response.failed" => Some(vec![NormalizedStreamEvent::UpstreamError(
            responses_failure_message(value),
        )]),
        "response.completed" => value
            .get("response")
            .and_then(|response| response.get("stop_reason"))
            .and_then(Value::as_str)
            .map(|reason| vec![NormalizedStreamEvent::StopReason(map_stop_reason(reason))]),
        "error" => Some(vec![NormalizedStreamEvent::UpstreamError(
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .or_else(|| value.get("message").and_then(Value::as_str))
                .unwrap_or("upstream streaming error")
                .to_string(),
        )]),
        _ => {
            if event_type.starts_with("response.") {
                let seen = UNHANDLED_RESPONSES_EVENT_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                if seen < MAX_UNHANDLED_RESPONSES_EVENT_LOGS {
                    tracing::info!(
                        event_type,
                        event_preview = %preview_text(&value.to_string(), 300),
                        "unhandled responses stream event"
                    );
                } else if seen == MAX_UNHANDLED_RESPONSES_EVENT_LOGS {
                    tracing::info!(
                        max_logs = MAX_UNHANDLED_RESPONSES_EVENT_LOGS,
                        "suppressing additional unhandled responses stream event logs"
                    );
                }
            }
            None
        }
    }
}

fn responses_output_text_from_content_part(value: &Value) -> Option<String> {
    let part = value.get("part")?;
    if part.get("type").and_then(Value::as_str) != Some("output_text") {
        return None;
    }

    Some(
        part.get("text")
            .and_then(Value::as_str)
            .or_else(|| value.get("delta").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string(),
    )
}

fn responses_failure_message(value: &Value) -> String {
    value
        .get("response")
        .and_then(|response| response.get("error"))
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .get("response")
                .and_then(|response| response.get("status"))
                .and_then(Value::as_str)
        })
        .unwrap_or("upstream request failed")
        .to_string()
}

fn normalized_stream_events_from_responses_output_item(
    item: &Value,
) -> Option<Vec<NormalizedStreamEvent>> {
    match item.get("type")?.as_str()? {
        "message" => {
            let mut snapshot = String::new();
            for part in item
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(part_text) = part.get("text").and_then(Value::as_str)
                    && !part_text.is_empty()
                {
                    snapshot.push_str(part_text);
                }
            }

            if snapshot.is_empty() {
                None
            } else {
                Some(vec![NormalizedStreamEvent::TextSnapshot(snapshot)])
            }
        }
        "function_call" => {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let id = responses_function_call_id(item).unwrap_or_default();
            let input_json = responses_json_like_text(item.get("arguments"));
            tracing::info!(
                call_id = %id,
                name = %name,
                arguments_shape = responses_value_shape(item.get("arguments")),
                arguments_preview = %preview_text(&input_json, 180),
                "responses stream function_call done"
            );

            Some(vec![NormalizedStreamEvent::ToolUseSnapshot {
                id,
                name,
                input_json,
            }])
        }
        _ => None,
    }
}

fn responses_function_call_id(item: &Value) -> Option<String> {
    item.get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .map(|id| id.to_string())
}

fn responses_value_shape(value: Option<&Value>) -> &'static str {
    match value {
        Some(Value::String(_)) => "string",
        Some(Value::Object(_)) => "object",
        Some(Value::Array(_)) => "array",
        Some(Value::Number(_)) => "number",
        Some(Value::Bool(_)) => "boolean",
        Some(Value::Null) => "null",
        None => "missing",
    }
}

fn responses_json_like_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

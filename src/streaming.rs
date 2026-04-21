use async_stream::try_stream;
use axum::body::Bytes;
use futures_util::TryStreamExt;
use serde_json::{Value, json};

use crate::{
    config::ApiMode,
    error::{ProxyError, is_context_window_error_message, normalize_upstream_error_message},
    logging::{append_error_log, preview_text},
    normalized::{NormalizedStreamEvent, normalized_stream_events_from_openai},
    protocol::{AnthropicContentResponseBlock, AnthropicMessagesResponse, estimate_token_count},
};

pub(crate) fn anthropic_sse_stream(
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
        let mut saw_text_delta = false;
        let mut stop_reason = "end_turn".to_string();
        let mut next_block_index = 0usize;
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
                        for normalized_event in normalized_stream_events_from_openai(api_mode, &event) {
                            match normalized_event {
                                NormalizedStreamEvent::ToolUseStart { id, name } => {
                                    let should_start = match &active_block {
                                        Some(StreamBlockState::ToolUse { id: current_id, name: current_name, .. }) => {
                                            current_id != &id || current_name != &name
                                        }
                                        Some(StreamBlockState::Text { .. }) => true,
                                        None => true,
                                    };

                                    if should_start {
                                        if let Some(previous) = active_block.take() {
                                            yield content_block_stop_event(previous.index())?;
                                        }
                                        let index = next_block_index;
                                        next_block_index += 1;
                                        yield sse_event_bytes(
                                            "content_block_start",
                                            json!({
                                                "type": "content_block_start",
                                                "index": index,
                                                "content_block": {
                                                    "type": "tool_use",
                                                    "id": id,
                                                    "name": name,
                                                    "input": {}
                                                }
                                            }),
                                        )?;
                                        active_block = Some(StreamBlockState::ToolUse {
                                            index,
                                            id,
                                            name,
                                            saw_input_delta: false,
                                        });
                                    }
                                }
                                NormalizedStreamEvent::ToolInputDelta { partial_json } => {
                                    if !partial_json.is_empty() {
                                        let index = match active_block.as_ref() {
                                            Some(StreamBlockState::ToolUse { index, .. }) => Some(*index),
                                            _ => None,
                                        };

                                        let Some(index) = index else {
                                            continue;
                                        };

                                        yield sse_event_bytes(
                                            "content_block_delta",
                                            json!({
                                                "type": "content_block_delta",
                                                "index": index,
                                                "delta": {
                                                    "type": "input_json_delta",
                                                    "partial_json": partial_json
                                                }
                                            }),
                                        )?;
                                        if let Some(StreamBlockState::ToolUse { saw_input_delta, .. }) =
                                            active_block.as_mut()
                                        {
                                            *saw_input_delta = true;
                                        }
                                    }
                                }
                                NormalizedStreamEvent::TextDelta(text) => {
                                    if !text.is_empty() {
                                        saw_text_delta = true;
                                        let index = match active_block.as_ref() {
                                            Some(StreamBlockState::Text { index, .. }) => *index,
                                            _ => {
                                                if let Some(previous) = active_block.take() {
                                                    yield content_block_stop_event(previous.index())?;
                                                }
                                                let index = next_block_index;
                                                next_block_index += 1;
                                                yield sse_event_bytes(
                                                    "content_block_start",
                                                    json!({
                                                        "type": "content_block_start",
                                                        "index": index,
                                                        "content_block": {
                                                            "type": "text",
                                                            "text": ""
                                                        }
                                                    }),
                                                )?;
                                                active_block = Some(StreamBlockState::Text {
                                                    index,
                                                    saw_delta: false,
                                                });
                                                index
                                            }
                                        };
                                        output_text.push_str(&text);
                                        yield sse_event_bytes(
                                            "content_block_delta",
                                            json!({
                                                "type": "content_block_delta",
                                                "index": index,
                                                "delta": {
                                                    "type": "text_delta",
                                                    "text": text
                                                }
                                            }),
                                        )?;
                                        if let Some(StreamBlockState::Text { saw_delta, .. }) =
                                            active_block.as_mut()
                                        {
                                            *saw_delta = true;
                                        }
                                    }
                                }
                                NormalizedStreamEvent::TextSnapshot(text) => {
                                    if !text.is_empty() && !saw_text_delta {
                                        let existing_text_block = match active_block.as_ref() {
                                            Some(StreamBlockState::Text { index, saw_delta }) => {
                                                Some((*index, *saw_delta))
                                            }
                                            _ => None,
                                        };

                                        let index = if let Some((_, true)) = existing_text_block {
                                            continue;
                                        } else if let Some((index, false)) = existing_text_block {
                                            index
                                        } else {
                                            if let Some(previous) = active_block.take() {
                                                yield content_block_stop_event(previous.index())?;
                                            }
                                            let index = next_block_index;
                                            next_block_index += 1;
                                            yield sse_event_bytes(
                                                "content_block_start",
                                                json!({
                                                    "type": "content_block_start",
                                                    "index": index,
                                                    "content_block": {
                                                        "type": "text",
                                                        "text": ""
                                                    }
                                                }),
                                            )?;
                                            active_block = Some(StreamBlockState::Text {
                                                index,
                                                saw_delta: false,
                                            });
                                            index
                                        };
                                        output_text.push_str(&text);
                                        yield sse_event_bytes(
                                            "content_block_delta",
                                            json!({
                                                "type": "content_block_delta",
                                                "index": index,
                                                "delta": {
                                                    "type": "text_delta",
                                                    "text": text
                                                }
                                            }),
                                        )?;
                                        if let Some(StreamBlockState::Text { saw_delta, .. }) =
                                            active_block.as_mut()
                                        {
                                            *saw_delta = true;
                                        }
                                    }
                                }
                                NormalizedStreamEvent::ToolUseSnapshot { id, name, input_json } => {
                                    let existing_tool = match active_block.as_ref() {
                                        Some(StreamBlockState::ToolUse {
                                            index,
                                            id: current_id,
                                            name: current_name,
                                            saw_input_delta,
                                        }) if current_id == &id && current_name == &name => {
                                            Some((*index, *saw_input_delta))
                                        }
                                        _ => None,
                                    };

                                    let (index, should_emit_input) =
                                        if let Some((index, saw_input_delta)) = existing_tool {
                                            (index, !saw_input_delta)
                                        } else {
                                            if let Some(previous) = active_block.take() {
                                                yield content_block_stop_event(previous.index())?;
                                            }
                                            let new_index = next_block_index;
                                            yield sse_event_bytes(
                                                "content_block_start",
                                                json!({
                                                    "type": "content_block_start",
                                                    "index": new_index,
                                                    "content_block": {
                                                        "type": "tool_use",
                                                        "id": id,
                                                        "name": name,
                                                        "input": {}
                                                    }
                                                }),
                                            )?;
                                            tracing::info!(
                                                index = new_index,
                                                tool_name = %name,
                                                call_id = %id,
                                                input_preview = %preview_text(&input_json, 200),
                                                "anthropic stream finalizing completed tool call"
                                            );
                                            active_block = Some(StreamBlockState::ToolUse {
                                                index: new_index,
                                                id: id.clone(),
                                                name: name.clone(),
                                                saw_input_delta: false,
                                            });
                                            (new_index, true)
                                        };

                                    if should_emit_input && !input_json.is_empty() {
                                        yield sse_event_bytes(
                                            "content_block_delta",
                                            json!({
                                                "type": "content_block_delta",
                                                "index": index,
                                                "delta": {
                                                    "type": "input_json_delta",
                                                    "partial_json": input_json
                                                }
                                            }),
                                        )?;
                                    }
                                    if let Some(StreamBlockState::ToolUse { saw_input_delta, .. }) =
                                        active_block.as_mut()
                                    {
                                        *saw_input_delta = true;
                                    }

                                    if existing_tool.is_some() {
                                        tracing::info!(
                                            index,
                                            tool_name = %name,
                                            call_id = %id,
                                            input_preview = %preview_text(&input_json, 200),
                                            "anthropic stream finalizing completed tool call"
                                        );
                                    }
                                    if let Some(previous) = active_block.take() {
                                        yield content_block_stop_event(previous.index())?;
                                    }
                                    stop_reason = "tool_use".to_string();
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
                                    return;
                                }
                                NormalizedStreamEvent::UpstreamError(message) => {
                                    let normalized_message = normalize_upstream_error_message(&message);
                                    tracing::warn!(message = %normalized_message, "anthropic upstream stream failed");
                                    if let Some(previous) = active_block.take() {
                                        yield content_block_stop_event(previous.index())?;
                                    }
                                    yield sse_event_bytes(
                                        "error",
                                        json!({
                                            "type": "error",
                                            "error": {
                                                "type": "api_error",
                                                "message": normalized_message
                                            }
                                        }),
                                    )?;
                                    return;
                                }
                                NormalizedStreamEvent::StopReason(reason) => {
                                    stop_reason = reason;
                                    if stop_reason == "tool_use"
                                        && matches!(active_block, Some(StreamBlockState::ToolUse { .. }))
                                    {
                                        tracing::info!("anthropic stream finalizing on upstream tool_use stop_reason");
                                        if let Some(previous) = active_block.take() {
                                            yield content_block_stop_event(previous.index())?;
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
                                        return;
                                    }
                                }
                            }
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

pub(crate) fn anthropic_single_response_sse(
    response: AnthropicMessagesResponse,
) -> Result<Bytes, ProxyError> {
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
            AnthropicContentResponseBlock::Thinking { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("thinking", data)
            }),
            AnthropicContentResponseBlock::RedactedThinking { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("redacted_thinking", data)
            }),
            AnthropicContentResponseBlock::Image { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("image", data)
            }),
            AnthropicContentResponseBlock::Document { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("document", data)
            }),
            AnthropicContentResponseBlock::ServerToolUse { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("server_tool_use", data)
            }),
            AnthropicContentResponseBlock::McpToolUse { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("mcp_tool_use", data)
            }),
            AnthropicContentResponseBlock::McpToolResult { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("mcp_tool_result", data)
            }),
            AnthropicContentResponseBlock::CodeExecutionToolResult { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("code_execution_tool_result", data)
            }),
            AnthropicContentResponseBlock::ContainerUpload { data } => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_with_type("container_upload", data)
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
            AnthropicContentResponseBlock::ServerToolUse { .. }
            | AnthropicContentResponseBlock::Thinking { .. }
            | AnthropicContentResponseBlock::RedactedThinking { .. }
            | AnthropicContentResponseBlock::Image { .. }
            | AnthropicContentResponseBlock::Document { .. }
            | AnthropicContentResponseBlock::McpToolUse { .. }
            | AnthropicContentResponseBlock::McpToolResult { .. }
            | AnthropicContentResponseBlock::CodeExecutionToolResult { .. }
            | AnthropicContentResponseBlock::ContainerUpload { .. } => {}
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

fn block_with_type(block_type: &str, data: &serde_json::Map<String, Value>) -> Value {
    let mut object = data.clone();
    object.insert("type".to_string(), Value::String(block_type.to_string()));
    Value::Object(object)
}

pub(crate) fn collect_text_from_sse(api_mode: ApiMode, body: &[u8]) -> Result<String, ProxyError> {
    let text = String::from_utf8_lossy(body);
    let mut parser = SseParser::default();
    let events = parser.push(&text);
    let mut output = String::new();
    let mut saw_text_delta = false;

    for event in events {
        if event == "[DONE]" {
            continue;
        }
        for normalized_event in normalized_stream_events_from_openai(api_mode, &event) {
            match normalized_event {
                NormalizedStreamEvent::TextDelta(delta) => {
                    saw_text_delta = true;
                    output.push_str(&delta);
                }
                NormalizedStreamEvent::TextSnapshot(snapshot) => {
                    if !saw_text_delta {
                        output.push_str(&snapshot);
                    }
                }
                NormalizedStreamEvent::UpstreamError(message) => {
                    return Err(classify_upstream_stream_error(&message));
                }
                _ => {}
            }
        }
    }

    if output.is_empty() {
        append_error_log(
            "failed to extract text from upstream sse response",
            &format!(
                "api_mode: {:?}\nsse_preview: {}",
                api_mode,
                preview_text(&text, 400)
            ),
        );
        return Err(ProxyError::bad_gateway(
            "failed to extract text from upstream sse response",
        ));
    }

    Ok(output)
}

fn classify_upstream_stream_error(message: &str) -> ProxyError {
    let normalized = normalize_upstream_error_message(message);
    if is_context_window_error_message(message) {
        return ProxyError::bad_request(format!("upstream stream failed: {normalized}"));
    }

    ProxyError::bad_gateway(format!("upstream stream failed: {normalized}"))
}

fn io_error<E: std::fmt::Display>(error: E) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

enum StreamBlockState {
    Text {
        index: usize,
        saw_delta: bool,
    },
    ToolUse {
        index: usize,
        id: String,
        name: String,
        saw_input_delta: bool,
    },
}

impl StreamBlockState {
    fn index(&self) -> usize {
        match self {
            StreamBlockState::Text { index, .. } => *index,
            StreamBlockState::ToolUse { index, .. } => *index,
        }
    }
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

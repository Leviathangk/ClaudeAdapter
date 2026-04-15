use async_stream::try_stream;
use axum::body::Bytes;
use futures_util::TryStreamExt;
use serde_json::{json, Value};

use crate::{
    config::ApiMode,
    error::ProxyError,
    logging::{append_error_log, preview_text},
    protocol::{
        estimate_token_count, map_stop_reason, AnthropicContentResponseBlock,
        AnthropicMessagesResponse,
    },
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

pub(crate) fn collect_text_from_sse(api_mode: ApiMode, body: &[u8]) -> Result<String, ProxyError> {
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

fn io_error<E: std::fmt::Display>(error: E) -> std::io::Error {
    std::io::Error::other(error.to_string())
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

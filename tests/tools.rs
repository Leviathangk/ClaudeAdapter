mod common;

use axum::{body::{Body, to_bytes}, http::{Request, StatusCode}};
use claude_adapter::{ApiMode, build_router};
use httpmock::{Method::POST, MockServer};
use serde_json::{Value, json};
use tower::ServiceExt;

use common::test_config;

#[tokio::test]
async fn forwards_anthropic_tools_to_chat_completions() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST)
            .path("/chat/completions")
            .json_body(json!({
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
                "tool_choice": "required"
            }));
        then.status(200).header("content-type", "application/json").json_body(json!({"choices": [{"message": {"content": "hi"}}]}));
    });

    let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
    let response = app
        .oneshot(Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-sonnet-4.6",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
                "tool_choice": {"type": "any"}
            }).to_string())).unwrap())
        .await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    upstream.assert();
}

#[tokio::test]
async fn forwards_anthropic_tools_to_responses_in_responses_format() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST)
            .path("/responses")
            .json_body(json!({
                "model": "o3",
                "input": "User: hello",
                "tools": [{
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }],
                "tool_choice": {"type": "function", "name": "Read"}
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"output": [{"content": [{"text": "done"}]}]}));
    });

    let app = build_router(test_config(server.base_url(), ApiMode::Responses));
    let response = app
        .oneshot(Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
                "tool_choice": {"type": "tool", "name": "Read"}
            }).to_string())).unwrap())
        .await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    upstream.assert();
}

#[tokio::test]
async fn converts_openai_tool_calls_to_anthropic_tool_use() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/chat/completions");
        then.status(200).header("content-type", "application/json").json_body(json!({
            "choices": [{"message": {"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "Read", "arguments": "{\"path\":\"src/main.rs\"}"}}]}}]
        }));
    });

    let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
    let response = app
        .oneshot(Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({"model": "claude-sonnet-4.6", "messages": [{"role": "user", "content": "hello"}]}).to_string())).unwrap())
        .await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice::<Value>(&body).unwrap();
    assert_eq!(json["stop_reason"], "tool_use");
    assert_eq!(json["content"][0]["type"], "tool_use");
    upstream.assert();
}

#[tokio::test]
async fn forwards_tool_result_to_chat_completions() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST)
            .path("/chat/completions")
            .json_body(json!({
                "model": "gpt-4.1-mini",
                "messages": [
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "Read", "arguments": "{\"path\":\"src/main.rs\"}"}}]},
                    {"role": "tool", "tool_call_id": "call_123", "content": "file content"}
                ]
            }));
        then.status(200).header("content-type", "application/json").json_body(json!({"choices": [{"message": {"content": "done"}}]}));
    });

    let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
    let response = app
        .oneshot(Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-sonnet-4.6",
                "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "call_123", "name": "Read", "input": {"path": "src/main.rs"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_123", "content": "file content"}]}
                ]
            }).to_string())).unwrap())
        .await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    upstream.assert();
}

#[tokio::test]
async fn preserves_claude_code_like_multiturn_context_for_chat_completions() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/chat/completions");
        then.status(200).header("content-type", "application/json").json_body(json!({"choices": [{"message": {"content": "done"}}]}));
    });

    let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
    let response = app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-sonnet-4.6",
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.107"},
                    {"type": "text", "text": "You are Claude Code."}
                ],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "read src/main.rs"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "I will inspect the file."}, {"type": "tool_use", "id": "call_123", "name": "Read", "input": {"path": "src/main.rs"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_123", "content": "fn main() {}"}, {"type": "text", "text": "continue"}]}
                ]
            }).to_string())).unwrap()
    ).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    upstream.assert();
}

mod common;

use std::sync::{Arc, Mutex};

use axum::{Router, extract::Json, http::Response, routing::post};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
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
        when.method(POST).path("/responses").json_body(json!({
            "model": "o3",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "hello"
                }]
            }],
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
async fn forwards_claude_code_multiturn_context_to_responses_as_structured_items() {
    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured_for_route = Arc::clone(&captured);

    let upstream_app = Router::new().route(
        "/responses",
        post(move |Json(payload): Json<Value>| {
            let captured_for_route = Arc::clone(&captured_for_route);
            async move {
                captured_for_route.lock().unwrap().push(payload);
                Response::builder()
                    .status(StatusCode::OK)
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"output": [{"content": [{"text": "done"}]}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream_app).await.unwrap();
    });

    let app = build_router(test_config(format!("http://{address}"), ApiMode::Responses));
    let response = app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-opus-4-6",
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.107"},
                    {"type": "text", "text": "You are Claude Code."}
                ],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "read src/main.rs"}]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "I will inspect the file."},
                        {"type": "tool_use", "id": "call_123", "name": "Read", "input": {"path": "src/main.rs"}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "call_123", "content": [
                            {"type": "text", "text": "fn main() {}"}
                        ]},
                        {"type": "text", "text": "continue"}
                    ]}
                ]
            }).to_string())).unwrap()
    ).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    let input = requests[0]["input"].as_array().unwrap();

    assert_eq!(input[0]["role"], "system");
    assert_eq!(
        input[0]["content"][0]["text"],
        "x-anthropic-billing-header: cc_version=2.1.107\nYou are Claude Code."
    );

    assert_eq!(input[1]["role"], "user");
    assert_eq!(input[1]["content"][0]["type"], "input_text");
    assert_eq!(input[1]["content"][0]["text"], "read src/main.rs");

    assert_eq!(input[2]["role"], "assistant");
    assert_eq!(input[2]["content"][0]["type"], "output_text");
    assert_eq!(input[2]["content"][0]["text"], "I will inspect the file.");

    assert_eq!(input[3]["type"], "function_call");
    assert_eq!(input[3]["call_id"], "call_123");
    assert_eq!(input[3]["name"], "Read");
    assert_eq!(input[3]["arguments"], "{\"path\":\"src/main.rs\"}");

    assert_eq!(input[4]["type"], "function_call_output");
    assert_eq!(input[4]["call_id"], "call_123");
    assert_eq!(input[4]["output"][0]["type"], "input_text");
    assert_eq!(input[4]["output"][0]["text"], "fn main() {}");

    assert_eq!(input[5]["role"], "user");
    assert_eq!(input[5]["content"][0]["text"], "continue");

    handle.abort();
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
async fn prefers_call_id_from_responses_function_calls() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/responses");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "output": [{
                    "type": "function_call",
                    "id": "fc_internal_123",
                    "call_id": "call_123",
                    "name": "Read",
                    "arguments": "{\"path\":\"src/main.rs\"}"
                }]
            }));
    });

    let app = build_router(test_config(server.base_url(), ApiMode::Responses));
    let response = app
        .oneshot(Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hello"}]}).to_string())).unwrap())
        .await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice::<Value>(&body).unwrap();
    assert_eq!(json["content"][0]["type"], "tool_use");
    assert_eq!(json["content"][0]["id"], "call_123");
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
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"choices": [{"message": {"content": "done"}}]}));
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

#[tokio::test]
async fn hoists_tool_result_before_text_in_user_message_for_chat_completions() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream_app).await.unwrap();
    });

    let app = build_router(test_config(
        format!("http://{address}"),
        ApiMode::ChatCompletions,
    ));
    let response = app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-sonnet-4.6",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "continue"},
                        {"type": "tool_result", "tool_use_id": "call_123", "content": "file content"}
                    ]
                }]
            }).to_string())).unwrap()
    ).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "tool");
    assert_eq!(messages[0]["tool_call_id"], "call_123");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "continue");
    handle.abort();
}

#[tokio::test]
async fn merges_adjacent_user_messages_before_chat_completion_request() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
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
                            {"role": "user", "content": "hello"},
                            {"role": "user", "content": "world"}
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
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "hello\nworld");
    handle.abort();
}

#[tokio::test]
async fn merges_adjacent_assistant_messages_before_chat_completion_request() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream_app).await.unwrap();
    });

    let app = build_router(test_config(
        format!("http://{address}"),
        ApiMode::ChatCompletions,
    ));
    let response = app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-sonnet-4.6",
                "messages": [
                    {"role": "assistant", "content": "first"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "call_123", "name": "Read", "input": {"path": "src/main.rs"}}]}
                ]
            }).to_string())).unwrap()
    ).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"], "first");
    assert_eq!(messages[0]["tool_calls"][0]["function"]["name"], "Read");
    handle.abort();
}

#[tokio::test]
async fn normalizes_error_tool_result_content_to_text() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
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
                        "messages": [{
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": "call_123",
                                "is_error": true,
                                "content": [
                                    {"type": "text", "text": "permission denied"},
                                    {"type": "image", "source": {"type": "base64", "data": "abc"}}
                                ]
                            }]
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "tool");
    assert!(
        messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("permission denied")
    );
    handle.abort();
}

#[tokio::test]
async fn concatenates_multi_text_tool_result_content_for_chat_completions() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream_app).await.unwrap();
    });

    let app = build_router(test_config(
        format!("http://{address}"),
        ApiMode::ChatCompletions,
    ));
    let response = app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-sonnet-4.6",
                "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "call_123", "name": "Read", "input": {"path": "src/main.rs"}}]},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": [
                            {"type": "text", "text": "line 1"},
                            {"type": "text", "text": "line 2"}
                        ]
                    }]}
                ]
            }).to_string())).unwrap()
    ).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["content"], "line 1\nline 2");
    handle.abort();
}

#[tokio::test]
async fn forwards_multimodal_user_content_to_chat_completions() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
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
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "inspect these attachments"},
                                {"type": "image", "detail": "high", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "document body"}}
                            ]
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    let content = requests[0]["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "inspect these attachments");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,abc");
    assert_eq!(content[1]["image_url"]["detail"], "high");
    assert_eq!(content[2]["type"], "text");
    assert_eq!(content[2]["text"], "document body");
    handle.abort();
}

#[tokio::test]
async fn strips_tool_reference_blocks_from_chat_completions_tool_results() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
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
                            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_123", "name": "ToolSearch", "input": {"query": "glob"}}]},
                            {"role": "user", "content": [{
                                "type": "tool_result",
                                "tool_use_id": "call_123",
                                "content": [
                                    {"type": "tool_reference", "tool_name": "Glob"},
                                    {"type": "text", "text": "tool loaded"}
                                ]
                            }]}
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
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["content"], "tool loaded");
    handle.abort();
}

#[tokio::test]
async fn does_not_leak_thinking_blocks_into_chat_completions_messages() {
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
                        json!({"choices": [{"message": {"content": "done"}}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream_app).await.unwrap();
    });

    let app = build_router(test_config(
        format!("http://{address}"),
        ApiMode::ChatCompletions,
    ));
    let response = app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-sonnet-4.6",
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "internal plan", "signature": "sig_1"},
                        {"type": "text", "text": "visible answer"}
                    ]
                }]
            }).to_string())).unwrap()
    ).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    let messages = requests[0]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"], "visible answer");
    handle.abort();
}

#[tokio::test]
async fn does_not_leak_thinking_blocks_into_responses_input() {
    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured_for_route = Arc::clone(&captured);

    let upstream_app = Router::new().route(
        "/responses",
        post(move |Json(payload): Json<Value>| {
            let captured_for_route = Arc::clone(&captured_for_route);
            async move {
                captured_for_route.lock().unwrap().push(payload);
                Response::builder()
                    .status(StatusCode::OK)
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"output": [{"content": [{"text": "done"}]}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream_app).await.unwrap();
    });

    let app = build_router(test_config(format!("http://{address}"), ApiMode::Responses));
    let response = app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer incoming-secret")
            .body(Body::from(json!({
                "model": "claude-opus-4-6",
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "internal plan", "signature": "sig_1"},
                        {"type": "text", "text": "visible answer"}
                    ]
                }]
            }).to_string())).unwrap()
    ).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    assert_eq!(requests[0]["input"][0]["role"], "assistant");
    assert_eq!(requests[0]["input"][0]["content"][0]["type"], "output_text");
    assert_eq!(
        requests[0]["input"][0]["content"][0]["text"],
        "visible answer"
    );
    handle.abort();
}

#[tokio::test]
async fn strips_tool_reference_blocks_from_responses_tool_outputs() {
    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured_for_route = Arc::clone(&captured);

    let upstream_app = Router::new().route(
        "/responses",
        post(move |Json(payload): Json<Value>| {
            let captured_for_route = Arc::clone(&captured_for_route);
            async move {
                captured_for_route.lock().unwrap().push(payload);
                Response::builder()
                    .status(StatusCode::OK)
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"output": [{"content": [{"text": "done"}]}]}).to_string(),
                    ))
                    .unwrap()
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
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
                        "messages": [
                            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_123", "name": "ToolSearch", "input": {"query": "glob"}}]},
                            {"role": "user", "content": [{
                                "type": "tool_result",
                                "tool_use_id": "call_123",
                                "content": [
                                    {"type": "tool_reference", "tool_name": "Glob"}
                                ]
                            }]}
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
    assert_eq!(requests[0]["input"][0]["type"], "function_call");
    assert_eq!(requests[0]["input"][1]["type"], "function_call_output");
    assert_eq!(
        requests[0]["input"][1]["output"][0]["text"],
        "[Tool references removed - unsupported by upstream]"
    );
    handle.abort();
}

#[tokio::test]
async fn preserves_server_tool_use_block_in_anthropic_response() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/responses");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "output": [
                    {
                        "type": "server_tool_use",
                        "id": "srv_123",
                        "name": "web_search",
                        "input": {"query": "rust"}
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
    assert_eq!(json["content"][0]["type"], "server_tool_use");
    assert_eq!(json["content"][0]["name"], "web_search");
    upstream.assert();
}

#[tokio::test]
async fn preserves_mcp_tool_blocks_in_anthropic_response() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/responses");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "output": [
                    {
                        "type": "mcp_tool_use",
                        "id": "mcp_use_1",
                        "name": "browser.open",
                        "server_name": "playwright",
                        "input": {"url": "https://example.com"}
                    },
                    {
                        "type": "mcp_tool_result",
                        "tool_use_id": "mcp_use_1",
                        "content": [{"type": "text", "text": "opened"}]
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
    assert_eq!(json["content"][0]["type"], "mcp_tool_use");
    assert_eq!(json["content"][0]["name"], "browser.open");
    assert_eq!(json["content"][1]["type"], "mcp_tool_result");
    assert_eq!(json["content"][1]["tool_use_id"], "mcp_use_1");
    upstream.assert();
}

#[tokio::test]
async fn preserves_code_execution_and_container_upload_blocks_in_anthropic_response() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/responses");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "output": [
                    {
                        "type": "code_execution_tool_result",
                        "tool_use_id": "code_1",
                        "content": [{"type": "text", "text": "exit code 0"}]
                    },
                    {
                        "type": "container_upload",
                        "container_id": "ctr_1",
                        "path": "/workspace/result.txt"
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
    assert_eq!(json["content"][0]["type"], "code_execution_tool_result");
    assert_eq!(json["content"][0]["tool_use_id"], "code_1");
    assert_eq!(json["content"][1]["type"], "container_upload");
    assert_eq!(json["content"][1]["container_id"], "ctr_1");
    upstream.assert();
}

#[tokio::test]
async fn preserves_thinking_and_redacted_thinking_blocks_in_anthropic_response() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/responses");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "output": [
                    {
                        "type": "thinking",
                        "thinking": "plan first",
                        "signature": "sig_1"
                    },
                    {
                        "type": "redacted_thinking",
                        "data": "opaque"
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
    assert_eq!(json["content"][0]["type"], "thinking");
    assert_eq!(json["content"][0]["thinking"], "plan first");
    assert_eq!(json["content"][1]["type"], "redacted_thinking");
    upstream.assert();
}

#[tokio::test]
async fn preserves_image_and_document_blocks_in_anthropic_response() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/responses");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "output": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc"}
                    },
                    {
                        "type": "document",
                        "source": {"type": "text", "media_type": "text/plain", "data": "hello"}
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
    assert_eq!(json["content"][0]["type"], "image");
    assert_eq!(json["content"][1]["type"], "document");
    upstream.assert();
}

#[tokio::test]
async fn strips_system_reminder_from_visible_text_output() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [{
                    "message": {
                        "content": "hello <system-reminder>internal note</system-reminder> world"
                    }
                }]
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
    assert_eq!(json["content"][0]["text"], "hello  world");
    upstream.assert();
}

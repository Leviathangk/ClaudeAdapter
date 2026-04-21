mod common;

use axum::{Router, extract::Json, http::Response, routing::post};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use claude_adapter::{ApiMode, build_router};
use httpmock::{Method::POST, MockServer};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

use common::test_config;

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
                "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]
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
    assert_eq!(json["content"][0]["text"], "hi");
    assert_eq!(json["stop_reason"], "end_turn");
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
                    json!({"model": "claude-sonnet-4.6", "messages": []}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.headers().get("x-should-retry").unwrap(), "false");
}

#[tokio::test]
async fn rejects_anthropic_messages_before_parsing_body_when_token_is_invalid() {
    let app = build_router(test_config(
        "http://127.0.0.1:1".to_string(),
        ApiMode::ChatCompletions,
    ));
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer wrong-token")
                .body(Body::from("{not-json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.headers().get("x-should-retry").unwrap(), "false");
}

#[tokio::test]
async fn normalizes_upstream_json_error_for_anthropic() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/chat/completions");
        then.status(429)
            .header("content-type", "application/json")
            .json_body(json!({"error": {"message": "rate limit exceeded"}}));
    });

    let app = build_router(test_config(server.base_url(), ApiMode::ChatCompletions));
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer incoming-secret")
                .body(Body::from(json!({"model": "claude-sonnet-4.6", "messages": [{"role": "user", "content": "hello"}]}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice::<Value>(&body).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["message"], "rate limit exceeded");
    upstream.assert();
}

#[tokio::test]
async fn normalizes_context_window_json_error_to_prompt_too_long() {
    let server = MockServer::start();
    let upstream = server.mock(|when, then| {
        when.method(POST).path("/responses");
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
                "error": {
                    "code": "context_length_exceeded",
                    "message": "Your input exceeds the context window of this model. Please adjust your input and try again."
                }
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice::<Value>(&body).unwrap();
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(message.starts_with("Prompt is too long:"));
    assert!(message.contains("context window"));
    upstream.assert();
}

#[tokio::test]
async fn normalizes_context_window_sse_failure_to_prompt_too_long() {
    let upstream_app = Router::new().route(
        "/responses",
        post(|Json(_payload): Json<Value>| async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(
                    "event: response.failed\n\
                     data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_123\",\"status\":\"failed\",\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"Your input exceeds the context window of this model. Please adjust your input and try again.\"}}}\n\n",
                ))
                .unwrap()
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
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice::<Value>(&body).unwrap();
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("Prompt is too long"));
    assert!(message.contains("context window"));

    handle.abort();
}

#[tokio::test]
async fn concatenates_all_system_text_blocks_before_chat_completion_request() {
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
                        "system": [
                            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.107"},
                            {"type": "text", "text": "You are Claude Code."}
                        ],
                        "messages": [{"role": "user", "content": "hello"}]
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
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(
        messages[0]["content"],
        "x-anthropic-billing-header: cc_version=2.1.107\nYou are Claude Code."
    );
    handle.abort();
}

#[tokio::test]
async fn maps_claude_code_output_config_to_chat_completions_request() {
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
                        "messages": [{"role": "user", "content": "hello"}],
                        "metadata": {"user_id": "session_1"},
                        "output_config": {
                            "effort": "max",
                            "format": {
                                "type": "json_schema",
                                "schema": {
                                    "type": "object",
                                    "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"]
                                }
                            }
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    assert_eq!(requests[0]["metadata"]["user_id"], "session_1");
    assert_eq!(requests[0]["reasoning_effort"], "xhigh");
    assert_eq!(requests[0]["response_format"]["type"], "json_schema");
    assert_eq!(
        requests[0]["response_format"]["json_schema"]["name"],
        "claude_code_output"
    );
    assert_eq!(
        requests[0]["response_format"]["json_schema"]["schema"]["required"][0],
        "ok"
    );
    handle.abort();
}

#[tokio::test]
async fn maps_claude_code_output_config_to_responses_request() {
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
                        "messages": [{"role": "user", "content": "hello"}],
                        "metadata": {"user_id": "session_1"},
                        "output_config": {
                            "effort": "max",
                            "format": {
                                "type": "json_schema",
                                "schema": {
                                    "type": "object",
                                    "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"]
                                }
                            }
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    assert_eq!(requests[0]["client_metadata"]["user_id"], "session_1");
    assert_eq!(requests[0]["stream"], false);
    assert_eq!(requests[0]["parallel_tool_calls"], false);
    assert_eq!(requests[0]["store"], false);
    assert_eq!(requests[0]["include"], json!([]));
    assert_eq!(requests[0]["reasoning"]["effort"], "xhigh");
    assert_eq!(requests[0]["text"]["format"]["type"], "json_schema");
    assert_eq!(requests[0]["text"]["format"]["name"], "codex_output_schema");
    assert_eq!(requests[0]["text"]["format"]["strict"], true);
    assert_eq!(requests[0]["text"]["format"]["schema"]["required"][0], "ok");
    assert!(requests[0].get("metadata").is_none());
    handle.abort();
}

#[tokio::test]
async fn maps_claude_code_metadata_to_responses_client_metadata_for_codex_provider() {
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
                        "messages": [{"role": "user", "content": "hello"}],
                        "metadata": {
                            "user_id": "session_1",
                            "flags": ["a", "b"],
                            "tracking": {"origin": "claude-code"},
                            "enabled": true
                        }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = captured.lock().unwrap();
    assert_eq!(requests[0]["client_metadata"]["user_id"], "session_1");
    assert_eq!(requests[0]["client_metadata"]["flags"], "[\"a\",\"b\"]");
    assert_eq!(
        requests[0]["client_metadata"]["tracking"],
        "{\"origin\":\"claude-code\"}"
    );
    assert_eq!(requests[0]["client_metadata"]["enabled"], "true");
    assert!(requests[0].get("metadata").is_none());
    handle.abort();
}

#[tokio::test]
async fn forwards_anthropic_stream_flag_to_responses_upstream_request() {
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
    let requests = captured.lock().unwrap();
    assert_eq!(requests[0]["stream"], true);
    assert_eq!(requests[0]["parallel_tool_calls"], false);
    assert_eq!(requests[0]["store"], false);
    assert_eq!(requests[0]["include"], json!([]));
    handle.abort();
}

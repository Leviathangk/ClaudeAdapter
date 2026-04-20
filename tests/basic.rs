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
    assert_eq!(requests[0]["metadata"]["user_id"], "session_1");
    assert_eq!(requests[0]["reasoning"]["effort"], "xhigh");
    assert_eq!(requests[0]["text"]["format"]["type"], "json_schema");
    assert_eq!(requests[0]["text"]["format"]["name"], "claude_code_output");
    assert_eq!(requests[0]["text"]["format"]["schema"]["required"][0], "ok");
    handle.abort();
}

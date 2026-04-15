mod common;

use axum::{body::{Body, to_bytes}, http::{Request, StatusCode}};
use claude_adapter::{ApiMode, build_router};
use httpmock::{Method::POST, MockServer};
use serde_json::{Value, json};
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
    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["id"], "chatcmpl_123");
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
    let app = build_router(test_config("http://127.0.0.1:1".to_string(), ApiMode::ChatCompletions));
    let response = app
        .oneshot(
            Request::post("/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json!({"model": "claude-sonnet-4.6", "messages": []}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
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

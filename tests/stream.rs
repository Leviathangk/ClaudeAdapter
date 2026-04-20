mod common;

use std::convert::Infallible;

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::Json,
    http::{Request, Response, StatusCode},
    routing::post,
};
use claude_adapter::{ApiMode, build_router};
use serde_json::{Value, json};
use tower::ServiceExt;

use common::test_config;

#[tokio::test]
async fn streams_anthropic_messages_from_chat_completions() {
    let upstream_app = Router::new().route(
        "/chat/completions",
        post(|Json(_payload): Json<Value>| async move {
            let stream = futures_util::stream::iter(vec![
                Ok::<Bytes, Infallible>(Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
                )),
                Ok::<Bytes, Infallible>(Bytes::from(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
                )),
                Ok::<Bytes, Infallible>(Bytes::from(
                    "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n",
                )),
                Ok::<Bytes, Infallible>(Bytes::from("data: [DONE]\n\n")),
            ]);
            Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
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
    let response = app.oneshot(Request::post("/v1/messages").header("content-type", "application/json").header("authorization", "Bearer incoming-secret").body(Body::from(json!({"model": "claude-sonnet-4.6", "stream": true, "messages": [{"role": "user", "content": "hello"}]}).to_string())).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("event: message_start"));
    assert!(text.contains("event: content_block_delta"));
    assert!(text.contains("event: message_stop"));
    handle.abort();
}

#[tokio::test]
async fn streams_openai_tool_calls_to_anthropic_tool_use_events() {
    let upstream_app = Router::new().route(
        "/chat/completions",
        post(|Json(_payload): Json<Value>| async move {
            let stream = futures_util::stream::iter(vec![
                Ok::<Bytes, Infallible>(Bytes::from("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_123\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"path\\\":\\\"src/main.rs\\\"}\"}}]}}]}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("data: [DONE]\n\n")),
            ]);
            Response::builder().status(StatusCode::OK).header(axum::http::header::CONTENT_TYPE, "text/event-stream").body(Body::from_stream(stream)).unwrap()
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
    let response = app.oneshot(Request::post("/v1/messages").header("content-type", "application/json").header("authorization", "Bearer incoming-secret").body(Body::from(json!({"model": "claude-sonnet-4.6", "stream": true, "messages": [{"role": "user", "content": "use a tool"}]}).to_string())).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("\"type\":\"tool_use\""));
    assert!(text.contains("\"type\":\"input_json_delta\""));
    handle.abort();
}

#[tokio::test]
async fn streams_openai_responses_tool_events_to_anthropic_tool_use() {
    let upstream_app = Router::new().route(
        "/responses",
        post(|Json(_payload): Json<Value>| async move {
            let stream = futures_util::stream::iter(vec![
                Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_123\",\"call_id\":\"call_123\",\"name\":\"Read\"}}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_123\",\"delta\":\"{\\\"path\\\":\\\"src/main.rs\\\"}\"}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"stop_reason\":\"function_call\"}}\n\n")),
            ]);
            Response::builder().status(StatusCode::OK).header(axum::http::header::CONTENT_TYPE, "text/event-stream").body(Body::from_stream(stream)).unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream_app).await.unwrap();
    });

    let app = build_router(test_config(format!("http://{address}"), ApiMode::Responses));
    let response = app.oneshot(Request::post("/v1/messages").header("content-type", "application/json").header("authorization", "Bearer incoming-secret").body(Body::from(json!({"model": "claude-opus-4-6", "stream": true, "messages": [{"role": "user", "content": "use a tool"}]}).to_string())).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("\"type\":\"tool_use\""));
    assert!(text.contains("\"id\":\"call_123\""));
    assert!(text.contains("\"stop_reason\":\"tool_use\""));
    handle.abort();
}

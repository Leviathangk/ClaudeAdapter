mod common;

use std::convert::Infallible;
use std::time::Duration;

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

#[tokio::test]
async fn streams_openai_responses_output_item_done_messages_to_anthropic_text() {
    let upstream_app = Router::new().route(
        "/responses",
        post(|Json(_payload): Json<Value>| async move {
            let stream = futures_util::stream::iter(vec![
                Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_123\"}}\n\n")),
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
    let response = app.oneshot(Request::post("/v1/messages").header("content-type", "application/json").header("authorization", "Bearer incoming-secret").body(Body::from(json!({"model": "claude-opus-4-6", "stream": true, "messages": [{"role": "user", "content": "say hello"}]}).to_string())).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("\"type\":\"text_delta\""));
    assert!(text.contains("\"text\":\"hello\""));
    assert!(text.contains("\"stop_reason\":\"end_turn\""));
    handle.abort();
}

#[tokio::test]
async fn streams_openai_responses_output_item_done_function_calls_to_anthropic_tool_use() {
    let upstream_app = Router::new().route(
        "/responses",
        post(|Json(_payload): Json<Value>| async move {
            let stream = futures_util::stream::iter(vec![
                Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_123\",\"call_id\":\"call_123\",\"name\":\"Read\",\"arguments\":\"{\\\"path\\\":\\\"src/main.rs\\\"}\"}}\n\n")),
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
    assert!(text.contains("\"partial_json\":\"{\\\"path\\\":\\\"src/main.rs\\\"}\""));
    assert!(text.contains("\"stop_reason\":\"tool_use\""));
    handle.abort();
}

#[tokio::test]
async fn deduplicates_responses_delta_and_done_events_and_preserves_block_indexes() {
    let upstream_app = Router::new().route(
        "/responses",
        post(|Json(_payload): Json<Value>| async move {
            let stream = futures_util::stream::iter(vec![
                Ok::<Bytes, Infallible>(Bytes::from("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_123\",\"call_id\":\"call_123\",\"name\":\"Grep\"}}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_123\",\"delta\":\"{\\\"pattern\\\":\\\"tauri\\\",\\\"path\\\":\\\"src\\\"}\"}\n\n")),
                Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_123\",\"call_id\":\"call_123\",\"name\":\"Grep\",\"arguments\":\"{\\\"pattern\\\":\\\"tauri\\\",\\\"path\\\":\\\"src\\\"}\"}}\n\n")),
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
    let response = app.oneshot(Request::post("/v1/messages").header("content-type", "application/json").header("authorization", "Bearer incoming-secret").body(Body::from(json!({"model": "claude-opus-4-6", "stream": true, "messages": [{"role": "user", "content": "inspect this repo"}]}).to_string())).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();

    assert_eq!(text.matches("\"text\":\"hello\"").count(), 1);
    assert_eq!(
        text.matches(
            "\"partial_json\":\"{\\\"pattern\\\":\\\"tauri\\\",\\\"path\\\":\\\"src\\\"}\""
        )
        .count(),
        1
    );
    assert_eq!(text.matches("\"type\":\"content_block_start\"").count(), 2);
    assert!(text.contains("\"index\":0"));
    assert!(text.contains("\"index\":1"));
    assert!(text.contains("\"type\":\"tool_use\""));
    assert!(text.contains("\"id\":\"call_123\""));
    assert!(text.contains("\"name\":\"Grep\""));
    handle.abort();
}

#[tokio::test]
async fn finalizes_tool_use_stream_without_waiting_for_upstream_completion() {
    let upstream_app = Router::new().route(
        "/responses",
        post(|Json(_payload): Json<Value>| async move {
            let stream = async_stream::stream! {
                yield Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_123\",\"call_id\":\"call_123\",\"name\":\"Grep\"}}\n\n"));
                yield Ok::<Bytes, Infallible>(Bytes::from("event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_123\",\"delta\":\"{\\\"pattern\\\":\\\"tauri\\\"}\"}\n\n"));
                yield Ok::<Bytes, Infallible>(Bytes::from("event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_123\",\"call_id\":\"call_123\",\"name\":\"Grep\",\"arguments\":\"{\\\"pattern\\\":\\\"tauri\\\"}\"}}\n\n"));
                tokio::time::sleep(Duration::from_millis(250)).await;
                yield Ok::<Bytes, Infallible>(Bytes::from("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"stop_reason\":\"function_call\"}}\n\n"));
            };
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
    let body = tokio::time::timeout(
        Duration::from_millis(150),
        to_bytes(response.into_body(), usize::MAX),
    )
    .await
    .expect("adapter should close tool_use turns without waiting for upstream stream end")
    .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("\"type\":\"tool_use\""));
    assert!(text.contains("\"id\":\"call_123\""));
    assert!(text.contains("\"stop_reason\":\"tool_use\""));
    assert!(text.contains("event: message_stop"));
    handle.abort();
}

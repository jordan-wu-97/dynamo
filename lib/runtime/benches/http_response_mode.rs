// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark comparing TCP Callback vs HTTP/2 Bidi Direct Response Mode
//!
//! This benchmark measures the performance difference between two response streaming methods:
//!
//! 1. **TCP Callback Mode**: Router sends HTTP request, worker connects back via TCP to send responses
//! 2. **HTTP/2 Bidi Direct Response Mode**: Worker responds directly on the same HTTP/2 stream as ndjson
//!
//! Run with: `cargo bench --bench http_response_mode`

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    pipeline::{
        Context,
        network::{
            ConnectionInfo, NetworkStreamWrapper, ResponseService, StreamOptions,
            codec::{TwoPartCodec, TwoPartMessage},
            egress::{
                http_router::HttpRequestClient,
                unified_client::{RequestPlaneClient, RequestResponseChannel},
            },
            tcp::{TcpStreamConnectionInfo, client::TcpClient, server::TcpStreamServer},
        },
    },
};
use futures::StreamExt;
use hyper_util::{
    rt::TokioExecutor,
    server::conn::auto::Builder as ConnBuilder,
    service::TowerToHyperService,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// Request sent to the worker for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchRequest {
    /// Size of each response chunk in bytes
    response_size: usize,
    /// Number of response chunks to send
    response_count: usize,
}

/// Response chunk sent back from worker
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResponse {
    chunk_id: usize,
    data: Vec<u8>,
}

/// Control message sent to worker with connection info for TCP callback
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ControlMsg {
    connection_info: ConnectionInfo,
}

/// Shared state for benchmark servers
struct BenchServerState {
    response_tx: mpsc::Sender<(Context<()>, ConnectionInfo, BenchRequest)>,
}

// ============================================================================
// TCP Callback Mode Implementation
// ============================================================================

/// Worker side: Handles TCP callback requests by connecting back via TCP
async fn tcp_callback_worker_handler(
    State(state): State<Arc<BenchServerState>>,
    body: Bytes,
) -> impl IntoResponse {
    // Decode the TwoPartMessage (header=control_msg, data=request)
    let codec = TwoPartCodec::default();
    let msg = match codec.decode_message(body) {
        Ok(m) => m,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(format!("Failed to decode message: {}", e)))
                .unwrap();
        }
    };

    let control_msg: ControlMsg = match msg.header() {
        Some(header) => match serde_json::from_slice(header) {
            Ok(c) => c,
            Err(e) => {
                return Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(Body::from(format!("Failed to decode control message: {}", e)))
                    .unwrap();
            }
        },
        None => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from("Missing control message header"))
                .unwrap();
        }
    };

    let request: BenchRequest = match msg.data() {
        Some(data) => match serde_json::from_slice(data) {
            Ok(r) => r,
            Err(e) => {
                return Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(Body::from(format!("Failed to decode request: {}", e)))
                    .unwrap();
            }
        },
        None => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from("Missing request data"))
                .unwrap();
        }
    };

    // Create context for TCP callback
    let tcp_info = match TcpStreamConnectionInfo::try_from(control_msg.connection_info.clone()) {
        Ok(info) => info,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(format!("Invalid connection info: {}", e)))
                .unwrap();
        }
    };

    let context = Context::with_id((), tcp_info.context.clone());

    // Send to background task for TCP callback processing
    if state
        .response_tx
        .send((context, control_msg.connection_info, request))
        .await
        .is_err()
    {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from("Failed to queue response"))
            .unwrap();
    }

    Response::builder()
        .status(StatusCode::OK)
        .body(Body::from("OK"))
        .unwrap()
}

/// Background task that handles TCP callback response streaming
async fn tcp_callback_response_task(
    mut rx: mpsc::Receiver<(Context<()>, ConnectionInfo, BenchRequest)>,
) {
    while let Some((context, connection_info, request)) = rx.recv().await {
        // Create response stream via TCP callback
        let mut publisher = match TcpClient::create_response_stream(
            context.context(),
            connection_info,
        )
        .await
        {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create response stream: {:?}", e);
                continue;
            }
        };

        // Send prologue
        if let Err(e) = publisher.send_prologue(None).await {
            tracing::error!("Failed to send prologue: {:?}", e);
            continue;
        }

        // Generate and send response chunks
        let data = vec![0u8; request.response_size];
        for chunk_id in 0..request.response_count {
            let response = BenchResponse {
                chunk_id,
                data: data.clone(),
            };
            let wrapper = NetworkStreamWrapper {
                data: Some(response),
                complete_final: false,
            };
            let resp_bytes = serde_json::to_vec(&wrapper).unwrap();
            if publisher.send(resp_bytes.into()).await.is_err() {
                break;
            }
        }

        // Send final marker
        let final_wrapper = NetworkStreamWrapper::<BenchResponse> {
            data: None,
            complete_final: true,
        };
        let final_bytes = serde_json::to_vec(&final_wrapper).unwrap();
        let _ = publisher.send(final_bytes.into()).await;
    }
}

// ============================================================================
// HTTP/2 Bidi Mode Implementation
// ============================================================================

/// Worker side: Handles bidi requests by streaming responses on the same HTTP connection
async fn bidi_worker_handler(headers: HeaderMap, body: Bytes) -> impl IntoResponse {
    // Check for bidi support header
    let accepts_bidi = headers
        .get("x-dynamo-accept-bidi-stream")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "true")
        .unwrap_or(false);

    if !accepts_bidi {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from("Missing x-dynamo-accept-bidi-stream header"))
            .unwrap();
    }

    // Decode request
    let request: BenchRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(format!("Failed to decode request: {}", e)))
                .unwrap();
        }
    };

    // Create streaming response
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    // Spawn task to generate responses
    tokio::spawn(async move {
        let data = vec![0u8; request.response_size];
        for chunk_id in 0..request.response_count {
            let response = BenchResponse {
                chunk_id,
                data: data.clone(),
            };
            let wrapper = NetworkStreamWrapper {
                data: Some(response),
                complete_final: false,
            };
            let line = serde_json::to_string(&wrapper).unwrap() + "\n";
            if tx.send(Ok(line)).await.is_err() {
                break;
            }
        }

        // Send final marker
        let final_wrapper = NetworkStreamWrapper::<BenchResponse> {
            data: None,
            complete_final: true,
        };
        let final_line = serde_json::to_string(&final_wrapper).unwrap() + "\n";
        let _ = tx.send(Ok(final_line)).await;
    });

    // Convert channel to stream
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header("x-dynamo-bidi-stream", "true")
        .header("content-type", "application/x-ndjson")
        .body(body)
        .unwrap()
}

// ============================================================================
// Benchmark Helper Functions
// ============================================================================

/// Start a TCP callback worker server
async fn start_tcp_callback_worker() -> (String, tokio::task::JoinHandle<()>) {
    let (response_tx, response_rx) = mpsc::channel(100);

    // Start the background response task
    tokio::spawn(tcp_callback_response_task(response_rx));

    let state = Arc::new(BenchServerState { response_tx });

    let app = Router::new()
        .route("/process", post(tcp_callback_worker_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let worker_url = format!("http://{}/process", addr);

    let handle = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let app = app.clone();
            tokio::spawn(async move {
                let conn_builder = ConnBuilder::new(TokioExecutor::new());
                let io = hyper_util::rt::TokioIo::new(stream);
                let tower_service = app.into_service();
                let hyper_service = TowerToHyperService::new(tower_service);
                let _ = conn_builder.serve_connection(io, hyper_service).await;
            });
        }
    });

    (worker_url, handle)
}

/// Start a bidi worker server
async fn start_bidi_worker() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route("/process", post(bidi_worker_handler));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let worker_url = format!("http://{}/process", addr);

    let handle = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let app = app.clone();
            tokio::spawn(async move {
                let conn_builder = ConnBuilder::new(TokioExecutor::new());
                let io = hyper_util::rt::TokioIo::new(stream);
                let tower_service = app.into_service();
                let hyper_service = TowerToHyperService::new(tower_service);
                let _ = conn_builder.serve_connection(io, hyper_service).await;
            });
        }
    });

    (worker_url, handle)
}

/// Run a single TCP callback benchmark iteration using HttpRequestClient
async fn run_tcp_callback_iteration(
    client: &HttpRequestClient,
    worker_url: &str,
    tcp_server: &Arc<TcpStreamServer>,
    request: &BenchRequest,
) -> usize {
    let context = Context::new(());

    // Register for response stream
    let options = StreamOptions::builder()
        .context(context.context())
        .enable_request_stream(false)
        .enable_response_stream(true)
        .build()
        .unwrap();

    let pending = tcp_server.register(options).await;
    let recv_stream_reg = pending.recv_stream.unwrap();
    let connection_info = recv_stream_reg.connection_info.clone();

    // Create control message with connection info
    let control_msg = ControlMsg { connection_info };
    let control_bytes = serde_json::to_vec(&control_msg).unwrap();
    let request_bytes = serde_json::to_vec(&request).unwrap();

    // Encode as TwoPartMessage
    let msg = TwoPartMessage::new(control_bytes.into(), request_bytes.into());
    let codec = TwoPartCodec::default();
    let payload = codec.encode_message(msg).unwrap();

    // Send request to worker using HttpRequestClient
    let response = client
        .send_request(worker_url.to_string(), payload, HashMap::new())
        .await
        .unwrap();

    // Should get TcpCallback response (worker doesn't support bidi)
    match response {
        RequestResponseChannel::TcpCallback(_) => {
            // Wait for TCP callback connection and receive responses
            let mut recv_stream = recv_stream_reg.stream_provider.await.unwrap().unwrap();
            let mut count = 0;

            while let Some(data) = recv_stream.recv().await {
                let wrapper: NetworkStreamWrapper<BenchResponse> =
                    serde_json::from_slice(&data).unwrap();
                if wrapper.complete_final {
                    break;
                }
                count += 1;
            }
            count
        }
        RequestResponseChannel::DirectResponse(_) => {
            panic!("Expected TcpCallback but got DirectResponse");
        }
    }
}

/// Run a single bidi benchmark iteration using HttpRequestClient
async fn run_bidi_iteration(
    client: &HttpRequestClient,
    worker_url: &str,
    request: &BenchRequest,
) -> usize {
    let request_bytes = serde_json::to_vec(&request).unwrap();

    // Send request using HttpRequestClient (automatically adds bidi header)
    let response = client
        .send_request(worker_url.to_string(), request_bytes.into(), HashMap::new())
        .await
        .unwrap();

    // Should get DirectResponse (bidi stream)
    match response {
        RequestResponseChannel::DirectResponse(mut stream) => {
            let mut count = 0;
            while let Some(result) = stream.next().await {
                let bytes = result.unwrap();
                if !bytes.is_empty() {
                    let wrapper: NetworkStreamWrapper<BenchResponse> =
                        serde_json::from_slice(&bytes).unwrap();
                    if wrapper.complete_final {
                        return count;
                    }
                    count += 1;
                }
            }
            count
        }
        RequestResponseChannel::TcpCallback(_) => {
            panic!("Expected DirectResponse but got TcpCallback");
        }
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark comparing response modes across different payload sizes
fn bench_response_size(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("response_size");
    group.sample_size(20);

    // Response size comparison (fixed count=10)
    const PAYLOAD_SIZES: &[usize] = &[64, 1024, 16384, 65536];
    const RESPONSE_COUNT: usize = 10;

    // Setup servers
    let (tcp_server, tcp_worker_url, tcp_handle, bidi_worker_url, bidi_handle) =
        runtime.block_on(async {
            let tcp_server = TcpStreamServer::new(
                dynamo_runtime::pipeline::network::tcp::server::ServerOptions::builder()
                    .port(0)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

            let (tcp_worker_url, tcp_handle) = start_tcp_callback_worker().await;
            let (bidi_worker_url, bidi_handle) = start_bidi_worker().await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            (
                tcp_server,
                tcp_worker_url,
                tcp_handle,
                bidi_worker_url,
                bidi_handle,
            )
        });

    let client = HttpRequestClient::new().unwrap();

    for &size in PAYLOAD_SIZES {
        let request = BenchRequest {
            response_size: size,
            response_count: RESPONSE_COUNT,
        };

        // TCP Callback benchmark
        group.bench_with_input(BenchmarkId::new("tcp_callback", size), &size, |b, _| {
            b.to_async(&runtime).iter(|| async {
                run_tcp_callback_iteration(&client, &tcp_worker_url, &tcp_server, &request).await
            });
        });

        // Bidi benchmark
        group.bench_with_input(BenchmarkId::new("bidi", size), &size, |b, _| {
            b.to_async(&runtime).iter(|| async {
                run_bidi_iteration(&client, &bidi_worker_url, &request).await
            });
        });
    }

    group.finish();

    // Cleanup
    tcp_handle.abort();
    bidi_handle.abort();
}

/// Benchmark comparing response modes across different stream lengths
fn bench_stream_length(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("stream_length");
    group.sample_size(20);

    // Stream length comparison (fixed size=1KB)
    const PAYLOAD_SIZE: usize = 1024;
    const STREAM_LENGTHS: &[usize] = &[1, 10, 100, 1000];

    // Setup servers
    let (tcp_server, tcp_worker_url, tcp_handle, bidi_worker_url, bidi_handle) =
        runtime.block_on(async {
            let tcp_server = TcpStreamServer::new(
                dynamo_runtime::pipeline::network::tcp::server::ServerOptions::builder()
                    .port(0)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

            let (tcp_worker_url, tcp_handle) = start_tcp_callback_worker().await;
            let (bidi_worker_url, bidi_handle) = start_bidi_worker().await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            (
                tcp_server,
                tcp_worker_url,
                tcp_handle,
                bidi_worker_url,
                bidi_handle,
            )
        });

    let client = HttpRequestClient::new().unwrap();

    for &count in STREAM_LENGTHS {
        let request = BenchRequest {
            response_size: PAYLOAD_SIZE,
            response_count: count,
        };

        // TCP Callback benchmark
        group.bench_with_input(BenchmarkId::new("tcp_callback", count), &count, |b, _| {
            b.to_async(&runtime).iter(|| async {
                run_tcp_callback_iteration(&client, &tcp_worker_url, &tcp_server, &request).await
            });
        });

        // Bidi benchmark
        group.bench_with_input(BenchmarkId::new("bidi", count), &count, |b, _| {
            b.to_async(&runtime).iter(|| async {
                run_bidi_iteration(&client, &bidi_worker_url, &request).await
            });
        });
    }

    group.finish();

    // Cleanup
    tcp_handle.abort();
    bidi_handle.abort();
}

/// Benchmark comparing response modes under concurrent load
fn bench_concurrent_load(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("concurrent_load");
    group.sample_size(10);

    // Concurrent load (fixed size=1KB, count=10)
    const PAYLOAD_SIZE: usize = 1024;
    const RESPONSE_COUNT: usize = 10;
    const CONCURRENT_REQUESTS: &[usize] = &[1, 10, 50, 100];

    // Setup servers
    let (tcp_server, tcp_worker_url, tcp_handle, bidi_worker_url, bidi_handle) =
        runtime.block_on(async {
            let tcp_server = TcpStreamServer::new(
                dynamo_runtime::pipeline::network::tcp::server::ServerOptions::builder()
                    .port(0)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

            let (tcp_worker_url, tcp_handle) = start_tcp_callback_worker().await;
            let (bidi_worker_url, bidi_handle) = start_bidi_worker().await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            (
                tcp_server,
                tcp_worker_url,
                tcp_handle,
                bidi_worker_url,
                bidi_handle,
            )
        });

    let client = Arc::new(HttpRequestClient::new().unwrap());

    for &concurrency in CONCURRENT_REQUESTS {
        let request = BenchRequest {
            response_size: PAYLOAD_SIZE,
            response_count: RESPONSE_COUNT,
        };

        // TCP Callback concurrent benchmark
        let client_clone = client.clone();
        let tcp_server_clone = tcp_server.clone();
        let tcp_url = tcp_worker_url.clone();
        group.bench_with_input(
            BenchmarkId::new("tcp_callback", concurrency),
            &concurrency,
            |b, _| {
                b.to_async(&runtime).iter(|| {
                    let client = client_clone.clone();
                    let tcp_server = tcp_server_clone.clone();
                    let url = tcp_url.clone();
                    let req = request.clone();
                    async move {
                        let handles: Vec<_> = (0..concurrency)
                            .map(|_| {
                                let client = client.clone();
                                let tcp_server = tcp_server.clone();
                                let url = url.clone();
                                let req = req.clone();
                                tokio::spawn(async move {
                                    run_tcp_callback_iteration(&client, &url, &tcp_server, &req)
                                        .await
                                })
                            })
                            .collect();

                        let mut total = 0;
                        for handle in handles {
                            total += handle.await.unwrap();
                        }
                        total
                    }
                });
            },
        );

        // Bidi concurrent benchmark
        let client_clone = client.clone();
        let bidi_url = bidi_worker_url.clone();
        group.bench_with_input(
            BenchmarkId::new("bidi", concurrency),
            &concurrency,
            |b, _| {
                b.to_async(&runtime).iter(|| {
                    let client = client_clone.clone();
                    let url = bidi_url.clone();
                    let req = request.clone();
                    async move {
                        let handles: Vec<_> = (0..concurrency)
                            .map(|_| {
                                let client = client.clone();
                                let url = url.clone();
                                let req = req.clone();
                                tokio::spawn(
                                    async move { run_bidi_iteration(&client, &url, &req).await },
                                )
                            })
                            .collect();

                        let mut total = 0;
                        for handle in handles {
                            total += handle.await.unwrap();
                        }
                        total
                    }
                });
            },
        );
    }

    group.finish();

    // Cleanup
    tcp_handle.abort();
    bidi_handle.abort();
}

criterion_group!(
    benches,
    bench_response_size,
    bench_stream_length,
    bench_concurrent_load
);
criterion_main!(benches);

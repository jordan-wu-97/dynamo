// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 Bidirectional endpoint for receiving requests and streaming responses
//!
//! Unlike `http_endpoint.rs` which returns HTTP 202 immediately and streams
//! responses via a separate TCP connection, this endpoint streams the response
//! directly back on the same HTTP/2 connection using NDJSON format.

use super::h2_bidi_handler::H2BidiWorkHandler;
use super::unified_server::RequestPlaneServer;
use crate::SystemHealth;
use crate::config::HealthStatus;
use crate::logging::TraceParent;
use crate::pipeline::network::PushWorkHandler;
use anyhow::Result;
use axum::{
    Router,
    body::{Body, Bytes},
    extract::{Path, State as AxumState},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use dashmap::DashMap;
use futures::StreamExt;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as Http2Builder;
use hyper_util::service::TowerToHyperService;
use parking_lot::Mutex;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

/// Default root path for H2Bidi RPC endpoints
const DEFAULT_H2BIDI_RPC_ROOT_PATH: &str = "/v1/rpc/bidi";

/// Shared HTTP/2 Bidirectional server that handles multiple endpoints
pub struct H2BidiServer {
    handlers: Arc<DashMap<String, Arc<H2BidiEndpointHandler>>>,
    bind_addr: SocketAddr,
    cancellation_token: CancellationToken,
}

/// Handler for a specific H2Bidi endpoint
struct H2BidiEndpointHandler {
    service_handler: Arc<dyn H2BidiWorkHandler>,
    instance_id: u64,
    namespace: Arc<String>,
    component_name: Arc<String>,
    endpoint_name: Arc<String>,
    system_health: Arc<Mutex<SystemHealth>>,
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
}

impl H2BidiServer {
    pub fn new(bind_addr: SocketAddr, cancellation_token: CancellationToken) -> Arc<Self> {
        Arc::new(Self {
            handlers: Arc::new(DashMap::new()),
            bind_addr,
            cancellation_token,
        })
    }

    /// Register an H2Bidi endpoint handler
    #[allow(clippy::too_many_arguments)]
    pub async fn register_h2bidi_endpoint(
        &self,
        subject: String,
        service_handler: Arc<dyn H2BidiWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        endpoint_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let handler = Arc::new(H2BidiEndpointHandler {
            service_handler,
            instance_id,
            namespace: Arc::new(namespace),
            component_name: Arc::new(component_name),
            endpoint_name: Arc::new(endpoint_name.clone()),
            system_health: system_health.clone(),
            inflight: Arc::new(AtomicU64::new(0)),
            notify: Arc::new(Notify::new()),
        });

        // Insert handler FIRST to ensure it's ready to receive requests
        let subject_clone = subject.clone();
        self.handlers.insert(subject, handler);

        // THEN set health status to Ready (after handler is registered)
        system_health
            .lock()
            .set_endpoint_health_status(&endpoint_name, HealthStatus::Ready);

        tracing::debug!(
            "Registered H2Bidi endpoint handler for subject: {}",
            subject_clone
        );
        Ok(())
    }

    /// Unregister an endpoint handler
    pub async fn unregister_h2bidi_endpoint(&self, subject: &str, endpoint_name: &str) {
        if let Some((_, handler)) = self.handlers.remove(subject) {
            handler
                .system_health
                .lock()
                .set_endpoint_health_status(endpoint_name, HealthStatus::NotReady);
            tracing::debug!(
                endpoint_name = %endpoint_name,
                subject = %subject,
                "Unregistered H2Bidi endpoint handler"
            );

            // Wait for inflight requests to complete
            let inflight_count = handler.inflight.load(Ordering::SeqCst);
            if inflight_count > 0 {
                tracing::info!(
                    endpoint_name = %endpoint_name,
                    inflight_count = inflight_count,
                    "Waiting for inflight H2Bidi requests to complete"
                );
                while handler.inflight.load(Ordering::SeqCst) > 0 {
                    handler.notify.notified().await;
                }
                tracing::info!(
                    endpoint_name = %endpoint_name,
                    "All inflight H2Bidi requests completed"
                );
            }
        }
    }

    /// Start the H2Bidi server
    pub async fn start(self: Arc<Self>) -> Result<()> {
        let rpc_root_path = std::env::var("DYN_H2BIDI_RPC_ROOT_PATH")
            .unwrap_or_else(|_| DEFAULT_H2BIDI_RPC_ROOT_PATH.to_string());
        let route_pattern = format!("{}/{{*endpoint}}", rpc_root_path);

        let app = Router::new()
            .route(&route_pattern, post(handle_h2bidi_request))
            .layer(TraceLayer::new_for_http())
            .with_state(self.clone());

        tracing::info!(
            "Starting H2Bidi server on {} at path {}/:endpoint",
            self.bind_addr,
            rpc_root_path
        );

        let listener = tokio::net::TcpListener::bind(&self.bind_addr).await?;
        let cancellation_token = self.cancellation_token.clone();

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, _addr)) => {
                            let app_clone = app.clone();
                            let cancel_clone = cancellation_token.clone();

                            tokio::spawn(async move {
                                let http2_builder = Http2Builder::new(TokioExecutor::new());
                                let io = TokioIo::new(stream);
                                let tower_service = app_clone.into_service();
                                let hyper_service = TowerToHyperService::new(tower_service);

                                tokio::select! {
                                    result = http2_builder.serve_connection(io, hyper_service) => {
                                        if let Err(e) = result {
                                            tracing::debug!("H2Bidi connection error: {}", e);
                                        }
                                    }
                                    _ = cancel_clone.cancelled() => {
                                        tracing::trace!("H2Bidi connection cancelled");
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Failed to accept H2Bidi connection: {}", e);
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    tracing::info!("H2BidiServer received cancellation signal, shutting down");
                    return Ok(());
                }
            }
        }
    }

    /// Bind and start the server, returning the actual bound address
    pub async fn bind_and_start(self: Arc<Self>) -> Result<SocketAddr> {
        let rpc_root_path = std::env::var("DYN_H2BIDI_RPC_ROOT_PATH")
            .unwrap_or_else(|_| DEFAULT_H2BIDI_RPC_ROOT_PATH.to_string());
        let route_pattern = format!("{}/{{*endpoint}}", rpc_root_path);

        let app = Router::new()
            .route(&route_pattern, post(handle_h2bidi_request))
            .layer(TraceLayer::new_for_http())
            .with_state(self.clone());

        let listener = tokio::net::TcpListener::bind(&self.bind_addr).await?;
        let actual_addr = listener.local_addr()?;

        tracing::info!(
            "H2Bidi server bound on {} at path {}/:endpoint",
            actual_addr,
            rpc_root_path
        );

        let cancellation_token = self.cancellation_token.clone();

        // Spawn the server loop in the background
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    accept_result = listener.accept() => {
                        match accept_result {
                            Ok((stream, _addr)) => {
                                let app_clone = app.clone();
                                let cancel_clone = cancellation_token.clone();

                                tokio::spawn(async move {
                                    let http2_builder = Http2Builder::new(TokioExecutor::new());
                                    let io = TokioIo::new(stream);
                                    let tower_service = app_clone.into_service();
                                    let hyper_service = TowerToHyperService::new(tower_service);

                                    tokio::select! {
                                        result = http2_builder.serve_connection(io, hyper_service) => {
                                            if let Err(e) = result {
                                                tracing::debug!("H2Bidi connection error: {}", e);
                                            }
                                        }
                                        _ = cancel_clone.cancelled() => {
                                            tracing::trace!("H2Bidi connection cancelled");
                                        }
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::error!("Failed to accept H2Bidi connection: {}", e);
                            }
                        }
                    }
                    _ = cancellation_token.cancelled() => {
                        tracing::info!("H2BidiServer received cancellation signal, shutting down");
                        break;
                    }
                }
            }
        });

        Ok(actual_addr)
    }

    /// Wait for all inflight requests
    pub async fn wait_for_inflight(&self) {
        for handler in self.handlers.iter() {
            while handler.value().inflight.load(Ordering::SeqCst) > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}

/// HTTP handler for H2Bidi requests - returns streaming NDJSON response
async fn handle_h2bidi_request(
    AxumState(server): AxumState<Arc<H2BidiServer>>,
    Path(endpoint_path): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Look up the handler for this endpoint
    let handler = match server.handlers.get(&endpoint_path) {
        Some(h) => h.clone(),
        None => {
            tracing::warn!("No H2Bidi handler found for endpoint: {}", endpoint_path);
            return (StatusCode::NOT_FOUND, "Endpoint not found").into_response();
        }
    };

    // Increment inflight counter
    handler.inflight.fetch_add(1, Ordering::SeqCst);

    // Extract tracing headers
    let traceparent = TraceParent::from_axum_headers(&headers);
    let instance_id = handler.instance_id;
    let namespace = handler.namespace.clone();
    let component_name = handler.component_name.clone();
    let endpoint_name = handler.endpoint_name.clone();

    tracing::trace!(instance_id, "Handling H2Bidi request");

    // Call the handler to get the response stream
    let stream_result = handler
        .service_handler
        .handle_streaming(body)
        .instrument(tracing::info_span!(
            "h2bidi_handle_streaming",
            component = component_name.as_ref(),
            endpoint = endpoint_name.as_ref(),
            namespace = namespace.as_ref(),
            instance_id = instance_id,
            trace_id = traceparent.trace_id,
            parent_id = traceparent.parent_id,
            x_request_id = traceparent.x_request_id,
            x_dynamo_request_id = traceparent.x_dynamo_request_id,
            tracestate = traceparent.tracestate
        ))
        .await;

    let inflight = handler.inflight.clone();
    let notify = handler.notify.clone();

    match stream_result {
        Ok(response_stream) => {
            // Wrap the stream to decrement inflight when done
            let wrapped_stream = InFlightStream {
                inner: response_stream,
                inflight: inflight.clone(),
                notify: notify.clone(),
                finished: false,
            };

            // Return streaming response with NDJSON content type
            let body = Body::from_stream(wrapped_stream);

            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/x-ndjson")
                .body(body)
                .unwrap()
        }
        Err(e) => {
            tracing::warn!(instance_id, "H2Bidi request failed: {}", e);

            // Decrement inflight on error
            inflight.fetch_sub(1, Ordering::SeqCst);
            notify.notify_one();

            // Return error as JSON
            let error_json = serde_json::json!({
                "error": e.to_string()
            });

            (StatusCode::INTERNAL_SERVER_ERROR, error_json.to_string()).into_response()
        }
    }
}

/// Stream wrapper that decrements inflight counter when stream ends
struct InFlightStream {
    inner: std::pin::Pin<Box<dyn futures::Stream<Item = Bytes> + Send>>,
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
    finished: bool,
}

impl futures::Stream for InFlightStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(bytes)) => std::task::Poll::Ready(Some(Ok(bytes))),
            std::task::Poll::Ready(None) => {
                if !self.finished {
                    self.finished = true;
                    self.inflight.fetch_sub(1, Ordering::SeqCst);
                    self.notify.notify_one();
                }
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl Drop for InFlightStream {
    fn drop(&mut self) {
        if !self.finished {
            // Stream was dropped before completion (client disconnect)
            self.inflight.fetch_sub(1, Ordering::SeqCst);
            self.notify.notify_one();
        }
    }
}

// Implement RequestPlaneServer trait for H2BidiServer
//
// Note: This implementation adapts H2BidiWorkHandler to the RequestPlaneServer
// interface which expects PushWorkHandler. For H2Bidi mode, the handler registered
// must implement H2BidiWorkHandler.
#[async_trait::async_trait]
impl RequestPlaneServer for H2BidiServer {
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        // For H2Bidi, we need the handler to implement H2BidiWorkHandler
        // Try to downcast or use an adapter
        // For now, we'll create an adapter that wraps PushWorkHandler
        let h2bidi_handler = Arc::new(PushWorkHandlerAdapter {
            inner: service_handler,
        });

        self.register_h2bidi_endpoint(
            endpoint_name.clone(),
            h2bidi_handler,
            instance_id,
            namespace,
            component_name,
            endpoint_name,
            system_health,
        )
        .await
    }

    async fn unregister_endpoint(&self, endpoint_name: &str) -> Result<()> {
        self.unregister_h2bidi_endpoint(endpoint_name, endpoint_name)
            .await;
        Ok(())
    }

    fn address(&self) -> String {
        format!("http://{}:{}", self.bind_addr.ip(), self.bind_addr.port())
    }

    fn transport_name(&self) -> &'static str {
        "h2bidi"
    }

    fn is_healthy(&self) -> bool {
        true
    }
}

/// Adapter to use PushWorkHandler as H2BidiWorkHandler
///
/// This adapter allows existing PushWorkHandler implementations to be used
/// with the H2Bidi server. Note that this is a temporary adapter - ideally
/// the handler should implement H2BidiWorkHandler directly.
struct PushWorkHandlerAdapter {
    inner: Arc<dyn PushWorkHandler>,
}

#[async_trait::async_trait]
impl H2BidiWorkHandler for PushWorkHandlerAdapter {
    async fn handle_streaming(
        &self,
        _payload: Bytes,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Bytes> + Send>>, crate::pipeline::PipelineError>
    {
        // This adapter cannot properly convert fire-and-forget to streaming
        // It should only be used when the underlying handler actually implements H2BidiWorkHandler
        Err(crate::pipeline::PipelineError::Generic(
            "PushWorkHandler cannot be used with H2Bidi - handler must implement H2BidiWorkHandler directly".to_string()
        ))
    }

    fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        self.inner.add_metrics(endpoint, metrics_labels)
    }

    fn set_endpoint_health_check_notifier(
        &self,
        notifier: Arc<tokio::sync::Notify>,
    ) -> Result<()> {
        self.inner.set_endpoint_health_check_notifier(notifier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2bidi_server_creation() {
        use std::net::{IpAddr, Ipv4Addr};
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let token = CancellationToken::new();

        let server = H2BidiServer::new(bind_addr, token);
        assert!(server.handlers.is_empty());
    }
}

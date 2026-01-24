// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 Bidirectional Work Handler
//!
//! This module defines the `H2BidiWorkHandler` trait for processing requests
//! and returning streaming responses directly. Unlike `PushWorkHandler` which
//! returns `()` and streams responses via a TCP callback, this handler returns
//! a `Stream` of response bytes.

use super::*;
use crate::metrics::prometheus_names::work_handler;
use crate::protocols::maybe_error::MaybeError;
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

/// Work handler trait for HTTP/2 bidirectional streaming
///
/// Unlike `PushWorkHandler` which uses a fire-and-forget model with TCP callback
/// for responses, this trait returns a stream of response bytes directly.
/// This enables true HTTP/2 bidirectional streaming where request and response
/// use the same connection.
#[async_trait]
pub trait H2BidiWorkHandler: Send + Sync {
    /// Handle a request payload and return a streaming response
    ///
    /// # Arguments
    ///
    /// * `payload` - JSON-encoded request payload
    ///
    /// # Returns
    ///
    /// Returns a stream of NDJSON response bytes. Each item in the stream
    /// is a complete JSON object followed by a newline character.
    /// The stream ends when generation is complete.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The payload cannot be deserialized
    /// - The underlying engine fails to generate a response
    async fn handle_streaming(
        &self,
        payload: Bytes,
    ) -> Result<Pin<Box<dyn Stream<Item = Bytes> + Send>>, PipelineError>;

    /// Add metrics to the handler
    fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()>;

    /// Set the endpoint-specific notifier for health check timer resets
    fn set_endpoint_health_check_notifier(&self, _notifier: Arc<tokio::sync::Notify>) -> Result<()> {
        // Default implementation for backwards compatibility
        Ok(())
    }
}

// RAII guard for metrics
struct H2BidiRequestMetricsGuard {
    inflight_requests: prometheus::IntGauge,
    request_duration: prometheus::Histogram,
    start_time: Instant,
}

impl Drop for H2BidiRequestMetricsGuard {
    fn drop(&mut self) {
        self.inflight_requests.dec();
        self.request_duration
            .observe(self.start_time.elapsed().as_secs_f64());
    }
}

#[async_trait]
impl<T: Data, U: Data> H2BidiWorkHandler for Ingress<SingleIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de> + std::fmt::Debug,
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    fn add_metrics(
        &self,
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        // Reuse the existing Ingress add_metrics implementation
        use crate::pipeline::network::Ingress;
        Ingress::add_metrics(self, endpoint, metrics_labels)
    }

    fn set_endpoint_health_check_notifier(&self, notifier: Arc<tokio::sync::Notify>) -> Result<()> {
        self.endpoint_health_check_notifier
            .set(notifier)
            .map_err(|_| anyhow::anyhow!("Endpoint health check notifier already set"))?;
        Ok(())
    }

    async fn handle_streaming(
        &self,
        payload: Bytes,
    ) -> Result<Pin<Box<dyn Stream<Item = Bytes> + Send>>, PipelineError> {
        let start_time = std::time::Instant::now();

        // Increment inflight and ensure it's decremented via RAII guard
        let metrics_guard = self.metrics().map(|m| {
            m.request_counter.inc();
            m.inflight_requests.inc();
            m.request_bytes.inc_by(payload.len() as u64);
            H2BidiRequestMetricsGuard {
                inflight_requests: m.inflight_requests.clone(),
                request_duration: m.request_duration.clone(),
                start_time,
            }
        });

        // For H2Bidi, the payload is just the request JSON directly (no two-part message)
        let request: T = serde_json::from_slice(&payload)?;

        // Generate a request ID
        let request_id = uuid::Uuid::new_v4().to_string();
        tracing::trace!(?request, request_id, "H2Bidi received request");

        // Create context and wrap request
        let request: context::Context<T> = Context::with_id(request, request_id.clone());

        // Call the engine to generate response stream
        tracing::trace!(request_id, "H2Bidi calling generate");
        let stream_result = self
            .segment
            .get()
            .expect("segment not set")
            .generate(request)
            .await
            .map_err(|e| {
                if let Some(m) = self.metrics() {
                    m.error_counter
                        .with_label_values(&[work_handler::error_types::GENERATE])
                        .inc();
                }
                PipelineError::GenerateError(e)
            });

        let stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                drop(metrics_guard);
                return Err(e);
            }
        };

        let _engine_ctx = stream.context();
        let metrics = self.metrics().cloned();
        let notifier = self.endpoint_health_check_notifier.get().cloned();

        // Transform the response stream to NDJSON bytes
        // Each response item becomes a JSON line followed by newline
        let ndjson_stream = futures::StreamExt::filter_map(stream, move |resp| {
            let metrics = metrics.clone();

            async move {
                // Serialize response to JSON
                match serde_json::to_vec(&resp) {
                    Ok(mut json_bytes) => {
                        // Add newline for NDJSON format
                        json_bytes.push(b'\n');

                        if let Some(m) = metrics.as_ref() {
                            m.response_bytes.inc_by(json_bytes.len() as u64);
                        }

                        Some(Bytes::from(json_bytes))
                    }
                    Err(err) => {
                        tracing::error!(%err, "Failed to serialize response to JSON");
                        if let Some(m) = metrics.as_ref() {
                            m.error_counter
                                .with_label_values(&["serialization"])
                                .inc();
                        }
                        None
                    }
                }
            }
        });

        // Wrap the stream to handle cleanup when it ends
        let cleanup_stream = CleanupStream {
            inner: Box::pin(ndjson_stream),
            metrics_guard,
            notifier,
            finished: false,
        };

        Ok(Box::pin(cleanup_stream))
    }
}

/// A stream wrapper that handles cleanup when the stream ends or is dropped
struct CleanupStream {
    inner: Pin<Box<dyn Stream<Item = Bytes> + Send>>,
    metrics_guard: Option<H2BidiRequestMetricsGuard>,
    notifier: Option<Arc<tokio::sync::Notify>>,
    finished: bool,
}

impl Stream for CleanupStream {
    type Item = Bytes;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(item)) => std::task::Poll::Ready(Some(item)),
            std::task::Poll::Ready(None) => {
                if !self.finished {
                    self.finished = true;
                    // Notify health check manager that generation is complete
                    if let Some(notifier) = self.notifier.as_ref() {
                        notifier.notify_one();
                    }
                    // Drop the metrics guard to record duration
                    drop(self.metrics_guard.take());
                }
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl Drop for CleanupStream {
    fn drop(&mut self) {
        if !self.finished {
            // Stream was dropped before completion (e.g., client disconnected)
            if let Some(notifier) = self.notifier.as_ref() {
                notifier.notify_one();
            }
        }
        // metrics_guard is dropped automatically
    }
}

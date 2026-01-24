// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 Bidirectional Router
//!
//! This router sends JSON requests and receives NDJSON streaming responses
//! over a single HTTP/2 connection. Unlike `AddressedPushRouter` which uses
//! a separate TCP connection for response streaming, this router receives
//! the response directly on the HTTP/2 stream.

use super::h2_bidi_client::{H2BidiClient, H2BidiStreamingClient};
use crate::engine::{AsyncEngine, AsyncEngineContextProvider, Data};
use crate::logging::inject_trace_headers_into_map;
use crate::pipeline::{Context, ManyOut, PipelineError, ResponseStream, SingleIn, context};
use crate::protocols::maybe_error::MaybeError;

use anyhow::{Error, Result};
use bytes::{Buf, Bytes, BytesMut};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request wrapper with target address
pub struct H2BidiAddressedRequest<T> {
    request: T,
    address: String,
}

impl<T> H2BidiAddressedRequest<T> {
    pub fn new(request: T, address: String) -> Self {
        Self { request, address }
    }

    pub(crate) fn into_parts(self) -> (T, String) {
        (self.request, self.address)
    }
}

/// HTTP/2 Bidirectional Router
///
/// Unlike `AddressedPushRouter` which:
/// 1. Sends request via HTTP/TCP/NATS
/// 2. Receives response via separate TCP "call-home" connection
///
/// This router:
/// 1. Sends JSON request via HTTP/2 POST
/// 2. Receives NDJSON response on same HTTP/2 stream
pub struct H2BidiRouter {
    client: Arc<dyn H2BidiStreamingClient>,
}

impl H2BidiRouter {
    /// Create a new H2BidiRouter with the default H2Bidi client
    pub fn new() -> Result<Arc<Self>> {
        let client = H2BidiClient::new()?;
        Ok(Arc::new(Self {
            client: Arc::new(client),
        }))
    }

    /// Create a new H2BidiRouter with a custom streaming client
    pub fn with_client(client: Arc<dyn H2BidiStreamingClient>) -> Arc<Self> {
        Arc::new(Self { client })
    }

    /// Create from environment configuration
    pub fn from_env() -> Result<Arc<Self>> {
        let client = H2BidiClient::from_env()?;
        Ok(Arc::new(Self {
            client: Arc::new(client),
        }))
    }
}

#[async_trait::async_trait]
impl<T, U> AsyncEngine<SingleIn<H2BidiAddressedRequest<T>>, ManyOut<U>, Error> for H2BidiRouter
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(
        &self,
        request: SingleIn<H2BidiAddressedRequest<T>>,
    ) -> Result<ManyOut<U>, Error> {
        let request_id = request.context().id().to_string();
        let (addressed_request, context) = request.transfer(());
        let (request_data, address) = addressed_request.into_parts();
        let engine_ctx = context.context();

        // Serialize the request to JSON
        let payload = serde_json::to_vec(&request_data)?;

        tracing::trace!(
            request_id,
            transport = self.client.transport_name(),
            address = %address,
            payload_size = payload.len(),
            "H2Bidi sending request"
        );

        // Prepare trace headers
        let mut headers = std::collections::HashMap::new();
        inject_trace_headers_into_map(&mut headers);

        // Send the request and get the streaming response
        let response_stream = self
            .client
            .send_streaming_request(address, Bytes::from(payload), headers)
            .await?;

        tracing::trace!(request_id, "H2Bidi received response stream");

        // Create a line buffer for NDJSON parsing
        let engine_ctx_clone = engine_ctx.clone();

        // Transform the byte stream into parsed NDJSON objects
        let ndjson_stream = NdjsonStream {
            inner: response_stream,
            buffer: BytesMut::new(),
            engine_ctx: engine_ctx_clone.clone(),
            finished: false,
        };

        // Map the NDJSON stream to the output type
        let output_stream = ndjson_stream.filter_map(move |result| {
            let engine_ctx = engine_ctx_clone.clone();
            async move {
                match result {
                    Ok(json_bytes) => {
                        // Try to deserialize the JSON line
                        match serde_json::from_slice::<U>(&json_bytes) {
                            Ok(item) => Some(item),
                            Err(err) => {
                                let json_str = String::from_utf8_lossy(&json_bytes);
                                tracing::warn!(%err, %json_str, "Failed to deserialize NDJSON line");
                                Some(U::from_err(Error::new(err).into()))
                            }
                        }
                    }
                    Err(err) => {
                        if engine_ctx.is_stopped() {
                            // Graceful cancellation
                            tracing::debug!("H2Bidi request cancelled");
                            None
                        } else {
                            // Unexpected error
                            tracing::warn!(%err, "H2Bidi stream error");
                            Some(U::from_err(err.into()))
                        }
                    }
                }
            }
        });

        Ok(ResponseStream::new(Box::pin(output_stream), engine_ctx))
    }
}

/// Stream that buffers bytes and emits complete NDJSON lines
struct NdjsonStream {
    inner: std::pin::Pin<Box<dyn futures::Stream<Item = Result<Bytes>> + Send>>,
    buffer: BytesMut,
    engine_ctx: Arc<dyn crate::engine::AsyncEngineContext>,
    finished: bool,
}

impl futures::Stream for NdjsonStream {
    type Item = Result<Bytes>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            // First, check if we have a complete line in the buffer
            if let Some(newline_pos) = self.buffer.iter().position(|&b| b == b'\n') {
                // Extract the line (without the newline)
                let line = self.buffer.split_to(newline_pos);
                // Skip the newline character
                self.buffer.advance(1);

                // Skip empty lines
                if line.is_empty() {
                    continue;
                }

                return std::task::Poll::Ready(Some(Ok(line.freeze())));
            }

            // No complete line, need more data
            if self.finished {
                // Stream ended - check if there's remaining data without trailing newline
                if !self.buffer.is_empty() {
                    let remaining = self.buffer.split().freeze();
                    return std::task::Poll::Ready(Some(Ok(remaining)));
                }
                return std::task::Poll::Ready(None);
            }

            // Check for cancellation
            if self.engine_ctx.is_killed() || self.engine_ctx.is_stopped() {
                self.finished = true;
                return std::task::Poll::Ready(None);
            }

            // Poll the inner stream for more data
            match self.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    // Append new bytes to buffer
                    self.buffer.extend_from_slice(&bytes);
                    // Continue loop to check for complete lines
                }
                std::task::Poll::Ready(Some(Err(err))) => {
                    self.finished = true;
                    return std::task::Poll::Ready(Some(Err(err)));
                }
                std::task::Poll::Ready(None) => {
                    self.finished = true;
                    // Continue loop to drain remaining buffer
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2bidi_addressed_request() {
        let request = H2BidiAddressedRequest::new("test_data", "http://localhost:8889/v1/rpc/bidi/generate".to_string());
        let (data, address) = request.into_parts();
        assert_eq!(data, "test_data");
        assert_eq!(address, "http://localhost:8889/v1/rpc/bidi/generate");
    }

    #[test]
    fn test_h2bidi_router_creation() {
        let router = H2BidiRouter::new();
        assert!(router.is_ok());
    }
}

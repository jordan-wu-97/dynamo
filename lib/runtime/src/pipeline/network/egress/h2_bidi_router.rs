// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 Bidirectional Streaming Router
//!
//! This module provides an AsyncEngine implementation that sends requests and
//! receives streaming responses over a single HTTP/2 connection. It replaces
//! the two-connection architecture (AddressedPushRouter) for H2Bidi mode.
//!
//! ## Wire Protocol
//!
//! **Request:**
//! ```text
//! POST /v1/rpc/bidi/{endpoint} HTTP/2
//! Content-Type: application/json
//! x-request-id: {uuid}
//! traceparent: {trace_context}
//!
//! {
//!   "id": "request-uuid",
//!   "request_type": "single_in",
//!   "response_type": "many_out",
//!   "request": <serialized T>
//! }
//! ```
//!
//! **Response:**
//! ```text
//! HTTP/2 200 OK
//! Content-Type: application/x-ndjson
//!
//! {"data": <U>, "complete_final": false}
//! {"data": <U>, "complete_final": false}
//! {"data": null, "complete_final": true}
//! ```

use std::sync::Arc;

use super::h2_bidi_client::H2BidiClient;
use crate::engine::{AsyncEngine, AsyncEngineContextProvider, Data};
use crate::logging::inject_trace_headers_into_map;
use crate::pipeline::network::NetworkStreamWrapper;
use crate::pipeline::network::STREAM_ERR_MSG;
use crate::pipeline::{ManyOut, ResponseStream, SingleIn};
use crate::protocols::maybe_error::MaybeError;

use super::addressed_router::AddressedRequest;
use anyhow::{Error, Result};
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

/// Error types specific to H2Bidi operations
#[derive(Debug, thiserror::Error)]
pub enum H2BidiError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(#[from] reqwest::Error),

    #[error("HTTP error: status {status}, body: {body}")]
    HttpError { status: u16, body: String },

    #[error("Stream ended unexpectedly")]
    UnexpectedStreamEnd,

    #[error("Failed to parse NDJSON: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Request cancelled")]
    Cancelled,
}

/// Request type marker for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RequestType {
    SingleIn,
    ManyIn,
}

/// Response type marker for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResponseType {
    SingleOut,
    ManyOut,
}

/// Request message sent to worker via H2Bidi
///
/// Note: Unlike the legacy AddressedPushRouter, this message does NOT include
/// ConnectionInfo since the response comes back on the same connection.
#[derive(Debug, Serialize)]
struct BidiRequestMessage<T> {
    /// Unique request identifier
    id: String,

    /// Type of request (single_in or many_in)
    request_type: RequestType,

    /// Expected response type (single_out or many_out)
    response_type: ResponseType,

    /// The actual request payload
    request: T,
}

/// State machine for parsing NDJSON from byte chunks
///
/// NDJSON (Newline-Delimited JSON) is a format where each line is a valid JSON
/// object. This parser handles:
/// - Lines split across multiple HTTP/2 frames
/// - Multiple complete lines in a single frame
/// - Partial JSON buffered until newline arrives
struct NdjsonParser {
    /// Buffer for incomplete lines
    buffer: BytesMut,
}

impl NdjsonParser {
    /// Create a new NDJSON parser
    fn new() -> Self {
        Self {
            buffer: BytesMut::new(),
        }
    }

    /// Feed a chunk of bytes and return any complete lines
    ///
    /// This method buffers partial lines and returns complete lines when
    /// a newline character is found. Multiple lines may be returned if
    /// the chunk contains multiple newlines.
    fn feed(&mut self, chunk: Bytes) -> Vec<Bytes> {
        self.buffer.extend_from_slice(&chunk);

        let mut lines = Vec::new();

        // Find all complete lines (ending with \n)
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            // Split off the line (including newline)
            let line = self.buffer.split_to(pos + 1);
            let line_bytes = line.freeze();

            // Trim the newline and any trailing \r
            let trimmed_len = if line_bytes.ends_with(b"\r\n") {
                line_bytes.len() - 2
            } else {
                line_bytes.len() - 1
            };

            let trimmed = line_bytes.slice(..trimmed_len);

            // Skip empty lines
            if !trimmed.is_empty() {
                lines.push(trimmed);
            }
        }

        lines
    }

    /// Check if there's remaining data in the buffer
    fn has_remaining(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get remaining buffer content (for error reporting)
    fn remaining(&self) -> &[u8] {
        &self.buffer
    }
}

/// HTTP/2 bidirectional router
///
/// This router implements AsyncEngine to send requests and receive streaming
/// responses over a single HTTP/2 connection. It is the H2Bidi counterpart
/// to AddressedPushRouter.
pub struct H2BidiRouter {
    client: Arc<H2BidiClient>,
}

impl H2BidiRouter {
    /// Create a new H2BidiRouter with the given client
    pub fn new(client: Arc<H2BidiClient>) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl<T, U> AsyncEngine<SingleIn<AddressedRequest<T>>, ManyOut<U>, Error> for H2BidiRouter
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
        let request_id = request.context().id().to_string();
        let (addressed_request, context) = request.transfer(());
        let (request_data, address) = addressed_request.into_parts();
        let engine_ctx = context.context();
        let engine_ctx_for_stream = engine_ctx.clone();

        // Build the request message (no ConnectionInfo needed!)
        let bidi_message = BidiRequestMessage {
            id: engine_ctx.id().to_string(),
            request_type: RequestType::SingleIn,
            response_type: ResponseType::ManyOut,
            request: request_data,
        };

        // Serialize request to JSON
        let body = serde_json::to_vec(&bidi_message)?;

        // Build URL: append /bidi to the address
        // The address is typically like "http://host:port/v1/rpc/{endpoint}"
        // We transform it to "http://host:port/v1/rpc/bidi/{endpoint}"
        let url = transform_url_to_bidi(&address);

        tracing::trace!(
            request_id,
            url = %url,
            body_len = body.len(),
            "Sending H2Bidi request"
        );

        // Prepare trace headers
        let mut headers = std::collections::HashMap::new();
        inject_trace_headers_into_map(&mut headers);
        headers.insert("x-request-id".to_string(), request_id.clone());

        // Send streaming request
        let response_stream = self
            .client
            .send_streaming_request(url, Bytes::from(body), headers)
            .await?;

        tracing::trace!(request_id, "H2Bidi request sent, processing response stream");

        // Track stream completion
        let mut is_complete_final = false;

        // Create response stream that parses NDJSON
        let stream = async_stream::stream! {
            let mut parser = NdjsonParser::new();
            let mut response_stream = std::pin::pin!(response_stream);

            loop {
                tokio::select! {
                    biased;

                    // Check for cancellation
                    _ = engine_ctx_for_stream.killed() => {
                        tracing::debug!(request_id, "H2Bidi request cancelled by context");
                        break;
                    }

                    // Process next chunk from stream
                    chunk = response_stream.next() => {
                        match chunk {
                            Some(Ok(bytes)) => {
                                // Parse NDJSON lines from the chunk
                                for line in parser.feed(bytes) {
                                    match parse_ndjson_line::<U>(&line) {
                                        Ok(Some(item)) => {
                                            is_complete_final = item.complete_final;
                                            if let Some(data) = item.data {
                                                yield data;
                                            } else if item.complete_final {
                                                // Final sentinel with no data - stream is complete
                                                break;
                                            } else {
                                                // Empty data without complete_final - error
                                                yield U::from_err(
                                                    Error::msg("Empty response received - this should never happen").into()
                                                );
                                            }
                                        }
                                        Ok(None) => {
                                            // Empty line, skip
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                request_id,
                                                error = %e,
                                                line = %String::from_utf8_lossy(&line),
                                                "Failed to parse NDJSON line"
                                            );
                                            yield U::from_err(Error::new(e).into());
                                        }
                                    }
                                }

                                // If we've seen the final message, stop processing
                                if is_complete_final {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                tracing::warn!(request_id, error = %e, "H2Bidi stream error");
                                yield U::from_err(e.into());
                                break;
                            }
                            None => {
                                // Stream ended
                                if !is_complete_final {
                                    // Check if we're stopped gracefully
                                    if engine_ctx_for_stream.is_stopped() {
                                        tracing::debug!(request_id, "H2Bidi stream ended after stop_generating()");
                                    } else {
                                        // Stream ended unexpectedly
                                        if parser.has_remaining() {
                                            tracing::warn!(
                                                request_id,
                                                remaining = %String::from_utf8_lossy(parser.remaining()),
                                                "H2Bidi stream ended with partial data in buffer"
                                            );
                                        }
                                        tracing::debug!(request_id, "{}", STREAM_ERR_MSG);
                                        yield U::from_err(Error::msg(STREAM_ERR_MSG).into());
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
    }
}

/// Transform a standard RPC URL to a bidi URL
///
/// Example:
/// - Input: `http://host:port/v1/rpc/generate`
/// - Output: `http://host:port/v1/rpc/bidi/generate`
fn transform_url_to_bidi(url: &str) -> String {
    // Find the last path component and insert "bidi" before it
    if let Some(idx) = url.rfind('/') {
        let (base, endpoint) = url.split_at(idx);

        // Check if base ends with /v1/rpc or similar pattern
        if base.ends_with("/rpc") || base.ends_with("/v1/rpc") {
            return format!("{}/bidi{}", base, endpoint);
        }
    }

    // Fallback: just append /bidi
    format!("{}/bidi", url)
}

/// Parse a single NDJSON line into a NetworkStreamWrapper
fn parse_ndjson_line<U>(line: &[u8]) -> Result<Option<NetworkStreamWrapper<U>>, serde_json::Error>
where
    U: for<'de> Deserialize<'de>,
{
    if line.is_empty() {
        return Ok(None);
    }
    let wrapper: NetworkStreamWrapper<U> = serde_json::from_slice(line)?;
    Ok(Some(wrapper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndjson_parser_single_line() {
        let mut parser = NdjsonParser::new();
        let lines = parser.feed(Bytes::from("{\"data\": 1}\n"));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].as_ref(), b"{\"data\": 1}");
    }

    #[test]
    fn test_ndjson_parser_multiple_lines() {
        let mut parser = NdjsonParser::new();
        let lines = parser.feed(Bytes::from("{\"a\": 1}\n{\"b\": 2}\n{\"c\": 3}\n"));
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].as_ref(), b"{\"a\": 1}");
        assert_eq!(lines[1].as_ref(), b"{\"b\": 2}");
        assert_eq!(lines[2].as_ref(), b"{\"c\": 3}");
    }

    #[test]
    fn test_ndjson_parser_split_across_chunks() {
        let mut parser = NdjsonParser::new();

        // First chunk: partial line
        let lines1 = parser.feed(Bytes::from("{\"data\":"));
        assert!(lines1.is_empty());

        // Second chunk: rest of line + complete line
        let lines2 = parser.feed(Bytes::from(" 1}\n{\"data\": 2}\n"));
        assert_eq!(lines2.len(), 2);
        assert_eq!(lines2[0].as_ref(), b"{\"data\": 1}");
        assert_eq!(lines2[1].as_ref(), b"{\"data\": 2}");
    }

    #[test]
    fn test_ndjson_parser_crlf() {
        let mut parser = NdjsonParser::new();
        let lines = parser.feed(Bytes::from("{\"data\": 1}\r\n{\"data\": 2}\r\n"));
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].as_ref(), b"{\"data\": 1}");
        assert_eq!(lines[1].as_ref(), b"{\"data\": 2}");
    }

    #[test]
    fn test_ndjson_parser_empty_lines() {
        let mut parser = NdjsonParser::new();
        let lines = parser.feed(Bytes::from("{\"a\": 1}\n\n{\"b\": 2}\n"));
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].as_ref(), b"{\"a\": 1}");
        assert_eq!(lines[1].as_ref(), b"{\"b\": 2}");
    }

    #[test]
    fn test_ndjson_parser_has_remaining() {
        let mut parser = NdjsonParser::new();
        assert!(!parser.has_remaining());

        parser.feed(Bytes::from("{\"incomplete"));
        assert!(parser.has_remaining());

        parser.feed(Bytes::from("\"}\n"));
        assert!(!parser.has_remaining());
    }

    #[test]
    fn test_transform_url_to_bidi() {
        assert_eq!(
            transform_url_to_bidi("http://localhost:8080/v1/rpc/generate"),
            "http://localhost:8080/v1/rpc/bidi/generate"
        );
        assert_eq!(
            transform_url_to_bidi("http://worker:9000/rpc/process"),
            "http://worker:9000/rpc/bidi/process"
        );
    }

    #[test]
    fn test_parse_ndjson_line() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct TestData {
            value: i32,
        }

        let line = b"{\"data\": {\"value\": 42}, \"complete_final\": false}";
        let result: NetworkStreamWrapper<TestData> =
            parse_ndjson_line(line).unwrap().unwrap();
        assert_eq!(result.data.unwrap().value, 42);
        assert!(!result.complete_final);

        let final_line = b"{\"data\": null, \"complete_final\": true}";
        let result: NetworkStreamWrapper<TestData> =
            parse_ndjson_line(final_line).unwrap().unwrap();
        assert!(result.data.is_none());
        assert!(result.complete_final);
    }

    #[test]
    fn test_parse_ndjson_line_empty() {
        let result: Result<Option<NetworkStreamWrapper<i32>>, _> = parse_ndjson_line(b"");
        assert!(result.unwrap().is_none());
    }
}

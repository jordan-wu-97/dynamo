// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 Bidirectional client for request plane
//!
//! This client sends JSON requests and receives NDJSON streaming responses
//! on the same HTTP/2 connection. Unlike the standard `HttpRequestClient`
//! which returns immediately with an ack, this client streams the response.

use super::http_router::Http2Config;
use super::unified_client::Headers;
use crate::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use std::pin::Pin;
use std::time::Duration;

/// Default timeout for the entire streaming response (not just the ack)
const DEFAULT_H2BIDI_RESPONSE_TIMEOUT_SECS: u64 = 300; // 5 minutes for long generations

/// HTTP/2 Bidirectional configuration
#[derive(Debug, Clone)]
pub struct H2BidiConfig {
    /// Base HTTP/2 configuration
    pub http2: Http2Config,
    /// Timeout for the entire streaming response
    pub response_timeout: Duration,
}

impl Default for H2BidiConfig {
    fn default() -> Self {
        Self {
            http2: Http2Config::default(),
            response_timeout: Duration::from_secs(DEFAULT_H2BIDI_RESPONSE_TIMEOUT_SECS),
        }
    }
}

impl H2BidiConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let http2 = Http2Config::from_env();

        let response_timeout = std::env::var("DYN_H2BIDI_RESPONSE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(DEFAULT_H2BIDI_RESPONSE_TIMEOUT_SECS));

        Self {
            http2,
            response_timeout,
        }
    }
}

/// Trait for HTTP/2 bidirectional streaming clients
///
/// Unlike `RequestPlaneClient` which returns a simple ack, this trait
/// returns a stream of response chunks for NDJSON parsing.
#[async_trait]
pub trait H2BidiStreamingClient: Send + Sync {
    /// Send a request and receive a streaming response
    ///
    /// # Arguments
    ///
    /// * `address` - Full URL to send the request to
    /// * `payload` - JSON request body
    /// * `headers` - Custom headers for tracing, etc.
    ///
    /// # Returns
    ///
    /// Returns a stream of Bytes chunks that form the NDJSON response.
    /// Each chunk may contain partial lines, complete lines, or multiple lines.
    /// The caller is responsible for line buffering and JSON parsing.
    async fn send_streaming_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>>;

    /// Get the transport name
    fn transport_name(&self) -> &'static str;

    /// Check if the client is healthy
    fn is_healthy(&self) -> bool;
}

/// HTTP/2 bidirectional streaming client implementation
pub struct H2BidiClient {
    client: reqwest::Client,
    config: H2BidiConfig,
}

impl H2BidiClient {
    /// Create a new H2Bidi client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(H2BidiConfig::default())
    }

    /// Create a new H2Bidi client with custom configuration
    pub fn with_config(config: H2BidiConfig) -> Result<Self> {
        // Build reqwest client without per-request timeout
        // (the response_timeout is for the entire streaming response)
        let builder = reqwest::Client::builder()
            .pool_max_idle_per_host(config.http2.pool_max_idle_per_host)
            .pool_idle_timeout(config.http2.pool_idle_timeout);
        // Note: We don't set timeout here because we want the stream to live longer

        let client = builder.build()?;

        Ok(Self { client, config })
    }

    /// Create from environment configuration
    pub fn from_env() -> Result<Self> {
        Self::with_config(H2BidiConfig::from_env())
    }

    /// Get the current configuration
    pub fn config(&self) -> &H2BidiConfig {
        &self.config
    }
}

impl Default for H2BidiClient {
    fn default() -> Self {
        Self::new().expect("Failed to create H2Bidi client")
    }
}

#[async_trait]
impl H2BidiStreamingClient for H2BidiClient {
    async fn send_streaming_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>> {
        let mut req = self
            .client
            .post(&address)
            .header("Content-Type", "application/json")
            .header("Accept", "application/x-ndjson")
            .body(payload);

        // Add custom headers
        for (key, value) in headers {
            req = req.header(key, value);
        }

        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("H2Bidi request failed with status {}: {}", status, text);
        }

        // Convert reqwest BytesStream to our stream type
        let stream = response.bytes_stream();

        // Map the stream to convert reqwest errors to anyhow errors
        let mapped_stream = futures::StreamExt::map(stream, |result| {
            result.map_err(|e| anyhow::anyhow!("Stream error: {}", e))
        });

        Ok(Box::pin(mapped_stream))
    }

    fn transport_name(&self) -> &'static str {
        "h2bidi"
    }

    fn is_healthy(&self) -> bool {
        // HTTP client is stateless and always healthy if created successfully
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2bidi_client_creation() {
        let client = H2BidiClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_h2bidi_config_default() {
        let config = H2BidiConfig::default();
        assert_eq!(
            config.response_timeout,
            Duration::from_secs(DEFAULT_H2BIDI_RESPONSE_TIMEOUT_SECS)
        );
    }

    #[test]
    fn test_h2bidi_config_from_env() {
        // Set environment variable
        unsafe {
            std::env::set_var("DYN_H2BIDI_RESPONSE_TIMEOUT_SECS", "600");
        }

        let config = H2BidiConfig::from_env();
        assert_eq!(config.response_timeout, Duration::from_secs(600));

        // Clean up
        unsafe {
            std::env::remove_var("DYN_H2BIDI_RESPONSE_TIMEOUT_SECS");
        }
    }

    #[tokio::test]
    async fn test_h2bidi_client_invalid_url() {
        let client = H2BidiClient::new().unwrap();
        let result = client
            .send_streaming_request(
                "http://invalid-host-that-does-not-exist:9999/test".to_string(),
                Bytes::from(r#"{"test": "data"}"#),
                std::collections::HashMap::new(),
            )
            .await;
        assert!(result.is_err());
    }
}

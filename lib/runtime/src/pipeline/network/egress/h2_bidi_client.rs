// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 Bidirectional Streaming Client
//!
//! This module provides an HTTP/2 client that supports bidirectional streaming,
//! allowing a single connection to send a request and receive a streaming response.
//! This replaces the two-connection architecture (request + callback) with a simpler
//! single-connection model.

use crate::Result;
use bytes::Bytes;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;

/// Default connection timeout for HTTP/2 handshake
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Default response buffer size
const DEFAULT_RESPONSE_BUFFER_SIZE: usize = 8192;

/// HTTP/2 Performance Configuration Constants (matching http_router.rs)
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 100;
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 90;
const DEFAULT_HTTP2_KEEP_ALIVE_INTERVAL_SECS: u64 = 30;
const DEFAULT_HTTP2_KEEP_ALIVE_TIMEOUT_SECS: u64 = 10;

/// Configuration for H2Bidi client
///
/// This configuration is optimized for streaming responses. Unlike the standard
/// HTTP client, there is no request timeout since streaming responses can be
/// long-lived. Only the connection timeout is enforced.
#[derive(Debug, Clone)]
pub struct H2BidiConfig {
    /// Maximum number of idle connections per host
    pub pool_max_idle_per_host: usize,

    /// How long to keep idle connections alive
    pub pool_idle_timeout: Duration,

    /// Timeout for establishing a connection (TCP + TLS + HTTP/2 handshake)
    pub connect_timeout: Duration,

    /// HTTP/2 keep-alive ping interval
    pub keep_alive_interval: Duration,

    /// Timeout for keep-alive ping responses
    pub keep_alive_timeout: Duration,

    /// Size hint for response buffer
    pub response_buffer_size: usize,
}

impl Default for H2BidiConfig {
    fn default() -> Self {
        Self {
            pool_max_idle_per_host: DEFAULT_POOL_MAX_IDLE_PER_HOST,
            pool_idle_timeout: Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS),
            connect_timeout: Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS),
            keep_alive_interval: Duration::from_secs(DEFAULT_HTTP2_KEEP_ALIVE_INTERVAL_SECS),
            keep_alive_timeout: Duration::from_secs(DEFAULT_HTTP2_KEEP_ALIVE_TIMEOUT_SECS),
            response_buffer_size: DEFAULT_RESPONSE_BUFFER_SIZE,
        }
    }
}

impl H2BidiConfig {
    /// Create configuration from environment variables
    ///
    /// Reads from:
    /// - `DYN_HTTP2_POOL_MAX_IDLE_PER_HOST`
    /// - `DYN_HTTP2_POOL_IDLE_TIMEOUT_SECS`
    /// - `DYN_HTTP2_CONNECT_TIMEOUT_SECS`
    /// - `DYN_HTTP2_KEEP_ALIVE_INTERVAL_SECS`
    /// - `DYN_HTTP2_KEEP_ALIVE_TIMEOUT_SECS`
    /// - `DYN_H2BIDI_RESPONSE_BUFFER_SIZE`
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("DYN_HTTP2_POOL_MAX_IDLE_PER_HOST")
            && let Ok(pool_size) = val.parse::<usize>()
        {
            config.pool_max_idle_per_host = pool_size;
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_POOL_IDLE_TIMEOUT_SECS")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.pool_idle_timeout = Duration::from_secs(timeout);
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_CONNECT_TIMEOUT_SECS")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.connect_timeout = Duration::from_secs(timeout);
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_KEEP_ALIVE_INTERVAL_SECS")
            && let Ok(interval) = val.parse::<u64>()
        {
            config.keep_alive_interval = Duration::from_secs(interval);
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_KEEP_ALIVE_TIMEOUT_SECS")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.keep_alive_timeout = Duration::from_secs(timeout);
        }

        if let Ok(val) = std::env::var("DYN_H2BIDI_RESPONSE_BUFFER_SIZE")
            && let Ok(size) = val.parse::<usize>()
        {
            config.response_buffer_size = size;
        }

        config
    }
}

/// HTTP/2 bidirectional streaming client
///
/// This client sends a request and receives a streaming response on the same
/// HTTP/2 connection. Unlike the standard HTTP client which expects a single
/// response body, this client returns a stream of bytes that can be processed
/// incrementally (e.g., as NDJSON lines).
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
        let builder = reqwest::Client::builder()
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .pool_idle_timeout(config.pool_idle_timeout)
            .connect_timeout(config.connect_timeout);
            // Note: reqwest automatically negotiates HTTP/2 via ALPN when available
            // We don't set a request timeout since streaming responses are long-lived

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

    /// Send a streaming request and receive a streaming response
    ///
    /// This method sends an HTTP/2 POST request and returns a stream of bytes
    /// from the response body. The stream can be processed incrementally as
    /// data arrives from the server.
    ///
    /// # Arguments
    ///
    /// * `url` - The full URL to send the request to
    /// * `body` - The request body (typically JSON-serialized request message)
    /// * `headers` - Additional headers to include in the request
    ///
    /// # Returns
    ///
    /// Returns a boxed stream of `Result<Bytes>` that yields chunks as they arrive.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Connection to the server fails
    /// - The server returns a non-2xx status code
    pub async fn send_streaming_request(
        &self,
        url: String,
        body: Bytes,
        headers: HashMap<String, String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>> {
        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body);

        // Add custom headers
        for (key, value) in headers {
            request = request.header(key, value);
        }

        let response = request.send().await?;

        // Check status before returning stream
        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "HTTP error: status {}, body: {}",
                status.as_u16(),
                body_text
            );
        }

        // Return the bytes stream
        let stream = response.bytes_stream();

        // Map the stream to convert reqwest::Error to anyhow::Error
        let mapped_stream =
            futures::StreamExt::map(stream, |result| result.map_err(anyhow::Error::from));

        Ok(Box::pin(mapped_stream))
    }
}

impl Default for H2BidiClient {
    fn default() -> Self {
        Self::new().expect("Failed to create H2Bidi client")
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
        assert_eq!(config.pool_max_idle_per_host, DEFAULT_POOL_MAX_IDLE_PER_HOST);
        assert_eq!(
            config.pool_idle_timeout,
            Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS)
        );
        assert_eq!(
            config.connect_timeout,
            Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS)
        );
        assert_eq!(config.response_buffer_size, DEFAULT_RESPONSE_BUFFER_SIZE);
    }

    #[test]
    fn test_h2bidi_config_from_env() {
        // Set environment variables
        unsafe {
            std::env::set_var("DYN_HTTP2_POOL_MAX_IDLE_PER_HOST", "200");
            std::env::set_var("DYN_HTTP2_CONNECT_TIMEOUT_SECS", "15");
            std::env::set_var("DYN_H2BIDI_RESPONSE_BUFFER_SIZE", "16384");
        }

        let config = H2BidiConfig::from_env();

        assert_eq!(config.pool_max_idle_per_host, 200);
        assert_eq!(config.connect_timeout, Duration::from_secs(15));
        assert_eq!(config.response_buffer_size, 16384);

        // Clean up
        unsafe {
            std::env::remove_var("DYN_HTTP2_POOL_MAX_IDLE_PER_HOST");
            std::env::remove_var("DYN_HTTP2_CONNECT_TIMEOUT_SECS");
            std::env::remove_var("DYN_H2BIDI_RESPONSE_BUFFER_SIZE");
        }
    }

    #[test]
    fn test_h2bidi_client_with_config() {
        let config = H2BidiConfig {
            pool_max_idle_per_host: 50,
            pool_idle_timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(5),
            keep_alive_interval: Duration::from_secs(20),
            keep_alive_timeout: Duration::from_secs(5),
            response_buffer_size: 4096,
        };

        let client = H2BidiClient::with_config(config.clone());
        assert!(client.is_ok());

        let client = client.unwrap();
        assert_eq!(client.config().pool_max_idle_per_host, 50);
        assert_eq!(client.config().connect_timeout, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_h2bidi_client_invalid_url() {
        let client = H2BidiClient::new().unwrap();
        let result = client
            .send_streaming_request(
                "http://invalid-host-that-does-not-exist:9999/test".to_string(),
                Bytes::from("test"),
                HashMap::new(),
            )
            .await;
        assert!(result.is_err());
    }
}

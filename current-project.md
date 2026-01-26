# Plan: HTTP/2 Bidirectional Streaming for AddressedPushRouter

## Goal
Enable `AddressedPushRouter` to receive responses over HTTP/2 response body (newline-delimited) instead of requiring a separate TCP callback connection, while maintaining backward compatibility.

## Key Constraint
**Always register with TcpStreamServer BEFORE sending the HTTP request** (maintains current behavior). If server responds with `x-bidi-stream: true`, **deregister** from TcpStreamServer and use HTTP response body instead.

## Key Changes

### 1. Add `deregister` method to `TcpStreamServer` (tcp/server.rs)

**Why this is needed**: Despite the comment at line 207 claiming RAII cleanup, there's no `Drop` impl for `RegisteredStream`. Cleanup only happens when a TCP connection arrives (line 434). Without explicit deregister, entries leak when bidi streaming bypasses TCP callback.

```rust
impl TcpStreamServer {
    /// Deregister a pending response stream by connection_info
    pub async fn deregister_response_stream(&self, connection_info: &TcpStreamConnectionInfo) {
        // deregister the stream
    }
}
```

### 2. Modify `RequestPlaneClient` trait (unified_client.rs)

Add new types and method:

```rust
/// Stream of response bytes
pub type ResponseByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

/// Context for stream handling
pub struct StreamContext {
    pub engine_context: Arc<dyn AsyncEngineContext>,
    pub tcp_server: Arc<TcpStreamServer>,
    /// Pre-registered pending connection (registration happens before send)
    pub pending_recv_stream: RegisteredStream<StreamReceiver>,
}

impl Drop for StreamContext {
    fn drop(&mut self) {
        // deregister
    }
}

impl StreamContext {
    pub fn await_transport_handshake(&self) -> Result<ResponseByteStream> {
        // await the transport handshake
    }
}

#[async_trait]
pub trait RequestPlaneClient: Send + Sync {
    /// Send request and return response stream
    ///
    /// For HTTP: checks x-bidi-stream header to decide stream source
    /// For TCP/NATS: always uses TCP callback
    async fn send_request_streaming(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
        stream_context: StreamContext,
    ) -> Result<ResponseByteStream>;

    // Existing methods unchanged...
}
```

### 3. Implement `send_request_streaming` for `HttpRequestClient` (http_router.rs)

```rust
impl HttpRequestClient {
    async fn send_request_streaming(
        &self,
        address: String,
        payload: Bytes,
        mut headers: Headers,
        stream_context: StreamContext,
    ) -> Result<ResponseByteStream> {
        // Signal we support bidi streaming
        headers.insert("x-accept-bidi-stream".to_string(), "true".to_string());

        let response = self.client.post(&address)
            .header("Content-Type", "application/octet-stream")
            .headers(/* convert headers */)
            .body(payload)
            .send().await?;

        if !response.status().is_success() {
            bail!("HTTP request failed: {}", response.status());
        }

        // Check if server supports bidi streaming
        let is_bidi = response.headers()
            .get("x-bidi-stream")
            .map(|v| v == "true")
            .unwrap_or(false);

        if is_bidi {
            // Deregister from TCP server - we won't need the callback
            let subject = stream_context.pending_recv_stream.connection_info
                .as_tcp().map(|info| info.subject.clone());
            if let Some(subject) = subject {
                stream_context.tcp_server.deregister(&subject).await;
            }

            // Response body is newline-delimited stream
            // Use tokio_util::codec::LinesCodec for line parsing
            use tokio_util::codec::{FramedRead, LinesCodec};
            use tokio_util::io::StreamReader;

            let body_stream = response.bytes_stream()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
            let reader = StreamReader::new(body_stream);
            let lines = FramedRead::new(reader, LinesCodec::new());

            Ok(Box::pin(lines.map(|line| {
                line.map(|s| Bytes::from(s))
                    .map_err(|e| anyhow::anyhow!("Line codec error: {}", e))
            })))
        } else {
            // Use TCP callback as before
            let (_, response_stream_provider) = stream_context.pending_recv_stream.into_parts();
            let response_stream = response_stream_provider.await
                .map_err(|_| PipelineError::DetachedStreamReceiver)?
                .map_err(PipelineError::ConnectionFailed)?;

            Ok(Box::pin(ReceiverStream::new(response_stream.rx).map(Ok)))
        }
    }
}
```

### 4. Update `TcpRequestClient` and `NatsRequestClient`

```rust
impl RequestPlaneClient for TcpRequestClient {
    async fn send_request_streaming(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
        stream_context: StreamContext,
    ) -> Result<ResponseByteStream> {
        // Send request via TCP
        self.send_request(address, payload, headers).await?;

        // Always use TCP callback (no bidi support)
        let (_, response_stream_provider) = stream_context.pending_recv_stream.into_parts();
        let response_stream = stream_context.await_transport_handshake()?;

        Ok(response_stream)
    }
}
```

### 6. Update `AddressedPushRouter::generate()` (addressed_router.rs)

```rust
async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
    /// ... keep similar logic as before

    // Build stream context with pre-registered pending connection
    let stream_context = StreamContext {
        engine_context: engine_ctx.clone(),
        tcp_server: self.resp_transport.clone(),
        pending_recv_stream,
    };

    // Send request via streaming API
    let response_stream = self.req_client
        .send_request_streaming(address, buffer, headers, stream_context)
        .await?;

    // ... keep similar logic as before, but use the response_stream

    Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
}
```

## Protocol Flow

```
1. Register with TcpStreamServer → get subject UUID
2. Build request with connection_info (includes subject)
3. Send HTTP request with x-accept-bidi-stream: true
4. Receive HTTP response
   ├─ If x-bidi-stream: true
   │   ├─ Deregister subject from TcpStreamServer
   │   └─ Stream response from HTTP body (newline-delimited)
   └─ If x-bidi-stream: false/missing
       └─ Await TCP callback as before
```

## Protocol Headers

- **Request**: `x-accept-bidi-stream: true` - client supports bidi streaming
- **Response**: `x-bidi-stream: true` - server is using bidi streaming

## Response Format (when bidi enabled)

Newline-delimited JSON, each line is a `NetworkStreamWrapper<U>`:
```
{"data": {...}, "complete_final": false}\n
{"data": {...}, "complete_final": false}\n
{"data": null, "complete_final": true}\n
```

## Integration Tests

- Create a mock HTTP server that interacts with the updated `AddressedPushRouter` and `HttpRequestClient` when `x-bidi-stream: true`
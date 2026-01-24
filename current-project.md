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
    /// Deregister a pending response stream by subject
    /// Called when bidi streaming is used instead of TCP callback
    pub async fn deregister(&self, subject: &str) {
        let mut state = self.state.lock().await;
        state.rx_subjects.remove(subject);
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
        let response_stream = response_stream_provider.await
            .map_err(|_| PipelineError::DetachedStreamReceiver)?
            .map_err(PipelineError::ConnectionFailed)?;

        Ok(Box::pin(ReceiverStream::new(response_stream.rx).map(Ok)))
    }
}
```

### 6. Update `AddressedPushRouter::generate()` (addressed_router.rs)

```rust
async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
    // ... extract request, address, engine_ctx ...

    // 1. Register with TCP server FIRST (unchanged)
    let options = StreamOptions::builder()
        .context(engine_ctx.clone())
        .enable_request_stream(false)
        .enable_response_stream(true)
        .build()?;

    let pending_connections = self.resp_transport.register(options).await;
    let pending_recv_stream = match pending_connections.into_parts() {
        (None, Some(recv)) => recv,
        _ => panic!("Invalid registration"),
    };

    // 2. Get connection_info for control message (unchanged)
    let connection_info = pending_recv_stream.connection_info.clone();

    // 3. Build control message with connection_info (unchanged)
    let control_message = RequestControlMessage { ... connection_info ... };
    let buffer = /* encode as before */;

    // 4. Build stream context with pre-registered pending connection
    let stream_context = StreamContext {
        engine_context: engine_ctx.clone(),
        tcp_server: self.resp_transport.clone(),
        pending_recv_stream,
    };

    // 5. Send request via streaming API
    let response_stream = self.req_client
        .send_request_streaming(address, buffer, headers, stream_context)
        .await?;

    // 6. Process stream (unchanged - same NetworkStreamWrapper<U> decoding)
    let stream = response_stream.filter_map(move |res| {
        match res {
            Ok(bytes) => /* deserialize NetworkStreamWrapper<U> as before */,
            Err(e) => Some(U::from_err(e.into())),
        }
    });

    Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
}
```

## Files to Modify

| File | Changes |
|------|---------|
| `lib/runtime/src/pipeline/network/tcp/server.rs` | Add `deregister()` method |
| `lib/runtime/src/pipeline/network/egress/unified_client.rs` | Add `ResponseByteStream`, `StreamContext`, `send_request_streaming` |
| `lib/runtime/src/pipeline/network/egress/http_router.rs` | Implement bidi streaming (use `LinesCodec`) |
| `lib/runtime/src/pipeline/network/egress/addressed_router.rs` | Use `send_request_streaming`, pass pre-registered connection |
| `lib/runtime/src/pipeline/network/egress/tcp_client.rs` | Implement `send_request_streaming` (use TCP callback) |
| `lib/runtime/src/pipeline/network/egress/nats_client.rs` | Implement `send_request_streaming` (use TCP callback) |

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

## Verification

1. **Integration tests**:
    - HTTP bidi: server returns `x-bidi-stream: true`, verify deregister called
    - HTTP fallback: server without header, verify TCP callback used
2. **Regression tests**: TCP and NATS transports unchanged behavior

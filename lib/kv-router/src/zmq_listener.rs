// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ-based KV event listener for receiving cache events from vLLM workers.
//!
//! This module provides functionality to listen for KV cache events over ZMQ,
//! parse msgpack-encoded event batches, and convert them to the internal
//! `KvCacheEvent` representation used by the router.
//!
//! # Example
//!
//! ```ignore
//! use dynamo_kv_router::zmq_listener::start_zmq_listener;
//! use tokio::sync::mpsc;
//! use tokio_util::sync::CancellationToken;
//!
//! let (tx, mut rx) = mpsc::unbounded_channel();
//! let cancel = CancellationToken::new();
//!
//! tokio::spawn(start_zmq_listener(
//!     "tcp://localhost:5555".to_string(),
//!     "kv-events".to_string(),
//!     tx,
//!     cancel,
//!     64, // kv_block_size
//! ));
//!
//! while let Some(event) = rx.recv().await {
//!     println!("Received event: {:?}", event);
//! }
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rmp_serde as rmps;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv, SubSocket};

use crate::protocols::{
    BlockExtraInfo, DpRank, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoredBlockData, KvCacheStoreData, compute_block_hash_for_seq,
};

// Error handling configuration for ZMQ operations
/// Initial backoff delay in milliseconds for retrying failed ZMQ operations.
pub const INITIAL_BACKOFF_MS: u64 = 10;
/// Maximum backoff delay in milliseconds.
pub const MAX_BACKOFF_MS: u64 = 5000;
/// Maximum number of consecutive errors before giving up.
pub const MAX_CONSECUTIVE_ERRORS: u32 = 10;
/// Maximum exponent for exponential backoff (caps at 2^8 = 256x multiplier).
pub const MAX_BACKOFF_EXPONENT: u32 = 8;

/// Configuration for connecting to a ZMQ KV event source.
#[derive(Debug, Clone)]
pub struct ZmqConfig {
    /// ZMQ endpoint URL (e.g., "tcp://localhost:5555")
    pub endpoint: String,
    /// ZMQ topic to subscribe to (empty string for all topics)
    pub topic: String,
}

impl ZmqConfig {
    /// Create a new ZMQ configuration.
    pub fn new(endpoint: impl Into<String>, topic: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            topic: topic.into(),
        }
    }
}

/// Calculate exponential backoff duration based on consecutive error count.
///
/// Returns a delay in milliseconds that increases exponentially with the number
/// of consecutive errors, capped at [`MAX_BACKOFF_MS`].
pub fn calculate_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_BACKOFF_EXPONENT)),
        MAX_BACKOFF_MS,
    )
}

/// Start a ZMQ listener that receives KV cache events and sends them through the channel.
///
/// This function connects to a ZMQ PUB socket, subscribes to the specified topic,
/// and processes incoming msgpack-encoded event batches. Events are converted to
/// `KvCacheEvent` and sent through the provided channel.
///
/// # Arguments
///
/// * `zmq_endpoint` - The ZMQ endpoint to connect to (e.g., "tcp://localhost:5555")
/// * `zmq_topic` - The topic to subscribe to (empty string for all topics)
/// * `tx` - Channel sender for forwarding parsed events
/// * `cancellation_token` - Token for graceful shutdown
/// * `kv_block_size` - Size of KV cache blocks in tokens
///
/// # Notes
///
/// The listener implements exponential backoff on errors and will terminate
/// after `MAX_CONSECUTIVE_ERRORS` consecutive failures.
pub async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let warning_count = Arc::new(AtomicU32::new(0));

    let mut socket = SubSocket::new();

    // Subscribe to the requested topic (empty string == all topics)
    if let Err(e) = socket.subscribe(&zmq_topic).await {
        tracing::error!("Failed to subscribe on ZMQ socket: {}", e);
        return;
    }

    if let Err(e) = socket.connect(&zmq_endpoint).await {
        tracing::error!("Failed to connect ZMQ SUB socket: {}", e);
        return;
    }

    let mut consecutive_errors = 0u32;
    #[allow(unused_assignments)]
    let mut exit_reason = "unknown";
    let mut messages_processed = 0u64;

    'main: loop {
        tokio::select! {
            biased;

            // Check for cancellation
            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                exit_reason = "cancellation token cancelled";
                break 'main;
            }

            // Receive message
            msg_result = socket.recv() => {
                let Ok(msg) = msg_result else {
                    let e = msg_result.unwrap_err();
                    consecutive_errors += 1;

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            error=%e,
                            consecutive_errors=%consecutive_errors,
                            "Too many consecutive ZMQ errors, terminating listener"
                        );
                        exit_reason = "too many consecutive errors";
                        break 'main;
                    }

                    // Simple exponential backoff with max exponent to prevent overflow
                    let backoff_ms = calculate_backoff_ms(consecutive_errors);

                    tracing::warn!(
                        error=%e,
                        consecutive_errors=%consecutive_errors,
                        backoff_ms=%backoff_ms,
                        "Error reading from ZMQ socket, applying exponential backoff"
                    );

                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                };
                // Reset error count on successful message
                consecutive_errors = 0;

                // We expect multipart frames: [topic, seq, payload]
                let mut frames: Vec<Vec<u8>> = msg.into_vec().into_iter().map(|frame| frame.to_vec()).collect();

                if frames.len() != 3 {
                    tracing::warn!("Received unexpected ZMQ frame count: expected 3, actual {}", frames.len());
                    continue;
                }

                // Extract the payload and sequence number.
                let payload = frames.pop().unwrap();
                let seq_bytes = frames.pop().unwrap();

                if seq_bytes.len() != 8 {
                    tracing::warn!("Invalid sequence number byte length: expected 8, actual {}", seq_bytes.len());
                    continue;
                }

                let seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());

                // Decode our batch of events.
                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
                let Ok(batch) = batch_result else {
                    let e = batch_result.unwrap_err();
                    tracing::warn!("Failed to decode KVEventBatch msgpack: {e}");
                    continue;
                };

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    seq,
                    batch.data_parallel_rank.unwrap_or(0)
                );

                let dp_rank = batch.data_parallel_rank.unwrap_or(0) as u32;
                for raw_event in batch.events.into_iter() {
                    let event = convert_event(raw_event, seq, kv_block_size, dp_rank, &warning_count);
                    if tx.send(event).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        exit_reason = "channel receiver dropped";
                        break 'main;
                    }
                    messages_processed += 1;
                }
            }
        }
    }
    tracing::debug!(
        "ZMQ listener exiting, reason: {}, messages processed: {}",
        exit_reason,
        messages_processed
    );
}

/// Convert a raw event coming from the ZMQ channel into the internal
/// [`KvCacheEvent`] representation used by the router.
pub fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    dp_rank: DpRank,
    warning_count: &Arc<AtomicU32>,
) -> KvCacheEvent {
    match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_id,
            block_mm_infos,
            ..
        } => {
            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            let block_hashes_u64: Vec<u64> = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash
                        .map(BlockHashValue::into_u64)
                        .map(ExternalSequenceBlockHash::from),
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes_u64,
                        lora_id.unwrap_or(0),
                        warning_count,
                        block_mm_infos.as_deref(),
                    ),
                }),
                dp_rank,
            }
        }
        RawKvEvent::BlockRemoved { block_hashes, .. } => {
            let hashes = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .map(ExternalSequenceBlockHash::from)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
                dp_rank,
            }
        }
        RawKvEvent::AllBlocksCleared => KvCacheEvent {
            event_id,
            data: KvCacheEventData::Cleared,
            dp_rank,
        },
    }
}

/// Create a stored block from its component parts.
///
/// # Arguments
///
/// * `kv_block_size` - Size of KV cache blocks
/// * `block_hash` - External sequence block hash
/// * `token_ids` - Token IDs in this block
/// * `_lora_id` - LoRA adapter ID (currently unused)
/// * `mm_extra_info` - Optional multimodal metadata
pub fn create_stored_block_from_parts(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    _lora_id: u64,
    mm_extra_info: Option<BlockExtraInfo>,
) -> KvCacheStoredBlockData {
    // Compute tokens_hash including MM info if present
    let block_mm_infos = mm_extra_info.as_ref().map(|info| vec![Some(info.clone())]);
    let tokens_hash =
        compute_block_hash_for_seq(token_ids, kv_block_size, block_mm_infos.as_deref())[0];

    tracing::trace!(
        "Creating stored block: external_block_hash={}, tokens_hash={}, token_ids={:?}, kv_block_size={}, mm_extra_info={:?}",
        block_hash,
        tokens_hash.0,
        token_ids,
        kv_block_size,
        mm_extra_info
    );
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
        mm_extra_info,
    }
}

/// Create stored blocks from token IDs and block hashes.
///
/// # Arguments
///
/// * `kv_block_size` - Size of KV cache blocks
/// * `token_ids` - All token IDs across blocks
/// * `num_block_tokens` - Number of tokens in each block
/// * `block_hashes` - External hash for each block
/// * `lora_id` - LoRA adapter ID
/// * `warning_count` - Counter for rate-limiting warnings
/// * `block_mm_infos` - Optional multimodal metadata per block
pub fn create_stored_blocks(
    kv_block_size: u32,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[u64],
    lora_id: u64,
    warning_count: &Arc<AtomicU32>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for (block_idx, (num_tokens_it, block_hash_it)) in
        num_block_tokens.iter().zip(block_hashes.iter()).enumerate()
    {
        if *num_tokens_it != kv_block_size as u64 {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens_it
                );
            }
            break;
        }

        let tokens = &token_ids[token_offset..(token_offset + *num_tokens_it as usize)];
        let mm_extra_info = block_mm_infos
            .and_then(|infos| infos.get(block_idx))
            .and_then(|opt| opt.clone());

        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash_it,
            tokens,
            lora_id,
            mm_extra_info,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

// -------------------------------------------------------------------------
// Types mirroring the Python msgspec-defined structures
// -------------------------------------------------------------------------

/// A batch of KV cache events received from a vLLM worker.
///
/// This struct mirrors the Python msgspec-defined structure used by vLLM
/// to send KV cache events over ZMQ.
#[derive(Debug, Serialize)]
pub struct KvEventBatch {
    /// Timestamp of the batch
    pub ts: f64,
    /// Events in this batch
    pub events: Vec<RawKvEvent>,
    /// Data parallel rank of the worker that sent this batch
    #[serde(alias = "dp_rank")]
    pub data_parallel_rank: Option<i32>,
}

impl<'de> Deserialize<'de> for KvEventBatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize from array format: [timestamp, [events], data_parallel_rank]
        let arr: (f64, Vec<RawKvEvent>, Option<i32>) = Deserialize::deserialize(deserializer)?;
        Ok(KvEventBatch {
            ts: arr.0,
            events: arr.1,
            data_parallel_rank: arr.2,
        })
    }
}

/// A block hash value that can be either signed or unsigned.
///
/// vLLM can send block hashes as either signed or unsigned 64-bit integers,
/// depending on Python's handling of large integers.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(untagged)]
pub enum BlockHashValue {
    /// Signed 64-bit integer hash (from Python's negative integer handling)
    Signed(i64),
    /// Unsigned 64-bit integer hash
    Unsigned(u64),
}

impl BlockHashValue {
    /// Convert the hash value to an unsigned 64-bit integer.
    pub fn into_u64(self) -> u64 {
        match self {
            BlockHashValue::Signed(v) => v as u64,
            BlockHashValue::Unsigned(v) => v,
        }
    }
}

/// A raw KV cache event as received from vLLM over ZMQ.
///
/// This enum represents the different types of events that can be received:
/// - `BlockStored`: One or more blocks were added to the KV cache
/// - `BlockRemoved`: One or more blocks were removed from the KV cache
/// - `AllBlocksCleared`: The entire KV cache was cleared
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum RawKvEvent {
    /// A block was stored in the KV cache
    BlockStored {
        /// Hashes of the stored blocks
        block_hashes: Vec<BlockHashValue>,
        /// Hash of the parent block (for prefix caching)
        parent_block_hash: Option<BlockHashValue>,
        /// Token IDs stored in these blocks
        token_ids: Vec<u32>,
        /// Size of each block in tokens
        block_size: usize,
        /// LoRA adapter ID if applicable
        lora_id: Option<u64>,
        /// Storage medium (optional)
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
        /// LoRA adapter name (optional)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        /// Multimodal block metadata (optional)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    },
    /// A block was removed from the KV cache
    BlockRemoved {
        /// Hashes of the removed blocks
        block_hashes: Vec<BlockHashValue>,
        /// Storage medium (optional)
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    /// All blocks were cleared from the KV cache
    AllBlocksCleared,
}

/// Custom deserializer for RawKvEvent that handles both map and sequence formats,
/// and ignores unknown fields for forward compatibility.
impl<'de> Deserialize<'de> for RawKvEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(RawKvEventVisitor)
    }
}

struct RawKvEventVisitor;

impl<'de> Visitor<'de> for RawKvEventVisitor {
    type Value = RawKvEvent;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a kv event encoded as a tagged map or sequence")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut event_type: Option<String> = None;
        let mut block_hashes: Option<Vec<BlockHashValue>> = None;
        let mut parent_block_hash: Option<Option<BlockHashValue>> = None;
        let mut token_ids: Option<Vec<u32>> = None;
        let mut block_size: Option<usize> = None;
        let mut lora_id: Option<Option<u64>> = None;
        let mut medium: Option<Option<String>> = None;
        let mut lora_name: Option<Option<String>> = None;
        let mut block_mm_infos: Option<Option<Vec<Option<BlockExtraInfo>>>> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => event_type = Some(map.next_value()?),
                "block_hashes" => block_hashes = Some(map.next_value()?),
                "parent_block_hash" => parent_block_hash = Some(map.next_value()?),
                "token_ids" => token_ids = Some(map.next_value()?),
                "block_size" => block_size = Some(map.next_value()?),
                "lora_id" => lora_id = Some(map.next_value()?),
                "medium" => medium = Some(map.next_value()?),
                "lora_name" => lora_name = Some(map.next_value()?),
                "block_mm_infos" => block_mm_infos = Some(map.next_value()?),
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        match event_type.as_deref() {
            Some("BlockStored") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                let token_ids = token_ids.ok_or_else(|| de::Error::missing_field("token_ids"))?;
                let block_size =
                    block_size.ok_or_else(|| de::Error::missing_field("block_size"))?;
                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash: parent_block_hash.unwrap_or(None),
                    token_ids,
                    block_size,
                    lora_id: lora_id.unwrap_or(None),
                    medium: medium.unwrap_or(None),
                    lora_name: lora_name.unwrap_or(None),
                    block_mm_infos: block_mm_infos.unwrap_or(None),
                })
            }
            Some("BlockRemoved") => {
                let block_hashes =
                    block_hashes.ok_or_else(|| de::Error::missing_field("block_hashes"))?;
                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium: medium.unwrap_or(None),
                })
            }
            Some("AllBlocksCleared") => Ok(RawKvEvent::AllBlocksCleared),
            Some(other) => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared"],
            )),
            None => Err(de::Error::missing_field("type")),
        }
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let tag: Option<String> = seq.next_element()?;
        let Some(tag) = tag else {
            return Err(de::Error::invalid_length(
                0,
                &"sequence must start with event tag",
            ));
        };

        match tag.as_str() {
            "BlockStored" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let parent_block_hash: Option<BlockHashValue> = seq.next_element()?.unwrap_or(None);
                let token_ids: Vec<u32> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &"missing token_ids"))?;
                let block_size: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &"missing block_size"))?;
                let lora_id: Option<u64> = seq.next_element()?.unwrap_or(None);
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);
                let lora_name: Option<String> = seq.next_element()?.unwrap_or(None);
                let block_mm_infos: Option<Vec<Option<BlockExtraInfo>>> =
                    seq.next_element()?.unwrap_or(None);

                // Drain any extra elements for forward compatibility
                while seq.next_element::<IgnoredAny>()?.is_some() {}

                Ok(RawKvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids,
                    block_size,
                    lora_id,
                    medium,
                    lora_name,
                    block_mm_infos,
                })
            }
            "BlockRemoved" => {
                let block_hashes: Vec<BlockHashValue> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"missing block_hashes"))?;
                let medium: Option<String> = seq.next_element()?.unwrap_or(None);

                while seq.next_element::<IgnoredAny>()?.is_some() {}

                Ok(RawKvEvent::BlockRemoved {
                    block_hashes,
                    medium,
                })
            }
            "AllBlocksCleared" => {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawKvEvent::AllBlocksCleared)
            }
            other => Err(de::Error::unknown_variant(
                other,
                &["BlockStored", "BlockRemoved", "AllBlocksCleared"],
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_backoff_ms() {
        assert_eq!(calculate_backoff_ms(0), 10);
        assert_eq!(calculate_backoff_ms(1), 20);
        assert_eq!(calculate_backoff_ms(2), 40);
        assert_eq!(calculate_backoff_ms(8), 2560); // 10 * 2^8 = 2560
        // MAX_BACKOFF_EXPONENT caps at 8, so values above that are capped
        assert_eq!(calculate_backoff_ms(9), 2560); // Capped by MAX_BACKOFF_EXPONENT
        // Very high values should also cap at MAX_BACKOFF_MS
        assert_eq!(calculate_backoff_ms(100), 2560); // Capped by MAX_BACKOFF_EXPONENT
    }

    #[test]
    fn test_zmq_config() {
        let config = ZmqConfig::new("tcp://localhost:5555", "kv-events");
        assert_eq!(config.endpoint, "tcp://localhost:5555");
        assert_eq!(config.topic, "kv-events");
    }

    #[test]
    fn test_create_stored_block_from_parts() {
        let kv_block_size = 4;
        let token_ids = vec![10, 20, 30, 40];
        let blk_hash = 0xdead_beef;

        let stored = create_stored_block_from_parts(kv_block_size, blk_hash, &token_ids, 0, None);

        assert_eq!(stored.block_hash.0, blk_hash);
        let expected_hash = compute_block_hash_for_seq(&token_ids, 4, None)[0];
        assert_eq!(stored.tokens_hash, expected_hash);
        assert!(stored.mm_extra_info.is_none());
    }

    #[test]
    fn test_create_stored_blocks_ok() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let num_block_tokens = vec![4_u64, 4_u64];
        let block_hashes = vec![111_u64, 222_u64];

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            0,
            &Arc::new(AtomicU32::new(0)),
            None,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_hash.0, 111);
        assert_eq!(blocks[1].block_hash.0, 222);
    }

    #[test]
    fn test_create_stored_blocks_wrong_size_triggers_warning() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7];
        let num_block_tokens = vec![4_u64, 3_u64];
        let block_hashes = vec![111_u64, 222_u64];
        let warning_count = Arc::new(AtomicU32::new(0));

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            0,
            &warning_count,
            None,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(warning_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_block_hash_value_conversion() {
        let signed = BlockHashValue::Signed(-1);
        let unsigned = BlockHashValue::Unsigned(42);

        assert_eq!(signed.into_u64(), u64::MAX); // -1 as u64
        assert_eq!(unsigned.into_u64(), 42);
    }
}

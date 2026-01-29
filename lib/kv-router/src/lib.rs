// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Router - Radix tree data structures for LLM KV cache routing.
//!
//! This crate provides the core radix tree implementation and protocols for
//! efficient KV cache lookup and routing in distributed LLM inference systems.
//!
//! # Features
//!
//! - **Core** (always available): Radix tree, protocols, indexer
//! - **`zmq`**: ZMQ-based event listener for receiving KV cache events from vLLM
//! - **`scheduler`**: Worker selection logic for KV-aware routing
//! - **`metrics`**: Prometheus metrics integration (requires `dynamo-runtime`)
//! - **`full`**: All features enabled
//!
//! # Example
//!
//! ```ignore
//! use dynamo_kv_router::{
//!     indexer::KvIndexer,
//!     protocols::{RouterEvent, compute_block_hash_for_seq},
//! };
//!
//! // Create an indexer
//! let indexer = KvIndexer::new(cancel_token, block_size, metrics);
//!
//! // Find matches for a token sequence
//! let hashes = compute_block_hash_for_seq(&tokens, block_size, None);
//! let overlaps = indexer.find_matches(hashes).await?;
//! ```

pub mod approx;
pub mod flat_hashmap;
pub mod indexer;
pub mod protocols;
pub mod radix_tree;
pub mod worker_config;

// Feature-gated modules
#[cfg(feature = "zmq")]
pub mod zmq_listener;

#[cfg(feature = "scheduler")]
pub mod scheduler;

// Re-export key types for convenience
pub use flat_hashmap::FlatHashMap;
pub use indexer::MaybeError;
pub use protocols::{
    KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, WorkerId,
    compute_block_hash_for_seq,
};
pub use radix_tree::RadixTree;
pub use worker_config::WorkerConfig;

// Feature-gated re-exports
#[cfg(feature = "zmq")]
pub use zmq_listener::{start_zmq_listener, ZmqConfig};

#[cfg(feature = "scheduler")]
pub use scheduler::{
    DefaultWorkerSelector, SchedulerConfig, SchedulerError, SelectionRequest, SelectionResult,
    WorkerSelector,
};

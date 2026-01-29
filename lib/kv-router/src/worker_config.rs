// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable worker configuration types for KV routing.
//!
//! This module provides a lightweight configuration struct that captures
//! the essential worker properties needed for KV cache routing decisions,
//! without requiring the full `dynamo-runtime` dependency.

use serde::{Deserialize, Serialize};

/// Configuration for a worker relevant to KV routing decisions.
///
/// This is a portable subset of the full model runtime configuration,
/// containing only the fields needed for:
/// - Worker selection/scheduling
/// - Data parallel routing
/// - Local indexer queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Number of data parallel replicas for this worker.
    /// Defaults to 1 (no data parallelism).
    #[serde(default = "default_data_parallel_size")]
    pub data_parallel_size: u32,

    /// Total number of KV cache blocks available on this worker.
    /// Used for load balancing decisions.
    #[serde(default)]
    pub total_kv_blocks: Option<u64>,

    /// Whether this worker has a local KV indexer enabled.
    /// When true, the router can query this worker directly for its KV state.
    #[serde(default)]
    pub enable_local_indexer: bool,
}

fn default_data_parallel_size() -> u32 {
    1
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            data_parallel_size: default_data_parallel_size(),
            total_kv_blocks: None,
            enable_local_indexer: false,
        }
    }
}

impl WorkerConfig {
    /// Create a new WorkerConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a WorkerConfig with specified data parallel size.
    pub fn with_data_parallel_size(mut self, size: u32) -> Self {
        self.data_parallel_size = size;
        self
    }

    /// Create a WorkerConfig with specified total KV blocks.
    pub fn with_total_kv_blocks(mut self, blocks: u64) -> Self {
        self.total_kv_blocks = Some(blocks);
        self
    }

    /// Create a WorkerConfig with local indexer enabled.
    pub fn with_local_indexer(mut self, enabled: bool) -> Self {
        self.enable_local_indexer = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WorkerConfig::default();
        assert_eq!(config.data_parallel_size, 1);
        assert_eq!(config.total_kv_blocks, None);
        assert!(!config.enable_local_indexer);
    }

    #[test]
    fn test_builder_pattern() {
        let config = WorkerConfig::new()
            .with_data_parallel_size(4)
            .with_total_kv_blocks(1024)
            .with_local_indexer(true);

        assert_eq!(config.data_parallel_size, 4);
        assert_eq!(config.total_kv_blocks, Some(1024));
        assert!(config.enable_local_indexer);
    }

    #[test]
    fn test_serialization() {
        let config = WorkerConfig::new()
            .with_data_parallel_size(2)
            .with_total_kv_blocks(512);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WorkerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.data_parallel_size, 2);
        assert_eq!(deserialized.total_kv_blocks, Some(512));
    }
}

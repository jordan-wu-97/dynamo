// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker selection and scheduling for KV-aware routing.
//!
//! This module provides the core worker selection logic for KV cache-aware routing,
//! allowing requests to be directed to workers that have the most relevant cached data.
//!
//! # Overview
//!
//! The scheduler uses a cost function that balances:
//! - **Overlap score**: How many blocks the worker already has cached
//! - **Load balancing**: Current prefill and decode load on each worker
//!
//! # Example
//!
//! ```
//! use dynamo_kv_router::scheduler::{DefaultWorkerSelector, WorkerSelector, SchedulerConfig};
//! use dynamo_kv_router::worker_config::WorkerConfig;
//! use dynamo_kv_router::protocols::{OverlapScores, WorkerWithDpRank};
//! use std::collections::HashMap;
//!
//! // Set up workers
//! let mut workers = HashMap::new();
//! workers.insert(1u64, Some(WorkerConfig::new().with_data_parallel_size(1)));
//! workers.insert(2u64, Some(WorkerConfig::new().with_data_parallel_size(1)));
//!
//! // Create overlap scores (from indexer.find_matches)
//! let mut overlaps = OverlapScores::new();
//! overlaps.scores.insert(WorkerWithDpRank::from_worker_id(1), 10);
//! overlaps.scores.insert(WorkerWithDpRank::from_worker_id(2), 5);
//!
//! // Select best worker
//! let selector = DefaultWorkerSelector::new(SchedulerConfig::default());
//! let request = dynamo_kv_router::scheduler::SelectionRequest {
//!     isl_tokens: 128,
//!     overlaps,
//!     decode_blocks: HashMap::new(),
//!     prefill_tokens: HashMap::new(),
//! };
//!
//! let result = selector.select_worker(&workers, &request, 64);
//! println!("Selected worker: {:?}", result.unwrap().worker);
//! ```

use std::collections::HashMap;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::protocols::{OverlapScores, WorkerId, WorkerWithDpRank};
use crate::worker_config::WorkerConfig;

/// Configuration for the KV router scheduler.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Weight for overlap score in the cost function.
    /// Higher values prioritize KV cache reuse over load balancing.
    pub overlap_score_weight: f64,

    /// Temperature for softmax sampling during worker selection.
    /// - 0.0: Always select the best worker (deterministic)
    /// - Higher values: More randomness in selection
    pub router_temperature: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            overlap_score_weight: 1.0,
            router_temperature: 0.0,
        }
    }
}

impl SchedulerConfig {
    /// Create a new scheduler config with specified overlap weight.
    pub fn with_overlap_weight(mut self, weight: f64) -> Self {
        self.overlap_score_weight = weight;
        self
    }

    /// Create a new scheduler config with specified temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.router_temperature = temp;
        self
    }
}

/// Errors that can occur during scheduling.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("no workers available")]
    NoWorkers,

    #[error("all workers busy")]
    AllWorkersBusy,
}

/// Result of worker selection.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// The selected worker with dp_rank
    pub worker: WorkerWithDpRank,

    /// Total blocks required to prefill this request
    pub required_blocks: u64,

    /// Number of blocks the selected worker may already have cached
    pub overlap_blocks: u32,
}

/// Request information for worker selection.
#[derive(Debug, Clone)]
pub struct SelectionRequest {
    /// Number of input tokens (ISL = Input Sequence Length)
    pub isl_tokens: usize,

    /// Overlap scores from the KV indexer
    pub overlaps: OverlapScores,

    /// Current decode blocks per worker (for load balancing)
    pub decode_blocks: HashMap<WorkerWithDpRank, usize>,

    /// Current prefill tokens per worker (for load balancing)
    pub prefill_tokens: HashMap<WorkerWithDpRank, usize>,
}

/// Optional overrides for scheduler config on a per-request basis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectionOverrides {
    /// Override for overlap_score_weight
    pub overlap_score_weight: Option<f64>,

    /// Override for router_temperature
    pub router_temperature: Option<f64>,
}

/// Trait for implementing custom worker selection logic.
pub trait WorkerSelector: Send + Sync {
    /// Select the best worker for a request.
    ///
    /// # Arguments
    ///
    /// * `workers` - Map of worker IDs to their configurations
    /// * `request` - The selection request containing overlap scores and load info
    /// * `block_size` - KV cache block size in tokens
    ///
    /// # Returns
    ///
    /// The selected worker and metadata about the selection.
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, Option<WorkerConfig>>,
        request: &SelectionRequest,
        block_size: u32,
    ) -> Result<SelectionResult, SchedulerError>;

    /// Select worker with per-request config overrides.
    fn select_worker_with_overrides(
        &self,
        workers: &HashMap<WorkerId, Option<WorkerConfig>>,
        request: &SelectionRequest,
        block_size: u32,
        _overrides: Option<&SelectionOverrides>,
    ) -> Result<SelectionResult, SchedulerError> {
        // Default implementation ignores overrides
        self.select_worker(workers, request, block_size)
    }
}

/// Default worker selector using a cost-based approach.
///
/// The cost function balances:
/// - Prefill cost: Tokens that need to be prefilled (lower is better)
/// - Decode load: Current decode blocks on the worker
///
/// Workers with more cached blocks have lower prefill cost.
#[derive(Debug, Clone)]
pub struct DefaultWorkerSelector {
    config: SchedulerConfig,
}

impl Default for DefaultWorkerSelector {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

impl DefaultWorkerSelector {
    /// Create a new default worker selector with the given config.
    pub fn new(config: SchedulerConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

impl WorkerSelector for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, Option<WorkerConfig>>,
        request: &SelectionRequest,
        block_size: u32,
    ) -> Result<SelectionResult, SchedulerError> {
        self.select_worker_with_overrides(workers, request, block_size, None)
    }

    fn select_worker_with_overrides(
        &self,
        workers: &HashMap<WorkerId, Option<WorkerConfig>>,
        request: &SelectionRequest,
        block_size: u32,
        overrides: Option<&SelectionOverrides>,
    ) -> Result<SelectionResult, SchedulerError> {
        if workers.is_empty() {
            return Err(SchedulerError::NoWorkers);
        }

        let isl = request.isl_tokens;
        assert!(isl > 0, "ISL must be positive");

        let request_blocks = isl.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;
        let decode_blocks = &request.decode_blocks;
        let prefill_tokens = &request.prefill_tokens;

        // Get effective config values (with overrides)
        let overlap_weight = overrides
            .and_then(|o| o.overlap_score_weight)
            .unwrap_or(self.config.overlap_score_weight);
        let temperature = overrides
            .and_then(|o| o.router_temperature)
            .unwrap_or(self.config.router_temperature);

        let mut worker_logits = HashMap::new();

        // Calculate logits for each worker with dp_rank
        for (worker_id, config) in workers.iter() {
            let data_parallel_size = config.as_ref().map(|c| c.data_parallel_size).unwrap_or(1);

            for dp_rank in 0..data_parallel_size {
                let worker = WorkerWithDpRank::new(*worker_id, dp_rank);

                // Get overlap for this worker (defaults to 0)
                let overlap = *overlaps.get(&worker).unwrap_or(&0) as usize;

                // Prefill tokens if scheduled to this worker
                // If prefill_tokens is provided, use it; otherwise compute from ISL and overlap
                let prefill_token = prefill_tokens.get(&worker).copied().unwrap_or_else(|| {
                    // Tokens that still need prefilling = ISL - (overlap * block_size)
                    isl.saturating_sub(overlap * block_size as usize)
                });
                let potential_prefill_block = (prefill_token as f64) / (block_size as f64);

                // Decode blocks if scheduled to this worker
                let decode_block = *decode_blocks
                    .get(&worker)
                    .unwrap_or(&(potential_prefill_block.floor() as usize))
                    as f64;

                // Calculate logit (lower is better)
                let logit = overlap_weight * potential_prefill_block + decode_block;

                worker_logits.insert(worker, logit);

                tracing::debug!(
                    "Cost for worker_id={} dp_rank={}: {:.3} = {:.1} * {:.3} + {:.3} (overlap={} blocks)",
                    worker.worker_id,
                    worker.dp_rank,
                    logit,
                    overlap_weight,
                    potential_prefill_block,
                    decode_block,
                    overlap
                );
            }
        }

        // Use softmax sampling to select worker(s)
        let candidates = softmax_sample(&worker_logits, temperature);

        // If multiple candidates (tied), use tree size as tie-breaker
        let best_worker = if candidates.len() > 1 {
            tracing::debug!(
                "Multiple workers tied with same logit, using tree size as tie-breaker"
            );
            *candidates
                .iter()
                .min_by_key(|worker| {
                    request
                        .overlaps
                        .tree_sizes
                        .get(worker)
                        .copied()
                        .unwrap_or(0)
                })
                .expect("candidates should not be empty")
        } else {
            candidates[0]
        };

        let best_overlap = *overlaps.get(&best_worker).unwrap_or(&0);
        let tree_size = request
            .overlaps
            .tree_sizes
            .get(&best_worker)
            .copied()
            .unwrap_or(0);

        let total_blocks_info = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.as_ref())
            .and_then(|cfg| cfg.total_kv_blocks)
            .map(|blocks| format!(", total blocks: {}", blocks))
            .unwrap_or_default();

        tracing::info!(
            "Selected worker: worker_id={} dp_rank={}, logit: {:.3}, cached blocks: {}, tree size: {}{}",
            best_worker.worker_id,
            best_worker.dp_rank,
            worker_logits[&best_worker],
            best_overlap,
            tree_size,
            total_blocks_info
        );

        Ok(SelectionResult {
            worker: best_worker,
            required_blocks: request_blocks as u64,
            overlap_blocks: best_overlap,
        })
    }
}

/// Softmax sampling for worker selection.
///
/// Returns a vec of workers: multiple if tied (temperature=0), single if sampled.
///
/// # Arguments
///
/// * `logits` - Map of workers to their logit values (lower is better)
/// * `temperature` - Sampling temperature (0 = deterministic, higher = more random)
fn softmax_sample(
    logits: &HashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> Vec<WorkerWithDpRank> {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Temperature 0: return all keys with the smallest logit value (ties)
    if temperature == 0.0 {
        let min_logit = logits.values().fold(f64::INFINITY, |a, &b| a.min(b));
        let min_keys: Vec<_> = logits
            .iter()
            .filter(|&(_, &v)| v == min_logit)
            .map(|(k, _)| *k)
            .collect();
        return min_keys;
    }

    let keys: Vec<_> = logits.keys().copied().collect();
    let values: Vec<_> = logits.values().copied().collect();

    // Find min and max for normalization
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        // All values are the same, uniform probability
        vec![1.0 / keys.len() as f64; keys.len()]
    } else {
        // Normalize values (lower is better, so negate)
        let normalized: Vec<_> = values
            .iter()
            .map(|&v| {
                let norm = v / (max_val - min_val);
                -norm
            })
            .collect();

        // Apply temperature and softmax
        let scaled: Vec<_> = normalized.iter().map(|&v| v / temperature).collect();
        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<_> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
        let sum_exp: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&v| v / sum_exp).collect()
    };

    // Sample from the probability distribution
    let mut rng = rand::rng();
    let sample: f64 = rng.random();

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return vec![keys[i]];
        }
    }

    // Fallback to last key
    vec![keys[keys.len() - 1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_workers(ids: &[u64]) -> HashMap<WorkerId, Option<WorkerConfig>> {
        ids.iter()
            .map(|&id| (id, Some(WorkerConfig::new())))
            .collect()
    }

    #[test]
    fn test_default_selector_single_worker() {
        let workers = make_workers(&[1]);
        let mut overlaps = OverlapScores::new();
        overlaps
            .scores
            .insert(WorkerWithDpRank::from_worker_id(1), 5);

        let request = SelectionRequest {
            isl_tokens: 128,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
        };

        let selector = DefaultWorkerSelector::default();
        let result = selector.select_worker(&workers, &request, 64).unwrap();

        assert_eq!(result.worker.worker_id, 1);
        assert_eq!(result.overlap_blocks, 5);
    }

    #[test]
    fn test_default_selector_prefers_cached() {
        let workers = make_workers(&[1, 2]);
        let mut overlaps = OverlapScores::new();
        overlaps
            .scores
            .insert(WorkerWithDpRank::from_worker_id(1), 10);
        overlaps
            .scores
            .insert(WorkerWithDpRank::from_worker_id(2), 2);

        let request = SelectionRequest {
            isl_tokens: 128,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
        };

        let selector = DefaultWorkerSelector::default();
        let result = selector.select_worker(&workers, &request, 64).unwrap();

        // Worker 1 has more cached blocks, so lower prefill cost
        assert_eq!(result.worker.worker_id, 1);
    }

    #[test]
    fn test_softmax_sample_single_key() {
        let mut logits = HashMap::new();
        let worker = WorkerWithDpRank::from_worker_id(42);
        logits.insert(worker, 0.5);

        for temperature in &[0.1, 1.0, 10.0] {
            let result = softmax_sample(&logits, *temperature);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], worker);
        }
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        let mut logits = HashMap::new();
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);

        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0); // Smallest logit
        logits.insert(worker3, 7.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker2);
    }

    #[test]
    fn test_softmax_sample_tied_values() {
        let mut logits = HashMap::new();
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);

        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0); // Tied
        logits.insert(worker3, 3.0); // Tied

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&worker2));
        assert!(result.contains(&worker3));
    }

    #[test]
    fn test_no_workers_error() {
        let workers: HashMap<WorkerId, Option<WorkerConfig>> = HashMap::new();
        let request = SelectionRequest {
            isl_tokens: 128,
            overlaps: OverlapScores::new(),
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
        };

        let selector = DefaultWorkerSelector::default();
        let result = selector.select_worker(&workers, &request, 64);

        assert!(matches!(result, Err(SchedulerError::NoWorkers)));
    }

    #[test]
    fn test_data_parallel_workers() {
        let mut workers = HashMap::new();
        workers.insert(1u64, Some(WorkerConfig::new().with_data_parallel_size(2)));

        // 256 tokens with block_size 64 = 4 blocks needed
        // dp_rank=0: 1 block cached = 3 blocks to prefill
        // dp_rank=1: 3 blocks cached = 1 block to prefill
        let mut overlaps = OverlapScores::new();
        overlaps.scores.insert(WorkerWithDpRank::new(1, 0), 1);
        overlaps.scores.insert(WorkerWithDpRank::new(1, 1), 3);

        let request = SelectionRequest {
            isl_tokens: 256,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
        };

        let selector = DefaultWorkerSelector::default();
        let result = selector.select_worker(&workers, &request, 64).unwrap();

        // dp_rank=1 has more cached blocks, so less prefill needed
        assert_eq!(result.worker.worker_id, 1);
        assert_eq!(result.worker.dp_rank, 1);
    }

    #[test]
    fn test_selection_overrides() {
        let workers = make_workers(&[1, 2]);
        let mut overlaps = OverlapScores::new();
        overlaps
            .scores
            .insert(WorkerWithDpRank::from_worker_id(1), 10);
        overlaps
            .scores
            .insert(WorkerWithDpRank::from_worker_id(2), 2);

        let request = SelectionRequest {
            isl_tokens: 128,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
        };

        // With overlap_weight=0, cache doesn't matter - should pick based on load alone
        let overrides = SelectionOverrides {
            overlap_score_weight: Some(0.0),
            router_temperature: None,
        };

        let selector = DefaultWorkerSelector::default();
        let result = selector
            .select_worker_with_overrides(&workers, &request, 64, Some(&overrides))
            .unwrap();

        // Both have same decode load, so either could be selected
        assert!(result.worker.worker_id == 1 || result.worker.worker_id == 2);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion functions between internal LLM types and protobuf types.

// #[cfg(feature = "proto")]
mod proto_convert {
    use crate::kv_router::RouterConfigOverride;
    use crate::protocols::common::preprocessor::{
        BootstrapInfo, MultimodalData, MultimodalDataMap, PrefillResult, PreprocessedRequest,
        RoutingHints,
    };
    use crate::protocols::common::{
        GuidedDecodingOptions, OutputOptions, SamplingOptions, StopConditions,
    };

    use dynamo_proto::request_plane::v1 as proto;

    // ============================================================================
    // RoutingHints conversions
    // ============================================================================

    impl From<RoutingHints> for proto::RoutingHints {
        fn from(r: RoutingHints) -> Self {
            proto::RoutingHints {
                backend_instance_id: r.backend_instance_id,
                prefill_worker_id: r.prefill_worker_id,
                decode_worker_id: r.decode_worker_id,
                dp_rank: r.dp_rank,
                enable_local_updates: r.enable_local_updates,
                expected_output_tokens: r.expected_output_tokens,
            }
        }
    }

    impl From<proto::RoutingHints> for RoutingHints {
        fn from(p: proto::RoutingHints) -> Self {
            RoutingHints {
                backend_instance_id: p.backend_instance_id,
                prefill_worker_id: p.prefill_worker_id,
                decode_worker_id: p.decode_worker_id,
                dp_rank: p.dp_rank,
                enable_local_updates: p.enable_local_updates,
                expected_output_tokens: p.expected_output_tokens,
            }
        }
    }

    // ============================================================================
    // BootstrapInfo conversions
    // ============================================================================

    impl From<BootstrapInfo> for proto::BootstrapInfo {
        fn from(b: BootstrapInfo) -> Self {
            proto::BootstrapInfo {
                bootstrap_host: b.bootstrap_host,
                bootstrap_port: b.bootstrap_port as u32,
                bootstrap_room: b.bootstrap_room,
            }
        }
    }

    impl From<proto::BootstrapInfo> for BootstrapInfo {
        fn from(p: proto::BootstrapInfo) -> Self {
            BootstrapInfo {
                bootstrap_host: p.bootstrap_host,
                bootstrap_port: p.bootstrap_port as u16,
                bootstrap_room: p.bootstrap_room,
            }
        }
    }

    // ============================================================================
    // PrefillResult conversions
    // ============================================================================

    impl From<PrefillResult> for proto::PrefillResult {
        fn from(p: PrefillResult) -> Self {
            let disaggregated_params =
                serde_json::from_value(p.disaggregated_params).unwrap_or_default();

            let prompt_tokens_details = p.prompt_tokens_details.map(|d| proto::PromptTokensDetails {
                cached_tokens: d.cached_tokens,
                audio_tokens: d.audio_tokens,
            });

            proto::PrefillResult {
                disaggregated_params: Some(disaggregated_params),
                prompt_tokens_details,
            }
        }
    }

    impl From<proto::PrefillResult> for PrefillResult {
        fn from(p: proto::PrefillResult) -> Self {
            let disaggregated_params = p
                .disaggregated_params
                .map(|s| serde_json::to_value(s).unwrap_or_default())
                .unwrap_or(serde_json::Value::Null);

            let prompt_tokens_details =
                p.prompt_tokens_details
                    .map(|d| dynamo_async_openai::types::PromptTokensDetails {
                        cached_tokens: d.cached_tokens,
                        audio_tokens: d.audio_tokens,
                    });

            PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
            }
        }
    }

    // ============================================================================
    // MultimodalData conversions
    // ============================================================================

    impl From<MultimodalData> for proto::MultimodalData {
        fn from(m: MultimodalData) -> Self {
            match m {
                MultimodalData::Url(url) => proto::MultimodalData {
                    data: Some(proto::multimodal_data::Data::Url(url.to_string())),
                },
                #[cfg(feature = "media-nixl")]
                MultimodalData::Decoded(desc) => {
                    let bytes = serde_json::to_vec(&desc).unwrap_or_default();
                    proto::MultimodalData {
                        data: Some(proto::multimodal_data::Data::RdmaDescriptor(bytes)),
                    }
                }
            }
        }
    }

    impl From<proto::MultimodalData> for MultimodalData {
        fn from(p: proto::MultimodalData) -> Self {
            match p.data {
                Some(proto::multimodal_data::Data::Url(url_str)) => {
                    let url = url::Url::parse(&url_str).expect("invalid URL in proto MultimodalData");
                    MultimodalData::Url(url)
                }
                #[cfg(feature = "media-nixl")]
                Some(proto::multimodal_data::Data::RdmaDescriptor(bytes)) => {
                    let desc = serde_json::from_slice(&bytes)
                        .expect("invalid RDMA descriptor in proto MultimodalData");
                    MultimodalData::Decoded(desc)
                }
                #[cfg(not(feature = "media-nixl"))]
                Some(proto::multimodal_data::Data::RdmaDescriptor(_)) => {
                    panic!("RDMA descriptor requires media-nixl feature")
                }
                None => panic!("MultimodalData has no data"),
            }
        }
    }

    // ============================================================================
    // StopConditions conversions
    // ============================================================================

    impl From<StopConditions> for proto::StopConditions {
        fn from(s: StopConditions) -> Self {
            proto::StopConditions {
                max_tokens: s.max_tokens,
                stop: s.stop.unwrap_or_default(),
                stop_token_ids_hidden: s.stop_token_ids_hidden.unwrap_or_default(),
                min_tokens: s.min_tokens,
                ignore_eos: s.ignore_eos,
                max_thinking_tokens: s.max_thinking_tokens,
            }
        }
    }

    impl From<proto::StopConditions> for StopConditions {
        fn from(p: proto::StopConditions) -> Self {
            StopConditions {
                max_tokens: p.max_tokens,
                stop: if p.stop.is_empty() {
                    None
                } else {
                    Some(p.stop)
                },
                stop_token_ids_hidden: if p.stop_token_ids_hidden.is_empty() {
                    None
                } else {
                    Some(p.stop_token_ids_hidden)
                },
                min_tokens: p.min_tokens,
                ignore_eos: p.ignore_eos,
                max_thinking_tokens: p.max_thinking_tokens,
            }
        }
    }

    // ============================================================================
    // GuidedDecodingOptions conversions
    // ============================================================================

    impl From<GuidedDecodingOptions> for proto::GuidedDecodingOptions {
        fn from(g: GuidedDecodingOptions) -> Self {
            proto::GuidedDecodingOptions {
                json: g.json.map(|v| serde_json::from_value(v).unwrap_or_default()),
                regex: g.regex,
                choice: g.choice.unwrap_or_default(),
                grammar: g.grammar,
                backend: g.backend,
                whitespace_pattern: g.whitespace_pattern,
            }
        }
    }

    impl From<proto::GuidedDecodingOptions> for GuidedDecodingOptions {
        fn from(p: proto::GuidedDecodingOptions) -> Self {
            GuidedDecodingOptions {
                json: p.json.map(|s| serde_json::to_value(s).unwrap_or_default()),
                regex: p.regex,
                choice: if p.choice.is_empty() {
                    None
                } else {
                    Some(p.choice)
                },
                grammar: p.grammar,
                backend: p.backend,
                whitespace_pattern: p.whitespace_pattern,
            }
        }
    }

    // ============================================================================
    // SamplingOptions conversions
    // ============================================================================

    impl From<SamplingOptions> for proto::SamplingOptions {
        fn from(s: SamplingOptions) -> Self {
            proto::SamplingOptions {
                n: s.n.map(|v| v as u32),
                best_of: s.best_of.map(|v| v as u32),
                presence_penalty: s.presence_penalty,
                frequency_penalty: s.frequency_penalty,
                repetition_penalty: s.repetition_penalty,
                temperature: s.temperature,
                top_p: s.top_p,
                top_k: s.top_k,
                min_p: s.min_p,
                use_beam_search: s.use_beam_search,
                length_penalty: s.length_penalty,
                seed: s.seed,
                include_stop_str_in_output: s.include_stop_str_in_output,
                guided_decoding: s.guided_decoding.map(|g| g.into()),
            }
        }
    }

    impl From<proto::SamplingOptions> for SamplingOptions {
        fn from(p: proto::SamplingOptions) -> Self {
            SamplingOptions {
                n: p.n.map(|v| v as u8),
                best_of: p.best_of.map(|v| v as u8),
                presence_penalty: p.presence_penalty,
                frequency_penalty: p.frequency_penalty,
                repetition_penalty: p.repetition_penalty,
                temperature: p.temperature,
                top_p: p.top_p,
                top_k: p.top_k,
                min_p: p.min_p,
                use_beam_search: p.use_beam_search,
                length_penalty: p.length_penalty,
                seed: p.seed,
                include_stop_str_in_output: p.include_stop_str_in_output,
                guided_decoding: p.guided_decoding.map(|g| g.into()),
            }
        }
    }

    // ============================================================================
    // OutputOptions conversions
    // ============================================================================

    impl From<OutputOptions> for proto::OutputOptions {
        fn from(o: OutputOptions) -> Self {
            proto::OutputOptions {
                logprobs: o.logprobs,
                prompt_logprobs: o.prompt_logprobs,
                skip_special_tokens: o.skip_special_tokens,
                formatted_prompt: o.formatted_prompt,
            }
        }
    }

    impl From<proto::OutputOptions> for OutputOptions {
        fn from(p: proto::OutputOptions) -> Self {
            OutputOptions {
                logprobs: p.logprobs,
                prompt_logprobs: p.prompt_logprobs,
                skip_special_tokens: p.skip_special_tokens,
                formatted_prompt: p.formatted_prompt,
            }
        }
    }

    // ============================================================================
    // RouterConfigOverride conversions
    // ============================================================================

    impl From<RouterConfigOverride> for proto::RouterConfigOverride {
        fn from(r: RouterConfigOverride) -> Self {
            proto::RouterConfigOverride {
                overlap_score_weight: r.overlap_score_weight,
                router_temperature: r.router_temperature,
            }
        }
    }

    impl From<proto::RouterConfigOverride> for RouterConfigOverride {
        fn from(p: proto::RouterConfigOverride) -> Self {
            RouterConfigOverride {
                overlap_score_weight: p.overlap_score_weight,
                router_temperature: p.router_temperature,
            }
        }
    }

    // ============================================================================
    // PreprocessedRequest conversions
    // ============================================================================

    impl From<PreprocessedRequest> for proto::PreprocessedRequest {
        fn from(r: PreprocessedRequest) -> Self {
            // Convert multimodal data map to repeated entries
            let multi_modal_data: Vec<proto::MultimodalDataEntry> = r
                .multi_modal_data
                .unwrap_or_default()
                .into_iter()
                .map(|(media_type, items)| proto::MultimodalDataEntry {
                    media_type,
                    items: items.into_iter().map(|m| m.into()).collect(),
                })
                .collect();

            proto::PreprocessedRequest {
                model: r.model,
                token_ids: r.token_ids,
                prompt_embeds: r.prompt_embeds,
                multi_modal_data,
                stop_conditions: Some(r.stop_conditions.into()),
                sampling_options: Some(r.sampling_options.into()),
                output_options: Some(r.output_options.into()),
                eos_token_ids: r.eos_token_ids,
                mdc_sum: r.mdc_sum,
                annotations: r.annotations,
                routing: r.routing.map(|r| r.into()),
                router_config_override: r.router_config_override.map(|r| r.into()),
                prefill_result: r.prefill_result.map(|p| p.into()),
                bootstrap_info: r.bootstrap_info.map(|b| b.into()),
                extra_args: r
                    .extra_args
                    .map(|v| serde_json::from_value(v).unwrap_or_default()),
            }
        }
    }

    impl From<proto::PreprocessedRequest> for PreprocessedRequest {
        fn from(p: proto::PreprocessedRequest) -> Self {
            // Convert multimodal data entries back to map
            let multi_modal_data: MultimodalDataMap = p
                .multi_modal_data
                .into_iter()
                .map(|entry| {
                    let items: Vec<MultimodalData> =
                        entry.items.into_iter().map(|m| m.into()).collect();
                    (entry.media_type, items)
                })
                .collect();

            PreprocessedRequest {
                model: p.model,
                token_ids: p.token_ids,
                prompt_embeds: p.prompt_embeds,
                multi_modal_data: if multi_modal_data.is_empty() {
                    None
                } else {
                    Some(multi_modal_data)
                },
                stop_conditions: p.stop_conditions.map(|s| s.into()).unwrap_or_default(),
                sampling_options: p.sampling_options.map(|s| s.into()).unwrap_or_default(),
                output_options: p.output_options.map(|o| o.into()).unwrap_or_default(),
                eos_token_ids: p.eos_token_ids,
                mdc_sum: p.mdc_sum,
                annotations: p.annotations,
                routing: p.routing.map(|r| r.into()),
                router_config_override: p.router_config_override.map(|r| r.into()),
                prefill_result: p.prefill_result.map(|p| p.into()),
                bootstrap_info: p.bootstrap_info.map(|b| b.into()),
                extra_args: p
                    .extra_args
                    .map(|v| serde_json::to_value(v).unwrap_or_default()),
                tracker: None,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_routing_hints_roundtrip() {
            let original = RoutingHints {
                backend_instance_id: Some(42),
                prefill_worker_id: Some(1),
                decode_worker_id: Some(2),
                dp_rank: Some(3),
                enable_local_updates: Some(true),
                expected_output_tokens: Some(100),
            };

            let proto: proto::RoutingHints = original.clone().into();
            let back: RoutingHints = proto.into();

            assert_eq!(original.backend_instance_id, back.backend_instance_id);
            assert_eq!(original.prefill_worker_id, back.prefill_worker_id);
            assert_eq!(original.decode_worker_id, back.decode_worker_id);
            assert_eq!(original.dp_rank, back.dp_rank);
            assert_eq!(original.enable_local_updates, back.enable_local_updates);
            assert_eq!(original.expected_output_tokens, back.expected_output_tokens);
        }

        #[test]
        fn test_stop_conditions_roundtrip() {
            let original = StopConditions {
                max_tokens: Some(100),
                stop: Some(vec!["stop1".to_string(), "stop2".to_string()]),
                stop_token_ids_hidden: Some(vec![1, 2, 3]),
                min_tokens: Some(10),
                ignore_eos: Some(false),
                max_thinking_tokens: Some(50),
            };

            let proto: proto::StopConditions = original.clone().into();
            let back: StopConditions = proto.into();

            assert_eq!(original.max_tokens, back.max_tokens);
            assert_eq!(original.stop, back.stop);
            assert_eq!(original.stop_token_ids_hidden, back.stop_token_ids_hidden);
            assert_eq!(original.min_tokens, back.min_tokens);
            assert_eq!(original.ignore_eos, back.ignore_eos);
            assert_eq!(original.max_thinking_tokens, back.max_thinking_tokens);
        }

        #[test]
        fn test_sampling_options_roundtrip() {
            let original = SamplingOptions {
                n: Some(2),
                best_of: Some(3),
                presence_penalty: Some(0.5),
                frequency_penalty: Some(0.3),
                repetition_penalty: Some(1.1),
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(50),
                min_p: Some(0.1),
                use_beam_search: Some(false),
                length_penalty: Some(1.0),
                seed: Some(42),
                include_stop_str_in_output: Some(true),
                guided_decoding: None,
            };

            let proto: proto::SamplingOptions = original.clone().into();
            let back: SamplingOptions = proto.into();

            assert_eq!(original.n, back.n);
            assert_eq!(original.best_of, back.best_of);
            assert_eq!(original.temperature, back.temperature);
            assert_eq!(original.top_p, back.top_p);
            assert_eq!(original.top_k, back.top_k);
            assert_eq!(original.seed, back.seed);
        }

        #[test]
        fn test_preprocessed_request_roundtrip() {
            let original = PreprocessedRequest {
                model: "test-model".to_string(),
                token_ids: vec![1, 2, 3, 4, 5],
                prompt_embeds: None,
                multi_modal_data: None,
                stop_conditions: StopConditions {
                    max_tokens: Some(100),
                    ..Default::default()
                },
                sampling_options: SamplingOptions {
                    temperature: Some(0.7),
                    ..Default::default()
                },
                output_options: OutputOptions {
                    logprobs: Some(5),
                    ..Default::default()
                },
                eos_token_ids: vec![0],
                mdc_sum: Some("abc123".to_string()),
                annotations: vec!["ann1".to_string()],
                routing: Some(RoutingHints {
                    backend_instance_id: Some(42),
                    ..Default::default()
                }),
                router_config_override: None,
                prefill_result: None,
                bootstrap_info: None,
                extra_args: None,
                tracker: None,
            };

            let proto: proto::PreprocessedRequest = original.clone().into();
            let back: PreprocessedRequest = proto.into();

            assert_eq!(original.model, back.model);
            assert_eq!(original.token_ids, back.token_ids);
            assert_eq!(original.eos_token_ids, back.eos_token_ids);
            assert_eq!(original.mdc_sum, back.mdc_sum);
            assert_eq!(original.annotations, back.annotations);
            assert_eq!(
                original.stop_conditions.max_tokens,
                back.stop_conditions.max_tokens
            );
            assert_eq!(
                original.sampling_options.temperature,
                back.sampling_options.temperature
            );
            assert_eq!(original.output_options.logprobs, back.output_options.logprobs);
            assert_eq!(
                original.routing.as_ref().unwrap().backend_instance_id,
                back.routing.as_ref().unwrap().backend_instance_id
            );
        }
    }
}

// The proto_convert module contains impl blocks for From/TryFrom traits.
// When the "proto" feature is enabled, these conversions are automatically
// available via the trait implementations.

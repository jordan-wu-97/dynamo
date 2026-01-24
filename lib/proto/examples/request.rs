use dynamo_proto::request_plane::v1::{PreprocessedRequest, SamplingOptions, StopConditions};

fn main() {
    let request = PreprocessedRequest {
        model: "model".to_string(),
        token_ids: vec![1, 2, 3],
        prompt_embeds: None,
        multi_modal_data: vec![],
        stop_conditions: Some(StopConditions::default()),
        sampling_options: Some(SamplingOptions::default()),
        output_options: None,
        eos_token_ids: vec![4, 5, 6],
        mdc_sum: None,
        annotations: vec![],
        routing: None,
        router_config_override: None,
        prefill_result: None,
        bootstrap_info: None,
        extra_args: None,
    };

    println!("{:?}", request);
}
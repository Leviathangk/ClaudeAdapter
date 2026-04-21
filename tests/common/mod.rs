use std::collections::HashMap;

use claude_adapter::{ApiMode, Config, ProviderConfig, ResponsesMetadataMode};

pub fn test_config(base_url: String, api_mode: ApiMode) -> Config {
    Config {
        bind: "127.0.0.1:8787".to_string(),
        incoming_api_key: Some("incoming-secret".to_string()),
        activate_provider: "active-provider".to_string(),
        providers: HashMap::from([(
            "active-provider".to_string(),
            ProviderConfig {
                base_url,
                api_mode,
                api_key: "provider-secret".to_string(),
                headers: HashMap::from([("x-extra-header".to_string(), "adapter".to_string())]),
                model_default: "fallback-model".to_string(),
                model_map: HashMap::from([
                    ("claude-sonnet-4.6".to_string(), "gpt-4.1-mini".to_string()),
                    ("claude-opus-4-6".to_string(), "o3".to_string()),
                ]),
                responses_metadata_mode: ResponsesMetadataMode::ClientMetadata,
            },
        )]),
    }
}

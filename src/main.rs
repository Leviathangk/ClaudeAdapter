use anyhow::Result;
use claude_adapter::{load_config, run};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let config_path =
        std::env::var("CLAUDE_ADAPTER_CONFIG").unwrap_or_else(|_| "config.yaml".to_string());
    let config = load_config(&config_path)?;
    run(config).await
}

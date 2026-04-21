use std::{
    collections::HashMap,
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow};
use notify::{Config as NotifyConfig, Event, RecommendedWatcher, RecursiveMode, Watcher};
use serde::Deserialize;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub config_path: PathBuf,
    pub env_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default)]
    pub incoming_api_key: Option<String>,
    pub activate_provider: String,
    pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_mode: ApiMode,
    pub api_key: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub model_default: String,
    #[serde(default)]
    pub model_map: HashMap<String, String>,
    #[serde(default = "default_responses_metadata_mode")]
    pub responses_metadata_mode: ResponsesMetadataMode,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiMode {
    ChatCompletions,
    Responses,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesMetadataMode {
    #[serde(alias = "metadata")]
    ClientMetadata,
    Omit,
}

pub(crate) fn default_bind() -> String {
    "127.0.0.1:8787".to_string()
}

fn default_responses_metadata_mode() -> ResponsesMetadataMode {
    ResponsesMetadataMode::ClientMetadata
}

pub fn load_config(path: &Path) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;
    let config: Config = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse yaml config: {}", path.display()))?;
    Ok(config)
}

pub fn load_env_file(path: &Path) -> Result<Option<Vec<String>>> {
    if !path.exists() {
        return Ok(None);
    }

    let entries = dotenvy::from_path_iter(path)
        .with_context(|| format!("failed to read env file: {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to parse env file: {}", path.display()))?;

    let mut keys = Vec::new();
    for (key, value) in entries {
        unsafe {
            env::set_var(&key, value);
        }
        keys.push(key);
    }

    keys.sort();
    keys.dedup();
    tracing::info!(path = %path.display(), env_keys = %keys.join(", "), ".env file loaded");
    Ok(Some(keys))
}

pub(crate) fn start_watchers(config: Arc<RwLock<Config>>, options: RunOptions) -> Result<()> {
    let config_path = options.config_path;
    let env_path = options.env_path;
    let watch_paths = collect_watch_paths(&config_path, env_path.as_deref())?;

    std::thread::Builder::new()
        .name("claude-adapter-watch".to_string())
        .spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut watcher = RecommendedWatcher::new(
                move |result| {
                    let _ = tx.send(result);
                },
                NotifyConfig::default(),
            )
            .expect("failed to create file watcher");

            for watch_path in &watch_paths {
                watcher
                    .watch(watch_path, RecursiveMode::NonRecursive)
                    .expect("failed to watch path");
            }

            while let Ok(event) = rx.recv() {
                match event {
                    Ok(event) => {
                        handle_watch_event(&config, &config_path, env_path.as_deref(), event)
                    }
                    Err(error) => tracing::error!(error = %error, "file watcher error"),
                }
            }
        })
        .context("failed to start file watcher thread")?;

    Ok(())
}

fn collect_watch_paths(config_path: &Path, env_path: Option<&Path>) -> Result<Vec<PathBuf>> {
    let mut paths = vec![parent_dir(config_path)?];
    if let Some(env_path) = env_path {
        let env_parent = parent_dir(env_path)?;
        if !paths.iter().any(|path| path == &env_parent) {
            paths.push(env_parent);
        }
    }
    Ok(paths)
}

fn parent_dir(path: &Path) -> Result<PathBuf> {
    path.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))
}

fn handle_watch_event(
    shared_config: &Arc<RwLock<Config>>,
    config_path: &Path,
    env_path: Option<&Path>,
    event: Event,
) {
    if event
        .paths
        .iter()
        .any(|path| matches_target(path, config_path))
    {
        match load_config(config_path) {
            Ok(config) => {
                let previous = shared_config.blocking_read().clone();
                *shared_config.blocking_write() = config.clone();
                log_startup_config(&config);
                log_config_change(&previous, &config, config_path);
            }
            Err(error) => {
                tracing::error!(path = %config_path.display(), error = %error, "failed to reload config");
            }
        }
    }

    if let Some(env_path) =
        env_path.filter(|target| event.paths.iter().any(|path| matches_target(path, target)))
    {
        match load_env_file(env_path) {
            Ok(Some(keys)) => {
                tracing::info!(path = %env_path.display(), refreshed_keys = %keys.join(", "), ".env reloaded")
            }
            Ok(None) => {
                tracing::info!(path = %env_path.display(), ".env not found, skipping reload")
            }
            Err(error) => {
                tracing::error!(path = %env_path.display(), error = %error, "failed to reload .env")
            }
        }
    }
}

fn matches_target(path: &Path, target: &Path) -> bool {
    path == target || (path.file_name() == target.file_name() && path.parent() == target.parent())
}

pub(crate) fn log_startup_config(config: &Config) {
    match config.providers.get(&config.activate_provider) {
        Some(provider) => {
            let mut mappings: Vec<String> = provider
                .model_map
                .iter()
                .map(|(from, to)| format!("{from} -> {to}"))
                .collect();
            mappings.sort();

            tracing::info!(
                provider = %config.activate_provider,
                model_default = %provider.model_default,
                model_map = %mappings.join(", "),
                "active provider loaded"
            );
        }
        None => {
            tracing::error!(provider = %config.activate_provider, "active provider missing from config");
        }
    }
}

fn log_config_change(previous: &Config, current: &Config, path: &Path) {
    let provider_change = format!(
        "{} -> {}",
        previous.activate_provider, current.activate_provider
    );
    let old_default = previous
        .providers
        .get(&previous.activate_provider)
        .map(|provider| provider.model_default.as_str())
        .unwrap_or("<missing>");
    let new_default = current
        .providers
        .get(&current.activate_provider)
        .map(|provider| provider.model_default.as_str())
        .unwrap_or("<missing>");

    tracing::info!(
        path = %path.display(),
        provider_change = %provider_change,
        model_default_change = %format!("{old_default} -> {new_default}"),
        "config reloaded"
    );
}

pub(crate) fn mapped_model(provider: &ProviderConfig, requested_model: &str) -> String {
    provider
        .model_map
        .get(requested_model)
        .cloned()
        .unwrap_or_else(|| provider.model_default.clone())
}

pub(crate) fn validate_bind(bind: &str) -> Result<SocketAddr> {
    bind.parse()
        .with_context(|| format!("invalid bind address: {bind}"))
}

#[cfg(test)]
mod tests {
    use super::{ProviderConfig, ResponsesMetadataMode};

    fn provider_yaml(extra: &str) -> String {
        format!(
            "\
base_url: http://127.0.0.1:8787
api_mode: responses
api_key: provider-secret
headers: {{}}
model_default: gpt-5.4
model_map: {{}}
{extra}"
        )
    }

    #[test]
    fn responses_metadata_mode_defaults_to_client_metadata() {
        let provider: ProviderConfig = serde_yaml::from_str(&provider_yaml("")).unwrap();
        assert_eq!(
            provider.responses_metadata_mode,
            ResponsesMetadataMode::ClientMetadata
        );
    }

    #[test]
    fn legacy_metadata_mode_alias_maps_to_client_metadata() {
        let provider: ProviderConfig =
            serde_yaml::from_str(&provider_yaml("responses_metadata_mode: metadata\n")).unwrap();
        assert_eq!(
            provider.responses_metadata_mode,
            ResponsesMetadataMode::ClientMetadata
        );
    }
}

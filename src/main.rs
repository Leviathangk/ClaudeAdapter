use anyhow::Result;
use claude_adapter::{
    RunOptions, append_error_log, install_panic_logger, load_config, load_env_file, run,
};
use std::io::{self, Write};
use std::path::PathBuf;

struct CliOptions {
    config_path: PathBuf,
    env_path: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    install_panic_logger();

    if let Err(error) = real_main().await {
        eprintln!("Error: {error:#}");
        append_error_log("startup/runtime error", &format!("error: {error:#}"));
        wait_for_enter();
        return Err(error);
    }

    Ok(())
}

async fn real_main() -> Result<()> {
    let options = parse_cli_options()?;

    if let Some(env_path) = &options.env_path {
        let _ = load_env_file(env_path)?;
    }

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let config = load_config(&options.config_path)?;
    run(
        config,
        RunOptions {
            config_path: options.config_path,
            env_path: options.env_path,
        },
    )
    .await
}

fn wait_for_enter() {
    eprint!("Press Enter to exit...");
    let _ = io::stderr().flush();
    let mut buffer = String::new();
    let _ = io::stdin().read_line(&mut buffer);
}

fn parse_cli_options() -> Result<CliOptions> {
    let cwd = std::env::current_dir()?;
    let mut config_path = cwd.join("config.yaml");
    let mut env_path = Some(cwd.join(".env"));

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-y" | "--yaml" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {arg}"))?;
                config_path = resolve_path(&cwd, value);
            }
            "-e" | "--env" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {arg}"))?;
                env_path = Some(resolve_path(&cwd, value));
            }
            _ => return Err(anyhow::anyhow!("unknown argument: {arg}")),
        }
    }

    if !config_path.exists() {
        return Err(anyhow::anyhow!(format!(
            "config file not found: {}",
            config_path.display()
        )));
    }

    Ok(CliOptions {
        config_path,
        env_path,
    })
}

fn resolve_path(cwd: &std::path::Path, value: String) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

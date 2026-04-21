use std::{
    env,
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};
use tracing_subscriber::{
    EnvFilter, fmt::writer::MakeWriter, layer::SubscriberExt, util::SubscriberInitExt,
};

static DEBUG_LOG_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

pub fn append_error_log(stage: &str, details: &str) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();

    let path = match env::current_dir() {
        Ok(dir) => dir.join("error.log"),
        Err(error) => {
            tracing::error!(error = %error, "failed to resolve current directory for error.log");
            return;
        }
    };

    let mut file = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(file) => file,
        Err(error) => {
            tracing::error!(path = %path.display(), error = %error, "failed to open error.log");
            return;
        }
    };

    let record =
        format!("[{timestamp}] {stage}\n{details}\n----------------------------------------\n");

    if let Err(error) = file.write_all(record.as_bytes()) {
        tracing::error!(path = %path.display(), error = %error, "failed to write error.log");
    }
}

pub fn install_panic_logger() {
    std::panic::set_hook(Box::new(|panic_info| {
        let location = panic_info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "unknown".to_string());

        let payload = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };

        append_error_log(
            "panic",
            &format!("location: {location}\npayload: {payload}"),
        );
    }));
}

pub fn init_tracing(env_filter: EnvFilter) {
    #[cfg(debug_assertions)]
    let debug_log_path = initialize_debug_log_file();

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_ansi(true);

    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer);

    #[cfg(debug_assertions)]
    let subscriber = subscriber.with(
        tracing_subscriber::fmt::layer()
            .with_writer(debug_log_writer())
            .with_ansi(false),
    );

    subscriber.init();

    #[cfg(debug_assertions)]
    if let Some(path) = debug_log_path {
        tracing::info!(path = %path.display(), "local debug log enabled");
    }
}

pub fn preview_text(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut preview: String = normalized.chars().take(limit).collect();
    if normalized.chars().count() > limit {
        preview.push_str("...");
    }
    preview
}

fn initialize_debug_log_file() -> Option<PathBuf> {
    let path = debug_log_path()?.clone();
    match OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
    {
        Ok(_) => Some(path),
        Err(error) => {
            eprintln!(
                "failed to initialize local debug log {}: {error}",
                path.display()
            );
            None
        }
    }
}

fn debug_log_path() -> Option<&'static PathBuf> {
    DEBUG_LOG_PATH
        .get_or_init(|| env::current_dir().ok().map(|dir| dir.join("debug.log")))
        .as_ref()
}

fn debug_log_writer() -> impl for<'writer> MakeWriter<'writer> + Send + Sync + 'static {
    move || -> Box<dyn Write + Send> {
        let Some(path) = debug_log_path() else {
            return Box::new(io::sink());
        };

        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => Box::new(file),
            Err(error) => {
                eprintln!("failed to open local debug log {}: {error}", path.display());
                Box::new(io::sink())
            }
        }
    }
}

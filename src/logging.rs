use std::{
    env,
    fs::OpenOptions,
    io::Write,
    time::{SystemTime, UNIX_EPOCH},
};

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

pub fn preview_text(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut preview: String = normalized.chars().take(limit).collect();
    if normalized.chars().count() > limit {
        preview.push_str("...");
    }
    preview
}

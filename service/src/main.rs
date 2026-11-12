//! HTTP front for the resident audio-transcribe ASR worker (issue #4).
//!
//! The service owns HTTP, the request queue, timeouts and the worker lifecycle;
//! ASR itself lives in `python -m audio_transcribe.asr_worker`, reached over a
//! Unix socket with JSON lines. `POST /v1/audio/transcriptions` keeps the
//! OpenAI-compatible shape (file, model, language, prompt, response_format) so a
//! client can switch to it by configuration only.
//!
//! Configuration comes from the environment:
//!   ASR_BIND           bind address, default 127.0.0.1:8765
//!   ASR_SOCKET         worker socket, default $HOME/.audio-transcribe/asr.sock
//!   ASR_TOKEN_FILE     file with the bearer token (0600); unset = no auth
//!   ASR_WORKER_CMD     worker command, default "python3 -m audio_transcribe.asr_worker"
//!   ASR_TONE_MODEL_DIR T-one model directory; unset = no degraded fallback
//!   ASR_TONE_PROMPT    "1" feeds the T-one text to the final pass as initial_prompt
//!   ASR_WHISPER_MODEL  default "large-v3"
//!   ASR_LANGUAGE       language used when the request omits it, default "ru"
//!   ASR_TIMEOUT_S      per-request timeout, default 60
//!   ASR_LOG_FILE       JSON-lines event log, default $HOME/.audio-transcribe/asr-events.jsonl
//!   ASR_LOG_MAX_BYTES  rotate above this size, default 4194304
//!   ASR_LOG_KEEP       rotated files kept next to the current one, default 5

use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::timeout;

const MAX_UPLOAD_BYTES: usize = 30 * 1024 * 1024;

struct Config {
    bind: String,
    socket: PathBuf,
    token: Option<String>,
    worker_cmd: Vec<String>,
    tone_model_dir: Option<PathBuf>,
    tone_prompt: bool,
    whisper_model: String,
    timeout: Duration,
    log_file: PathBuf,
    log_max_bytes: u64,
    log_keep: u32,
}

/// Bounded JSON-lines event log: one line per request and lifecycle event,
/// rotated by size so the service can never fill the disk. The analyzer in
/// `scripts/asr_log_review.py` reads these files; the launchd stdio files keep
/// only startup lines, because per-request output does not go there.
struct Logger {
    path: PathBuf,
    max_bytes: u64,
    keep: u32,
    lock: std::sync::Mutex<()>,
}

impl Logger {
    fn new(path: PathBuf, max_bytes: u64, keep: u32) -> Self {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Self {
            path,
            max_bytes: max_bytes.max(1024),
            keep: keep.max(1),
            lock: std::sync::Mutex::new(()),
        }
    }

    fn write(&self, event: &str, mut fields: Value) {
        let _guard = match self.lock.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        fields["ts"] = json!(now_ms());
        fields["event"] = json!(event);
        let mut line = fields.to_string();
        line.push('\n');
        self.rotate_if_needed();
        use std::io::Write;
        match std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            Ok(mut file) => {
                let _ = file.write_all(line.as_bytes());
            }
            Err(error) => eprintln!("asr log {}: {error}", self.path.display()),
        }
    }

    fn rotate_if_needed(&self) {
        let size = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        if size < self.max_bytes {
            return;
        }
        for index in (1..self.keep).rev() {
            let from = self.numbered(index);
            let to = self.numbered(index + 1);
            let _ = std::fs::rename(&from, &to);
        }
        let _ = std::fs::rename(&self.path, self.numbered(1));
        for index in (1..=self.keep).rev() {
            if index > self.keep {
                continue;
            }
            let _ = std::fs::remove_file(self.numbered(index + 1));
        }
    }

    fn numbered(&self, index: u32) -> PathBuf {
        let mut name = self.path.clone().into_os_string();
        name.push(format!(".{index}"));
        PathBuf::from(name)
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

struct Conn {
    _child: Child,
    read: BufReader<OwnedReadHalf>,
    write: OwnedWriteHalf,
    next_id: u64,
}

struct App {
    cfg: Arc<Config>,
    worker: Mutex<Option<Conn>>,
    counter: AtomicU64,
    log: Arc<Logger>,
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn load_config() -> Result<Config, String> {
    let home = env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
    let token = match env::var("ASR_TOKEN_FILE") {
        Ok(path) if !path.trim().is_empty() => {
            let raw = std::fs::read_to_string(&path)
                .map_err(|e| format!("read ASR_TOKEN_FILE {path}: {e}"))?;
            let trimmed = raw.trim().to_string();
            if trimmed.is_empty() {
                return Err(format!("ASR_TOKEN_FILE {path} is empty"));
            }
            Some(trimmed)
        }
        _ => None,
    };
    let worker_cmd = env_or("ASR_WORKER_CMD", "python3 -m audio_transcribe.asr_worker")
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if worker_cmd.is_empty() {
        return Err("ASR_WORKER_CMD is empty".to_string());
    }
    let tone_model_dir = env::var("ASR_TONE_MODEL_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from);
    Ok(Config {
        bind: env_or("ASR_BIND", "127.0.0.1:8765"),
        socket: PathBuf::from(env_or(
            "ASR_SOCKET",
            &format!("{home}/.audio-transcribe/asr.sock"),
        )),
        token,
        worker_cmd,
        tone_model_dir,
        tone_prompt: matches!(
            env_or("ASR_TONE_PROMPT", "0").as_str(),
            "1" | "true" | "yes"
        ),
        whisper_model: env_or("ASR_WHISPER_MODEL", "large-v3"),
        log_file: PathBuf::from(env_or(
            "ASR_LOG_FILE",
            &format!("{home}/.audio-transcribe/asr-events.jsonl"),
        )),
        log_max_bytes: env_or("ASR_LOG_MAX_BYTES", "4194304")
            .parse()
            .map_err(|e| format!("ASR_LOG_MAX_BYTES: {e}"))?,
        log_keep: env_or("ASR_LOG_KEEP", "5")
            .parse()
            .map_err(|e| format!("ASR_LOG_KEEP: {e}"))?,
        timeout: Duration::from_secs(
            env_or("ASR_TIMEOUT_S", "60")
                .parse()
                .map_err(|e| format!("ASR_TIMEOUT_S: {e}"))?,
        ),
    })
}

fn authorized(cfg: &Config, headers: &HeaderMap) -> bool {
    let Some(expected) = cfg.token.as_deref() else {
        return true;
    };
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(|got| got.trim() == expected)
        .unwrap_or(false)
}

async fn spawn_worker(cfg: &Config) -> Result<Conn, String> {
    let mut command = Command::new(&cfg.worker_cmd[0]);
    command.args(&cfg.worker_cmd[1..]);
    command.arg("--socket").arg(&cfg.socket);
    if let Some(dir) = &cfg.tone_model_dir {
        command.arg("--tone-model-dir").arg(dir);
        if cfg.tone_prompt {
            command.arg("--tone-prompt");
        }
    }
    command.arg("--whisper-model").arg(&cfg.whisper_model);
    command.stdin(Stdio::null());
    // Per-request worker output goes to the JSONL event log, not to launchd's
    // stdio files, so those files cannot grow with traffic.
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());
    command.kill_on_drop(true);
    if let Some(parent) = cfg.socket.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    eprintln!("starting ASR worker on {}", cfg.socket.display());
    let child = command
        .spawn()
        .map_err(|e| format!("spawn ASR worker {:?}: {e}", cfg.worker_cmd))?;
    let mut last_error = String::new();
    for _ in 0..150 {
        match UnixStream::connect(&cfg.socket).await {
            Ok(stream) => {
                let (read, write) = stream.into_split();
                return Ok(Conn {
                    _child: child,
                    read: BufReader::new(read),
                    write,
                    next_id: 1,
                });
            }
            Err(error) => {
                last_error = error.to_string();
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
    Err(format!(
        "ASR worker did not open {}: {last_error}",
        cfg.socket.display()
    ))
}

impl App {
    /// Send one request to the worker, restarting it if it died or timed out.
    async fn request(&self, mut payload: Value) -> Result<Value, String> {
        let mut guard = self.worker.lock().await;
        if guard.is_none() {
            match spawn_worker(&self.cfg).await {
                Ok(conn) => {
                    *guard = Some(conn);
                    self.log.write("worker_start", json!({"socket": self.cfg.socket.display().to_string()}));
                }
                Err(error) => {
                    self.log.write("worker_start_failed", json!({"error": error}));
                    return Err(error);
                }
            }
        }
        let conn = guard.as_mut().expect("worker connection");
        let id = conn.next_id;
        conn.next_id += 1;
        payload["id"] = json!(id);

        let mut line = serde_json::to_vec(&payload).map_err(|e| e.to_string())?;
        line.push(b'\n');
        let exchange = async {
            conn.write.write_all(&line).await?;
            conn.write.flush().await?;
            let mut buf = String::new();
            conn.read.read_line(&mut buf).await?;
            Ok::<String, std::io::Error>(buf)
        };
        match timeout(self.cfg.timeout, exchange).await {
            Ok(Ok(buf)) if !buf.trim().is_empty() => {
                let value: Value = serde_json::from_str(buf.trim())
                    .map_err(|e| format!("worker returned invalid json: {e}"))?;
                if value.get("ok").and_then(Value::as_bool) == Some(true) {
                    Ok(value)
                } else {
                    Err(value
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("worker failed")
                        .to_string())
                }
            }
            Ok(Ok(_)) => {
                *guard = None;
                self.log.write("worker_dropped", json!({"reason": "closed the connection"}));
                Err("worker closed the connection".to_string())
            }
            Ok(Err(error)) => {
                *guard = None;
                self.log
                    .write("worker_dropped", json!({"reason": format!("io error: {error}")}));
                Err(format!("worker io error: {error}"))
            }
            Err(_) => {
                // shortcut: a timed-out worker is killed and respawned instead of
                // cancelled in place; cheaper than thread-level cancellation.
                *guard = None;
                self.log.write("worker_dropped", json!({"reason": "timeout"}));
                Err(format!(
                    "transcription timed out after {}s",
                    self.cfg.timeout.as_secs()
                ))
            }
        }
    }
}

async fn health(State(app): State<Arc<App>>) -> Response {
    let worker = if app.worker.lock().await.is_some() {
        "ready"
    } else {
        "idle"
    };
    Json(json!({
        "ok": true,
        "worker": worker,
        "model": app.cfg.whisper_model,
        "tone_model": app.cfg.tone_model_dir.as_ref().map(|p| p.display().to_string()),
    }))
    .into_response()
}

fn extension_for(file_name: Option<&str>, content_type: Option<&str>) -> String {
    if let Some(name) = file_name {
        if let Some(ext) = Path::new(name).extension().and_then(|e| e.to_str()) {
            if !ext.is_empty() && ext.len() <= 8 {
                return ext.to_lowercase();
            }
        }
    }
    match content_type.unwrap_or("").split(';').next().unwrap_or("") {
        "audio/webm" => "webm",
        "audio/ogg" | "audio/opus" => "ogg",
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => "m4a",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        _ => "bin",
    }
    .to_string()
}

async fn transcribe(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    if !authorized(&app.cfg, &headers) {
        app.log.write("unauthorized", json!({"path": "/v1/audio/transcriptions"}));
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"message": "missing or invalid bearer token"}})),
        )
            .into_response();
    }

    let mut audio: Option<(String, Bytes)> = None;
    let mut language = env_or("ASR_LANGUAGE", "ru");
    let mut prompt = String::new();
    let mut response_format = "json".to_string();
    let mut model = app.cfg.whisper_model.clone();

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": {"message": format!("invalid multipart body: {error}")}})),
                )
                    .into_response()
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        let file_name = field.file_name().map(str::to_string);
        let content_type = field.content_type().map(str::to_string);
        match name.as_str() {
            "file" => {
                let bytes = match field.bytes().await {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({"error": {"message": format!("read file field: {error}")}})),
                        )
                            .into_response()
                    }
                };
                let ext = extension_for(file_name.as_deref(), content_type.as_deref());
                audio = Some((ext, bytes));
            }
            "language" => language = field.text().await.unwrap_or_default().trim().to_string(),
            "prompt" => prompt = field.text().await.unwrap_or_default(),
            "response_format" => {
                response_format = field.text().await.unwrap_or_default().trim().to_string()
            }
            "model" => model = field.text().await.unwrap_or_default().trim().to_string(),
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let Some((ext, bytes)) = audio else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": "missing 'file' field"}})),
        )
            .into_response();
    };
    if bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": "empty audio upload"}})),
        )
            .into_response();
    }

    let stamp = app.counter.fetch_add(1, Ordering::Relaxed);
    let dir = env::temp_dir().join(format!("asr-{}-{stamp}", std::process::id()));
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": format!("temp dir: {error}")}})),
        )
            .into_response();
    }
    let path = dir.join(format!("audio.{ext}"));
    if let Err(error) = std::fs::write(&path, &bytes) {
        let _ = std::fs::remove_dir_all(&dir);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": format!("write upload: {error}")}})),
        )
            .into_response();
    }

    let started = std::time::Instant::now();
    let result = app
        .request(json!({
            "audio": path.display().to_string(),
            "language": if language.is_empty() { Value::Null } else { json!(language) },
            "prompt": prompt,
            "model": model,
        }))
        .await;
    let _ = std::fs::remove_dir_all(&dir);

    match result {
        Ok(value) => {
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let duration_ms = started.elapsed().as_millis() as u64;
            app.log.write(
                "transcribe",
                json!({
                    "ok": true,
                    "degraded": value.get("degraded").and_then(Value::as_bool).unwrap_or(false),
                    "audio_bytes": bytes.len(),
                    "duration_ms": duration_ms,
                    "language": language,
                    "chars": text.chars().count(),
                    "timings_ms": value.get("timings_ms").cloned().unwrap_or(Value::Null),
                    "tone_error": value.get("tone_error").cloned().unwrap_or(Value::Null),
                }),
            );
            if response_format == "text" {
                return ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text)
                    .into_response();
            }
            Json(json!({"text": text})).into_response()
        }
        Err(message) => {
            app.log.write(
                "transcribe",
                json!({
                    "ok": false,
                    "audio_bytes": bytes.len(),
                    "duration_ms": started.elapsed().as_millis() as u64,
                    "language": language,
                    "error": message,
                }),
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": message}})),
            )
                .into_response()
        }
    }
}

#[tokio::main]
async fn main() {
    let cfg = match load_config() {
        Ok(cfg) => Arc::new(cfg),
        Err(error) => {
            eprintln!("configuration error: {error}");
            std::process::exit(2);
        }
    };
    let log = Arc::new(Logger::new(
        cfg.log_file.clone(),
        cfg.log_max_bytes,
        cfg.log_keep,
    ));
    log.write(
        "service_start",
        json!({
            "bind": cfg.bind,
            "model": cfg.whisper_model,
            "tone_model": cfg.tone_model_dir.as_ref().map(|p| p.display().to_string()),
            "auth": cfg.token.is_some(),
            "timeout_s": cfg.timeout.as_secs(),
            "log_max_bytes": cfg.log_max_bytes,
            "log_keep": cfg.log_keep,
        }),
    );
    let app = Arc::new(App {
        cfg: cfg.clone(),
        worker: Mutex::new(None),
        counter: AtomicU64::new(0),
        log,
    });
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/audio/transcriptions", post(transcribe))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .with_state(app);

    let listener = match tokio::net::TcpListener::bind(&cfg.bind).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("bind {}: {error}", cfg.bind);
            std::process::exit(2);
        }
    };
    eprintln!("audio-transcribe service listening on {}", cfg.bind);
    if let Err(error) = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        eprintln!("server error: {error}");
        std::process::exit(1);
    }
    // Dropping the router drops `App`, and with it the worker child
    // (`kill_on_drop`), so a stopped service does not leave an orphan worker.
}

async fn shutdown_signal() {
    let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(signal) => signal,
        Err(error) => {
            eprintln!("SIGTERM handler: {error}");
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(token: Option<&str>) -> Config {
        Config {
            bind: "127.0.0.1:0".into(),
            socket: PathBuf::from("/tmp/asr-test.sock"),
            token: token.map(str::to_string),
            worker_cmd: vec!["python3".into()],
            tone_model_dir: None,
            tone_prompt: false,
            whisper_model: "large-v3".into(),
            timeout: Duration::from_secs(1),
            log_file: PathBuf::from("/tmp/asr-test-events.jsonl"),
            log_max_bytes: 1_048_576,
            log_keep: 2,
        }
    }

    #[test]
    fn token_is_required_when_configured() {
        let cfg = config(Some("secret"));
        let mut headers = HeaderMap::new();
        assert!(!authorized(&cfg, &headers));
        headers.insert(header::AUTHORIZATION, "Bearer wrong".parse().unwrap());
        assert!(!authorized(&cfg, &headers));
        headers.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        assert!(authorized(&cfg, &headers));
    }

    #[test]
    fn auth_is_optional_without_token_file() {
        assert!(authorized(&config(None), &HeaderMap::new()));
    }

    #[test]
    fn the_event_log_rotates_and_stays_bounded() {
        let dir = std::env::temp_dir().join(format!("asr-log-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.jsonl");
        let logger = Logger::new(path.clone(), 2048, 3);
        for index in 0..200 {
            logger.write("transcribe", json!({"index": index, "padding": "x".repeat(80)}));
        }
        let mut total = 0u64;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            total += entry.metadata().unwrap().len();
        }
        // keep=3 means the current file plus two rotations; nothing unbounded.
        assert!(total <= 2048 * 3, "log grew to {total} bytes");
        assert!(path.exists());
        assert!(path.with_extension("jsonl.1").exists());
        assert!(!path.with_extension("jsonl.3").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn upload_extension_prefers_filename_then_content_type() {
        assert_eq!(extension_for(Some("note.M4A"), Some("audio/webm")), "m4a");
        assert_eq!(extension_for(None, Some("audio/webm; codecs=opus")), "webm");
        assert_eq!(extension_for(None, None), "bin");
    }
}

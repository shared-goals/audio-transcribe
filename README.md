# audio-transcribe

Local audio transcription pipeline that processes recorded meetings into structured Obsidian notes. Runs entirely on macOS with no cloud APIs.

## Pipeline

```
Audio (WAV/M4A/MP3)
  → preprocess (ffmpeg: 16kHz mono WAV)
  → transcribe (WhisperX / MLX)
  → align (wav2vec2)
  → diarize (pyannote, optional)
  → format (Markdown meeting note)
```

## Install

```zsh
curl -fsSL https://raw.githubusercontent.com/shared-goals/audio-transcribe/main/install.sh | zsh
```

The installer handles Homebrew, ffmpeg, uv, the Python package, PATH setup, and HuggingFace token configuration.

The installer uses the latest supported tagged release (`0.5.0`) and installs the complete optional ML stack. For a lightweight CLI-only development environment, use `uv sync`. Backend-specific development installs use `uv sync --extra mlx`, `--extra whisperx`, or `--extra diarization`; `--extra ml` installs everything.

### Requirements

- macOS (Apple Silicon)
- HuggingFace token with accepted pyannote licenses ([speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0))

## Usage

### Transcribe a meeting

```zsh
# Fast pass (default): transcribe + align → meeting note
audio-transcribe process recording.m4a

# Save output to a specific directory
audio-transcribe process recording.m4a -o meetings/

# Full pass: includes speaker diarization (slower)
audio-transcribe process recording.m4a -o meetings/ --full

# Also generate a plain transcript
audio-transcribe process recording.m4a -o meetings/ --transcript transcript.md
```

### Choose a backend

Three transcription backends are available:

```zsh
# mlx-vad (default) — Apple Silicon GPU with VAD chunking, fastest
audio-transcribe process recording.m4a --backend mlx-vad

# mlx — Apple Silicon GPU, single-pass
audio-transcribe process recording.m4a --backend mlx

# whisperx — CPU via ctranslate2, slowest but most compatible
audio-transcribe process recording.m4a --backend whisperx
```

### Language and model

```zsh
# Set language (default: ru)
audio-transcribe process recording.m4a -l en

# Set Whisper model (default: large-v3)
audio-transcribe process recording.m4a -m medium
```

### Skip stages

```zsh
# Skip alignment
audio-transcribe process recording.m4a --no-align

# Skip diarization (already skipped in fast pass)
audio-transcribe process recording.m4a --full --no-diarize
```

### Post-process existing meeting notes

```zsh
# Add speaker diarization to an existing note
audio-transcribe diarize meetings/2026-03-01-standup.md

# Auto-identify speakers using voice database
audio-transcribe identify meetings/2026-03-01-standup.md

# Apply speaker mapping from frontmatter
audio-transcribe update meetings/2026-03-01-standup.md
```

### Speaker management

```zsh
audio-transcribe speakers list
audio-transcribe speakers forget "Name"
```

### Statistics and recommendations

```zsh
# View historical run stats
audio-transcribe stats --last 5

# Get backend recommendation for a file
audio-transcribe recommend recording.m4a

# Learn corrections from an edited transcript
audio-transcribe learn corrected-transcript.md
```

### Machine-readable output

```zsh
# JSON-lines output (no TUI, for scripting)
audio-transcribe process recording.m4a --json
```

### Private ASR and diarization benchmarks

Benchmark inputs and outputs stay in an explicitly selected private dataset directory. `prepare` imports the current note only as an editable draft and creates a draft vocabulary. Before running candidates, accept the transcript as `reference-transcript.md`, set `reference_status` and `reference_sha256`, review `vocabulary.txt`, set `vocabulary_status` and `vocabulary_sha256`, and add explicit `models.baseline` and `models.russian` definitions to `benchmark.json`.

```zsh
audio-transcribe benchmark prepare recording.mp3 --current-note current-note.md --dataset private-benchmark
audio-transcribe benchmark run-asr --dataset private-benchmark --candidate baseline
audio-transcribe benchmark run-asr --dataset private-benchmark --candidate vocabulary
audio-transcribe benchmark run-asr --dataset private-benchmark --candidate russian
audio-transcribe benchmark evaluate-asr --dataset private-benchmark
audio-transcribe benchmark report --dataset private-benchmark
audio-transcribe benchmark run-asr --dataset private-benchmark --candidate winner-context
audio-transcribe benchmark run-diarization --dataset private-benchmark --candidate pyannote-3.1
audio-transcribe benchmark run-diarization --dataset private-benchmark --candidate community-1
audio-transcribe benchmark run-diarization --dataset private-benchmark --candidate community-1-exclusive
audio-transcribe benchmark evaluate-diarization --dataset private-benchmark
audio-transcribe benchmark report --dataset private-benchmark
```

The four ASR candidates are the generic baseline, the baseline with reviewed `vocabulary.txt` passed as Whisper's `initial_prompt`, the explicitly configured Russian model, and the initial winner retested with previous-text context. Diarization requires an accepted `reference-speakers.json` and the frozen winning aligned-ASR checksum, then compares pyannote 3.1, Community-1, and Community-1 exclusive word assignment. Every ASR and diarization evaluation is bound to the exact candidate `result.json` checksum in addition to the accepted audio, transcript, vocabulary, aligned ASR, and speaker-reference checksums; rerunning either kind of candidate invalidates its previous evaluation before artifact replacement begins, so even a partial write cannot leave trusted stale evidence, while a payload that cannot be serialized leaves the previous artifacts and evaluation untouched. Legacy unbound evaluations must be regenerated. The combined report recomputes the ASR winner from current checksum-validated runs, rejects an aligned result from any other candidate, validates all diarization metrics before ranking, and ranks safety-eligible candidates by DER, confusion, word-speaker accuracy, fragmentation, and total ASR+alignment+diarization runtime, with 30/60/120-minute forecasts and a 2× gate against the original ASR+alignment+pyannote 3.1 baseline. Failed gated-model access is retained without deleting successful sibling runs. These commands do not change the production model or backend defaults.

### Profiles, resume, and templates

Interrupted runs resume automatically from `~/.audio-transcribe/runs/`. Use `--force` for a clean run, `--restart-from align` to invalidate one stage and everything after it, or `--keep-workdir` to retain normalized audio.

Profiles live in `~/.config/audio-transcribe/config.toml`:

```toml
[profiles.fast]
language = "ru"
model = "large-v3"
backend = "mlx-vad"
no_diarize = true

[profiles.full]
backend = "mlx-vad"
no_diarize = false
template = "~/templates/meeting.md"
```

Templates may use `{{title}}`, `{{meeting_note}}`, and `{{transcript}}`; at least one content placeholder is required.

```zsh
audio-transcribe process recording.m4a --profile fast
```

### Unattended worker

```zsh
audio-transcribe doctor
audio-transcribe worker enqueue recording.m4a -o meetings/
audio-transcribe worker run --once
audio-transcribe worker run --watch incoming/ -o meetings/
audio-transcribe worker status --json
audio-transcribe worker retry 42
```

The queue is SQLite-backed and survives restarts. Failed jobs retry with exponential backoff and move to a visible dead-letter state after their attempt limit.

For launchd, copy `scripts/com.gnezim.audio-transcribe-worker.plist.template`, replace its four placeholders, and point `__WORKER_SCRIPT__` at the executable `scripts/audio-transcribe-worker.sh`. `audio-transcribe worker status --json` is the health check; a running LaunchAgent alone is not considered healthy.

### ASR service for short requests

`audio-transcribe process` loads the model on every call (9 s for a 7 s recording), which is too slow for voice input in an agent composer. The service keeps the model warm:

```zsh
# build once, on the same architecture as the target host
cargo build --release --manifest-path service/Cargo.toml

# the Rust front owns HTTP, the queue and timeouts; the Python worker owns ASR
python -m audio_transcribe.asr_worker --socket ~/.audio-transcribe/asr.sock \
  --tone-model-dir ~/.local/share/sherpa-onnx-streaming-t-one-russian-2025-09-08

ASR_WORKER_CMD="$PWD/.venv/bin/python -m audio_transcribe.asr_worker" \
ASR_TONE_MODEL_DIR=~/.local/share/sherpa-onnx-streaming-t-one-russian-2025-09-08 \
ASR_TOKEN_FILE=~/asr-token ASR_BIND=127.0.0.1:8765 \
  ./service/target/release/audio-transcribe-service

curl -H "Authorization: Bearer $(cat ~/asr-token)" \
  -F file=@voice.webm -F language=ru http://127.0.0.1:8765/v1/audio/transcriptions
curl http://127.0.0.1:8765/health
```

`POST /v1/audio/transcriptions` accepts the OpenAI-compatible multipart fields `file`, `model`, `language`, `prompt` and `response_format` (`json` by default, `text` for plain output). The final text comes from the production `mlx-vad` path with `mlx-whisper large-v3`, so punctuation and casing match `process`. T-one is the degraded answer when the final pass fails, and `--tone-prompt` feeds its text to the final pass as `initial_prompt` — off by default: an unpunctuated prompt makes whisper drop punctuation too.

Measured 2026-11-12 on a Mac mini M4 (`chez-mac`): first call 7.2 s (model load), then 1.8 s for a 9 s recording; 27 s of audio takes 13.6 s and 63 s takes 31.8 s, so the service targets short requests, not meetings.

For launchd, copy `scripts/com.gnezim.audio-transcribe-asr.plist.template`, replace the placeholders, and keep `ASR_BIND` on loopback unless the client reaches the host over the overlay network; `ASR_TOKEN_FILE` must stay `0600`. The worker is a child of the service and is killed with it, so stopping the service leaves no orphan process.

Configuration is read from the environment: `ASR_BIND`, `ASR_SOCKET`, `ASR_TOKEN_FILE`, `ASR_WORKER_CMD`, `ASR_TONE_MODEL_DIR`, `ASR_TONE_PROMPT`, `ASR_WHISPER_MODEL`, `ASR_LANGUAGE` (default `ru`), `ASR_TIMEOUT_S` (default 60).

### Service logs

The service appends one JSON line per request and lifecycle event to `ASR_LOG_FILE` (`service_start`, `worker_start`, `worker_dropped`, `transcribe`, `unauthorized`) and rotates that file by size: `ASR_LOG_MAX_BYTES` (default 4 MiB) with `ASR_LOG_KEEP` rotations (default 5), so the log is bounded at 24 MiB and never needs a cleanup job. Per-request worker output does not go to launchd's stdio files, which therefore only carry startup lines.

```zsh
scripts/asr_log_review.py --hours 24          # human summary
scripts/asr_log_review.py --hours 24 --json   # machine summary
scripts/asr_log_review.py --hours 24 --only-anomalies   # silent when clean (cron)
scripts/asr_log_review.py --hours 24 --fail-on-anomaly  # exit 2 on findings
```

The review flags error rate above 5 %, p95 above 15 s, worker restarts, token rejections, broken lines and an empty window — coarse thresholds meant to catch a broken service, not to grade latency.

## Updates

Updates are explicit and always select the latest stable release tag. The CLI never mutates its own environment during an unrelated command:

```zsh
audio-transcribe self-update
```

## Output

Each run produces a Markdown meeting note with YAML frontmatter (speakers, audio data path, timestamps) and a Transcript section. Optional downstream LLM post-processing can add sections such as Summary, Key Points, Decisions, and Action Items. This analysis step is outside `audio-transcribe` and may use any local or hosted model or provider.

## Development

```zsh
uv sync
uv sync --extra ml        # required for real transcription
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest
```

Run the real Apple Silicon smoke test after changing the ML stack:

```zsh
AUDIO_TRANSCRIBE_SMOKE_AUDIO=/path/to/short-trusted.wav uv run --extra ml pytest -m ml_smoke
```

## License

[MIT](LICENSE)

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

The installer uses the latest supported tagged release (`0.4.0`) and installs the optional ML stack. For a lightweight CLI-only development environment, use `uv sync`; for transcription backends, use `uv sync --extra ml`.

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

## Updates

Updates are explicit and always select the latest stable release tag. The CLI never mutates its own environment during an unrelated command:

```zsh
audio-transcribe self-update
```

## Output

Each run produces a Markdown meeting note with YAML frontmatter (speakers, audio data path, timestamps) and a Transcript section. Post-processing with Claude adds Summary, Key Points, Decisions, and Action Items.

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

# audio-transcribe

Local audio transcription pipeline that processes recorded meetings into structured Obsidian notes. Runs entirely on MacBook Air M4 (24 GB RAM) with no cloud APIs.

## Pipeline

```
Audio (WAV/M4A/MP3)
  → preprocess (ffmpeg)
  → transcribe (WhisperX / MLX)
  → align (wav2vec2)
  → diarize (pyannote)
  → format (Markdown meeting note)
```

## Install

```zsh
curl -fsSL https://git.gnerim.ru/gnezim/audio-transcribe/raw/branch/main/install.sh | zsh
```

The installer handles Homebrew, ffmpeg, uv, the Python package, PATH setup, and HuggingFace token configuration.

### Requirements

- macOS (Apple Silicon)
- HuggingFace token with accepted pyannote licenses ([speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0))

## Usage

```zsh
# Fast pass: transcribe + align → meeting note
audio-transcribe process recording.m4a -o meetings/

# Full pass: includes speaker diarization
audio-transcribe process recording.m4a -o meetings/ --full

# Post-process existing meeting notes
audio-transcribe diarize meetings/2026-03-01-standup.md
audio-transcribe identify meetings/2026-03-01-standup.md
audio-transcribe update meetings/2026-03-01-standup.md

# Speaker management
audio-transcribe speakers list
audio-transcribe speakers forget "Name"

# Utilities
audio-transcribe stats --last 5
audio-transcribe recommend recording.m4a
audio-transcribe learn corrected-transcript.md
audio-transcribe self-update
```

## Updates

The tool checks for updates automatically once per day (silently, on first run). To force an immediate update:

```zsh
audio-transcribe self-update
```

## Output

Each run produces a Markdown meeting note with YAML frontmatter (speakers, audio data path, timestamps) and sections for Transcript. Post-processing with Claude adds Summary, Key Points, Decisions, and Action Items.

## Development

```zsh
uv sync
uv run ruff check .
uv run mypy .
uv run pytest
```

## License

Private project.

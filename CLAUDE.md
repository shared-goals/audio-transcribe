# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

End-to-end **local** pipeline that processes recorded meetings into structured Obsidian notes. Runs entirely on MacBook Air M4 (24 GB RAM). No cloud APIs.

Full pipeline:
```
Audio (WAV/M4A/MP3)
  ↓ audio-transcribe process — preprocess + transcribe + align + diarize + format
  ↓ Claude (via Claudian /process-meeting) — structured summary + vault note
```

## Setup

```bash
uv sync                          # install dependencies
export HF_TOKEN=hf_...           # required for diarization (pyannote)
```

HuggingFace token requires accepting the pyannote license at:
https://huggingface.co/pyannote/speaker-diarization-3.1

External dependency: `ffmpeg` must be installed (`brew install ffmpeg`).
For LLM summarization: `brew install ollama && ollama pull gemma3:27b`

## Code Quality Stack

Follows the same conventions as `bft/svod-excel-generator`:

- **Python**: `>=3.12` (bump `requires-python` when adding new scripts)
- **Formatter**: `black` — line length 120
- **Linter**: `ruff` — rules `E, F, I, N, W, B, ANN`; line length 120
- **Type checker**: `mypy` — strict mode (`disallow_untyped_defs`, `warn_return_any`)
- **Tests**: `pytest` + `pytest-cov`

```bash
uv run ruff check .
uv run ruff format .        # or: uv run black .
uv run mypy .
uv run pytest
uv run pytest tests/path/to/test_file.py::test_name   # single test
```

Add dev dependencies to `pyproject.toml` under `[dependency-groups] dev` when setting up linting/testing.

## Running the Pipeline

```bash
# Full pipeline: transcribe + format transcript
audio-transcribe process input.m4a -o result.json --transcript transcript.md

# Options
audio-transcribe process input.m4a -l ru -m large-v3 --backend mlx-vad -o result.json
audio-transcribe process input.m4a --no-align --no-diarize -o result.json

# View historical run statistics
audio-transcribe stats --last 5

# Get backend recommendations for a file
audio-transcribe recommend input.m4a

# Learn corrections from an edited transcript
audio-transcribe learn corrected-transcript.md

# Test Ollama LLM connectivity and Russian summarization
uv run test_ollama.py
uv run test_ollama.py --list-models
uv run test_ollama.py -m qwen2.5:14b
```

## Critical M4 Constraint

**Always use `device="cpu"` and `compute_type="int8"`** for WhisperX on Apple Silicon. float16 crashes with ctranslate2 on M4. This is already hardcoded in the scripts.

## Model Choices

| Stage | Model | Notes |
|-------|-------|-------|
| ASR | `antony66/whisper-large-v3-russian` | 6.39% WER (vs 9.84% for base large-v3); WhisperX loads HF models directly |
| Alignment | `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | WhisperX default for `ru`; upgrade to `wav2vec2-xls-r-1b-russian` for better precision |
| Diarization | `pyannote/speaker-diarization-3.1` | Bundled with WhisperX; needs `HF_TOKEN` |
| LLM | `gemma3:27b` via Ollama | ~16 GB RAM; sequential execution means Whisper unloads before LLM starts |

## Output Format

`audio-transcribe process` outputs JSON:
```json
{
  "audio_file": "...", "language": "ru", "model": "large-v3",
  "processing_time_s": 42.0,
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "...", "speaker": "SPEAKER_00",
     "words": [{"word": "...", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_00"}]}
  ]
}
```

## Package Structure

`audio_transcribe/` Python package:
- `stages/` — preprocess, transcribe, align, diarize, format, correct
- `progress/` — events, json_reporter (JSONL), tui (rich.live)
- `stats/` — store (history.json), estimator (ETA), recommender, hardware
- `quality/` — scorecard (graded quality metrics)

Remaining utility script: `test_ollama.py`.

## Current Phase & Roadmap

**Phase 3 (Unified CLI)** — complete.

Unified `audio-transcribe` CLI replaces old loose scripts.

Planned phases:
- **Phase 4**: Enhancements — task extraction, people cards, file watcher
- **Phase 5**: Local LLM fallback — Ollama/Gemma offline pipeline

Vault lives at `/Users/gnezim/_projects/gnezim/knowledge/`. Project spec at `knowledge/projects/personal/audio-transcribe/`.

**Always update the Obsidian vault** when making progress — update the roadmap, project page, and task files at `knowledge/projects/personal/audio-transcribe/`.

## Memory Budget (24 GB M4)

Sequential execution is intentional — Whisper (~6 GB) unloads before Ollama/Gemma 27B (~16 GB) loads. Do not attempt to run both simultaneously.

## Git Conventions

Do not include `Co-Authored-By` lines in commit messages.

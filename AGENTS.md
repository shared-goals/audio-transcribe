# AGENTS.md

This file provides guidance to coding agents working with this repository.

## Project Purpose

End-to-end **local** pipeline that processes recorded meetings into structured Obsidian notes. Runs entirely on macOS. No cloud APIs.

Full pipeline:
```
Audio (WAV/M4A/MP3)
  ↓ audio-transcribe process — preprocess + transcribe + align + diarize + format
  ↓ optional external LLM workflow — summary, decisions, action items, and other analysis
```

LLM analysis is intentionally outside the `audio-transcribe` runtime. Do not assume a specific model, provider, API, or local inference engine when changing the project contract or user-facing documentation.

## Setup

```bash
uv sync --extra ml               # install development and transcription dependencies
export HF_TOKEN=hf_...           # required for diarization (pyannote)
```

HuggingFace token requires accepting the pyannote license at:
https://huggingface.co/pyannote/speaker-diarization-3.1

External dependency: `ffmpeg` must be installed (`brew install ffmpeg`).

## Code Quality Stack

Follows the same conventions as `bft/svod-excel-generator`:

- **Python**: `>=3.12` (bump `requires-python` when adding new scripts)
- **Formatter**: `ruff format` — line length 120
- **Linter**: `ruff` — rules `E, F, I, N, W, B, ANN`; line length 120
- **Type checker**: `mypy` — strict mode (`disallow_untyped_defs`, `warn_return_any`)
- **Tests**: `pytest` + `pytest-cov`

```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest
uv run pytest tests/path/to/test_file.py::test_name   # single test
```

Add dev dependencies to `pyproject.toml` under `[dependency-groups] dev` when setting up linting/testing. Keep heavy transcription dependencies in the `ml` optional extra so core CLI and CI environments remain lightweight.

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

```

## Critical M4 Constraint

**Always use `device="cpu"` and `compute_type="int8"`** for WhisperX on Apple Silicon. float16 crashes with ctranslate2 on M4. This is already hardcoded in the scripts.

## Transcription Model Choices

| Stage | Model | Notes |
|-------|-------|-------|
| ASR | `antony66/whisper-large-v3-russian` | 6.39% WER (vs 9.84% for base large-v3); WhisperX loads HF models directly |
| Alignment | `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | WhisperX default for `ru`; upgrade to `wav2vec2-xls-r-1b-russian` for better precision |
| Diarization | `pyannote/speaker-diarization-3.1` | Bundled with WhisperX; needs `HF_TOKEN` |

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
- `stages/` — preprocess, transcribe, align, diarize, format, correct, diarize_update, identify, update, loader
- `markdown/` — parser (MeetingDoc), updater (sections, frontmatter, speaker mapping)
- `speakers/` — embeddings (pyannote wespeaker), database (file-based voice DB)
- `progress/` — events, json_reporter (JSONL), tui (rich.live), composite
- `stats/` — store (history.json), estimator (ETA), recommender, hardware
- `quality/` — scorecard (graded quality metrics)
- `preflight.py` — pre-flight validation (ffmpeg, input file, HF_TOKEN)
- `util.py` — atomic file writes (crash-safe)
- `log.py` — centralized logging configuration

## Current Phase & Roadmap

**Phases 1–4** — complete. **Phase 5 (Enhancements)** — active (~95%).

Completed: unified CLI, reactive pipeline, task extraction, people cards, speakers legend, auto-diarization, meetings index, pipeline hardening.

Remaining:
- **Phase 5**: File watcher, template system
- **Phase 6**: Optional provider-neutral LLM post-processing integration

Vault lives at `/Users/gnezim/_projects/gnezim/knowledge/`. Project spec at `knowledge/projects/personal/audio-transcribe/`.

**Always update the Obsidian vault** when making progress — update the roadmap, project page, and task files at `knowledge/projects/personal/audio-transcribe/`.

## Memory Budget (24 GB M4)

Keep memory-intensive downstream analysis outside the transcription process. Workflows that run a local LLM should schedule it after transcription resources are released and choose a model appropriate for the host's available memory.

## Markdown Style

Do not wrap or break lines in markdown files. Write each paragraph or list item as a single long line.

## Release & Changelog

This project uses [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/). Version is tracked in `pyproject.toml`, `audio_transcribe/__init__.py`, the installer default, and the generated `uv.lock`.

**Per-commit rule**: When committing a `fix:`, `feat:`, or breaking change, also add a line to the `[Unreleased]` section of `CHANGELOG.md` under the appropriate heading (`### Added`, `### Fixed`, `### Changed`, `### Removed`). This keeps the changelog current while context is fresh. A release must also refresh `uv.lock` and pass `uv lock --check`.

**Releasing**: Use `/release` to bump version, stamp changelog, refresh the lockfile, commit, tag, and optionally push. The skill auto-detects the bump level from commit prefixes (`fix:` → patch, `feat:` → minor, `BREAKING CHANGE` → major) and lets you override.

## Git Conventions

Do not include `Co-Authored-By` lines in commit messages.

# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.2.1] - 2026-03-05

### Fixed
- Installer pins Python 3.12 explicitly; fix curl piped to zsh stdin interleaving
- Mirror script now pushes tags; fix zsh PATH variable name collision

## [0.2.0] - 2026-03-05

First distributable release. Summarizes all work from Phases 1-5.

### Added
- Unified CLI (`audio-transcribe`) with commands: process, diarize, identify, update, speakers, stats, recommend, learn, self-update
- Full transcription pipeline: preprocess, transcribe (whisperx/mlx/mlx-vad), align, diarize, format
- Meeting markdown parser and updater (frontmatter, sections, speaker mapping)
- Speaker voice embedding database (enroll, match, forget) with pyannote wespeaker
- Reactive pipeline: fast pass (no diarize) and full pass with `--full` flag
- Post-process commands: `diarize`, `identify`, `update` for incremental meeting enrichment
- Quality scorecard with graded metrics (A/B/C/D)
- Stats store with run history, ETA estimator, and smart recommender
- Rich TUI progress display with live spinners
- JSON-lines machine-readable output mode (`--json`)
- Language-scoped corrections system with learn/apply workflow
- Pre-flight validation (ffmpeg, input file, HF_TOKEN)
- Atomic file writes for crash safety (speaker DB, stats store)
- Logging infrastructure with centralized configuration
- Composite reporter for multi-target event dispatch
- Install script (`install.sh`) for one-command setup on macOS
- Auto-update: daily background check + `self-update` command

### Fixed
- Speaker legend format matching parser expectations
- Timestamp collision in diarize_update (subsecond precision)
- Double-letter speaker labels (AA, AB, ...) for >26 speakers
- Phrase-level replacement in learn_corrections
- Embedding dimension validation in SpeakerDB
- Zero-vector prevention in extract_speaker_embedding
- Subprocess timeouts for ffmpeg calls
- Safe filesystem paths for speaker names
- Backup before diarize overwrite

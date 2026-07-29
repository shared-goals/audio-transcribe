# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Checksum-bound diarization benchmark CLI, evaluation, combined ASR+diarization ranking, and runtime forecasts
- pyannote 3.1 and Community-1 benchmark adapters with exclusive speaker assignment
- Permutation-invariant DER and word-level speaker metrics
- Private two-speaker reference format for diarization benchmarks
- Private reference dataset preparation for reproducible ASR benchmarks
- WER, CER, terminology, missed-speech, hallucination, and VAD-boundary metrics
- Isolated MLX-VAD benchmark candidates with vocabulary and context controls
- Checksum-bound benchmark manifests, quality ranking, and two-hour runtime forecasts
- Private ASR benchmark CLI for reference preparation, candidate runs, evaluation, and reports

### Fixed
- Bind ASR evaluations and winner-context reports to exact candidate results and invalidate them on rerun
- Bind diarization evaluations to exact run results, recompute the current ASR winner, and reject malformed quality evidence
- Convert oversized diarization benchmark numeric inputs into stable validation errors
- Reject undefined collar-only diarization scores and normalize validated speaker labels consistently
- Require finite positive audio duration in diarization references and normalize speaker IDs before validation
- Reject malformed ASR winner candidate identities as clean CLI evidence errors before model launch
- Require completed, identity-matching, valid-metrics evidence before rerunning the ASR winner with context
- Reject stale winner-context rankings and legacy ASR run manifests without a required vocabulary checksum
- Bind ASR benchmark runs and evaluations to accepted audio, transcript, and vocabulary checksums; use source duration for RTF and preserve failures when Git provenance is unavailable
- Reject duplicate, inconsistent, or non-finite ASR benchmark runs before writing strict JSON and Markdown reports
- Preserve mixed reviewed/unclear candidate text and detect partial-overlap hallucinations in ASR benchmarks
- Normalize underscores as separators and require whole-token terminology matches in ASR benchmarks
- Keep benchmark-only VAD boundaries out of production transcription checkpoints

## [0.5.1] - 2026-07-28

### Fixed
- Build PyAV 15 against Homebrew FFmpeg 7 on macOS and patch TorchCodec RPATHs with ad-hoc signing so diarization loads both libraries through one compatible FFmpeg ABI
- Use the current MLX cache-limit API without emitting a deprecation warning

## [0.5.0] - 2026-07-13

### Added
- Resumable, content-addressed pipeline checkpoints with per-run JSON manifests, selective `--restart-from`, `--force`, and normalized-audio cleanup
- Durable SQLite worker queue with atomic claims, crash recovery, exponential retry, dead-letter state, manual retry, stable-file watching, and machine-readable health
- Named TOML configuration profiles and dependency-free meeting-note templates
- `doctor`, `batch`, and `worker enqueue|run|status|retry` commands for unattended operation
- Release automation for checksums, CycloneDX SBOMs, provenance attestations, and opt-in/scheduled Apple Silicon smoke tests
- Current, exact GitHub Actions releases (`checkout` 7.0.0, `setup-uv` 8.3.2, build provenance 4.1.1)
- LaunchAgent worker templates for replacing ad-hoc polling scripts
- Separate `mlx`, `whisperx`, and `diarization` installation extras while retaining the aggregate `ml` extra

### Changed
- Make `mlx-vad` the consistent default across CLI, profiles, and documentation
- Persist intermediate stage results only in the private state directory, not beside source recordings
- Add advisory locking around concurrent statistics and speaker-database updates
- Validate backend names and speaker ranges instead of silently falling back to WhisperX

### Fixed
- Prevent checkpoint collisions for byte-identical recordings stored at different paths
- Validate NumPy audio buffers before constructing pyannote tensors
- Preserve empty-preprocess rejection while bringing its test coverage into the release gate
- Ensure public-mirror release tags target the sanitized commit and retain safe worker assets

## [0.4.0] - 2026-07-13

### Added
- Cross-platform CI for lockfile validation, Ruff formatting and linting, strict mypy, tests with coverage, dependency auditing, and package builds
- Dependabot groups for development tools, ML dependencies, and GitHub Actions
- Security policy documenting the upstream PyTorch compatibility exception and trusted-input requirements
- Opt-in real-audio Apple Silicon smoke test for ML backend upgrades
- `--version` CLI option
- Clear pre-flight diagnostics when the optional ML dependencies are not installed

### Changed
- Refresh and constrain the compatible dependency stack, including WhisperX 3.8.6 and current CLI/development tooling
- Move heavyweight transcription libraries into the optional `ml` extra; the installer continues to install the complete stack
- Standardize formatting on Ruff and remove Black
- Install and update only stable tagged releases; remove silent background self-upgrades from normal CLI commands
- Restrict wheel and source-distribution contents and add complete package metadata
- Move `PipelineError` into a shared errors module to remove loader-to-orchestrator coupling
- Use atomic writes for generated JSON, meeting notes, and transcripts
- Log best-effort statistics persistence failures at debug level
- Narrow missing-import suppression to untyped ML libraries

### Fixed
- Keep `uv.lock` synchronized with the released package version
- Quote wiki-link values in YAML frontmatter so generated meeting notes remain parseable
- Use the current MLX cache-release API instead of the deprecated Metal-specific method

## [0.3.0] - 2026-03-05

### Added
- HF_TOKEN fallback: sync token bidirectionally with `~/.cache/huggingface/token` so GUI-launched processes (Obsidian/Claudian) can access it

### Changed
- Installer: harden with `set -eo pipefail`, dependency checks, trap cleanup on INT/TERM, idempotent PATH setup, `ZDOTDIR` support, shadowed binary detection, version pinning (`AUDIO_TRANSCRIBE_VERSION`), already-installed skip, quiet mode (`QUIET=1`), and precise `HF_TOKEN` grep

### Fixed
- Mirror workflow: always checkout main branch on tag push events
- Installer: clean up temp file after curl-piped re-exec

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

# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.13.0] - 2026-11-13

### Added
- `--expected-speakers N` for `process` and `diarize`: the calendar's participant count is treated as a weak prior — it
  never forces `num_speakers`, it only produces a review warning when the cluster count differs (guests joined, someone
  did not come or stayed silent, clusters merged). `--speakers N` remains the explicit "I know exactly" option
- `diarization.planned_speakers` in the result JSON, so a report shows what the plan was next to what was found

## [0.12.2] - 2026-11-13

### Added
- `diarize --speakers N` re-diarizes an existing note with an exact speaker count instead of the `min/max` range, for
  the long-recording experiment without re-transcribing the audio

## [0.12.1] - 2026-11-13

### Tests
- The diarization stage forwards `num_speakers` and drops `min/max` when an exact count is given; without a hint the
  range is passed unchanged

## [0.12.0] - 2026-11-13

### Added
- `process --speakers N` (and profile `num_speakers`) passes an exact speaker count to pyannote instead of the
  `min/max` range. The hint wins over `--min-speakers/--max-speakers`; when it is absent the old range behaviour is
  unchanged
- The diarization report compares the resulting cluster count with the hint and warns on a mismatch, so a wrong hint
  (guests, a shared room account, a silent participant) is visible instead of silently trusted

## [0.11.0] - 2026-11-13

### Added
- Cluster hygiene after diarization: a cluster shorter than 30 s or holding under 2 % of the speech is not a speaker;
  its segments move to the temporally nearest labelled cluster and the merge is logged. On the checked 46-minute call
  this removes the 8-second `SPEAKER_02` artefact
- `diarization` block in the result JSON: per-cluster seconds and share, `UNKNOWN` seconds and share, and warnings for
  a cluster shorter than 30 s, an `UNKNOWN` share above 5 % and a cluster count that differs from the expected one.
  Warnings are also written to the log, so oversplit is visible without listening

### Checks
- 471 tests, `ruff` and `mypy` clean; measured on two real artifacts: a clean 9-minute two-person call (2 clusters,
  0 % UNKNOWN, no warnings) and the 46-minute call (4 clusters → 3 after hygiene, `SPEAKER_02` 7.9 s merged)

## [0.10.2] - 2026-11-13

### Fixed
- `retime` recomputes every stamp from the shifted segments instead of adding the offset to the already floored
  `MM:SS`, so a line at 45.04 s prints `00:45` (before: `00:44`, one second early on shifted notes). Line count and
  segment count must match, otherwise the note is left untouched with a clear message
- `retime --restamp` rewrites display stamps from the artifact without changing the offset, for notes already shifted
  with the off-by-one stamps

## [0.10.1] - 2026-11-13

### Fixed
- Empty transcript segments are written as `[MM:SS] Speaker:` without a trailing space, and `retime` strips trailing
  whitespace from the lines it rewrites; generated notes no longer fail `git diff --check`

## [0.10.0] - 2026-11-13

### Added
- `retime --all --meetings-dir <dir>` restores absolute timings for every note in a directory that still needs it, and
  skips notes whose recorded offset already matches. `--dry-run` reports the planned shifts per note
- New runs record `timing_offset_s` in the `.audio-data` JSON as well as in the note, so `retime` can never shift the
  same artifact twice (the 0.9.0 incident where the JSON moved and the note did not)

## [0.9.5] - 2026-11-13

### Fixed
- CI matrix uses `macos-latest`: the `macos-14` label resolved to an unsupported runner configuration
  (`macos-14-arm64-us-default`) and every macOS job was cancelled before it started

## [0.9.4] - 2026-11-13

### Fixed
- CI on the published tree is green: the two help tests no longer require the private `benchmark` command, `retime`
  and `attribute` print a readable message when the segments artifact is missing instead of a raw `PipelineError`,
  and coverage is measured over shipped modules only (`benchmark/*` and the interactive TUI are omitted) with new CLI
  surface tests bringing the gate to 82.6 %

## [0.9.3] - 2026-11-13

### Fixed
- The benchmark typer is loaded through `importlib`, so mypy in the published tree no longer fails with
  `import-untyped` for the private module that the mirror does not contain (CI on GitHub was red since 0.6.3)
- GitHub releases are created even when `docs/releases/<tag>.md` is absent: notes are taken from the CHANGELOG section
  by `scripts/changelog_section.py` (release workflow failed on every tag after v0.5.0)
- The weekly `Apple Silicon ML smoke` schedule is commented out: no self-hosted macOS runner is attached, so every
  scheduled run sat until the 24-hour cancel

## [0.9.2] - 2026-11-13

### Changed
- `httpx2[socks]` is declared in the `mlx`, `whisperx`, `diarization` and `ml` extras, so Hugging Face downloads work
  behind a SOCKS proxy (ported from the stale public PR #18 instead of merging it)
- `typer>=0.27.0,<0.28` with 0.27.3 in the lock; the full CLI suite and `--help` were rerun on it
- GitHub Actions: `checkout` 7.0.1, `setup-uv` 10.2.0, `attest-build-provenance` 4.2.2

### Checks
- `pytest` 457 passed, `ruff` and `mypy` clean, `uv lock --check` OK with `--extra ml` installed

## [0.9.1] - 2026-11-13

### Fixed
- `retime` no longer re-renders the note: the offset is inserted textually and only `[MM:SS]` stamps change, so the
  diff is the timestamps plus one frontmatter line instead of reformatted sections and spacing

## [0.9.0] - 2026-11-13

### Fixed
- Timestamps now refer to the original recording: preprocessing trims leading silence, but the removed lead
  (measured from both durations) is added back to every segment and word before the note is written. Before this,
  transcripts were early by ~1.6 s on the checked recording, so the first replies played silence or the wrong speech

### Added
- `retime <note>` restores absolute timings in an existing note and its `.audio-data` JSON: it measures the removed
  lead again, rewrites only timestamps (text and speaker labels untouched) and records `timing_offset_s`, so a second
  run cannot shift the note twice. `--dry-run` reports without writing

## [0.8.0] - 2026-11-13

### Added
- `attribute --label` on a single segment or a partial range now labels **only those transcript lines** and
  leaves `frontmatter.speakers` untouched, so a line can carry an actor while the rest of the cluster keeps its
  label. The log records `partial: true`, a whole-cluster label still updates the legend as before

## [0.7.2] - 2026-11-13

### Fixed
- Type the optional benchmark typer and isolate the import-guard test in a subprocess, so the full suite
  passes again and `mypy .` is clean

## [0.7.1] - 2026-11-13

### Fixed
- The CLI imports the benchmark typer lazily, so a package built without the private benchmark module still
  starts; before this, every command failed with `ModuleNotFoundError: audio_transcribe.benchmark`

### Changed
- The public mirror now installs the stripped tree into a temporary environment and runs
  `audio-transcribe --version` before pushing, so a broken public package cannot be published again

## [0.7.0] - 2026-11-13

### Added
- `attribute --audio-data PATH`: apply a speaker correction when the segments JSON is not next to the note,
  e.g. a cached copy fetched from the processing host. Lets a client correct speakers without moving or
  rewriting the vault artifact

## [0.6.4] - 2026-11-13

### Changed
- The public mirror also excludes `docs/superpowers` (internal benchmark and quality design notes), so no
  benchmark material is published outside the internal remote

## [0.6.3] - 2026-11-13

### Changed
- The public GitHub mirror excludes the private benchmark CLI (`audio_transcribe/benchmark`,
  `tests/benchmark`) and its README section; the benchmark package stays on the internal remote only

## [0.6.2] - 2026-11-13

### Fixed
- `attribute --versions` no longer demands `--speaker`; a correction without `--speaker` exits 2 with a
  readable message instead of a bare option error

## [0.6.1] - 2026-11-13

### Fixed
- `attribute` matches the transcript line of a speaker that has no entry in `frontmatter.speakers`
  (`UNKNOWN` is rendered as `Unknown`, other clusters as their raw id), so correcting an
  undiarized recording no longer aborts with "строка транскрипта не найдена"

## [0.6.0] - 2026-11-13

### Added
- `audio-transcribe attribute`: manual speaker corrections as append-only versions. The log lives in
  `meetings/.attribution/<stem>.jsonl` and is created only when the first manual correction happens;
  each line records version, base version, author, reason, scope, speaker before/after, affected segment
  count and the note hash before the edit
- Correction scopes: one segment, a time range over whole segments, or a whole acoustic cluster. Moving
  part of a cluster to an existing cluster is allowed; a new label is accepted only for the whole cluster,
  because `frontmatter.speakers` maps one cluster to one label
- Stale `--base-version` is refused before any write; a hand-edited transcript line that cannot be found
  aborts the correction; a failed note write truncates the appended log line back
- `attribute --versions` prints the recorded versions inspectably
- ASR service for short requests: a resident worker (`audio_transcribe.asr_worker`) keeps `mlx-whisper large-v3` warm over a Unix socket, and a Rust front (`service/`, axum) owns the queue, timeouts, bearer token and worker lifecycle behind `POST /v1/audio/transcriptions` and `GET /health`
- Optional streaming T-one Russian first pass in the worker: degraded answer when the final pass fails, or `initial_prompt` with `--tone-prompt`
- `keep_model` flag for `transcribe_mlx_vad` so a resident caller keeps the MLX model cached
- `asr` extra (`sherpa-onnx`, `mlx-whisper`) and a launchd template for the service

- Bounded JSON-lines event log in the service (`ASR_LOG_FILE`/`ASR_LOG_MAX_BYTES`/`ASR_LOG_KEEP`, default 4 MiB × 5) with rotation on write; per-request output no longer reaches the launchd stdio files
- `scripts/asr_log_review.py`: 24 h summary of traffic, latency percentiles, error rate, degraded answers, worker restarts and token rejections; `--json`, `--only-anomalies`, `--fail-on-anomaly`

### Measured
- Mac mini M4: 9 s recording 1.8 s warm (7.2 s first call); 27 s 13.6 s; 63 s 31.8 s. Feeding the T-one text as `initial_prompt` removes punctuation and casing from the final text, so it stays opt-in

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
- Preserve actionable failed-run evidence and return exit 1 when an ASR benchmark discovers a missing optional ML package during execution
- Keep optional torch/pyannote benchmark runners lazy so core CLI and help work without the `ml` extra
- Bind machine and Markdown benchmark reports to one validated generation with rollback-safe publication
- Report the combined sequential pipeline peak as the maximum of ASR and diarization component peaks
- Replace private workstation, recording, dataset, vault, and model-cache paths in tracked benchmark documentation with reusable variables
- Include validated end-to-end combined RTF in machine-readable and Markdown benchmark reports
- Include selected ASR metrics, stage timings, DER components, and checksum evidence in combined benchmark reports
- Invalidate benchmark evaluations before artifact replacement so partial writes cannot preserve stale trusted evidence
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

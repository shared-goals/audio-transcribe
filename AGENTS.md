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

Vault lives outside the repository at `$KNOWLEDGE_VAULT`. Project spec is under `$KNOWLEDGE_VAULT/projects/personal/audio-transcribe/`.

**Always update the Obsidian vault** when making progress — update the roadmap, project page, and task files at `knowledge/projects/personal/audio-transcribe/`.

## Memory Budget (24 GB M4)

Keep memory-intensive downstream analysis outside the transcription process. Workflows that run a local LLM should schedule it after transcription resources are released and choose a model appropriate for the host's available memory.

## Markdown Style

Do not wrap or break lines in markdown files. Write each paragraph or list item as a single long line.

## Tracker

The project tracker is this repository's Gitea issues. Keep one `wayfinder:map` issue as the index of
decisions: Destination / Notes / Current state — <date> / Decisions so far / Next action / Out of scope.
Children carry their own `wayfinder:*` (grilling, research, prototype, task); implementation issues use
`bug|enhancement` plus `needs-*`. The Vault project hub links the map in one line and does not copy it.
Tracker mutations follow an explicit instruction.

## Release & Changelog

This project uses [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/). Version is tracked in `pyproject.toml`, `audio_transcribe/__init__.py`, the installer default, and the generated `uv.lock`.

**Per-commit rule**: When committing a `fix:`, `feat:`, or breaking change, also add a line to the `[Unreleased]` section of `CHANGELOG.md` under the appropriate heading (`### Added`, `### Fixed`, `### Changed`, `### Removed`). This keeps the changelog current while context is fresh. A release must also refresh `uv.lock` and pass `uv lock --check`.

**Releasing**: Use `/release` to bump version, stamp changelog, refresh the lockfile, commit, tag, and optionally push. The skill auto-detects the bump level from commit prefixes (`fix:` → patch, `feat:` → minor, `BREAKING CHANGE` → major) and lets you override.

## Versioned publication

Every change published to origin, including documentation or project rules, ships as a versioned
changeset: bump `version` in `pyproject.toml` and `audio_transcribe/__init__.py`, stamp `CHANGELOG.md`
(move `[Unreleased]` entries into the new version heading with the current date), refresh `uv.lock` and
pass `uv lock --check`, then commit. Create an annotated `v<version>` tag at that commit and publish a
Gitea release through `tea` with the same version and notes. Run `scripts/mirror-to-github.sh` when the
release must reach `audio-transcribe self-update`, which reads tags from the public GitHub mirror. Verify
that the remote commit, tag, both version files and the release agree. Preserve published tags; a retry
completes missing steps at the same commit. Unfinished local edits need no version.

Choose PATCH by default for compatible refinements of existing behaviour, internal implementation, tests
and documentation. Use MINOR for a new command, a new artifact or log format, a new CLI contract, or a
substantial functional slice; say why in the release notes. Before 1.0, an incompatible change to the
transcript artifact, the `.audio-data` JSON shape or the attribution log requires an explicit migration
decision and the next MINOR. After 1.0, incompatible public changes take MAJOR. This follows
[SemVer 2.0.0](https://semver.org/).

A release that changes runtime behaviour on `chez-mac` also needs the update applied on that host through
the project's own upgrade path (`audio-transcribe self-update`) and one short real recording verified end
to end. Report the local upgrade, the remote host state and the observed processing separately.

## Git Conventions

Do not include `Co-Authored-By` lines in commit messages.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **audio-transcribe** (1481 symbols, 2671 relationships, 71 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/audio-transcribe/context` | Codebase overview, check index freshness |
| `gitnexus://repo/audio-transcribe/clusters` | All functional areas |
| `gitnexus://repo/audio-transcribe/processes` | All execution flows |
| `gitnexus://repo/audio-transcribe/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

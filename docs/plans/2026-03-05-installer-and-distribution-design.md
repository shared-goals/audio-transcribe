# Installer and Distribution Design

**Date**: 2026-03-05
**Status**: Approved

## Problem

The tool is not distributable. The knowledge vault scripts hardcode
`$HOME/_projects/gnezim/audio-transcribe` and require `uv run` from within the
project directory. Non-technical users cannot install the tool without manual
steps. There is no versioning, changelog, or update mechanism.

## Goals

- One copy-paste command to install on any Mac (no prior knowledge of uv/Python)
- Auto-update on first daily run, plus on-demand `self-update` command
- Proper semver tagging and changelog
- Dual-remote support (Gitea primary, GitHub future public mirror)

## Target Platform

macOS only (Apple Silicon). All users have zsh as default shell.

## Design

### 1. Install Script (`install.sh`)

Hosted in the repo root. User runs:

```zsh
curl -fsSL https://git.gnerim.ru/gnezim/audio-transcribe/raw/branch/main/install.sh | zsh
```

The script is `#!/bin/zsh` (not bash) because:
- Default macOS shell is zsh
- Environment variables are stored in `~/.zshrc`
- `source ~/.zshrc` only works from zsh

**Steps in order:**

1. **Homebrew** -- check for `brew`, if missing install via the official
   Homebrew installer. Source the shell env so it is available immediately.

2. **ffmpeg** -- `brew install ffmpeg` if missing.

3. **uv** -- check for `uv`, if missing install via
   `curl -LsSf https://astral.sh/uv/install.sh | sh`. Source
   `~/.local/bin/env`.

4. **Install the tool** --
   `uv tool install git+https://git.gnerim.ru/gnezim/audio-transcribe.git`

5. **PATH setup** -- check if `~/.local/bin` is in PATH. If not, append
   `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` and source it.

6. **HF_TOKEN wizard**:
   - Check if `HF_TOKEN` is already set in env or `~/.zshrc`.
   - If missing, print explanation of what it is for.
   - Open browser tabs (via `open` on macOS):
     - https://huggingface.co/settings/tokens -- create/copy token
     - https://huggingface.co/pyannote/speaker-diarization-3.1 -- accept license
     - https://huggingface.co/pyannote/segmentation-3.0 -- accept license
     - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM -- accept license
   - Prompt user to paste their token.
   - Validate token via
     `curl -sS -H "Authorization: Bearer $token" https://huggingface.co/api/whoami`.
   - On success: append `export HF_TOKEN="..."` to `~/.zshrc`.
   - On failure: show error, let user retry or skip.

7. **Verify** -- run `audio-transcribe --help` to confirm installation works.

### 2. Auto-Update Mechanism

Built into the Python CLI in `audio_transcribe/cli.py`.

**On every CLI invocation (before any command runs):**

1. Check timestamp file at `~/.audio-transcribe/.last-update`.
2. If file is missing or older than 24 hours:
   - Run `uv tool upgrade audio-transcribe` (subprocess, capture output).
   - If network unreachable or upgrade fails -- silently skip, do not block
     the user.
   - On success -- update the timestamp file.
3. If within 24 hours -- skip, zero overhead.

**On-demand command:**

```zsh
audio-transcribe self-update
```

Forces an immediate `uv tool upgrade` regardless of timestamp. Shows output to
the user.

**Implementation:** The daily check runs as a Typer callback (before any
subcommand). `self-update` is a regular Typer command.

### 3. Versioning and Changelog

**Semantic versioning** with annotated git tags:
- Format: `vMAJOR.MINOR.PATCH` (e.g., `v0.1.0`, `v0.2.0`, `v1.0.0`)
- `pyproject.toml` `version` field stays in sync with tags.
- Tags are annotated: `git tag -a v0.2.0 -m "description"`.

**CHANGELOG.md** at repo root, following
[Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

## [Unreleased]

## [0.2.0] - 2026-03-05
### Added
- ...
### Fixed
- ...
```

**Release workflow:**

1. Update `CHANGELOG.md` -- move Unreleased items to new version section.
2. Bump version in `pyproject.toml`.
3. Commit: `release: v0.2.0`.
4. Tag: `git tag -a v0.2.0 -m "v0.2.0"`.
5. Push: `git push && git push --tags`.

### 4. Dual Remote (Future)

Gitea remains primary (`origin`). GitHub added as second remote (`github`)
when the project is ready for public release.

Push to both on release:

```zsh
git push origin --tags && git push github --tags
```

The install script URL is easy to swap -- a single variable at the top of
`install.sh`. When GitHub is added, the script can default to GitHub as the
public install source while Gitea remains for development.

### 5. Knowledge Vault Integration Changes

Once installed globally, the knowledge vault scripts simplify:

- **`scripts/process-audio-local.sh`**: Remove `WHISPERX_DIR` and `cd`. Call
  `audio-transcribe process` directly.
- **`.claude/commands/process-meeting.md`**: Remove
  `cd /Users/gnezim/_projects/gnezim/audio-transcribe` lines. Call
  `audio-transcribe diarize` and `audio-transcribe identify` directly.

These changes happen in the knowledge vault repo, not in this repo.

## Rejected Alternatives

### Homebrew Tap

Homebrew formulas work well for simple CLI tools. For a project with torch,
whisperx, and pyannote (~2 GB of ML dependencies), maintaining a formula is
painful: each dependency needs a `resource` block with pinned URL and sha256,
torch pulls dozens of transitive deps, and every version bump means
regenerating all resource blocks. `uv tool install` handles ML dependency
resolution effortlessly.

### Pure Shell Wrapper

A shell script at `/usr/local/bin/audio-transcribe` handling install, updates,
and delegation to the real binary. Fragile -- the wrapper can get out of sync
with the Python package, and having two "audio-transcribe" entities (wrapper +
real binary) creates confusion.

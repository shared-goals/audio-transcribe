# HF Token File Fallback

## Problem

When `audio-transcribe` is launched from a GUI app (e.g., Obsidian/Claudian plugin), macOS GUI apps don't inherit shell environment variables from `~/.zshrc`. This means `HF_TOKEN` is unavailable and diarization silently skips.

## Design

At CLI startup (in the `main()` typer callback in `cli.py`), before any command runs:

1. If `HF_TOKEN` is in env and `~/.cache/huggingface/token` doesn't exist → write token to cache file (create dir if needed)
2. If `HF_TOKEN` is NOT in env and cache file exists → set `os.environ["HF_TOKEN"]` from file
3. Otherwise → do nothing

## Rationale

- **Approach B (env var at startup)**: single change at the entry point, zero changes to the 4 consumer sites (`preflight.py`, `pipeline.py`, `speakers/embeddings.py`, `stages/diarize_update.py`)
- Bidirectional sync: terminal runs "seed" the cache file for future GUI sessions
- Uses the standard HuggingFace cache path (`~/.cache/huggingface/token`), same as `huggingface-cli login`

## Changes

- `cli.py`: add `_sync_hf_token()` to the `main()` callback
- Tests: mock env + filesystem to cover both directions

## Not changed (deferred)

- `install.sh`: could also write to cache file during install, but deferred to a separate change
- Preflight warning message: unchanged

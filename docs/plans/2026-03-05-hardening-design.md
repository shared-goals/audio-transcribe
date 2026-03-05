# Pipeline Hardening Design

Fix bugs, logic errors, and robustness gaps across the audio-transcribe pipeline. No new features — pure quality improvements.

## Motivation

Code audit identified 6 categories of issues: data loss bugs, unhandled errors, logic/correctness errors, robustness gaps, architectural constraints, and quality-of-life gaps. This design addresses all of them with incremental, backwards-compatible changes.

## Scope

~390 lines of changes across ~18 files. 3 new files. No architectural overhauls.

---

## Area 1: Stats Wiring + Data Loss Bugs

### 1A. Wire stats_store.append() into Pipeline

`Pipeline._run_stage()` measures timing and RSS but discards both after emitting events. No `RunRecord` is ever constructed or persisted.

**Fix**: Accumulate stage stats in `Pipeline.run()`, construct `RunRecord` at the end, persist via `stats_store.append()`.

Changes to `pipeline.py`:
- Add `self._stage_stats: dict[str, StageStats] = {}` in `__init__`
- In `_run_stage`, store timing: `self._stage_stats[name] = StageStats(...)`
- After `PipelineComplete` event, build `RunRecord` from accumulated data
- Call `detect_hardware()`, `compute_quality(segments)`, compute `realtime_ratio`
- Call `self.stats_store.append(record)` if stats_store is not None
- Wrap in try/except — stats failure must not crash the pipeline

Wire ETA estimation:
- In `_run_stage`, call `estimate_stage(name, duration, history)` if estimator history available
- Pass `eta_s` to `StageStart` event

Get audio duration early (feeds both stats and TUI):
- Call ffprobe before preprocessing to get actual duration
- Pass to `PipelineStart` event (currently hardcoded 0.0)

### 1B. Delete orphaned test

Delete `tests/test_ollama_utils.py` — imports deleted `test_ollama` module, breaks pytest collection.

### 1C. Embedding dimension validation

Add `_EMBEDDING_DIM = 256` constant to `database.py`.
- `enroll()`: raise `ValueError` if embedding shape doesn't match
- `match()`: log warning and skip entries with wrong shape (data corruption)

---

## Area 2: Error Handling

Design principle: don't scatter try/except in every stage. Add three layers: pre-flight validation, orchestrator-level stage wrapping, and error events.

### 2A. Pre-flight validation

New file: `audio_transcribe/preflight.py`

`check(audio_file, backend, skip_diarize) -> PreflightResult`

Validates:
- ffmpeg binary exists in PATH
- Input file exists and is non-empty
- HF_TOKEN set (warning if missing and diarization enabled)
- Disk space (rough check: 10x input file size)

Returns `PreflightResult(errors: list[str], warnings: list[str])` with `.ok` property.

Called at top of `Pipeline.run()`. Errors raise `PipelineError`. Warnings emitted through reporter.

### 2B. PipelineError exception

New exception class with stage context:

```python
class PipelineError(Exception):
    def __init__(self, message: str, stage: str | None = None, elapsed_s: float = 0.0): ...
```

### 2C. Stage-level error wrapping

In `Pipeline._run_stage()`, wrap `fn()` in try/except:
- `FileNotFoundError` → `PipelineError` with "file not found" context
- `MemoryError` → `PipelineError` with RSS info and "try smaller model" hint
- `Exception` → `PipelineError` with stage name and original error chained via `from e`

One place to maintain. Stages stay clean. Original traceback preserved.

### 2D. Error event type

New dataclass in `events.py`:

```python
@dataclass
class StageError:
    stage: str
    error: str
    time_s: float = 0.0
```

Add `on_stage_error` to both reporters:
- TuiReporter: red X + stage name + error message
- JsonReporter: `{"event": "stage_error", ...}`

### 2E. CLI error handling

Catch `PipelineError` in `cli.py` process command, display clean message, exit code 1.

### 2F. Shared JSON loader

Extract `_load_audio_data(meeting_path, doc) -> dict` helper for identify.py, update.py, diarize_update.py. Validates path exists, catches JSONDecodeError, raises PipelineError with clear message.

---

## Area 3: Logic / Correctness Fixes

### 3A. Speaker legend format mismatch (CRITICAL)

`format_meeting_note()` writes `- SPEAKER_00: Speaker A` but `parse_speaker_legend()` expects `- **Speaker A**: SPEAKER_00`.

Fix: Change `format_meeting_note()` and `format_transcript()` legend output to `- **{label}**: {speaker_id}`. Matches what `diarize_and_update()` already produces.

### 3B. Timestamp collision in diarize_update (CRITICAL)

`format_time()` truncates to integer seconds. Multiple segments within the same second overwrite each other in `ts_to_speaker`.

Fix: Use `f"{start:.1f}"` as the mapping key instead of `format_time(start)`. Add `_ts_to_seconds(ts)` helper to convert formatted timestamps back to float for lookup.

### 3C. >26 speaker labels

`chr(65 + idx)` produces non-letter characters after Z.

Fix: Add `_speaker_label(idx)` helper using double letters after Z (AA, AB, ...). Update prefix-stripping regex to `Speaker [A-Z]{1,2}`.

### 3D. Over-aggressive prefix stripping

`while _pfx.match(text_part)` strips all matching prefixes, including legitimate text.

Fix: Replace `while` with `if` — strip at most one prefix per line.

### 3E. learn_corrections() improvements

Only captures exact-count word replacements. Misses phrase-level corrections.

Fix: For `replace` opcodes with unequal word counts, join into phrase-level substitutions. Log warning when line counts differ (currently silent via `strict=False`).

---

## Area 4: Robustness / Edge Cases

### 4A. Atomic file writes

Every `Path.write_text()` and `np.save()` can corrupt files on crash.

New file: `audio_transcribe/util.py`
- `atomic_write_text(path, content)` — write to temp file in same directory, then `Path.replace()` (atomic on same filesystem)
- `atomic_np_save(path, arr)` — same pattern for numpy arrays
- Both clean up temp file on failure

Apply to: `database.py` (index + embeddings), `diarize_update.py` (JSON + markdown), `stats/store.py`.

Not applied to first-time pipeline output writes (no existing data to corrupt).

### 4B. FFmpeg subprocess timeout

`_load_audio_ffmpeg()` and `preprocess()` call ffmpeg with no timeout.

Fix: Add `timeout=300` (5 min) for embedding extraction, `timeout=600` (10 min) for preprocessing. `TimeoutExpired` handled by Area 2 orchestrator wrapping.

### 4C. Zero vector → None

`extract_speaker_embedding()` returns `np.zeros(256)` when no usable segments found. This flows into `enroll()` and creates entries that never match.

Fix: Return `None` instead. Change return type to `NDArray[np.float32] | None`. Update all 3 call sites (diarize_update, identify, update) to check for None. Mypy strict catches any missed call site.

### 4D. Speaker name sanitization

`_embedding_path()` only replaces spaces. Names with `/` or `\` create bad paths.

Fix: Use `re.sub(r"[^\w\-]", "_", name)` to keep only word characters and hyphens (Cyrillic-safe via `\w`). Use counter-based filenames (`name_01.npy`) to prevent collisions. Store filename in index, look up existing filename before generating new one.

### 4E. Preprocessed file cleanup

If ffmpeg crashes after partial write, corrupted `.16k.wav` remains.

Fix: Write to temp file with `.tmp` suffix, rename on success, delete on failure (try/finally).

---

## Area 5: Architecture / Design

### 5A. Logging infrastructure

17 `print(file=sys.stderr)` calls scattered across stages.

New file: `audio_transcribe/log.py`
- `configure(verbose=False)` — sets up `logging.getLogger("audio_transcribe")` with stderr handler
- Called once from CLI via `--verbose / -v` flag

Replace all 17 print statements with `logger.info()` / `logger.debug()` calls. Mechanical change.

When TUI is active, set logging level to WARNING so log output doesn't interfere with Rich Live.

### 5B. Format consolidation

`format_transcript()` and `format_meeting_note()` serve different purposes (legacy vs reactive). Keep both, but fix the legend format inconsistency (covered in 3A). Add comment documenting the distinction.

### 5C. Language-scoped corrections

Single `corrections.yaml` applies to all languages.

Fix: Support both flat (legacy) and language-scoped format in `load_corrections()`:

```yaml
# Language-scoped format:
ru:
  substitutions: { ... }
en:
  substitutions: { ... }
```

Add `language` parameter to `load_corrections()`. Flat format (no language keys) continues working. Pass `effective_language` from pipeline.

### 5D. Diarization versioning

No way to detect stale diarization.

Fix: Add `diarization_ts: <ISO timestamp>` to frontmatter after diarization. Informational only — no behavioral changes.

### 5E. Reporter composability

TUI and JSON reporters are mutually exclusive.

Add `CompositeReporter` class that dispatches events to a list of reporters. Keep existing CLI behavior (single reporter). Composite available for future use. Add `on_stage_error` method to all reporters.

---

## Area 6: Quality of Life

### 6A. Pre-flight checks

Covered in Area 2A.

### 6B. Backup before diarization overwrite

Create `.md.bak` and `.json.bak` copies at the start of `diarize_and_update()`. Only keep last backup. Add `--no-backup` CLI flag for automation.

### 6C. Orphan test cleanup

Covered in Area 1B.

### 6D. Silent HF_TOKEN skip notification

When HF_TOKEN is empty, emit a StageComplete event with `extra={"skipped": "HF_TOKEN not set"}` instead of silently skipping. TUI shows: `✓ diarize  0.0s  (skipped: HF_TOKEN not set)`.

### 6E. PipelineStart duration fix

Covered in Area 1A — call ffprobe before preprocessing.

---

## Implementation Order

1. **Area 3** (logic fixes) — correctness first, prevents wrong output
2. **Area 1** (stats wiring) — unlocks dormant ETA/recommend features
3. **Area 2** (error handling) — user experience on failures
4. **Area 4** (robustness) — data safety
5. **Area 6** (QoL) — polish
6. **Area 5** (architecture) — long-term maintainability

## Files Changed

**New files (3):**
- `audio_transcribe/preflight.py`
- `audio_transcribe/util.py`
- `audio_transcribe/log.py`

**Modified files (~15):**
- `audio_transcribe/pipeline.py` — stats wiring, stage wrapping, preflight, duration
- `audio_transcribe/stages/format.py` — legend format, speaker label helper
- `audio_transcribe/stages/diarize_update.py` — timestamp collision, prefix strip, backup, versioning, shared loader
- `audio_transcribe/stages/correct.py` — language scoping, learn_corrections improvement
- `audio_transcribe/stages/identify.py` — shared loader, zero vector check
- `audio_transcribe/stages/update.py` — shared loader, zero vector check
- `audio_transcribe/stages/preprocess.py` — timeout, temp file, logging
- `audio_transcribe/speakers/database.py` — dimension validation, atomic writes, name sanitization
- `audio_transcribe/speakers/embeddings.py` — return None, timeout
- `audio_transcribe/progress/events.py` — StageError event
- `audio_transcribe/progress/tui.py` — on_stage_error, on_warning
- `audio_transcribe/progress/json_reporter.py` — on_stage_error
- `audio_transcribe/cli.py` — PipelineError handling, --verbose, --no-backup, language passthrough
- `audio_transcribe/markdown/parser.py` — no changes needed

**Deleted files (1):**
- `tests/test_ollama_utils.py`

**New tests:**
- Stats wiring: pipeline produces RunRecord after run
- Preflight: checks pass/fail for various conditions
- Legend format: parse_speaker_legend works on format_meeting_note output
- Timestamp collision: segments within same second get correct speakers
- >26 speakers: labels and regex work correctly
- Zero vector: None returned, enrollment skipped
- Atomic writes: temp file cleaned up on failure
- Error wrapping: PipelineError carries stage context

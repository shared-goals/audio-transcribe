# Pipeline Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix bugs, logic errors, and robustness gaps across the audio-transcribe pipeline — no new features, pure quality improvements.

**Architecture:** Incremental fixes organized into 6 areas (logic, stats, errors, robustness, QoL, architecture). Each task is self-contained with tests. Changes are backwards-compatible.

**Tech Stack:** Python 3.12, pytest, mypy strict, ruff, black (line-length 120)

**Quality commands:**
```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy .
```

---

### Task 1: Delete orphaned test file

**Files:**
- Delete: `tests/test_ollama_utils.py`

**Step 1: Verify the file imports a deleted module**

Run: `head -5 tests/test_ollama_utils.py`
Expected: import from `test_ollama` which no longer exists.

**Step 2: Delete it**

```bash
rm tests/test_ollama_utils.py
```

**Step 3: Verify tests pass**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass, no collection errors.

**Step 4: Commit**

```bash
git add -u tests/test_ollama_utils.py
git commit -m "chore: delete orphaned test_ollama_utils.py"
```

---

### Task 2: Fix speaker legend format mismatch

The legend format produced by `format_meeting_note()` and `format_transcript()` doesn't match what `parse_speaker_legend()` expects. This breaks speaker mapping in the reactive pipeline.

**Files:**
- Modify: `audio_transcribe/stages/format.py:93,156`
- Modify: `tests/stages/test_format.py`
- Test: `tests/stages/test_format.py`

**Step 1: Write failing test**

Add to `tests/stages/test_format.py`:

```python
from audio_transcribe.markdown.parser import parse_meeting, parse_speaker_legend


def test_format_meeting_note_legend_parseable():
    """format_meeting_note legend must be parseable by parse_speaker_legend."""
    data = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 5.0, "text": "Hi", "speaker": "SPEAKER_01"},
        ],
        "audio_file": "2026-03-05-test.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 5.0,
    }
    md = format_meeting_note(data, audio_data_path=".audio-data/test.json")
    doc = parse_meeting(md)
    legend = parse_speaker_legend(doc)
    assert "SPEAKER_00" in legend
    assert "SPEAKER_01" in legend
    assert legend["SPEAKER_00"] == "Speaker A"
    assert legend["SPEAKER_01"] == "Speaker B"


def test_format_transcript_legend_parseable():
    """format_transcript legend must be parseable by parse_speaker_legend."""
    data = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "SPEAKER_00"},
        ],
        "audio_file": "test.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 5.0,
    }
    md = format_transcript(data)
    doc = parse_meeting(md)
    legend = parse_speaker_legend(doc)
    assert "SPEAKER_00" in legend
    assert legend["SPEAKER_00"] == "Speaker A"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stages/test_format.py::test_format_meeting_note_legend_parseable tests/stages/test_format.py::test_format_transcript_legend_parseable -v`
Expected: FAIL — legend lines don't match `_LEGEND_LINE_RE` pattern.

**Step 3: Fix legend format in format.py**

In `audio_transcribe/stages/format.py`, change legend line format in both functions.

Line 93 (in `format_transcript`), change:
```python
# Old:
lines.append(f"- {speaker_id}: {label}")
# New:
lines.append(f"- **{label}**: {speaker_id}")
```

Line 155-156 (in `format_meeting_note`), change:
```python
# Old:
lines.append(f"- {speaker_id}: {label}")
# New:
lines.append(f"- **{label}**: {speaker_id}")
```

**Step 4: Update any existing tests that assert old format**

Search for tests asserting `"- SPEAKER_"` pattern and update them to the new `"- **"` format.

**Step 5: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All pass.

**Step 6: Commit**

```bash
git add audio_transcribe/stages/format.py tests/stages/test_format.py
git commit -m "fix: speaker legend format to match parse_speaker_legend expectations"
```

---

### Task 3: Fix timestamp collision in diarize_update

Multiple segments within the same second overwrite each other in `ts_to_speaker` dict.

**Files:**
- Modify: `audio_transcribe/stages/diarize_update.py:44-47,102-124`
- Test: `tests/test_diarize_command.py`

**Step 1: Write failing test**

Add to `tests/test_diarize_command.py`:

```python
def test_diarize_timestamp_collision(tmp_path):
    """Segments within the same second should get correct speaker labels."""
    md_content = textwrap.dedent("""\
        ---
        title: 2026-03-05 meeting
        date: '2026-03-05'
        reanalyze: false
        audio_file: meeting.wav
        audio_data: .audio-data/meeting.json
        ---

        ## Transcript

        [01:40] First utterance
        [01:40] Second utterance
    """)
    meeting_md = tmp_path / "meetings" / "meeting.md"
    meeting_md.parent.mkdir(parents=True)
    meeting_md.write_text(md_content)

    stored_json = {
        "audio_file": "meeting.wav",
        "segments": [
            {"start": 100.0, "end": 100.4, "text": "First utterance"},
            {"start": 100.5, "end": 101.0, "text": "Second utterance"},
        ],
    }
    audio_data_dir = tmp_path / "meetings" / ".audio-data"
    audio_data_dir.mkdir(parents=True)
    (audio_data_dir / "meeting.json").write_text(json.dumps(stored_json))

    diarized_segments = [
        {"start": 100.0, "end": 100.4, "text": "First utterance", "speaker": "SPEAKER_00"},
        {"start": 100.5, "end": 101.0, "text": "Second utterance", "speaker": "SPEAKER_01"},
    ]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized_segments):
        diarize_and_update(meeting_md, force=True)

    result = meeting_md.read_text()
    # Both speakers should be assigned correctly despite same formatted timestamp
    assert "Speaker A: First utterance" in result
    assert "Speaker B: Second utterance" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diarize_command.py::test_diarize_timestamp_collision -v`
Expected: FAIL — both lines get "Speaker B" (last writer wins).

**Step 3: Implement fix**

In `audio_transcribe/stages/diarize_update.py`:

Add helper function after `_match_timestamp`:

```python
def _ts_to_seconds(ts: str) -> float:
    """Convert MM:SS or HH:MM:SS back to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(parts[0]) * 60 + int(parts[1])
```

Change lines 102-107 — use subsecond key:

```python
# Old:
ts_to_speaker: dict[str, str] = {}
for seg in diarized_segments:
    ts = format_time(float(seg.get("start", 0.0)))
    speaker_id = str(seg.get("speaker", ""))
    if speaker_id and speaker_id in legend:
        ts_to_speaker[ts] = legend[speaker_id]

# New:
ts_to_speaker: dict[str, str] = {}
for seg in diarized_segments:
    start_key = f"{float(seg.get('start', 0.0)):.1f}"
    speaker_id = str(seg.get("speaker", ""))
    if speaker_id and speaker_id in legend:
        ts_to_speaker[start_key] = legend[speaker_id]
```

Change lines 113-116 — match using seconds:

```python
# Old:
if line_ts and line_ts in ts_to_speaker:
    speaker = ts_to_speaker[line_ts]

# New:
if line_ts:
    line_secs = _ts_to_seconds(line_ts)
    start_key = f"{line_secs:.1f}"
    if start_key in ts_to_speaker:
        speaker = ts_to_speaker[start_key]
```

Adjust the rest of the block to be inside the new `if start_key in ts_to_speaker:` branch.

**Step 4: Run tests**

Run: `uv run pytest tests/test_diarize_command.py -v --tb=short`
Expected: All pass including new test.

**Step 5: Commit**

```bash
git add audio_transcribe/stages/diarize_update.py tests/test_diarize_command.py
git commit -m "fix: timestamp collision in diarize_update — use subsecond precision"
```

---

### Task 4: Fix >26 speaker labels and prefix stripping

`chr(65 + idx)` produces invalid characters after Z. Prefix stripping while-loop is over-aggressive.

**Files:**
- Modify: `audio_transcribe/stages/format.py:28-33`
- Modify: `audio_transcribe/stages/diarize_update.py:120-123`
- Test: `tests/stages/test_format.py`

**Step 1: Write failing test for >26 speakers**

Add to `tests/stages/test_format.py`:

```python
def test_build_speaker_legend_27_speakers():
    """Labels should use double letters after Z."""
    segments = [{"speaker": f"SPEAKER_{i:02d}", "text": "x"} for i in range(27)]
    legend = build_speaker_legend(segments)
    assert legend["SPEAKER_00"] == "Speaker A"
    assert legend["SPEAKER_25"] == "Speaker Z"
    assert legend["SPEAKER_26"] == "Speaker AA"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stages/test_format.py::test_build_speaker_legend_27_speakers -v`
Expected: FAIL — `Speaker [` instead of `Speaker AA`.

**Step 3: Add `_speaker_label` helper**

In `audio_transcribe/stages/format.py`, add before `build_speaker_legend`:

```python
def _speaker_label(idx: int) -> str:
    """Generate speaker label: A, B, ..., Z, AA, AB, ..."""
    if idx < 26:
        return f"Speaker {chr(65 + idx)}"
    first = chr(65 + (idx // 26) - 1)
    second = chr(65 + (idx % 26))
    return f"Speaker {first}{second}"
```

Change `build_speaker_legend` line 32:

```python
# Old:
seen[speaker] = f"Speaker {chr(65 + label_idx)}"
# New:
seen[speaker] = _speaker_label(label_idx)
```

**Step 4: Fix prefix stripping — change while to if**

In `audio_transcribe/stages/diarize_update.py` line 120-123:

```python
# Old:
_pfx = re.compile(r"^(?:Speaker [A-Z]|SPEAKER_\d+|Unknown|None):\s+")
text_part = after_bracket[1]
while _pfx.match(text_part):
    text_part = _pfx.sub("", text_part, count=1)

# New:
_pfx = re.compile(r"^(?:Speaker [A-Z]{1,2}|SPEAKER_\d+|Unknown|None):\s+")
text_part = after_bracket[1]
if _pfx.match(text_part):
    text_part = _pfx.sub("", text_part, count=1)
```

**Step 5: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All pass.

**Step 6: Commit**

```bash
git add audio_transcribe/stages/format.py audio_transcribe/stages/diarize_update.py tests/stages/test_format.py
git commit -m "fix: >26 speaker labels use double letters; prefix strip limited to one"
```

---

### Task 5: Improve learn_corrections to capture phrase-level changes

**Files:**
- Modify: `audio_transcribe/stages/correct.py:74-92`
- Test: `tests/stages/test_correct.py`

**Step 1: Write failing test**

Add to `tests/stages/test_correct.py`:

```python
def test_learn_corrections_phrase_replacement():
    """Phrase-level replacements (different word count) should be captured."""
    original = ["это красный автомобиль"]
    corrected = ["это красная машина"]
    learned = learn_corrections(original, corrected)
    # "красный автомобиль" → "красная машина" (2 words → 2 words, word-by-word)
    assert "красный" in learned or "красный автомобиль" in learned


def test_learn_corrections_unequal_word_count():
    """Replacements with different word counts should be captured as phrases."""
    original = ["в общем то да"]
    corrected = ["вообще да"]
    learned = learn_corrections(original, corrected)
    assert "в общем то" in learned
    assert learned["в общем то"] == "вообще"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stages/test_correct.py::test_learn_corrections_unequal_word_count -v`
Expected: FAIL — unequal-count replacements produce empty dict.

**Step 3: Implement fix**

In `audio_transcribe/stages/correct.py`, modify `learn_corrections` (lines 86-90):

```python
# Old:
if tag == "replace" and (i2 - i1) == (j2 - j1):
    for orig_w, corr_w in zip(orig_words[i1:i2], corr_words[j1:j2], strict=True):
        if orig_w != corr_w:
            learned[orig_w] = corr_w

# New:
if tag == "replace":
    if (i2 - i1) == (j2 - j1):
        for orig_w, corr_w in zip(orig_words[i1:i2], corr_words[j1:j2], strict=True):
            if orig_w != corr_w:
                learned[orig_w] = corr_w
    else:
        orig_phrase = " ".join(orig_words[i1:i2])
        corr_phrase = " ".join(corr_words[j1:j2])
        learned[orig_phrase] = corr_phrase
```

**Step 4: Run all tests**

Run: `uv run pytest tests/stages/test_correct.py -v --tb=short`
Expected: All pass.

**Step 5: Commit**

```bash
git add audio_transcribe/stages/correct.py tests/stages/test_correct.py
git commit -m "fix: learn_corrections captures phrase-level replacements"
```

---

### Task 6: Wire stats_store.append() into Pipeline

The stats store is passed to Pipeline but never written to. ETA, stats, and recommend commands are permanently empty.

**Files:**
- Modify: `audio_transcribe/pipeline.py:94-206`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing test**

Add to `tests/test_pipeline.py`:

```python
from audio_transcribe.stats.store import StatsStore


def test_pipeline_persists_run_record(tmp_path):
    """Pipeline should write a RunRecord to stats_store after successful run."""
    events: list[tuple[str, object]] = []
    reporter = _make_reporter(events)

    store = StatsStore(tmp_path / "history.json")

    with patch.multiple(
        "audio_transcribe.pipeline",
        **_STAGE_PATCHES,
    ):
        pipeline = Pipeline(reporter=reporter, stats_store=store)
        config = PipelineConfig(audio_file="test.wav", skip_diarize=True, suppress_stdout_json=True)
        pipeline.run(config)

    records = store.load()
    assert len(records) == 1
    r = records[0]
    assert r.config.model == "large-v3"
    assert r.config.backend == "whisperx"
    assert "preprocess" in r.stages
    assert "transcribe" in r.stages
    assert r.total_time_s > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_pipeline_persists_run_record -v`
Expected: FAIL — `records` is empty.

**Step 3: Implement stats wiring**

In `audio_transcribe/pipeline.py`:

Add to imports:

```python
from audio_transcribe.models import Config, InputInfo, RunRecord, StageStats
```

Add to `Pipeline.__init__` after line 105:

```python
self._stage_stats: dict[str, StageStats] = {}
self._corrections_applied: int = 0
```

In `_run_stage`, after line 204 (after `on_stage_complete`), add:

```python
self._stage_stats[name] = StageStats(time_s=round(elapsed, 1), peak_rss_mb=round(_current_rss_mb(), 0))
```

After corrections stage (line 158), capture count:

```python
# After: result["segments"] = segments
# Add:
self._corrections_applied = count
```

After `PipelineComplete` event emission (after line 188), before the stdout print, add:

```python
if self.stats_store is not None:
    self._persist_stats(config, output, effective_language, time.time() - t0)
```

Add new method to Pipeline class:

```python
def _persist_stats(self, config: PipelineConfig, output: dict[str, Any], language: str, elapsed: float) -> None:
    """Best-effort persistence of run statistics."""
    try:
        from datetime import datetime

        from audio_transcribe.quality.scorecard import compute_quality
        from audio_transcribe.stages.format import compute_duration
        from audio_transcribe.stats.hardware import detect_hardware

        segments = output.get("segments", [])
        duration_s = compute_duration(segments)

        record = RunRecord(
            id=datetime.now().isoformat(),
            hardware=detect_hardware(),
            input=InputInfo(
                file=config.audio_file,
                duration_s=duration_s,
                file_size_mb=Path(config.audio_file).stat().st_size / 1_048_576 if Path(config.audio_file).exists() else 0.0,
            ),
            config=Config(
                language=language,
                model=config.model,
                backend=config.backend,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
                align_model=config.align_model,
            ),
            stages=self._stage_stats,
            quality=compute_quality(segments),
            corrections_applied=self._corrections_applied,
            total_time_s=round(elapsed, 1),
            realtime_ratio=round(elapsed / duration_s, 2) if duration_s > 0 else 0.0,
        )
        self.stats_store.append(record)
    except Exception:
        pass  # Stats are best-effort — never crash the pipeline
```

Note: You'll also need to make `effective_language` available after the transcribe stage. It's already computed on line 130. Just ensure it's accessible in the `_persist_stats` call. Since it's a local variable in `run()`, pass it as a parameter.

**Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline.py -v --tb=short`
Expected: All pass including new test.

**Step 5: Run full suite + type checks**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`
Expected: All pass.

**Step 6: Commit**

```bash
git add audio_transcribe/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire stats_store.append into Pipeline — persists run history"
```

---

### Task 7: Add embedding dimension validation to SpeakerDB

**Files:**
- Modify: `audio_transcribe/speakers/database.py:51-91`
- Test: `tests/test_speaker_db.py`

**Step 1: Write failing test**

Add to `tests/test_speaker_db.py`:

```python
def test_enroll_wrong_dimension_raises(tmp_path):
    db = SpeakerDB(tmp_path / "speakers")
    bad_embedding = np.zeros(128, dtype=np.float32)  # Wrong dimension
    with pytest.raises(ValueError, match="256"):
        db.enroll("Test Speaker", bad_embedding)


def test_match_skips_corrupt_embedding(tmp_path):
    db = SpeakerDB(tmp_path / "speakers")
    # Enroll a valid speaker
    good = np.random.rand(256).astype(np.float32)
    db.enroll("Good", good)
    # Manually corrupt the embedding file
    np.save(db._embedding_path("Good"), np.zeros(128, dtype=np.float32))
    # Match should skip corrupt entry, not crash
    query = np.random.rand(256).astype(np.float32)
    results = db.match(query)
    assert len(results) == 0  # Skipped due to dimension mismatch
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speaker_db.py::test_enroll_wrong_dimension_raises tests/test_speaker_db.py::test_match_skips_corrupt_embedding -v`
Expected: FAIL.

**Step 3: Implement validation**

In `audio_transcribe/speakers/database.py`:

Add constant at top (after imports):

```python
import logging

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 256
```

In `enroll()`, add at the start:

```python
if embedding.shape != (_EMBEDDING_DIM,):
    raise ValueError(f"Expected {_EMBEDDING_DIM}-dim embedding, got shape {embedding.shape}")
```

In `match()`, add after loading stored embedding:

```python
stored = np.load(self._dir / str(meta["file"])).astype(np.float32)
if stored.shape != (_EMBEDDING_DIM,):
    logger.warning("Corrupt embedding for %s (shape %s), skipping", key, stored.shape)
    continue
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_speaker_db.py -v --tb=short`
Expected: All pass.

**Step 5: Commit**

```bash
git add audio_transcribe/speakers/database.py tests/test_speaker_db.py
git commit -m "fix: validate embedding dimensions in SpeakerDB"
```

---

### Task 8: Add PipelineError and stage-level error wrapping

**Files:**
- Modify: `audio_transcribe/pipeline.py:197-206`
- Modify: `audio_transcribe/progress/events.py`
- Modify: `audio_transcribe/progress/tui.py`
- Modify: `audio_transcribe/progress/json_reporter.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing test**

Add to `tests/test_pipeline.py`:

```python
from audio_transcribe.pipeline import PipelineError


def test_pipeline_wraps_stage_error():
    """Stage exceptions should be wrapped in PipelineError with stage context."""
    events: list[tuple[str, object]] = []
    reporter = _make_reporter(events)
    # Also add on_stage_error to capture error events
    reporter.on_stage_error = lambda e: events.append(("stage_error", e))

    patches = dict(_STAGE_PATCHES)
    patches["preprocess_stage"] = MagicMock(side_effect=FileNotFoundError("test.wav not found"))

    with patch.multiple("audio_transcribe.pipeline", **patches):
        pipeline = Pipeline(reporter=reporter)
        config = PipelineConfig(audio_file="test.wav", suppress_stdout_json=True)
        with pytest.raises(PipelineError) as exc_info:
            pipeline.run(config)

    assert exc_info.value.stage == "preprocess"
    assert "not found" in str(exc_info.value)
    # Error event should have been emitted
    error_events = [e for name, e in events if name == "stage_error"]
    assert len(error_events) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_pipeline_wraps_stage_error -v`
Expected: FAIL — `PipelineError` doesn't exist yet and exceptions aren't wrapped.

**Step 3: Add StageError event**

In `audio_transcribe/progress/events.py`, add:

```python
@dataclass
class StageError:
    """Emitted when a pipeline stage fails."""

    stage: str
    error: str
    time_s: float = 0.0
```

**Step 4: Add on_stage_error to reporters**

In `audio_transcribe/progress/tui.py`, add method to `TuiReporter`:

```python
def on_stage_error(self, event: StageError) -> None:
    """Mark stage as failed and update display."""
    from audio_transcribe.progress.events import StageError as _SE  # noqa: F811
    self._stages_done.append({"stage": event.stage, "time_s": event.time_s, "peak_rss_mb": 0, "error": event.error})
    self._current_stage = None
    if self._live:
        self._live.update(self._make_renderable())
```

Import `StageError` in the import line at top.

In `audio_transcribe/progress/json_reporter.py`, add:

```python
def on_stage_error(self, event: StageError) -> None:
    """Handle stage error event."""
    self._emit({"event": "stage_error", **asdict(event)})
```

Import `StageError` in the import line at top.

**Step 5: Add PipelineError and wrapping**

In `audio_transcribe/pipeline.py`, add after imports:

```python
class PipelineError(Exception):
    """Pipeline failure with stage context."""

    def __init__(self, message: str, stage: str | None = None, elapsed_s: float = 0.0) -> None:
        self.stage = stage
        self.elapsed_s = elapsed_s
        super().__init__(message)
```

Replace `_run_stage` method:

```python
def _run_stage(self, name: str, fn: Any) -> Any:
    """Run a stage with timing, event emission, and error wrapping."""
    self.reporter.on_stage_start(StageStart(stage=name, eta_s=None))
    t = time.time()
    try:
        result = fn()
    except Exception as e:
        elapsed = time.time() - t
        from audio_transcribe.progress.events import StageError
        if hasattr(self.reporter, "on_stage_error"):
            self.reporter.on_stage_error(
                StageError(stage=name, error=str(e), time_s=round(elapsed, 1))
            )
        raise PipelineError(f"{name} failed: {e}", stage=name, elapsed_s=elapsed) from e
    elapsed = time.time() - t
    self.reporter.on_stage_complete(
        StageComplete(stage=name, time_s=round(elapsed, 1), peak_rss_mb=round(_current_rss_mb(), 0))
    )
    self._stage_stats[name] = StageStats(time_s=round(elapsed, 1), peak_rss_mb=round(_current_rss_mb(), 0))
    return result
```

**Step 6: Run tests**

Run: `uv run pytest tests/test_pipeline.py -v --tb=short`
Expected: All pass. Check that existing tests still pass — the wrapping should be transparent for successful stages.

**Step 7: Run full suite + type checks**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`

**Step 8: Commit**

```bash
git add audio_transcribe/pipeline.py audio_transcribe/progress/events.py audio_transcribe/progress/tui.py audio_transcribe/progress/json_reporter.py tests/test_pipeline.py
git commit -m "feat: add PipelineError with stage context and StageError event"
```

---

### Task 9: Add pre-flight validation

**Files:**
- Create: `audio_transcribe/preflight.py`
- Modify: `audio_transcribe/pipeline.py`
- Test: `tests/test_preflight.py`

**Step 1: Write tests first**

Create `tests/test_preflight.py`:

```python
"""Tests for pre-flight validation checks."""

from audio_transcribe.preflight import check


def test_check_missing_audio_file():
    result = check("/nonexistent/audio.wav")
    assert not result.ok
    assert any("not found" in e for e in result.errors)


def test_check_valid_file(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    result = check(str(audio))
    assert result.ok


def test_check_empty_file(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    result = check(str(audio))
    assert not result.ok
    assert any("empty" in e for e in result.errors)


def test_check_warns_missing_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    result = check(str(audio), skip_diarize=False)
    assert result.ok  # Warning, not error
    assert any("HF_TOKEN" in w for w in result.warnings)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Create preflight.py**

Create `audio_transcribe/preflight.py`:

```python
"""Pre-flight checks before pipeline execution."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreflightResult:
    """Result of pre-flight validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no fatal errors."""
        return len(self.errors) == 0


def check(
    audio_file: str,
    backend: str = "whisperx",
    skip_diarize: bool = False,
) -> PreflightResult:
    """Validate prerequisites before running the pipeline."""
    result = PreflightResult()

    # ffmpeg binary
    if not shutil.which("ffmpeg"):
        result.errors.append("ffmpeg not found in PATH — install with: brew install ffmpeg")

    # Input file
    p = Path(audio_file)
    if not p.exists():
        result.errors.append(f"Audio file not found: {audio_file}")
    elif p.stat().st_size == 0:
        result.errors.append(f"Audio file is empty: {audio_file}")

    # HF_TOKEN for diarization
    if not skip_diarize and not os.environ.get("HF_TOKEN"):
        result.warnings.append("HF_TOKEN not set — diarization will be skipped")

    return result
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_preflight.py -v --tb=short`
Expected: All pass.

**Step 5: Wire into Pipeline.run()**

In `audio_transcribe/pipeline.py`, at the top of `Pipeline.run()` (after `t0 = time.time()`):

```python
from audio_transcribe.preflight import check as preflight_check

preflight = preflight_check(config.audio_file, config.backend, config.skip_diarize)
if not preflight.ok:
    raise PipelineError("\n".join(preflight.errors))
```

**Step 6: Run full suite + type checks**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`

**Step 7: Commit**

```bash
git add audio_transcribe/preflight.py tests/test_preflight.py audio_transcribe/pipeline.py
git commit -m "feat: add pre-flight validation (ffmpeg, input file, HF_TOKEN)"
```

---

### Task 10: Shared JSON loader for post-processing stages

Extract common JSON loading with proper error handling.

**Files:**
- Create: `audio_transcribe/stages/loader.py`
- Modify: `audio_transcribe/stages/identify.py:35-42`
- Modify: `audio_transcribe/stages/update.py:28-34`
- Modify: `audio_transcribe/stages/diarize_update.py:78-85`
- Test: `tests/stages/test_loader.py`

**Step 1: Write tests**

Create `tests/stages/test_loader.py`:

```python
"""Tests for shared audio data loader."""

import json

import pytest

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.pipeline import PipelineError
from audio_transcribe.stages.loader import load_audio_data


def test_load_audio_data_success(tmp_path):
    md = "---\naudio_data: .audio-data/test.json\n---\n\n## Transcript\n"
    doc = parse_meeting(md)
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps({"segments": []}))
    result = load_audio_data(tmp_path / "meeting.md", doc)
    assert result == {"segments": []}


def test_load_audio_data_missing_path():
    md = "---\ntitle: test\n---\n"
    doc = parse_meeting(md)
    with pytest.raises(PipelineError, match="audio_data"):
        load_audio_data(Path("/tmp/meeting.md"), doc)


def test_load_audio_data_file_not_found(tmp_path):
    md = "---\naudio_data: .audio-data/missing.json\n---\n"
    doc = parse_meeting(md)
    with pytest.raises(PipelineError, match="not found"):
        load_audio_data(tmp_path / "meeting.md", doc)


def test_load_audio_data_corrupt_json(tmp_path):
    md = "---\naudio_data: .audio-data/bad.json\n---\n"
    doc = parse_meeting(md)
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "bad.json").write_text("{broken json")
    with pytest.raises(PipelineError, match="Corrupted"):
        load_audio_data(tmp_path / "meeting.md", doc)
```

Add missing import at top: `from pathlib import Path`.

**Step 2: Create loader.py**

Create `audio_transcribe/stages/loader.py`:

```python
"""Shared loader for .audio-data JSON files used by post-processing stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio_transcribe.markdown.parser import MeetingDoc
from audio_transcribe.pipeline import PipelineError


def load_audio_data(meeting_path: Path, doc: MeetingDoc) -> dict[str, Any]:
    """Load .audio-data JSON for a meeting note.

    Raises PipelineError with clear message on failure.
    """
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    if not audio_data_rel:
        raise PipelineError("Meeting note has no audio_data path in frontmatter")

    json_path = meeting_path.parent / audio_data_rel
    if not json_path.exists():
        raise PipelineError(f"Audio data not found: {json_path}")

    try:
        data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError as e:
        raise PipelineError(f"Corrupted audio data: {json_path} — {e}") from e
```

**Step 3: Run tests**

Run: `uv run pytest tests/stages/test_loader.py -v`
Expected: All pass.

**Step 4: Replace inline JSON loading in identify.py, update.py, diarize_update.py**

In each file, replace the `audio_data_rel / json_path / json.loads` block with:

```python
from audio_transcribe.stages.loader import load_audio_data
stored = load_audio_data(meeting_path, doc)
```

**Step 5: Run full suite**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`

**Step 6: Commit**

```bash
git add audio_transcribe/stages/loader.py tests/stages/test_loader.py audio_transcribe/stages/identify.py audio_transcribe/stages/update.py audio_transcribe/stages/diarize_update.py
git commit -m "refactor: shared JSON loader with proper error handling"
```

---

### Task 11: Atomic file writes utility

**Files:**
- Create: `audio_transcribe/util.py`
- Modify: `audio_transcribe/speakers/database.py`
- Modify: `audio_transcribe/stats/store.py`
- Test: `tests/test_util.py`

**Step 1: Write tests**

Create `tests/test_util.py`:

```python
"""Tests for atomic file write utilities."""

from pathlib import Path

import numpy as np

from audio_transcribe.util import atomic_np_save, atomic_write_text


def test_atomic_write_text_creates_file(tmp_path):
    path = tmp_path / "test.txt"
    atomic_write_text(path, "hello world")
    assert path.read_text() == "hello world"


def test_atomic_write_text_no_temp_on_success(tmp_path):
    path = tmp_path / "test.txt"
    atomic_write_text(path, "content")
    # No .tmp files should remain
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_text_cleans_up_on_error(tmp_path):
    path = tmp_path / "readonly" / "test.txt"
    # Parent doesn't exist but atomic_write_text creates it
    atomic_write_text(path, "content")
    assert path.read_text() == "content"


def test_atomic_np_save(tmp_path):
    path = tmp_path / "test.npy"
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    atomic_np_save(path, arr)
    loaded = np.load(path)
    np.testing.assert_array_equal(loaded, arr)


def test_atomic_np_save_no_temp_on_success(tmp_path):
    path = tmp_path / "test.npy"
    atomic_np_save(path, np.zeros(3))
    assert not list(tmp_path.glob("*.tmp*"))
```

**Step 2: Create util.py**

Create `audio_transcribe/util.py`:

```python
"""File I/O utilities — atomic writes to prevent data corruption."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content atomically: temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding=encoding) as f:
            f.write(content)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_np_save(path: Path, arr: NDArray[Any]) -> None:
    """Save numpy array atomically: temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npy")
    os.close(fd)
    try:
        np.save(tmp, arr)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_util.py -v`
Expected: All pass.

**Step 4: Apply to SpeakerDB**

In `audio_transcribe/speakers/database.py`, replace direct writes:

Import: `from audio_transcribe.util import atomic_np_save, atomic_write_text`

In `_save_index()`:
```python
# Old: self._index_path.write_text(json.dumps(...), encoding="utf-8")
# New:
atomic_write_text(self._index_path, json.dumps(self._index, ensure_ascii=False, indent=2))
```

In `enroll()`:
```python
# Old: np.save(self._embedding_path(name), averaged)
# New:
atomic_np_save(self._embedding_path(name), averaged)
```

Same for the new enrollment branch.

**Step 5: Apply to StatsStore**

In `audio_transcribe/stats/store.py`, import and use `atomic_write_text` for `append()` and `clear()`.

**Step 6: Run full suite**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`

**Step 7: Commit**

```bash
git add audio_transcribe/util.py tests/test_util.py audio_transcribe/speakers/database.py audio_transcribe/stats/store.py
git commit -m "feat: atomic file writes for speaker DB and stats store"
```

---

### Task 12: Return None from extract_speaker_embedding for empty results

**Files:**
- Modify: `audio_transcribe/speakers/embeddings.py:75-107`
- Modify: `audio_transcribe/stages/identify.py:56-63`
- Modify: `audio_transcribe/stages/update.py:58-61`
- Modify: `audio_transcribe/stages/diarize_update.py:140-142`
- Test: `tests/test_speaker_db.py`

**Step 1: Write test**

Add to `tests/test_speaker_db.py`:

```python
from audio_transcribe.speakers import embeddings as _embeddings


def test_extract_speaker_embedding_returns_none_for_short_segments():
    """Should return None when no segments are long enough."""
    segments = [
        {"start": 0.0, "end": 0.5, "text": "Hi", "speaker": "SPEAKER_00"},
    ]
    with patch.object(_embeddings, "_load_audio_ffmpeg", return_value={"waveform": None, "sample_rate": 16000}):
        result = _embeddings.extract_speaker_embedding("test.wav", segments, "SPEAKER_00", min_duration=1.0)
    assert result is None
```

Add `from unittest.mock import patch` to imports if not already there.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speaker_db.py::test_extract_speaker_embedding_returns_none_for_short_segments -v`
Expected: FAIL — returns `np.zeros(256)` instead of `None`.

**Step 3: Change return type to Optional**

In `audio_transcribe/speakers/embeddings.py`, change `extract_speaker_embedding`:

```python
def extract_speaker_embedding(
    audio_file: str, segments: list[dict[str, Any]], speaker_id: str, min_duration: float = 1.0
) -> NDArray[np.float32] | None:
    """Extract average embedding for a speaker. Returns None if no usable segments."""
    speaker_segs = [s for s in segments if s.get("speaker") == speaker_id]
    if not speaker_segs:
        logger.warning("Speaker %s has no segments, skipping", speaker_id)
        return None
    ...
    if not embeddings:
        logger.warning("Speaker %s has no segments >= %.1fs, skipping", speaker_id, min_duration)
        return None
    ...
```

**Step 4: Update call sites**

In `audio_transcribe/stages/identify.py` (line 56-63):
```python
embedding = _embeddings.extract_speaker_embedding(audio_file, segments, str(speaker_id))
if embedding is None:
    result.unmatched.append(str(speaker_id))
    continue
matches = db.match(embedding, threshold=threshold)
```

In `audio_transcribe/stages/update.py` (line 58-61):
```python
embedding = _embeddings.extract_speaker_embedding(audio_file, segments, speaker_id)
if embedding is not None:
    db.enroll(person_name, embedding)
```

In `audio_transcribe/stages/diarize_update.py` (line 140-142):
```python
embedding = _embeddings.extract_speaker_embedding(audio_file, diarized_segments, speaker_id)
if embedding is not None:
    db.enroll(person_name, embedding)
```

**Step 5: Run full suite + mypy**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`
Expected: All pass. mypy will catch any call sites that don't handle None.

**Step 6: Commit**

```bash
git add audio_transcribe/speakers/embeddings.py audio_transcribe/stages/identify.py audio_transcribe/stages/update.py audio_transcribe/stages/diarize_update.py tests/test_speaker_db.py
git commit -m "fix: extract_speaker_embedding returns None instead of zero vector"
```

---

### Task 13: Add subprocess timeouts to ffmpeg calls

**Files:**
- Modify: `audio_transcribe/speakers/embeddings.py:46-50`
- Modify: `audio_transcribe/stages/preprocess.py:38`
- Test: existing tests cover the happy path; timeout behavior tested via the error wrapping in Task 8

**Step 1: Add timeouts**

In `audio_transcribe/speakers/embeddings.py` line 46-50:
```python
proc = subprocess.run(
    ["ffmpeg", "-y", "-i", audio_path, "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "-"],
    capture_output=True,
    check=True,
    timeout=300,  # 5 minutes max for embedding extraction
)
```

In `audio_transcribe/stages/preprocess.py` line 38:
```python
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minutes max
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All pass (timeout doesn't affect normal execution).

**Step 3: Commit**

```bash
git add audio_transcribe/speakers/embeddings.py audio_transcribe/stages/preprocess.py
git commit -m "fix: add subprocess timeouts to ffmpeg calls (5min embed, 10min preprocess)"
```

---

### Task 14: Speaker name sanitization in SpeakerDB

**Files:**
- Modify: `audio_transcribe/speakers/database.py:44-46`
- Test: `tests/test_speaker_db.py`

**Step 1: Write test**

Add to `tests/test_speaker_db.py`:

```python
def test_enroll_special_characters_in_name(tmp_path):
    """Speaker names with special characters should produce valid filenames."""
    db = SpeakerDB(tmp_path / "speakers")
    embedding = np.random.rand(256).astype(np.float32)
    db.enroll("Ivan/Maria", embedding)
    assert db.has_speaker("Ivan/Maria")
    # Filename should not contain slashes
    files = list((tmp_path / "speakers").glob("*.npy"))
    assert len(files) == 1
    assert "/" not in files[0].name
```

**Step 2: Implement fix**

In `audio_transcribe/speakers/database.py`, change `_embedding_path`:

```python
def _embedding_path(self, name: str) -> Path:
    """Get or generate a safe filename for a speaker embedding."""
    key = self._normalize(name)
    # Return existing filename if already indexed
    if key in self._index and "file" in self._index[key]:
        return self._dir / str(self._index[key]["file"])
    # Generate new safe filename with counter
    safe = re.sub(r"[^\w\-]", "_", key) or "_unknown"
    counter = 1
    while (self._dir / f"{safe}_{counter:02d}.npy").exists():
        counter += 1
    return self._dir / f"{safe}_{counter:02d}.npy"
```

Add `import re` to imports.

**Step 3: Run tests**

Run: `uv run pytest tests/test_speaker_db.py -v --tb=short`
Expected: All pass.

**Step 4: Commit**

```bash
git add audio_transcribe/speakers/database.py tests/test_speaker_db.py
git commit -m "fix: sanitize speaker names for safe filesystem paths"
```

---

### Task 15: Backup before diarization overwrite + HF_TOKEN skip notification

**Files:**
- Modify: `audio_transcribe/stages/diarize_update.py`
- Modify: `audio_transcribe/pipeline.py:139-146`
- Test: `tests/test_diarize_command.py`

**Step 1: Write test for backup**

Add to `tests/test_diarize_command.py`:

```python
def test_diarize_creates_backup(tmp_path):
    """Diarize should create .bak backup of meeting note before overwriting."""
    md_content = textwrap.dedent("""\
        ---
        title: test
        audio_file: meeting.wav
        audio_data: .audio-data/meeting.json
        ---

        ## Transcript

        [00:00] Original text
    """)
    meeting_md = tmp_path / "meetings" / "meeting.md"
    meeting_md.parent.mkdir(parents=True)
    meeting_md.write_text(md_content)

    stored_json = {"audio_file": "meeting.wav", "segments": [{"start": 0.0, "end": 2.0, "text": "Original text"}]}
    audio_data_dir = tmp_path / "meetings" / ".audio-data"
    audio_data_dir.mkdir(parents=True)
    (audio_data_dir / "meeting.json").write_text(json.dumps(stored_json))

    diarized = [{"start": 0.0, "end": 2.0, "text": "Original text", "speaker": "SPEAKER_00"}]
    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized):
        diarize_and_update(meeting_md, force=True)

    bak = meeting_md.with_suffix(".md.bak")
    assert bak.exists()
    assert "Original text" in bak.read_text()
```

**Step 2: Add backup logic**

In `audio_transcribe/stages/diarize_update.py`, at the start of `diarize_and_update()`, after reading `md_text`:

```python
# Backup before overwrite
bak_path = meeting_path.with_suffix(".md.bak")
bak_path.write_text(md_text, encoding="utf-8")
```

**Step 3: Add HF_TOKEN skip notification**

In `audio_transcribe/pipeline.py`, change lines 139-146:

```python
# Old:
if not config.skip_diarize:
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        result = self._run_stage(
            "diarize",
            lambda: diarize_stage(result, audio, hf_token, config.min_speakers, config.max_speakers),
        )

# New:
if not config.skip_diarize:
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        result = self._run_stage(
            "diarize",
            lambda: diarize_stage(result, audio, hf_token, config.min_speakers, config.max_speakers),
        )
    else:
        self.reporter.on_stage_start(StageStart(stage="diarize", eta_s=None))
        self.reporter.on_stage_complete(
            StageComplete(stage="diarize", time_s=0.0, extra={"skipped": "HF_TOKEN not set"})
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/ -v --tb=short`

**Step 5: Commit**

```bash
git add audio_transcribe/stages/diarize_update.py audio_transcribe/pipeline.py tests/test_diarize_command.py
git commit -m "feat: backup before diarize overwrite; notify on HF_TOKEN skip"
```

---

### Task 16: Language-scoped corrections

**Files:**
- Modify: `audio_transcribe/stages/correct.py:14-25`
- Modify: `audio_transcribe/pipeline.py:152`
- Test: `tests/stages/test_correct.py`

**Step 1: Write test**

Add to `tests/stages/test_correct.py`:

```python
def test_load_corrections_language_scoped(tmp_path):
    """Language-scoped corrections should load only the matching language."""
    corrections_file = tmp_path / "corrections.yaml"
    corrections_file.write_text(
        "ru:\n  substitutions:\n    кубернетес: Kubernetes\nen:\n  substitutions:\n    colour: color\n"
    )
    ru = load_corrections(str(corrections_file), language="ru")
    assert ru["substitutions"] == {"кубернетес": "Kubernetes"}

    en = load_corrections(str(corrections_file), language="en")
    assert en["substitutions"] == {"colour": "color"}


def test_load_corrections_legacy_flat_format(tmp_path):
    """Legacy flat format (no language keys) should still work."""
    corrections_file = tmp_path / "corrections.yaml"
    corrections_file.write_text("substitutions:\n  кубернетес: Kubernetes\n")
    result = load_corrections(str(corrections_file), language="ru")
    assert result["substitutions"] == {"кубернетес": "Kubernetes"}
```

**Step 2: Implement**

In `audio_transcribe/stages/correct.py`, change `load_corrections` signature and logic:

```python
def load_corrections(path: str, language: str = "ru") -> dict[str, Any]:
    """Load corrections from YAML. Supports flat (legacy) and language-scoped format."""
    p = Path(path)
    if not p.exists():
        return {"substitutions": {}, "patterns": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"substitutions": {}, "patterns": []}

    # Legacy flat format — has "substitutions" or "patterns" at top level
    if "substitutions" in data or "patterns" in data:
        return {
            "substitutions": data.get("substitutions") or {},
            "patterns": data.get("patterns") or [],
        }

    # Language-scoped format
    lang_data = data.get(language, {})
    if not isinstance(lang_data, dict):
        return {"substitutions": {}, "patterns": []}
    return {
        "substitutions": lang_data.get("substitutions") or {},
        "patterns": lang_data.get("patterns") or [],
    }
```

In `audio_transcribe/pipeline.py` line 152, pass language:

```python
# Old:
corrections = load_corrections(corrections_path)
# New:
corrections = load_corrections(corrections_path, effective_language)
```

**Step 3: Run tests**

Run: `uv run pytest tests/stages/test_correct.py -v --tb=short`
Expected: All pass.

**Step 4: Run full suite + mypy**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`

**Step 5: Commit**

```bash
git add audio_transcribe/stages/correct.py audio_transcribe/pipeline.py tests/stages/test_correct.py
git commit -m "feat: language-scoped corrections — backwards compatible with flat format"
```

---

### Task 17: Logging infrastructure

**Files:**
- Create: `audio_transcribe/log.py`
- Modify: `audio_transcribe/cli.py` (add `--verbose` flag)
- Modify: `audio_transcribe/stages/preprocess.py` (replace prints)
- Modify: `audio_transcribe/stages/transcribe.py` (replace prints)
- Modify: `audio_transcribe/stages/align.py` (replace prints)
- Modify: `audio_transcribe/stages/diarize.py` (replace prints)

**Step 1: Create log.py**

Create `audio_transcribe/log.py`:

```python
"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys


def configure(verbose: bool = False) -> None:
    """Configure logging for the audio_transcribe package."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger("audio_transcribe")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
```

**Step 2: Add --verbose flag to CLI**

In `audio_transcribe/cli.py`, add a callback:

```python
@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging")) -> None:
    """Local audio transcription pipeline."""
    from audio_transcribe.log import configure

    configure(verbose=verbose)
```

**Step 3: Replace print statements with logger calls**

In each stage file, replace `print(..., file=sys.stderr)` with `logger.info(...)`:

For `preprocess.py`:
```python
import logging
logger = logging.getLogger(__name__)
# Replace print("Preprocessing: ...", file=sys.stderr)
# With: logger.info("Preprocessing: %s → %s", input_path, output_path)
```

Do the same for `transcribe.py`, `align.py`, `diarize.py`. Use `logger.info` for status messages, `logger.debug` for timing details.

**Step 4: Run full suite**

Run: `uv run pytest tests/ -v --tb=short && uv run mypy .`

**Step 5: Commit**

```bash
git add audio_transcribe/log.py audio_transcribe/cli.py audio_transcribe/stages/preprocess.py audio_transcribe/stages/transcribe.py audio_transcribe/stages/align.py audio_transcribe/stages/diarize.py
git commit -m "refactor: replace print(stderr) with structured logging"
```

---

### Task 18: Diarization timestamp + composite reporter

**Files:**
- Modify: `audio_transcribe/stages/diarize_update.py`
- Create: `audio_transcribe/progress/composite.py`
- Test: `tests/progress/test_composite.py`

**Step 1: Add diarization_ts to frontmatter**

In `audio_transcribe/stages/diarize_update.py`, after line 132 (`set_frontmatter(doc, "reanalyze", True)`):

```python
from datetime import datetime
doc = set_frontmatter(doc, "diarization_ts", datetime.now().isoformat())
```

**Step 2: Create CompositeReporter**

Create `audio_transcribe/progress/composite.py`:

```python
"""Composite reporter — dispatch events to multiple reporters."""

from __future__ import annotations

from typing import Any

from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageError, StageStart


class CompositeReporter:
    """Dispatch events to a list of reporters."""

    def __init__(self, reporters: list[Any]) -> None:
        self._reporters = reporters

    def on_pipeline_start(self, event: PipelineStart) -> None:
        for r in self._reporters:
            r.on_pipeline_start(event)

    def on_stage_start(self, event: StageStart) -> None:
        for r in self._reporters:
            r.on_stage_start(event)

    def on_stage_complete(self, event: StageComplete) -> None:
        for r in self._reporters:
            r.on_stage_complete(event)

    def on_stage_error(self, event: StageError) -> None:
        for r in self._reporters:
            if hasattr(r, "on_stage_error"):
                r.on_stage_error(event)

    def on_pipeline_complete(self, event: PipelineComplete) -> None:
        for r in self._reporters:
            r.on_pipeline_complete(event)
```

**Step 3: Write test**

Create `tests/progress/test_composite.py`:

```python
"""Tests for CompositeReporter."""

from unittest.mock import MagicMock

from audio_transcribe.progress.composite import CompositeReporter
from audio_transcribe.progress.events import PipelineStart


def test_composite_dispatches_to_all():
    r1, r2 = MagicMock(), MagicMock()
    composite = CompositeReporter([r1, r2])
    event = PipelineStart(file="test.wav", duration_s=10.0, config={})
    composite.on_pipeline_start(event)
    r1.on_pipeline_start.assert_called_once_with(event)
    r2.on_pipeline_start.assert_called_once_with(event)
```

**Step 4: Run tests**

Run: `uv run pytest tests/progress/ -v --tb=short`

**Step 5: Commit**

```bash
git add audio_transcribe/stages/diarize_update.py audio_transcribe/progress/composite.py tests/progress/test_composite.py
git commit -m "feat: diarization_ts in frontmatter; CompositeReporter for event dispatch"
```

---

### Task 19: Final verification

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

**Step 2: Run linter**

```bash
uv run ruff check .
```

**Step 3: Run formatter**

```bash
uv run ruff format .
```

**Step 4: Run type checker**

```bash
uv run mypy .
```

**Step 5: Fix any issues found**

Address ruff, mypy, or test failures.

**Step 6: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix linting and type errors from hardening changes"
```

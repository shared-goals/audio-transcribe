# Unified CLI Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the audio-transcribe project from loose scripts into a unified CLI tool with rich progress display, historical stats for ETA prediction, quality scoring, and a correction feedback loop.

**Architecture:** Python package (`audio_transcribe/`) with `stages/`, `progress/`, `stats/`, `quality/` modules. A pipeline orchestrator emits typed events consumed by either a Rich TUI or JSON-lines reporter. Historical stats stored in `~/.audio-transcribe/history.json`, corrections in `~/.audio-transcribe/corrections.yaml`.

**Tech Stack:** Python >=3.12, typer (CLI), rich (TUI), PyYAML (corrections file). Existing deps: torch, whisperx.

**Design doc:** `docs/plans/2026-02-28-unified-cli-design.md`

---

## Task 1: Package Scaffold + Data Models

**Files:**
- Create: `audio_transcribe/__init__.py`
- Create: `audio_transcribe/models.py`
- Create: `audio_transcribe/stages/__init__.py`
- Create: `audio_transcribe/progress/__init__.py`
- Create: `audio_transcribe/stats/__init__.py`
- Create: `audio_transcribe/quality/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_models.py`

**Step 1: Write failing tests for data models**

```python
# tests/test_models.py
"""Tests for audio_transcribe.models — result dataclasses and stats schema."""

from audio_transcribe.models import (
    AlignResult,
    Config,
    DiarizeResult,
    HardwareInfo,
    InputInfo,
    PipelineResult,
    QualityMetrics,
    RunRecord,
    StageStats,
    TranscribeResult,
)


def test_transcribe_result_fields():
    r = TranscribeResult(segments=[{"start": 0.0, "end": 1.0, "text": "hi"}], language="ru", text="hi")
    assert r.language == "ru"
    assert len(r.segments) == 1


def test_align_result_inherits_segments():
    r = AlignResult(
        segments=[{"start": 0.0, "end": 1.0, "text": "hi", "words": []}],
        language="ru",
        text="hi",
        words_total=1,
        words_aligned=1,
    )
    assert r.words_total == 1
    assert r.words_aligned == 1


def test_diarize_result_has_speakers():
    r = DiarizeResult(
        segments=[{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}],
        language="ru",
        text="hi",
        words_total=1,
        words_aligned=1,
        speakers_detected=1,
        speaker_transitions=0,
    )
    assert r.speakers_detected == 1


def test_config_defaults():
    c = Config(language="ru", model="large-v3", backend="whisperx")
    assert c.min_speakers == 2
    assert c.max_speakers == 6
    assert c.align_model is None


def test_hardware_info_creation():
    h = HardwareInfo(chip="Apple M4", cores_physical=10, memory_gb=24, os="macOS 15.3", python="3.12.8")
    assert h.chip == "Apple M4"


def test_stage_stats():
    s = StageStats(time_s=42.3, peak_rss_mb=6200)
    assert s.time_s == 42.3


def test_quality_metrics():
    q = QualityMetrics(
        segments=142,
        words_total=4200,
        words_aligned=4050,
        alignment_pct=96.4,
        speakers_detected=3,
        speaker_coverage_pct=94.2,
        speaker_transitions=87,
    )
    assert q.alignment_pct == 96.4


def test_quality_metrics_grade():
    q_a = QualityMetrics(
        segments=100, words_total=100, words_aligned=96, alignment_pct=96.0,
        speakers_detected=3, speaker_coverage_pct=95.0, speaker_transitions=50,
    )
    assert q_a.grade == "A"

    q_b = QualityMetrics(
        segments=100, words_total=100, words_aligned=90, alignment_pct=90.0,
        speakers_detected=3, speaker_coverage_pct=80.0, speaker_transitions=50,
    )
    assert q_b.grade == "B"

    q_c = QualityMetrics(
        segments=100, words_total=100, words_aligned=70, alignment_pct=70.0,
        speakers_detected=3, speaker_coverage_pct=60.0, speaker_transitions=50,
    )
    assert q_c.grade == "C"


def test_run_record_total_time():
    r = RunRecord(
        id="2026-02-28T14:30:00Z",
        hardware=HardwareInfo(chip="Apple M4", cores_physical=10, memory_gb=24, os="macOS 15.3", python="3.12.8"),
        input=InputInfo(file="test.wav", duration_s=60.0, file_size_mb=1.9),
        config=Config(language="ru", model="large-v3", backend="whisperx"),
        stages={"transcribe": StageStats(time_s=10.0, peak_rss_mb=3000)},
        quality=None,
        corrections_applied=0,
        total_time_s=10.0,
        realtime_ratio=0.167,
    )
    assert r.total_time_s == 10.0


def test_pipeline_result_creation():
    r = PipelineResult(
        audio_file="test.wav",
        language="ru",
        model="large-v3",
        segments=[],
        processing_time_s=10.0,
    )
    assert r.audio_file == "test.wav"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_transcribe'`

**Step 3: Update pyproject.toml**

Add to `pyproject.toml`:

```toml
[project.scripts]
audio-transcribe = "audio_transcribe.cli:app"

[project]
# ... existing fields ...
dependencies = [
    "torch>=2.8.0",
    "torchaudio>=2.8.0",
    "whisperx>=3.8.0",
    "typer>=0.15.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
]
```

Add `typer`, `rich`, `pyyaml` to dependencies. Add `[project.scripts]` entry.

Also add ruff ignore for `audio_transcribe/__init__.py` and update per-file-ignores for tests subdirs:

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["ANN"]
```

**Step 4: Create package scaffold**

Create empty `__init__.py` files:
- `audio_transcribe/__init__.py` — with `__version__ = "0.2.0"`
- `audio_transcribe/stages/__init__.py`
- `audio_transcribe/progress/__init__.py`
- `audio_transcribe/stats/__init__.py`
- `audio_transcribe/quality/__init__.py`

**Step 5: Implement data models**

```python
# audio_transcribe/models.py
"""Data models for the audio-transcribe pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscribeResult:
    """Output from the transcription stage."""

    segments: list[dict[str, Any]]
    language: str
    text: str


@dataclass
class AlignResult:
    """Output from the alignment stage."""

    segments: list[dict[str, Any]]
    language: str
    text: str
    words_total: int
    words_aligned: int


@dataclass
class DiarizeResult:
    """Output from the diarization stage."""

    segments: list[dict[str, Any]]
    language: str
    text: str
    words_total: int
    words_aligned: int
    speakers_detected: int
    speaker_transitions: int


@dataclass
class Config:
    """Pipeline configuration."""

    language: str
    model: str
    backend: str
    min_speakers: int = 2
    max_speakers: int = 6
    align_model: str | None = None


@dataclass
class HardwareInfo:
    """Machine hardware fingerprint."""

    chip: str
    cores_physical: int
    memory_gb: int
    os: str
    python: str


@dataclass
class InputInfo:
    """Input audio file characteristics."""

    file: str
    duration_s: float
    file_size_mb: float
    sample_rate: int = 16000


@dataclass
class StageStats:
    """Timing and memory stats for a single pipeline stage."""

    time_s: float
    peak_rss_mb: float


@dataclass
class QualityMetrics:
    """Quality scorecard for a transcription run."""

    segments: int
    words_total: int
    words_aligned: int
    alignment_pct: float
    speakers_detected: int
    speaker_coverage_pct: float
    speaker_transitions: int

    @property
    def grade(self) -> str:
        """Letter grade: A (>95% aligned, >90% speaker), B (>85%, >75%), C (below)."""
        if self.alignment_pct > 95 and self.speaker_coverage_pct > 90:
            return "A"
        if self.alignment_pct > 85 and self.speaker_coverage_pct > 75:
            return "B"
        return "C"


@dataclass
class RunRecord:
    """A single historical run entry for ~/.audio-transcribe/history.json."""

    id: str
    hardware: HardwareInfo
    input: InputInfo
    config: Config
    stages: dict[str, StageStats]
    quality: QualityMetrics | None
    corrections_applied: int
    total_time_s: float
    realtime_ratio: float


@dataclass
class PipelineResult:
    """Final output of the full pipeline."""

    audio_file: str
    language: str
    model: str
    segments: list[dict[str, Any]]
    processing_time_s: float
    quality: QualityMetrics | None = None
    processing: RunRecord | None = None
```

**Step 6: Create placeholder cli.py**

```python
# audio_transcribe/cli.py
"""CLI entry point for audio-transcribe."""

import typer

app = typer.Typer(name="audio-transcribe", help="Local audio transcription pipeline.")


@app.command()
def process() -> None:
    """Run the full transcription pipeline."""
    typer.echo("Not implemented yet.")


if __name__ == "__main__":
    app()
```

**Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: all PASS

**Step 8: Run linters**

Run: `uv run ruff check audio_transcribe/ tests/test_models.py && uv run mypy audio_transcribe/models.py`
Expected: clean

**Step 9: Commit**

```bash
git add audio_transcribe/ tests/test_models.py pyproject.toml
git commit -m "feat: add package scaffold, data models, and CLI entry point"
```

---

## Task 2: Stats Store + Hardware Detection

**Files:**
- Create: `audio_transcribe/stats/store.py`
- Create: `audio_transcribe/stats/hardware.py`
- Test: `tests/stats/test_store.py`
- Test: `tests/stats/test_hardware.py`

**Step 1: Write failing tests for stats store**

```python
# tests/stats/__init__.py  (empty)

# tests/stats/test_store.py
"""Tests for stats store — read/write/query ~/.audio-transcribe/history.json."""

import json

from audio_transcribe.models import (
    Config,
    HardwareInfo,
    InputInfo,
    RunRecord,
    StageStats,
)
from audio_transcribe.stats.store import StatsStore


def _make_record(id: str = "2026-01-01T00:00:00Z", duration_s: float = 60.0, model: str = "large-v3") -> RunRecord:
    return RunRecord(
        id=id,
        hardware=HardwareInfo(chip="Apple M4", cores_physical=10, memory_gb=24, os="macOS 15.3", python="3.12.8"),
        input=InputInfo(file="test.wav", duration_s=duration_s, file_size_mb=1.9),
        config=Config(language="ru", model=model, backend="whisperx"),
        stages={"transcribe": StageStats(time_s=10.0, peak_rss_mb=3000)},
        quality=None,
        corrections_applied=0,
        total_time_s=10.0,
        realtime_ratio=0.167,
    )


def test_store_creates_file_on_first_write(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    store.append(_make_record())
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 1


def test_store_appends_multiple_records(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    store.append(_make_record("r1"))
    store.append(_make_record("r2"))
    data = json.loads(path.read_text())
    assert len(data) == 2


def test_store_load_empty(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    assert store.load() == []


def test_store_load_returns_records(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    store.append(_make_record())
    records = store.load()
    assert len(records) == 1
    assert records[0].id == "2026-01-01T00:00:00Z"


def test_store_query_by_config(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    store.append(_make_record("r1", model="large-v3"))
    store.append(_make_record("r2", model="small"))
    store.append(_make_record("r3", model="large-v3"))
    matches = store.query(model="large-v3")
    assert len(matches) == 2


def test_store_query_by_hardware(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    store.append(_make_record("r1"))
    matches = store.query(chip="Apple M4")
    assert len(matches) == 1
    matches_none = store.query(chip="Apple M4 Pro")
    assert len(matches_none) == 0


def test_store_last_n(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    for i in range(10):
        store.append(_make_record(f"r{i}"))
    last3 = store.last(3)
    assert len(last3) == 3
    assert last3[0].id == "r7"


def test_store_clear(tmp_path):
    path = tmp_path / "history.json"
    store = StatsStore(path)
    store.append(_make_record())
    store.clear()
    assert store.load() == []
```

```python
# tests/stats/test_hardware.py
"""Tests for hardware detection."""

from audio_transcribe.stats.hardware import detect_hardware
from audio_transcribe.models import HardwareInfo


def test_detect_hardware_returns_hardware_info():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.cores_physical > 0
    assert hw.memory_gb > 0
    assert len(hw.chip) > 0
    assert len(hw.os) > 0
    assert len(hw.python) > 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/stats/ -v`
Expected: FAIL — imports fail

**Step 3: Implement hardware detection**

```python
# audio_transcribe/stats/hardware.py
"""Detect hardware characteristics for the current machine."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

from audio_transcribe.models import HardwareInfo


def detect_hardware() -> HardwareInfo:
    """Return hardware fingerprint for the current machine."""
    chip = _detect_chip()
    cores = os.cpu_count() or 1
    memory_gb = _detect_memory_gb()
    os_version = f"{platform.system()} {platform.release()}"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return HardwareInfo(
        chip=chip,
        cores_physical=cores,
        memory_gb=memory_gb,
        os=os_version,
        python=python_version,
    )


def _detect_chip() -> str:
    """Detect CPU/chip name. macOS: sysctl, Linux: /proc/cpuinfo, fallback: platform."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return platform.processor() or platform.machine()


def _detect_memory_gb() -> int:
    """Detect total RAM in GB. macOS: sysctl, Linux: /proc/meminfo, fallback: 0."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) // (1024 ** 3)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // (1024 ** 2)
        except (FileNotFoundError, ValueError):
            pass
    return 0
```

**Step 4: Implement stats store**

`audio_transcribe/stats/store.py` — implements `StatsStore` class with:
- `__init__(path)` — path to history.json, creates parent dir if needed
- `append(record)` — serialize RunRecord to JSON, append to file
- `load()` — deserialize all records
- `query(**filters)` — filter by chip, model, backend, language
- `last(n)` — return last N records
- `clear()` — empty the file

Use `dataclasses.asdict()` for serialization and reconstruct dataclasses on load.

**Step 5: Run tests**

Run: `uv run pytest tests/stats/ -v`
Expected: all PASS

**Step 6: Run linters**

Run: `uv run ruff check audio_transcribe/stats/ tests/stats/ && uv run mypy audio_transcribe/stats/`

**Step 7: Commit**

```bash
git add audio_transcribe/stats/ tests/stats/
git commit -m "feat: add stats store and hardware detection"
```

---

## Task 3: Quality Scorecard

**Files:**
- Create: `audio_transcribe/quality/scorecard.py`
- Test: `tests/quality/test_scorecard.py`

**Step 1: Write failing tests**

```python
# tests/quality/__init__.py  (empty)

# tests/quality/test_scorecard.py
"""Tests for quality scorecard computation."""

from audio_transcribe.quality.scorecard import compute_quality


def test_compute_quality_basic():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello world", "speaker": "SPEAKER_00",
         "words": [
             {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
             {"word": "world", "start": 0.6, "end": 1.0, "speaker": "SPEAKER_00"},
         ]},
    ]
    q = compute_quality(segments)
    assert q.segments == 1
    assert q.words_total == 2
    assert q.words_aligned == 2
    assert q.alignment_pct == 100.0
    assert q.speakers_detected == 1
    assert q.speaker_coverage_pct == 100.0
    assert q.speaker_transitions == 0


def test_compute_quality_missing_speaker():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "UNKNOWN"},
        {"start": 1.5, "end": 2.0, "text": "world", "speaker": "SPEAKER_00"},
    ]
    q = compute_quality(segments)
    assert q.speakers_detected == 1  # UNKNOWN not counted
    assert q.speaker_coverage_pct == 50.0


def test_compute_quality_unaligned_words():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello world",
         "words": [
             {"word": "hello", "start": 0.0, "end": 0.5},
             {"word": "world"},  # no start = unaligned
         ]},
    ]
    q = compute_quality(segments)
    assert q.words_total == 2
    assert q.words_aligned == 1
    assert q.alignment_pct == 50.0


def test_compute_quality_speaker_transitions():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 3.0, "text": "c", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 4.0, "text": "d", "speaker": "SPEAKER_00"},
    ]
    q = compute_quality(segments)
    assert q.speaker_transitions == 2  # 00->01, 01->00
    assert q.speakers_detected == 2


def test_compute_quality_empty():
    q = compute_quality([])
    assert q.segments == 0
    assert q.words_total == 0
    assert q.alignment_pct == 0.0
    assert q.speaker_coverage_pct == 0.0


def test_compute_quality_no_words_key():
    """Segments without words: count segment text words as total, 0 aligned."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello world", "speaker": "SPEAKER_00"},
    ]
    q = compute_quality(segments)
    assert q.words_total == 2
    assert q.words_aligned == 0
    assert q.alignment_pct == 0.0
```

**Step 2: Implement scorecard**

```python
# audio_transcribe/quality/scorecard.py
"""Compute quality metrics from pipeline output segments."""

from __future__ import annotations

from typing import Any

from audio_transcribe.models import QualityMetrics


def compute_quality(segments: list[dict[str, Any]]) -> QualityMetrics:
    """Compute quality scorecard from transcription segments."""
    if not segments:
        return QualityMetrics(
            segments=0, words_total=0, words_aligned=0, alignment_pct=0.0,
            speakers_detected=0, speaker_coverage_pct=0.0, speaker_transitions=0,
        )

    words_total = 0
    words_aligned = 0
    speakers: set[str] = set()
    segments_with_speaker = 0
    transitions = 0
    prev_speaker: str | None = None

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker != "UNKNOWN":
            speakers.add(speaker)
            segments_with_speaker += 1

        if prev_speaker is not None and speaker != "UNKNOWN" and prev_speaker != "UNKNOWN" and speaker != prev_speaker:
            transitions += 1
        if speaker != "UNKNOWN":
            prev_speaker = speaker

        if "words" in seg:
            for w in seg["words"]:
                words_total += 1
                if "start" in w:
                    words_aligned += 1
        else:
            # Count text words as total, 0 aligned
            words_total += len(seg.get("text", "").split())

    alignment_pct = (words_aligned / words_total * 100) if words_total > 0 else 0.0
    speaker_coverage_pct = (segments_with_speaker / len(segments) * 100) if segments else 0.0

    return QualityMetrics(
        segments=len(segments),
        words_total=words_total,
        words_aligned=words_aligned,
        alignment_pct=round(alignment_pct, 1),
        speakers_detected=len(speakers),
        speaker_coverage_pct=round(speaker_coverage_pct, 1),
        speaker_transitions=transitions,
    )
```

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/quality/ -v
uv run ruff check audio_transcribe/quality/ tests/quality/
uv run mypy audio_transcribe/quality/
git add audio_transcribe/quality/ tests/quality/
git commit -m "feat: add quality scorecard computation"
```

---

## Task 4: Corrections System

**Files:**
- Create: `audio_transcribe/stages/correct.py`
- Test: `tests/stages/test_correct.py`

**Step 1: Write failing tests**

```python
# tests/stages/__init__.py  (empty)

# tests/stages/test_correct.py
"""Tests for the corrections stage — load, apply, and learn corrections."""

import yaml

from audio_transcribe.stages.correct import apply_corrections, load_corrections, learn_corrections


def test_load_corrections_empty(tmp_path):
    path = tmp_path / "corrections.yaml"
    c = load_corrections(str(path))
    assert c["substitutions"] == {}
    assert c["patterns"] == []


def test_load_corrections_from_file(tmp_path):
    path = tmp_path / "corrections.yaml"
    path.write_text(yaml.dump({
        "substitutions": {"кубернетес": "Kubernetes"},
        "patterns": [],
    }))
    c = load_corrections(str(path))
    assert c["substitutions"]["кубернетес"] == "Kubernetes"


def test_apply_corrections_substitution():
    corrections = {"substitutions": {"кубернетес": "Kubernetes"}, "patterns": []}
    segments = [{"text": "Мы используем кубернетес для деплоя", "start": 0.0, "end": 3.0}]
    result, count = apply_corrections(segments, corrections)
    assert "Kubernetes" in result[0]["text"]
    assert count == 1


def test_apply_corrections_case_insensitive():
    corrections = {"substitutions": {"кубернетес": "Kubernetes"}, "patterns": []}
    segments = [{"text": "Кубернетес работает", "start": 0.0, "end": 2.0}]
    result, count = apply_corrections(segments, corrections)
    assert "Kubernetes" in result[0]["text"]
    assert count == 1


def test_apply_corrections_word_level():
    corrections = {"substitutions": {"хелло": "hello"}, "patterns": []}
    segments = [
        {"text": "хелло world", "start": 0.0, "end": 1.0,
         "words": [
             {"word": "хелло", "start": 0.0, "end": 0.5},
             {"word": "world", "start": 0.6, "end": 1.0},
         ]},
    ]
    result, count = apply_corrections(segments, corrections)
    assert result[0]["words"][0]["word"] == "hello"
    assert count == 1


def test_apply_corrections_pattern():
    corrections = {"substitutions": {}, "patterns": [{"match": "\\bпиар\\b", "replace": "PR"}]}
    segments = [{"text": "нужен пиар для проекта", "start": 0.0, "end": 2.0}]
    result, count = apply_corrections(segments, corrections)
    assert "PR" in result[0]["text"]
    assert count == 1


def test_apply_corrections_no_match():
    corrections = {"substitutions": {"nonexistent": "replacement"}, "patterns": []}
    segments = [{"text": "normal text", "start": 0.0, "end": 1.0}]
    result, count = apply_corrections(segments, corrections)
    assert result[0]["text"] == "normal text"
    assert count == 0


def test_learn_corrections_finds_diff():
    original = ["привет мир кубернетес"]
    corrected = ["привет мир Kubernetes"]
    learned = learn_corrections(original, corrected)
    assert "кубернетес" in learned
    assert learned["кубернетес"] == "Kubernetes"


def test_learn_corrections_multiple():
    original = ["кубернетес и дженкинс работают"]
    corrected = ["Kubernetes и Jenkins работают"]
    learned = learn_corrections(original, corrected)
    assert len(learned) == 2


def test_learn_corrections_no_diff():
    original = ["привет мир"]
    corrected = ["привет мир"]
    learned = learn_corrections(original, corrected)
    assert len(learned) == 0
```

**Step 2: Implement corrections module**

`audio_transcribe/stages/correct.py`:
- `load_corrections(path)` — read YAML, return dict with `substitutions` and `patterns`
- `apply_corrections(segments, corrections)` — apply substitutions (case-insensitive) and regex patterns to segment text and word text. Return `(modified_segments, count)`.
- `learn_corrections(original_lines, corrected_lines)` — word-level diff between original and corrected text lines using `difflib.SequenceMatcher`. Return dict of `{wrong: correct}` substitutions.

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/stages/test_correct.py -v
uv run ruff check audio_transcribe/stages/correct.py tests/stages/
uv run mypy audio_transcribe/stages/correct.py
git add audio_transcribe/stages/ tests/stages/
git commit -m "feat: add corrections system — load, apply, learn"
```

---

## Task 5: Extract Preprocess Stage

**Files:**
- Create: `audio_transcribe/stages/preprocess.py`
- Test: `tests/stages/test_preprocess.py`
- Reference: `preprocess.py` (current root-level script)

**Step 1: Write failing tests**

Copy and adapt tests from `tests/test_preprocess.py`. The function signature stays identical:

```python
def preprocess(input_path: str, output_path: str | None = None, remove_silence: bool = True,
               silence_threshold_db: str = "-35dB", silence_duration: float = 0.3) -> str
```

Tests import from `audio_transcribe.stages.preprocess` instead of `preprocess`.

**Step 2: Extract core logic**

Copy the `preprocess()` function from `preprocess.py` into `audio_transcribe/stages/preprocess.py`. Remove the `main()` and argparse code — only the pure function.

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/stages/test_preprocess.py -v
git add audio_transcribe/stages/preprocess.py tests/stages/test_preprocess.py
git commit -m "feat: extract preprocess stage from standalone script"
```

---

## Task 6: Extract Transcribe Stage

**Files:**
- Create: `audio_transcribe/stages/transcribe.py`
- Test: `tests/stages/test_transcribe.py`
- Reference: `transcribe_whisperx.py` (functions: `transcribe`, `transcribe_mlx`, `transcribe_mlx_vad`, `_offset_segments`, `_clear_mlx_cache`, `build_output`, `MLX_MODEL_MAP`)

**Step 1: Write failing tests**

Adapt tests from `tests/test_transcribe_whisperx.py`. The module contains all three backends plus `build_output`, `_offset_segments`, and `MLX_MODEL_MAP`.

Tests import from `audio_transcribe.stages.transcribe`.

**Step 2: Extract functions**

Copy all transcription-related functions into `audio_transcribe/stages/transcribe.py`:
- `MLX_MODEL_MAP`
- `transcribe(audio_path, model_size, language)` (whisperx backend)
- `transcribe_mlx(audio_path, model_size, language)`
- `transcribe_mlx_vad(audio_path, model_size, language)`
- `_offset_segments(segments, offset)`
- `_clear_mlx_cache()`
- `build_output(result, audio_file, language, model, elapsed)`

No argparse, no `main()`.

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/stages/test_transcribe.py -v
git commit -m "feat: extract transcribe stage with all backends"
```

---

## Task 7: Extract Align + Diarize Stages

**Files:**
- Create: `audio_transcribe/stages/align.py`
- Create: `audio_transcribe/stages/diarize.py`
- Test: `tests/stages/test_align.py`
- Test: `tests/stages/test_diarize.py`
- Reference: `transcribe_whisperx.py` (functions: `align`, `diarize`)

**Step 1: Extract align**

From `transcribe_whisperx.py`, extract:
```python
def align(result: dict[str, Any], audio: Any, language: str, align_model: str | None = None) -> dict[str, Any]
```

Into `audio_transcribe/stages/align.py`.

**Step 2: Extract diarize**

From `transcribe_whisperx.py`, extract:
```python
def diarize(result: dict[str, Any], audio: Any, hf_token: str, min_speakers: int, max_speakers: int) -> dict[str, Any]
```

Into `audio_transcribe/stages/diarize.py`.

**Step 3: Write tests**

These stages are thin wrappers around whisperx library calls with heavy dependencies. Tests will be minimal — verify function signatures exist and can be imported. Integration testing happens at the pipeline level.

**Step 4: Run tests, lint, commit**

```bash
uv run pytest tests/stages/ -v
git commit -m "feat: extract align and diarize stages"
```

---

## Task 8: Extract Format Stage

**Files:**
- Create: `audio_transcribe/stages/format.py`
- Test: `tests/stages/test_format.py`
- Reference: `format_transcript.py`

**Step 1: Write failing tests**

Adapt tests from `tests/test_format_transcript.py`. Import from `audio_transcribe.stages.format`.

Functions to extract:
- `format_time(seconds) -> str`
- `build_speaker_legend(segments) -> dict[str, str]`
- `format_segment(segment, legend) -> str`
- `compute_duration(segments) -> float`
- `format_transcript(data) -> str`

**Step 2: Copy pure functions**

Copy all functions from `format_transcript.py` into `audio_transcribe/stages/format.py`. No `main()`, no argparse.

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/stages/test_format.py -v
git commit -m "feat: extract format stage from standalone script"
```

---

## Task 9: Progress Events + Reporters

**Files:**
- Create: `audio_transcribe/progress/events.py`
- Create: `audio_transcribe/progress/tui.py`
- Create: `audio_transcribe/progress/json_reporter.py`
- Test: `tests/progress/test_events.py`
- Test: `tests/progress/test_json_reporter.py`

**Step 1: Write failing tests for events**

```python
# tests/progress/__init__.py  (empty)

# tests/progress/test_events.py
"""Tests for progress event types."""

from audio_transcribe.progress.events import StageStart, StageComplete, PipelineStart, PipelineComplete


def test_stage_start():
    e = StageStart(stage="transcribe", eta_s=85.0)
    assert e.stage == "transcribe"
    assert e.eta_s == 85.0


def test_stage_start_no_eta():
    e = StageStart(stage="transcribe", eta_s=None)
    assert e.eta_s is None


def test_stage_complete():
    e = StageComplete(stage="transcribe", time_s=42.3, peak_rss_mb=6200, extra={"segments": 142})
    assert e.time_s == 42.3
    assert e.extra["segments"] == 142


def test_pipeline_start():
    e = PipelineStart(file="meeting.wav", duration_s=3600.0, config={"model": "large-v3"})
    assert e.file == "meeting.wav"


def test_pipeline_complete():
    e = PipelineComplete(total_time_s=118.6, output="result.json", transcript="transcript.md")
    assert e.total_time_s == 118.6
```

```python
# tests/progress/test_json_reporter.py
"""Tests for JSON-lines progress reporter."""

import json

from audio_transcribe.progress.events import PipelineStart, StageStart, StageComplete, PipelineComplete
from audio_transcribe.progress.json_reporter import JsonReporter


def test_json_reporter_pipeline_start(capsys):
    reporter = JsonReporter()
    reporter.on_pipeline_start(PipelineStart(file="test.wav", duration_s=60.0, config={}))
    line = capsys.readouterr().out.strip()
    data = json.loads(line)
    assert data["event"] == "start"
    assert data["file"] == "test.wav"


def test_json_reporter_stage_events(capsys):
    reporter = JsonReporter()
    reporter.on_stage_start(StageStart(stage="transcribe", eta_s=10.0))
    reporter.on_stage_complete(StageComplete(stage="transcribe", time_s=8.5, peak_rss_mb=3000))
    lines = capsys.readouterr().out.strip().split("\n")
    assert len(lines) == 2
    start = json.loads(lines[0])
    assert start["event"] == "stage_start"
    complete = json.loads(lines[1])
    assert complete["event"] == "stage_complete"
    assert complete["time_s"] == 8.5


def test_json_reporter_pipeline_complete(capsys):
    reporter = JsonReporter()
    reporter.on_pipeline_complete(PipelineComplete(total_time_s=10.0, output="r.json", transcript="t.md"))
    line = capsys.readouterr().out.strip()
    data = json.loads(line)
    assert data["event"] == "complete"
```

**Step 2: Implement events dataclasses**

```python
# audio_transcribe/progress/events.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PipelineStart:
    file: str
    duration_s: float
    config: dict[str, Any]

@dataclass
class StageStart:
    stage: str
    eta_s: float | None

@dataclass
class StageComplete:
    stage: str
    time_s: float
    peak_rss_mb: float = 0
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineComplete:
    total_time_s: float
    output: str
    transcript: str | None = None
```

**Step 3: Implement JSON reporter**

```python
# audio_transcribe/progress/json_reporter.py
"""JSON-lines progress reporter for machine consumers."""

import json
import sys
from dataclasses import asdict
from audio_transcribe.progress.events import PipelineStart, StageStart, StageComplete, PipelineComplete


class JsonReporter:
    def on_pipeline_start(self, event: PipelineStart) -> None:
        self._emit({"event": "start", **asdict(event)})

    def on_stage_start(self, event: StageStart) -> None:
        self._emit({"event": "stage_start", **asdict(event)})

    def on_stage_complete(self, event: StageComplete) -> None:
        self._emit({"event": "stage_complete", **asdict(event)})

    def on_pipeline_complete(self, event: PipelineComplete) -> None:
        self._emit({"event": "complete", **asdict(event)})

    def _emit(self, data: dict) -> None:
        print(json.dumps(data, ensure_ascii=False), flush=True)
```

**Step 4: Implement TUI reporter**

`audio_transcribe/progress/tui.py` — uses `rich.live.Live`, `rich.table.Table`, `rich.progress.Progress`. Implements same interface as `JsonReporter` (duck typing or protocol). This is the most visual component — test manually, unit test just the data formatting helpers.

Key elements:
- `TuiReporter` class with same `on_*` methods as `JsonReporter`
- Uses `rich.live.Live` context manager for live updates
- Progress bar per stage
- Memory display from `resource.getrusage`
- Quality grade display on completion

**Step 5: Run tests, lint, commit**

```bash
uv run pytest tests/progress/ -v
git commit -m "feat: add progress events, JSON reporter, and TUI reporter"
```

---

## Task 10: ETA Estimator

**Files:**
- Create: `audio_transcribe/stats/estimator.py`
- Test: `tests/stats/test_estimator.py`

**Step 1: Write failing tests**

```python
# tests/stats/test_estimator.py
"""Tests for ETA estimation from historical data."""

from audio_transcribe.models import (
    Config, HardwareInfo, InputInfo, RunRecord, StageStats,
)
from audio_transcribe.stats.estimator import estimate_stage, EstimateResult


def _make_hw():
    return HardwareInfo(chip="Apple M4", cores_physical=10, memory_gb=24, os="macOS 15.3", python="3.12.8")


def _make_record(duration_s: float, transcribe_time: float) -> RunRecord:
    return RunRecord(
        id="test",
        hardware=_make_hw(),
        input=InputInfo(file="test.wav", duration_s=duration_s, file_size_mb=duration_s * 0.03),
        config=Config(language="ru", model="large-v3", backend="whisperx"),
        stages={"transcribe": StageStats(time_s=transcribe_time, peak_rss_mb=3000)},
        quality=None,
        corrections_applied=0,
        total_time_s=transcribe_time,
        realtime_ratio=transcribe_time / duration_s,
    )


def test_estimate_no_history():
    result = estimate_stage("transcribe", 60.0, [])
    assert result is None


def test_estimate_insufficient_history():
    records = [_make_record(60.0, 10.0)]
    result = estimate_stage("transcribe", 60.0, records)
    assert result is None  # need >= 3


def test_estimate_linear_relationship():
    # 60s audio -> 10s, 120s audio -> 20s, 180s audio -> 30s (linear: 1:6 ratio)
    records = [
        _make_record(60.0, 10.0),
        _make_record(120.0, 20.0),
        _make_record(180.0, 30.0),
    ]
    result = estimate_stage("transcribe", 240.0, records)
    assert result is not None
    assert abs(result.eta_s - 40.0) < 2.0  # ~40s expected
    assert result.confident  # R^2 should be ~1.0


def test_estimate_low_confidence():
    # Noisy data
    records = [
        _make_record(60.0, 10.0),
        _make_record(120.0, 50.0),  # outlier
        _make_record(180.0, 15.0),  # outlier
    ]
    result = estimate_stage("transcribe", 240.0, records)
    assert result is not None
    assert not result.confident  # high variance


def test_estimate_missing_stage():
    records = [
        _make_record(60.0, 10.0),
        _make_record(120.0, 20.0),
        _make_record(180.0, 30.0),
    ]
    result = estimate_stage("diarize", 60.0, records)  # stage not in records
    assert result is None
```

**Step 2: Implement estimator**

```python
# audio_transcribe/stats/estimator.py
"""ETA estimation using linear regression on historical run data."""

from __future__ import annotations
from dataclasses import dataclass
from audio_transcribe.models import RunRecord


@dataclass
class EstimateResult:
    eta_s: float
    confident: bool  # True if R^2 >= 0.7
    sample_size: int


def estimate_stage(stage: str, audio_duration_s: float, history: list[RunRecord]) -> EstimateResult | None:
    """Estimate stage duration from historical data using linear regression."""
    # Filter records that have this stage
    points = []
    for r in history:
        if stage in r.stages:
            points.append((r.input.duration_s, r.stages[stage].time_s))

    if len(points) < 3:
        return None

    # Simple linear regression: time = a * duration + b
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return None

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n

    eta = max(0, a * audio_duration_s + b)

    # R^2
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return EstimateResult(eta_s=round(eta, 1), confident=r_squared >= 0.7, sample_size=n)
```

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/stats/test_estimator.py -v
git commit -m "feat: add ETA estimator with linear regression"
```

---

## Task 11: Smart Recommender

**Files:**
- Create: `audio_transcribe/stats/recommender.py`
- Test: `tests/stats/test_recommender.py`

**Step 1: Write failing tests**

```python
# tests/stats/test_recommender.py
"""Tests for smart recommendation engine."""

from audio_transcribe.models import (
    Config, HardwareInfo, InputInfo, RunRecord, StageStats,
)
from audio_transcribe.stats.recommender import recommend


def _make_hw():
    return HardwareInfo(chip="Apple M4", cores_physical=10, memory_gb=24, os="macOS 15.3", python="3.12.8")


def _make_record(backend: str, duration_s: float, total_time_s: float) -> RunRecord:
    return RunRecord(
        id="test",
        hardware=_make_hw(),
        input=InputInfo(file="test.wav", duration_s=duration_s, file_size_mb=duration_s * 0.03),
        config=Config(language="ru", model="large-v3", backend=backend),
        stages={"transcribe": StageStats(time_s=total_time_s, peak_rss_mb=3000)},
        quality=None,
        corrections_applied=0,
        total_time_s=total_time_s,
        realtime_ratio=total_time_s / duration_s,
    )


def test_recommend_insufficient_history():
    recs = recommend(duration_s=60.0, history=[])
    assert recs.backend is None  # no recommendation possible


def test_recommend_best_backend():
    history = [
        _make_record("whisperx", 60.0, 30.0),
        _make_record("whisperx", 120.0, 60.0),
        _make_record("mlx-vad", 60.0, 15.0),
        _make_record("mlx-vad", 120.0, 30.0),
        _make_record("mlx-vad", 180.0, 45.0),
    ]
    recs = recommend(duration_s=60.0, history=history)
    assert recs.backend == "mlx-vad"
    assert recs.speedup_factor is not None
    assert recs.speedup_factor > 1.0
```

**Step 2: Implement recommender**

`audio_transcribe/stats/recommender.py`:
- `recommend(duration_s, history)` — analyze historical runs, compare backends by average realtime_ratio, return `Recommendation` dataclass with `backend`, `model`, `speedup_factor`, `tips`.
- Needs 5+ runs total to make recommendations.

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/stats/test_recommender.py -v
git commit -m "feat: add smart recommender engine"
```

---

## Task 12: Pipeline Orchestrator

**Files:**
- Create: `audio_transcribe/pipeline.py`
- Test: `tests/test_pipeline.py`

This is the core module that wires stages together and emits progress events.

**Step 1: Write failing tests**

```python
# tests/test_pipeline.py
"""Tests for pipeline orchestrator — stage sequencing and event emission."""

from unittest.mock import MagicMock, patch

from audio_transcribe.pipeline import Pipeline, PipelineConfig
from audio_transcribe.progress.events import PipelineStart, StageStart, StageComplete, PipelineComplete


def test_pipeline_config_defaults():
    cfg = PipelineConfig(audio_file="test.wav")
    assert cfg.language == "ru"
    assert cfg.model == "large-v3"
    assert cfg.backend == "whisperx"
    assert cfg.skip_align is False
    assert cfg.skip_diarize is False


def test_pipeline_emits_events():
    """Pipeline should emit start/complete events for each stage."""
    events = []
    pipeline = Pipeline(reporter=MagicMock())
    pipeline.reporter.on_pipeline_start = lambda e: events.append(("pipeline_start", e))
    pipeline.reporter.on_stage_start = lambda e: events.append(("stage_start", e))
    pipeline.reporter.on_stage_complete = lambda e: events.append(("stage_complete", e))
    pipeline.reporter.on_pipeline_complete = lambda e: events.append(("pipeline_complete", e))

    # Mock all stages
    with patch("audio_transcribe.pipeline.preprocess_stage") as mock_pre, \
         patch("audio_transcribe.pipeline.transcribe_stage") as mock_trans, \
         patch("audio_transcribe.pipeline.align_stage") as mock_align, \
         patch("audio_transcribe.pipeline.diarize_stage") as mock_diarize, \
         patch("audio_transcribe.pipeline.format_stage") as mock_format:

        mock_pre.return_value = "clean.wav"
        mock_trans.return_value = ({"segments": [], "text": ""}, None)
        mock_align.return_value = {"segments": []}
        mock_diarize.return_value = {"segments": []}
        mock_format.return_value = "# Transcript"

        cfg = PipelineConfig(audio_file="test.wav", output="out.json")
        pipeline.run(cfg)

    stage_starts = [e for name, e in events if name == "stage_start"]
    stage_completes = [e for name, e in events if name == "stage_complete"]
    assert len(stage_starts) >= 4  # preprocess, transcribe, align, diarize, (correct), format
    assert len(stage_completes) >= 4
```

**Step 2: Implement pipeline**

`audio_transcribe/pipeline.py`:

```python
@dataclass
class PipelineConfig:
    audio_file: str
    language: str = "ru"
    model: str = "large-v3"
    backend: str = "whisperx"
    min_speakers: int = 2
    max_speakers: int = 6
    align_model: str | None = None
    skip_align: bool = False
    skip_diarize: bool = False
    output: str | None = None
    transcript_output: str | None = None
    corrections_path: str | None = None


class Pipeline:
    def __init__(self, reporter, stats_store=None, estimator_history=None):
        self.reporter = reporter
        self.stats_store = stats_store
        self.estimator_history = estimator_history or []

    def run(self, config: PipelineConfig) -> PipelineResult:
        # 1. Emit pipeline start
        # 2. For each stage: emit start, run, emit complete, measure time+rss
        # 3. Compute quality scorecard
        # 4. Record stats
        # 5. Emit pipeline complete
        ...
```

The pipeline:
- Calls each stage function with timing via `time.time()` and `resource.getrusage()`
- Emits events to the reporter before/after each stage
- Uses the estimator to provide ETAs in stage_start events
- Computes quality scorecard after diarization
- Appends a RunRecord to the stats store
- Returns PipelineResult

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/test_pipeline.py -v
git commit -m "feat: add pipeline orchestrator with event emission"
```

---

## Task 13: CLI — process Command

**Files:**
- Modify: `audio_transcribe/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
"""Tests for CLI commands using typer test client."""

from typer.testing import CliRunner
from audio_transcribe.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "process" in result.output


def test_process_help():
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    assert "--language" in result.output
    assert "--model" in result.output
    assert "--backend" in result.output
    assert "--json" in result.output


def test_process_missing_file():
    result = runner.invoke(app, ["process", "nonexistent.wav"])
    assert result.exit_code != 0


def test_stats_help():
    result = runner.invoke(app, ["stats", "--help"])
    assert result.exit_code == 0
    assert "--last" in result.output


def test_recommend_help():
    result = runner.invoke(app, ["recommend", "--help"])
    assert result.exit_code == 0


def test_learn_help():
    result = runner.invoke(app, ["learn", "--help"])
    assert result.exit_code == 0
```

**Step 2: Implement CLI**

```python
# audio_transcribe/cli.py
"""CLI entry point for audio-transcribe."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="audio-transcribe", help="Local audio transcription pipeline.")


@app.command()
def process(
    audio_file: Path = typer.Argument(..., help="Input audio file (WAV, M4A, MP3)"),
    language: str = typer.Option("ru", "-l", "--language", help="Language code"),
    model: str = typer.Option("large-v3", "-m", "--model", help="Whisper model size"),
    backend: str = typer.Option("whisperx", "--backend", help="Transcription backend"),
    min_speakers: int = typer.Option(2, "--min-speakers", help="Minimum speakers for diarization"),
    max_speakers: int = typer.Option(6, "--max-speakers", help="Maximum speakers for diarization"),
    align_model: str | None = typer.Option(None, "--align-model", help="Custom alignment model"),
    no_align: bool = typer.Option(False, "--no-align", help="Skip alignment"),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip diarization"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output JSON path"),
    transcript: Path | None = typer.Option(None, "--transcript", help="Output Markdown path"),
    json_mode: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
) -> None:
    """Run the full transcription pipeline."""
    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    from audio_transcribe.pipeline import Pipeline, PipelineConfig
    from audio_transcribe.progress.json_reporter import JsonReporter
    from audio_transcribe.progress.tui import TuiReporter
    from audio_transcribe.stats.store import StatsStore

    store = StatsStore()
    reporter = JsonReporter() if json_mode or not sys.stdout.isatty() else TuiReporter()

    config = PipelineConfig(
        audio_file=str(audio_file),
        language=language,
        model=model,
        backend=backend,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        align_model=align_model,
        skip_align=no_align,
        skip_diarize=no_diarize,
        output=str(output) if output else None,
        transcript_output=str(transcript) if transcript else None,
    )

    pipeline = Pipeline(reporter=reporter, stats_store=store)
    pipeline.run(config)


@app.command()
def stats(
    last: int = typer.Option(10, "--last", "-n", help="Show last N runs"),
    clear: bool = typer.Option(False, "--clear", help="Clear all history"),
) -> None:
    """View historical run statistics."""
    ...


@app.command()
def recommend(
    audio_file: Path = typer.Argument(..., help="Audio file to analyze"),
) -> None:
    """Suggest optimal settings based on historical performance."""
    ...


@app.command()
def learn(
    corrected_md: Path = typer.Argument(..., help="Corrected Markdown transcript"),
    original: Path | None = typer.Option(None, "--original", help="Original JSON output"),
) -> None:
    """Learn corrections from an edited transcript."""
    ...


if __name__ == "__main__":
    app()
```

**Step 3: Run tests, lint, commit**

```bash
uv run pytest tests/test_cli.py -v
git commit -m "feat: add CLI with process, stats, recommend, learn commands"
```

---

## Task 14: CLI — stats, recommend, learn Commands

**Files:**
- Modify: `audio_transcribe/cli.py`
- Test: `tests/test_cli.py` (add more tests)

**Step 1: Implement stats command**

Loads history from store, formats as a rich table (or JSON with `--json`). Shows last N runs with: date, file, duration, total time, realtime ratio, quality grade.

**Step 2: Implement recommend command**

Gets audio file duration (via ffprobe or a quick load), loads history, calls `recommend()`, displays results as formatted text.

**Step 3: Implement learn command**

- Reads corrected markdown file
- Finds original JSON (from `--original` flag or by matching audio_file in YAML header)
- Strips timestamps/speaker labels from markdown to get plain text
- Calls `learn_corrections()` to diff
- Presents discovered corrections to user (via `typer.confirm`)
- Appends to `~/.audio-transcribe/corrections.yaml`

**Step 4: Run tests, lint, commit**

```bash
uv run pytest tests/test_cli.py -v
git commit -m "feat: implement stats, recommend, and learn CLI commands"
```

---

## Task 15: TUI Reporter Polish

**Files:**
- Modify: `audio_transcribe/progress/tui.py`
- Test: manual testing (TUI is visual)

**Step 1: Implement full TUI**

Use `rich.live.Live` with a custom layout:
- Header: filename + duration
- Stage list: checkmark/spinner/pending indicator + name + time + progress bar + ETA
- Footer: memory gauge + config summary
- Completion screen: 2-column summary + quality grade + output paths

**Step 2: Test manually**

Run with a short audio file:
```bash
uv run audio-transcribe process test_data/short.wav --backend whisperx
```

Verify:
- Progress bars update live
- ETAs shown when history exists
- Completion summary is clean
- JSON mode (`--json`) outputs clean JSONL

**Step 3: Commit**

```bash
git commit -m "feat: polish rich TUI progress display"
```

---

## Task 16: Migration — Remove Old Scripts + Update Tests

**Files:**
- Delete: `preprocess.py`, `transcribe_whisperx.py`, `format_transcript.py`
- Delete: `benchmark.py`, `compare_align.py`, `verify_diarize.py`
- Delete: `tests/test_preprocess.py`, `tests/test_transcribe_whisperx.py`, `tests/test_format_transcript.py`
- Delete: `tests/test_benchmark.py`, `tests/test_compare_align.py`, `tests/test_verify_diarize.py`
- Keep: `test_ollama.py`, `tests/test_ollama_utils.py`
- Modify: `CLAUDE.md` (update docs)

**Step 1: Verify all new tests pass**

```bash
uv run pytest tests/ -v
```

All tests under `tests/stages/`, `tests/stats/`, `tests/quality/`, `tests/progress/`, `tests/test_pipeline.py`, `tests/test_cli.py`, `tests/test_models.py` must pass.

**Step 2: Remove old scripts**

```bash
git rm preprocess.py transcribe_whisperx.py format_transcript.py
git rm benchmark.py compare_align.py verify_diarize.py
git rm tests/test_preprocess.py tests/test_transcribe_whisperx.py tests/test_format_transcript.py
git rm tests/test_benchmark.py tests/test_compare_align.py tests/test_verify_diarize.py
```

**Step 3: Verify no import breakage**

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy .
```

**Step 4: Update CLAUDE.md**

Update the running instructions, pipeline description, and script references to reflect the new `audio-transcribe` CLI:

```bash
# Old:
uv run preprocess.py input.m4a -o clean.wav
uv run transcribe_whisperx.py clean.wav -o result.json
uv run format_transcript.py result.json -o transcript.md

# New:
audio-transcribe process input.m4a -o result.json --transcript transcript.md
audio-transcribe stats --last 5
audio-transcribe recommend input.m4a
audio-transcribe learn corrected-transcript.md
```

**Step 5: Run full suite, commit**

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy .
git add -A
git commit -m "refactor: remove old scripts, migrate to unified CLI package"
```

---

## Task 17: Integration Test + Final Verification

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration smoke test**

```python
# tests/test_integration.py
"""Integration smoke tests for the full CLI."""

from typer.testing import CliRunner
from audio_transcribe.cli import app

runner = CliRunner()


def test_process_json_mode_with_mock(tmp_path, monkeypatch):
    """End-to-end test with mocked ML stages, verifying JSON output format."""
    # Create a dummy audio file
    audio = tmp_path / "test.wav"
    audio.touch()
    output = tmp_path / "result.json"

    # Mock the heavy ML imports to avoid loading torch/whisperx
    # ... (mock preprocess, transcribe, align, diarize stages)

    result = runner.invoke(app, [
        "process", str(audio),
        "--json", "-o", str(output),
    ])
    # Verify JSON lines were emitted
    # Verify output file was created
    ...


def test_stats_empty_history(tmp_path, monkeypatch):
    """Stats command with no history should not crash."""
    monkeypatch.setenv("HOME", str(tmp_path))
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0


def test_full_roundtrip_learn(tmp_path):
    """Process -> edit markdown -> learn corrections roundtrip."""
    # Create mock original JSON and corrected markdown
    # Run learn command
    # Verify corrections.yaml was created
    ...
```

**Step 2: Run everything**

```bash
uv run pytest tests/ -v --tb=short
uv run ruff check .
uv run mypy .
```

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke tests for CLI"
```

---

## Summary

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Package scaffold + data models | None |
| 2 | Stats store + hardware detection | Task 1 |
| 3 | Quality scorecard | Task 1 |
| 4 | Corrections system | Task 1 |
| 5 | Extract preprocess stage | Task 1 |
| 6 | Extract transcribe stage | Task 1 |
| 7 | Extract align + diarize stages | Task 1 |
| 8 | Extract format stage | Task 1 |
| 9 | Progress events + reporters | Task 1 |
| 10 | ETA estimator | Task 2 |
| 11 | Smart recommender | Task 2 |
| 12 | Pipeline orchestrator | Tasks 2-9 |
| 13 | CLI — process command | Task 12 |
| 14 | CLI — stats, recommend, learn | Tasks 10, 11, 4 |
| 15 | TUI reporter polish | Task 9 |
| 16 | Migration — remove old scripts | Tasks 5-8, 13 |
| 17 | Integration test | Tasks 13, 14 |

**Parallelizable:** Tasks 3-8 can run in parallel (they only depend on Task 1). Tasks 10-11 can run in parallel (both depend on Task 2).

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


def _make_record(id="2026-01-01T00:00:00Z", duration_s=60.0, model="large-v3"):
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

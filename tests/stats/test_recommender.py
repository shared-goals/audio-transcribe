"""Tests for smart recommendation engine."""

from audio_transcribe.models import (
    Config,
    HardwareInfo,
    InputInfo,
    RunRecord,
    StageStats,
)
from audio_transcribe.stats.recommender import Recommendation, recommend


def _make_hw() -> HardwareInfo:
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


def test_recommend_insufficient_total_runs():
    """Need 5+ runs total to make recommendations."""
    history = [
        _make_record("whisperx", 60.0, 30.0),
        _make_record("mlx-vad", 60.0, 15.0),
    ]
    recs = recommend(duration_s=60.0, history=history)
    assert recs.backend is None


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


def test_recommend_single_backend():
    """With only one backend, still recommend it if enough data."""
    history = [
        _make_record("whisperx", 60.0, 10.0),
        _make_record("whisperx", 120.0, 20.0),
        _make_record("whisperx", 180.0, 30.0),
        _make_record("whisperx", 240.0, 40.0),
        _make_record("whisperx", 300.0, 50.0),
    ]
    recs = recommend(duration_s=60.0, history=history)
    assert recs.backend == "whisperx"
    assert recs.speedup_factor is None  # no comparison available


def test_recommendation_fields():
    """Verify Recommendation has expected fields."""
    r = Recommendation(backend="mlx-vad", speedup_factor=2.0, tips=["Use mlx-vad for faster processing"])
    assert r.backend == "mlx-vad"
    assert r.speedup_factor == 2.0
    assert len(r.tips) == 1

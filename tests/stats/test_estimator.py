"""Tests for ETA estimation from historical data."""

from audio_transcribe.models import (
    Config,
    HardwareInfo,
    InputInfo,
    RunRecord,
    StageStats,
)
from audio_transcribe.stats.estimator import EstimateResult, estimate_stage


def _make_hw() -> HardwareInfo:
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


def test_estimate_result_fields():
    """Verify EstimateResult has expected fields."""
    r = EstimateResult(eta_s=10.0, confident=True, sample_size=5)
    assert r.eta_s == 10.0
    assert r.confident is True
    assert r.sample_size == 5


def _make_record_with_backend(duration_s, transcribe_time, backend):
    return RunRecord(
        id="test",
        hardware=_make_hw(),
        input=InputInfo(file="test.wav", duration_s=duration_s, file_size_mb=duration_s * 0.03),
        config=Config(language="ru", model="large-v3", backend=backend),
        stages={"transcribe": StageStats(time_s=transcribe_time, peak_rss_mb=3000)},
        quality=None,
        corrections_applied=0,
        total_time_s=transcribe_time,
        realtime_ratio=transcribe_time / duration_s,
    )


def test_estimate_filters_by_backend():
    """Only history from the same backend is used for estimation."""
    records = [
        _make_record_with_backend(60.0, 10.0, "mlx-vad"),
        _make_record_with_backend(120.0, 20.0, "mlx-vad"),
        _make_record_with_backend(180.0, 30.0, "mlx-vad"),
        _make_record_with_backend(60.0, 60.0, "whisperx"),  # much slower
        _make_record_with_backend(120.0, 120.0, "whisperx"),
        _make_record_with_backend(180.0, 180.0, "whisperx"),
    ]
    mlx_result = estimate_stage("transcribe", 240.0, records, backend="mlx-vad")
    wx_result = estimate_stage("transcribe", 240.0, records, backend="whisperx")
    assert mlx_result is not None
    assert wx_result is not None
    assert mlx_result.eta_s < wx_result.eta_s  # mlx-vad is faster


def test_estimate_no_backend_filter_uses_all():
    """Without backend filter, all history is used."""
    records = [
        _make_record_with_backend(60.0, 10.0, "mlx-vad"),
        _make_record_with_backend(120.0, 20.0, "whisperx"),
        _make_record_with_backend(180.0, 30.0, "mlx"),
    ]
    result = estimate_stage("transcribe", 240.0, records)
    assert result is not None

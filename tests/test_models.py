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
        segments=100,
        words_total=100,
        words_aligned=96,
        alignment_pct=96.0,
        speakers_detected=3,
        speaker_coverage_pct=95.0,
        speaker_transitions=50,
    )
    assert q_a.grade == "A"

    q_b = QualityMetrics(
        segments=100,
        words_total=100,
        words_aligned=90,
        alignment_pct=90.0,
        speakers_detected=3,
        speaker_coverage_pct=80.0,
        speaker_transitions=50,
    )
    assert q_b.grade == "B"

    q_c = QualityMetrics(
        segments=100,
        words_total=100,
        words_aligned=70,
        alignment_pct=70.0,
        speakers_detected=3,
        speaker_coverage_pct=60.0,
        speaker_transitions=50,
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

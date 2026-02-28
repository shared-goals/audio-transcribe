"""Tests for progress event types."""

from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageStart


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

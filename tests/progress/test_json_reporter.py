"""Tests for JSON-lines progress reporter."""

import json

from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageStart
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

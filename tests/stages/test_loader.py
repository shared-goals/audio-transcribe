"""Tests for shared audio data loader."""

import json
from pathlib import Path

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

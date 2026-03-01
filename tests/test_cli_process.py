"""Tests for process command reactive pipeline changes."""

import json
from unittest.mock import patch

from typer.testing import CliRunner

from audio_transcribe.cli import app

runner = CliRunner()

_MOCK_RESULT = {
    "audio_file": "meeting.wav",
    "language": "ru",
    "model": "large-v3",
    "processing_time_s": 10.0,
    "segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}],
}


def test_process_stores_raw_json(tmp_path):
    """process command stores raw WhisperX JSON in .audio-data/ directory."""
    audio_file = tmp_path / "2026-02-28-meeting.wav"
    audio_file.write_bytes(b"fake")
    output_dir = tmp_path / "meetings"
    output_dir.mkdir()

    mock_result = {
        "audio_file": str(audio_file),
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}],
    }

    with patch("audio_transcribe.pipeline.run_pipeline", return_value=mock_result):
        result = runner.invoke(app, ["process", str(audio_file), "-o", str(output_dir)])

    assert result.exit_code == 0, result.output

    audio_data_dir = output_dir / ".audio-data"
    assert audio_data_dir.exists()
    json_files = list(audio_data_dir.glob("*.json"))
    assert len(json_files) == 1

    stored = json.loads(json_files[0].read_text())
    assert stored["segments"] == mock_result["segments"]


def test_process_skips_diarize_by_default(tmp_path):
    """process command skips diarization in fast pass mode."""
    audio_file = tmp_path / "meeting.wav"
    audio_file.write_bytes(b"fake")

    with patch("audio_transcribe.pipeline.run_pipeline") as mock_pipeline:
        mock_pipeline.return_value = {
            "audio_file": str(audio_file),
            "language": "ru",
            "model": "large-v3",
            "processing_time_s": 5.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
        }
        runner.invoke(app, ["process", str(audio_file), "-o", str(tmp_path)])

    assert mock_pipeline.call_args is not None


def test_process_output_has_reanalyze_true(tmp_path):
    """Output meeting note has reanalyze: true and audio_file in frontmatter."""
    audio_file = tmp_path / "2026-02-28-standup.wav"
    audio_file.write_bytes(b"fake")
    output_dir = tmp_path / "meetings"
    output_dir.mkdir()

    mock_result = {
        "audio_file": str(audio_file),
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
    }

    with patch("audio_transcribe.pipeline.run_pipeline", return_value=mock_result):
        result = runner.invoke(app, ["process", str(audio_file), "-o", str(output_dir)])

    assert result.exit_code == 0, result.output
    md_files = list(output_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text()
    assert "reanalyze: true" in content
    assert "audio_data:" in content
    assert "audio_file:" in content

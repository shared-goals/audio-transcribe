"""Integration smoke tests — full CLI with mocked ML stages."""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from audio_transcribe.cli import app

runner = CliRunner()

# Shared mock return values
_MOCK_TRANSCRIBE_RV = ({"segments": [], "text": "", "language": "ru"}, None)
_MOCK_ALIGN_RV: dict[str, list[object]] = {"segments": []}
_MOCK_CORRECTIONS: dict[str, object] = {"substitutions": {}, "patterns": []}
_MOCK_OUTPUT = {
    "audio_file": "test.wav",
    "language": "ru",
    "model": "large-v3",
    "processing_time_s": 1.0,
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}],
}


def test_process_json_mode_with_mock(tmp_path, monkeypatch):
    """End-to-end process command with mocked ML stages verifies JSON output format."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Redirect history/corrections away from home directory
    monkeypatch.setattr("audio_transcribe.cli._DEFAULT_HISTORY", tmp_path / "history.json")
    monkeypatch.setattr("audio_transcribe.cli._DEFAULT_CORRECTIONS", tmp_path / "corrections.yaml")

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value=str(audio)),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=_MOCK_TRANSCRIBE_RV),
        patch("audio_transcribe.pipeline.align_stage", return_value=_MOCK_ALIGN_RV),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_MOCK_CORRECTIONS),
        patch("audio_transcribe.pipeline.build_output_stage", return_value=_MOCK_OUTPUT),
    ):
        result = runner.invoke(app, ["process", str(audio), "--json", "-o", str(output_dir), "--no-diarize"])

    assert result.exit_code == 0, result.output

    # JSON stored in .audio-data/
    json_path = output_dir / ".audio-data" / "test.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert "segments" in data
    assert data["language"] == "ru"

    # --json flag emits JSONL events to stdout
    lines = [line for line in result.output.strip().splitlines() if line]
    events = [json.loads(line) for line in lines]
    event_types = {e.get("event") for e in events}
    assert "start" in event_types
    assert "complete" in event_types


def test_process_with_transcript_output(tmp_path, monkeypatch):
    """Process command should write a Markdown transcript when --transcript is given."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    transcript = tmp_path / "transcript.md"

    monkeypatch.setattr("audio_transcribe.cli._DEFAULT_HISTORY", tmp_path / "history.json")
    monkeypatch.setattr("audio_transcribe.cli._DEFAULT_CORRECTIONS", tmp_path / "corrections.yaml")

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value=str(audio)),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=_MOCK_TRANSCRIBE_RV),
        patch("audio_transcribe.pipeline.align_stage", return_value=_MOCK_ALIGN_RV),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_MOCK_CORRECTIONS),
        patch("audio_transcribe.pipeline.build_output_stage", return_value=_MOCK_OUTPUT),
    ):
        result = runner.invoke(
            app,
            ["process", str(audio), "--json", "-o", str(output_dir), "--transcript", str(transcript), "--no-diarize"],
        )

    assert result.exit_code == 0, result.output
    assert transcript.exists()
    assert "Transcript" in transcript.read_text()


def test_full_roundtrip_learn(tmp_path, monkeypatch):
    """Create mock original JSON + corrected markdown, run learn, verify corrections.yaml."""
    monkeypatch.setattr("audio_transcribe.cli._DEFAULT_CORRECTIONS", tmp_path / "corrections.yaml")

    # Original JSON with a typo ("мор" should be "мир")
    original = tmp_path / "result.json"
    original.write_text(
        json.dumps({"segments": [{"text": "Привет мор", "start": 0.0, "end": 1.0}]}),
        encoding="utf-8",
    )

    # Corrected markdown in expected format: [MM:SS] Speaker: text
    corrected_md = tmp_path / "result.md"
    corrected_md.write_text("[00:00] Speaker A: Привет мир\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["learn", str(corrected_md), "--original", str(original)],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "мор" in result.output
    assert "мир" in result.output

    corrections_file = tmp_path / "corrections.yaml"
    assert corrections_file.exists()

    import yaml

    data = yaml.safe_load(corrections_file.read_text(encoding="utf-8"))
    assert data["substitutions"]["мор"] == "мир"

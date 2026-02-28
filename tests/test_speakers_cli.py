"""Tests for speakers CLI subcommands."""

import numpy as np
from typer.testing import CliRunner

from audio_transcribe.cli import app
from audio_transcribe.speakers.database import SpeakerDB

runner = CliRunner()


def test_speakers_list_empty(tmp_path):
    result = runner.invoke(app, ["speakers", "list", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No speakers" in result.output


def test_speakers_list_shows_enrolled(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))
    db.enroll("Maria", np.random.randn(256).astype(np.float32))

    result = runner.invoke(app, ["speakers", "list", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Andrey" in result.output
    assert "Maria" in result.output


def test_speakers_forget(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))

    result = runner.invoke(app, ["speakers", "forget", "Andrey", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0

    db2 = SpeakerDB(tmp_path)
    assert not db2.has_speaker("Andrey")


def test_speakers_forget_unknown(tmp_path):
    result = runner.invoke(app, ["speakers", "forget", "Nobody", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "not found" in result.output.lower()

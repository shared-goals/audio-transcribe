"""Tests for CLI commands using typer test client."""

from typer.testing import CliRunner

from audio_transcribe.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "process" in result.output


def test_process_help():
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    assert "--language" in result.output
    assert "--model" in result.output
    assert "--backend" in result.output
    assert "--json" in result.output


def test_process_missing_file():
    result = runner.invoke(app, ["process", "nonexistent.wav"])
    assert result.exit_code != 0


def test_stats_help():
    result = runner.invoke(app, ["stats", "--help"])
    assert result.exit_code == 0
    assert "--last" in result.output


def test_recommend_help():
    result = runner.invoke(app, ["recommend", "--help"])
    assert result.exit_code == 0


def test_learn_help():
    result = runner.invoke(app, ["learn", "--help"])
    assert result.exit_code == 0


def test_stats_empty_history(tmp_path, monkeypatch):
    """Stats command with no history should not crash."""
    monkeypatch.setattr("audio_transcribe.cli._DEFAULT_HISTORY", tmp_path / "history.json")
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "No history" in result.output


def test_stats_clear(tmp_path, monkeypatch):
    """Stats --clear should clear history."""
    import json

    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps([{"id": "test"}]))
    monkeypatch.setattr("audio_transcribe.cli._DEFAULT_HISTORY", history_file)
    result = runner.invoke(app, ["stats", "--clear"])
    assert result.exit_code == 0
    assert "cleared" in result.output


def test_learn_missing_file():
    """Learn with nonexistent file should exit with error."""
    result = runner.invoke(app, ["learn", "nonexistent.md"])
    assert result.exit_code != 0

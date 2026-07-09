"""Tests for CLI commands using typer test client."""

import os

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


def test_sync_hf_token_env_to_cache(tmp_path, monkeypatch):
    """When HF_TOKEN is set but cache file missing, write it."""
    import audio_transcribe.cli as cli_mod

    cache_file = tmp_path / "huggingface" / "token"
    monkeypatch.setattr(cli_mod, "_HF_TOKEN_CACHE", cache_file)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")

    cli_mod._sync_hf_token()

    assert cache_file.exists()
    assert cache_file.read_text(encoding="utf-8") == "hf_test_token_123"


def test_sync_hf_token_cache_to_env(tmp_path, monkeypatch):
    """When HF_TOKEN is unset but cache file exists, load it."""
    import audio_transcribe.cli as cli_mod

    cache_file = tmp_path / "huggingface" / "token"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("hf_cached_token_456\n", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_HF_TOKEN_CACHE", cache_file)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    cli_mod._sync_hf_token()

    assert os.environ["HF_TOKEN"] == "hf_cached_token_456"


def test_sync_hf_token_no_overwrite(tmp_path, monkeypatch):
    """When both env and cache exist, do nothing (don't overwrite cache)."""
    import audio_transcribe.cli as cli_mod

    cache_file = tmp_path / "huggingface" / "token"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("hf_old_token", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_HF_TOKEN_CACHE", cache_file)
    monkeypatch.setenv("HF_TOKEN", "hf_new_token")

    cli_mod._sync_hf_token()

    assert cache_file.read_text(encoding="utf-8") == "hf_old_token"


def test_sync_hf_token_neither_set(tmp_path, monkeypatch):
    """When neither env nor cache exist, do nothing."""
    import audio_transcribe.cli as cli_mod

    cache_file = tmp_path / "nonexistent" / "token"
    monkeypatch.setattr(cli_mod, "_HF_TOKEN_CACHE", cache_file)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    cli_mod._sync_hf_token()

    assert not cache_file.exists()
    assert "HF_TOKEN" not in os.environ

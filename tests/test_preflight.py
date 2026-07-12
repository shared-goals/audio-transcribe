"""Tests for pre-flight validation checks."""

from pathlib import Path
from unittest.mock import patch

from audio_transcribe.preflight import check


def _module_available(_name: str) -> object:
    return object()


def test_check_missing_audio_file() -> None:
    with (
        patch("audio_transcribe.preflight.find_spec", side_effect=_module_available),
        patch("audio_transcribe.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
    ):
        result = check("/nonexistent/audio.wav")
    assert not result.ok
    assert any("not found" in e for e in result.errors)


def test_check_valid_file(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    with (
        patch("audio_transcribe.preflight.find_spec", side_effect=_module_available),
        patch("audio_transcribe.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
    ):
        result = check(str(audio))
    assert result.ok


def test_check_empty_file(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    with patch("audio_transcribe.preflight.find_spec", side_effect=_module_available):
        result = check(str(audio))
    assert not result.ok
    assert any("empty" in e for e in result.errors)


def test_check_warns_missing_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    with (
        patch("audio_transcribe.preflight.find_spec", side_effect=_module_available),
        patch("audio_transcribe.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
    ):
        result = check(str(audio), skip_diarize=False)
    assert result.ok
    assert any("HF_TOKEN" in w for w in result.warnings)


def test_check_reports_missing_ml_extra(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")
    with patch("audio_transcribe.preflight.find_spec", return_value=None):
        result = check(str(audio), backend="mlx-vad")
    assert not result.ok
    assert any("audio-transcribe[ml]" in error for error in result.errors)

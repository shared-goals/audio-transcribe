"""Tests for preprocess.py — FFmpeg audio preprocessing."""

from unittest.mock import MagicMock, patch

import pytest

from preprocess import preprocess


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        preprocess("nonexistent_audio_file.wav")


def test_default_output_path(tmp_path):
    input_file = tmp_path / "audio.m4a"
    input_file.touch()
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
    ):
        result = preprocess(str(input_file))
    assert result == str(tmp_path / "audio.16k.wav")


def test_custom_output_path(tmp_path):
    input_file = tmp_path / "audio.wav"
    input_file.touch()
    output_file = str(tmp_path / "out.wav")
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
    ):
        result = preprocess(str(input_file), output_file)
    assert result == output_file


def test_ffmpeg_failure(tmp_path):
    input_file = tmp_path / "audio.wav"
    input_file.touch()
    with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="ffmpeg error")):
        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            preprocess(str(input_file), str(tmp_path / "out.wav"))


def test_silence_removal_excluded(tmp_path):
    input_file = tmp_path / "audio.wav"
    input_file.touch()
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        preprocess(str(input_file), str(tmp_path / "out.wav"), remove_silence=False)
    cmd = mock_run.call_args[0][0]
    af_value = cmd[cmd.index("-af") + 1]
    assert "silenceremove" not in af_value


def test_silence_removal_included(tmp_path):
    input_file = tmp_path / "audio.wav"
    input_file.touch()
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        preprocess(str(input_file), str(tmp_path / "out.wav"), remove_silence=True)
    cmd = mock_run.call_args[0][0]
    af_value = cmd[cmd.index("-af") + 1]
    assert "silenceremove" in af_value


def test_silence_threshold_applied(tmp_path):
    input_file = tmp_path / "audio.wav"
    input_file.touch()
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        preprocess(str(input_file), str(tmp_path / "out.wav"), silence_threshold_db="-50dB")
    cmd = mock_run.call_args[0][0]
    af_value = cmd[cmd.index("-af") + 1]
    assert "-50dB" in af_value


def test_output_always_includes_resample(tmp_path):
    input_file = tmp_path / "audio.wav"
    input_file.touch()
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        preprocess(str(input_file), str(tmp_path / "out.wav"), remove_silence=False)
    cmd = mock_run.call_args[0][0]
    af_value = cmd[cmd.index("-af") + 1]
    assert "aresample=16000" in af_value
    assert "channel_layouts=mono" in af_value


def test_ffmpeg_overwrite_flag(tmp_path):
    """Verify -y (overwrite) flag is always passed."""
    input_file = tmp_path / "audio.wav"
    input_file.touch()
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)),
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        preprocess(str(input_file), str(tmp_path / "out.wav"))
    cmd = mock_run.call_args[0][0]
    assert "-y" in cmd
    assert cmd[0] == "ffmpeg"

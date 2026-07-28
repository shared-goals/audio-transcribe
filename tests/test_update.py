"""Tests for explicit self-update logic."""

import subprocess
from pathlib import Path
from unittest.mock import patch

from audio_transcribe.update import _latest_release, force_upgrade


def test_latest_release_selects_highest_semver() -> None:
    output = "\n".join(
        [
            "abc\trefs/tags/v0.3.0",
            "def\trefs/tags/v0.10.0",
            "ghi\trefs/tags/not-a-release",
        ]
    )
    with patch("audio_transcribe.update.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout=output)
        assert _latest_release() == "0.10.0"


def test_latest_release_failure() -> None:
    with patch("audio_transcribe.update.subprocess.run", side_effect=subprocess.TimeoutExpired([], 20)):
        assert _latest_release() is None


def test_force_upgrade_success() -> None:
    with (
        patch("audio_transcribe.update._latest_release", return_value="0.6.0"),
        patch("audio_transcribe.update.subprocess.run") as mock_run,
        patch("audio_transcribe.update.sys.platform", "darwin"),
        patch("audio_transcribe.update.shutil.which", return_value="/opt/homebrew/bin/brew"),
        patch("audio_transcribe.update._prepare_macos_ffmpeg") as prepare,
        patch("audio_transcribe.update._repair_installed_tool") as repair,
    ):
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        prepare.return_value = (
            Path("/opt/homebrew/opt/ffmpeg@7"),
            {"UV_NO_BINARY_PACKAGE": "av"},
        )
        assert force_upgrade() is True
        mock_run.assert_called_once_with(
            [
                "uv",
                "tool",
                "install",
                "--python",
                "3.12",
                "--force",
                "audio-transcribe[ml] @ git+https://github.com/shared-goals/audio-transcribe.git@v0.6.0",
            ],
            timeout=300.0,
            check=True,
            env={"UV_NO_BINARY_PACKAGE": "av"},
        )
        prepare.assert_called_once_with()
        repair.assert_called_once_with(Path("/opt/homebrew/opt/ffmpeg@7"))


def test_force_upgrade_failure():
    with (
        patch("audio_transcribe.update._latest_release", return_value="0.6.0"),
        patch("audio_transcribe.update.subprocess.run", side_effect=subprocess.TimeoutExpired([], 300)),
    ):
        assert force_upgrade() is False


def test_force_upgrade_without_release() -> None:
    with patch("audio_transcribe.update._latest_release", return_value=None):
        assert force_upgrade() is False


def test_force_upgrade_does_not_downgrade() -> None:
    with (
        patch("audio_transcribe.update._latest_release", return_value="0.3.0"),
        patch("audio_transcribe.update.subprocess.run") as mock_run,
    ):
        assert force_upgrade() is True
        mock_run.assert_not_called()

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from audio_transcribe.doctor import checks_json, run_checks


def test_doctor_reports_runtime_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    with (
        patch("audio_transcribe.doctor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("audio_transcribe.doctor.importlib.util.find_spec", return_value=object()),
        patch(
            "audio_transcribe.doctor.shutil.disk_usage",
            return_value=SimpleNamespace(free=10 * 1_073_741_824),
        ),
    ):
        checks = run_checks(tmp_path, "mlx-vad")
    assert all(check.ok for check in checks)
    assert checks_json(checks)[0]["name"] == "ffmpeg"


def test_doctor_reports_missing_dependencies_and_low_disk(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    with (
        patch("audio_transcribe.doctor.shutil.which", return_value=None),
        patch("audio_transcribe.doctor.importlib.util.find_spec", return_value=None),
        patch("audio_transcribe.doctor.shutil.disk_usage", return_value=SimpleNamespace(free=1)),
    ):
        checks = run_checks(tmp_path / "state", "whisperx")
    by_name = {check.name: check for check in checks}
    assert by_name["ffmpeg"].ok is False
    assert by_name["torch"].ok is False
    assert by_name["HF token"].ok is False
    assert by_name["free disk"].ok is False


def test_doctor_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        run_checks(tmp_path, "typo")

"""Tests for auto-update logic."""

import subprocess
import time
from unittest.mock import patch

from audio_transcribe.update import _needs_update, _touch_timestamp, check_for_update, force_upgrade


def test_needs_update_missing_file(tmp_path):
    assert _needs_update(tmp_path / ".last-update")


def test_needs_update_fresh_timestamp(tmp_path):
    ts_file = tmp_path / ".last-update"
    ts_file.write_text(str(time.time()))
    assert not _needs_update(ts_file)


def test_needs_update_stale_timestamp(tmp_path):
    ts_file = tmp_path / ".last-update"
    ts_file.write_text(str(time.time() - 90000))  # >24h ago
    assert _needs_update(ts_file)


def test_needs_update_corrupted_file(tmp_path):
    ts_file = tmp_path / ".last-update"
    ts_file.write_text("not-a-number")
    assert _needs_update(ts_file)


def test_touch_timestamp_creates_file(tmp_path):
    ts_file = tmp_path / "sub" / ".last-update"
    _touch_timestamp(ts_file)
    assert ts_file.exists()
    ts = float(ts_file.read_text().strip())
    assert time.time() - ts < 5


def test_check_for_update_skips_when_fresh(tmp_path):
    ts_file = tmp_path / ".last-update"
    ts_file.write_text(str(time.time()))
    with patch("audio_transcribe.update._run_upgrade") as mock_upgrade:
        check_for_update(last_update_path=ts_file)
        mock_upgrade.assert_not_called()


def test_check_for_update_triggers_when_stale(tmp_path):
    ts_file = tmp_path / ".last-update"
    ts_file.write_text(str(time.time() - 90000))
    with patch("audio_transcribe.update._run_upgrade", return_value=True) as mock_upgrade:
        check_for_update(last_update_path=ts_file)
        mock_upgrade.assert_called_once()
    # Timestamp should be refreshed
    ts = float(ts_file.read_text().strip())
    assert time.time() - ts < 5


def test_check_for_update_silent_on_failure(tmp_path):
    ts_file = tmp_path / ".last-update"
    with patch("audio_transcribe.update._run_upgrade", return_value=False):
        check_for_update(last_update_path=ts_file)
    # Timestamp should NOT be written on failure
    assert not ts_file.exists()


def test_force_upgrade_success():
    with patch("audio_transcribe.update.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        with patch("audio_transcribe.update._touch_timestamp"):
            assert force_upgrade() is True


def test_force_upgrade_failure():
    with patch("audio_transcribe.update.subprocess.run", side_effect=subprocess.TimeoutExpired([], 60)):
        assert force_upgrade() is False

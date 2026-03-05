"""Auto-update logic for audio-transcribe."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

_STATE_DIR = Path.home() / ".audio-transcribe"
_LAST_UPDATE = _STATE_DIR / ".last-update"
_UPDATE_INTERVAL_S = 86400  # 24 hours


def _needs_update(last_update_path: Path = _LAST_UPDATE, interval_s: float = _UPDATE_INTERVAL_S) -> bool:
    """Return True if enough time has passed since the last update check."""
    if not last_update_path.exists():
        return True
    try:
        ts = float(last_update_path.read_text().strip())
    except (ValueError, OSError):
        return True
    return (time.time() - ts) > interval_s


def _run_upgrade(timeout: float = 30.0) -> bool:
    """Run uv tool upgrade. Return True on success."""
    try:
        subprocess.run(
            ["uv", "tool", "upgrade", "audio-transcribe"],
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _touch_timestamp(last_update_path: Path = _LAST_UPDATE) -> None:
    """Write current timestamp to the update marker file."""
    last_update_path.parent.mkdir(parents=True, exist_ok=True)
    last_update_path.write_text(str(time.time()))


def check_for_update(
    last_update_path: Path = _LAST_UPDATE,
    interval_s: float = _UPDATE_INTERVAL_S,
) -> None:
    """Check once per day and silently upgrade if due."""
    if not _needs_update(last_update_path, interval_s):
        return
    if _run_upgrade():
        _touch_timestamp(last_update_path)


def force_upgrade() -> bool:
    """Force immediate upgrade, showing output. Return True on success."""
    try:
        subprocess.run(
            ["uv", "tool", "upgrade", "audio-transcribe"],
            timeout=60.0,
            check=True,
        )
        _touch_timestamp()
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False

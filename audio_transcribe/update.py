"""Auto-update logic for audio-transcribe."""

from __future__ import annotations

import re
import subprocess

_REPO_URL = "https://github.com/shared-goals/audio-transcribe.git"


def _latest_release(timeout: float = 20.0) -> str | None:
    """Return the highest stable vX.Y.Z tag advertised by the release repository."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", _REPO_URL],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None

    versions: list[tuple[tuple[int, int, int], str]] = []
    for tag in re.findall(r"refs/tags/v(\d+\.\d+\.\d+)$", result.stdout, flags=re.MULTILINE):
        major, minor, patch = (int(part) for part in tag.split("."))
        versions.append(((major, minor, patch), tag))
    return max(versions)[1] if versions else None


def force_upgrade() -> bool:
    """Install the latest stable tagged release. Return True on success."""
    from audio_transcribe import __version__

    version = _latest_release()
    if version is None:
        return False
    version_key = tuple(int(part) for part in version.split("."))
    current_key = tuple(int(part) for part in __version__.split("."))
    if version_key <= current_key:
        return True
    spec = f"audio-transcribe[ml] @ git+{_REPO_URL}@v{version}"
    try:
        subprocess.run(
            ["uv", "tool", "install", "--python", "3.12", "--force", spec],
            timeout=300.0,
            check=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False

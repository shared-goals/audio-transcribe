"""Auto-update logic for audio-transcribe."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_URL = "https://github.com/shared-goals/audio-transcribe.git"


def _prepare_macos_ffmpeg() -> tuple[Path, dict[str, str]]:
    """Ensure build tools exist and return the FFmpeg 7 source-build environment."""
    from audio_transcribe.macos_ffmpeg import ffmpeg7_build_environment

    if shutil.which("brew") is None:
        raise FileNotFoundError("Homebrew is required for the macOS FFmpeg 7 runtime")
    subprocess.run(["brew", "install", "ffmpeg@7", "pkgconf"], timeout=600.0, check=True)
    prefix_result = subprocess.run(
        ["brew", "--prefix", "ffmpeg@7"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=True,
    )
    prefix = Path(prefix_result.stdout.strip())
    return prefix, ffmpeg7_build_environment(prefix)


def _repair_installed_tool(prefix: Path) -> None:
    """Patch TorchCodec in the uv tool environment after installation."""
    from audio_transcribe.macos_ffmpeg import patch_torchcodec_rpath

    tool_dir_result = subprocess.run(
        ["uv", "tool", "dir"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=True,
    )
    tool_root = Path(tool_dir_result.stdout.strip()) / "audio-transcribe"
    candidates = list(tool_root.glob("lib/python*/site-packages/torchcodec"))
    if len(candidates) != 1:
        raise RuntimeError(f"could not locate installed TorchCodec package under {tool_root}")
    if patch_torchcodec_rpath(candidates[0], prefix) == 0:
        raise RuntimeError(f"no FFmpeg 7 TorchCodec libraries found under {candidates[0]}")


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
        prefix: Path | None = None
        install_options: dict[str, Any] = {}
        if sys.platform == "darwin":
            prefix, env = _prepare_macos_ffmpeg()
            install_options["env"] = env
        subprocess.run(
            ["uv", "tool", "install", "--python", "3.12", "--force", spec],
            timeout=300.0,
            check=True,
            **install_options,
        )
        if prefix is not None:
            _repair_installed_tool(prefix)
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError, RuntimeError):
        return False

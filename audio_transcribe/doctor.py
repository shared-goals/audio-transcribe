"""Runtime diagnostics for local and unattended installations."""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from audio_transcribe.config import Backend
from audio_transcribe.worker import JobQueue


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(state_dir: Path, backend: str = Backend.MLX_VAD) -> list[Check]:
    """Inspect dependencies, tokens, storage, state, and queue integrity."""
    checks: list[Check] = []
    checks.append(Check("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found"))
    checks.append(Check("ffprobe", shutil.which("ffprobe") is not None, shutil.which("ffprobe") or "not found"))
    required_modules = {
        Backend.MLX_VAD: ("mlx_whisper", "whisperx"),
        Backend.MLX: ("mlx_whisper", "whisperx"),
        Backend.WHISPERX: ("whisperx", "torch"),
    }[Backend(backend)]
    for module in required_modules:
        found = importlib.util.find_spec(module) is not None
        checks.append(Check(module, found, "installed" if found else "missing; install the matching extra"))
    token = bool(os.environ.get("HF_TOKEN") or (Path.home() / ".cache" / "huggingface" / "token").is_file())
    checks.append(Check("HF token", token, "available" if token else "missing; required only for diarization"))
    state_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(state_dir).free / 1_073_741_824
    checks.append(Check("free disk", free_gb >= 5, f"{free_gb:.1f} GiB available"))
    queue_health = JobQueue(state_dir / "jobs.sqlite3").health()
    checks.append(Check("queue", bool(queue_health["ok"]), str(queue_health)))
    return checks


def checks_json(checks: list[Check]) -> list[dict[str, object]]:
    return [asdict(check) for check in checks]

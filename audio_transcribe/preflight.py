"""Pre-flight checks before pipeline execution."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreflightResult:
    """Result of pre-flight validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True if no errors were found."""
        return len(self.errors) == 0


def check(
    audio_file: str,
    backend: str = "whisperx",
    skip_diarize: bool = False,
) -> PreflightResult:
    """Validate prerequisites before running the pipeline."""
    result = PreflightResult()

    if not shutil.which("ffmpeg"):
        result.errors.append("ffmpeg not found in PATH — install with: brew install ffmpeg")

    p = Path(audio_file)
    if not p.exists():
        result.errors.append(f"Audio file not found: {audio_file}")
    elif p.stat().st_size == 0:
        result.errors.append(f"Audio file is empty: {audio_file}")

    if not skip_diarize and not os.environ.get("HF_TOKEN"):
        result.warnings.append("HF_TOKEN not set — diarization will be skipped")

    return result

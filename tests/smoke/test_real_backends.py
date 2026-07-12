"""Opt-in Apple Silicon smoke test for the real ML backend stack."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from audio_transcribe.pipeline import run_pipeline
from audio_transcribe.progress.json_reporter import JsonReporter


@pytest.mark.ml_smoke
def test_real_backend_with_audio() -> None:
    """Transcribe a user-supplied fixture without mocking model calls."""
    raw_path = os.environ.get("AUDIO_TRANSCRIBE_SMOKE_AUDIO")
    if not raw_path:
        pytest.skip("set AUDIO_TRANSCRIBE_SMOKE_AUDIO to a trusted short recording")
    audio = Path(raw_path)
    if not audio.is_file():
        pytest.fail(f"smoke audio does not exist: {audio}")

    backend = os.environ.get("AUDIO_TRANSCRIBE_SMOKE_BACKEND", "mlx-vad")
    model = os.environ.get("AUDIO_TRANSCRIBE_SMOKE_MODEL", "tiny")
    result = run_pipeline(
        str(audio),
        backend=backend,
        model=model,
        no_align=True,
        no_diarize=True,
        reporter=JsonReporter(),
    )
    assert result["segments"]
    assert result["language"]

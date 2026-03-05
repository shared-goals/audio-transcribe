"""Shared loader for .audio-data JSON files used by post-processing stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio_transcribe.markdown.parser import MeetingDoc
from audio_transcribe.pipeline import PipelineError


def load_audio_data(meeting_path: Path, doc: MeetingDoc) -> dict[str, Any]:
    """Load .audio-data JSON for a meeting note."""
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    if not audio_data_rel:
        raise PipelineError("Meeting note has no audio_data path in frontmatter")
    json_path = meeting_path.parent / audio_data_rel
    if not json_path.exists():
        raise PipelineError(f"Audio data not found: {json_path}")
    try:
        data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError as e:
        raise PipelineError(f"Corrupted audio data: {json_path} — {e}") from e

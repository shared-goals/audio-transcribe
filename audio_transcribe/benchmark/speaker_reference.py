"""Load and strictly validate private two-speaker diarization references."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

from audio_transcribe.benchmark.models import SpeakerReference, SpeakerTurn


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"speaker reference {field} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"speaker reference {field} must be a finite number")
    return number


def _checksum(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"speaker reference {field} checksum is required")
    return value


def load_speaker_reference(path: Path) -> SpeakerReference:
    """Load sorted, duration-bounded turns for exactly two non-empty speakers."""
    payload: object = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("speaker reference must be a JSON object")

    source_sha256 = _checksum(payload, "source_sha256")
    aligned_asr_sha256 = _checksum(payload, "aligned_asr_sha256")
    raw_turns = payload.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("speaker reference turns must be a list")

    if "audio_duration_s" not in payload:
        raise ValueError("speaker reference audio duration is required")
    audio_duration_s = _number(payload["audio_duration_s"], "audio duration")
    if audio_duration_s <= 0.0:
        raise ValueError("speaker reference audio duration must be positive")

    turns: list[SpeakerTurn] = []
    speaker_ends: dict[str, float] = {}
    previous_key: tuple[float, float] | None = None
    for index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"speaker reference turn {index} must be an object")
        start_s = _number(raw_turn.get("start"), f"turn {index} start")
        end_s = _number(raw_turn.get("end"), f"turn {index} end")
        speaker = raw_turn.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"speaker reference turn {index} speaker must be non-empty")
        speaker = speaker.strip()
        if start_s < 0.0 or end_s <= start_s:
            raise ValueError(f"speaker reference turn {index} must have positive duration")
        if end_s > audio_duration_s:
            raise ValueError(f"speaker reference turn {index} exceeds audio duration")
        key = (start_s, end_s)
        if previous_key is not None and key < previous_key:
            raise ValueError("speaker reference turns must be sorted")
        if start_s < speaker_ends.get(speaker, 0.0):
            raise ValueError(f"speaker reference same-speaker turns overlap for {speaker}")
        speaker_ends[speaker] = end_s
        previous_key = key
        turns.append(SpeakerTurn(start_s, end_s, speaker))

    if len({turn.speaker for turn in turns}) != 2:
        raise ValueError("speaker reference must contain exactly two speakers")
    return SpeakerReference(source_sha256, aligned_asr_sha256, audio_duration_s, tuple(turns))

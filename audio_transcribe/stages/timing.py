"""Timestamp shifting.

`preprocess` may trim leading silence, and every later stage then reports times
relative to the trimmed file. The note points at the original recording, so the
removed lead has to be added back to every segment and word.
"""

from __future__ import annotations

from typing import Any


def shift_timings(result: dict[str, Any], offset_s: float) -> dict[str, Any]:
    """Return a copy with all segment and word times moved by `offset_s`."""
    if offset_s <= 0:
        return result
    shifted = dict(result)
    segments: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        item = dict(segment)
        item["start"] = round(float(item.get("start", 0.0)) + offset_s, 3)
        item["end"] = round(float(item.get("end", 0.0)) + offset_s, 3)
        if "words" in item:
            item["words"] = [
                {
                    **word,
                    "start": round(float(word.get("start", 0.0)) + offset_s, 3),
                    "end": round(float(word.get("end", 0.0)) + offset_s, 3),
                }
                for word in item["words"]
            ]
        segments.append(item)
    shifted["segments"] = segments
    return shifted


def parse_stamp(value: str) -> float:
    """`MM:SS` or `H:MM:SS` to seconds."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(value)
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def format_stamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

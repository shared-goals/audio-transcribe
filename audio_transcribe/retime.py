"""Restore absolute timestamps in an already produced meeting note.

Older runs transcribed a silence-trimmed copy, so the note's transcript and its
`.audio-data` JSON are early by the removed lead. This rewrites timestamps only:
text and speaker labels stay byte-identical, and the offset is recorded so a
second run cannot shift the note twice.

Stamps are recomputed from the shifted segments, not from the previous display
stamps: adding a fractional offset to an already floored `MM:SS` moves lines to
the wrong second (45.04 s must print as 00:45 even if the old line said 00:33).
"""

from __future__ import annotations

import re
from typing import Any

from audio_transcribe.stages.timing import format_stamp, parse_stamp, shift_timings

_LINE_RE = re.compile(r"^\[([\d:]+)\]\s(.*)$")


class RetimeError(Exception):
    """The note and the artifact cannot be paired safely."""


def _transcript_lines(lines: list[str]) -> list[int]:
    indexes: list[int] = []
    in_transcript = False
    for index, line in enumerate(lines):
        if line.startswith("## "):
            in_transcript = line.strip() == "## Transcript"
            continue
        if in_transcript and _LINE_RE.match(line):
            indexes.append(index)
    return indexes


def retime_document(text: str, data: dict[str, Any], offset_s: float) -> tuple[str, dict[str, Any], int]:
    """Shift transcript lines and segments by `offset_s`; return (text, data, lines)."""
    shifted = shift_timings(data, offset_s) if offset_s > 0 else data
    if offset_s > 0:
        shifted["timing_offset_s"] = round(data_offset(data) + offset_s, 3)

    lines = text.split("\n")
    line_indexes = _transcript_lines(lines)
    segments = list(shifted.get("segments", []))
    if len(line_indexes) != len(segments):
        raise RetimeError(
            f"строк транскрипта {len(line_indexes)}, сегментов {len(segments)} — заметка правилась вручную, "
            "автоматическая перестановка отменена"
        )
    for line_index, segment in zip(line_indexes, segments, strict=True):
        match = _LINE_RE.match(lines[line_index])
        assert match is not None
        stamp = format_stamp(float(segment["start"]))
        lines[line_index] = f"[{stamp}] {match.group(2)}".rstrip()
    return "\n".join(lines), shifted, len(segments)


def with_offset_marker(text: str, offset_s: float) -> str:
    """Add or bump `timing_offset_s` in the frontmatter."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return text
    block = match.group(1)
    existing = re.search(r"^timing_offset_s:\s*([0-9.]+)\s*$", block, re.MULTILINE)
    total = offset_s + (float(existing.group(1)) if existing else 0.0)
    line = f"timing_offset_s: {round(total, 3)}"
    if existing:
        block = block[: existing.start()] + line + block[existing.end() :]
    else:
        block = f"{block}\n{line}"
    return f"---\n{block}\n---\n" + text[match.end() :]


def current_offset(text: str) -> float:
    match = re.search(r"^timing_offset_s:\s*([0-9.]+)\s*$", text, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def data_offset(data: dict[str, Any]) -> float:
    value = data.get("timing_offset_s")
    return float(value) if isinstance(value, (int, float)) else 0.0


def parse_any_stamp(value: str) -> float:
    return parse_stamp(value)

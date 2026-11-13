"""Restore absolute timestamps in an already produced meeting note.

Older runs transcribed a silence-trimmed copy, so the note's transcript and its
`.audio-data` JSON are early by the removed lead. This rewrites timestamps only:
text and speaker labels stay byte-identical, and the offset is recorded so a
second run cannot shift the note twice.
"""

from __future__ import annotations

import re
from typing import Any

from audio_transcribe.stages.timing import format_stamp, parse_stamp, shift_timings

_LINE_RE = re.compile(r"^\[([\d:]+)\]\s(.*)$")


def retime_document(text: str, data: dict[str, Any], offset_s: float) -> tuple[str, dict[str, Any], int]:
    """Shift transcript lines and segments by `offset_s`; return (text, data, lines)."""
    if offset_s <= 0:
        return text, data, 0
    lines = text.split("\n")
    changed = 0
    in_transcript = False
    for index, line in enumerate(lines):
        if line.startswith("## "):
            in_transcript = line.strip() == "## Transcript"
            continue
        if not in_transcript:
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        stamp = format_stamp(parse_stamp(match.group(1)) + offset_s)
        lines[index] = f"[{stamp}] {match.group(2)}"
        changed += 1
    return "\n".join(lines), shift_timings(data, offset_s), changed


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

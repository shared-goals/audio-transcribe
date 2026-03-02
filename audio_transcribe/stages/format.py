"""Format stage: convert WhisperX JSON output to readable Markdown transcript."""

from __future__ import annotations

import re
from datetime import date as date_type
from typing import Any

import yaml


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_speaker_legend(segments: list[dict[str, Any]]) -> dict[str, str]:
    """Map speaker IDs to friendly labels in order of first appearance.

    Returns e.g. {"SPEAKER_00": "Speaker A", "SPEAKER_01": "Speaker B"}.
    """
    seen: dict[str, str] = {}
    label_idx = 0
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker not in seen and speaker != "UNKNOWN":
            seen[speaker] = f"Speaker {chr(65 + label_idx)}"
            label_idx += 1
    return seen


def format_segment(segment: dict[str, Any], legend: dict[str, str] | None = None) -> str:
    """Format a single segment as '[MM:SS] Speaker: text'."""
    start = segment.get("start", 0.0)
    speaker_id = str(segment.get("speaker", "UNKNOWN"))
    text = str(segment.get("text", "")).strip()

    if legend and speaker_id in legend:
        speaker = legend[speaker_id]
    elif speaker_id == "UNKNOWN":
        speaker = "Unknown"
    else:
        speaker = speaker_id

    return f"[{format_time(float(start))}] {speaker}: {text}"


def compute_duration(segments: list[dict[str, Any]]) -> float:
    """Compute total audio duration from segment timestamps."""
    if not segments:
        return 0.0
    return max(float(seg.get("end", 0.0)) for seg in segments)


def format_transcript(data: dict[str, Any]) -> str:
    """Format full WhisperX JSON as Markdown transcript.

    Includes YAML metadata header, speaker legend, and timestamped transcript.
    """
    segments: list[dict[str, Any]] = data.get("segments", [])
    audio_file = str(data.get("audio_file", "unknown"))
    language = str(data.get("language", "unknown"))
    model = str(data.get("model", "unknown"))
    processing_time = float(data.get("processing_time_s", 0.0))

    legend = build_speaker_legend(segments)
    duration = compute_duration(segments)
    speaker_count = len(legend)

    lines: list[str] = []

    # YAML metadata header
    lines.append("---")
    lines.append(f"audio_file: {audio_file}")
    lines.append(f"duration: {format_time(duration)}")
    lines.append(f"language: {language}")
    lines.append(f"model: {model}")
    lines.append(f"speakers: {speaker_count}")
    lines.append(f"processing_time_s: {processing_time}")
    lines.append("---")
    lines.append("")

    # Speaker legend
    if legend:
        lines.append("## Speakers")
        lines.append("")
        for speaker_id, label in legend.items():
            lines.append(f"- {speaker_id}: {label}")
        lines.append("")

    # Transcript
    lines.append("## Transcript")
    lines.append("")

    for seg in segments:
        lines.append(format_segment(seg, legend))

    lines.append("")
    return "\n".join(lines)


def format_meeting_note(data: dict[str, Any], audio_data_path: str) -> str:
    """Format WhisperX JSON as a reactive pipeline meeting note.

    Produces markdown with reanalyze frontmatter flag.
    Includes speaker section only if segments have speaker labels.
    """
    segments: list[dict[str, Any]] = data.get("segments", [])
    audio_file = str(data.get("audio_file", "unknown"))
    language = str(data.get("language", "unknown"))
    model = str(data.get("model", "unknown"))
    processing_time = float(data.get("processing_time_s", 0.0))

    has_speakers = any(seg.get("speaker") for seg in segments)
    legend = build_speaker_legend(segments) if has_speakers else {}
    duration = compute_duration(segments)

    # Extract date from audio filename, fall back to today
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", audio_file)
    date_str = date_match.group(1) if date_match else str(date_type.today())

    # Build frontmatter
    fm: dict[str, object] = {
        "title": f"{date_str} meeting",
        "date": date_str,
        "duration": format_time(duration),
        "language": language,
        "model": model,
        "processing_time_s": processing_time,
        "reanalyze": True,
        "audio_file": audio_file,
        "audio_data": audio_data_path,
    }

    if legend:
        fm["speakers"] = {sid: label for sid, label in legend.items()}

    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False).rstrip())
    lines.append("---")
    lines.append("")

    # Speaker legend (only if diarized)
    if legend:
        lines.append("## Speakers")
        lines.append("")
        for speaker_id, label in legend.items():
            lines.append(f"- {speaker_id}: {label}")
        lines.append("")

    # Transcript — no speaker prefixes; diarize step adds them after speaker ID
    lines.append("## Transcript")
    lines.append("")
    for seg in segments:
        lines.append(format_segment(seg, None))
    lines.append("")

    return "\n".join(lines)

#!/usr/bin/env python3
"""Convert WhisperX JSON output to readable Markdown transcript.

Usage:
    uv run format_transcript.py result.json -o transcript.md
    uv run format_transcript.py result.json  # stdout
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
            lines.append(f"- **{label}**: {speaker_id}")
        lines.append("")

    # Transcript
    lines.append("## Transcript")
    lines.append("")

    for seg in segments:
        lines.append(format_segment(seg, legend))

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WhisperX JSON to Markdown transcript")
    parser.add_argument("json_file", help="WhisperX JSON output file")
    parser.add_argument("-o", "--output", help="Output Markdown file (default: stdout)")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = format_transcript(data)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"Saved: {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()

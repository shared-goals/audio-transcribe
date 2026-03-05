"""Diarize an existing meeting note and update it in-place."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.markdown.updater import extract_wiki_links, replace_section, set_frontmatter
from audio_transcribe.speakers import embeddings as _embeddings
from audio_transcribe.stages.format import build_speaker_legend, format_time

if TYPE_CHECKING:
    from audio_transcribe.speakers.database import SpeakerDB


def run_diarization(
    audio_file: str,
    segments: list[dict[str, Any]],
    min_speakers: int = 1,
    max_speakers: int = 6,
) -> list[dict[str, Any]]:
    """Run pyannote diarization on audio and return segments with speaker labels."""
    import os

    import whisperx
    from whisperx.diarize import DiarizationPipeline

    hf_token = os.environ.get("HF_TOKEN", "")
    audio = whisperx.load_audio(audio_file)
    result: dict[str, Any] = {"segments": segments}
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1", token=hf_token, device="cpu"
    )
    diarize_segs = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segs, result)
    return list(result.get("segments", []))


def _match_timestamp(line: str) -> str | None:
    """Extract MM:SS or HH:MM:SS timestamp from a transcript line."""
    match = re.match(r"\[(\d[\d:]+)]", line)
    return match.group(1) if match else None


def diarize_and_update(
    meeting_path: Path,
    min_speakers: int = 1,
    max_speakers: int = 6,
    force: bool = False,
    audio_file_override: str | None = None,
    db: SpeakerDB | None = None,
) -> None:
    """Run diarization and update the meeting note in-place.

    Preserves existing transcript text — only adds speaker label prefixes.
    Raises RuntimeError if already diarized unless force=True.
    If db is provided and frontmatter has [[wiki-link]] speakers, enroll them.
    """
    md_text = meeting_path.read_text(encoding="utf-8")
    doc = parse_meeting(md_text)

    # Check if already diarized
    if "Speakers" in doc.sections and not force:
        raise RuntimeError("already diarized — pass force=True to re-diarize")

    # Capture pre-existing wiki-link speaker mappings before overwriting
    pre_speakers = doc.frontmatter.get("speakers", {})
    pre_wiki_links: dict[str, str] = {}
    if isinstance(pre_speakers, dict) and db is not None:
        pre_wiki_links = extract_wiki_links({str(k): str(v) for k, v in pre_speakers.items()})

    # Load stored JSON
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    json_path = meeting_path.parent / audio_data_rel
    stored = json.loads(json_path.read_text(encoding="utf-8"))

    audio_file: str = audio_file_override if audio_file_override is not None else str(
        doc.frontmatter.get("audio_file", stored.get("audio_file", ""))
    )
    segments: list[dict[str, Any]] = stored.get("segments", [])

    # Run diarization
    diarized_segments = run_diarization(audio_file, segments, min_speakers, max_speakers)

    # Update stored JSON
    stored["segments"] = diarized_segments
    json_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build speaker legend
    legend = build_speaker_legend(diarized_segments)

    # Build speakers section content
    speaker_lines = [f"- **{label}**: {sid}" for sid, label in legend.items()]
    speakers_content = "\n".join(speaker_lines)

    # Build timestamp → speaker label mapping from diarized segments.
    # Multiple segments can share the same formatted timestamp (sub-second
    # differences get truncated), so we collect *all* entries per timestamp
    # and pop them in order when matching transcript lines.
    ts_to_speakers: dict[str, list[tuple[str, str]]] = {}
    for seg in diarized_segments:
        ts = format_time(float(seg.get("start", 0.0)))
        speaker_id = str(seg.get("speaker", ""))
        if speaker_id and speaker_id in legend:
            ts_to_speakers.setdefault(ts, []).append((legend[speaker_id], str(seg.get("text", ""))))

    # Preserve existing transcript text; only add speaker label prefixes
    existing_transcript = doc.sections.get("Transcript", "")
    new_lines: list[str] = []
    for line in existing_transcript.split("\n"):
        line_ts = _match_timestamp(line)
        out_line = line
        if line_ts and line_ts in ts_to_speakers:
            entries = ts_to_speakers[line_ts]
            if entries:
                speaker = entries.pop(0)[0]
                after_bracket = line.split("] ", 1)
                if len(after_bracket) == 2:
                    # Strip all stacked speaker prefixes (e.g. "Speaker A: Unknown: text" → "text")
                    _pfx = re.compile(r"^(?:Speaker [A-Z]{1,2}|SPEAKER_\d+|Unknown|None):\s+")
                    text_part = after_bracket[1]
                    if _pfx.match(text_part):
                        text_part = _pfx.sub("", text_part, count=1)
                    out_line = f"[{line_ts}] {speaker}: {text_part}"
        new_lines.append(out_line)
    transcript_content = "\n".join(new_lines)

    # Update document
    doc = replace_section(doc, "Speakers", speakers_content, before="Transcript")
    doc = replace_section(doc, "Transcript", transcript_content)
    doc = set_frontmatter(doc, "speakers", {sid: label for sid, label in legend.items()})
    doc = set_frontmatter(doc, "reanalyze", True)

    meeting_path.write_text(doc.to_markdown(), encoding="utf-8")

    # Auto-enroll pre-existing wiki-link speakers if db provided
    if db is not None and pre_wiki_links:
        for speaker_id, person_name in pre_wiki_links.items():
            if not db.has_speaker(person_name):
                embedding = _embeddings.extract_speaker_embedding(audio_file, diarized_segments, speaker_id)
                if np.any(embedding):
                    db.enroll(person_name, embedding)

"""Auto-identify speakers by matching voice embeddings against known speakers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.markdown.updater import apply_speaker_mapping, set_frontmatter
from audio_transcribe.speakers import embeddings as _embeddings
from audio_transcribe.speakers.database import SpeakerDB


@dataclass
class IdentifyResult:
    """Result of speaker identification."""

    matched: dict[str, str] = field(default_factory=dict)   # speaker_id -> person_name
    unmatched: list[str] = field(default_factory=list)       # speaker_ids with no match


def identify_speakers(
    meeting_path: Path,
    db: SpeakerDB,
    threshold: float = 0.5,
    update_file: bool = True,
    audio_file_override: str | None = None,
) -> IdentifyResult:
    """Identify speakers in a meeting note using the voice embedding DB."""
    md_text = meeting_path.read_text(encoding="utf-8")
    doc = parse_meeting(md_text)

    # Load stored JSON
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    json_path = meeting_path.parent / audio_data_rel
    stored: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    audio_file = audio_file_override if audio_file_override is not None else str(
        doc.frontmatter.get("audio_file", stored.get("audio_file", ""))
    )
    segments: list[dict[str, Any]] = stored.get("segments", [])

    # Get current speaker mapping
    speakers = doc.frontmatter.get("speakers", {})
    if not isinstance(speakers, dict):
        speakers = {}

    result = IdentifyResult()

    # For each unidentified speaker (no wiki-link), try to match
    for speaker_id, current_label in speakers.items():
        if "[[" in str(current_label):
            continue  # Already identified

        embedding = _embeddings.extract_speaker_embedding(audio_file, segments, str(speaker_id))
        matches = db.match(embedding, threshold=threshold)

        if matches:
            person_name = matches[0][0]
            result.matched[str(speaker_id)] = person_name
        else:
            result.unmatched.append(str(speaker_id))

    # Update the meeting note if matches found
    if update_file and result.matched:
        label_mapping: dict[str, str] = {}
        new_speakers = dict(speakers)
        for speaker_id, person_name in result.matched.items():
            old_label = str(speakers[speaker_id])
            new_label = f"[[{person_name}]]"
            label_mapping[old_label] = new_label
            new_speakers[speaker_id] = new_label

        doc = apply_speaker_mapping(doc, label_mapping)
        doc = set_frontmatter(doc, "speakers", new_speakers)
        doc = set_frontmatter(doc, "reanalyze", True)
        meeting_path.write_text(doc.to_markdown(), encoding="utf-8")

    return result

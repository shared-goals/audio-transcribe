"""Apply speaker mapping and enroll new speakers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting, parse_speaker_legend
from audio_transcribe.markdown.updater import apply_speaker_mapping, extract_wiki_links, set_frontmatter
from audio_transcribe.speakers import embeddings as _embeddings
from audio_transcribe.speakers.database import SpeakerDB


def update_meeting(meeting_path: Path, db: SpeakerDB) -> None:
    """Apply speaker mapping from frontmatter and enroll new speakers."""
    md_text = meeting_path.read_text(encoding="utf-8")
    doc = parse_meeting(md_text)

    speakers = doc.frontmatter.get("speakers", {})
    if not isinstance(speakers, dict):
        return

    # Load stored JSON for enrollment
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    json_path = meeting_path.parent / audio_data_rel
    stored: dict[str, object] = json.loads(json_path.read_text(encoding="utf-8"))
    audio_file = str(doc.frontmatter.get("audio_file", stored.get("audio_file", "")))
    raw_segments = stored.get("segments", [])
    segments: list[dict[str, object]] = raw_segments if isinstance(raw_segments, list) else []

    # Build mapping from current legend labels to frontmatter values
    legend = parse_speaker_legend(doc)  # {SPEAKER_ID: current_label}
    label_mapping: dict[str, str] = {}

    for speaker_id, new_label in speakers.items():
        old_label = legend.get(str(speaker_id))
        if old_label and old_label != str(new_label):
            label_mapping[old_label] = str(new_label)

    # Apply mapping
    if label_mapping:
        doc = apply_speaker_mapping(doc, label_mapping)

    doc = set_frontmatter(doc, "reanalyze", True)

    # Enroll new wiki-link speakers in voice DB
    wiki_links = extract_wiki_links({str(k): str(v) for k, v in speakers.items()})
    for speaker_id, person_name in wiki_links.items():
        if not db.has_speaker(person_name):
            embedding = _embeddings.extract_speaker_embedding(audio_file, segments, speaker_id)
            if np.any(embedding):
                db.enroll(person_name, embedding)

    meeting_path.write_text(doc.to_markdown(), encoding="utf-8")

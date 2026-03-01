"""Integration test for the full reactive pipeline workflow."""

import json
from unittest.mock import patch

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.diarize_update import diarize_and_update
from audio_transcribe.stages.format import format_meeting_note
from audio_transcribe.stages.update import update_meeting


def test_full_reactive_workflow(tmp_path):
    """Simulate the complete reactive pipeline end-to-end."""

    # === Step 1: Fast pass (process command output) ===
    whisperx_result = {
        "audio_file": "2026-02-28-standup.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 15.0,
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "Привет, давайте начнём"},
            {"start": 3.5, "end": 6.0, "text": "Да, у меня есть обновления"},
            {"start": 6.5, "end": 10.0, "text": "Отлично, расскажи подробнее"},
        ],
    }

    # Store JSON
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    data_dir = meetings_dir / ".audio-data"
    data_dir.mkdir()
    json_path = data_dir / "2026-02-28-standup.json"
    json_path.write_text(json.dumps(whisperx_result, ensure_ascii=False, indent=2))

    # Format meeting note (fast pass — no speakers)
    markdown = format_meeting_note(whisperx_result, audio_data_path=".audio-data/2026-02-28-standup.json")
    md_path = meetings_dir / "2026-02-28-standup.md"
    md_path.write_text(markdown)

    # Verify fast pass output
    doc = parse_meeting(md_path.read_text())
    assert doc.frontmatter["reanalyze"] is True
    assert "Speakers" not in doc.sections
    assert "Привет" in doc.sections["Transcript"]

    # === Step 2: Diarize ===
    diarized_segments = [
        {"start": 0.0, "end": 3.0, "text": "Привет, давайте начнём", "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 6.0, "text": "Да, у меня есть обновления", "speaker": "SPEAKER_01"},
        {"start": 6.5, "end": 10.0, "text": "Отлично, расскажи подробнее", "speaker": "SPEAKER_00"},
    ]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized_segments):
        diarize_and_update(md_path)

    doc = parse_meeting(md_path.read_text())
    assert "Speakers" in doc.sections
    assert doc.frontmatter["reanalyze"] is True
    assert "Speaker A" in doc.sections["Transcript"]
    assert "Speaker B" in doc.sections["Transcript"]

    # === Step 3: User maps speakers in frontmatter ===
    content = md_path.read_text()
    content = content.replace("SPEAKER_00: Speaker A", 'SPEAKER_00: "[[Andrey]]"')
    content = content.replace("SPEAKER_01: Speaker B", 'SPEAKER_01: "[[Maria]]"')
    md_path.write_text(content)

    # === Step 4: Update (apply mapping + enroll) ===
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)

    mock_embedding = np.random.randn(256).astype(np.float32)
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=mock_embedding):
        update_meeting(md_path, db)

    doc = parse_meeting(md_path.read_text())
    assert "[[Andrey]]" in doc.sections["Transcript"]
    assert "[[Maria]]" in doc.sections["Transcript"]
    assert doc.frontmatter["reanalyze"] is True

    # Verify enrollment
    assert db.has_speaker("Andrey")
    assert db.has_speaker("Maria")

    # === Step 5: Verify speakers DB works for future meetings ===
    speakers = db.list_speakers()
    names = {s["name"] for s in speakers}
    assert names == {"Andrey", "Maria"}

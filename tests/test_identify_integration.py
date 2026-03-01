"""Integration test for speaker identification across meetings."""

import json
from unittest.mock import patch

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.format import format_meeting_note
from audio_transcribe.stages.identify import identify_speakers
from audio_transcribe.stages.update import update_meeting


def test_identify_matches_enrolled_speaker_from_previous_meeting(tmp_path):
    """Speakers enrolled from meeting 1 are identified in meeting 2."""
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)

    # === Meeting 1: enroll Andrey ===
    m1_dir = tmp_path / "meeting1"
    m1_dir.mkdir()
    data_dir1 = m1_dir / ".audio-data"
    data_dir1.mkdir()

    m1_data = {
        "audio_file": "meeting1.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Привет", "speaker": "SPEAKER_00"},
        ],
    }
    (data_dir1 / "meeting1.json").write_text(json.dumps(m1_data))

    m1_md = format_meeting_note(m1_data, audio_data_path=".audio-data/meeting1.json")
    m1_path = m1_dir / "meeting1.md"
    m1_path.write_text(m1_md)

    # User maps speaker and runs update → enrolls Andrey
    content = m1_path.read_text()
    content = content.replace("SPEAKER_00: Speaker A", 'SPEAKER_00: "[[Andrey]]"')
    m1_path.write_text(content)

    andrey_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=andrey_embedding):
        update_meeting(m1_path, db)

    assert db.has_speaker("Andrey")

    # === Meeting 2: identify Andrey ===
    m2_dir = tmp_path / "meeting2"
    m2_dir.mkdir()
    data_dir2 = m2_dir / ".audio-data"
    data_dir2.mkdir()

    m2_data = {
        "audio_file": "meeting2.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 8.0,
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "Добрый день", "speaker": "SPEAKER_00"},
        ],
    }
    (data_dir2 / "meeting2.json").write_text(json.dumps(m2_data))

    m2_md = format_meeting_note(m2_data, audio_data_path=".audio-data/meeting2.json")
    m2_path = m2_dir / "meeting2.md"
    m2_path.write_text(m2_md)

    # Identify should match SPEAKER_00 → Andrey (embedding close to [1,0,0])
    similar_embedding = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=similar_embedding):
        result = identify_speakers(m2_path, db)

    assert result.matched.get("SPEAKER_00") == "Andrey"

    # Verify the meeting file was updated
    doc = parse_meeting(m2_path.read_text())
    assert doc.frontmatter["speakers"]["SPEAKER_00"] == "[[Andrey]]"

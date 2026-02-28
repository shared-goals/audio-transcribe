"""Tests for speaker identification logic."""

import json
import textwrap
from unittest.mock import patch

import numpy as np

from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.identify import identify_speakers


def test_identify_matches_known_speaker(tmp_path):
    """Known speaker in DB gets matched and mapped."""
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)
    known_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db.enroll("Andrey", known_embedding)

    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: Speaker A
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    with patch(
        "audio_transcribe.speakers.embeddings.extract_speaker_embedding",
        return_value=np.array([0.95, 0.05, 0.0], dtype=np.float32),
    ):
        result = identify_speakers(meeting_path, db)

    assert len(result.matched) >= 1
    assert result.matched["SPEAKER_00"] == "Andrey"


def test_identify_no_match_for_unknown(tmp_path):
    """Unknown speaker gets no match."""
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)
    db.enroll("Andrey", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: Speaker A
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    with patch(
        "audio_transcribe.speakers.embeddings.extract_speaker_embedding",
        return_value=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ):
        result = identify_speakers(meeting_path, db)

    assert len(result.matched) == 0
    assert "SPEAKER_00" in result.unmatched

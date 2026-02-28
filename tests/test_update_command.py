"""Tests for update subcommand logic."""

import json
import textwrap
from unittest.mock import patch

import numpy as np

from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.update import update_meeting


def test_update_applies_speaker_mapping(tmp_path):
    """Speaker mapping from frontmatter is applied to transcript body."""
    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: "[[Andrey]]"
          SPEAKER_01: "[[Maria]]"
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00
        - **Speaker B**: SPEAKER_01

        ## Transcript

        [00:00] Speaker A: Hello
        [00:05] Speaker B: Hi there
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 7.5, "text": "Hi there", "speaker": "SPEAKER_01"},
        ],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    db_dir = tmp_path / "speakers"

    with patch(
        "audio_transcribe.speakers.embeddings.extract_speaker_embedding",
        return_value=np.random.randn(256).astype(np.float32),
    ):
        update_meeting(meeting_path, SpeakerDB(db_dir))

    result = meeting_path.read_text()
    assert "[[Andrey]]" in result
    assert "[[Maria]]" in result
    assert "Speaker A" not in result.split("## Transcript")[1]
    assert "reanalyze: true" in result


def test_update_enrolls_new_wiki_link_speakers(tmp_path):
    """New [[wiki-link]] speakers get enrolled in voice DB."""
    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: "[[Andrey]]"
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

    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)

    mock_embedding = np.random.randn(256).astype(np.float32)
    with patch(
        "audio_transcribe.speakers.embeddings.extract_speaker_embedding",
        return_value=mock_embedding,
    ):
        update_meeting(meeting_path, db)

    assert db.has_speaker("Andrey")

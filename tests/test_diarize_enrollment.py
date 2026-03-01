"""Tests for automatic voice enrollment during diarization."""

import json
import textwrap
from unittest.mock import patch

import numpy as np

from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.diarize_update import diarize_and_update


def test_diarize_enrolls_wiki_link_speakers(tmp_path):
    """If frontmatter has [[wiki-link]] speakers, enroll them after diarization."""
    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: "[[Andrey]]"
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **[[Andrey]]**: SPEAKER_00

        ## Transcript

        [00:00] [[Andrey]]: Hello
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

    diarized = [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}]
    mock_embedding = np.random.randn(256).astype(np.float32)

    with (
        patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized),
        patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=mock_embedding),
    ):
        diarize_and_update(meeting_path, db=db, force=True)

    assert db.has_speaker("Andrey")

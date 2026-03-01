"""Tests for the diarize subcommand logic."""

import json
import textwrap
from unittest.mock import patch

import pytest

from audio_transcribe.stages.diarize_update import diarize_and_update


def test_diarize_update_adds_speakers(tmp_path):
    """Diarize updates transcript with speaker labels and adds speaker legend."""
    md_content = textwrap.dedent("""\
        ---
        title: 2026-02-28 meeting
        date: '2026-02-28'
        reanalyze: false
        audio_file: meeting.wav
        audio_data: .audio-data/meeting.json
        ---

        ## Transcript

        [00:00] Привет
        [00:05] Здравствуйте
    """)
    meeting_md = tmp_path / "meetings" / "meeting.md"
    meeting_md.parent.mkdir(parents=True)
    meeting_md.write_text(md_content)

    stored_json = {
        "audio_file": "meeting.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет"},
            {"start": 5.0, "end": 7.5, "text": "Здравствуйте"},
        ],
    }
    audio_data_dir = tmp_path / "meetings" / ".audio-data"
    audio_data_dir.mkdir(parents=True)
    (audio_data_dir / "meeting.json").write_text(json.dumps(stored_json))

    diarized_segments = [
        {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 7.5, "text": "Здравствуйте", "speaker": "SPEAKER_01"},
    ]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized_segments):
        diarize_and_update(meeting_md)

    result = meeting_md.read_text()
    assert "Speaker A" in result
    assert "Speaker B" in result
    assert "SPEAKER_00" in result
    assert "reanalyze: true" in result
    assert "## Speakers" in result


def test_diarize_update_stores_diarized_json(tmp_path):
    """Diarize updates the stored JSON with speaker labels."""
    md_content = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        audio_data: .audio-data/test.json
        ---

        ## Transcript

        [00:00] Hello
    """)
    meeting_md = tmp_path / "test.md"
    meeting_md.write_text(md_content)

    stored = {
        "audio_file": "test.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 5.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    json_path = data_dir / "test.json"
    json_path.write_text(json.dumps(stored))

    diarized = [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"}]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized):
        diarize_and_update(meeting_md)

    updated_json = json.loads(json_path.read_text())
    assert updated_json["segments"][0]["speaker"] == "SPEAKER_00"


def test_diarize_refuses_if_already_diarized(tmp_path):
    """Diarize refuses if Speakers section exists unless force=True."""
    md_content = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    meeting_md = tmp_path / "meeting.md"
    meeting_md.write_text(md_content)

    with pytest.raises(RuntimeError, match="already diarized"):
        diarize_and_update(meeting_md)


def test_diarize_preserves_user_text_edits(tmp_path):
    """Diarize adds speaker labels but preserves user-edited transcript text."""
    md_content = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        audio_data: .audio-data/test.json
        ---

        ## Transcript

        [00:00] Привет, коллеги!
        [00:05] Добрый день всем
    """)
    meeting_md = tmp_path / "meeting.md"
    meeting_md.write_text(md_content)

    stored = {
        "audio_file": "test.wav",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет коллеги"},
            {"start": 5.0, "end": 7.5, "text": "Добрый день"},
        ],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    diarized = [
        {"start": 0.0, "end": 2.5, "text": "Привет коллеги", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 7.5, "text": "Добрый день", "speaker": "SPEAKER_01"},
    ]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized):
        diarize_and_update(meeting_md)

    result = meeting_md.read_text()
    assert "Привет, коллеги!" in result
    assert "Добрый день всем" in result
    assert "Speaker A:" in result
    assert "Speaker B:" in result

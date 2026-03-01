"""Tests for audio_transcribe.stages.format — transcript formatting."""

import json

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.stages.format import (
    build_speaker_legend,
    compute_duration,
    format_meeting_note,
    format_segment,
    format_time,
    format_transcript,
)

# --- format_time ---


def test_format_time_zero():
    assert format_time(0.0) == "00:00"


def test_format_time_seconds_only():
    assert format_time(45.0) == "00:45"


def test_format_time_minutes_and_seconds():
    assert format_time(125.0) == "02:05"


def test_format_time_exact_minute():
    assert format_time(60.0) == "01:00"


def test_format_time_over_one_hour():
    assert format_time(3661.0) == "1:01:01"


def test_format_time_fractional_truncated():
    assert format_time(90.9) == "01:30"


def test_format_time_large_hours():
    assert format_time(7200.0) == "2:00:00"


# --- build_speaker_legend ---


def test_build_speaker_legend_empty():
    assert build_speaker_legend([]) == {}


def test_build_speaker_legend_single_speaker():
    segs = [{"speaker": "SPEAKER_00", "text": "hi"}]
    legend = build_speaker_legend(segs)
    assert legend == {"SPEAKER_00": "Speaker A"}


def test_build_speaker_legend_preserves_first_appearance_order():
    segs = [
        {"speaker": "SPEAKER_02", "text": "first"},
        {"speaker": "SPEAKER_00", "text": "second"},
        {"speaker": "SPEAKER_02", "text": "again"},
    ]
    legend = build_speaker_legend(segs)
    assert legend == {"SPEAKER_02": "Speaker A", "SPEAKER_00": "Speaker B"}


def test_build_speaker_legend_unknown_excluded():
    segs = [
        {"speaker": "UNKNOWN", "text": "a"},
        {"speaker": "SPEAKER_00", "text": "b"},
    ]
    legend = build_speaker_legend(segs)
    assert "UNKNOWN" not in legend
    assert legend == {"SPEAKER_00": "Speaker A"}


def test_build_speaker_legend_no_speaker_key():
    segs = [{"text": "no speaker"}]
    legend = build_speaker_legend(segs)
    assert legend == {}


def test_build_speaker_legend_three_speakers():
    segs = [
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
        {"speaker": "SPEAKER_02"},
    ]
    legend = build_speaker_legend(segs)
    assert legend == {
        "SPEAKER_00": "Speaker A",
        "SPEAKER_01": "Speaker B",
        "SPEAKER_02": "Speaker C",
    }


# --- format_segment ---


def test_format_segment_basic():
    seg = {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"}
    legend = {"SPEAKER_00": "Speaker A"}
    assert format_segment(seg, legend) == "[00:00] Speaker A: Привет"


def test_format_segment_with_timestamp():
    seg = {"start": 125.0, "end": 130.0, "text": "Test", "speaker": "SPEAKER_01"}
    legend = {"SPEAKER_01": "Speaker B"}
    assert format_segment(seg, legend) == "[02:05] Speaker B: Test"


def test_format_segment_unknown_speaker():
    seg = {"start": 10.0, "end": 12.0, "text": "hello", "speaker": "UNKNOWN"}
    assert format_segment(seg) == "[00:10] Unknown: hello"


def test_format_segment_no_legend():
    seg = {"start": 0.0, "end": 1.0, "text": "test", "speaker": "SPEAKER_00"}
    assert format_segment(seg) == "[00:00] SPEAKER_00: test"


def test_format_segment_missing_speaker():
    seg = {"start": 0.0, "end": 1.0, "text": "test"}
    assert format_segment(seg) == "[00:00] Unknown: test"


def test_format_segment_text_stripped():
    seg = {"start": 0.0, "end": 1.0, "text": "  extra spaces  ", "speaker": "SPEAKER_00"}
    assert format_segment(seg) == "[00:00] SPEAKER_00: extra spaces"


# --- compute_duration ---


def test_compute_duration_empty():
    assert compute_duration([]) == 0.0


def test_compute_duration_single_segment():
    assert compute_duration([{"start": 0.0, "end": 5.5}]) == 5.5


def test_compute_duration_multiple_segments():
    segs = [
        {"start": 0.0, "end": 2.0},
        {"start": 3.0, "end": 10.0},
        {"start": 11.0, "end": 8.0},
    ]
    assert compute_duration(segs) == 10.0


def test_compute_duration_missing_end():
    segs = [{"start": 0.0}]
    assert compute_duration(segs) == 0.0


# --- format_transcript (integration) ---


def _make_data(segments=None, **kwargs):
    data = {
        "audio_file": kwargs.get("audio_file", "test.wav"),
        "language": kwargs.get("language", "ru"),
        "model": kwargs.get("model", "large-v3"),
        "processing_time_s": kwargs.get("processing_time_s", 42.0),
        "segments": segments or [],
    }
    return data


def test_format_transcript_empty_segments():
    md = format_transcript(_make_data())
    assert "---" in md
    assert "audio_file: test.wav" in md
    assert "speakers: 0" in md
    assert "## Transcript" in md
    assert "## Speakers" not in md


def test_format_transcript_metadata_header():
    md = format_transcript(_make_data(language="en", model="base", processing_time_s=10.5))
    assert "language: en" in md
    assert "model: base" in md
    assert "processing_time_s: 10.5" in md


def test_format_transcript_with_segments():
    segs = [
        {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.0, "text": "Мир", "speaker": "SPEAKER_01"},
    ]
    md = format_transcript(_make_data(segs))
    assert "speakers: 2" in md
    assert "## Speakers" in md
    assert "**Speaker A**: SPEAKER_00" in md
    assert "**Speaker B**: SPEAKER_01" in md
    assert "[00:00] Speaker A: Привет" in md
    assert "[00:03] Speaker B: Мир" in md


def test_format_transcript_duration_from_segments():
    segs = [
        {"start": 0.0, "end": 30.0, "text": "a", "speaker": "SPEAKER_00"},
        {"start": 30.0, "end": 125.0, "text": "b", "speaker": "SPEAKER_00"},
    ]
    md = format_transcript(_make_data(segs))
    assert "duration: 02:05" in md


def test_format_transcript_roundtrip_json(tmp_path):
    """Verify format_transcript works with JSON loaded from a file."""
    data = _make_data([
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
    ])
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    md = format_transcript(loaded)
    assert "[00:00] Speaker A: hello" in md


def test_format_transcript_unknown_speakers_not_in_legend():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": "UNKNOWN"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_00"},
    ]
    md = format_transcript(_make_data(segs))
    assert "speakers: 1" in md
    assert "[00:00] Unknown: a" in md
    assert "[00:01] Speaker A: b" in md


# --- format_meeting_note ---


def test_format_fast_pass_no_speakers():
    """Fast pass output has no speaker section when diarization was skipped."""
    data = {
        "audio_file": "meeting.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет"},
            {"start": 2.5, "end": 5.0, "text": "Здравствуйте"},
        ],
    }
    result = format_meeting_note(data, audio_data_path=".audio-data/meeting.json")
    doc = parse_meeting(result)

    assert doc.frontmatter["reanalyze"] is True
    assert doc.frontmatter["audio_data"] == ".audio-data/meeting.json"
    assert "Speakers" not in doc.sections
    assert "Transcript" in doc.sections
    assert "Привет" in doc.sections["Transcript"]


def test_format_with_speakers():
    """When segments have speaker labels, include speaker section."""
    data = {
        "audio_file": "meeting.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 5.0, "text": "Здравствуйте", "speaker": "SPEAKER_01"},
        ],
    }
    result = format_meeting_note(data, audio_data_path=".audio-data/meeting.json")
    doc = parse_meeting(result)

    assert "Speakers" in doc.sections
    assert doc.frontmatter["speakers"]["SPEAKER_00"] == "Speaker A"
    assert doc.frontmatter["speakers"]["SPEAKER_01"] == "Speaker B"
    # Transcript has clean text without speaker prefixes (diarize step adds them)
    assert "Привет" in doc.sections["Transcript"]
    assert "Speaker A" not in doc.sections["Transcript"]


def test_format_frontmatter_has_audio_file():
    data = {
        "audio_file": "recordings/2026-02-28-standup.mp3",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
    }
    result = format_meeting_note(data, audio_data_path=".audio-data/test.json")
    doc = parse_meeting(result)
    assert doc.frontmatter["audio_file"] == "recordings/2026-02-28-standup.mp3"
    assert doc.frontmatter["date"] == "2026-02-28"


def test_format_date_fallback_to_today():
    data = {
        "audio_file": "standup.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
    }
    from datetime import date
    result = format_meeting_note(data, audio_data_path=".audio-data/test.json")
    doc = parse_meeting(result)
    assert doc.frontmatter["date"] == str(date.today())

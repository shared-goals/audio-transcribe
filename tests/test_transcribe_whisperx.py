"""Tests for transcribe_whisperx.py — build_output (pure function)."""

from transcribe_whisperx import build_output


def test_build_output_empty_segments():
    output = build_output({}, "audio.wav", "ru", "large-v3", 10.0)
    assert output["audio_file"] == "audio.wav"
    assert output["language"] == "ru"
    assert output["model"] == "large-v3"
    assert output["processing_time_s"] == 10.0
    assert output["segments"] == []


def test_build_output_text_stripped():
    result_in = {"segments": [{"start": 0.0, "end": 2.5, "text": "  Привет  "}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert output["segments"][0]["text"] == "Привет"


def test_build_output_unknown_speaker_default():
    result_in = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert output["segments"][0]["speaker"] == "UNKNOWN"


def test_build_output_speaker_assigned():
    result_in = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "hi", "speaker": "SPEAKER_01"}]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert output["segments"][0]["speaker"] == "SPEAKER_01"


def test_build_output_words_included():
    result_in = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                    {"word": "world", "start": 0.6, "end": 1.0, "speaker": "SPEAKER_00"},
                ],
            }
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    words = output["segments"][0]["words"]
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert words[1]["word"] == "world"


def test_build_output_words_without_start_excluded():
    """Words missing 'start' timestamp should be dropped."""
    result_in = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "no_time"},  # no 'start' key
                ],
            }
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert len(output["segments"][0]["words"]) == 1


def test_build_output_timestamps_rounded():
    result_in = {"segments": [{"start": 1.23456789, "end": 5.98765432, "text": "test"}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 9.9999)
    seg = output["segments"][0]
    assert seg["start"] == 1.235
    assert seg["end"] == 5.988
    assert output["processing_time_s"] == 10.0


def test_build_output_multiple_segments():
    result_in = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "first", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "text": "second", "speaker": "SPEAKER_01"},
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 5.0)
    assert len(output["segments"]) == 2
    assert output["segments"][1]["speaker"] == "SPEAKER_01"


def test_build_output_no_words_key_when_absent():
    """Segments without 'words' key should not include 'words' in output."""
    result_in = {"segments": [{"start": 0.0, "end": 1.0, "text": "test"}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert "words" not in output["segments"][0]

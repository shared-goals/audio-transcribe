"""Tests for quality scorecard computation."""

from audio_transcribe.quality.scorecard import compute_quality


def test_compute_quality_basic():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                {"word": "world", "start": 0.6, "end": 1.0, "speaker": "SPEAKER_00"},
            ],
        },
    ]
    q = compute_quality(segments)
    assert q.segments == 1
    assert q.words_total == 2
    assert q.words_aligned == 2
    assert q.alignment_pct == 100.0
    assert q.speakers_detected == 1
    assert q.speaker_coverage_pct == 100.0
    assert q.speaker_transitions == 0


def test_compute_quality_missing_speaker():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "UNKNOWN"},
        {"start": 1.5, "end": 2.0, "text": "world", "speaker": "SPEAKER_00"},
    ]
    q = compute_quality(segments)
    assert q.speakers_detected == 1  # UNKNOWN not counted
    assert q.speaker_coverage_pct == 50.0


def test_compute_quality_unaligned_words():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world"},  # no start = unaligned
            ],
        },
    ]
    q = compute_quality(segments)
    assert q.words_total == 2
    assert q.words_aligned == 1
    assert q.alignment_pct == 50.0


def test_compute_quality_speaker_transitions():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 3.0, "text": "c", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 4.0, "text": "d", "speaker": "SPEAKER_00"},
    ]
    q = compute_quality(segments)
    assert q.speaker_transitions == 2  # 00->01, 01->00
    assert q.speakers_detected == 2


def test_compute_quality_empty():
    q = compute_quality([])
    assert q.segments == 0
    assert q.words_total == 0
    assert q.alignment_pct == 0.0
    assert q.speaker_coverage_pct == 0.0


def test_compute_quality_no_words_key():
    """Segments without words: count segment text words as total, 0 aligned."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello world", "speaker": "SPEAKER_00"},
    ]
    q = compute_quality(segments)
    assert q.words_total == 2
    assert q.words_aligned == 0
    assert q.alignment_pct == 0.0

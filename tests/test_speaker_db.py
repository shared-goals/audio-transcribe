"""Tests for speaker embedding database."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from audio_transcribe.speakers import embeddings as _embeddings
from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.speakers.embeddings import cosine_distance


def test_cosine_distance_identical():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(a, b) < 0.01


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_distance(a, b) - 1.0) < 0.01


def test_cosine_distance_opposite():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    assert cosine_distance(a, b) > 1.5


def test_speaker_db_enroll(tmp_path):
    db = SpeakerDB(tmp_path)
    embedding = np.random.randn(256).astype(np.float32)
    db.enroll("Andrey", embedding)

    assert db.has_speaker("Andrey")
    loaded = db.get_embedding("Andrey")
    np.testing.assert_array_almost_equal(loaded, embedding)


def test_speaker_db_enroll_updates_average(tmp_path):
    db = SpeakerDB(tmp_path)
    e1 = np.zeros(256, dtype=np.float32)
    e1[0] = 1.0
    e2 = np.zeros(256, dtype=np.float32)
    e2[1] = 1.0

    db.enroll("Andrey", e1)
    db.enroll("Andrey", e2)

    avg = db.get_embedding("Andrey")
    expected = np.zeros(256, dtype=np.float32)
    expected[0] = 0.5
    expected[1] = 0.5
    np.testing.assert_array_almost_equal(avg, expected)


def test_speaker_db_match(tmp_path):
    db = SpeakerDB(tmp_path)
    e_andrey = np.zeros(256, dtype=np.float32)
    e_andrey[0] = 1.0
    e_maria = np.zeros(256, dtype=np.float32)
    e_maria[1] = 1.0
    db.enroll("Andrey", e_andrey)
    db.enroll("Maria", e_maria)

    query = np.zeros(256, dtype=np.float32)
    query[0] = 0.95
    query[1] = 0.05
    matches = db.match(query, threshold=0.5)
    assert len(matches) >= 1
    assert matches[0][0] == "Andrey"


def test_speaker_db_match_no_match(tmp_path):
    db = SpeakerDB(tmp_path)
    e = np.zeros(256, dtype=np.float32)
    e[0] = 1.0
    db.enroll("Andrey", e)

    query = np.zeros(256, dtype=np.float32)
    query[1] = 1.0
    matches = db.match(query, threshold=0.5)
    assert len(matches) == 0


def test_speaker_db_list(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))
    db.enroll("Maria", np.random.randn(256).astype(np.float32))

    speakers = db.list_speakers()
    assert len(speakers) == 2
    names = {s["name"] for s in speakers}
    assert names == {"Andrey", "Maria"}


def test_speaker_db_forget(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))
    assert db.has_speaker("Andrey")

    db.forget("Andrey")
    assert not db.has_speaker("Andrey")


def test_speaker_db_persistence(tmp_path):
    db1 = SpeakerDB(tmp_path)
    embedding = np.random.randn(256).astype(np.float32)
    db1.enroll("Andrey", embedding)

    db2 = SpeakerDB(tmp_path)
    assert db2.has_speaker("Andrey")
    np.testing.assert_array_almost_equal(db2.get_embedding("Andrey"), embedding)


def test_enroll_wrong_dimension_raises(tmp_path):
    db = SpeakerDB(tmp_path / "speakers")
    bad_embedding = np.zeros(128, dtype=np.float32)
    with pytest.raises(ValueError, match="256"):
        db.enroll("Test Speaker", bad_embedding)


def test_extract_speaker_embedding_returns_none_for_short_segments() -> None:
    """Should return None when no segments are long enough."""
    segments = [
        {"start": 0.0, "end": 0.5, "text": "Hi", "speaker": "SPEAKER_00"},
    ]
    with patch.object(_embeddings, "_load_audio_ffmpeg", return_value={"waveform": None, "sample_rate": 16000}):
        result = _embeddings.extract_speaker_embedding("test.wav", segments, "SPEAKER_00", min_duration=1.0)
    assert result is None


def test_enroll_special_characters_in_name(tmp_path: Path) -> None:
    """Speaker names with special characters should produce valid filenames."""
    db = SpeakerDB(tmp_path / "speakers")
    embedding = np.random.rand(256).astype(np.float32)
    db.enroll("Ivan/Maria", embedding)
    assert db.has_speaker("Ivan/Maria")
    files = list((tmp_path / "speakers").glob("*.npy"))
    assert len(files) == 1
    assert "/" not in files[0].name


def test_match_skips_corrupt_embedding(tmp_path):
    db = SpeakerDB(tmp_path / "speakers")
    good = np.random.rand(256).astype(np.float32)
    db.enroll("Good", good)
    # Manually corrupt the embedding file
    np.save(db._embedding_path("Good"), np.zeros(128, dtype=np.float32))
    query = np.random.rand(256).astype(np.float32)
    results = db.match(query)
    assert len(results) == 0  # Skipped due to dimension mismatch

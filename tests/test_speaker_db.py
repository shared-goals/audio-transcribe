"""Tests for speaker embedding database."""

import numpy as np

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
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    db.enroll("Andrey", e1)
    db.enroll("Andrey", e2)

    avg = db.get_embedding("Andrey")
    np.testing.assert_array_almost_equal(avg, np.array([0.5, 0.5, 0.0]))


def test_speaker_db_match(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    db.enroll("Maria", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    query = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    matches = db.match(query, threshold=0.5)
    assert len(matches) >= 1
    assert matches[0][0] == "Andrey"


def test_speaker_db_match_no_match(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
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

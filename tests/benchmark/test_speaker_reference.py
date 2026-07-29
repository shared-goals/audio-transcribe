"""Validation tests for private two-speaker diarization references."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_transcribe.benchmark.speaker_reference import load_speaker_reference


def _write_reference(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "source_sha256": "audio",
        "aligned_asr_sha256": "asr",
        "audio_duration_s": 10.0,
        "turns": [
            {"start": 0.0, "end": 2.0, "speaker": "speaker_a"},
            {"start": 2.0, "end": 4.0, "speaker": "speaker_b"},
        ],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_speaker_reference_accepts_exactly_two_speakers(tmp_path: Path) -> None:
    path = tmp_path / "reference-speakers.json"
    _write_reference(path)

    reference = load_speaker_reference(path)

    assert reference.source_sha256 == "audio"
    assert reference.aligned_asr_sha256 == "asr"
    assert reference.audio_duration_s == 10.0
    assert {turn.speaker for turn in reference.turns} == {"speaker_a", "speaker_b"}


def test_load_speaker_reference_accepts_cross_speaker_overlap(tmp_path: Path) -> None:
    path = tmp_path / "reference-speakers.json"
    _write_reference(
        path,
        turns=[
            {"start": 0.0, "end": 3.0, "speaker": "speaker_a"},
            {"start": 2.0, "end": 4.0, "speaker": "speaker_b"},
        ],
    )

    assert len(load_speaker_reference(path).turns) == 2


@pytest.mark.parametrize(
    ("turns", "message"),
    [
        (
            [
                {"start": 0.0, "end": 3.0, "speaker": "speaker_a"},
                {"start": 2.0, "end": 4.0, "speaker": "speaker_a"},
                {"start": 4.0, "end": 5.0, "speaker": "speaker_b"},
            ],
            "same-speaker turns overlap",
        ),
        (
            [
                {"start": 0.0, "end": 0.0, "speaker": "speaker_a"},
                {"start": 2.0, "end": 4.0, "speaker": "speaker_b"},
            ],
            "positive duration",
        ),
        (
            [
                {"start": 0.0, "end": 2.0, "speaker": "speaker_a"},
                {"start": 2.0, "end": 4.0, "speaker": "speaker_b"},
                {"start": 4.0, "end": 5.0, "speaker": "speaker_c"},
            ],
            "exactly two speakers",
        ),
        (
            [
                {"start": 0.0, "end": 2.0, "speaker": "speaker_a"},
                {"start": 9.0, "end": 11.0, "speaker": "speaker_b"},
            ],
            "audio duration",
        ),
        (
            [
                {"start": 2.0, "end": 4.0, "speaker": "speaker_a"},
                {"start": 0.0, "end": 2.0, "speaker": "speaker_b"},
            ],
            "sorted",
        ),
    ],
)
def test_load_speaker_reference_rejects_invalid_turns(
    tmp_path: Path,
    turns: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "reference-speakers.json"
    _write_reference(path, turns=turns)

    with pytest.raises(ValueError, match=message):
        load_speaker_reference(path)


@pytest.mark.parametrize("field", ("source_sha256", "aligned_asr_sha256"))
def test_load_speaker_reference_rejects_missing_checksums(tmp_path: Path, field: str) -> None:
    path = tmp_path / "reference-speakers.json"
    payload = {
        "source_sha256": "audio",
        "aligned_asr_sha256": "asr",
        "turns": [
            {"start": 0.0, "end": 2.0, "speaker": "speaker_a"},
            {"start": 2.0, "end": 4.0, "speaker": "speaker_b"},
        ],
    }
    del payload[field]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_speaker_reference(path)


@pytest.mark.parametrize("duration", (None, 0.0, -1.0, float("nan"), float("inf")))
def test_load_speaker_reference_requires_positive_finite_audio_duration(
    tmp_path: Path,
    duration: float | None,
) -> None:
    path = tmp_path / "reference-speakers.json"
    if duration is None:
        _write_reference(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["audio_duration_s"]
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        _write_reference(path, audio_duration_s=duration)

    with pytest.raises(ValueError, match="audio duration"):
        load_speaker_reference(path)


def test_load_speaker_reference_normalizes_speaker_ids(tmp_path: Path) -> None:
    path = tmp_path / "reference-speakers.json"
    _write_reference(
        path,
        turns=[
            {"start": 0.0, "end": 2.0, "speaker": " speaker_a "},
            {"start": 2.0, "end": 4.0, "speaker": "speaker_b\t"},
        ],
    )

    reference = load_speaker_reference(path)

    assert tuple(turn.speaker for turn in reference.turns) == ("speaker_a", "speaker_b")


def test_load_speaker_reference_rejects_empty_speaker_after_normalization(tmp_path: Path) -> None:
    path = tmp_path / "reference-speakers.json"
    _write_reference(
        path,
        turns=[
            {"start": 0.0, "end": 2.0, "speaker": " \t "},
            {"start": 2.0, "end": 4.0, "speaker": "speaker_b"},
        ],
    )

    with pytest.raises(ValueError, match="speaker must be non-empty"):
        load_speaker_reference(path)


def test_load_speaker_reference_detects_self_overlap_after_speaker_normalization(tmp_path: Path) -> None:
    path = tmp_path / "reference-speakers.json"
    _write_reference(
        path,
        turns=[
            {"start": 0.0, "end": 3.0, "speaker": "speaker_a"},
            {"start": 2.0, "end": 4.0, "speaker": " speaker_a "},
            {"start": 4.0, "end": 5.0, "speaker": "speaker_b"},
        ],
    )

    with pytest.raises(ValueError, match="same-speaker turns overlap"):
        load_speaker_reference(path)

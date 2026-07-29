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

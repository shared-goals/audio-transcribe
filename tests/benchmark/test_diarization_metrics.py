"""Exact behavioral tests for permutation-invariant diarization metrics."""

from __future__ import annotations

import pytest

from audio_transcribe.benchmark.diarization_metrics import evaluate_diarization
from audio_transcribe.benchmark.models import SpeakerTurn


def test_swapped_anonymous_labels_are_not_confusion() -> None:
    reference = (
        SpeakerTurn(0.0, 5.0, "speaker_a"),
        SpeakerTurn(5.0, 10.0, "speaker_b"),
    )
    predicted = (
        SpeakerTurn(0.0, 5.0, "SPEAKER_01"),
        SpeakerTurn(5.0, 10.0, "SPEAKER_00"),
    )

    metrics = evaluate_diarization(reference, predicted, aligned_words=[])

    assert metrics.confusion_s == 0.0
    assert metrics.der == 0.0
    assert metrics.label_mapping == {"SPEAKER_01": "speaker_a", "SPEAKER_00": "speaker_b"}


@pytest.mark.parametrize(
    ("predicted", "missed", "false_alarm", "confusion", "der"),
    [
        ((SpeakerTurn(0.0, 3.0, "p0"), SpeakerTurn(4.0, 8.0, "p1")), 1.0, 0.0, 0.0, 0.125),
        ((SpeakerTurn(0.0, 4.0, "p0"), SpeakerTurn(4.0, 9.0, "p1")), 0.0, 1.0, 0.0, 0.125),
        ((SpeakerTurn(0.0, 5.0, "p0"), SpeakerTurn(5.0, 8.0, "p1")), 0.0, 0.0, 1.0, 0.125),
    ],
)
def test_reports_exact_one_second_der_components(
    predicted: tuple[SpeakerTurn, ...],
    missed: float,
    false_alarm: float,
    confusion: float,
    der: float,
) -> None:
    reference = (
        SpeakerTurn(0.0, 4.0, "a"),
        SpeakerTurn(4.0, 8.0, "b"),
    )

    metrics = evaluate_diarization(reference, predicted, aligned_words=[])

    assert metrics.missed_speech_s == missed
    assert metrics.false_alarm_s == false_alarm
    assert metrics.confusion_s == confusion
    assert metrics.der == der


def test_overlap_counts_speaker_time_and_scores_both_active_speakers() -> None:
    reference = (
        SpeakerTurn(0.0, 3.0, "a"),
        SpeakerTurn(2.0, 4.0, "b"),
    )
    predicted = (
        SpeakerTurn(0.0, 3.0, "p0"),
        SpeakerTurn(2.0, 4.0, "p1"),
    )

    metrics = evaluate_diarization(reference, predicted, aligned_words=[])

    assert metrics.der == 0.0
    assert metrics.label_mapping == {"p0": "a", "p1": "b"}


def test_word_accuracy_uses_midpoint_only_for_unambiguous_reference_speech() -> None:
    reference = (
        SpeakerTurn(0.0, 3.0, "a"),
        SpeakerTurn(2.0, 5.0, "b"),
    )
    predicted = (
        SpeakerTurn(0.0, 3.0, "p0"),
        SpeakerTurn(2.0, 5.0, "p1"),
    )
    words = [
        {"start": 0.2, "end": 0.8, "speaker": "p0"},
        {"start": 3.2, "end": 3.8, "speaker": "p0"},
        {"start": 2.2, "end": 2.8, "speaker": "p1"},
        {"start": 6.0, "end": 6.5, "speaker": "p0"},
    ]

    metrics = evaluate_diarization(reference, predicted, aligned_words=words)

    assert metrics.word_speaker_accuracy == 0.5


def test_counts_predicted_fragments_strictly_shorter_than_half_second() -> None:
    reference = (
        SpeakerTurn(0.0, 2.0, "a"),
        SpeakerTurn(2.0, 4.0, "b"),
    )
    predicted = (
        SpeakerTurn(0.0, 0.49, "p0"),
        SpeakerTurn(0.49, 2.0, "p0"),
        SpeakerTurn(2.0, 2.5, "p1"),
        SpeakerTurn(2.5, 4.0, "p1"),
    )

    metrics = evaluate_diarization(reference, predicted, aligned_words=[])

    assert metrics.fragmented_turns == 1


def test_mapping_tie_is_deterministic() -> None:
    reference = (
        SpeakerTurn(0.0, 1.0, "a"),
        SpeakerTurn(1.0, 2.0, "b"),
    )
    predicted = (SpeakerTurn(3.0, 4.0, "p0"),)

    metrics = evaluate_diarization(reference, predicted, aligned_words=[])

    assert metrics.label_mapping == {"p0": "a"}


def test_collar_excludes_reference_boundary_regions() -> None:
    reference = (
        SpeakerTurn(0.0, 2.0, "a"),
        SpeakerTurn(2.0, 4.0, "b"),
    )
    predicted = (
        SpeakerTurn(0.0, 1.75, "p0"),
        SpeakerTurn(2.25, 4.0, "p1"),
    )

    metrics = evaluate_diarization(reference, predicted, aligned_words=[], collar_s=0.25)

    assert metrics.der == 0.0


@pytest.mark.parametrize(
    ("reference", "predicted", "words", "collar"),
    [
        (
            (SpeakerTurn(0.0, 1.0, "a"), SpeakerTurn(1.0, 2.0, "a")),
            (),
            [],
            0.0,
        ),
        (
            (SpeakerTurn(0.0, 1.0, "a"), SpeakerTurn(1.0, 2.0, "b")),
            (SpeakerTurn(0.0, 1.0, ""),),
            [],
            0.0,
        ),
        (
            (SpeakerTurn(0.0, 1.0, "a"), SpeakerTurn(1.0, 2.0, "b")),
            (),
            [{"start": 1.0, "end": 0.0, "speaker": "p0"}],
            0.0,
        ),
        (
            (SpeakerTurn(0.0, 1.0, "a"), SpeakerTurn(1.0, 2.0, "b")),
            (),
            [],
            -0.1,
        ),
    ],
)
def test_rejects_invalid_metric_inputs(
    reference: tuple[SpeakerTurn, ...],
    predicted: tuple[SpeakerTurn, ...],
    words: list[dict[str, object]],
    collar: float,
) -> None:
    with pytest.raises(ValueError):
        evaluate_diarization(reference, predicted, aligned_words=words, collar_s=collar)

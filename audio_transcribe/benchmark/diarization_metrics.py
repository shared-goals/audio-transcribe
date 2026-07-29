"""Permutation-invariant diarization quality metrics for two-speaker meetings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import permutations
from math import isfinite
from typing import cast

from audio_transcribe.benchmark.models import DiarizationMetrics, SpeakerTurn

_FRAGMENT_THRESHOLD_S = 0.5


def _coerce_finite_float(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _normalize_turns(
    turns: object,
    *,
    name: str,
    require_two_speakers: bool,
) -> tuple[tuple[SpeakerTurn, ...], tuple[str, ...]]:
    if not isinstance(turns, Sequence):
        raise ValueError(f"{name} turns must be a sequence")
    speakers: set[str] = set()
    normalized_turns: list[SpeakerTurn] = []
    for turn in turns:
        if not isinstance(turn, SpeakerTurn):
            raise ValueError(f"invalid {name} turn")
        start = _coerce_finite_float(turn.start_s, name=f"{name} turn start")
        end = _coerce_finite_float(turn.end_s, name=f"{name} turn end")
        if start < 0.0 or end <= start or not isinstance(turn.speaker, str) or not turn.speaker.strip():
            raise ValueError(f"invalid {name} turn")
        speaker = turn.speaker.strip()
        speakers.add(speaker)
        normalized_turns.append(SpeakerTurn(float(start), float(end), speaker))
    if require_two_speakers and len(speakers) != 2:
        raise ValueError("reference must contain exactly two speakers")
    if len(speakers) > 2:
        raise ValueError(f"{name} must contain at most two speakers")
    return tuple(normalized_turns), tuple(sorted(speakers))


def _normalize_words(aligned_words: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(aligned_words, Sequence):
        raise ValueError("aligned words must be a sequence")
    normalized_words: list[Mapping[str, object]] = []
    for word in aligned_words:
        if not isinstance(word, Mapping):
            raise ValueError("invalid aligned word")
        start = _coerce_finite_float(word.get("start"), name="aligned word start")
        end = _coerce_finite_float(word.get("end"), name="aligned word end")
        if start < 0.0 or end <= start:
            raise ValueError("invalid aligned word timestamps")
        speaker = word.get("speaker")
        if speaker is not None and (not isinstance(speaker, str) or not speaker.strip()):
            raise ValueError("invalid aligned word speaker")
        normalized_words.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker.strip() if isinstance(speaker, str) else None,
            }
        )
    return tuple(normalized_words)


def _active_speakers(turns: Sequence[SpeakerTurn], midpoint: float) -> set[str]:
    return {turn.speaker for turn in turns if turn.start_s <= midpoint < turn.end_s}


def _is_in_collar(midpoint: float, reference_boundaries: set[float], collar_s: float) -> bool:
    return collar_s > 0.0 and any(abs(midpoint - boundary) < collar_s for boundary in reference_boundaries)


def _score_mapping(
    reference: Sequence[SpeakerTurn],
    predicted: Sequence[SpeakerTurn],
    mapping: dict[str, str],
    boundaries: Sequence[float],
    reference_boundaries: set[float],
    collar_s: float,
) -> tuple[float, float, float, float]:
    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0
    reference_speech = 0.0
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        duration = end - start
        midpoint = start + duration / 2.0
        if _is_in_collar(midpoint, reference_boundaries, collar_s):
            continue
        reference_active = _active_speakers(reference, midpoint)
        predicted_active = {mapping[speaker] for speaker in _active_speakers(predicted, midpoint)}
        reference_count = len(reference_active)
        predicted_count = len(predicted_active)
        correct_count = len(reference_active & predicted_active)
        reference_speech += reference_count * duration
        missed += max(0, reference_count - predicted_count) * duration
        false_alarm += max(0, predicted_count - reference_count) * duration
        confusion += (min(reference_count, predicted_count) - correct_count) * duration
    return missed, false_alarm, confusion, reference_speech


def _mapping_candidates(
    predicted_speakers: tuple[str, ...],
    reference_speakers: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    if not predicted_speakers:
        return ({},)
    return tuple(
        dict(zip(predicted_speakers, reference_permutation, strict=True))
        for reference_permutation in permutations(reference_speakers, len(predicted_speakers))
    )


def _word_speaker_accuracy(
    reference: Sequence[SpeakerTurn],
    aligned_words: Sequence[Mapping[str, object]],
    mapping: dict[str, str],
) -> float:
    eligible = 0
    correct = 0
    for word in aligned_words:
        start = float(cast(int | float, word["start"]))
        end = float(cast(int | float, word["end"]))
        reference_active = _active_speakers(reference, start + (end - start) / 2.0)
        if len(reference_active) != 1:
            continue
        eligible += 1
        predicted_speaker = word.get("speaker")
        if isinstance(predicted_speaker, str) and mapping.get(predicted_speaker) in reference_active:
            correct += 1
    return correct / eligible if eligible else 0.0


def evaluate_diarization(
    reference: Sequence[SpeakerTurn],
    predicted_turns: Sequence[SpeakerTurn],
    aligned_words: Sequence[Mapping[str, object]],
    collar_s: float = 0.0,
) -> DiarizationMetrics:
    """Score a two-speaker prediction after choosing the best anonymous-label permutation."""
    normalized_collar = _coerce_finite_float(collar_s, name="collar_s")
    if normalized_collar < 0.0:
        raise ValueError("collar_s must be finite and non-negative")
    normalized_reference, reference_speakers = _normalize_turns(
        reference,
        name="reference",
        require_two_speakers=True,
    )
    normalized_prediction, predicted_speakers = _normalize_turns(
        predicted_turns,
        name="prediction",
        require_two_speakers=False,
    )
    normalized_words = _normalize_words(aligned_words)

    reference_boundaries = {edge for turn in normalized_reference for edge in (turn.start_s, turn.end_s)}
    raw_boundaries = reference_boundaries | {
        edge for turn in normalized_prediction for edge in (turn.start_s, turn.end_s)
    }
    if normalized_collar > 0.0:
        raw_boundaries |= {
            collar_edge
            for boundary in reference_boundaries
            for collar_edge in (
                max(0.0, boundary - normalized_collar),
                boundary + normalized_collar,
            )
        }
    boundaries = tuple(sorted(raw_boundaries))

    scored = []
    for mapping in _mapping_candidates(predicted_speakers, reference_speakers):
        components = _score_mapping(
            normalized_reference,
            normalized_prediction,
            mapping,
            boundaries,
            reference_boundaries,
            normalized_collar,
        )
        scored.append((sum(components[:3]), tuple(mapping.items()), mapping, components))
    _, _, best_mapping, (missed, false_alarm, confusion, reference_speech) = min(scored)
    if reference_speech <= 0.0:
        raise ValueError("collar leaves no scored reference speaker-time")
    der = (missed + false_alarm + confusion) / reference_speech

    return DiarizationMetrics(
        der=der,
        missed_speech_s=missed,
        false_alarm_s=false_alarm,
        confusion_s=confusion,
        word_speaker_accuracy=_word_speaker_accuracy(normalized_reference, normalized_words, best_mapping),
        fragmented_turns=sum(turn.end_s - turn.start_s < _FRAGMENT_THRESHOLD_S for turn in normalized_prediction),
        label_mapping=best_mapping,
    )

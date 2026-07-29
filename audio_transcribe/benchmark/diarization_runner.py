"""Isolated execution for pyannote diarization benchmark candidates."""

from __future__ import annotations

import gc
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Audio, Pipeline

from audio_transcribe.benchmark.models import DiarizationCandidate, DiarizationRunResult, SpeakerTurn


def _peak_rss_mb() -> float:
    """Return process peak RSS in MB using the repository's platform conversion."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return float(usage.ru_maxrss) / (1024 * 1024)
    return float(usage.ru_maxrss) / 1024


def _validate_candidate(candidate: DiarizationCandidate) -> None:
    if (
        not isinstance(candidate.name, str)
        or not candidate.name.strip()
        or not isinstance(candidate.model_id, str)
        or not candidate.model_id.strip()
        or not isinstance(candidate.exclusive, bool)
    ):
        raise ValueError("invalid diarization candidate")


def _turns_from_annotation(annotation: Any) -> tuple[SpeakerTurn, ...]:
    turns: list[SpeakerTurn] = []
    for segment, _, raw_speaker in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        speaker = str(raw_speaker).strip()
        if not isfinite(start) or not isfinite(end) or start < 0.0 or end <= start or not speaker:
            raise ValueError("pyannote returned an invalid speaker turn")
        turns.append(SpeakerTurn(start, end, speaker))
    return tuple(turns)


def _select_annotation(output: Any, *, exclusive: bool) -> Any:
    attribute = "exclusive_speaker_diarization" if exclusive else "speaker_diarization"
    if hasattr(output, attribute):
        return getattr(output, attribute)
    if exclusive:
        raise ValueError("candidate did not return exclusive_speaker_diarization")
    if hasattr(output, "itertracks"):
        return output
    raise ValueError("candidate did not return speaker_diarization")


def _aligned_words(aligned_result: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(aligned_result, Mapping):
        raise ValueError("aligned result must be a mapping")
    segments = aligned_result.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("aligned result segments must be a sequence")
    words: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("aligned result contains an invalid segment")
        raw_words = segment.get("words", ())
        if not isinstance(raw_words, Sequence) or isinstance(raw_words, (str, bytes)):
            raise ValueError("aligned segment words must be a sequence")
        for raw_word in raw_words:
            if not isinstance(raw_word, Mapping):
                raise ValueError("aligned result contains an invalid word")
            start_value = raw_word.get("start")
            end_value = raw_word.get("end")
            if (
                not isinstance(start_value, (int, float))
                or isinstance(start_value, bool)
                or not isinstance(end_value, (int, float))
                or isinstance(end_value, bool)
            ):
                raise ValueError("aligned word timestamps must be finite and increasing")
            start = float(start_value)
            end = float(end_value)
            if not isfinite(start) or not isfinite(end) or start < 0.0 or end <= start:
                raise ValueError("aligned word timestamps must be finite and increasing")
            word = dict(raw_word)
            word["start"] = start
            word["end"] = end
            words.append(word)
    return tuple(words)


def _assign_words(
    aligned_result: object,
    turns: tuple[SpeakerTurn, ...],
) -> tuple[dict[str, Any], ...]:
    assigned: list[dict[str, Any]] = []
    for word in _aligned_words(aligned_result):
        start = float(word["start"])
        end = float(word["end"])
        overlaps = tuple(max(0.0, min(end, turn.end_s) - max(start, turn.start_s)) for turn in turns)
        best_index = max(range(len(turns)), key=overlaps.__getitem__) if turns else None
        word["speaker"] = turns[best_index].speaker if best_index is not None and overlaps[best_index] > 0.0 else None
        assigned.append(word)
    return tuple(assigned)


def _is_gated_access_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, PermissionError) or any(
        marker in message for marker in ("401", "403", "unauthorized", "forbidden", "gated")
    )


def run_diarization_candidate(
    audio: Path,
    aligned_result: dict[str, Any],
    token: str,
    candidate: DiarizationCandidate,
) -> DiarizationRunResult:
    """Run one two-speaker pyannote candidate without allowing its failure to escape."""
    _validate_candidate(candidate)
    started = time.perf_counter()
    pipeline: Any = None
    try:
        pipeline = Pipeline.from_pretrained(candidate.model_id, token=token)
        if pipeline is None:
            raise RuntimeError("pyannote did not return a pipeline")
        pipeline.to(torch.device("cpu"))
        waveform, sample_rate = Audio(sample_rate=16_000, mono="downmix")(audio)
        output = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=2,
        )
        turns = _turns_from_annotation(_select_annotation(output, exclusive=candidate.exclusive))
        word_assignments = _assign_words(aligned_result, turns)
        return DiarizationRunResult(
            candidate=candidate,
            turns=turns,
            word_assignments=word_assignments,
            time_s=time.perf_counter() - started,
            peak_rss_mb=_peak_rss_mb(),
            status="completed",
            error=None,
        )
    except Exception as exc:
        if _is_gated_access_failure(exc):
            error = (
                f"Cannot access gated model {candidate.model_id}. Set a valid HF_TOKEN and accept the model license "
                f"on Hugging Face. Original error: {exc}"
            )
        else:
            error = str(exc)
        return DiarizationRunResult(
            candidate=candidate,
            turns=(),
            word_assignments=(),
            time_s=time.perf_counter() - started,
            peak_rss_mb=_peak_rss_mb(),
            status="failed",
            error=error,
        )
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()

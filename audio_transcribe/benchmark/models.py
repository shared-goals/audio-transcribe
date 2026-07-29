"""Immutable data models shared by ASR benchmark modules."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from audio_transcribe.stages.transcribe import MlxVadOptions


@dataclass(frozen=True)
class ReferenceSpan:
    """One manually reviewed interval in an ASR reference transcript."""

    start_s: float
    end_s: float
    text: str
    excluded: bool


@dataclass(frozen=True)
class ReferenceDocument:
    """An immutable reference transcript bound to its source file checksum."""

    source_sha256: str
    spans: tuple[ReferenceSpan, ...]


@dataclass(frozen=True)
class SpeakerTurn:
    """One verified speaker interval in a diarization reference."""

    start_s: float
    end_s: float
    speaker: str


@dataclass(frozen=True)
class SpeakerReference:
    """Verified two-speaker turns bound to source audio and aligned ASR."""

    source_sha256: str
    aligned_asr_sha256: str
    audio_duration_s: float
    turns: tuple[SpeakerTurn, ...]


@dataclass(frozen=True)
class DiarizationMetrics:
    """Permutation-invariant timeline and word-level diarization measurements."""

    der: float
    missed_speech_s: float
    false_alarm_s: float
    confusion_s: float
    word_speaker_accuracy: float
    fragmented_turns: int
    label_mapping: dict[str, str]


@dataclass(frozen=True)
class AsrMetrics:
    """Immutable quality measurements for one candidate ASR transcript."""

    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int
    term_hits: int
    term_total: int
    term_recall: float
    hallucinated_phrases: tuple[str, ...]
    missed_phrases: tuple[str, ...]
    boundary_errors: int


@dataclass(frozen=True)
class AsrCandidate:
    """One isolated MLX-VAD model and inference configuration."""

    name: str
    model: str
    options: MlxVadOptions


@dataclass(frozen=True)
class AsrRunResult:
    """Outcome and resource measurements from one ASR candidate."""

    candidate: AsrCandidate
    result: dict[str, Any] | None
    stage_times_s: dict[str, float]
    peak_rss_mb: float
    vad_boundaries: tuple[float, ...]
    status: str
    error: str | None


@dataclass(frozen=True)
class RunManifest:
    """Reproducibility, environment, timing, and status evidence for one benchmark run."""

    candidate_name: str
    source_sha256: str
    reference_sha256: str
    vocabulary_sha256: str
    git_commit: str
    model: str
    model_revision: str | None
    options: dict[str, Any]
    hardware: dict[str, str]
    stage_times_s: dict[str, float]
    peak_rss_mb: float
    status: str
    error: str | None


@dataclass(frozen=True)
class EvaluatedAsrRun:
    """One persisted run paired with optional ASR metrics and real-time factor."""

    manifest: RunManifest
    metrics: AsrMetrics | None
    rtf: float | None


@dataclass(frozen=True)
class Forecast:
    """Linear runtime forecast with a symmetric planning range."""

    duration_minutes: int
    seconds: float
    low_seconds: float
    high_seconds: float


@dataclass(frozen=True)
class Ranking:
    """Default and optional quality-profile recommendation with auditable reasons."""

    default_winner: str
    quality_profile: str | None
    reasons: tuple[str, ...]


def validate_run_manifest(manifest: RunManifest) -> None:
    """Reject incoherent status, timing, and resource evidence before persistence or ranking."""
    stage_times = tuple(manifest.stage_times_s.values())
    total_time = sum(stage_times)
    if (
        manifest.status not in {"completed", "failed"}
        or any(not isfinite(value) or value < 0.0 for value in stage_times)
        or not isfinite(total_time)
        or not isfinite(manifest.peak_rss_mb)
        or manifest.peak_rss_mb < 0.0
    ):
        raise ValueError("invalid benchmark run: status, timing, or memory")
    if manifest.status == "completed":
        if manifest.error is not None or total_time <= 0.0:
            raise ValueError("invalid benchmark run: completed state")
    elif not manifest.error:
        raise ValueError("invalid benchmark run: failed state")


def validate_asr_metrics(metrics: AsrMetrics) -> None:
    """Reject non-finite, negative, or internally impossible ASR quality values."""
    counts = (
        metrics.substitutions,
        metrics.deletions,
        metrics.insertions,
        metrics.reference_words,
        metrics.hypothesis_words,
        metrics.term_hits,
        metrics.term_total,
        metrics.boundary_errors,
    )
    if (
        not isfinite(metrics.wer)
        or metrics.wer < 0.0
        or not isfinite(metrics.cer)
        or metrics.cer < 0.0
        or not isfinite(metrics.term_recall)
        or not 0.0 <= metrics.term_recall <= 1.0
        or any(value < 0 for value in counts)
        or metrics.term_hits > metrics.term_total
    ):
        raise ValueError("invalid ASR metrics")


def validate_evaluated_asr_run(run: EvaluatedAsrRun) -> None:
    """Enforce the completed/failed metrics and real-time-factor state machine."""
    validate_run_manifest(run.manifest)
    if run.manifest.status == "completed":
        if run.metrics is None or run.rtf is None or not isfinite(run.rtf) or run.rtf <= 0.0:
            raise ValueError("invalid benchmark run: completed metrics or RTF")
        validate_asr_metrics(run.metrics)
    elif run.metrics is not None or run.rtf is not None:
        raise ValueError("invalid benchmark run: failed metrics or RTF")

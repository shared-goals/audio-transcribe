"""Immutable data models shared by ASR benchmark modules."""

from __future__ import annotations

from dataclasses import dataclass
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

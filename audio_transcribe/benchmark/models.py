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

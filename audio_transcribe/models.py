"""Data models for the audio-transcribe pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TranscribeResult:
    """Output from the transcription stage."""

    segments: list[dict[str, Any]]
    language: str
    text: str


@dataclass
class AlignResult:
    """Output from the alignment stage."""

    segments: list[dict[str, Any]]
    language: str
    text: str
    words_total: int
    words_aligned: int


@dataclass
class DiarizeResult:
    """Output from the diarization stage."""

    segments: list[dict[str, Any]]
    language: str
    text: str
    words_total: int
    words_aligned: int
    speakers_detected: int
    speaker_transitions: int


@dataclass
class Config:
    """Pipeline configuration."""

    language: str
    model: str
    backend: str
    min_speakers: int = 2
    max_speakers: int = 6
    align_model: str | None = None


@dataclass
class HardwareInfo:
    """Machine hardware fingerprint."""

    chip: str
    cores_physical: int
    memory_gb: int
    os: str
    python: str


@dataclass
class InputInfo:
    """Input audio file characteristics."""

    file: str
    duration_s: float
    file_size_mb: float
    sample_rate: int = 16000


@dataclass
class StageStats:
    """Timing and memory stats for a single pipeline stage."""

    time_s: float
    peak_rss_mb: float


@dataclass
class QualityMetrics:
    """Quality scorecard for a transcription run."""

    segments: int
    words_total: int
    words_aligned: int
    alignment_pct: float
    speakers_detected: int
    speaker_coverage_pct: float
    speaker_transitions: int

    @property
    def grade(self) -> str:
        """Letter grade: A (>95% aligned, >90% speaker), B (>85%, >75%), C (below)."""
        if self.alignment_pct > 95 and self.speaker_coverage_pct > 90:
            return "A"
        if self.alignment_pct > 85 and self.speaker_coverage_pct > 75:
            return "B"
        return "C"


@dataclass
class RunRecord:
    """A single historical run entry for ~/.audio-transcribe/history.json."""

    id: str
    hardware: HardwareInfo
    input: InputInfo
    config: Config
    stages: dict[str, StageStats]
    quality: QualityMetrics | None
    corrections_applied: int
    total_time_s: float
    realtime_ratio: float


@dataclass
class PipelineResult:
    """Final output of the full pipeline."""

    audio_file: str
    language: str
    model: str
    segments: list[dict[str, Any]]
    processing_time_s: float
    quality: QualityMetrics | None = None
    processing: RunRecord | None = None

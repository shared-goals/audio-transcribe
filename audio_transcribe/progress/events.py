"""Typed progress events emitted by the pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineStart:
    """Emitted when the pipeline begins processing."""

    file: str
    duration_s: float
    config: dict[str, Any]


@dataclass
class StageStart:
    """Emitted when a pipeline stage begins."""

    stage: str
    eta_s: float | None


@dataclass
class StageComplete:
    """Emitted when a pipeline stage finishes."""

    stage: str
    time_s: float
    peak_rss_mb: float = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageError:
    """Emitted when a pipeline stage fails."""

    stage: str
    error: str
    time_s: float = 0.0


@dataclass
class PipelineComplete:
    """Emitted when the entire pipeline finishes."""

    total_time_s: float
    output: str
    transcript: str | None = None

"""Composite reporter — dispatch events to multiple reporters."""

from __future__ import annotations

from typing import Any

from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageError, StageStart


class CompositeReporter:
    """Dispatch events to a list of reporters."""

    def __init__(self, reporters: list[Any]) -> None:
        self._reporters = reporters

    def on_pipeline_start(self, event: PipelineStart) -> None:
        for r in self._reporters:
            r.on_pipeline_start(event)

    def on_stage_start(self, event: StageStart) -> None:
        for r in self._reporters:
            r.on_stage_start(event)

    def on_stage_complete(self, event: StageComplete) -> None:
        for r in self._reporters:
            r.on_stage_complete(event)

    def on_stage_error(self, event: StageError) -> None:
        for r in self._reporters:
            if hasattr(r, "on_stage_error"):
                r.on_stage_error(event)

    def on_pipeline_complete(self, event: PipelineComplete) -> None:
        for r in self._reporters:
            r.on_pipeline_complete(event)

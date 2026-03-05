"""JSON-lines progress reporter for machine consumers."""

from __future__ import annotations

import json
from dataclasses import asdict

from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageError, StageStart


class JsonReporter:
    """Emit pipeline progress events as JSON lines to stdout."""

    def on_pipeline_start(self, event: PipelineStart) -> None:
        """Handle pipeline start event."""
        self._emit({"event": "start", **asdict(event)})

    def on_stage_start(self, event: StageStart) -> None:
        """Handle stage start event."""
        self._emit({"event": "stage_start", **asdict(event)})

    def on_stage_complete(self, event: StageComplete) -> None:
        """Handle stage complete event."""
        self._emit({"event": "stage_complete", **asdict(event)})

    def on_stage_error(self, event: StageError) -> None:
        """Handle stage error event."""
        self._emit({"event": "stage_error", **asdict(event)})

    def on_pipeline_complete(self, event: PipelineComplete) -> None:
        """Handle pipeline complete event."""
        self._emit({"event": "complete", **asdict(event)})

    def _emit(self, data: dict[str, object]) -> None:
        """Write a JSON line to stdout."""
        print(json.dumps(data, ensure_ascii=False), flush=True)

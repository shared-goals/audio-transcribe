"""Tests for CompositeReporter."""

from unittest.mock import MagicMock

from audio_transcribe.progress.composite import CompositeReporter
from audio_transcribe.progress.events import PipelineStart


def test_composite_dispatches_to_all() -> None:
    r1, r2 = MagicMock(), MagicMock()
    composite = CompositeReporter([r1, r2])
    event = PipelineStart(file="test.wav", duration_s=10.0, config={})
    composite.on_pipeline_start(event)
    r1.on_pipeline_start.assert_called_once_with(event)
    r2.on_pipeline_start.assert_called_once_with(event)

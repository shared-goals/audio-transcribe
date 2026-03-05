"""Diarization stage: speaker identification via pyannote/WhisperX."""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def diarize(
    result: dict[str, Any], audio: Any, hf_token: str, min_speakers: int, max_speakers: int
) -> dict[str, Any]:
    """Assign speaker labels to segments using pyannote diarization."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    logger.debug("Diarizing (%d-%d speakers)", min_speakers, max_speakers)
    t = time.time()
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1", token=hf_token, device="cpu"
    )
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    logger.debug("Diarization done in %.1fs", time.time() - t)

    del diarize_model
    gc.collect()
    return result

"""Diarization stage: speaker identification via pyannote/WhisperX."""

from __future__ import annotations

import gc
import sys
import time
from typing import Any


def diarize(
    result: dict[str, Any], audio: Any, hf_token: str, min_speakers: int, max_speakers: int
) -> dict[str, Any]:
    """Assign speaker labels to segments using pyannote diarization."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    print(f"[3/3] Diarizing ({min_speakers}-{max_speakers} speakers)...", file=sys.stderr)
    t = time.time()
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1", token=hf_token, device="cpu"
    )
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(f"      Done in {time.time() - t:.1f}s", file=sys.stderr)

    del diarize_model
    gc.collect()
    return result

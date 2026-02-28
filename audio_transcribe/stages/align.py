"""Alignment stage: wav2vec2 word-level timestamps via WhisperX."""

from __future__ import annotations

import gc
import sys
import time
from typing import Any


def align(
    result: dict[str, Any], audio: Any, language: str, align_model: str | None = None
) -> dict[str, Any]:
    """Align transcription segments with word-level timestamps using WhisperX."""
    import whisperx

    label = align_model.split("/")[-1] if align_model else "default"
    print(f"[2/3] Aligning word timestamps ({label})...", file=sys.stderr)
    t = time.time()
    align_kwargs: dict[str, Any] = {"language_code": language, "device": "cpu"}
    if align_model:
        align_kwargs["model_name"] = align_model
    model_a, metadata = whisperx.load_align_model(**align_kwargs)
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device="cpu",
        return_char_alignments=False,
    )
    print(f"      Done in {time.time() - t:.1f}s", file=sys.stderr)

    del model_a
    gc.collect()
    return result

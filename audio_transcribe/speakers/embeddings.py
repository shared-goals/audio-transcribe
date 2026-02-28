"""Voice embedding extraction and comparison."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine distance between two vectors. 0 = identical, 2 = opposite."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 2.0
    return 1.0 - dot / (norm_a * norm_b)


@functools.lru_cache(maxsize=1)
def _get_model() -> Any:
    """Lazily load and cache the pyannote embedding model."""
    from pyannote.audio import Model

    return Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        token=os.environ.get("HF_TOKEN"),
    )


def extract_embedding(audio_path: str, start: float, end: float) -> NDArray[np.float32]:
    """Extract speaker embedding from an audio segment using pyannote wespeaker model."""
    from pyannote.audio import Inference
    from pyannote.core import Segment

    model = _get_model()
    inference = Inference(model, window="whole")
    segment = Segment(start, end)
    embedding = inference.crop(audio_path, segment)
    result: NDArray[np.float32] = np.array(embedding, dtype=np.float32).flatten()
    return result


def extract_speaker_embedding(
    audio_file: str, segments: list[dict[str, Any]], speaker_id: str, min_duration: float = 1.0
) -> NDArray[np.float32]:
    """Extract average embedding for a speaker from their segments.

    Returns zero vector if no segments >= min_duration seconds.
    """
    speaker_segs = [s for s in segments if s.get("speaker") == speaker_id]
    if not speaker_segs:
        logger.warning("Speaker %s has no segments, skipping embedding extraction", speaker_id)
        return np.zeros(256, dtype=np.float32)

    embeddings = []
    for seg in speaker_segs:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end - start >= min_duration:
            emb = extract_embedding(audio_file, start, end)
            embeddings.append(emb)

    if not embeddings:
        logger.warning(
            "Speaker %s has no segments >= %.1fs, skipping voice enrollment",
            speaker_id,
            min_duration,
        )
        return np.zeros(256, dtype=np.float32)

    mean_arr: NDArray[np.float32] = np.mean(embeddings, axis=0).astype(np.float32)
    return mean_arr

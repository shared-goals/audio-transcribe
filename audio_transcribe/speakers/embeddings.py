"""Voice embedding extraction and comparison."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
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


def _load_audio_ffmpeg(audio_path: str, sample_rate: int = 16000) -> dict[str, Any]:
    """Decode audio to a torch tensor dict via ffmpeg.

    Returns {'waveform': (1, samples) float32 tensor, 'sample_rate': int}.
    Uses ffmpeg directly to avoid the torchcodec/AudioDecoder incompatibility.
    """
    import torch

    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def extract_embedding(
    audio: str | dict[str, Any], start: float, end: float
) -> NDArray[np.float32]:
    """Extract speaker embedding from an audio segment using pyannote wespeaker model.

    audio: path string (decoded via ffmpeg) or preloaded {'waveform': tensor, 'sample_rate': int}.
    """
    from pyannote.audio import Inference
    from pyannote.core import Segment

    audio_input: dict[str, Any] = _load_audio_ffmpeg(audio) if isinstance(audio, str) else audio
    model = _get_model()
    inference = Inference(model, window="whole")
    segment = Segment(start, end)
    embedding = inference.crop(audio_input, segment)
    arr: NDArray[np.float32] = np.array(embedding, dtype=np.float32).flatten()
    return arr


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

    # Preload audio once for all segments to avoid repeated ffmpeg decoding
    audio_preloaded = _load_audio_ffmpeg(audio_file)

    embeddings = []
    for seg in speaker_segs:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end - start >= min_duration:
            emb = extract_embedding(audio_preloaded, start, end)
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

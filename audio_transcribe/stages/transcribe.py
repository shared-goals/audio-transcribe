"""Transcription stage: WhisperX, MLX, and MLX-VAD backends.

All backends return (result_dict, audio_array) for downstream align/diarize stages.
"""

import gc
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

logging.getLogger("whisperx").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.utilities.migration.utils").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)

# Maps whisper model size names to mlx-community HuggingFace repos (Apple Silicon MLX backend)
MLX_MODEL_MAP: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def transcribe(audio_path: str, model_size: str, language: str) -> tuple[dict[str, Any], Any]:
    """Transcribe using WhisperX (CTranslate2/CPU backend)."""
    import whisperx

    logger.debug("Transcribing with %s (int8, cpu)", model_size)
    t = time.time()
    model = whisperx.load_model(model_size, device="cpu", compute_type="int8")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)
    logger.debug("%d segments in %.1fs", len(result.get("segments", [])), time.time() - t)

    del model
    gc.collect()
    return result, audio


def _clear_mlx_cache() -> None:
    """Release MLX model singleton and free Metal GPU memory."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = None
    ModelHolder.model_path = None
    gc.collect()
    mx.metal.clear_cache()


def transcribe_mlx(audio_path: str, model_size: str, language: str) -> tuple[dict[str, Any], Any]:
    """Transcribe using mlx-whisper (Apple Silicon GPU backend)."""
    import mlx_whisper
    import whisperx  # needed for load_audio — returns float32 numpy array at 16kHz

    mlx_repo = MLX_MODEL_MAP.get(model_size)
    if mlx_repo is None:
        logger.warning(
            "'%s' not in MLX model map; using as HF repo directly. Known sizes: %s",
            model_size,
            list(MLX_MODEL_MAP.keys()),
        )
        mlx_repo = model_size

    logger.debug("Transcribing with mlx-whisper (%s)", mlx_repo)
    t = time.time()

    result: dict[str, Any] = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=mlx_repo,
        language=language if language else None,
        word_timestamps=False,  # wav2vec2 alignment step handles word timestamps
    )

    # Free MLX model (~3 GB Metal GPU memory) before align/diarize stages
    _clear_mlx_cache()

    # Load audio array separately for the align/diarize stages
    audio: Any = whisperx.load_audio(audio_path)

    logger.debug("%d segments in %.1fs", len(result.get("segments", [])), time.time() - t)
    return result, audio


def _offset_segments(segments: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    """Shift segment timestamps by offset seconds (chunk start time -> absolute time)."""
    for seg in segments:
        seg["start"] += offset
        seg["end"] += offset
    return segments


def transcribe_mlx_vad(audio_path: str, model_size: str, language: str) -> tuple[dict[str, Any], Any]:
    """Transcribe using pyannote VAD pre-segmentation + mlx-whisper decoder.

    Combines whisperx's pyannote VAD (splits audio into <=30s speech chunks)
    with mlx-whisper's fast Apple Silicon decoder for better accuracy than
    plain mlx on conversational audio.
    """
    import mlx_whisper
    import torch
    import whisperx
    from whisperx.audio import SAMPLE_RATE
    from whisperx.vads.pyannote import Pyannote, load_vad_model

    mlx_repo = MLX_MODEL_MAP.get(model_size)
    if mlx_repo is None:
        logger.warning(
            "'%s' not in MLX model map; using as HF repo directly. Known sizes: %s",
            model_size,
            list(MLX_MODEL_MAP.keys()),
        )
        mlx_repo = model_size

    logger.debug("Transcribing with mlx-whisper + VAD (%s)", mlx_repo)
    t = time.time()

    # Load audio (float32 numpy array at 16kHz)
    audio: Any = whisperx.load_audio(audio_path)

    # Run pyannote VAD to find speech regions
    vad_pipeline: Any = load_vad_model(device="cpu")
    waveform: Any = torch.from_numpy(audio).unsqueeze(0)
    vad_result: Any = vad_pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})

    # Merge speech regions into <=30s chunks
    chunks: list[dict[str, Any]] = Pyannote.merge_chunks(vad_result, chunk_size=30, onset=0.5, offset=0.363)

    # Free VAD model before mlx-whisper decoder loads
    del vad_pipeline
    gc.collect()

    # Cap Metal buffer cache to prevent unbounded growth across chunks
    import mlx.core as mx

    mx.metal.set_cache_limit(100_000_000)  # 100 MB

    # Transcribe each chunk independently with mlx-whisper
    all_segments: list[dict[str, Any]] = []
    for chunk in chunks:
        start_sample = int(chunk["start"] * SAMPLE_RATE)
        end_sample = int(chunk["end"] * SAMPLE_RATE)
        chunk_audio: Any = audio[start_sample:end_sample]

        chunk_result: dict[str, Any] = mlx_whisper.transcribe(
            chunk_audio,
            path_or_hf_repo=mlx_repo,
            language=language if language else None,
            word_timestamps=False,
            condition_on_previous_text=False,
        )

        # Offset chunk-relative timestamps to absolute positions
        all_segments.extend(_offset_segments(chunk_result.get("segments", []), chunk["start"]))

    # Free MLX model (~3 GB Metal GPU memory) before align/diarize stages
    _clear_mlx_cache()

    result: dict[str, Any] = {
        "text": " ".join(seg.get("text", "") for seg in all_segments),
        "language": language,
        "segments": all_segments,
    }

    n_seg, n_ch = len(all_segments), len(chunks)
    logger.debug("%d segments from %d VAD chunks in %.1fs", n_seg, n_ch, time.time() - t)
    return result, audio


def build_output(
    result: dict[str, Any], audio_file: str, language: str, model: str, elapsed: float
) -> dict[str, Any]:
    """Build the final JSON output from pipeline result."""
    segments = []
    for seg in result.get("segments", []):
        s: dict[str, Any] = {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
            "speaker": seg.get("speaker", "UNKNOWN"),
        }
        if "words" in seg:
            s["words"] = [
                {
                    "word": w["word"],
                    "start": round(w.get("start", 0), 3),
                    "end": round(w.get("end", 0), 3),
                    "speaker": w.get("speaker", "UNKNOWN"),
                }
                for w in seg["words"]
                if "start" in w
            ]
        segments.append(s)
    return {
        "audio_file": audio_file,
        "language": language,
        "model": model,
        "processing_time_s": round(elapsed, 1),
        "segments": segments,
    }

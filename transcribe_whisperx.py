#!/usr/bin/env python3
"""WhisperX pipeline: transcribe + align + diarize.

Optimized for Apple Silicon M4 (int8, cpu).

Usage:
    uv run transcribe_whisperx.py audio.wav -o result.json
    uv run transcribe_whisperx.py audio.wav --no-diarize -o result.json

Prerequisites:
    export HF_TOKEN=<your_huggingface_token>
    Accept licenses at:
      https://huggingface.co/pyannote/speaker-diarization-3.1
      https://huggingface.co/pyannote/speaker-diarization-community-1
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Suppress third-party noise that doesn't affect pipeline functionality
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly", category=UserWarning)
warnings.filterwarnings("ignore", message="Lightning automatically upgraded", category=UserWarning)
logging.getLogger("whisperx").setLevel(logging.WARNING)

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
    import whisperx

    print(f"[1/3] Transcribing with {model_size} (int8, cpu)...", file=sys.stderr)
    t = time.time()
    model = whisperx.load_model(model_size, device="cpu", compute_type="int8")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)
    print(f"      {len(result.get('segments', []))} segments in {time.time() - t:.1f}s", file=sys.stderr)

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
    import mlx_whisper
    import whisperx  # needed for load_audio — returns float32 numpy array at 16kHz

    mlx_repo = MLX_MODEL_MAP.get(model_size)
    if mlx_repo is None:
        print(
            f"Warning: '{model_size}' not in MLX model map; using as HF repo directly. "
            f"Requires MLX-converted weights. Known sizes: {list(MLX_MODEL_MAP.keys())}",
            file=sys.stderr,
        )
        mlx_repo = model_size

    print(f"[1/3] Transcribing with mlx-whisper ({mlx_repo})...", file=sys.stderr)
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

    print(f"      {len(result.get('segments', []))} segments in {time.time() - t:.1f}s", file=sys.stderr)
    return result, audio


def _offset_segments(segments: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    """Shift segment timestamps by offset seconds (chunk start time → absolute time)."""
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
        print(
            f"Warning: '{model_size}' not in MLX model map; using as HF repo directly. "
            f"Requires MLX-converted weights. Known sizes: {list(MLX_MODEL_MAP.keys())}",
            file=sys.stderr,
        )
        mlx_repo = model_size

    print(f"[1/3] Transcribing with mlx-whisper + VAD ({mlx_repo})...", file=sys.stderr)
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
    print(f"      {n_seg} segments from {n_ch} VAD chunks in {time.time() - t:.1f}s", file=sys.stderr)
    return result, audio


def align(result: dict[str, Any], audio: Any, language: str, align_model: str | None = None) -> dict[str, Any]:
    import whisperx

    label = align_model.split("/")[-1] if align_model else "default"
    print(f"[2/3] Aligning word timestamps ({label})...", file=sys.stderr)
    t = time.time()
    align_kwargs: dict[str, Any] = {"language_code": language, "device": "cpu"}
    if align_model:
        align_kwargs["model_name"] = align_model
    model_a, metadata = whisperx.load_align_model(**align_kwargs)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio,
        device="cpu", return_char_alignments=False,
    )
    print(f"      Done in {time.time() - t:.1f}s", file=sys.stderr)

    del model_a
    gc.collect()
    return result


def diarize(
    result: dict[str, Any], audio: Any, hf_token: str, min_speakers: int, max_speakers: int
) -> dict[str, Any]:
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    print(f"[3/3] Diarizing ({min_speakers}-{max_speakers} speakers)...", file=sys.stderr)
    t = time.time()
    diarize_model = DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", token=hf_token, device="cpu")
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(f"      Done in {time.time() - t:.1f}s", file=sys.stderr)

    del diarize_model
    gc.collect()
    return result


def build_output(result: dict[str, Any], audio_file: str, language: str, model: str, elapsed: float) -> dict[str, Any]:
    segments = []
    for seg in result.get("segments", []):
        s = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperX pipeline: transcribe + align + diarize")
    parser.add_argument("audio_file")
    parser.add_argument("-l", "--language", default="ru")
    parser.add_argument("-m", "--model", default="large-v3")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument(
        "--align-model",
        help="Alignment model HF repo (default: whisperx built-in for language). "
        "E.g. jonatasgrosman/wav2vec2-xls-r-1b-russian",
    )
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "--backend",
        choices=["whisperx", "mlx", "mlx-vad"],
        default="whisperx",
        help="'whisperx' (CTranslate2/CPU, default), 'mlx' (Apple Silicon GPU), "
        "or 'mlx-vad' (pyannote VAD + mlx-whisper; uv sync --extra mlx)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")
    skip_diarize = args.no_diarize
    if not skip_diarize and not hf_token:
        print("HF_TOKEN not set — skipping diarization (set HF_TOKEN to enable speaker labels)", file=sys.stderr)
        skip_diarize = True

    t0 = time.time()

    if args.backend == "mlx":
        result, audio = transcribe_mlx(str(audio_path), args.model, args.language)
    elif args.backend == "mlx-vad":
        result, audio = transcribe_mlx_vad(str(audio_path), args.model, args.language)
    else:
        result, audio = transcribe(str(audio_path), args.model, args.language)

    # Use auto-detected language if user didn't force one (both backends return result["language"])
    effective_language: str = result.get("language") or args.language

    if not args.no_align:
        result = align(result, audio, effective_language, args.align_model)

    if not skip_diarize:
        result = diarize(result, audio, hf_token, args.min_speakers, args.max_speakers)

    elapsed = time.time() - t0
    output = build_output(result, str(audio_path), effective_language, args.model, elapsed)
    print(f"\nDone in {elapsed:.1f}s — {len(output['segments'])} segments", file=sys.stderr)

    json_str = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Saved: {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()

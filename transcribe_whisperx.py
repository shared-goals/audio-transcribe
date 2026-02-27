#!/usr/bin/env python3
"""WhisperX pipeline: transcribe + align + diarize.

Optimized for Apple Silicon M4 (int8, cpu).

Usage:
    uv run transcribe_whisperx.py audio.wav -o result.json
    uv run transcribe_whisperx.py audio.wav --no-diarize -o result.json

Prerequisites:
    export HF_TOKEN=<your_huggingface_token>
    Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


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


def align(result: dict[str, Any], audio: Any, language: str) -> dict[str, Any]:
    import whisperx

    print("[2/3] Aligning word timestamps...", file=sys.stderr)
    t = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=language, device="cpu")
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
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device="cpu")
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
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument("-o", "--output")
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
    result, audio = transcribe(str(audio_path), args.model, args.language)

    if not args.no_align:
        result = align(result, audio, args.language)

    if not skip_diarize:
        result = diarize(result, audio, hf_token, args.min_speakers, args.max_speakers)

    elapsed = time.time() - t0
    output = build_output(result, str(audio_path), args.language, args.model, elapsed)
    print(f"\nDone in {elapsed:.1f}s — {len(output['segments'])} segments", file=sys.stderr)

    json_str = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Saved: {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()

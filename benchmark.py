#!/usr/bin/env python3
"""Benchmark WhisperX pipeline stages on M4.

Measures wall time and peak RSS memory per stage.

Usage:
    uv run benchmark.py audio.wav
    uv run benchmark.py audio.wav --stages transcribe align   # skip diarize
"""

import argparse
import gc
import logging
import os
import resource
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Suppress third-party noise that doesn't affect pipeline functionality
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly", category=UserWarning)
warnings.filterwarnings("ignore", message="Lightning automatically upgraded", category=UserWarning)
logging.getLogger("whisperx").setLevel(logging.WARNING)


def rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns kilobytes
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def bench_transcribe(audio_path: str, model_size: str, language: str) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import whisperx

    gc.collect()
    before = rss_mb()
    t0 = time.time()
    model = whisperx.load_model(model_size, device="cpu", compute_type="int8")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)
    elapsed = time.time() - t0
    after = rss_mb()
    del model
    gc.collect()

    return result, audio, {
        "stage": "transcribe",
        "time_s": round(elapsed, 2),
        "peak_rss_mb": round(after, 0),
        "delta_rss_mb": round(after - before, 0),
        "segments": len(result.get("segments", [])),
    }


def bench_align(
    result: dict[str, Any], audio: Any, language: str, align_model: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    import whisperx

    gc.collect()
    before = rss_mb()
    t0 = time.time()
    align_kwargs: dict[str, Any] = {"language_code": language, "device": "cpu"}
    if align_model:
        align_kwargs["model_name"] = align_model
    model_a, metadata = whisperx.load_align_model(**align_kwargs)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio,
        device="cpu", return_char_alignments=False,
    )
    elapsed = time.time() - t0
    after = rss_mb()
    del model_a
    gc.collect()

    align_label = f"align ({align_model.split('/')[-1]})" if align_model else "align (default)"
    return result, {
        "stage": align_label,
        "time_s": round(elapsed, 2),
        "peak_rss_mb": round(after, 0),
        "delta_rss_mb": round(after - before, 0),
    }


def bench_diarize(
    result: dict[str, Any], audio: Any, hf_token: str, min_speakers: int, max_speakers: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    gc.collect()
    before = rss_mb()
    t0 = time.time()
    diarize_model = DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", token=hf_token, device="cpu")
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    elapsed = time.time() - t0
    after = rss_mb()
    del diarize_model
    gc.collect()

    return result, {
        "stage": "diarize",
        "time_s": round(elapsed, 2),
        "peak_rss_mb": round(after, 0),
        "delta_rss_mb": round(after - before, 0),
    }


def print_results(stats: list[dict[str, Any]], audio_duration_s: float | None) -> None:
    print("\n## Benchmark Results\n")
    col = max(12, max((len(s["stage"]) for s in stats), default=12))
    header = f"| {'Stage':<{col}} | Time (s) | Peak RSS (MB) | Delta RSS (MB) |"
    sep = f"|{'-' * (col + 2)}|----------|---------------|----------------|"
    print(header)
    print(sep)
    for s in stats:
        print(f"| {s['stage']:<{col}} | {s['time_s']:>8.2f} | {s['peak_rss_mb']:>13.0f} | {s['delta_rss_mb']:>14.0f} |")
    total = sum(s["time_s"] for s in stats)
    print(f"| {'**Total**':<{col}} | {total:>8.2f} | {'':>13} | {'':>14} |")
    if audio_duration_s:
        ratio = total / audio_duration_s
        print(f"\nAudio:      {audio_duration_s:.0f}s ({audio_duration_s / 60:.1f} min)")
        print(f"Processing: {total:.1f}s  →  {ratio:.2f}x realtime")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark WhisperX pipeline on M4")
    parser.add_argument("audio_file")
    parser.add_argument("-l", "--language", default="ru")
    parser.add_argument("-m", "--model", default="large-v3")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument(
        "--align-model",
        help="Alignment model HF repo (default: whisperx built-in for language). "
        "E.g. jonatasgrosman/wav2vec2-xls-r-1b-russian",
    )
    parser.add_argument(
        "--stages", nargs="+",
        choices=["transcribe", "align", "diarize"],
        default=["transcribe", "align", "diarize"],
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")
    if "diarize" in args.stages and not hf_token:
        print("HF_TOKEN not set — skipping diarize benchmark", file=sys.stderr)
        args.stages = [s for s in args.stages if s != "diarize"]

    try:
        import whisperx
        audio_duration_s = len(whisperx.load_audio(str(audio_path))) / 16000
    except Exception:
        audio_duration_s = None

    print(f"Benchmarking: {audio_path.name}  |  stages: {args.stages}", file=sys.stderr)

    all_stats: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    loaded_audio: Any = None

    if "transcribe" in args.stages:
        result, loaded_audio, stats = bench_transcribe(str(audio_path), args.model, args.language)
        all_stats.append(stats)
        print(f"transcribe: {stats['time_s']}s  peak={stats['peak_rss_mb']}MB", file=sys.stderr)

    if "align" in args.stages and result is not None:
        result, stats = bench_align(result, loaded_audio, args.language, args.align_model)
        all_stats.append(stats)
        print(f"align:      {stats['time_s']}s  peak={stats['peak_rss_mb']}MB", file=sys.stderr)

    if "diarize" in args.stages and result is not None:
        result, stats = bench_diarize(result, loaded_audio, hf_token, args.min_speakers, args.max_speakers)
        all_stats.append(stats)
        print(f"diarize:    {stats['time_s']}s  peak={stats['peak_rss_mb']}MB", file=sys.stderr)

    print_results(all_stats, audio_duration_s)


if __name__ == "__main__":
    main()

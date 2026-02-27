#!/usr/bin/env python3
"""Compare Russian wav2vec2 alignment models side-by-side.

Transcribes once, aligns with both default and 1B models, and
prints a Markdown comparison table with alignment quality metrics.

Usage:
    uv run compare_align.py audio.wav
    uv run compare_align.py audio.wav -l ru -m large-v3
"""

import argparse
import gc
import logging
import resource
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress third-party noise that doesn't affect pipeline functionality
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly", category=UserWarning)
warnings.filterwarnings("ignore", message="Lightning automatically upgraded", category=UserWarning)
logging.getLogger("whisperx").setLevel(logging.WARNING)

DEFAULT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
UPGRADE_MODEL = "jonatasgrosman/wav2vec2-xls-r-1b-russian"


@dataclass
class AlignStats:
    """Alignment quality metrics for a single model run."""

    model_name: str
    total_words: int
    aligned_words: int
    missing_words: int
    aligned_pct: float
    mean_duration: float
    median_duration: float
    max_duration: float
    mean_gap: float
    median_gap: float
    max_gap: float
    overlapping_words: int
    time_s: float
    peak_rss_mb: float


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def compute_word_stats(words: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute alignment quality stats from a list of word dicts.

    Each word dict is expected to have optional 'start' and 'end' keys.
    Words without both 'start' and 'end' are counted as missing.
    """
    total = len(words)
    if total == 0:
        return {
            "total_words": 0,
            "aligned_words": 0,
            "missing_words": 0,
            "aligned_pct": 0.0,
            "durations": [],
            "gaps": [],
            "overlapping_words": 0,
        }

    aligned: list[dict[str, Any]] = [w for w in words if "start" in w and "end" in w]
    missing = total - len(aligned)

    durations = [w["end"] - w["start"] for w in aligned]

    gaps: list[float] = []
    overlapping = 0
    for i in range(1, len(aligned)):
        gap = aligned[i]["start"] - aligned[i - 1]["end"]
        gaps.append(gap)
        if gap < 0:
            overlapping += 1

    return {
        "total_words": total,
        "aligned_words": len(aligned),
        "missing_words": missing,
        "aligned_pct": (len(aligned) / total * 100) if total > 0 else 0.0,
        "durations": durations,
        "gaps": gaps,
        "overlapping_words": overlapping,
    }


def _median(values: list[float]) -> float:
    """Compute median of a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def collect_all_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all words from aligned whisperx result segments."""
    words: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        words.extend(seg.get("words", []))
    return words


def build_align_stats(
    model_name: str,
    result: dict[str, Any],
    time_s: float,
    peak_rss_mb: float,
) -> AlignStats:
    """Build AlignStats from an aligned whisperx result."""
    words = collect_all_words(result)
    stats = compute_word_stats(words)
    durations: list[float] = stats["durations"]
    gaps: list[float] = stats["gaps"]

    return AlignStats(
        model_name=model_name,
        total_words=stats["total_words"],
        aligned_words=stats["aligned_words"],
        missing_words=stats["missing_words"],
        aligned_pct=stats["aligned_pct"],
        mean_duration=_mean(durations),
        median_duration=_median(durations),
        max_duration=max(durations) if durations else 0.0,
        mean_gap=_mean(gaps),
        median_gap=_median(gaps),
        max_gap=max(gaps) if gaps else 0.0,
        overlapping_words=stats["overlapping_words"],
        time_s=time_s,
        peak_rss_mb=peak_rss_mb,
    )


def run_alignment(
    result: dict[str, Any],
    audio: Any,
    language: str,
    model_name: str | None,
) -> tuple[dict[str, Any], float, float]:
    """Run whisperx alignment and return (result, time_s, peak_rss_mb)."""
    import copy

    import whisperx

    # Deep-copy segments so each model gets a fresh input
    result_copy: dict[str, Any] = {"segments": copy.deepcopy(result["segments"])}

    gc.collect()
    before_rss = _rss_mb()
    t0 = time.time()

    align_kwargs: dict[str, Any] = {"language_code": language, "device": "cpu"}
    if model_name:
        align_kwargs["model_name"] = model_name
    model_a, metadata = whisperx.load_align_model(**align_kwargs)
    aligned = whisperx.align(
        result_copy["segments"],
        model_a,
        metadata,
        audio,
        device="cpu",
        return_char_alignments=False,
    )

    elapsed = time.time() - t0
    peak_rss = _rss_mb()

    del model_a
    gc.collect()

    return aligned, elapsed, max(peak_rss, before_rss)


def print_comparison(stats_list: list[AlignStats], sample_segments: list[tuple[str, list[dict[str, Any]]]]) -> None:
    """Print a Markdown comparison table and sample segments."""
    print("\n## Alignment Model Comparison\n")

    # Header
    labels = [s.model_name.split("/")[-1] for s in stats_list]
    col_w = max(22, max(len(lb) for lb in labels) + 2)
    metric_w = 24

    header = f"| {'Metric':<{metric_w}} |"
    sep = f"|{'-' * (metric_w + 2)}|"
    for lb in labels:
        header += f" {lb:>{col_w}} |"
        sep += f"{'-' * (col_w + 2)}|"
    print(header)
    print(sep)

    rows: list[tuple[str, list[str]]] = [
        ("Total words", [str(s.total_words) for s in stats_list]),
        ("Aligned words", [str(s.aligned_words) for s in stats_list]),
        ("Missing timestamps", [str(s.missing_words) for s in stats_list]),
        ("Aligned %", [f"{s.aligned_pct:.1f}%" for s in stats_list]),
        ("Mean word duration (s)", [f"{s.mean_duration:.3f}" for s in stats_list]),
        ("Median word duration (s)", [f"{s.median_duration:.3f}" for s in stats_list]),
        ("Max word duration (s)", [f"{s.max_duration:.3f}" for s in stats_list]),
        ("Mean inter-word gap (s)", [f"{s.mean_gap:.3f}" for s in stats_list]),
        ("Median inter-word gap (s)", [f"{s.median_gap:.3f}" for s in stats_list]),
        ("Max inter-word gap (s)", [f"{s.max_gap:.3f}" for s in stats_list]),
        ("Overlapping words", [str(s.overlapping_words) for s in stats_list]),
        ("Time (s)", [f"{s.time_s:.1f}" for s in stats_list]),
        ("Peak RSS (MB)", [f"{s.peak_rss_mb:.0f}" for s in stats_list]),
    ]

    for metric, values in rows:
        row = f"| {metric:<{metric_w}} |"
        for v in values:
            row += f" {v:>{col_w}} |"
        print(row)

    # Sample segments for manual inspection
    if sample_segments:
        print("\n## Sample Aligned Segments (first 3)\n")
        for model_label, segments in sample_segments:
            short = model_label.split("/")[-1]
            print(f"### {short}\n")
            for seg in segments[:3]:
                text = seg.get("text", "").strip()
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                n_words = len(seg.get("words", []))
                aligned_w = len([w for w in seg.get("words", []) if "start" in w and "end" in w])
                print(f"  [{start:.2f}-{end:.2f}] ({aligned_w}/{n_words} words) {text}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Russian wav2vec2 alignment models")
    parser.add_argument("audio_file")
    parser.add_argument("-l", "--language", default="ru")
    parser.add_argument("-m", "--model", default="large-v3")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    models = [DEFAULT_MODEL, UPGRADE_MODEL]

    # Step 1: transcribe once
    from transcribe_whisperx import transcribe

    result, audio = transcribe(str(audio_path), args.model, args.language)

    # Step 2: align with each model
    all_stats: list[AlignStats] = []
    sample_segments: list[tuple[str, list[dict[str, Any]]]] = []

    for model_name in models:
        short = model_name.split("/")[-1]
        print(f"\nAligning with {short}...", file=sys.stderr)
        aligned, elapsed, peak_rss = run_alignment(result, audio, args.language, model_name)
        stats = build_align_stats(model_name, aligned, elapsed, peak_rss)
        all_stats.append(stats)
        sample_segments.append((model_name, aligned.get("segments", [])[:3]))

    # Step 3: print comparison
    print_comparison(all_stats, sample_segments)


if __name__ == "__main__":
    main()

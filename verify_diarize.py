#!/usr/bin/env python3
"""Verify pyannote diarization on multi-speaker Russian meetings.

Transcribes once, aligns once, then runs diarization with multiple
min/max speaker configs and compares results side-by-side.

Usage:
    uv run verify_diarize.py audio.wav
    uv run verify_diarize.py audio.wav --configs "2-4,2-6,3-6"
    uv run verify_diarize.py audio.wav --min-speakers 2 --max-speakers 4
"""

import argparse
import copy
import gc
import logging
import os
import resource
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Suppress third-party noise that doesn't affect pipeline functionality
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly", category=UserWarning)
warnings.filterwarnings("ignore", message="Lightning automatically upgraded", category=UserWarning)
logging.getLogger("whisperx").setLevel(logging.WARNING)


@dataclass
class DiarizeConfig:
    """A single diarization speaker range configuration."""

    min_speakers: int
    max_speakers: int

    @property
    def label(self) -> str:
        return f"{self.min_speakers}-{self.max_speakers}"


@dataclass
class SpeakerStats:
    """Per-speaker breakdown metrics."""

    speaker: str
    segments: int
    words: int
    duration_s: float
    duration_pct: float
    mean_segment_duration: float
    median_segment_duration: float


@dataclass
class DiarizeStats:
    """Aggregate diarization metrics for one config run."""

    config_label: str
    speakers_detected: int
    total_segments: int
    segments_with_speaker: int
    segment_coverage_pct: float
    total_words: int
    words_with_speaker: int
    word_coverage_pct: float
    speaker_transitions: int
    time_s: float
    peak_rss_mb: float
    per_speaker: list[SpeakerStats] = field(default_factory=list)


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def parse_configs(config_str: str) -> list[DiarizeConfig]:
    """Parse comma-separated 'min-max' configs, e.g. '2-4,2-6,3-6'."""
    configs: list[DiarizeConfig] = []
    for part in config_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid config format '{part}': expected 'min-max'")
        pieces = part.split("-")
        if len(pieces) != 2:
            raise ValueError(f"Invalid config format '{part}': expected exactly one '-'")
        try:
            min_s, max_s = int(pieces[0]), int(pieces[1])
        except ValueError as err:
            raise ValueError(f"Invalid config format '{part}': min and max must be integers") from err
        if min_s < 1:
            raise ValueError(f"Invalid config '{part}': min_speakers must be >= 1")
        if max_s < min_s:
            raise ValueError(f"Invalid config '{part}': max_speakers must be >= min_speakers")
        configs.append(DiarizeConfig(min_speakers=min_s, max_speakers=max_s))
    return configs


def count_speaker_transitions(segments: list[dict[str, Any]]) -> int:
    """Count speaker changes between consecutive segments (skipping UNKNOWN)."""
    transitions = 0
    prev_speaker: str | None = None
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker == "UNKNOWN":
            continue
        if prev_speaker is not None and speaker != prev_speaker:
            transitions += 1
        prev_speaker = speaker
    return transitions


def compute_speaker_durations(segments: list[dict[str, Any]]) -> dict[str, float]:
    """Total speaking duration per speaker from segments."""
    durations: dict[str, float] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        dur = seg.get("end", 0.0) - seg.get("start", 0.0)
        durations[speaker] = durations.get(speaker, 0.0) + dur
    return durations


def compute_speaker_segment_counts(segments: list[dict[str, Any]]) -> dict[str, int]:
    """Number of segments per speaker."""
    counts: dict[str, int] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def compute_speaker_word_counts(segments: list[dict[str, Any]]) -> dict[str, int]:
    """Number of words per speaker (from segment word lists)."""
    counts: dict[str, int] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        n_words = len(seg.get("words", []))
        counts[speaker] = counts.get(speaker, 0) + n_words
    return counts


def compute_segment_durations_per_speaker(segments: list[dict[str, Any]]) -> dict[str, list[float]]:
    """List of individual segment durations per speaker."""
    durations: dict[str, list[float]] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        dur = seg.get("end", 0.0) - seg.get("start", 0.0)
        durations.setdefault(speaker, []).append(dur)
    return durations


def build_diarize_stats(
    config_label: str,
    result: dict[str, Any],
    time_s: float,
    peak_rss_mb: float,
) -> DiarizeStats:
    """Build DiarizeStats from a diarized whisperx result."""
    segments: list[dict[str, Any]] = result.get("segments", [])
    total_segments = len(segments)
    segments_with_speaker = sum(1 for s in segments if s.get("speaker", "UNKNOWN") != "UNKNOWN")

    total_words = sum(len(s.get("words", [])) for s in segments)
    words_with_speaker = sum(
        sum(1 for w in s.get("words", []) if w.get("speaker", "UNKNOWN") != "UNKNOWN") for s in segments
    )

    # Unique real speakers (exclude UNKNOWN)
    all_speakers = {s.get("speaker", "UNKNOWN") for s in segments}
    real_speakers = sorted(sp for sp in all_speakers if sp != "UNKNOWN")

    speaker_durations = compute_speaker_durations(segments)
    speaker_seg_counts = compute_speaker_segment_counts(segments)
    speaker_word_counts = compute_speaker_word_counts(segments)
    seg_durations_per = compute_segment_durations_per_speaker(segments)

    total_duration = sum(speaker_durations.values())

    per_speaker: list[SpeakerStats] = []
    for sp in real_speakers:
        dur = speaker_durations.get(sp, 0.0)
        seg_durs = seg_durations_per.get(sp, [])
        per_speaker.append(
            SpeakerStats(
                speaker=sp,
                segments=speaker_seg_counts.get(sp, 0),
                words=speaker_word_counts.get(sp, 0),
                duration_s=dur,
                duration_pct=(dur / total_duration * 100) if total_duration > 0 else 0.0,
                mean_segment_duration=statistics.mean(seg_durs) if seg_durs else 0.0,
                median_segment_duration=statistics.median(seg_durs) if seg_durs else 0.0,
            )
        )

    return DiarizeStats(
        config_label=config_label,
        speakers_detected=len(real_speakers),
        total_segments=total_segments,
        segments_with_speaker=segments_with_speaker,
        segment_coverage_pct=(segments_with_speaker / total_segments * 100) if total_segments > 0 else 0.0,
        total_words=total_words,
        words_with_speaker=words_with_speaker,
        word_coverage_pct=(words_with_speaker / total_words * 100) if total_words > 0 else 0.0,
        speaker_transitions=count_speaker_transitions(segments),
        time_s=time_s,
        peak_rss_mb=peak_rss_mb,
        per_speaker=per_speaker,
    )


def run_diarizations(
    result: dict[str, Any],
    audio: Any,
    hf_token: str,
    configs: list[DiarizeConfig],
) -> list[DiarizeStats]:
    """Load diarization model once, run each config, return stats list."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    print("Loading DiarizationPipeline...", file=sys.stderr)
    diarize_model: Any = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1", token=hf_token, device="cpu"
    )

    all_stats: list[DiarizeStats] = []
    for cfg in configs:
        print(f"\nDiarizing with {cfg.label} speakers...", file=sys.stderr)
        gc.collect()
        before_rss = _rss_mb()
        t0 = time.time()

        diarize_segments: Any = diarize_model(audio, min_speakers=cfg.min_speakers, max_speakers=cfg.max_speakers)
        result_copy: dict[str, Any] = {"segments": copy.deepcopy(result["segments"])}
        diarized: dict[str, Any] = whisperx.assign_word_speakers(diarize_segments, result_copy)

        elapsed = time.time() - t0
        peak_rss = max(_rss_mb(), before_rss)

        stats = build_diarize_stats(cfg.label, diarized, elapsed, peak_rss)
        all_stats.append(stats)
        msg = f"  {stats.speakers_detected} speakers, {stats.speaker_transitions} transitions in {elapsed:.1f}s"
        print(msg, file=sys.stderr)

    del diarize_model
    gc.collect()
    return all_stats


def print_comparison(stats_list: list[DiarizeStats]) -> None:
    """Print a Markdown comparison table."""
    print("\n## Diarization Config Comparison\n")

    labels = [s.config_label for s in stats_list]
    col_w = max(16, max(len(lb) for lb in labels) + 2)
    metric_w = 24

    header = f"| {'Metric':<{metric_w}} |"
    sep = f"|{'-' * (metric_w + 2)}|"
    for lb in labels:
        header += f" {lb:>{col_w}} |"
        sep += f"{'-' * (col_w + 2)}|"
    print(header)
    print(sep)

    rows: list[tuple[str, list[str]]] = [
        ("Speakers detected", [str(s.speakers_detected) for s in stats_list]),
        ("Total segments", [str(s.total_segments) for s in stats_list]),
        ("Segments with speaker", [str(s.segments_with_speaker) for s in stats_list]),
        ("Segment coverage %", [f"{s.segment_coverage_pct:.1f}%" for s in stats_list]),
        ("Total words", [str(s.total_words) for s in stats_list]),
        ("Words with speaker", [str(s.words_with_speaker) for s in stats_list]),
        ("Word coverage %", [f"{s.word_coverage_pct:.1f}%" for s in stats_list]),
        ("Speaker transitions", [str(s.speaker_transitions) for s in stats_list]),
        ("Time (s)", [f"{s.time_s:.1f}" for s in stats_list]),
        ("Peak RSS (MB)", [f"{s.peak_rss_mb:.0f}" for s in stats_list]),
    ]

    for metric, values in rows:
        row = f"| {metric:<{metric_w}} |"
        for v in values:
            row += f" {v:>{col_w}} |"
        print(row)


def print_speaker_breakdown(stats_list: list[DiarizeStats]) -> None:
    """Print per-speaker breakdown tables for each config."""
    for stats in stats_list:
        if not stats.per_speaker:
            continue
        print(f"\n### Speaker breakdown: {stats.config_label}\n")
        col_w = 12
        header = (
            f"| {'Speaker':<12} | {'Segments':>{col_w}} | {'Words':>{col_w}} "
            f"| {'Duration (s)':>{col_w}} | {'Duration %':>{col_w}} "
            f"| {'Mean seg (s)':>{col_w}} | {'Med seg (s)':>{col_w}} |"
        )
        sep = (
            f"|{'-' * 14}|{'-' * (col_w + 2)}|{'-' * (col_w + 2)}"
            f"|{'-' * (col_w + 2)}|{'-' * (col_w + 2)}"
            f"|{'-' * (col_w + 2)}|{'-' * (col_w + 2)}|"
        )
        print(header)
        print(sep)
        for sp in stats.per_speaker:
            print(
                f"| {sp.speaker:<12} | {sp.segments:>{col_w}} | {sp.words:>{col_w}} "
                f"| {sp.duration_s:>{col_w}.1f} | {sp.duration_pct:>{col_w}.1f}% "
                f"| {sp.mean_segment_duration:>{col_w}.2f} | {sp.median_segment_duration:>{col_w}.2f} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify pyannote diarization on multi-speaker audio")
    parser.add_argument("audio_file")
    parser.add_argument("-l", "--language", default="ru")
    parser.add_argument("-m", "--model", default="large-v3")
    parser.add_argument(
        "--align-model",
        help="Alignment model HF repo (default: whisperx built-in for language)",
    )
    parser.add_argument(
        "--configs",
        default="2-4,2-6,3-6",
        help="Comma-separated min-max speaker configs (default: 2-4,2-6,3-6)",
    )
    parser.add_argument("--min-speakers", type=int, help="Single config: min speakers (overrides --configs)")
    parser.add_argument("--max-speakers", type=int, help="Single config: max speakers (overrides --configs)")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Error: HF_TOKEN not set (required for diarization)", file=sys.stderr)
        sys.exit(1)

    # Build config list
    if args.min_speakers is not None and args.max_speakers is not None:
        configs = [DiarizeConfig(min_speakers=args.min_speakers, max_speakers=args.max_speakers)]
    elif args.min_speakers is not None or args.max_speakers is not None:
        parser.error("--min-speakers and --max-speakers must be used together")
    else:
        configs = parse_configs(args.configs)

    if not configs:
        print("Error: no valid configs provided", file=sys.stderr)
        sys.exit(1)

    # Step 1: transcribe
    from transcribe_whisperx import align, transcribe

    result, audio = transcribe(str(audio_path), args.model, args.language)
    effective_language: str = result.get("language") or args.language

    # Step 2: align
    result = align(result, audio, effective_language, args.align_model)

    # Step 3: run diarization with each config
    all_stats = run_diarizations(result, audio, hf_token, configs)

    # Step 4: print results
    print_comparison(all_stats)
    print_speaker_breakdown(all_stats)


if __name__ == "__main__":
    main()

"""Isolated execution for configurable MLX-VAD benchmark candidates."""

from __future__ import annotations

import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from audio_transcribe.benchmark.models import AsrCandidate, AsrRunResult
from audio_transcribe.stages.align import align
from audio_transcribe.stages.preprocess import preprocess
from audio_transcribe.stages.transcribe import transcribe_mlx_vad


def _peak_rss_mb() -> float:
    """Return process peak RSS in MB using the repository's platform conversion."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return float(usage.ru_maxrss) / (1024 * 1024)
    return float(usage.ru_maxrss) / 1024


def run_asr_candidate(audio: Path, candidate: AsrCandidate, align_model: str | None) -> AsrRunResult:
    """Run one ASR candidate without allowing its failure to escape."""
    stage_times_s: dict[str, float] = {}
    vad_boundaries: tuple[float, ...] = ()
    with tempfile.NamedTemporaryFile(
        prefix="audio-transcribe-benchmark-",
        suffix=".16k.wav",
        delete=False,
    ) as temp_audio:
        normalized_audio = Path(temp_audio.name)

    try:
        try:
            started = time.perf_counter()
            preprocess(str(audio), str(normalized_audio))
        finally:
            stage_times_s["preprocess"] = time.perf_counter() - started

        try:
            started = time.perf_counter()
            transcribed, audio_array = transcribe_mlx_vad(
                str(normalized_audio),
                candidate.model,
                "ru",
                candidate.options,
            )
        finally:
            stage_times_s["transcribe"] = time.perf_counter() - started

        raw_boundaries: Any = transcribed.pop("_benchmark_vad_boundaries", ())
        vad_boundaries = tuple(float(boundary) for boundary in raw_boundaries)
        try:
            started = time.perf_counter()
            aligned = align(transcribed, audio_array, "ru", align_model)
        finally:
            stage_times_s["align"] = time.perf_counter() - started

        return AsrRunResult(
            candidate=candidate,
            result=aligned,
            stage_times_s=stage_times_s,
            peak_rss_mb=_peak_rss_mb(),
            vad_boundaries=vad_boundaries,
            status="completed",
            error=None,
        )
    except ModuleNotFoundError as exc:
        dependency = exc.name or "unknown"
        return AsrRunResult(
            candidate=candidate,
            result=None,
            stage_times_s=stage_times_s,
            peak_rss_mb=_peak_rss_mb(),
            vad_boundaries=vad_boundaries,
            status="failed",
            error=f"Missing optional ML dependency '{dependency}'. Run `uv sync --extra ml`.",
            failure_kind="missing_optional_dependency",
        )
    except Exception as exc:
        return AsrRunResult(
            candidate=candidate,
            result=None,
            stage_times_s=stage_times_s,
            peak_rss_mb=_peak_rss_mb(),
            vad_boundaries=vad_boundaries,
            status="failed",
            error=str(exc),
        )
    finally:
        normalized_audio.unlink(missing_ok=True)

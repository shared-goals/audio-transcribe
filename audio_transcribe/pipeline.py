"""Pipeline orchestrator — wires stages together and emits progress events."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_transcribe.models import Config, InputInfo, RunRecord, StageStats
from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageStart
from audio_transcribe.stages.correct import apply_corrections, load_corrections
from audio_transcribe.stages.format import format_transcript
from audio_transcribe.stages.preprocess import preprocess as preprocess_stage
from audio_transcribe.stages.transcribe import (
    build_output as build_output_stage,
)
from audio_transcribe.stages.transcribe import (
    transcribe as _transcribe_whisperx,
)
from audio_transcribe.stages.transcribe import (
    transcribe_mlx as _transcribe_mlx,
)
from audio_transcribe.stages.transcribe import (
    transcribe_mlx_vad as _transcribe_mlx_vad,
)

# Stage function aliases for easy mocking in tests
preprocess_stage = preprocess_stage
format_stage = format_transcript
build_output_stage = build_output_stage


def transcribe_stage(
    audio_path: str, model_size: str, language: str, backend: str
) -> tuple[dict[str, Any], Any]:
    """Dispatch to the correct transcription backend."""
    if backend == "mlx":
        return _transcribe_mlx(audio_path, model_size, language)
    elif backend == "mlx-vad":
        return _transcribe_mlx_vad(audio_path, model_size, language)
    else:
        return _transcribe_whisperx(audio_path, model_size, language)


def align_stage(
    result: dict[str, Any], audio: Any, language: str, align_model: str | None = None
) -> dict[str, Any]:
    """Run alignment stage."""
    from audio_transcribe.stages.align import align

    return align(result, audio, language, align_model)


def diarize_stage(
    result: dict[str, Any], audio: Any, hf_token: str, min_speakers: int, max_speakers: int
) -> dict[str, Any]:
    """Run diarization stage."""
    from audio_transcribe.stages.diarize import diarize

    return diarize(result, audio, hf_token, min_speakers, max_speakers)


def _current_rss_mb() -> float:
    """Get current process peak RSS in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    audio_file: str
    language: str = "ru"
    model: str = "large-v3"
    backend: str = "whisperx"
    min_speakers: int = 2
    max_speakers: int = 6
    align_model: str | None = None
    skip_align: bool = False
    skip_diarize: bool = False
    output: str | None = None
    transcript_output: str | None = None
    corrections_path: str | None = None
    suppress_stdout_json: bool = False  # Don't print JSON to stdout (when output handled externally)


class Pipeline:
    """Orchestrate the transcription pipeline with progress events."""

    def __init__(
        self,
        reporter: Any,
        stats_store: Any | None = None,
        estimator_history: list[Any] | None = None,
    ) -> None:
        self.reporter = reporter
        self.stats_store = stats_store
        self.estimator_history = estimator_history or []
        self._stage_stats: dict[str, StageStats] = {}
        self._corrections_applied: int = 0

    def run(self, config: PipelineConfig) -> dict[str, Any]:
        """Execute the full pipeline."""
        t0 = time.time()

        # Emit pipeline start
        cfg_dict = {"model": config.model, "backend": config.backend}
        self.reporter.on_pipeline_start(
            PipelineStart(file=config.audio_file, duration_s=0.0, config=cfg_dict)
        )

        # Stage 1: Preprocess
        clean_path = self._run_stage(
            "preprocess",
            lambda: preprocess_stage(config.audio_file),
        )

        # Stage 2: Transcribe
        result, audio = self._run_stage(
            "transcribe",
            lambda: transcribe_stage(clean_path, config.model, config.language, config.backend),
        )

        # Use auto-detected language if available
        effective_language: str = result.get("language") or config.language

        # Stage 3: Align (optional)
        if not config.skip_align:
            result = self._run_stage(
                "align",
                lambda: align_stage(result, audio, effective_language, config.align_model),
            )

        # Stage 4: Diarize (optional)
        if not config.skip_diarize:
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                result = self._run_stage(
                    "diarize",
                    lambda: diarize_stage(result, audio, hf_token, config.min_speakers, config.max_speakers),
                )

        # Stage 5: Corrections (optional)
        corrections_path = config.corrections_path or str(
            Path.home() / ".audio-transcribe" / "corrections.yaml"
        )
        corrections = load_corrections(corrections_path)
        if corrections["substitutions"] or corrections["patterns"]:
            segments, count = self._run_stage(
                "correct",
                lambda: apply_corrections(result.get("segments", []), corrections),
            )
            result["segments"] = segments
            self._corrections_applied = count

        # Stage 6: Build output
        elapsed = time.time() - t0
        output = self._run_stage(
            "format",
            lambda: build_output_stage(result, config.audio_file, effective_language, config.model, elapsed),
        )

        # Write JSON output
        if config.output:
            json_str = json.dumps(output, ensure_ascii=False, indent=2)
            Path(config.output).write_text(json_str, encoding="utf-8")

        # Stage 7: Format transcript (optional)
        transcript_md: str | None = None
        if config.transcript_output:
            transcript_md = self._run_stage(
                "transcript",
                lambda: format_stage(output),
            )
            Path(config.transcript_output).write_text(transcript_md, encoding="utf-8")

        # Emit pipeline complete
        self.reporter.on_pipeline_complete(
            PipelineComplete(
                total_time_s=round(time.time() - t0, 1),
                output=config.output or "<stdout>",
                transcript=config.transcript_output,
            )
        )

        if self.stats_store is not None:
            self._persist_stats(config, output, effective_language, time.time() - t0)

        # Print JSON to stdout if no output file specified and not suppressed
        if not config.output and not config.suppress_stdout_json:
            print(json.dumps(output, ensure_ascii=False, indent=2))

        result_dict: dict[str, Any] = output
        return result_dict

    def _run_stage(self, name: str, fn: Any) -> Any:
        """Run a stage with timing and event emission."""
        self.reporter.on_stage_start(StageStart(stage=name, eta_s=None))
        t = time.time()
        result = fn()
        elapsed = time.time() - t
        self.reporter.on_stage_complete(
            StageComplete(stage=name, time_s=round(elapsed, 1), peak_rss_mb=round(_current_rss_mb(), 0))
        )
        self._stage_stats[name] = StageStats(time_s=round(elapsed, 1), peak_rss_mb=round(_current_rss_mb(), 0))
        return result

    def _persist_stats(self, config: PipelineConfig, output: dict[str, Any], language: str, elapsed: float) -> None:
        """Best-effort persistence of run statistics."""
        try:
            from datetime import datetime

            from audio_transcribe.quality.scorecard import compute_quality
            from audio_transcribe.stages.format import compute_duration
            from audio_transcribe.stats.hardware import detect_hardware

            segments = output.get("segments", [])
            duration_s = compute_duration(segments)

            record = RunRecord(
                id=datetime.now().isoformat(),
                hardware=detect_hardware(),
                input=InputInfo(
                    file=config.audio_file,
                    duration_s=duration_s,
                    file_size_mb=(
                        Path(config.audio_file).stat().st_size / 1_048_576
                        if Path(config.audio_file).exists()
                        else 0.0
                    ),
                ),
                config=Config(
                    language=language,
                    model=config.model,
                    backend=config.backend,
                    min_speakers=config.min_speakers,
                    max_speakers=config.max_speakers,
                    align_model=config.align_model,
                ),
                stages=self._stage_stats,
                quality=compute_quality(segments),
                corrections_applied=self._corrections_applied,
                total_time_s=round(elapsed, 1),
                realtime_ratio=round(elapsed / duration_s, 2) if duration_s > 0 else 0.0,
            )
            assert self.stats_store is not None  # caller checks before calling
            self.stats_store.append(record)
        except Exception:
            pass  # Stats are best-effort — never crash the pipeline


def run_pipeline(
    audio_file: str,
    language: str = "ru",
    model: str = "large-v3",
    backend: str = "whisperx",
    min_speakers: int = 2,
    max_speakers: int = 6,
    align_model: str | None = None,
    no_align: bool = False,
    no_diarize: bool = False,
    corrections_path: str | None = None,
    reporter: Any = None,
    stats_store: Any = None,
) -> dict[str, Any]:
    """Run pipeline and return output dict. Output file handling is the caller's responsibility."""
    config = PipelineConfig(
        audio_file=audio_file,
        language=language,
        model=model,
        backend=backend,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        align_model=align_model,
        skip_align=no_align,
        skip_diarize=no_diarize,
        corrections_path=corrections_path,
        suppress_stdout_json=True,
    )
    p = Pipeline(reporter=reporter, stats_store=stats_store)
    return p.run(config)

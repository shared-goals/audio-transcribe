"""Pipeline orchestrator — wires stages together and emits progress events."""

from __future__ import annotations

import json
import logging
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_transcribe.checkpoints import CheckpointStore
from audio_transcribe.config import Backend
from audio_transcribe.errors import PipelineError
from audio_transcribe.models import Config, InputInfo, RunRecord, StageStats
from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageError, StageStart
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
from audio_transcribe.util import atomic_write_text

logger = logging.getLogger(__name__)

# Stage function aliases for easy mocking in tests
preprocess_stage = preprocess_stage
format_stage = format_transcript
build_output_stage = build_output_stage


def transcribe_stage(audio_path: str, model_size: str, language: str, backend: str) -> tuple[dict[str, Any], Any]:
    """Dispatch to the correct transcription backend."""
    selected = Backend(backend)
    if selected is Backend.MLX:
        return _transcribe_mlx(audio_path, model_size, language)
    if selected is Backend.MLX_VAD:
        return _transcribe_mlx_vad(audio_path, model_size, language)
    return _transcribe_whisperx(audio_path, model_size, language)


def load_audio_stage(audio_path: str) -> Any:
    """Reload normalized audio when resuming after transcription."""
    import whisperx

    return whisperx.load_audio(audio_path)


def align_stage(result: dict[str, Any], audio: Any, language: str, align_model: str | None = None) -> dict[str, Any]:
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


def _probe_duration(audio_file: str) -> float:
    """Get audio duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_file,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Could not probe audio duration for %s", audio_file)
        return 0.0


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
    resume: bool = False
    force: bool = False
    restart_from: str | None = None
    keep_workdir: bool = False
    state_dir: str | None = None


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
        self._audio_duration_s: float = 0.0
        self._backend: str = ""
        self._checkpoints: CheckpointStore | None = None

    def run(self, config: PipelineConfig) -> dict[str, Any]:
        """Execute the full pipeline."""
        t0 = time.time()

        try:
            Backend(config.backend)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        if config.min_speakers < 1 or config.max_speakers < config.min_speakers:
            raise PipelineError("speaker range must satisfy 1 <= min_speakers <= max_speakers")

        # A Pipeline instance may be reused; per-run state must not leak.
        self._stage_stats = {}
        self._corrections_applied = 0
        self._checkpoints = None
        if config.resume:
            checkpoint_config = {
                "language": config.language,
                "model": config.model,
                "backend": config.backend,
                "min_speakers": config.min_speakers,
                "max_speakers": config.max_speakers,
                "align_model": config.align_model,
                "skip_align": config.skip_align,
                "skip_diarize": config.skip_diarize,
            }
            state_root = Path(config.state_dir or Path.home() / ".audio-transcribe") / "runs"
            self._checkpoints = CheckpointStore(Path(config.audio_file), checkpoint_config, state_root, config.force)
            if config.restart_from:
                self._checkpoints.invalidate_from(
                    config.restart_from,
                    ["preprocess", "transcribe", "align", "diarize", "correct", "format"],
                )

        from audio_transcribe.preflight import check as preflight_check

        preflight = preflight_check(config.audio_file, config.backend, config.skip_diarize)
        if not preflight.ok:
            raise PipelineError("\n".join(preflight.errors))

        # Get audio duration for ETA estimation and TUI display
        self._audio_duration_s = _probe_duration(config.audio_file)
        self._backend = config.backend

        # Emit pipeline start
        cfg_dict = {"model": config.model, "backend": config.backend}
        self.reporter.on_pipeline_start(
            PipelineStart(file=config.audio_file, duration_s=self._audio_duration_s, config=cfg_dict)
        )

        try:
            result = self._run_stages(config, t0)
            if self._checkpoints:
                self._checkpoints.finish(config.output)
            return result
        except Exception as exc:
            if self._checkpoints:
                self._checkpoints.fail(str(exc))
            raise
        except KeyboardInterrupt:
            # Ensure TUI Live display is stopped so terminal is restored
            if self._checkpoints:
                self._checkpoints.fail("interrupted")
            if hasattr(self.reporter, "_live") and self.reporter._live:
                self.reporter._live.stop()
            raise

    def _run_stages(self, config: PipelineConfig, t0: float) -> dict[str, Any]:
        """Execute all pipeline stages."""

        # Stage 1: Preprocess
        cached_clean = self._load_checkpoint("preprocess")
        if isinstance(cached_clean, str) and Path(cached_clean).exists():
            clean_path = self._resume_stage("preprocess", cached_clean)
        else:
            workspace_output = str(self._checkpoints.root / "audio.16k.wav") if self._checkpoints else None
            clean_path = self._run_stage(
                "preprocess",
                lambda: preprocess_stage(config.audio_file, workspace_output),
            )
            self._save_checkpoint("preprocess", clean_path)

        # Stage 2: Transcribe
        cached_transcribe = self._load_checkpoint("transcribe")
        if isinstance(cached_transcribe, dict):
            result = self._resume_stage("transcribe", cached_transcribe)
            audio = load_audio_stage(clean_path)
        else:
            result, audio = self._run_stage(
                "transcribe",
                lambda: transcribe_stage(clean_path, config.model, config.language, config.backend),
            )
            self._save_checkpoint("transcribe", result)

        # Use auto-detected language if available
        effective_language: str = result.get("language") or config.language

        # Stage 3: Align (optional)
        if not config.skip_align:
            cached_align = self._load_checkpoint("align")
            if isinstance(cached_align, dict):
                result = self._resume_stage("align", cached_align)
            else:
                result = self._run_stage(
                    "align",
                    lambda: align_stage(result, audio, effective_language, config.align_model),
                )
                self._save_checkpoint("align", result)

        # Stage 4: Diarize (optional)
        if not config.skip_diarize:
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                cached_diarize = self._load_checkpoint("diarize")
                if isinstance(cached_diarize, dict):
                    result = self._resume_stage("diarize", cached_diarize)
                else:
                    result = self._run_stage(
                        "diarize",
                        lambda: diarize_stage(result, audio, hf_token, config.min_speakers, config.max_speakers),
                    )
                    self._save_checkpoint("diarize", result)
            else:
                self.reporter.on_stage_start(StageStart(stage="diarize", eta_s=None))
                self.reporter.on_stage_complete(
                    StageComplete(stage="diarize", time_s=0.0, extra={"skipped": "HF_TOKEN not set"})
                )

        # Stage 5: Corrections (optional)
        corrections_path = config.corrections_path or str(Path.home() / ".audio-transcribe" / "corrections.yaml")
        corrections = load_corrections(corrections_path, effective_language)
        if corrections["substitutions"] or corrections["patterns"]:
            segments, count = self._run_stage(
                "correct",
                lambda: apply_corrections(result.get("segments", []), corrections),
            )
            result["segments"] = segments
            self._corrections_applied = count
            self._save_checkpoint("correct", {"segments": segments, "count": count})

        # Stage 6: Build output
        elapsed = time.time() - t0
        output = self._run_stage(
            "format",
            lambda: build_output_stage(result, config.audio_file, effective_language, config.model, elapsed),
        )
        self._save_checkpoint("format", output)

        # Write JSON output
        if config.output:
            json_str = json.dumps(output, ensure_ascii=False, indent=2)
            atomic_write_text(Path(config.output), json_str)

        # Stage 7: Format transcript (optional)
        transcript_md: str | None = None
        if config.transcript_output:
            transcript_md = self._run_stage(
                "transcript",
                lambda: format_stage(output),
            )
            atomic_write_text(Path(config.transcript_output), transcript_md)

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
        if self._checkpoints and not config.keep_workdir:
            clean = Path(clean_path).resolve()
            if clean.is_relative_to(self._checkpoints.root.resolve()):
                clean.unlink(missing_ok=True)
        return result_dict

    def _load_checkpoint(self, stage: str) -> Any | None:
        return self._checkpoints.load(stage) if self._checkpoints else None

    def _save_checkpoint(self, stage: str, value: Any) -> None:
        if self._checkpoints:
            elapsed = self._stage_stats.get(stage, StageStats(0.0, 0.0)).time_s
            self._checkpoints.complete_stage(stage, value, elapsed)

    def _resume_stage(self, stage: str, value: Any) -> Any:
        self.reporter.on_stage_start(StageStart(stage=stage, eta_s=0.0))
        self.reporter.on_stage_complete(StageComplete(stage=stage, time_s=0.0, extra={"resumed": True}))
        self._stage_stats[stage] = StageStats(time_s=0.0, peak_rss_mb=round(_current_rss_mb(), 0))
        return value

    def _estimate_eta(self, stage: str) -> float | None:
        """Estimate stage ETA from history, or None if insufficient data."""
        if not self.estimator_history or self._audio_duration_s <= 0:
            return None
        from audio_transcribe.stats.estimator import estimate_stage

        est = estimate_stage(stage, self._audio_duration_s, self.estimator_history, backend=self._backend)
        return est.eta_s if est else None

    def _run_stage(self, name: str, fn: Any) -> Any:
        """Run a stage with timing, event emission, and error wrapping."""
        self.reporter.on_stage_start(StageStart(stage=name, eta_s=self._estimate_eta(name)))
        t = time.time()
        try:
            result = fn()
        except Exception as e:
            elapsed = time.time() - t
            if hasattr(self.reporter, "on_stage_error"):
                self.reporter.on_stage_error(StageError(stage=name, error=str(e), time_s=round(elapsed, 1)))
            raise PipelineError(f"{name} failed: {e}", stage=name, elapsed_s=elapsed) from e
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
                        Path(config.audio_file).stat().st_size / 1_048_576 if Path(config.audio_file).exists() else 0.0
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
            logger.debug("Could not persist pipeline statistics", exc_info=True)


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
    estimator_history: list[Any] | None = None,
    resume: bool = False,
    force: bool = False,
    restart_from: str | None = None,
    keep_workdir: bool = False,
    state_dir: str | None = None,
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
        resume=resume,
        force=force,
        restart_from=restart_from,
        keep_workdir=keep_workdir,
        state_dir=state_dir,
    )
    p = Pipeline(reporter=reporter, stats_store=stats_store, estimator_history=estimator_history)
    return p.run(config)

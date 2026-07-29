"""Deterministic ASR benchmark ranking, forecasts, and aggregate reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any

from audio_transcribe.benchmark.models import (
    EvaluatedAsrRun,
    EvaluatedDiarizationRun,
    Forecast,
    Ranking,
    validate_diarization_metrics,
    validate_evaluated_asr_run,
)
from audio_transcribe.util import atomic_write_text

_FORECAST_DURATIONS = (30, 60, 120)
_PLANNING_RANGE = 0.15
_DEFAULT_RUNTIME_LIMIT = 2.0


def forecast(rtf: float, duration_minutes: int) -> Forecast:
    """Forecast processing time from a measured real-time factor."""
    if not isfinite(rtf) or rtf <= 0.0:
        raise ValueError("real-time factor must be finite and positive")
    if duration_minutes <= 0:
        raise ValueError("forecast duration must be positive")
    seconds = rtf * duration_minutes * 60
    low_seconds = seconds * (1.0 - _PLANNING_RANGE)
    high_seconds = seconds * (1.0 + _PLANNING_RANGE)
    if not all(isfinite(value) for value in (seconds, low_seconds, high_seconds)):
        raise ValueError("forecast seconds must be finite")
    return Forecast(
        duration_minutes=duration_minutes,
        seconds=seconds,
        low_seconds=low_seconds,
        high_seconds=high_seconds,
    )


def _total_time(run: EvaluatedAsrRun) -> float:
    return sum(run.manifest.stage_times_s.values())


def _successful(run: EvaluatedAsrRun) -> bool:
    return run.manifest.status == "completed" and run.metrics is not None and run.rtf is not None


def _safety_eligible(run: EvaluatedAsrRun, baseline: EvaluatedAsrRun) -> bool:
    if run.metrics is None or baseline.metrics is None:
        return False
    return len(run.metrics.hallucinated_phrases) <= len(baseline.metrics.hallucinated_phrases) and len(
        run.metrics.missed_phrases
    ) <= len(baseline.metrics.missed_phrases)


def rank_asr(results: tuple[EvaluatedAsrRun, ...], baseline_name: str) -> Ranking:
    """Rank successful ASR runs while enforcing safety and the two-times default gate."""
    candidate_names = tuple(run.manifest.candidate_name for run in results)
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("duplicate benchmark candidate name")
    for run in results:
        validate_evaluated_asr_run(run)

    baseline = next(
        (run for run in results if run.manifest.candidate_name == baseline_name and _successful(run)),
        None,
    )
    if baseline is None:
        raise ValueError("successful baseline run is required")
    baseline_total = _total_time(baseline)
    if baseline_total <= 0.0:
        raise ValueError("baseline runtime must be positive")

    successful = tuple(run for run in results if _successful(run))
    eligibility = {run.manifest.candidate_name: _safety_eligible(run, baseline) for run in successful}

    def ranking_key(run: EvaluatedAsrRun) -> tuple[bool, float, float, float, float, str]:
        assert run.metrics is not None
        return (
            not eligibility[run.manifest.candidate_name],
            run.metrics.wer,
            -run.metrics.term_recall,
            run.metrics.cer,
            _total_time(run),
            run.manifest.candidate_name,
        )

    ordered = tuple(sorted(successful, key=ranking_key))
    default = next(
        run
        for run in ordered
        if eligibility[run.manifest.candidate_name] and _total_time(run) <= baseline_total * _DEFAULT_RUNTIME_LIMIT
    )
    quality = next(run for run in ordered if eligibility[run.manifest.candidate_name])
    quality_profile = (
        quality.manifest.candidate_name if quality.manifest.candidate_name != default.manifest.candidate_name else None
    )
    reasons = [
        (
            f"{default.manifest.candidate_name} is the best safety-eligible candidate within "
            f"the {_DEFAULT_RUNTIME_LIMIT:.2f}x baseline runtime limit."
        )
    ]
    if quality_profile is not None:
        reasons.append(
            f"{quality_profile} has the best eligible quality but exceeds the "
            f"{_DEFAULT_RUNTIME_LIMIT:.2f}x default runtime limit."
        )
    reasons.extend(
        (
            f"{run.manifest.candidate_name} was excluded by safety gating because it increased "
            "missed or hallucinated phrases."
        )
        for run in ordered
        if not eligibility[run.manifest.candidate_name]
    )
    failed_names = tuple(run.manifest.candidate_name for run in results if not _successful(run))
    if failed_names:
        reasons.append(f"Failed or incomplete runs were excluded from ranking: {', '.join(failed_names)}.")
    return Ranking(default.manifest.candidate_name, quality_profile, tuple(reasons))


def _result_record(
    run: EvaluatedAsrRun,
    baseline_total: float,
) -> dict[str, Any]:
    successful = _successful(run)
    total_time = _total_time(run)
    forecasts = (
        [asdict(forecast(run.rtf, duration)) for duration in _FORECAST_DURATIONS]
        if successful and run.rtf is not None
        else []
    )
    return {
        "candidate": run.manifest.candidate_name,
        "manifest": asdict(run.manifest),
        "metrics": asdict(run.metrics) if run.metrics is not None else None,
        "total_time_s": total_time,
        "rtf": run.rtf,
        "baseline_ratio": total_time / baseline_total if successful else None,
        "forecasts": forecasts,
        "peak_rss_mb": run.manifest.peak_rss_mb,
        "status": run.manifest.status,
        "error": run.manifest.error,
    }


def _format_number(value: float | int | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.3f}{suffix}"


def _format_forecast(value: Forecast | None) -> str:
    if value is None:
        return "—"
    return f"{value.seconds:.1f} s ({value.low_seconds:.1f}–{value.high_seconds:.1f})"


def _markdown_cell(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>").replace("|", "\\|")


def _report_row(run: EvaluatedAsrRun, baseline_total: float) -> str:
    metrics = run.metrics
    successful = _successful(run)
    total_time = _total_time(run)
    forecasts = (
        {duration: forecast(run.rtf, duration) for duration in _FORECAST_DURATIONS}
        if successful and run.rtf is not None
        else {}
    )
    status = run.manifest.status
    if run.manifest.error:
        status = f"{status}: {run.manifest.error}"
    values = (
        run.manifest.candidate_name,
        _format_number(metrics.wer if metrics else None),
        _format_number(metrics.cer if metrics else None),
        _format_number(metrics.term_recall if metrics else None),
        str(metrics.substitutions) if metrics else "—",
        str(metrics.deletions) if metrics else "—",
        str(metrics.insertions) if metrics else "—",
        str(len(metrics.missed_phrases)) if metrics else "—",
        str(len(metrics.hallucinated_phrases)) if metrics else "—",
        str(metrics.boundary_errors) if metrics else "—",
        _format_number(total_time, " s"),
        _format_number(total_time / baseline_total if successful else None, "x"),
        _format_forecast(forecasts.get(30)),
        _format_forecast(forecasts.get(60)),
        _format_forecast(forecasts.get(120)),
        _format_number(run.manifest.peak_rss_mb, " MB"),
        status,
    )
    return "| " + " | ".join(_markdown_cell(value) for value in values) + " |"


def _markdown_report(
    results: tuple[EvaluatedAsrRun, ...],
    baseline_total: float,
    ranking: Ranking,
) -> str:
    header = (
        "| Candidate | WER | CER | Term recall | Substitutions | Deletions | Insertions | Missed phrases | "
        "Hallucinated phrases | Boundary errors | Runtime | Baseline ratio | 30 min forecast | 60 min forecast | "
        "120 min forecast | Peak RAM | Status |"
    )
    separator = "| " + " | ".join("---" for _ in range(17)) + " |"
    lines = [
        "# ASR benchmark report",
        "",
        header,
        separator,
        *(_report_row(run, baseline_total) for run in results),
        "",
        "## Recommendation",
        "",
        f"Default winner: **{ranking.default_winner}**.",
    ]
    if ranking.quality_profile is not None:
        lines.extend(("", f"Optional quality profile: **{ranking.quality_profile}**."))
    lines.extend(("", *(f"- {reason}" for reason in ranking.reasons)))
    lines.extend(
        (
            "",
            "Forecasts are linear planning estimates with a ±15% range; fixed model-loading overhead "
            "and other non-linear runtime behavior can change actual processing time.",
            "",
        )
    )
    return "\n".join(lines)


def _validate_diarization_run(run: EvaluatedDiarizationRun) -> None:
    if (
        run.status not in {"completed", "failed"}
        or not isfinite(run.time_s)
        or run.time_s < 0.0
        or not isfinite(run.peak_rss_mb)
        or run.peak_rss_mb < 0.0
    ):
        raise ValueError("invalid diarization run")
    if run.status == "completed":
        if run.error is not None or run.metrics is None or run.time_s <= 0.0:
            raise ValueError("invalid completed diarization run")
        validate_diarization_metrics(run.metrics)
    elif not run.error or run.metrics is not None:
        raise ValueError("invalid failed diarization run")


def _rank_diarization(
    results: tuple[EvaluatedDiarizationRun, ...],
    combined_times: dict[str, float],
    baseline_combined: float,
) -> Ranking:
    names = tuple(run.candidate_name for run in results)
    if len(names) != len(set(names)):
        raise ValueError("duplicate diarization candidate name")
    for run in results:
        _validate_diarization_run(run)
    completed = tuple(run for run in results if run.status == "completed" and run.metrics is not None)
    baseline = next((run for run in completed if run.candidate_name == "pyannote-3.1"), None)
    if baseline is None or baseline.metrics is None:
        raise ValueError("successful pyannote-3.1 baseline is required")

    def eligible(run: EvaluatedDiarizationRun) -> bool:
        assert run.metrics is not None
        assert baseline.metrics is not None
        return (
            run.metrics.missed_speech_s <= baseline.metrics.missed_speech_s
            and run.metrics.false_alarm_s <= baseline.metrics.false_alarm_s
        )

    def key(run: EvaluatedDiarizationRun) -> tuple[bool, float, float, float, int, float, str]:
        assert run.metrics is not None
        return (
            not eligible(run),
            run.metrics.der,
            run.metrics.confusion_s,
            -run.metrics.word_speaker_accuracy,
            run.metrics.fragmented_turns,
            combined_times[run.candidate_name],
            run.candidate_name,
        )

    ordered = tuple(sorted(completed, key=key))
    default = next(
        run
        for run in ordered
        if eligible(run) and combined_times[run.candidate_name] <= baseline_combined * _DEFAULT_RUNTIME_LIMIT
    )
    quality = next(run for run in ordered if eligible(run))
    optional = quality.candidate_name if quality.candidate_name != default.candidate_name else None
    reasons = (
        f"{default.candidate_name} is the best safety-eligible diarization candidate within the "
        "2.00x combined baseline.",
    )
    return Ranking(default.candidate_name, optional, reasons)


def _combined_report(
    asr_results: tuple[EvaluatedAsrRun, ...],
    diarization_results: tuple[EvaluatedDiarizationRun, ...],
    selected_asr_name: str,
    audio_duration_s: float,
) -> tuple[dict[str, Any], str]:
    if not isfinite(audio_duration_s) or audio_duration_s <= 0.0:
        raise ValueError("audio duration must be finite and positive")
    selected = next(
        (run for run in asr_results if run.manifest.candidate_name == selected_asr_name and _successful(run)),
        None,
    )
    if selected is None:
        raise ValueError("successful selected ASR run is required")
    original_asr = next(
        (run for run in asr_results if run.manifest.candidate_name == "baseline" and _successful(run)),
        None,
    )
    if original_asr is None:
        raise ValueError("successful original ASR baseline is required")
    selected_asr_time = _total_time(selected)
    combined_times = {run.candidate_name: selected_asr_time + run.time_s for run in diarization_results}
    baseline_diarization = next(
        (run for run in diarization_results if run.candidate_name == "pyannote-3.1" and run.status == "completed"),
        None,
    )
    if baseline_diarization is None:
        raise ValueError("successful pyannote-3.1 baseline is required")
    baseline_combined = _total_time(original_asr) + baseline_diarization.time_s
    ranking = _rank_diarization(diarization_results, combined_times, baseline_combined)
    records: list[dict[str, Any]] = []
    rows: list[str] = []
    for run in diarization_results:
        successful = run.status == "completed" and run.metrics is not None
        combined = combined_times[run.candidate_name]
        rtf = combined / audio_duration_s if successful else None
        forecasts = [asdict(forecast(rtf, duration)) for duration in _FORECAST_DURATIONS] if rtf else []
        records.append(
            {
                "candidate": run.candidate_name,
                "source_sha256": run.source_sha256,
                "aligned_asr_sha256": run.aligned_asr_sha256,
                "speaker_reference_sha256": run.speaker_reference_sha256,
                "model": run.model,
                "exclusive": run.exclusive,
                "metrics": asdict(run.metrics) if run.metrics else None,
                "diarization_time_s": run.time_s,
                "combined_time_s": combined,
                "combined_baseline_ratio": combined / baseline_combined if successful else None,
                "forecasts": forecasts,
                "peak_rss_mb": run.peak_rss_mb,
                "status": run.status,
                "error": run.error,
            }
        )
        metrics = run.metrics
        by_duration = {item["duration_minutes"]: Forecast(**item) for item in forecasts}
        status = run.status if not run.error else f"{run.status}: {run.error}"
        values = (
            run.candidate_name,
            _format_number(metrics.der if metrics else None),
            _format_number(metrics.confusion_s if metrics else None, " s"),
            _format_number(metrics.word_speaker_accuracy if metrics else None),
            str(metrics.fragmented_turns) if metrics else "—",
            _format_number(run.time_s, " s"),
            _format_number(combined, " s"),
            _format_number(combined / baseline_combined if successful else None, "x"),
            _format_forecast(by_duration.get(30)),
            _format_forecast(by_duration.get(60)),
            _format_forecast(by_duration.get(120)),
            _format_number(run.peak_rss_mb, " MB"),
            status,
        )
        rows.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    header = (
        "| Candidate | DER | Confusion | Word speaker accuracy | Fragmented turns | Diarization runtime | "
        "Combined runtime | Combined baseline ratio | 30 min forecast | 60 min forecast | 120 min forecast | "
        "Peak RAM | Status |"
    )
    markdown = "\n".join(
        (
            "# Combined ASR and diarization benchmark report",
            "",
            f"Selected ASR: **{selected_asr_name}**.",
            "",
            header,
            "| " + " | ".join("---" for _ in range(13)) + " |",
            *rows,
            "",
            "## Recommendation",
            "",
            f"Default diarization winner: **{ranking.default_winner}**.",
            "",
            *(f"- {reason}" for reason in ranking.reasons),
            "",
            "Forecasts include ASR, alignment, and diarization and are linear planning estimates with a ±15% range.",
            "",
        )
    )
    return {"diarization_ranking": asdict(ranking), "diarization_results": records}, markdown


def write_report(
    dataset: Path,
    results: tuple[EvaluatedAsrRun, ...],
    *,
    diarization_results: tuple[EvaluatedDiarizationRun, ...] | None = None,
    selected_asr_name: str | None = None,
    audio_duration_s: float | None = None,
) -> tuple[Path, Path]:
    """Atomically write machine-readable metrics and a human-readable comparison."""
    ranking = rank_asr(results, "baseline")
    baseline = next(run for run in results if run.manifest.candidate_name == "baseline" and _successful(run))
    baseline_total = _total_time(baseline)
    aggregate = {
        "forecast_durations_minutes": list(_FORECAST_DURATIONS),
        "ranking": asdict(ranking),
        "results": [_result_record(run, baseline_total) for run in results],
    }
    expanded_dataset = dataset.expanduser().resolve()
    metrics_path = expanded_dataset / "metrics.json"
    report_path = expanded_dataset / "report.md"
    report_content = _markdown_report(results, baseline_total, ranking)
    if diarization_results is not None:
        if selected_asr_name is None or audio_duration_s is None:
            raise ValueError("combined report requires selected ASR and audio duration")
        combined, report_content = _combined_report(results, diarization_results, selected_asr_name, audio_duration_s)
        aggregate.update(combined)
    metrics_content = json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write_text(metrics_path, metrics_content)
    atomic_write_text(report_path, report_content)
    return metrics_path, report_path

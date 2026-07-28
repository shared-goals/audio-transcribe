"""Deterministic ASR benchmark ranking, forecasts, and aggregate reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audio_transcribe.benchmark.models import EvaluatedAsrRun, Forecast, Ranking
from audio_transcribe.util import atomic_write_text

_FORECAST_DURATIONS = (30, 60, 120)
_PLANNING_RANGE = 0.15
_DEFAULT_RUNTIME_LIMIT = 2.0


def forecast(rtf: float, duration_minutes: int) -> Forecast:
    """Forecast processing time from a measured real-time factor."""
    if rtf < 0.0:
        raise ValueError("real-time factor must not be negative")
    if duration_minutes <= 0:
        raise ValueError("forecast duration must be positive")
    seconds = rtf * duration_minutes * 60
    return Forecast(
        duration_minutes=duration_minutes,
        seconds=seconds,
        low_seconds=seconds * (1.0 - _PLANNING_RANGE),
        high_seconds=seconds * (1.0 + _PLANNING_RANGE),
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
    return "| " + " | ".join(values) + " |"


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


def write_report(dataset: Path, results: tuple[EvaluatedAsrRun, ...]) -> tuple[Path, Path]:
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
    atomic_write_text(metrics_path, json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(report_path, _markdown_report(results, baseline_total, ranking))
    return metrics_path, report_path

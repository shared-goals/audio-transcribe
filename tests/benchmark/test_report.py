"""Tests for ASR benchmark forecasts, ranking, and aggregate reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_transcribe.benchmark.models import AsrMetrics, EvaluatedAsrRun, RunManifest
from audio_transcribe.benchmark.report import forecast, rank_asr, write_report


def _metrics(
    *,
    wer: float,
    cer: float = 0.2,
    term_recall: float = 0.8,
    missed: tuple[str, ...] = (),
    hallucinated: tuple[str, ...] = (),
) -> AsrMetrics:
    return AsrMetrics(
        wer=wer,
        cer=cer,
        substitutions=1,
        deletions=2,
        insertions=3,
        reference_words=10,
        hypothesis_words=10,
        term_hits=4,
        term_total=5,
        term_recall=term_recall,
        hallucinated_phrases=hallucinated,
        missed_phrases=missed,
        boundary_errors=1,
    )


def _run(
    name: str,
    *,
    total: float,
    wer: float,
    rtf: float = 0.5,
    cer: float = 0.2,
    term_recall: float = 0.8,
    missed: tuple[str, ...] = (),
    hallucinated: tuple[str, ...] = (),
) -> EvaluatedAsrRun:
    manifest = RunManifest(
        candidate_name=name,
        source_sha256="audio-sha",
        reference_sha256="reference-sha",
        git_commit="abc123",
        model=f"example/{name}",
        model_revision=None,
        options={"chunk_size_s": 30},
        hardware={"chip": "Apple M4"},
        stage_times_s={"transcribe": total},
        peak_rss_mb=512.0,
        status="completed",
        error=None,
    )
    return EvaluatedAsrRun(
        manifest,
        _metrics(
            wer=wer,
            cer=cer,
            term_recall=term_recall,
            missed=missed,
            hallucinated=hallucinated,
        ),
        rtf,
    )


def _failed_run(name: str) -> EvaluatedAsrRun:
    return EvaluatedAsrRun(
        RunManifest(
            candidate_name=name,
            source_sha256="audio-sha",
            reference_sha256="reference-sha",
            git_commit="abc123",
            model=f"example/{name}",
            model_revision=None,
            options={"chunk_size_s": 30},
            hardware={"chip": "Apple M4"},
            stage_times_s={"transcribe": 3.0},
            peak_rss_mb=128.0,
            status="failed",
            error="model unavailable",
        ),
        None,
        None,
    )


def test_forecast_includes_planning_range() -> None:
    """The two-hour estimate must expose the agreed symmetric 15% planning range."""
    value = forecast(0.5, 120)

    assert value.duration_minutes == 120
    assert value.seconds == 3600.0
    assert value.low_seconds == 3060.0
    assert value.high_seconds == 4140.0


def test_rank_rejects_candidate_over_two_times_baseline() -> None:
    """A slower quality leader must remain optional rather than become the default."""
    ranking = rank_asr(
        (
            _run("baseline", total=100.0, wer=0.3),
            _run("candidate", total=201.0, wer=0.1),
        ),
        "baseline",
    )

    assert ranking.default_winner == "baseline"
    assert ranking.quality_profile == "candidate"
    assert any("2.00" in reason for reason in ranking.reasons)


@pytest.mark.parametrize(
    ("missed", "hallucinated"),
    (
        (("пропуск",), ()),
        ((), ("галлюцинация",)),
    ),
)
def test_rank_rejects_candidate_with_worse_safety_signals(
    missed: tuple[str, ...],
    hallucinated: tuple[str, ...],
) -> None:
    """Lower WER cannot compensate for increasing either missed or hallucinated speech."""
    ranking = rank_asr(
        (
            _run("baseline", total=100.0, wer=0.3),
            _run(
                "unsafe",
                total=90.0,
                wer=0.1,
                missed=missed,
                hallucinated=hallucinated,
            ),
        ),
        "baseline",
    )

    assert ranking.default_winner == "baseline"
    assert ranking.quality_profile is None
    assert any("unsafe" in reason and "safety" in reason for reason in ranking.reasons)


def test_rank_uses_term_recall_then_cer_then_runtime_as_tie_breakers() -> None:
    """The specified deterministic ordering must resolve equal-WER candidates."""
    ranking = rank_asr(
        (
            _run("baseline", total=100.0, wer=0.3),
            _run("lower-recall", total=80.0, wer=0.2, term_recall=0.7, cer=0.1),
            _run("higher-cer", total=70.0, wer=0.2, term_recall=0.9, cer=0.2),
            _run("winner", total=90.0, wer=0.2, term_recall=0.9, cer=0.1),
        ),
        "baseline",
    )

    assert ranking.default_winner == "winner"


def test_rank_excludes_failed_runs() -> None:
    """A failed candidate must not participate even when it has no quality metrics."""
    ranking = rank_asr((_run("baseline", total=100.0, wer=0.3), _failed_run("failed")), "baseline")

    assert ranking.default_winner == "baseline"
    assert ranking.quality_profile is None


def test_write_report_preserves_failures_and_required_comparison_fields(tmp_path: Path) -> None:
    """Machine and Markdown reports must retain failures and every required comparison dimension."""
    results = (
        _run("baseline", total=100.0, wer=0.3),
        _run("candidate", total=150.0, wer=0.2, rtf=0.75),
        _failed_run("failed"),
    )

    metrics_path, report_path = write_report(tmp_path, results)

    assert metrics_path == tmp_path / "metrics.json"
    assert report_path == tmp_path / "report.md"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["ranking"]["default_winner"] == "candidate"
    assert metrics["forecast_durations_minutes"] == [30, 60, 120]
    failed = next(value for value in metrics["results"] if value["candidate"] == "failed")
    assert failed["status"] == "failed"
    assert failed["error"] == "model unavailable"
    assert failed["forecasts"] == []

    report = report_path.read_text(encoding="utf-8")
    for heading in (
        "Candidate",
        "WER",
        "CER",
        "Term recall",
        "Substitutions",
        "Deletions",
        "Insertions",
        "Missed phrases",
        "Hallucinated phrases",
        "Boundary errors",
        "Runtime",
        "Baseline ratio",
        "30 min forecast",
        "60 min forecast",
        "120 min forecast",
        "Peak RAM",
        "Status",
    ):
        assert heading in report
    assert "Default winner: **candidate**" in report
    assert "failed: model unavailable" in report
    assert "±15%" in report

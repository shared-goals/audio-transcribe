"""Tests for ASR benchmark forecasts, ranking, and aggregate reports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from audio_transcribe.benchmark.models import (
    AsrMetrics,
    DiarizationMetrics,
    EvaluatedAsrRun,
    EvaluatedDiarizationRun,
    RunManifest,
)
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
        vocabulary_sha256="vocabulary-sha",
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
            vocabulary_sha256="vocabulary-sha",
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


def test_write_report_combines_diarization_quality_runtime_and_forecasts(tmp_path: Path) -> None:
    asr = (_run("baseline", total=100.0, wer=0.3, rtf=0.5),)
    diarization = (
        EvaluatedDiarizationRun(
            candidate_name="pyannote-3.1",
            source_sha256="audio-sha",
            aligned_asr_sha256="aligned-sha",
            speaker_reference_sha256="speaker-sha",
            model="pyannote/speaker-diarization-3.1",
            exclusive=False,
            time_s=20.0,
            peak_rss_mb=700.0,
            status="completed",
            error=None,
            metrics=DiarizationMetrics(0.2, 1.0, 2.0, 3.0, 0.9, 4, {"P0": "A", "P1": "B"}),
        ),
        EvaluatedDiarizationRun(
            candidate_name="community-1",
            source_sha256="audio-sha",
            aligned_asr_sha256="aligned-sha",
            speaker_reference_sha256="speaker-sha",
            model="pyannote/speaker-diarization-community-1",
            exclusive=False,
            time_s=10.0,
            peak_rss_mb=600.0,
            status="completed",
            error=None,
            metrics=DiarizationMetrics(0.1, 0.5, 0.5, 1.0, 0.95, 2, {"P0": "A", "P1": "B"}),
        ),
    )

    metrics_path, report_path = write_report(
        tmp_path,
        asr,
        diarization_results=diarization,
        selected_asr_name="baseline",
        audio_duration_s=200.0,
    )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["diarization_ranking"]["default_winner"] == "community-1"
    candidate = next(item for item in payload["diarization_results"] if item["candidate"] == "community-1")
    assert candidate["combined_time_s"] == 110.0
    assert candidate["combined_baseline_ratio"] == pytest.approx(110 / 120)
    assert [item["duration_minutes"] for item in candidate["forecasts"]] == [30, 60, 120]
    report = report_path.read_text(encoding="utf-8")
    for heading in (
        "DER",
        "Confusion",
        "Word speaker accuracy",
        "Fragmented turns",
        "Diarization runtime",
        "Combined baseline ratio",
        "30 min forecast",
        "60 min forecast",
        "120 min forecast",
    ):
        assert heading in report


@pytest.mark.parametrize(
    "metrics",
    (
        DiarizationMetrics(float("nan"), 0.0, 0.0, 0.0, 1.0, 0, {}),
        DiarizationMetrics(float("inf"), 0.0, 0.0, 0.0, 1.0, 0, {}),
        DiarizationMetrics(-0.1, 0.0, 0.0, 0.0, 1.0, 0, {}),
        DiarizationMetrics(0.1, -0.1, 0.0, 0.0, 1.0, 0, {}),
        DiarizationMetrics(0.1, 0.0, float("inf"), 0.0, 1.0, 0, {}),
        DiarizationMetrics(0.1, 0.0, 0.0, -0.1, 1.0, 0, {}),
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, -0.1, 0, {}),
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, 1.1, 0, {}),
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, 1.0, -1, {}),
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, 1.0, True, {}),
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, 1.0, 0, {"": "A"}),
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, 1.0, 0, {"P0": ""}),
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, 1.0, 0, {1: "A"}),  # type: ignore[dict-item]
        DiarizationMetrics(0.1, 0.0, 0.0, 0.0, 1.0, 0, {"P0": 1}),  # type: ignore[dict-item]
    ),
)
def test_combined_report_rejects_invalid_diarization_metrics(
    tmp_path: Path,
    metrics: DiarizationMetrics,
) -> None:
    run = EvaluatedDiarizationRun(
        candidate_name="pyannote-3.1",
        source_sha256="audio-sha",
        aligned_asr_sha256="aligned-sha",
        speaker_reference_sha256="speaker-sha",
        model="pyannote/speaker-diarization-3.1",
        exclusive=False,
        time_s=20.0,
        peak_rss_mb=700.0,
        status="completed",
        error=None,
        metrics=metrics,
    )

    with pytest.raises(ValueError, match="invalid diarization metrics"):
        write_report(
            tmp_path,
            (_run("baseline", total=100.0, wer=0.3, rtf=0.5),),
            diarization_results=(run,),
            selected_asr_name="baseline",
            audio_duration_s=200.0,
        )


@pytest.mark.parametrize("rtf", (0.0, -0.1, float("nan"), float("inf"), float("-inf")))
def test_forecast_rejects_non_finite_or_non_positive_rtf(rtf: float) -> None:
    """Forecasts must never emit invalid, zero, or non-finite processing times."""
    with pytest.raises(ValueError, match="real-time factor must be finite and positive"):
        forecast(rtf, 120)


def test_forecast_rejects_finite_rtf_that_overflows_duration() -> None:
    """A finite input must still be rejected if the projected seconds overflow."""
    with pytest.raises(ValueError, match="forecast seconds must be finite"):
        forecast(1e308, 120)


@pytest.mark.parametrize(
    "results",
    (
        (
            _run("baseline", total=100.0, wer=0.3),
            replace(
                _run("baseline", total=90.0, wer=0.2),
                manifest=replace(
                    _run("baseline", total=90.0, wer=0.2).manifest,
                    error="also invalid",
                ),
            ),
        ),
        (
            _run("baseline", total=100.0, wer=0.3),
            _run("candidate", total=90.0, wer=0.2),
            _run("candidate", total=80.0, wer=0.1),
        ),
    ),
)
def test_rank_rejects_duplicate_candidate_names(results: tuple[EvaluatedAsrRun, ...]) -> None:
    """Duplicate names, including duplicate baselines, must fail before ranking state is keyed by name."""
    with pytest.raises(ValueError, match="duplicate benchmark candidate name"):
        rank_asr(results, "baseline")


@pytest.mark.parametrize(
    "run",
    (
        replace(
            _run("baseline", total=100.0, wer=0.3),
            manifest=replace(
                _run("baseline", total=100.0, wer=0.3).manifest,
                error="completed with an error",
            ),
        ),
        replace(_run("baseline", total=100.0, wer=0.3), metrics=None),
        replace(_run("baseline", total=100.0, wer=0.3), rtf=None),
        replace(
            _failed_run("baseline"),
            manifest=replace(_failed_run("baseline").manifest, error=None),
        ),
        replace(_failed_run("baseline"), metrics=_metrics(wer=0.1)),
        replace(_failed_run("baseline"), rtf=0.5),
    ),
)
def test_rank_rejects_inconsistent_status_error_metrics_and_rtf(run: EvaluatedAsrRun) -> None:
    """Malformed completed and failed states must be rejected rather than ranked or silently ignored."""
    with pytest.raises(ValueError, match="invalid benchmark run"):
        rank_asr((run,), "baseline")


@pytest.mark.parametrize(
    "run",
    (
        replace(_run("baseline", total=100.0, wer=0.3), rtf=0.0),
        replace(_run("baseline", total=100.0, wer=0.3), rtf=float("nan")),
        replace(
            _run("baseline", total=100.0, wer=0.3),
            manifest=replace(
                _run("baseline", total=100.0, wer=0.3).manifest,
                stage_times_s={"transcribe": float("inf")},
            ),
        ),
        replace(
            _run("baseline", total=100.0, wer=0.3),
            manifest=replace(
                _run("baseline", total=100.0, wer=0.3).manifest,
                stage_times_s={"transcribe": -1.0, "align": 101.0},
            ),
        ),
    ),
)
def test_rank_rejects_invalid_rtf_or_runtime(run: EvaluatedAsrRun) -> None:
    """Ranking requires finite positive RTF and finite non-negative stages with positive total runtime."""
    with pytest.raises(ValueError, match="invalid benchmark run"):
        rank_asr((run,), "baseline")


@pytest.mark.parametrize(
    "metrics",
    (
        replace(_metrics(wer=0.3), wer=float("nan")),
        replace(_metrics(wer=0.3), wer=-0.1),
        replace(_metrics(wer=0.3), cer=float("inf")),
        replace(_metrics(wer=0.3), cer=-0.1),
        replace(_metrics(wer=0.3), term_recall=float("nan")),
        replace(_metrics(wer=0.3), term_recall=-0.1),
        replace(_metrics(wer=0.3), term_recall=1.1),
        replace(_metrics(wer=0.3), deletions=-1),
        replace(_metrics(wer=0.3), term_hits=6),
    ),
)
def test_rank_rejects_invalid_quality_values(metrics: AsrMetrics) -> None:
    """Error rates must be finite/non-negative, recall bounded, and metric counts internally valid."""
    run = replace(_run("baseline", total=100.0, wer=0.3), metrics=metrics)

    with pytest.raises(ValueError, match="invalid ASR metrics"):
        rank_asr((run,), "baseline")


def test_write_report_rejects_nested_non_finite_json_before_writing(tmp_path: Path) -> None:
    """Strict JSON serialization must reject hidden non-finite option values before creating either report."""
    baseline = _run("baseline", total=100.0, wer=0.3)
    baseline = replace(
        baseline,
        manifest=replace(baseline.manifest, options={"temperature": float("nan")}),
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        write_report(tmp_path, (baseline,))

    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "report.md").exists()


def test_write_report_escapes_pipes_and_normalizes_newlines_in_cells(tmp_path: Path) -> None:
    """Candidate and error cells must stay within one Markdown row even with adversarial text."""
    failed = _failed_run("failed|candidate")
    failed = replace(
        failed,
        manifest=replace(failed.manifest, error="bad | model\r\ntry again\nlater"),
    )

    _, report_path = write_report(tmp_path, (_run("baseline", total=100.0, wer=0.3), failed))

    report_lines = report_path.read_text(encoding="utf-8").splitlines()
    failed_row = next(line for line in report_lines if "failed\\|candidate" in line)
    assert "failed: bad \\| model<br>try again<br>later" in failed_row

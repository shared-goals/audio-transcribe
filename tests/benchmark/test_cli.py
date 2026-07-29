"""Tests for the explicit private ASR benchmark command surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from audio_transcribe.benchmark.cli import _git_commit
from audio_transcribe.benchmark.models import (
    AsrCandidate,
    AsrMetrics,
    AsrRunResult,
    DiarizationCandidate,
    DiarizationMetrics,
    DiarizationRunResult,
    RunManifest,
)
from audio_transcribe.cli import app
from audio_transcribe.preflight import PreflightResult
from audio_transcribe.stages.transcribe import MlxVadOptions

runner = CliRunner()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_result_sha(dataset: Path, name: str = "baseline") -> str:
    result_path = dataset / "runs" / name / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"segments": [], "candidate": name}), encoding="utf-8")
    return _sha256(result_path)


def _accepted_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "runs").mkdir()
    audio = dataset / "meeting.mp3"
    audio.write_bytes(b"synthetic audio fixture")
    reference = dataset / "reference-transcript.md"
    reference.write_text("# Reference transcript\n\n[00:00.000 --> 00:10.000] Первый этап.\n", encoding="utf-8")
    vocabulary = dataset / "vocabulary.txt"
    vocabulary.write_text("ЕИРЦ\n", encoding="utf-8")
    (dataset / "benchmark.json").write_text(
        json.dumps(
            {
                "audio_path": str(audio),
                "audio_sha256": _sha256(audio),
                "reference_status": "accepted",
                "reference_sha256": _sha256(reference),
                "vocabulary_status": "accepted",
                "vocabulary_sha256": _sha256(vocabulary),
                "models": {"baseline": "example/baseline", "russian": "example/russian"},
            }
        ),
        encoding="utf-8",
    )
    return dataset, audio


def _manifest(dataset: Path, **overrides: object) -> RunManifest:
    benchmark = json.loads((dataset / "benchmark.json").read_text(encoding="utf-8"))
    values: dict[str, object] = {
        "candidate_name": "baseline",
        "source_sha256": benchmark["audio_sha256"],
        "reference_sha256": benchmark["reference_sha256"],
        "git_commit": "abc123",
        "model": "example/baseline",
        "model_revision": None,
        "options": asdict(MlxVadOptions()),
        "hardware": {"machine": "arm64"},
        "stage_times_s": {"transcribe": 10.0},
        "peak_rss_mb": 100.0,
        "status": "completed",
        "error": None,
        "vocabulary_sha256": benchmark["vocabulary_sha256"],
    }
    values.update(overrides)
    return RunManifest(**values)  # type: ignore[arg-type]


def _valid_metrics() -> dict[str, object]:
    return asdict(
        AsrMetrics(
            wer=0.1,
            cer=0.05,
            substitutions=1,
            deletions=0,
            insertions=0,
            reference_words=10,
            hypothesis_words=10,
            term_hits=1,
            term_total=1,
            term_recall=1.0,
            hallucinated_phrases=(),
            missed_phrases=(),
            boundary_errors=0,
        )
    )


def _invalid_metrics() -> dict[str, object]:
    metrics = _valid_metrics()
    metrics["term_hits"] = 2
    return metrics


def test_benchmark_help_lists_commands() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    for command in ("prepare", "run-asr", "evaluate-asr", "run-diarization", "evaluate-diarization", "report"):
        assert command in result.output


def _diarization_dataset(tmp_path: Path, *, status: str = "accepted") -> Path:
    dataset, _audio = _accepted_dataset(tmp_path)
    aligned = dataset / "runs" / "winner-context" / "result.json"
    aligned.parent.mkdir()
    aligned.write_text(
        json.dumps({"segments": [{"words": [{"word": "тест", "start": 0.0, "end": 1.0}]}]}),
        encoding="utf-8",
    )
    reference = dataset / "reference-speakers.json"
    reference.write_text(
        json.dumps(
            {
                "source_sha256": json.loads((dataset / "benchmark.json").read_text())["audio_sha256"],
                "aligned_asr_sha256": _sha256(aligned),
                "audio_duration_s": 2.0,
                "turns": [
                    {"start": 0.0, "end": 1.0, "speaker": "A"},
                    {"start": 1.0, "end": 2.0, "speaker": "B"},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads((dataset / "benchmark.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "speaker_reference_status": status,
            "speaker_reference_sha256": _sha256(reference),
            "aligned_asr_path": str(aligned),
            "aligned_asr_sha256": _sha256(aligned),
        }
    )
    (dataset / "benchmark.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, wer in (("baseline", 0.3), ("winner-context", 0.1)):
        run_dir = dataset / "runs" / name
        run_dir.mkdir(exist_ok=True)
        run_manifest = _manifest(dataset, candidate_name=name)
        (run_dir / "manifest.json").write_text(json.dumps(asdict(run_manifest)), encoding="utf-8")
        if name == "baseline":
            (run_dir / "result.json").write_text('{"segments":[]}', encoding="utf-8")
        metrics = _valid_metrics()
        metrics["wer"] = wer
        (run_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "source_sha256": run_manifest.source_sha256,
                    "reference_sha256": run_manifest.reference_sha256,
                    "vocabulary_sha256": run_manifest.vocabulary_sha256,
                    "result_sha256": _sha256(run_dir / "result.json"),
                    "metrics": metrics,
                    "rtf": 0.5,
                }
            ),
            encoding="utf-8",
        )
    return dataset


def test_run_diarization_rejects_draft_speaker_reference(tmp_path: Path) -> None:
    dataset = _diarization_dataset(tmp_path, status="draft")

    result = runner.invoke(
        app,
        ["benchmark", "run-diarization", "--dataset", str(dataset), "--candidate", "pyannote-3.1"],
    )

    assert result.exit_code == 1
    assert "accepted speaker reference" in result.output


def test_run_diarization_rejects_aligned_asr_checksum_mismatch_before_model(tmp_path: Path) -> None:
    dataset = _diarization_dataset(tmp_path)
    aligned = dataset / "runs" / "winner-context" / "result.json"
    aligned.write_text('{"segments":[]}', encoding="utf-8")

    with patch(
        "audio_transcribe.benchmark.cli.run_diarization_candidate",
        side_effect=AssertionError("model must not run"),
    ):
        result = runner.invoke(
            app,
            ["benchmark", "run-diarization", "--dataset", str(dataset), "--candidate", "pyannote-3.1"],
        )

    assert result.exit_code == 1
    assert "ASR result checksum mismatch" in result.output


def test_run_diarization_persists_failed_model_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _diarization_dataset(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "token")
    failed = DiarizationRunResult(
        DiarizationCandidate("community-1", "pyannote/speaker-diarization-community-1", False),
        (),
        (),
        1.5,
        100.0,
        "failed",
        "gated model",
    )

    with patch("audio_transcribe.benchmark.cli.run_diarization_candidate", return_value=failed):
        stale_evaluation = dataset / "runs" / "diarization" / "community-1" / "evaluation.json"
        stale_evaluation.parent.mkdir(parents=True)
        stale_evaluation.write_text('{"metrics":{"der":0.0}}', encoding="utf-8")
        result = runner.invoke(
            app,
            ["benchmark", "run-diarization", "--dataset", str(dataset), "--candidate", "community-1"],
        )

    assert result.exit_code == 0
    persisted = json.loads((dataset / "runs" / "diarization" / "community-1" / "result.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["error"] == "gated model"
    assert not stale_evaluation.exists()


def test_run_diarization_rejects_aligned_result_from_non_winning_asr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _diarization_dataset(tmp_path)
    manifest = json.loads((dataset / "benchmark.json").read_text(encoding="utf-8"))
    baseline_result = dataset / "runs" / "baseline" / "result.json"
    manifest["aligned_asr_path"] = str(baseline_result)
    manifest["aligned_asr_sha256"] = _sha256(baseline_result)
    reference = dataset / "reference-speakers.json"
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    reference_payload["aligned_asr_sha256"] = _sha256(baseline_result)
    reference.write_text(json.dumps(reference_payload), encoding="utf-8")
    manifest["speaker_reference_sha256"] = _sha256(reference)
    (dataset / "benchmark.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "token")

    with patch(
        "audio_transcribe.benchmark.cli.run_diarization_candidate",
        side_effect=AssertionError("model must not run"),
    ):
        result = runner.invoke(
            app,
            ["benchmark", "run-diarization", "--dataset", str(dataset), "--candidate", "pyannote-3.1"],
        )

    assert result.exit_code == 1
    assert "winning ASR candidate" in result.output


def test_report_rejects_evaluation_for_previous_diarization_result(tmp_path: Path) -> None:
    dataset = _diarization_dataset(tmp_path)
    run_dir = dataset / "runs" / "diarization" / "pyannote-3.1"
    run_dir.mkdir(parents=True)
    benchmark = json.loads((dataset / "benchmark.json").read_text(encoding="utf-8"))
    result_payload = {
        "candidate": {
            "name": "pyannote-3.1",
            "model_id": "pyannote/speaker-diarization-3.1",
            "exclusive": False,
        },
        "source_sha256": benchmark["audio_sha256"],
        "aligned_asr_sha256": benchmark["aligned_asr_sha256"],
        "speaker_reference_sha256": benchmark["speaker_reference_sha256"],
        "turns": [],
        "word_assignments": [],
        "time_s": 2.0,
        "peak_rss_mb": 100.0,
        "status": "completed",
        "error": None,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "source_sha256": benchmark["audio_sha256"],
                "aligned_asr_sha256": benchmark["aligned_asr_sha256"],
                "speaker_reference_sha256": benchmark["speaker_reference_sha256"],
                "result_sha256": "previous-result",
                "metrics": asdict(DiarizationMetrics(0.1, 0.0, 0.0, 0.1, 0.9, 0, {"P0": "A", "P1": "B"})),
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])

    assert result.exit_code == 1
    assert "diarization result checksum mismatch" in result.output


def test_evaluate_diarization_binds_metrics_to_exact_result_checksum(tmp_path: Path) -> None:
    dataset = _diarization_dataset(tmp_path)
    benchmark = json.loads((dataset / "benchmark.json").read_text(encoding="utf-8"))
    run_dir = dataset / "runs" / "diarization" / "pyannote-3.1"
    run_dir.mkdir(parents=True)
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "name": "pyannote-3.1",
                    "model_id": "pyannote/speaker-diarization-3.1",
                    "exclusive": False,
                },
                "source_sha256": benchmark["audio_sha256"],
                "aligned_asr_sha256": benchmark["aligned_asr_sha256"],
                "speaker_reference_sha256": benchmark["speaker_reference_sha256"],
                "turns": [
                    {"start_s": 0.0, "end_s": 1.0, "speaker": "P0"},
                    {"start_s": 1.0, "end_s": 2.0, "speaker": "P1"},
                ],
                "word_assignments": [{"start": 0.1, "end": 0.9, "speaker": "P0"}],
                "time_s": 2.0,
                "peak_rss_mb": 100.0,
                "status": "completed",
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "evaluate-diarization", "--dataset", str(dataset)])

    assert result.exit_code == 0
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["result_sha256"] == _sha256(result_path)


def test_report_ignores_stale_metrics_ranking_and_uses_current_asr_runs(tmp_path: Path) -> None:
    dataset = _diarization_dataset(tmp_path)
    (dataset / "metrics.json").write_text(
        json.dumps({"ranking": {"default_winner": "baseline"}}),
        encoding="utf-8",
    )
    benchmark = json.loads((dataset / "benchmark.json").read_text(encoding="utf-8"))
    run_dir = dataset / "runs" / "diarization" / "pyannote-3.1"
    run_dir.mkdir(parents=True)
    result_payload = {
        "candidate": {
            "name": "pyannote-3.1",
            "model_id": "pyannote/speaker-diarization-3.1",
            "exclusive": False,
        },
        "source_sha256": benchmark["audio_sha256"],
        "aligned_asr_sha256": benchmark["aligned_asr_sha256"],
        "speaker_reference_sha256": benchmark["speaker_reference_sha256"],
        "turns": [],
        "word_assignments": [],
        "time_s": 2.0,
        "peak_rss_mb": 100.0,
        "status": "completed",
        "error": None,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "source_sha256": benchmark["audio_sha256"],
                "aligned_asr_sha256": benchmark["aligned_asr_sha256"],
                "speaker_reference_sha256": benchmark["speaker_reference_sha256"],
                "result_sha256": _sha256(result_path),
                "metrics": asdict(DiarizationMetrics(0.1, 0.0, 0.0, 0.1, 0.9, 0, {"P0": "A", "P1": "B"})),
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])

    assert result.exit_code == 0
    assert "Selected ASR: **winner-context**" in (dataset / "report.md").read_text(encoding="utf-8")


def test_prepare_rejects_missing_audio(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("## Transcript\n\n[00:00] Тест.\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "benchmark",
            "prepare",
            str(tmp_path / "missing.mp3"),
            "--current-note",
            str(note),
            "--dataset",
            str(tmp_path / "dataset"),
        ],
    )

    assert result.exit_code != 0
    assert "audio must be an existing file" in result.output


def test_run_asr_rejects_draft_reference(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "benchmark.json").write_text(
        json.dumps({"reference_status": "draft"}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "baseline"])

    assert result.exit_code == 1
    assert "accepted reference" in result.output


def test_report_rejects_reference_checksum_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    reference = dataset / "reference-transcript.md"
    reference.write_text("# Reference transcript\n\n[00:00.000 --> 00:01.000] Тест.\n", encoding="utf-8")
    (dataset / "benchmark.json").write_text(
        json.dumps(
            {
                "reference_status": "accepted",
                "reference_sha256": hashlib.sha256(b"different").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])

    assert result.exit_code != 0
    assert "checksum mismatch" in result.output


def test_run_asr_rejects_edited_accepted_vocabulary_before_model_execution(tmp_path: Path) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    (dataset / "vocabulary.txt").write_text("изменено\n", encoding="utf-8")

    with patch(
        "audio_transcribe.benchmark.cli.run_asr_candidate",
        side_effect=AssertionError("model must not run"),
    ):
        result = runner.invoke(app, ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "baseline"])

    assert result.exit_code == 1
    assert "vocabulary checksum mismatch" in result.output


def test_winner_context_rejects_ranking_from_previous_accepted_evidence(tmp_path: Path) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    stale_manifest = asdict(
        _manifest(
            dataset,
            reference_sha256="previous-reference",
            vocabulary_sha256="previous-vocabulary",
        )
    )
    (dataset / "metrics.json").write_text(
        json.dumps(
            {
                "ranking": {"default_winner": "baseline"},
                "results": [{"candidate": "baseline", "manifest": stale_manifest}],
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "audio_transcribe.benchmark.cli.run_asr_candidate",
        side_effect=AssertionError("model must not run"),
    ):
        result = runner.invoke(
            app,
            ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "winner-context"],
        )

    assert result.exit_code == 1
    assert "winner-context evidence checksum mismatch" in result.output


@pytest.mark.parametrize(
    ("result_status", "result_error", "metrics", "manifest_overrides"),
    (
        ("failed", "model unavailable", None, {"candidate_name": "russian", "status": "failed", "error": "failed"}),
        ("completed", None, None, {}),
        ("completed", None, _valid_metrics(), {"candidate_name": "russian"}),
        ("completed", None, _invalid_metrics(), {}),
    ),
)
def test_winner_context_rejects_invalid_winner_result_before_model_launch(
    tmp_path: Path,
    result_status: str,
    result_error: str | None,
    metrics: dict[str, object] | None,
    manifest_overrides: dict[str, object],
) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    result_manifest = asdict(_manifest(dataset, **manifest_overrides))
    result_sha256 = _candidate_result_sha(dataset)
    (dataset / "metrics.json").write_text(
        json.dumps(
            {
                "ranking": {"default_winner": "baseline"},
                "results": [
                    {
                        "candidate": "baseline",
                        "manifest": result_manifest,
                        "status": result_status,
                        "error": result_error,
                        "metrics": metrics,
                        "result_sha256": result_sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "audio_transcribe.benchmark.cli.run_asr_candidate",
        side_effect=AssertionError("model must not run"),
    ):
        result = runner.invoke(
            app,
            ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "winner-context"],
        )

    assert result.exit_code == 1
    assert "winner-context winner evidence is invalid" in result.output


@pytest.mark.parametrize(
    ("result_candidate", "manifest_candidate"),
    (
        (["baseline"], "baseline"),
        ("baseline", {"name": "baseline"}),
    ),
)
def test_winner_context_rejects_malformed_candidate_identity_cleanly(
    tmp_path: Path,
    result_candidate: object,
    manifest_candidate: object,
) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    result_manifest = asdict(_manifest(dataset))
    result_manifest["candidate_name"] = manifest_candidate
    (dataset / "metrics.json").write_text(
        json.dumps(
            {
                "ranking": {"default_winner": "baseline"},
                "results": [
                    {
                        "candidate": result_candidate,
                        "manifest": result_manifest,
                        "status": "completed",
                        "error": None,
                        "metrics": _valid_metrics(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "audio_transcribe.benchmark.cli.run_asr_candidate",
        side_effect=AssertionError("model must not run"),
    ):
        result = runner.invoke(
            app,
            ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "winner-context"],
        )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "winner-context winner evidence is invalid" in result.output


def test_report_rejects_stale_run_manifest_checksums(tmp_path: Path) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    run_dir = dataset / "runs" / "baseline"
    run_dir.mkdir()
    manifest = _manifest(dataset, reference_sha256="stale-reference")
    result_sha256 = _candidate_result_sha(dataset)
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "source_sha256": manifest.source_sha256,
                "reference_sha256": manifest.reference_sha256,
                "vocabulary_sha256": "stale-vocabulary",
                "result_sha256": result_sha256,
                "metrics": None,
                "rtf": None,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])

    assert result.exit_code == 1
    assert "run checksum mismatch" in result.output


def test_report_rejects_stale_evaluation_checksums(tmp_path: Path) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    run_dir = dataset / "runs" / "baseline"
    run_dir.mkdir()
    manifest = _manifest(dataset, status="failed", error="model unavailable")
    result_sha256 = _candidate_result_sha(dataset)
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "source_sha256": manifest.source_sha256,
                "reference_sha256": manifest.reference_sha256,
                "vocabulary_sha256": "stale-vocabulary",
                "result_sha256": result_sha256,
                "metrics": None,
                "rtf": None,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])

    assert result.exit_code == 1
    assert "run checksum mismatch" in result.output


@pytest.mark.parametrize("result_sha256", (None, "previous-result"))
def test_report_rejects_unbound_or_stale_asr_result_evaluation(
    tmp_path: Path,
    result_sha256: str | None,
) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    run_dir = dataset / "runs" / "baseline"
    run_dir.mkdir()
    manifest = _manifest(dataset)
    result_path = run_dir / "result.json"
    result_path.write_text('{"segments":[{"text":"current"}]}', encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    evaluation = {
        "source_sha256": manifest.source_sha256,
        "reference_sha256": manifest.reference_sha256,
        "vocabulary_sha256": manifest.vocabulary_sha256,
        "metrics": _valid_metrics(),
        "rtf": 0.5,
    }
    if result_sha256 is not None:
        evaluation["result_sha256"] = result_sha256
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")

    result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])

    assert result.exit_code == 1
    assert "ASR result checksum mismatch" in result.output


def test_report_preserves_asr_only_datasets_without_speaker_reference(tmp_path: Path) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    run_dir = dataset / "runs" / "baseline"
    run_dir.mkdir()
    manifest = _manifest(dataset)
    result_sha256 = _candidate_result_sha(dataset)
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "source_sha256": manifest.source_sha256,
                "reference_sha256": manifest.reference_sha256,
                "vocabulary_sha256": manifest.vocabulary_sha256,
                "result_sha256": result_sha256,
                "metrics": _valid_metrics(),
                "rtf": 0.5,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])

    assert result.exit_code == 0
    assert (dataset / "report.md").read_text(encoding="utf-8").startswith("# ASR benchmark report")


def test_evaluate_asr_uses_source_audio_duration_for_rtf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    run_dir = dataset / "runs" / "baseline"
    run_dir.mkdir()
    manifest = _manifest(dataset)
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 10.0, "text": "Первый этап."}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("audio_transcribe.benchmark.cli._probe_audio_duration", lambda _audio: 100.0, raising=False)

    result = runner.invoke(app, ["benchmark", "evaluate-asr", "--dataset", str(dataset)])

    assert result.exit_code == 0
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["rtf"] == 0.1
    assert evaluation["result_sha256"] == _sha256(run_dir / "result.json")


def test_winner_context_rejects_report_after_ranked_asr_result_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    run_dir = dataset / "runs" / "baseline"
    run_dir.mkdir()
    manifest = _manifest(dataset)
    result_path = run_dir / "result.json"
    result_path.write_text('{"segments":[{"text":"ranked"}]}', encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "source_sha256": manifest.source_sha256,
                "reference_sha256": manifest.reference_sha256,
                "vocabulary_sha256": manifest.vocabulary_sha256,
                "result_sha256": _sha256(result_path),
                "metrics": _valid_metrics(),
                "rtf": 0.5,
            }
        ),
        encoding="utf-8",
    )
    report_result = runner.invoke(app, ["benchmark", "report", "--dataset", str(dataset)])
    assert report_result.exit_code == 0
    result_path.write_text('{"segments":[{"text":"mutated"}]}', encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "unused")

    with patch(
        "audio_transcribe.benchmark.cli.run_asr_candidate",
        side_effect=AssertionError("model must not run"),
    ):
        result = runner.invoke(
            app,
            ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "winner-context"],
        )

    assert result.exit_code == 1
    assert "winner-context result checksum mismatch" in result.output


@pytest.mark.parametrize("duration_s", (0.0, float("nan"), float("inf")))
def test_evaluate_asr_rejects_invalid_source_audio_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration_s: float,
) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    monkeypatch.setattr("audio_transcribe.benchmark.cli._probe_duration", lambda _audio: duration_s)

    result = runner.invoke(app, ["benchmark", "evaluate-asr", "--dataset", str(dataset)])

    assert result.exit_code == 1
    assert "source audio duration is unavailable" in result.output


def test_run_asr_stops_on_preflight_error_without_starting_model(tmp_path: Path) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)

    with (
        patch(
            "audio_transcribe.preflight.check",
            return_value=PreflightResult(errors=["mlx backend unavailable"]),
        ),
        patch(
            "audio_transcribe.benchmark.cli.run_asr_candidate",
            side_effect=AssertionError("model must not run"),
        ),
    ):
        result = runner.invoke(app, ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "baseline"])

    assert result.exit_code == 1
    assert "mlx backend unavailable" in result.output


def test_run_asr_persists_model_failure_when_invoked_outside_repository(tmp_path: Path) -> None:
    dataset, _audio = _accepted_dataset(tmp_path)
    candidate = AsrCandidate("baseline", "example/baseline", MlxVadOptions())
    failed = AsrRunResult(
        candidate=candidate,
        result=None,
        stage_times_s={"transcribe": 2.0},
        peak_rss_mb=100.0,
        vad_boundaries=(),
        status="failed",
        error="model unavailable",
    )

    with (
        patch("audio_transcribe.preflight.check", return_value=PreflightResult()),
        patch("audio_transcribe.benchmark.cli.run_asr_candidate", return_value=failed),
        patch("audio_transcribe.benchmark.cli.subprocess.run", side_effect=subprocess.CalledProcessError(128, "git")),
    ):
        result = runner.invoke(
            app,
            ["benchmark", "run-asr", "--dataset", str(dataset), "--candidate", "baseline"],
            catch_exceptions=True,
        )

    assert result.exit_code == 0
    persisted = json.loads((dataset / "runs" / "baseline" / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["error"] == "model unavailable"
    assert persisted["git_commit"] == "unavailable"


def test_git_commit_provenance_is_resolved_from_repository_not_caller_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    commit = _git_commit()

    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)

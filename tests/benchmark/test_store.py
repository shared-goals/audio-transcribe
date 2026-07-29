"""Tests for checksum-bound ASR benchmark run persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from audio_transcribe.benchmark.cli import _evaluated_runs
from audio_transcribe.benchmark.models import DiarizationCandidate, DiarizationRunResult, RunManifest
from audio_transcribe.benchmark.store import write_diarization_run, write_run
from audio_transcribe.util import atomic_write_text


def _manifest(**overrides: object) -> RunManifest:
    values: dict[str, object] = {
        "candidate_name": "baseline",
        "source_sha256": "audio-sha",
        "reference_sha256": "reference-sha",
        "git_commit": "abc123",
        "model": "example/model",
        "model_revision": "revision-1",
        "options": {"chunk_size_s": 30},
        "hardware": {"chip": "Apple M4"},
        "stage_times_s": {"transcribe": 10.0, "align": 2.0},
        "peak_rss_mb": 512.0,
        "status": "completed",
        "error": None,
        "vocabulary_sha256": "vocabulary-sha",
    }
    values.update(overrides)
    return RunManifest(**values)  # type: ignore[arg-type]


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "benchmark"
    dataset.mkdir()
    (dataset / "benchmark.json").write_text(
        json.dumps(
            {
                "audio_sha256": "audio-sha",
                "reference_sha256": "reference-sha",
                "vocabulary_sha256": "vocabulary-sha",
            }
        ),
        encoding="utf-8",
    )
    return dataset


@pytest.mark.parametrize(
    "overrides",
    (
        {"source_sha256": "different-audio"},
        {"reference_sha256": "different-reference"},
    ),
)
def test_write_run_rejects_checksum_mismatch(tmp_path: Path, overrides: dict[str, object]) -> None:
    """A manifest from different source or reference evidence must never be persisted."""
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="^benchmark checksum mismatch$"):
        write_run(dataset, _manifest(**overrides), {"segments": []})

    assert not (dataset / "runs").exists()


def test_write_run_rejects_manifest_without_vocabulary_binding(tmp_path: Path) -> None:
    """Legacy unbound manifests must fail migration instead of being silently accepted."""
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="^benchmark checksum mismatch$"):
        write_run(dataset, _manifest(vocabulary_sha256=None), {"segments": []})

    assert not (dataset / "runs").exists()


def test_write_run_persists_manifest_and_result_atomically(tmp_path: Path) -> None:
    """Both durable run artifacts must cross the repository's atomic-write boundary."""
    dataset = _dataset(tmp_path)
    manifest = _manifest()
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "Тест"}]}

    with patch("audio_transcribe.benchmark.store.atomic_write_text", wraps=atomic_write_text) as atomic_write:
        manifest_path, result_path = write_run(dataset, manifest, result)

    assert manifest_path == dataset / "runs" / "baseline" / "manifest.json"
    assert result_path == dataset / "runs" / "baseline" / "result.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["git_commit"] == "abc123"
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert [call.args[0] for call in atomic_write.call_args_list] == [manifest_path, result_path]


def test_write_run_invalidates_previous_evaluation_after_replacing_result(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    evaluation = dataset / "runs" / "baseline" / "evaluation.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text('{"result_sha256":"stale"}', encoding="utf-8")

    write_run(dataset, _manifest(), {"segments": [{"text": "new result"}]})

    assert not evaluation.exists()


def test_write_run_invalidates_evaluation_before_partial_artifact_failure(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run_dir = dataset / "runs" / "baseline"
    run_dir.mkdir(parents=True)
    evaluation = run_dir / "evaluation.json"
    evaluation.write_text('{"result_sha256":"old"}', encoding="utf-8")
    (run_dir / "result.json").write_text('{"segments":[{"text":"old"}]}', encoding="utf-8")
    writes = 0

    def write_manifest_then_fail(path: Path, content: str) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("result write failed")
        atomic_write_text(path, content)

    with (
        patch("audio_transcribe.benchmark.store.atomic_write_text", side_effect=write_manifest_then_fail),
        pytest.raises(OSError, match="result write failed"),
    ):
        write_run(dataset, _manifest(), {"segments": [{"text": "new"}]})

    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["git_commit"] == "abc123"
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["segments"][0]["text"] == "old"
    assert not evaluation.exists()
    benchmark = json.loads((dataset / "benchmark.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="no evaluated ASR runs"):
        _evaluated_runs(dataset, benchmark)


def test_write_diarization_run_invalidates_evaluation_before_artifact_failure(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark"
    run_dir = dataset / "runs" / "diarization" / "pyannote-3.1"
    run_dir.mkdir(parents=True)
    evaluation = run_dir / "evaluation.json"
    evaluation.write_text('{"result_sha256":"old"}', encoding="utf-8")
    result_path = run_dir / "result.json"
    result_path.write_text('{"status":"completed","old":true}', encoding="utf-8")
    run = DiarizationRunResult(
        DiarizationCandidate("pyannote-3.1", "pyannote/speaker-diarization-3.1", False),
        (),
        (),
        1.0,
        100.0,
        "completed",
        None,
    )

    with (
        patch("audio_transcribe.benchmark.store.atomic_write_text", side_effect=OSError("result write failed")),
        pytest.raises(OSError, match="result write failed"),
    ):
        write_diarization_run(
            dataset,
            run,
            source_sha256="audio",
            aligned_asr_sha256="asr",
            speaker_reference_sha256="speakers",
        )

    assert json.loads(result_path.read_text(encoding="utf-8"))["old"] is True
    assert not evaluation.exists()


def test_write_run_keeps_previous_evaluation_when_new_payload_cannot_serialize(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    evaluation = dataset / "runs" / "baseline" / "evaluation.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text('{"result_sha256":"still-valid"}', encoding="utf-8")

    with (
        patch("audio_transcribe.benchmark.store.atomic_write_text") as atomic_write,
        pytest.raises(ValueError, match="Out of range float values"),
    ):
        write_run(dataset, _manifest(), {"score": float("nan")})

    atomic_write.assert_not_called()
    assert evaluation.is_file()


def test_write_run_serializes_both_payloads_before_first_write(tmp_path: Path) -> None:
    """An invalid result must not leave behind an otherwise valid manifest-only run."""
    dataset = _dataset(tmp_path)

    with (
        patch("audio_transcribe.benchmark.store.atomic_write_text") as atomic_write,
        pytest.raises(ValueError, match="Out of range float values"),
    ):
        write_run(dataset, _manifest(), {"score": float("nan")})

    atomic_write.assert_not_called()
    assert not (dataset / "runs").exists()


@pytest.mark.parametrize(
    "manifest",
    (
        _manifest(status="completed", error="unexpected error"),
        _manifest(status="failed", error=None),
        _manifest(status="unknown", error=None),
        _manifest(stage_times_s={"transcribe": float("inf")}),
        _manifest(stage_times_s={"transcribe": 1e308, "align": 1e308}),
        _manifest(peak_rss_mb=float("nan")),
    ),
)
def test_write_run_rejects_invalid_manifest_invariants(tmp_path: Path, manifest: RunManifest) -> None:
    """Persisted manifests must have coherent status, error, timing, and memory evidence."""
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="invalid benchmark run"):
        write_run(dataset, manifest, {"segments": []})

    assert not (dataset / "runs").exists()

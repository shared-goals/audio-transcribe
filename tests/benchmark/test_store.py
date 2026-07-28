"""Tests for checksum-bound ASR benchmark run persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from audio_transcribe.benchmark.models import RunManifest
from audio_transcribe.benchmark.store import write_run
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
    }
    values.update(overrides)
    return RunManifest(**values)  # type: ignore[arg-type]


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "benchmark"
    dataset.mkdir()
    (dataset / "benchmark.json").write_text(
        json.dumps({"audio_sha256": "audio-sha", "reference_sha256": "reference-sha"}),
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

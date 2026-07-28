"""Tests for the explicit private ASR benchmark command surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from audio_transcribe.cli import app

runner = CliRunner()


def test_benchmark_help_lists_commands() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    for command in ("prepare", "run-asr", "evaluate-asr", "report"):
        assert command in result.output


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

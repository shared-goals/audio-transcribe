"""Every command must at least build its options and print help."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audio_transcribe.cli import app

runner = CliRunner()


def _commands() -> list[list[str]]:
    paths: list[list[str]] = []
    for command in app.registered_commands:
        name = command.name or (command.callback.__name__ if command.callback else None)
        if name:
            paths.append([name])
    for group in app.registered_groups:
        assert group.name
        paths.append([group.name])
        typer_app = group.typer_instance
        for sub in typer_app.registered_commands if typer_app else []:
            name = sub.name or (sub.callback.__name__ if sub.callback else None)
            if name:
                paths.append([group.name, name])
    return paths


def test_every_command_and_group_prints_help() -> None:
    paths = _commands()
    assert len(paths) >= 8, paths
    for path in paths:
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output)
        assert "Usage" in result.output, path


def test_retime_rejects_a_note_without_audio(tmp_path: Path) -> None:
    note = tmp_path / "demo.md"
    note.write_text("---\ntype: meeting\ntitle: x\n---\n\n## Transcript\n\n[00:00] A: x\n", encoding="utf-8")
    result = runner.invoke(app, ["retime", str(note)])
    assert result.exit_code == 2
    assert "артефакта сегментов" in result.output or "audio_file" in result.output


def test_attribute_reports_a_missing_artifact(tmp_path: Path) -> None:
    note = tmp_path / "demo.md"
    note.write_text(
        "---\ntype: meeting\naudio_data: .audio-data/demo.json\n---\n\n## Transcript\n\n[00:00] A: x\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["attribute", str(note), "--speaker", "SPEAKER_00", "--label", "X"])
    assert result.exit_code == 1
    assert "Отказ" in result.output
    assert "Audio data not found" in result.output


def test_retime_rewrites_timestamps_and_records_the_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from audio_transcribe.stages import preprocess as preprocess_module

    monkeypatch.setattr(preprocess_module, "preprocess", lambda *args, **kwargs: "/tmp/trimmed.wav")
    monkeypatch.setattr(preprocess_module, "leading_trim_offset", lambda *args, **kwargs: 1.5)

    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 32)
    note = tmp_path / "demo.md"
    note.write_text(
        f"---\ntype: meeting\naudio_file: {audio}\naudio_data: .audio-data/demo.json\n---\n\n"
        "## Transcript\n\n[00:00] A: привет\n",
        encoding="utf-8",
    )
    data = note.parent / ".audio-data" / "demo.json"
    data.parent.mkdir()
    data.write_text(json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "привет"}]}), encoding="utf-8")

    result = runner.invoke(app, ["retime", str(note)])
    assert result.exit_code == 0, result.output
    assert "timing_offset_s: 1.5" in note.read_text(encoding="utf-8")
    assert "[00:01] A: привет" in note.read_text(encoding="utf-8")
    assert json.loads(data.read_text(encoding="utf-8"))["segments"][0]["start"] == 1.5

    again = runner.invoke(app, ["retime", str(note)])
    assert again.exit_code == 0
    assert "уже сдвинуто" in again.output


def test_worker_status_reads_an_empty_queue(tmp_path: Path) -> None:
    queue = tmp_path / "queue.db"
    text = runner.invoke(app, ["worker", "status", "--queue", str(queue)])
    assert text.exit_code == 0, text.output
    assert "health" in text.output

    payload = runner.invoke(app, ["worker", "status", "--queue", str(queue), "--json"])
    assert payload.exit_code == 0, payload.output
    assert json.loads(payload.output)["jobs"] == []


def test_process_reports_a_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(app, ["process", str(tmp_path / "absent.wav")])
    assert result.exit_code == 1
    assert "file not found" in result.output

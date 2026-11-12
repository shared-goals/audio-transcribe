import json
import shutil
from pathlib import Path

import pytest

from audio_transcribe.attribution import AttributionError, Scope, apply_correction, load_versions, log_path

NOTE = """---
type: meeting
title: 2026-11-10 meeting
date: '2026-11-10'
audio_file: /Users/demo/Movies/demo.mp3
audio_data: .audio-data/demo.json
speakers:
  SPEAKER_00: Speaker A
  SPEAKER_01: Speaker B
---

## Speakers

- **Speaker A**: SPEAKER_00
- **Speaker B**: SPEAKER_01

## Transcript

[00:01] Speaker A: первая реплика
[00:05] Speaker B: вторая реплика
[00:10] Speaker A: третья реплика, текст не трогаем
"""

SEGMENTS = {
    "audio_file": "/Users/demo/Movies/demo.mp3",
    "language": "ru",
    "model": "large-v3",
    "processing_time_s": 1.0,
    "segments": [
        {"start": 1.0, "end": 4.0, "text": "первая реплика", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 8.0, "text": "вторая реплика", "speaker": "SPEAKER_01"},
        {"start": 10.0, "end": 14.0, "text": "третья реплика, текст не трогаем", "speaker": "SPEAKER_00"},
    ],
}


@pytest.fixture()
def meeting(tmp_path: Path) -> Path:
    note = tmp_path / "meetings" / "demo.md"
    note.parent.mkdir(parents=True)
    note.write_text(NOTE, encoding="utf-8")
    data = note.parent / ".audio-data" / "demo.json"
    data.parent.mkdir()
    data.write_text(json.dumps(SEGMENTS), encoding="utf-8")
    return note


def test_whole_cluster_rename_updates_note_legend_and_log(meeting: Path) -> None:
    version = apply_correction(
        meeting,
        Scope("cluster"),
        "SPEAKER_00",
        label="[[actors/demo|Демо]]",
        author="anton",
        reason="узнал голос",
        now="2026-11-13T10:00:00+00:00",
    )

    text = meeting.read_text(encoding="utf-8")
    assert version.version == 1
    assert "SPEAKER_00: '[[actors/demo|Демо]]'" in text or 'SPEAKER_00: "[[actors/demo|Демо]]"' in text
    assert "attribution_version: 1" in text
    assert "reanalyze: true" in text
    assert "[00:01] [[actors/demo|Демо]]: первая реплика" in text
    assert "[00:10] [[actors/demo|Демо]]: третья реплика, текст не трогаем" in text
    assert "- **[[actors/demo|Демо]]**: SPEAKER_00" in text
    assert "[00:05] Speaker B: вторая реплика" in text

    entries = load_versions(log_path(meeting))
    assert [e.version for e in entries] == [1]
    assert entries[0].operations[0]["segments"] == 2
    assert entries[0].reason == "узнал голос"
    assert entries[0].note_before_sha256


def test_partial_scope_moves_segments_to_existing_cluster(meeting: Path) -> None:
    apply_correction(
        meeting,
        Scope("range", start=0.0, end=6.0),
        "SPEAKER_00",
        target_speaker_id="SPEAKER_01",
        author="anton",
    )
    text = meeting.read_text(encoding="utf-8")
    assert "[00:01] Speaker B: первая реплика" in text
    assert "[00:10] Speaker A: третья реплика, текст не трогаем" in text
    assert "SPEAKER_00: Speaker A" in text


def test_new_label_on_partial_cluster_is_refused(meeting: Path) -> None:
    with pytest.raises(AttributionError, match="только для всего кластера"):
        apply_correction(
            meeting,
            Scope("segment", index=0),
            "SPEAKER_00",
            label="[[actors/demo|Демо]]",
            author="anton",
        )
    assert not log_path(meeting).exists()


def test_stale_base_version_is_refused_and_nothing_is_written(meeting: Path) -> None:
    apply_correction(meeting, Scope("cluster"), "SPEAKER_00", label="Speaker C", author="anton")
    before = meeting.read_text(encoding="utf-8")
    with pytest.raises(AttributionError, match="устаревшая база"):
        apply_correction(
            meeting,
            Scope("cluster"),
            "SPEAKER_01",
            label="Speaker D",
            author="anton",
            base_version=0,
        )
    assert meeting.read_text(encoding="utf-8") == before
    assert len(load_versions(log_path(meeting))) == 1


def test_second_correction_keeps_previous_version(meeting: Path) -> None:
    apply_correction(meeting, Scope("cluster"), "SPEAKER_00", label="Speaker C", author="anton")
    first = load_versions(log_path(meeting))[0]
    apply_correction(meeting, Scope("cluster"), "SPEAKER_01", label="Speaker D", author="anton")

    entries = load_versions(log_path(meeting))
    assert [e.version for e in entries] == [1, 2]
    assert entries[0] == first
    assert entries[1].base_version == 1
    text = meeting.read_text(encoding="utf-8")
    assert "[00:01] Speaker C: первая реплика" in text
    assert "[00:05] Speaker D: вторая реплика" in text


def test_hand_edited_line_aborts_without_writing(meeting: Path) -> None:
    edited = NOTE.replace("[00:05] Speaker B: вторая реплика", "[00:05] Кто-то: вторая реплика")
    meeting.write_text(edited, encoding="utf-8")
    with pytest.raises(AttributionError, match="не найдена строка"):
        apply_correction(meeting, Scope("cluster"), "SPEAKER_01", label="Speaker D", author="anton")
    assert meeting.read_text(encoding="utf-8") == edited
    assert not log_path(meeting).exists()


def test_unknown_speaker_is_refused(meeting: Path) -> None:
    with pytest.raises(AttributionError, match="отсутствует в транскрипте"):
        apply_correction(meeting, Scope("cluster"), "SPEAKER_09", label="X", author="anton")


def test_log_dir_is_tracked_not_gitignored(meeting: Path) -> None:
    apply_correction(meeting, Scope("cluster"), "SPEAKER_00", label="Speaker C", author="anton")
    assert log_path(meeting).parent.name == ".attribution"
    assert log_path(meeting).exists()


def test_failed_note_write_removes_appended_log_line(meeting: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import audio_transcribe.attribution as attribution

    def boom(path: Path, content: str, encoding: str = "utf-8") -> None:
        raise OSError("disk full")

    monkeypatch.setattr(attribution, "atomic_write_text", boom)
    with pytest.raises(OSError):
        apply_correction(meeting, Scope("cluster"), "SPEAKER_00", label="Speaker C", author="anton")
    assert load_versions(log_path(meeting)) == []
    shutil.rmtree(meeting.parent / ".attribution", ignore_errors=True)


UNKNOWN_NOTE = """---
type: meeting
title: demo
audio_file: /Users/demo/Movies/demo.mp3
audio_data: .audio-data/demo.json
---

## Transcript

[00:00] Unknown: одна реплика
"""


def test_unknown_speaker_line_is_matched_and_relabelled(tmp_path: Path) -> None:
    note = tmp_path / "meetings" / "demo.md"
    note.parent.mkdir(parents=True)
    note.write_text(UNKNOWN_NOTE, encoding="utf-8")
    data = note.parent / ".audio-data" / "demo.json"
    data.parent.mkdir()
    data.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 5.0, "text": "одна реплика", "speaker": "UNKNOWN"}],
            }
        ),
        encoding="utf-8",
    )

    apply_correction(note, Scope("cluster"), "UNKNOWN", label="[[actors/demo|Демо]]", author="anton")

    text = note.read_text(encoding="utf-8")
    assert "[00:00] [[actors/demo|Демо]]: одна реплика" in text
    assert "UNKNOWN: '[[actors/demo|Демо]]'" in text or 'UNKNOWN: "[[actors/demo|Демо]]"' in text


def test_versions_can_be_listed_without_a_speaker(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from audio_transcribe.cli import app

    note = tmp_path / "demo.md"
    note.write_text(NOTE, encoding="utf-8")
    result = CliRunner().invoke(app, ["attribute", str(note), "--versions"])
    assert result.exit_code == 0, result.output
    assert "Версий нет" in result.output


def test_speaker_is_required_for_a_correction(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from audio_transcribe.cli import app

    note = tmp_path / "demo.md"
    note.write_text(NOTE, encoding="utf-8")
    result = CliRunner().invoke(app, ["attribute", str(note), "--label", "X"])
    assert result.exit_code == 2

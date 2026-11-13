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


def test_partial_label_rewrites_only_that_line(meeting: Path) -> None:
    version = apply_correction(
        meeting,
        Scope("segment", index=0),
        "SPEAKER_00",
        label="[[actors/demo|Демо]]",
        author="anton",
    )
    text = meeting.read_text(encoding="utf-8")
    assert "[00:01] [[actors/demo|Демо]]: первая реплика" in text
    assert "[00:10] Speaker A: третья реплика, текст не трогаем" in text
    assert "SPEAKER_00: Speaker A" in text, "cluster label must stay for the rest"
    assert version.operations[0]["partial"] is True


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


def test_cluster_rename_replaces_a_line_labelled_by_hand(meeting: Path) -> None:
    edited = NOTE.replace("[00:05] Speaker B: вторая реплика", "[00:05] Кто-то: вторая реплика")
    meeting.write_text(edited, encoding="utf-8")
    version = apply_correction(meeting, Scope("cluster"), "SPEAKER_01", label="Speaker D", author="anton")
    text = meeting.read_text(encoding="utf-8")
    assert "[00:05] Speaker D: вторая реплика" in text
    assert version.operations[0]["relabelled_lines"] == 1


def test_missing_transcript_line_aborts_without_writing(meeting: Path) -> None:
    edited = NOTE.replace("[00:05] Speaker B: вторая реплика\n", "")
    meeting.write_text(edited, encoding="utf-8")
    with pytest.raises(AttributionError, match="разошлись"):
        apply_correction(meeting, Scope("cluster"), "SPEAKER_01", label="Speaker D", author="anton")
    assert meeting.read_text(encoding="utf-8") == edited
    assert not log_path(meeting).exists()


def test_single_segment_edit_replaces_a_line_labelled_by_hand(meeting: Path) -> None:
    edited = NOTE.replace(
        "[00:05] Speaker B: вторая реплика",
        "[00:05] [[actors/kto-to|Кто-то]]: вторая реплика",
    )
    meeting.write_text(edited, encoding="utf-8")
    version = apply_correction(
        meeting,
        Scope("segment", index=1),
        "SPEAKER_01",
        label="[[actors/drugoi|Другой]]",
        author="anton",
    )
    text = meeting.read_text(encoding="utf-8")
    assert "[00:05] [[actors/drugoi|Другой]]: вторая реплика" in text
    assert "[00:01] Speaker A: первая реплика" in text
    assert version.operations[0]["segments"] == 1


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


def test_audio_data_override_is_used(tmp_path: Path) -> None:
    note = tmp_path / "meetings" / "demo.md"
    note.parent.mkdir(parents=True)
    note.write_text(NOTE, encoding="utf-8")
    cache = tmp_path / "cache" / "demo.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(SEGMENTS), encoding="utf-8")

    apply_correction(note, Scope("cluster"), "SPEAKER_00", label="Speaker C", author="anton", audio_data=cache)
    assert "Speaker C" in note.read_text(encoding="utf-8")


def test_shift_timings_moves_segments_and_words() -> None:
    from audio_transcribe.stages.timing import shift_timings

    result = {
        "segments": [
            {"start": 0.37, "end": 1.5, "text": "x", "words": [{"word": "x", "start": 0.37, "end": 0.9}]},
            {"start": 4.0, "end": 5.0, "text": "y"},
        ]
    }
    shifted = shift_timings(result, 1.597)
    assert shifted["segments"][0]["start"] == 1.967
    assert shifted["segments"][0]["words"][0]["end"] == 2.497
    assert shifted["segments"][1]["start"] == 5.597
    assert result["segments"][0]["start"] == 0.37, "input must not be mutated"

    from audio_transcribe.stages.timing import format_stamp, parse_stamp

    assert parse_stamp("01:02") == 62
    assert parse_stamp("1:00:01") == 3601
    assert format_stamp(62) == "01:02"
    assert format_stamp(3601) == "1:00:01"


def test_retime_shifts_only_transcript_lines_and_records_offset() -> None:
    from audio_transcribe.retime import current_offset, retime_document, with_offset_marker

    text = NOTE
    data = {
        "segments": [
            {"start": 0.37, "end": 1.5, "text": "x", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 8.0, "text": "y", "speaker": "SPEAKER_01"},
            {"start": 10.0, "end": 14.0, "text": "z", "speaker": "SPEAKER_00"},
        ]
    }
    updated, shifted, lines = retime_document(text, data, 1.597)
    assert lines == 3
    # Stamps come from the shifted segments (0.37+1.597=1.967 -> 00:01), not from
    # adding the offset to the old display stamp.
    assert "[00:01] Speaker A: первая реплика" in updated
    assert "[00:06] Speaker B: вторая реплика" in updated
    assert "[00:11] Speaker A: третья реплика, текст не трогаем" in updated
    assert shifted["segments"][0]["start"] == 1.967

    marked = with_offset_marker(updated, 1.597)
    assert "timing_offset_s: 1.597" in marked
    assert current_offset(marked) == 1.597


def test_retime_changes_only_timestamps_and_the_offset_line() -> None:
    from audio_transcribe.retime import retime_document, with_offset_marker

    segments = [
        {"start": 1.0, "end": 2.0, "text": "x"},
        {"start": 5.0, "end": 6.0, "text": "y"},
        {"start": 10.0, "end": 11.0, "text": "z"},
    ]
    updated, _data, _lines = retime_document(NOTE, {"segments": segments}, 1.5)
    marked = with_offset_marker(updated, 1.5)
    old_lines = NOTE.splitlines()
    new_lines = [line for line in marked.splitlines() if not line.startswith("timing_offset_s:")]
    assert len(old_lines) == len(new_lines)
    changed = [(a, b) for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    for old, new in changed:
        assert old.startswith("[") and new.startswith("["), (old, new)
    assert marked.count("timing_offset_s: 1.5") == 1


def test_bulk_retime_skips_notes_that_are_already_shifted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from audio_transcribe.cli import app
    from audio_transcribe.stages import preprocess as preprocess_module

    monkeypatch.setattr(preprocess_module, "preprocess", lambda *args, **kwargs: "/tmp/trimmed.wav")
    monkeypatch.setattr(preprocess_module, "leading_trim_offset", lambda *args, **kwargs: 2.0)

    meetings = tmp_path / "meetings"
    meetings.mkdir()
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 16)
    for name, offset in (("first", None), ("second", 2.0)):
        note = meetings / f"{name}.md"
        extra = "" if offset is None else f"timing_offset_s: {offset}\n"
        note.write_text(
            f"---\ntype: meeting\naudio_file: {audio}\naudio_data: .audio-data/{name}.json\n{extra}---\n\n"
            "## Transcript\n\n[00:00] A: x\n",
            encoding="utf-8",
        )
        data = meetings / ".audio-data" / f"{name}.json"
        data.parent.mkdir(exist_ok=True)
        payload: dict[str, object] = {"segments": [{"start": 0.0, "end": 1.0, "text": "x"}]}
        if offset is not None:
            payload["timing_offset_s"] = offset
        data.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["retime", "--all", "--meetings-dir", str(meetings)])
    assert result.exit_code == 0, result.output
    assert "first.md: сдвинуто на 2.000" in result.output
    assert "second.md: уже сдвинуто на 2.000" in result.output
    assert "Изменено: 1" in result.output


def test_empty_segment_has_no_trailing_space() -> None:
    from audio_transcribe.stages.format import format_segment

    assert format_segment({"start": 0.0, "text": "", "speaker": "UNKNOWN"}) == "[00:00] Unknown:"
    assert format_segment({"start": 61.0, "text": "привет", "speaker": "UNKNOWN"}) == "[01:01] Unknown: привет"

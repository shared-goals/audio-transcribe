"""Tests for private ASR reference preparation and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from audio_transcribe.benchmark.models import ReferenceSpan
from audio_transcribe.benchmark.reference import parse_reference, prepare_reference, reference_text


def test_parse_reference_excludes_unclear_span(tmp_path: Path) -> None:
    """An unclear interval must not contribute to ASR error-rate reference text."""
    reference = tmp_path / "reference-transcript.md"
    reference.write_text(
        "# Reference transcript\n\n[00:00.000 --> 00:02.500] Первый этап.\n[00:02.500 --> 00:04.000] [неразборчиво]\n",
        encoding="utf-8",
    )

    document = parse_reference(reference)

    assert document.spans[0] == ReferenceSpan(0.0, 2.5, "Первый этап.", False)
    assert document.spans[1].excluded is True
    assert reference_text(document) == "Первый этап."
    assert document.source_sha256 == hashlib.sha256(reference.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("[00:00.000 --> 00:01.000] Первая строка\nnot a span\n", "invalid reference line"),
        (
            "[00:00.000 --> 00:02.000] Первая строка\n[00:01.999 --> 00:03.000] Перекрытие\n",
            "overlaps",
        ),
        ("[00:02.000 --> 00:01.999] Отрицательная длительность\n", "negative duration"),
        ("# Reference transcript\n", "no spans"),
    ],
)
def test_parse_reference_rejects_invalid_document(content: str, reason: str, tmp_path: Path) -> None:
    """Malformed or unusable reference documents must fail before a benchmark can run."""
    reference = tmp_path / "reference-transcript.md"
    reference.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=reason):
        parse_reference(reference)


def test_prepare_reference_creates_private_draft_without_modifying_note(tmp_path: Path) -> None:
    """Preparation must preserve the current note while creating editable private benchmark inputs."""
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"private audio bytes")
    current_note = tmp_path / "meeting.md"
    original_note = (
        "---\n"
        "type: meeting\n"
        "---\n\n"
        "## Speakers\n\n"
        "- **Speaker A**: SPEAKER_00\n\n"
        "## Transcript\n\n"
        "[00:00] Speaker A: Первый этап.\n"
        "[00:02] Speaker B: Второй этап.\n"
    )
    current_note.write_text(original_note, encoding="utf-8")
    dataset = tmp_path / "datasets" / "pilot"
    dataset.parent.mkdir()

    draft_path, manifest_path = prepare_reference(audio, current_note, dataset)

    assert draft_path == dataset / "reference-transcript.draft.md"
    assert manifest_path == dataset / "benchmark.json"
    assert draft_path.read_text(encoding="utf-8") == (
        "# Reference transcript draft\n\n[00:00] Первый этап.\n[00:02] Второй этап.\n"
    )
    assert (dataset / "vocabulary.txt").read_text(encoding="utf-8") == ""
    assert (dataset / "runs").is_dir()
    assert current_note.read_text(encoding="utf-8") == original_note

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["audio_path"] == str(audio.resolve())
    assert manifest["audio_sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert manifest["current_note_path"] == str(current_note.resolve())
    assert manifest["reference_status"] == "draft"
    assert manifest["vocabulary_status"] == "draft"
    assert "vocabulary_sha256" not in manifest
    assert manifest["created_at"].endswith("+00:00")


def test_prepare_reference_rejects_dataset_without_existing_parent(tmp_path: Path) -> None:
    """Preparation must not create an arbitrary parent tree outside the selected private dataset area."""
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"private audio bytes")
    current_note = tmp_path / "meeting.md"
    current_note.write_text("## Transcript\n\n[00:00] Первый этап.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset parent"):
        prepare_reference(audio, current_note, tmp_path / "missing" / "pilot")

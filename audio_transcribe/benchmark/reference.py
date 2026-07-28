"""Parse accepted ASR references and prepare private editable drafts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from audio_transcribe.benchmark.models import ReferenceDocument, ReferenceSpan
from audio_transcribe.markdown.parser import parse_meeting, parse_speaker_legend
from audio_transcribe.util import atomic_write_text

_REFERENCE_HEADER = "# Reference transcript"
_REFERENCE_LINE = re.compile(r"^\[(\d\d):(\d\d)\.(\d{3}) --> (\d\d):(\d\d)\.(\d{3})\] (.+)$")
_NOTE_LINE = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s+(.+)$")
_STANDARD_SPEAKER_LABEL = re.compile(
    r"^(?:(?:Speaker\s+[A-Z]+|Unknown|SPEAKER_\d+|\[\[[^\]]+\]\])\s*:\s+)(.+)$",
    re.IGNORECASE,
)
_UNCLEAR_TEXT = "[неразборчиво]"


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_seconds(minutes: str, seconds: str, milliseconds: str) -> float:
    """Convert one strict reference timestamp into seconds."""
    second_value = int(seconds)
    if second_value >= 60:
        raise ValueError("reference timestamp seconds must be below 60")
    return int(minutes) * 60 + second_value + int(milliseconds) / 1000


def parse_reference(path: Path) -> ReferenceDocument:
    """Load a strictly formatted, manually reviewed ASR reference transcript."""
    source = path.expanduser().resolve()
    spans: list[ReferenceSpan] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line == _REFERENCE_HEADER:
            continue
        match = _REFERENCE_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid reference line {line_number}")
        start_s = _timestamp_seconds(match.group(1), match.group(2), match.group(3))
        end_s = _timestamp_seconds(match.group(4), match.group(5), match.group(6))
        if end_s < start_s:
            raise ValueError(f"negative duration at reference line {line_number}")
        if spans and start_s < spans[-1].end_s:
            raise ValueError(f"reference span overlaps at line {line_number}")
        text = match.group(7)
        spans.append(ReferenceSpan(start_s, end_s, text, text.casefold() == _UNCLEAR_TEXT))
    if not spans:
        raise ValueError("reference document has no spans")
    return ReferenceDocument(_file_sha256(source), tuple(spans))


def reference_text(document: ReferenceDocument) -> str:
    """Return evaluable reference speech, omitting intentionally unclear spans."""
    return " ".join(span.text for span in document.spans if not span.excluded)


def _strip_speaker_label(text: str, speaker_labels: set[str]) -> str:
    """Remove only known meeting-note speaker prefixes from transcript text."""
    for label in speaker_labels:
        prefix = f"{label}:"
        if text.startswith(prefix):
            return text[len(prefix) :].lstrip()
    match = _STANDARD_SPEAKER_LABEL.fullmatch(text)
    return match.group(1) if match is not None else text


def _draft_transcript(note: Path) -> str:
    """Extract timestamped transcript lines from a meeting note for manual review."""
    document = parse_meeting(note.read_text(encoding="utf-8"))
    transcript = document.sections.get("Transcript")
    if transcript is None:
        raise ValueError("current note has no Transcript section")
    speaker_labels = set(parse_speaker_legend(document).values())
    lines = ["# Reference transcript draft", ""]
    for line in transcript.splitlines():
        match = _NOTE_LINE.fullmatch(line)
        if match is not None:
            text = _strip_speaker_label(match.group(2), speaker_labels)
            lines.append(f"[{match.group(1)}] {text}")
    if len(lines) == 2:
        raise ValueError("current note has no timestamped transcript text")
    return "\n".join(lines) + "\n"


def prepare_reference(audio: Path, current_note: Path, dataset: Path) -> tuple[Path, Path]:
    """Create a private benchmark draft and metadata without changing the current note."""
    expanded_audio = audio.expanduser().resolve()
    expanded_note = current_note.expanduser().resolve()
    expanded_dataset = dataset.expanduser().resolve()
    if not expanded_audio.is_file():
        raise ValueError("audio must be an existing file")
    if not expanded_note.is_file():
        raise ValueError("current note must be an existing file")
    if not expanded_dataset.parent.is_dir():
        raise ValueError("dataset parent must already exist")

    draft = _draft_transcript(expanded_note)
    expanded_dataset.mkdir(exist_ok=True)
    runs = expanded_dataset / "runs"
    runs.mkdir(exist_ok=True)
    draft_path = expanded_dataset / "reference-transcript.draft.md"
    manifest_path = expanded_dataset / "benchmark.json"
    vocabulary_path = expanded_dataset / "vocabulary.txt"
    manifest = {
        "audio_path": str(expanded_audio),
        "audio_sha256": _file_sha256(expanded_audio),
        "current_note_path": str(expanded_note),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_status": "draft",
    }
    atomic_write_text(draft_path, draft)
    atomic_write_text(vocabulary_path, "")
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return draft_path, manifest_path

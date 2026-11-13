"""Versioned speaker attribution: append-only log plus in-place note update.

Every manual correction is a new version. The log holds operations only, so it is
small enough to live in Git; the transcript text and previous versions are never
rewritten. Only Audio Transcribe writes here — the GUI sends an intent.

Limitation (deliberate): a cluster keeps one label in `frontmatter.speakers`, so
a *new* label can only be given to a scope that covers the whole cluster. Moving
part of a cluster to another existing cluster is always allowed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.markdown.updater import replace_section, set_frontmatter
from audio_transcribe.stages.format import format_time
from audio_transcribe.stages.loader import load_audio_data
from audio_transcribe.util import atomic_write_text

ScopeKind = Literal["segment", "range", "cluster"]

_LINE_RE = re.compile(r"^\[([\d:]+)\]\s(.*?):\s(.*)$")


def rendered_label(speaker_id: str, speakers: dict[str, Any]) -> str:
    """The label `format_segment` would print for this acoustic id."""
    if speaker_id in speakers:
        return str(speakers[speaker_id])
    return "Unknown" if speaker_id == "UNKNOWN" else speaker_id


LOG_DIR = ".attribution"


class AttributionError(Exception):
    """A correction that cannot be applied without violating the contract."""


@dataclass(frozen=True)
class Scope:
    kind: ScopeKind
    index: int | None = None
    start: float | None = None
    end: float | None = None

    def to_json(self) -> dict[str, Any]:
        fields = (("kind", self.kind), ("index", self.index), ("start", self.start), ("end", self.end))
        return {key: value for key, value in fields if value is not None}

    def __post_init__(self) -> None:
        if self.kind == "segment" and self.index is None:
            raise AttributionError("segment scope requires index")
        if self.kind == "range" and (self.start is None or self.end is None or self.end <= self.start):
            raise AttributionError("range scope requires start < end")
        if self.kind == "cluster" and (self.start is None) != (self.end is None):
            raise AttributionError("cluster scope range needs both bounds")


@dataclass
class Version:
    version: int
    base_version: int
    at: str
    author: str
    operations: list[dict[str, Any]] = field(default_factory=list)
    note_before_sha256: str = ""
    reason: str = ""


def log_path(meeting_path: Path) -> Path:
    return meeting_path.parent / LOG_DIR / f"{meeting_path.stem}.jsonl"


def load_versions(path: Path) -> list[Version]:
    if not path.exists():
        return []
    versions: list[Version] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AttributionError(f"{path}:{number}: некорректная строка лога: {exc}") from exc
        versions.append(
            Version(
                version=int(payload["version"]),
                base_version=int(payload["base_version"]),
                at=str(payload["at"]),
                author=str(payload["author"]),
                operations=list(payload.get("operations") or []),
                note_before_sha256=str(payload.get("note_before_sha256", "")),
                reason=str(payload.get("reason", "")),
            )
        )
    return versions


def current_version(path: Path) -> int:
    versions = load_versions(path)
    return versions[-1].version if versions else 0


def _affected(segments: list[dict[str, Any]], scope: Scope) -> list[int]:
    if scope.kind == "segment":
        if scope.index is None or not 0 <= scope.index < len(segments):
            raise AttributionError(f"сегмент {scope.index} вне диапазона 0..{len(segments) - 1}")
        return [scope.index]
    if scope.kind == "range":
        assert scope.start is not None and scope.end is not None
        picked = [i for i, s in enumerate(segments) if s["start"] >= scope.start and s["end"] <= scope.end]
        if not picked:
            raise AttributionError("в указанном диапазоне нет целых реплик")
        return picked
    if scope.kind == "cluster":
        picked = [
            i
            for i, s in enumerate(segments)
            if (scope.start is None or (s["start"] >= scope.start and s["end"] <= (scope.end or 0)))
        ]
        if not picked:
            raise AttributionError("в указанном диапазоне нет реплик")
        return picked
    raise AttributionError(f"неизвестный тип области: {scope.kind}")


def plan_correction(
    meeting_path: Path,
    scope: Scope,
    speaker_id: str,
    label: str | None,
    target_speaker_id: str | None,
    version: int,
    audio_data: Path | None = None,
) -> tuple[str, dict[str, Any], list[int]]:
    """Return (updated markdown, operation, affected indices, segments).

    Nothing is written here; every validation happens before the first write.
    """
    doc = parse_meeting(meeting_path.read_text(encoding="utf-8"))
    stored = load_audio_data(meeting_path, doc) if audio_data is None else _read_audio_data(audio_data)
    segments: list[dict[str, Any]] = list(stored.get("segments", []))
    if not segments:
        raise AttributionError("в артефакте нет сегментов: правка спикера невозможна")

    known = {str(s.get("speaker", "UNKNOWN")) for s in segments}
    if speaker_id not in known:
        raise AttributionError(f"спикер {speaker_id} отсутствует в транскрипте")

    picked = _affected(segments, scope)
    if scope.kind == "cluster":
        picked = [i for i in picked if str(segments[i].get("speaker", "UNKNOWN")) == speaker_id]
        if not picked:
            raise AttributionError(f"реплик {speaker_id} в указанной области нет")

    speakers = doc.frontmatter.get("speakers")
    speakers = dict(speakers) if isinstance(speakers, dict) else {}
    old_label = rendered_label(speaker_id, speakers)

    target_id = target_speaker_id or speaker_id
    if target_id != speaker_id and target_id not in known:
        raise AttributionError(f"целевой спикер {target_id} отсутствует в транскрипте")
    if target_id == speaker_id and not label:
        raise AttributionError("нужна новая метка или другой существующий спикер")

    cluster_all = [i for i, s in enumerate(segments) if str(s.get("speaker", "UNKNOWN")) == speaker_id]
    partial_label = bool(label) and target_id == speaker_id and sorted(picked) != cluster_all
    new_label = label if (label and target_id == speaker_id) else rendered_label(target_id, speakers)
    if not new_label or new_label == target_id:
        raise AttributionError(f"у спикера {target_id} нет метки в заметке")

    # Rewrite only the label part of the affected transcript lines.
    section = doc.sections.get("Transcript", "")
    lines = section.split("\n")
    used: set[int] = set()
    for index in picked:
        segment = segments[index]
        stamp = format_time(float(segment["start"]))
        original = old_label if target_id == speaker_id else rendered_label(str(segment.get("speaker")), speakers)
        match_index = next(
            (
                n
                for n, line in enumerate(lines)
                if n not in used and (m := _LINE_RE.match(line)) and m.group(1) == stamp and m.group(2) == original
            ),
            None,
        )
        if match_index is None:
            raise AttributionError(
                f"не найдена строка транскрипта для реплики {stamp} с меткой «{original}»; "
                "заметка правилась вручную — правка отменена"
            )
        used.add(match_index)
        match = _LINE_RE.match(lines[match_index])
        assert match is not None
        lines[match_index] = f"[{match.group(1)}] {new_label}: {match.group(3)}"

    doc = replace_section(doc, "Transcript", "\n".join(lines))
    if target_id == speaker_id and not partial_label:
        speakers[speaker_id] = new_label
        doc = set_frontmatter(doc, "speakers", speakers)
        legend = doc.sections.get("Speakers", "")
        if legend:
            doc = replace_section(doc, "Speakers", legend.replace(f"**{old_label}**", f"**{new_label}**"))

    doc = set_frontmatter(doc, "attribution_version", version)
    doc = set_frontmatter(doc, "reanalyze", True)

    operation = {
        "scope": scope.to_json(),
        "speaker": speaker_id,
        "target_speaker": target_id,
        "label_before": old_label,
        "label_after": new_label,
        "segments": len(picked),
        # A partial label applies to these lines only; the cluster keeps its label.
        "partial": partial_label,
    }
    return doc.to_markdown(), operation, picked


def _read_audio_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AttributionError(f"нет артефакта сегментов: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AttributionError(f"повреждён артефакт сегментов: {path} — {exc}") from exc
    if not isinstance(data, dict):
        raise AttributionError(f"артефакт сегментов не объект: {path}")
    return data


def apply_correction(
    meeting_path: Path,
    scope: Scope,
    speaker_id: str,
    *,
    label: str | None = None,
    target_speaker_id: str | None = None,
    author: str,
    reason: str = "",
    base_version: int | None = None,
    now: str | None = None,
    audio_data: Path | None = None,
) -> Version:
    """Append a version and update the note in place.

    Order: validate everything, write the log line, then replace the note. If the
    note write fails the appended line is truncated away, so a failed correction
    never leaves a version that was applied to nothing.
    """
    if not author.strip():
        raise AttributionError("нужен автор правки")
    path = log_path(meeting_path)
    versions = load_versions(path)
    current = versions[-1].version if versions else 0
    if base_version is not None and base_version != current:
        raise AttributionError(f"устаревшая база: ожидалась версия {base_version}, текущая {current}")

    before = meeting_path.read_text(encoding="utf-8")
    updated, operation, _picked = plan_correction(
        meeting_path, scope, speaker_id, label, target_speaker_id, current + 1, audio_data
    )

    import hashlib
    from datetime import UTC, datetime

    version = Version(
        version=current + 1,
        base_version=current,
        at=now or datetime.now(UTC).isoformat(timespec="seconds"),
        author=author,
        operations=[operation],
        note_before_sha256=hashlib.sha256(before.encode("utf-8")).hexdigest(),
        reason=reason,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    previous_size = path.stat().st_size if path.exists() else 0
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(version.__dict__, ensure_ascii=False) + "\n")
        handle.flush()
    try:
        atomic_write_text(meeting_path, updated)
    except BaseException:
        with path.open("r+", encoding="utf-8") as handle:
            handle.truncate(previous_size)
        raise
    return version

"""Deterministic clean-up and reporting for diarization clusters.

A cluster shorter than `min_seconds` (or than `min_share` of the speech) is not a
speaker: those fragments are moved to the temporally nearest labelled cluster, so
an 8-second artefact cannot become `SPEAKER_03`. The report makes oversplit and
unassigned speech visible without listening.
"""

from __future__ import annotations

from typing import Any

MIN_CLUSTER_SECONDS = 30.0
MIN_CLUSTER_SHARE = 0.02
UNKNOWN_WARN_SHARE = 0.05

UNKNOWN = "UNKNOWN"


def _duration(segment: dict[str, Any]) -> float:
    return max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))


def cluster_seconds(segments: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for segment in segments:
        speaker = str(segment.get("speaker") or UNKNOWN)
        totals[speaker] = totals.get(speaker, 0.0) + _duration(segment)
    return totals


def small_clusters(
    segments: list[dict[str, Any]],
    min_seconds: float = MIN_CLUSTER_SECONDS,
    min_share: float = MIN_CLUSTER_SHARE,
) -> list[str]:
    totals = cluster_seconds(segments)
    total = sum(totals.values())
    return [
        speaker
        for speaker, seconds in sorted(totals.items())
        if speaker != UNKNOWN and (seconds < min_seconds or (total > 0 and seconds / total < min_share))
    ]


def _nearest_label(segment: dict[str, Any], segments: list[dict[str, Any]], small: set[str]) -> str | None:
    """Nearest labelled neighbour by time distance to the segment's mid-point."""
    middle = (float(segment.get("start", 0.0)) + float(segment.get("end", 0.0))) / 2
    best: tuple[float, str] | None = None
    for other in segments:
        speaker = str(other.get("speaker") or UNKNOWN)
        if speaker in small or speaker == UNKNOWN:
            continue
        other_middle = (float(other.get("start", 0.0)) + float(other.get("end", 0.0))) / 2
        distance = abs(other_middle - middle)
        if best is None or distance < best[0]:
            best = (distance, speaker)
    return best[1] if best else None


def merge_small_clusters(
    segments: list[dict[str, Any]],
    min_seconds: float = MIN_CLUSTER_SECONDS,
    min_share: float = MIN_CLUSTER_SHARE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Move tiny clusters to their nearest neighbour; return (segments, merged)."""
    small = set(small_clusters(segments, min_seconds, min_share))
    if not small:
        return segments, []
    merged: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for segment in segments:
        speaker = str(segment.get("speaker") or UNKNOWN)
        item = segment
        if speaker in small:
            target = _nearest_label(segment, segments, small) or UNKNOWN
            item = {**segment, "speaker": target}
            entry = merged.setdefault(speaker, {"from": speaker, "to": target, "segments": 0, "seconds": 0.0})
            entry["segments"] += 1
            entry["seconds"] = round(float(entry["seconds"]) + _duration(segment), 3)
        result.append(item)
    return result, sorted(merged.values(), key=lambda entry: entry["from"])


def diarization_report(
    segments: list[dict[str, Any]],
    expected_speakers: int | None = None,
    planned_speakers: int | None = None,
) -> dict[str, Any]:
    totals = cluster_seconds(segments)
    total = sum(totals.values())
    labelled = {speaker: seconds for speaker, seconds in totals.items() if speaker != UNKNOWN}
    unknown_share = (totals.get(UNKNOWN, 0.0) / total) if total > 0 else 0.0
    warnings: list[str] = []
    if expected_speakers is not None and len(labelled) != expected_speakers:
        warnings.append(f"кластеров {len(labelled)}, ожидалось {expected_speakers}")
    if planned_speakers is not None and expected_speakers is None and len(labelled) != planned_speakers:
        # A calendar plan is a weak prior: invitees may be absent, guests may join.
        hint = "гости или дробление" if len(labelled) > planned_speakers else "кто-то молчал или кластеры склеены"
        warnings.append(f"кластеров {len(labelled)}, по приглашению {planned_speakers} — возможно, {hint}")
    for speaker, seconds in sorted(labelled.items()):
        if seconds < MIN_CLUSTER_SECONDS:
            warnings.append(f"{speaker}: {seconds:.0f} с — короче {MIN_CLUSTER_SECONDS:.0f} с")
    if unknown_share > UNKNOWN_WARN_SHARE:
        warnings.append(f"UNKNOWN {unknown_share:.0%} речи")
    return {
        "clusters": [
            {"speaker": speaker, "seconds": round(seconds, 1), "share": round(seconds / total, 3) if total else 0.0}
            for speaker, seconds in sorted(labelled.items(), key=lambda kv: -kv[1])
        ],
        "unknown_seconds": round(totals.get(UNKNOWN, 0.0), 1),
        "unknown_share": round(unknown_share, 3),
        "planned_speakers": planned_speakers,
        "warnings": warnings,
    }

"""Compute quality metrics from pipeline output segments."""

from __future__ import annotations

from typing import Any

from audio_transcribe.models import QualityMetrics


def compute_quality(segments: list[dict[str, Any]]) -> QualityMetrics:
    """Compute quality scorecard from transcription segments."""
    if not segments:
        return QualityMetrics(
            segments=0,
            words_total=0,
            words_aligned=0,
            alignment_pct=0.0,
            speakers_detected=0,
            speaker_coverage_pct=0.0,
            speaker_transitions=0,
        )

    words_total = 0
    words_aligned = 0
    speakers: set[str] = set()
    segments_with_speaker = 0
    transitions = 0
    prev_speaker: str | None = None

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker != "UNKNOWN":
            speakers.add(speaker)
            segments_with_speaker += 1

        if prev_speaker is not None and speaker != "UNKNOWN" and prev_speaker != "UNKNOWN" and speaker != prev_speaker:
            transitions += 1
        if speaker != "UNKNOWN":
            prev_speaker = speaker

        if "words" in seg:
            for w in seg["words"]:
                words_total += 1
                if "start" in w:
                    words_aligned += 1
        else:
            words_total += len(seg.get("text", "").split())

    alignment_pct = (words_aligned / words_total * 100) if words_total > 0 else 0.0
    speaker_coverage_pct = (segments_with_speaker / len(segments) * 100) if segments else 0.0

    return QualityMetrics(
        segments=len(segments),
        words_total=words_total,
        words_aligned=words_aligned,
        alignment_pct=round(alignment_pct, 1),
        speakers_detected=len(speakers),
        speaker_coverage_pct=round(speaker_coverage_pct, 1),
        speaker_transitions=transitions,
    )

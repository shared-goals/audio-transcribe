"""Evaluate candidate ASR transcripts against reviewed reference intervals."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from jiwer import process_characters, process_words

from audio_transcribe.benchmark.models import AsrMetrics, ReferenceDocument, ReferenceSpan

_PUNCTUATION = re.compile(r"[\W_]+", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Normalize Russian text before word- and character-error measurement."""
    normalized = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", normalized)).strip()


def _interval_overlap(start_s: float, end_s: float, span: ReferenceSpan) -> float:
    """Return the shared duration between a candidate interval and a reference span."""
    return max(0.0, min(end_s, span.end_s) - max(start_s, span.start_s))


def _best_overlap(start_s: float, end_s: float, spans: tuple[ReferenceSpan, ...]) -> ReferenceSpan | None:
    """Return the reference span with the greatest positive overlap to an interval."""
    best_span: ReferenceSpan | None = None
    best_duration = 0.0
    for span in spans:
        duration = _interval_overlap(start_s, end_s, span)
        if duration > best_duration:
            best_span = span
            best_duration = duration
    return best_span


def _has_reviewed_overlap(start_s: float, end_s: float, spans: tuple[ReferenceSpan, ...]) -> bool:
    """Return whether an interval overlaps any evaluable reference speech."""
    return any(not span.excluded and _interval_overlap(start_s, end_s, span) > 0.0 for span in spans)


def _has_uncovered_time(start_s: float, end_s: float, spans: tuple[ReferenceSpan, ...]) -> bool:
    """Return whether an interval contains time not covered by any reference span."""
    covered_until = start_s
    overlaps = sorted(
        (
            max(start_s, span.start_s),
            min(end_s, span.end_s),
        )
        for span in spans
        if _interval_overlap(start_s, end_s, span) > 0.0
    )
    for overlap_start, overlap_end in overlaps:
        if overlap_start > covered_until:
            return True
        covered_until = max(covered_until, overlap_end)
    return covered_until < end_s


def _contains_token_sequence(tokens: list[str], term_tokens: list[str]) -> bool:
    """Return whether a normalized vocabulary term appears as consecutive whole tokens."""
    term_length = len(term_tokens)
    return bool(term_tokens) and any(tokens[index : index + term_length] == term_tokens for index in range(len(tokens)))


def _candidate_values(segment: dict[str, Any]) -> tuple[float, float, str]:
    """Extract one candidate's timing and text from the pipeline segment contract."""
    return float(segment["start"]), float(segment["end"]), str(segment.get("text", ""))


def _near_boundary(start_s: float, end_s: float, boundaries: tuple[float, ...]) -> bool:
    """Return whether either segment edge falls within one second of a VAD boundary."""
    return any(min(abs(start_s - boundary), abs(end_s - boundary)) <= 1.0 for boundary in boundaries)


def evaluate_asr(
    reference: ReferenceDocument,
    segments: list[dict[str, Any]],
    terms: tuple[str, ...],
    boundaries: tuple[float, ...],
) -> AsrMetrics:
    """Calculate transcript accuracy, terminology recall, and time-based safety signals."""
    reference_text = " ".join(_normalize(span.text) for span in reference.spans if not span.excluded)
    candidate_texts: list[str] = []
    hallucinated_phrases: list[str] = []
    boundary_errors = 0

    for segment in segments:
        start_s, end_s, text = _candidate_values(segment)
        normalized_text = _normalize(text)
        best_span = _best_overlap(start_s, end_s, reference.spans)
        has_reviewed_overlap = _has_reviewed_overlap(start_s, end_s, reference.spans)
        has_uncovered_time = _has_uncovered_time(start_s, end_s, reference.spans)
        if normalized_text and has_uncovered_time:
            hallucinated_phrases.append(text)
        if has_reviewed_overlap or has_uncovered_time:
            candidate_texts.append(normalized_text)
        if best_span is None or best_span.excluded:
            continue
        is_boundary_mismatch = (
            normalized_text and best_span is not None and normalized_text != _normalize(best_span.text)
        )
        if is_boundary_mismatch and _near_boundary(start_s, end_s, boundaries):
            boundary_errors += 1

    hypothesis_text = " ".join(candidate_texts)
    word_output = process_words(reference_text, hypothesis_text)
    character_output = process_characters(reference_text, hypothesis_text)
    normalized_terms = tuple(_normalize(term).split() for term in terms)
    reference_tokens = reference_text.split()
    hypothesis_tokens = hypothesis_text.split()
    term_hits = sum(
        _contains_token_sequence(reference_tokens, term_tokens)
        and _contains_token_sequence(hypothesis_tokens, term_tokens)
        for term_tokens in normalized_terms
    )
    missed_phrases = tuple(
        span.text
        for span in reference.spans
        if not span.excluded
        and _normalize(span.text)
        and not any(
            _normalize(_candidate_values(segment)[2])
            and _interval_overlap(_candidate_values(segment)[0], _candidate_values(segment)[1], span) > 0.0
            for segment in segments
        )
    )
    reference_words = len(reference_text.split())
    hypothesis_words = len(hypothesis_text.split())
    term_total = len(normalized_terms)
    return AsrMetrics(
        wer=word_output.wer,
        cer=character_output.cer,
        substitutions=word_output.substitutions,
        deletions=word_output.deletions,
        insertions=word_output.insertions,
        reference_words=reference_words,
        hypothesis_words=hypothesis_words,
        term_hits=term_hits,
        term_total=term_total,
        term_recall=term_hits / term_total if term_total else 0.0,
        hallucinated_phrases=tuple(hallucinated_phrases),
        missed_phrases=missed_phrases,
        boundary_errors=boundary_errors,
    )

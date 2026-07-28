"""Tests for ASR benchmark quality metrics."""

from __future__ import annotations

from audio_transcribe.benchmark.asr_metrics import evaluate_asr
from audio_transcribe.benchmark.models import ReferenceDocument, ReferenceSpan


def test_evaluate_asr_counts_edits_and_terms() -> None:
    """Word edits and reviewed terminology must be measured independently."""
    reference = ReferenceDocument(
        "sha",
        (ReferenceSpan(0.0, 4.0, "Следующий этап и ЧТЗ.", False),),
    )

    metrics = evaluate_asr(
        reference,
        [{"start": 0.0, "end": 4.0, "text": "Следственный этап."}],
        ("следующий этап", "ЧТЗ"),
        (),
    )

    assert metrics.substitutions == 1
    assert metrics.deletions == 2
    assert metrics.term_hits == 0
    assert metrics.term_total == 2


def test_evaluate_asr_normalizes_russian_case_punctuation_and_yo() -> None:
    """Equivalent Russian punctuation, case, and ё spelling must not create errors."""
    reference = ReferenceDocument("sha", (ReferenceSpan(0.0, 2.0, "Ёлка, ТЕСТ!", False),))

    metrics = evaluate_asr(reference, [{"start": 0.0, "end": 2.0, "text": "елка тест"}], (), ())

    assert metrics.wer == 0.0
    assert metrics.cer == 0.0
    assert metrics.reference_words == 2
    assert metrics.hypothesis_words == 2


def test_evaluate_asr_excludes_unclear_reference_intervals() -> None:
    """Candidate speech aligned to a reviewed unclear interval must not affect error rates."""
    reference = ReferenceDocument(
        "sha",
        (
            ReferenceSpan(0.0, 2.0, "Первый этап.", False),
            ReferenceSpan(2.0, 4.0, "[неразборчиво]", True),
        ),
    )

    metrics = evaluate_asr(
        reference,
        [
            {"start": 0.0, "end": 2.0, "text": "Первый этап"},
            {"start": 2.0, "end": 4.0, "text": "лишние слова"},
        ],
        (),
        (),
    )

    assert metrics.wer == 0.0
    assert metrics.reference_words == 2
    assert metrics.hypothesis_words == 2


def test_evaluate_asr_reports_hallucinated_candidate_phrase() -> None:
    """Speech in a candidate-only time interval must be exposed as a hallucination."""
    reference = ReferenceDocument("sha", (ReferenceSpan(0.0, 2.0, "Первый этап.", False),))

    metrics = evaluate_asr(
        reference,
        [
            {"start": 0.0, "end": 2.0, "text": "Первый этап"},
            {"start": 3.0, "end": 5.0, "text": "Лишняя фраза"},
        ],
        (),
        (),
    )

    assert metrics.hallucinated_phrases == ("Лишняя фраза",)


def test_evaluate_asr_reports_candidate_with_uncovered_tail_as_hallucinated() -> None:
    """A long candidate covering reference speech and extra time must remain a hallucination signal."""
    reference = ReferenceDocument("sha", (ReferenceSpan(0.0, 2.0, "Первый этап.", False),))

    metrics = evaluate_asr(
        reference,
        [{"start": 0.0, "end": 5.0, "text": "Первый этап. Лишняя фраза."}],
        (),
        (),
    )

    assert metrics.hallucinated_phrases == ("Первый этап. Лишняя фраза.",)


def test_evaluate_asr_keeps_candidate_text_with_reviewed_and_unclear_overlap() -> None:
    """A mixed reviewed/unclear segment must not discard its reviewed candidate speech."""
    reference = ReferenceDocument(
        "sha",
        (
            ReferenceSpan(0.0, 2.0, "Первый этап.", False),
            ReferenceSpan(2.0, 5.0, "[неразборчиво]", True),
        ),
    )

    metrics = evaluate_asr(
        reference,
        [{"start": 0.0, "end": 5.0, "text": "Первый этап"}],
        (),
        (),
    )

    assert metrics.wer == 0.0
    assert metrics.hypothesis_words == 2


def test_evaluate_asr_reports_missing_reference_phrase() -> None:
    """A reviewed speech interval without candidate text must be exposed as missed."""
    reference = ReferenceDocument(
        "sha",
        (
            ReferenceSpan(0.0, 2.0, "Первый этап.", False),
            ReferenceSpan(2.0, 4.0, "Второй этап.", False),
        ),
    )

    metrics = evaluate_asr(reference, [{"start": 0.0, "end": 2.0, "text": "Первый этап"}], (), ())

    assert metrics.missed_phrases == ("Второй этап.",)


def test_evaluate_asr_counts_mismatch_near_vad_boundary() -> None:
    """A text mismatch within one second of a VAD boundary is a boundary error signal."""
    reference = ReferenceDocument(
        "sha",
        (
            ReferenceSpan(0.0, 2.0, "Первый этап.", False),
            ReferenceSpan(2.0, 4.0, "Второй этап.", False),
        ),
    )

    metrics = evaluate_asr(
        reference,
        [
            {"start": 0.0, "end": 2.0, "text": "Первый этап"},
            {"start": 2.4, "end": 4.0, "text": "Третий этап"},
        ],
        (),
        (2.0,),
    )

    assert metrics.boundary_errors == 1


def test_evaluate_asr_normalizes_fullwidth_underscore_as_word_separator() -> None:
    """NFKC-normalized underscores must separate words like other punctuation."""
    reference = ReferenceDocument("sha", (ReferenceSpan(0.0, 2.0, "Проект＿А", False),))

    metrics = evaluate_asr(reference, [{"start": 0.0, "end": 2.0, "text": "проект а"}], (), ())

    assert metrics.wer == 0.0
    assert metrics.cer == 0.0


def test_evaluate_asr_does_not_count_term_substrings_as_hits() -> None:
    """A vocabulary token must not match only as part of a longer recognized word."""
    reference = ReferenceDocument("sha", (ReferenceSpan(0.0, 2.0, "Задача выполнена.", False),))

    metrics = evaluate_asr(reference, [{"start": 0.0, "end": 2.0, "text": "задача выполнена"}], ("за",), ())

    assert metrics.term_hits == 0
    assert metrics.term_recall == 0.0

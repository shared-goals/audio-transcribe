"""Tests for compare_align.py — pure metrics functions (no whisperx needed)."""

from compare_align import (
    AlignStats,
    build_align_stats,
    collect_all_words,
    compute_word_stats,
    print_comparison,
)

# --- compute_word_stats ---


def test_compute_word_stats_empty():
    stats = compute_word_stats([])
    assert stats["total_words"] == 0
    assert stats["aligned_words"] == 0
    assert stats["missing_words"] == 0
    assert stats["aligned_pct"] == 0.0
    assert stats["durations"] == []
    assert stats["gaps"] == []
    assert stats["overlapping_words"] == 0


def test_compute_word_stats_all_aligned():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
        {"word": "test", "start": 1.1, "end": 1.5},
    ]
    stats = compute_word_stats(words)
    assert stats["total_words"] == 3
    assert stats["aligned_words"] == 3
    assert stats["missing_words"] == 0
    assert stats["aligned_pct"] == 100.0
    assert len(stats["durations"]) == 3
    assert stats["durations"][0] == 0.5
    assert stats["durations"][1] == 0.4  # 1.0 - 0.6
    assert len(stats["gaps"]) == 2
    assert abs(stats["gaps"][0] - 0.1) < 1e-9  # 0.6 - 0.5
    assert abs(stats["gaps"][1] - 0.1) < 1e-9  # 1.1 - 1.0
    assert stats["overlapping_words"] == 0


def test_compute_word_stats_some_missing():
    words: list[dict[str, object]] = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "missing"},  # no start/end
        {"word": "world", "start": 0.6, "end": 1.0},
    ]
    stats = compute_word_stats(words)
    assert stats["total_words"] == 3
    assert stats["aligned_words"] == 2
    assert stats["missing_words"] == 1
    assert abs(stats["aligned_pct"] - 66.666) < 0.1


def test_compute_word_stats_all_missing():
    words = [{"word": "a"}, {"word": "b"}]
    stats = compute_word_stats(words)
    assert stats["total_words"] == 2
    assert stats["aligned_words"] == 0
    assert stats["missing_words"] == 2
    assert stats["aligned_pct"] == 0.0
    assert stats["durations"] == []
    assert stats["gaps"] == []


def test_compute_word_stats_overlapping():
    words = [
        {"word": "a", "start": 0.0, "end": 0.5},
        {"word": "b", "start": 0.3, "end": 0.8},  # overlaps with 'a'
        {"word": "c", "start": 0.9, "end": 1.2},
    ]
    stats = compute_word_stats(words)
    assert stats["overlapping_words"] == 1
    assert stats["gaps"][0] < 0  # 0.3 - 0.5 = -0.2


def test_compute_word_stats_single_word():
    words = [{"word": "solo", "start": 1.0, "end": 1.5}]
    stats = compute_word_stats(words)
    assert stats["total_words"] == 1
    assert stats["aligned_words"] == 1
    assert stats["gaps"] == []
    assert stats["overlapping_words"] == 0
    assert stats["durations"] == [0.5]


# --- collect_all_words ---


def test_collect_all_words_empty():
    assert collect_all_words({}) == []
    assert collect_all_words({"segments": []}) == []


def test_collect_all_words_extracts_from_segments():
    result = {
        "segments": [
            {"text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]},
            {"text": "world", "words": [{"word": "world", "start": 0.6, "end": 1.0}]},
        ]
    }
    words = collect_all_words(result)
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert words[1]["word"] == "world"


def test_collect_all_words_segments_without_words():
    result = {"segments": [{"text": "hello"}]}
    assert collect_all_words(result) == []


# --- build_align_stats ---


def test_build_align_stats():
    result = {
        "segments": [
            {
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ],
            }
        ]
    }
    stats = build_align_stats("test/model", result, time_s=2.5, peak_rss_mb=1500.0)
    assert isinstance(stats, AlignStats)
    assert stats.model_name == "test/model"
    assert stats.total_words == 2
    assert stats.aligned_words == 2
    assert stats.missing_words == 0
    assert stats.aligned_pct == 100.0
    assert stats.time_s == 2.5
    assert stats.peak_rss_mb == 1500.0
    assert abs(stats.mean_duration - 0.45) < 1e-9  # (0.4 + 0.5) / 2
    assert abs(stats.mean_gap - 0.1) < 1e-9  # 0.5 - 0.4


def test_build_align_stats_empty_result():
    stats = build_align_stats("empty/model", {"segments": []}, time_s=0.1, peak_rss_mb=100.0)
    assert stats.total_words == 0
    assert stats.aligned_words == 0
    assert stats.mean_duration == 0.0
    assert stats.max_duration == 0.0


# --- print_comparison ---


def test_print_comparison_renders_table(capsys):
    stats = [
        AlignStats(
            model_name="org/model-a",
            total_words=100,
            aligned_words=95,
            missing_words=5,
            aligned_pct=95.0,
            mean_duration=0.3,
            median_duration=0.25,
            max_duration=1.2,
            mean_gap=0.05,
            median_gap=0.03,
            max_gap=0.5,
            overlapping_words=2,
            time_s=10.0,
            peak_rss_mb=2000.0,
        ),
        AlignStats(
            model_name="org/model-b",
            total_words=100,
            aligned_words=98,
            missing_words=2,
            aligned_pct=98.0,
            mean_duration=0.28,
            median_duration=0.24,
            max_duration=1.0,
            mean_gap=0.04,
            median_gap=0.02,
            max_gap=0.4,
            overlapping_words=1,
            time_s=15.0,
            peak_rss_mb=3000.0,
        ),
    ]
    print_comparison(stats, [])
    out = capsys.readouterr().out
    assert "model-a" in out
    assert "model-b" in out
    assert "95.0%" in out
    assert "98.0%" in out
    assert "Aligned %" in out
    assert "Total words" in out


def test_print_comparison_with_samples(capsys):
    stats = [
        AlignStats(
            model_name="org/model-a",
            total_words=10,
            aligned_words=10,
            missing_words=0,
            aligned_pct=100.0,
            mean_duration=0.3,
            median_duration=0.3,
            max_duration=0.5,
            mean_gap=0.05,
            median_gap=0.05,
            max_gap=0.1,
            overlapping_words=0,
            time_s=5.0,
            peak_rss_mb=1000.0,
        ),
    ]
    sample_segments = [
        (
            "org/model-a",
            [
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.4},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ],
                }
            ],
        ),
    ]
    print_comparison(stats, sample_segments)
    out = capsys.readouterr().out
    assert "Sample Aligned Segments" in out
    assert "model-a" in out
    assert "hello world" in out
    assert "2/2 words" in out

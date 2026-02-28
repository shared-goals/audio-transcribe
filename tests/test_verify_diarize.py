"""Tests for verify_diarize.py — pure metrics functions (no whisperx needed)."""

import pytest

from verify_diarize import (
    DiarizeConfig,
    DiarizeStats,
    SpeakerStats,
    build_diarize_stats,
    compute_segment_durations_per_speaker,
    compute_speaker_durations,
    compute_speaker_segment_counts,
    compute_speaker_word_counts,
    count_speaker_transitions,
    parse_configs,
    print_comparison,
    print_speaker_breakdown,
)

# --- parse_configs ---


def test_parse_configs_basic():
    configs = parse_configs("2-4,2-6,3-6")
    assert len(configs) == 3
    assert configs[0] == DiarizeConfig(min_speakers=2, max_speakers=4)
    assert configs[1] == DiarizeConfig(min_speakers=2, max_speakers=6)
    assert configs[2] == DiarizeConfig(min_speakers=3, max_speakers=6)


def test_parse_configs_single():
    configs = parse_configs("2-4")
    assert len(configs) == 1
    assert configs[0].min_speakers == 2
    assert configs[0].max_speakers == 4


def test_parse_configs_whitespace():
    configs = parse_configs("  2-4 , 3-5  ")
    assert len(configs) == 2
    assert configs[0].label == "2-4"
    assert configs[1].label == "3-5"


def test_parse_configs_empty_parts():
    configs = parse_configs("2-4,,3-5")
    assert len(configs) == 2


def test_parse_configs_invalid_no_dash():
    with pytest.raises(ValueError, match="expected 'min-max'"):
        parse_configs("24")


def test_parse_configs_invalid_too_many_dashes():
    with pytest.raises(ValueError, match="expected exactly one"):
        parse_configs("2-4-6")


def test_parse_configs_invalid_not_int():
    with pytest.raises(ValueError, match="must be integers"):
        parse_configs("a-b")


def test_parse_configs_invalid_min_zero():
    with pytest.raises(ValueError, match="min_speakers must be >= 1"):
        parse_configs("0-4")


def test_parse_configs_invalid_max_less_than_min():
    with pytest.raises(ValueError, match="max_speakers must be >= min_speakers"):
        parse_configs("5-3")


def test_parse_configs_min_equals_max():
    configs = parse_configs("3-3")
    assert len(configs) == 1
    assert configs[0].min_speakers == 3
    assert configs[0].max_speakers == 3


# --- DiarizeConfig ---


def test_diarize_config_label():
    cfg = DiarizeConfig(min_speakers=2, max_speakers=6)
    assert cfg.label == "2-6"


# --- count_speaker_transitions ---


def test_transitions_empty():
    assert count_speaker_transitions([]) == 0


def test_transitions_single_segment():
    assert count_speaker_transitions([{"speaker": "SPEAKER_00"}]) == 0


def test_transitions_same_speaker():
    segs = [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_00"}]
    assert count_speaker_transitions(segs) == 0


def test_transitions_alternating():
    segs = [
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
    ]
    assert count_speaker_transitions(segs) == 3


def test_transitions_skip_unknown():
    segs = [
        {"speaker": "SPEAKER_00"},
        {"speaker": "UNKNOWN"},
        {"speaker": "SPEAKER_01"},
    ]
    assert count_speaker_transitions(segs) == 1


def test_transitions_all_unknown():
    segs = [{"speaker": "UNKNOWN"}, {"speaker": "UNKNOWN"}]
    assert count_speaker_transitions(segs) == 0


def test_transitions_no_speaker_key():
    segs = [{"text": "hello"}, {"text": "world"}]
    assert count_speaker_transitions(segs) == 0


# --- compute_speaker_durations ---


def test_speaker_durations_basic():
    segs = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 5.0},
        {"speaker": "SPEAKER_00", "start": 5.0, "end": 7.0},
    ]
    durs = compute_speaker_durations(segs)
    assert abs(durs["SPEAKER_00"] - 4.0) < 1e-9
    assert abs(durs["SPEAKER_01"] - 3.0) < 1e-9


def test_speaker_durations_empty():
    assert compute_speaker_durations([]) == {}


def test_speaker_durations_unknown():
    segs = [{"start": 0.0, "end": 1.0}]
    durs = compute_speaker_durations(segs)
    assert "UNKNOWN" in durs
    assert abs(durs["UNKNOWN"] - 1.0) < 1e-9


# --- compute_speaker_segment_counts ---


def test_speaker_segment_counts_basic():
    segs = [
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
        {"speaker": "SPEAKER_00"},
    ]
    counts = compute_speaker_segment_counts(segs)
    assert counts["SPEAKER_00"] == 2
    assert counts["SPEAKER_01"] == 1


def test_speaker_segment_counts_empty():
    assert compute_speaker_segment_counts([]) == {}


# --- compute_speaker_word_counts ---


def test_speaker_word_counts_basic():
    segs = [
        {"speaker": "SPEAKER_00", "words": [{"word": "a"}, {"word": "b"}]},
        {"speaker": "SPEAKER_01", "words": [{"word": "c"}]},
        {"speaker": "SPEAKER_00", "words": [{"word": "d"}]},
    ]
    counts = compute_speaker_word_counts(segs)
    assert counts["SPEAKER_00"] == 3
    assert counts["SPEAKER_01"] == 1


def test_speaker_word_counts_no_words():
    segs = [{"speaker": "SPEAKER_00"}]
    counts = compute_speaker_word_counts(segs)
    assert counts["SPEAKER_00"] == 0


def test_speaker_word_counts_empty():
    assert compute_speaker_word_counts([]) == {}


# --- compute_segment_durations_per_speaker ---


def test_segment_durations_per_speaker_basic():
    segs = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
        {"speaker": "SPEAKER_00", "start": 5.0, "end": 8.0},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 5.0},
    ]
    durs = compute_segment_durations_per_speaker(segs)
    assert len(durs["SPEAKER_00"]) == 2
    assert abs(durs["SPEAKER_00"][0] - 2.0) < 1e-9
    assert abs(durs["SPEAKER_00"][1] - 3.0) < 1e-9
    assert len(durs["SPEAKER_01"]) == 1
    assert abs(durs["SPEAKER_01"][0] - 3.0) < 1e-9


def test_segment_durations_per_speaker_empty():
    assert compute_segment_durations_per_speaker([]) == {}


# --- build_diarize_stats ---


def _make_segments() -> list[dict[str, object]]:
    return [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 2.0,
            "text": "hello",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                {"word": "world", "start": 0.6, "end": 1.0, "speaker": "SPEAKER_00"},
            ],
        },
        {
            "speaker": "SPEAKER_01",
            "start": 2.0,
            "end": 5.0,
            "text": "test",
            "words": [
                {"word": "test", "start": 2.0, "end": 2.5, "speaker": "SPEAKER_01"},
            ],
        },
        {
            "speaker": "SPEAKER_00",
            "start": 5.0,
            "end": 7.0,
            "text": "again",
            "words": [
                {"word": "again", "start": 5.0, "end": 5.5, "speaker": "SPEAKER_00"},
            ],
        },
    ]


def test_build_diarize_stats_basic():
    result = {"segments": _make_segments()}
    stats = build_diarize_stats("2-4", result, time_s=10.0, peak_rss_mb=2000.0)
    assert isinstance(stats, DiarizeStats)
    assert stats.config_label == "2-4"
    assert stats.speakers_detected == 2
    assert stats.total_segments == 3
    assert stats.segments_with_speaker == 3
    assert stats.segment_coverage_pct == 100.0
    assert stats.total_words == 4  # 2 + 1 + 1 words across 3 segments
    assert stats.words_with_speaker == 4
    assert stats.word_coverage_pct == 100.0
    assert stats.speaker_transitions == 2
    assert stats.time_s == 10.0
    assert stats.peak_rss_mb == 2000.0


def test_build_diarize_stats_per_speaker():
    result = {"segments": _make_segments()}
    stats = build_diarize_stats("2-4", result, time_s=1.0, peak_rss_mb=100.0)
    assert len(stats.per_speaker) == 2

    sp0 = next(s for s in stats.per_speaker if s.speaker == "SPEAKER_00")
    assert sp0.segments == 2
    assert sp0.words == 3
    assert abs(sp0.duration_s - 4.0) < 1e-9  # 2.0 + 2.0
    assert sp0.mean_segment_duration == 2.0
    assert sp0.median_segment_duration == 2.0

    sp1 = next(s for s in stats.per_speaker if s.speaker == "SPEAKER_01")
    assert sp1.segments == 1
    assert sp1.words == 1
    assert abs(sp1.duration_s - 3.0) < 1e-9


def test_build_diarize_stats_empty():
    stats = build_diarize_stats("2-4", {"segments": []}, time_s=0.1, peak_rss_mb=50.0)
    assert stats.speakers_detected == 0
    assert stats.total_segments == 0
    assert stats.segment_coverage_pct == 0.0
    assert stats.word_coverage_pct == 0.0
    assert stats.speaker_transitions == 0
    assert stats.per_speaker == []


def test_build_diarize_stats_with_unknown():
    segs = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "words": [{"word": "a", "speaker": "SPEAKER_00"}]},
        {"speaker": "UNKNOWN", "start": 2.0, "end": 3.0, "words": [{"word": "b", "speaker": "UNKNOWN"}]},
    ]
    stats = build_diarize_stats("test", {"segments": segs}, time_s=1.0, peak_rss_mb=100.0)
    assert stats.speakers_detected == 1
    assert stats.segments_with_speaker == 1
    assert stats.segment_coverage_pct == 50.0
    assert stats.words_with_speaker == 1
    assert stats.word_coverage_pct == 50.0
    # Only real speakers in per_speaker
    assert len(stats.per_speaker) == 1
    assert stats.per_speaker[0].speaker == "SPEAKER_00"


def test_build_diarize_stats_duration_pct():
    segs = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "words": []},
        {"speaker": "SPEAKER_01", "start": 3.0, "end": 7.0, "words": []},
    ]
    stats = build_diarize_stats("test", {"segments": segs}, time_s=1.0, peak_rss_mb=100.0)
    sp0 = next(s for s in stats.per_speaker if s.speaker == "SPEAKER_00")
    sp1 = next(s for s in stats.per_speaker if s.speaker == "SPEAKER_01")
    # Total duration = 7.0, sp0 = 3.0 (42.86%), sp1 = 4.0 (57.14%)
    assert abs(sp0.duration_pct - 42.857) < 0.1
    assert abs(sp1.duration_pct - 57.143) < 0.1


# --- print_comparison ---


def test_print_comparison_renders_table(capsys):
    stats = [
        DiarizeStats(
            config_label="2-4",
            speakers_detected=3,
            total_segments=20,
            segments_with_speaker=18,
            segment_coverage_pct=90.0,
            total_words=100,
            words_with_speaker=95,
            word_coverage_pct=95.0,
            speaker_transitions=10,
            time_s=5.0,
            peak_rss_mb=2000.0,
        ),
        DiarizeStats(
            config_label="2-6",
            speakers_detected=4,
            total_segments=20,
            segments_with_speaker=20,
            segment_coverage_pct=100.0,
            total_words=100,
            words_with_speaker=100,
            word_coverage_pct=100.0,
            speaker_transitions=15,
            time_s=6.0,
            peak_rss_mb=2100.0,
        ),
    ]
    print_comparison(stats)
    out = capsys.readouterr().out
    assert "2-4" in out
    assert "2-6" in out
    assert "Speakers detected" in out
    assert "90.0%" in out
    assert "100.0%" in out
    assert "Speaker transitions" in out


def test_print_comparison_single_config(capsys):
    stats = [
        DiarizeStats(
            config_label="3-6",
            speakers_detected=5,
            total_segments=30,
            segments_with_speaker=30,
            segment_coverage_pct=100.0,
            total_words=200,
            words_with_speaker=200,
            word_coverage_pct=100.0,
            speaker_transitions=20,
            time_s=8.0,
            peak_rss_mb=2500.0,
        ),
    ]
    print_comparison(stats)
    out = capsys.readouterr().out
    assert "3-6" in out
    assert "Diarization Config Comparison" in out


# --- print_speaker_breakdown ---


def test_print_speaker_breakdown_renders(capsys):
    stats = [
        DiarizeStats(
            config_label="2-4",
            speakers_detected=2,
            total_segments=10,
            segments_with_speaker=10,
            segment_coverage_pct=100.0,
            total_words=50,
            words_with_speaker=50,
            word_coverage_pct=100.0,
            speaker_transitions=5,
            time_s=3.0,
            peak_rss_mb=1500.0,
            per_speaker=[
                SpeakerStats(
                    speaker="SPEAKER_00",
                    segments=6,
                    words=30,
                    duration_s=15.0,
                    duration_pct=60.0,
                    mean_segment_duration=2.5,
                    median_segment_duration=2.3,
                ),
                SpeakerStats(
                    speaker="SPEAKER_01",
                    segments=4,
                    words=20,
                    duration_s=10.0,
                    duration_pct=40.0,
                    mean_segment_duration=2.5,
                    median_segment_duration=2.0,
                ),
            ],
        ),
    ]
    print_speaker_breakdown(stats)
    out = capsys.readouterr().out
    assert "Speaker breakdown: 2-4" in out
    assert "SPEAKER_00" in out
    assert "SPEAKER_01" in out
    assert "60.0%" in out
    assert "40.0%" in out


def test_print_speaker_breakdown_no_speakers(capsys):
    stats = [
        DiarizeStats(
            config_label="2-4",
            speakers_detected=0,
            total_segments=0,
            segments_with_speaker=0,
            segment_coverage_pct=0.0,
            total_words=0,
            words_with_speaker=0,
            word_coverage_pct=0.0,
            speaker_transitions=0,
            time_s=1.0,
            peak_rss_mb=100.0,
        ),
    ]
    print_speaker_breakdown(stats)
    out = capsys.readouterr().out
    assert out == ""  # no output for empty per_speaker

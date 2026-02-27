"""Tests for benchmark.py — rss_mb and print_results (pure/side-effect functions)."""

from benchmark import print_results, rss_mb


def test_rss_mb_returns_positive_float():
    result = rss_mb()
    assert isinstance(result, float)
    assert result > 0


def test_print_results_no_duration(capsys):
    stats = [
        {"stage": "transcribe", "time_s": 12.5, "peak_rss_mb": 3000.0, "delta_rss_mb": 500.0},
        {"stage": "align", "time_s": 5.2, "peak_rss_mb": 3200.0, "delta_rss_mb": 200.0},
    ]
    print_results(stats, None)
    out = capsys.readouterr().out
    assert "transcribe" in out
    assert "align" in out
    assert "Total" in out
    assert "realtime" not in out


def test_print_results_with_duration(capsys):
    stats = [
        {"stage": "transcribe", "time_s": 60.0, "peak_rss_mb": 3000.0, "delta_rss_mb": 500.0}
    ]
    print_results(stats, 120.0)
    out = capsys.readouterr().out
    assert "realtime" in out
    assert "0.50x" in out  # 60s / 120s = 0.50x realtime


def test_print_results_total_time(capsys):
    stats = [
        {"stage": "transcribe", "time_s": 30.0, "peak_rss_mb": 3000.0, "delta_rss_mb": 400.0},
        {"stage": "align", "time_s": 10.0, "peak_rss_mb": 3100.0, "delta_rss_mb": 100.0},
    ]
    print_results(stats, None)
    out = capsys.readouterr().out
    assert "40.00" in out  # total = 30 + 10


def test_print_results_audio_duration_shown(capsys):
    stats = [{"stage": "transcribe", "time_s": 30.0, "peak_rss_mb": 3000.0, "delta_rss_mb": 500.0}]
    print_results(stats, 600.0)  # 10-minute audio
    out = capsys.readouterr().out
    assert "600" in out
    assert "10.0 min" in out


def test_print_results_empty_stats(capsys):
    print_results([], None)
    out = capsys.readouterr().out
    assert "Total" in out
    assert "0.00" in out  # total time = 0

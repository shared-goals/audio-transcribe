import json
import time

from scripts.asr_log_review import main, summarize


def _event(kind, **fields):
    return {"ts": int(time.time() * 1000), "event": kind, **fields}


def _write(tmp_path, events, rotations=()):
    path = tmp_path / "asr-events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    for index, extra in enumerate(rotations, start=1):
        (tmp_path / f"asr-events.jsonl.{index}").write_text(
            "\n".join(json.dumps(e) for e in extra) + "\n", encoding="utf-8"
        )
    return path


def test_clean_window_reports_ok(tmp_path, capsys):
    path = _write(
        tmp_path,
        [
            _event("service_start", bind="0.0.0.0:8788"),
            _event("worker_start", socket="/tmp/asr.sock"),
            _event("transcribe", ok=True, duration_ms=1800, timings_ms={"whisper": 1700}),
            _event("transcribe", ok=True, duration_ms=2100, timings_ms={"whisper": 1900}),
            _event("transcribe", ok=True, duration_ms=2000, timings_ms={"whisper": 1800}),
        ],
    )

    assert main(["--log", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ASR-сервис: ok")
    assert "3 запросов" in out


def test_failures_and_restarts_are_reported(tmp_path, capsys):
    path = _write(
        tmp_path,
        [
            _event("worker_dropped", reason="Broken pipe"),
            _event("transcribe", ok=False, error="worker io error: Broken pipe", duration_ms=900),
            _event("transcribe", ok=False, error="timeout", duration_ms=60_000),
            _event("transcribe", ok=False, error="timeout", duration_ms=60_000),
            _event("unauthorized", path="/v1/audio/transcriptions"),
        ],
    )

    assert main(["--log", str(path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["requests"] == 3
    assert summary["failed"] == 3
    assert any("доля ошибок" in item for item in summary["anomalies"])
    assert any("воркер перезапускался" in item for item in summary["anomalies"])
    assert any("токену" in item for item in summary["anomalies"])

    assert main(["--log", str(path), "--fail-on-anomaly"]) == 2
    assert "ВНИМАНИЕ" in capsys.readouterr().out


def test_rotations_are_read_and_broken_lines_counted(tmp_path, capsys):
    path = _write(
        tmp_path,
        [_event("transcribe", ok=True, duration_ms=1500)],
        rotations=([_event("transcribe", ok=True, duration_ms=1600)],),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    summary = summarize(
        [
            _event("transcribe", ok=True, duration_ms=1500),
            _event("transcribe", ok=True, duration_ms=1600),
        ],
        1,
        24,
    )
    assert summary["requests"] == 2
    assert summary["broken_lines"] == 1
    assert any("битых строк" in item for item in summary["anomalies"])

    assert main(["--log", str(path), "--hours", "24"]) == 0
    assert "битых строк" in capsys.readouterr().out


def test_slow_whisper_passes_are_counted(tmp_path, capsys):
    path = _write(
        tmp_path,
        [
            _event("transcribe", ok=True, duration_ms=12000, timings_ms={"whisper": 11800}),
            _event("transcribe", ok=True, duration_ms=1800, timings_ms={"whisper": 1700}),
        ],
    )

    assert main(["--log", str(path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["slow_whisper_passes"] == 1

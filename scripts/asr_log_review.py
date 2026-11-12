#!/usr/bin/env python3
"""Review the JSON-lines event log of the ASR service.

The service writes one line per request and lifecycle event and rotates the file
by size, so this script only reads. It answers three questions: did the service
answer, how fast, and did anything look wrong.

Usage:
    scripts/asr_log_review.py [--log ~/.audio-transcribe/asr-events.jsonl] [--hours 24]
                              [--json] [--fail-on-anomaly]

Exit codes: 0 = report written (anomalies included), 2 = anomalies found and
--fail-on-anomaly was set. The daily cron uses the default (0) and delivers the
printed summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path.home() / ".audio-transcribe" / "asr-events.jsonl"

# Thresholds are deliberately coarse: they exist to catch a broken service, not
# to grade latency. Measured 2026-11-12 on chez-mac: 1.8 s for 9 s of audio.
ERROR_RATE_LIMIT = 0.05
P95_LIMIT_MS = 15_000
MIN_REQUESTS_FOR_RATE = 3


def log_files(path: Path) -> list[Path]:
    """Current file plus its rotations, oldest first."""
    if not path.exists():
        return []
    rotations = sorted(
        (p for p in path.parent.glob(f"{path.name}.*") if p.name[len(path.name) + 1 :].isdigit()),
        key=lambda p: int(p.name[len(path.name) + 1 :]),
        reverse=True,
    )
    return [*rotations, path]


def read_events(path: Path, since_ms: int) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    broken = 0
    for file in log_files(path):
        try:
            with file.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        broken += 1
                        continue
                    if not isinstance(event, dict):
                        broken += 1
                        continue
                    if int(event.get("ts") or 0) >= since_ms:
                        events.append(event)
        except OSError:
            broken += 1
    return events, broken


def iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, UTC).astimezone().isoformat(timespec="seconds")


def summarize(events: list[dict[str, Any]], broken: int, hours: int) -> dict[str, Any]:
    requests = [e for e in events if e.get("event") == "transcribe"]
    ok = [e for e in requests if e.get("ok")]
    failed = [e for e in requests if not e.get("ok")]
    durations = sorted(int(e.get("duration_ms") or 0) for e in ok)
    cold = sum(1 for e in ok if int((e.get("timings_ms") or {}).get("whisper") or 0) >= 4000)
    starts = [e for e in events if e.get("event") == "service_start"]
    worker_starts = [e for e in events if e.get("event") == "worker_start"]
    dropped = [e for e in events if e.get("event") == "worker_dropped"]
    unauthorized = [e for e in events if e.get("event") == "unauthorized"]
    degraded = [e for e in ok if e.get("degraded")]

    def pct(part: float, whole: float) -> float:
        return round(part / whole, 4) if whole else 0.0

    error_rate = pct(len(failed), len(requests))
    p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))] if durations else 0
    anomalies: list[str] = []
    if len(requests) >= MIN_REQUESTS_FOR_RATE and error_rate > ERROR_RATE_LIMIT:
        anomalies.append(f"доля ошибок {error_rate:.0%} за {hours} ч ({len(failed)} из {len(requests)})")
    if p95 > P95_LIMIT_MS:
        anomalies.append(f"p95 {p95} мс выше {P95_LIMIT_MS} мс")
    if dropped:
        anomalies.append(
            f"воркер перезапускался {len(dropped)} раз: {', '.join(sorted({str(e.get('reason')) for e in dropped}))}"
        )
    if unauthorized:
        anomalies.append(f"отказов по токену: {len(unauthorized)}")
    if broken:
        anomalies.append(f"битых строк в логе: {broken}")
    if not requests:
        anomalies.append(f"за {hours} ч нет ни одного запроса к сервису")

    return {
        "window_hours": hours,
        "requests": len(requests),
        "ok": len(ok),
        "failed": len(failed),
        "degraded": len(degraded),
        "error_rate": error_rate,
        "duration_ms": {
            "p50": int(statistics.median(durations)) if durations else 0,
            "p95": p95,
            "max": durations[-1] if durations else 0,
        },
        "slow_whisper_passes": cold,
        "service_starts": len(starts),
        "worker_starts": len(worker_starts),
        "worker_drops": len(dropped),
        "unauthorized": len(unauthorized),
        "broken_lines": broken,
        "anomalies": anomalies,
    }


def render(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if not summary["anomalies"]:
        lines.append(
            f"ASR-сервис: ok — {summary['requests']} запросов за {summary['window_hours']} ч, "
            f"p50 {summary['duration_ms']['p50']} мс, p95 {summary['duration_ms']['p95']} мс, "
            f"ошибок {summary['failed']}, деградаций {summary['degraded']}"
        )
        return "\n".join(lines)

    lines.append(f"ASR-сервис: ВНИМАНИЕ за {summary['window_hours']} ч")
    lines.extend(f"- {item}" for item in summary["anomalies"])
    lines.append(
        f"- запросов {summary['requests']}, ошибок {summary['failed']}, "
        f"p50 {summary['duration_ms']['p50']} мс, p95 {summary['duration_ms']['p95']} мс, "
        f"макс {summary['duration_ms']['max']} мс"
    )
    recent_failures = [e for e in events if e.get("event") == "transcribe" and not e.get("ok")][-3:]
    for event in recent_failures:
        lines.append(f"- {iso(int(event.get('ts') or 0))}: {event.get('error')}")
    lines.append(f"- лог: {summary['log_file']} (плюс ротации .1…n)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review the ASR service event log.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    parser.add_argument("--fail-on-anomaly", action="store_true", help="exit 2 when anomalies are found")
    parser.add_argument(
        "--only-anomalies",
        action="store_true",
        help="print nothing when the window is clean (daily cron delivery)",
    )
    args = parser.parse_args(argv)

    since_ms = int((datetime.now(UTC) - timedelta(hours=args.hours)).timestamp() * 1000)
    events, broken = read_events(args.log, since_ms)
    summary = summarize(events, broken, int(args.hours))
    summary["log_file"] = str(args.log)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif not (args.only_anomalies and not summary["anomalies"]):
        print(render(summary, events))
    return 2 if (args.fail_on_anomaly and summary["anomalies"]) else 0


if __name__ == "__main__":
    sys.exit(main())

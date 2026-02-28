"""ETA estimation using linear regression on historical run data."""

from __future__ import annotations

from dataclasses import dataclass

from audio_transcribe.models import RunRecord


@dataclass
class EstimateResult:
    """Result of an ETA estimation."""

    eta_s: float
    confident: bool  # True if R^2 >= 0.7
    sample_size: int


def estimate_stage(
    stage: str, audio_duration_s: float, history: list[RunRecord]
) -> EstimateResult | None:
    """Estimate stage duration from historical data using linear regression.

    Returns None if fewer than 3 relevant data points exist.
    """
    # Filter records that have this stage
    points: list[tuple[float, float]] = []
    for r in history:
        if stage in r.stages:
            points.append((r.input.duration_s, r.stages[stage].time_s))

    if len(points) < 3:
        return None

    # Simple linear regression: time = a * duration + b
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return None

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n

    eta = max(0.0, a * audio_duration_s + b)

    # R^2 coefficient of determination
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys, strict=True))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return EstimateResult(eta_s=round(eta, 1), confident=r_squared >= 0.7, sample_size=n)

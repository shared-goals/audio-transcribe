"""Smart recommendation engine based on historical run performance."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from audio_transcribe.models import RunRecord

_MIN_TOTAL_RUNS = 5


@dataclass
class Recommendation:
    """Recommendation for optimal pipeline settings."""

    backend: str | None
    speedup_factor: float | None = None
    tips: list[str] = field(default_factory=list)


def recommend(duration_s: float, history: list[RunRecord]) -> Recommendation:
    """Suggest optimal settings based on historical performance.

    Requires at least 5 total runs to make a recommendation.
    Compares backends by average realtime_ratio (lower is better).
    """
    if len(history) < _MIN_TOTAL_RUNS:
        return Recommendation(backend=None, tips=["Not enough history — run at least 5 transcriptions first."])

    # Group by backend, compute average realtime_ratio
    backend_ratios: dict[str, list[float]] = defaultdict(list)
    for r in history:
        backend_ratios[r.config.backend].append(r.realtime_ratio)

    if not backend_ratios:
        return Recommendation(backend=None, tips=["No backend data available."])

    # Find best backend (lowest average realtime_ratio = fastest)
    backend_avg: dict[str, float] = {
        backend: sum(ratios) / len(ratios) for backend, ratios in backend_ratios.items()
    }

    best_backend = min(backend_avg, key=lambda b: backend_avg[b])
    tips: list[str] = []

    # Calculate speedup vs second-best if multiple backends
    speedup_factor: float | None = None
    if len(backend_avg) > 1:
        sorted_backends = sorted(backend_avg, key=lambda b: backend_avg[b])
        second_best = sorted_backends[1]
        if backend_avg[best_backend] > 0:
            speedup_factor = round(backend_avg[second_best] / backend_avg[best_backend], 1)
            tips.append(
                f"{best_backend} is ~{speedup_factor}x faster than {second_best} "
                f"(avg ratio: {backend_avg[best_backend]:.3f} vs {backend_avg[second_best]:.3f})"
            )

    # Add tips based on data patterns
    best_count = len(backend_ratios[best_backend])
    if best_count < 3:
        tips.append(f"Only {best_count} runs with {best_backend} — more data will improve accuracy.")

    return Recommendation(backend=best_backend, speedup_factor=speedup_factor, tips=tips)

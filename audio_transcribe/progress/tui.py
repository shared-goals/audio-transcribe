"""Rich TUI progress reporter for interactive terminal use."""

from __future__ import annotations

import resource
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageStart


def _current_rss_mb() -> float:
    """Get current process RSS in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KB
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


class TuiReporter:
    """Rich TUI reporter for interactive terminal progress display."""

    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._stages: list[dict[str, Any]] = []

    def on_pipeline_start(self, event: PipelineStart) -> None:
        """Display pipeline start banner."""
        self._console.print(f"\n[bold blue]Processing:[/] {event.file}")
        duration_str = _format_time(event.duration_s) if event.duration_s > 0 else "unknown"
        self._console.print(f"  Duration: {duration_str}")
        self._stages = []

    def on_stage_start(self, event: StageStart) -> None:
        """Display stage start with optional ETA."""
        eta_str = f" (ETA: {_format_time(event.eta_s)})" if event.eta_s is not None else ""
        self._console.print(f"\n[bold yellow]>>>[/] {event.stage}{eta_str}")

    def on_stage_complete(self, event: StageComplete) -> None:
        """Display stage completion with timing."""
        self._stages.append({"stage": event.stage, "time_s": event.time_s, "peak_rss_mb": event.peak_rss_mb})
        extra_str = ""
        if event.extra:
            parts = [f"{k}={v}" for k, v in event.extra.items()]
            extra_str = f" ({', '.join(parts)})"
        self._console.print(
            f"[bold green]<<<[/] {event.stage}: {_format_time(event.time_s)}, "
            f"{event.peak_rss_mb:.0f} MB RSS{extra_str}"
        )

    def on_pipeline_complete(self, event: PipelineComplete) -> None:
        """Display pipeline summary table."""
        self._console.print()

        table = Table(title="Pipeline Summary", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Time", justify="right")
        table.add_column("Peak RSS", justify="right")

        for s in self._stages:
            table.add_row(s["stage"], _format_time(s["time_s"]), f"{s['peak_rss_mb']:.0f} MB")

        table.add_row("[bold]Total[/]", f"[bold]{_format_time(event.total_time_s)}[/]", "")
        self._console.print(table)

        self._console.print(f"\n[bold green]Done![/] Output: {event.output}")
        if event.transcript:
            self._console.print(f"  Transcript: {event.transcript}")

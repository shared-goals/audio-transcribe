"""Rich TUI progress reporter for interactive terminal use."""

from __future__ import annotations

import resource
import sys
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

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
    """Rich TUI reporter using rich.live.Live for interactive terminal progress display."""

    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._live: Live | None = None
        self._pipeline_start: PipelineStart | None = None
        self._stages_done: list[dict[str, Any]] = []
        self._current_stage: str | None = None
        self._current_eta: float | None = None

    def _make_renderable(self) -> Group:
        """Build the current live renderable from pipeline state."""
        lines: list[Any] = []

        # Header: filename + config
        if self._pipeline_start:
            fname = Path(self._pipeline_start.file).name
            dur_s = self._pipeline_start.duration_s
            dur_str = f"  [dim]{_format_time(dur_s)}[/]" if dur_s > 0 else ""
            lines.append(Text.from_markup(f"[bold blue]Processing:[/] {fname}{dur_str}"))
            cfg = self._pipeline_start.config
            model = cfg.get("model", "")
            backend = cfg.get("backend", "")
            lines.append(Text.from_markup(f"[dim]  {backend} / {model}[/]"))
            lines.append(Text(""))  # blank line separator

        # Completed stages — green checkmark + timing + memory
        for s in self._stages_done:
            rss = f"  [dim]{s['peak_rss_mb']:.0f} MB[/]" if s.get("peak_rss_mb") else ""
            lines.append(
                Text.from_markup(
                    f"[green]✓[/] [cyan]{s['stage']:<12}[/] [bold]{_format_time(s['time_s'])}[/]{rss}"
                )
            )

        # Current stage — spinner with optional ETA
        if self._current_stage:
            eta_str = f"  [dim]ETA: {_format_time(self._current_eta)}[/]" if self._current_eta is not None else ""
            spinner = Spinner("dots", text=f" [yellow]{self._current_stage}[/]{eta_str}")
            lines.append(spinner)

        return Group(*lines)

    def on_pipeline_start(self, event: PipelineStart) -> None:
        """Start the Live display."""
        self._pipeline_start = event
        self._stages_done = []
        self._current_stage = None
        self._current_eta = None
        self._live = Live(
            self._make_renderable(),
            console=self._console,
            refresh_per_second=10,
        )
        self._live.start()

    def on_stage_start(self, event: StageStart) -> None:
        """Update live display to show stage as running."""
        self._current_stage = event.stage
        self._current_eta = event.eta_s
        if self._live:
            self._live.update(self._make_renderable())

    def on_stage_complete(self, event: StageComplete) -> None:
        """Mark stage as done and update display."""
        self._stages_done.append(
            {
                "stage": event.stage,
                "time_s": event.time_s,
                "peak_rss_mb": event.peak_rss_mb,
            }
        )
        self._current_stage = None
        self._current_eta = None
        if self._live:
            self._live.update(self._make_renderable())

    def on_pipeline_complete(self, event: PipelineComplete) -> None:
        """Stop live display and print summary table."""
        # Final render with all stages done
        if self._live:
            self._live.update(self._make_renderable())
            self._live.stop()
            self._live = None

        # Summary table
        self._console.print()
        table = Table(title="Pipeline Summary", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Time", justify="right")
        table.add_column("Peak RSS", justify="right")

        for s in self._stages_done:
            table.add_row(s["stage"], _format_time(s["time_s"]), f"{s['peak_rss_mb']:.0f} MB")

        table.add_row("[bold]Total[/]", f"[bold]{_format_time(event.total_time_s)}[/]", "")
        self._console.print(table)

        self._console.print(f"\n[bold green]Done![/] Output: {event.output}")
        if event.transcript:
            self._console.print(f"  Transcript: {event.transcript}")

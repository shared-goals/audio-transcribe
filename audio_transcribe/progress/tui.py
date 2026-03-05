"""Rich TUI progress reporter for interactive terminal use."""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from audio_transcribe.progress.events import PipelineComplete, PipelineStart, StageComplete, StageError, StageStart

# Maximum number of pipeline stages (preprocess, transcribe, align, diarize, correct, format)
_MAX_STAGES = 6


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


class _LiveDisplay:
    """Rich renderable that re-evaluates on every render tick for live elapsed updates."""

    def __init__(self, reporter: TuiReporter) -> None:
        self._reporter = reporter

    def __rich_console__(self, console: Any, options: Any) -> Any:
        yield self._reporter._build_display()


class TuiReporter:
    """Rich TUI reporter using rich.live.Live for interactive terminal progress display."""

    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._live: Live | None = None
        self._pipeline_start: PipelineStart | None = None
        self._stages_done: list[dict[str, Any]] = []
        self._current_stage: str | None = None
        self._current_eta: float | None = None
        self._stage_start_time: float = 0.0
        self._spinner = Spinner("dots")

    def _build_display(self) -> Group:
        """Build the current live renderable from pipeline state."""
        lines: list[RenderableType] = []

        # Header: filename + config (always 3 lines for stable height)
        if self._pipeline_start:
            fname = Path(self._pipeline_start.file).name
            dur_s = self._pipeline_start.duration_s
            dur_str = f"  [dim]{_format_time(dur_s)}[/]" if dur_s > 0 else ""
            lines.append(Text.from_markup(f"[bold blue]Processing:[/] {fname}{dur_str}"))
            cfg = self._pipeline_start.config
            model = cfg.get("model", "")
            backend = cfg.get("backend", "")
            lines.append(Text.from_markup(f"[dim]  {backend} / {model}[/]"))
            lines.append(Text(""))
        else:
            lines.extend([Text(""), Text(""), Text("")])

        # Completed stages
        for s in self._stages_done:
            rss = f"  [dim]{s['peak_rss_mb']:.0f} MB[/]" if s.get("peak_rss_mb") else ""
            lines.append(
                Text.from_markup(
                    f"[green]✓[/] [cyan]{s['stage']:<12}[/] [bold]{_format_time(s['time_s'])}[/]{rss}"
                )
            )

        # Current stage — persistent spinner + live elapsed
        if self._current_stage:
            elapsed = time.time() - self._stage_start_time
            elapsed_str = f"  [dim]{_format_time(elapsed)}[/]"
            eta_str = f"  [dim]ETA: {_format_time(self._current_eta)}[/]" if self._current_eta is not None else ""
            self._spinner.text = Text.from_markup(f" [yellow]{self._current_stage}[/]{elapsed_str}{eta_str}")
            lines.append(self._spinner)

        # Pad to fixed height to prevent Rich Live from leaving artifacts
        # Total fixed height = 3 (header) + _MAX_STAGES (stage rows)
        n_header = 3 if self._pipeline_start else 0
        n_done = len(self._stages_done)
        n_active = 1 if self._current_stage else 0
        target_height = 3 + _MAX_STAGES
        padding = target_height - n_header - n_done - n_active
        for _ in range(max(0, padding)):
            lines.append(Text(""))

        return Group(*lines)

    def on_pipeline_start(self, event: PipelineStart) -> None:
        """Start the Live display."""
        self._pipeline_start = event
        self._stages_done = []
        self._current_stage = None
        self._current_eta = None
        self._spinner = Spinner("dots")
        self._live = Live(
            _LiveDisplay(self),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.start()

    def on_stage_start(self, event: StageStart) -> None:
        """Update live display to show stage as running."""
        self._current_stage = event.stage
        self._current_eta = event.eta_s
        self._stage_start_time = time.time()
        self._spinner = Spinner("dots")

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

    def on_stage_error(self, event: StageError) -> None:
        """Mark stage as failed and update display."""
        self._stages_done.append(
            {"stage": event.stage, "time_s": event.time_s, "peak_rss_mb": 0, "error": event.error}
        )
        self._current_stage = None

    def on_pipeline_complete(self, event: PipelineComplete) -> None:
        """Stop live display and print summary table."""
        if self._live:
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

        if event.output and event.output != "<stdout>":
            self._console.print(f"\n[bold green]Done![/] Output: {event.output}")
        else:
            self._console.print("\n[bold green]Done![/]")
        if event.transcript:
            self._console.print(f"  Transcript: {event.transcript}")

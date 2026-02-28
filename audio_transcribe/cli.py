"""CLI entry point for audio-transcribe."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="audio-transcribe", help="Local audio transcription pipeline.")

_DEFAULT_HISTORY = Path.home() / ".audio-transcribe" / "history.json"
_DEFAULT_CORRECTIONS = Path.home() / ".audio-transcribe" / "corrections.yaml"


@app.command()
def process(
    audio_file: Path = typer.Argument(..., help="Input audio file (WAV, M4A, MP3)"),
    language: str = typer.Option("ru", "-l", "--language", help="Language code"),
    model: str = typer.Option("large-v3", "-m", "--model", help="Whisper model size"),
    backend: str = typer.Option(
        "whisperx",
        "--backend",
        help="Transcription backend: whisperx (CPU), mlx, mlx-vad (Apple Silicon)",
    ),
    min_speakers: int = typer.Option(2, "--min-speakers", help="Minimum speakers for diarization"),
    max_speakers: int = typer.Option(6, "--max-speakers", help="Maximum speakers for diarization"),
    align_model: Optional[str] = typer.Option(None, "--align-model", help="Custom alignment model HF repo"),
    no_align: bool = typer.Option(False, "--no-align", help="Skip alignment stage"),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip diarization stage"),
    full: bool = typer.Option(False, "--full", help="Include diarization (slower). Default is fast pass."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory for meeting notes"),
    transcript: Optional[Path] = typer.Option(None, "--transcript", help="Output Markdown transcript path"),
    json_mode: bool = typer.Option(False, "--json", help="Machine-readable JSON-lines output (no TUI)"),
) -> None:
    """Fast pass: transcribe + align → meeting note. Use --full to include diarization."""
    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    from audio_transcribe.pipeline import run_pipeline
    from audio_transcribe.progress.json_reporter import JsonReporter
    from audio_transcribe.progress.tui import TuiReporter
    from audio_transcribe.stages.format import format_meeting_note, format_transcript
    from audio_transcribe.stats.store import StatsStore

    store = StatsStore(_DEFAULT_HISTORY)
    reporter = JsonReporter() if json_mode or not sys.stdout.isatty() else TuiReporter()

    # Fast pass by default; --full enables diarization; --no-diarize also forces skip
    skip_diarize = no_diarize or not full

    result = run_pipeline(
        audio_file=str(audio_file),
        language=language,
        model=model,
        backend=backend,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        align_model=align_model,
        no_align=no_align,
        no_diarize=skip_diarize,
        corrections_path=str(_DEFAULT_CORRECTIONS),
        reporter=reporter,
        stats_store=store,
    )

    # Determine output directory
    output_dir = output if output is not None else Path(".")
    stem = audio_file.stem

    # Store raw JSON in .audio-data/
    audio_data_dir = output_dir / ".audio-data"
    audio_data_dir.mkdir(parents=True, exist_ok=True)
    json_path = audio_data_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Format and write meeting note
    relative_json = f".audio-data/{stem}.json"
    markdown = format_meeting_note(result, audio_data_path=relative_json)
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    # Optional legacy transcript
    if transcript:
        transcript.write_text(format_transcript(result), encoding="utf-8")


@app.command()
def diarize(
    meeting: Path = typer.Argument(..., help="Path to meeting markdown file"),
    min_speakers: int = typer.Option(1, "--min-speakers"),
    max_speakers: int = typer.Option(6, "--max-speakers"),
    force: bool = typer.Option(False, "--force", help="Re-diarize even if already diarized"),
    audio_file: Optional[str] = typer.Option(None, "--audio-file", help="Override audio file path"),
) -> None:
    """Add speaker diarization to an existing meeting note."""
    from audio_transcribe.stages.diarize_update import diarize_and_update

    if not meeting.exists():
        typer.echo(f"Error: file not found: {meeting}", err=True)
        raise typer.Exit(1)

    try:
        diarize_and_update(
            meeting,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            force=force,
            audio_file_override=audio_file,
        )
        typer.echo(f"Diarized: {meeting} (reanalyze: true)")
    except RuntimeError as e:
        typer.echo(f"Error: {e}. Use --force to re-diarize.", err=True)
        raise typer.Exit(1) from e


@app.command()
def stats(
    last: int = typer.Option(10, "--last", "-n", help="Show last N runs"),
    clear: bool = typer.Option(False, "--clear", help="Clear all history"),
) -> None:
    """View historical run statistics."""
    from rich.console import Console
    from rich.table import Table

    from audio_transcribe.stats.store import StatsStore

    store = StatsStore(_DEFAULT_HISTORY)

    if clear:
        store.clear()
        typer.echo("History cleared.")
        return

    records = store.last(last)
    if not records:
        typer.echo("No history yet. Run 'audio-transcribe process' on an audio file first.")
        return

    console = Console()
    table = Table(title=f"Last {len(records)} runs", show_header=True)
    table.add_column("Date", style="cyan")
    table.add_column("File")
    table.add_column("Duration", justify="right")
    table.add_column("Total time", justify="right")
    table.add_column("RT ratio", justify="right")
    table.add_column("Model")
    table.add_column("Backend")

    for r in records:
        date = r.id[:16].replace("T", " ")
        dur = f"{r.input.duration_s:.0f}s"
        total = f"{r.total_time_s:.1f}s"
        ratio = f"{r.realtime_ratio:.2f}x"
        table.add_row(date, Path(r.input.file).name, dur, total, ratio, r.config.model, r.config.backend)

    console.print(table)


@app.command()
def recommend(
    audio_file: Path = typer.Argument(..., help="Audio file to analyze"),
) -> None:
    """Suggest optimal settings based on historical performance."""
    import subprocess

    from audio_transcribe.stats.recommender import recommend as _recommend
    from audio_transcribe.stats.store import StatsStore

    # Get audio duration via ffprobe
    duration_s = 0.0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)],
            capture_output=True, text=True, check=False,
        )
        duration_s = float(result.stdout.strip())
    except (ValueError, FileNotFoundError):
        pass

    store = StatsStore(_DEFAULT_HISTORY)
    history = store.load()
    rec = _recommend(duration_s=duration_s, history=history)

    if rec.backend is None:
        typer.echo("Not enough history for recommendations.")
        for tip in rec.tips:
            typer.echo(f"  • {tip}")
        return

    typer.echo(f"\nRecommended backend: {rec.backend}")
    if rec.speedup_factor:
        typer.echo(f"  Speedup: {rec.speedup_factor}x vs next best")
    for tip in rec.tips:
        typer.echo(f"  • {tip}")


@app.command()
def learn(
    corrected_md: Path = typer.Argument(..., help="Corrected Markdown transcript"),
    original: Optional[Path] = typer.Option(None, "--original", help="Original JSON output (auto-detected if omitted)"),
) -> None:
    """Learn corrections from an edited transcript."""
    import json
    import re

    import yaml

    from audio_transcribe.stages.correct import learn_corrections

    if not corrected_md.exists():
        typer.echo(f"Error: file not found: {corrected_md}", err=True)
        raise typer.Exit(1)

    # Strip timestamps/speaker labels from corrected markdown to get plain text
    md_text = corrected_md.read_text(encoding="utf-8")
    corrected_lines: list[str] = []
    for line in md_text.splitlines():
        # Match lines like "[00:12] Speaker A: some text"
        m = re.match(r"^\[\d+:\d+(?::\d+)?\]\s+[^:]+:\s+(.+)$", line)
        if m:
            corrected_lines.append(m.group(1).strip())

    if not corrected_lines:
        typer.echo("No transcript lines found in the Markdown file.")
        raise typer.Exit(1)

    # Find original JSON to diff against
    original_path = original
    if original_path is None:
        # Try to find matching JSON next to the markdown
        candidate = corrected_md.with_suffix(".json")
        if not candidate.exists():
            typer.echo(
                "Could not find original JSON. Pass --original path/to/result.json",
                err=True,
            )
            raise typer.Exit(1)
        original_path = candidate

    data = json.loads(original_path.read_text(encoding="utf-8"))
    original_lines = [seg.get("text", "").strip() for seg in data.get("segments", [])]

    learned = learn_corrections(original_lines, corrected_lines)
    if not learned:
        typer.echo("No corrections found — transcripts appear identical.")
        return

    typer.echo(f"\nFound {len(learned)} correction(s):")
    for wrong, correct in learned.items():
        typer.echo(f"  {wrong!r:30s} → {correct!r}")

    if not typer.confirm("\nAdd these to corrections.yaml?"):
        return

    # Load existing corrections and merge
    _DEFAULT_CORRECTIONS.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if _DEFAULT_CORRECTIONS.exists():
        existing = yaml.safe_load(_DEFAULT_CORRECTIONS.read_text(encoding="utf-8")) or {}

    raw_subs = existing.get("substitutions", {})
    subs: dict[str, str] = raw_subs if isinstance(raw_subs, dict) else {}
    subs.update(learned)
    existing["substitutions"] = subs

    _DEFAULT_CORRECTIONS.write_text(yaml.dump(existing, allow_unicode=True), encoding="utf-8")
    typer.echo(f"Saved to {_DEFAULT_CORRECTIONS}")


if __name__ == "__main__":
    app()

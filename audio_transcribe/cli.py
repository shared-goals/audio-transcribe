"""CLI entry point for audio-transcribe."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import typer

# Suppress harmless warnings from torchcodec and Lightning before any ML imports
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec", category=UserWarning)
warnings.filterwarnings("ignore", message=r"(?s).*Lightning automatically upgraded", category=UserWarning)

# Suppress noisy tqdm progress bars from huggingface_hub file downloads
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

app = typer.Typer(name="audio-transcribe", help="Local audio transcription pipeline.", add_completion=False)

speakers_app = typer.Typer(help="Manage known speaker voice embeddings.")
app.add_typer(speakers_app, name="speakers")
worker_app = typer.Typer(help="Manage the durable unattended transcription queue.")
app.add_typer(worker_app, name="worker")

_DEFAULT_HISTORY = Path.home() / ".audio-transcribe" / "history.json"
_DEFAULT_CORRECTIONS = Path.home() / ".audio-transcribe" / "corrections.yaml"
_HF_TOKEN_CACHE = Path.home() / ".cache" / "huggingface" / "token"
_DEFAULT_STATE = Path.home() / ".audio-transcribe"
_DEFAULT_QUEUE = _DEFAULT_STATE / "jobs.sqlite3"


def _version_callback(value: bool) -> None:
    """Print the package version and exit before command initialization."""
    if value:
        from audio_transcribe import __version__

        typer.echo(__version__)
        raise typer.Exit()


def _sync_hf_token() -> None:
    """Sync HF_TOKEN between environment and ~/.cache/huggingface/token.

    If HF_TOKEN is set in env but the cache file is missing, write it.
    If HF_TOKEN is not set but the cache file exists, load it into env.
    """
    env_token = os.environ.get("HF_TOKEN", "")
    if env_token:
        if not _HF_TOKEN_CACHE.exists():
            _HF_TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _HF_TOKEN_CACHE.write_text(env_token, encoding="utf-8")
    else:
        if _HF_TOKEN_CACHE.is_file():
            cached = _HF_TOKEN_CACHE.read_text(encoding="utf-8").strip()
            if cached:
                os.environ["HF_TOKEN"] = cached


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Local audio transcription pipeline."""
    from audio_transcribe.log import configure

    configure(verbose=verbose)
    _sync_hf_token()


@app.command()
def process(
    audio_file: Path = typer.Argument(..., help="Input audio file (WAV, M4A, MP3)"),
    language: Optional[str] = typer.Option(None, "-l", "--language", help="Language code (profile default: ru)"),
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Whisper model size"),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="mlx-vad: Apple Silicon + VAD (fastest) | mlx: Apple Silicon | whisperx: CPU (slowest)",
    ),
    min_speakers: Optional[int] = typer.Option(None, "--min-speakers", help="Minimum speakers for diarization"),
    max_speakers: Optional[int] = typer.Option(None, "--max-speakers", help="Maximum speakers for diarization"),
    align_model: Optional[str] = typer.Option(None, "--align-model", help="Custom alignment model HF repo"),
    no_align: bool = typer.Option(False, "--no-align", help="Skip alignment stage"),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip diarization stage"),
    full: bool = typer.Option(False, "--full", help="Include diarization (slower). Default is fast pass."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory for meeting notes"),
    transcript: Optional[Path] = typer.Option(None, "--transcript", help="Output Markdown transcript path"),
    json_mode: bool = typer.Option(False, "--json", help="Machine-readable JSON-lines output (no TUI)"),
    profile: str = typer.Option("default", "--profile", help="Named profile from config.toml"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Configuration TOML path"),
    template: Optional[Path] = typer.Option(None, "--template", help="Meeting-note template path"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume matching interrupted runs"),
    force: bool = typer.Option(False, "--force", help="Ignore matching checkpoints"),
    restart_from: Optional[str] = typer.Option(None, "--restart-from", help="Invalidate this stage and later stages"),
    keep_workdir: bool = typer.Option(False, "--keep-workdir", help="Keep normalized audio after success"),
) -> None:
    """Fast pass: transcribe + align → meeting note. Use --full to include diarization."""
    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    from audio_transcribe.config import load_profile
    from audio_transcribe.pipeline import run_pipeline
    from audio_transcribe.progress.json_reporter import JsonReporter
    from audio_transcribe.progress.tui import TuiReporter
    from audio_transcribe.stages.format import format_meeting_note, format_transcript
    from audio_transcribe.stats.store import StatsStore
    from audio_transcribe.util import atomic_write_text

    try:
        selected = load_profile(profile, config_file)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    language = language or selected.language
    model = model or selected.model
    backend = backend or selected.backend
    min_speakers = min_speakers if min_speakers is not None else selected.min_speakers
    max_speakers = max_speakers if max_speakers is not None else selected.max_speakers
    align_model = align_model or selected.align_model
    template = template or (Path(selected.template).expanduser() if selected.template else None)

    store = StatsStore(_DEFAULT_HISTORY)
    reporter = JsonReporter() if json_mode or not sys.stdout.isatty() else TuiReporter()

    # Fast pass by default; --full enables diarization; --no-diarize also forces skip
    skip_diarize = no_diarize or (not full and selected.no_diarize)

    result = run_pipeline(
        audio_file=str(audio_file),
        language=language,
        model=model,
        backend=backend,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        align_model=align_model,
        no_align=no_align or selected.no_align,
        no_diarize=skip_diarize,
        corrections_path=str(_DEFAULT_CORRECTIONS),
        reporter=reporter,
        stats_store=store,
        estimator_history=store.load(),
        resume=resume,
        force=force,
        restart_from=restart_from,
        keep_workdir=keep_workdir,
        state_dir=str(_DEFAULT_STATE),
    )

    # Determine output directory
    output_dir = output if output is not None else Path(".")
    stem = audio_file.stem

    # Store raw JSON in .audio-data/
    audio_data_dir = output_dir / ".audio-data"
    audio_data_dir.mkdir(parents=True, exist_ok=True)
    json_path = audio_data_dir / f"{stem}.json"
    atomic_write_text(json_path, json.dumps(result, ensure_ascii=False, indent=2))

    # Format and write meeting note
    relative_json = f".audio-data/{stem}.json"
    markdown = format_meeting_note(result, audio_data_path=relative_json)
    if template:
        from audio_transcribe.templates import render_template

        markdown = render_template(template, markdown, stem, format_transcript(result))
    md_path = output_dir / f"{stem}.md"
    atomic_write_text(md_path, markdown)

    # Optional legacy transcript
    if transcript:
        atomic_write_text(transcript, format_transcript(result))

    if not json_mode and sys.stdout.isatty():
        typer.echo(f"Meeting note: {md_path}", err=True)
        if transcript:
            typer.echo(f"Transcript:   {transcript}", err=True)


@app.command()
def doctor(
    backend: str = typer.Option("mlx-vad", "--backend"),
    json_mode: bool = typer.Option(False, "--json"),
    state_dir: Path = typer.Option(_DEFAULT_STATE, "--state-dir"),
) -> None:
    """Check runtime dependencies, state storage, queue integrity, and ML extras."""
    from audio_transcribe.doctor import checks_json, run_checks

    try:
        checks = run_checks(state_dir, backend)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if json_mode:
        typer.echo(json.dumps(checks_json(checks), ensure_ascii=False))
    else:
        for check in checks:
            typer.echo(f"{'OK' if check.ok else 'FAIL':4} {check.name}: {check.detail}")
    if any(not check.ok and check.name != "HF token" for check in checks):
        raise typer.Exit(1)


@app.command()
def batch(
    inputs: list[Path] = typer.Argument(..., help="Audio files to enqueue"),
    output: Path = typer.Option(Path("."), "-o", "--output"),
    profile: str = typer.Option("default", "--profile"),
    queue: Path = typer.Option(_DEFAULT_QUEUE, "--queue"),
) -> None:
    """Idempotently enqueue several recordings."""
    from audio_transcribe.worker import JobQueue

    jobs = JobQueue(queue)
    for path in inputs:
        try:
            job_id = jobs.enqueue(path, output, profile)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"queued {job_id}: {path}")


@worker_app.command("enqueue")
def worker_enqueue(
    audio_file: Path = typer.Argument(...),
    output: Path = typer.Option(Path("."), "-o", "--output"),
    profile: str = typer.Option("default", "--profile"),
    queue: Path = typer.Option(_DEFAULT_QUEUE, "--queue"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
) -> None:
    """Add one recording to the durable queue."""
    from audio_transcribe.worker import JobQueue

    try:
        job_id = JobQueue(queue).enqueue(audio_file, output, profile, max_attempts)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(job_id))


@worker_app.command("status")
def worker_status(
    queue: Path = typer.Option(_DEFAULT_QUEUE, "--queue"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Show queue health and recent jobs."""
    from audio_transcribe.worker import JobQueue

    jobs = JobQueue(queue)
    if json_mode:
        typer.echo(jobs.as_json())
        return
    health = jobs.health()
    typer.echo(f"health: {'ok' if health['ok'] else 'degraded'}; jobs: {health['jobs']}")
    for job in jobs.list(limit=20):
        typer.echo(f"{job.id:4} {job.status:8} attempt {job.attempts}/{job.max_attempts} {job.input_path}")


@worker_app.command("retry")
def worker_retry(
    job_id: int = typer.Argument(...),
    queue: Path = typer.Option(_DEFAULT_QUEUE, "--queue"),
) -> None:
    """Move a failed or dead-letter job back to the queue."""
    from audio_transcribe.worker import JobQueue

    try:
        JobQueue(queue).retry(job_id)
    except KeyError as exc:
        typer.echo(f"Error: job not found: {job_id}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"queued {job_id}")


@worker_app.command("run")
def worker_run(
    queue: Path = typer.Option(_DEFAULT_QUEUE, "--queue"),
    once: bool = typer.Option(False, "--once", help="Exit when no ready job remains"),
    watch: Optional[Path] = typer.Option(None, "--watch", help="Scan a directory for stable audio files"),
    output: Path = typer.Option(Path("."), "-o", "--output"),
    profile: str = typer.Option("default", "--profile"),
    stable_for: float = typer.Option(30.0, "--stable-for"),
    poll: float = typer.Option(10.0, "--poll"),
) -> None:
    """Process queued recordings with retries and dead-letter handling."""
    import time

    from audio_transcribe.worker import JobQueue, stable_audio_files

    jobs = JobQueue(queue)
    jobs.recover_stale()
    while True:
        if watch:
            for audio in stable_audio_files(watch, stable_for):
                jobs.enqueue(audio, output, profile)
        job = jobs.claim()
        if job is None:
            if once:
                return
            time.sleep(max(poll, 0.1))
            continue
        try:
            # Reuse the CLI command so worker and interactive behavior stay identical.
            process(
                audio_file=Path(job.input_path),
                output=Path(job.output_dir),
                profile=job.profile,
                language=None,
                model=None,
                backend=None,
                min_speakers=None,
                max_speakers=None,
                align_model=None,
                no_align=False,
                no_diarize=False,
                full=False,
                transcript=None,
                json_mode=True,
                config_file=None,
                template=None,
                resume=True,
                force=False,
                restart_from=None,
                keep_workdir=False,
            )
        except (Exception, SystemExit) as exc:
            status = jobs.fail(job.id, str(exc))
            typer.echo(f"job {job.id} {status}: {exc}", err=True)
        else:
            jobs.complete(job.id)


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
        date = r.id[:16].replace("T", " ") if r.id[:4].isdigit() else r.id[:12]
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
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_file),
            ],
            capture_output=True,
            text=True,
            check=False,
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


@app.command()
def identify(
    meeting: Path = typer.Argument(..., help="Path to meeting markdown file"),
    threshold: float = typer.Option(0.5, "--threshold", help="Cosine distance threshold for matching"),
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir"),
    audio_file: Optional[str] = typer.Option(None, "--audio-file", help="Override audio file path"),
) -> None:
    """Auto-identify speakers using voice embedding database."""
    from audio_transcribe.speakers.database import SpeakerDB
    from audio_transcribe.stages.identify import identify_speakers

    if not meeting.exists():
        typer.echo(f"Error: file not found: {meeting}", err=True)
        raise typer.Exit(1)

    db = SpeakerDB(db_dir)
    result = identify_speakers(meeting, db, threshold=threshold, audio_file_override=audio_file)

    if result.matched:
        for sid, name in result.matched.items():
            typer.echo(f"  Matched {sid} → [[{name}]]")
        typer.echo(f"Updated: {meeting} (reanalyze: true)")
    else:
        typer.echo("No speakers matched.")

    if result.unmatched:
        typer.echo(f"  Unmatched: {', '.join(result.unmatched)}")


@app.command()
def update(
    meeting: Path = typer.Argument(..., help="Path to meeting markdown file"),
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir"),
) -> None:
    """Apply speaker mapping from frontmatter and enroll new voices."""
    from audio_transcribe.speakers.database import SpeakerDB
    from audio_transcribe.stages.update import update_meeting

    if not meeting.exists():
        typer.echo(f"Error: file not found: {meeting}", err=True)
        raise typer.Exit(1)

    db = SpeakerDB(db_dir)
    update_meeting(meeting, db)
    typer.echo(f"Updated: {meeting} (reanalyze: true)")


@speakers_app.command("list")
def speakers_list(
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir"),
) -> None:
    """List all known speakers in the voice embedding database."""
    from audio_transcribe.speakers.database import SpeakerDB

    db = SpeakerDB(db_dir)
    known = db.list_speakers()

    if not known:
        typer.echo("No speakers enrolled yet.")
        return

    for s in known:
        name = s["name"]
        samples = s.get("samples", 0)
        last_seen = s.get("last_seen", "unknown")
        typer.echo(f"  {name} ({samples} samples, last seen {last_seen})")


@speakers_app.command("forget")
def speakers_forget(
    name: str = typer.Argument(..., help="Speaker name to remove"),
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir"),
) -> None:
    """Remove a speaker from the voice embedding database."""
    from audio_transcribe.speakers.database import SpeakerDB

    db = SpeakerDB(db_dir)
    if not db.has_speaker(name):
        typer.echo(f"Speaker not found: {name}")
        return

    db.forget(name)
    typer.echo(f"Removed: {name}")


@app.command("self-update")
def self_update() -> None:
    """Force an immediate upgrade to the latest version."""
    from audio_transcribe.update import force_upgrade

    typer.echo("Upgrading audio-transcribe...")
    if force_upgrade():
        typer.echo("Done.")
    else:
        typer.echo("Upgrade failed. Check your network connection.", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

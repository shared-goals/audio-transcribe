"""CLI entry point for audio-transcribe."""

import typer

app = typer.Typer(name="audio-transcribe", help="Local audio transcription pipeline.")


@app.command()
def process() -> None:
    """Run the full transcription pipeline."""
    typer.echo("Not implemented yet.")


if __name__ == "__main__":
    app()

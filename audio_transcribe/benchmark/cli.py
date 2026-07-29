"""Explicit command surface for private, checksum-bound ASR benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Annotated, Any

import typer

from audio_transcribe import preflight
from audio_transcribe.benchmark.asr_metrics import evaluate_asr
from audio_transcribe.benchmark.diarization_metrics import evaluate_diarization
from audio_transcribe.benchmark.models import (
    AsrCandidate,
    AsrMetrics,
    DiarizationCandidate,
    DiarizationMetrics,
    EvaluatedAsrRun,
    EvaluatedDiarizationRun,
    RunManifest,
    SpeakerTurn,
    validate_asr_metrics,
    validate_diarization_metrics,
)
from audio_transcribe.benchmark.reference import parse_reference, prepare_reference
from audio_transcribe.benchmark.report import rank_asr, write_report
from audio_transcribe.benchmark.speaker_reference import load_speaker_reference
from audio_transcribe.benchmark.store import write_diarization_run, write_run
from audio_transcribe.pipeline import _probe_duration
from audio_transcribe.stages.transcribe import MlxVadOptions
from audio_transcribe.util import atomic_write_text

benchmark_app = typer.Typer(help="Run private, reproducible transcription benchmarks.")
_CANDIDATES = ("baseline", "vocabulary", "russian", "winner-context")
_DIARIZATION_CANDIDATES = {
    "pyannote-3.1": DiarizationCandidate(
        "pyannote-3.1",
        "pyannote/speaker-diarization-3.1",
        False,
    ),
    "community-1": DiarizationCandidate(
        "community-1",
        "pyannote/speaker-diarization-community-1",
        False,
    ),
    "community-1-exclusive": DiarizationCandidate(
        "community-1-exclusive",
        "pyannote/speaker-diarization-community-1",
        True,
    ),
}


def run_asr_candidate(audio: Path, candidate: AsrCandidate, align_model: str | None) -> Any:
    """Load the optional ML ASR runner only when an ASR benchmark is executed."""
    try:
        from audio_transcribe.benchmark.asr_runner import run_asr_candidate as execute
    except ModuleNotFoundError as exc:
        raise RuntimeError("ASR benchmarks require optional ML packages; run `uv sync --extra ml`.") from exc
    return execute(audio, candidate, align_model)


def run_diarization_candidate(
    audio: Path,
    aligned_result: dict[str, Any],
    token: str,
    candidate: DiarizationCandidate,
) -> Any:
    """Load torch and pyannote only when a diarization benchmark is executed."""
    try:
        from audio_transcribe.benchmark.diarization_runner import run_diarization_candidate as execute
    except ModuleNotFoundError as exc:
        raise RuntimeError("Diarization benchmarks require optional ML packages; run `uv sync --extra ml`.") from exc
    return execute(audio, aligned_result, token, candidate)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(dataset: Path) -> tuple[Path, dict[str, Any], Path, Path]:
    root = dataset.expanduser().resolve()
    manifest_path = root / "benchmark.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("benchmark dataset is missing benchmark.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("benchmark.json must contain an object")
    reference = root / "reference-transcript.md"
    if manifest.get("reference_status") != "accepted" or not reference.is_file():
        raise ValueError("benchmark requires an accepted reference")
    reference_sha256 = manifest.get("reference_sha256")
    if not isinstance(reference_sha256, str) or _sha256(reference) != reference_sha256:
        raise ValueError("benchmark reference checksum mismatch")
    vocabulary = root / "vocabulary.txt"
    vocabulary_sha256 = manifest.get("vocabulary_sha256")
    if manifest.get("vocabulary_status") != "accepted" or not vocabulary.is_file():
        raise ValueError("benchmark requires an accepted vocabulary")
    if not isinstance(vocabulary_sha256, str) or _sha256(vocabulary) != vocabulary_sha256:
        raise ValueError("benchmark vocabulary checksum mismatch")
    return root, manifest, reference, vocabulary


def _audio_path(manifest: dict[str, Any]) -> Path:
    raw_path = manifest.get("audio_path")
    expected_sha256 = manifest.get("audio_sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
        raise ValueError("benchmark audio definition is missing")
    audio = Path(raw_path).expanduser().resolve()
    if not audio.is_file() or _sha256(audio) != expected_sha256:
        raise ValueError("benchmark audio checksum mismatch")
    return audio


def _vocabulary(path: Path) -> str:
    terms = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return ", ".join(terms)


def _candidate_definition(
    dataset: Path,
    manifest: dict[str, Any],
    vocabulary_path: Path,
    candidate_name: str,
) -> tuple[AsrCandidate, str | None, str | None]:
    if candidate_name not in _CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate_name}")
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("benchmark model definitions are missing")
    selected_name = candidate_name
    condition_on_previous_text = False
    if candidate_name == "winner-context":
        metrics_path = dataset / "metrics.json"
        if not metrics_path.is_file():
            raise ValueError("winner-context requires an existing benchmark report")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        ranking = metrics.get("ranking") if isinstance(metrics, dict) else None
        winner_name = ranking.get("default_winner") if isinstance(ranking, dict) else None
        if not isinstance(winner_name, str) or winner_name not in {"baseline", "vocabulary", "russian"}:
            raise ValueError("winner-context requires an initial ASR winner")
        results = metrics.get("results") if isinstance(metrics, dict) else None
        if not isinstance(results, list) or not results:
            raise ValueError("winner-context requires checksum-bound ranking evidence")
        expected = (
            manifest["audio_sha256"],
            manifest["reference_sha256"],
            manifest["vocabulary_sha256"],
        )
        winner_verified = False
        for result in results:
            result_manifest = result.get("manifest") if isinstance(result, dict) else None
            if not isinstance(result_manifest, dict):
                raise ValueError("winner-context requires checksum-bound ranking evidence")
            result_checksums = (
                result_manifest.get("source_sha256"),
                result_manifest.get("reference_sha256"),
                result_manifest.get("vocabulary_sha256"),
            )
            if result_checksums != expected:
                raise ValueError("winner-context evidence checksum mismatch")
            result_candidate = result.get("candidate")
            manifest_candidate = result_manifest.get("candidate_name")
            if (
                not isinstance(result_candidate, str)
                or not result_candidate
                or not isinstance(manifest_candidate, str)
                or not manifest_candidate
            ):
                raise ValueError("winner-context winner evidence is invalid")
            result_sha256 = result.get("result_sha256")
            result_path = dataset / "runs" / result_candidate / "result.json"
            if (
                result_candidate not in _CANDIDATES
                or not isinstance(result_sha256, str)
                or not result_path.is_file()
                or _sha256(result_path) != result_sha256
            ):
                raise ValueError("winner-context result checksum mismatch")
            if result_candidate == winner_name or manifest_candidate == winner_name:
                metrics_payload = result.get("metrics")
                if (
                    result_candidate != winner_name
                    or manifest_candidate != winner_name
                    or result.get("status") != "completed"
                    or result.get("error") is not None
                    or result_manifest.get("status") != "completed"
                    or result_manifest.get("error") is not None
                    or not isinstance(metrics_payload, dict)
                ):
                    raise ValueError("winner-context winner evidence is invalid")
                try:
                    validate_asr_metrics(_asr_metrics(metrics_payload))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("winner-context winner evidence is invalid") from exc
                winner_verified = True
        if not winner_verified:
            raise ValueError("winner-context winner evidence is invalid")
        selected_name = winner_name
        condition_on_previous_text = True
    model_key = "russian" if selected_name == "russian" else "baseline"
    model = models.get(model_key)
    if not isinstance(model, str) or not model:
        raise ValueError(f"benchmark model definition is missing: {model_key}")
    prompt = _vocabulary(vocabulary_path) if selected_name == "vocabulary" else None
    options = MlxVadOptions(
        chunk_size_s=30,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=prompt or None,
    )
    revision = models.get(f"{model_key}_revision")
    align_model = manifest.get("align_model")
    return (
        AsrCandidate(candidate_name, model, options),
        revision if isinstance(revision, str) else None,
        align_model if isinstance(align_model, str) else None,
    )


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    value = completed.stdout.strip()
    return value if value else "unavailable"


def _probe_audio_duration(audio: Path) -> float:
    duration_s = _probe_duration(str(audio))
    if not isinstance(duration_s, (int, float)) or not isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("benchmark source audio duration is unavailable")
    return float(duration_s)


def _manifest_for_run(
    dataset_manifest: dict[str, Any],
    candidate: AsrCandidate,
    model_revision: str | None,
    stage_times_s: dict[str, float],
    peak_rss_mb: float,
    status: str,
    error: str | None,
) -> RunManifest:
    return RunManifest(
        candidate_name=candidate.name,
        source_sha256=str(dataset_manifest["audio_sha256"]),
        reference_sha256=str(dataset_manifest["reference_sha256"]),
        git_commit=_git_commit(),
        model=candidate.model,
        model_revision=model_revision,
        options=asdict(candidate.options),
        hardware={"machine": platform.machine(), "platform": platform.platform()},
        stage_times_s=stage_times_s,
        peak_rss_mb=peak_rss_mb,
        status=status,
        error=error,
        vocabulary_sha256=str(dataset_manifest["vocabulary_sha256"]),
    )


def _run_manifest(payload: dict[str, Any]) -> RunManifest:
    return RunManifest(**payload)


def _asr_metrics(payload: dict[str, Any]) -> AsrMetrics:
    converted = dict(payload)
    converted["hallucinated_phrases"] = tuple(converted["hallucinated_phrases"])
    converted["missed_phrases"] = tuple(converted["missed_phrases"])
    return AsrMetrics(**converted)


def _evaluated_runs(dataset: Path, dataset_manifest: dict[str, Any]) -> tuple[EvaluatedAsrRun, ...]:
    runs_dir = dataset / "runs"
    values: list[EvaluatedAsrRun] = []
    for candidate_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        evaluation_path = candidate_dir / "evaluation.json"
        manifest_path = candidate_dir / "manifest.json"
        result_path = candidate_dir / "result.json"
        if not evaluation_path.is_file() or not manifest_path.is_file() or not result_path.is_file():
            continue
        manifest = _run_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        expected = (
            dataset_manifest["audio_sha256"],
            dataset_manifest["reference_sha256"],
            dataset_manifest["vocabulary_sha256"],
        )
        run_checksums = (manifest.source_sha256, manifest.reference_sha256, manifest.vocabulary_sha256)
        evaluation_checksums = (
            evaluation.get("source_sha256"),
            evaluation.get("reference_sha256"),
            evaluation.get("vocabulary_sha256"),
        )
        if run_checksums != expected or evaluation_checksums != expected:
            raise ValueError(f"benchmark run checksum mismatch: {manifest.candidate_name}")
        result_sha256 = _sha256(result_path)
        if evaluation.get("result_sha256") != result_sha256:
            raise ValueError(f"ASR result checksum mismatch: {manifest.candidate_name}")
        metrics_payload = evaluation.get("metrics")
        values.append(
            EvaluatedAsrRun(
                manifest,
                _asr_metrics(metrics_payload) if isinstance(metrics_payload, dict) else None,
                evaluation.get("rtf"),
                result_sha256,
            )
        )
    if not values:
        raise ValueError("benchmark has no evaluated ASR runs")
    return tuple(values)


def _diarization_inputs(
    root: Path,
    manifest: dict[str, Any],
    selected_asr_name: str,
) -> tuple[Path, dict[str, Any], str, Path, str, tuple[SpeakerTurn, ...], float]:
    reference_path = root / "reference-speakers.json"
    if manifest.get("speaker_reference_status") != "accepted" or not reference_path.is_file():
        raise ValueError("benchmark requires an accepted speaker reference")
    speaker_sha256 = manifest.get("speaker_reference_sha256")
    if not isinstance(speaker_sha256, str) or _sha256(reference_path) != speaker_sha256:
        raise ValueError("speaker reference checksum mismatch")
    raw_aligned_path = manifest.get("aligned_asr_path")
    aligned_sha256 = manifest.get("aligned_asr_sha256")
    if not isinstance(raw_aligned_path, str) or not isinstance(aligned_sha256, str):
        raise ValueError("winning aligned ASR definition is missing")
    aligned_path = Path(raw_aligned_path).expanduser().resolve()
    if not aligned_path.is_file() or _sha256(aligned_path) != aligned_sha256:
        raise ValueError("aligned ASR checksum mismatch")
    expected_aligned_path = (root / "runs" / selected_asr_name / "result.json").resolve()
    if aligned_path != expected_aligned_path:
        raise ValueError("aligned ASR does not belong to the current winning ASR candidate")
    reference = load_speaker_reference(reference_path)
    if reference.source_sha256 != manifest["audio_sha256"] or reference.aligned_asr_sha256 != aligned_sha256:
        raise ValueError("speaker reference evidence checksum mismatch")
    aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
    if not isinstance(aligned, dict):
        raise ValueError("aligned ASR result must contain an object")
    return (
        _audio_path(manifest),
        aligned,
        aligned_sha256,
        reference_path,
        speaker_sha256,
        reference.turns,
        reference.audio_duration_s,
    )


def _diarization_metrics(payload: dict[str, Any]) -> DiarizationMetrics:
    converted = dict(payload)
    mapping = converted.get("label_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("invalid diarization label mapping")
    metrics = DiarizationMetrics(**converted)
    validate_diarization_metrics(metrics)
    return metrics


def _evaluated_diarization_runs(
    dataset: Path,
    manifest: dict[str, Any],
    selected_asr_name: str,
) -> tuple[EvaluatedDiarizationRun, ...]:
    root = dataset / "runs" / "diarization"
    if not root.is_dir():
        return ()
    _audio, _aligned, aligned_sha256, _reference, speaker_sha256, _turns, _duration = _diarization_inputs(
        dataset,
        manifest,
        selected_asr_name,
    )
    values: list[EvaluatedDiarizationRun] = []
    for candidate_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        result_path = candidate_dir / "result.json"
        evaluation_path = candidate_dir / "evaluation.json"
        if not result_path.is_file() or not evaluation_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        result_sha256 = _sha256(result_path)
        expected = (manifest["audio_sha256"], aligned_sha256, speaker_sha256)
        if (
            not isinstance(result, dict)
            or not isinstance(evaluation, dict)
            or (
                result.get("source_sha256"),
                result.get("aligned_asr_sha256"),
                result.get("speaker_reference_sha256"),
            )
            != expected
            or (
                evaluation.get("source_sha256"),
                evaluation.get("aligned_asr_sha256"),
                evaluation.get("speaker_reference_sha256"),
            )
            != expected
            or evaluation.get("result_sha256") != result_sha256
        ):
            if isinstance(evaluation, dict) and evaluation.get("result_sha256") != result_sha256:
                raise ValueError(f"diarization result checksum mismatch: {candidate_dir.name}")
            raise ValueError(f"diarization evidence checksum mismatch: {candidate_dir.name}")
        candidate = result.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("invalid diarization candidate evidence")
        metrics = evaluation.get("metrics")
        values.append(
            EvaluatedDiarizationRun(
                candidate_name=str(candidate["name"]),
                source_sha256=str(result["source_sha256"]),
                aligned_asr_sha256=str(result["aligned_asr_sha256"]),
                speaker_reference_sha256=str(result["speaker_reference_sha256"]),
                model=str(candidate["model_id"]),
                exclusive=bool(candidate["exclusive"]),
                time_s=float(result["time_s"]),
                peak_rss_mb=float(result["peak_rss_mb"]),
                status=str(result["status"]),
                error=result.get("error"),
                metrics=_diarization_metrics(metrics) if isinstance(metrics, dict) else None,
            )
        )
    return tuple(values)


@benchmark_app.command("prepare")
def prepare(
    audio: Annotated[Path, typer.Argument(help="Private source audio.")],
    current_note: Annotated[Path, typer.Option("--current-note", help="Current transcript note used only as a draft.")],
    dataset: Annotated[Path, typer.Option("--dataset", help="Private benchmark dataset directory.")],
) -> None:
    """Prepare an editable reference draft without changing the source note."""
    try:
        draft_path, manifest_path = prepare_reference(audio, current_note, dataset)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Draft: {draft_path}")
    typer.echo(f"Manifest: {manifest_path}")


@benchmark_app.command("run-asr")
def run_asr(
    dataset: Annotated[Path, typer.Option("--dataset", help="Private benchmark dataset directory.")],
    candidate: Annotated[str, typer.Option("--candidate", help="baseline|vocabulary|russian|winner-context")],
) -> None:
    """Run and persist one isolated ASR candidate."""
    try:
        root, dataset_manifest, _reference, vocabulary_path = _load_dataset(dataset)
        audio = _audio_path(dataset_manifest)
        definition, revision, align_model = _candidate_definition(root, dataset_manifest, vocabulary_path, candidate)
        preflight_result = preflight.check(str(audio), backend="mlx-vad", skip_diarize=True)
        if not preflight_result.ok:
            raise ValueError("\n".join(preflight_result.errors))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    result = run_asr_candidate(audio, definition, align_model)
    manifest = _manifest_for_run(
        dataset_manifest,
        definition,
        revision,
        result.stage_times_s,
        result.peak_rss_mb,
        result.status,
        result.error,
    )
    try:
        persisted_result = dict(result.result or {"segments": []})
        persisted_result["vad_boundaries"] = list(result.vad_boundaries)
        write_run(root, manifest, persisted_result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if result.status == "failed":
        typer.echo(f"{candidate}: failed: {result.error}", err=True)
        return
    typer.echo(f"{candidate}: completed")


@benchmark_app.command("evaluate-asr")
def evaluate_asr_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Private benchmark dataset directory.")],
) -> None:
    """Evaluate all persisted ASR runs against the accepted reference."""
    try:
        root, dataset_manifest, reference_path, vocabulary_path = _load_dataset(dataset)
        audio = _audio_path(dataset_manifest)
        duration_s = _probe_audio_duration(audio)
        reference = parse_reference(reference_path)
        vocabulary = tuple(
            line.strip() for line in vocabulary_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        evaluated = 0
        for candidate_dir in sorted(path for path in (root / "runs").iterdir() if path.is_dir()):
            manifest_path = candidate_dir / "manifest.json"
            result_path = candidate_dir / "result.json"
            if not manifest_path.is_file() or not result_path.is_file():
                continue
            manifest = _run_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
            if (
                manifest.source_sha256 != dataset_manifest["audio_sha256"]
                or manifest.reference_sha256 != dataset_manifest["reference_sha256"]
                or manifest.vocabulary_sha256 != dataset_manifest["vocabulary_sha256"]
            ):
                raise ValueError("benchmark checksum mismatch")
            metrics: AsrMetrics | None = None
            rtf: float | None = None
            if manifest.status == "completed":
                result = json.loads(result_path.read_text(encoding="utf-8"))
                segments = result.get("segments", [])
                boundaries = tuple(float(value) for value in result.get("vad_boundaries", ()))
                metrics = evaluate_asr(reference, segments, vocabulary, boundaries)
                rtf = sum(manifest.stage_times_s.values()) / duration_s
            payload = {
                "source_sha256": dataset_manifest["audio_sha256"],
                "reference_sha256": dataset_manifest["reference_sha256"],
                "vocabulary_sha256": dataset_manifest["vocabulary_sha256"],
                "result_sha256": _sha256(result_path),
                "metrics": asdict(metrics) if metrics is not None else None,
                "rtf": rtf,
            }
            atomic_write_text(
                candidate_dir / "evaluation.json",
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )
            evaluated += 1
        if evaluated == 0:
            raise ValueError("benchmark has no persisted ASR runs")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Evaluated ASR runs: {evaluated}")


@benchmark_app.command("run-diarization")
def run_diarization_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Private benchmark dataset directory.")],
    candidate: Annotated[
        str,
        typer.Option("--candidate", help="pyannote-3.1|community-1|community-1-exclusive"),
    ],
) -> None:
    """Run and persist one isolated two-speaker diarization candidate."""
    try:
        root, manifest, _reference, _vocabulary = _load_dataset(dataset)
        asr_runs = _evaluated_runs(root, manifest)
        selected_asr_name = rank_asr(asr_runs, "baseline").default_winner
        audio, aligned, aligned_sha256, _speaker_path, speaker_sha256, _turns, _duration = _diarization_inputs(
            root,
            manifest,
            selected_asr_name,
        )
        definition = _DIARIZATION_CANDIDATES.get(candidate)
        if definition is None:
            raise ValueError(f"unknown diarization candidate: {candidate}")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required for diarization")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    try:
        result = run_diarization_candidate(audio, aligned, token, definition)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    try:
        write_diarization_run(
            root,
            result,
            source_sha256=str(manifest["audio_sha256"]),
            aligned_asr_sha256=aligned_sha256,
            speaker_reference_sha256=speaker_sha256,
        )
    except (OSError, ValueError, TypeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if result.status == "failed":
        typer.echo(f"{candidate}: failed: {result.error}", err=True)
        return
    typer.echo(f"{candidate}: completed")


@benchmark_app.command("evaluate-diarization")
def evaluate_diarization_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Private benchmark dataset directory.")],
) -> None:
    """Evaluate all persisted diarization candidates against the accepted speaker reference."""
    try:
        root, manifest, _reference, _vocabulary = _load_dataset(dataset)
        asr_runs = _evaluated_runs(root, manifest)
        selected_asr_name = rank_asr(asr_runs, "baseline").default_winner
        (
            _audio,
            _aligned,
            aligned_sha256,
            _speaker_path,
            speaker_sha256,
            reference_turns,
            _duration,
        ) = _diarization_inputs(root, manifest, selected_asr_name)
        runs_root = root / "runs" / "diarization"
        evaluated = 0
        for candidate_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            result_path = candidate_dir / "result.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            expected = (manifest["audio_sha256"], aligned_sha256, speaker_sha256)
            if (
                not isinstance(result, dict)
                or (
                    result.get("source_sha256"),
                    result.get("aligned_asr_sha256"),
                    result.get("speaker_reference_sha256"),
                )
                != expected
            ):
                raise ValueError(f"diarization evidence checksum mismatch: {candidate_dir.name}")
            metrics: DiarizationMetrics | None = None
            if result.get("status") == "completed":
                turns = tuple(SpeakerTurn(**turn) for turn in result.get("turns", ()))
                words = result.get("word_assignments", ())
                if not isinstance(words, list):
                    raise ValueError("invalid diarization word assignments")
                metrics = evaluate_diarization(reference_turns, turns, words)
            payload = {
                "source_sha256": manifest["audio_sha256"],
                "aligned_asr_sha256": aligned_sha256,
                "speaker_reference_sha256": speaker_sha256,
                "result_sha256": _sha256(result_path),
                "metrics": asdict(metrics) if metrics else None,
            }
            atomic_write_text(
                candidate_dir / "evaluation.json",
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )
            evaluated += 1
        if evaluated == 0:
            raise ValueError("benchmark has no persisted diarization runs")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Evaluated diarization runs: {evaluated}")


@benchmark_app.command("report")
def report(
    dataset: Annotated[Path, typer.Option("--dataset", help="Private benchmark dataset directory.")],
) -> None:
    """Aggregate evaluated runs and write forecasts and recommendations."""
    try:
        root, dataset_manifest, _reference, _vocabulary_path = _load_dataset(dataset)
        _audio_path(dataset_manifest)
        asr_runs = _evaluated_runs(root, dataset_manifest)
        selected_asr = rank_asr(asr_runs, "baseline").default_winner
        diarization_runs = _evaluated_diarization_runs(root, dataset_manifest, selected_asr)
        if diarization_runs:
            _audio, _aligned, _aligned_sha, _speaker, _speaker_sha, _turns, duration_s = _diarization_inputs(
                root,
                dataset_manifest,
                selected_asr,
            )
            metrics_path, report_path = write_report(
                root,
                asr_runs,
                diarization_results=diarization_runs,
                selected_asr_name=selected_asr,
                audio_duration_s=duration_s,
            )
        else:
            metrics_path, report_path = write_report(root, asr_runs)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Metrics: {metrics_path}")
    typer.echo(f"Report: {report_path}")

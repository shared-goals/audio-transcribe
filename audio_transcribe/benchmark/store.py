"""Checksum-bound persistence for private ASR benchmark run artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audio_transcribe.benchmark.models import RunManifest, validate_run_manifest
from audio_transcribe.util import atomic_write_text


def write_run(dataset: Path, manifest: RunManifest, result: dict[str, Any]) -> tuple[Path, Path]:
    """Validate evidence checksums and atomically persist one candidate run."""
    expanded_dataset = dataset.expanduser().resolve()
    benchmark_path = expanded_dataset / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (
        not isinstance(benchmark, dict)
        or manifest.source_sha256 != benchmark.get("audio_sha256")
        or manifest.reference_sha256 != benchmark.get("reference_sha256")
        or (manifest.vocabulary_sha256 is not None and manifest.vocabulary_sha256 != benchmark.get("vocabulary_sha256"))
    ):
        raise ValueError("benchmark checksum mismatch")

    candidate_path = Path(manifest.candidate_name)
    if candidate_path.name != manifest.candidate_name or manifest.candidate_name in {"", ".", ".."}:
        raise ValueError("invalid benchmark candidate name")

    validate_run_manifest(manifest)
    manifest_content = json.dumps(asdict(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    result_content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    run_dir = expanded_dataset / "runs" / manifest.candidate_name
    manifest_path = run_dir / "manifest.json"
    result_path = run_dir / "result.json"
    atomic_write_text(manifest_path, manifest_content)
    atomic_write_text(result_path, result_content)
    return manifest_path, result_path

"""Resumable run workspaces and machine-readable manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audio_transcribe import __version__
from audio_transcribe.util import atomic_write_text


def _now() -> str:
    return datetime.now(UTC).isoformat()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunManifest:
    """Persistent state for one input/configuration combination."""

    run_id: str
    version: str
    input_path: str
    input_sha256: str
    config: dict[str, Any]
    started_at: str
    updated_at: str
    status: str = "running"
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    output: str | None = None
    error: str | None = None


class CheckpointStore:
    """Store JSON-safe stage results in a stable per-run workspace."""

    def __init__(self, input_path: Path, config: dict[str, Any], root: Path, force: bool = False) -> None:
        input_hash = file_sha256(input_path)
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        identity = f"{__version__}:{input_path.resolve()}:{input_hash}:{config_json}"
        run_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
        self.root = root / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists() and not force:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.manifest = RunManifest(**raw)
            self.manifest.status = "running"
            self.manifest.error = None
        else:
            now = _now()
            self.manifest = RunManifest(
                run_id=run_id,
                version=__version__,
                input_path=str(input_path.resolve()),
                input_sha256=input_hash,
                config=config,
                started_at=now,
                updated_at=now,
            )
        self.save_manifest()

    def stage_path(self, stage: str) -> Path:
        """Return the checkpoint path for a stage."""
        return self.root / f"{stage}.json"

    def load(self, stage: str) -> Any | None:
        """Load a completed checkpoint, or return None."""
        entry = self.manifest.stages.get(stage, {})
        path = self.stage_path(stage)
        if entry.get("status") != "complete" or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def complete_stage(self, stage: str, value: Any, elapsed_s: float) -> None:
        """Atomically persist a JSON-safe stage result."""
        atomic_write_text(self.stage_path(stage), json.dumps(value, ensure_ascii=False))
        self.manifest.stages[stage] = {"status": "complete", "time_s": round(elapsed_s, 3), "at": _now()}
        self.save_manifest()

    def invalidate_from(self, stage: str, order: list[str]) -> None:
        """Delete a stage and all downstream checkpoints."""
        if stage not in order:
            raise ValueError(f"unknown restart stage {stage!r}; choose one of: {', '.join(order)}")
        for name in order[order.index(stage) :]:
            self.manifest.stages.pop(name, None)
            self.stage_path(name).unlink(missing_ok=True)
        self.save_manifest()

    def finish(self, output: str | None = None) -> None:
        self.manifest.status = "complete"
        self.manifest.output = output
        self.save_manifest()

    def fail(self, error: str) -> None:
        self.manifest.status = "failed"
        self.manifest.error = error
        self.save_manifest()

    def save_manifest(self) -> None:
        self.manifest.updated_at = _now()
        atomic_write_text(self.manifest_path, json.dumps(asdict(self.manifest), ensure_ascii=False, indent=2))

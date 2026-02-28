"""Stats store — read/write/query historical run records."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audio_transcribe.models import (
    Config,
    HardwareInfo,
    InputInfo,
    QualityMetrics,
    RunRecord,
    StageStats,
)

_DEFAULT_PATH = Path.home() / ".audio-transcribe" / "history.json"


class StatsStore:
    """File-based store for historical pipeline run records."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH

    def append(self, record: RunRecord) -> None:
        """Serialize and append a RunRecord to the history file."""
        records = self._load_raw()
        records.append(asdict(record))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> list[RunRecord]:
        """Deserialize all records from disk."""
        return [_dict_to_record(d) for d in self._load_raw()]

    def query(self, **filters: str) -> list[RunRecord]:
        """Filter records by chip, model, backend, or language."""
        records = self.load()
        result: list[RunRecord] = []
        for r in records:
            if _matches(r, filters):
                result.append(r)
        return result

    def last(self, n: int) -> list[RunRecord]:
        """Return the last N records."""
        records = self.load()
        return records[-n:]

    def clear(self) -> None:
        """Empty the history file."""
        if self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    def _load_raw(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        text = self._path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return data


def _matches(record: RunRecord, filters: dict[str, str]) -> bool:
    """Check if a record matches all given filters."""
    for key, value in filters.items():
        if key == "chip":
            if record.hardware.chip != value:
                return False
        elif key == "model":
            if record.config.model != value:
                return False
        elif key == "backend":
            if record.config.backend != value:
                return False
        elif key == "language":
            if record.config.language != value:
                return False
    return True


def _dict_to_record(d: dict[str, Any]) -> RunRecord:
    """Reconstruct a RunRecord from a dict."""
    hw = HardwareInfo(**d["hardware"])
    inp = InputInfo(**d["input"])
    cfg = Config(**d["config"])
    stages = {name: StageStats(**s) for name, s in d["stages"].items()}
    quality = QualityMetrics(**d["quality"]) if d.get("quality") else None
    return RunRecord(
        id=d["id"],
        hardware=hw,
        input=inp,
        config=cfg,
        stages=stages,
        quality=quality,
        corrections_applied=d["corrections_applied"],
        total_time_s=d["total_time_s"],
        realtime_ratio=d["realtime_ratio"],
    )

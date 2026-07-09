"""Speaker embedding database — store, match, and manage known speakers."""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from audio_transcribe.speakers.embeddings import cosine_distance
from audio_transcribe.util import atomic_np_save, atomic_write_text

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 256


class SpeakerDB:
    """File-based speaker embedding database.

    Names are normalized to lowercase for all lookups.
    Display names are preserved in index.json under "display_name".
    """

    def __init__(self, db_dir: Path) -> None:
        self._dir = db_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._index: dict[str, dict[str, object]] = self._load_index()

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower()

    def _load_index(self) -> dict[str, dict[str, object]]:
        if self._index_path.exists():
            data: dict[str, dict[str, object]] = json.loads(self._index_path.read_text(encoding="utf-8"))
            return data
        return {}

    def _save_index(self) -> None:
        atomic_write_text(self._index_path, json.dumps(self._index, ensure_ascii=False, indent=2))

    def _embedding_path(self, name: str) -> Path:
        """Get or generate a safe filename for a speaker embedding."""
        key = self._normalize(name)
        # Return existing filename if already indexed
        if key in self._index and "file" in self._index[key]:
            return self._dir / str(self._index[key]["file"])
        # Generate new safe filename
        safe = re.sub(r"[^\w\-]", "_", key) or "_unknown"
        counter = 1
        while (self._dir / f"{safe}_{counter:02d}.npy").exists():
            counter += 1
        return self._dir / f"{safe}_{counter:02d}.npy"

    def has_speaker(self, name: str) -> bool:
        return self._normalize(name) in self._index

    def enroll(self, name: str, embedding: NDArray[np.float32]) -> None:
        """Add or update a speaker's embedding. Averages with existing if present."""
        if embedding.shape != (_EMBEDDING_DIM,):
            msg = f"Expected embedding dimension ({_EMBEDDING_DIM},), got {embedding.shape}"
            raise ValueError(msg)
        key = self._normalize(name)
        if key in self._index:
            existing = self.get_embedding(name)
            raw_count = self._index[key].get("samples", 1)
            count = int(raw_count) if isinstance(raw_count, (int, float, str)) else 1
            averaged = (existing * count + embedding) / (count + 1)
            atomic_np_save(self._embedding_path(name), averaged)
            self._index[key]["samples"] = count + 1
            self._index[key]["last_seen"] = str(date.today())
        else:
            path = self._embedding_path(name)
            atomic_np_save(path, embedding)
            self._index[key] = {
                "display_name": name,
                "file": path.name,
                "samples": 1,
                "last_seen": str(date.today()),
            }
        self._save_index()

    def get_embedding(self, name: str) -> NDArray[np.float32]:
        """Load a speaker's embedding from disk."""
        path = self._embedding_path(name)
        arr: NDArray[np.float32] = np.load(path).astype(np.float32)
        return arr

    def match(self, query: NDArray[np.float32], threshold: float = 0.5) -> list[tuple[str, float]]:
        """Find speakers matching the query embedding.

        Returns list of (display_name, distance) sorted by distance, filtered by threshold.
        """
        results: list[tuple[str, float]] = []
        for key, meta in self._index.items():
            stored = np.load(self._dir / str(meta["file"])).astype(np.float32)
            if stored.shape != (_EMBEDDING_DIM,):
                logger.warning(
                    "Skipping speaker %s: embedding shape %s, expected (%d,)",
                    key, stored.shape, _EMBEDDING_DIM,
                )
                continue
            dist = cosine_distance(query, stored)
            if dist < threshold:
                display_name = str(meta.get("display_name", key))
                results.append((display_name, dist))
        results.sort(key=lambda x: x[1])
        return results

    def list_speakers(self) -> list[dict[str, object]]:
        """List all known speakers with metadata."""
        return [
            {"name": str(meta.get("display_name", key)), **{k: v for k, v in meta.items() if k != "display_name"}}
            for key, meta in self._index.items()
        ]

    def forget(self, name: str) -> None:
        """Remove a speaker from the database."""
        key = self._normalize(name)
        if key in self._index:
            path = self._embedding_path(name)
            if path.exists():
                path.unlink()
            del self._index[key]
            self._save_index()

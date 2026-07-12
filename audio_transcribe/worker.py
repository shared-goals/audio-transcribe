"""Durable SQLite-backed transcription job queue."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Job:
    id: int
    input_path: str
    output_dir: str
    profile: str
    status: str
    attempts: int
    max_attempts: int
    available_at: float
    created_at: str
    updated_at: str
    last_error: str | None


class JobQueue:
    """A transactional queue safe for multiple worker processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    profile TEXT NOT NULL DEFAULT 'default',
                    status TEXT NOT NULL DEFAULT 'queued',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    available_at REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT,
                    UNIQUE(input_path, output_dir)
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(status, available_at, id)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
        finally:
            db.close()

    def enqueue(self, input_path: Path, output_dir: Path, profile: str = "default", max_attempts: int = 3) -> int:
        """Add a job idempotently and return its id."""
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        now = _now()
        with self._connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO jobs
                (input_path, output_dir, profile, max_attempts, available_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (str(input_path.resolve()), str(output_dir.resolve()), profile, max_attempts, time.time(), now, now),
            )
            row = db.execute(
                "SELECT id FROM jobs WHERE input_path = ? AND output_dir = ?",
                (str(input_path.resolve()), str(output_dir.resolve())),
            ).fetchone()
        assert row is not None
        return int(row["id"])

    def claim(self) -> Job | None:
        """Atomically claim the next ready job."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status = 'queued' AND available_at <= ? ORDER BY id LIMIT 1",
                (time.time(),),
            ).fetchone()
            if row is None:
                db.execute("COMMIT")
                return None
            db.execute(
                "UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = ? WHERE id = ?",
                (_now(), row["id"]),
            )
            db.execute("COMMIT")
            return self.get(int(row["id"]))

    def complete(self, job_id: int) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status = 'complete', updated_at = ?, last_error = NULL WHERE id = ?",
                (_now(), job_id),
            )

    def fail(self, job_id: int, error: str, base_delay_s: float = 30.0) -> str:
        """Retry with exponential backoff or move a job to the dead-letter state."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        status = "dead" if job.attempts >= job.max_attempts else "queued"
        delay = 0.0 if status == "dead" else base_delay_s * (2 ** max(job.attempts - 1, 0))
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status = ?, available_at = ?, updated_at = ?, last_error = ? WHERE id = ?",
                (status, time.time() + delay, _now(), error[-4000:], job_id),
            )
        return status

    def recover_stale(self, older_than_s: float = 3600.0) -> int:
        """Return abandoned running jobs to the queue after a worker crash."""
        cutoff = datetime.fromtimestamp(time.time() - older_than_s, UTC).isoformat()
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status = 'queued', available_at = ?, updated_at = ? "
                "WHERE status = 'running' AND updated_at < ?",
                (time.time(), _now(), cutoff),
            )
        return cursor.rowcount

    def retry(self, job_id: int) -> None:
        """Explicitly move a failed/dead job back to the ready queue."""
        if self.get(job_id) is None:
            raise KeyError(job_id)
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status = 'queued', available_at = ?, updated_at = ?, last_error = NULL WHERE id = ?",
                (time.time(), _now(), job_id),
            )

    def get(self, job_id: int) -> Job | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job(**dict(row)) if row else None

    def list(self, status: str | None = None, limit: int = 50) -> list[Job]:
        query = "SELECT * FROM jobs"
        args: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            args = (status,)
        query += " ORDER BY id DESC LIMIT ?"
        args += (limit,)
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [Job(**dict(row)) for row in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def health(self) -> dict[str, Any]:
        """Return queue state suitable for monitoring."""
        with self._connect() as db:
            integrity = str(db.execute("PRAGMA quick_check").fetchone()[0])
        counts = self.counts()
        return {"ok": integrity == "ok" and counts.get("dead", 0) == 0, "integrity": integrity, "jobs": counts}

    def as_json(self) -> str:
        return json.dumps({"health": self.health(), "jobs": [asdict(job) for job in self.list()]}, indent=2)


def stable_audio_files(directory: Path, stable_for_s: float = 30.0) -> Iterator[Path]:
    """Yield supported audio files that have stopped changing."""
    cutoff = time.time() - stable_for_s
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".wav", ".m4a", ".mp3", ".flac", ".aac"} and path.is_file():
            if path.stat().st_mtime <= cutoff:
                yield path

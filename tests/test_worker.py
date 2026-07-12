import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from audio_transcribe.worker import JobQueue, stable_audio_files


def test_queue_lifecycle_is_idempotent(tmp_path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"audio")
    queue = JobQueue(tmp_path / "jobs.sqlite3")
    first = queue.enqueue(audio, tmp_path / "out")
    assert queue.enqueue(audio, tmp_path / "out") == first
    job = queue.claim()
    assert job is not None and job.status == "running" and job.attempts == 1
    queue.complete(job.id)
    completed = queue.get(job.id)
    assert completed is not None and completed.status == "complete"
    assert queue.health()["ok"] is True


def test_retry_backoff_dead_letter_and_manual_retry(tmp_path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"audio")
    queue = JobQueue(tmp_path / "jobs.sqlite3")
    job_id = queue.enqueue(audio, tmp_path / "out", max_attempts=2)
    first = queue.claim()
    assert first is not None
    assert queue.fail(job_id, "first", base_delay_s=0) == "queued"
    second = queue.claim()
    assert second is not None
    assert queue.fail(job_id, "second", base_delay_s=0) == "dead"
    assert queue.health()["ok"] is False
    queue.retry(job_id)
    retried = queue.get(job_id)
    assert retried is not None and retried.status == "queued"
    with pytest.raises(KeyError):
        queue.retry(999)


def test_recover_stale_and_status_json(tmp_path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"audio")
    queue = JobQueue(tmp_path / "jobs.sqlite3")
    queue.enqueue(audio, tmp_path / "out")
    assert queue.claim() is not None
    assert queue.recover_stale(older_than_s=-1) == 1
    assert '"health"' in queue.as_json()


def test_stable_audio_files(tmp_path):
    old = tmp_path / "old.m4a"
    new = tmp_path / "new.wav"
    ignored = tmp_path / "note.txt"
    for path in (old, new, ignored):
        path.write_bytes(b"x")
    os.utime(old, (time.time() - 60, time.time() - 60))
    assert list(stable_audio_files(tmp_path, stable_for_s=30)) == [old]


def test_concurrent_claims_are_unique(tmp_path):
    queue = JobQueue(tmp_path / "jobs.sqlite3")
    for index in range(8):
        audio = tmp_path / f"{index}.wav"
        audio.write_bytes(b"audio")
        queue.enqueue(audio, tmp_path / "out")
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: queue.claim(), range(8)))
    ids = [job.id for job in claimed if job is not None]
    assert len(ids) == len(set(ids)) == 8

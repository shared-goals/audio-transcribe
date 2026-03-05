"""Tests for atomic file write utilities."""
from pathlib import Path

import numpy as np

from audio_transcribe.util import atomic_np_save, atomic_write_text


def test_atomic_write_text_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    atomic_write_text(path, "hello world")
    assert path.read_text() == "hello world"


def test_atomic_write_text_no_temp_on_success(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    atomic_write_text(path, "content")
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_np_save(tmp_path: Path) -> None:
    path = tmp_path / "test.npy"
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    atomic_np_save(path, arr)
    loaded = np.load(path)
    np.testing.assert_array_equal(loaded, arr)


def test_atomic_np_save_no_temp_on_success(tmp_path: Path) -> None:
    path = tmp_path / "test.npy"
    atomic_np_save(path, np.zeros(3))
    assert not list(tmp_path.glob("*.tmp*"))

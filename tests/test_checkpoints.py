import json

import pytest

from audio_transcribe.checkpoints import CheckpointStore, file_sha256


def test_checkpoint_roundtrip_and_manifest(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    store = CheckpointStore(audio, {"backend": "mlx"}, tmp_path / "runs")
    store.complete_stage("transcribe", {"segments": []}, 1.25)
    assert store.load("transcribe") == {"segments": []}
    store.finish("out.json")
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["input_sha256"] == file_sha256(audio)


def test_checkpoint_identity_includes_path(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    a = CheckpointStore(first, {}, tmp_path / "runs")
    b = CheckpointStore(second, {}, tmp_path / "runs")
    assert a.root != b.root


def test_invalidate_and_failure(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    store = CheckpointStore(audio, {}, tmp_path / "runs")
    store.complete_stage("preprocess", "clean.wav", 0.1)
    store.complete_stage("transcribe", {"text": "hello"}, 1)
    store.invalidate_from("transcribe", ["preprocess", "transcribe", "align"])
    assert store.load("preprocess") == "clean.wav"
    assert store.load("transcribe") is None
    store.fail("boom")
    assert store.manifest.status == "failed"
    with pytest.raises(ValueError, match="unknown restart stage"):
        store.invalidate_from("oops", ["preprocess"])

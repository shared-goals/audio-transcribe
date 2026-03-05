"""Tests for pre-flight validation checks."""

from audio_transcribe.preflight import check


def test_check_missing_audio_file() -> None:
    result = check("/nonexistent/audio.wav")
    assert not result.ok
    assert any("not found" in e for e in result.errors)


def test_check_valid_file(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    result = check(str(audio))
    assert result.ok


def test_check_empty_file(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    result = check(str(audio))
    assert not result.ok
    assert any("empty" in e for e in result.errors)


def test_check_warns_missing_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 1024)
    result = check(str(audio), skip_diarize=False)
    assert result.ok
    assert any("HF_TOKEN" in w for w in result.warnings)

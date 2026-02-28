"""Tests for audio_transcribe.stages.diarize — speaker diarization stage."""

from audio_transcribe.stages.diarize import diarize


def test_diarize_signature():
    """Verify diarize function exists with expected signature."""
    import inspect

    sig = inspect.signature(diarize)
    params = list(sig.parameters.keys())
    assert params == ["result", "audio", "hf_token", "min_speakers", "max_speakers"]

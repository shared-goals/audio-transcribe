"""Tests for audio_transcribe.stages.align — word alignment stage."""

from audio_transcribe.stages.align import align


def test_align_signature():
    """Verify align function exists with expected signature."""
    import inspect

    sig = inspect.signature(align)
    params = list(sig.parameters.keys())
    assert params == ["result", "audio", "language", "align_model"]


def test_align_has_default_align_model():
    """Verify align_model parameter defaults to None."""
    import inspect

    sig = inspect.signature(align)
    assert sig.parameters["align_model"].default is None

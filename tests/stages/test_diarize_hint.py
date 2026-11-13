"""The exact speaker count must reach pyannote instead of the min/max range."""

import sys
import types

import pytest

from audio_transcribe.stages.diarize import diarize


class _FakePipeline:
    calls: list[dict[str, object]] = []

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, audio, **kwargs):
        _FakePipeline.calls.append(kwargs)
        return []


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    diarize_module = types.ModuleType("whisperx.diarize")
    diarize_module.DiarizationPipeline = _FakePipeline  # type: ignore[attr-defined]
    whisperx_module = types.ModuleType("whisperx")
    whisperx_module.assign_word_speakers = lambda turns, result: result  # type: ignore[attr-defined]
    whisperx_module.diarize = diarize_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisperx", whisperx_module)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize_module)
    _FakePipeline.calls = []


def test_exact_count_replaces_the_range(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    diarize({"segments": []}, None, "token", 2, 6, 2)
    assert _FakePipeline.calls == [{"num_speakers": 2, "min_speakers": None, "max_speakers": None}]


def test_without_a_hint_the_range_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    diarize({"segments": []}, None, "token", 2, 5)
    assert _FakePipeline.calls == [{"num_speakers": None, "min_speakers": 2, "max_speakers": 5}]

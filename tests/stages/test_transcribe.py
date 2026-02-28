"""Tests for audio_transcribe.stages.transcribe — build_output (pure function) and MLX integration."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio_transcribe.stages.transcribe import MLX_MODEL_MAP, _offset_segments, build_output


def test_build_output_empty_segments():
    output = build_output({}, "audio.wav", "ru", "large-v3", 10.0)
    assert output["audio_file"] == "audio.wav"
    assert output["language"] == "ru"
    assert output["model"] == "large-v3"
    assert output["processing_time_s"] == 10.0
    assert output["segments"] == []


def test_build_output_text_stripped():
    result_in = {"segments": [{"start": 0.0, "end": 2.5, "text": "  Привет  "}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert output["segments"][0]["text"] == "Привет"


def test_build_output_unknown_speaker_default():
    result_in = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert output["segments"][0]["speaker"] == "UNKNOWN"


def test_build_output_speaker_assigned():
    result_in = {"segments": [{"start": 0.0, "end": 2.0, "text": "hi", "speaker": "SPEAKER_01"}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert output["segments"][0]["speaker"] == "SPEAKER_01"


def test_build_output_words_included():
    result_in = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                    {"word": "world", "start": 0.6, "end": 1.0, "speaker": "SPEAKER_00"},
                ],
            }
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    words = output["segments"][0]["words"]
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert words[1]["word"] == "world"


def test_build_output_words_without_start_excluded():
    """Words missing 'start' timestamp should be dropped."""
    result_in = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "no_time"},  # no 'start' key
                ],
            }
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert len(output["segments"][0]["words"]) == 1


def test_build_output_timestamps_rounded():
    result_in = {"segments": [{"start": 1.23456789, "end": 5.98765432, "text": "test"}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 9.9999)
    seg = output["segments"][0]
    assert seg["start"] == 1.235
    assert seg["end"] == 5.988
    assert output["processing_time_s"] == 10.0


def test_build_output_multiple_segments():
    result_in = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "first", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "text": "second", "speaker": "SPEAKER_01"},
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 5.0)
    assert len(output["segments"]) == 2
    assert output["segments"][1]["speaker"] == "SPEAKER_01"


def test_build_output_no_words_key_when_absent():
    """Segments without 'words' key should not include 'words' in output."""
    result_in = {"segments": [{"start": 0.0, "end": 1.0, "text": "test"}]}
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    assert "words" not in output["segments"][0]


# --- MLX model map tests ---


def test_mlx_model_map_contains_standard_sizes():
    for size in ["tiny", "base", "small", "medium", "large-v2", "large-v3"]:
        assert size in MLX_MODEL_MAP
        assert MLX_MODEL_MAP[size].startswith("mlx-community/")


def test_mlx_model_map_unknown_falls_back_to_name():
    assert MLX_MODEL_MAP.get("custom-model", "custom-model") == "custom-model"


def test_build_output_accepts_mlx_style_segments():
    """mlx-whisper segments have extra keys that build_output must silently ignore."""
    result_in = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Привет",
                "tokens": [50364, 1057],
                "temperature": 0.0,
                "avg_logprob": -0.3,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.01,
            }
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 5.0)
    assert output["segments"][0]["text"] == "Привет"
    assert output["segments"][0]["start"] == 0.0


def test_build_output_mlx_words_with_probability_field():
    """mlx word entries use 'probability' not 'score'; build_output should not include it."""
    result_in = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": "world", "start": 0.6, "end": 1.0, "probability": 0.95},
                ],
            }
        ]
    }
    output = build_output(result_in, "a.wav", "ru", "large-v3", 1.0)
    words = output["segments"][0]["words"]
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert "probability" not in words[0]


# --- _offset_segments (pure function) ---


def test_offset_segments_shifts_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": "hello"},
        {"start": 3.0, "end": 5.0, "text": "world"},
    ]
    result = _offset_segments(segments, 10.0)
    assert result[0]["start"] == 10.0
    assert result[0]["end"] == 12.5
    assert result[1]["start"] == 13.0
    assert result[1]["end"] == 15.0


def test_offset_segments_empty_list():
    assert _offset_segments([], 5.0) == []


# --- MLX backend tests (require mocking heavy deps) ---


def _add_mlx_cache_mocks(mp):
    """Add mlx.core and mlx_whisper.transcribe mocks needed by _clear_mlx_cache()."""
    mock_mx = MagicMock()
    mock_model_holder = MagicMock()
    mock_model_holder.model = "loaded_model"
    mock_model_holder.model_path = "/some/path"
    mock_mlx_transcribe_mod = MagicMock()
    mock_mlx_transcribe_mod.ModelHolder = mock_model_holder

    mock_mlx_pkg = MagicMock()
    mock_mlx_pkg.core = mock_mx
    mp.setitem(sys.modules, "mlx", mock_mlx_pkg)
    mp.setitem(sys.modules, "mlx.core", mock_mx)
    mp.setitem(sys.modules, "mlx_whisper.transcribe", mock_mlx_transcribe_mod)
    return mock_mx


def test_transcribe_mlx_maps_model_name_and_calls_correctly(tmp_path):
    """transcribe_mlx should resolve 'large-v3' to the correct mlx-community repo."""
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()

    mock_result = {
        "text": "Привет мир",
        "language": "ru",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Привет мир"}],
    }
    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = mock_result
    mock_wx = MagicMock()
    mock_wx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "mlx_whisper", mock_mlx)
        mp.setitem(sys.modules, "whisperx", mock_wx)
        _add_mlx_cache_mocks(mp)
        import importlib

        import audio_transcribe.stages.transcribe as mod

        importlib.reload(mod)
        result, audio = mod.transcribe_mlx(str(dummy_audio), "large-v3", "ru")

    mock_mlx.transcribe.assert_called_once_with(
        str(dummy_audio),
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="ru",
        word_timestamps=False,
    )
    assert result["language"] == "ru"
    assert len(result["segments"]) == 1


def test_transcribe_mlx_unknown_model_warns_and_passes_through(tmp_path, capsys):
    """An unknown model name should be passed through directly with a warning."""
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()

    mock_result = {"text": "", "language": "ru", "segments": []}
    mock_mlx = MagicMock()
    mock_mlx.transcribe.return_value = mock_result
    mock_wx = MagicMock()
    mock_wx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "mlx_whisper", mock_mlx)
        mp.setitem(sys.modules, "whisperx", mock_wx)
        _add_mlx_cache_mocks(mp)
        import importlib

        import audio_transcribe.stages.transcribe as mod

        importlib.reload(mod)
        mod.transcribe_mlx(str(dummy_audio), "my-custom/model", "ru")

    mock_mlx.transcribe.assert_called_once_with(
        str(dummy_audio),
        path_or_hf_repo="my-custom/model",
        language="ru",
        word_timestamps=False,
    )
    captured = capsys.readouterr()
    assert "not in MLX model map" in captured.err


# --- mlx-vad backend tests ---


def _setup_mlx_vad_mocks(mp, audio_length_s=30, vad_chunks=None, transcribe_results=None):
    """Set up sys.modules mocks for transcribe_mlx_vad and reload the module."""
    import importlib

    sample_rate = 16000

    mock_mlx = MagicMock()
    if transcribe_results is not None:
        mock_mlx.transcribe.side_effect = transcribe_results
    else:
        mock_mlx.transcribe.return_value = {"text": "", "language": "ru", "segments": []}

    mock_wx = MagicMock()
    mock_wx.load_audio.return_value = np.zeros(int(audio_length_s * sample_rate), dtype=np.float32)

    mock_wx_audio = MagicMock()
    mock_wx_audio.SAMPLE_RATE = sample_rate

    mock_pyannote_mod = MagicMock()
    mock_pyannote_mod.load_vad_model.return_value = MagicMock()
    mock_pyannote_mod.Pyannote.merge_chunks.return_value = vad_chunks or []

    mock_torch = MagicMock()

    mock_mx = MagicMock()
    mock_model_holder = MagicMock()
    mock_model_holder.model = "loaded_model"
    mock_model_holder.model_path = "/some/path"
    mock_mlx_transcribe_mod = MagicMock()
    mock_mlx_transcribe_mod.ModelHolder = mock_model_holder

    mock_mlx_pkg = MagicMock()
    mock_mlx_pkg.core = mock_mx

    mp.setitem(sys.modules, "mlx_whisper", mock_mlx)
    mp.setitem(sys.modules, "mlx_whisper.transcribe", mock_mlx_transcribe_mod)
    mp.setitem(sys.modules, "mlx", mock_mlx_pkg)
    mp.setitem(sys.modules, "mlx.core", mock_mx)
    mp.setitem(sys.modules, "whisperx", mock_wx)
    mp.setitem(sys.modules, "whisperx.audio", mock_wx_audio)
    mp.setitem(sys.modules, "whisperx.vads", MagicMock())
    mp.setitem(sys.modules, "whisperx.vads.pyannote", mock_pyannote_mod)
    mp.setitem(sys.modules, "torch", mock_torch)

    import audio_transcribe.stages.transcribe as mod

    importlib.reload(mod)
    return mock_mlx, mod, mock_mx


def test_transcribe_mlx_vad_offsets_timestamps(tmp_path):
    """Chunk-relative timestamps should be shifted to absolute positions."""
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()

    chunk_result = {
        "text": "Привет мир",
        "language": "ru",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет"},
            {"start": 3.0, "end": 5.0, "text": "мир"},
        ],
    }

    with pytest.MonkeyPatch().context() as mp:
        mock_mlx, mod, _mock_mx = _setup_mlx_vad_mocks(
            mp,
            vad_chunks=[{"start": 10.0, "end": 25.0, "segments": [(10.0, 25.0)]}],
            transcribe_results=[chunk_result],
        )
        result, _audio = mod.transcribe_mlx_vad(str(dummy_audio), "large-v3", "ru")

    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 10.0
    assert result["segments"][0]["end"] == 12.5
    assert result["segments"][1]["start"] == 13.0
    assert result["segments"][1]["end"] == 15.0


def test_transcribe_mlx_vad_processes_multiple_chunks(tmp_path):
    """Two VAD chunks should produce segments from both in the output."""
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()

    chunk1_result = {
        "text": "первый",
        "language": "ru",
        "segments": [{"start": 0.0, "end": 2.0, "text": "первый"}],
    }
    chunk2_result = {
        "text": "второй",
        "language": "ru",
        "segments": [{"start": 0.0, "end": 3.0, "text": "второй"}],
    }

    with pytest.MonkeyPatch().context() as mp:
        _mock_mlx, mod, _mock_mx = _setup_mlx_vad_mocks(
            mp,
            audio_length_s=120,
            vad_chunks=[
                {"start": 5.0, "end": 20.0, "segments": [(5.0, 20.0)]},
                {"start": 60.0, "end": 80.0, "segments": [(60.0, 80.0)]},
            ],
            transcribe_results=[chunk1_result, chunk2_result],
        )
        result, _audio = mod.transcribe_mlx_vad(str(dummy_audio), "large-v3", "ru")

    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 5.0
    assert result["segments"][0]["end"] == 7.0
    assert result["segments"][1]["start"] == 60.0
    assert result["segments"][1]["end"] == 63.0


def test_transcribe_mlx_vad_empty_vad(tmp_path):
    """No VAD segments should produce an empty result."""
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()

    with pytest.MonkeyPatch().context() as mp:
        mock_mlx, mod, _mock_mx = _setup_mlx_vad_mocks(mp, vad_chunks=[])
        result, _audio = mod.transcribe_mlx_vad(str(dummy_audio), "large-v3", "ru")

    assert result["segments"] == []
    assert result["text"] == ""
    mock_mlx.transcribe.assert_not_called()


def test_transcribe_mlx_vad_clears_mlx_cache(tmp_path):
    """After transcription, ModelHolder should be cleared and Metal cache freed."""
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()

    with pytest.MonkeyPatch().context() as mp:
        _mock_mlx, mod, mock_mx = _setup_mlx_vad_mocks(
            mp,
            vad_chunks=[{"start": 0.0, "end": 10.0, "segments": [(0.0, 10.0)]}],
            transcribe_results=[{"text": "test", "language": "ru", "segments": []}],
        )
        mock_model_holder = sys.modules["mlx_whisper.transcribe"].ModelHolder

        mod.transcribe_mlx_vad(str(dummy_audio), "large-v3", "ru")

        assert mock_model_holder.model is None
        assert mock_model_holder.model_path is None
        mock_mx.metal.set_cache_limit.assert_called_once_with(100_000_000)
        mock_mx.metal.clear_cache.assert_called_once()


def test_transcribe_mlx_clears_mlx_cache(tmp_path):
    """After transcription, ModelHolder should be cleared and Metal cache freed."""
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.touch()

    import importlib

    mock_result = {"text": "test", "language": "ru", "segments": []}
    mock_mlx_whisper = MagicMock()
    mock_mlx_whisper.transcribe.return_value = mock_result
    mock_wx = MagicMock()
    mock_wx.load_audio.return_value = np.zeros(16000, dtype=np.float32)

    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "mlx_whisper", mock_mlx_whisper)
        mp.setitem(sys.modules, "whisperx", mock_wx)
        mock_mx = _add_mlx_cache_mocks(mp)
        mock_model_holder = sys.modules["mlx_whisper.transcribe"].ModelHolder

        import audio_transcribe.stages.transcribe as mod

        importlib.reload(mod)
        mod.transcribe_mlx(str(dummy_audio), "large-v3", "ru")

        assert mock_model_holder.model is None
        assert mock_model_holder.model_path is None
        mock_mx.metal.clear_cache.assert_called_once()

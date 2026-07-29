"""Tests for isolated pyannote diarization benchmark candidates."""

from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch

from audio_transcribe.benchmark.diarization_runner import run_diarization_candidate
from audio_transcribe.benchmark.models import DiarizationCandidate


class _Annotation:
    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, *, yield_label: bool = False) -> Iterator[tuple[SimpleNamespace, None, str]]:
        assert yield_label
        for start, end, speaker in self._turns:
            yield SimpleNamespace(start=start, end=end), None, speaker


def _aligned_result() -> dict[str, object]:
    return {
        "segments": [
            {
                "words": [
                    {"word": "one", "start": 0.0, "end": 1.0},
                    {"word": "tie", "start": 0.5, "end": 1.5},
                    {"word": "two", "start": 1.0, "end": 2.0},
                ]
            }
        ]
    }


def test_normal_candidate_loads_on_cpu_and_assigns_words_by_maximum_overlap(tmp_path: Path) -> None:
    candidate = DiarizationCandidate("pyannote-3.1", "pyannote/speaker-diarization-3.1", False)
    pipeline = MagicMock()
    pipeline.return_value = _Annotation([(0.0, 1.2, "SPEAKER_00"), (0.8, 2.0, "SPEAKER_01")])
    waveform = torch.zeros((1, 32))

    with (
        patch(
            "audio_transcribe.benchmark.diarization_runner.Pipeline.from_pretrained",
            return_value=pipeline,
        ) as load_pipeline,
        patch("audio_transcribe.benchmark.diarization_runner.Audio") as audio_loader,
        patch("audio_transcribe.benchmark.diarization_runner.time.perf_counter", side_effect=(10.0, 12.5)),
        patch("audio_transcribe.benchmark.diarization_runner._peak_rss_mb", return_value=321.0),
        patch("audio_transcribe.benchmark.diarization_runner.gc.collect") as collect,
    ):
        audio_loader.return_value.return_value = (waveform, 16_000)
        result = run_diarization_candidate(tmp_path / "meeting.mp3", _aligned_result(), "hf_secret", candidate)

    load_pipeline.assert_called_once_with(candidate.model_id, token="hf_secret")
    pipeline.to.assert_called_once()
    assert pipeline.to.call_args.args[0] == torch.device("cpu")
    pipeline.assert_called_once_with({"waveform": waveform, "sample_rate": 16_000}, num_speakers=2)
    assert [turn.speaker for turn in result.turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert [word["speaker"] for word in result.word_assignments] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert result.time_s == 2.5
    assert result.peak_rss_mb == 321.0
    assert result.status == "completed"
    assert result.error is None
    collect.assert_called_once_with()


def test_exclusive_candidate_selects_exclusive_turns_for_output_and_assignment(tmp_path: Path) -> None:
    candidate = DiarizationCandidate(
        "community-1-exclusive",
        "pyannote/speaker-diarization-community-1",
        True,
    )
    pipeline = MagicMock()
    pipeline.return_value = SimpleNamespace(
        speaker_diarization=_Annotation([(0.0, 2.0, "NORMAL")]),
        exclusive_speaker_diarization=_Annotation([(0.0, 1.0, "EXCLUSIVE_0"), (1.0, 2.0, "EXCLUSIVE_1")]),
    )

    with (
        patch("audio_transcribe.benchmark.diarization_runner.Pipeline.from_pretrained", return_value=pipeline),
        patch("audio_transcribe.benchmark.diarization_runner.Audio") as audio_loader,
        patch("audio_transcribe.benchmark.diarization_runner.gc.collect"),
    ):
        audio_loader.return_value.return_value = (torch.zeros((1, 32)), 16_000)
        result = run_diarization_candidate(tmp_path / "meeting.mp3", _aligned_result(), "hf_secret", candidate)

    assert [turn.speaker for turn in result.turns] == ["EXCLUSIVE_0", "EXCLUSIVE_1"]
    assert [word["speaker"] for word in result.word_assignments] == ["EXCLUSIVE_0", "EXCLUSIVE_0", "EXCLUSIVE_1"]


def test_gated_model_failure_is_actionable_and_isolated(tmp_path: Path) -> None:
    candidate = DiarizationCandidate("community-1", "pyannote/speaker-diarization-community-1", False)

    with (
        patch(
            "audio_transcribe.benchmark.diarization_runner.Pipeline.from_pretrained",
            side_effect=PermissionError("401 Unauthorized"),
        ),
        patch("audio_transcribe.benchmark.diarization_runner.Audio") as audio_loader,
        patch("audio_transcribe.benchmark.diarization_runner.gc.collect") as collect,
    ):
        result = run_diarization_candidate(tmp_path / "meeting.mp3", _aligned_result(), "bad-token", candidate)

    audio_loader.assert_not_called()
    assert result.status == "failed"
    assert result.turns == ()
    assert result.word_assignments == ()
    assert result.error is not None
    assert candidate.model_id in result.error
    assert "HF_TOKEN" in result.error
    assert "license" in result.error.lower()
    collect.assert_called_once_with()


@pytest.mark.parametrize(
    "candidate",
    [
        DiarizationCandidate("", "model", False),
        DiarizationCandidate("candidate", "", False),
    ],
)
def test_invalid_candidate_is_rejected(candidate: DiarizationCandidate, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate"):
        run_diarization_candidate(tmp_path / "meeting.mp3", _aligned_result(), "token", candidate)


def test_invalid_aligned_words_return_failed_result(tmp_path: Path) -> None:
    candidate = DiarizationCandidate("candidate", "model", False)
    pipeline = MagicMock()
    pipeline.return_value = _Annotation([(0.0, 1.0, "SPEAKER_00")])

    with (
        patch("audio_transcribe.benchmark.diarization_runner.Pipeline.from_pretrained", return_value=pipeline),
        patch("audio_transcribe.benchmark.diarization_runner.Audio") as audio_loader,
        patch("audio_transcribe.benchmark.diarization_runner.gc.collect"),
    ):
        audio_loader.return_value.return_value = (torch.zeros((1, 32)), 16_000)
        result = run_diarization_candidate(
            tmp_path / "meeting.mp3",
            {"segments": [{"words": [{"word": "broken", "start": 2.0, "end": 1.0}]}]},
            "token",
            candidate,
        )

    assert result.status == "failed"
    assert result.error == "aligned word timestamps must be finite and increasing"


def test_exclusive_candidate_fails_when_model_does_not_publish_exclusive_output(tmp_path: Path) -> None:
    candidate = DiarizationCandidate("exclusive", "model", True)
    pipeline = MagicMock()
    pipeline.return_value = SimpleNamespace(speaker_diarization=_Annotation([(0.0, 1.0, "SPEAKER_00")]))

    with (
        patch("audio_transcribe.benchmark.diarization_runner.Pipeline.from_pretrained", return_value=pipeline),
        patch("audio_transcribe.benchmark.diarization_runner.Audio") as audio_loader,
        patch("audio_transcribe.benchmark.diarization_runner.gc.collect"),
    ):
        audio_loader.return_value.return_value = (torch.zeros((1, 32)), 16_000)
        result = run_diarization_candidate(tmp_path / "meeting.mp3", _aligned_result(), "token", candidate)

    assert result.status == "failed"
    assert result.error is not None
    assert "exclusive_speaker_diarization" in result.error

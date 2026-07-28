"""Tests for isolated ASR benchmark candidate execution."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

from audio_transcribe.benchmark.asr_runner import run_asr_candidate
from audio_transcribe.benchmark.models import AsrCandidate
from audio_transcribe.stages.transcribe import MlxVadOptions


def test_run_asr_candidate_times_stages_and_cleans_normalized_audio(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.m4a"
    audio.touch()
    candidate = AsrCandidate("baseline", "large-v3", MlxVadOptions())
    transcribed = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}],
        "_benchmark_vad_boundaries": (0.0, 12.5),
    }
    aligned = {"segments": [{"start": 0.0, "end": 1.0, "text": "Привет", "words": []}]}

    with (
        patch(
            "audio_transcribe.benchmark.asr_runner.preprocess",
            side_effect=lambda _source, output: output,
        ) as preprocess,
        patch(
            "audio_transcribe.benchmark.asr_runner.transcribe_mlx_vad",
            return_value=(transcribed, object()),
        ) as transcribe,
        patch("audio_transcribe.benchmark.asr_runner.align", return_value=aligned) as align,
        patch(
            "audio_transcribe.benchmark.asr_runner.time.perf_counter",
            side_effect=(0.0, 1.0, 1.0, 3.0, 3.0, 6.0),
        ),
        patch(
            "audio_transcribe.benchmark.asr_runner.resource.getrusage",
            return_value=SimpleNamespace(ru_maxrss=512 * 1024 * 1024),
        ),
        patch("audio_transcribe.benchmark.asr_runner.sys.platform", "darwin"),
    ):
        run_result = run_asr_candidate(audio, candidate, "align/model")

    normalized_audio = Path(preprocess.call_args.args[1])
    assert not normalized_audio.exists()
    preprocess.assert_called_once_with(str(audio), ANY)
    transcribe.assert_called_once_with(str(normalized_audio), "large-v3", "ru", candidate.options)
    align.assert_called_once_with(
        {"segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}]},
        ANY,
        "ru",
        "align/model",
    )
    assert run_result.candidate == candidate
    assert run_result.result == aligned
    assert run_result.stage_times_s == {"preprocess": 1.0, "transcribe": 2.0, "align": 3.0}
    assert run_result.peak_rss_mb == 512.0
    assert run_result.vad_boundaries == (0.0, 12.5)
    assert run_result.status == "completed"
    assert run_result.error is None


def test_run_asr_candidate_isolates_failure_and_cleans_normalized_audio(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.m4a"
    audio.touch()
    candidate = AsrCandidate("broken", "large-v3", MlxVadOptions())

    with (
        patch(
            "audio_transcribe.benchmark.asr_runner.preprocess",
            side_effect=lambda _source, output: output,
        ) as preprocess,
        patch(
            "audio_transcribe.benchmark.asr_runner.transcribe_mlx_vad",
            side_effect=RuntimeError("candidate exploded"),
        ),
        patch("audio_transcribe.benchmark.asr_runner.align") as align,
        patch(
            "audio_transcribe.benchmark.asr_runner.time.perf_counter",
            side_effect=(0.0, 1.0, 1.0, 2.5),
        ),
        patch(
            "audio_transcribe.benchmark.asr_runner.resource.getrusage",
            return_value=SimpleNamespace(ru_maxrss=256 * 1024 * 1024),
        ),
        patch("audio_transcribe.benchmark.asr_runner.sys.platform", "darwin"),
    ):
        run_result = run_asr_candidate(audio, candidate, None)

    assert not Path(preprocess.call_args.args[1]).exists()
    align.assert_not_called()
    assert run_result.result is None
    assert run_result.stage_times_s == {"preprocess": 1.0, "transcribe": 1.5}
    assert run_result.peak_rss_mb == 256.0
    assert run_result.vad_boundaries == ()
    assert run_result.status == "failed"
    assert run_result.error == "candidate exploded"

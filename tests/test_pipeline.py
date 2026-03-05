"""Tests for pipeline orchestrator — stage sequencing and event emission."""

from unittest.mock import MagicMock, patch

from audio_transcribe.pipeline import Pipeline, PipelineConfig
from audio_transcribe.progress.events import StageStart
from audio_transcribe.stats.store import StatsStore

# Common patches for all pipeline tests — mock all external stages
_STAGE_PATCHES = {
    "preprocess_stage": "clean.wav",
    "transcribe_stage": ({"segments": [], "text": "", "language": "ru"}, None),
    "align_stage": {"segments": []},
    "diarize_stage": {"segments": []},
    "format_stage": "# Transcript",
    "build_output_stage": {
        "segments": [],
        "audio_file": "test.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 1.0,
    },
    "load_corrections": {"substitutions": {}, "patterns": []},
}


def _make_reporter(events: list[tuple[str, object]]) -> MagicMock:
    reporter = MagicMock()
    reporter.on_pipeline_start = lambda e: events.append(("pipeline_start", e))
    reporter.on_stage_start = lambda e: events.append(("stage_start", e))
    reporter.on_stage_complete = lambda e: events.append(("stage_complete", e))
    reporter.on_pipeline_complete = lambda e: events.append(("pipeline_complete", e))
    return reporter


def test_pipeline_config_defaults():
    cfg = PipelineConfig(audio_file="test.wav")
    assert cfg.language == "ru"
    assert cfg.model == "large-v3"
    assert cfg.backend == "whisperx"
    assert cfg.skip_align is False
    assert cfg.skip_diarize is False


def test_pipeline_config_custom():
    cfg = PipelineConfig(
        audio_file="test.wav",
        language="en",
        model="base",
        backend="mlx",
        skip_align=True,
        skip_diarize=True,
    )
    assert cfg.language == "en"
    assert cfg.backend == "mlx"
    assert cfg.skip_align is True


def test_pipeline_emits_events():
    """Pipeline should emit start/complete events for each stage."""
    events: list[tuple[str, object]] = []
    reporter = _make_reporter(events)
    pipeline = Pipeline(reporter=reporter)

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value=_STAGE_PATCHES["preprocess_stage"]),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=_STAGE_PATCHES["transcribe_stage"]),
        patch("audio_transcribe.pipeline.align_stage", return_value=_STAGE_PATCHES["align_stage"]),
        patch("audio_transcribe.pipeline.diarize_stage", return_value=_STAGE_PATCHES["diarize_stage"]),
        patch("audio_transcribe.pipeline.build_output_stage", return_value=_STAGE_PATCHES["build_output_stage"]),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_STAGE_PATCHES["load_corrections"]),
    ):
        cfg = PipelineConfig(audio_file="test.wav", output="out.json")
        pipeline.run(cfg)

    stage_starts = [e for name, e in events if name == "stage_start"]
    stage_completes = [e for name, e in events if name == "stage_complete"]
    pipeline_starts = [e for name, e in events if name == "pipeline_start"]
    pipeline_completes = [e for name, e in events if name == "pipeline_complete"]

    assert len(pipeline_starts) == 1
    assert len(pipeline_completes) == 1
    assert len(stage_starts) >= 4  # preprocess, transcribe, align, diarize, format
    assert len(stage_completes) >= 4


def test_pipeline_skips_align():
    """Pipeline should skip align when skip_align=True."""
    events: list[tuple[str, object]] = []
    reporter = _make_reporter(events)
    pipeline = Pipeline(reporter=reporter)

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value=_STAGE_PATCHES["preprocess_stage"]),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=_STAGE_PATCHES["transcribe_stage"]),
        patch("audio_transcribe.pipeline.align_stage") as mock_align,
        patch("audio_transcribe.pipeline.diarize_stage", return_value=_STAGE_PATCHES["diarize_stage"]),
        patch("audio_transcribe.pipeline.build_output_stage", return_value=_STAGE_PATCHES["build_output_stage"]),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_STAGE_PATCHES["load_corrections"]),
    ):
        cfg = PipelineConfig(audio_file="test.wav", skip_align=True)
        pipeline.run(cfg)

    mock_align.assert_not_called()
    stage_names = [e.stage for _, e in events if isinstance(e, StageStart)]
    assert "align" not in stage_names


def test_pipeline_skips_diarize():
    """Pipeline should skip diarize when skip_diarize=True."""
    events: list[tuple[str, object]] = []
    reporter = _make_reporter(events)
    pipeline = Pipeline(reporter=reporter)

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value=_STAGE_PATCHES["preprocess_stage"]),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=_STAGE_PATCHES["transcribe_stage"]),
        patch("audio_transcribe.pipeline.align_stage", return_value=_STAGE_PATCHES["align_stage"]),
        patch("audio_transcribe.pipeline.diarize_stage") as mock_diarize,
        patch("audio_transcribe.pipeline.build_output_stage", return_value=_STAGE_PATCHES["build_output_stage"]),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_STAGE_PATCHES["load_corrections"]),
    ):
        cfg = PipelineConfig(audio_file="test.wav", skip_diarize=True)
        pipeline.run(cfg)

    mock_diarize.assert_not_called()
    stage_names = [e.stage for _, e in events if isinstance(e, StageStart)]
    assert "diarize" not in stage_names


def test_pipeline_writes_output(tmp_path):
    """Pipeline should write JSON output when output path is specified."""
    reporter = MagicMock()
    pipeline = Pipeline(reporter=reporter)

    output_file = tmp_path / "result.json"
    output_data = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}],
        "audio_file": "test.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 1.0,
    }

    seg = {"start": 0.0, "end": 1.0, "text": "hi"}
    seg_spk = {**seg, "speaker": "SPEAKER_00"}
    trans_rv = ({"segments": [seg], "text": "hi", "language": "ru"}, None)

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value="clean.wav"),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=trans_rv),
        patch("audio_transcribe.pipeline.align_stage", return_value={"segments": [seg]}),
        patch("audio_transcribe.pipeline.diarize_stage", return_value={"segments": [seg_spk]}),
        patch("audio_transcribe.pipeline.build_output_stage", return_value=output_data),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_STAGE_PATCHES["load_corrections"]),
    ):
        cfg = PipelineConfig(audio_file="test.wav", output=str(output_file))
        pipeline.run(cfg)

    assert output_file.exists()
    import json

    data = json.loads(output_file.read_text())
    assert "segments" in data


def test_pipeline_writes_transcript(tmp_path):
    """Pipeline should write Markdown transcript when transcript_output is specified."""
    reporter = MagicMock()
    pipeline = Pipeline(reporter=reporter)

    transcript_file = tmp_path / "transcript.md"

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value=_STAGE_PATCHES["preprocess_stage"]),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=_STAGE_PATCHES["transcribe_stage"]),
        patch("audio_transcribe.pipeline.align_stage", return_value=_STAGE_PATCHES["align_stage"]),
        patch("audio_transcribe.pipeline.diarize_stage", return_value=_STAGE_PATCHES["diarize_stage"]),
        patch("audio_transcribe.pipeline.build_output_stage", return_value=_STAGE_PATCHES["build_output_stage"]),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_STAGE_PATCHES["load_corrections"]),
    ):
        cfg = PipelineConfig(audio_file="test.wav", transcript_output=str(transcript_file))
        pipeline.run(cfg)

    assert transcript_file.exists()
    assert "Transcript" in transcript_file.read_text()


def test_pipeline_persists_run_record(tmp_path):
    """Pipeline should write a RunRecord to stats_store after successful run."""
    events: list[tuple[str, object]] = []
    reporter = _make_reporter(events)
    store = StatsStore(tmp_path / "history.json")

    with (
        patch("audio_transcribe.pipeline.preprocess_stage", return_value=_STAGE_PATCHES["preprocess_stage"]),
        patch("audio_transcribe.pipeline.transcribe_stage", return_value=_STAGE_PATCHES["transcribe_stage"]),
        patch("audio_transcribe.pipeline.align_stage", return_value=_STAGE_PATCHES["align_stage"]),
        patch("audio_transcribe.pipeline.build_output_stage", return_value=_STAGE_PATCHES["build_output_stage"]),
        patch("audio_transcribe.pipeline.load_corrections", return_value=_STAGE_PATCHES["load_corrections"]),
    ):
        pipeline = Pipeline(reporter=reporter, stats_store=store)
        config = PipelineConfig(audio_file="test.wav", skip_diarize=True, suppress_stdout_json=True)
        pipeline.run(config)

    records = store.load()
    assert len(records) == 1
    r = records[0]
    assert r.config.model == "large-v3"
    assert r.config.backend == "whisperx"
    assert "preprocess" in r.stages
    assert "transcribe" in r.stages
    assert r.total_time_s >= 0

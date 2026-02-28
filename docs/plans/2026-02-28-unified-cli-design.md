# Unified CLI Tool Design

Date: 2026-02-28

## Goal

Replace the current collection of loose scripts with a single `audio-transcribe` CLI tool featuring rich terminal progress, historical stats for ETA prediction, quality scoring, and a feedback loop for transcription corrections.

## Package Structure

```
audio_transcribe/
  __init__.py
  cli.py                      # typer app with subcommands
  pipeline.py                 # stage orchestration + event emission
  stages/
    __init__.py
    preprocess.py             # FFmpeg: mono 16kHz, silence removal
    transcribe.py             # WhisperX/MLX transcription
    align.py                  # wav2vec2 word alignment
    diarize.py                # pyannote speaker diarization
    correct.py                # apply learned corrections
    format.py                 # JSON -> Markdown formatting
  progress/
    __init__.py
    events.py                 # dataclasses: StageStart, StageProgress, StageComplete
    tui.py                    # rich Live display: progress bars, ETA, memory
    json_reporter.py          # machine-readable JSON lines to stdout
  stats/
    __init__.py
    store.py                  # read/write ~/.audio-transcribe/history.json
    estimator.py              # ETA from historical linear regression
    recommender.py            # suggest optimal backend/model from history
  quality/
    __init__.py
    scorecard.py              # alignment %, speaker coverage, confidence
tests/
  ... (mirrors package structure)
```

## CLI Subcommands

```
audio-transcribe process <file> [--language ru] [--model large-v3] [--backend auto] [--output result.json] [--json]
audio-transcribe stats [--last N] [--clear]
audio-transcribe recommend <file>
audio-transcribe learn <corrected.md> [--original result.json]
```

- `process` -- full pipeline with rich TUI or JSON output mode.
- `stats` -- view historical run statistics.
- `recommend` -- suggest optimal settings based on history.
- `learn` -- diff corrected markdown against original, extract corrections.

### Entry Point

```toml
[project.scripts]
audio-transcribe = "audio_transcribe.cli:app"
```

Works with `pip install -e .` / `pipx install .` and `uv run audio-transcribe`.

## Pipeline Stages

The pipeline runs these stages sequentially:

```
preprocess -> transcribe -> align -> diarize -> correct -> format
```

Each stage is a module with a clear function signature returning a typed result dataclass:

```python
# stages/transcribe.py
def transcribe(audio_path: str, model: str, language: str, backend: str) -> TranscribeResult

# stages/align.py
def align(result: TranscribeResult, audio: ndarray, language: str, model: str | None) -> AlignResult

# stages/diarize.py
def diarize(result: AlignResult, audio: ndarray, min_speakers: int, max_speakers: int) -> DiarizeResult

# stages/correct.py
def correct(result: DiarizeResult, corrections_path: str) -> CorrectedResult

# stages/format.py
def format_transcript(result: CorrectedResult, ...) -> str
```

The pipeline orchestrator (`pipeline.py`) calls each stage and emits events (stage_start, stage_progress, stage_complete) consumed by either the TUI or JSON reporter.

## Progress Display

### Interactive Mode (default): Rich TUI

Live-updating display using `rich`:

```
 Processing: meeting-2026-02-28.wav (60:00)

 v Preprocess     2.1s          (est: ~2s based on 12 runs)
 * Transcribe     ============  42.3s / ~85s  [ETA: 43s]
   Align
   Diarize
   Correct
   Format

 Memory: 6.2 GB peak  |  Model: large-v3 (mlx-vad)
```

Completion summary:

```
 v Complete: meeting-2026-02-28.wav

 Preprocess     2.1s     Transcribe   85.3s
 Align         12.4s     Diarize      18.7s
 Correct        0.0s     Format        0.1s
 Total:       118.6s (0.03x realtime)

 Quality: A (96.4% aligned | 3 speakers | 94.2% coverage)

 Output: /path/to/result.json
 Transcript: /path/to/transcript.md
```

### Machine Mode (--json or piped stdout)

JSON lines for machine consumers:

```jsonl
{"event":"start","file":"meeting.wav","duration_s":3600}
{"event":"stage_start","stage":"preprocess","eta_s":2.0}
{"event":"stage_complete","stage":"preprocess","time_s":2.1}
{"event":"stage_start","stage":"transcribe","eta_s":85.0}
{"event":"stage_complete","stage":"transcribe","time_s":85.3,"segments":142}
...
{"event":"complete","total_time_s":118.6,"output":"result.json","transcript":"transcript.md"}
```

## Historical Stats

### Storage

Location: `~/.audio-transcribe/history.json`

Each run appends an entry:

```json
{
  "id": "2026-02-28T14:30:00Z",
  "hardware": {
    "chip": "Apple M4",
    "cores_physical": 10,
    "memory_gb": 24,
    "os": "macOS 15.3",
    "python": "3.12.8"
  },
  "input": {
    "file": "meeting-2026-02-28.wav",
    "duration_s": 3600.0,
    "file_size_mb": 115.2,
    "sample_rate": 16000
  },
  "config": {
    "language": "ru",
    "model": "large-v3",
    "backend": "mlx-vad",
    "min_speakers": 2,
    "max_speakers": 6,
    "align_model": null
  },
  "stages": {
    "preprocess": {"time_s": 2.1, "peak_rss_mb": 120},
    "transcribe": {"time_s": 85.3, "peak_rss_mb": 6200},
    "align":      {"time_s": 12.4, "peak_rss_mb": 3100},
    "diarize":    {"time_s": 18.7, "peak_rss_mb": 2800},
    "correct":    {"time_s": 0.0, "peak_rss_mb": 50},
    "format":     {"time_s": 0.1, "peak_rss_mb": 50}
  },
  "quality": {
    "segments": 142,
    "words_total": 4200,
    "words_aligned": 4050,
    "alignment_pct": 96.4,
    "speakers_detected": 3,
    "speaker_coverage_pct": 94.2,
    "speaker_transitions": 87
  },
  "corrections_applied": 12,
  "total_time_s": 118.6,
  "realtime_ratio": 0.033
}
```

Stats are also embedded in the output JSON under a `processing` key.

### ETA Estimation

Weighted linear regression: `audio_duration -> stage_time`, filtered by:

1. Matching hardware fingerprint (chip + memory)
2. Matching config (model, backend, language)
3. Fallback to all data if <3 matching runs

Cold start (no history): shows "~unknown" instead of wrong estimates.

Confidence: when R^2 < 0.7, show `~85s?` with question mark.

### Smart Recommendations

After 5+ runs, `audio-transcribe recommend <file>` suggests:
- Best backend for the file's duration
- Best model for quality/speed tradeoff
- Tips based on stage time distribution

Purely informational -- never auto-changes settings.

## Quality Scorecard

Computed after each run, embedded in output JSON and displayed in TUI.

| Metric | Source | Description |
|--------|--------|-------------|
| alignment_pct | align | % of words with timestamps |
| speaker_coverage_pct | diarize | % of segments with speaker labels |
| speaker_transitions | diarize | Speaker change count |
| speakers_detected | diarize | Unique speakers |
| words_total / words_aligned | align | Raw counts |
| segments_total | transcribe | Segment count |
| avg_segment_duration_s | transcribe | Mean segment length |
| silence_ratio | preprocess | % silence removed |

### Grading (TUI only)

- A: alignment > 95% AND speaker coverage > 90%
- B: alignment > 85% AND speaker coverage > 75%
- C: below B thresholds

JSON mode outputs raw numbers only.

## Feedback Loop: Transcription Corrections

### Workflow

1. `audio-transcribe process meeting.wav` generates transcript.md
2. User edits transcript.md in Obsidian (fixing errors)
3. `audio-transcribe learn corrected.md` diffs against original
4. Corrections stored in `~/.audio-transcribe/corrections.yaml`
5. Future runs auto-apply corrections after diarization

### Corrections File

```yaml
# ~/.audio-transcribe/corrections.yaml
# Auto-learned + manually editable

substitutions:
  wrong_word: correct_word
  another_error: fixed_version

patterns:
  - match: "\\bregex_pattern\\b"
    replace: "replacement"
```

### Learn Command

```
audio-transcribe learn <corrected.md> [--original result.json]
```

- Parses corrected markdown (strips timestamps, speaker labels)
- Loads original text from associated JSON (via YAML header metadata)
- Word-level diff to extract substitutions
- Presents corrections for user approval before saving

### Application

Corrections applied as a post-processing step:
`preprocess -> transcribe -> align -> diarize -> [correct] -> format`

Simple text replacement over segment and word text. Fast, deterministic.

## Migration

| Current Script | Becomes | Notes |
|----------------|---------|-------|
| preprocess.py | audio_transcribe/stages/preprocess.py | Core logic extracted |
| transcribe_whisperx.py | stages/transcribe.py, stages/align.py, stages/diarize.py | Split into 3 |
| format_transcript.py | stages/format.py | Core logic extracted |
| benchmark.py | Removed | Replaced by stats system |
| compare_align.py | Removed | Served its purpose |
| verify_diarize.py | Removed | Served its purpose |
| test_ollama.py | Stays as-is | Separate concern |
| process-audio-local.sh | Removed | Replaced by CLI |

## New Dependencies

- `typer` -- CLI framework (type-hint based, built on click)
- `rich` -- TUI progress display (also used by typer)
- `pyyaml` -- YAML parsing for meeting markdown frontmatter (used by reactive pipeline's markdown parser)

## Install Modes

- **Developer**: `uv run audio-transcribe process file.wav`
- **User**: `pip install .` or `pipx install .` then `audio-transcribe process file.wav`

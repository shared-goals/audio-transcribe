# ASR Quality Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private-reference ASR benchmark that compares the current Russian meeting transcription baseline, meeting vocabulary prompting, and a reproducibly converted Russian fine-tuned Whisper model, then reports quality, runtime, memory, and forecasts up to two hours.

**Architecture:** Add an isolated `audio_transcribe.benchmark` package and an `audio-transcribe benchmark` CLI group. Reference data and real outputs remain in the private vault; repository code handles parsing, metrics, manifests, candidate execution, and reporting through typed interfaces. Production defaults and the original meeting note do not change during this plan.

**Tech Stack:** Python 3.12, Typer, mlx-whisper 0.4.x, WhisperX alignment, jiwer, dataclasses, JSON, pytest, Ruff, strict mypy.

## Global Constraints

- Use `/Users/gnezim/Movies/2026-07-28_17-31-24_busurina-gk-plan-2.mp3` as the only pilot recording.
- Store private benchmark data under `/Users/gnezim/_projects/gnezim/knowledge/projects/personal/audio-transcribe/benchmarks/2026-07-28-busurina-gk-plan-2/`.
- Do not overwrite the existing meeting note or commit real audio, reference text, vocabulary, or run outputs to the public repository.
- Keep the current 30-second VAD chunks; 60-second variants are excluded.
- Keep `condition_on_previous_text=False` except for the final retest of the best initial candidate.
- Meeting vocabulary uses Whisper `initial_prompt`; no LLM or additional inference is allowed.
- The Russian model conversion must use `ml-explore/mlx-examples` commit `796f5b53cab69a3d48a44233ce21aae889e94a08` and `antony66/whisper-large-v3-russian` revision `3c24e46392ff376009a176ad290fe83e368edf7a`.
- A default candidate must improve quality and take no more than 2× the baseline total time.
- Forecast 30-, 60-, and 120-minute recordings and show a ±15% planning range.
- Preserve the M4 constraint: WhisperX uses `device="cpu"` and `compute_type="int8"`.
- Follow TDD, update `[Unreleased]` in `CHANGELOG.md` for every `feat:` or `fix:` commit, and update the Obsidian project page, roadmap, and benchmark task as progress is made.

---

## File Structure

- Create `audio_transcribe/benchmark/__init__.py` as the package boundary.
- Create `audio_transcribe/benchmark/models.py` for immutable reference, candidate, manifest, metric, and forecast dataclasses.
- Create `audio_transcribe/benchmark/reference.py` for reference Markdown parsing, exclusions, checksums, and draft generation.
- Create `audio_transcribe/benchmark/asr_metrics.py` for normalization, WER/CER details, term recall, and safety counts.
- Create `audio_transcribe/benchmark/asr_runner.py` for candidate execution and per-stage timing/memory.
- Create `audio_transcribe/benchmark/store.py` for atomic manifests and run artifacts.
- Create `audio_transcribe/benchmark/report.py` for ranking, 2× gating, forecasts, JSON, and Markdown.
- Create `audio_transcribe/benchmark/cli.py` for the benchmark Typer commands.
- Modify `audio_transcribe/cli.py` only to mount `benchmark_app`.
- Modify `audio_transcribe/stages/transcribe.py` to accept an immutable options object while keeping existing defaults unchanged.
- Add focused tests under `tests/benchmark/`.

### Task 1: Reference Format and Private Dataset Preparation

**Files:**
- Create: `audio_transcribe/benchmark/__init__.py`
- Create: `audio_transcribe/benchmark/models.py`
- Create: `audio_transcribe/benchmark/reference.py`
- Test: `tests/benchmark/__init__.py`
- Test: `tests/benchmark/test_reference.py`

**Interfaces:**
- Produces: `ReferenceSpan(start_s: float, end_s: float, text: str, excluded: bool)`.
- Produces: `ReferenceDocument(source_sha256: str, spans: tuple[ReferenceSpan, ...])`.
- Produces: `parse_reference(path: Path) -> ReferenceDocument`.
- Produces: `prepare_reference(audio: Path, current_note: Path, dataset: Path) -> tuple[Path, Path]`.
- Produces: `reference_text(document: ReferenceDocument) -> str`, excluding marked spans.

- [ ] **Step 1: Add jiwer as a lightweight benchmark dependency and write failing reference tests**

Add `"jiwer>=4.0.0,<5"` to `[project.dependencies]`, then create tests covering accepted spans and exclusions:

```python
def test_parse_reference_excludes_unclear_span(tmp_path: Path) -> None:
    reference = tmp_path / "reference-transcript.md"
    reference.write_text(
        "# Reference transcript\n\n"
        "[00:00.000 --> 00:02.500] Первый этап.\n"
        "[00:02.500 --> 00:04.000] [неразборчиво]\n",
        encoding="utf-8",
    )
    document = parse_reference(reference)
    assert document.spans[0] == ReferenceSpan(0.0, 2.5, "Первый этап.", False)
    assert document.spans[1].excluded is True
    assert reference_text(document) == "Первый этап."
```

- [ ] **Step 2: Run the focused test and verify the missing-module failure**

Run: `uv run pytest tests/benchmark/test_reference.py -q`

Expected: FAIL because `audio_transcribe.benchmark.reference` does not exist.

- [ ] **Step 3: Implement strict parsing and private draft preparation**

Use the exact line grammar `^\[(\d\d):(\d\d)\.(\d{3}) --> (\d\d):(\d\d)\.(\d{3})\] (.+)$`. Reject overlaps, negative duration, empty documents, and a dataset path outside an existing parent. `prepare_reference` writes these files atomically:

```text
dataset/
  benchmark.json
  reference-transcript.draft.md
  vocabulary.txt
  runs/
```

`benchmark.json` stores the expanded audio path, SHA-256, current-note path, creation timestamp, and `reference_status: "draft"`. The draft imports timestamped transcript text but strips speaker labels; it never modifies the source note.

- [ ] **Step 4: Run reference tests and type checks**

Run: `uv run pytest tests/benchmark/test_reference.py -q && uv run mypy audio_transcribe/benchmark/models.py audio_transcribe/benchmark/reference.py`

Expected: PASS.

- [ ] **Step 5: Update changelog and commit**

Add under `[Unreleased] / Added`: `Private reference dataset preparation for reproducible ASR benchmarks`.

```bash
git add pyproject.toml uv.lock CHANGELOG.md audio_transcribe/benchmark tests/benchmark
git commit -m "feat: add ASR benchmark reference format"
```

### Task 2: ASR Quality Metrics

**Files:**
- Create: `audio_transcribe/benchmark/asr_metrics.py`
- Modify: `audio_transcribe/benchmark/models.py`
- Test: `tests/benchmark/test_asr_metrics.py`

**Interfaces:**
- Consumes: `ReferenceDocument`, candidate segment dictionaries, reviewed vocabulary terms.
- Produces: `AsrMetrics(wer, cer, substitutions, deletions, insertions, reference_words, hypothesis_words, term_hits, term_total, term_recall, hallucinated_phrases, missed_phrases, boundary_errors)`.
- Produces: `evaluate_asr(reference: ReferenceDocument, segments: list[dict[str, Any]], terms: tuple[str, ...], boundaries: tuple[float, ...]) -> AsrMetrics`.

- [ ] **Step 1: Write failing tests for normalization, edit counts, terms, and safety signals**

```python
def test_evaluate_asr_counts_edits_and_terms() -> None:
    reference = ReferenceDocument(
        "sha",
        (ReferenceSpan(0.0, 4.0, "Следующий этап и ЧТЗ.", False),),
    )
    metrics = evaluate_asr(
        reference,
        [{"start": 0.0, "end": 4.0, "text": "Следственный этап."}],
        ("следующий этап", "ЧТЗ"),
        (),
    )
    assert metrics.substitutions == 1
    assert metrics.deletions == 2
    assert metrics.term_hits == 0
    assert metrics.term_total == 2
```

Add separate tests proving punctuation/case normalization, `[неразборчиво]` exclusion, an extra long candidate phrase counts as a hallucinated phrase, a missing reference span counts as missed, and a mismatched segment within 1.0 second of a supplied boundary increments `boundary_errors`.

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/benchmark/test_asr_metrics.py -q`

Expected: FAIL because `evaluate_asr` and `AsrMetrics` do not exist.

- [ ] **Step 3: Implement metrics using jiwer process functions**

Normalize Russian text with Unicode NFKC, lowercase, `ё → е`, punctuation-to-space, and collapsed whitespace. Use `jiwer.process_words` for word edits and `jiwer.process_characters` for CER. Define safety phrase matching by maximum overlap of time intervals; a non-empty candidate in an interval with no reference speech is hallucinated, and a non-empty reference interval with no overlapping candidate text is missed.

- [ ] **Step 4: Run tests, Ruff, and mypy**

Run: `uv run pytest tests/benchmark/test_asr_metrics.py -q && uv run ruff check audio_transcribe/benchmark tests/benchmark && uv run mypy audio_transcribe/benchmark`

Expected: PASS.

- [ ] **Step 5: Update changelog and commit**

Add under `[Unreleased] / Added`: `WER, CER, terminology, missed-speech, hallucination, and VAD-boundary metrics`.

```bash
git add CHANGELOG.md audio_transcribe/benchmark tests/benchmark
git commit -m "feat: add ASR benchmark quality metrics"
```

### Task 3: Configurable MLX-VAD Candidate Execution

**Files:**
- Modify: `audio_transcribe/stages/transcribe.py`
- Create: `audio_transcribe/benchmark/asr_runner.py`
- Modify: `audio_transcribe/benchmark/models.py`
- Test: `tests/stages/test_transcribe.py`
- Test: `tests/benchmark/test_asr_runner.py`

**Interfaces:**
- Produces: `MlxVadOptions(chunk_size_s: int = 30, condition_on_previous_text: bool = False, initial_prompt: str | None = None)`.
- Changes: `transcribe_mlx_vad(audio_path: str, model_size: str, language: str, options: MlxVadOptions | None = None)`.
- Produces: `AsrCandidate(name: str, model: str, options: MlxVadOptions)`.
- Produces: `AsrRunResult(candidate, result, stage_times_s, peak_rss_mb, vad_boundaries, status, error)`.
- Produces: `run_asr_candidate(audio: Path, candidate: AsrCandidate, align_model: str | None) -> AsrRunResult`.

- [ ] **Step 1: Extend existing mock tests with a failing options contract**

```python
options = MlxVadOptions(
    chunk_size_s=30,
    condition_on_previous_text=False,
    initial_prompt="Бусурина, ЧТЗ, опытная эксплуатация",
)
mod.transcribe_mlx_vad(str(dummy_audio), "large-v3", "ru", options)
mock_mlx.transcribe.assert_called_once_with(
    ANY,
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    language="ru",
    word_timestamps=False,
    condition_on_previous_text=False,
    initial_prompt="Бусурина, ЧТЗ, опытная эксплуатация",
)
```

Also assert `Pyannote.merge_chunks(..., chunk_size=30, ...)` receives `options.chunk_size_s`.

- [ ] **Step 2: Run tests and verify the signature failure**

Run: `uv run pytest tests/stages/test_transcribe.py -q`

Expected: FAIL because the fourth positional argument is unsupported.

- [ ] **Step 3: Implement options without changing production defaults**

Put `MlxVadOptions` in `audio_transcribe/stages/transcribe.py`. When `initial_prompt is None`, omit that keyword entirely so existing call assertions and mlx-whisper behavior remain unchanged. Record VAD chunk boundaries in the result under private key `_benchmark_vad_boundaries`; `build_output` must not copy private keys into normal pipeline JSON.

- [ ] **Step 4: Write runner failure-isolation and timing tests**

Mock preprocess, transcription, alignment, `resource.getrusage`, and monotonic time. Assert a raised candidate exception becomes `status="failed"` with error text and does not escape `run_asr_candidate`.

- [ ] **Step 5: Implement the runner**

The runner creates a temporary normalized WAV, calls `transcribe_mlx_vad`, calls existing `align`, measures each stage with `time.perf_counter`, samples peak RSS using the repository's macOS conversion rule, and deletes temporary audio in `finally`.

- [ ] **Step 6: Run focused and regression tests**

Run: `uv run pytest tests/stages/test_transcribe.py tests/benchmark/test_asr_runner.py tests/test_pipeline.py -q`

Expected: PASS with existing production behavior unchanged.

- [ ] **Step 7: Update changelog and commit**

Add under `[Unreleased] / Added`: `Isolated MLX-VAD benchmark candidates with vocabulary and context controls`.

```bash
git add CHANGELOG.md audio_transcribe/stages/transcribe.py audio_transcribe/benchmark tests/stages/test_transcribe.py tests/benchmark
git commit -m "feat: add configurable ASR benchmark runner"
```

### Task 4: Run Store, Forecasts, Ranking, and Reports

**Files:**
- Create: `audio_transcribe/benchmark/store.py`
- Create: `audio_transcribe/benchmark/report.py`
- Modify: `audio_transcribe/benchmark/models.py`
- Test: `tests/benchmark/test_store.py`
- Test: `tests/benchmark/test_report.py`

**Interfaces:**
- Produces: `RunManifest` with source/reference checksums, Git commit, model/revision, options, hardware, stage times, peak RSS, status, and error.
- Produces: `Forecast(duration_minutes: int, seconds: float, low_seconds: float, high_seconds: float)`.
- Produces: `forecast(rtf: float, duration_minutes: int) -> Forecast`.
- Produces: `rank_asr(results: tuple[EvaluatedAsrRun, ...], baseline_name: str) -> Ranking`.
- Produces: `write_report(dataset: Path, results: tuple[EvaluatedAsrRun, ...]) -> tuple[Path, Path]`.

- [ ] **Step 1: Write failing checksum and atomic-store tests**

Assert a run whose source or reference SHA differs from `benchmark.json` is rejected with `ValueError("benchmark checksum mismatch")`. Assert `manifest.json` and `result.json` are written through `atomic_write_text`.

- [ ] **Step 2: Write failing forecast and ranking tests**

```python
def test_forecast_includes_planning_range() -> None:
    value = forecast(0.5, 120)
    assert value.seconds == 3600.0
    assert value.low_seconds == 3060.0
    assert value.high_seconds == 4140.0

def test_rank_rejects_candidate_over_two_times_baseline() -> None:
    ranking = rank_asr((baseline_run(total=100, wer=0.3), candidate_run(total=201, wer=0.1)), "baseline")
    assert ranking.default_winner == "baseline"
    assert ranking.quality_profile == "candidate"
```

Add tests proving a candidate with better WER but more hallucinated or missed phrases cannot win.

- [ ] **Step 3: Run tests and verify failures**

Run: `uv run pytest tests/benchmark/test_store.py tests/benchmark/test_report.py -q`

Expected: FAIL because store/report modules do not exist.

- [ ] **Step 4: Implement store, forecasts, and deterministic ranking**

Use exact forecast durations `(30, 60, 120)`. Sort successful candidates by safety eligibility, WER, negative term recall, CER, then total time. Preserve failed runs in the report but exclude them from ranking.

- [ ] **Step 5: Implement `metrics.json` and `report.md`**

The Markdown table columns are candidate, WER, CER, term recall, substitutions/deletions/insertions, missed/hallucinated phrases, boundary errors, runtime, baseline ratio, 30/60/120-minute forecasts, peak RAM, and status. The recommendation states the default winner, optional quality profile, and reasons.

- [ ] **Step 6: Run focused tests and static checks**

Run: `uv run pytest tests/benchmark/test_store.py tests/benchmark/test_report.py -q && uv run ruff check audio_transcribe/benchmark tests/benchmark && uv run mypy audio_transcribe/benchmark`

Expected: PASS.

- [ ] **Step 7: Update changelog and commit**

Add under `[Unreleased] / Added`: `Checksum-bound benchmark manifests, quality ranking, and two-hour runtime forecasts`.

```bash
git add CHANGELOG.md audio_transcribe/benchmark tests/benchmark
git commit -m "feat: add ASR benchmark reports and forecasts"
```

### Task 5: Benchmark CLI

**Files:**
- Create: `audio_transcribe/benchmark/cli.py`
- Modify: `audio_transcribe/cli.py`
- Test: `tests/benchmark/test_cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces CLI group: `audio-transcribe benchmark`.
- Produces commands: `prepare`, `run-asr`, `evaluate-asr`, and `report`.

- [ ] **Step 1: Write failing CLI help and validation tests**

```python
def test_benchmark_help_lists_commands() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    for command in ("prepare", "run-asr", "evaluate-asr", "report"):
        assert command in result.output
```

Add tests that `prepare` rejects a missing MP3, `run-asr` rejects a draft reference, and `report` exits non-zero on checksum mismatch.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `uv run pytest tests/benchmark/test_cli.py tests/test_cli.py -q`

Expected: FAIL because the `benchmark` group is absent.

- [ ] **Step 3: Implement commands with explicit inputs**

Use these signatures:

```text
audio-transcribe benchmark prepare AUDIO --current-note NOTE --dataset DIR
audio-transcribe benchmark run-asr --dataset DIR --candidate baseline|vocabulary|russian|winner-context
audio-transcribe benchmark evaluate-asr --dataset DIR
audio-transcribe benchmark report --dataset DIR
```

`run-asr` reads the accepted vocabulary and model definitions from `benchmark.json`. It returns exit code 1 only for dataset/preflight failures; a model failure is persisted and printed but does not delete earlier runs.

- [ ] **Step 4: Run CLI and full non-ML tests**

Run: `uv run pytest tests/benchmark tests/test_cli.py tests/test_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Update documentation and commit**

Document the private benchmark commands in `README.md` without including the real path or transcript. Add the CLI feature to `[Unreleased]`.

```bash
git add README.md CHANGELOG.md audio_transcribe/cli.py audio_transcribe/benchmark tests/benchmark tests/test_cli.py
git commit -m "feat: add ASR benchmark CLI"
```

### Task 6: Prepare and Approve the Pilot Reference

**Files:**
- Create privately: `knowledge/projects/personal/audio-transcribe/benchmarks/2026-07-28-busurina-gk-plan-2/reference-transcript.draft.md`
- Create privately: `knowledge/projects/personal/audio-transcribe/benchmarks/2026-07-28-busurina-gk-plan-2/reference-transcript.md`
- Create privately: `knowledge/projects/personal/audio-transcribe/benchmarks/2026-07-28-busurina-gk-plan-2/vocabulary.txt`
- Modify privately: `knowledge/projects/personal/audio-transcribe/tasks/benchmark-asr-diarization-alternatives.md`

**Interfaces:**
- Produces an accepted `reference-transcript.md` and vocabulary consumed by Tasks 7 and the diarization plan.

- [ ] **Step 1: Create the dataset**

Run:

```bash
uv run audio-transcribe benchmark prepare \
  /Users/gnezim/Movies/2026-07-28_17-31-24_busurina-gk-plan-2.mp3 \
  --current-note /Users/gnezim/_projects/gnezim/knowledge/meetings/2026-07-28_17-31-24_busurina-gk-plan-2.md \
  --dataset /Users/gnezim/_projects/gnezim/knowledge/projects/personal/audio-transcribe/benchmarks/2026-07-28-busurina-gk-plan-2
```

Expected: draft files and `benchmark.json` are created without modifying the meeting note.

- [ ] **Step 2: Prepare the reference draft from the audio**

Listen in timestamped intervals, correct the imported text, and write one line per utterance using `[MM:SS.mmm --> MM:SS.mmm] text`. Mark only genuinely uncertain speech as `[неразборчиво]`; do not guess terminology.

- [ ] **Step 3: Prepare reviewed vocabulary**

Write one term per line, preserving intended spelling. Include only terms heard or strongly implied by the accepted reference, such as participant names, `ЧТЗ`, `РПР`, `опытная эксплуатация`, `приёмочные испытания`, `модернизация`, and `развитие`.

- [ ] **Step 4: User review gate**

Present every `[неразборчиво]` interval plus a link to the draft. Do not rename the draft to `reference-transcript.md` until the user approves corrections.

- [ ] **Step 5: Accept the reference**

Copy the approved content to `reference-transcript.md`, set `reference_status` to `accepted`, store its SHA-256 in `benchmark.json`, and update the vault benchmark task with the review date. Do not commit private data to this repository.

### Task 7: Convert the Russian Model and Run the ASR Benchmark

**Files:**
- Create locally outside Git: `/Users/gnezim/.cache/audio-transcribe/models/whisper-large-v3-russian-mlx/`
- Create privately: benchmark run directories, `metrics.json`, and `report.md`
- Modify privately: vault project page, roadmap, and benchmark task

**Interfaces:**
- Consumes accepted reference and CLI from Tasks 1–6.
- Produces the winning ASR candidate and aligned output consumed by the diarization plan.

- [ ] **Step 1: Convert the pinned Russian model reproducibly**

Run:

```bash
conversion_dir="$(mktemp -d)"
model_source="/Users/gnezim/.cache/audio-transcribe/models/whisper-large-v3-russian-hf"
git clone --filter=blob:none https://github.com/ml-explore/mlx-examples.git "$conversion_dir/mlx-examples"
git -C "$conversion_dir/mlx-examples" checkout 796f5b53cab69a3d48a44233ce21aae889e94a08
hf download antony66/whisper-large-v3-russian \
  --revision 3c24e46392ff376009a176ad290fe83e368edf7a \
  --local-dir "$model_source"
uv run python "$conversion_dir/mlx-examples/whisper/convert.py" \
  --torch-name-or-path "$model_source" \
  --mlx-path /Users/gnezim/.cache/audio-transcribe/models/whisper-large-v3-russian-mlx
```

Write `conversion-manifest.json` beside the converted model with the two pinned revisions, the SHA-256 of the source `model.safetensors`, the conversion command, and the conversion timestamp.

- [ ] **Step 2: Smoke-test the converted model**

Run a 30-second clip through `mlx_whisper.transcribe` with `language="ru"` and `condition_on_previous_text=False`. Expected: non-empty Russian text and no model-shape error.

- [ ] **Step 3: Run initial candidates**

Run:

```bash
dataset_dir="/Users/gnezim/_projects/gnezim/knowledge/projects/personal/audio-transcribe/benchmarks/2026-07-28-busurina-gk-plan-2"
for candidate in baseline vocabulary russian; do
  uv run audio-transcribe benchmark run-asr --dataset "$dataset_dir" --candidate "$candidate"
done
uv run audio-transcribe benchmark evaluate-asr --dataset "$dataset_dir"
uv run audio-transcribe benchmark report --dataset "$dataset_dir"
```

- [ ] **Step 4: Run the winner with previous-text context**

Read `metrics.json` to identify the best initial candidate, then run:

```bash
uv run audio-transcribe benchmark run-asr --dataset "$dataset_dir" --candidate winner-context
uv run audio-transcribe benchmark evaluate-asr --dataset "$dataset_dir"
uv run audio-transcribe benchmark report --dataset "$dataset_dir"
```

- [ ] **Step 5: Verify report completeness**

Confirm all successful candidates show WER, CER, term recall, safety counts, stage times, RTF, peak RAM, baseline ratio, and forecasts for 30/60/120 minutes. Confirm failed candidates remain visible with error text and are excluded from ranking.

- [ ] **Step 6: Run the repository quality gate**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
uv lock --check
```

Expected: all required checks pass; opt-in real-model tests may be skipped unless explicitly enabled.

- [ ] **Step 7: Update vault artifacts and commit only repository changes**

Update the project page, roadmap, and benchmark task with measured ASR results and the two-hour forecast. Keep private benchmark files in the vault. Commit any final repository documentation or test adjustments with a conventional commit and the matching changelog entry.

- [ ] **Step 8: ASR decision gate**

Present the report and request approval of the winning ASR input for the diarization plan. Do not change production defaults or the original meeting note yet.

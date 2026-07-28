# Diarization Quality Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare pyannote 3.1, Community-1, and Community-1 exclusive assignment against a verified two-speaker reference using the same winning aligned ASR output, then produce combined quality and two-hour performance recommendations.

**Architecture:** Extend the isolated benchmark package after the ASR plan has selected and frozen one aligned transcript. Add a speaker-reference parser, permutation-invariant diarization metrics, model adapters, and combined reporting. Production diarization remains unchanged until the user approves the final report.

**Tech Stack:** Python 3.12, pyannote.audio 4.x, WhisperX word timestamps, dataclasses, JSON, pytest, Ruff, strict mypy.

## Global Constraints

- Start only after the ASR plan has produced an approved winning aligned output and checksum.
- Use the same single pilot recording and private dataset directory.
- Evaluate exactly two speakers with `min_speakers=2` and `max_speakers=2`.
- Evaluate anonymous speaker assignment only; do not identify people by voice.
- Compare `pyannote/speaker-diarization-3.1`, `pyannote/speaker-diarization-community-1`, and Community-1 `exclusive_speaker_diarization`.
- Preserve source audio, speaker identities, references, and real outputs only in the private vault.
- A new default must improve diarization quality and keep the combined pipeline within 2× the original baseline total time.
- Forecast the combined pipeline for 30, 60, and 120 minutes with ±15% ranges.
- A failed or inaccessible gated model is recorded as failed and does not abort other candidates.
- Follow TDD and update changelog and required vault artifacts for progress.

---

## File Structure

- Create `audio_transcribe/benchmark/speaker_reference.py` for JSON validation and accepted turns.
- Create `audio_transcribe/benchmark/diarization_metrics.py` for collar-based DER components, label permutation, word accuracy, and fragmentation.
- Create `audio_transcribe/benchmark/diarization_runner.py` for pyannote model adapters and exclusive output.
- Extend `audio_transcribe/benchmark/models.py`, `store.py`, `report.py`, and `cli.py`.
- Add focused tests under `tests/benchmark/`.
- Modify `audio_transcribe/stages/diarize.py` only after final approval, and only to introduce the selected model/assignment behind a compatible default.

### Task 1: Speaker Reference Format

**Files:**
- Create: `audio_transcribe/benchmark/speaker_reference.py`
- Modify: `audio_transcribe/benchmark/models.py`
- Test: `tests/benchmark/test_speaker_reference.py`

**Interfaces:**
- Produces: `SpeakerTurn(start_s: float, end_s: float, speaker: str)`.
- Produces: `SpeakerReference(source_sha256: str, aligned_asr_sha256: str, turns: tuple[SpeakerTurn, ...])`.
- Produces: `load_speaker_reference(path: Path) -> SpeakerReference`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_load_speaker_reference_accepts_two_speakers(tmp_path: Path) -> None:
    path = tmp_path / "reference-speakers.json"
    path.write_text(
        '{"source_sha256":"audio","aligned_asr_sha256":"asr","turns":['
        '{"start":0.0,"end":2.0,"speaker":"speaker_a"},'
        '{"start":2.0,"end":4.0,"speaker":"speaker_b"}]}',
        encoding="utf-8",
    )
    reference = load_speaker_reference(path)
    assert {turn.speaker for turn in reference.turns} == {"speaker_a", "speaker_b"}
```

Add tests rejecting overlaps for one speaker, non-positive duration, unknown third speaker, missing checksums, and turns outside the probed audio duration.

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/benchmark/test_speaker_reference.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement strict loading**

Require exactly two non-empty speaker IDs and sorted turns. Permit overlapping turns between different speakers because real overlap is valid; reject self-overlap for the same speaker.

- [ ] **Step 4: Run tests and checks**

Run: `uv run pytest tests/benchmark/test_speaker_reference.py -q && uv run mypy audio_transcribe/benchmark/speaker_reference.py`

Expected: PASS.

- [ ] **Step 5: Update changelog and commit**

Add `[Unreleased] / Added`: `Private two-speaker reference format for diarization benchmarks`.

```bash
git add CHANGELOG.md audio_transcribe/benchmark tests/benchmark
git commit -m "feat: add diarization speaker reference"
```

### Task 2: Permutation-Invariant Diarization Metrics

**Files:**
- Create: `audio_transcribe/benchmark/diarization_metrics.py`
- Modify: `audio_transcribe/benchmark/models.py`
- Test: `tests/benchmark/test_diarization_metrics.py`

**Interfaces:**
- Produces: `DiarizationMetrics(der, missed_speech_s, false_alarm_s, confusion_s, word_speaker_accuracy, fragmented_turns, label_mapping)`.
- Produces: `evaluate_diarization(reference, predicted_turns, aligned_words, collar_s=0.0) -> DiarizationMetrics`.

- [ ] **Step 1: Write failing tests for label swaps and DER components**

```python
def test_swapped_anonymous_labels_are_not_confusion() -> None:
    reference = (
        SpeakerTurn(0.0, 5.0, "speaker_a"),
        SpeakerTurn(5.0, 10.0, "speaker_b"),
    )
    predicted = (
        SpeakerTurn(0.0, 5.0, "SPEAKER_01"),
        SpeakerTurn(5.0, 10.0, "SPEAKER_00"),
    )
    metrics = evaluate_diarization(reference, predicted, aligned_words=[])
    assert metrics.confusion_s == 0.0
    assert metrics.der == 0.0
    assert metrics.label_mapping == {"SPEAKER_01": "speaker_a", "SPEAKER_00": "speaker_b"}
```

Add exact fixtures for 1 second missed speech, 1 second false alarm, 1 second confusion, overlapped reference speech, word-level speaker assignment, and fragments shorter than 0.5 seconds.

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/benchmark/test_diarization_metrics.py -q`

Expected: FAIL because metrics are absent.

- [ ] **Step 3: Implement timeline scoring**

Build atomic boundaries from every reference and predicted turn edge, score each interval by active speaker sets, and search both possible two-speaker label mappings. Calculate `DER = (missed + false_alarm + confusion) / reference_speech`. Assign each aligned word by midpoint and calculate accuracy only for words whose midpoint has exactly one reference speaker.

- [ ] **Step 4: Run tests, Ruff, and mypy**

Run: `uv run pytest tests/benchmark/test_diarization_metrics.py -q && uv run ruff check audio_transcribe/benchmark tests/benchmark && uv run mypy audio_transcribe/benchmark`

Expected: PASS.

- [ ] **Step 5: Update changelog and commit**

Add `[Unreleased] / Added`: `Permutation-invariant DER and word-level speaker metrics`.

```bash
git add CHANGELOG.md audio_transcribe/benchmark tests/benchmark
git commit -m "feat: add diarization benchmark metrics"
```

### Task 3: Pyannote Candidate Adapters

**Files:**
- Create: `audio_transcribe/benchmark/diarization_runner.py`
- Modify: `audio_transcribe/benchmark/models.py`
- Test: `tests/benchmark/test_diarization_runner.py`

**Interfaces:**
- Produces: `DiarizationCandidate(name: str, model_id: str, exclusive: bool)`.
- Produces: `DiarizationRunResult(candidate, turns, word_assignments, time_s, peak_rss_mb, status, error)`.
- Produces: `run_diarization_candidate(audio: Path, aligned_result: dict[str, Any], token: str, candidate: DiarizationCandidate) -> DiarizationRunResult`.

- [ ] **Step 1: Write failing adapter tests**

Mock `pyannote.audio.Pipeline.from_pretrained`. Assert every call uses the candidate model ID, moves the pipeline to CPU, and calls it with `num_speakers=2`. For the exclusive candidate, return an object with both `speaker_diarization` and `exclusive_speaker_diarization` and assert exclusive turns are selected.

- [ ] **Step 2: Write a failing gated-access test**

Make `from_pretrained` raise an authorization exception and assert the returned run has `status="failed"` and actionable text naming the gated model and HF token.

- [ ] **Step 3: Run tests and verify failure**

Run: `uv run pytest tests/benchmark/test_diarization_runner.py -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 4: Implement adapters and word assignment**

Load audio through the coherent TorchCodec/PyAV runtime fixed in v0.5.1. Convert pyannote annotations to `SpeakerTurn` tuples. For normal outputs, assign each aligned word to the speaker with maximum temporal overlap; for Community-1 exclusive, use exclusive turns. Release pipeline objects and run `gc.collect()` after each candidate.

- [ ] **Step 5: Run focused tests and real import smoke test**

Run:

```bash
uv run pytest tests/benchmark/test_diarization_runner.py -q
uv run python -W error -c "import pyannote.audio; import torchcodec"
```

Expected: tests pass and imports produce no TorchCodec warning.

- [ ] **Step 6: Update changelog and commit**

Add `[Unreleased] / Added`: `pyannote 3.1 and Community-1 benchmark adapters with exclusive speaker assignment`.

```bash
git add CHANGELOG.md audio_transcribe/benchmark tests/benchmark
git commit -m "feat: add pyannote diarization benchmark runner"
```

### Task 4: Diarization CLI and Combined Report

**Files:**
- Modify: `audio_transcribe/benchmark/cli.py`
- Modify: `audio_transcribe/benchmark/store.py`
- Modify: `audio_transcribe/benchmark/report.py`
- Modify: `audio_transcribe/benchmark/models.py`
- Test: `tests/benchmark/test_cli.py`
- Test: `tests/benchmark/test_report.py`

**Interfaces:**
- Adds commands: `run-diarization`, `evaluate-diarization`.
- Extends `report` with combined ASR, alignment, and diarization time and quality.

- [ ] **Step 1: Write failing CLI tests**

Assert help includes the two new commands. Assert `run-diarization` refuses a draft/missing speaker reference, an aligned-ASR checksum mismatch, or any speaker range other than exactly two.

- [ ] **Step 2: Write failing combined-report tests**

Create synthetic ASR and diarization results and assert the final table includes DER, confusion, word speaker accuracy, fragmented turns, diarization runtime, combined baseline ratio, and 30/60/120-minute forecasts.

- [ ] **Step 3: Run tests and verify failure**

Run: `uv run pytest tests/benchmark/test_cli.py tests/benchmark/test_report.py -q`

Expected: FAIL because commands and combined fields are absent.

- [ ] **Step 4: Implement commands**

Use:

```text
audio-transcribe benchmark run-diarization --dataset DIR --candidate pyannote-3.1|community-1|community-1-exclusive
audio-transcribe benchmark evaluate-diarization --dataset DIR
audio-transcribe benchmark report --dataset DIR
```

Persist failed model access without removing successful candidates.

- [ ] **Step 5: Extend ranking and forecasts**

Rank by safety eligibility, DER, confusion, negative word-speaker accuracy, fragmented turns, then total combined runtime. Enforce the 2× gate against the original ASR+alignment+pyannote-3.1 baseline.

- [ ] **Step 6: Run benchmark and regression tests**

Run: `uv run pytest tests/benchmark tests/test_cli.py tests/stages/test_diarize.py -q`

Expected: PASS.

- [ ] **Step 7: Update README, changelog, and commit**

```bash
git add README.md CHANGELOG.md audio_transcribe/benchmark tests/benchmark
git commit -m "feat: add diarization benchmark CLI and report"
```

### Task 5: Prepare the Two-Speaker Reference

**Files:**
- Create privately: `reference-speakers.draft.json`
- Create privately: `reference-speakers.json`
- Modify privately: `benchmark.json`
- Modify privately: vault benchmark task

**Interfaces:**
- Produces accepted speaker turns bound to the source and winning aligned-ASR checksums.

- [ ] **Step 1: Generate a draft from the current diarization**

Export the current pyannote 3.1 turns into `reference-speakers.draft.json`, using stable private IDs `speaker_a` and `speaker_b`.

- [ ] **Step 2: Correct speaker turns against the audio**

Review every transition and overlap. Split wrong merged turns, join fragmentation belonging to one continuous speaker, retain genuine overlap, and never infer identity from voice embeddings.

- [ ] **Step 3: User review gate**

Present a compact list of disputed transitions with timestamps and the draft path. Do not accept the speaker reference until the user confirms it.

- [ ] **Step 4: Accept and checksum the reference**

Write `reference-speakers.json`, bind it to the source MP3 SHA and winning aligned ASR SHA, set `speaker_reference_status: "accepted"` in `benchmark.json`, and update the vault task.

### Task 6: Run Diarization Candidates and Produce the Final Recommendation

**Files:**
- Create privately: diarization run directories, final `metrics.json`, and `report.md`
- Modify privately: vault project page, roadmap, and benchmark task

**Interfaces:**
- Produces the final ASR+diarization recommendation and production-change approval gate.

- [ ] **Step 1: Verify gated model access**

Run a preflight load for `pyannote/speaker-diarization-community-1` using the cached `HF_TOKEN`. If access fails, record the failure in the report and provide the exact Hugging Face license/access action; continue with pyannote 3.1.

- [ ] **Step 2: Run all diarization candidates**

```bash
dataset_dir="/Users/gnezim/_projects/gnezim/knowledge/projects/personal/audio-transcribe/benchmarks/2026-07-28-busurina-gk-plan-2"
for candidate in pyannote-3.1 community-1 community-1-exclusive; do
  uv run audio-transcribe benchmark run-diarization --dataset "$dataset_dir" --candidate "$candidate"
done
uv run audio-transcribe benchmark evaluate-diarization --dataset "$dataset_dir"
uv run audio-transcribe benchmark report --dataset "$dataset_dir"
```

- [ ] **Step 3: Verify the final report**

Confirm the report contains ASR metrics, diarization metrics, stage times, combined RTF, peak RAM, baseline ratios, and forecasts for 30/60/120 minutes with ±15% ranges. Confirm the recommended default satisfies the 2× gate.

- [ ] **Step 4: Run the complete quality gate**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
uv lock --check
```

Expected: all required checks pass.

- [ ] **Step 5: Update vault artifacts**

Update the project page, roadmap, and benchmark task with measured DER, speaker confusion, word assignment accuracy, memory, runtimes, and two-hour forecast. Do not overwrite the original meeting note.

- [ ] **Step 6: Production-change gate**

Present the selected ASR and diarization configuration and request explicit approval before modifying production defaults, chez-mac, or the original meeting note.

### Task 7: Apply an Approved Production Default

**Files:**
- Modify conditionally: `audio_transcribe/stages/transcribe.py`
- Modify conditionally: `audio_transcribe/stages/diarize.py`
- Modify conditionally: `audio_transcribe/config.py`
- Modify conditionally: relevant tests
- Modify conditionally: `CHANGELOG.md`, `README.md`, vault project page, roadmap, and benchmark task

**Interfaces:**
- Consumes explicit user approval from Task 6.
- Produces a backwards-compatible default with the benchmark winner and a retained baseline profile.

- [ ] **Step 1: Write failing production-default tests from the approved report**

Assert the default model, vocabulary/context behavior, diarization model, and assignment mode match the approved winner. Add a named `baseline` profile preserving v0.5.1 behavior.

- [ ] **Step 2: Run focused tests and verify failure**

Run the exact config, transcribe, diarize, and CLI tests covering the approved fields.

- [ ] **Step 3: Implement only the approved changes**

Do not introduce any unbenchmarked option. Keep exactly-two-speaker forcing limited to the benchmark; production retains its configured min/max range unless separately approved.

- [ ] **Step 4: Run the full quality gate and one real pilot inference**

Run all repository checks, then process the pilot MP3 into a new output directory. Compare its checksums, metrics, and report to the approved benchmark run.

- [ ] **Step 5: Version, document, commit, and request release approval**

Add the correct semantic version and changelog entry, update vault artifacts, commit without `Co-Authored-By`, and stop before tag/push/deploy unless explicitly authorized.

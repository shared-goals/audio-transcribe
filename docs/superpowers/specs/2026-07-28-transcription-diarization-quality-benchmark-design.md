# Transcription and Diarization Quality Benchmark Design

## Objective

Improve the default transcription and diarization quality using one representative Russian meeting while keeping total processing time within twice the current baseline. The benchmark must identify which individual change improves quality, report measured runtime and memory, and forecast processing time for recordings up to two hours.

The pilot recording is `$BENCHMARK_AUDIO`. Its existing meeting note at `$BENCHMARK_NOTE` is evidence of the current quality problem, but neither that note nor the source audio will be overwritten or committed to the public repository.

## Decision Boundary

This benchmark uses one meeting as sufficient evidence for changing production defaults. A candidate may become the default only if it improves the agreed quality metrics and its total processing time is no more than twice the current baseline on the same recording. A slower quality winner may be retained as an optional quality profile.

ASR and diarization are evaluated sequentially. ASR is selected first because changing recognized text and word timing changes the input used for speaker assignment. Diarization candidates then receive the same aligned output from the winning ASR configuration.

## Private Reference Dataset

The benchmark dataset lives at `$BENCHMARK_DATASET` outside the repository and contains:

- The original MP3 by reference, without copying it into the repository.
- `reference-transcript.md`, containing accurate text with time intervals but no speaker labels.
- `reference-speakers.json`, containing verified intervals for exactly two real speakers.
- A meeting vocabulary file containing only reviewed names, abbreviations, and domain terms.
- One directory per benchmark run with its manifest, raw output, metrics, logs, runtime, and peak memory.
- `metrics.json` with machine-readable aggregate results.
- `report.md` with the human-readable comparison, forecasts, and recommendation.

Codex prepares the first transcript draft by listening to the audio and marks uncertain passages as `[неразборчиво]`. The user reviews only those passages and any other suspected errors before the draft becomes the reference. Intervals marked `[неразборчиво]` are excluded from automatic WER and CER calculations.

The source audio, reference transcript, vocabulary, speaker identities, and real meeting outputs must not enter the public GitHub mirror. The repository contains only generic benchmark code and synthetic fixtures.

## Benchmark Flow

```text
MP3
 ├─ draft reference → user review → accepted transcript and speaker reference
 ├─ generic Whisper baseline
 ├─ generic Whisper with meeting vocabulary
 └─ Russian fine-tuned Whisper converted to MLX
          ↓
   ASR metrics and winner selection
          ↓
   winner retested with previous-text context
          ↓
   final ASR winner and aligned words
          ├─ pyannote 3.1, exactly two speakers
          ├─ pyannote Community-1, exactly two speakers
          └─ Community-1 exclusive assignment, exactly two speakers
                    ↓
        diarization metrics, forecasts, and recommendation
```

Preprocessing and alignment remain identical across candidates unless a model contract makes that impossible. Any unavoidable difference is recorded in the run manifest and highlighted in the report.

## ASR Candidates

The following candidates run in order:

1. Current production baseline: `mlx-community/whisper-large-v3-mlx`, 30-second VAD chunks, `condition_on_previous_text=False`, no initial prompt.
2. The same baseline with a short reviewed meeting vocabulary passed through `initial_prompt`.
3. `antony66/whisper-large-v3-russian` converted into an MLX-compatible model, with 30-second VAD chunks and previous-text context disabled.
4. The best of candidates 1–3 retested with 30-second chunks and `condition_on_previous_text=True`.

The 60-second chunk variants are excluded because the pilot experiment already showed merged and missed speech and a hallucinated quiet-tail phrase. Meeting vocabulary is supplied directly to Whisper in the same inference call; it does not require an LLM or any additional inference stage.

The first vocabulary is a versioned, manually reviewed file specific to the pilot meeting. Automatic extraction from the vault is outside this scope.

## Diarization Candidates

The winning aligned ASR output is evaluated with:

1. `pyannote/speaker-diarization-3.1`.
2. `pyannote/speaker-diarization-community-1`.
3. Community-1 using `exclusive_speaker_diarization` for word-to-speaker assignment.

Every candidate is told that the recording contains exactly two speakers by setting both minimum and maximum speaker counts to two. Automatic speaker counting is a separate future experiment. Voice identification and mapping clusters to known people are also excluded; the benchmark evaluates segmentation and speaker assignment only.

## Metrics

### ASR

- Word error rate.
- Character error rate.
- Recall for reviewed names, abbreviations, and domain terms.
- Substitutions, deletions, and insertions.
- Missed phrases.
- Hallucinated phrases.
- Errors near VAD chunk boundaries.
- Number and duration of reference intervals excluded as `[неразборчиво]`.

### Diarization

- Diarization error rate.
- Speaker confusion.
- Missed speech.
- False alarm speech.
- Word-level speaker assignment accuracy.
- Count and duration of implausibly short fragmented speaker turns.

Speaker labels are permutation-invariant: the evaluator finds the best mapping between anonymous predicted labels and the two reference speakers before calculating confusion and word assignment accuracy.

### Performance

- Preprocessing time.
- ASR time.
- Alignment time.
- Diarization time.
- Total time.
- Real-time factor for each stage and the total pipeline.
- Peak resident memory.
- Number of VAD chunks and ASR segments.

## Forecast

For each successful candidate, the report forecasts processing time for 30-, 60-, and 120-minute recordings using the measured real-time factor:

`forecast_seconds = recording_duration_seconds * measured_real_time_factor`

The report shows the linear forecast and a ±15% planning range. It also shows the candidate's ratio to the baseline and whether it remains within the two-times limit. Forecasts are planning estimates, not latency guarantees; fixed model-loading overhead and non-linear diarization behavior are disclosed in the report.

## Selection Rules

ASR candidates are ranked primarily by WER, with hallucinations, missed phrases, and reviewed-term recall used as safety and tie-breaking signals. A candidate cannot win if it materially increases hallucinated or missed speech even when aggregate WER improves.

Diarization candidates are ranked primarily by DER and speaker confusion, with word-level speaker accuracy as a tie-breaker. A candidate cannot win if it improves DER by creating excessive fragmented turns that make the transcript harder to read.

The final default must satisfy the two-times runtime limit. The report retains all raw metrics and explains any judgment where the metrics disagree.

## Components

The benchmark is exposed through a new `audio-transcribe benchmark` CLI group with these bounded commands:

- `benchmark prepare` creates the private dataset structure and imports the current transcript only as a draft.
- `benchmark evaluate-asr` evaluates one or more ASR outputs against the accepted transcript reference.
- `benchmark evaluate-diarization` evaluates predicted turns and word assignments against the accepted speaker reference.
- `benchmark report` aggregates runs, calculates forecasts, applies selection rules, and writes `metrics.json` and `report.md`.

Every run has a manifest containing the Git commit, model repository and revision, parameters, hardware description, input checksum, reference checksum, timestamps, status, and failure information. A changed reference checksum invalidates prior aggregate conclusions until affected runs are evaluated again.

## Failure Handling

A failed candidate is marked `failed` with its traceback and environment captured in its run directory. Other candidates continue. Failed or incomplete runs never participate in winner selection.

The benchmark refuses to compare results created from different source or reference checksums unless explicitly producing a historical comparison. It reports missing model access, unsupported MLX conversion, unavailable Hugging Face licenses, and malformed reference intervals as actionable errors.

Production defaults and the original meeting note remain unchanged until the final report selects a winner and the user approves the production change.

## Testing

Repository tests include:

- Unit tests for WER, CER, term recall, exclusions, DER components, label permutation, word speaker accuracy, real-time factor, forecasts, and the two-times gate.
- Synthetic fixtures with known substitutions, deletions, insertions, hallucinations, missed speech, false alarms, and swapped speaker labels.
- Manifest and checksum validation tests.
- Failure-isolation tests proving one failed candidate does not abort the benchmark.
- Report snapshot or structural tests for the comparison table and recommendation.
- One opt-in end-to-end local benchmark using the pilot MP3 and downloaded models.

The final verification includes the existing Ruff, strict mypy, pytest, lockfile, and real-model macOS checks required by the repository.

## Scope Exclusions

The first implementation does not include:

- Automatic vocabulary extraction from the entire vault.
- LLM inference or post-processing.
- Voice identification or automatic person naming.
- Automatic speaker counting.
- Qwen3-ASR, Parakeet, VibeVoice, or Sortformer.
- A multi-meeting corpus.
- Automatic replacement of the original meeting note.

These exclusions keep the benchmark causal, private, and small enough to implement and validate as one project.

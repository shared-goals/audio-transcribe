# Reactive Pipeline Design

Incremental, multi-pass pipeline that delivers fast initial results and progressively enriches meeting notes through diarization, speaker identification, and user corrections.

## Motivation

Current pipeline is batch sequential — run everything, get results at the end. This design splits the pipeline into independent passes so you get a usable meeting note in minutes, with speakers and refined analysis arriving later.

## Architecture

CLI handles audio processing. Claudian handles all LLM analysis. A frontmatter flag (`reanalyze`) bridges them.

**Important:** All passes are sequential — do not run multiple passes on the same file concurrently. The `reanalyze` flag coordinates handoff between CLI and Claudian.

```
Audio file
  ↓ audio-transcribe process     (fast: preprocess → transcribe → align)
  ↓ Meeting note in vault         (transcript only, reanalyze: true)
  ↓ /process-meeting in Claudian  (summary, decisions, tasks → reanalyze: false)
  ↓ audio-transcribe diarize      (add speakers → reanalyze: true)
  ↓ audio-transcribe identify     (auto-match voices → reanalyze: true)
  ↓ User edits in Obsidian        (fix text, map speakers → reanalyze: true)
  ↓ /process-meeting in Claudian  (re-analyze → reanalyze: false)
```

## CLI Subcommands

All subcommands validate inputs at the CLI boundary and print clear error messages (e.g., `Error: audio_data not found in frontmatter`).

### `audio-transcribe process <audio-file> [-o output-dir]`

Fast pass: preprocess → transcribe → align → write meeting note.

- `-o` specifies the output directory; filename is derived from audio file
- Outputs meeting markdown with transcript (no speakers, no summary)
- Stores raw WhisperX JSON in `<output-dir>/.audio-data/<stem>.json`
- Stores original audio path in `audio_file` frontmatter field
- Sets `reanalyze: true` in frontmatter
- If no date found in audio filename, falls back to today's date

### `audio-transcribe diarize <meeting.md> [--force]`

Adds speaker diarization to an existing meeting note.

- Refuses if `## Speakers` section already exists unless `--force` is passed
- Reads stored JSON from `audio_data` frontmatter path
- Reads audio from `audio_file` frontmatter path (override with `--audio-file`)
- Runs pyannote diarization (supports `--min-speakers=1`)
- Adds speaker labels to existing transcript lines (preserves user text corrections)
- Adds `speakers` mapping to frontmatter
- Adds `## Speakers` legend to markdown
- If frontmatter `speakers` contains `[[wiki-links]]` not yet in speaker DB, enrolls their voice embeddings
- Warns if a speaker has no segments >= 1s for enrollment
- Sets `reanalyze: true`

### `audio-transcribe identify <meeting.md>`

Auto-matches speakers against known voice embeddings.

- Reads audio from `audio_file` frontmatter path (override with `--audio-file`)
- Extracts per-speaker embeddings from audio using stored JSON timestamps
- Cosine similarity against speaker DB
- High confidence (distance < 0.5): auto-maps in frontmatter + legend
- Low confidence: prints candidates for manual confirmation
- Sets `reanalyze: true` if any mappings changed

### `audio-transcribe update <meeting.md>`

Applies speaker mapping changes from frontmatter to transcript body.

- Reads `speakers` mapping from frontmatter
- Replaces speaker references using targeted regex (matches label positions only, not arbitrary text)
- Enrolls new `[[wiki-link]]` speakers in voice DB
- Warns if enrollment skipped due to short segments
- Sets `reanalyze: true`

### `audio-transcribe speakers list`

Shows all known speakers in the voice embedding DB with meeting count and last seen date.

### `audio-transcribe speakers forget <name>`

Removes a speaker's voice embeddings from the DB.

## Meeting Markdown Format

### After `process` (fast pass — no speakers)

```markdown
---
title: 2026-02-28 meeting
date: 2026-02-28
duration: 45:12
model: large-v3
reanalyze: true
audio_file: recordings/2026-02-28-meeting.mp3
audio_data: .audio-data/2026-02-28-meeting.json
---

## Transcript

[00:00] Привет, давайте начнём.
[00:05] Да, у меня есть обновления по проекту.
```

### After `diarize` (speakers added)

```markdown
---
title: 2026-02-28 meeting
date: 2026-02-28
duration: 45:12
model: large-v3
reanalyze: true
audio_file: recordings/2026-02-28-meeting.mp3
speakers:
  SPEAKER_00: Speaker A
  SPEAKER_01: Speaker B
audio_data: .audio-data/2026-02-28-meeting.json
---

## Speakers
- **Speaker A**: SPEAKER_00
- **Speaker B**: SPEAKER_01

## Transcript

[00:00] Speaker A: Привет, давайте начнём.
[00:05] Speaker B: Да, у меня есть обновления по проекту.
```

### After user maps speakers + Claudian analyzes

```markdown
---
title: Встреча по проекту
date: 2026-02-28
duration: 45:12
model: large-v3
reanalyze: false
audio_file: recordings/2026-02-28-meeting.mp3
speakers:
  SPEAKER_00: "[[Andrey]]"
  SPEAKER_01: "[[Maria]]"
audio_data: .audio-data/2026-02-28-meeting.json
---

## Speakers
- **[[Andrey]]**: SPEAKER_00
- **[[Maria]]**: SPEAKER_01

## Summary
- Обсудили статус проекта...

## Decisions
- ...

## Action Items
- [ ] @[[Andrey]]: подготовить отчёт к пятнице
- [ ] @[[Maria]]: обновить документацию

## Transcript

[00:00] [[Andrey]]: Привет, давайте начнём.
[00:05] [[Maria]]: Да, у меня есть обновления по проекту.
```

### Section ordering

Enforced by `SECTION_ORDER` constant in the markdown parser. `to_markdown()` sorts sections by this order. Unknown sections are appended at the end.

```
frontmatter
## Speakers          ← written by CLI (diarize), edited by user
## Summary           ← written/updated by Claudian
## Key Points        ← written/updated by Claudian
## Decisions         ← written/updated by Claudian
## Action Items      ← written/updated by Claudian
## Transcript        ← written by CLI, edited by user
```

### Data files per meeting

- `meetings/YYYY-MM-DD-title.md` — the living meeting note (user-facing)
- `meetings/.audio-data/YYYY-MM-DD-title.json` — raw WhisperX JSON (internal)

## Voice Embeddings & Speaker Identification

### Model

Uses `pyannote/wespeaker-voxceleb-resnet34-LM` — the same embedding model that `speaker-diarization-3.1` uses internally. Zero new dependencies.

```python
from pyannote.audio import Model, Inference
from pyannote.core import Segment

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")

# Extract embedding from a speaker's segment
embedding = inference.crop("audio.wav", Segment(13.37, 19.81))
```

### Enrollment (automatic)

When user maps `SPEAKER_00: "[[Andrey]]"` in frontmatter:

1. `diarize` or `update` detects new `[[wiki-link]]` not yet in speaker DB
2. Extracts embeddings from all segments belonging to SPEAKER_00 (>= 1s each)
3. Averages (centroid) into a single profile embedding
4. Stores to `~/.audio-transcribe/speakers/`
5. If no qualifying segments (all < 1s), logs a warning and skips enrollment

More meetings → more samples → better matching accuracy.

### Matching

`audio-transcribe identify meeting.md`:

1. Extracts per-speaker embeddings from audio using stored JSON timestamps
2. Cosine similarity against all known speakers
3. Distance < 0.5 → match, > 0.5 → unknown
4. Auto-maps high-confidence, prints uncertain for confirmation

### Storage

Speaker names are normalized to lowercase for all lookups. Display names are preserved in `index.json`.

```
~/.audio-transcribe/speakers/
  index.json           # {"andrey": {"display_name": "Andrey", "file": "andrey.npy", "samples": 5, "last_seen": "2026-02-28"}}
  andrey.npy           # numpy embedding vector
  maria.npy
```

### Upgrade path

If pyannote embedding quality is insufficient for verification, swap in:
- **SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) — better verification accuracy
- **WeSpeaker standalone** — built-in `register()`/`recognize()` API with MPS support

Same cosine matching, different embedding model.

## Claudian Integration

### `/process-meeting` command updates

The existing `/process-meeting` command handles both initial analysis and re-analysis:

1. **Read** meeting markdown
2. **Check** `reanalyze` flag
3. **If first analysis** (no Summary section):
   - Generate summary, decisions, action items from transcript
   - Create stub people cards for `[[Person]]` links not yet in vault
   - Insert sections between `## Speakers` and `## Transcript`
4. **If re-analysis** (`reanalyze: true`, Summary exists):
   - Re-read the edited transcript
   - Regenerate summary, decisions, action items
   - Update people card stubs for new `[[Person]]` links
   - Overwrite analysis sections in-place
5. **Set** `reanalyze: false`
6. **Auto-commit**

### People card stubs

Auto-created at `people/<Name>.md` when `[[Person]]` link doesn't exist in vault:

```markdown
---
type: person
created: 2026-02-28
---

# Name

First seen in [[2026-02-28-meeting-title]].
```

## End-to-End Workflow

```
1. Record meeting → meeting.m4a

2. Fast pass (minutes):
   $ audio-transcribe process meeting.m4a -o meetings/
   → meetings/2026-02-28-meeting.md  (transcript, reanalyze: true)

3. First analysis (Claudian):
   /process-meeting meetings/2026-02-28-meeting.md
   → adds Summary, Decisions, Action Items → reanalyze: false

4. Diarize (anytime after step 2, but sequential with step 3):
   $ audio-transcribe diarize meetings/2026-02-28-meeting.md
   → adds speaker labels → reanalyze: true

5. Auto-identify speakers (optional):
   $ audio-transcribe identify meetings/2026-02-28-meeting.md
   → matches voices against known speakers

6. Review + fix in Obsidian:
   - Fix transcript typos
   - Map speakers: Speaker A → [[Andrey]]
   - Set reanalyze: true

7. Re-analysis (Claudian):
   /process-meeting meetings/2026-02-28-meeting.md
   → re-generates analysis with correct speakers → reanalyze: false

8. Voice enrollment (automatic):
   Next diarize/identify run extracts [[Andrey]]'s embeddings
   for future auto-matching.
```

Steps 2-3 deliver a usable meeting note in minutes. Steps 4-8 progressively enrich it.

## Relationship to Unified CLI Design

This design builds on the existing unified CLI design (17-task plan). The package structure, stage abstraction, and models from that design remain. This adds:

- Multi-pass pipeline orchestration (process → diarize → identify → update)
- `reanalyze` frontmatter flag for Claudian integration
- Voice embedding speaker DB and identification
- In-place markdown updates across passes (diarize preserves user text edits)
- People card auto-creation
- Updated `/process-meeting` Claude command

Note: `pyyaml` must be added to unified CLI plan dependencies — the markdown parser requires it.

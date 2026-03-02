# People Card Updater + Speakers Legend — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Speakers legend format in `format.py`, extend `/process-meeting` to write the Speakers section with resolved names, and add a People Cards phase that updates participant cards with meeting backlinks.

**Architecture:** Three changes — (1) Python TDD fix to `format.py` to flip legend order from `**Speaker A**: SPEAKER_00` to `SPEAKER_00: Speaker A`; (2) prose addition to `knowledge/.claude/commands/process-meeting.md` to write `## Speakers` from `speakers:` frontmatter; (3) prose People Cards phase in the same command file. No new Python modules.

**Tech Stack:** Python, pytest, YAML frontmatter, Obsidian Markdown (vault Claude command)

**Design doc:** `docs/plans/2026-03-02-people-card-updater-design.md`

---

### Task 1: Fix Speakers legend format in `format.py`

**Files:**
- Modify: `audio_transcribe/stages/format.py:93` (format_transcript) and `:156` (format_meeting_note)
- Modify: `tests/stages/test_format.py`

The current code writes `- **Speaker A**: SPEAKER_00`. The new format is `- SPEAKER_00: Speaker A`. This affects both `format_transcript()` and `format_meeting_note()`.

**Step 1: Write failing test for `format_transcript` new legend format**

Add to `tests/stages/test_format.py`, replacing the assertion in `test_format_transcript_with_segments`:

```python
def test_format_transcript_with_segments():
    segs = [
        {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.0, "text": "Мир", "speaker": "SPEAKER_01"},
    ]
    md = format_transcript(_make_data(segs))
    assert "speakers: 2" in md
    assert "## Speakers" in md
    assert "SPEAKER_00: Speaker A" in md   # NEW: ID first
    assert "SPEAKER_01: Speaker B" in md   # NEW: ID first
    assert "**Speaker A**: SPEAKER_00" not in md  # OLD format must be gone
    assert "[00:00] Speaker A: Привет" in md
    assert "[00:03] Speaker B: Мир" in md
```

**Step 2: Add failing test for `format_meeting_note` legend format**

Add a new test to `tests/stages/test_format.py`:

```python
def test_format_with_speakers_legend_format():
    """Speakers section uses 'SPEAKER_ID: label' format (ID first)."""
    data = {
        "audio_file": "meeting.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 5.0, "text": "Пока", "speaker": "SPEAKER_01"},
        ],
    }
    result = format_meeting_note(data, audio_data_path=".audio-data/meeting.json")
    assert "SPEAKER_00: Speaker A" in result
    assert "SPEAKER_01: Speaker B" in result
    assert "**Speaker A**: SPEAKER_00" not in result
```

**Step 3: Run tests to confirm they fail**

```bash
cd /Users/gnezim/_projects/gnezim/audio-transcribe
uv run pytest tests/stages/test_format.py::test_format_transcript_with_segments tests/stages/test_format.py::test_format_with_speakers_legend_format -v
```

Expected: FAIL — `"**Speaker A**: SPEAKER_00"` assertion still passes (old format present) and new assertions fail.

**Step 4: Fix `format_transcript` in `format.py`**

In `format_transcript()` (around line 93), change:
```python
        lines.append(f"- **{label}**: {speaker_id}")
```
to:
```python
        lines.append(f"- {speaker_id}: {label}")
```

**Step 5: Fix `format_meeting_note` in `format.py`**

In `format_meeting_note()` (around line 156), change:
```python
        lines.append(f"- **{label}**: {speaker_id}")
```
to:
```python
        lines.append(f"- {speaker_id}: {label}")
```

**Step 6: Run targeted tests to confirm they pass**

```bash
uv run pytest tests/stages/test_format.py::test_format_transcript_with_segments tests/stages/test_format.py::test_format_with_speakers_legend_format -v
```

Expected: both PASS.

**Step 7: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass. (No other tests reference `**Speaker A**` — confirmed above.)

**Step 8: Commit**

```bash
git add audio_transcribe/stages/format.py tests/stages/test_format.py
git commit -m "fix: flip speakers legend to SPEAKER_ID: label format"
```

---

### Task 2: Add Speakers section writing to `/process-meeting`

**Files:**
- Modify: `knowledge/.claude/commands/process-meeting.md`

This is a prose instruction change. No Python. Work in the vault repo at `/Users/gnezim/_projects/gnezim/knowledge/`.

**Step 1: Read the current file**

Read `/Users/gnezim/_projects/gnezim/knowledge/.claude/commands/process-meeting.md`. Locate the "Analyze the transcript" section (currently lines 15–20), which lists sections to write as items 1–4.

**Step 2: Add Speakers section writing instruction**

Replace the "Analyze the transcript" block:

Old:
```markdown
**Analyze the transcript** and write these sections (in Russian) BEFORE `## Transcript`:

1. **## Summary** — 3-5 bullet points covering main topics
2. **## Key Points** — specific facts, numbers, statuses mentioned
3. **## Decisions** — decisions made, with rationale if available
4. **## Action Items** — as checkboxes: `- [ ] [[Person]]: description` (use wiki-links from speaker mapping)
```

New:
```markdown
**Analyze the transcript** and write these sections BEFORE `## Transcript`:

0. **## Speakers** — resolved speaker mapping from `speakers:` frontmatter (write this first, before Summary):
   - If `speakers:` is absent or empty, skip this section.
   - For each entry, write: `- SPEAKER_ID: [[Name]]` (if wiki-link resolved) or `- SPEAKER_ID: label` (if not yet resolved).
   - Example:
     ```
     ## Speakers

     - SPEAKER_00: [[Давыдов Денис]]
     - SPEAKER_01: [[Лукашев_Антон_Викторович]]
     ```
   - If section already exists (re-analysis), overwrite it.

1. **## Summary** — 3-5 bullet points covering main topics (in Russian)
2. **## Key Points** — specific facts, numbers, statuses mentioned (in Russian)
3. **## Decisions** — decisions made, with rationale if available (in Russian)
4. **## Action Items** — as checkboxes: `- [ ] [[Person]]: description` (use wiki-links from speaker mapping)
```

**Step 3: Save and verify**

Read the file back to confirm the change looks correct.

**Step 4: Commit in vault repo**

```bash
cd /Users/gnezim/_projects/gnezim/knowledge
git add .claude/commands/process-meeting.md
git commit -m "feat: write resolved Speakers section in process-meeting"
```

Expected: commit succeeds.

---

### Task 3: Add People Cards phase to `/process-meeting`

**Files:**
- Modify: `knowledge/.claude/commands/process-meeting.md`

Add the People Cards phase after the existing Task Creation phase. The file currently ends at the Task Creation "Confirm" block (line ~147). The new People Cards phase goes after it.

**Step 1: Read the current file end**

Read the last 15 lines of `/Users/gnezim/_projects/gnezim/knowledge/.claude/commands/process-meeting.md` to confirm the exact ending text.

**Step 2: Append the People Cards phase**

After the current final "**Confirm** to the user:" block (which ends with `- People card stubs created`), append the following new phase:

```markdown

**Update people cards** from the attendees:

**Check for attendees:** If `attendees:` frontmatter is absent or empty, skip people card updates entirely and go to the final summary.

**Step 1 — For each name in `attendees:`:**

a. **Find their card**: Check `people/{Name}.md`. If not found, skip this person (their stub was just created above; it will be enriched on the next run).

b. **Deduplication check**: If `### Key Conversations` already contains `[[{meeting-note-stem}]]`, skip this person.

c. **Infer topic**: From the meeting's `## Summary`, write a short Russian phrase (≤80 characters) describing what was discussed with or relevant to this person — their role, their tasks, or the key topic they were involved in.

d. **Prepend entry** under `### Key Conversations` (newest first):
```
- **{meeting-date}**: [[{meeting-note-stem}]] — {topic phrase}
```
If the section contains only template placeholder text (`- **YYYY-MM-DD**: Topic or key discussion point`), replace the placeholder with this entry.

e. **Update `last_contact` frontmatter**: Set `last_contact: {meeting-date}` only if the meeting date is newer than the current `last_contact` value, or if `last_contact` is empty.

f. **Save the updated person card** using the Edit tool.

**Step 2 — Auto-commit people cards:**
```bash
./scripts/auto-commit.sh people/
```

**Final summary** to the user:
- Meeting note path and sections written
- Task notes created (list filenames + project), any skipped duplicates
- People cards updated (list names), any skipped (not found or duplicate)
- People card stubs created
```

**Step 3: Save and verify**

Read the full file back to confirm the phase is appended correctly and the structure is coherent.

**Step 4: Commit in vault repo**

```bash
cd /Users/gnezim/_projects/gnezim/knowledge
git add .claude/commands/process-meeting.md
git commit -m "feat: add People Cards phase to process-meeting"
```

Expected: commit succeeds.

---

### Task 4: Update vault roadmap

**Files:**
- Modify: `knowledge/projects/personal/audio-transcribe/tasks/build-people-card-updater.md`
- Modify: `knowledge/projects/personal/audio-transcribe/roadmap.md`

**Step 1: Mark task done**

In `build-people-card-updater.md`:
- Set frontmatter `status: done`
- Change all `- [ ]` acceptance criteria to `- [x]`

**Step 2: Update roadmap**

In `roadmap.md`:
- Change `- [ ] Build people card updater (meeting backlinks)` to `- [x] Build people card updater (meeting backlinks)`
- Set frontmatter `overall_progress: 85`
- Set frontmatter `last_updated: 2026-03-02`
- Prepend to `## Change Log`:
  ```
  - 2026-03-02: People card updater + Speakers legend complete. `/process-meeting`
    now writes resolved `## Speakers` section from frontmatter and adds a People Cards
    phase: for each attendee, prepends a dated entry to `### Key Conversations` with
    a 1-line topic summary, updates `last_contact`, deduplicates on re-run. Also fixed
    `format.py` legend order to `SPEAKER_ID: label`. Roadmap at 85%.
  ```

**Step 3: Commit**

```bash
cd /Users/gnezim/_projects/gnezim/knowledge
git add projects/personal/audio-transcribe/tasks/build-people-card-updater.md
git add projects/personal/audio-transcribe/roadmap.md
git commit -m "chore: mark people-card-updater complete, roadmap to 85%"
```

---

### Task 5: Commit plan in audio-transcribe repo

```bash
cd /Users/gnezim/_projects/gnezim/audio-transcribe
git add docs/plans/2026-03-02-people-card-updater-plan.md
git commit -m "docs: add people card updater implementation plan"
```

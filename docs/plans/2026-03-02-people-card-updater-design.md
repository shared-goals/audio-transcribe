# People Card Updater + Speakers Legend — Design

**Date:** 2026-03-02
**Phase:** 5 (Enhancements)
**Scope:** Two related changes — (1) fix the Speakers legend format in `format.py`, (2) extend `/process-meeting` to write the Speakers section and update people cards after meeting analysis.

---

## Problem

1. The `## Speakers` section written by `format_meeting_note()` uses `**Speaker A**: SPEAKER_00` order — label first, ID second. The user wants ID first so the legend reads as a mapping: `SPEAKER_00: [[Name]]`.

2. `/process-meeting` does not currently write the `## Speakers` section with resolved real names from the `speakers:` frontmatter.

3. After a meeting is processed, participant people cards are not updated — no meeting backlinks, no `last_contact` date.

---

## Solution

Three changes, two locations:

### Change 1: `format.py` — flip Speakers legend order

**File:** `audio_transcribe/stages/format.py`

In `format_meeting_note()`, change the Speakers section line from:
```python
lines.append(f"- **{label}**: {speaker_id}")
```
to:
```python
lines.append(f"- {speaker_id}: {label}")
```

Also apply the same fix in `format_transcript()` (the standalone transcript formatter).

**Result — initial creation output:**
```markdown
## Speakers

- SPEAKER_00: Speaker A
- SPEAKER_01: Speaker B
```

**No section written** for fast pass (no diarization) — same as current behaviour.

---

### Change 2: `/process-meeting` — write Speakers section

**File:** `knowledge/.claude/commands/process-meeting.md`

Add to the "Analyze the transcript" step: after writing Summary/Key Points/Decisions/Action Items, also write/overwrite `## Speakers`.

Source: the `speakers:` frontmatter field (dict of `SPEAKER_ID → label`).

**If `speakers:` has resolved wiki-links** (e.g. after `audio-transcribe update`):
```markdown
## Speakers

- SPEAKER_00: [[Давыдов Денис]]
- SPEAKER_01: [[Лукашев_Антон_Викторович]]
- SPEAKER_02: [[Басалов Алексей]]
```

**If `speakers:` has unresolved labels** (e.g. `Speaker A`):
```markdown
## Speakers

- SPEAKER_00: Speaker A
- SPEAKER_01: Speaker B
```

**If `speakers:` is absent or empty:** skip the section (leave existing content untouched).

The `## Speakers` section is written BEFORE `## Summary` (it is already first in SECTION_ORDER).

---

### Change 3: `/process-meeting` — People Cards phase

**File:** `knowledge/.claude/commands/process-meeting.md`

New phase added after the Task Creation phase.

#### Trigger
Read `attendees:` list from meeting frontmatter. This list is set by the existing analysis step (wiki-link values from `speakers:` frontmatter, e.g. `[Давыдов Денис, Лукашев_Антон_Викторович]`). If list is empty or absent, skip the entire phase.

#### Per-attendee steps

For each name in `attendees:`:

**a. Find the person's card**
Check `people/{Name}.md`. If not found, skip (the people card stub step earlier in the command handles new people; the card will be present on the next run).

**b. Deduplication check**
If `### Key Conversations` already contains a link to this meeting's stem (e.g. `[[2026-02-27-aisa-tasks-role-overview]]`), skip this person.

**c. Infer topic**
From the meeting's `## Summary` section, write a short (≤80 char) Russian phrase describing what was discussed with or relevant to this person — their role, their tasks, or the key topic they were involved in.

**d. Prepend entry under `### Key Conversations`** (newest first):
```markdown
- **2026-02-27**: [[2026-02-27-aisa-tasks-role-overview]] — Обзор задач АИСА, роль в проекте КФ
```

If `### Key Conversations` contains only template placeholder text (`- **YYYY-MM-DD**: Topic or key discussion point`), replace it with the new entry.

**e. Update `last_contact` frontmatter**
Set `last_contact: YYYY-MM-DD` (meeting date) only if the meeting date is newer than the current `last_contact` value (or if `last_contact` is empty).

#### Auto-commit
```bash
./scripts/auto-commit.sh people/
```

#### Report
List updated people cards (names + files). List any skipped (not found, or duplicate).

---

## Edge Cases

| Situation | Behaviour |
|---|---|
| Fast pass (no diarization) | No `## Speakers` section in initial note; `/process-meeting` skips it if `speakers:` absent |
| `speakers:` has partial wiki-links | Write what's there; unresolved IDs show raw label |
| `attendees:` absent from frontmatter | Skip People Cards phase entirely |
| Person card not in `people/` | Skip that person (stub created earlier in same command run) |
| Meeting link already in Key Conversations | Skip person (deduplication) |
| `last_contact` already newer than meeting date | Don't overwrite |
| `### Key Conversations` has only template text | Replace placeholder, insert real entry |

---

## Constraints

- **`format.py` change is Python** — requires tests (TDD)
- **`/process-meeting` changes are prose** — Claude command, no Python
- **No new commands** — everything stays in existing files
- **Vault-only** for `/process-meeting` changes

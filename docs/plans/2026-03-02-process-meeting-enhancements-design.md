# Process-Meeting Enhancements Design

## Date: 2026-03-02

## Problem

1. `/process-meeting` assumes speakers already exist in frontmatter, but the default `audio-transcribe process` runs in fast-pass mode (no diarization). Users must remember to run `--full` or manually diarize — this is not obvious and leads to meeting notes without speaker labels.

2. After processing a meeting, neither the central meetings index (`meetings/meetings.md`) nor the relevant project page are updated with a link to the new meeting note. The user must manually add these references.

## Approach

Extend the existing `/process-meeting` command with two new phases (Approach 1 — single orchestrator, no new commands).

## Design

### Phase 0: Auto-Diarize (before analysis)

After reading the meeting note and checking `reanalyze`, but before reading the transcript:

1. Check if `speakers:` frontmatter is absent, `null`, or empty `{}`.
2. If missing, check that `audio_data:` frontmatter points to a valid `.audio-data/*.json` file.
3. Run:
   ```zsh
   cd /Users/gnezim/_projects/gnezim/audio-transcribe
   uv run audio-transcribe diarize <absolute-path-to-meeting-note>
   uv run audio-transcribe identify <absolute-path-to-meeting-note>
   ```
4. Re-read the meeting note (diarize/identify modify it in-place).
5. If `audio_data:` is missing or the commands fail, warn the user and continue without speakers (current behavior).

Slots in after the frontmatter check (current line 7-9) and before "Read the transcript" (current line 11).

### Final Phase: Update Meetings Index + Project Page (after people cards)

After people card updates, before the final summary.

#### Part A — Central meetings index (`meetings/meetings.md`)

1. Check if `meetings/meetings.md` exists.
2. Look for a `## Recent Meetings` section — if absent, append it before `## Related Documentation` (or at end).
3. Prepend a new entry under `## Recent Meetings` (newest first):
   ```
   - **YYYY-MM-DD HH:MM** — [[meeting-note-stem]] — {project name} — {one-line Russian topic from Summary}
   ```
   Date/time from frontmatter `date:` and `time:` fields.
4. Deduplication: if `[[meeting-note-stem]]` already in section, skip.

#### Part B — Project page

1. Use the same project chosen during task creation. If task creation was skipped, prompt the user for which project using the same numbered menu.
2. Find the project's main `.md` file (the one with `type: project` frontmatter) in `projects/{chosen-project}/`.
3. Look for a `## Meetings` section. If absent, insert before `## Tasks` (or append at end).
4. Prepend a new entry (newest first):
   ```
   - [[meeting-note-stem]] — {one-line Russian topic}
   ```
5. Deduplication: if `[[meeting-note-stem]]` already exists, skip.

#### Auto-commit

```zsh
./scripts/auto-commit.sh meetings/meetings.md projects/{chosen-project}/
```

## Changes Required

Only one file changes: `/Users/gnezim/_projects/gnezim/knowledge/.claude/commands/process-meeting.md`

No Python code changes needed — the CLI `diarize` and `identify` commands already exist.

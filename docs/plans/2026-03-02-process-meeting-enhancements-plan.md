# Process-Meeting Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `/process-meeting` command with auto-diarization and meetings index updates.

**Architecture:** Two new phases added to the existing command file. Phase 0 (auto-diarize) inserts before transcript reading. Final phase (index updates) inserts after people card updates and before the final summary. No Python code changes — uses existing CLI commands.

**Tech Stack:** Claude command (Markdown prompt file), `audio-transcribe diarize/identify` CLI

---

### Task 1: Insert auto-diarize phase into process-meeting command

**Files:**
- Modify: `/Users/gnezim/_projects/gnezim/knowledge/.claude/commands/process-meeting.md:10-13`

**Step 1: Insert the auto-diarize block**

Between the current line 9 (`proceed with analysis.`) and line 11 (`**Read the transcript**`), insert this new phase:

```markdown
**Auto-diarize if needed:**
- Check if `speakers:` frontmatter is absent, `null`, or empty (`{}`).
- If speakers are missing AND `audio_data:` frontmatter points to a valid file path:
  1. Run diarization:
     ```zsh
     cd /Users/gnezim/_projects/gnezim/audio-transcribe
     uv run audio-transcribe diarize <absolute-path-to-meeting-note>
     ```
  2. Run speaker identification:
     ```zsh
     uv run audio-transcribe identify <absolute-path-to-meeting-note>
     ```
  3. Re-read the meeting note (both commands modify it in-place).
  4. Report which speakers were detected.
- If `audio_data:` is missing, or either command fails, warn the user and continue without speakers (treat as single-speaker, same as current behavior).
```

**Step 2: Update the speaker mapping instruction**

The existing line 13 says: `If no speakers mapping exists, treat as single-speaker.`

Change this to: `If no speakers mapping exists after auto-diarize, treat as single-speaker.`

**Step 3: Commit**

```zsh
cd /Users/gnezim/_projects/gnezim/knowledge
./scripts/auto-commit.sh .claude/commands/process-meeting.md
```

If `auto-commit.sh` is not available, commit manually:
```zsh
git add .claude/commands/process-meeting.md
git commit -m "feat: add auto-diarize phase to /process-meeting command"
```

---

### Task 2: Add meetings index + project page update phase

**Files:**
- Modify: `/Users/gnezim/_projects/gnezim/knowledge/.claude/commands/process-meeting.md:190-197`

**Step 1: Insert the new phase between people card auto-commit and final summary**

After the people cards auto-commit block (current line 186-188: `./scripts/auto-commit.sh people/`) and before the final summary section (current line 192), insert a new `---` separated phase:

```markdown
---

**Update meetings index and project page:**

**Part A — Central meetings index (`meetings/meetings.md`):**

1. Check if `meetings/meetings.md` exists. If not, skip Part A.
2. Look for a `## Recent Meetings` section. If absent, insert it before `## Related Documentation` (or at the end of the file if that heading doesn't exist either).
3. **Deduplication**: If `[[{meeting-note-stem}]]` already appears in the `## Recent Meetings` section, skip Part A.
4. Prepend a new entry immediately after the `## Recent Meetings` heading (newest first):
   ```
   - **{date} {time}** — [[{meeting-note-stem}]] — {project name} — {one-line Russian topic from Summary}
   ```
   Where `{date}` and `{time}` come from frontmatter `date:` and `time:` fields. `{project name}` is the readable project name from the chosen project path (last segment, e.g. `eirc`, `audio-transcribe`). `{one-line Russian topic}` is a ≤80 char phrase summarizing the meeting from `## Summary`.

**Part B — Project page:**

1. Determine the project: use the same project chosen during task creation (Step 1 of the task phase). If task creation was skipped (user chose "0" or no action items existed), present the same numbered project menu now and ask the user to pick. Add option `0. Skip project page update`.
2. If user chose "0" or skipped, skip Part B.
3. Find the project's main `.md` file — the file with `type: project` in its frontmatter inside `projects/{chosen-project}/`. Read it.
4. Look for a `## Meetings` section. If absent, insert it before `## Tasks` (or at the end of the file if `## Tasks` doesn't exist).
5. **Deduplication**: If `[[{meeting-note-stem}]]` already appears in the `## Meetings` section, skip Part B.
6. Prepend a new entry immediately after the `## Meetings` heading (newest first):
   ```
   - [[{meeting-note-stem}]] — {one-line Russian topic}
   ```

**Auto-commit index updates:**
```zsh
./scripts/auto-commit.sh meetings/meetings.md projects/{chosen-project}/
```

If `auto-commit.sh` is not available:
```zsh
git add meetings/meetings.md "projects/{chosen-project}/"
git commit -m "docs: update meetings index and project page for {meeting-note-stem}"
```
```

**Step 2: Update the final summary to include new outputs**

Replace the current final summary block (line 192-197) with:

```markdown
**Final summary** to the user:
- Meeting note path and sections written/updated
- Auto-diarization: whether it ran, how many speakers detected (or skipped/failed)
- Task notes created (list filenames + project), any skipped duplicates
- People cards updated (list names), any skipped (not found or duplicate)
- People card stubs created
- Meetings index updated (or skipped — duplicate/missing)
- Project page updated (or skipped — duplicate/user skipped)
```

**Step 3: Commit**

```zsh
cd /Users/gnezim/_projects/gnezim/knowledge
./scripts/auto-commit.sh .claude/commands/process-meeting.md
```

If `auto-commit.sh` is not available:
```zsh
git add .claude/commands/process-meeting.md
git commit -m "feat: add meetings index + project page updates to /process-meeting"
```

---

### Task 3: Update vault project page with new task entries

**Files:**
- Modify: `/Users/gnezim/_projects/gnezim/knowledge/projects/personal/audio-transcribe/audio-transcribe.md:191-194`

**Step 1: Move task entries to correct sections**

In the `## Tasks` section, move the completed task entries and update the Active section to reflect the new enhancements work. The `build-task-extraction` and `build-people-card-updater` tasks are done (Phase 5 completed 2026-03-02 per memory). Update accordingly.

**Step 2: Commit**

```zsh
cd /Users/gnezim/_projects/gnezim/knowledge
git add projects/personal/audio-transcribe/audio-transcribe.md
git commit -m "docs: update audio-transcribe task statuses"
```

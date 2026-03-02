# Task Note Extractor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `/process-meeting` Claude command to create vault task notes from action items after meeting analysis.

**Architecture:** Single file change — append a "Task Creation" phase to `knowledge/.claude/commands/process-meeting.md`. Claude discovers available `tasks/` project directories, asks the user which project to use, then creates one task note per action item with enriched frontmatter. No Python code changes.

**Tech Stack:** Claude command (Markdown prose instructions), Obsidian vault, `knowledge/templates/task-note.md`

**Design doc:** `docs/plans/2026-03-02-task-extraction-design.md`

---

### Task 1: Extend `/process-meeting` with Task Creation phase

**Files:**
- Modify: `knowledge/.claude/commands/process-meeting.md` (vault repo, not audio-transcribe)

This is a prose instruction change — no tests to write. The implementation is the new instructions themselves.

**Step 1: Read the current command**

Open `knowledge/.claude/commands/process-meeting.md`. Note the current last step is "Confirm to the user". The new Task Creation phase goes after the auto-commit step, just before the final confirmation.

**Step 2: Write the new Task Creation section**

Replace the final two steps (Auto-commit + Confirm) with the expanded version below.

The complete replacement for lines 48–53 (`**Auto-commit:**` through end of file):

```markdown
**Auto-commit:**
```bash
./scripts/auto-commit.sh <meeting-note-path>
```

**Create task notes** from the Action Items:

**Step 1 — List available project task directories.**
Use the Glob tool to find all `tasks/` directories under `projects/`:

Pattern: `projects/**/tasks`

Present a numbered menu of matches (strip the `projects/` prefix and `/tasks` suffix for readability). Add option `0. Skip task creation`. If the meeting frontmatter has a `project:` field, mark that entry as `(default)`.

Example output:
```
Found 5 action items. Where should task notes go?

 1. work/bft/eirc
 2. work/bft/svod-excel-generator
 3. personal/audio-transcribe
 4. personal/homelab
 ... (all projects)
 0. Skip task creation
```

**Step 2 — Wait for user to choose** a number (or "skip" / "0").
If user says "skip" or "0", skip task creation entirely and go to the final confirmation.

**Step 3 — For each action item checkbox in `## Action Items`:**

Parse format: `- [ ] [[Person]]: description text`

For each item:

a. **Compute slug**: 2-4 Latin keyword words from the description (transliterate or extract key nouns). Use only lowercase letters and hyphens. Examples: `plan-migration`, `review-chatbot`, `prepare-report`.

b. **Compute filename**: `{meeting-date}-{slug}.md` where `meeting-date` is the `date:` field from frontmatter.

c. **Check for duplicate**: If `projects/{chosen-project}/tasks/{filename}` already exists — skip this item and note it as skipped.

d. **Infer enrichment from description and meeting context:**
- `priority`:
  - `high` if description contains: срочно, критично, немедленно, asap, до [конкретной даты в ближайшую неделю]
  - `low` if description contains: при возможности, в перспективе, когда будет время, не срочно
  - `medium` otherwise
- `due`: extract an explicit date if mentioned (ISO format YYYY-MM-DD), otherwise `null`
- `tags`: 1-3 lowercase topic keywords from the description content (e.g. `migration`, `postgresql`, `reporting`, `chatbot`)

e. **Write task note** at `projects/{chosen-project}/tasks/{filename}`:

```yaml
---
type: task
status: backlog
parent: {chosen-project}/{project-id}
assignee: [[Person]]
meeting: [[{meeting-note-stem}]]
created: {meeting-date}
due: {due or null}
priority: {inferred priority}
milestone: false
tags: {inferred tags list}
depends_on: []
blocks: []
permalink: knowledge/projects/{chosen-project}/tasks/{slug-without-date}
---

# {Task title — Russian, from action item description}

## Description

{Full action item text. If the meeting context adds relevant detail (e.g. a collaborator mentioned, a specific system named), include it here.}

## Related

- [[meetings/{meeting-note-stem}]]
```

Where `{project-id}` is the last path segment of the project directory repeated (e.g. `eirc/eirc`, `audio-transcribe/audio-transcribe`).

f. **Update the action item checkbox** in the meeting note — append ` → [[{filename-without-ext}]]` to the line:

Before: `- [ ] [[Anton]]: Проработать план миграции`
After:  `- [ ] [[Anton]]: Проработать план миграции → [[2026-02-27-plan-migration]]`

**Step 4 — Auto-commit the new task files:**
```bash
./scripts/auto-commit.sh projects/{chosen-project}/tasks/
```

**Confirm** to the user:
- Meeting note path
- Sections written/updated
- Task notes created (list filenames + project)
- Any skipped items (duplicates)
- People card stubs created
```

**Step 3: Save the modified command file**

Write the updated `process-meeting.md` using the Edit tool. The file lives in the vault at `knowledge/.claude/commands/process-meeting.md`.

**Step 4: Commit in the vault repo**

```bash
cd /Users/gnezim/_projects/gnezim/knowledge
git add .claude/commands/process-meeting.md
git commit -m "feat: extend process-meeting with task note creation"
```

Expected: commit succeeds.

---

### Task 2: Smoke test — create tasks from a real meeting

This is a manual verification step using an existing meeting note.

**Step 1: Set `reanalyze: true` on the test meeting**

Edit frontmatter of `meetings/2026-02-27-aisa-tasks-role-overview.md`:
```yaml
reanalyze: true
```

**Step 2: Run `/process-meeting` in the vault**

In a Claudian session in the vault, run:
```
/process-meeting meetings/2026-02-27-aisa-tasks-role-overview.md
```

**Step 3: Verify the project selection prompt appears**

Expected: Claude lists available project directories and asks which one to use.

**Step 4: Select `work/bft/eirc`**

Expected: Claude creates task files under `projects/work/bft/eirc/tasks/`.

**Step 5: Verify task files created**

```bash
ls /Users/gnezim/_projects/gnezim/knowledge/projects/work/bft/eirc/tasks/
```

Expected: 5-6 new files named `2026-02-27-*.md`.

**Step 6: Spot-check one task file**

Open one file. Verify:
- [ ] Frontmatter has `type: task`, `assignee`, `meeting`, `priority`, `tags`
- [ ] `## Description` contains Russian text from the action item
- [ ] `## Related` links back to the meeting note

**Step 7: Verify meeting note action items updated**

Open `meetings/2026-02-27-aisa-tasks-role-overview.md`. Verify action item lines end with `→ [[2026-02-27-*]]` wiki-links.

**Step 8: Verify deduplication — re-run without clearing files**

Set `reanalyze: true` again and re-run `/process-meeting` on the same meeting. Select the same project.

Expected: Claude reports all items as skipped (already exist), no duplicate files created.

---

### Task 3: Update vault roadmap

**Files:**
- Modify: `knowledge/projects/personal/audio-transcribe/roadmap.md`
- Modify: `knowledge/projects/personal/audio-transcribe/tasks/build-task-extraction.md`

**Step 1: Mark task as complete in the vault task file**

In `tasks/build-task-extraction.md`, update frontmatter:
```yaml
status: done
```

And tick all acceptance criteria checkboxes.

**Step 2: Update roadmap**

In `roadmap.md`, under Phase 5, change:
```markdown
- [ ] Build task note extractor (separate notes in `projects/.../tasks/`)
```
to:
```markdown
- [x] Build task note extractor (separate notes in `projects/.../tasks/`)
```

Also update `overall_progress` from `75` to `80` and add a changelog entry.

**Step 3: Commit vault changes**

```bash
cd /Users/gnezim/_projects/gnezim/knowledge
git add projects/personal/audio-transcribe/
git commit -m "chore: mark task-extraction complete, update roadmap"
```

---

### Task 4: Commit design + plan in audio-transcribe repo

**Files:**
- Already committed: `docs/plans/2026-03-02-task-extraction-design.md`
- Commit: `docs/plans/2026-03-02-task-extraction-plan.md`

**Step 1: Commit the plan file**

```bash
cd /Users/gnezim/_projects/gnezim/audio-transcribe
git add docs/plans/2026-03-02-task-extraction-plan.md
git commit -m "docs: add task extraction implementation plan"
```

Expected: commit succeeds.

# Task Note Extractor — Design

**Date:** 2026-03-02
**Phase:** 5 (Enhancements)
**Scope:** Extend `/process-meeting` Claude command to create vault task notes from action items

---

## Problem

After `/process-meeting` runs, action items live only as checkboxes in the meeting note. They are not discoverable as tasks in the relevant project's `tasks/` directory, have no assignee frontmatter, no priority, and no backlink from the project.

## Solution

Extend the existing `/process-meeting` command with a **Task Creation** phase that runs after the analysis is complete. No new CLI subcommand or separate Claude command. No code changes to `audio_transcribe/` Python package.

---

## Flow

### Existing steps (unchanged)

1. Read meeting note + frontmatter
2. Check `reanalyze` flag
3. Analyze transcript → write Summary, Key Points, Decisions, Action Items
4. Update frontmatter (`reanalyze: false`, `title`, `attendees`)
5. Create people card stubs
6. Save file, auto-commit

### New step 7: Task Creation

After saving the meeting note:

**7a. Discover project directories**
Claude lists all `tasks/` subdirectories in the vault, presents as a numbered menu:

```
Found 6 action items. Which project should tasks go in?

 1. work/bft/eirc
 2. work/bft/svod-excel-generator
 3. personal/audio-transcribe
 4. personal/homelab
 ... (all projects with tasks/ dirs)
 0. Skip task creation
```

**7b. User selects**
User replies with a number (or "0" / "skip"). If the meeting note already has a `project:` field in frontmatter, that project is highlighted as the default.

**7c. Create task notes**
For each action item checkbox:
- Parse assignee (`[[Person]]`) and description text
- Infer: priority (high/medium/low from urgency language), due date (from explicit date mentions or null), tags (from content keywords)
- Check for existing file with same slug — skip if found (deduplication)
- Write task note file (see format below)
- Append `→ [[task-filename]]` wiki-link to the action item checkbox in the meeting note

**7d. Report**
Confirm: list of created files, any skipped (duplicate), path to project tasks dir.

---

## Task Note Format

**Filename:** `{meeting-date}-{slug}.md`
- `meeting-date`: from frontmatter `date:` field
- `slug`: short Latin keyword(s) from the title (e.g., `plan-migration`, `review-chatbot`)
- Placed in: `projects/{selected-project}/tasks/`

**Content:**

```yaml
---
type: task
status: backlog
parent: {project-path}/{project-id}
assignee: [[Person]]
meeting: [[meeting-filename]]
created: YYYY-MM-DD
due: null                    # or inferred date if mentioned explicitly
priority: medium             # inferred: high if urgent/deadline language
milestone: false
tags: []                     # inferred from description content
depends_on: []
blocks: []
permalink: knowledge/projects/{project-path}/tasks/{filename-without-ext}
---

# Task title (Russian, from action item)

## Description

Full action item text, expanded with any relevant context from the meeting.

## Related

- [[meetings/meeting-filename]]
```

**Enrichment rules:**
- `priority: high` — if description contains deadline, urgency words (срочно, критично, немедленно, до [date])
- `priority: low` — if prefaced with "при возможности", "в перспективе", "когда будет время"
- `priority: medium` — default
- `due:` — extract explicit dates mentioned in the action item (e.g., "до вторника", "к концу недели", ISO date)
- `tags:` — 1-3 topic keywords from description (e.g., `migration`, `postgresql`, `reporting`)

---

## Updated Meeting Note

Each action item checkbox gets a wiki-link appended after creation:

```markdown
## Action Items

- [ ] [[Anton]]: Проработать план миграции → [[2026-03-02-plan-migration]]
- [ ] [[Denis]]: Связаться с Антоном во вторник → [[2026-03-02-contact-anton]]
```

---

## Deduplication

Before creating each task file:
1. Compute the target filename `{date}-{slug}.md`
2. If that file already exists in the target `tasks/` dir → skip, report to user
3. Re-running `/process-meeting` with `reanalyze: true` will not create duplicate task files

---

## Constraints

- **No Python changes** — implemented entirely as a Claude command (markdown instructions)
- **No new commands** — extends `knowledge/.claude/commands/process-meeting.md`
- **Vault-only** — reads/writes within the vault; no interaction with `audio_transcribe/` package
- **Russian-first** — task titles and descriptions remain in Russian; slug is Latin keywords only

---

## Out of Scope

- Mapping tasks to multiple projects (one project per meeting)
- Automated priority ML inference (rule-based keyword matching only)
- Syncing task status back to meeting note checkboxes
- People card updater (separate Phase 5 task)

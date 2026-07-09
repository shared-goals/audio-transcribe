# /release Skill + Changelog Rule

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automate versioning, changelog, and tagging via a `/release` slash command skill, with an incremental CLAUDE.md rule to keep changelog updated per-commit.

**Architecture:** A `.claude/commands/release.md` skill that Claude Code executes as a slash command. It reads git state, proposes a semver bump, updates version + changelog, commits, tags, and optionally pushes. A CLAUDE.md rule ensures `[Unreleased]` stays current between releases.

---

### Task 1: Add changelog rule to CLAUDE.md

**Files:** Modify: `CLAUDE.md`

Add a `## Release & Changelog` section with the incremental rule: when committing fix:/feat:/breaking changes, also update `[Unreleased]` in CHANGELOG.md.

### Task 2: Create `/release` slash command skill

**Files:** Create: `.claude/commands/release.md`

The skill instructs Claude to:
1. Run `git tag --sort=-v:refname | head -1` to find last tag
2. Run `git log <tag>..HEAD --oneline` to see commits since
3. Read `CHANGELOG.md` `[Unreleased]` section
4. Fill any gaps — commits not reflected in `[Unreleased]`
5. Auto-detect bump: fix:→patch, feat:→minor, BREAKING CHANGE→major
6. Present recommendation, let user override
7. Bump version in `pyproject.toml` and `audio_transcribe/__init__.py`
8. Stamp `[Unreleased]` → `[X.Y.Z] - YYYY-MM-DD`, add fresh `[Unreleased]`
9. Commit as `release: vX.Y.Z`, tag `vX.Y.Z`
10. Ask to push with `--follow-tags`

### Task 3: Perform immediate v0.2.1 release

Use the new workflow to release v0.2.1 for the 2 pending fix commits.

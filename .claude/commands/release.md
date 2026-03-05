Release a new version. Follows semver, Keep a Changelog format.

## Steps

### 1. Gather context

Run these commands and read the results:
- `git tag --sort=-v:refname | head -1` — find the last release tag
- `git log <last-tag>..HEAD --oneline` — commits since last release
- Read `CHANGELOG.md` — check the `[Unreleased]` section
- Read current version from `pyproject.toml` (the `version = "..."` line)

If there are no commits since the last tag, stop and tell the user there's nothing to release.

### 2. Fill changelog gaps

Compare the commit list against the `[Unreleased]` section. For any `fix:` or `feat:` commits not already reflected in `[Unreleased]`, add them under the appropriate heading (`### Added` for `feat:`, `### Fixed` for `fix:`, `### Changed` for `refactor:` or other changes). Use concise, user-facing descriptions (not raw commit messages). Group related commits into single entries when appropriate.

Present the updated `[Unreleased]` section to the user for review before proceeding.

### 3. Detect and confirm bump level

Count commit types since the last tag:
- Any `BREAKING CHANGE` in commit body or `!` after type → **major**
- Any `feat:` → **minor**
- Only `fix:`, `docs:`, `refactor:`, etc. → **patch**

Present: "This looks like a **patch** bump (v0.2.0 → v0.2.1): N fixes, N features. Override? [patch/minor/major]"

Wait for user confirmation or override before proceeding.

### 4. Apply version bump

Update the version string in both files:
- `pyproject.toml`: the `version = "X.Y.Z"` line
- `audio_transcribe/__init__.py`: the `__version__ = "X.Y.Z"` line

### 5. Stamp the changelog

In `CHANGELOG.md`:
- Replace `## [Unreleased]` content by renaming it to `## [X.Y.Z] - YYYY-MM-DD` (use today's date)
- Add a fresh empty `## [Unreleased]` section above it

### 6. Commit and tag

```bash
git add pyproject.toml audio_transcribe/__init__.py CHANGELOG.md
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
```

### 7. Offer to push

Ask the user: "Push commit + tag to origin? [y/n]"

If yes:
```bash
git push origin main --follow-tags
```

If no, remind them to push later with `git push origin main --follow-tags`.

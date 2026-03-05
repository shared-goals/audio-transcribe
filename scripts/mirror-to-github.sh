#!/bin/zsh
# Mirror audio-transcribe to GitHub, excluding private files.
#
# Excluded:
#   docs/plans/     — internal design docs with personal paths/references
#   CLAUDE.md       — Claude Code instructions with personal paths
#   .claude/        — Claude Code project config
#
# The script creates a temporary clean clone, removes excluded paths,
# and force-pushes to the GitHub remote.
#
# Usage: zsh scripts/mirror-to-github.sh
set -e

GITHUB_REMOTE="git@github.com:shared-goals/audio-transcribe.git"
BRANCH="main"

# Paths to exclude from the public mirror
EXCLUDE_PATHS=(
    "docs/plans"
    "CLAUDE.md"
    ".claude"
    ".gitea"
    "scripts"
)

# PII patterns to scan for (fail-safe)
PII_PATTERNS=(
    "gnerim\.ru"
    "/Users/gnezim"
    "hf_[a-zA-Z0-9]{20,}"
)

info()  { printf '\033[1;34m==>\033[0m %s\n' "$1" }
ok()    { printf '\033[1;32m✓\033[0m %s\n' "$1" }
err()   { printf '\033[1;31m✗\033[0m %s\n' "$1" }

REPO_ROOT="$(git rev-parse --show-toplevel)"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

info "Creating clean clone..."
git clone --single-branch --branch "$BRANCH" "$REPO_ROOT" "$TMPDIR/repo" --quiet

cd "$TMPDIR/repo"

info "Removing excluded paths..."
for path in "${EXCLUDE_PATHS[@]}"; do
    if [ -e "$path" ]; then
        git rm -rf "$path" --quiet 2>/dev/null || true
        ok "Removed $path"
    fi
done

# Commit the removals
git commit --quiet -m "mirror: remove private files" --allow-empty

info "Scanning for PII leaks..."
FOUND_PII=0
for pattern in "${PII_PATTERNS[@]}"; do
    # Search tracked files only (exclude .git)
    MATCHES=$(grep -rn --include='*.py' --include='*.sh' --include='*.md' --include='*.toml' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.txt' -E "$pattern" . 2>/dev/null | grep -v '\.git/' || true)
    if [ -n "$MATCHES" ]; then
        err "PII pattern '$pattern' found:"
        echo "$MATCHES" | head -10
        FOUND_PII=1
    fi
done

if [ "$FOUND_PII" -eq 1 ]; then
    err "PII check FAILED. Fix the issues above before mirroring."
    exit 1
fi
ok "PII scan clean"

info "Pushing to GitHub..."
git remote add github "$GITHUB_REMOTE"
git push github "$BRANCH" --force

ok "Mirrored to $GITHUB_REMOTE"

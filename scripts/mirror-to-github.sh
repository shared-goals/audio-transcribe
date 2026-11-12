#!/bin/zsh
# Mirror audio-transcribe to GitHub, excluding private files.
#
# Excluded:
#   docs/plans/     — internal design docs with personal paths/references
#   AGENTS.md       — development-agent instructions with personal paths
#   .claude/        — local development-agent project config
#
# The script creates a temporary clean clone, removes excluded paths,
# retargets only release tags attached to the tip, and force-pushes the public tree.
#
# Usage: zsh scripts/mirror-to-github.sh
set -e

GITHUB_REMOTE="https://github.com/shared-goals/audio-transcribe.git"
BRANCH="main"

# Paths to exclude from the public mirror
EXCLUDE_PATHS=(
    "docs/plans"
    "AGENTS.md"
    ".claude"
    ".gitea"
    "scripts/mirror-to-github.sh"
    # Benchmark CLI and its private dataset tooling stay on the internal remote.
    "audio_transcribe/benchmark"
    "tests/benchmark"
    "docs/superpowers"
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
WORK_DIR=$(mktemp -d)
CLONE="$WORK_DIR/repo"
trap "rm -rf $WORK_DIR" EXIT

info "Creating clean clone..."
git clone --single-branch --branch "$BRANCH" "$REPO_ROOT" "$CLONE" --quiet

# Configure git in clone
git -C "$CLONE" config user.name "mirror-bot"
git -C "$CLONE" config user.email "mirror-bot@noreply"

info "Removing excluded paths..."
TIP_TAGS=("${(@f)$(git -C "$CLONE" tag --points-at HEAD)}")
for p in "${EXCLUDE_PATHS[@]}"; do
    if [ -e "$CLONE/$p" ]; then
        rm -rf "$CLONE/$p"
        ok "Removed $p"
    fi
done

# Amend the tip commit to exclude private files (no extra mirror commit)
git -C "$CLONE" add -A
git -C "$CLONE" commit --quiet --amend --no-edit
for tag in "${TIP_TAGS[@]}"; do
    [[ -n "$tag" ]] && git -C "$CLONE" tag --force "$tag" HEAD
done

info "Stripping private benchmark docs from README..."
python3 - "$CLONE/README.md" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
out, skipping = [], False
for line in lines:
    if line.startswith("### Private ASR and diarization benchmarks"):
        skipping = True
        continue
    if skipping and line.startswith("### "):
        skipping = False
    if not skipping:
        out.append(line)
path.write_text("".join(out), encoding="utf-8")
PY
git -C "$CLONE" add -A

info "Scanning for PII leaks..."
FOUND_PII=0
for pattern in "${PII_PATTERNS[@]}"; do
    MATCHES=$(grep -rn --include='*.py' --include='*.sh' --include='*.md' --include='*.toml' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.txt' -E "$pattern" "$CLONE" 2>/dev/null | grep -v '\.git/' || true)
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

info "Verifying that the public package imports and runs..."
VERIFY_DIR=$(mktemp -d)
uv venv "$VERIFY_DIR/venv" --python 3.12 >/dev/null
uv pip install --python "$VERIFY_DIR/venv/bin/python" "$CLONE" --quiet
"$VERIFY_DIR/venv/bin/audio-transcribe" --version
rm -rf "$VERIFY_DIR"

info "Pushing to GitHub..."
git -C "$CLONE" remote add github "$GITHUB_REMOTE"
git -C "$CLONE" push github "$BRANCH" --force
for tag in "${TIP_TAGS[@]}"; do
    [[ -n "$tag" ]] && git -C "$CLONE" push github "refs/tags/$tag" --force
done

ok "Mirrored to $GITHUB_REMOTE"

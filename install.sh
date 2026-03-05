#!/bin/zsh
# Installer for audio-transcribe
# Usage: curl -fsSL https://raw.githubusercontent.com/shared-goals/audio-transcribe/main/install.sh | zsh
#
# When piped via curl, re-exec from a temp file to avoid stdin interleaving.
if [[ ! -f "$0" || "$0" == "zsh" ]]; then
    _tmp=$(mktemp)
    cat > "$_tmp"
    exec zsh "$_tmp"
fi
set -e

REPO_URL="https://github.com/shared-goals/audio-transcribe.git"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$1" }
ok()    { printf '\033[1;32m✓\033[0m %s\n' "$1" }
warn()  { printf '\033[1;33m!\033[0m %s\n' "$1" }
error() { printf '\033[1;31m✗\033[0m %s\n' "$1" >&2 }

# --- 1. Homebrew ---
info "Checking Homebrew..."
if command -v brew &>/dev/null; then
    ok "Homebrew found"
else
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Source brew env for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    ok "Homebrew installed"
fi

# --- 2. ffmpeg ---
info "Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found"
else
    info "Installing ffmpeg..."
    brew install ffmpeg
    ok "ffmpeg installed"
fi

# --- 3. uv ---
info "Checking uv..."
if command -v uv &>/dev/null; then
    ok "uv found"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || true
    ok "uv installed"
fi

# --- 4. Install audio-transcribe ---
info "Installing audio-transcribe..."
uv tool install --python 3.12 "git+${REPO_URL}"
ok "audio-transcribe installed"

# --- 5. PATH setup ---
info "Checking PATH..."
if [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
    ok "\$HOME/.local/bin already in PATH"
else
    info "Adding ~/.local/bin to PATH in ~/.zshrc..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
    export PATH="$HOME/.local/bin:$PATH"
    ok "PATH updated"
fi

# --- 6. HF_TOKEN wizard ---
info "Checking HF_TOKEN..."
if [[ -n "$HF_TOKEN" ]] || grep -q 'HF_TOKEN' "$HOME/.zshrc" 2>/dev/null; then
    ok "HF_TOKEN already configured"
else
    echo ""
    warn "HuggingFace token is required for speaker diarization."
    echo "  You need to:"
    echo "  1. Create a token at https://huggingface.co/settings/tokens"
    echo "  2. Accept licenses for the pyannote models"
    echo ""
    info "Opening browser tabs for token creation and license acceptance..."
    open "https://huggingface.co/settings/tokens" 2>/dev/null || true
    sleep 1
    open "https://huggingface.co/pyannote/speaker-diarization-3.1" 2>/dev/null || true
    open "https://huggingface.co/pyannote/segmentation-3.0" 2>/dev/null || true
    open "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM" 2>/dev/null || true
    echo ""

    while true; do
        printf "Paste your HuggingFace token (or 'skip' to set up later): "
        read -r hf_token

        if [[ "$hf_token" == "skip" ]]; then
            warn "Skipping HF_TOKEN setup. Diarization will not work until configured."
            warn "Add 'export HF_TOKEN=\"hf_...\"' to ~/.zshrc when ready."
            break
        fi

        # Validate token
        status=$(curl -sS -o /dev/null -w '%{http_code}' \
            -H "Authorization: Bearer $hf_token" \
            "https://huggingface.co/api/whoami")

        if [[ "$status" == "200" ]]; then
            echo "export HF_TOKEN=\"$hf_token\"" >> "$HOME/.zshrc"
            export HF_TOKEN="$hf_token"
            ok "HF_TOKEN saved to ~/.zshrc"
            break
        else
            error "Token validation failed (HTTP $status). Please try again."
        fi
    done
fi

# --- 7. Verify ---
echo ""
info "Verifying installation..."
if audio-transcribe --help &>/dev/null; then
    ok "audio-transcribe is ready!"
    echo ""
    echo "  Usage:  audio-transcribe process recording.m4a -o meetings/"
    echo "  Help:   audio-transcribe --help"
    echo ""
    echo "  Restart your terminal or run: source ~/.zshrc"
    echo ""
else
    error "Installation verification failed. Restart your terminal and try: audio-transcribe --help"
    exit 1
fi

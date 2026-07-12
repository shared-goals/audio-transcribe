#!/bin/zsh
set -eu

: "${AUDIO_TRANSCRIBE_WATCH:?set AUDIO_TRANSCRIBE_WATCH}"
: "${AUDIO_TRANSCRIBE_OUTPUT:?set AUDIO_TRANSCRIBE_OUTPUT}"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
exec audio-transcribe worker run \
  --watch "$AUDIO_TRANSCRIBE_WATCH" \
  --output "$AUDIO_TRANSCRIBE_OUTPUT" \
  --profile "${AUDIO_TRANSCRIBE_PROFILE:-default}"

#!/bin/zsh
set -eo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    print -u2 "This helper is only for macOS."
    exit 1
fi

brew install ffmpeg@7 pkgconf
ffmpeg7_prefix="$(brew --prefix ffmpeg@7)"

env \
    UV_NO_BINARY_PACKAGE=av \
    PKG_CONFIG_PATH="$ffmpeg7_prefix/lib/pkgconfig" \
    CPPFLAGS="-I$ffmpeg7_prefix/include" \
    LDFLAGS="-L$ffmpeg7_prefix/lib" \
    uv sync --all-extras --all-groups --reinstall-package av

uv run python -m audio_transcribe.macos_ffmpeg "$ffmpeg7_prefix"

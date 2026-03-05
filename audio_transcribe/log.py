"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys


def configure(verbose: bool = False) -> None:
    """Configure logging for the audio_transcribe package."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger("audio_transcribe")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)

"""Corrections system — load, apply, and learn text corrections."""

from __future__ import annotations

import difflib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_corrections(path: str, language: str = "ru") -> dict[str, Any]:
    """Load corrections from YAML. Supports flat (legacy) and language-scoped format."""
    p = Path(path)
    if not p.exists():
        return {"substitutions": {}, "patterns": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"substitutions": {}, "patterns": []}
    # Legacy flat format
    if "substitutions" in data or "patterns" in data:
        return {
            "substitutions": data.get("substitutions") or {},
            "patterns": data.get("patterns") or [],
        }
    # Language-scoped format
    lang_data = data.get(language, {})
    if not isinstance(lang_data, dict):
        return {"substitutions": {}, "patterns": []}
    return {
        "substitutions": lang_data.get("substitutions") or {},
        "patterns": lang_data.get("patterns") or [],
    }


def apply_corrections(
    segments: list[dict[str, Any]], corrections: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """Apply substitutions and regex patterns to segment text and words.

    Returns (modified_segments, replacement_count).
    """
    segments = deepcopy(segments)
    count = 0

    substitutions: dict[str, str] = corrections.get("substitutions", {})
    patterns: list[dict[str, str]] = corrections.get("patterns", [])

    for seg in segments:
        text = seg.get("text", "")

        # Apply substitutions (case-insensitive)
        for wrong, correct in substitutions.items():
            new_text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
            if new_text != text:
                count += 1
                text = new_text
        seg["text"] = text

        # Apply word-level substitutions
        if "words" in seg:
            for w in seg["words"]:
                word_text = w.get("word", "")
                for wrong, correct in substitutions.items():
                    new_word = re.sub(re.escape(wrong), correct, word_text, flags=re.IGNORECASE)
                    if new_word != word_text:
                        word_text = new_word
                w["word"] = word_text

        # Apply regex patterns
        for pat in patterns:
            match_re = pat.get("match", "")
            replace_str = pat.get("replace", "")
            new_text = re.sub(match_re, replace_str, seg["text"])
            if new_text != seg["text"]:
                count += 1
                seg["text"] = new_text

    return segments, count


def learn_corrections(original: list[str], corrected: list[str]) -> dict[str, str]:
    """Diff original and corrected text lines to discover word-level substitutions.

    Returns dict of {wrong_word: correct_word}.
    """
    learned: dict[str, str] = {}

    for orig_line, corr_line in zip(original, corrected, strict=False):
        orig_words = orig_line.split()
        corr_words = corr_line.split()

        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                if (i2 - i1) == (j2 - j1):
                    for orig_w, corr_w in zip(orig_words[i1:i2], corr_words[j1:j2], strict=True):
                        if orig_w != corr_w:
                            learned[orig_w] = corr_w
                else:
                    orig_phrase = " ".join(orig_words[i1:i2])
                    corr_phrase = " ".join(corr_words[j1:j2])
                    learned[orig_phrase] = corr_phrase

    return learned

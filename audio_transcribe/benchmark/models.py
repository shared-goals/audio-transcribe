"""Immutable data models shared by ASR benchmark modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSpan:
    """One manually reviewed interval in an ASR reference transcript."""

    start_s: float
    end_s: float
    text: str
    excluded: bool


@dataclass(frozen=True)
class ReferenceDocument:
    """An immutable reference transcript bound to its source file checksum."""

    source_sha256: str
    spans: tuple[ReferenceSpan, ...]

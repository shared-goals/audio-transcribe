"""Typed configuration profiles for the CLI and worker."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any


class Backend(StrEnum):
    """Supported transcription engines."""

    MLX_VAD = "mlx-vad"
    MLX = "mlx"
    WHISPERX = "whisperx"


@dataclass(frozen=True)
class Profile:
    """User-overridable pipeline defaults."""

    language: str = "ru"
    model: str = "large-v3"
    backend: str = Backend.MLX_VAD
    min_speakers: int = 2
    max_speakers: int = 6
    align_model: str | None = None
    no_align: bool = False
    no_diarize: bool = True
    template: str | None = None

    def validated(self) -> Profile:
        """Return this profile after validating user-controlled values."""
        try:
            Backend(self.backend)
        except ValueError as exc:
            choices = ", ".join(Backend)
            raise ValueError(f"unsupported backend {self.backend!r}; choose one of: {choices}") from exc
        if not self.language.strip():
            raise ValueError("language must not be empty")
        if self.min_speakers < 1:
            raise ValueError("min_speakers must be at least 1")
        if self.max_speakers < self.min_speakers:
            raise ValueError("max_speakers must be greater than or equal to min_speakers")
        return self


def default_config_path() -> Path:
    """Return the XDG-compatible configuration path."""
    return Path.home() / ".config" / "audio-transcribe" / "config.toml"


def load_profile(name: str = "default", path: Path | None = None) -> Profile:
    """Load and validate a named TOML profile, inheriting built-in defaults."""
    config_path = path or default_config_path()
    if not config_path.exists():
        if name != "default":
            raise ValueError(f"profile {name!r} not found: {config_path}")
        return Profile().validated()
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    profiles = document.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("config [profiles] must be a table")
    raw = profiles.get(name, {})
    if not isinstance(raw, dict) or (name != "default" and name not in profiles):
        raise ValueError(f"profile {name!r} not found: {config_path}")
    allowed = {field.name for field in fields(Profile)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown profile option(s): {', '.join(unknown)}")
    values: dict[str, Any] = asdict(Profile())
    values.update(raw)
    return Profile(**values).validated()

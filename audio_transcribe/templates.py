"""Small, dependency-free meeting-note template renderer."""

from __future__ import annotations

from pathlib import Path


def render_template(template_path: Path, meeting_note: str, title: str, transcript: str) -> str:
    """Render documented placeholders and reject a template that drops the note."""
    template = template_path.read_text(encoding="utf-8")
    if "{{meeting_note}}" not in template and "{{transcript}}" not in template:
        raise ValueError("template must include {{meeting_note}} or {{transcript}}")
    return (
        template.replace("{{meeting_note}}", meeting_note)
        .replace("{{title}}", title)
        .replace("{{transcript}}", transcript)
    )

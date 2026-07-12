import pytest

from audio_transcribe.templates import render_template


def test_render_template(tmp_path):
    path = tmp_path / "template.md"
    path.write_text("# {{title}}\n{{meeting_note}}\n{{transcript}}")
    assert render_template(path, "NOTE", "Meeting", "WORDS") == "# Meeting\nNOTE\nWORDS"


def test_template_requires_content_placeholder(tmp_path):
    path = tmp_path / "template.md"
    path.write_text("# {{title}}")
    with pytest.raises(ValueError, match="must include"):
        render_template(path, "NOTE", "Meeting", "WORDS")

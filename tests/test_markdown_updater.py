"""Tests for meeting markdown updater."""

import textwrap

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.markdown.updater import (
    apply_speaker_mapping,
    extract_wiki_links,
    replace_section,
    set_frontmatter,
)


def test_replace_existing_section():
    md = textwrap.dedent("""\
        ---
        title: Test
        ---

        ## Summary

        - Old point

        ## Transcript

        [00:00] Hello
    """)
    doc = parse_meeting(md)
    result = replace_section(doc, "Summary", "- New point\n- Another point")
    assert "- New point" in result.sections["Summary"]
    assert "- Old point" not in result.sections["Summary"]
    assert "Hello" in result.sections["Transcript"]


def test_insert_new_section_before_transcript():
    md = textwrap.dedent("""\
        ---
        title: Test
        ---

        ## Transcript

        [00:00] Hello
    """)
    doc = parse_meeting(md)
    result = replace_section(doc, "Summary", "- A point", before="Transcript")
    assert result._section_order.index("Summary") < result._section_order.index("Transcript")


def test_set_frontmatter_field():
    md = textwrap.dedent("""\
        ---
        title: Test
        reanalyze: true
        ---

        ## Transcript

        [00:00] Hello
    """)
    doc = parse_meeting(md)
    result = set_frontmatter(doc, "reanalyze", False)
    assert result.frontmatter["reanalyze"] is False


def test_apply_speaker_mapping():
    md = textwrap.dedent("""\
        ---
        title: Test
        speakers:
          SPEAKER_00: "[[Andrey]]"
          SPEAKER_01: "[[Maria]]"
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00
        - **Speaker B**: SPEAKER_01

        ## Transcript

        [00:00] Speaker A: Hello
        [00:05] Speaker B: Hi there
    """)
    doc = parse_meeting(md)
    mapping = {"Speaker A": "[[Andrey]]", "Speaker B": "[[Maria]]"}
    result = apply_speaker_mapping(doc, mapping)
    assert "[[Andrey]]" in result.sections["Transcript"]
    assert "[[Maria]]" in result.sections["Transcript"]
    assert "[[Andrey]]" in result.sections["Speakers"]
    assert "Speaker A" not in result.sections["Transcript"]


def test_extract_wiki_links():
    mapping = {"SPEAKER_00": "[[Andrey]]", "SPEAKER_01": "Speaker B"}
    links = extract_wiki_links(mapping)
    assert links == {"SPEAKER_00": "Andrey"}

"""Tests for meeting markdown parser."""

import textwrap

from audio_transcribe.markdown.parser import MeetingDoc, parse_meeting


def test_parse_frontmatter():
    md = textwrap.dedent("""\
        ---
        title: Test meeting
        date: 2026-02-28
        reanalyze: true
        ---

        ## Transcript

        [00:00] Hello world
    """)
    doc = parse_meeting(md)
    assert doc.frontmatter["title"] == "Test meeting"
    assert doc.frontmatter["reanalyze"] is True


def test_parse_sections():
    md = textwrap.dedent("""\
        ---
        title: Test
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Summary

        - Point one

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    doc = parse_meeting(md)
    assert "Speakers" in doc.sections
    assert "Summary" in doc.sections
    assert "Transcript" in doc.sections
    assert "Speaker A: Hello" in doc.sections["Transcript"]


def test_parse_speakers_mapping():
    md = textwrap.dedent("""\
        ---
        title: Test
        speakers:
          SPEAKER_00: "[[Andrey]]"
          SPEAKER_01: Speaker B
        ---

        ## Transcript

        [00:00] Hello
    """)
    doc = parse_meeting(md)
    assert doc.frontmatter["speakers"]["SPEAKER_00"] == "[[Andrey]]"
    assert doc.frontmatter["speakers"]["SPEAKER_01"] == "Speaker B"


def test_parse_no_frontmatter():
    md = "## Transcript\n\n[00:00] Hello\n"
    doc = parse_meeting(md)
    assert doc.frontmatter == {}
    assert "Transcript" in doc.sections


def test_semantic_roundtrip():
    md = textwrap.dedent("""\
        ---
        title: Test meeting
        date: 2026-02-28
        reanalyze: true
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello world
    """)
    doc = parse_meeting(md)
    doc2 = parse_meeting(doc.to_markdown())
    assert doc2.frontmatter == doc.frontmatter
    assert doc2.sections == doc.sections
    assert doc2._section_order == doc._section_order


def test_wiki_link_roundtrip_through_yaml():
    """Wiki-links with [[ ]] survive YAML dump/load cycle."""
    md = textwrap.dedent("""\
        ---
        title: Test
        speakers:
          SPEAKER_00: "[[Andrey]]"
          SPEAKER_01: "[[Maria]]"
        ---

        ## Transcript

        [00:00] Hello
    """)
    doc = parse_meeting(md)
    doc2 = parse_meeting(doc.to_markdown())
    assert doc2.frontmatter["speakers"]["SPEAKER_00"] == "[[Andrey]]"
    assert doc2.frontmatter["speakers"]["SPEAKER_01"] == "[[Maria]]"


def test_section_order_enforced():
    """Sections are output in canonical SECTION_ORDER regardless of insertion order."""
    md = textwrap.dedent("""\
        ---
        title: Test
        ---

        ## Transcript

        [00:00] Hello

        ## Speakers

        - **Speaker A**: SPEAKER_00
    """)
    doc = parse_meeting(md)
    output = doc.to_markdown()
    speakers_pos = output.index("## Speakers")
    transcript_pos = output.index("## Transcript")
    assert speakers_pos < transcript_pos


def test_parse_speaker_legend():
    md = textwrap.dedent("""\
        ---
        title: Test
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00
        - **[[Andrey]]**: SPEAKER_01

        ## Transcript

        [00:00] Hello
    """)
    doc = parse_meeting(md)
    from audio_transcribe.markdown.parser import parse_speaker_legend
    legend = parse_speaker_legend(doc)
    assert legend == {"SPEAKER_00": "Speaker A", "SPEAKER_01": "[[Andrey]]"}

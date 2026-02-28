"""Update meeting markdown sections and frontmatter in-place."""

from __future__ import annotations

import re
from copy import deepcopy

from audio_transcribe.markdown.parser import MeetingDoc


def replace_section(doc: MeetingDoc, name: str, content: str, before: str | None = None) -> MeetingDoc:
    """Replace or insert a section. If 'before' is given, insert before that section."""
    result = deepcopy(doc)
    result.sections[name] = content

    if name not in result._section_order:
        if before and before in result._section_order:
            idx = result._section_order.index(before)
            result._section_order.insert(idx, name)
        else:
            result._section_order.append(name)

    return result


def set_frontmatter(doc: MeetingDoc, key: str, value: object) -> MeetingDoc:
    """Set a frontmatter field."""
    result = deepcopy(doc)
    result.frontmatter[key] = value
    return result


def apply_speaker_mapping(doc: MeetingDoc, mapping: dict[str, str]) -> MeetingDoc:
    """Replace speaker labels in targeted positions only (not arbitrary text).

    In Transcript: matches '] Speaker A:' at line start.
    In Speakers legend: matches '**Speaker A**'.
    """
    result = deepcopy(doc)
    for section_name in result._section_order:
        content = result.sections[section_name]
        for old_name, new_name in mapping.items():
            if section_name == "Transcript":
                # Match speaker label at start of transcript line: '] Speaker A:'
                content = re.sub(
                    r"(\] )" + re.escape(old_name) + r":",
                    r"\1" + new_name + ":",
                    content,
                )
            elif section_name == "Speakers":
                # Match bold label in legend: '**Speaker A**'
                content = content.replace(f"**{old_name}**", f"**{new_name}**")
        result.sections[section_name] = content
    return result


_WIKI_LINK_RE = re.compile(r"\[\[(.+?)]]")


def extract_wiki_links(speakers: dict[str, str]) -> dict[str, str]:
    """Extract speaker IDs that have [[wiki-link]] values.

    Returns {speaker_id: person_name} for entries like {"SPEAKER_00": "[[Andrey]]"}.
    """
    result: dict[str, str] = {}
    for speaker_id, label in speakers.items():
        match = _WIKI_LINK_RE.search(label)
        if match:
            result[speaker_id] = match.group(1)
    return result

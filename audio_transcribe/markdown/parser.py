"""Parse meeting markdown into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

# Canonical section order. to_markdown() sorts by this. Unknown sections appended at end.
SECTION_ORDER: list[str] = [
    "Speakers",
    "Summary",
    "Key Points",
    "Decisions",
    "Action Items",
    "Transcript",
]

_WIKI_LINK_RE = re.compile(r"\[\[.+?]]")


class _QuotedStr(str):
    """String subclass that forces yaml.dump to use double quotes."""


def _quoted_representer(dumper: yaml.Dumper, data: _QuotedStr) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(_QuotedStr, _quoted_representer)


def _force_quote_wiki_links(fm: dict[str, object]) -> dict[str, object]:
    """Ensure values containing [[ are yaml.dump-safe (quoted strings).

    Without this, yaml.dump may output [[Name]] unquoted, which yaml.safe_load
    parses as a nested list instead of a string.
    """
    result = dict(fm)
    speakers = result.get("speakers")
    if isinstance(speakers, dict):
        result["speakers"] = {
            k: _QuotedStr(v) if isinstance(v, str) and "[[" in v else v
            for k, v in speakers.items()
        }
    return result


@dataclass
class MeetingDoc:
    """Parsed meeting markdown document."""

    frontmatter: dict[str, object] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    _section_order: list[str] = field(default_factory=list)
    _raw_frontmatter: str = ""

    def to_markdown(self) -> str:
        """Render back to markdown string."""
        lines: list[str] = []

        if self.frontmatter:
            # Force-quote wiki-link values to survive YAML roundtrip
            safe_fm = _force_quote_wiki_links(self.frontmatter)
            lines.append("---")
            lines.append(yaml.dump(safe_fm, allow_unicode=True, default_flow_style=False, sort_keys=False).rstrip())
            lines.append("---")
            lines.append("")

        # Sort sections by canonical order; unknown sections appended at end
        sorted_sections = sorted(
            self._section_order,
            key=lambda s: (SECTION_ORDER.index(s) if s in SECTION_ORDER else len(SECTION_ORDER), s),
        )

        for name in sorted_sections:
            lines.append(f"## {name}")
            lines.append("")
            content = self.sections.get(name, "")
            if content:
                lines.append(content)
                if not content.endswith("\n"):
                    lines.append("")

        return "\n".join(lines) + "\n" if lines else ""


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_LEGEND_LINE_RE = re.compile(r"- \*\*(.+?)\*\*: (SPEAKER_\d+)")


def parse_meeting(text: str) -> MeetingDoc:
    """Parse meeting markdown into a MeetingDoc."""
    doc = MeetingDoc()
    body = text

    # Parse frontmatter
    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        doc._raw_frontmatter = fm_match.group(1)
        doc.frontmatter = yaml.safe_load(doc._raw_frontmatter) or {}
        body = text[fm_match.end():]

    # Parse sections
    section_matches = list(_SECTION_RE.finditer(body))
    for i, match in enumerate(section_matches):
        name = match.group(1)
        start = match.end() + 1  # skip newline after heading
        if i + 1 < len(section_matches):
            end = section_matches[i + 1].start()
        else:
            end = len(body)

        content = body[start:end].strip("\n")
        doc.sections[name] = content
        doc._section_order.append(name)

    return doc


def parse_speaker_legend(doc: MeetingDoc) -> dict[str, str]:
    """Parse the Speakers section into {SPEAKER_ID: current_label}.

    E.g. "- **Speaker A**: SPEAKER_00" → {"SPEAKER_00": "Speaker A"}
    """
    legend: dict[str, str] = {}
    speakers_section = doc.sections.get("Speakers", "")
    for match in _LEGEND_LINE_RE.finditer(speakers_section):
        label = match.group(1)
        speaker_id = match.group(2)
        legend[speaker_id] = label
    return legend

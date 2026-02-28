# Reactive Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the unified CLI with multi-pass pipeline, voice-based speaker identification, and Claudian re-analysis integration so meetings evolve from raw transcript to fully analyzed notes through progressive enrichment.

**Architecture:** Adds `speakers/` module (voice embeddings via pyannote wespeaker) and `markdown/` module (parse/update meeting notes in-place) to the existing `audio_transcribe/` package. New CLI subcommands: `diarize`, `identify`, `update`, `speakers`. A `reanalyze` frontmatter flag bridges CLI processing and manual Claudian analysis.

**Tech Stack:** Python >=3.12, typer (CLI from unified plan), pyannote.audio (embeddings — already installed), numpy (embedding storage), pyyaml (frontmatter). Existing deps: torch, whisperx.

**Prerequisite:** The unified CLI plan (17 tasks) must be completed first. This plan extends that package.

**Design doc:** `docs/plans/2026-02-28-reactive-pipeline-design.md`

---

## Task 1: Meeting Markdown Parser

Parse meeting markdown files into structured data: frontmatter (YAML), named sections, and raw content. This is the foundation for all in-place update operations.

**Files:**
- Create: `audio_transcribe/markdown/__init__.py`
- Create: `audio_transcribe/markdown/parser.py`
- Test: `tests/test_markdown_parser.py`

**Step 1: Write failing tests**

```python
# tests/test_markdown_parser.py
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_markdown_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_transcribe.markdown'`

**Step 3: Implement the parser**

```python
# audio_transcribe/markdown/__init__.py
"""Meeting markdown parsing and updating."""
```

```python
# audio_transcribe/markdown/parser.py
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_markdown_parser.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add audio_transcribe/markdown/ tests/test_markdown_parser.py
git commit -m "feat: add meeting markdown parser"
```

---

## Task 2: Meeting Markdown Updater

Update specific sections of a meeting markdown in-place: replace sections, update frontmatter fields, and apply speaker mapping changes throughout the document.

**Files:**
- Create: `audio_transcribe/markdown/updater.py`
- Test: `tests/test_markdown_updater.py`

**Step 1: Write failing tests**

```python
# tests/test_markdown_updater.py
"""Tests for meeting markdown updater."""

import textwrap

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.markdown.updater import (
    replace_section,
    set_frontmatter,
    apply_speaker_mapping,
    extract_wiki_links,
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_markdown_updater.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement the updater**

```python
# audio_transcribe/markdown/updater.py
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_markdown_updater.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add audio_transcribe/markdown/updater.py tests/test_markdown_updater.py
git commit -m "feat: add meeting markdown updater for in-place section updates"
```

---

## Task 3: Modify Format Stage for Fast Pass

Update the format stage to produce the reactive pipeline markdown format: `reanalyze: true` flag, `audio_data` path in frontmatter, no speakers when diarization is skipped. This modifies the existing `audio_transcribe/stages/format.py` from the unified CLI plan.

**Files:**
- Modify: `audio_transcribe/stages/format.py`
- Modify: `tests/test_format_stage.py` (or whatever test file exists from unified plan Task 8)

**Step 1: Write failing tests**

Add to the existing format stage test file:

```python
# tests/test_format_stage.py — add these tests

def test_format_fast_pass_no_speakers():
    """Fast pass output has no speaker section when diarization was skipped."""
    data = {
        "audio_file": "meeting.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет"},
            {"start": 2.5, "end": 5.0, "text": "Здравствуйте"},
        ],
    }
    result = format_meeting_note(data, audio_data_path=".audio-data/meeting.json")
    doc = parse_meeting(result)

    assert doc.frontmatter["reanalyze"] is True
    assert doc.frontmatter["audio_data"] == ".audio-data/meeting.json"
    assert "Speakers" not in doc.sections
    assert "Transcript" in doc.sections
    assert "Привет" in doc.sections["Transcript"]


def test_format_with_speakers():
    """When segments have speaker labels, include speaker section."""
    data = {
        "audio_file": "meeting.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 5.0, "text": "Здравствуйте", "speaker": "SPEAKER_01"},
        ],
    }
    result = format_meeting_note(data, audio_data_path=".audio-data/meeting.json")
    doc = parse_meeting(result)

    assert "Speakers" in doc.sections
    assert doc.frontmatter["speakers"]["SPEAKER_00"] == "Speaker A"
    assert doc.frontmatter["speakers"]["SPEAKER_01"] == "Speaker B"
    assert "Speaker A" in doc.sections["Transcript"]


def test_format_frontmatter_has_audio_file():
    data = {
        "audio_file": "recordings/2026-02-28-standup.mp3",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
    }
    result = format_meeting_note(data, audio_data_path=".audio-data/test.json")
    doc = parse_meeting(result)
    assert doc.frontmatter["audio_file"] == "recordings/2026-02-28-standup.mp3"
    assert doc.frontmatter["date"] == "2026-02-28"


def test_format_date_fallback_to_today():
    data = {
        "audio_file": "standup.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
    }
    from datetime import date
    result = format_meeting_note(data, audio_data_path=".audio-data/test.json")
    doc = parse_meeting(result)
    assert doc.frontmatter["date"] == str(date.today())
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_format_stage.py -v -k "fast_pass or with_speakers or date_from_filename"`
Expected: FAIL — `format_meeting_note` not found

**Step 3: Implement format_meeting_note**

Add to `audio_transcribe/stages/format.py`:

```python
def format_meeting_note(data: dict[str, object], audio_data_path: str) -> str:
    """Format WhisperX JSON as a reactive pipeline meeting note.

    Produces markdown with reanalyze frontmatter flag.
    Includes speaker section only if segments have speaker labels.
    """
    segments: list[dict[str, object]] = data.get("segments", [])
    audio_file = str(data.get("audio_file", "unknown"))
    language = str(data.get("language", "unknown"))
    model = str(data.get("model", "unknown"))
    processing_time = float(data.get("processing_time_s", 0.0))

    has_speakers = any(seg.get("speaker") for seg in segments)
    legend = build_speaker_legend(segments) if has_speakers else {}
    duration = compute_duration(segments)

    # Extract date from audio filename, fall back to today
    from datetime import date as date_type
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", audio_file)
    date_str = date_match.group(1) if date_match else str(date_type.today())

    # Build frontmatter
    fm: dict[str, object] = {
        "title": f"{date_str} meeting" if date_str else Path(audio_file).stem,
        "date": date_str,
        "duration": format_time(duration),
        "language": language,
        "model": model,
        "reanalyze": True,
        "audio_file": audio_file,
        "audio_data": audio_data_path,
    }

    if legend:
        fm["speakers"] = {sid: label for sid, label in legend.items()}

    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False).rstrip())
    lines.append("---")
    lines.append("")

    # Speaker legend (only if diarized)
    if legend:
        lines.append("## Speakers")
        lines.append("")
        for speaker_id, label in legend.items():
            lines.append(f"- **{label}**: {speaker_id}")
        lines.append("")

    # Transcript
    lines.append("## Transcript")
    lines.append("")
    for seg in segments:
        lines.append(format_segment(seg, legend if has_speakers else None))
    lines.append("")

    return "\n".join(lines)
```

Note: Import `re`, `yaml`, and `Path` at the top of the file. The existing `format_transcript`, `format_time`, `build_speaker_legend`, `format_segment`, `compute_duration` functions are already in this module from unified plan Task 8.

**Step 4: Run tests**

Run: `uv run pytest tests/test_format_stage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add audio_transcribe/stages/format.py tests/test_format_stage.py
git commit -m "feat: add format_meeting_note for reactive pipeline fast pass"
```

---

## Task 4: Modify Process Command — Store JSON + Skip Diarize

Modify the `audio-transcribe process` command to store raw WhisperX JSON in `.audio-data/` and skip diarization by default (fast pass). Add `--full` flag to opt-in to diarization.

**Files:**
- Modify: `audio_transcribe/cli.py`
- Modify: `audio_transcribe/pipeline.py` (or wherever the orchestrator lives from unified plan Task 12)
- Test: `tests/test_cli_process.py`

**Step 1: Write failing tests**

```python
# tests/test_cli_process.py — add these tests

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner
from audio_transcribe.cli import app


runner = CliRunner()


def test_process_stores_raw_json(tmp_path):
    """process command stores raw WhisperX JSON in .audio-data/ directory."""
    audio_file = tmp_path / "2026-02-28-meeting.wav"
    audio_file.write_bytes(b"fake")
    output_dir = tmp_path / "meetings"
    output_dir.mkdir()

    mock_result = {
        "audio_file": str(audio_file),
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}],
    }

    with patch("audio_transcribe.pipeline.run_pipeline", return_value=mock_result):
        result = runner.invoke(app, ["process", str(audio_file), "-o", str(output_dir)])

    assert result.exit_code == 0

    # Check .audio-data/ JSON was created
    audio_data_dir = output_dir / ".audio-data"
    assert audio_data_dir.exists()
    json_files = list(audio_data_dir.glob("*.json"))
    assert len(json_files) == 1

    stored = json.loads(json_files[0].read_text())
    assert stored["segments"] == mock_result["segments"]


def test_process_skips_diarize_by_default(tmp_path):
    """process command skips diarization in fast pass mode."""
    audio_file = tmp_path / "meeting.wav"
    audio_file.write_bytes(b"fake")

    with patch("audio_transcribe.pipeline.run_pipeline") as mock_pipeline:
        mock_pipeline.return_value = {
            "audio_file": str(audio_file),
            "language": "ru",
            "model": "large-v3",
            "processing_time_s": 5.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}],
        }
        result = runner.invoke(app, ["process", str(audio_file), "-o", str(tmp_path)])

    # Pipeline should have been called with diarize=False
    call_kwargs = mock_pipeline.call_args
    assert call_kwargs is not None
    # The exact kwarg name depends on unified plan's pipeline.run_pipeline signature.
    # Verify no_diarize=True or diarize=False was passed.


def test_process_output_has_reanalyze_true(tmp_path):
    """Output meeting note has reanalyze: true and audio_file in frontmatter."""
    audio_file = tmp_path / "2026-02-28-standup.wav"
    audio_file.write_bytes(b"fake")
    output_dir = tmp_path / "meetings"
    output_dir.mkdir()

    mock_result = {
        "audio_file": str(audio_file),
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
    }

    with patch("audio_transcribe.pipeline.run_pipeline", return_value=mock_result):
        result = runner.invoke(app, ["process", str(audio_file), "-o", str(output_dir)])

    md_files = list(output_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text()
    assert "reanalyze: true" in content
    assert "audio_data:" in content
    assert "audio_file:" in content
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_process.py -v -k "stores_raw_json or skips_diarize or reanalyze_true"`
Expected: FAIL

**Step 3: Modify CLI process command**

In `audio_transcribe/cli.py`, modify the `process` subcommand:

1. Add `--full` flag (default False) — when True, include diarization
2. After pipeline completes, store raw JSON in `<output_dir>/.audio-data/<stem>.json`
3. Use `format_meeting_note()` instead of `format_transcript()` to produce the output
4. Pass `audio_data` relative path to `format_meeting_note()`

Key implementation changes (pseudocode — adapt to actual unified plan structure):

```python
@app.command()
def process(
    audio_file: Path,
    output: Path = typer.Option(None, "-o"),
    full: bool = typer.Option(False, "--full", help="Include diarization (slower)"),
    language: str = typer.Option("ru", "-l"),
    model: str = typer.Option("large-v3", "-m"),
):
    """Fast pass: transcribe + align → meeting note. Use --full to include diarization."""
    result = run_pipeline(
        audio_file=audio_file,
        language=language,
        model=model,
        no_diarize=not full,
        # ... other args from unified plan
    )

    # Determine output paths
    output_dir = output or Path(".")
    stem = audio_file.stem
    audio_data_dir = output_dir / ".audio-data"
    audio_data_dir.mkdir(parents=True, exist_ok=True)

    # Store raw JSON
    json_path = audio_data_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # Format meeting note
    relative_json = f".audio-data/{stem}.json"
    markdown = format_meeting_note(result, audio_data_path=relative_json)

    md_path = output_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_cli_process.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add audio_transcribe/cli.py audio_transcribe/pipeline.py tests/test_cli_process.py
git commit -m "feat: process command stores JSON and skips diarize by default"
```

---

## Task 5: Diarize Subcommand

Add `audio-transcribe diarize <meeting.md>` that reads the stored JSON, runs pyannote diarization, and updates the meeting note in-place with speaker labels. Preserves existing transcript text (only adds speaker label prefixes). Refuses if already diarized unless `--force`.

**Files:**
- Modify: `audio_transcribe/cli.py`
- Create: `audio_transcribe/stages/diarize_update.py`
- Test: `tests/test_diarize_command.py`

**Step 1: Write failing tests**

```python
# tests/test_diarize_command.py
"""Tests for the diarize subcommand logic."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

from audio_transcribe.stages.diarize_update import diarize_and_update


def test_diarize_update_adds_speakers(tmp_path):
    """Diarize updates transcript with speaker labels and adds speaker legend."""
    # Create meeting note
    md_content = textwrap.dedent("""\
        ---
        title: 2026-02-28 meeting
        date: '2026-02-28'
        reanalyze: false
        audio_file: meeting.wav
        audio_data: .audio-data/meeting.json
        ---

        ## Transcript

        [00:00] Привет
        [00:05] Здравствуйте
    """)
    meeting_md = tmp_path / "meetings" / "meeting.md"
    meeting_md.parent.mkdir(parents=True)
    meeting_md.write_text(md_content)

    # Create stored JSON
    stored_json = {
        "audio_file": "meeting.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет"},
            {"start": 5.0, "end": 7.5, "text": "Здравствуйте"},
        ],
    }
    audio_data_dir = tmp_path / "meetings" / ".audio-data"
    audio_data_dir.mkdir(parents=True)
    (audio_data_dir / "meeting.json").write_text(json.dumps(stored_json))

    # Mock diarization to add speaker labels
    diarized_segments = [
        {"start": 0.0, "end": 2.5, "text": "Привет", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 7.5, "text": "Здравствуйте", "speaker": "SPEAKER_01"},
    ]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized_segments):
        diarize_and_update(meeting_md)

    # Verify updated markdown
    result = meeting_md.read_text()
    assert "Speaker A" in result
    assert "Speaker B" in result
    assert "SPEAKER_00" in result
    assert "reanalyze: true" in result
    assert "## Speakers" in result


def test_diarize_update_stores_diarized_json(tmp_path):
    """Diarize updates the stored JSON with speaker labels."""
    md_content = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        audio_data: .audio-data/test.json
        ---

        ## Transcript

        [00:00] Hello
    """)
    meeting_md = tmp_path / "test.md"
    meeting_md.write_text(md_content)

    stored = {
        "audio_file": "test.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 5.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    json_path = data_dir / "test.json"
    json_path.write_text(json.dumps(stored))

    diarized = [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"}]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized):
        diarize_and_update(meeting_md)

    updated_json = json.loads(json_path.read_text())
    assert updated_json["segments"][0]["speaker"] == "SPEAKER_00"


def test_diarize_refuses_if_already_diarized(tmp_path):
    """Diarize refuses if Speakers section exists unless force=True."""
    md_content = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    meeting_md = tmp_path / "meeting.md"
    meeting_md.write_text(md_content)

    import pytest
    with pytest.raises(RuntimeError, match="already diarized"):
        diarize_and_update(meeting_md)


def test_diarize_preserves_user_text_edits(tmp_path):
    """Diarize adds speaker labels but preserves user-edited transcript text."""
    md_content = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        audio_data: .audio-data/test.json
        ---

        ## Transcript

        [00:00] Привет, коллеги!
        [00:05] Добрый день всем
    """)
    meeting_md = tmp_path / "meeting.md"
    meeting_md.write_text(md_content)

    stored = {
        "audio_file": "test.wav",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Привет коллеги"},
            {"start": 5.0, "end": 7.5, "text": "Добрый день"},
        ],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    diarized = [
        {"start": 0.0, "end": 2.5, "text": "Привет коллеги", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 7.5, "text": "Добрый день", "speaker": "SPEAKER_01"},
    ]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized):
        diarize_and_update(meeting_md)

    result = meeting_md.read_text()
    # User's corrected text preserved (note: "коллеги!" with comma and exclamation)
    assert "Привет, коллеги!" in result
    assert "Добрый день всем" in result
    # Speaker labels added
    assert "Speaker A:" in result
    assert "Speaker B:" in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diarize_command.py -v`
Expected: FAIL

**Step 3: Implement diarize_and_update**

```python
# audio_transcribe/stages/diarize_update.py
"""Diarize an existing meeting note and update it in-place."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.markdown.updater import replace_section, set_frontmatter
from audio_transcribe.stages.format import build_speaker_legend, format_time


def run_diarization(audio_file: str, segments: list[dict[str, Any]], min_speakers: int = 1, max_speakers: int = 6) -> list[dict[str, Any]]:
    """Run pyannote diarization on audio. Wraps the existing diarize stage."""
    # Import here to avoid loading torch at module level
    from audio_transcribe.stages.diarize import diarize  # from unified plan Task 7

    return diarize(audio_file, segments, min_speakers=min_speakers, max_speakers=max_speakers)


def _match_timestamp(line: str) -> str | None:
    """Extract MM:SS or HH:MM:SS timestamp from a transcript line."""
    match = re.match(r"\[(\d[\d:]+)]", line)
    return match.group(1) if match else None


def diarize_and_update(
    meeting_path: Path,
    min_speakers: int = 1,
    max_speakers: int = 6,
    force: bool = False,
    audio_file_override: str | None = None,
) -> None:
    """Run diarization and update the meeting note in-place.

    Preserves existing transcript text — only adds speaker label prefixes.
    Refuses if already diarized unless force=True.
    """
    # Parse meeting
    md_text = meeting_path.read_text(encoding="utf-8")
    doc = parse_meeting(md_text)

    # Check if already diarized
    if "Speakers" in doc.sections and not force:
        raise RuntimeError("already diarized — pass force=True to re-diarize")

    # Load stored JSON
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    json_path = meeting_path.parent / audio_data_rel
    stored = json.loads(json_path.read_text(encoding="utf-8"))

    audio_file = audio_file_override or str(doc.frontmatter.get("audio_file", stored.get("audio_file", "")))
    segments = stored.get("segments", [])

    # Run diarization
    diarized_segments = run_diarization(audio_file, segments, min_speakers, max_speakers)

    # Update stored JSON
    stored["segments"] = diarized_segments
    json_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build speaker legend
    legend = build_speaker_legend(diarized_segments)
    speakers_mapping = {sid: label for sid, label in legend.items()}

    # Build speakers section
    speaker_lines = [f"- **{label}**: {sid}" for sid, label in legend.items()]
    speakers_content = "\n".join(speaker_lines)

    # Build timestamp → speaker mapping from diarized segments
    ts_to_speaker: dict[str, str] = {}
    for seg in diarized_segments:
        ts = format_time(float(seg.get("start", 0.0)))
        speaker_id = seg.get("speaker", "")
        if speaker_id and speaker_id in legend:
            ts_to_speaker[ts] = legend[speaker_id]

    # Preserve existing transcript text, only add speaker prefixes
    existing_transcript = doc.sections.get("Transcript", "")
    new_lines: list[str] = []
    for line in existing_transcript.split("\n"):
        ts = _match_timestamp(line)
        if ts and ts in ts_to_speaker:
            speaker = ts_to_speaker[ts]
            # Only add prefix if not already present
            after_bracket = line.split("] ", 1)
            if len(after_bracket) == 2 and not after_bracket[1].startswith(speaker + ":"):
                line = f"[{ts}] {speaker}: {after_bracket[1]}"
        new_lines.append(line)
    transcript_content = "\n".join(new_lines)

    # Update document
    doc = replace_section(doc, "Speakers", speakers_content, before="Transcript")
    doc = replace_section(doc, "Transcript", transcript_content)
    doc = set_frontmatter(doc, "speakers", speakers_mapping)
    doc = set_frontmatter(doc, "reanalyze", True)

    # Write back
    meeting_path.write_text(doc.to_markdown(), encoding="utf-8")
```

**Step 4: Add CLI subcommand**

In `audio_transcribe/cli.py`:

```python
@app.command()
def diarize(
    meeting: Path = typer.Argument(help="Path to meeting markdown file"),
    min_speakers: int = typer.Option(1, "--min-speakers"),
    max_speakers: int = typer.Option(6, "--max-speakers"),
    force: bool = typer.Option(False, "--force", help="Re-diarize even if already diarized"),
    audio_file: str = typer.Option(None, "--audio-file", help="Override audio file path"),
):
    """Add speaker diarization to an existing meeting note."""
    from audio_transcribe.stages.diarize_update import diarize_and_update

    try:
        diarize_and_update(
            meeting,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            force=force,
            audio_file_override=audio_file,
        )
        typer.echo(f"Diarized: {meeting} (reanalyze: true)")
    except RuntimeError as e:
        typer.echo(f"Error: {e}. Use --force to re-diarize.", err=True)
        raise typer.Exit(1)
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_diarize_command.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add audio_transcribe/stages/diarize_update.py audio_transcribe/cli.py tests/test_diarize_command.py
git commit -m "feat: add diarize subcommand for post-hoc speaker labeling"
```

---

## Task 6: Speaker Embedding Database

Create the speaker embedding module that extracts, stores, and matches voice embeddings using pyannote's wespeaker model.

**Files:**
- Create: `audio_transcribe/speakers/__init__.py`
- Create: `audio_transcribe/speakers/embeddings.py`
- Create: `audio_transcribe/speakers/database.py`
- Test: `tests/test_speaker_db.py`

**Step 1: Write failing tests**

```python
# tests/test_speaker_db.py
"""Tests for speaker embedding database."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.speakers.embeddings import cosine_distance


def test_cosine_distance_identical():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(a, b) < 0.01


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_distance(a, b) - 1.0) < 0.01


def test_cosine_distance_opposite():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    assert cosine_distance(a, b) > 1.5


def test_speaker_db_enroll(tmp_path):
    db = SpeakerDB(tmp_path)
    embedding = np.random.randn(256).astype(np.float32)
    db.enroll("Andrey", embedding)

    assert db.has_speaker("Andrey")
    loaded = db.get_embedding("Andrey")
    np.testing.assert_array_almost_equal(loaded, embedding)


def test_speaker_db_enroll_updates_average(tmp_path):
    db = SpeakerDB(tmp_path)
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    db.enroll("Andrey", e1)
    db.enroll("Andrey", e2)

    avg = db.get_embedding("Andrey")
    # Average of [1,0,0] and [0,1,0] = [0.5, 0.5, 0]
    np.testing.assert_array_almost_equal(avg, np.array([0.5, 0.5, 0.0]))


def test_speaker_db_match(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    db.enroll("Maria", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    query = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    matches = db.match(query, threshold=0.5)
    assert len(matches) >= 1
    assert matches[0][0] == "Andrey"


def test_speaker_db_match_no_match(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    matches = db.match(query, threshold=0.5)
    assert len(matches) == 0


def test_speaker_db_list(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))
    db.enroll("Maria", np.random.randn(256).astype(np.float32))

    speakers = db.list_speakers()
    assert len(speakers) == 2
    names = {s["name"] for s in speakers}
    assert names == {"Andrey", "Maria"}


def test_speaker_db_forget(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))
    assert db.has_speaker("Andrey")

    db.forget("Andrey")
    assert not db.has_speaker("Andrey")


def test_speaker_db_persistence(tmp_path):
    db1 = SpeakerDB(tmp_path)
    embedding = np.random.randn(256).astype(np.float32)
    db1.enroll("Andrey", embedding)

    # New instance reads from disk
    db2 = SpeakerDB(tmp_path)
    assert db2.has_speaker("Andrey")
    np.testing.assert_array_almost_equal(db2.get_embedding("Andrey"), embedding)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speaker_db.py -v`
Expected: FAIL

**Step 3: Implement embeddings module**

```python
# audio_transcribe/speakers/__init__.py
"""Speaker identification via voice embeddings."""
```

```python
# audio_transcribe/speakers/embeddings.py
"""Voice embedding extraction and comparison."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine distance between two vectors. 0 = identical, 2 = opposite."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 2.0
    return 1.0 - dot / (norm_a * norm_b)


@functools.lru_cache(maxsize=1)
def _get_model() -> Any:
    """Lazily load and cache the pyannote embedding model."""
    from pyannote.audio import Model

    return Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        token=os.environ.get("HF_TOKEN"),
    )


def extract_embedding(audio_path: str, start: float, end: float) -> NDArray[np.float32]:
    """Extract speaker embedding from an audio segment using pyannote wespeaker model."""
    from pyannote.audio import Inference
    from pyannote.core import Segment

    model = _get_model()
    inference = Inference(model, window="whole")
    segment = Segment(start, end)
    embedding = inference.crop(audio_path, segment)
    return np.array(embedding, dtype=np.float32).flatten()


def extract_speaker_embedding(
    audio_file: str, segments: list[dict[str, Any]], speaker_id: str, min_duration: float = 1.0
) -> NDArray[np.float32]:
    """Extract average embedding for a speaker from their segments.

    Returns zero vector if no segments >= min_duration seconds.
    Logs a warning if no qualifying segments found.
    """
    speaker_segs = [s for s in segments if s.get("speaker") == speaker_id]
    if not speaker_segs:
        logger.warning("Speaker %s has no segments, skipping embedding extraction", speaker_id)
        return np.zeros(256, dtype=np.float32)

    embeddings = []
    for seg in speaker_segs:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end - start >= min_duration:
            emb = extract_embedding(audio_file, start, end)
            embeddings.append(emb)

    if not embeddings:
        logger.warning(
            "Speaker %s has no segments >= %.1fs, skipping voice enrollment",
            speaker_id,
            min_duration,
        )
        return np.zeros(256, dtype=np.float32)

    return np.mean(embeddings, axis=0).astype(np.float32)
```

**Step 4: Implement database module**

```python
# audio_transcribe/speakers/database.py
"""Speaker embedding database — store, match, and manage known speakers."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from audio_transcribe.speakers.embeddings import cosine_distance


class SpeakerDB:
    """File-based speaker embedding database.

    Names are normalized to lowercase for all lookups.
    Display names are preserved in index.json under "display_name".
    """

    def __init__(self, db_dir: Path) -> None:
        self._dir = db_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._index: dict[str, dict[str, object]] = self._load_index()

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower()

    def _load_index(self) -> dict[str, dict[str, object]]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _embedding_path(self, name: str) -> Path:
        safe_name = self._normalize(name).replace(" ", "_")
        return self._dir / f"{safe_name}.npy"

    def has_speaker(self, name: str) -> bool:
        return self._normalize(name) in self._index

    def enroll(self, name: str, embedding: NDArray[np.float32]) -> None:
        """Add or update a speaker's embedding. Averages with existing if present."""
        key = self._normalize(name)
        if key in self._index:
            existing = self.get_embedding(name)
            count = int(self._index[key].get("samples", 1))
            # Running average
            averaged = (existing * count + embedding) / (count + 1)
            np.save(self._embedding_path(name), averaged)
            self._index[key]["samples"] = count + 1
            self._index[key]["last_seen"] = str(date.today())
        else:
            np.save(self._embedding_path(name), embedding)
            self._index[key] = {
                "display_name": name,
                "file": self._embedding_path(name).name,
                "samples": 1,
                "last_seen": str(date.today()),
            }
        self._save_index()

    def get_embedding(self, name: str) -> NDArray[np.float32]:
        """Load a speaker's embedding from disk."""
        path = self._embedding_path(name)
        return np.load(path).astype(np.float32)

    def match(self, query: NDArray[np.float32], threshold: float = 0.5) -> list[tuple[str, float]]:
        """Find speakers matching the query embedding.

        Returns list of (display_name, distance) sorted by distance, filtered by threshold.
        """
        results: list[tuple[str, float]] = []
        for key, meta in self._index.items():
            stored = np.load(self._dir / str(meta["file"])).astype(np.float32)
            dist = cosine_distance(query, stored)
            if dist < threshold:
                display_name = str(meta.get("display_name", key))
                results.append((display_name, dist))
        results.sort(key=lambda x: x[1])
        return results

    def list_speakers(self) -> list[dict[str, object]]:
        """List all known speakers with metadata."""
        return [
            {"name": str(meta.get("display_name", key)), **{k: v for k, v in meta.items() if k != "display_name"}}
            for key, meta in self._index.items()
        ]

    def forget(self, name: str) -> None:
        """Remove a speaker from the database."""
        key = self._normalize(name)
        if key in self._index:
            path = self._embedding_path(name)
            if path.exists():
                path.unlink()
            del self._index[key]
            self._save_index()
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_speaker_db.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add audio_transcribe/speakers/ tests/test_speaker_db.py
git commit -m "feat: add speaker embedding database with enroll/match/forget"
```

---

## Task 7: Identify Subcommand

Add `audio-transcribe identify <meeting.md>` that auto-matches speakers against the voice embedding DB and updates the meeting note.

**Files:**
- Create: `audio_transcribe/stages/identify.py`
- Modify: `audio_transcribe/cli.py`
- Test: `tests/test_identify_command.py`

**Step 1: Write failing tests**

```python
# tests/test_identify_command.py
"""Tests for speaker identification logic."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.identify import identify_speakers, IdentifyResult


def test_identify_matches_known_speaker(tmp_path):
    """Known speaker in DB gets matched and mapped."""
    # Setup speaker DB with known embedding
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)
    known_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db.enroll("Andrey", known_embedding)

    # Create meeting note with unidentified speakers
    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: Speaker A
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    # Mock embedding extraction to return something close to Andrey
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=np.array([0.95, 0.05, 0.0], dtype=np.float32)):
        result = identify_speakers(meeting_path, db)

    assert len(result.matched) >= 1
    assert result.matched["SPEAKER_00"] == "Andrey"


def test_identify_no_match_for_unknown(tmp_path):
    """Unknown speaker gets no match."""
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)
    db.enroll("Andrey", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: Speaker A
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    # Return embedding far from Andrey
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
        result = identify_speakers(meeting_path, db)

    assert len(result.matched) == 0
    assert "SPEAKER_00" in result.unmatched
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_identify_command.py -v`
Expected: FAIL

**Step 3: Implement identify module**

```python
# audio_transcribe/stages/identify.py
"""Auto-identify speakers by matching voice embeddings against known speakers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.markdown.updater import apply_speaker_mapping, set_frontmatter
from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.speakers.embeddings import extract_speaker_embedding


@dataclass
class IdentifyResult:
    """Result of speaker identification."""

    matched: dict[str, str] = field(default_factory=dict)  # speaker_id -> person_name
    unmatched: list[str] = field(default_factory=list)  # speaker_ids with no match


def identify_speakers(
    meeting_path: Path,
    db: SpeakerDB,
    threshold: float = 0.5,
    update_file: bool = True,
    audio_file_override: str | None = None,
) -> IdentifyResult:
    """Identify speakers in a meeting note using the voice embedding DB."""
    md_text = meeting_path.read_text(encoding="utf-8")
    doc = parse_meeting(md_text)

    # Load stored JSON
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    json_path = meeting_path.parent / audio_data_rel
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    audio_file = audio_file_override or str(doc.frontmatter.get("audio_file", stored.get("audio_file", "")))
    segments = stored.get("segments", [])

    # Get current speaker mapping
    speakers = doc.frontmatter.get("speakers", {})
    if not isinstance(speakers, dict):
        speakers = {}

    result = IdentifyResult()

    # For each unidentified speaker (no wiki-link), try to match
    for speaker_id, current_label in speakers.items():
        if "[[" in str(current_label):
            continue  # Already identified

        embedding = extract_speaker_embedding(audio_file, segments, speaker_id)
        matches = db.match(embedding, threshold=threshold)

        if matches:
            person_name = matches[0][0]
            result.matched[speaker_id] = person_name
        else:
            result.unmatched.append(speaker_id)

    # Update the meeting note if matches found
    if update_file and result.matched:
        # Build label mapping: old label -> new wiki-link
        label_mapping: dict[str, str] = {}
        new_speakers = dict(speakers)
        for speaker_id, person_name in result.matched.items():
            old_label = str(speakers[speaker_id])
            new_label = f"[[{person_name}]]"
            label_mapping[old_label] = new_label
            new_speakers[speaker_id] = new_label

        doc = apply_speaker_mapping(doc, label_mapping)
        doc = set_frontmatter(doc, "speakers", new_speakers)
        doc = set_frontmatter(doc, "reanalyze", True)
        meeting_path.write_text(doc.to_markdown(), encoding="utf-8")

    return result
```

**Step 4: Add CLI subcommand**

In `audio_transcribe/cli.py`:

```python
@app.command()
def identify(
    meeting: Path = typer.Argument(help="Path to meeting markdown file"),
    threshold: float = typer.Option(0.5, "--threshold", help="Cosine distance threshold for matching"),
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir", help="Speaker DB directory"),
    audio_file: str = typer.Option(None, "--audio-file", help="Override audio file path"),
):
    """Auto-identify speakers using voice embedding database."""
    from audio_transcribe.speakers.database import SpeakerDB
    from audio_transcribe.stages.identify import identify_speakers

    db = SpeakerDB(db_dir)
    result = identify_speakers(meeting, db, threshold=threshold, audio_file_override=audio_file)

    if result.matched:
        for sid, name in result.matched.items():
            typer.echo(f"  Matched {sid} → [[{name}]]")
        typer.echo(f"Updated: {meeting} (reanalyze: true)")
    else:
        typer.echo("No speakers matched.")

    if result.unmatched:
        typer.echo(f"  Unmatched: {', '.join(result.unmatched)}")
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_identify_command.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add audio_transcribe/stages/identify.py audio_transcribe/cli.py tests/test_identify_command.py
git commit -m "feat: add identify subcommand for voice-based speaker matching"
```

---

## Task 8: Update Subcommand + Voice Enrollment

Add `audio-transcribe update <meeting.md>` that applies speaker mapping from frontmatter to transcript body and enrolls new `[[wiki-link]]` speakers in the voice DB.

**Files:**
- Create: `audio_transcribe/stages/update.py`
- Modify: `audio_transcribe/cli.py`
- Test: `tests/test_update_command.py`

**Step 1: Write failing tests**

```python
# tests/test_update_command.py
"""Tests for update subcommand logic."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.update import update_meeting


def test_update_applies_speaker_mapping(tmp_path):
    """Speaker mapping from frontmatter is applied to transcript body."""
    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: "[[Andrey]]"
          SPEAKER_01: "[[Maria]]"
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00
        - **Speaker B**: SPEAKER_01

        ## Transcript

        [00:00] Speaker A: Hello
        [00:05] Speaker B: Hi there
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 7.5, "text": "Hi there", "speaker": "SPEAKER_01"},
        ],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    db_dir = tmp_path / "speakers"

    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=np.random.randn(256).astype(np.float32)):
        update_meeting(meeting_path, SpeakerDB(db_dir))

    result = meeting_path.read_text()
    assert "[[Andrey]]" in result
    assert "[[Maria]]" in result
    assert "Speaker A" not in result.split("## Transcript")[1]
    assert "reanalyze: true" in result


def test_update_enrolls_new_wiki_link_speakers(tmp_path):
    """New [[wiki-link]] speakers get enrolled in voice DB."""
    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: "[[Andrey]]"
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **Speaker A**: SPEAKER_00

        ## Transcript

        [00:00] Speaker A: Hello
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)

    mock_embedding = np.random.randn(256).astype(np.float32)
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=mock_embedding):
        update_meeting(meeting_path, db)

    assert db.has_speaker("Andrey")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_update_command.py -v`
Expected: FAIL

**Step 3: Implement update module**

```python
# audio_transcribe/stages/update.py
"""Apply speaker mapping and enroll new speakers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting, parse_speaker_legend
from audio_transcribe.markdown.updater import apply_speaker_mapping, extract_wiki_links, set_frontmatter
from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.speakers.embeddings import extract_speaker_embedding


def update_meeting(meeting_path: Path, db: SpeakerDB) -> None:
    """Apply speaker mapping from frontmatter and enroll new speakers."""
    md_text = meeting_path.read_text(encoding="utf-8")
    doc = parse_meeting(md_text)

    speakers = doc.frontmatter.get("speakers", {})
    if not isinstance(speakers, dict):
        return

    # Load stored JSON for enrollment
    audio_data_rel = str(doc.frontmatter.get("audio_data", ""))
    json_path = meeting_path.parent / audio_data_rel
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    audio_file = str(doc.frontmatter.get("audio_file", stored.get("audio_file", "")))
    segments = stored.get("segments", [])

    # Build mapping from current legend labels to frontmatter values
    legend = parse_speaker_legend(doc)  # {SPEAKER_ID: current_label}
    label_mapping: dict[str, str] = {}

    for speaker_id, new_label in speakers.items():
        old_label = legend.get(speaker_id)
        if old_label and old_label != new_label:
            label_mapping[old_label] = str(new_label)

    # Apply mapping
    if label_mapping:
        doc = apply_speaker_mapping(doc, label_mapping)

    doc = set_frontmatter(doc, "reanalyze", True)

    # Enroll new wiki-link speakers in voice DB
    wiki_links = extract_wiki_links(speakers)
    for speaker_id, person_name in wiki_links.items():
        if not db.has_speaker(person_name):
            embedding = extract_speaker_embedding(audio_file, segments, speaker_id)
            if np.any(embedding):
                db.enroll(person_name, embedding)

    meeting_path.write_text(doc.to_markdown(), encoding="utf-8")
```

**Step 4: Add CLI subcommand**

In `audio_transcribe/cli.py`:

```python
@app.command()
def update(
    meeting: Path = typer.Argument(help="Path to meeting markdown file"),
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir"),
):
    """Apply speaker mapping from frontmatter and enroll new voices."""
    from audio_transcribe.speakers.database import SpeakerDB
    from audio_transcribe.stages.update import update_meeting

    db = SpeakerDB(db_dir)
    update_meeting(meeting, db)
    typer.echo(f"Updated: {meeting} (reanalyze: true)")
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_update_command.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add audio_transcribe/stages/update.py audio_transcribe/cli.py tests/test_update_command.py
git commit -m "feat: add update subcommand for speaker mapping and voice enrollment"
```

---

## Task 9: Speakers List and Forget Subcommands

Add `audio-transcribe speakers list` and `audio-transcribe speakers forget <name>` CLI subcommands.

**Files:**
- Modify: `audio_transcribe/cli.py`
- Test: `tests/test_speakers_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_speakers_cli.py
"""Tests for speakers CLI subcommands."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from audio_transcribe.cli import app
from audio_transcribe.speakers.database import SpeakerDB


runner = CliRunner()


def test_speakers_list_empty(tmp_path):
    result = runner.invoke(app, ["speakers", "list", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No speakers" in result.output


def test_speakers_list_shows_enrolled(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))
    db.enroll("Maria", np.random.randn(256).astype(np.float32))

    result = runner.invoke(app, ["speakers", "list", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Andrey" in result.output
    assert "Maria" in result.output


def test_speakers_forget(tmp_path):
    db = SpeakerDB(tmp_path)
    db.enroll("Andrey", np.random.randn(256).astype(np.float32))

    result = runner.invoke(app, ["speakers", "forget", "Andrey", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0

    db2 = SpeakerDB(tmp_path)
    assert not db2.has_speaker("Andrey")


def test_speakers_forget_unknown(tmp_path):
    result = runner.invoke(app, ["speakers", "forget", "Nobody", "--db-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "not found" in result.output.lower()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speakers_cli.py -v`
Expected: FAIL

**Step 3: Implement CLI subcommands**

In `audio_transcribe/cli.py`, add a subcommand group:

```python
speakers_app = typer.Typer(help="Manage known speaker voice embeddings.")
app.add_typer(speakers_app, name="speakers")


@speakers_app.command("list")
def speakers_list(
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir"),
):
    """List all known speakers in the voice embedding database."""
    from audio_transcribe.speakers.database import SpeakerDB

    db = SpeakerDB(db_dir)
    speakers = db.list_speakers()

    if not speakers:
        typer.echo("No speakers enrolled yet.")
        return

    for s in speakers:
        name = s["name"]
        samples = s.get("samples", 0)
        last_seen = s.get("last_seen", "unknown")
        typer.echo(f"  {name} ({samples} samples, last seen {last_seen})")


@speakers_app.command("forget")
def speakers_forget(
    name: str = typer.Argument(help="Speaker name to remove"),
    db_dir: Path = typer.Option(Path.home() / ".audio-transcribe" / "speakers", "--db-dir"),
):
    """Remove a speaker from the voice embedding database."""
    from audio_transcribe.speakers.database import SpeakerDB

    db = SpeakerDB(db_dir)
    if not db.has_speaker(name):
        typer.echo(f"Speaker not found: {name}")
        return

    db.forget(name)
    typer.echo(f"Removed: {name}")
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_speakers_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add audio_transcribe/cli.py tests/test_speakers_cli.py
git commit -m "feat: add speakers list and forget subcommands"
```

---

## Task 10: Update /process-meeting Claude Command

Update the Claudian `/process-meeting` command to handle the `reanalyze` flag, create people card stubs, and do in-place section updates.

**Files:**
- Modify: `/Users/gnezim/_projects/gnezim/knowledge/.claude/commands/process-meeting.md`

**Step 1: No test needed** — this is a Claude command (prompt file), not code.

**Step 2: Update the command**

Replace the content of `process-meeting.md` with the updated version that:

1. Checks `reanalyze` frontmatter flag
2. Reads `speakers` mapping for `[[wiki-link]]` names (uses them as attendees)
3. On first analysis: creates Summary, Key Points, Decisions, Action Items sections before Transcript
4. On re-analysis (`reanalyze: true` + Summary exists): overwrites those sections in-place
5. Sets `reanalyze: false` after analysis
6. Creates stub people cards for `[[Person]]` links not yet in vault
7. Uses `@[[Person]]` format in action items

Updated command content:

```markdown
Process a meeting transcript into a structured meeting note, or re-analyze after edits.

$ARGUMENTS — path to the meeting Markdown file (relative to vault root), e.g. `meetings/2026-02-28-standup.md`

**Read the meeting note** at the path provided in `$ARGUMENTS`.

**Check the frontmatter:**
- If `reanalyze` is `false` or missing, and Summary section already exists, say "Already analyzed. Set `reanalyze: true` in frontmatter to re-analyze." and stop.
- If `reanalyze` is `true`, or no Summary section exists, proceed with analysis.

**Read the transcript** from the `## Transcript` section.

**Read speaker mapping** from frontmatter `speakers` field. Use `[[wiki-link]]` names as real attendees. If no speakers mapping exists, treat as single-speaker.

**Analyze the transcript** and write these sections (in Russian) BEFORE `## Transcript`:

1. **## Summary** — 3-5 bullet points covering main topics
2. **## Key Points** — specific facts, numbers, statuses mentioned
3. **## Decisions** — decisions made, with rationale if available
4. **## Action Items** — as checkboxes: `- [ ] @[[Person]]: description` (use wiki-links from speaker mapping)

**If sections already exist** (re-analysis): overwrite their content. Do NOT duplicate sections.

**Update frontmatter:**
- Set `reanalyze: false`
- Set `title` to a descriptive Russian title based on content
- Set `attendees` list from speaker mapping values

**Create people card stubs** for any `[[Person]]` links in speaker mapping where the person file doesn't exist yet:

For each person name from `[[Name]]` links:
1. Check if `people/Name.md` or `people/*/Name.md` exists
2. If not, create a stub at `people/Name.md`:

```yaml
---
type: person
created: YYYY-MM-DD
---

# Name

First seen in [[meeting-filename]].
```

**Write the updated file** using Edit tool (preserve sections you didn't change).

**Auto-commit:**
```bash
./scripts/auto-commit.sh <meeting-note-path>
```

**Confirm** to the user: show the meeting note path and what was extracted/updated.
```

**Step 3: Commit**

```bash
cd /Users/gnezim/_projects/gnezim/knowledge
git add .claude/commands/process-meeting.md
git commit -m "feat: update process-meeting for reactive pipeline with reanalyze flag"
```

---

## Task 11: Diarize Auto-Enrollment Hook

Modify `diarize_and_update` to automatically enroll speakers when the meeting note already has `[[wiki-link]]` mappings in frontmatter (from a previous `identify` or manual mapping).

**Files:**
- Modify: `audio_transcribe/stages/diarize_update.py`
- Test: `tests/test_diarize_enrollment.py`

**Step 1: Write failing test**

```python
# tests/test_diarize_enrollment.py
"""Tests for automatic voice enrollment during diarization."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.diarize_update import diarize_and_update


def test_diarize_enrolls_wiki_link_speakers(tmp_path):
    """If frontmatter has [[wiki-link]] speakers, enroll them after diarization."""
    md = textwrap.dedent("""\
        ---
        title: Test
        audio_file: test.wav
        speakers:
          SPEAKER_00: "[[Andrey]]"
        audio_data: .audio-data/test.json
        ---

        ## Speakers

        - **[[Andrey]]**: SPEAKER_00

        ## Transcript

        [00:00] [[Andrey]]: Hello
    """)
    meeting_path = tmp_path / "meeting.md"
    meeting_path.write_text(md)

    stored = {
        "audio_file": "test.wav",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}],
    }
    data_dir = tmp_path / ".audio-data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(json.dumps(stored))

    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)

    diarized = [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"}]
    mock_embedding = np.random.randn(256).astype(np.float32)

    with (
        patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized),
        patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=mock_embedding),
    ):
        diarize_and_update(meeting_path, db=db)

    assert db.has_speaker("Andrey")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diarize_enrollment.py -v`
Expected: FAIL

**Step 3: Modify diarize_and_update**

Add optional `db` parameter and enrollment logic to `audio_transcribe/stages/diarize_update.py`. Import `extract_speaker_embedding` from the shared `speakers.embeddings` module (no duplication):

```python
from audio_transcribe.speakers.embeddings import extract_speaker_embedding


def diarize_and_update(
    meeting_path: Path,
    min_speakers: int = 2,
    max_speakers: int = 6,
    db: SpeakerDB | None = None,
) -> None:
    """Run diarization and update the meeting note in-place.

    If db is provided and frontmatter has [[wiki-link]] speakers,
    enroll their voice embeddings.
    """
    # ... existing diarization code ...

    # After updating the document, check for wiki-link speakers to enroll
    if db is not None:
        from audio_transcribe.markdown.updater import extract_wiki_links

        updated_doc = parse_meeting(meeting_path.read_text(encoding="utf-8"))
        speakers_fm = updated_doc.frontmatter.get("speakers", {})
        if isinstance(speakers_fm, dict):
            wiki_links = extract_wiki_links(speakers_fm)
            for speaker_id, person_name in wiki_links.items():
                if not db.has_speaker(person_name):
                    embedding = extract_speaker_embedding(audio_file, diarized_segments, speaker_id)
                    if np.any(embedding):
                        db.enroll(person_name, embedding)
```

Also add `import numpy as np` to imports. Update the `diarize` CLI command to pass `db`. The `extract_speaker_embedding` function is already imported from `speakers.embeddings` — no local definition needed.

**Step 4: Run tests**

Run: `uv run pytest tests/test_diarize_enrollment.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add audio_transcribe/stages/diarize_update.py tests/test_diarize_enrollment.py
git commit -m "feat: auto-enroll wiki-link speakers during diarization"
```

---

## Task 12: Integration Test — Full Reactive Workflow

End-to-end test simulating the full workflow: process → diarize → identify → update → verify final state.

**Files:**
- Create: `tests/test_reactive_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_reactive_integration.py
"""Integration test for the full reactive pipeline workflow."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.format import format_meeting_note
from audio_transcribe.stages.diarize_update import diarize_and_update
from audio_transcribe.stages.identify import identify_speakers
from audio_transcribe.stages.update import update_meeting


def test_full_reactive_workflow(tmp_path):
    """Simulate the complete reactive pipeline end-to-end."""

    # === Step 1: Fast pass (process command output) ===
    whisperx_result = {
        "audio_file": "2026-02-28-standup.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 15.0,
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "Привет, давайте начнём"},
            {"start": 3.5, "end": 6.0, "text": "Да, у меня есть обновления"},
            {"start": 6.5, "end": 10.0, "text": "Отлично, расскажи подробнее"},
        ],
    }

    # Store JSON
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    data_dir = meetings_dir / ".audio-data"
    data_dir.mkdir()
    json_path = data_dir / "2026-02-28-standup.json"
    json_path.write_text(json.dumps(whisperx_result, ensure_ascii=False, indent=2))

    # Format meeting note (fast pass — no speakers)
    markdown = format_meeting_note(whisperx_result, audio_data_path=".audio-data/2026-02-28-standup.json")
    md_path = meetings_dir / "2026-02-28-standup.md"
    md_path.write_text(markdown)

    # Verify fast pass output
    doc = parse_meeting(md_path.read_text())
    assert doc.frontmatter["reanalyze"] is True
    assert "Speakers" not in doc.sections
    assert "Привет" in doc.sections["Transcript"]

    # === Step 2: Diarize ===
    diarized_segments = [
        {"start": 0.0, "end": 3.0, "text": "Привет, давайте начнём", "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 6.0, "text": "Да, у меня есть обновления", "speaker": "SPEAKER_01"},
        {"start": 6.5, "end": 10.0, "text": "Отлично, расскажи подробнее", "speaker": "SPEAKER_00"},
    ]

    with patch("audio_transcribe.stages.diarize_update.run_diarization", return_value=diarized_segments):
        diarize_and_update(md_path)

    doc = parse_meeting(md_path.read_text())
    assert "Speakers" in doc.sections
    assert doc.frontmatter["reanalyze"] is True
    assert "Speaker A" in doc.sections["Transcript"]
    assert "Speaker B" in doc.sections["Transcript"]

    # === Step 3: User maps speakers in frontmatter ===
    content = md_path.read_text()
    content = content.replace("SPEAKER_00: Speaker A", 'SPEAKER_00: "[[Andrey]]"')
    content = content.replace("SPEAKER_01: Speaker B", 'SPEAKER_01: "[[Maria]]"')
    md_path.write_text(content)

    # === Step 4: Update (apply mapping + enroll) ===
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)

    mock_embedding = np.random.randn(256).astype(np.float32)
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=mock_embedding):
        update_meeting(md_path, db)

    doc = parse_meeting(md_path.read_text())
    assert "[[Andrey]]" in doc.sections["Transcript"]
    assert "[[Maria]]" in doc.sections["Transcript"]
    assert doc.frontmatter["reanalyze"] is True

    # Verify enrollment
    assert db.has_speaker("Andrey")
    assert db.has_speaker("Maria")

    # === Step 5: Verify speakers DB works for future meetings ===
    speakers = db.list_speakers()
    names = {s["name"] for s in speakers}
    assert names == {"Andrey", "Maria"}
```

**Step 2: Run the integration test**

Run: `uv run pytest tests/test_reactive_integration.py -v`
Expected: PASS (all mocks in place, testing the glue logic)

**Step 3: Commit**

```bash
git add tests/test_reactive_integration.py
git commit -m "test: add integration test for full reactive pipeline workflow"
```

---

## Task 13: Add numpy + pyyaml to Dependencies

Ensure `numpy` (for embeddings) and `pyyaml` (for frontmatter parsing) are in project dependencies. numpy is already a transitive dep of torch but should be explicit. pyyaml is needed by the markdown parser.

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Add to dependencies:

```toml
[project]
dependencies = [
    "torch>=2.8.0",
    "torchaudio>=2.8.0",
    "whisperx>=3.8.0",
    "typer>=0.15.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "numpy>=1.26.0",
]
```

Note: cosine distance is implemented with pure numpy (see `embeddings.py`), avoiding a scipy dependency. `pyannote.audio` is already a transitive dep of `whisperx`. `pyyaml` is needed by the markdown parser — also add it to the unified CLI plan dependencies.

**Step 2: Sync**

Run: `uv sync`
Expected: Clean install

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add numpy as explicit dependency for speaker embeddings"
```

---

---

## Task 14: Identify Integration Test

Separate integration test proving that speakers enrolled from meeting 1 are correctly identified in meeting 2 via voice matching.

**Files:**
- Create: `tests/test_identify_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_identify_integration.py
"""Integration test for speaker identification across meetings."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_transcribe.markdown.parser import parse_meeting
from audio_transcribe.speakers.database import SpeakerDB
from audio_transcribe.stages.format import format_meeting_note
from audio_transcribe.stages.identify import identify_speakers
from audio_transcribe.stages.update import update_meeting


def test_identify_matches_enrolled_speaker_from_previous_meeting(tmp_path):
    """Speakers enrolled from meeting 1 are identified in meeting 2."""
    db_dir = tmp_path / "speakers"
    db = SpeakerDB(db_dir)

    # === Meeting 1: enroll Andrey ===
    m1_dir = tmp_path / "meeting1"
    m1_dir.mkdir()
    data_dir1 = m1_dir / ".audio-data"
    data_dir1.mkdir()

    m1_data = {
        "audio_file": "meeting1.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 10.0,
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Привет", "speaker": "SPEAKER_00"},
        ],
    }
    (data_dir1 / "meeting1.json").write_text(json.dumps(m1_data))

    m1_md = format_meeting_note(m1_data, audio_data_path=".audio-data/meeting1.json")
    m1_path = m1_dir / "meeting1.md"
    m1_path.write_text(m1_md)

    # User maps speaker and runs update → enrolls Andrey
    content = m1_path.read_text()
    content = content.replace("SPEAKER_00: Speaker A", 'SPEAKER_00: "[[Andrey]]"')
    m1_path.write_text(content)

    andrey_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=andrey_embedding):
        update_meeting(m1_path, db)

    assert db.has_speaker("Andrey")

    # === Meeting 2: identify Andrey ===
    m2_dir = tmp_path / "meeting2"
    m2_dir.mkdir()
    data_dir2 = m2_dir / ".audio-data"
    data_dir2.mkdir()

    m2_data = {
        "audio_file": "meeting2.wav",
        "language": "ru",
        "model": "large-v3",
        "processing_time_s": 8.0,
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "Добрый день", "speaker": "SPEAKER_00"},
        ],
    }
    (data_dir2 / "meeting2.json").write_text(json.dumps(m2_data))

    m2_md = format_meeting_note(m2_data, audio_data_path=".audio-data/meeting2.json")
    m2_path = m2_dir / "meeting2.md"
    m2_path.write_text(m2_md)

    # Identify should match SPEAKER_00 → Andrey (embedding close to [1,0,0])
    similar_embedding = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    with patch("audio_transcribe.speakers.embeddings.extract_speaker_embedding", return_value=similar_embedding):
        result = identify_speakers(m2_path, db)

    assert result.matched.get("SPEAKER_00") == "Andrey"

    # Verify the meeting file was updated
    doc = parse_meeting(m2_path.read_text())
    assert doc.frontmatter["speakers"]["SPEAKER_00"] == "[[Andrey]]"
```

**Step 2: Run the test**

Run: `uv run pytest tests/test_identify_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_identify_integration.py
git commit -m "test: add integration test for cross-meeting speaker identification"
```

---

## Task Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | Meeting markdown parser (+ SECTION_ORDER, parse_speaker_legend, wiki-link quoting) | Unified plan complete |
| 2 | Meeting markdown updater (targeted regex replacement) | Task 1 |
| 3 | Format stage for fast pass (audio_file frontmatter, date fallback) | Tasks 1, 2 + unified plan Task 8 |
| 4 | Process command — store JSON, skip diarize | Task 3 + unified plan Task 13 |
| 5 | Diarize subcommand (--force, --audio-file, preserve text) | Tasks 1, 2, 3 + unified plan Task 7 |
| 6 | Speaker embedding database (case-normalized, short-segment warnings) | — (independent) |
| 7 | Identify subcommand (--audio-file) | Tasks 1, 2, 6 |
| 8 | Update subcommand + voice enrollment (uses parse_speaker_legend) | Tasks 1, 2, 6 |
| 9 | Speakers list/forget CLI | Task 6 |
| 10 | Update /process-meeting Claude command (people stubs at people/Name.md) | — (independent) |
| 11 | Diarize auto-enrollment hook | Tasks 5, 6 |
| 12 | Integration test — full reactive workflow | Tasks 1-9, 11 |
| 13 | Add numpy + pyyaml dependencies | — (do anytime) |
| 14 | Integration test — cross-meeting speaker identification | Tasks 6, 7, 8 |

import subprocess
import sys
from pathlib import Path

from scripts.changelog_section import section

TEXT = """# Changelog

## [Unreleased]

## [0.9.2] - 2026-11-13

### Changed
- one

## [0.9.1] - 2026-11-13

### Fixed
- two
"""


def test_section_stops_at_the_next_version() -> None:
    body = section(TEXT, "0.9.2")
    assert "### Changed" in body and "two" not in body
    assert section(TEXT, "9.9.9") == ""


def test_cli_uses_the_repository_changelog() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/changelog_section.py", "v0.9.2"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "httpx2" in completed.stdout

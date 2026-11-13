"""Print the CHANGELOG section for a release tag, for GitHub release notes."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def section(text: str, version: str) -> str:
    pattern = re.compile(rf"^## \[{re.escape(version)}\].*?$(.*?)(?=^## \[|\Z)", re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: changelog_section.py vX.Y.Z", file=sys.stderr)
        return 2
    version = argv[1].removeprefix("v")
    body = section(Path("CHANGELOG.md").read_text(encoding="utf-8"), version)
    print(body or f"Release {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

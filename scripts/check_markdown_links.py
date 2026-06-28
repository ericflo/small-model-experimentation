#!/usr/bin/env python3
"""Check local markdown links in repository navigation surfaces."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
DEFAULT_TARGETS = [
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "docs",
    "knowledge",
    "research_programs",
    "templates",
]


def iter_markdown(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = ROOT / raw_path
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".md":
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob("*.md")))
    return sorted(set(found))


def is_external(target: str) -> bool:
    return "://" in target or target.startswith("mailto:")


def normalized_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target.split("#", 1)[0]


def check(paths: list[str]) -> list[str]:
    errors: list[str] = []
    for md in iter_markdown(paths):
        text = md.read_text(encoding="utf-8", errors="replace")
        for match in LINK_RE.finditer(text):
            raw_target = match.group(1)
            target = normalized_target(raw_target)
            if not target or target.startswith("#") or is_external(target):
                continue
            candidate = (md.parent / target).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                errors.append(f"{md.relative_to(ROOT)} links outside repo: {raw_target}")
                continue
            if not candidate.exists():
                errors.append(f"{md.relative_to(ROOT)} missing link target: {raw_target}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=DEFAULT_TARGETS, help="files or directories to check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = check(args.paths)
    if errors:
        print("markdown link check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("markdown link check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

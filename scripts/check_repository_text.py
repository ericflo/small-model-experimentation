#!/usr/bin/env python3
"""Check repository guidance for stale framing and temporary scaffold residue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = [
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "docs",
    "knowledge",
    "research_programs",
    "templates",
    "scripts",
    "experiments",
]
EXCLUDED_PARTS = {".git", ".ipynb_checkpoints", "__pycache__"}
EXCLUDED_FILES = {"scripts/check_repository_text.py"}
STALE_PHRASES = [
    "two intensive tracks",
    "two-week tracks",
    "two original lines",
    "Cross-Track",
    "cross-track findings",
    "Track Y imports",
    "Track Z imports",
    "temp_scaffold",
    "temp_make",
]
TEXT_EXTENSIONS = {".md", ".py", ".yaml", ".yml", ".json", ".csv", ".txt", ".toml", ".gitignore", ".gitattributes"}
EXPERIMENT_TEXT_EXTENSIONS = {".md", ".py", ".yaml", ".yml", ".txt", ".toml"}


def should_scan(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in EXCLUDED_FILES:
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if "experiments" in path.parts:
        return path.suffix in EXPERIMENT_TEXT_EXTENSIONS
    if path.name in {".gitignore", ".gitattributes"}:
        return True
    return path.suffix in TEXT_EXTENSIONS


def iter_files(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = ROOT / raw_path
        if not path.exists():
            continue
        if path.is_file() and should_scan(path):
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and should_scan(candidate)))
    return sorted(set(found))


def check(paths: list[str]) -> list[str]:
    errors: list[str] = []
    for path in iter_files(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for phrase in STALE_PHRASES:
                if phrase in line:
                    errors.append(f"{path.relative_to(ROOT)}:{line_number}: stale text: {phrase}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=DEFAULT_TARGETS, help="files or directories to check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = check(args.paths)
    if errors:
        print("repository text check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("repository text check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

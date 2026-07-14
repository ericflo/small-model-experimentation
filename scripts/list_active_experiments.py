#!/usr/bin/env python3
"""Print the canonical experiment lifecycle roster for release-time review."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
STATUS_RE = re.compile(
    r"(?im)^[ \t]*\*\*status:\*\*[ \t]*in-progress\b([^\n]*)$"
)
SINCE_RE = re.compile(r"(?i)\bsince[ \t]+(\d{4}-\d{2}-\d{2})\b")


def active_experiments() -> list[tuple[str, str, str]]:
    active: list[tuple[str, str, str]] = []
    for experiment in sorted(EXPERIMENTS.iterdir()):
        if not experiment.is_dir():
            continue
        readme = experiment / "README.md"
        if not readme.exists():
            continue
        match = STATUS_RE.search(readme.read_text(encoding="utf-8", errors="replace"))
        if not match:
            continue
        tail = match.group(1)
        since_match = SINCE_RE.search(tail)
        since = since_match.group(1) if since_match else "unknown"
        reason = tail[since_match.end() :] if since_match else tail
        reason = reason.strip().lstrip("·—–-:*").strip()
        active.append((experiment.name, since, reason))
    return active


def main() -> int:
    active = active_experiments()
    print(f"in-progress experiments: {len(active)}")
    for experiment_id, since, reason in active:
        print(f"- {experiment_id} (since {since}): {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

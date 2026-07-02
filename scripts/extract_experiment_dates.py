#!/usr/bin/env python3
"""Audit knowledge/experiment_dates.json against in-repo run records.

Git history collapses most experiment dirs onto the 2026-06-28 corpus import
commit, so the committed dates file records each experiment's real run window,
recovered from records inside the dir. This script keeps that file honest:

- verifies every recorded start/end date is supported by at least one evidence
  class found in the experiment dir (literal date string, epoch timestamp,
  date-shaped seed, git commit date, or file mtime, each with one day of
  UTC-vs-local slack);
- lists experiment dirs that have no entry, with a suggested window when the
  dir contains high-precision records (dated log headings, config date keys,
  epoch timestamp fields).

Exits non-zero when a recorded date has no supporting evidence or a catalog
experiment is missing from the dates file. Use --suggest to print JSON entries
for missing dirs, ready to paste into the dates file after review.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DATES_PATH = ROOT / "knowledge" / "experiment_dates.json"
CATALOG_PATH = ROOT / "knowledge" / "experiment_catalog.csv"

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
SEED_RE = re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b")
EPOCH_RE = re.compile(r"(?<![\d.])(1(?:7[7-9]|8[0-9])\d{7})(?:\.\d+)?(?![\d])")
HEADING_DATE_RE = re.compile(r"^#{1,6}\s.*?\b(20\d{2}-\d{2}-\d{2})\b", re.M)
LOG_STAMP_RE = re.compile(r"^[-*]\s*(?:Started|Finished|Date|Time UTC)\b[^\n]*?\b(20\d{2}-\d{2}-\d{2})\b", re.M | re.I)
CONFIG_DATE_KEY_RE = re.compile(r'"(?:date|date_utc|run_date|started(?:_at)?)"\s*:\s*"(20\d{2}-\d{2}-\d{2})')
EPOCH_KEY_RE = re.compile(r'"(?:created_unix|start_unix|end_unix|timestamp)"\s*:\s*(1\d{9})(?:\.\d+)?')

TEXT_SUFFIXES = {".md", ".txt", ".log", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".py", ".sh", ".toml", ".cfg"}
SKIP_NAMES = {"metadata.yaml"}  # generated_on there is the catalog sweep, not a run date
MAX_TEXT_BYTES = 30_000_000


def parse_epoch(value: str) -> str | None:
    try:
        stamp = dt.datetime.fromtimestamp(float(value), dt.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return stamp.strftime("%Y-%m-%d")


def valid_date(text: str) -> bool:
    try:
        dt.date.fromisoformat(text)
    except ValueError:
        return False
    return True


class Evidence:
    """All date evidence found inside one experiment dir, bucketed by class."""

    def __init__(self) -> None:
        self.by_class: dict[str, set[str]] = {
            "literal": set(),
            "epoch": set(),
            "seed": set(),
            "git": set(),
            "mtime": set(),
        }
        self.precise: set[str] = set()  # from headings, log stamps, config keys, epoch fields

    def supports(self, date: str) -> list[str]:
        kinds = [name for name, pool in self.by_class.items() if date in pool]
        if kinds:
            return kinds
        day = dt.date.fromisoformat(date)
        for delta in (-1, 1):
            near = (day + dt.timedelta(days=delta)).isoformat()
            for name, pool in self.by_class.items():
                if near in pool:
                    return [name + "~1d"]
        return []


def gather_evidence(exp_dir: Path) -> Evidence:
    evidence = Evidence()
    for path in sorted(exp_dir.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        mtime = dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc)
        evidence.by_class["mtime"].add(mtime.strftime("%Y-%m-%d"))
        if path.suffix.lower() not in TEXT_SUFFIXES or stat.st_size > MAX_TEXT_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in DATE_RE.finditer(text):
            if valid_date(match.group(1)):
                evidence.by_class["literal"].add(match.group(1))
        for match in SEED_RE.finditer(text):
            candidate = "-".join(match.groups())
            if valid_date(candidate):
                evidence.by_class["seed"].add(candidate)
        for match in EPOCH_RE.finditer(text):
            start = match.start()
            if start >= 2 and text[start - 1] == "." and text[start - 2].isdigit():
                continue  # decimal fraction of a metric, not a timestamp
            day = parse_epoch(match.group(1))
            if day:
                evidence.by_class["epoch"].add(day)
        if path.suffix.lower() in {".md", ".txt", ".log"}:
            for regex in (HEADING_DATE_RE, LOG_STAMP_RE):
                for match in regex.finditer(text):
                    if valid_date(match.group(1)):
                        evidence.precise.add(match.group(1))
        if path.suffix.lower() in {".json", ".jsonl"}:
            for match in CONFIG_DATE_KEY_RE.finditer(text):
                if valid_date(match.group(1)):
                    evidence.precise.add(match.group(1))
            for match in EPOCH_KEY_RE.finditer(text):
                day = parse_epoch(match.group(1))
                if day:
                    evidence.precise.add(day)
    result = subprocess.run(
        ["git", "log", "--format=%ad", "--date=short", "--", str(exp_dir.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.strip():
            evidence.by_class["git"].add(line.strip())
    return evidence


def catalog_ids() -> list[str]:
    if not CATALOG_PATH.exists():
        return sorted(p.name for p in EXPERIMENTS.iterdir() if p.is_dir())
    with CATALOG_PATH.open(newline="", encoding="utf-8") as handle:
        return [row["id"] for row in csv.DictReader(handle)]


def suggest_entry(evidence: Evidence) -> dict[str, str] | None:
    if not evidence.precise:
        return None
    days = sorted(evidence.precise)
    return {
        "start": days[0],
        "end": days[-1],
        "confidence": "medium",
        "sources": ["suggested by scripts/extract_experiment_dates.py from dated headings/log stamps/config keys — review before committing"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suggest", action="store_true", help="print JSON entries for dirs missing from the dates file")
    args = parser.parse_args()

    recorded: dict[str, dict] = {}
    if DATES_PATH.exists():
        payload = json.loads(DATES_PATH.read_text(encoding="utf-8"))
        recorded = payload.get("experiments", {}) if isinstance(payload, dict) else {}

    problems: list[str] = []
    missing: list[str] = []
    verified = 0
    for exp_id in catalog_ids():
        exp_dir = EXPERIMENTS / exp_id
        entry = recorded.get(exp_id)
        if entry is None:
            missing.append(exp_id)
            continue
        if not entry.get("start"):
            continue  # searched, no recoverable record — nothing to verify
        if not exp_dir.is_dir():
            problems.append(f"{exp_id}: has a dates entry but experiments/{exp_id}/ does not exist")
            continue
        evidence = gather_evidence(exp_dir)
        entry_ok = True
        for label in ("start", "end"):
            date = str(entry.get(label, ""))
            if not date:
                continue
            if not valid_date(date):
                problems.append(f"{exp_id}: {label} {date!r} is not a valid YYYY-MM-DD date")
                entry_ok = False
                continue
            kinds = evidence.supports(date)
            if not kinds:
                problems.append(f"{exp_id}: {label} {date} has no supporting evidence in the dir or git history")
                entry_ok = False
        if entry_ok:
            verified += 1

    dated_total = sum(1 for entry in recorded.values() if entry.get("start"))
    print(f"experiment dates: {verified}/{dated_total} dated entries evidence-supported; {len(missing)} dir(s) missing an entry")
    for line in problems:
        print(f"- {line}")
    if missing:
        for exp_id in missing:
            note = ""
            if args.suggest:
                suggestion = suggest_entry(gather_evidence(EXPERIMENTS / exp_id))
                if suggestion:
                    note = " " + json.dumps({exp_id: suggestion})
                else:
                    note = " (no high-precision records found; add with confidence 'none')"
            print(f"- missing entry: {exp_id}{note}")
    return 1 if problems or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

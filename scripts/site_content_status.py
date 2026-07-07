#!/usr/bin/env python3
"""Report site-content coverage so the generated site stays maintained.

For every experiment in the catalog, checks whether the three curated,
committed content artifacts have an entry:

  knowledge/experiment_dates.json  — run window (dates)
  knowledge/experiment_viz.json    — native result charts
  knowledge/experiment_brief.json  — plain-language practitioner brief

Dates auto-fill deterministically (scripts/extract_experiment_dates.py --apply);
charts and briefs are authored by agent enrichment passes. The site build
degrades gracefully when any are missing, so this is an informational report by
default. Use --strict to exit non-zero when charts or briefs are missing (e.g.
to gate a release), and --json for machine-readable output listing the gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
CATALOG = KNOWLEDGE / "experiment_catalog.csv"


def catalog_ids() -> list[str]:
    if not CATALOG.exists():
        return []
    with CATALOG.open(newline="", encoding="utf-8") as handle:
        return [row["id"] for row in csv.DictReader(handle)]


def load_entries(name: str) -> dict:
    path = KNOWLEDGE / name
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    entries = payload.get("experiments", {}) if isinstance(payload, dict) else {}
    return entries if isinstance(entries, dict) else {}


def has_dates(entry: object) -> bool:
    return isinstance(entry, dict)  # present at all (empty = searched, no record)


def has_charts(entry: object) -> bool:
    return isinstance(entry, dict) and bool(entry.get("charts"))


def has_brief(entry: object) -> bool:
    return isinstance(entry, dict) and bool(str(entry.get("plain_answer", "")).strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any experiment lacks a practitioner brief (the enforced layer; charts/dates stay informational)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON of the gaps")
    args = parser.parse_args()

    ids = catalog_ids()
    dates = load_entries("experiment_dates.json")
    viz = load_entries("experiment_viz.json")
    briefs = load_entries("experiment_brief.json")

    gaps = {"dates": [], "charts": [], "brief": []}
    for exp_id in ids:
        if not has_dates(dates.get(exp_id)):
            gaps["dates"].append(exp_id)
        if not has_charts(viz.get(exp_id)):
            gaps["charts"].append(exp_id)
        if not has_brief(briefs.get(exp_id)):
            gaps["brief"].append(exp_id)

    total = len(ids)
    covered = {kind: total - len(missing) for kind, missing in gaps.items()}

    if args.json:
        print(json.dumps({"total": total, "covered": covered, "gaps": gaps}, indent=1))
    else:
        print(f"site content coverage over {total} experiments:")
        print(f"  dates   {covered['dates']:>4}/{total}")
        print(f"  charts  {covered['charts']:>4}/{total}")
        print(f"  briefs  {covered['brief']:>4}/{total}")
        for kind, label, fix in (
            ("dates", "no dates entry", "scripts/extract_experiment_dates.py --apply (git) or record extraction"),
            ("charts", "no result charts", "run the chart enrichment pass (docs/site_maintenance.md)"),
            ("brief", "no practitioner brief", "run the brief enrichment pass (docs/site_maintenance.md)"),
        ):
            missing = gaps[kind]
            if missing:
                shown = ", ".join(missing[:12]) + (f" … (+{len(missing) - 12})" if len(missing) > 12 else "")
                print(f"- {len(missing)} {label} → {fix}\n    {shown}")

    if args.strict and gaps["brief"]:
        print(
            f"\nSTRICT: {len(gaps['brief'])} experiment(s) lack a practitioner brief. "
            "Author them before committing — see docs/site_maintenance.md "
            "(run scripts/enrichment/enrich_briefs.workflow.js for the ids above, then merge_briefs.py)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

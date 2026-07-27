#!/usr/bin/env python3
"""Surface PARKED ideas that match a bottleneck you just found. Retrieval, not memory.

WHY THIS EXISTS (2026-07-26). A session independently rediscovered mid-stack layer looping, proposed
adaptive sequential best-of-N, and identified termination as the binding constraint on two surfaces --
and all three were already written down in this repo as parked `next_tests`:

  C59  "Intermediate-LAYER injection / layer-looping ... rather than re-embedding at the input"
  C63  "Adaptive selection at deploy: spend samples only on tasks not yet solved"
  C62  "A termination/finishing intervention ... train a stop-when-stuck signal -- test whether
        reducing the ~100% timeout rate lifts pass rate"
  C52  "Run long-context (16k+) loop-FTPO as its own experiment"

One was re-derived from scratch; the others were never surfaced at all. The knowledge was not missing,
the RETRIEVAL was: `make related` matches an idea against prior EXPERIMENTS, and nothing matched a
finding against the corpus's own unrun follow-ups. Worse, a technique can be filed as dead from a
negative on a different target -- FTPO reads as "failed" in the ledger (C52, reasoning-fork preference)
while the same doc names loop-targeted FTPO as the outstanding experiment.

USE IT when a result changes what you think the bottleneck is -- not only when starting a new idea:

    make parked QUERY="termination timeout non-convergence commit"

Searches claim next_tests AND avoid lists, open questions, the future queue, and docs/ prose.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _terms(query: str):
    return [t for t in re.split(r"[\s,]+", query.strip()) if len(t) > 2]


def _match(text: str, terms):
    low = text.lower()
    return sum(1 for t in terms if t.lower() in low)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    query = " ".join(sys.argv[1:])
    terms = _terms(query)
    print(f"parked ideas matching: {terms}\n")
    found = 0

    # 1) claim ledger: next_tests are the corpus's own explicit unrun follow-ups; `avoid` entries are
    #    the traps a re-attempt would otherwise walk into.
    led = ROOT / "knowledge" / "claims" / "claim_ledger.json"
    if led.exists():
        d = json.loads(led.read_text())
        claims = d["claims"] if isinstance(d, dict) and "claims" in d else d
        rows = []
        for c in claims:
            for field in ("next_tests", "avoid"):
                for item in (c.get(field) or []):
                    score = _match(item, terms)
                    if score:
                        rows.append((score, c.get("id", "?"), field, item.strip()))
        rows.sort(key=lambda r: -r[0])
        if rows:
            print(f"== CLAIM LEDGER ({len(rows)} hits) ==")
            for score, cid, field, item in rows[:12]:
                tag = "NEXT" if field == "next_tests" else "AVOID"
                print(f"  [{cid} {tag}] {item[:200]}")
            found += len(rows)
            print()

    # 2) future queue + open questions
    for rel in ("knowledge/future_experiment_queue.json", "knowledge/open_questions.md",
                "knowledge/future_program_seeds.md", "knowledge/research_roadmap.md"):
        p = ROOT / rel
        if not p.exists():
            continue
        if p.suffix == ".json":
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            items = data if isinstance(data, list) else data.get("items", data.get("proposals", []))
            rows = [(_match(json.dumps(i), terms), i) for i in items if isinstance(i, dict)]
            rows = sorted([r for r in rows if r[0]], key=lambda r: -r[0])
            if rows:
                print(f"== {rel} ({len(rows)} hits) ==")
                for score, i in rows[:8]:
                    print(f"  - {(i.get('title') or i.get('id') or '')[:160]}")
                found += len(rows)
                print()
        else:
            hits = [ln.strip() for ln in p.read_text().splitlines() if _match(ln, terms) >= 2]
            if hits:
                print(f"== {rel} ({len(hits)} hits) ==")
                for h in hits[:8]:
                    print(f"  {h[:190]}")
                found += len(hits)
                print()

    # 3) docs prose -- a technique's status can live in a doc while the ledger reads "failed"
    doc_hits = []
    for p in sorted((ROOT / "docs").glob("*.md")):
        for i, ln in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if _match(ln, terms) >= 2 and len(ln.strip()) > 40:
                doc_hits.append((p.name, i, ln.strip()))
    if doc_hits:
        print(f"== docs/ ({len(doc_hits)} lines) ==")
        for name, i, ln in doc_hits[:10]:
            print(f"  {name}:{i}: {ln[:170]}")
        found += len(doc_hits)
        print()

    print(f"total {found} parked items. A technique filed as FAILED may have failed on a DIFFERENT "
          f"target -- check the doc, not just the ledger verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

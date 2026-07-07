#!/usr/bin/env python3
"""Clean, lint, and merge authored practitioner briefs into the committed artifact.

Reads a JSON file of newly authored briefs (either {"experiments": {id: brief}}
or {id: brief}), normalizes them (HTML-unescape, arrow glyphs, drop stray keys),
lints each against the no-jargon ban-list and the schema shape, applies safe
mechanical fixes (held-out -> unseen, "op type" -> "operation type"), merges the
clean ones into knowledge/experiment_brief.json, and prints any ids that still
need re-authoring. Deterministic — no model in the loop; safe to run from the
maintenance cron.

    python3 scripts/enrichment/merge_briefs.py --in new_briefs.json
    python3 scripts/enrichment/merge_briefs.py --in new_briefs.json --json   # machine-readable

Exit code is 0 on success; the flagged list (if any) is reported for follow-up.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRIEFS = ROOT / "knowledge" / "experiment_brief.json"

PLAIN_FIELDS = ["verdict_tag", "concept_primer", "plain_question", "plain_answer", "why_it_matters"]
REQUIRED = PLAIN_FIELDS + ["verdict_tone", "key_numbers", "charts"]
TONES = {"positive", "negative", "mixed", "neutral"}
BAN = re.compile(
    r"\bC\d{1,2}\b|top-1|top1|pass@|greedy@|\bAUROC\b|\bWilson\b|channel-matched|parse-immune"
    r"|\bQLoRA\b|\bLoRA\b|\bSFT\b|\bDPO\b|\bGRPO\b|\bMBPP\b|HumanEval|\bDSL\b|\bpass@k\b"
    r"|\boracle\b|held[\s-]?out|holdout|\bAUC\b|\bELBO\b|logit|likelihood ranking"
    r"|next-op|\bops?\b|banking|self-distill|expert iteration|min-depth",
    re.I,
)
DECIMAL = re.compile(r"(?<![\d.])0\.\d{2,}")


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = text.replace(" -> ", " → ").replace("->", "→")
    # safe mechanical de-jargon
    text = re.sub(r"\bheld[\s-]?out\b", "unseen", text, flags=re.I)
    text = re.sub(r"\bholdout\b", "unseen", text, flags=re.I)
    text = re.sub(r"\bop type\b", "operation type", text)
    text = re.sub(r"\bop types\b", "operation types", text)
    return text.strip()


def clean_brief(raw: dict) -> dict:
    brief: dict = {}
    for key in PLAIN_FIELDS:
        if key in raw:
            brief[key] = normalize(str(raw[key]))
    brief["verdict_tone"] = str(raw.get("verdict_tone", "neutral"))
    brief["key_numbers"] = [
        {k: normalize(str(n.get(k, ""))) for k in ("label", "value", "sub")}
        for n in (raw.get("key_numbers") or [])
        if isinstance(n, dict)
    ][:4]
    charts = []
    for chart in raw.get("charts") or []:
        if isinstance(chart, dict) and isinstance(chart.get("index"), int):
            charts.append(
                {
                    "index": chart["index"],
                    "chart_plain_title": normalize(str(chart.get("chart_plain_title", ""))),
                    "chart_read": normalize(str(chart.get("chart_read", ""))),
                    "chart_takeaway": normalize(str(chart.get("chart_takeaway", ""))),
                }
            )
    brief["charts"] = charts
    return brief


def lint(exp_id: str, brief: dict) -> list[str]:
    issues = []
    for key in REQUIRED:
        value = brief.get(key)
        if key not in brief or (isinstance(value, str) and not value.strip()):
            issues.append(f"missing/empty {key}")
    if brief.get("verdict_tone") not in TONES:
        issues.append(f"bad verdict_tone {brief.get('verdict_tone')!r}")
    tag = str(brief.get("verdict_tag", ""))
    if tag.endswith("."):
        issues.append("verdict_tag ends with a period")
    if len(tag.split()) > 8:
        issues.append(f"verdict_tag too long ({len(tag.split())} words)")
    texts = [(f, str(brief.get(f, ""))) for f in PLAIN_FIELDS]
    texts.append(("key_numbers", " ".join(f"{n.get('label','')} {n.get('sub','')}" for n in brief.get("key_numbers", []))))
    for i, c in enumerate(brief.get("charts", [])):
        texts += [(f"chart[{i}]", " ".join(str(c.get(k, "")) for k in ("chart_plain_title", "chart_read", "chart_takeaway")))]
    for label, text in texts:
        for hit in set(BAN.findall(text)):
            issues.append(f"jargon in {label}: {hit!r}")
        if DECIMAL.search(text):
            issues.append(f"raw decimal in {label}")
    return issues


def load_input(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], dict):
        payload = payload["result"]
    if isinstance(payload, dict) and "experiments" in payload:
        return payload["experiments"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", required=True, help="JSON of newly authored briefs")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    incoming = load_input(Path(args.infile))
    existing = {"experiments": {}}
    if BRIEFS.exists():
        loaded = json.loads(BRIEFS.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    existing.setdefault("experiments", {})

    merged, flagged = [], {}
    for exp_id, raw in incoming.items():
        if not isinstance(raw, dict):
            flagged[exp_id] = ["not an object"]
            continue
        brief = clean_brief(raw)
        issues = lint(exp_id, brief)
        if issues:
            flagged[exp_id] = issues
        else:
            existing["experiments"][exp_id] = brief
            merged.append(exp_id)

    if merged:
        existing["experiments"] = dict(sorted(existing["experiments"].items()))
        BRIEFS.write_text(json.dumps(existing, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps({"merged": sorted(merged), "flagged": flagged}, indent=1))
    else:
        print(f"merged {len(merged)} brief(s); {len(flagged)} flagged for re-authoring")
        for exp_id, issues in sorted(flagged.items()):
            print(f"- {exp_id}: {'; '.join(sorted(set(issues))[:6])}")
        if flagged:
            print("re-author:", json.dumps(sorted(flagged.keys())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

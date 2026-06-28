#!/usr/bin/env python3
"""Find related programs, claims, and experiments for a rough research idea."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "only",
    "or",
    "should",
    "that",
    "the",
    "then",
    "to",
    "under",
    "use",
    "when",
    "with",
}


def tokens(text: str) -> Counter[str]:
    parts = re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower().replace("_", " "))
    expanded: list[str] = []
    for part in parts:
        expanded.extend(piece for piece in re.split(r"[._-]+", part) if piece)
        expanded.append(part)
    return Counter(token for token in expanded if len(token) > 2 and token not in STOPWORDS)


def parse_inline_list(value: str) -> list[str]:
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return []
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip('"').strip("'") for item in inner.split(",") if item.strip()]


def load_programs() -> list[dict[str, str]]:
    registry = ROOT / "research_programs" / "registry.yaml"
    programs: list[dict[str, str]] = []
    current: dict[str, object] | None = None
    active_list: str | None = None
    for raw_line in registry.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "programs:":
            continue
        if stripped.startswith("- id:"):
            if current:
                programs.append(flatten_program(current))
            current = {"id": stripped.split(":", 1)[1].strip().strip('"'), "title": "", "focus": "", "seed_tags": []}
            active_list = None
            continue
        if current is None:
            continue
        if stripped.startswith("- ") and active_list:
            current[active_list].append(stripped[2:].strip().strip('"'))  # type: ignore[index, union-attr]
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "seed_tags":
            parsed = parse_inline_list(value)
            current[key] = parsed
            active_list = key if not parsed else None
        elif key in {"title", "focus", "charter"}:
            current[key] = value.strip('"')
            active_list = None
    if current:
        programs.append(flatten_program(current))
    return programs


def flatten_program(program: dict[str, object]) -> dict[str, str]:
    program_id = str(program.get("id", ""))
    program_dir = ROOT / "research_programs" / program_id
    program_text = []
    for filename in ["charter.md", "backlog.md", "evidence.md"]:
        path = program_dir / filename
        if path.exists():
            program_text.append(path.read_text(encoding="utf-8", errors="replace"))
    return {
        "id": program_id,
        "title": str(program.get("title", "")),
        "focus": str(program.get("focus", "")),
        "tags": " ".join(str(tag) for tag in program.get("seed_tags", [])),
        "path": f"research_programs/{program_id}/charter.md",
        "text": "\n".join(program_text),
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def document_text(kind: str, row: dict[str, str]) -> str:
    if kind == "program":
        return " ".join([row["id"], row["title"], row["focus"], row["tags"], row["text"]])
    if kind == "claim":
        return " ".join([row["id"], row["title"], row["status"], row["programs"], row["evidence"], row["summary"], row["implication"]])
    if kind == "future_queue":
        return " ".join(
            [
                row["id"],
                row["title"],
                row["status"],
                row["priority"],
                row["effort"],
                row["programs"],
                row["question"],
                row["hypothesis"],
                row["success_signal"],
                row["failure_signal"],
                row["next_step"],
            ]
        )
    return " ".join([row["id"], row["title"], row["tags"], row["research_programs"], row["summary"]])


def score(query_terms: Counter[str], doc_terms: Counter[str]) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    overlap = set(query_terms) & set(doc_terms)
    if not overlap:
        return 0.0
    weighted_overlap = sum((1 + math.log(query_terms[token] + 1)) * (1 + math.log(doc_terms[token] + 1)) for token in overlap)
    coverage = len(overlap) / len(query_terms)
    density = len(overlap) / len(doc_terms)
    return weighted_overlap * (1.0 + coverage) * (1.0 + min(density * 8, 1.0))


def scored(kind: str, rows: list[dict[str, str]], query_terms: Counter[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in rows:
        doc_terms = tokens(document_text(kind, row))
        value = score(query_terms, doc_terms)
        if value <= 0:
            continue
        overlap = sorted(set(query_terms) & set(doc_terms))
        result = dict(row)
        result.pop("text", None)
        result["score"] = round(value, 3)
        result["matched_terms"] = overlap[:12]
        results.append(result)
    return sorted(results, key=lambda item: (-float(item["score"]), str(item.get("id", ""))))


def load_related(query: str, top: int) -> dict[str, list[dict[str, object]]]:
    query_terms = tokens(query)
    programs = scored("program", load_programs(), query_terms)[:top]
    claims = scored("claim", load_csv(ROOT / "knowledge" / "claims" / "index.csv"), query_terms)[:top]
    experiments = scored("experiment", load_csv(ROOT / "knowledge" / "experiment_catalog.csv"), query_terms)[:top]
    future_queue = scored("future_queue", load_csv(ROOT / "knowledge" / "future_experiment_queue.csv"), query_terms)[:top]
    return {"programs": programs, "claims": claims, "experiments": experiments, "future_queue": future_queue}


def md_result(query: str, results: dict[str, list[dict[str, object]]]) -> str:
    lines = ["# Related Work", "", f"Query: {query}", ""]
    sections = [
        ("Programs", "programs", ["id", "title", "focus", "path"]),
        ("Queued Future Work", "future_queue", ["id", "title", "priority", "status", "programs", "next_step"]),
        ("Claims", "claims", ["id", "title", "status", "programs"]),
        ("Experiments", "experiments", ["id", "title", "research_programs", "primary_report"]),
    ]
    for title, key, fields in sections:
        lines.extend([f"## {title}", ""])
        rows = results[key]
        if not rows:
            lines.extend(["No strong lexical matches.", ""])
            continue
        for row in rows:
            lines.append(f"### {row.get('id', '')}")
            lines.append(f"- Score: {row['score']}")
            lines.append(f"- Matched terms: {', '.join(row['matched_terms'])}")
            for field in fields:
                value = str(row.get(field, "")).strip()
                if value:
                    lines.append(f"- {field}: {value}")
            if key == "experiments":
                summary = str(row.get("summary", ""))
                if summary:
                    lines.append(f"- summary: {summary}")
            if key == "claims":
                implication = str(row.get("implication", ""))
                if implication:
                    lines.append(f"- implication: {implication}")
            if key == "future_queue":
                question = str(row.get("question", ""))
                if question:
                    lines.append(f"- question: {question}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_intake(path: Path, query: str, results: dict[str, list[dict[str, object]]]) -> None:
    programs = results["programs"]
    claims = results["claims"]
    experiments = results["experiments"]
    future_queue = results.get("future_queue", [])
    lines = [
        "# Idea Intake",
        "",
        "## Program Fit",
        "",
        f"- Program: {programs[0]['id'] if programs else ''}",
        "- Existing or new program: existing",
        "- Closest program scorecard reviewed: knowledge/program_scorecards.md",
        f"- Related future queue item: {future_queue[0]['id'] if future_queue else ''}",
        "",
        "## Prior Evidence",
        "",
    ]
    for index in range(3):
        value = experiments[index]["id"] if index < len(experiments) else ""
        lines.append(f"- Anchor {index + 1}: {value}")
    lines.extend(
        [
            f"- Closest duplicate or near-duplicate: {experiments[0]['id'] if experiments else ''}",
            "",
            "## Novelty Claim",
            "",
            query,
            "",
            "## Related Claims",
            "",
        ]
    )
    for claim in claims[:3]:
        lines.append(f"- {claim['id']}: {claim['title']} ({claim['status']})")
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            "Why should this work, and what would make that explanation false?",
            "",
            "## Control Plan",
            "",
            "- Baseline:",
            "- Mechanism-falsifying control:",
            "- Shift or robustness check:",
            "- Hidden-label boundary:",
            "",
            "## Evidence Output",
            "",
            "- Program evidence update:",
            "- Claim ledger or synthesis update:",
            "- Reusable artifact:",
            "- Stop or branch condition:",
            "",
            "## Decision",
            "",
            "- Run experiment:",
            "- Create program:",
            "- Write synthesis only:",
            "- Defer:",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="rough idea, mechanism, task, or question")
    parser.add_argument("--top", type=int, default=5, help="results per section")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    parser.add_argument("--write-intake", type=Path, help="write a prefilled idea intake note")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        print("usage: scripts/find_related.py <rough idea>", file=sys.stderr)
        return 2
    results = load_related(query, args.top)
    if args.write_intake:
        write_intake(args.write_intake, query, results)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(md_result(query, results), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

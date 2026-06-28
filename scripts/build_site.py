#!/usr/bin/env python3
"""Build the static research website from generated repository data."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
PROGRAMS = ROOT / "research_programs"
TEMPLATE = ROOT / "templates" / "site"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def split_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def parse_inline_list(value: str) -> list[str]:
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return []
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip('"').strip("'") for item in inner.split(",") if item.strip()]


def load_programs() -> list[dict[str, object]]:
    registry = PROGRAMS / "registry.yaml"
    programs: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    active_list: str | None = None
    for raw_line in registry.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "programs:":
            continue
        if stripped.startswith("- id:"):
            if current:
                programs.append(current)
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
        if key in {"seed_tags", "seed_experiments"}:
            parsed = parse_inline_list(value)
            current[key] = parsed
            active_list = key if not parsed else None
        else:
            current[key] = value.strip('"')
            active_list = None
    if current:
        programs.append(current)
    return programs


def first_paragraph(path: Path, limit: int = 420) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith("#") or stripped.startswith("- ") or stripped.startswith("|"):
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append(" ".join(current))
    for block in blocks:
        cleaned = re.sub(r"\s+", " ", block).strip()
        if len(cleaned) > 50:
            return cleaned[: limit - 3].rstrip() + "..." if len(cleaned) > limit else cleaned
    return ""


def int_value(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def merge_experiments() -> list[dict[str, object]]:
    catalog = read_csv(KNOWLEDGE / "experiment_catalog.csv")
    readiness_by_id = {row["id"]: row for row in read_csv(KNOWLEDGE / "experiment_readiness.csv")}
    experiments: list[dict[str, object]] = []
    for row in catalog:
        ready = readiness_by_id.get(row["id"], {})
        experiment = {
            "id": row["id"],
            "title": row["title"],
            "source_track": row["source_track"],
            "tags": split_list(row["tags"]),
            "programs": split_list(row["research_programs"]),
            "summary": row["summary"],
            "path": row["path"],
            "primary_readme": row["primary_readme"],
            "primary_report": row["primary_report"],
            "total_files": int_value(row["total_files"]),
            "total_size_bytes": int_value(row["total_size_bytes"]),
            "readme_status": ready.get("readme_status", ""),
            "experiment_log": ready.get("experiment_log", ""),
            "run_surface": ready.get("run_surface", ""),
            "smoke_command": ready.get("smoke_command", ""),
            "artifact_manifest": ready.get("artifact_manifest", ""),
            "manifest_kinds": split_list(ready.get("manifest_kinds", "")),
            "recognized_artifacts": split_list(ready.get("recognized_artifacts", "")),
            "anchor_ready": ready.get("anchor_ready", ""),
            "needs": [] if ready.get("needs") == "none" else split_list(ready.get("needs", "")),
        }
        experiments.append(experiment)
    return experiments


def build_data() -> dict[str, object]:
    experiments = merge_experiments()
    future_queue = read_json(KNOWLEDGE / "future_experiment_queue.json")
    queue_rows = read_csv(KNOWLEDGE / "future_experiment_queue.csv")
    claims = read_csv(KNOWLEDGE / "claims" / "index.csv")
    artifacts = read_csv(KNOWLEDGE / "artifact_manifest_index.csv")
    program_index = read_csv(KNOWLEDGE / "research_program_index.csv")
    manifest = read_json(KNOWLEDGE / "experiment_manifest.json")
    programs = load_programs()

    program_counts = Counter()
    program_ready = Counter()
    program_queue = Counter()
    program_claims = Counter()
    for experiment in experiments:
        for program_id in experiment["programs"]:  # type: ignore[index]
            program_counts[str(program_id)] += 1
            if experiment["anchor_ready"] == "yes":
                program_ready[str(program_id)] += 1
    for row in queue_rows:
        program_queue.update(split_list(row["programs"]))
    for row in claims:
        program_claims.update(split_list(row["programs"]))

    program_cards = []
    for program in programs:
        program_id = str(program["id"])
        program_cards.append(
            {
                "id": program_id,
                "title": str(program.get("title", "")),
                "focus": str(program.get("focus", "")),
                "path": f"research_programs/{program_id}/charter.md",
                "seed_tags": program.get("seed_tags", []),
                "experiment_count": program_counts[program_id],
                "anchor_ready_count": program_ready[program_id],
                "queue_count": program_queue[program_id],
                "claim_count": program_claims[program_id],
                "excerpt": first_paragraph(PROGRAMS / program_id / "charter.md"),
            }
        )

    tag_counts = Counter()
    need_counts = Counter()
    readiness_counts = Counter()
    run_surface_counts = Counter()
    source_counts = Counter()
    for experiment in experiments:
        tag_counts.update(experiment["tags"])  # type: ignore[arg-type]
        need_counts.update(experiment["needs"])  # type: ignore[arg-type]
        readiness_counts[str(experiment["anchor_ready"])] += 1
        run_surface_counts[str(experiment["run_surface"])] += 1
        source_counts[str(experiment["source_track"])] += 1

    artifact_counts = Counter(row["kind"] for row in artifacts)
    manifest_counts = Counter(row["experiment_id"] for row in artifacts)
    extension_counts: Counter[str] = Counter()
    top_dirs: Counter[str] = Counter()
    if isinstance(manifest, list):
        for item in manifest:
            if not isinstance(item, dict):
                continue
            extension_counts.update({str(key): int_value(value) for key, value in dict(item.get("file_counts", {})).items()})
            top_dirs.update(str(name) for name in item.get("top_level_dirs", []))

    def counter_rows(counter: Counter[str], limit: int | None = None) -> list[dict[str, object]]:
        rows = [{"id": key, "value": value} for key, value in counter.most_common(limit)]
        return rows

    queue_proposals = []
    for proposal in future_queue.get("proposals", []):  # type: ignore[union-attr]
        if not isinstance(proposal, dict):
            continue
        clean = dict(proposal)
        clean["programs"] = [str(program_id) for program_id in proposal.get("programs", [])]
        clean["expected_artifacts"] = [str(item) for item in proposal.get("expected_artifacts", [])]
        queue_proposals.append(clean)

    candidate_programs = [
        dict(item) for item in future_queue.get("candidate_programs", []) if isinstance(item, dict)  # type: ignore[union-attr]
    ]

    summary = {
        "experiments": len(experiments),
        "programs": len(program_cards),
        "claims": len(claims),
        "future_proposals": len(queue_proposals),
        "candidate_programs": len(candidate_programs),
        "anchor_ready": readiness_counts["yes"],
        "needs_curation": readiness_counts["no"],
        "artifact_manifests": len(manifest_counts),
        "total_files": sum(int(experiment["total_files"]) for experiment in experiments),
        "total_size_bytes": sum(int(experiment["total_size_bytes"]) for experiment in experiments),
    }

    return {
        "generated_at": __import__("datetime").date.today().isoformat(),
        "repo": {
            "name": "small-model-experimentation",
            "github": "https://github.com/ericflo/small-model-experimentation",
        },
        "summary": summary,
        "programs": program_cards,
        "experiments": experiments,
        "claims": claims,
        "queue": queue_proposals,
        "candidate_programs": candidate_programs,
        "artifacts": artifacts,
        "program_index": program_index,
        "charts": {
            "tags": counter_rows(tag_counts, 30),
            "needs": counter_rows(need_counts),
            "readiness": counter_rows(readiness_counts),
            "run_surfaces": counter_rows(run_surface_counts),
            "source_tracks": counter_rows(source_counts),
            "artifact_kinds": counter_rows(artifact_counts),
            "extensions": counter_rows(extension_counts, 24),
            "top_dirs": counter_rows(top_dirs),
            "queue_status": counter_rows(Counter(str(item.get("status", "")) for item in queue_proposals)),
            "queue_priority": counter_rows(Counter(str(item.get("priority", "")) for item in queue_proposals)),
        },
        "narrative": {
            "synthesis": first_paragraph(KNOWLEDGE / "synthesis.md", 520),
            "roadmap": first_paragraph(KNOWLEDGE / "research_roadmap.md", 520),
            "patterns": first_paragraph(KNOWLEDGE / "patterns.md", 520),
        },
    }


def write_site(out_dir: Path) -> None:
    data = build_data()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    html = (TEMPLATE / "index.html").read_text(encoding="utf-8")
    embedded_data = json.dumps(data, sort_keys=True).replace("</", "<\\/")
    html = html.replace("__SITE_DATA__", embedded_data)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    shutil.copyfile(TEMPLATE / "assets" / "styles.css", out_dir / "assets" / "styles.css")
    shutil.copyfile(TEMPLATE / "assets" / "app.js", out_dir / "assets" / "app.js")
    (out_dir / "data" / "site-data.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="site", type=Path, help="output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    write_site(out_dir)
    print(f"built site: {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

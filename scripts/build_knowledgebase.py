#!/usr/bin/env python3
"""Build repository catalogs from self-contained experiment folders."""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
KNOWLEDGE = ROOT / "knowledge"

TRACK_Z_NON_QWEN35 = {
    "bridge_dose_recombination_curriculum",
    "counterexample_rule_repair",
    "execution_conditioned_repair",
    "factor_recombination_ladder",
    "feature_factorized_rule_diversity",
    "real_transform_abi_gate_with_counterexamples",
    "rule_family_diversity_scaling",
    "targeted_bridge_allocation",
    "trace_keyed_symbol_repair",
}

TAG_KEYWORDS = {
    "abi": ["abi", "bytecode", "compiler"],
    "active-learning": ["active", "acquisition", "interrogation", "query"],
    "bridge": ["bridge"],
    "coverage": ["coverage", "passk", "ceiling"],
    "counterexamples": ["counterexample", "hard_negative", "negative"],
    "curriculum": ["curriculum", "ladder", "scheduled"],
    "distillation": ["distill", "teacher", "dagger"],
    "execution": ["executor", "executable", "execution", "program"],
    "foofah": ["foofah"],
    "latent-state": ["latent", "register", "slot", "state"],
    "lora": ["lora", "qlora", "adapter"],
    "memory": ["memory", "rag", "retrieval"],
    "operator-search": ["operator", "inventory", "shortlister"],
    "policy": ["policy", "grpo", "rl", "mdp"],
    "repair": ["repair", "refiner"],
    "small-model": ["qwen", "qwen35", "4b"],
    "table-transform": ["table", "foofah", "transform"],
    "tools": ["tool"],
    "verification": ["verifier", "verify", "verified", "oracle", "judge"],
}

STANDARD_DIRS = [
    "README.md",
    "metadata.yaml",
    "experiment_log.md",
    "src/",
    "scripts/",
    "configs/",
    "data/",
    "runs/",
    "analysis/",
    "reports/",
]


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def excluded(path: Path) -> bool:
    parts = path.parts
    if "__pycache__" in parts:
        return True
    if path.name.endswith(":Zone.Identifier") or path.suffix == ".pyc":
        return True
    for index, part in enumerate(parts[:-1]):
        if part == "reports" and parts[index + 1] == "adapters":
            return True
    return False


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    return path.relative_to(ROOT).as_posix()


def title_from_slug(slug: str) -> str:
    words = slug.replace("qwen35_4b", "qwen3.5_4b").split("_")
    return " ".join(w.upper() if w in {"abi", "rag", "vm", "dpo", "rl"} else w.capitalize() for w in words)


def source_track(slug: str) -> str:
    if slug.startswith("qwen35_4b_") or slug in TRACK_Z_NON_QWEN35:
        return "track-z"
    return "track-y"


def first_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def first_paragraph(text: str) -> str:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith("#") or stripped.startswith("|") or stripped.startswith("!") or stripped.startswith("- "):
            continue
        if not stripped:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append(" ".join(current))
    for block in blocks:
        if len(block) > 40:
            return re.sub(r"\s+", " ", block).strip()
    return ""


def generated_stub_summary(text: str) -> str:
    marker = "generated during repository normalization"
    if marker not in text.lower():
        return ""
    match = re.search(r"^## Summary\s+(.+?)(?:\n## |\Z)", text, flags=re.M | re.S)
    if not match:
        return ""
    return first_paragraph(match.group(1))


def find_primary_report(exp: Path) -> Path | None:
    candidates: list[Path] = []
    reports = exp / "reports"
    if reports.exists():
        for pattern in [
            "final_report.md",
            "*_report.md",
            "report.md",
            "*_paper.md",
            "*summary.md",
            "*.md",
        ]:
            candidates.extend(sorted(reports.rglob(pattern)))
    analysis_summary = exp / "analysis" / "summary.md"
    if analysis_summary.exists():
        candidates.append(analysis_summary)
    seen = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique[0] if unique else None


def summarize(exp: Path, primary_report: Path | None) -> tuple[str, str]:
    readme = exp / "README.md"
    text = read_text(readme, 6000)
    if not text and primary_report:
        text = read_text(primary_report, 6000)
    title = first_heading(text) or title_from_slug(exp.name)
    summary = generated_stub_summary(text) or first_paragraph(text)
    if not summary and primary_report:
        summary = first_paragraph(read_text(primary_report, 6000))
    if not summary:
        summary = "Imported standalone experiment. See the local reports, analysis outputs, and source files."
    return title, summary


def extension(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "[none]"


def file_stats(exp: Path) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    total_size = 0
    total_files = 0
    for path in exp.rglob("*"):
        if not path.is_file():
            continue
        if excluded(path.relative_to(ROOT)):
            continue
        counts[extension(path)] += 1
        total_files += 1
        total_size += path.stat().st_size
    return counts, total_files, total_size


def top_level_dirs(exp: Path) -> list[str]:
    return sorted(path.name for path in exp.iterdir() if path.is_dir())


def tags_for(slug: str, title: str, summary: str) -> list[str]:
    haystack = f"{slug} {title} {summary}".lower()
    tags = []
    for tag, needles in TAG_KEYWORDS.items():
        if any(needle in haystack for needle in needles):
            tags.append(tag)
    return tags or ["experiment"]


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_metadata(record: dict[str, object]) -> None:
    path = ROOT / str(record["path"]) / "metadata.yaml"
    counts = record["file_counts"]
    lines = [
        "# Generated by scripts/build_knowledgebase.py. Edit only when adding human curation.",
        f"id: {yaml_scalar(str(record['id']))}",
        f"title: {yaml_scalar(str(record['title']))}",
        f"source_track: {yaml_scalar(str(record['source_track']))}",
        f"path: {yaml_scalar(str(record['path']))}",
        f"primary_readme: {yaml_scalar(str(record['primary_readme']))}",
        f"primary_report: {yaml_scalar(str(record['primary_report']))}",
        f"summary: {yaml_scalar(str(record['summary']))}",
        "tags:",
    ]
    for tag in record["tags"]:
        lines.append(f"  - {yaml_scalar(str(tag))}")
    lines.extend(
        [
            "top_level_dirs:",
            *[f"  - {yaml_scalar(name)}" for name in record["top_level_dirs"]],
            "file_counts:",
            *[f"  {key.lstrip('.') if key.startswith('.') else key}: {counts[key]}" for key in sorted(counts)],
            f"total_files: {record['total_files']}",
            f"total_size_bytes: {record['total_size_bytes']}",
            f"generated_on: {yaml_scalar(str(record['generated_on']))}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_missing_readme(record: dict[str, object]) -> bool:
    readme = ROOT / str(record["path"]) / "README.md"
    if readme.exists():
        return False
    report = str(record["primary_report"])
    report_rel = Path(report).relative_to(record["path"]).as_posix() if report else ""
    lines = [
        f"# {record['title']}",
        "",
        "This top-level README was generated during repository normalization because the imported experiment did not include one.",
        "",
        f"- Source track: `{record['source_track']}`",
        f"- Primary report: [{report_rel}]({report_rel})" if report_rel else "- Primary report: not detected",
        "- Metadata: [metadata.yaml](metadata.yaml)",
        "",
        "## How To Read",
        "",
        "Start with the primary report, then inspect `data/`, `reports/`, `analysis/`, `src/`, and `scripts/` as available. This folder remains self-contained; do not move its run data into shared directories.",
        "",
        "## Summary",
        "",
        str(record["summary"]),
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    return True


def collect_records() -> list[dict[str, object]]:
    if not EXPERIMENTS.exists():
        raise SystemExit("experiments/ does not exist")
    generated_on = dt.date.today().isoformat()
    records: list[dict[str, object]] = []
    for exp in sorted(path for path in EXPERIMENTS.iterdir() if path.is_dir()):
        primary_report = find_primary_report(exp)
        title, summary = summarize(exp, primary_report)
        counts, total_files, total_size = file_stats(exp)
        record = {
            "id": exp.name,
            "title": title,
            "source_track": source_track(exp.name),
            "path": rel(exp),
            "primary_readme": rel(exp / "README.md") if (exp / "README.md").exists() else "",
            "primary_report": rel(primary_report),
            "summary": summary,
            "tags": tags_for(exp.name, title, summary),
            "top_level_dirs": top_level_dirs(exp),
            "file_counts": dict(counts),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "generated_on": generated_on,
        }
        records.append(record)
    return records


def write_catalog_csv(records: list[dict[str, object]]) -> None:
    path = KNOWLEDGE / "experiment_catalog.csv"
    fieldnames = [
        "id",
        "title",
        "source_track",
        "tags",
        "summary",
        "path",
        "primary_readme",
        "primary_report",
        "total_files",
        "total_size_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: ";".join(record[key]) if key == "tags" else record[key]
                    for key in fieldnames
                }
            )


def md_link(label: str, target: str) -> str:
    return f"[{label}](../{target})" if target else ""


def write_catalog_md(records: list[dict[str, object]]) -> None:
    path = KNOWLEDGE / "experiment_catalog.md"
    track_counts = Counter(str(record["source_track"]) for record in records)
    lines = [
        "# Experiment Catalog",
        "",
        f"Generated from `experiments/` on {dt.date.today().isoformat()}.",
        "",
        f"- Experiments: {len(records)}",
        f"- Track Y imports: {track_counts['track-y']}",
        f"- Track Z imports: {track_counts['track-z']}",
        "",
        "| Track | Experiment | Tags | Summary | Read | Report |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        summary = str(record["summary"]).replace("|", "\\|")
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        tags = ", ".join(record["tags"])
        lines.append(
            "| {track} | `{id}` | {tags} | {summary} | {readme} | {report} |".format(
                track=record["source_track"],
                id=record["id"],
                tags=tags,
                summary=summary,
                readme=md_link("README", str(record["primary_readme"])),
                report=md_link("report", str(record["primary_report"])),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tag_index(records: list[dict[str, object]]) -> None:
    by_tag: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        for tag in record["tags"]:
            by_tag[str(tag)].append(record)
    lines = [
        "# Tag Index",
        "",
        "Tags are generated from experiment ids, titles, and summaries. Treat them as navigation aids, not final taxonomy.",
        "",
    ]
    for tag in sorted(by_tag):
        lines.extend([f"## {tag}", ""])
        for record in by_tag[tag]:
            lines.append(
                f"- `{record['id']}` ({record['source_track']}): {md_link('README', str(record['primary_readme']))}"
            )
        lines.append("")
    (KNOWLEDGE / "tag_index.md").write_text("\n".join(lines), encoding="utf-8")


def write_artifact_index(records: list[dict[str, object]]) -> None:
    ext_counts: Counter[str] = Counter()
    dir_counts: Counter[str] = Counter()
    largest: list[tuple[int, str]] = []
    for record in records:
        ext_counts.update(record["file_counts"])
        for dirname in record["top_level_dirs"]:
            dir_counts[str(dirname)] += 1
    for path in EXPERIMENTS.rglob("*"):
        if path.is_file() and not excluded(path.relative_to(ROOT)):
            largest.append((path.stat().st_size, rel(path)))
    largest.sort(reverse=True)

    lines = [
        "# Artifact Index",
        "",
        "This is a repository-level inventory. Each experiment remains the source of truth for its own artifacts.",
        "",
        "## Standard Experiment Shape",
        "",
    ]
    for item in STANDARD_DIRS:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Top-Level Directory Coverage", "", "| Directory | Experiments |", "| --- | ---: |"])
    for dirname, count in sorted(dir_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{dirname}/` | {count} |")
    lines.extend(["", "## File Extensions", "", "| Extension | Files |", "| --- | ---: |"])
    for ext, count in sorted(ext_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{ext}` | {count} |")
    lines.extend(["", "## Largest Files", "", "| Size MB | File |", "| ---: | --- |"])
    for size, path in largest[:30]:
        lines.append(f"| {size / 1024 / 1024:.1f} | `{path}` |")
    (KNOWLEDGE / "artifact_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_tracks(records: list[dict[str, object]]) -> None:
    lines = [
        "# Source Track Provenance",
        "",
        "The raw import contained two independent tracks. The final repository uses one flattened `experiments/<id>/` namespace; source track is preserved here and in each `metadata.yaml`.",
        "",
    ]
    for track in ["track-y", "track-z"]:
        track_records = [record for record in records if record["source_track"] == track]
        lines.extend([f"## {track}", "", f"Experiments: {len(track_records)}", ""])
        for record in track_records:
            lines.append(f"- `{record['id']}`")
        lines.append("")
    (KNOWLEDGE / "source_tracks.md").write_text("\n".join(lines), encoding="utf-8")

    with (KNOWLEDGE / "source_tracks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source_track", "path"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": record["id"], "source_track": record["source_track"], "path": record["path"]})


def write_json_manifest(records: list[dict[str, object]]) -> None:
    serializable = []
    for record in records:
        clean = dict(record)
        clean["file_counts"] = dict(record["file_counts"])
        serializable.append(clean)
    (KNOWLEDGE / "experiment_manifest.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generated_readme_ids(records: list[dict[str, object]], generated: list[str]) -> list[str]:
    ids = set(generated)
    marker = "generated during repository normalization"
    for record in records:
        readme = ROOT / str(record["primary_readme"]) if record["primary_readme"] else None
        if readme and readme.exists() and marker in read_text(readme, 1000).lower():
            ids.add(str(record["id"]))
    return sorted(ids)


def write_readme_gap_report(records: list[dict[str, object]], generated: list[str]) -> None:
    generated = generated_readme_ids(records, generated)
    lines = [
        "# README Coverage",
        "",
        "Top-level README files are required for all experiments. Generated entries should be replaced with human-authored summaries when the experiment is revisited.",
        "",
    ]
    if generated:
        lines.extend(["## Generated During Normalization", ""])
        for exp_id in generated:
            lines.append(f"- `{exp_id}`")
        lines.append("")
    lines.extend(["## Current Status", "", "| Experiment | README | Report |", "| --- | --- | --- |"])
    for record in records:
        lines.append(
            f"| `{record['id']}` | {md_link('README', str(record['primary_readme']))} | {md_link('report', str(record['primary_report']))} |"
        )
    (KNOWLEDGE / "readme_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    generated_readmes: list[str] = []
    for record in records:
        if write_missing_readme(record):
            generated_readmes.append(str(record["id"]))

    records = collect_records()
    for record in records:
        write_metadata(record)

    records = collect_records()
    write_catalog_csv(records)
    write_catalog_md(records)
    write_tag_index(records)
    write_artifact_index(records)
    write_source_tracks(records)
    write_json_manifest(records)
    write_readme_gap_report(records, generated_readmes)
    print(f"indexed {len(records)} experiments")
    if generated_readmes:
        print(f"generated {len(generated_readmes)} README stubs")


if __name__ == "__main__":
    main()

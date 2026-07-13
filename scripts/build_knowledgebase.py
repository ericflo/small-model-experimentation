#!/usr/bin/env python3
"""Build repository catalogs from self-contained experiment folders.

Output must be a pure function of repo content — byte-stable regardless of run
date. Never write wall-clock values into the generated files (enforced by
scripts/validate_repository.py); the old generated_on stamps dirtied every
branch at UTC midnight.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
KNOWLEDGE = ROOT / "knowledge"
PROGRAMS = ROOT / "research_programs"
PROGRAM_REGISTRY = PROGRAMS / "registry.yaml"
CLAIMS = KNOWLEDGE / "claims"
CLAIM_LEDGER = CLAIMS / "claim_ledger.json"
FUTURE_QUEUE = KNOWLEDGE / "future_experiment_queue.json"

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

EXCLUDED_DIR_NAMES = {
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}

COMMAND_PATTERNS = [
    r"\bpython(?:3)?\s+",
    r"\bpython\s+-m\b",
    r"\buv\s+run\b",
    r"\bmake\s+",
    r"\bbash\s+",
    r"\bpytest\b",
    r"\bnpm\s+",
]

SCRIPT_EXTENSIONS = {".py", ".sh", ".bash", ".ipynb", ".R", ".jl"}


def parse_inline_list(value: str) -> list[str]:
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return []
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip('"').strip("'") for item in inner.split(",") if item.strip()]


def load_programs() -> list[dict[str, object]]:
    if not PROGRAM_REGISTRY.exists():
        return []
    programs: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    active_list: str | None = None
    for raw_line in PROGRAM_REGISTRY.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "programs:":
            continue
        if stripped.startswith("- id:"):
            if current:
                programs.append(current)
            current = {
                "id": stripped.split(":", 1)[1].strip().strip('"'),
                "title": "",
                "charter": "",
                "focus": "",
                "seed_tags": [],
                "seed_experiments": [],
            }
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


PROGRAMS_CACHE = load_programs()


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def metadata_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    prefix = f"{key}: "
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value
    return ""


def metadata_list(path: Path, key: str) -> list[str]:
    if not path.exists():
        return []
    values: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_list = False
    for line in lines:
        if line.startswith(f"{key}:"):
            in_list = True
            continue
        if in_list:
            if line.startswith("  - "):
                values.append(line[4:].strip().strip('"'))
            elif line and not line.startswith(" "):
                break
    return values


def excluded(path: Path) -> bool:
    parts = path.parts
    if EXCLUDED_DIR_NAMES.intersection(parts):
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


def source_track_for(exp: Path) -> str:
    metadata = exp / "metadata.yaml"
    existing = metadata_value(metadata, "source_track")
    if existing:
        return existing
    return "new"


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
        # Skip the canonical "**Status:** finished|in-progress …" lifecycle line
        # (every experiment README carries one, right after the title) so it never
        # becomes the extracted summary. See scripts/build_site.py parse_readme_status.
        if in_fence or stripped.startswith("#") or stripped.startswith("|") or stripped.startswith("!") or stripped.startswith("- ") or stripped.lower().startswith("**status:"):
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
    return sorted(
        path.name
        for path in exp.iterdir()
        if path.is_dir() and path.name not in EXCLUDED_DIR_NAMES
    )


def tags_for(slug: str, title: str, summary: str) -> list[str]:
    haystack = f"{slug} {title} {summary}".lower()
    tags = []
    for tag, needles in TAG_KEYWORDS.items():
        if any(needle in haystack for needle in needles):
            tags.append(tag)
    return tags or ["experiment"]


def programs_for(exp_id: str, tags: list[str]) -> list[str]:
    assigned: list[str] = []
    tag_set = set(tags)
    for program in PROGRAMS_CACHE:
        seed_experiments = set(program.get("seed_experiments", []))
        seed_tags = set(program.get("seed_tags", []))
        if exp_id in seed_experiments or tag_set.intersection(seed_tags):
            assigned.append(str(program["id"]))
    return sorted(set(assigned)) or ["program_review_needed"]


def research_programs_for(exp: Path, tags: list[str]) -> list[str]:
    generated = set(programs_for(exp.name, tags))
    existing = set(metadata_list(exp / "metadata.yaml", "research_programs"))
    assigned = (generated | existing) - {"program_review_needed"}
    if assigned:
        return sorted(assigned)
    return ["program_review_needed"]


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
    lines.append("research_programs:")
    for program in record["research_programs"]:
        lines.append(f"  - {yaml_scalar(str(program))}")
    lines.extend(
        [
            "top_level_dirs:",
            *[f"  - {yaml_scalar(name)}" for name in record["top_level_dirs"]],
            "file_counts:",
            *[f"  {key.lstrip('.') if key.startswith('.') else key}: {counts[key]}" for key in sorted(counts)],
            f"total_files: {record['total_files']}",
            f"total_size_bytes: {record['total_size_bytes']}",
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
    records: list[dict[str, object]] = []
    for exp in sorted(path for path in EXPERIMENTS.iterdir() if path.is_dir()):
        primary_report = find_primary_report(exp)
        title, summary = summarize(exp, primary_report)
        counts, total_files, total_size = file_stats(exp)
        tags = tags_for(exp.name, title, summary)
        record = {
            "id": exp.name,
            "title": title,
            "source_track": source_track_for(exp),
            "path": rel(exp),
            "primary_readme": rel(exp / "README.md") if (exp / "README.md").exists() else "",
            "primary_report": rel(primary_report),
            "summary": summary,
            "tags": tags,
            "research_programs": research_programs_for(exp, tags),
            "top_level_dirs": top_level_dirs(exp),
            "file_counts": dict(counts),
            "total_files": total_files,
            "total_size_bytes": total_size,
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
        "research_programs",
        "summary",
        "path",
        "primary_readme",
        "primary_report",
        "total_files",
        "total_size_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {}
            for key in fieldnames:
                row[key] = ";".join(record[key]) if key in {"tags", "research_programs"} else record[key]
            writer.writerow(row)


def md_link(label: str, target: str) -> str:
    return f"[{label}](../{target})" if target else ""


def write_catalog_md(records: list[dict[str, object]]) -> None:
    path = KNOWLEDGE / "experiment_catalog.md"
    track_counts = Counter(str(record["source_track"]) for record in records)
    lines = [
        "# Experiment Catalog",
        "",
        "Generated from `experiments/` by `scripts/build_knowledgebase.py`.",
        "",
        f"- Experiments: {len(records)}",
        f"- Source track Y provenance: {track_counts['track-y']}",
        f"- Source track Z provenance: {track_counts['track-z']}",
        "",
        "| Programs | Track | Experiment | Tags | Summary | Read | Report |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        summary = str(record["summary"]).replace("|", "\\|")
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        tags = ", ".join(record["tags"])
        programs = ", ".join(record["research_programs"])
        lines.append(
            "| {programs} | {track} | `{id}` | {tags} | {summary} | {readme} | {report} |".format(
                programs=programs,
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


def write_program_index(records: list[dict[str, object]]) -> None:
    by_program: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        for program_id in record["research_programs"]:
            by_program[str(program_id)].append(record)

    lines = [
        "# Research Program Index",
        "",
        "Generated from `research_programs/registry.yaml` and experiment metadata. The imported tracks are seed evidence for durable future research programs, not the boundary of the repository.",
        "",
        f"- Programs: {len(PROGRAMS_CACHE)}",
        f"- Experiments: {len(records)}",
        "",
    ]
    csv_rows: list[dict[str, str]] = []
    for program in PROGRAMS_CACHE:
        program_id = str(program["id"])
        program_records = sorted(by_program.get(program_id, []), key=lambda item: str(item["id"]))
        lines.extend(
            [
                f"## {program['title']}",
                "",
                str(program["focus"]),
                "",
                f"- Charter: [{program_id}](../{program['charter']})",
                f"- Assigned experiments: {len(program_records)}",
                "",
            ]
        )
        seed_ids = set(program.get("seed_experiments", []))
        if seed_ids:
            lines.extend(["### Seed Evidence", ""])
            for exp_id in sorted(seed_ids):
                record = next((candidate for candidate in records if candidate["id"] == exp_id), None)
                if record:
                    lines.append(f"- `{exp_id}`: {md_link('README', str(record['primary_readme']))}")
                else:
                    lines.append(f"- `{exp_id}`: not present in `experiments/`")
            lines.append("")
        lines.extend(["### Assigned Experiments", ""])
        for record in program_records:
            lines.append(
                f"- `{record['id']}` ({record['source_track']}): {md_link('README', str(record['primary_readme']))}"
            )
            csv_rows.append(
                {
                    "program_id": program_id,
                    "program_title": str(program["title"]),
                    "experiment_id": str(record["id"]),
                    "source_track": str(record["source_track"]),
                    "primary_readme": str(record["primary_readme"]),
                    "primary_report": str(record["primary_report"]),
                }
            )
        lines.append("")

    if by_program.get("program_review_needed"):
        lines.extend(["## Program Review Needed", ""])
        for record in by_program["program_review_needed"]:
            lines.append(f"- `{record['id']}`: {md_link('README', str(record['primary_readme']))}")
        lines.append("")

    (KNOWLEDGE / "research_program_index.md").write_text("\n".join(lines), encoding="utf-8")
    with (KNOWLEDGE / "research_program_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["program_id", "program_title", "experiment_id", "source_track", "primary_readme", "primary_report"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)


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


def manifest_kind(path: Path) -> str:
    name = path.name
    if name == "artifact_manifest.yaml":
        return "standard-artifact"
    if name == "large_artifacts_manifest.md":
        return "large-artifact"
    if name == "checkpoint_manifest.csv":
        return "checkpoint"
    if name == "dataset_manifest.json":
        return "dataset"
    if name.endswith(".manifest.json"):
        return "run-data"
    if "manifest" in name:
        return "other"
    return "unknown"


def is_artifact_manifest_file(path: Path) -> bool:
    name = path.name
    if name in {"artifact_manifest.yaml", "large_artifacts_manifest.md", "checkpoint_manifest.csv", "split_manifest.json"}:
        return True
    if name.endswith(".manifest.json") or name.endswith("_manifest.json"):
        return True
    return False


def artifact_manifest_rows(records: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    record_ids = {str(record["id"]) for record in records}
    for exp in sorted(path for path in EXPERIMENTS.iterdir() if path.is_dir() and path.name in record_ids):
        for path in sorted(exp.rglob("*")):
            if not path.is_file() or excluded(path.relative_to(ROOT)):
                continue
            if not is_artifact_manifest_file(path):
                continue
            rows.append(
                {
                    "experiment_id": exp.name,
                    "kind": manifest_kind(path),
                    "path": rel(path),
                }
            )
    return rows


def readme_status(record: dict[str, object]) -> str:
    readme = ROOT / str(record["primary_readme"]) if record["primary_readme"] else None
    if not readme or not readme.exists():
        return "missing"
    marker = "generated during repository normalization"
    return "generated-stub" if marker in read_text(readme, 1000).lower() else "human-authored"


def experiment_docs(record: dict[str, object]) -> str:
    paths = []
    if record["primary_readme"]:
        paths.append(ROOT / str(record["primary_readme"]))
    if record["primary_report"]:
        paths.append(ROOT / str(record["primary_report"]))
    return "\n".join(read_text(path, 20000) for path in paths)


def experiment_script_paths(exp: Path) -> list[Path]:
    paths: list[Path] = []
    scripts = exp / "scripts"
    if scripts.exists():
        paths.extend(path for path in scripts.rglob("*") if path.is_file() and path.suffix in SCRIPT_EXTENSIONS)
    for name in ["Makefile", "run.sh", "train.py", "eval.py", "evaluate.py"]:
        path = exp / name
        if path.exists() and path.is_file():
            paths.append(path)
    return sorted(set(paths))


def has_command_signal(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in COMMAND_PATTERNS)


def has_smoke_signal(record: dict[str, object], scripts: list[Path]) -> bool:
    haystack = experiment_docs(record).lower()
    if "smoke" in haystack or "--smoke" in haystack:
        return True
    for path in scripts[:25]:
        if "smoke" in path.name.lower():
            return True
        if "smoke" in read_text(path, 12000).lower():
            return True
    return False


def run_surface(record: dict[str, object], scripts: list[Path]) -> str:
    exp = ROOT / str(record["path"])
    docs = experiment_docs(record)
    documented = has_command_signal(docs)
    if documented and scripts:
        return "documented-scripts"
    if documented:
        return "documented-command"
    if scripts:
        return "scripts-undocumented"
    if any((exp / dirname).exists() for dirname in ["src", "analysis"]):
        return "source-or-analysis"
    if any((exp / dirname).exists() for dirname in ["data", "runs", "reports"]):
        return "artifact-only"
    return "unknown"


def recognized_artifacts(exp: Path) -> list[str]:
    names = [name for name in ["src", "scripts", "configs", "data", "runs", "analysis", "reports"] if (exp / name).exists()]
    names.extend(name for name in ["experiment_log.md", "checkpoint_manifest.csv"] if (exp / name).exists())
    return names


def readiness_rows(records: list[dict[str, object]]) -> list[dict[str, str]]:
    manifest_rows = artifact_manifest_rows(records)
    manifest_by_experiment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        manifest_by_experiment[row["experiment_id"]].append(row)

    rows: list[dict[str, str]] = []
    for record in records:
        exp = ROOT / str(record["path"])
        exp_id = str(record["id"])
        scripts = experiment_script_paths(exp)
        status = readme_status(record)
        report = bool(record["primary_report"])
        log = (exp / "experiment_log.md").exists()
        artifact_names = recognized_artifacts(exp)
        manifests = manifest_by_experiment.get(exp_id, [])
        manifest_kinds = sorted(set(row["kind"] for row in manifests))
        surface = run_surface(record, scripts)
        smoke = has_smoke_signal(record, scripts)
        programs = [str(program) for program in record["research_programs"]]

        needs: list[str] = []
        if status == "missing":
            needs.append("add-readme")
        elif status == "generated-stub":
            needs.append("replace-generated-readme")
        if not report:
            needs.append("add-primary-report")
        if not log:
            needs.append("add-experiment-log")
        if surface in {"scripts-undocumented", "source-or-analysis", "artifact-only", "unknown"}:
            needs.append("document-run-path")
        if not smoke:
            needs.append("add-smoke-command")
        if not manifests:
            needs.append("add-artifact-manifest")
        if "program_review_needed" in programs:
            needs.append("review-program-assignment")

        anchor_ready = status == "human-authored" and report and bool(artifact_names) and "program_review_needed" not in programs

        rows.append(
            {
                "id": exp_id,
                "title": str(record["title"]),
                "source_track": str(record["source_track"]),
                "research_programs": ";".join(programs),
                "readme_status": status,
                "primary_report": str(record["primary_report"]),
                "experiment_log": yes_no(log),
                "run_surface": surface,
                "smoke_command": yes_no(smoke),
                "artifact_manifest": yes_no(bool(manifests)),
                "manifest_kinds": ";".join(manifest_kinds),
                "recognized_artifacts": ";".join(artifact_names),
                "anchor_ready": yes_no(anchor_ready),
                "needs": ";".join(needs) if needs else "none",
            }
        )
    return rows


def write_experiment_readiness(records: list[dict[str, object]]) -> None:
    rows = readiness_rows(records)
    fieldnames = [
        "id",
        "title",
        "source_track",
        "research_programs",
        "readme_status",
        "primary_report",
        "experiment_log",
        "run_surface",
        "smoke_command",
        "artifact_manifest",
        "manifest_kinds",
        "recognized_artifacts",
        "anchor_ready",
        "needs",
    ]
    with (KNOWLEDGE / "experiment_readiness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    readme_counts = Counter(row["readme_status"] for row in rows)
    run_counts = Counter(row["run_surface"] for row in rows)
    need_counts: Counter[str] = Counter()
    for row in rows:
        if row["needs"] != "none":
            need_counts.update(row["needs"].split(";"))

    lines = [
        "# Experiment Readiness Matrix",
        "",
        "Generated from tracked experiment contents. Use this as a triage surface for turning individual experiment folders into reusable anchors for future research programs.",
        "",
        "Anchor-ready means the experiment has a human-authored README, a detected primary report, recognized local artifacts, and an assigned research program. It does not mean the result is correct or final.",
        "",
        f"- Experiments: {len(rows)}",
        f"- Anchor-ready: {sum(1 for row in rows if row['anchor_ready'] == 'yes')}",
        f"- Human-authored READMEs: {readme_counts['human-authored']}",
        f"- Generated README stubs: {readme_counts['generated-stub']}",
        f"- Primary reports: {sum(1 for row in rows if row['primary_report'])}",
        f"- Experiment logs: {sum(1 for row in rows if row['experiment_log'] == 'yes')}",
        f"- Smoke commands: {sum(1 for row in rows if row['smoke_command'] == 'yes')}",
        f"- Artifact manifests: {sum(1 for row in rows if row['artifact_manifest'] == 'yes')}",
        "",
        "## Run Surface Counts",
        "",
        "| Run surface | Experiments |",
        "| --- | ---: |",
    ]
    for surface, count in sorted(run_counts.items()):
        lines.append(f"| `{surface}` | {count} |")

    lines.extend(["", "## Curation Needs", "", "| Need | Experiments |", "| --- | ---: |"])
    for need, count in sorted(need_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{need}` | {count} |")
    if not need_counts:
        lines.append("| `none` | 0 |")

    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| Ready | Experiment | Programs | README | Report | Log | Run surface | Smoke | Manifests | Needs |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        programs = row["research_programs"].replace(";", ", ")
        report = md_link("report", row["primary_report"]) if row["primary_report"] else ""
        manifests = row["manifest_kinds"].replace(";", ", ") if row["manifest_kinds"] else ""
        needs = row["needs"].replace(";", ", ")
        lines.append(
            "| {ready} | {experiment} | {programs} | `{readme}` | {report} | {log} | `{surface}` | {smoke} | {manifests} | {needs} |".format(
                ready=row["anchor_ready"],
                experiment=md_link(f"`{row['id']}`", f"experiments/{row['id']}/README.md"),
                programs=programs,
                readme=row["readme_status"],
                report=report,
                log=row["experiment_log"],
                surface=row["run_surface"],
                smoke=row["smoke_command"],
                manifests=manifests,
                needs=needs,
            )
        )

    (KNOWLEDGE / "experiment_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact_manifest_index(records: list[dict[str, object]]) -> None:
    rows = artifact_manifest_rows(records)
    by_kind = Counter(row["kind"] for row in rows)
    by_experiment = Counter(row["experiment_id"] for row in rows)
    lines = [
        "# Artifact Manifest Index",
        "",
        "Generated from manifest-like files under `experiments/`. Each experiment remains the source of truth for its own artifacts.",
        "",
        f"- Experiments with manifests: {len(by_experiment)}",
        f"- Manifest files: {len(rows)}",
        "",
        "## Manifest Types",
        "",
        "| Type | Files |",
        "| --- | ---: |",
    ]
    for kind, count in sorted(by_kind.items()):
        lines.append(f"| `{kind}` | {count} |")
    lines.extend(["", "## Manifests", "", "| Experiment | Type | Manifest |", "| --- | --- | --- |"])
    for row in rows:
        lines.append(f"| `{row['experiment_id']}` | `{row['kind']}` | {md_link('manifest', row['path'])} |")
    (KNOWLEDGE / "artifact_manifest_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (KNOWLEDGE / "artifact_manifest_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "kind", "path"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
        writer = csv.DictWriter(handle, fieldnames=["id", "source_track", "path"], lineterminator="\n")
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


def load_claims() -> list[dict[str, object]]:
    if not CLAIM_LEDGER.exists():
        return []
    data = json.loads(CLAIM_LEDGER.read_text(encoding="utf-8"))
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise SystemExit("knowledge/claims/claim_ledger.json must contain a claims list")
    return claims


def claim_link(label: str, target: str) -> str:
    return f"[{label}](../../{target})" if target else label


def evidence_label(evidence: dict[str, object], record_by_id: dict[str, dict[str, object]]) -> str:
    kind = str(evidence.get("kind", ""))
    if kind == "experiment":
        exp_id = str(evidence.get("id", ""))
        record = record_by_id.get(exp_id)
        target = str(record.get("primary_report") or record.get("primary_readme")) if record else ""
        return claim_link(f"`{exp_id}`", target)
    if kind == "program":
        program_id = str(evidence.get("id", ""))
        target = str(evidence.get("path") or f"research_programs/{program_id}/charter.md")
        return claim_link(f"`{program_id}`", target)
    target = str(evidence.get("path", ""))
    return claim_link(f"`{target}`", target)


def write_claim_index(records: list[dict[str, object]]) -> None:
    claims = load_claims()
    record_by_id = {str(record["id"]): record for record in records}
    by_status = Counter(str(claim.get("status", "")) for claim in claims)
    by_program: Counter[str] = Counter()
    for claim in claims:
        by_program.update(str(program_id) for program_id in claim.get("programs", []))

    lines = [
        "# Claim Index",
        "",
        "Generated from `knowledge/claims/claim_ledger.json`. Edit the ledger, not this file.",
        "",
        f"- Claims: {len(claims)}",
        "",
        "## Status Counts",
        "",
        "| Status | Claims |",
        "| --- | ---: |",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Program Counts", "", "| Program | Claims |", "| --- | ---: |"])
    for program_id, count in sorted(by_program.items()):
        lines.append(f"| `{program_id}` | {count} |")
    lines.append("")

    csv_rows: list[dict[str, str]] = []
    for claim in claims:
        claim_id = str(claim.get("id", ""))
        title = str(claim.get("title", ""))
        status = str(claim.get("status", ""))
        programs = [str(program_id) for program_id in claim.get("programs", [])]
        evidence_items = [item for item in claim.get("evidence", []) if isinstance(item, dict)]
        lines.extend(
            [
                f"## {claim_id}: {title}",
                "",
                f"- Status: `{status}`",
                f"- Programs: {', '.join(f'`{program_id}`' for program_id in programs)}",
                f"- Summary: {claim.get('summary', '')}",
                f"- Implication: {claim.get('implication', '')}",
                "",
                "### Evidence",
                "",
            ]
        )
        for evidence in evidence_items:
            lines.append(f"- {evidence_label(evidence, record_by_id)}")
        lines.extend(["", "### Next Tests", ""])
        for test in claim.get("next_tests", []):
            lines.append(f"- {test}")
        lines.extend(["", "### Avoid", ""])
        for item in claim.get("avoid", []):
            lines.append(f"- {item}")
        lines.append("")

        csv_rows.append(
            {
                "id": claim_id,
                "title": title,
                "status": status,
                "programs": ";".join(programs),
                "evidence": ";".join(
                    str(evidence.get("id") or evidence.get("path", "")) for evidence in evidence_items
                ),
                "summary": str(claim.get("summary", "")),
                "implication": str(claim.get("implication", "")),
            }
        )

    (CLAIMS / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    with (CLAIMS / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "title", "status", "programs", "evidence", "summary", "implication"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)


def load_future_queue() -> dict[str, object]:
    if not FUTURE_QUEUE.exists():
        return {"candidate_programs": [], "proposals": []}
    data = json.loads(FUTURE_QUEUE.read_text(encoding="utf-8"))
    proposals = data.get("proposals", [])
    if not isinstance(proposals, list):
        raise SystemExit("knowledge/future_experiment_queue.json must contain a proposals list")
    candidates = data.get("candidate_programs", [])
    if not isinstance(candidates, list):
        raise SystemExit("knowledge/future_experiment_queue.json must contain a candidate_programs list")
    return data


def program_title_map(candidate_programs: list[dict[str, object]]) -> dict[str, str]:
    titles = {str(program["id"]): str(program["title"]) for program in PROGRAMS_CACHE}
    for candidate in candidate_programs:
        titles[str(candidate.get("id", ""))] = str(candidate.get("title", ""))
    return titles


def program_queue_link(program_id: str, title: str, candidate_ids: set[str]) -> str:
    if program_id in candidate_ids:
        return f"`{program_id}`"
    return md_link(title or program_id, f"research_programs/{program_id}/charter.md")


def write_future_experiment_queue() -> None:
    data = load_future_queue()
    candidate_programs = [item for item in data.get("candidate_programs", []) if isinstance(item, dict)]
    proposals = [item for item in data.get("proposals", []) if isinstance(item, dict)]
    candidate_ids = {str(item.get("id", "")) for item in candidate_programs}
    titles = program_title_map(candidate_programs)
    by_status = Counter(str(item.get("status", "")) for item in proposals)
    by_priority = Counter(str(item.get("priority", "")) for item in proposals)
    by_program: dict[str, list[dict[str, object]]] = defaultdict(list)
    for proposal in proposals:
        for program_id in proposal.get("programs", []):
            by_program[str(program_id)].append(proposal)

    existing_ids = {str(program["id"]) for program in PROGRAMS_CACHE}
    existing_covered = existing_ids.intersection(by_program)
    lines = [
        "# Future Experiment Queue",
        "",
        "Generated from `knowledge/future_experiment_queue.json`. Edit the JSON source, not this file.",
        "",
        "This queue is intentionally broader than the imported prototype corpus. It is a launchpad for future experiments, candidate programs, infrastructure work, and falsifiable probes.",
        "",
        f"- Proposals: {len(proposals)}",
        f"- Existing research programs covered: {len(existing_covered)} / {len(existing_ids)}",
        f"- Candidate program lines: {len(candidate_programs)}",
        "",
        "## Status Counts",
        "",
        "| Status | Proposals |",
        "| --- | ---: |",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"| `{status}` | {count} |")

    lines.extend(["", "## Priority Counts", "", "| Priority | Proposals |", "| --- | ---: |"])
    for priority, count in sorted(by_priority.items()):
        lines.append(f"| `{priority}` | {count} |")

    lines.extend(["", "## Candidate Program Lines", ""])
    for candidate in candidate_programs:
        lines.extend(
            [
                f"### {candidate.get('title', candidate.get('id', ''))}",
                "",
                f"- Candidate id: `{candidate.get('id', '')}`",
                f"- Focus: {candidate.get('focus', '')}",
                "",
            ]
        )

    lines.extend(["## By Program", ""])
    program_order = [str(program["id"]) for program in PROGRAMS_CACHE] + sorted(candidate_ids)
    for program_id in program_order:
        program_proposals = sorted(by_program.get(program_id, []), key=lambda item: str(item.get("priority", "")))
        if not program_proposals:
            continue
        title = titles.get(program_id, program_id)
        lines.extend([f"### {title}", "", f"- Proposals: {len(program_proposals)}", ""])
        for proposal in program_proposals:
            lines.append(
                f"- `{proposal.get('id', '')}` (`{proposal.get('priority', '')}`, `{proposal.get('status', '')}`): {proposal.get('question', '')}"
            )
        lines.append("")

    lines.extend(
        [
            "## Queue",
            "",
            "| Priority | Status | Effort | Proposal | Programs | Question | Next step | Source |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for proposal in proposals:
        programs = ", ".join(
            program_queue_link(program_id, titles.get(program_id, program_id), candidate_ids)
            for program_id in [str(item) for item in proposal.get("programs", [])]
        )
        question = str(proposal.get("question", "")).replace("|", "\\|")
        next_step = str(proposal.get("next_step", "")).replace("|", "\\|")
        source = md_link("source", str(proposal.get("source", "")))
        lines.append(
            "| {priority} | {status} | {effort} | `{id}` | {programs} | {question} | {next_step} | {source} |".format(
                priority=proposal.get("priority", ""),
                status=proposal.get("status", ""),
                effort=proposal.get("effort", ""),
                id=proposal.get("id", ""),
                programs=programs,
                question=question,
                next_step=next_step,
                source=source,
            )
        )

    (KNOWLEDGE / "future_experiment_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (KNOWLEDGE / "future_experiment_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "id",
            "title",
            "status",
            "priority",
            "effort",
            "programs",
            "question",
            "hypothesis",
            "success_signal",
            "failure_signal",
            "next_step",
            "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for proposal in proposals:
            writer.writerow(
                {
                    "id": proposal.get("id", ""),
                    "title": proposal.get("title", ""),
                    "status": proposal.get("status", ""),
                    "priority": proposal.get("priority", ""),
                    "effort": proposal.get("effort", ""),
                    "programs": ";".join(str(item) for item in proposal.get("programs", [])),
                    "question": proposal.get("question", ""),
                    "hypothesis": proposal.get("hypothesis", ""),
                    "success_signal": proposal.get("success_signal", ""),
                    "failure_signal": proposal.get("failure_signal", ""),
                    "next_step": proposal.get("next_step", ""),
                    "source": proposal.get("source", ""),
                }
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
    write_program_index(records)
    write_artifact_index(records)
    write_artifact_manifest_index(records)
    write_experiment_readiness(records)
    write_source_tracks(records)
    write_json_manifest(records)
    write_claim_index(records)
    write_future_experiment_queue()
    write_readme_gap_report(records, generated_readmes)
    print(f"indexed {len(records)} experiments")
    if generated_readmes:
        print(f"generated {len(generated_readmes)} README stubs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate repository organization invariants."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
KNOWLEDGE = ROOT / "knowledge"
PROGRAMS = ROOT / "research_programs"
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
MIN_RESEARCH_PROGRAMS = 8


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def metadata_value(path: Path, key: str) -> str:
    prefix = f"{key}: "
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value
    return ""


def metadata_list(path: Path, key: str) -> list[str]:
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


def registry_program_ids() -> list[str]:
    registry = PROGRAMS / "registry.yaml"
    if not registry.exists():
        return []
    ids: list[str] = []
    for line in registry.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            ids.append(stripped.split(":", 1)[1].strip().strip('"'))
    return ids


def validate() -> int:
    errors: list[str] = []

    if not EXPERIMENTS.exists():
        fail(errors, "missing experiments/")
        return report(errors)
    if (ROOT / "tracks").exists():
        fail(errors, "legacy tracks/ directory still exists")

    program_ids = registry_program_ids()
    if len(program_ids) < MIN_RESEARCH_PROGRAMS:
        fail(errors, f"research program registry has {len(program_ids)} programs; expected at least {MIN_RESEARCH_PROGRAMS}")
    if len(set(program_ids)) != len(program_ids):
        fail(errors, "research program registry contains duplicate ids")
    for program_id in program_ids:
        program_dir = PROGRAMS / program_id
        for required in ["charter.md", "backlog.md", "evidence.md"]:
            if not (program_dir / required).exists():
                fail(errors, f"missing research program file: {rel(program_dir / required)}")

    for required in [
        PROGRAMS / "README.md",
        PROGRAMS / "registry.yaml",
        KNOWLEDGE / "research_program_index.md",
        KNOWLEDGE / "research_program_index.csv",
        ROOT / "scripts" / "scaffold_research_program.py",
        ROOT / "scripts" / "scaffold_experiment.py",
        ROOT / "scripts" / "check_markdown_links.py",
        ROOT / "scripts" / "check_python_syntax.py",
        ROOT / "scripts" / "check_repository_text.py",
        ROOT / ".github" / "workflows" / "validate.yml",
        ROOT / "templates" / "research_program" / "charter.md",
        ROOT / "templates" / "research_program" / "backlog.md",
        ROOT / "templates" / "research_program" / "evidence.md",
        ROOT / "templates" / "experiment" / "README.md",
        ROOT / "templates" / "experiment" / "metadata.yaml",
    ]:
        if not required.exists():
            fail(errors, f"missing research-program scaffold file: {rel(required)}")

    experiments = sorted(path for path in EXPERIMENTS.iterdir() if path.is_dir())
    if not experiments:
        fail(errors, "experiments/ contains no experiment directories")

    for exp in experiments:
        readme = exp / "README.md"
        metadata = exp / "metadata.yaml"
        if not readme.exists():
            fail(errors, f"missing README: {rel(readme)}")
        if not metadata.exists():
            fail(errors, f"missing metadata: {rel(metadata)}")
        else:
            meta_id = metadata_value(metadata, "id")
            meta_path = metadata_value(metadata, "path")
            meta_source_track = metadata_value(metadata, "source_track")
            meta_programs = metadata_list(metadata, "research_programs")
            if meta_id != exp.name:
                fail(errors, f"metadata id mismatch in {rel(metadata)}: {meta_id!r} != {exp.name!r}")
            if meta_path != rel(exp):
                fail(errors, f"metadata path mismatch in {rel(metadata)}: {meta_path!r} != {rel(exp)!r}")
            if not meta_source_track:
                fail(errors, f"metadata missing source_track: {rel(metadata)}")
            if not meta_programs:
                fail(errors, f"metadata missing research_programs: {rel(metadata)}")
            unknown_programs = sorted(set(meta_programs) - set(program_ids) - {"program_review_needed"})
            if unknown_programs:
                fail(errors, f"metadata references unknown research programs in {rel(metadata)}: {', '.join(unknown_programs)}")
        useful_dirs = [name for name in ["src", "scripts", "data", "runs", "analysis", "reports"] if (exp / name).exists()]
        useful_files = [name for name in ["experiment_log.md", "checkpoint_manifest.csv"] if (exp / name).exists()]
        if not useful_dirs and not useful_files:
            fail(errors, f"experiment has no recognized artifacts: {rel(exp)}")

    for pattern in ["*:Zone.Identifier", "*.pyc"]:
        for path in ROOT.rglob(pattern):
            if ".git" not in path.parts:
                fail(errors, f"generated/copy artifact should not be tracked: {rel(path)}")
    for path in ROOT.rglob("__pycache__"):
        if ".git" not in path.parts:
            fail(errors, f"python cache directory should not be present: {rel(path)}")

    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.stat().st_size > MAX_GITHUB_FILE_BYTES:
            fail(errors, f"file exceeds GitHub hard limit: {rel(path)}")

    if (ROOT / ".git").exists():
        tracked_adapters = subprocess.run(
            ["git", "ls-files", "experiments/**/reports/adapters/**"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if tracked_adapters.stdout.strip():
            fail(errors, "adapter outputs are tracked:\n" + tracked_adapters.stdout.strip())

    catalog = KNOWLEDGE / "experiment_catalog.csv"
    if not catalog.exists():
        fail(errors, "missing knowledge/experiment_catalog.csv")
    else:
        with catalog.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(experiments):
            fail(errors, f"catalog row count {len(rows)} != experiment count {len(experiments)}")
        catalog_ids = {row["id"] for row in rows}
        exp_ids = {path.name for path in experiments}
        missing = sorted(exp_ids - catalog_ids)
        extra = sorted(catalog_ids - exp_ids)
        if missing:
            fail(errors, "catalog missing experiments: " + ", ".join(missing))
        if extra:
            fail(errors, "catalog has extra experiments: " + ", ".join(extra))

    program_index = KNOWLEDGE / "research_program_index.csv"
    if program_index.exists():
        with program_index.open(newline="", encoding="utf-8") as handle:
            program_rows = list(csv.DictReader(handle))
        indexed_programs = {row["program_id"] for row in program_rows}
        missing_programs = sorted(set(program_ids) - indexed_programs)
        if missing_programs:
            fail(errors, "program index has no experiment rows for programs: " + ", ".join(missing_programs))

    gitattributes = ROOT / ".gitattributes"
    if not gitattributes.exists():
        fail(errors, "missing .gitattributes")
    else:
        content = gitattributes.read_text(encoding="utf-8", errors="replace")
        if "*.safetensors filter=lfs" not in content:
            fail(errors, ".gitattributes does not LFS-track safetensors")

    return report(errors)


def report(errors: list[str]) -> int:
    if errors:
        print("repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("repository validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(validate())

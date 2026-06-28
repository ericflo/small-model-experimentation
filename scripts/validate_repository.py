#!/usr/bin/env python3
"""Validate repository organization invariants."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
KNOWLEDGE = ROOT / "knowledge"
PROGRAMS = ROOT / "research_programs"
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
MIN_RESEARCH_PROGRAMS = 8
CLAIM_STATUSES = {"Confirmed", "Promising", "Negative", "Open", "Retired"}
ARTIFACT_MANIFEST_FIELDS = ["schema_version:", "external_artifacts:", "omitted_artifacts:", "reproducibility:"]


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


def registry_programs() -> list[dict[str, str]]:
    registry = PROGRAMS / "registry.yaml"
    if not registry.exists():
        return []
    programs: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in registry.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current:
                programs.append(current)
            current = {"id": stripped.split(":", 1)[1].strip().strip('"'), "title": ""}
        elif current and stripped.startswith("title:"):
            current["title"] = stripped.split(":", 1)[1].strip().strip('"')
    if current:
        programs.append(current)
    return programs


def registry_program_ids() -> list[str]:
    return [program["id"] for program in registry_programs()]


def validate_claim_ledger(errors: list[str], program_ids: set[str], exp_ids: set[str]) -> None:
    ledger = KNOWLEDGE / "claims" / "claim_ledger.json"
    if not ledger.exists():
        fail(errors, "missing claims ledger: knowledge/claims/claim_ledger.json")
        return
    try:
        data = json.loads(ledger.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(errors, f"claims ledger is invalid JSON: {exc}")
        return
    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        fail(errors, "claims ledger must contain a non-empty claims list")
        return

    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            fail(errors, "claims ledger contains a non-object claim")
            continue
        claim_id = str(claim.get("id", ""))
        if not claim_id:
            fail(errors, "claims ledger contains a claim without an id")
        elif claim_id in seen:
            fail(errors, f"claims ledger contains duplicate id: {claim_id}")
        seen.add(claim_id)

        status = str(claim.get("status", ""))
        if status not in CLAIM_STATUSES:
            fail(errors, f"claim {claim_id} has invalid status: {status!r}")

        programs = claim.get("programs", [])
        if not isinstance(programs, list) or not programs:
            fail(errors, f"claim {claim_id} must name at least one program")
        else:
            unknown_programs = sorted(set(str(program_id) for program_id in programs) - program_ids)
            if unknown_programs:
                fail(errors, f"claim {claim_id} references unknown programs: {', '.join(unknown_programs)}")

        evidence_items = claim.get("evidence", [])
        if not isinstance(evidence_items, list) or not evidence_items:
            fail(errors, f"claim {claim_id} must include evidence")
            continue
        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                fail(errors, f"claim {claim_id} contains non-object evidence")
                continue
            kind = str(evidence.get("kind", ""))
            if kind == "experiment":
                exp_id = str(evidence.get("id", ""))
                if exp_id not in exp_ids:
                    fail(errors, f"claim {claim_id} references missing experiment evidence: {exp_id}")
            elif kind == "program":
                program_id = str(evidence.get("id", ""))
                if program_id not in program_ids:
                    fail(errors, f"claim {claim_id} references missing program evidence: {program_id}")
            elif kind == "doc":
                path = ROOT / str(evidence.get("path", ""))
                if not path.exists():
                    fail(errors, f"claim {claim_id} references missing doc evidence: {rel(path)}")
            else:
                fail(errors, f"claim {claim_id} has unknown evidence kind: {kind!r}")

    claim_index = KNOWLEDGE / "claims" / "index.csv"
    if claim_index.exists():
        with claim_index.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        indexed_ids = {row["id"] for row in rows}
        ledger_ids = {str(claim.get("id", "")) for claim in claims if isinstance(claim, dict)}
        if indexed_ids != ledger_ids:
            fail(errors, "claim index ids do not match claim ledger ids")


def experiment_has_artifact_manifest(exp: Path) -> bool:
    candidates = [
        exp / "reports" / "artifact_manifest.yaml",
        exp / "artifact_manifest.yaml",
        exp / "large_artifacts_manifest.md",
        exp / "checkpoint_manifest.csv",
    ]
    return any(candidate.exists() for candidate in candidates)


def is_artifact_manifest_file(path: Path) -> bool:
    name = path.name
    if name in {"artifact_manifest.yaml", "large_artifacts_manifest.md", "checkpoint_manifest.csv", "split_manifest.json"}:
        return True
    if name.endswith(".manifest.json") or name.endswith("_manifest.json"):
        return True
    return False


def validate_standard_artifact_manifest(errors: list[str], path: Path, exp: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for field in ARTIFACT_MANIFEST_FIELDS:
        if field not in text:
            fail(errors, f"artifact manifest missing {field} in {rel(path)}")
    expected = f'experiment_id: "{exp.name}"'
    if "experiment_id:" in text and expected not in text:
        fail(errors, f"artifact manifest experiment_id mismatch in {rel(path)}; expected {expected}")


def validate() -> int:
    errors: list[str] = []

    if not EXPERIMENTS.exists():
        fail(errors, "missing experiments/")
        return report(errors)
    if (ROOT / "tracks").exists():
        fail(errors, "legacy tracks/ directory still exists")

    registry = registry_programs()
    program_ids = [program["id"] for program in registry]
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
        KNOWLEDGE / "program_scorecards.md",
        KNOWLEDGE / "artifact_manifest_index.md",
        KNOWLEDGE / "artifact_manifest_index.csv",
        KNOWLEDGE / "claims" / "claim_ledger.json",
        KNOWLEDGE / "claims" / "index.md",
        KNOWLEDGE / "claims" / "index.csv",
        KNOWLEDGE / "decision_records" / "README.md",
        ROOT / "docs" / "idea_intake_protocol.md",
        ROOT / "docs" / "artifact_policy.md",
        ROOT / "scripts" / "scaffold_research_program.py",
        ROOT / "scripts" / "scaffold_experiment.py",
        ROOT / "scripts" / "check_markdown_links.py",
        ROOT / "scripts" / "check_python_syntax.py",
        ROOT / "scripts" / "check_repository_text.py",
        ROOT / "scripts" / "find_related.py",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "experiment_proposal.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research_program.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "synthesis_update.yml",
        ROOT / ".github" / "pull_request_template.md",
        ROOT / ".github" / "workflows" / "validate.yml",
        ROOT / "templates" / "research_program" / "charter.md",
        ROOT / "templates" / "research_program" / "backlog.md",
        ROOT / "templates" / "research_program" / "evidence.md",
        ROOT / "templates" / "experiment" / "README.md",
        ROOT / "templates" / "experiment" / "metadata.yaml",
        ROOT / "templates" / "experiment" / "reports" / "artifact_manifest.yaml",
        ROOT / "templates" / "idea_intake.md",
        ROOT / "templates" / "decision_record.md",
    ]:
        if not required.exists():
            fail(errors, f"missing research-program scaffold file: {rel(required)}")

    scorecards = KNOWLEDGE / "program_scorecards.md"
    if scorecards.exists():
        content = scorecards.read_text(encoding="utf-8", errors="replace")
        for program in registry:
            title = program["title"]
            if title and f"## {title}" not in content:
                fail(errors, f"program scorecards missing section for {title!r}")

    experiments = sorted(path for path in EXPERIMENTS.iterdir() if path.is_dir())
    if not experiments:
        fail(errors, "experiments/ contains no experiment directories")
    exp_ids = {path.name for path in experiments}

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
            if meta_source_track == "new" and not experiment_has_artifact_manifest(exp):
                fail(errors, f"new experiment missing artifact manifest: {rel(exp)}")
            if not meta_programs:
                fail(errors, f"metadata missing research_programs: {rel(metadata)}")
            unknown_programs = sorted(set(meta_programs) - set(program_ids) - {"program_review_needed"})
            if unknown_programs:
                fail(errors, f"metadata references unknown research programs in {rel(metadata)}: {', '.join(unknown_programs)}")
        useful_dirs = [name for name in ["src", "scripts", "data", "runs", "analysis", "reports"] if (exp / name).exists()]
        useful_files = [name for name in ["experiment_log.md", "checkpoint_manifest.csv"] if (exp / name).exists()]
        if not useful_dirs and not useful_files:
            fail(errors, f"experiment has no recognized artifacts: {rel(exp)}")
        standard_manifest = exp / "reports" / "artifact_manifest.yaml"
        if standard_manifest.exists():
            validate_standard_artifact_manifest(errors, standard_manifest, exp)
        if (exp / "reports" / "adapters").exists() and not experiment_has_artifact_manifest(exp):
            fail(errors, f"adapter directory exists without an external artifact manifest: {rel(exp)}")

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

    artifact_manifest_index = KNOWLEDGE / "artifact_manifest_index.csv"
    if artifact_manifest_index.exists():
        with artifact_manifest_index.open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        indexed_paths = {row["path"] for row in manifest_rows}
        expected_paths = set()
        for exp in experiments:
            for path in exp.rglob("*"):
                if path.is_file() and ".git" not in path.parts and is_artifact_manifest_file(path):
                    expected_paths.add(rel(path))
            standard = exp / "reports" / "artifact_manifest.yaml"
            if standard.exists():
                expected_paths.add(rel(standard))
        missing_manifest_paths = sorted(expected_paths - indexed_paths)
        if missing_manifest_paths:
            fail(errors, "artifact manifest index missing paths: " + ", ".join(missing_manifest_paths[:20]))

    validate_claim_ledger(errors, set(program_ids), exp_ids)

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

#!/usr/bin/env python3
"""Validate repository organization invariants."""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
KNOWLEDGE = ROOT / "knowledge"
PROGRAMS = ROOT / "research_programs"
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
# Working-tree directories that are gitignored / ephemeral and must never be scanned
# for "tracked" hygiene checks (large files, stray caches): they cannot be committed.
IGNORED_DIRS = {
    ".git", ".venv", ".venv-vllm", "venv", "large_artifacts", "site", "node_modules",
    ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache", ".mypy_cache",
}
MIN_RESEARCH_PROGRAMS = 8
MIN_FUTURE_PROPOSALS = 24
CLAIM_STATUSES = {"Confirmed", "Promising", "Negative", "Open", "Retired"}
QUEUE_STATUSES = {"ready-for-intake", "program-seed", "needs-design", "infrastructure"}
QUEUE_PRIORITIES = {"P0", "P1", "P2"}
QUEUE_EFFORTS = {"small", "medium", "large"}
# How long an experiment may stay flagged in-progress before CI forces a review.
# Runs here conclude within days; a genuinely long one just re-affirms `since`.
# This is the anti-footgun: nothing can silently rot stuck in "in progress".
STALE_INPROGRESS_DAYS = 45
ARTIFACT_MANIFEST_FIELDS = ["schema_version:", "external_artifacts:", "omitted_artifacts:", "reproducibility:"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def in_ignored_dir(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


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

        for field in ("next_tests", "avoid"):
            values = claim.get(field, [])
            if not isinstance(values, list):
                fail(errors, f"claim {claim_id} field {field} must be a list")
            elif any(not str(value).strip() for value in values):
                fail(errors, f"claim {claim_id} field {field} contains an empty item")

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


def validate_future_queue(errors: list[str], program_ids: set[str]) -> None:
    source = KNOWLEDGE / "future_experiment_queue.json"
    if not source.exists():
        fail(errors, "missing future experiment queue: knowledge/future_experiment_queue.json")
        return
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(errors, f"future experiment queue is invalid JSON: {exc}")
        return

    candidate_programs = data.get("candidate_programs", [])
    proposals = data.get("proposals", [])
    if not isinstance(candidate_programs, list) or not candidate_programs:
        fail(errors, "future experiment queue must contain candidate_programs")
        candidate_programs = []
    if not isinstance(proposals, list) or len(proposals) < MIN_FUTURE_PROPOSALS:
        count = len(proposals) if isinstance(proposals, list) else 0
        fail(errors, f"future experiment queue has {count} proposals; expected at least {MIN_FUTURE_PROPOSALS}")
        proposals = []

    candidate_ids: set[str] = set()
    for candidate in candidate_programs:
        if not isinstance(candidate, dict):
            fail(errors, "future experiment queue contains non-object candidate program")
            continue
        candidate_id = str(candidate.get("id", ""))
        if not candidate_id:
            fail(errors, "future experiment queue contains candidate program without id")
        elif candidate_id in candidate_ids:
            fail(errors, f"future experiment queue contains duplicate candidate program id: {candidate_id}")
        candidate_ids.add(candidate_id)
        for field in ["title", "focus"]:
            if not str(candidate.get(field, "")).strip():
                fail(errors, f"future queue candidate {candidate_id} missing {field}")

    required_fields = {
        "id",
        "title",
        "status",
        "priority",
        "effort",
        "programs",
        "question",
        "hypothesis",
        "minimal_protocol",
        "success_signal",
        "failure_signal",
        "expected_artifacts",
        "next_step",
        "avoid",
        "source",
    }
    seen: set[str] = set()
    referenced_existing_programs: set[str] = set()
    known_programs = program_ids | candidate_ids
    for proposal in proposals:
        if not isinstance(proposal, dict):
            fail(errors, "future experiment queue contains non-object proposal")
            continue
        proposal_id = str(proposal.get("id", ""))
        if not proposal_id:
            fail(errors, "future experiment queue contains proposal without id")
        elif proposal_id in seen:
            fail(errors, f"future experiment queue contains duplicate proposal id: {proposal_id}")
        seen.add(proposal_id)

        missing_fields = sorted(field for field in required_fields if not proposal.get(field))
        if missing_fields:
            fail(errors, f"future queue proposal {proposal_id} missing fields: {', '.join(missing_fields)}")

        status = str(proposal.get("status", ""))
        if status not in QUEUE_STATUSES:
            fail(errors, f"future queue proposal {proposal_id} has invalid status: {status!r}")
        priority = str(proposal.get("priority", ""))
        if priority not in QUEUE_PRIORITIES:
            fail(errors, f"future queue proposal {proposal_id} has invalid priority: {priority!r}")
        effort = str(proposal.get("effort", ""))
        if effort not in QUEUE_EFFORTS:
            fail(errors, f"future queue proposal {proposal_id} has invalid effort: {effort!r}")

        programs = proposal.get("programs", [])
        if not isinstance(programs, list) or not programs:
            fail(errors, f"future queue proposal {proposal_id} must reference at least one program")
        else:
            unknown = sorted(set(str(program_id) for program_id in programs) - known_programs)
            if unknown:
                fail(errors, f"future queue proposal {proposal_id} references unknown programs: {', '.join(unknown)}")
            referenced_existing_programs.update(str(program_id) for program_id in programs if str(program_id) in program_ids)

        artifacts = proposal.get("expected_artifacts", [])
        if not isinstance(artifacts, list) or not artifacts:
            fail(errors, f"future queue proposal {proposal_id} must list expected artifacts")

        source_path = ROOT / str(proposal.get("source", ""))
        if not source_path.exists():
            fail(errors, f"future queue proposal {proposal_id} references missing source: {rel(source_path)}")

    missing_programs = sorted(program_ids - referenced_existing_programs)
    if missing_programs:
        fail(errors, "future experiment queue has no proposal for programs: " + ", ".join(missing_programs))

    queue_csv = KNOWLEDGE / "future_experiment_queue.csv"
    if queue_csv.exists():
        with queue_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(proposals):
            fail(errors, f"future queue csv row count {len(rows)} != proposal count {len(proposals)}")
        indexed_ids = {row["id"] for row in rows}
        if indexed_ids != seen:
            fail(errors, "future queue csv ids do not match source proposal ids")


def experiment_has_artifact_manifest(exp: Path) -> bool:
    candidates = [
        exp / "reports" / "artifact_manifest.yaml",
        exp / "artifact_manifest.yaml",
        exp / "large_artifacts_manifest.md",
        exp / "checkpoint_manifest.csv",
    ]
    return any(candidate.exists() for candidate in candidates)


def _valid_chart_spec(spec: object) -> bool:
    """Mirror of build_site._valid_spec: a spec renders as a native site chart iff it is a bar/line with
    well-formed series. Kept in sync deliberately (a chart that won't render must not satisfy the gate)."""
    if not isinstance(spec, dict) or spec.get("kind") not in {"bar", "line"}:
        return False
    series = spec.get("series")
    if not isinstance(series, list) or not series:
        return False
    if spec["kind"] == "bar":
        cats = spec.get("categories")
        if not isinstance(cats, list) or not cats:
            return False
        for entry in series:
            values = entry.get("values") if isinstance(entry, dict) else None
            if not isinstance(values, list) or len(values) != len(cats):
                return False
            if not all(isinstance(v, (int, float)) for v in values):
                return False
    else:
        for entry in series:
            points = entry.get("points") if isinstance(entry, dict) else None
            if not isinstance(points, list) or not points:
                return False
            if not all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in points):
                return False
    return True


def experiment_chart_count(exp_id: str, viz: dict) -> int:
    """Number of valid native chart specs registered for an experiment in experiment_viz.json."""
    entry = viz.get(exp_id) if isinstance(viz, dict) else None
    charts = entry.get("charts", []) if isinstance(entry, dict) else []
    return sum(1 for spec in charts if _valid_chart_spec(spec))


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


def validate_root_readme(errors: list[str]) -> None:
    # The root README is principles-only by policy (owner decision, 2026-07-08):
    # findings, per-claim narratives, and corpus counts go stale with every
    # pipeline commit and belong on the generated site, which is always current.
    readme = ROOT / "README.md"
    if not readme.exists():
        fail(errors, "missing README.md")
        return
    text = readme.read_text(encoding="utf-8", errors="replace")
    claim_anchors = sorted(set(re.findall(r"claims/#c\d+", text, flags=re.IGNORECASE)))
    if claim_anchors:
        fail(errors, "root README references specific claim anchors (" + ", ".join(claim_anchors[:5]) +
                     "): the README is principles-only — link the live claim ledger instead of individual claims")
    counts = re.findall(r"\*\*\d[\d,]*\*\*\s*(?:result |evidence-linked )?(?:experiments?|research programs?|programs?|claims?|charts?)",
                        text, flags=re.IGNORECASE)
    if counts:
        fail(errors, "root README hardcodes corpus counts (" + "; ".join(counts[:5]) +
                     "): counts drift with every pipeline commit — describe without numbers; the site carries live counts")


def validate_benchmark_firewall(errors: list[str]) -> None:
    if not (ROOT / "benchmarks").exists():
        return
    forbidden_patterns = [
        "menagerie/families",
        "benchmarks.menagerie",
        "benchmarks/menagerie",
        "from families",
        "import families",
    ]
    for path in EXPERIMENTS.rglob("*.py"):
        if in_ignored_dir(path):
            continue
        experiment_root = next((parent for parent in path.parents if parent.parent == EXPERIMENTS), None)
        text = path.read_text(encoding="utf-8", errors="replace")
        experiment_src = experiment_root / "src" if experiment_root is not None else None
        has_local_families = (path.parent / "families.py").exists() or (
            experiment_src is not None
            and (experiment_src / "families.py").exists()
            and (
                path.is_relative_to(experiment_src)
                or bool(re.search(r'''sys\.path\.(?:insert|append)\([^)]*(?:/|joinpath\()\s*["']src["']''', text))
            )
        )
        for pattern in forbidden_patterns:
            if pattern in text:
                if pattern in {"from families", "import families"} and has_local_families:
                    continue
                fail(errors, f"benchmark firewall violation in {rel(path)}: matched {pattern!r}; "
                             "experiments may only invoke benchmarks/menagerie/run.py as a subprocess "
                             "(that usage does not contain the forbidden strings)")


# Generators of tracked files must be pure functions of repo content: a wall-clock
# stamp in their output dirties tracked files at every UTC rollover (the generated_on
# failure mode removed on 2026-07-09). Content-derived timestamps (epochs, mtimes)
# are fine; only "now" calls are forbidden. Exemptions: build_site.py writes only the
# gitignored site/ tree, and menagerie run.py records genuine per-run GPU provenance.
WALL_CLOCK_RE = re.compile(r"datetime\.now\(|date\.today\(|utcnow\(")
# build_site stamps the generation date; the validator reads today's date only to
# age-check in-progress entries (it writes no tracked output). Both are exempt.
WALL_CLOCK_EXEMPT = {"scripts/build_site.py", "scripts/validate_repository.py", "benchmarks/menagerie/run.py"}


def validate_no_wall_clock_stamps(errors: list[str]) -> None:
    for scan_root in [ROOT / "scripts", ROOT / "benchmarks"]:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            if in_ignored_dir(path) or rel(path) in WALL_CLOCK_EXEMPT:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if WALL_CLOCK_RE.search(line):
                    fail(
                        errors,
                        f"wall-clock call in {rel(path)}:{lineno}: tracked generated files must be "
                        "byte-stable across regeneration — derive dates from content or keep the "
                        "stamp out of tracked output (see docs/quality_gates.md)",
                    )


# Scaffold-template filler prose. Any of these surviving in an experiment's
# README or reports means a section was never filled in after the run — and
# the site publishes README/report text verbatim (a placeholder shipped to
# the site on 2026-07-10 despite full results living in reports/report.md).
# Model diligence does not scale; this gate does. Keep in sync with
# templates/experiment/.
TEMPLATE_PLACEHOLDER_PHRASES = (
    "Fill this after the run.",
    "What specific uncertainty does this experiment resolve?",
    "State the mechanism you expect to work and why it should beat the baseline.",
    "What changed after this result? What is now more likely, less likely, or still unknown?",
    "Update this before reporting results.",
    "Update `artifact_manifest.yaml` before considering this result complete.",
)


def validate_no_template_placeholders(errors: list[str]) -> None:
    for exp in sorted(EXPERIMENTS.iterdir()):
        if not exp.is_dir():
            continue
        candidates = [exp / "README.md", *sorted((exp / "reports").glob("*.md"))]
        for path in candidates:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for phrase in TEMPLATE_PLACEHOLDER_PHRASES:
                if phrase in text:
                    fail(
                        errors,
                        f"scaffold placeholder still present in {rel(path)}: {phrase!r} — "
                        "fill the section in (the site publishes this text verbatim)",
                    )


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
        KNOWLEDGE / "experiment_readiness.md",
        KNOWLEDGE / "experiment_readiness.csv",
        KNOWLEDGE / "future_experiment_queue.json",
        KNOWLEDGE / "future_experiment_queue.md",
        KNOWLEDGE / "future_experiment_queue.csv",
        KNOWLEDGE / "claims" / "claim_ledger.json",
        KNOWLEDGE / "claims" / "index.md",
        KNOWLEDGE / "claims" / "index.csv",
        KNOWLEDGE / "decision_records" / "README.md",
        ROOT / "docs" / "idea_intake_protocol.md",
        ROOT / "docs" / "artifact_policy.md",
        ROOT / "docs" / "vllm_inference.md",
        ROOT / "requirements-vllm.txt",
        ROOT / "requirements-vllm.lock.txt",
        ROOT / "scripts" / "scaffold_research_program.py",
        ROOT / "scripts" / "scaffold_experiment.py",
        ROOT / "scripts" / "check_markdown_links.py",
        ROOT / "scripts" / "check_python_syntax.py",
        ROOT / "scripts" / "check_repository_text.py",
        ROOT / "scripts" / "build_site.py",
        ROOT / "scripts" / "check_site.py",
        ROOT / "scripts" / "find_related.py",
        ROOT / "scripts" / "scaffold_from_queue.py",
        ROOT / "templates" / "site" / "index.html",
        ROOT / "templates" / "site" / "assets" / "app.js",
        ROOT / "templates" / "site" / "assets" / "styles.css",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "experiment_proposal.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research_program.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "synthesis_update.yml",
        ROOT / ".github" / "pull_request_template.md",
        ROOT / ".github" / "workflows" / "validate.yml",
        ROOT / ".github" / "workflows" / "pages.yml",
        ROOT / "templates" / "research_program" / "charter.md",
        ROOT / "templates" / "research_program" / "backlog.md",
        ROOT / "templates" / "research_program" / "evidence.md",
        ROOT / "templates" / "experiment" / "README.md",
        ROOT / "templates" / "experiment" / "metadata.yaml",
        ROOT / "templates" / "experiment" / "reports" / "artifact_manifest.yaml",
        ROOT / "templates" / "experiment" / "src" / "vllm_runner.py",
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

    viz_path = KNOWLEDGE / "experiment_viz.json"
    viz_experiments: dict = {}
    if not viz_path.exists():
        fail(errors, "missing knowledge/experiment_viz.json (holds the native chart specs)")
    else:
        try:
            viz_payload = json.loads(viz_path.read_text(encoding="utf-8"))
            viz_experiments = viz_payload.get("experiments", {}) if isinstance(viz_payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            fail(errors, "knowledge/experiment_viz.json is not valid JSON")

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
        if experiment_chart_count(exp.name, viz_experiments) < 1:
            fail(errors, f"experiment has no native chart: {rel(exp)} — add at least one bar/line spec "
                         f"under experiments[\"{exp.name}\"].charts in knowledge/experiment_viz.json so the "
                         f"site shows a graph (matplotlib PNGs do not count). See knowledge/experiment_viz.json "
                         f"for the format; one spec should set \"headline\": true.")
        standard_manifest = exp / "reports" / "artifact_manifest.yaml"
        if standard_manifest.exists():
            validate_standard_artifact_manifest(errors, standard_manifest, exp)
        if (exp / "reports" / "adapters").exists() and not experiment_has_artifact_manifest(exp):
            fail(errors, f"adapter directory exists without an external artifact manifest: {rel(exp)}")

    for pattern in ["*:Zone.Identifier", "*.pyc"]:
        for path in ROOT.rglob(pattern):
            if not in_ignored_dir(path):
                fail(errors, f"generated/copy artifact should not be tracked: {rel(path)}")
    for path in ROOT.rglob("__pycache__"):
        if not in_ignored_dir(path):
            fail(errors, f"python cache directory should not be present: {rel(path)}")

    # git cannot track empty directories, so an empty dir exists locally but not in a fresh
    # CI checkout — catalog regeneration then diverges and generated-clean fails only in CI
    for path in EXPERIMENTS.rglob("*"):
        if path.is_dir() and not in_ignored_dir(path) and not any(path.iterdir()):
            fail(errors, f"empty directory (untrackable by git; will diverge in CI): {rel(path)}")

    for path in ROOT.rglob("*"):
        if path.is_file() and not in_ignored_dir(path) and path.stat().st_size > MAX_GITHUB_FILE_BYTES:
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

    readiness = KNOWLEDGE / "experiment_readiness.csv"
    if not readiness.exists():
        fail(errors, "missing knowledge/experiment_readiness.csv")
    else:
        with readiness.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required_columns = {
            "id",
            "readme_status",
            "primary_report",
            "experiment_log",
            "run_surface",
            "smoke_command",
            "artifact_manifest",
            "anchor_ready",
            "needs",
        }
        missing_columns = sorted(required_columns - set(rows[0].keys() if rows else []))
        if missing_columns:
            fail(errors, "readiness index missing columns: " + ", ".join(missing_columns))
        if len(rows) != len(experiments):
            fail(errors, f"readiness row count {len(rows)} != experiment count {len(experiments)}")
        readiness_ids = {row["id"] for row in rows if "id" in row}
        missing = sorted(exp_ids - readiness_ids)
        extra = sorted(readiness_ids - exp_ids)
        if missing:
            fail(errors, "readiness index missing experiments: " + ", ".join(missing))
        if extra:
            fail(errors, "readiness index has extra experiments: " + ", ".join(extra))
        invalid_ready = sorted(row.get("id", "") for row in rows if row.get("anchor_ready") not in {"yes", "no"})
        if invalid_ready:
            fail(errors, "readiness index has invalid anchor_ready values: " + ", ".join(invalid_ready[:20]))

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
    validate_future_queue(errors, set(program_ids))
    validate_experiment_status(errors)
    validate_root_readme(errors)
    validate_benchmark_firewall(errors)
    validate_no_wall_clock_stamps(errors)
    validate_no_template_placeholders(errors)

    gitattributes = ROOT / ".gitattributes"
    if not gitattributes.exists():
        fail(errors, "missing .gitattributes")
    else:
        content = gitattributes.read_text(encoding="utf-8", errors="replace")
        if "*.safetensors filter=lfs" not in content:
            fail(errors, ".gitattributes does not LFS-track safetensors")

    return report(errors)


# Keep in sync with _STATUS_LINE_RE / _STATUS_SINCE_RE in scripts/build_site.py.
STATUS_LINE_RE = re.compile(r"(?im)^[ \t]*\*\*status:\*\*[ \t]*(finished|in-progress)\b([^\n]*)$")
STATUS_SINCE_RE = re.compile(r"(?i)\bsince[ \t]+(\d{4}-\d{2}-\d{2})\b")


def validate_experiment_status(errors: list[str]) -> None:
    """Guard the finished/in-progress lifecycle, which lives as a canonical line
    at the top of each experiment's OWN README (the single source of truth):

        **Status:** finished
        **Status:** in-progress · since YYYY-MM-DD · <what remains>

    Co-locating it with the work means the agent that concludes an experiment
    updates it in place, and new experiments inherit it from the template — so it
    is far harder to leave stale than a central file. Every experiment must carry
    exactly one such line. An in-progress line must carry a parseable ISO `since`
    date and a reason, must not be future-dated, and must not be stale
    (> STALE_INPROGRESS_DAYS) — the anti-footgun that forces a concluded experiment
    out of "in progress". Status is never inferred from prose (preregistrations and
    finished ablations share the same "verdict / negative / not run" vocabulary, so
    any guess mislabels); the author declares it, CI keeps it honest.
    """
    today = dt.date.today()
    for exp in sorted(EXPERIMENTS.iterdir()):
        if not exp.is_dir():
            continue
        readme = exp / "README.md"
        if not readme.exists():
            fail(errors, f"{rel(readme)}: experiment has no README.md")
            continue
        text = readme.read_text(encoding="utf-8", errors="replace")
        found = list(STATUS_LINE_RE.finditer(text))
        if not found:
            fail(errors, (f"{rel(readme)}: missing the canonical status line. Add, right after the title, "
                          f"'**Status:** finished' or '**Status:** in-progress · since YYYY-MM-DD · <what remains>'."))
            continue
        if len(found) > 1:
            fail(errors, f"{rel(readme)}: {len(found)} '**Status:**' lines found; keep exactly one")
            continue
        status, tail = found[0].group(1).lower(), found[0].group(2)
        if status != "in-progress":
            continue
        since_match = STATUS_SINCE_RE.search(tail)
        reason = (tail[since_match.end():] if since_match else tail).strip().lstrip("·—–-:*").strip()
        if not reason:
            fail(errors, (f"{rel(readme)}: in-progress status needs a reason "
                          f"('**Status:** in-progress · since YYYY-MM-DD · <what remains>')"))
        if not since_match:
            fail(errors, f"{rel(readme)}: in-progress status needs 'since YYYY-MM-DD'")
            continue
        raw_since = since_match.group(1)
        try:
            since = dt.date.fromisoformat(raw_since)
        except ValueError:
            fail(errors, f"{rel(readme)}: status 'since {raw_since}' is not a valid ISO date")
            continue
        if since > today:
            fail(errors, f"{rel(readme)}: status 'since {raw_since}' is in the future")
            continue
        age = (today - since).days
        if age > STALE_INPROGRESS_DAYS:
            fail(errors, (f"{rel(readme)}: flagged in-progress for {age} days (limit {STALE_INPROGRESS_DAYS}). "
                          f"Set '**Status:** finished' if it concluded, or bump the 'since' date to re-affirm it is "
                          f"still running."))


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

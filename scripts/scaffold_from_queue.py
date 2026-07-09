#!/usr/bin/env python3
"""Create an experiment scaffold from a future experiment queue item."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True

import scaffold_experiment  # noqa: E402


QUEUE = ROOT / "knowledge" / "future_experiment_queue.json"
EXPERIMENTS = ROOT / "experiments"


def load_queue() -> dict[str, object]:
    if not QUEUE.exists():
        raise SystemExit("missing knowledge/future_experiment_queue.json")
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def find_proposal(queue: dict[str, object], proposal_id: str) -> dict[str, object]:
    proposals = queue.get("proposals", [])
    if not isinstance(proposals, list):
        raise SystemExit("future experiment queue does not contain a proposals list")
    for proposal in proposals:
        if isinstance(proposal, dict) and proposal.get("id") == proposal_id:
            return proposal
    raise SystemExit(f"unknown future queue proposal: {proposal_id}")


def candidate_program_ids(queue: dict[str, object]) -> set[str]:
    candidates = queue.get("candidate_programs", [])
    if not isinstance(candidates, list):
        return set()
    return {str(candidate.get("id", "")) for candidate in candidates if isinstance(candidate, dict)}


def experiment_id_for(proposal: dict[str, object], override: str) -> str:
    experiment_id = (override or str(proposal.get("id", ""))).strip()
    if not scaffold_experiment.SLUG_RE.match(experiment_id):
        raise SystemExit(f"invalid experiment id {experiment_id!r}; use lower snake_case")
    return experiment_id


def write_queue_context(exp_dir: Path, proposal: dict[str, object]) -> None:
    (exp_dir / "queue_proposal.json").write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_idea_intake(
    exp_dir: Path,
    proposal: dict[str, object],
    programs: list[str],
    candidate_programs: list[str],
) -> None:
    lines = [
        "# Idea Intake",
        "",
        "## Program Fit",
        "",
        f"- Program: {', '.join(programs)}",
        "- Existing or new program: existing" if programs else "- Existing or new program: new",
        "- Closest program scorecard reviewed: knowledge/program_scorecards.md",
        f"- Related future queue item: {proposal.get('id', '')}",
        f"- Candidate program line: {', '.join(candidate_programs)}" if candidate_programs else "- Candidate program line:",
        "",
        "## Prior Evidence",
        "",
        "- Anchor 1:",
        "- Anchor 2:",
        "- Anchor 3:",
        "- Closest duplicate or near-duplicate:",
        "",
        "## Novelty Claim",
        "",
        str(proposal.get("question", "")),
        "",
        "## Mechanism",
        "",
        str(proposal.get("hypothesis", "")),
        "",
        "## Control Plan",
        "",
        "- Baseline:",
        f"- Mechanism-falsifying control: {proposal.get('failure_signal', '')}",
        "- Shift or robustness check:",
        "- Hidden-label boundary:",
        "",
        "## Minimal Protocol",
        "",
        str(proposal.get("minimal_protocol", "")),
        "",
        "## Evidence Output",
        "",
        "- Program evidence update:",
        "- Claim ledger or synthesis update:",
        "- Reusable artifact: " + ", ".join(str(item) for item in proposal.get("expected_artifacts", [])),
        "- Stop or branch condition:",
        "",
        "## Decision",
        "",
        "- Run experiment:",
        "- Create program:",
        "- Write synthesis only:",
        "- Defer:",
        "",
        "## Queue Context",
        "",
        f"- Priority: {proposal.get('priority', '')}",
        f"- Status: {proposal.get('status', '')}",
        f"- Effort: {proposal.get('effort', '')}",
        f"- Success signal: {proposal.get('success_signal', '')}",
        f"- Avoid: {proposal.get('avoid', '')}",
        f"- Source: {proposal.get('source', '')}",
        "",
    ]
    (exp_dir / "idea_intake.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme_context(exp_dir: Path, proposal: dict[str, object], candidate_programs: list[str]) -> None:
    readme = exp_dir / "README.md"
    text = readme.read_text(encoding="utf-8")
    context = [
        "## Queue Context",
        "",
        f"- Queue item: `{proposal.get('id', '')}`",
        f"- Priority: `{proposal.get('priority', '')}`",
        f"- Status at scaffold time: `{proposal.get('status', '')}`",
        f"- Candidate program line: {', '.join(f'`{program_id}`' for program_id in candidate_programs) if candidate_programs else ''}",
        f"- Source: `{proposal.get('source', '')}`",
        "",
        "See [idea_intake.md](idea_intake.md) and [queue_proposal.json](queue_proposal.json) before running the full experiment.",
        "",
    ]
    marker = "## Setup"
    if marker in text:
        text = text.replace(marker, "\n".join(context) + marker, 1)
    else:
        text = text.rstrip() + "\n\n" + "\n".join(context)
    readme.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_id", help="id from knowledge/future_experiment_queue.json")
    parser.add_argument("--experiment-id", default="", help="override the experiment folder id")
    parser.add_argument("--tag", action="append", default=[], help="extra navigation tag; repeat as needed")
    parser.add_argument("--dry-run", action="store_true", help="validate routing without creating files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = load_queue()
    proposal = find_proposal(queue, args.proposal_id.strip())
    experiment_id = experiment_id_for(proposal, args.experiment_id)
    exp_dir = EXPERIMENTS / experiment_id
    if exp_dir.exists():
        print(f"experiment already exists: experiments/{experiment_id}", file=sys.stderr)
        return 2

    known_programs = scaffold_experiment.registry_ids()
    candidates = candidate_program_ids(queue)
    proposal_programs = [str(program_id) for program_id in proposal.get("programs", [])]
    programs = [program_id for program_id in proposal_programs if program_id in known_programs]
    candidate_programs = [program_id for program_id in proposal_programs if program_id in candidates]
    unknown = sorted(set(proposal_programs) - known_programs - candidates)
    if unknown:
        print("proposal references unknown program(s): " + ", ".join(unknown), file=sys.stderr)
        return 2
    if not programs:
        print("proposal has no registered research program; create or assign a program first", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"proposal: {proposal.get('id', '')}")
        print(f"experiment: experiments/{experiment_id}")
        print("programs: " + ", ".join(programs))
        if candidate_programs:
            print("candidate program lines: " + ", ".join(candidate_programs))
        return 0

    for dirname in ["src", "scripts", "configs", "data", "runs", "analysis", "reports"]:
        (exp_dir / dirname).mkdir(parents=True, exist_ok=True)

    title = str(proposal.get("title") or scaffold_experiment.title_from_slug(experiment_id))
    summary = str(proposal.get("question", ""))
    tags = ["queued-proposal"] + [tag.strip() for tag in args.tag if tag.strip()]

    scaffold_experiment.write_readme(exp_dir, title, summary, programs)
    scaffold_experiment.write_metadata(exp_dir, experiment_id, title, summary, tags, programs)
    scaffold_experiment.write_report(exp_dir, title)
    scaffold_experiment.write_run_script(exp_dir, experiment_id)
    (exp_dir / "experiment_log.md").write_text(
        f"# {title} Experiment Log\n\n## Scaffold\n\nCreated from future queue proposal `{proposal.get('id', '')}`.\n",
        encoding="utf-8",
    )
    (exp_dir / "configs" / "default.yaml").write_text("# Experiment configuration goes here.\n", encoding="utf-8")
    (exp_dir / "src" / "README.md").write_text("# Source\n\nPut experiment-local code here.\n", encoding="utf-8")
    shutil.copyfile(
        scaffold_experiment.TEMPLATE / "src" / "vllm_runner.py",
        exp_dir / "src" / "vllm_runner.py",
    )
    write_idea_intake(exp_dir, proposal, programs, candidate_programs)
    write_queue_context(exp_dir, proposal)
    write_readme_context(exp_dir, proposal, candidate_programs)

    print(f"created experiment: experiments/{experiment_id}")
    print("next: complete idea_intake.md anchors, implement the smoke path, then run make check")
    return 0


if __name__ == "__main__":
    sys.exit(main())

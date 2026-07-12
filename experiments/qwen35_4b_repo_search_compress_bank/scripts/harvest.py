#!/usr/bin/env python3
"""Search procedural training repositories with the frozen C53 coding policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import harness  # noqa: E402
import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def summarize(rows: list[dict], tasks: list[repo_tasks.RepoTask]) -> dict:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    family = {}
    for name in sorted({task.family for task in tasks}):
        members = [task for task in tasks if task.family == name]
        covered = sum(any(row["workspace_success"] for row in by_task[task.task_id]) for task in members)
        family[name] = {"tasks": len(members), "covered": covered, "coverage": covered / len(members)}
    covered = sum(any(row["workspace_success"] for row in by_task[task.task_id]) for task in tasks)
    return {
        "tasks": len(tasks),
        "trajectories": len(rows),
        "covered_tasks": covered,
        "task_coverage": covered / len(tasks),
        "successful_trajectories": sum(row["workspace_success"] for row in rows),
        "submitted_trajectories": sum(row["submitted"] for row in rows),
        "verified_after_final_patch": sum(row["verified_after_final_patch"] for row in rows),
        "commit_after_pass": sum(row["commit_after_pass"] for row in rows),
        "sampled_tokens": sum(row["sampled_tokens"] for row in rows),
        "per_family": family,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--tasks-per-family", type=int, default=None)
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    hcfg = cfg["harvest"]
    artifact_root = args.artifact_root or resolve(cfg["artifacts"]["root"])
    model = args.model or resolve(cfg["model"]["search_teacher"])
    if not (model / "config.json").is_file():
        raise SystemExit(f"search teacher is not a merged checkpoint: {model}")
    n_tasks = args.tasks_per_family or int(hcfg["tasks_per_family"])
    n_trajectories = args.trajectories or int(hcfg["trajectories"])
    max_turns = args.max_turns or int(hcfg["max_turns"])
    tasks = repo_tasks.make_tasks(
        tuple(cfg["families"]["train"]), n_tasks, int(hcfg["seed"]), str(hcfg["split"])
    )

    runner = harness.make_runner(cfg["engine"], model_override=str(model))
    sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(hcfg["think_budget"]),
        n=1,
        answer_max_tokens=int(hcfg["answer_max_tokens"]),
        greedy=False,
        temperature=float(hcfg["temperature"]),
        top_p=float(hcfg["top_p"]),
        top_k=int(hcfg["top_k"]),
        run_seed=int(hcfg["run_seed"]),
    )
    specs = [(task, trajectory) for task in tasks for trajectory in range(n_trajectories)]
    rows = repo_agent.run_episodes(runner, specs, sampling, max_turns=max_turns)
    summary = summarize(rows, tasks)
    payload = {
        "schema_version": 1,
        "model": str(model.resolve()),
        "model_config_sha256": digest(model / "config.json"),
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "settings": {
            "seed": int(hcfg["seed"]),
            "run_seed": int(hcfg["run_seed"]),
            "tasks_per_family": n_tasks,
            "trajectories": n_trajectories,
            "max_turns": max_turns,
            "think_budget": int(hcfg["think_budget"]),
            "answer_max_tokens": int(hcfg["answer_max_tokens"]),
            "temperature": float(hcfg["temperature"]),
            "top_p": float(hcfg["top_p"]),
            "top_k": int(hcfg["top_k"]),
        },
        "summary": summary,
        "trajectories": rows,
    }
    bank.assert_firewall_clean(payload, tasks)
    output = args.output or artifact_root / "harvest" / "trajectories.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

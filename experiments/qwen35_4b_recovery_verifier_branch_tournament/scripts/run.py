#!/usr/bin/env python3
"""Resumable public-verifier recovery branch tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_paths(cfg: dict) -> dict[str, Path]:
    return {name: resolve(cfg["model"][name]) for name in ("base", "action", "candidate")}


def validate_inputs(cfg: dict) -> dict[str, str]:
    observed = {}
    for name, path in model_paths(cfg).items():
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_5":
            raise SystemExit(f"{name} is not Qwen/Qwen3.5-4B")
        observed[f"model/{name}"] = sha256_file(path / "model.safetensors")
        if observed[f"model/{name}"] != cfg["model"]["expected_weight_sha256"][name]:
            raise SystemExit(f"frozen {name} weight hash mismatch")
    for block, paths in cfg["retrospective_qualification"].items():
        for arm in ("candidate", "action"):
            path = resolve(paths[arm])
            if not path.is_file():
                raise SystemExit(f"missing retrospective {block}/{arm}: {path}")
            observed[f"retrospective/{block}/{arm}"] = sha256_file(path)
            if observed[f"retrospective/{block}/{arm}"] != paths[f"{arm}_sha256"]:
                raise SystemExit(f"retrospective {block}/{arm} hash mismatch")
    return observed


def run_command(command: list[str], allowed_returncodes: tuple[int, ...] = (0,)) -> int:
    print("[run] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def run_if_missing(
    output: Path,
    command: list[str],
    allowed_returncodes: tuple[int, ...] = (0,),
) -> int:
    if output.exists():
        print(f"[resume] {output} exists", flush=True)
        payload = json.loads(output.read_text(encoding="utf-8"))
        if "gate" in payload and not payload["gate"].get("passed", False):
            return 4
        return 0
    return run_command(command, allowed_returncodes)


def selector_command(cfg: dict, candidate: Path, action: Path, output: Path) -> list[str]:
    return [
        str(ROOT / ".venv" / "bin" / "python"),
        str(EXP / "scripts" / "select_tournament.py"),
        "--candidate", str(candidate),
        "--action", str(action),
        "--output", str(output),
    ]


def retrospective_qualification(cfg: dict, output_root: Path) -> dict:
    receipts = {}
    for block, paths in cfg["retrospective_qualification"].items():
        output = output_root / f"selector_{block}.json"
        run_if_missing(
            output,
            selector_command(cfg, resolve(paths["candidate"]), resolve(paths["action"]), output),
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload["public"]["success"] != 0.75:
            raise AssertionError(f"retrospective {block} public selector changed")
        if payload["oracle_union"]["success"] != 0.7875:
            raise AssertionError(f"retrospective {block} union changed")
        if payload["oracle_union"]["public_capture"] < 0.95:
            raise AssertionError(f"retrospective {block} capture changed")
        if payload["public"]["success"] - payload["expected_random"]["success"] != 0.0625:
            raise AssertionError(f"retrospective {block} expected-random contrast changed")
        receipts[block] = {
            "public_success": payload["public"]["success"],
            "oracle_union_success": payload["oracle_union"]["success"],
            "oracle_union_capture": payload["oracle_union"]["public_capture"],
            "expected_random_success": payload["expected_random"]["success"],
            "selection_counts": payload["selection_counts"]["public"],
        }
    return receipts


def cpu_smoke() -> dict:
    cfg = load_config()
    if cfg["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise AssertionError("single-model rule violated")
    hashes = validate_inputs(cfg)
    prospective = tuple(cfg["families"]["prospective"])
    if prospective != repo_tasks.PROSPECTIVE_FAMILIES:
        raise AssertionError("prospective family registry differs from config")
    if set(prospective) & set((*repo_tasks.TRAIN_FAMILIES, *repo_tasks.TRANSFER_FAMILIES)):
        raise AssertionError("prospective families overlap prior recovery families")

    tasks = repo_tasks.make_tasks(prospective, 1, seed=85210, split="smoke")
    for task in tasks:
        for state, expected in (("initial", (False, False)),
                                ("partial", (False, False)),
                                ("oracle", (True, True))):
            env = repo_tasks.RepoEnv(task)
            try:
                if state == "partial":
                    env.apply_partial()
                elif state == "oracle":
                    env.apply_oracle()
                observed = (env.visible_pass(), env.hidden_pass())
                if observed != expected:
                    raise AssertionError((task.task_id, state, observed, expected))
            finally:
                env.close()
    bank.assert_firewall_clean({"tasks": [task.public_manifest() for task in tasks]}, tasks)

    manifests = []
    for block, spec in cfg["evaluation"]["blocks"].items():
        block_tasks = repo_tasks.make_tasks(
            prospective, int(spec["tasks_per_family"]), int(spec["seed"]), block
        )
        manifests.append(repo_tasks.manifest_digest(block_tasks))
    if len(set(manifests)) != len(manifests):
        raise AssertionError("prospective blocks overlap")

    per_call = int(cfg["evaluation"]["think_budget"]) + int(
        cfg["evaluation"]["answer_max_tokens"]
    )
    branch_reserved = int(cfg["evaluation"]["deep_turns"]) * per_call
    tournament_reserved = 2 * branch_reserved
    sample_reserved = (
        int(cfg["evaluation"]["sample_more_trajectories"])
        * int(cfg["evaluation"]["sample_more_turns_each"])
        * per_call
    )
    if tournament_reserved != sample_reserved:
        raise AssertionError("tournament/sample-more reservation mismatch")

    retrospective = retrospective_qualification(cfg, EXP / "analysis")
    return {
        "schema_version": 1,
        "status": "PASS",
        "model": cfg["model"]["id"],
        "input_hashes": hashes,
        "prospective_families": list(prospective),
        "prospective_blocks_disjoint": True,
        "branch_reserved_sampled_tokens": branch_reserved,
        "tournament_reserved_sampled_tokens": tournament_reserved,
        "sample_more_reserved_sampled_tokens": sample_reserved,
        "retrospective_qualification": retrospective,
        "selector_uses_hidden_outcomes": False,
        "benchmark_content_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if sum((args.smoke, args.gpu_smoke, args.full)) != 1:
        parser.error("choose exactly one of --smoke, --gpu-smoke, --full")
    if args.smoke:
        receipt = cpu_smoke()
        output = EXP / "reports" / "smoke_receipt.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0

    cfg = load_config()
    validate_inputs(cfg)
    models = model_paths(cfg)
    artifact_root = resolve(cfg["artifacts"]["root"])
    py = str(ROOT / ".venv" / "bin" / "python")
    vpy = str(ROOT / ".venv-vllm" / "bin" / "python")

    def evaluate(
        arm: str,
        model: Path,
        block: str,
        mode: str,
        tasks_per_family: int | None = None,
        output_override: Path | None = None,
    ) -> Path:
        output = output_override or artifact_root / "eval" / (
            f"{block}_recovery_{arm}_{mode}.json"
        )
        command = [
            vpy,
            str(EXP / "scripts" / "eval_repo_agent.py"),
            "--arm", arm,
            "--model", str(model),
            "--block", block,
            "--mode", mode,
            "--output", str(output),
        ]
        if tasks_per_family is not None:
            command.extend(("--tasks-per-family", str(tasks_per_family)))
        run_if_missing(output, command)
        return output

    if args.gpu_smoke:
        block = "prospective_dev"
        candidate = evaluate(
            "candidate", models["candidate"], block, "deep", tasks_per_family=1,
            output_override=artifact_root / "smoke" / "candidate.json",
        )
        action = evaluate(
            "action", models["action"], block, "deep", tasks_per_family=1,
            output_override=artifact_root / "smoke" / "action.json",
        )
        tournament = artifact_root / "smoke" / "tournament.json"
        run_if_missing(tournament, selector_command(cfg, candidate, action, tournament))
        payload = json.loads(tournament.read_text(encoding="utf-8"))
        if payload["reserved_sampled_tokens_per_case"] != 12_288:
            raise SystemExit("GPU smoke tournament reservation differs from preregistration")
        if set(payload["selection_counts"]["public"]) - {"candidate", "action"}:
            raise SystemExit("GPU smoke emitted an unknown selector arm")
        return 0

    retrospective_qualification(cfg, EXP / "analysis")

    def run_block(block: str) -> int:
        # All trajectory controls precede theoretical feasibility and selector scoring.
        base = evaluate("base", models["base"], block, "deep")
        candidate = evaluate("candidate", models["candidate"], block, "deep")
        action = evaluate("action", models["action"], block, "deep")
        candidate_sample = evaluate(
            "candidate", models["candidate"], block, "sample_more"
        )
        action_sample = evaluate("action", models["action"], block, "sample_more")
        del base  # contextual control; not used to set the incumbent bar.

        feasibility = EXP / "analysis" / f"{block}_feasibility.json"
        feasibility_code = run_if_missing(feasibility, [
            py,
            str(EXP / "scripts" / "check_feasibility.py"),
            "--candidate", str(candidate),
            "--action", str(action),
            "--candidate-sample-more", str(candidate_sample),
            "--action-sample-more", str(action_sample),
            "--output", str(feasibility),
        ], allowed_returncodes=(0, 4))
        if feasibility_code == 4:
            print(f"[run] {block} tournament ceiling cannot clear frozen controls", flush=True)
            return 4

        tournament = artifact_root / "eval" / f"{block}_tournament.json"
        run_if_missing(tournament, selector_command(cfg, candidate, action, tournament))
        gate = EXP / "analysis" / f"{block}_gate.json"
        return run_if_missing(gate, [
            py,
            str(EXP / "scripts" / "analyze_gate.py"),
            "--tournament", str(tournament),
            "--candidate", str(candidate),
            "--action", str(action),
            "--candidate-sample-more", str(candidate_sample),
            "--action-sample-more", str(action_sample),
            "--output", str(gate),
        ], allowed_returncodes=(0, 4))

    if run_block("prospective_dev") == 4:
        print("[run] prospective development failed; confirmation and winner bank stay sealed")
        return 4
    if run_block("prospective_confirm") == 4:
        print("[run] prospective confirmation failed; winner bank and Menagerie stay sealed")
        return 4
    print(
        "[run] both public-tournament blocks passed; a new transition-balanced winner-bank "
        "experiment is authorized. Menagerie remains sealed in this harness-only result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

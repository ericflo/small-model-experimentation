#!/usr/bin/env python3
"""Apply every preregistered qualification gate to one specialist domain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import domain_families, load_config, read_jsonl, write_json  # noqa: E402


CORRECTION_MARKERS = ("check", "verify", "reconsider", "backtrack", "revise", "mistake", "instead")


def _rows(directory: Path) -> list[dict]:
    return read_jsonl(directory / "episode_rows.jsonl.gz")


def _scores(directory: Path, families: list[str], best_of_k: bool = False) -> dict:
    groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in _rows(directory):
        if row["family"] in families:
            groups[(row["family"], int(row["level"]), int(row["ep_seed"]))].append(float(row["score"]))
    if not groups:
        raise ValueError(f"no requested families in {directory}")
    values: dict[str, list[float]] = defaultdict(list)
    for key, samples in groups.items():
        if not best_of_k and len(samples) != 1:
            raise ValueError(f"expected greedy singleton for {key} in {directory}; got {len(samples)}")
        values[key[0]].append(max(samples) if best_of_k else samples[0])
    by_family = {family: sum(items) / len(items) for family, items in sorted(values.items())}
    if set(by_family) != set(families):
        raise ValueError(f"family mismatch in {directory}: {sorted(by_family)} != {sorted(families)}")
    return {
        "by_family": by_family,
        "macro": sum(by_family.values()) / len(by_family),
        "episode_groups": len(groups),
        "rollouts_per_group": sorted({len(samples) for samples in groups.values()}),
    }


def _entropy(logprob_rows: list | None) -> list[float]:
    output = []
    for token_row in logprob_rows or []:
        if not isinstance(token_row, dict):
            continue
        probabilities = [
            math.exp(float(item["logprob"]))
            for item in token_row.values()
            if isinstance(item, dict) and item.get("logprob") is not None
        ]
        mass = sum(probabilities)
        if mass > 1.0 + 1e-5:
            probabilities = [value / mass for value in probabilities]
            tail = 0.0
        else:
            tail = max(0.0, 1.0 - mass)
        value = -sum(p * math.log(max(p, 1e-30)) for p in probabilities)
        if tail:
            value -= tail * math.log(tail)
        output.append(value)
    return output


def _diagnostics(directory: Path, families: list[str]) -> dict:
    rows = [row for row in _rows(directory) if row["family"] in families]
    action_valid = []
    natural_close = []
    entropies = []
    correction = []
    sampled_tokens = 0
    logical_input_tokens = 0
    for row in rows:
        action_valid.append(float(row.get("action_valid_rate", 0.0)))
        natural_close.append(float(row.get("natural_close_rate", 0.0)))
        for turn in row["turns"]:
            policy = turn["policy"]
            text = str(policy.get("text", "")).lower()
            correction.append(float(any(marker in text for marker in CORRECTION_MARKERS)))
            entropies.extend(_entropy(policy.get("stage1_logprobs")))
            entropies.extend(_entropy(policy.get("stage2_logprobs")))
            sampled_tokens += int(policy.get("n_sampled_tokens") or 0)
            logical_input_tokens += int(policy.get("n_stage1_prompt_tokens") or 0)
            logical_input_tokens += int(policy.get("n_stage2_prompt_tokens") or 0)
    return {
        "episodes": len(rows),
        "action_valid_rate": sum(action_valid) / len(action_valid),
        "natural_close_rate": sum(natural_close) / len(natural_close),
        "entropy": sum(entropies) / len(entropies) if entropies else None,
        "entropy_positions": len(entropies),
        "correction_marker_rate": sum(correction) / len(correction) if correction else 0.0,
        "sampled_tokens": sampled_tokens,
        "logical_model_input_tokens": logical_input_tokens,
    }


def _atom_retention(incumbent: Path, candidate: Path) -> dict:
    def mapping(directory: Path) -> dict[tuple[str, str], float]:
        return {
            (row["family"], row["id"]): float(row["outputs"][0]["score"])
            for row in read_jsonl(directory / "atom_rows.jsonl.gz")
        }
    left, right = mapping(incumbent), mapping(candidate)
    if set(left) != set(right):
        raise ValueError("atom pairing mismatch")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left):
        grouped[key[0]].append(right[key] - left[key])
    by_family = {family: sum(values) / len(values) for family, values in sorted(grouped.items())}
    return {
        "n": len(left),
        "by_family_delta": by_family,
        "worst_family_delta": min(by_family.values()) if by_family else None,
    }


def _behavior_diff(source: Path, candidate: Path, families: list[str]) -> dict:
    def mapping(directory: Path) -> dict[tuple[str, int, int], dict]:
        return {
            (row["family"], int(row["level"]), int(row["ep_seed"])): row
            for row in _rows(directory)
            if row["family"] in families
        }
    left, right = mapping(source), mapping(candidate)
    if set(left) != set(right):
        raise ValueError("behavioral-canary pairing mismatch")
    changed_episodes = 0
    changed_first_turns = 0
    for key in sorted(left):
        left_turns, right_turns = left[key]["turns"], right[key]["turns"]
        left_tokens = [turn["policy"].get("token_ids") for turn in left_turns]
        right_tokens = [turn["policy"].get("token_ids") for turn in right_turns]
        changed = left_tokens != right_tokens
        changed_episodes += changed
        if left_turns and right_turns:
            changed_first_turns += left_tokens[0] != right_tokens[0]
    return {
        "n_paired_episodes": len(left),
        "changed_episodes": changed_episodes,
        "changed_first_turns": changed_first_turns,
        "passed": changed_episodes > 0,
    }


def _relative_retention(candidate: float | None, incumbent: float | None, max_drop: float) -> bool:
    if candidate is None or incumbent is None:
        return False
    if incumbent < 0.01:
        return candidate >= incumbent - 0.01
    return candidate >= incumbent * (1.0 - max_drop)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--domain", choices=("discover", "control", "tools", "compose"), required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--incumbent-best8", type=Path, required=True)
    parser.add_argument("--dagger", type=Path, required=True)
    parser.add_argument("--extra-sft", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    own = domain_families(config, args.domain)
    process = list(config["split"]["train_families"]) + list(config["split"]["transfer_families"])
    retention = [family for family in process if family not in own]
    scores = {
        "incumbent": _scores(args.incumbent, own),
        "incumbent_best8": _scores(args.incumbent_best8, own, best_of_k=True),
        "dagger": _scores(args.dagger, own),
        "extra_sft": _scores(args.extra_sft, own),
        "shuffled": _scores(args.shuffled, own),
        "specialist": _scores(args.specialist, own),
    }
    retention_scores = {
        "incumbent": _scores(args.incumbent, retention),
        "specialist": _scores(args.specialist, retention),
    }
    retention_delta = {
        family: retention_scores["specialist"]["by_family"][family]
        - retention_scores["incumbent"]["by_family"][family]
        for family in retention
    }
    diagnostics = {
        "incumbent": _diagnostics(args.incumbent, process),
        "specialist": _diagnostics(args.specialist, process),
    }
    atom = _atom_retention(args.incumbent, args.specialist)
    behavior = _behavior_diff(args.dagger, args.specialist, own)
    specialist_scores_receipt = json.loads(
        (args.specialist / "scores.json").read_text(encoding="utf-8")
    )
    merge = specialist_scores_receipt.get("model_fingerprint", {}).get("merge_receipt") or {}
    gate = config["gates"]
    candidate = scores["specialist"]["macro"]
    checks = {
        "vs_incumbent": candidate - scores["incumbent"]["macro"] >= float(gate["specialist_incumbent_delta"]),
        "vs_dagger": candidate - scores["dagger"]["macro"] >= float(gate["specialist_vs_dagger_delta"]),
        "vs_extra_sft": candidate - scores["extra_sft"]["macro"] >= float(gate["specialist_vs_extra_sft_delta"]),
        "vs_shuffled": candidate - scores["shuffled"]["macro"] >= float(gate["specialist_vs_shuffled_delta"]),
        "beats_incumbent_best8": candidate > scores["incumbent_best8"]["macro"],
        "inference_tokens_below_best8": (
            diagnostics["specialist"]["sampled_tokens"]
            < _diagnostics(args.incumbent_best8, process)["sampled_tokens"]
        ),
        "every_retention_family": min(retention_delta.values()) >= -float(gate["specialist_max_retention_regression"]),
        "atom_retention": atom["worst_family_delta"] is not None
        and atom["worst_family_delta"] >= -float(gate["specialist_max_retention_regression"]),
        "action_validity": diagnostics["specialist"]["action_valid_rate"]
        >= diagnostics["incumbent"]["action_valid_rate"] - float(gate["specialist_max_retention_regression"]),
        "natural_close": diagnostics["specialist"]["natural_close_rate"]
        >= diagnostics["incumbent"]["natural_close_rate"] - float(gate["specialist_max_natural_close_regression"]),
        "entropy": _relative_retention(
            diagnostics["specialist"]["entropy"], diagnostics["incumbent"]["entropy"],
            float(gate["specialist_max_entropy_relative_drop"]),
        ),
        "correction_markers": _relative_retention(
            diagnostics["specialist"]["correction_marker_rate"],
            diagnostics["incumbent"]["correction_marker_rate"],
            float(gate["specialist_max_correction_marker_relative_drop"]),
        ),
        "nonzero_behavioral_canary": behavior["passed"],
        "nonzero_merged_delta": int(merge.get("nonzero_lora_modules", 0)) > 0,
    }
    result = {
        "stage": "specialist_qualification",
        "domain": args.domain,
        "own_families": own,
        "retention_families": retention,
        "scores": scores,
        "retention_scores": retention_scores,
        "retention_family_delta": retention_delta,
        "diagnostics": diagnostics,
        "atom_retention": atom,
        "behavioral_canary_vs_dagger": behavior,
        "gate": {"passed": all(checks.values()), "checks": checks},
    }
    output = args.out or EXP / "analysis" / "specialists" / f"{args.domain}.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired proxy analysis and preregistered DAgger/RL mechanism gates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, read_jsonl, write_json  # noqa: E402


def _episode_map(directory: Path) -> dict[tuple, dict]:
    rows = read_jsonl(directory / "episode_rows.jsonl.gz")
    return {
        (row["family"], int(row["level"]), int(row["ep_seed"]), int(row["rollout"])): row
        for row in rows
    }


def _atom_map(directory: Path) -> dict[tuple, dict]:
    rows = read_jsonl(directory / "atom_rows.jsonl.gz")
    return {(row["family"], row["id"]): row for row in rows}


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(probability * (len(values) - 1))))
    return values[index]


def _paired_episode_delta(
    baseline_dir: Path,
    candidate_dir: Path,
    families: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    baseline = _episode_map(baseline_dir)
    candidate = _episode_map(candidate_dir)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) ^ set(candidate))[:5]
        raise ValueError(f"episode pairing mismatch: {missing}")
    by_family: dict[str, list[float]] = defaultdict(list)
    aux: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in sorted(baseline):
        family = key[0]
        if family not in families:
            continue
        left, right = baseline[key], candidate[key]
        by_family[family].append(float(right["score"]) - float(left["score"]))
        aux[family]["action_valid"].append(
            float(right.get("action_valid_rate", 0.0)) - float(left.get("action_valid_rate", 0.0))
        )
        aux[family]["natural_close"].append(
            float(right.get("natural_close_rate", 0.0)) - float(left.get("natural_close_rate", 0.0))
        )
    family_delta = {family: sum(values) / len(values) for family, values in sorted(by_family.items())}
    macro = sum(family_delta.values()) / len(family_delta) if family_delta else 0.0
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        per_family = []
        for family in sorted(by_family):
            values = by_family[family]
            resampled = [values[rng.randrange(len(values))] for _ in values]
            per_family.append(sum(resampled) / len(resampled))
        draws.append(sum(per_family) / len(per_family))
    action_valid = [value for family in aux.values() for value in family["action_valid"]]
    natural_close = [value for family in aux.values() for value in family["natural_close"]]
    return {
        "families": families,
        "n_pairs": sum(len(values) for values in by_family.values()),
        "family_delta": family_delta,
        "macro_delta": macro,
        "macro_delta_ci95": [_quantile(draws, 0.025), _quantile(draws, 0.975)],
        "mean_action_valid_delta": sum(action_valid) / len(action_valid) if action_valid else 0.0,
        "mean_natural_close_delta": sum(natural_close) / len(natural_close) if natural_close else 0.0,
    }


def _paired_atom_delta(baseline_dir: Path, candidate_dir: Path) -> dict:
    baseline = _atom_map(baseline_dir)
    candidate = _atom_map(candidate_dir)
    if set(baseline) != set(candidate):
        raise ValueError("atom pairing mismatch")
    by_family: dict[str, list[float]] = defaultdict(list)
    parse_by_family: dict[str, list[float]] = defaultdict(list)
    for key in sorted(baseline):
        left = baseline[key]["outputs"][0]
        right = candidate[key]["outputs"][0]
        by_family[key[0]].append(float(right["score"]) - float(left["score"]))
        parse_by_family[key[0]].append(
            float(right.get("answer_value") is not None) - float(left.get("answer_value") is not None)
        )
    family_delta = {family: sum(values) / len(values) for family, values in sorted(by_family.items())}
    parse_delta = {
        family: sum(values) / len(values) for family, values in sorted(parse_by_family.items())
    }
    return {
        "n_pairs": len(baseline),
        "family_delta": family_delta,
        "family_macro_delta": sum(family_delta.values()) / len(family_delta),
        "family_macro_parse_delta": sum(parse_delta.values()) / len(parse_delta),
        "worst_family_delta": min(family_delta.values()),
    }


def _comparison(
    baseline_dir: Path,
    candidate_dir: Path,
    config: dict,
    seed: int,
) -> dict:
    n_bootstrap = int(config["controls"]["paired_bootstrap_samples"])
    return {
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "train_episodes": _paired_episode_delta(
            baseline_dir,
            candidate_dir,
            list(config["split"]["train_families"]),
            n_bootstrap,
            seed,
        ),
        "transfer_episodes": _paired_episode_delta(
            baseline_dir,
            candidate_dir,
            list(config["split"]["transfer_families"]),
            n_bootstrap,
            seed + 1,
        ),
        "atom_retention": _paired_atom_delta(baseline_dir, candidate_dir),
    }


def _dagger_gate(comparison: dict, config: dict) -> dict:
    gate = config["gates"]
    train = comparison["train_episodes"]
    transfer = comparison["transfer_episodes"]
    atom = comparison["atom_retention"]
    checks = {
        "train_macro": train["macro_delta"] >= float(gate["dagger_train_macro_delta"]),
        "transfer_macro": transfer["macro_delta"] >= float(gate["dagger_transfer_macro_min_delta"]),
        "one_transfer_family": max(transfer["family_delta"].values()) >= float(gate["dagger_one_transfer_family_delta"]),
        "action_valid_retention": train["mean_action_valid_delta"] >= -float(gate["max_retention_regression"]),
        "natural_close_retention": train["mean_natural_close_delta"] >= -float(gate["max_retention_regression"]),
        "atom_score_retention": atom["family_macro_delta"] >= -float(gate["max_retention_regression"]),
        "atom_parse_retention": atom["family_macro_parse_delta"] >= -float(gate["max_retention_regression"]),
        "single_family_retention": atom["worst_family_delta"] >= -float(gate["max_single_family_regression"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("dagger", "rl"), required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dagger", type=Path)
    parser.add_argument("--matched-sft", type=Path)
    parser.add_argument("--shuffled", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config, _ = load_config(args.config)

    if args.phase == "dagger":
        comparison = _comparison(args.incumbent, args.candidate, config, seed=901)
        result = {
            "phase": "dagger",
            "incumbent_vs_candidate": comparison,
            "gate": _dagger_gate(comparison, config),
        }
    else:
        required = {"dagger": args.dagger, "matched_sft": args.matched_sft, "shuffled": args.shuffled}
        missing = [name for name, path in required.items() if path is None]
        if missing:
            raise SystemExit(f"RL phase requires: {', '.join(missing)}")
        incumbent_cmp = _comparison(args.incumbent, args.candidate, config, seed=911)
        dagger_cmp = _comparison(args.dagger, args.candidate, config, seed=921)
        sft_cmp = _comparison(args.matched_sft, args.candidate, config, seed=931)
        shuffled_cmp = _comparison(args.shuffled, args.candidate, config, seed=941)
        gate = config["gates"]
        improved = sum(
            delta > 0.0 for delta in dagger_cmp["train_episodes"]["family_delta"].values()
        )
        retention = _dagger_gate(incumbent_cmp, config)["checks"]
        checks = {
            "vs_dagger_train_macro": dagger_cmp["train_episodes"]["macro_delta"] >= float(gate["rl_vs_dagger_macro_delta"]),
            "vs_matched_sft_train_macro": sft_cmp["train_episodes"]["macro_delta"] >= float(gate["rl_vs_matched_sft_macro_delta"]),
            "vs_shuffled_train_macro": shuffled_cmp["train_episodes"]["macro_delta"] >= float(gate["rl_vs_shuffled_macro_delta"]),
            "improved_train_families": improved >= int(gate["rl_min_improved_train_families"]),
            "transfer_nonnegative_vs_dagger": dagger_cmp["transfer_episodes"]["macro_delta"] >= float(gate["rl_transfer_macro_min_delta"]),
            "action_valid_retention": retention["action_valid_retention"],
            "natural_close_retention": retention["natural_close_retention"],
            "atom_score_retention": retention["atom_score_retention"],
            "atom_parse_retention": retention["atom_parse_retention"],
            "single_family_retention": retention["single_family_retention"],
        }
        result = {
            "phase": "rl",
            "candidate_vs_incumbent": incumbent_cmp,
            "candidate_vs_dagger": dagger_cmp,
            "candidate_vs_matched_sft": sft_cmp,
            "candidate_vs_shuffled": shuffled_cmp,
            "improved_train_families": improved,
            "gate": {"passed": all(checks.values()), "checks": checks},
        }

    output = args.out or EXP / "analysis" / f"{args.phase}_gate.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

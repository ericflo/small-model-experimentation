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

from io_utils import (  # noqa: E402
    domain_families,
    load_config,
    read_jsonl,
    resolve_repo_path,
    sha256_file,
    write_json,
)


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
    entropy_sum = 0.0
    entropy_positions = 0
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
            stored_positions = int(
                policy.get("reported_top20_tail_lumped_entropy_positions") or 0
            )
            if stored_positions:
                entropy_sum += float(
                    policy["reported_top20_tail_lumped_entropy_sum"]
                )
                entropy_positions += stored_positions
            else:
                values = _entropy(policy.get("stage1_logprobs"))
                values.extend(_entropy(policy.get("stage2_logprobs")))
                entropy_sum += sum(values)
                entropy_positions += len(values)
            sampled_tokens += int(policy.get("n_sampled_tokens") or 0)
            logical_input_tokens += int(policy.get("n_stage1_prompt_tokens") or 0)
            logical_input_tokens += int(policy.get("n_stage2_prompt_tokens") or 0)
    return {
        "episodes": len(rows),
        "action_valid_rate": sum(action_valid) / len(action_valid),
        "natural_close_rate": sum(natural_close) / len(natural_close),
        "entropy": entropy_sum / entropy_positions if entropy_positions else None,
        "entropy_positions": entropy_positions,
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


def _checkpoint_integrity(
    eval_directory: Path,
    expected_model: Path,
    expected_adapter: Path,
) -> dict:
    """Tie an evaluation fingerprint to its exact adapter, merge, and training receipt."""
    scores_path = eval_directory / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    model_path = Path(scores["model"]).resolve()
    if model_path != expected_model.resolve():
        raise ValueError(f"unexpected model in {scores_path}: {model_path}")
    merge_path = model_path / "merge_receipt.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    fingerprint = scores.get("model_fingerprint", {})
    if fingerprint.get("merge_receipt_sha256") != sha256_file(merge_path):
        raise ValueError(f"stale merge fingerprint in {scores_path}")
    if fingerprint.get("merge_receipt") != merge:
        raise ValueError(f"embedded merge receipt mismatch in {scores_path}")
    adapter_path = Path(merge["adapter"]).resolve()
    if adapter_path != expected_adapter.resolve():
        raise ValueError(f"unexpected adapter in {merge_path}: {adapter_path}")
    if merge.get("adapter_config_sha256") != sha256_file(
        adapter_path / "adapter_config.json"
    ):
        raise ValueError(f"adapter-config hash mismatch in {merge_path}")
    if merge.get("adapter_weights_sha256") != sha256_file(
        adapter_path / "adapter_model.safetensors"
    ):
        raise ValueError(f"adapter-weight hash mismatch in {merge_path}")
    training_path = adapter_path / "training_receipt.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    structural_merge = (
        int(merge.get("applied_lora_modules", 0)) > 0
        and int(merge.get("nonzero_lora_modules", 0))
        == int(merge.get("applied_lora_modules", -1))
        and merge.get("merge_device") == "cuda"
        and merge.get("fp32_tf32_allowed") is False
    )
    return {
        "model": str(model_path),
        "adapter": str(adapter_path),
        "scores_sha256": sha256_file(scores_path),
        "merge_receipt_sha256": sha256_file(merge_path),
        "training_receipt_sha256": sha256_file(training_path),
        "merge": merge,
        "training": training,
        "structural_merge": structural_merge,
    }


def _sft_completed(receipt: dict, *, source: Path, max_steps: int | None, epochs: float) -> bool:
    return (
        receipt.get("method") == "emission_weighted_dagger_sft"
        and receipt.get("source_model") == str(source.resolve())
        and math.isclose(float(receipt.get("epochs", -1)), float(epochs))
        and int(receipt.get("max_steps", -2)) == (-1 if max_steps is None else max_steps)
        and int(receipt.get("optimizer_steps", 0)) > 0
        and (max_steps is None or int(receipt.get("optimizer_steps", -1)) == max_steps)
        and float(receipt.get("skip_rate", 1.0)) <= 0.15
    )


def _grpo_completed(receipt: dict, *, source: Path, steps: int, shuffled: bool) -> bool:
    return (
        receipt.get("method") == "guarded_sequence_grpo"
        and receipt.get("source_model") == str(source.resolve())
        and int(receipt.get("requested_steps", -1)) == steps
        and int(receipt.get("completed_steps", -1)) == steps
        and receipt.get("stopped_reason") is None
        and bool(receipt.get("shuffle_advantages")) is shuffled
        and float(receipt.get("policy_skip_rate", 1.0)) <= 0.15
    )


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
    root = resolve_repo_path(config["model"]["artifacts_root"])
    incumbent_model = root / "merged" / "incumbent_blend"
    dagger_model = root / "merged" / "dagger" / args.domain
    checkpoint_paths = {
        "dagger": (
            args.dagger,
            dagger_model,
            root / "adapters" / "dagger" / args.domain,
        ),
        "extra_sft": (
            args.extra_sft,
            root / "merged" / "extra_sft" / args.domain,
            root / "adapters" / "extra_sft" / args.domain,
        ),
        "shuffled": (
            args.shuffled,
            root / "merged" / "shuffled" / args.domain,
            root / "adapters" / "shuffled" / args.domain,
        ),
        "specialist": (
            args.specialist,
            root / "merged" / "specialist" / args.domain,
            root / "adapters" / "specialist" / args.domain,
        ),
    }
    checkpoints = {
        name: _checkpoint_integrity(*paths)
        for name, paths in checkpoint_paths.items()
    }
    dagger_training = checkpoints["dagger"]["training"]
    extra_training = checkpoints["extra_sft"]["training"]
    shuffled_training = checkpoints["shuffled"]["training"]
    specialist_training = checkpoints["specialist"]["training"]
    grpo_steps = int(config["rl_train"]["max_steps"])
    extra_steps = int(config["controls"]["matched_sft_steps"])
    forward_counts = ("reference_forwards", "policy_forwards", "anchor_forwards")
    specialist_forward_opportunities = sum(
        int(specialist_training.get("compute_ledger", {}).get(key, 0))
        for key in forward_counts
    )
    shuffled_forward_opportunities = sum(
        int(shuffled_training.get("compute_ledger", {}).get(key, 0))
        for key in forward_counts
    )
    training_integrity = {
        name: {
            key: checkpoint[key]
            for key in (
                "model",
                "adapter",
                "scores_sha256",
                "merge_receipt_sha256",
                "training_receipt_sha256",
                "structural_merge",
            )
        }
        for name, checkpoint in checkpoints.items()
    }
    training_integrity["compute"] = {
        "extra_sft_forward_opportunities": int(
            extra_training.get("forward_example_upper_bound", 0)
        ),
        "specialist_forward_opportunities": specialist_forward_opportunities,
        "shuffled_forward_opportunities": shuffled_forward_opportunities,
    }
    gate = config["gates"]
    candidate = scores["specialist"]["macro"]
    checks = {
        "all_merges_structurally_nonzero": all(
            checkpoint["structural_merge"] for checkpoint in checkpoints.values()
        ),
        "dagger_training_completed": _sft_completed(
            dagger_training,
            source=incumbent_model,
            max_steps=None,
            epochs=float(config["dagger_train"]["epochs"]),
        ),
        "extra_sft_training_completed": _sft_completed(
            extra_training,
            source=dagger_model,
            max_steps=extra_steps,
            epochs=1.0,
        ),
        "specialist_training_completed": _grpo_completed(
            specialist_training,
            source=dagger_model,
            steps=grpo_steps,
            shuffled=False,
        ),
        "shuffled_training_completed": _grpo_completed(
            shuffled_training,
            source=dagger_model,
            steps=grpo_steps,
            shuffled=True,
        ),
        "extra_sft_forward_overmatch": int(
            extra_training.get("forward_example_upper_bound", 0)
        )
        >= specialist_forward_opportunities,
        "shuffled_forward_match": shuffled_forward_opportunities
        == specialist_forward_opportunities,
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
        "nonzero_merged_delta": int(
            checkpoints["specialist"]["merge"].get("nonzero_lora_modules", 0)
        ) > 0,
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
        "training_integrity": training_integrity,
        "gate": {"passed": all(checks.values()), "checks": checks},
    }
    output = args.out or EXP / "analysis" / "specialists" / f"{args.domain}.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

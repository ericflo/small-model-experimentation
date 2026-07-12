"""Paired analysis and fail-closed verdict assignment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .config import config_sha256
from .mechanics import hierarchical_paired_bootstrap_interval


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_bundles(
    runs_dir: Path, expected_config_sha256: str
) -> list[dict[str, Any]]:
    bundles = []
    for summary_path in sorted(runs_dir.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "EVALUATION_COMPLETE":
            continue
        if summary.get("config_sha256") != expected_config_sha256:
            continue
        rows_path = Path(summary["row_file"])
        if not rows_path.is_absolute():
            rows_path = summary_path.parent / rows_path
        if not rows_path.exists():
            continue
        expected_rows_hash = summary.get("row_file_sha256")
        if expected_rows_hash is not None and _sha256(rows_path) != expected_rows_hash:
            raise RuntimeError(f"evaluation row hash mismatch: {rows_path}")
        metadata = summary.get("setup", {}).get("checkpoint_metadata", {})
        if (
            metadata.get("data_manifest_sha256") is not None
            and summary.get("data_manifest_sha256") is not None
            and metadata.get("data_manifest_sha256")
            != summary.get("data_manifest_sha256")
        ):
            raise RuntimeError(
                f"checkpoint/evaluation data-manifest mismatch: {summary_path}"
            )
        bundles.append(
            {
                "summary_path": summary_path,
                "summary": summary,
                "rows": _read_jsonl(rows_path),
                "checkpoint_metadata": metadata,
                "train_seed": metadata.get("train_seed"),
                "train_arm": summary.get("train_arm"),
                "eval_mode": summary.get("eval_mode"),
                "pilot": bool(summary.get("pilot", False)),
            }
        )
    return bundles


def _prefer_full_bundles(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Never let retained pilot rows overwrite or pool with full evaluations."""
    full = [bundle for bundle in bundles if not bundle["pilot"]]
    return full if full else bundles


def _index_rows(bundle: Mapping[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (row["id"], int(row["k"]), row["split"]): row
        for row in bundle["rows"]
    }


def _paired_carry_bag(
    bundles: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for bundle in bundles:
        seed = bundle["train_seed"]
        arm = bundle["train_arm"]
        mode = bundle["eval_mode"]
        if seed is None or arm not in {"carry", "bag"} or mode != arm:
            continue
        if arm in by_seed[int(seed)]:
            raise RuntimeError(f"duplicate full {arm} evaluation for seed {seed}")
        by_seed[int(seed)][arm] = bundle

    primary_depths = set(map(int, config["evaluation"]["primary_depths"]))
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    differences_by_seed: dict[int, list[float]] = defaultdict(list)
    per_depth_seed: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seed_receipts = []
    for seed, arms in sorted(by_seed.items()):
        if set(arms) != {"carry", "bag"}:
            continue
        carry_metadata = arms["carry"]["checkpoint_metadata"]
        bag_metadata = arms["bag"]["checkpoint_metadata"]
        allowed_k1_error = float(config["gates"]["k1_max_logit_abs_error"])
        for arm_name, bundle in arms.items():
            parity_error = bundle["summary"].get(
                "checkpoint_k1_max_logit_abs_error"
            )
            if (
                parity_error is None
                or not math.isfinite(float(parity_error))
                or float(parity_error) > allowed_k1_error
            ):
                raise RuntimeError(
                    f"missing or failed post-checkpoint K=1 parity for {arm_name} seed {seed}"
                )
        for receipt_key in (
            "data_manifest_sha256",
            "training_prompt_tokens",
            "training_layer_token_applications",
        ):
            if (
                receipt_key not in carry_metadata
                or carry_metadata.get(receipt_key) != bag_metadata.get(receipt_key)
            ):
                raise RuntimeError(
                    f"Carry/Bag training-compute receipt mismatch for seed {seed}: "
                    f"{receipt_key}"
                )
        carry_parameters = carry_metadata.get("trainable_parameters", {})
        bag_parameters = bag_metadata.get("trainable_parameters", {})
        for receipt_key in ("total", "names_sha256", "values_sha256"):
            if (
                receipt_key not in carry_parameters
                or carry_parameters.get(receipt_key) != bag_parameters.get(receipt_key)
            ):
                raise RuntimeError(
                    f"Carry/Bag initialization receipt mismatch for seed {seed}: "
                    f"{receipt_key}"
                )
        carry = _index_rows(arms["carry"])
        bag = _index_rows(arms["bag"])
        keys = sorted(set(carry).intersection(bag))
        paired = 0
        for key in keys:
            c, b = carry[key], bag[key]
            for receipt_key in ("prompt_tokens", "layer_token_applications"):
                if c.get(receipt_key) != b.get(receipt_key):
                    raise RuntimeError(
                        f"Carry/Bag evaluation-compute mismatch for seed {seed}, "
                        f"item {c['id']}: {receipt_key}"
                    )
            depth = int(c["depth"])
            if (
                c["split"] != "depth_extrapolation"
                or depth not in primary_depths
                or int(c["k"]) != depth
            ):
                continue
            difference = float(bool(c["correct"])) - float(bool(b["correct"]))
            differences_by_seed[seed].append(difference)
            per_depth_seed[depth][seed].append(difference)
            paired += 1
        seed_receipts.append(
            {
                "seed": seed,
                "paired_primary_rows": paired,
                "training_compute_equal": True,
                "initialization_equal": True,
                "post_checkpoint_k1_parity": True,
            }
        )

    if not differences_by_seed:
        return {
            "available": False,
            "reason": "no separately trained carry/bag matched-depth rows",
            "seed_receipts": seed_receipts,
        }
    mean, lower, upper = hierarchical_paired_bootstrap_interval(
        differences_by_seed, resamples=resamples, seed=bootstrap_seed
    )
    depth_results = {}
    for depth, grouped_values in sorted(per_depth_seed.items()):
        dmean, dlower, dupper = hierarchical_paired_bootstrap_interval(
            grouped_values, resamples=resamples, seed=bootstrap_seed + depth
        )
        depth_results[str(depth)] = {
            "n": sum(len(values) for values in grouped_values.values()),
            "difference": dmean,
            "ci95": [dlower, dupper],
        }
    positive_depths = sum(
        result["difference"] > 0 for result in depth_results.values()
    )
    required_per_depth = int(config["evaluation"]["min_items_per_cell"])
    required_per_seed_depth = int(
        config["substrate"]["evaluation_examples_per_split"]
    ) // len(config["substrate"]["extrapolation_depths"])
    required_seeds = set(map(int, config["training"]["train_seeds"]))
    complete_depths = sum(
        depth_results.get(str(depth), {}).get("n", 0) >= required_per_depth
        and set(per_depth_seed.get(depth, {})) == required_seeds
        and all(
            len(values) >= required_per_seed_depth
            for values in per_depth_seed.get(depth, {}).values()
        )
        for depth in primary_depths
    )
    return {
        "available": True,
        "n": sum(len(values) for values in differences_by_seed.values()),
        "train_seed_pairs": len(
            [receipt for receipt in seed_receipts if receipt["paired_primary_rows"]]
        ),
        "carry_minus_bag": mean,
        "ci95": [lower, upper],
        "per_depth": depth_results,
        "positive_depths": positive_depths,
        "required_items_per_primary_depth": required_per_depth,
        "required_items_per_seed_depth": required_per_seed_depth,
        "complete_primary_depths": complete_depths,
        "required_primary_depths": len(primary_depths),
        "seed_receipts": seed_receipts,
    }


def _unseen_k_scaling(
    bundles: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    train_k = int(config["training"]["train_k"])
    gains_by_seed: dict[int, list[float]] = defaultdict(list)
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] != "carry":
            continue
        rows_by_id: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in bundle["rows"]:
            if row["split"] == "depth_extrapolation":
                rows_by_id[row["id"]][int(row["k"])] = row
        for item in rows_by_id.values():
            depth = int(next(iter(item.values()))["depth"])
            if depth <= train_k or train_k not in item or depth not in item:
                continue
            gains_by_seed[int(bundle["train_seed"])].append(
                float(bool(item[depth]["correct"]))
                - float(bool(item[train_k]["correct"]))
            )
    if not gains_by_seed:
        return {
            "available": False,
            "reason": "no paired K=train_k versus K=depth carry rows",
        }
    mean, lower, upper = hierarchical_paired_bootstrap_interval(
        gains_by_seed,
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 100,
    )
    return {
        "available": True,
        "n": sum(len(values) for values in gains_by_seed.values()),
        "training_seeds": len(gains_by_seed),
        "gain": mean,
        "ci95": [lower, upper],
    }


def _state_sufficiency(
    bundles: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    node_values = []
    joint_values = []
    primary_depths = set(map(int, config["evaluation"]["primary_depths"]))
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] != "carry":
            continue
        for row in bundle["rows"]:
            depth = int(row["depth"])
            if (
                row["split"] == "depth_extrapolation"
                and depth in primary_depths
                and int(row["k"]) == depth
            ):
                node_values.append(float(row.get("node_step_accuracy", 0.0)))
                joint_values.append(float(row.get("joint_step_accuracy", 0.0)))
    if not node_values:
        return {"available": False}
    node = sum(node_values) / len(node_values)
    joint = sum(joint_values) / len(joint_values)
    return {
        "available": True,
        "n": len(node_values),
        "node_step_accuracy": node,
        "joint_step_accuracy": joint,
        "passes": node >= float(config["gates"]["min_state_node_accuracy"])
        or joint >= float(config["gates"]["min_state_joint_accuracy"]),
    }


def _swap_summary(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] != "carry":
            continue
        swap = bundle["summary"].get("counterfactual_swaps")
        if swap:
            summaries.append(
                {
                    "train_seed": bundle["train_seed"],
                    **swap,
                    "donor_follow_gain": swap["donor_follow_rate"]
                    - swap["recipient_preserve_rate"],
                }
            )
    return {"available": bool(summaries), "seeds": summaries}


def _edge_cut_summary(
    bundles: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair each Carry checkpoint with itself evaluated after cutting the carry edge."""
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] not in {"carry", "bag"}:
            continue
        seed = bundle["train_seed"]
        if seed is None:
            continue
        mode = str(bundle["eval_mode"])
        if mode in by_seed[int(seed)]:
            raise RuntimeError(f"duplicate full Carry-checkpoint {mode} evaluation for seed {seed}")
        by_seed[int(seed)][mode] = bundle

    primary_depths = set(map(int, config["evaluation"]["primary_depths"]))
    per_seed = []
    for seed, modes in sorted(by_seed.items()):
        if set(modes) != {"carry", "bag"}:
            continue
        intact = _index_rows(modes["carry"])
        cut = _index_rows(modes["bag"])
        differences = []
        for key in sorted(set(intact).intersection(cut)):
            intact_row, cut_row = intact[key], cut[key]
            depth = int(intact_row["depth"])
            if (
                intact_row["split"] == "depth_extrapolation"
                and depth in primary_depths
                and int(intact_row["k"]) == depth
            ):
                differences.append(
                    float(bool(intact_row["correct"]))
                    - float(bool(cut_row["correct"]))
                )
        if differences:
            per_seed.append(
                {
                    "train_seed": seed,
                    "n": len(differences),
                    "intact_minus_edge_cut": sum(differences) / len(differences),
                }
            )
    return {
        "available": bool(per_seed),
        "training_seed_pairs": len(per_seed),
        "seeds": per_seed,
        "diagnostic_only": True,
    }


def _sample_more(runs_dir: Path, expected_config_sha256: str) -> dict[str, Any]:
    rows = []
    seed_receipts = []
    seen_seeds: set[int] = set()
    sampled_tokens = 0
    generation_seconds = 0.0
    for summary_path in sorted(runs_dir.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "SAMPLE_MORE_COMPLETE":
            continue
        if summary.get("config_sha256") != expected_config_sha256:
            continue
        train_seed = int(summary["text_train_seed"])
        if train_seed in seen_seeds:
            raise RuntimeError(f"duplicate sample-more evaluation for seed {train_seed}")
        seen_seeds.add(train_seed)
        path = Path(summary["rows"])
        if not path.is_absolute():
            path = summary_path.parent / path
        if path.exists():
            expected_rows_hash = summary.get("rows_sha256")
            if expected_rows_hash is not None and _sha256(path) != expected_rows_hash:
                raise RuntimeError(f"sample-more row hash mismatch: {path}")
            seed_rows = _read_jsonl(path)
            rows.extend(seed_rows)
            seed_receipts.append({"train_seed": train_seed, "n": len(seed_rows)})
            sampled_tokens += int(summary.get("sampled_tokens", 0))
            generation_seconds += float(summary.get("generation_seconds", 0.0))
    if not rows:
        return {"available": False}
    return {
        "available": True,
        "n": len(rows),
        "majority_accuracy": sum(bool(row["majority_correct"]) for row in rows)
        / len(rows),
        "oracle_pass_at_n": sum(bool(row["pass_at_n"]) for row in rows) / len(rows),
        "parse_rate": sum(float(row["parse_rate"]) for row in rows) / len(rows),
        "training_seeds": len(seed_receipts),
        "seed_receipts": seed_receipts,
        "sampled_tokens": sampled_tokens,
        "generation_seconds": generation_seconds,
    }


def _deployment_comparison(
    bundles: list[dict[str, Any]],
    runs_dir: Path,
    expected_config_sha256: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Paired Carry-minus-explicit-CoT-oracle comparison, nested by train seed."""
    carry_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    carry_data_hashes: set[str] = set()
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] != "carry":
            continue
        seed = bundle["train_seed"]
        if seed is None:
            continue
        if int(seed) in carry_by_seed:
            raise RuntimeError(f"duplicate full Carry evaluation for seed {seed}")
        data_hash = bundle.get("summary", {}).get("data_manifest_sha256")
        if data_hash is not None:
            carry_data_hashes.add(str(data_hash))
        carry_by_seed[int(seed)] = {
            row["id"]: row
            for row in bundle["rows"]
            if row["split"] == "depth_extrapolation"
            and int(row["k"]) == int(row["depth"])
        }
    if len(carry_data_hashes) > 1:
        raise RuntimeError("Carry evaluations disagree on the data manifest")
    carry_data_hash = next(iter(carry_data_hashes), None)

    sample_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for summary_path in sorted(runs_dir.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            summary.get("status") != "SAMPLE_MORE_COMPLETE"
            or summary.get("config_sha256") != expected_config_sha256
        ):
            continue
        seed = int(summary["text_train_seed"])
        if (
            carry_data_hash is not None
            and summary.get("data_manifest_sha256") != carry_data_hash
        ):
            raise RuntimeError("Carry and sample-more data manifests differ")
        if seed in sample_by_seed:
            raise RuntimeError(f"duplicate sample-more evaluation for seed {seed}")
        path = Path(summary["rows"])
        if not path.is_absolute():
            path = summary_path.parent / path
        if path.exists():
            expected_rows_hash = summary.get("rows_sha256")
            if expected_rows_hash is not None and _sha256(path) != expected_rows_hash:
                raise RuntimeError(f"sample-more row hash mismatch: {path}")
            sample_by_seed[seed] = {row["id"]: row for row in _read_jsonl(path)}

    differences_by_seed: dict[int, list[float]] = defaultdict(list)
    seed_receipts = []
    for seed in sorted(set(carry_by_seed).intersection(sample_by_seed)):
        carry_rows = carry_by_seed[seed]
        sample_rows = sample_by_seed[seed]
        for item_id in sorted(set(carry_rows).intersection(sample_rows)):
            differences_by_seed[seed].append(
                float(bool(carry_rows[item_id]["correct"]))
                - float(bool(sample_rows[item_id]["pass_at_n"]))
            )
        seed_receipts.append(
            {"train_seed": seed, "paired_rows": len(differences_by_seed[seed])}
        )
    required_seeds = len(config["training"]["train_seeds"])
    if len(differences_by_seed) < required_seeds or any(
        not values for values in differences_by_seed.values()
    ):
        return {
            "available": False,
            "training_seed_pairs": len(differences_by_seed),
            "required_training_seed_pairs": required_seeds,
            "seed_receipts": seed_receipts,
        }
    mean, lower, upper = hierarchical_paired_bootstrap_interval(
        differences_by_seed,
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 200,
    )
    return {
        "available": True,
        "n": sum(len(values) for values in differences_by_seed.values()),
        "training_seed_pairs": len(differences_by_seed),
        "carry_minus_sample_more_oracle": mean,
        "ci95": [lower, upper],
        "seed_receipts": seed_receipts,
    }


def _write_curve_csv(path: Path, bundles: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    for bundle in bundles:
        for row in bundle["rows"]:
            groups[
                (
                    str(bundle["train_arm"]),
                    str(bundle["eval_mode"]),
                    str(row["split"]),
                    int(row["depth"]),
                    int(row["k"]),
                )
            ].append(float(bool(row["correct"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "train_arm",
                "eval_mode",
                "split",
                "depth",
                "k",
                "n",
                "accuracy",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for (train_arm, eval_mode, split, depth, k), values in sorted(groups.items()):
            writer.writerow(
                {
                    "train_arm": train_arm,
                    "eval_mode": eval_mode,
                    "split": split,
                    "depth": depth,
                    "k": k,
                    "n": len(values),
                    "accuracy": sum(values) / len(values),
                }
            )


def analyze_runs(
    config: Mapping[str, Any], runs_dir: Path, output: Path
) -> dict[str, Any]:
    bundles = _prefer_full_bundles(
        _evaluation_bundles(runs_dir, config_sha256(config))
    )
    carry_bag = _paired_carry_bag(bundles, config)
    scaling = _unseen_k_scaling(bundles, config)
    state_sufficiency = _state_sufficiency(bundles, config)
    swaps = _swap_summary(bundles)
    edge_cut = _edge_cut_summary(bundles, config)
    sample_more = _sample_more(runs_dir, config_sha256(config))
    deployment = _deployment_comparison(
        bundles, runs_dir, config_sha256(config), config
    )
    gate = config["gates"]

    if not carry_bag["available"]:
        verdict = "SETUP_ONLY"
    elif (
        carry_bag["train_seed_pairs"] < len(config["training"]["train_seeds"])
        or carry_bag["complete_primary_depths"]
        < carry_bag["required_primary_depths"]
    ):
        verdict = "UNDER_REPLICATED"
    elif (
        carry_bag["carry_minus_bag"] < float(gate["min_carry_minus_bag"])
        or carry_bag["ci95"][0] <= 0
        or carry_bag["positive_depths"] < int(gate["min_positive_primary_depths"])
    ):
        verdict = "NO_SERIAL_STATE_ADVANTAGE"
    elif (
        not scaling["available"]
        or scaling["gain"] <= 0
        or scaling["ci95"][0] <= 0
    ):
        verdict = "TRAINED_UNROLLING_ONLY"
    elif not state_sufficiency["available"] or not state_sufficiency["passes"]:
        verdict = "SERIAL_BUT_STATE_NOT_SUFFICIENT"
    elif (
        not edge_cut["available"]
        or edge_cut["training_seed_pairs"] < len(config["training"]["train_seeds"])
        or not swaps["available"]
        or len(swaps["seeds"]) < len(config["training"]["train_seeds"])
        or not all(
            seed["donor_follow_gain"] >= float(gate["min_donor_follow_gain"])
            for seed in swaps["seeds"]
        )
    ):
        verdict = "DEEP_BUT_NOT_CAUSALLY_IDENTIFIED"
    elif not sample_more["available"] or not deployment["available"]:
        verdict = "MECHANISTIC_DEPTH_POSITIVE"
    else:
        if (
            deployment["carry_minus_sample_more_oracle"] > 0
            and deployment["ci95"][0] > 0
        ):
            verdict = "DEPLOYABLE_DEPTH_BREAKTHROUGH"
        else:
            verdict = "MECHANISTIC_DEPTH_POSITIVE_SAMPLE_MORE_LOSS"

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "verdict": verdict,
        "evaluation_bundles": len(bundles),
        "carry_vs_bag": carry_bag,
        "unseen_k_scaling": scaling,
        "state_sufficiency": state_sufficiency,
        "trained_checkpoint_edge_cut": edge_cut,
        "counterfactual_swaps": swaps,
        "sample_more": sample_more,
        "deployment_comparison": deployment,
        "warning": (
            "Only DEPLOYABLE_DEPTH_BREAKTHROUGH licenses a deployment claim; "
            "mechanistic labels remain valuable but narrower."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_curve_csv(output.parent / "k_depth_curves.csv", bundles)
    return summary

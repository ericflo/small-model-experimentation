"""Pure counterfactual order-support selection and frozen gate analysis."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEPLOYABLE_NAMES = (
    "first_trace",
    "majority_real",
    "mean_real_probability",
    "max_confidence_trace",
    "minimum_entropy_trace",
)
PRIMARY_NAME = "mean_order_probability_delta"
MISMATCH_NAME = "oracle_alias_balanced_task_mismatch"
REVERSE_NAME = "reverse_order_probability_delta"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _argmin(values: list[float]) -> int:
    return min(range(len(values)), key=lambda index: (values[index], index))


def _mean(rows: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0]))]


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-30)) for value in probabilities)


def validate_and_group(
    real_rows: list[dict[str, Any]],
    shuffled_rows: list[dict[str, Any]],
    *,
    aliases: list[str],
    expected_tasks: int,
    traces_per_task: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate exact row pairing and return probability-only task matrices."""

    expected_rows = expected_tasks * traces_per_task
    if len(real_rows) != expected_rows or len(shuffled_rows) != expected_rows:
        raise ValueError("unexpected source row cardinality")
    real = {(str(row["task_id"]), int(row["trace_index"])): row for row in real_rows}
    shuffled = {
        (str(row["task_id"]), int(row["trace_index"])): row
        for row in shuffled_rows
    }
    if len(real) != expected_rows or set(real) != set(shuffled):
        raise ValueError("real/shuffled task-path keys do not pair exactly")
    alias_count = len(aliases)
    token_orders: set[tuple[int, ...]] = set()
    grouped: dict[str, dict[str, Any]] = {}
    gold: dict[str, str] = {}
    for key in sorted(real):
        task_id, trace_index = key
        row = real[key]
        control = shuffled[key]
        if row["thought_token_ids_sha256"] != control["source_thought_token_ids_sha256"]:
            raise ValueError("shuffle source does not match ordered thought")
        if row["correct_alias"] != control["correct_alias"]:
            raise ValueError("paired rows disagree on evaluation label")
        if int(row["thought_tokens"]) != int(control["thought_tokens"]):
            raise ValueError("paired rows disagree on thought length")
        if control.get("token_multiset_match") is not True:
            raise ValueError("shuffle token multiset control failed")
        if not bool(row.get("finite")) or not bool(control.get("finite")):
            raise ValueError("non-finite source row")
        ordered_values = row["alias_probabilities"]
        shuffled_values = control["alias_probabilities"]
        if not isinstance(ordered_values, dict) or not isinstance(shuffled_values, dict):
            raise ValueError("source alias probabilities must be named mappings")
        if set(ordered_values) != set(aliases) or set(shuffled_values) != set(aliases):
            raise ValueError("source alias probability names changed")
        ordered_p = [float(ordered_values[alias]) for alias in aliases]
        shuffled_p = [float(shuffled_values[alias]) for alias in aliases]
        if len(ordered_p) != alias_count or len(shuffled_p) != alias_count:
            raise ValueError("alias probability width changed")
        for probabilities in (ordered_p, shuffled_p):
            if not all(math.isfinite(value) and value >= 0.0 for value in probabilities):
                raise ValueError("invalid alias probability")
            if abs(sum(probabilities) - 1.0) > 2e-4:
                raise ValueError("constrained probabilities do not sum to one")
        token_orders.add(tuple(int(value) for value in row["alias_token_ids"]))
        token_orders.add(tuple(int(value) for value in control["alias_token_ids"]))
        ordered_choice = str(row["chosen_alias"])
        shuffled_choice = str(control["chosen_alias"])
        if ordered_choice not in aliases or shuffled_choice not in aliases:
            raise ValueError("stored choice is outside public aliases")
        if max(ordered_p) - ordered_p[aliases.index(ordered_choice)] > 1e-7:
            raise ValueError("stored ordered choice is not probability-maximal")
        if max(shuffled_p) - shuffled_p[aliases.index(shuffled_choice)] > 1e-7:
            raise ValueError("stored shuffled choice is not probability-maximal")
        if task_id not in grouped:
            grouped[task_id] = {
                "real": [None] * traces_per_task,
                "shuffled": [None] * traces_per_task,
                "real_choices": [None] * traces_per_task,
            }
            gold[task_id] = str(row["correct_alias"])
        elif gold[task_id] != str(row["correct_alias"]):
            raise ValueError("task label changes across traces")
        if trace_index < 0 or trace_index >= traces_per_task:
            raise ValueError("trace index outside frozen range")
        if grouped[task_id]["real"][trace_index] is not None:
            raise ValueError("duplicate trace index")
        grouped[task_id]["real"][trace_index] = ordered_p
        grouped[task_id]["shuffled"][trace_index] = shuffled_p
        grouped[task_id]["real_choices"][trace_index] = ordered_choice
    if len(grouped) != expected_tasks or len(token_orders) != 1:
        raise ValueError("task count or alias token order changed")
    for task_id, matrices in grouped.items():
        if any(
            row is None
            for row in matrices["real"]
            + matrices["shuffled"]
            + matrices["real_choices"]
        ):
            raise ValueError(f"incomplete task {task_id}")
        matrices["gold"] = gold[task_id]
    return grouped, {
        "tasks": len(grouped),
        "rows_per_arm": expected_rows,
        "traces_per_task": traces_per_task,
        "alias_count": alias_count,
        "alias_token_ids": list(next(iter(token_orders))),
        "paired_probability_rows": expected_rows,
        "token_multiset_controls_passed": expected_rows,
    }


def deployable_predictions(
    grouped: dict[str, dict[str, Any]], aliases: list[str]
) -> dict[str, dict[str, str]]:
    """Predict using probability matrices only; outcome fields are ignored."""

    output: dict[str, dict[str, str]] = {}
    for task_id in sorted(grouped):
        real = grouped[task_id]["real"]
        shuffled = grouped[task_id]["shuffled"]
        mean_real = _mean(real)
        mean_shuffled = _mean(shuffled)
        delta = [left - right for left, right in zip(mean_real, mean_shuffled)]
        stored_choices = grouped[task_id].get("real_choices")
        choices = (
            [aliases.index(str(alias)) for alias in stored_choices]
            if stored_choices is not None
            else [_argmax(row) for row in real]
        )
        counts = Counter(choices)
        majority = max(
            range(len(aliases)),
            key=lambda index: (counts[index], mean_real[index], -index),
        )
        max_confidence_trace = max(
            range(len(real)), key=lambda index: (max(real[index]), -index)
        )
        minimum_entropy_trace = min(
            range(len(real)), key=lambda index: (_entropy(real[index]), index)
        )
        output[task_id] = {
            PRIMARY_NAME: aliases[_argmax(delta)],
            REVERSE_NAME: aliases[_argmin(delta)],
            "first_trace": aliases[choices[0]],
            "majority_real": aliases[majority],
            "mean_real_probability": aliases[_argmax(mean_real)],
            "max_confidence_trace": aliases[choices[max_confidence_trace]],
            "minimum_entropy_trace": aliases[choices[minimum_entropy_trace]],
        }
    return output


def oracle_mismatch_predictions(
    grouped: dict[str, dict[str, Any]], aliases: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Gold-balanced mechanism control, explicitly not a deployable selector."""

    by_gold: dict[str, list[str]] = defaultdict(list)
    for task_id, task in grouped.items():
        by_gold[str(task["gold"])].append(task_id)
    donor_by_task: dict[str, str] = {}
    for gold_alias, task_ids in sorted(by_gold.items()):
        ordered = sorted(task_ids)
        if len(ordered) < 2:
            raise ValueError(f"mismatch stratum {gold_alias!r} has fewer than two tasks")
        for index, task_id in enumerate(ordered):
            donor_by_task[task_id] = ordered[(index + 1) % len(ordered)]
    predictions: dict[str, str] = {}
    for task_id in sorted(grouped):
        mean_real = _mean(grouped[task_id]["real"])
        mean_wrong_shuffle = _mean(grouped[donor_by_task[task_id]]["shuffled"])
        delta = [left - right for left, right in zip(mean_real, mean_wrong_shuffle)]
        predictions[task_id] = aliases[_argmax(delta)]
    return predictions, donor_by_task


def _stable_seed(base: int, name: str) -> int:
    digest = hashlib.blake2b(f"{base}\0{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def paired_bootstrap_lower(
    differences: list[float], *, seed: int, resamples: int
) -> float:
    generator = random.Random(seed)
    size = len(differences)
    draws = sorted(
        sum(differences[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    return float(draws[min(int(0.05 * resamples), resamples - 1)])


def analyze(
    grouped: dict[str, dict[str, Any]],
    *,
    aliases: list[str],
    gates: dict[str, Any],
    bootstrap_resamples: int,
    seed: int,
    split: str,
) -> dict[str, Any]:
    predictions = deployable_predictions(grouped, aliases)
    mismatch, donor_by_task = oracle_mismatch_predictions(grouped, aliases)
    gold = {task_id: str(task["gold"]) for task_id, task in grouped.items()}
    task_ids = sorted(grouped)
    all_names = (PRIMARY_NAME, *DEPLOYABLE_NAMES, REVERSE_NAME)
    outcomes = {
        name: {task_id: predictions[task_id][name] == gold[task_id] for task_id in task_ids}
        for name in all_names
    }
    outcomes[MISMATCH_NAME] = {
        task_id: mismatch[task_id] == gold[task_id] for task_id in task_ids
    }
    accuracies = {
        name: sum(values.values()) / len(task_ids) for name, values in outcomes.items()
    }
    primary = outcomes[PRIMARY_NAME]
    comparisons: dict[str, dict[str, Any]] = {}
    comparators = (*DEPLOYABLE_NAMES, MISMATCH_NAME)
    for name in comparators:
        differences = [
            float(primary[task_id]) - float(outcomes[name][task_id])
            for task_id in task_ids
        ]
        wins = sum(value > 0 for value in differences)
        losses = sum(value < 0 for value in differences)
        comparisons[name] = {
            "candidate_minus_comparator": sum(differences) / len(differences),
            "paired_one_sided_95_lower": paired_bootstrap_lower(
                differences,
                seed=_stable_seed(seed, f"{split}:{name}"),
                resamples=bootstrap_resamples,
            ),
            "wins": wins,
            "losses": losses,
            "ties": len(differences) - wins - losses,
        }
    candidate_predictions = {
        task_id: predictions[task_id][PRIMARY_NAME] for task_id in task_ids
    }
    successful_aliases = sorted(
        {gold[task_id] for task_id in task_ids if primary[task_id]}
    )
    predicted_aliases = sorted(set(candidate_predictions.values()))
    candidate_in_real_choice_pool = {
        task_id: candidate_predictions[task_id]
        in {aliases[_argmax(row)] for row in grouped[task_id]["real"]}
        for task_id in task_ids
    }
    maximum_candidate = float(gates["candidate_accuracy_max"])
    required_gain = float(gates["minimum_point_gain"])
    reachability = {
        name: accuracies[name] + required_gain <= maximum_candidate
        for name in comparators
    }
    gate_checks = {
        "all_comparisons_reachable": all(reachability.values()),
        "candidate_accuracy_range": float(gates["candidate_accuracy_min"])
        <= accuracies[PRIMARY_NAME]
        <= maximum_candidate,
        "all_point_gains": all(
            comparisons[name]["candidate_minus_comparator"] >= required_gain
            for name in comparators
        ),
        "all_paired_lowers": all(
            comparisons[name]["paired_one_sided_95_lower"]
            > float(gates["paired_one_sided_95_lower_min"])
            for name in comparators
        ),
        "chosen_alias_support": len(predicted_aliases)
        >= int(gates["chosen_alias_support_min"]),
        "correct_alias_support": len(successful_aliases)
        >= int(gates["correct_alias_support_min"]),
        "reverse_delta_gap": accuracies[PRIMARY_NAME] - accuracies[REVERSE_NAME]
        >= float(gates["reverse_delta_gap_min"]),
    }
    if not gate_checks["all_comparisons_reachable"]:
        decision = "GATE_INFEASIBLE"
        passed = False
    else:
        passed = all(gate_checks.values())
        if split == "qualification":
            decision = "ORDER_SUPPORT_QUALIFIED" if passed else "NO_ORDER_SUPPORT_SELECTOR"
        else:
            decision = (
                "RETROSPECTIVE_ORDER_SUPPORT_REPLICATED"
                if passed
                else "ORDER_SUPPORT_CONFIRMATION_FAIL"
            )
    task_rows = []
    for task_id in task_ids:
        row = {
            "task_id": task_id,
            "correct_alias": gold[task_id],
            "candidate_in_real_choice_pool": candidate_in_real_choice_pool[task_id],
            "oracle_mismatch_donor_task": donor_by_task[task_id],
        }
        for name in all_names:
            row[f"prediction_{name}"] = predictions[task_id][name]
            row[f"correct_{name}"] = outcomes[name][task_id]
        row[f"prediction_{MISMATCH_NAME}"] = mismatch[task_id]
        row[f"correct_{MISMATCH_NAME}"] = outcomes[MISMATCH_NAME][task_id]
        task_rows.append(row)
    return {
        "split": split,
        "decision": decision,
        "passed": passed,
        "scientific_scope": "retrospective_selector_signal_not_matched_compute_capability",
        "primary": PRIMARY_NAME,
        "accuracies": accuracies,
        "comparisons": comparisons,
        "reachability": reachability,
        "gate_checks": gate_checks,
        "candidate_predicted_aliases": predicted_aliases,
        "candidate_successful_aliases": successful_aliases,
        "candidate_in_real_choice_pool_rate": sum(candidate_in_real_choice_pool.values())
        / len(task_ids),
        "tasks": len(task_ids),
        "task_rows": task_rows,
    }

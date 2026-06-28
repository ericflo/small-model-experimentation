from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LETTERS = list("ABCDEFGH")


@dataclass(frozen=True)
class Operator:
    index: int
    name: str
    signature: str
    description: str
    family: str
    params: dict[str, int]

    def apply(self, xs: list[int]) -> int:
        p = self.params
        if self.family == "sum_mod":
            return (sum(xs) + p["bias"]) % p["mod"]
        if self.family == "weighted_mod":
            return (sum((i + 1 + p["offset"]) * x for i, x in enumerate(xs)) + p["bias"]) % p["mod"]
        if self.family == "count_ge":
            return sum(1 for x in xs if x >= p["threshold"])
        if self.family == "count_mod":
            return sum(1 for x in xs if x % p["mod"] == p["remainder"])
        if self.family == "max_shift":
            return max(xs) + p["shift"]
        if self.family == "min_shift":
            return min(xs) + p["shift"]
        if self.family == "sorted_pick":
            return sorted(xs)[p["index"]] + p["shift"]
        if self.family == "range_mod":
            return (max(xs) - min(xs) + p["bias"]) % p["mod"]
        if self.family == "pair_delta":
            return abs(xs[p["i"]] - xs[p["j"]]) + p["shift"]
        if self.family == "prefix_mod":
            return (sum(xs[: p["width"]]) + p["bias"]) % p["mod"]
        raise ValueError(f"unknown operator family: {self.family}")


def build_operator_library(size: int) -> list[Operator]:
    """Build a deterministic typed list[int]->int operator inventory."""
    families: list[tuple[str, dict[str, int], str]] = []
    mods = [5, 7, 9, 11, 13, 17, 19, 23]
    for mod in mods:
        for bias in range(mod):
            families.append(("sum_mod", {"mod": mod, "bias": bias}, f"(sum(xs)+{bias}) mod {mod}"))
    for mod in mods:
        for offset in range(3):
            for bias in range(0, mod, max(1, mod // 5)):
                families.append(
                    (
                        "weighted_mod",
                        {"mod": mod, "offset": offset, "bias": bias},
                        f"(weighted_sum_offset_{offset}(xs)+{bias}) mod {mod}",
                    )
                )
    for threshold in range(1, 13):
        families.append(("count_ge", {"threshold": threshold}, f"count values >= {threshold}"))
    for mod in [2, 3, 4, 5, 6, 7]:
        for remainder in range(mod):
            families.append(("count_mod", {"mod": mod, "remainder": remainder}, f"count values == {remainder} mod {mod}"))
    for shift in range(-4, 9):
        families.append(("max_shift", {"shift": shift}, f"max(xs)+{shift}"))
        families.append(("min_shift", {"shift": shift}, f"min(xs)+{shift}"))
    for index in range(5):
        for shift in range(-3, 6):
            families.append(("sorted_pick", {"index": index, "shift": shift}, f"sorted(xs)[{index}]+{shift}"))
    for mod in mods:
        for bias in range(0, mod, max(1, mod // 4)):
            families.append(("range_mod", {"mod": mod, "bias": bias}, f"(range(xs)+{bias}) mod {mod}"))
    for i in range(5):
        for j in range(i + 1, 5):
            for shift in range(0, 5):
                families.append(("pair_delta", {"i": i, "j": j, "shift": shift}, f"abs(xs[{i}]-xs[{j}])+{shift}"))
    for width in range(1, 6):
        for mod in mods:
            for bias in range(0, mod, max(1, mod // 3)):
                families.append(("prefix_mod", {"width": width, "mod": mod, "bias": bias}, f"(sum(xs[:{width}])+{bias}) mod {mod}"))

    if size > len(families):
        raise ValueError(f"requested {size} operators but only {len(families)} are defined")
    operators: list[Operator] = []
    for index, (family, params, desc) in enumerate(families[:size]):
        operators.append(
            Operator(
                index=index,
                name=f"op{index:03d}_{family}",
                signature="list[int] -> int",
                description=desc,
                family=family,
                params=params,
            )
        )
    return operators


def make_case(rng: random.Random) -> dict[str, list[int]]:
    return {
        "a": [rng.randint(0, 12) for _ in range(5)],
        "b": [rng.randint(0, 12) for _ in range(5)],
    }


def eval_pair(record: dict[str, Any], operators: list[Operator], left: int, right: int, case: dict[str, list[int]]) -> Any:
    lv = operators[left].apply(case["a"])
    rv = operators[right].apply(case["b"])
    params = record["template_params"]
    if record["template"] == "pair_affine_mod":
        return int((params["left_scale"] * lv + params["right_scale"] * rv + params["bias"]) % params["mod"])
    if record["template"] == "pair_compare_gate":
        margin = lv - rv - params["offset"]
        if margin > 0:
            return "LEFT"
        if margin < 0:
            return "RIGHT"
        return "TIE"
    raise ValueError(f"unknown template: {record['template']}")


def case_to_text(case: dict[str, list[int]]) -> str:
    return f"a={case['a']} b={case['b']}"


def template_text(record: dict[str, Any]) -> str:
    p = record["template_params"]
    if record["template"] == "pair_affine_mod":
        return f"output = ({p['left_scale']}*LEFT(a) + {p['right_scale']}*RIGHT(b) + {p['bias']}) mod {p['mod']}"
    if record["template"] == "pair_compare_gate":
        return f"output = compare(LEFT(a) - RIGHT(b) - {p['offset']}) as LEFT/RIGHT/TIE"
    raise ValueError(f"unknown template: {record['template']}")


def operator_values(operators: list[Operator], cases: list[dict[str, list[int]]], side: str) -> np.ndarray:
    values = np.zeros((len(operators), len(cases)), dtype=np.int16)
    key = "a" if side == "left" else "b"
    for i, op in enumerate(operators):
        for j, case in enumerate(cases):
            values[i, j] = op.apply(case[key])
    return values


def pair_output_matrix(record: dict[str, Any], left_values: np.ndarray, right_values: np.ndarray, case_index: int) -> np.ndarray:
    p = record["template_params"]
    lv = left_values[:, case_index][:, None].astype(np.int32)
    rv = right_values[:, case_index][None, :].astype(np.int32)
    if record["template"] == "pair_affine_mod":
        return ((p["left_scale"] * lv + p["right_scale"] * rv + p["bias"]) % p["mod"]).astype(np.int16)
    if record["template"] == "pair_compare_gate":
        margin = lv - rv - p["offset"]
        out = np.zeros_like(margin, dtype=np.int8)
        out[margin > 0] = 1
        out[margin < 0] = -1
        return out
    raise ValueError(f"unknown template: {record['template']}")


def encoded_output(value: Any) -> int:
    if value == "LEFT":
        return 1
    if value == "RIGHT":
        return -1
    if value == "TIE":
        return 0
    return int(value)


def decoded_output(value: int, template: str) -> Any:
    if template == "pair_compare_gate":
        return {1: "LEFT", -1: "RIGHT", 0: "TIE"}[int(value)]
    return int(value)


def all_cases(record: dict[str, Any]) -> list[dict[str, list[int]]]:
    return record["visible_cases"] + record["query_pool"] + record["hidden_cases"]


def observation_indices(record: dict[str, Any], query_indices: Iterable[int] = ()) -> list[int]:
    return list(range(len(record["visible_cases"]))) + [len(record["visible_cases"]) + idx for idx in query_indices]


def candidate_mask(record: dict[str, Any], operators: list[Operator], query_indices: Iterable[int] = ()) -> np.ndarray:
    n = record["library_size"]
    ops = operators[:n]
    cases = all_cases(record)
    left_values = operator_values(ops, cases, "left")
    right_values = operator_values(ops, cases, "right")
    mask = np.ones((n, n), dtype=bool)
    target_left, target_right = record["target_pair"]
    for case_index in observation_indices(record, query_indices):
        mat = pair_output_matrix(record, left_values, right_values, case_index)
        target_y = mat[target_left, target_right]
        mask &= mat == target_y
    return mask


def output_buckets_for_query(
    record: dict[str, Any],
    operators: list[Operator],
    mask: np.ndarray,
    query_index: int,
) -> tuple[dict[Any, int], np.ndarray]:
    n = record["library_size"]
    ops = operators[:n]
    cases = all_cases(record)
    case_index = len(record["visible_cases"]) + query_index
    left_values = operator_values(ops, cases, "left")
    right_values = operator_values(ops, cases, "right")
    mat = pair_output_matrix(record, left_values, right_values, case_index)
    vals, counts = np.unique(mat[mask], return_counts=True)
    buckets = {decoded_output(int(v), record["template"]): int(c) for v, c in zip(vals, counts)}
    return buckets, mat


def bucket_stats(buckets: dict[Any, int]) -> dict[str, float | int | str]:
    total = sum(buckets.values())
    if total <= 0:
        return {"unique": 0, "largest": 0, "expected_remaining": 0.0, "entropy": 0.0, "top": "{}"}
    probs = [count / total for count in buckets.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    expected_remaining = sum(count * count for count in buckets.values()) / total
    ordered = sorted(buckets.items(), key=lambda kv: (-kv[1], str(kv[0])))[:6]
    top = "{" + ", ".join(f"{k}:{v}" for k, v in ordered) + "}"
    return {
        "unique": len(buckets),
        "largest": max(buckets.values()),
        "expected_remaining": float(expected_remaining),
        "entropy": float(entropy),
        "top": top,
    }


def hidden_equivalent_count(record: dict[str, Any], operators: list[Operator], mask: np.ndarray) -> int:
    n = record["library_size"]
    ops = operators[:n]
    hidden = record["hidden_cases"]
    if not hidden:
        return 0
    left_values = operator_values(ops, hidden, "left")
    right_values = operator_values(ops, hidden, "right")
    target_left, target_right = record["target_pair"]
    equiv = mask.copy()
    for i in range(len(hidden)):
        mat = pair_output_matrix(record, left_values, right_values, i)
        equiv &= mat == mat[target_left, target_right]
    return int(equiv.sum())


def first_prior_pair(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    left, right = coords[0]
    return int(left), int(right)


def pair_hidden_matches(record: dict[str, Any], operators: list[Operator], pair: tuple[int, int] | None) -> bool:
    if pair is None:
        return False
    left, right = pair
    target_left, target_right = record["target_pair"]
    for case in record["hidden_cases"]:
        if eval_pair(record, operators, left, right, case) != eval_pair(record, operators, target_left, target_right, case):
            return False
    return True


def action_diagnostics(
    record: dict[str, Any],
    operators: list[Operator],
    query_indices: list[int],
    used_query_indices: list[int],
) -> dict[str, Any]:
    mask = candidate_mask(record, operators, used_query_indices)
    before = int(mask.sum())
    rows: list[dict[str, Any]] = []
    target_left, target_right = record["target_pair"]
    n = record["library_size"]
    ops = operators[:n]
    cases = all_cases(record)
    left_values = operator_values(ops, cases, "left")
    right_values = operator_values(ops, cases, "right")
    for letter, qidx in zip(LETTERS, query_indices):
        case_index = len(record["visible_cases"]) + qidx
        mat = pair_output_matrix(record, left_values, right_values, case_index)
        vals, counts = np.unique(mat[mask], return_counts=True)
        buckets = {decoded_output(int(v), record["template"]): int(c) for v, c in zip(vals, counts)}
        target_y = mat[target_left, target_right]
        after_mask = mask & (mat == target_y)
        after = int(after_mask.sum())
        stats = bucket_stats(buckets)
        reward = math.log2(max(before, 1) / max(after, 1))
        rows.append(
            {
                "letter": letter,
                "query_index": qidx,
                "case": record["query_pool"][qidx],
                "true_output": decoded_output(int(target_y), record["template"]),
                "survivors_if_taken": after,
                "reward": reward,
                **stats,
            }
        )
    best = max(rows, key=lambda row: (row["reward"], -row["expected_remaining"], row["entropy"]))
    return {
        "candidate_count": before,
        "used_query_indices": list(used_query_indices),
        "actions": rows,
        "oracle_action": best["letter"],
        "oracle_query_index": best["query_index"],
    }


def rank_queries_by_expected_split(
    record: dict[str, Any],
    operators: list[Operator],
    used_query_indices: list[int],
    candidate_query_indices: list[int],
) -> list[dict[str, Any]]:
    """Rank deployable probe candidates without using the target output."""
    mask = candidate_mask(record, operators, used_query_indices)
    n = record["library_size"]
    ops = operators[:n]
    cases = all_cases(record)
    left_values = operator_values(ops, cases, "left")
    right_values = operator_values(ops, cases, "right")
    rows: list[dict[str, Any]] = []
    for qidx in candidate_query_indices:
        case_index = len(record["visible_cases"]) + qidx
        mat = pair_output_matrix(record, left_values, right_values, case_index)
        vals, counts = np.unique(mat[mask], return_counts=True)
        buckets = {decoded_output(int(v), record["template"]): int(c) for v, c in zip(vals, counts)}
        stats = bucket_stats(buckets)
        rows.append(
            {
                "query_index": qidx,
                "case": record["query_pool"][qidx],
                **stats,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["expected_remaining"]),
            int(row["largest"]),
            -float(row["entropy"]),
            int(row["query_index"]),
        ),
    )


def select_action_queries(
    record: dict[str, Any],
    operators: list[Operator],
    used_query_indices: list[int],
    source: str,
    action_count: int = 8,
) -> list[int]:
    """Select the displayed action set for a process state."""
    used = set(used_query_indices)
    candidates = [idx for idx in range(len(record["query_pool"])) if idx not in used]
    if len(candidates) < action_count:
        return []
    if source == "random8":
        return candidates[:action_count]
    if source == "mined8":
        ranked = rank_queries_by_expected_split(record, operators, used_query_indices, candidates)
        chosen = [int(row["query_index"]) for row in ranked[:action_count]]
        # Randomize display order deterministically so "A" is not always the
        # target-independent max-split action.
        seed = f"{record['record_id']}:{len(used_query_indices)}:{source}"
        rng = random.Random(seed)
        rng.shuffle(chosen)
        return chosen
    raise ValueError(f"unknown action source: {source}")


def full_pool_choice(
    record: dict[str, Any],
    operators: list[Operator],
    used_query_indices: list[int],
    policy: str,
) -> dict[str, Any]:
    """Choose one query from the full remaining pool for non-model baselines."""
    used = set(used_query_indices)
    candidates = [idx for idx in range(len(record["query_pool"])) if idx not in used]
    if not candidates:
        raise ValueError("no remaining queries")
    if policy == "fullpool_max_split":
        row = rank_queries_by_expected_split(record, operators, used_query_indices, candidates)[0]
        return {"query_index": int(row["query_index"]), "reward": None, "oracle_query_index": None}
    if policy == "fullpool_oracle":
        mask = candidate_mask(record, operators, used_query_indices)
        before = int(mask.sum())
        target_left, target_right = record["target_pair"]
        n = record["library_size"]
        ops = operators[:n]
        cases = all_cases(record)
        left_values = operator_values(ops, cases, "left")
        right_values = operator_values(ops, cases, "right")
        rows: list[dict[str, Any]] = []
        for qidx in candidates:
            case_index = len(record["visible_cases"]) + qidx
            mat = pair_output_matrix(record, left_values, right_values, case_index)
            vals, counts = np.unique(mat[mask], return_counts=True)
            buckets = {decoded_output(int(v), record["template"]): int(c) for v, c in zip(vals, counts)}
            target_y = mat[target_left, target_right]
            after = int((mask & (mat == target_y)).sum())
            stats = bucket_stats(buckets)
            rows.append(
                {
                    "query_index": qidx,
                    "reward": math.log2(max(before, 1) / max(after, 1)),
                    **stats,
                }
            )
        best = max(rows, key=lambda row: (row["reward"], -row["expected_remaining"], row["entropy"]))
        return {
            "query_index": int(best["query_index"]),
            "reward": float(best["reward"]),
            "oracle_query_index": int(best["query_index"]),
        }
    raise ValueError(policy)


def generate_record(
    rng: random.Random,
    record_id: str,
    split: str,
    library_size: int,
    template: str,
    min_initial_candidates: int,
    query_pool_cases: int = 96,
    max_attempts: int = 200,
) -> dict[str, Any]:
    operators = build_operator_library(library_size)
    for attempt in range(max_attempts):
        if template == "pair_affine_mod":
            params = {
                "left_scale": rng.choice([2, 3, 5, 7]),
                "right_scale": rng.choice([3, 4, 6, 8]),
                "bias": rng.randint(0, 16),
                "mod": rng.choice([17, 19, 23, 29]),
            }
        elif template == "pair_compare_gate":
            params = {"offset": rng.choice([-2, -1, 0, 1, 2])}
        else:
            raise ValueError(template)
        record = {
            "record_id": record_id,
            "split": split,
            "library_size": library_size,
            "template": template,
            "template_params": params,
            "target_pair": [rng.randrange(library_size), rng.randrange(library_size)],
            "visible_cases": [make_case(rng) for _ in range(4)],
            "query_pool": [make_case(rng) for _ in range(query_pool_cases)],
            "hidden_cases": [make_case(rng) for _ in range(16)],
            "generation_attempt": attempt,
        }
        mask = candidate_mask(record, operators, [])
        if int(mask.sum()) >= min_initial_candidates and mask[tuple(record["target_pair"])]:
            return record
    return record


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

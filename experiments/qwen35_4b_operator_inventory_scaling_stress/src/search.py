from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operator_library import OperatorSpec
from .tasks import program_for


@dataclass(frozen=True)
class PreparedRecord:
    outputs: np.ndarray
    expected: np.ndarray
    m: np.ndarray
    offset: np.ndarray
    threshold: np.ndarray
    visible_idx: np.ndarray
    hidden_idx: np.ndarray
    query_idx: np.ndarray


def prepare_record(record: dict[str, Any], operators: list[OperatorSpec]) -> PreparedRecord:
    cases = record["visible"] + record["hidden"] + record["query_pool"]
    outputs = np.zeros((record["library_size"], len(cases)), dtype=np.int64)
    for case_index, case in enumerate(cases):
        xs = case["input"]["xs"]
        for op_index, operator in enumerate(operators[: record["library_size"]]):
            outputs[op_index, case_index] = operator.eval(xs)
    expected = np.array([case["expected"] for case in cases], dtype=np.int64)
    m = np.array([case["input"].get("m", 1) for case in cases], dtype=np.int64)
    offset = np.array([case["input"].get("offset", 0) for case in cases], dtype=np.int64)
    threshold = np.array([case["input"].get("threshold", 0) for case in cases], dtype=np.int64)
    visible_count = len(record["visible"])
    hidden_count = len(record["hidden"])
    return PreparedRecord(
        outputs=outputs,
        expected=expected,
        m=m,
        offset=offset,
        threshold=threshold,
        visible_idx=np.arange(0, visible_count),
        hidden_idx=np.arange(visible_count, visible_count + hidden_count),
        query_idx=np.arange(visible_count + hidden_count, len(cases)),
    )


def raw_candidate_ids(library_size: int, hole_count: int) -> np.ndarray:
    if hole_count == 1:
        return np.arange(library_size, dtype=np.int64)
    return np.arange(library_size * library_size, dtype=np.int64)


def target_candidate_id(record: dict[str, Any]) -> int:
    indexes = record["target_operator_indexes"]
    library_size = record["library_size"]
    if len(indexes) == 1:
        return int(indexes[0])
    return int(indexes[0] * library_size + indexes[1])


def candidate_operator_tuple(candidate_id: int, library_size: int, hole_count: int) -> tuple[int, ...]:
    if candidate_id < 0:
        return ()
    if hole_count == 1:
        return (candidate_id,)
    return (candidate_id // library_size, candidate_id % library_size)


def candidate_program(candidate_id: int, record: dict[str, Any], operators: list[OperatorSpec]) -> str:
    indexes = candidate_operator_tuple(candidate_id, record["library_size"], record["hole_count"])
    if not indexes:
        return ""
    names = tuple(operators[index].name for index in indexes)
    return program_for(record["template"], names)


def candidate_values(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray, case_index: int) -> np.ndarray:
    template = record["template"]
    library_size = record["library_size"]
    if record["hole_count"] == 1:
        op = prep.outputs[candidate_ids, case_index]
        if template == "single_mod":
            return op % prep.m[case_index]
        if template == "single_offset":
            return op + prep.offset[case_index]
        raise KeyError(template)

    left = candidate_ids // library_size
    right = candidate_ids % library_size
    left_values = prep.outputs[left, case_index]
    right_values = prep.outputs[right, case_index]
    if template == "pair_affine_mod":
        return (3 * left_values + right_values) % prep.m[case_index]
    if template == "pair_compare_gate":
        return ((left_values - right_values) > prep.threshold[case_index]).astype(np.int64)
    raise KeyError(template)


def pass_count(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray, case_indexes: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(candidate_ids), dtype=np.int16)
    for case_index in case_indexes:
        counts += candidate_values(record, prep, candidate_ids, int(case_index)) == prep.expected[case_index]
    return counts


def visible_consistent_ids(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray) -> np.ndarray:
    mask = np.ones(len(candidate_ids), dtype=bool)
    for case_index in prep.visible_idx:
        mask &= candidate_values(record, prep, candidate_ids, int(case_index)) == prep.expected[case_index]
    return candidate_ids[mask]


def choose_query(
    record: dict[str, Any],
    prep: PreparedRecord,
    candidate_ids: np.ndarray,
    used_query_positions: set[int],
    *,
    oracle: bool,
) -> int | None:
    if len(candidate_ids) <= 1:
        return None
    best_score: tuple[int, int, int, int] | None = None
    best_position: int | None = None
    for query_position, case_index in enumerate(prep.query_idx):
        if query_position in used_query_positions:
            continue
        values = candidate_values(record, prep, candidate_ids, int(case_index))
        unique, counts = np.unique(values, return_counts=True)
        true_value = prep.expected[case_index]
        true_matches = counts[unique == true_value]
        true_bucket = int(true_matches[0]) if len(true_matches) else 0
        largest_bucket = int(counts.max()) if len(counts) else 0
        if oracle:
            score = (len(candidate_ids) - true_bucket, len(unique), -largest_bucket, -query_position)
        else:
            score = (len(candidate_ids) - largest_bucket, len(unique), -largest_bucket, -query_position)
        if best_score is None or score > best_score:
            best_score = score
            best_position = query_position
    return best_position


def selected_row(
    record: dict[str, Any],
    prep: PreparedRecord,
    candidate_ids: np.ndarray,
    operators: list[OperatorSpec],
) -> dict[str, Any]:
    if len(candidate_ids) == 0:
        return {
            "selected_candidate_id": -1,
            "selected_program": "",
            "selected_hidden_passes": 0,
            "selected_hidden_all": False,
        }
    selected = int(candidate_ids[0])
    hidden_passes = int(pass_count(record, prep, np.array([selected], dtype=np.int64), prep.hidden_idx)[0])
    return {
        "selected_candidate_id": selected,
        "selected_program": candidate_program(selected, record, operators),
        "selected_hidden_passes": hidden_passes,
        "selected_hidden_all": hidden_passes == len(prep.hidden_idx),
    }


def evaluate_candidates(record: dict[str, Any], operators: list[OperatorSpec]) -> dict[str, Any]:
    library_size = record["library_size"]
    raw_ids = raw_candidate_ids(library_size, record["hole_count"])
    prep = prepare_record(record, operators)
    visible_ids = visible_consistent_ids(record, prep, raw_ids)
    target_id = target_candidate_id(record)
    hidden_passes = pass_count(record, prep, visible_ids, prep.hidden_idx) if len(visible_ids) else np.array([], dtype=np.int16)
    selected = selected_row(record, prep, visible_ids, operators)
    unique_operator_ids: set[int] = set()
    for candidate_id in visible_ids:
        unique_operator_ids.update(candidate_operator_tuple(int(candidate_id), library_size, record["hole_count"]))
    prefix_budgets = [1024, 4096, 16384]
    return {
        "raw_candidate_count": int(len(raw_ids)),
        "target_candidate_id": int(target_id),
        "target_rank": int(target_id + 1),
        "target_in_raw_candidates": True,
        "target_in_visible_candidates": bool(np.any(visible_ids == target_id)),
        "candidate_oracle_hidden_passes": int(hidden_passes.max()) if len(hidden_passes) else 0,
        "candidate_oracle_hidden_all": bool(np.any(hidden_passes == len(prep.hidden_idx))) if len(hidden_passes) else False,
        "visible_consistent_count": int(len(visible_ids)),
        "visible_consistent_operator_count": int(len(unique_operator_ids)),
        "visible_consistent_tuple_count": int(len(visible_ids)),
        "prefix_budget_hits": {str(budget): bool(target_id < budget) for budget in prefix_budgets},
        **selected,
    }


def active_trace(
    record: dict[str, Any],
    operators: list[OperatorSpec],
    *,
    budgets: list[int],
    policy: str,
) -> list[dict[str, Any]]:
    library_size = record["library_size"]
    prep = prepare_record(record, operators)
    raw_ids = raw_candidate_ids(library_size, record["hole_count"])
    candidate_ids = visible_consistent_ids(record, prep, raw_ids)
    used_query_positions: set[int] = set()
    rows: list[dict[str, Any]] = []
    max_budget = max(budgets)
    oracle = policy == "oracle_elimination"

    for step in range(max_budget + 1):
        selected = selected_row(record, prep, candidate_ids, operators)
        if step in budgets:
            rows.append(
                {
                    "budget": step,
                    "policy": policy,
                    "queries_used": len(used_query_positions),
                    "candidate_count": int(len(candidate_ids)),
                    "selected_hidden_all": selected["selected_hidden_all"],
                    "selected_hidden_passes": selected["selected_hidden_passes"],
                    "selected_candidate_id": selected["selected_candidate_id"],
                    "observed_count": len(prep.visible_idx) + len(used_query_positions),
                }
            )
        if step == max_budget:
            break
        query_position = choose_query(record, prep, candidate_ids, used_query_positions, oracle=oracle)
        if query_position is None:
            continue
        used_query_positions.add(query_position)
        case_index = int(prep.query_idx[query_position])
        values = candidate_values(record, prep, candidate_ids, case_index)
        candidate_ids = candidate_ids[values == prep.expected[case_index]]
    return rows


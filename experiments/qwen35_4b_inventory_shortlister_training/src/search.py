from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operator_library import OperatorSpec
from .tasks import template_eval


@dataclass(frozen=True)
class PreparedRecord:
    outputs: np.ndarray
    expected: np.ndarray
    m: np.ndarray
    threshold: np.ndarray
    visible_idx: np.ndarray
    hidden_idx: np.ndarray
    query_idx: np.ndarray


def prepare_record(record: dict[str, Any], operators: list[OperatorSpec]) -> PreparedRecord:
    cases = record["visible"] + record["hidden"] + record["query_pool"]
    library_size = record["library_size"]
    outputs = np.zeros((library_size, len(cases)), dtype=np.int64)
    for case_index, case in enumerate(cases):
        xs = case["input"]["xs"]
        for op_index, operator in enumerate(operators[:library_size]):
            outputs[op_index, case_index] = operator.eval(xs)
    expected = np.array([case["expected"] for case in cases], dtype=np.int64)
    m = np.array([case["input"].get("m", 1) for case in cases], dtype=np.int64)
    threshold = np.array([case["input"].get("threshold", 0) for case in cases], dtype=np.int64)
    visible_count = len(record["visible"])
    hidden_count = len(record["hidden"])
    return PreparedRecord(
        outputs=outputs,
        expected=expected,
        m=m,
        threshold=threshold,
        visible_idx=np.arange(0, visible_count),
        hidden_idx=np.arange(visible_count, visible_count + hidden_count),
        query_idx=np.arange(visible_count + hidden_count, len(cases)),
    )


def raw_candidate_ids(library_size: int) -> np.ndarray:
    return np.arange(library_size * library_size, dtype=np.int64)


def target_candidate_id(record: dict[str, Any]) -> int:
    return int(record["target_left_index"] * record["library_size"] + record["target_right_index"])


def candidate_values(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray, case_index: int) -> np.ndarray:
    library_size = record["library_size"]
    left = candidate_ids // library_size
    right = candidate_ids % library_size
    left_values = prep.outputs[left, case_index]
    right_values = prep.outputs[right, case_index]
    if record["template"] == "pair_affine_mod":
        return (3 * left_values + right_values) % prep.m[case_index]
    if record["template"] == "pair_compare_gate":
        return ((left_values - right_values) > prep.threshold[case_index]).astype(np.int64)
    raise KeyError(record["template"])


def visible_consistent_ids(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    mask = np.ones(len(candidate_ids), dtype=bool)
    for case_index in indexes:
        mask &= candidate_values(record, prep, candidate_ids, int(case_index)) == prep.expected[case_index]
    return candidate_ids[mask]


def pass_count(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(candidate_ids), dtype=np.int16)
    for case_index in indexes:
        counts += candidate_values(record, prep, candidate_ids, int(case_index)) == prep.expected[case_index]
    return counts


def selected_hidden_all(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray) -> bool:
    if len(candidate_ids) == 0:
        return False
    selected = np.array([int(candidate_ids[0])], dtype=np.int64)
    return bool(pass_count(record, prep, selected, prep.hidden_idx)[0] == len(prep.hidden_idx))


def exhaustive_summary(record: dict[str, Any], operators: list[OperatorSpec]) -> dict[str, Any]:
    raw_ids = raw_candidate_ids(record["library_size"])
    prep = prepare_record(record, operators)
    visible_ids = visible_consistent_ids(record, prep, raw_ids, prep.visible_idx)
    hidden_passes = pass_count(record, prep, visible_ids, prep.hidden_idx) if len(visible_ids) else np.array([], dtype=np.int16)
    target_id = target_candidate_id(record)
    return {
        "raw_candidate_count": int(len(raw_ids)),
        "visible_consistent_count": int(len(visible_ids)),
        "target_in_visible": bool(np.any(visible_ids == target_id)),
        "oracle_hidden_all": bool(np.any(hidden_passes == len(prep.hidden_idx))) if len(hidden_passes) else False,
        "selected_hidden_all": selected_hidden_all(record, prep, visible_ids),
    }


def choose_split_query(record: dict[str, Any], prep: PreparedRecord, candidate_ids: np.ndarray, available_indexes: list[int]) -> int | None:
    if len(candidate_ids) <= 1:
        return None
    best_score: tuple[int, int, int] | None = None
    best_index: int | None = None
    for case_index in available_indexes:
        values = candidate_values(record, prep, candidate_ids, case_index)
        unique, counts = np.unique(values, return_counts=True)
        largest_bucket = int(counts.max()) if len(counts) else 0
        score = (len(candidate_ids) - largest_bucket, len(unique), -case_index)
        if best_score is None or score > best_score:
            best_score = score
            best_index = case_index
    return best_index


def designed_probe_summary(record: dict[str, Any], operators: list[OperatorSpec], *, budget: int = 6) -> dict[str, Any]:
    raw_ids = raw_candidate_ids(record["library_size"])
    prep = prepare_record(record, operators)
    candidates = raw_ids
    available = list(map(int, prep.query_idx))
    chosen: list[int] = []
    for _ in range(budget):
        query_index = choose_split_query(record, prep, candidates, available)
        if query_index is None:
            break
        available.remove(query_index)
        chosen.append(query_index)
        values = candidate_values(record, prep, candidates, query_index)
        candidates = candidates[values == prep.expected[query_index]]
    target_id = target_candidate_id(record)
    hidden_passes = pass_count(record, prep, candidates, prep.hidden_idx) if len(candidates) else np.array([], dtype=np.int16)
    return {
        "designed_queries": len(chosen),
        "designed_visible_consistent_count": int(len(candidates)),
        "designed_target_in_visible": bool(np.any(candidates == target_id)),
        "designed_oracle_hidden_all": bool(np.any(hidden_passes == len(prep.hidden_idx))) if len(hidden_passes) else False,
        "designed_selected_hidden_all": selected_hidden_all(record, prep, candidates),
    }


#!/usr/bin/env python3
"""Content-blind termination gates and post-gate semantic smoke analysis.

Termination selection reads token counts, finish metadata, and token-id periodicity
only.  Decoded answers and hidden examples are read only by ``analyze_matrix``
after both fresh K=12 receipts at one rung verify and pass termination.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import macro_domain as domain  # noqa: E402
import model_harness as harness  # noqa: E402
import scientific_artifacts as store  # noqa: E402


ANSWER_MAX_TOKENS = 512
LOOP_TAIL_TOKENS = 8192
LOOP_MAX_PERIOD_TOKENS = 2048
LOOP_MIN_MATCH_RATE = 0.99
MAX_UNRESOLVED_RATE = 0.05
MAX_ANSWER_LIMIT_RATE = 0.05
MAX_LOOP_RATE = 0.25
MAX_SURFACE_CALLS = 5
MAX_EXPANDED_DEPTH = 5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            _require(bool(line.strip()), f"blank JSONL line at {path}:{line_number}")
            value = json.loads(line)
            _require(isinstance(value, dict), f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def _periodic_loop(output: Mapping[str, Any]) -> dict[str, Any]:
    raw = output.get("retained_thinking_token_ids", output.get("stage1_token_ids"))
    if not isinstance(raw, list) or len(raw) < LOOP_TAIL_TOKENS:
        return {"periodic_loop": False, "period_tokens": None, "match_rate": None}
    tail = [int(token) for token in raw[-LOOP_TAIL_TOKENS:]]
    best_rate = 0.0
    best_period: int | None = None
    for period in range(1, min(LOOP_MAX_PERIOD_TOKENS, len(tail) // 2) + 1):
        comparisons = len(tail) - period
        allowed = math.floor(comparisons * (1.0 - LOOP_MIN_MATCH_RATE) + 1e-12)
        mismatches = 0
        for index in range(period, len(tail)):
            if tail[index] != tail[index - period]:
                mismatches += 1
                if mismatches > allowed:
                    break
        if mismatches <= allowed:
            rate = (comparisons - mismatches) / comparisons
            if rate > best_rate:
                best_rate = rate
                best_period = period
                if rate == 1.0:
                    break
    return {
        "periodic_loop": best_period is not None,
        "period_tokens": best_period,
        "match_rate": best_rate if best_period is not None else None,
    }


def termination_metrics(
    rows: Sequence[Mapping[str, Any]], *, budget: int
) -> dict[str, Any]:
    """Recompute the frozen content-blind adequacy gate from runner-native rows."""

    _require(budget in store.SCIENTIFIC_BUDGETS, "unregistered termination budget")
    samples = unresolved = loops = answer_limit = 0
    stage1_length = forced = boundaries = answer_restarts = 0
    loop_periods: list[int] = []
    for row in rows:
        outputs = row.get("outputs")
        _require(isinstance(outputs, list) and bool(outputs), "runner row lacks outputs")
        for output in outputs:
            _require(isinstance(output, Mapping), "runner output must be an object")
            samples += 1
            thinking = int(output.get("n_thinking_tokens", -1))
            answer = int(output.get("n_answer_tokens", -1))
            _require(0 <= thinking <= budget and answer >= 0, "invalid token accounting")
            is_forced = bool(output.get("forced_close"))
            is_boundary = thinking + 1 >= budget
            is_stage1_length = str(output.get("stage1_finish_reason")) == "length"
            forced += int(is_forced)
            boundaries += int(is_boundary)
            stage1_length += int(is_stage1_length)
            answer_restarts += int(is_stage1_length and not is_forced and not is_boundary)
            if is_forced or is_boundary:
                loop = _periodic_loop(output)
                if loop["periodic_loop"]:
                    loops += 1
                    loop_periods.append(int(loop["period_tokens"]))
                else:
                    unresolved += 1
            is_answer_limit = (
                bool(output.get("truncated"))
                or str(output.get("finish_reason")) == "length"
                or answer >= ANSWER_MAX_TOKENS
            )
            answer_limit += int(is_answer_limit)
    _require(samples > 0, "termination metrics require at least one sample")
    unresolved_rate = unresolved / samples
    loop_rate = loops / samples
    answer_rate = answer_limit / samples
    adequate = (
        unresolved_rate < MAX_UNRESOLVED_RATE
        and answer_rate < MAX_ANSWER_LIMIT_RATE
        and loop_rate <= MAX_LOOP_RATE
    )
    return {
        "schema_version": 1,
        "selection_uses_decoded_or_scored_content": False,
        "selection_uses_token_identity_for_periodicity": True,
        "thinking_budget": budget,
        "answer_max_tokens": ANSWER_MAX_TOKENS,
        "samples": samples,
        "stage1_length_finishes": stage1_length,
        "forced_interventions": forced,
        "reasoning_boundary_contacts": boundaries,
        "answer_restarts_after_natural_close": answer_restarts,
        "unresolved_cap_contacts": unresolved,
        "unresolved_cap_contact_rate": unresolved_rate,
        "periodic_loop_contacts": loops,
        "periodic_loop_rate": loop_rate,
        "period_tokens": sorted(loop_periods),
        "answer_limit_contacts": answer_limit,
        "answer_limit_contact_rate": answer_rate,
        "thresholds": {
            "unresolved_cap_contact_rate_below": MAX_UNRESOLVED_RATE,
            "periodic_loop_rate_at_most": MAX_LOOP_RATE,
            "answer_limit_contact_rate_below": MAX_ANSWER_LIMIT_RATE,
        },
        "adequate": adequate,
    }


def _pairs(raw: Any, *, where: str) -> list[tuple[list[int], list[int]]]:
    _require(isinstance(raw, list) and bool(raw), f"{where} must be a nonempty list")
    pairs: list[tuple[list[int], list[int]]] = []
    for index, item in enumerate(raw):
        _require(isinstance(item, Mapping), f"{where}[{index}] must be an object")
        pairs.append((list(item["input"]), list(item["output"])))
    return pairs


def _score_program(program: Sequence[str], pairs: Sequence[tuple[list[int], list[int]]]) -> int:
    return sum(domain.execute_program(program, inputs) == expected for inputs, expected in pairs)


def _arm_semantics(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    tasks: Mapping[str, Mapping[str, Any]],
    library: Mapping[str, Any],
) -> dict[str, Any]:
    macro_map = {
        str(macro["token"]): tuple(str(token) for token in macro["expansion"])
        for macro in library["macros"]
    }
    allowed = set(domain.PRIMITIVES) | (set(macro_map) if arm != "base" else set())
    parsed = harness.parse_program_outputs(
        rows, allowed_tokens=allowed, max_surface_calls=MAX_SURFACE_CALLS
    )
    by_record: dict[str, list[Any]] = {}
    for completion in parsed:
        by_record.setdefault(completion.record_id, []).append(completion)
    task_results: list[dict[str, Any]] = []
    parsed_count = valid_count = macro_count = 0
    for record_id in sorted(by_record):
        suffix = f"::{arm}"
        _require(record_id.endswith(suffix), "record/arm mismatch during semantic analysis")
        task_id = record_id[: -len(suffix)]
        task = tasks[task_id]
        visible = _pairs(task["visible"], where=f"{task_id}.visible")
        hidden = _pairs(task["hidden"], where=f"{task_id}.hidden")
        candidates: list[dict[str, Any]] = []
        for completion in sorted(by_record[record_id], key=lambda item: item.sample_index):
            parsed_ok = completion.program is not None
            parsed_count += int(parsed_ok)
            candidate: dict[str, Any] = {
                "sample_index": completion.sample_index,
                "parsed": parsed_ok,
                "valid": False,
                "visible_correct": -1,
                "hidden_pass": False,
                "macro_used": False,
            }
            if parsed_ok:
                try:
                    expanded = tuple(domain.expand_program(completion.program, macro_map))
                    valid = len(expanded) <= MAX_EXPANDED_DEPTH
                except (KeyError, TypeError, ValueError):
                    expanded = ()
                    valid = False
                if valid:
                    candidate["valid"] = True
                    valid_count += 1
                    macro_used = any(token in macro_map for token in completion.program)
                    candidate["macro_used"] = macro_used
                    macro_count += int(macro_used)
                    candidate["visible_correct"] = _score_program(expanded, visible)
                    candidate["_expanded"] = expanded
            candidates.append(candidate)
        valid = [candidate for candidate in candidates if candidate["valid"]]
        selected = (
            min(
                (
                    candidate
                    for candidate in valid
                    if candidate["visible_correct"]
                    == max(item["visible_correct"] for item in valid)
                ),
                key=lambda candidate: candidate["sample_index"],
            )
            if valid
            else None
        )
        # Hidden labels enter only after the visible-only selected index is frozen.
        for candidate in valid:
            candidate["hidden_pass"] = _score_program(candidate["_expanded"], hidden) == len(hidden)
            candidate.pop("_expanded")
        task_results.append(
            {
                "task_id": task_id,
                "split": task["split"],
                "selected_sample_index": None if selected is None else selected["sample_index"],
                "selected_hidden_pass": bool(selected and selected["hidden_pass"]),
                "oracle_hidden_pass": any(candidate["hidden_pass"] for candidate in candidates),
                "valid_macro_candidate": any(
                    candidate["valid"] and candidate["macro_used"] for candidate in candidates
                ),
                "candidates": candidates,
            }
        )
    n_samples = sum(len(row["outputs"]) for row in rows)
    reuse = [task for task in task_results if task["split"] == "smoke_reuse"]
    return {
        "arm": arm,
        "tasks": task_results,
        "metrics": {
            "samples": n_samples,
            "parse_rate": parsed_count / n_samples,
            "valid_rate": valid_count / n_samples,
            "macro_using_candidate_rate": macro_count / n_samples,
            "selected_hidden_accuracy": sum(task["selected_hidden_pass"] for task in task_results)
            / len(task_results),
            "oracle_hidden_coverage": sum(task["oracle_hidden_pass"] for task in task_results)
            / len(task_results),
            "reuse_oracle_hidden_coverage": sum(task["oracle_hidden_pass"] for task in reuse)
            / len(reuse),
            "reuse_tasks_with_valid_macro_candidate": sum(
                task["valid_macro_candidate"] for task in reuse
            ),
        },
    }


def analyze_matrix(root: Path) -> dict[str, Any]:
    current_binding = store.build_protocol_binding(EXP)
    catalog_path = EXP / "analysis" / "scientific_smoke_artifact_catalog.json"
    selection_path = EXP / "analysis" / "smoke_budget_selection.json"
    catalog = store.verify_catalog(
        catalog_path,
        root,
        protocol_binding=current_binding,
        selection_file=selection_path,
    )
    store.validate_selection(
        selection_path,
        catalog,
        budget_ladder=store.SCIENTIFIC_BUDGETS,
        arms=store.SCIENTIFIC_MATRIX_ARMS,
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit_targets: list[tuple[int, str, Mapping[str, Any]]] = []
    for tier in selection["tiers"]:
        tier_budget = int(tier["budget"])
        probe_prefix = f"smoke_budget_probes/think_{tier_budget}/base"
        probe_receipt = store.verify_receipt(root, probe_prefix)
        comparable = store.comparable_protocol_identity(probe_receipt)
        audit_targets.append(
            (tier_budget, probe_prefix, tier["scientific_probe"]["termination"])
        )
        for arm, arm_state in tier["arms"].items():
            if arm_state["status"] != "complete":
                continue
            arm_prefix = f"smoke_tiers/think_{tier_budget}/{arm}"
            arm_receipt = store.verify_receipt(root, arm_prefix)
            _require(
                store.comparable_protocol_identity(arm_receipt) == comparable,
                f"tier {tier_budget}/{arm} protocol differs from its K4 probe",
            )
            audit_targets.append((tier_budget, arm_prefix, arm_state["termination"]))
    # Every receipt is verified before any row is opened. Recompute the entire
    # lower-tier history so an edited adequacy bit cannot outcome-shop a rung.
    for tier_budget, prefix, recorded in audit_targets:
        recomputed = termination_metrics(
            read_rows(store.bundle_paths(root, prefix).rows), budget=tier_budget
        )
        _require(recomputed == recorded, f"selection termination audit drift at {prefix}")
    budget, prefixes = store.selected_bundle_prefixes(
        catalog, store.SCIENTIFIC_MATRIX_ARMS
    )
    # Pass 1: verify both complete K12 identities before reading either row file.
    receipts: dict[str, dict[str, Any]] = {}
    paths_by_arm: dict[str, Any] = {}
    for arm in store.SCIENTIFIC_MATRIX_ARMS:
        prefix = prefixes[arm]
        receipt = store.verify_receipt(root, prefix)
        paths = store.bundle_paths(root, prefix)
        preflight = json.loads(paths.preflight.read_text(encoding="utf-8"))
        _require(
            preflight.get("protocol_binding") == current_binding,
            "matrix artifact is not bound to the current frozen analyzer/protocol",
        )
        receipts[arm] = receipt
        paths_by_arm[arm] = paths
    _require(
        store.comparable_protocol_identity(receipts["base"])
        == store.comparable_protocol_identity(receipts["designed_ceiling"]),
        "selected base/designed runtime or engine protocols differ",
    )

    # Pass 2: recompute both content-blind termination gates before any decode/grade.
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    terminations: dict[str, dict[str, Any]] = {}
    for arm in store.SCIENTIFIC_MATRIX_ARMS:
        paths = paths_by_arm[arm]
        rows = read_rows(paths.rows)
        termination = termination_metrics(rows, budget=budget)
        _require(termination["adequate"] is True, f"{arm} matrix termination is inadequate")
        rows_by_arm[arm] = rows
        terminations[arm] = termination

    # Pass 3: only a fully verified selected matrix may expose decoded/hidden content.
    tasks_payload = json.loads((EXP / "data" / "tasks.json").read_text(encoding="utf-8"))
    libraries_payload = json.loads(
        (EXP / "data" / "libraries.json").read_text(encoding="utf-8")
    )
    tasks = {
        str(task["id"]): task
        for task in tasks_payload["tasks"]
        if str(task.get("split", "")).startswith("smoke")
    }
    _require(len(tasks) == store.SCIENTIFIC_N_RECORDS, "frozen smoke task count drifted")
    arms: dict[str, Any] = {}
    for arm in store.SCIENTIFIC_MATRIX_ARMS:
        arms[arm] = {
            "rows_sha256": receipts[arm]["files"]["rows"]["sha256"],
            "termination": terminations[arm],
            "semantics": _arm_semantics(
                rows_by_arm[arm],
                arm=arm,
                tasks=tasks,
                library=libraries_payload["libraries"][arm],
            ),
        }
    base = arms["base"]["semantics"]["metrics"]
    designed = arms["designed_ceiling"]["semantics"]["metrics"]
    gates = {
        "both_parse_at_least_half": base["parse_rate"] >= 0.5
        and designed["parse_rate"] >= 0.5,
        "designed_uses_macros_on_two_reuse_tasks": designed[
            "reuse_tasks_with_valid_macro_candidate"
        ]
        >= 2,
        "designed_reuse_oracle_not_below_base": designed[
            "reuse_oracle_hidden_coverage"
        ]
        >= base["reuse_oracle_hidden_coverage"],
    }
    return {
        "schema_version": 1,
        "experiment_id": store.EXPERIMENT_ID,
        "thinking_budget": budget,
        "selection_boundary": (
            "visible examples choose the earliest maximal-score valid sample; hidden examples "
            "grade only after that index is frozen"
        ),
        "arms": arms,
        "smoke_gate": {"pass": all(gates.values()), "gates": gates},
        "protocol_binding": current_binding,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, default=EXP / "analysis" / "smoke_analysis.json")
    args = parser.parse_args(argv)
    root = store.resolve_artifact_root(args.artifact_root)
    result = analyze_matrix(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "budget": result["thinking_budget"],
                "smoke_pass": result["smoke_gate"]["pass"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

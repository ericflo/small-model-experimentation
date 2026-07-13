"""Frozen transport, mechanics generation, visible selection, and hidden scoring."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from calibration_stage import CalibrationInputs, sampling_configs
from interface_analysis import answer_cap_contact, score_interface_rows
from plans import freeze_taskwise_matches
from protocol import hidden_correct, parse_program, select_visible
from task_data import CONCRETE_OPERATIONS, OPERATION_TO_ALIAS, operation_from_record
from transactions import (
    artifact_paths,
    authenticate_registered_complete_chain,
    authenticate_registered_complete_prefix,
    inventory_state,
    read_canonical,
    run_transaction,
)


EXP = Path(__file__).resolve().parents[1]
MECHANICS_INVOCATION_ORDER = (
    "transport",
    "direct",
    "suffix_materialized",
    "suffix_name_only",
    "suffix_shuffled",
)
GENERATION_INVOCATIONS = MECHANICS_INVOCATION_ORDER[1:]
PREPARED_PATHS = {
    name: EXP / "runs" / "prepared" / f"{name}_requests.jsonl"
    for name in MECHANICS_INVOCATION_ORDER
}
EXPECTED_ROWS = {
    "transport": 24,
    "direct": 2_304,
    "suffix_materialized": 576,
    "suffix_name_only": 576,
    "suffix_shuffled": 576,
}
DEFAULT_MECHANICS_LOCK = EXP / "runs/mechanics/implementation_lock.json"
DEFAULT_MECHANICS_PREFLIGHT = EXP / "runs/mechanics/live_preflight.json"
DEFAULT_RUNNER_PATH = EXP / "src/vllm_runner.py"
RAW_DIR = EXP / "runs/mechanics/raw"
TRANSPORT_DECISION = EXP / "runs/mechanics/transport_decision.json"
VISIBLE_SELECTION = EXP / "runs/mechanics/visible_selection.json"
HIDDEN_RESULT = EXP / "runs/mechanics/hidden_result.json"
PUBLIC_PATH = EXP / "data/procedural/mechanics_public.jsonl"
GOLD_PATH = EXP / "data/procedural/mechanics_gold.jsonl"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"mechanics input is unsafe or absent: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid mechanics JSONL: {path}:{number}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object mechanics JSONL row: {path}:{number}")
        rows.append(value)
    return rows


def selected_interface(decision: Mapping[str, Any], inputs: CalibrationInputs) -> str:
    arms = tuple(inputs.config["interface"]["fixed_winner_priority"])
    winner = decision.get("winner")
    qualification = decision.get("qualification")
    if (
        decision.get("decision") != "CALIBRATION_INTERFACE_SELECTED"
        or winner not in arms
        or not isinstance(qualification, dict)
        or set(qualification) != set(arms)
        or qualification.get(winner) is not True
        or decision.get("fixed_priority") != list(arms)
        or decision.get("selection_uses_metric_ranking") is not False
    ):
        raise RuntimeError("calibration decision does not authorize mechanics")
    return str(winner)


def mechanics_sampling(
    decision: Mapping[str, Any], inputs: CalibrationInputs, *, direct_pool: bool = False
) -> dict[str, Any]:
    winner = selected_interface(decision, inputs)
    value = dataclasses.replace(
        sampling_configs(inputs)[winner],
        run_seed=int(
            inputs.config["seeds"]["direct_pool" if direct_pool else "mechanics"]
        ),
    )
    value.validate()
    return dataclasses.asdict(value)


def mechanics_sampling_plan(
    decision: Mapping[str, Any], inputs: CalibrationInputs
) -> dict[str, dict[str, Any]]:
    return {
        invocation: mechanics_sampling(
            decision,
            inputs,
            direct_pool=invocation == "direct",
        )
        for invocation in MECHANICS_INVOCATION_ORDER
    }


def mechanics_registrations(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, dict[str, Any]]:
    sampling = mechanics_sampling_plan(decision, inputs)
    return {
        invocation: {
            "prepared_path": PREPARED_PATHS[invocation],
            "expected_rows": EXPECTED_ROWS[invocation],
            "implementation_lock_path": mechanics_lock_path,
            "live_preflight_path": live_preflight_path,
            "runner_path": runner_path,
            "sampling": sampling[invocation],
        }
        for invocation in MECHANICS_INVOCATION_ORDER
    }


def _run_invocation(
    *,
    invocation: str,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    runner: Any,
    raw_dir: Path,
    mechanics_lock_path: Path,
    live_preflight_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    registration = mechanics_registrations(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )[invocation]

    def generate(
        rows: Sequence[dict[str, Any]], sampling: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from vllm_runner import SamplingConfig

        return runner.generate(rows, SamplingConfig(**dict(sampling)))

    return run_transaction(
        raw_dir=raw_dir,
        invocation=invocation,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        generate=generate,
        **registration,
    )


def run_transport_transaction(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    runner: Any,
    raw_dir: Path = RAW_DIR,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, Any]:
    state = inventory_state(raw_dir, "transport")
    registrations = mechanics_registrations(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
    if state == "complete":
        return authenticate_registered_complete_prefix(
            raw_dir=raw_dir,
            invocation_order=MECHANICS_INVOCATION_ORDER,
            registrations=registrations,
            through="transport",
        )
    if any(
        inventory_state(raw_dir, name) != "absent"
        for name in GENERATION_INVOCATIONS
    ):
        raise RuntimeError("mechanics generation exists before transport completes")
    _run_invocation(
        invocation="transport",
        decision=decision,
        inputs=inputs,
        runner=runner,
        raw_dir=raw_dir,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
    return authenticate_registered_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        registrations=registrations,
        through="transport",
    )


def analyze_transport(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    raw_dir: Path = RAW_DIR,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, Any]:
    selected_interface(decision, inputs)
    authenticate_registered_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        registrations=mechanics_registrations(
            decision=decision,
            inputs=inputs,
            mechanics_lock_path=mechanics_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
        ),
        through="transport",
    )
    bundle = read_canonical(artifact_paths(raw_dir, "transport")["bundle"])
    interface = inputs.config["interface"]
    metrics = score_interface_rows(
        bundle["rows"],
        answer_cap=int(interface["sampled_answer_cap"]),
        thinking_budget=int(interface["thinking_budget"]),
    )
    gate = interface["transport"]
    qualifies = bool(
        metrics["rows"] == int(gate["rows"])
        and metrics["exact_echo_successes"]
        >= int(gate["exact_echo_successes_min"])
        and metrics["parse_successes"] >= int(gate["parse_successes_min"])
        and metrics["answer_cap_contacts"]
        <= int(gate["answer_cap_contacts_max"])
        and all(
            metrics["by_arity"][str(arity)]["rows"]
            == int(gate["suffix_rows" if arity == 2 else "direct_rows"])
            and metrics["by_arity"][str(arity)]["exact_echo_successes"]
            >= int(gate["each_arity_exact_successes_min"])
            and metrics["by_arity"][str(arity)]["parse_successes"]
            >= int(gate["each_arity_parse_successes_min"])
            for arity in (2, 3)
        )
    )
    return {
        "schema_version": 1,
        "decision": (
            "SELECTED_INTERFACE_TRANSPORT_PASS"
            if qualifies
            else "SELECTED_INTERFACE_DID_NOT_TRANSPORT"
        ),
        "winner": decision["winner"],
        "qualifies": qualifies,
        "metrics": metrics,
        "bundle_sha256": canonical_sha256(bundle),
        "hidden_files_read": [],
        "benchmark_files_read": [],
    }


def run_generation_transactions(
    *,
    decision: Mapping[str, Any],
    transport: Mapping[str, Any],
    inputs: CalibrationInputs,
    runner: Any,
    raw_dir: Path = RAW_DIR,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, Any]:
    if (
        transport.get("decision") != "SELECTED_INTERFACE_TRANSPORT_PASS"
        or transport.get("winner") != selected_interface(decision, inputs)
        or transport.get("qualifies") is not True
    ):
        raise RuntimeError("failed transport cannot authorize mechanics generation")
    registrations = mechanics_registrations(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
    states = [inventory_state(raw_dir, name) for name in MECHANICS_INVOCATION_ORDER]
    if states[0] != "complete":
        raise RuntimeError("transport transaction is not complete")
    incomplete = [index for index, state in enumerate(states) if state != "complete"]
    if not incomplete:
        return authenticate_registered_complete_chain(
            raw_dir=raw_dir,
            invocation_order=MECHANICS_INVOCATION_ORDER,
            registrations=registrations,
        )
    first = incomplete[0]
    if first == 0 or any(state != "complete" for state in states[:first]) or any(
        state != "absent" for state in states[first + 1 :]
    ):
        raise RuntimeError("mechanics transaction prefix/inventory changed")
    for position, invocation in enumerate(
        MECHANICS_INVOCATION_ORDER[first:], start=first
    ):
        if inventory_state(raw_dir, invocation) == "absent":
            authenticate_registered_complete_prefix(
                raw_dir=raw_dir,
                invocation_order=MECHANICS_INVOCATION_ORDER,
                registrations=registrations,
                through=MECHANICS_INVOCATION_ORDER[position - 1],
            )
        _run_invocation(
            invocation=invocation,
            decision=decision,
            inputs=inputs,
            runner=runner,
            raw_dir=raw_dir,
            mechanics_lock_path=mechanics_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
        )
    return authenticate_registered_complete_chain(
        raw_dir=raw_dir,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        registrations=registrations,
    )


def _load_public(path: Path = PUBLIC_PATH) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [row.get("task_id") for row in rows]
    if len(rows) != 24 or len(set(ids)) != 24 or any(not isinstance(x, str) for x in ids):
        raise RuntimeError("mechanics public task identity changed")
    return rows


def _bundle_rows(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        name: read_canonical(artifact_paths(raw_dir, name)["bundle"])["rows"]
        for name in GENERATION_INVOCATIONS
    }


def _output_cost_fields(output: Mapping[str, Any]) -> dict[str, int]:
    return {
        field: output[field]
        for field in (
            "n_sampled_tokens",
            "n_stage1_prompt_tokens",
            "n_stage2_prompt_tokens",
        )
    }


def _visible_candidate(
    row: Mapping[str, Any], *, suffix: bool
) -> dict[str, Any]:
    output = row["outputs"][0]
    candidate = (
        operation_from_record(row["meta"]["candidate"]) if suffix else None
    )
    return {
        "candidate_id": row["id"],
        "candidate": candidate,
        "text": output["text"],
        "cost": _output_cost_fields(output),
        "sample_index": row["meta"].get("sample_index") if not suffix else None,
    }


def _generation_metrics(
    rows: Sequence[Mapping[str, Any]], *, arity: int, thinking_expected: bool, cap: int
) -> dict[str, Any]:
    parsed = 0
    contacts = 0
    for row in rows:
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("mechanics output geometry changed")
        output = outputs[0]
        expected_domain = "thought" if thinking_expected else "answer"
        if output.get("seed_domain_stage1") != expected_domain:
            raise RuntimeError("mechanics output reasoning policy changed")
        parsed += bool(
            parse_program(
                output.get("text"),
                arity=arity,
                thinking_expected=thinking_expected,
            )["parsed"]
        )
        contacts += answer_cap_contact(output, cap)
    return {
        "rows": len(rows),
        "parse_successes": parsed,
        "parse_rate": parsed / len(rows),
        "answer_cap_contacts": contacts,
        "answer_cap_contact_rate": contacts / len(rows),
    }


def analyze_visible(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    raw_dir: Path = RAW_DIR,
    public_path: Path = PUBLIC_PATH,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, Any]:
    winner = selected_interface(decision, inputs)
    chain = authenticate_registered_complete_chain(
        raw_dir=raw_dir,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        registrations=mechanics_registrations(
            decision=decision,
            inputs=inputs,
            mechanics_lock_path=mechanics_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
        ),
    )
    public = _load_public(public_path)
    public_by_id = {row["task_id"]: row for row in public}
    rows = _bundle_rows(raw_dir)
    thinking = winner.startswith("think512_")
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {
        task_id: defaultdict(list) for task_id in public_by_id
    }
    for invocation, invocation_rows in rows.items():
        suffix = invocation.startswith("suffix_")
        for row in invocation_rows:
            task_id = row["meta"].get("task_id")
            if task_id not in by_task:
                raise RuntimeError("mechanics result has an unknown task")
            by_task[task_id][invocation].append(
                _visible_candidate(row, suffix=suffix)
            )
    tasks: dict[str, Any] = {}
    for task_id, groups in by_task.items():
        direct = sorted(groups["direct"], key=lambda row: row["sample_index"])
        materialized = groups["suffix_materialized"]
        if (
            len(direct) != 96
            or [row["sample_index"] for row in direct] != list(range(96))
            or len(materialized) != 24
            or len(groups["suffix_name_only"]) != 24
            or len(groups["suffix_shuffled"]) != 24
        ):
            raise RuntimeError("taskwise mechanics row geometry changed")
        resource_plan = freeze_taskwise_matches(
            task_id=task_id,
            treatment_outputs=[row["cost"] for row in materialized],
            direct_outputs=[row["cost"] for row in direct],
        )
        if resource_plan["sampled"]["pool_exhausted"] or resource_plan["logical"][
            "pool_exhausted"
        ]:
            raise RuntimeError("direct master pool exhausted a mandatory match")
        sampled_k = int(resource_plan["sampled"]["first_over_k"])
        logical_k = int(resource_plan["logical"]["first_over_k"])
        controls = {
            "materialized": materialized,
            "name_only": groups["suffix_name_only"],
            "shuffled": groups["suffix_shuffled"],
            "direct_sampled": direct[:sampled_k],
            "direct_logical": direct[:logical_k],
            "direct_full_diagnostic": direct,
        }
        tasks[task_id] = {
            "resource_plan": resource_plan,
            "selections": {
                arm: select_visible(
                    public_by_id[task_id],
                    candidates,
                    thinking_expected=thinking,
                )
                for arm, candidates in controls.items()
            },
        }
    cap = int(inputs.config["interface"]["sampled_answer_cap"])
    metrics = {
        "direct": _generation_metrics(
            rows["direct"], arity=3, thinking_expected=thinking, cap=cap
        ),
        "materialized": _generation_metrics(
            rows["suffix_materialized"],
            arity=2,
            thinking_expected=thinking,
            cap=cap,
        ),
        "name_only": _generation_metrics(
            rows["suffix_name_only"],
            arity=2,
            thinking_expected=thinking,
            cap=cap,
        ),
        "shuffled": _generation_metrics(
            rows["suffix_shuffled"],
            arity=2,
            thinking_expected=thinking,
            cap=cap,
        ),
    }
    gate = inputs.config["mechanics"]
    abi_pass = all(
        value["parse_rate"] >= float(gate["all_generation_parse_rate_min"])
        and value["answer_cap_contact_rate"]
        <= float(gate["all_generation_answer_cap_contact_rate_max"])
        for value in metrics.values()
    )
    return {
        "schema_version": 1,
        "decision": (
            "MECHANICS_VISIBLE_SELECTION_FROZEN"
            if abi_pass
            else "MECHANICS_INTERFACE_NONTRANSPORT"
        ),
        "winner": winner,
        "generation_abi_pass": abi_pass,
        "generation_metrics": metrics,
        "tasks": tasks,
        "transaction_chain": chain,
        "public_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "selector_uses_hidden": False,
        "hidden_files_read": [],
        "benchmark_files_read": [],
    }


def _selected_program(selection: Mapping[str, Any]) -> tuple[Any, ...] | None:
    candidate_id = selection.get("selected_candidate_id")
    if candidate_id is None:
        return None
    matches = [
        row for row in selection.get("scored", []) if row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1 or matches[0].get("full_program") is None:
        raise RuntimeError("visible selection cannot recover its selected program")
    return tuple(tuple(operation) for operation in matches[0]["full_program"])


def score_hidden(
    *,
    visible: Mapping[str, Any],
    gold_path: Path = GOLD_PATH,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if visible.get("decision") != "MECHANICS_VISIBLE_SELECTION_FROZEN" or visible.get(
        "selector_uses_hidden"
    ) is not False:
        raise RuntimeError("hidden scoring requires a frozen passing visible receipt")
    gold_rows = read_jsonl(gold_path)
    gold = {row.get("task_id"): row for row in gold_rows}
    if len(gold_rows) != 24 or len(gold) != 24 or set(gold) != set(visible["tasks"]):
        raise RuntimeError("mechanics gold task identity changed")
    arms = (
        "materialized",
        "name_only",
        "shuffled",
        "direct_sampled",
        "direct_logical",
    )
    per_task: dict[str, Any] = {}
    selected_successes = {arm: 0 for arm in arms}
    oracle_successes = {arm: 0 for arm in arms}
    materialized_support: set[str] = set()
    for task_id, task in visible["tasks"].items():
        outcomes: dict[str, Any] = {}
        for arm in arms:
            selection = task["selections"][arm]
            selected = _selected_program(selection)
            selected_ok = hidden_correct(gold[task_id], selected)
            parsed_programs = [
                tuple(tuple(operation) for operation in row["full_program"])
                for row in selection["scored"]
                if row.get("full_program") is not None
            ]
            correct = [program for program in parsed_programs if hidden_correct(gold[task_id], program)]
            oracle = bool(correct)
            selected_successes[arm] += selected_ok
            oracle_successes[arm] += oracle
            if arm == "materialized":
                materialized_support.update(
                    OPERATION_TO_ALIAS[program[0]] for program in correct
                )
            outcomes[arm] = {
                "selected_candidate_id": selection["selected_candidate_id"],
                "selected_correct": selected_ok,
                "oracle_coverage": oracle,
                "hidden_correct_proposals": len(correct),
            }
        per_task[task_id] = outcomes
    selected_accuracy = {arm: selected_successes[arm] / 24 for arm in arms}
    oracle_coverage = {arm: oracle_successes[arm] / 24 for arm in arms}
    gate = config["mechanics"]
    comparisons = ("name_only", "shuffled", "direct_sampled", "direct_logical")
    selected_gain = {
        arm: selected_accuracy["materialized"] - selected_accuracy[arm]
        for arm in comparisons
    }
    oracle_gain = {
        arm: oracle_coverage["materialized"] - oracle_coverage[arm]
        for arm in comparisons
    }
    passes = bool(
        selected_accuracy["materialized"]
        >= float(gate["materialized_selected_accuracy_min"])
        and selected_successes["materialized"]
        >= int(gate["materialized_selected_successful_tasks_min"])
        and selected_gain["name_only"]
        >= float(gate["selected_accuracy_gain_vs_name_min"])
        and selected_gain["shuffled"]
        >= float(gate["selected_accuracy_gain_vs_shuffled_min"])
        and selected_gain["direct_sampled"]
        >= float(gate["selected_accuracy_gain_vs_direct_sampled_min"])
        and selected_gain["direct_logical"]
        >= float(gate["selected_accuracy_gain_vs_direct_logical_min"])
        and oracle_coverage["materialized"]
        >= float(gate["materialized_oracle_coverage_min"])
        and oracle_gain["name_only"]
        >= float(gate["oracle_coverage_gain_vs_name_min"])
        and oracle_gain["shuffled"]
        >= float(gate["oracle_coverage_gain_vs_shuffled_min"])
        and oracle_gain["direct_sampled"]
        >= float(gate["oracle_coverage_gain_vs_direct_sampled_min"])
        and oracle_gain["direct_logical"]
        >= float(gate["oracle_coverage_gain_vs_direct_logical_min"])
        and len(materialized_support)
        >= int(gate["materialized_oracle_first_operation_support_min"])
    )
    return {
        "schema_version": 1,
        "decision": (
            "MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_PASS"
            if passes
            else "MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_FAIL"
        ),
        "primary_selected_accuracy": selected_accuracy,
        "selected_successes": selected_successes,
        "selected_accuracy_gain": selected_gain,
        "oracle_proposal_coverage_diagnostic": oracle_coverage,
        "oracle_coverage_gain_diagnostic": oracle_gain,
        "materialized_oracle_first_operation_aliases": sorted(materialized_support),
        "per_task": per_task,
        "visible_selection_sha256": canonical_sha256(visible),
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "hidden_files_read": [str(gold_path)],
        "benchmark_files_read": [],
    }

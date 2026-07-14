"""Frozen transport, mechanics generation, visible selection, and hidden scoring."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from calibration_stage import (
    CalibrationInputs,
    sampling_configs,
)
from interface_analysis import choose_interface
from mechanics_protocol import hidden_correct, parse_program, score_echo, select_visible
from mechanics_runtime import (
    SELECTED_INTERFACE,
    authenticate_selected_interface_bundle,
    generate_selected_interface,
)
from plans import freeze_taskwise_matches
from stats import paired_report
from task_data import (
    DEPTH_THREE,
    OPERATION_TO_ALIAS,
    apply_pipeline,
    operation_from_record,
)
from mechanics_transactions import (
    artifact_paths,
    authenticate_registered_complete_chain,
    authenticate_registered_historical_prefix,
    authenticate_registered_complete_prefix,
    inventory_state,
    read_canonical,
    run_transaction,
)
from vllm_runner import SamplingConfig


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
RESOURCE_DECISION = EXP / "runs/mechanics/resource_decision.json"
HIDDEN_RESULT = EXP / "runs/mechanics/hidden_result.json"
PUBLIC_PATH = EXP / "data/procedural/mechanics_public.jsonl"
GOLD_CIPHERTEXT_PATH = EXP / "data/procedural/mechanics_gold.jsonl.aesgcm"
GOLD_KEY_PATH = EXP / ".secrets/mechanics_gold.aes256.key"
GOLD_MAGIC = b"AESGCM1\0"
GOLD_AAD = b"tokenizer-eos-residual-mechanics-fresh-replay-v1/mechanics-gold-v1"


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


def json_native(value: Any) -> Any:
    """Return the exact JSON-domain value used by durable receipts."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


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
    arms = tuple(inputs.config["interface"]["fixed_tokenizer_winner_priority"])
    expected_keys = {
        "schema_version",
        "stage",
        "model",
        "revision",
        "decision",
        "winner",
        "matched_hf_control",
        "fixed_tokenizer_priority",
        "qualification",
        "gate",
        "shared_thought",
        "boundary_pairs",
        "answer_requests",
        "all_pair_authentication",
        "condition_receipts",
        "cells",
        "transaction_chain",
        "calibration_input_receipt",
        "hidden_files_read",
        "mechanics_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
        "implementation_lock_sha256",
        "live_preflight_sha256",
        "runner_sha256",
    }
    winner = decision.get("winner")
    qualification = decision.get("qualification")
    cells = decision.get("cells")
    gate = decision.get("gate")
    if not isinstance(cells, dict) or not isinstance(gate, dict):
        raise RuntimeError("calibration decision cells/gate are incomplete")
    recomputed = choose_interface(cells, gate)
    if (
        set(decision) != expected_keys
        or decision.get("schema_version") != 1
        or decision.get("stage")
        != "authenticated_tokenizer_eos_calibration_decision"
        or decision.get("model") != "Qwen/Qwen3.5-4B"
        or decision.get("revision")
        != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or decision.get("decision")
        != "TOKENIZER_EOS_ONLY_INTERFACE_QUALIFIED"
        or winner != SELECTED_INTERFACE
        or not isinstance(qualification, dict)
        or set(qualification) != set(recomputed["qualification"])
        or qualification.get(winner) is not True
        or decision.get("fixed_tokenizer_priority") != list(arms)
        or decision.get("matched_hf_control")
        != "hf_model_eos_no_think_program_slot"
        or decision.get("all_pair_authentication") != "PASS"
        or decision.get("boundary_pairs") != 192
        or decision.get("answer_requests") != 384
        or any(decision.get(key) != value for key, value in recomputed.items())
        or any(
            not isinstance(decision.get(field), str)
            or len(decision[field]) != 64
            for field in (
                "implementation_lock_sha256",
                "live_preflight_sha256",
            )
        )
        or any(
            decision.get(field) != []
            for field in (
                "hidden_files_read",
                "mechanics_files_read",
                "qualification_files_read",
                "confirmation_files_read",
                "benchmark_files_read",
            )
        )
    ):
        raise RuntimeError("calibration decision does not authorize mechanics")
    return str(winner)


def mechanics_sampling(
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    *,
    invocation: str,
) -> dict[str, Any]:
    winner = selected_interface(decision, inputs)
    if winner != SELECTED_INTERFACE:
        raise RuntimeError("mechanics winner changed")
    if invocation not in MECHANICS_INVOCATION_ORDER:
        raise RuntimeError("unknown mechanics invocation")
    seed_name = (
        "transport"
        if invocation == "transport"
        else "direct_pool"
        if invocation == "direct"
        else "mechanics"
    )
    value = dataclasses.replace(
        sampling_configs(inputs)["no_think_program_slot_pairs"],
        run_seed=int(inputs.config["seeds"][seed_name]),
    )
    value.validate()
    return dataclasses.asdict(value)


def mechanics_sampling_plan(
    decision: Mapping[str, Any], inputs: CalibrationInputs
) -> dict[str, dict[str, Any]]:
    return {
        invocation: mechanics_sampling(decision, inputs, invocation=invocation)
        for invocation in MECHANICS_INVOCATION_ORDER
    }


def mechanics_registrations(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
    transport_decision_path: Path = TRANSPORT_DECISION,
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
            "authorization_paths": (
                {}
                if invocation == "transport"
                else {"transport_decision": transport_decision_path}
            ),
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
    transport_decision_path: Path,
) -> dict[str, Any]:
    registration = mechanics_registrations(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
    )[invocation]

    def generate(
        rows: Sequence[dict[str, Any]], sampling: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return generate_selected_interface(
            runner, rows, SamplingConfig(**dict(sampling))
        )

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
    transport_decision_path: Path = TRANSPORT_DECISION,
) -> dict[str, Any]:
    state = inventory_state(raw_dir, "transport")
    registrations = mechanics_registrations(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
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
        transport_decision_path=transport_decision_path,
    )
    return authenticate_registered_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        registrations=registrations,
        through="transport",
    )


def _score_transport_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        meta = row.get("meta")
        outputs = row.get("outputs")
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in seen
            or not isinstance(meta, Mapping)
            or not isinstance(outputs, list)
            or len(outputs) != 1
            or not isinstance(outputs[0], Mapping)
        ):
            raise RuntimeError("transport row geometry changed")
        seen.add(row_id)
        arity = meta.get("arity")
        expected = meta.get("expected")
        output = outputs[0]
        if type(arity) is not int or arity not in {2, 3} or not isinstance(
            expected, str
        ):
            raise RuntimeError("transport row registration changed")
        if type(output.get("answer_cap_contact")) is not bool:
            raise RuntimeError("transport cap-contact field changed")
        echo = score_echo(
            output.get("text"),
            expected=expected,
            arity=arity,
            thinking_expected=False,
        )
        scored.append(
            {
                "id": row_id,
                "task_id": meta.get("task_id"),
                "arity": arity,
                "exact_echo": bool(echo["exact_echo"]),
                "parsed": bool(echo["parsed"]),
                "answer_cap_contact": output["answer_cap_contact"],
            }
        )
    by_arity = {
        str(arity): {
            "rows": sum(row["arity"] == arity for row in scored),
            "exact_echo_successes": sum(
                row["arity"] == arity and row["exact_echo"] for row in scored
            ),
            "parse_successes": sum(
                row["arity"] == arity and row["parsed"] for row in scored
            ),
            "answer_cap_contacts": sum(
                row["arity"] == arity and row["answer_cap_contact"]
                for row in scored
            ),
        }
        for arity in (2, 3)
    }
    return {
        "rows": len(scored),
        "exact_echo_successes": sum(row["exact_echo"] for row in scored),
        "parse_successes": sum(row["parsed"] for row in scored),
        "answer_cap_contacts": sum(row["answer_cap_contact"] for row in scored),
        "by_arity": by_arity,
        "scored": scored,
    }


def _transport_decision_from_authenticated_transaction(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    raw_dir: Path = RAW_DIR,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
    transport_decision_path: Path = TRANSPORT_DECISION,
    tokenizer: Any,
) -> dict[str, Any]:
    selected_interface(decision, inputs)
    bundle = read_canonical(artifact_paths(raw_dir, "transport")["bundle"])
    engine_receipt = read_canonical(live_preflight_path)
    semantic_authentication = authenticate_selected_interface_bundle(
        records=read_jsonl(PREPARED_PATHS["transport"]),
        bundle=bundle,
        sampling=SamplingConfig(**mechanics_sampling_plan(decision, inputs)["transport"]),
        tokenizer=tokenizer,
        tokenizer_receipt=inputs.tokenizer_receipt,
        engine_receipt=engine_receipt,
    )
    interface = inputs.config["interface"]
    metrics = _score_transport_rows(bundle["rows"])
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
            == int(
                gate[
                    "suffix_arity_two_rows"
                    if arity == 2
                    else "direct_arity_three_rows"
                ]
            )
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
        "generation_authentication": semantic_authentication,
        "hidden_files_read": [],
        "benchmark_files_read": [],
    }


def authorize_initial_transport(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    raw_dir: Path = RAW_DIR,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
    transport_decision_path: Path = TRANSPORT_DECISION,
    tokenizer: Any,
) -> dict[str, Any]:
    """Analyze transport exactly once while every later invocation is absent."""

    authenticate_registered_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        registrations=mechanics_registrations(
            decision=decision,
            inputs=inputs,
            mechanics_lock_path=mechanics_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
            transport_decision_path=transport_decision_path,
        ),
        through="transport",
    )
    return _transport_decision_from_authenticated_transaction(
        decision=decision,
        inputs=inputs,
        raw_dir=raw_dir,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
        tokenizer=tokenizer,
    )


def authenticate_initial_transport_decision(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    raw_dir: Path = RAW_DIR,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
    transport_decision_path: Path = TRANSPORT_DECISION,
    tokenizer: Any,
) -> dict[str, Any]:
    observed = read_canonical(transport_decision_path)
    expected = authorize_initial_transport(
        decision=decision,
        inputs=inputs,
        raw_dir=raw_dir,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
        tokenizer=tokenizer,
    )
    if observed != expected:
        raise RuntimeError("recorded transport decision differs from exact analysis")
    if (
        observed["decision"] != "SELECTED_INTERFACE_TRANSPORT_PASS"
        or observed["qualifies"] is not True
    ):
        raise RuntimeError("failed transport cannot authorize mechanics generation")
    return observed


def authenticate_historical_transport(
    *,
    decision: Mapping[str, Any],
    inputs: CalibrationInputs,
    authenticated_chain: Mapping[str, Any],
    raw_dir: Path = RAW_DIR,
    mechanics_lock_path: Path = DEFAULT_MECHANICS_LOCK,
    live_preflight_path: Path = DEFAULT_MECHANICS_PREFLIGHT,
    runner_path: Path = DEFAULT_RUNNER_PATH,
    transport_decision_path: Path = TRANSPORT_DECISION,
    tokenizer: Any,
) -> dict[str, Any]:
    """Replay transport only after exact authentication of every descendant."""

    registrations = mechanics_registrations(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
    )
    authenticate_registered_historical_prefix(
        raw_dir=raw_dir,
        invocation_order=MECHANICS_INVOCATION_ORDER,
        registrations=registrations,
        through="transport",
        authenticated_chain=authenticated_chain,
    )
    observed = read_canonical(transport_decision_path)
    expected = _transport_decision_from_authenticated_transaction(
        decision=decision,
        inputs=inputs,
        raw_dir=raw_dir,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
        tokenizer=tokenizer,
    )
    if observed != expected:
        raise RuntimeError("recorded transport decision differs on historical replay")
    if (
        observed["decision"] != "SELECTED_INTERFACE_TRANSPORT_PASS"
        or observed["qualifies"] is not True
    ):
        raise RuntimeError("failed transport cannot authorize historical replay")
    return observed


def _read_passing_transport_decision(path: Path) -> dict[str, Any]:
    observed = read_canonical(path)
    expected_keys = {
        "schema_version",
        "decision",
        "winner",
        "qualifies",
        "metrics",
        "bundle_sha256",
        "generation_authentication",
        "hidden_files_read",
        "benchmark_files_read",
    }
    if (
        not isinstance(observed, dict)
        or set(observed) != expected_keys
        or type(observed.get("schema_version")) is not int
        or observed.get("schema_version") != 1
        or observed.get("decision") != "SELECTED_INTERFACE_TRANSPORT_PASS"
        or observed.get("qualifies") is not True
        or observed.get("winner") != SELECTED_INTERFACE
        or not isinstance(observed.get("metrics"), dict)
        or not isinstance(observed.get("generation_authentication"), dict)
        or not isinstance(observed.get("bundle_sha256"), str)
        or len(observed["bundle_sha256"]) != 64
        or observed.get("hidden_files_read") != []
        or observed.get("benchmark_files_read") != []
    ):
        raise RuntimeError("recorded transport decision is not a passing exact receipt")
    return observed


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
    transport_decision_path: Path = TRANSPORT_DECISION,
    tokenizer: Any,
) -> dict[str, Any]:
    registrations = mechanics_registrations(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
    )
    states = [inventory_state(raw_dir, name) for name in MECHANICS_INVOCATION_ORDER]
    if states[0] != "complete":
        raise RuntimeError("transport transaction is not complete")
    if all(state == "absent" for state in states[1:]):
        authenticated_transport = authenticate_initial_transport_decision(
            decision=decision,
            inputs=inputs,
            raw_dir=raw_dir,
            mechanics_lock_path=mechanics_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
            transport_decision_path=transport_decision_path,
            tokenizer=tokenizer,
        )
    else:
        # Descendant STARTED/COMPLETE receipts bind this file by hash. Their
        # transaction authentication below proves the historical authorization;
        # semantic replay waits for the complete-chain API.
        authenticated_transport = _read_passing_transport_decision(
            transport_decision_path
        )
    if dict(transport) != authenticated_transport:
        raise RuntimeError("caller transport differs from authenticated decision")
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
            transport_decision_path=transport_decision_path,
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


def _bundles(raw_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: read_canonical(artifact_paths(raw_dir, name)["bundle"])
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
        contact = output.get("answer_cap_contact")
        if type(contact) is not bool:
            raise RuntimeError("mechanics cap-contact field changed")
        contacts += contact
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
    transport_decision_path: Path = TRANSPORT_DECISION,
    tokenizer: Any,
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
            transport_decision_path=transport_decision_path,
        ),
    )
    authenticate_historical_transport(
        decision=decision,
        inputs=inputs,
        authenticated_chain=chain,
        raw_dir=raw_dir,
        mechanics_lock_path=mechanics_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        transport_decision_path=transport_decision_path,
        tokenizer=tokenizer,
    )
    public = _load_public(public_path)
    public_by_id = {row["task_id"]: row for row in public}
    bundles = _bundles(raw_dir)
    engine_receipt = read_canonical(live_preflight_path)
    sampling_plan = mechanics_sampling_plan(decision, inputs)
    generation_authentication = {
        name: authenticate_selected_interface_bundle(
            records=read_jsonl(PREPARED_PATHS[name]),
            bundle=bundle,
            sampling=SamplingConfig(**sampling_plan[name]),
            tokenizer=tokenizer,
            tokenizer_receipt=inputs.tokenizer_receipt,
            engine_receipt=engine_receipt,
        )
        for name, bundle in bundles.items()
    }
    rows = {name: bundle["rows"] for name, bundle in bundles.items()}
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
    resource_plans: dict[str, dict[str, Any]] = {}
    task_groups: dict[
        str, tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, list[dict[str, Any]]]]
    ] = {}
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
            direct_row_ids=[row["candidate_id"] for row in direct],
        )
        resource_plans[task_id] = resource_plan
        task_groups[task_id] = (direct, materialized, groups)
    exhausted = {
        task_id: [
            metric
            for metric in ("sampled", "logical")
            if plan[metric]["pool_exhausted"]
        ]
        for task_id, plan in resource_plans.items()
        if any(plan[metric]["pool_exhausted"] for metric in ("sampled", "logical"))
    }
    if exhausted:
        return json_native(
            {
                "schema_version": 1,
                "decision": inputs.config["outcomes"][
                    "direct_resource_match_pool_exhausted"
                ],
                "winner": winner,
                "resource_plans": resource_plans,
                "exhausted_tasks": exhausted,
                "generation_authentication": generation_authentication,
                "transaction_chain": chain,
                "public_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
                "selector_uses_hidden": False,
                "hidden_files_read": [],
                "benchmark_files_read": [],
            }
        )
    tasks: dict[str, Any] = {}
    for task_id, (direct, materialized, groups) in task_groups.items():
        resource_plan = resource_plans[task_id]
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
    return json_native(
        {
            "schema_version": 1,
            "decision": (
                "MECHANICS_VISIBLE_SELECTION_FROZEN"
                if abi_pass
                else "MECHANICS_INTERFACE_NONTRANSPORT"
            ),
            "winner": winner,
            "generation_abi_pass": abi_pass,
            "generation_metrics": metrics,
            "generation_authentication": generation_authentication,
            "tasks": tasks,
            "transaction_chain": chain,
            "public_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
            "selector_uses_hidden": False,
            "hidden_files_read": [],
            "benchmark_files_read": [],
        }
    )


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


def decrypt_hidden_gold(
    *,
    ciphertext_path: Path = GOLD_CIPHERTEXT_PATH,
    key_path: Path = GOLD_KEY_PATH,
    construction_path: Path = EXP / "runs/construction/summary.json",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open hidden plaintext only after the caller's publication authorization."""

    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    for path in (ciphertext_path, key_path, construction_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"hidden mechanics input is unsafe or absent: {path}")
    ciphertext = ciphertext_path.read_bytes()
    key = key_path.read_bytes()
    construction = read_canonical(construction_path)
    receipt = construction.get("hidden_ciphertext")
    if (
        not isinstance(receipt, dict)
        or receipt.get("algorithm") != "AES-256-GCM"
        or receipt.get("aad_utf8") != GOLD_AAD.decode("utf-8")
        or receipt.get("plaintext_rows") != 24
        or receipt.get("key_tracked") is not False
        or len(key) != 32
        or hashlib.sha256(key).hexdigest() != receipt.get("local_key_sha256")
        or hashlib.sha256(ciphertext).hexdigest()
        != receipt.get("ciphertext_sha256")
        or not ciphertext.startswith(GOLD_MAGIC)
        or len(ciphertext) <= len(GOLD_MAGIC) + 12
    ):
        raise RuntimeError("hidden ciphertext receipt changed")
    nonce_start = len(GOLD_MAGIC)
    nonce = ciphertext[nonce_start : nonce_start + 12]
    try:
        plaintext = AESGCM(key).decrypt(
            nonce, ciphertext[nonce_start + 12 :], GOLD_AAD
        )
    except InvalidTag as error:
        raise RuntimeError("hidden ciphertext authentication failed") from error
    if hashlib.sha256(plaintext).hexdigest() != receipt.get("plaintext_sha256"):
        raise RuntimeError("hidden plaintext hash changed")
    try:
        text = plaintext.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("hidden plaintext encoding changed") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid hidden JSONL row {number}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object hidden JSONL row {number}")
        canonical = json.dumps(value, sort_keys=True) + "\n"
        if canonical.encode("utf-8") != (line + "\n").encode("utf-8"):
            raise RuntimeError(f"noncanonical hidden JSONL row {number}")
        rows.append(value)
    if len(rows) != 24:
        raise RuntimeError("hidden plaintext row count changed")
    return rows, {
        "algorithm": "AES-256-GCM",
        "aad_utf8": GOLD_AAD.decode("utf-8"),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        "local_key_sha256": hashlib.sha256(key).hexdigest(),
        "hidden_files_read": [str(ciphertext_path), str(key_path)],
    }


def score_hidden(
    *,
    visible: Mapping[str, Any],
    gold_rows: Sequence[dict[str, Any]],
    gold_receipt: Mapping[str, Any],
    public_path: Path = PUBLIC_PATH,
    config: Mapping[str, Any],
    program_inventory: Sequence[tuple[Any, ...]] = DEPTH_THREE,
) -> dict[str, Any]:
    expected_gold_receipt = {
        "algorithm",
        "aad_utf8",
        "ciphertext_sha256",
        "plaintext_sha256",
        "local_key_sha256",
        "hidden_files_read",
    }
    if (
        not isinstance(gold_receipt, Mapping)
        or set(gold_receipt) != expected_gold_receipt
        or gold_receipt.get("algorithm") != "AES-256-GCM"
        or gold_receipt.get("aad_utf8") != GOLD_AAD.decode("utf-8")
        or any(
            not isinstance(gold_receipt.get(field), str)
            or len(gold_receipt[field]) != 64
            for field in (
                "ciphertext_sha256",
                "plaintext_sha256",
                "local_key_sha256",
            )
        )
        or not isinstance(gold_receipt.get("hidden_files_read"), list)
        or len(gold_receipt["hidden_files_read"]) != 2
        or any(
            not isinstance(path, str) or not path
            for path in gold_receipt["hidden_files_read"]
        )
    ):
        raise RuntimeError("hidden scoring receipt changed")
    expected_visible_keys = {
        "schema_version",
        "decision",
        "winner",
        "generation_abi_pass",
        "generation_metrics",
        "generation_authentication",
        "tasks",
        "transaction_chain",
        "public_sha256",
        "selector_uses_hidden",
        "hidden_files_read",
        "benchmark_files_read",
    }
    if (
        set(visible) != expected_visible_keys
        or visible.get("schema_version") != 1
        or visible.get("decision") != "MECHANICS_VISIBLE_SELECTION_FROZEN"
        or visible.get("generation_abi_pass") is not True
        or visible.get("selector_uses_hidden") is not False
        or visible.get("hidden_files_read") != []
        or visible.get("benchmark_files_read") != []
    ):
        raise RuntimeError("hidden scoring requires a frozen passing visible receipt")
    public_rows = _load_public(public_path)
    public = {row["task_id"]: row for row in public_rows}
    public_sha = hashlib.sha256(public_path.read_bytes()).hexdigest()
    if visible.get("public_sha256") != public_sha:
        raise RuntimeError("visible receipt differs from current public mechanics data")
    gold = {row.get("task_id"): row for row in gold_rows}
    if (
        len(gold_rows) != 24
        or len(gold) != 24
        or set(gold) != set(visible["tasks"])
        or set(public) != set(gold)
    ):
        raise RuntimeError("mechanics gold task identity changed")
    if not program_inventory:
        raise RuntimeError("exhaustive program inventory is empty")
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
    selected_vectors = {arm: [] for arm in arms}
    oracle_vectors = {arm: [] for arm in arms}
    materialized_support: set[str] = set()
    exhaustive: dict[str, Any] = {}
    for task_id in sorted(visible["tasks"]):
        task = visible["tasks"][task_id]
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
            selected_vectors[arm].append(bool(selected_ok))
            oracle_vectors[arm].append(oracle)
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
        visible_consistent = []
        for program in program_inventory:
            if all(
                apply_pipeline(row["input"], program) == row["output"]
                for row in public[task_id]["visible"]
            ):
                visible_consistent.append(program)
        ceiling_correct = sum(
            hidden_correct(gold[task_id], program) for program in visible_consistent
        )
        exhaustive[task_id] = {
            "visible_consistent_programs": len(visible_consistent),
            "hidden_correct_visible_consistent_programs": ceiling_correct,
            "any_hidden_correct": ceiling_correct > 0,
        }
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
    bootstrap_seed = int(config["seeds"]["bootstrap"])
    paired_inference = {
        "selected": {
            arm: paired_report(
                selected_vectors["materialized"],
                selected_vectors[arm],
                seed=bootstrap_seed + index,
            )
            for index, arm in enumerate(comparisons)
        },
        "oracle": {
            arm: paired_report(
                oracle_vectors["materialized"],
                oracle_vectors[arm],
                seed=bootstrap_seed + 100 + index,
            )
            for index, arm in enumerate(comparisons)
        },
        "affects_gate": False,
    }
    ceiling_successes = sum(row["any_hidden_correct"] for row in exhaustive.values())
    return {
        "schema_version": 1,
        "decision": (
            config["outcomes"]["capability_pass"]
            if passes
            else config["outcomes"]["capability_fail"]
        ),
        "primary_selected_accuracy": selected_accuracy,
        "selected_successes": selected_successes,
        "selected_accuracy_gain": selected_gain,
        "oracle_proposal_coverage_diagnostic": oracle_coverage,
        "oracle_coverage_gain_diagnostic": oracle_gain,
        "materialized_oracle_first_operation_aliases": sorted(materialized_support),
        "per_task": per_task,
        "report_only_paired_inference": paired_inference,
        "report_only_exhaustive_cpu_ceiling": {
            "program_inventory_size": len(program_inventory),
            "tasks": exhaustive,
            "tasks_with_any_hidden_correct_visible_consistent_program": ceiling_successes,
            "coverage": ceiling_successes / 24,
            "affects_gate": False,
        },
        "visible_selection_sha256": canonical_sha256(visible),
        "gold_plaintext_sha256": gold_receipt["plaintext_sha256"],
        "hidden_ciphertext_receipt": dict(gold_receipt),
        "hidden_files_read": list(gold_receipt["hidden_files_read"]),
        "benchmark_files_read": [],
    }

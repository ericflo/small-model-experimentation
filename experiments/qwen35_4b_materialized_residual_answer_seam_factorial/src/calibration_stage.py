"""Sealed calibration inputs, invocation plan, transactions, and scoring."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from interface_analysis import choose_interface, score_interface_rows
from transactions import (
    MODEL_ID,
    MODEL_REVISION,
    artifact_paths,
    authenticate_complete_prefix,
    authenticate_registered_complete_chain,
    inventory_state,
    read_canonical,
    run_transaction,
    sha256_file,
)
from vllm_runner import EngineConfig, SamplingConfig


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
CALIBRATION_RELATIVE_READS = (
    "configs/default.yaml",
    "runs/prepared/preoutcome_receipt.json",
    "runs/prepared/calibration_requests.jsonl",
    "runs/tokenizer/receipt_v3.json",
)
INVOCATION_ORDER = (
    "calibration_thoughts",
    "think512_freeform",
    "think512_program_slot",
    "no_think_freeform",
    "no_think_program_slot",
)
INTERFACE_ARMS = INVOCATION_ORDER[1:]
EXPECTED_ROWS = 48
DEFAULT_PREPARED_PATH = EXP / "runs/prepared/calibration_requests.jsonl"
DEFAULT_IMPLEMENTATION_LOCK_PATH = EXP / "runs/calibration/implementation_lock.json"
DEFAULT_LIVE_PREFLIGHT_PATH = EXP / "runs/calibration/live_preflight.json"
DEFAULT_RUNNER_PATH = EXP / "src/vllm_runner.py"


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


def _safe_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"calibration input is unsafe or absent: {path}")
    return path.read_bytes()


def _strict_json(data: bytes, *, source: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate JSON key in calibration input: {source}")
            value[key] = item
        return value

    try:
        return json.loads(data, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid calibration JSON: {source}") from error


def _strict_jsonl(data: bytes, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"invalid calibration JSONL encoding: {source}") from error
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        value = _strict_json(line.encode("utf-8"), source=f"{source}:{number}")
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object calibration row: {source}:{number}")
        rows.append(value)
    return rows


@dataclasses.dataclass(frozen=True)
class CalibrationInputs:
    config: dict[str, Any]
    preoutcome: dict[str, Any]
    tokenizer_receipt: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    read_receipt: dict[str, dict[str, Any]]


def load_calibration_inputs(exp: Path = EXP) -> CalibrationInputs:
    """Read only the four registered calibration artifacts.

    The caller supplies an experiment root so tests can prove identical loading
    when every mechanics/public/audit/gold/prepared artifact is absent.
    """
    paths = {relative: exp / relative for relative in CALIBRATION_RELATIVE_READS}
    raw = {relative: _safe_bytes(path) for relative, path in paths.items()}
    config = yaml.safe_load(raw["configs/default.yaml"])
    preoutcome = _strict_json(
        raw["runs/prepared/preoutcome_receipt.json"], source="preoutcome_receipt"
    )
    tokenizer_receipt = _strict_json(
        raw["runs/tokenizer/receipt_v3.json"], source="tokenizer_receipt_v3"
    )
    records = _strict_jsonl(
        raw["runs/prepared/calibration_requests.jsonl"],
        source="calibration_requests",
    )
    if not isinstance(config, dict):
        raise RuntimeError("calibration config is not an object")
    config_sha = hashlib.sha256(raw["configs/default.yaml"]).hexdigest()
    request_relative = (
        "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/"
        "runs/prepared/calibration_requests.jsonl"
    )
    request_entry = preoutcome.get("request_files", {}).get(request_relative)
    if (
        preoutcome.get("schema_version") != 1
        or preoutcome.get("decision") != "PREOUTCOME_PASS"
        or preoutcome.get("config_sha256") != config_sha
        or preoutcome.get("expected_invocation_rows", {}).get(
            "calibration_each_interface"
        )
        != EXPECTED_ROWS
        or request_entry
        != {
            "rows": EXPECTED_ROWS,
            "sha256": hashlib.sha256(
                raw["runs/prepared/calibration_requests.jsonl"]
            ).hexdigest(),
        }
        or preoutcome.get("model_calls") != 0
        or preoutcome.get("sampled_model_outputs") != 0
    ):
        raise RuntimeError("calibration preoutcome boundary changed")
    forbidden_fields = (
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    )
    if any(preoutcome.get(field) != [] for field in forbidden_fields):
        raise RuntimeError("preoutcome receipt records forbidden reads")
    if (
        tokenizer_receipt.get("schema_version") != 3
        or tokenizer_receipt.get("decision")
        != "TOKENIZER_AND_RENDERED_FRESHNESS_PASS"
        or tokenizer_receipt.get("model") != MODEL_ID
        or tokenizer_receipt.get("revision") != MODEL_REVISION
        or tokenizer_receipt.get("config_sha256") != config_sha
        or tokenizer_receipt.get("model_loaded") is not False
        or tokenizer_receipt.get("model_calls") != 0
        or tokenizer_receipt.get("sampled_model_outputs") != 0
        or any(tokenizer_receipt.get(field) != [] for field in forbidden_fields)
    ):
        raise RuntimeError("calibration tokenizer boundary changed")
    ids: list[str] = []
    for index, row in enumerate(records):
        if (
            set(row) != {"id", "messages", "meta"}
            or not isinstance(row["id"], str)
            or not isinstance(row["messages"], list)
            or len(row["messages"]) != 1
            or not isinstance(row["meta"], dict)
            or row["meta"].get("family") != "calibration"
            or row["meta"].get("arity") != (2 if index < 24 else 3)
        ):
            raise RuntimeError("calibration request schema/order changed")
        ids.append(row["id"])
    if len(records) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise RuntimeError("calibration request identity changed")
    read_receipt = {
        relative: {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        for relative, data in raw.items()
    }
    if tuple(read_receipt) != CALIBRATION_RELATIVE_READS:
        raise RuntimeError("calibration read allowlist changed")
    return CalibrationInputs(
        config=config,
        preoutcome=preoutcome,
        tokenizer_receipt=tokenizer_receipt,
        records=tuple(records),
        read_receipt=read_receipt,
    )


def engine_config(inputs: CalibrationInputs) -> EngineConfig:
    generation = inputs.config["generation"]
    value = EngineConfig(
        max_model_len=int(generation["max_model_len"]),
        gpu_memory_utilization=0.90,
        max_num_seqs=int(generation["max_num_seqs"]),
        max_num_batched_tokens=int(generation["max_num_batched_tokens"]),
        cudagraph_capture_sizes=tuple(generation["cudagraph_capture_sizes"]),
        enable_prefix_caching=bool(generation["enable_prefix_caching"]),
        enforce_eager=False,
    )
    value.validate()
    return value


def sampling_configs(inputs: CalibrationInputs) -> dict[str, SamplingConfig]:
    interface = inputs.config["interface"]
    if (
        tuple(interface["arms"]) != INTERFACE_ARMS
        or tuple(interface["fixed_winner_priority"]) != INTERFACE_ARMS
        or interface["n"] != 1
        or interface["unconstrained_sampling"] is not True
        or interface["grammar_mask"] is not False
        or interface["logit_bias"] is not False
        or interface["teacher_forced_answer_tokens"] is not False
        or interface["paired_answer_seed"] is not True
    ):
        raise RuntimeError("calibration interface inventory changed")
    common = {
        "n": 1,
        "max_tokens": int(interface["sampled_answer_cap"]),
        "answer_max_tokens": int(interface["sampled_answer_cap"]),
        "greedy": False,
        "temperature": float(interface["temperature"]),
        "top_p": float(interface["top_p"]),
        "top_k": int(interface["top_k"]),
        "run_seed": int(inputs.config["seeds"]["calibration"]),
        "paired_answer_seed": True,
    }
    thought = SamplingConfig(
        **common,
        thinking="budget",
        thinking_budget=int(interface["thinking_budget"]),
        force_answer_seam=True,
    )
    plan = {
        "calibration_thoughts": thought,
        "think512_freeform": thought,
        "think512_program_slot": dataclasses.replace(
            thought, answer_prefix=str(interface["answer_prefix_text"])
        ),
        "no_think_freeform": SamplingConfig(**common, thinking="off"),
        "no_think_program_slot": SamplingConfig(
            **common,
            thinking="off",
            answer_prefix=str(interface["answer_prefix_text"]),
        ),
    }
    if tuple(plan) != INVOCATION_ORDER:
        raise RuntimeError("calibration invocation order changed")
    for sampling in plan.values():
        sampling.validate()
    return plan


def _thought_bundle(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = read_canonical(
        artifact_paths(raw_dir, "calibration_thoughts")["bundle"]
    )
    if bundle.get("invocation") != "calibration_thoughts":
        raise RuntimeError("shared-thought transaction identity changed")
    return bundle["rows"], bundle["runner_metadata"]


def calibration_registrations(
    *,
    inputs: CalibrationInputs,
    prepared_path: Path = DEFAULT_PREPARED_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
    live_preflight_path: Path = DEFAULT_LIVE_PREFLIGHT_PATH,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, dict[str, Any]]:
    plan = sampling_configs(inputs)
    return {
        invocation: {
            "prepared_path": prepared_path,
            "expected_rows": EXPECTED_ROWS,
            "implementation_lock_path": implementation_lock_path,
            "live_preflight_path": live_preflight_path,
            "runner_path": runner_path,
            "sampling": dataclasses.asdict(plan[invocation]),
        }
        for invocation in INVOCATION_ORDER
    }


def authenticate_calibration_chain(
    *,
    inputs: CalibrationInputs,
    raw_dir: Path,
    prepared_path: Path = DEFAULT_PREPARED_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
    live_preflight_path: Path = DEFAULT_LIVE_PREFLIGHT_PATH,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, Any]:
    return authenticate_registered_complete_chain(
        raw_dir=raw_dir,
        invocation_order=INVOCATION_ORDER,
        registrations=calibration_registrations(
            inputs=inputs,
            prepared_path=prepared_path,
            implementation_lock_path=implementation_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
        ),
    )


def run_calibration_transactions(
    *,
    inputs: CalibrationInputs,
    runner: Any,
    raw_dir: Path,
    implementation_lock_path: Path,
    live_preflight_path: Path,
    runner_path: Path,
    prepared_path: Path | None = None,
) -> dict[str, Any]:
    plan = sampling_configs(inputs)
    prepared_path = prepared_path or DEFAULT_PREPARED_PATH
    if inputs.records != tuple(
        _strict_jsonl(_safe_bytes(prepared_path), source="live_calibration_requests")
    ):
        raise RuntimeError("live calibration requests differ from sealed inputs")
    states = [inventory_state(raw_dir, name) for name in INVOCATION_ORDER]
    incomplete = [index for index, state in enumerate(states) if state != "complete"]
    if not incomplete:
        return authenticate_calibration_chain(
            inputs=inputs,
            raw_dir=raw_dir,
            prepared_path=prepared_path,
            implementation_lock_path=implementation_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
        )
    first = incomplete[0]
    if any(state != "complete" for state in states[:first]) or any(
        state != "absent" for state in states[first + 1 :]
    ):
        raise RuntimeError("calibration transaction prefix/inventory changed")
    for position in range(first, len(INVOCATION_ORDER)):
        invocation = INVOCATION_ORDER[position]
        current_state = inventory_state(raw_dir, invocation)
        if position and current_state == "absent":
            authenticate_complete_prefix(
                raw_dir=raw_dir,
                invocation_order=INVOCATION_ORDER,
                through=INVOCATION_ORDER[position - 1],
            )

        def generate(
            rows: Sequence[dict[str, Any]],
            sampling_mapping: Mapping[str, Any],
            *,
            name: str = invocation,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            sampling = SamplingConfig(**dict(sampling_mapping))
            if name == "calibration_thoughts":
                return runner.generate_thought_prefixes(rows, sampling)
            if name.startswith("think512_"):
                thought_rows, thought_metadata = _thought_bundle(raw_dir)
                return runner.generate_from_thought_prefixes(
                    rows,
                    thought_rows,
                    thought_metadata,
                    sampling,
                )
            return runner.generate(rows, sampling)

        run_transaction(
            raw_dir=raw_dir,
            invocation=invocation,
            invocation_order=INVOCATION_ORDER,
            prepared_path=prepared_path,
            expected_rows=EXPECTED_ROWS,
            implementation_lock_path=implementation_lock_path,
            live_preflight_path=live_preflight_path,
            runner_path=runner_path,
            sampling=dataclasses.asdict(plan[invocation]),
            generate=generate,
        )
    return authenticate_calibration_chain(
        inputs=inputs,
        raw_dir=raw_dir,
        prepared_path=prepared_path,
        implementation_lock_path=implementation_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )


def load_calibration_bundles(
    *,
    inputs: CalibrationInputs,
    raw_dir: Path,
    prepared_path: Path = DEFAULT_PREPARED_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
    live_preflight_path: Path = DEFAULT_LIVE_PREFLIGHT_PATH,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, dict[str, Any]]:
    authenticate_calibration_chain(
        inputs=inputs,
        raw_dir=raw_dir,
        prepared_path=prepared_path,
        implementation_lock_path=implementation_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
    return {
        invocation: read_canonical(artifact_paths(raw_dir, invocation)["bundle"])
        for invocation in INVOCATION_ORDER
    }


def _authenticate_factorial_pairing(
    bundles: Mapping[str, dict[str, Any]], inputs: CalibrationInputs
) -> dict[str, Any]:
    ids = [row["id"] for row in inputs.records]
    thought_rows = bundles["calibration_thoughts"]["rows"]
    if [row.get("id") for row in thought_rows] != ids:
        raise RuntimeError("shared-thought row identity/order changed")
    thought_outputs = [row["outputs"][0] for row in thought_rows]
    prefix_ids = inputs.tokenizer_receipt["answer_prefix"]["token_ids"]
    arms = {name: bundles[name]["rows"] for name in INTERFACE_ARMS}
    if any([row["id"] for row in rows] != ids for rows in arms.values()):
        raise RuntimeError("factorial arm row identity/order changed")
    expected_modes = {
        "calibration_thoughts": "shared_thought_prefixes",
        "think512_freeform": "shared_thought_continuation",
        "think512_program_slot": "shared_thought_continuation",
        "no_think_freeform": "full_generation",
        "no_think_program_slot": "full_generation",
    }
    if any(
        bundles[name]["runner_metadata"].get("generation_mode") != mode
        for name, mode in expected_modes.items()
    ):
        raise RuntimeError("factorial generation mode changed")
    for index, thought in enumerate(thought_outputs):
        outputs = {name: arms[name][index]["outputs"][0] for name in INTERFACE_ARMS}
        for name in ("think512_freeform", "think512_program_slot"):
            if (
                outputs[name].get("stage1_token_ids")
                != thought.get("stage1_token_ids")
                or outputs[name].get("retained_thinking_token_ids")
                != thought.get("retained_thinking_token_ids")
                or outputs[name].get("seed_stage1") != thought.get("seed_stage1")
            ):
                raise RuntimeError("thinking cells do not share exact thought tokens")
        answer_seeds = {
            outputs[name].get(
                "seed_stage2" if name.startswith("think512_") else "seed_stage1"
            )
            for name in INTERFACE_ARMS
        }
        if len(answer_seeds) != 1 or None in answer_seeds:
            raise RuntimeError("factorial answer seed pairing changed")
        expected_prefixes = {
            "think512_freeform": [],
            "think512_program_slot": prefix_ids,
            "no_think_freeform": [],
            "no_think_program_slot": prefix_ids,
        }
        if any(
            outputs[name].get("answer_prefix_token_ids") != expected
            for name, expected in expected_prefixes.items()
        ):
            raise RuntimeError("factorial answer-prefix assignment changed")
    thought_source_sha = canonical_sha256(
        {
            "rows": thought_rows,
            "runner_metadata": bundles["calibration_thoughts"]["runner_metadata"],
        }
    )
    if any(
        bundles[name]["runner_metadata"].get("thought_source_sha256")
        != thought_source_sha
        for name in ("think512_freeform", "think512_program_slot")
    ):
        raise RuntimeError("thinking continuation source hash changed")
    thought_sampled = sum(
        len(output["stage1_token_ids"]) for output in thought_outputs
    )
    thought_counts = bundles["calibration_thoughts"]["runner_metadata"].get(
        "counts", {}
    )
    if (
        thought_counts.get("sampled_tokens") != thought_sampled
        or thought_counts.get("physical_sampled_tokens") != thought_sampled
        or thought_counts.get("reused_sampled_tokens") != 0
    ):
        raise RuntimeError("shared-thought physical accounting changed")
    for name in INTERFACE_ARMS:
        outputs = [row["outputs"][0] for row in arms[name]]
        counts = bundles[name]["runner_metadata"].get("counts", {})
        sampled = sum(output["n_sampled_tokens"] for output in outputs)
        if name.startswith("think512_"):
            physical = sum(len(output["stage2_token_ids"]) for output in outputs)
            reused = thought_sampled
        else:
            physical = sampled
            reused = 0
        if (
            counts.get("sampled_tokens") != sampled
            or counts.get("physical_sampled_tokens") != physical
            or counts.get("reused_sampled_tokens") != reused
            or sampled != physical + reused
        ):
            raise RuntimeError(f"factorial physical accounting changed: {name}")
    return {
        "rows": len(ids),
        "shared_thought_source_sha256": thought_source_sha,
        "exact_stage1_token_pairing": True,
        "exact_answer_seed_pairing": True,
        "registered_prefix_assignment": True,
        "physical_and_reused_sampled_token_accounting": True,
    }


def analyze_calibration(
    *,
    inputs: CalibrationInputs,
    raw_dir: Path,
    prepared_path: Path = DEFAULT_PREPARED_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
    live_preflight_path: Path = DEFAULT_LIVE_PREFLIGHT_PATH,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, Any]:
    bundles = load_calibration_bundles(
        inputs=inputs,
        raw_dir=raw_dir,
        prepared_path=prepared_path,
        implementation_lock_path=implementation_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
    pairing = _authenticate_factorial_pairing(bundles, inputs)
    interface = inputs.config["interface"]
    metrics_full = {
        arm: score_interface_rows(
            bundles[arm]["rows"],
            answer_cap=int(interface["sampled_answer_cap"]),
            thinking_budget=int(interface["thinking_budget"]),
        )
        for arm in INTERFACE_ARMS
    }
    decision = choose_interface(
        metrics_full,
        priority=INTERFACE_ARMS,
        gate=interface["calibration"],
    )
    metrics = {
        arm: {key: value for key, value in row.items() if key != "scored"}
        for arm, row in metrics_full.items()
    }
    scored_sha = {
        arm: canonical_sha256(row["scored"]) for arm, row in metrics_full.items()
    }
    chain = authenticate_calibration_chain(
        inputs=inputs,
        raw_dir=raw_dir,
        prepared_path=prepared_path,
        implementation_lock_path=implementation_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
    return {
        "schema_version": 1,
        "stage": "interface_calibration",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        **decision,
        "metrics": metrics,
        "scored_rows_sha256": scored_sha,
        "pairing": pairing,
        "transaction_chain": chain,
        "calibration_read_receipt": inputs.read_receipt,
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }

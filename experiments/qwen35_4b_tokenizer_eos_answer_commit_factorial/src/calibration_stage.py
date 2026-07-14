"""Sealed calibration inputs, append-only pair transactions, and decision logic."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from interface_analysis import (
    BoundaryAuthenticationError,
    authenticate_and_score_bundle,
    choose_interface,
)
from transactions import (
    MODEL_ID,
    MODEL_REVISION,
    artifact_paths,
    authenticate_registered_complete_chain,
    authenticate_registered_complete_prefix,
    canonical_sha256,
    inventory_state,
    read_canonical,
    run_transaction,
    sha256_file,
)
from vllm_runner import EngineConfig, SamplingConfig, _stable_seed


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
CALIBRATION_RELATIVE_READS = (
    "configs/default.yaml",
    "runs/prepared/preoutcome_receipt.json",
    "runs/prepared/calibration_requests.jsonl",
    "runs/tokenizer/receipt.json",
)
INVOCATION_ORDER = (
    "calibration_thoughts",
    "no_think_program_slot_pairs",
    "no_think_freeform_pairs",
    "think512_program_slot_pairs",
    "think512_freeform_pairs",
)
PAIR_INVOCATIONS = INVOCATION_ORDER[1:]
PAIR_CONDITION = {
    "no_think_program_slot_pairs": ("no_think", "program_slot"),
    "no_think_freeform_pairs": ("no_think", "freeform"),
    "think512_program_slot_pairs": ("think512", "program_slot"),
    "think512_freeform_pairs": ("think512", "freeform"),
}
EXPECTED_ROWS = 48
DEFAULT_PREPARED_PATH = EXP / "runs/prepared/calibration_requests.jsonl"
DEFAULT_IMPLEMENTATION_LOCK_PATH = EXP / "runs/calibration/implementation_lock.json"
DEFAULT_LIVE_PREFLIGHT_PATH = EXP / "runs/calibration/live_preflight.json"
DEFAULT_RUNNER_PATH = EXP / "src/vllm_runner.py"
DEFAULT_DECISION_PATH = EXP / "runs/calibration/decision.json"
RUNTIME_METADATA_KEYS = {
    "python",
    "python_executable",
    "platform",
    "packages",
    "environment_lock",
    "uv",
    "cuda_toolkit",
    "gpu",
    "vllm_enable_v1_multiprocessing",
    "git_commit",
    "git_dirty",
}


def _safe_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"calibration input is unsafe or absent: {path}")
    return path.read_bytes()


def _strict_json(data: bytes, *, source: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in calibration input: {source}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid calibration JSON: {source}") from error


def _strict_jsonl(data: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"invalid calibration JSONL encoding: {source}") from error
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        value = _strict_json(line.encode(), source=f"{source}:{number}")
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
    """Read only the four frozen calibration inputs; never mechanics or hidden data."""
    paths = {relative: exp / relative for relative in CALIBRATION_RELATIVE_READS}
    raw = {relative: _safe_bytes(path) for relative, path in paths.items()}
    config = yaml.safe_load(raw["configs/default.yaml"])
    preoutcome = _strict_json(
        raw["runs/prepared/preoutcome_receipt.json"], source="preoutcome"
    )
    tokenizer_receipt = _strict_json(
        raw["runs/tokenizer/receipt.json"], source="tokenizer_receipt"
    )
    records = _strict_jsonl(
        raw["runs/prepared/calibration_requests.jsonl"],
        source="calibration_requests",
    )
    config_sha = hashlib.sha256(raw["configs/default.yaml"]).hexdigest()
    preoutcome_sha = hashlib.sha256(
        raw["runs/prepared/preoutcome_receipt.json"]
    ).hexdigest()
    runner_sha = sha256_file(exp / "src/vllm_runner.py")
    request_relative = (
        "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/"
        "runs/prepared/calibration_requests.jsonl"
    )
    expected_request = {
        "rows": EXPECTED_ROWS,
        "sha256": hashlib.sha256(
            raw["runs/prepared/calibration_requests.jsonl"]
        ).hexdigest(),
    }
    forbidden = (
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    )
    if (
        not isinstance(config, dict)
        or preoutcome.get("schema_version") != 2
        or preoutcome.get("decision") != "PREOUTCOME_PASS"
        or preoutcome.get("config_sha256") != config_sha
        or preoutcome.get("expected_invocation_rows", {}).get(
            "calibration_each_cell"
        )
        != EXPECTED_ROWS
        or preoutcome.get("expected_invocation_rows", {}).get(
            "calibration_answer_total"
        )
        != 384
        or preoutcome.get("expected_invocation_rows", {}).get(
            "calibration_shared_thought"
        )
        != EXPECTED_ROWS
        or preoutcome.get("request_files", {}).get(request_relative)
        != expected_request
        or preoutcome.get("model_calls") != 0
        or preoutcome.get("sampled_model_outputs") != 0
        or any(preoutcome.get(field) != [] for field in forbidden)
    ):
        raise RuntimeError("calibration preoutcome boundary changed")
    if (
        tokenizer_receipt.get("schema_version") != 1
        or tokenizer_receipt.get("decision")
        != "TOKENIZER_GRAMMAR_PROMPT_FRESHNESS_PASS"
        or tokenizer_receipt.get("model") != MODEL_ID
        or tokenizer_receipt.get("revision") != MODEL_REVISION
        or tokenizer_receipt.get("config_sha256") != config_sha
        or tokenizer_receipt.get("preoutcome_sha256") != preoutcome_sha
        or tokenizer_receipt.get("runner_sha256") != runner_sha
        or tokenizer_receipt.get("model_loaded") is not False
        or tokenizer_receipt.get("model_calls") != 0
        or tokenizer_receipt.get("sampled_model_outputs") != 0
        or any(tokenizer_receipt.get(field) != [] for field in forbidden)
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
            or row["id"]
            not in tokenizer_receipt.get("calibration_prompt_token_ids", {})
            or row["id"]
            not in tokenizer_receipt.get("calibration_expected_token_ids", {})
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
        enable_prefix_caching=bool(generation["enable_prefix_caching"]),
        enforce_eager=False,
        cudagraph_capture_sizes=tuple(generation["cudagraph_capture_sizes"]),
    )
    value.validate()
    return value


def sampling_configs(inputs: CalibrationInputs) -> dict[str, SamplingConfig]:
    interface = inputs.config["interface"]
    if (
        interface["answer_boundaries"]["tokenizer_eos"]["stop_token_id"]
        != 248046
        or interface["answer_boundaries"]["hf_model_eos_control"][
            "stop_token_id"
        ]
        != 248044
        or interface["n"] != 1
        or interface["paired_answer_seed"] is not True
        or interface["paired_request_adjacency"] is not True
        or interface["one_shared_thought_transaction_per_task"] is not True
        or interface["unconstrained_sampling"] is not True
        or interface["grammar_mask"] is not False
        or interface["logit_bias"] is not False
        or interface["teacher_forced_answer_tokens"] is not False
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
    no_think = SamplingConfig(**common, thinking="off")
    prefix = str(interface["answer_prefix_text"])
    plan = {
        "calibration_thoughts": thought,
        "no_think_program_slot_pairs": dataclasses.replace(
            no_think, answer_prefix=prefix
        ),
        "no_think_freeform_pairs": no_think,
        "think512_program_slot_pairs": dataclasses.replace(
            thought, answer_prefix=prefix
        ),
        "think512_freeform_pairs": thought,
    }
    if tuple(plan) != INVOCATION_ORDER:
        raise RuntimeError("calibration invocation order changed")
    for sampling in plan.values():
        sampling.validate()
    return plan


def load_analysis_tokenizer(inputs: CalibrationInputs) -> Any:
    """Load the exact tokenizer only; this never loads model weights."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    receipt = inputs.tokenizer_receipt
    if (
        type(tokenizer).__name__ != receipt["tokenizer_class"]
        or len(tokenizer) != receipt["tokenizer_length"]
        or tokenizer.eos_token_id
        != receipt["termination"]["tokenizer_eos_token_id"]
        or tokenizer.encode("</think>\n\n", add_special_tokens=False)
        != receipt["think_token_ids"]["forced_close_sequence"]
        or tokenizer.encode("PROGRAM:", add_special_tokens=False)
        != receipt["program_slot_prefix_token_ids"]
    ):
        raise RuntimeError("analysis tokenizer differs from frozen receipt")
    return tokenizer


def calibration_registrations(
    *,
    inputs: CalibrationInputs,
    prepared_path: Path = DEFAULT_PREPARED_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
    live_preflight_path: Path = DEFAULT_LIVE_PREFLIGHT_PATH,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, dict[str, Any]]:
    plan = sampling_configs(inputs)
    authorization = {
        "config": EXP / "configs/default.yaml",
        "preoutcome": EXP / "runs/prepared/preoutcome_receipt.json",
        "tokenizer_receipt": EXP / "runs/tokenizer/receipt.json",
    }
    return {
        invocation: {
            "prepared_path": prepared_path,
            "expected_rows": EXPECTED_ROWS,
            "implementation_lock_path": implementation_lock_path,
            "live_preflight_path": live_preflight_path,
            "runner_path": runner_path,
            "sampling": dataclasses.asdict(plan[invocation]),
            "authorization_paths": authorization,
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


def _thought_bundle(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = read_canonical(artifact_paths(raw_dir, "calibration_thoughts")["bundle"])
    if value.get("invocation") != "calibration_thoughts":
        raise RuntimeError("shared-thought transaction identity changed")
    return value["rows"], value["runner_metadata"]


def run_calibration_transactions(
    *,
    inputs: CalibrationInputs,
    runner: Any,
    raw_dir: Path,
    implementation_lock_path: Path,
    live_preflight_path: Path,
    runner_path: Path,
    prepared_path: Path = DEFAULT_PREPARED_PATH,
) -> dict[str, Any]:
    plan = sampling_configs(inputs)
    if inputs.records != tuple(
        _strict_jsonl(_safe_bytes(prepared_path), source="live_calibration_requests")
    ):
        raise RuntimeError("live calibration requests differ from sealed inputs")
    registrations = calibration_registrations(
        inputs=inputs,
        prepared_path=prepared_path,
        implementation_lock_path=implementation_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
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
        if position and inventory_state(raw_dir, invocation) == "absent":
            authenticate_registered_complete_prefix(
                raw_dir=raw_dir,
                invocation_order=INVOCATION_ORDER,
                registrations=registrations,
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
                return runner.generate_boundary_pairs(
                    rows,
                    sampling,
                    thought_rows=thought_rows,
                    thought_metadata=thought_metadata,
                )
            return runner.generate_boundary_pairs(rows, sampling)

        registration = registrations[invocation]
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
            authorization_paths=registration["authorization_paths"],
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
        name: read_canonical(artifact_paths(raw_dir, name)["bundle"])
        for name in INVOCATION_ORDER
    }


def authenticate_thought_bundle(
    bundle: Mapping[str, Any],
    *,
    inputs: CalibrationInputs,
    tokenizer: Any,
) -> dict[str, Any]:
    rows = bundle.get("rows")
    metadata = bundle.get("runner_metadata")
    sampling = sampling_configs(inputs)["calibration_thoughts"]
    if (
        bundle.get("schema_version") != 1
        or bundle.get("invocation") != "calibration_thoughts"
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_ROWS
        or not isinstance(metadata, dict)
        or metadata.get("schema_version") != 6
        or metadata.get("generation_mode") != "shared_thought_prefixes"
        or metadata.get("model") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("runner_sha256") != sha256_file(DEFAULT_RUNNER_PATH)
        or canonical_sha256(metadata.get("sampling"))
        != canonical_sha256(dataclasses.asdict(sampling))
        or metadata.get("counts", {}).get("requests") != EXPECTED_ROWS
        or metadata.get("counts", {}).get("completions") != EXPECTED_ROWS
        or metadata.get("termination", {}).get("hf_model_eos_token_id") != 248044
    ):
        raise BoundaryAuthenticationError("shared-thought bundle metadata changed")
    total_prompt = 0
    total_sampled = 0
    total_injected = 0
    for record, row in zip(inputs.records, rows, strict=True):
        prompt = inputs.tokenizer_receipt["calibration_prompt_token_ids"][
            record["id"]
        ]["think512"]
        outputs = row.get("outputs")
        if (
            row.get("id") != record["id"]
            or row.get("meta") != record["meta"]
            or not isinstance(outputs, list)
            or len(outputs) != 1
            or row.get("prompt_sha256") != prompt["prompt_text_sha256"]
            or row.get("effective_prompt_sha256")
            != hashlib.sha256(
                b"".join(value.to_bytes(4, "big") for value in prompt["token_ids"])
            ).hexdigest()
            or row.get("n_prompt_tokens") != len(prompt["token_ids"])
            or row.get("n_original_prompt_tokens") != len(prompt["token_ids"])
            or row.get("prompt_channel") != "thinking"
            or row.get("answer_prefix_token_ids") != []
        ):
            raise BoundaryAuthenticationError("shared-thought row/prompt changed")
        output = outputs[0]
        sampled = output.get("stage1_token_ids")
        retained = output.get("retained_thinking_token_ids")
        if not isinstance(sampled, list) or not isinstance(retained, list):
            raise BoundaryAuthenticationError("shared-thought token schema changed")
        if output.get("finish_reason") == "stop":
            if (
                output.get("stop_reason") != 248044
                or not sampled
                or sampled[-1] != 248044
                or 248044 in sampled[:-1]
                or len(sampled) > 512
            ):
                raise BoundaryAuthenticationError("shared-thought stop geometry changed")
            without_eos = sampled[:-1]
        elif output.get("finish_reason") == "length":
            if (
                output.get("stop_reason") is not None
                or len(sampled) != 512
                or 248044 in sampled
            ):
                raise BoundaryAuthenticationError("shared-thought cap geometry changed")
            without_eos = sampled
        else:
            raise BoundaryAuthenticationError("shared-thought finish reason changed")
        close_id = int(inputs.tokenizer_receipt["think_token_ids"]["close"][0])
        close_index = without_eos.index(close_id) if close_id in without_eos else None
        expected_retained = (
            without_eos[:close_index] if close_index is not None else without_eos
        )
        expected_seed = _stable_seed(
            sampling.run_seed, record["id"], -1, "thought"
        )
        if (
            retained != expected_retained
            or output.get("token_ids") != retained
            or output.get("text")
            != tokenizer.decode(retained, skip_special_tokens=False)
            or output.get("sample_index") != 0
            or output.get("stage1_parent_seed") != expected_seed
            or output.get("seed_stage1") != expected_seed
            or output.get("seed_stage2") is not None
            or output.get("seed_domain_stage1") != "thought"
            or output.get("seed_domain_stage2") is not None
            or output.get("n_thinking_tokens") != len(retained)
            or output.get("n_answer_tokens") != 0
            or output.get("n_sampled_tokens") != len(sampled)
            or output.get("n_injected_tokens") != 0
            or output.get("n_completion_tokens") != len(retained)
            or output.get("n_terminal_tokens_trimmed")
            != len(sampled) - len(without_eos)
            or output.get("n_tokens_discarded_after_close")
            != len(without_eos) - len(retained)
            or output.get("n_stage1_prompt_tokens") != len(prompt["token_ids"])
            or output.get("n_stage2_prompt_tokens") != 0
            or output.get("thinking_closed") is not (close_index is not None)
            or output.get("forced_close") is not False
            or output.get("stage1_finish_reason") != output.get("finish_reason")
            or output.get("stage1_stop_reason") != output.get("stop_reason")
            or output.get("truncated")
            is not (output.get("finish_reason") == "length")
        ):
            raise BoundaryAuthenticationError("shared-thought token/seed fields changed")
        total_prompt += len(prompt["token_ids"])
        total_sampled += len(sampled)
        total_injected += int(output["n_injected_tokens"])
    counts = metadata["counts"]
    expected_counts = {
        "unique_input_prompt_tokens": total_prompt,
        "stage1_logical_prompt_tokens": total_prompt,
        "stage2_logical_prompt_tokens": 0,
        "logical_model_input_tokens": total_prompt,
        "logical_prompt_tokens": total_prompt,
        "physical_prompt_tokens": total_prompt,
        "reused_prompt_tokens": 0,
        "sampled_tokens": total_sampled,
        "physical_sampled_tokens": total_sampled,
        "reused_sampled_tokens": 0,
        "logical_model_tokens": total_prompt + total_sampled,
        "physical_model_tokens": total_prompt + total_sampled,
        "reused_model_tokens": 0,
        "injected_tokens": total_injected,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise BoundaryAuthenticationError("shared-thought token-cost summary changed")
    return {
        "decision": "SHARED_THOUGHT_TRANSACTION_AUTHENTICATED",
        "rows": EXPECTED_ROWS,
        "bundle_sha256": canonical_sha256(bundle),
        "source_sha256": hashlib.sha256(
            json.dumps(
                {"rows": rows, "runner_metadata": metadata},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }


def authenticate_pair_thought_reuse(
    pair_bundle: Mapping[str, Any],
    thought_bundle: Mapping[str, Any],
) -> None:
    """Bind each thinking continuation directly to its persisted source output."""
    pair_rows = pair_bundle.get("rows")
    thought_rows = thought_bundle.get("rows")
    pair_metadata = pair_bundle.get("runner_metadata")
    thought_metadata = thought_bundle.get("runner_metadata")
    if (
        not isinstance(pair_rows, list)
        or not isinstance(thought_rows, list)
        or len(pair_rows) != EXPECTED_ROWS
        or len(thought_rows) != EXPECTED_ROWS
        or not isinstance(pair_metadata, dict)
        or not isinstance(thought_metadata, dict)
    ):
        raise BoundaryAuthenticationError("thought-reuse bundle geometry changed")
    source_sha256 = hashlib.sha256(
        json.dumps(
            {"rows": thought_rows, "runner_metadata": thought_metadata},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if pair_metadata.get("thought_source_sha256") != source_sha256:
        raise BoundaryAuthenticationError("thinking cell used another thought source")
    source_fields = (
        "stage1_parent_seed",
        "seed_stage1",
        "stage1_token_ids",
        "retained_thinking_token_ids",
        "stage1_finish_reason",
        "stage1_stop_reason",
        "stage1_cumulative_logprob",
        "stage1_logprobs",
        "n_terminal_tokens_trimmed",
        "n_tokens_discarded_after_close",
    )
    for pair_row, thought_row in zip(pair_rows, thought_rows, strict=True):
        if not isinstance(pair_row, dict) or not isinstance(thought_row, dict):
            raise BoundaryAuthenticationError("thought-reuse row identity changed")
        pair_outputs = pair_row.get("outputs")
        thought_outputs = thought_row.get("outputs")
        if (
            pair_row.get("id") != thought_row.get("id")
            or pair_row.get("meta") != thought_row.get("meta")
            or not isinstance(pair_outputs, list)
            or len(pair_outputs) != 2
            or not isinstance(thought_outputs, list)
            or len(thought_outputs) != 1
        ):
            raise BoundaryAuthenticationError("thought-reuse row identity changed")
        source = thought_outputs[0]
        if not isinstance(source, dict) or any(
            not isinstance(output, dict)
            or any(output.get(field) != source.get(field) for field in source_fields)
            for output in pair_outputs
        ):
            raise BoundaryAuthenticationError(
                "thinking continuation differs from persisted thought"
            )


def authenticate_bundle_engine_preflight(
    bundle: Mapping[str, Any], live_preflight: Mapping[str, Any]
) -> None:
    """Bind a runner sidecar to the engine/runtime attested before generation."""
    metadata = bundle.get("runner_metadata")
    preflight_runtime = live_preflight.get("runtime")
    bundle_runtime = metadata.get("runtime") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or not isinstance(preflight_runtime, dict)
        or not isinstance(bundle_runtime, dict)
        or set(preflight_runtime) != RUNTIME_METADATA_KEYS
        or set(bundle_runtime) != RUNTIME_METADATA_KEYS
        or metadata.get("engine") != live_preflight.get("engine")
        or canonical_sha256(metadata.get("engine_args"))
        != live_preflight.get("engine_args_sha256")
        or metadata.get("resolved_cudagraph")
        != live_preflight.get("resolved_cudagraph")
        or metadata.get("resolved_logprobs_mode")
        != live_preflight.get("resolved_logprobs_mode")
    ):
        raise BoundaryAuthenticationError("bundle differs from live engine preflight")
    stable_keys = RUNTIME_METADATA_KEYS - {"git_dirty"}
    if any(
        bundle_runtime.get(key) != preflight_runtime.get(key) for key in stable_keys
    ):
        raise BoundaryAuthenticationError("bundle differs from live runtime preflight")


def score_calibration_bundles(
    bundles: Mapping[str, Mapping[str, Any]],
    *,
    inputs: CalibrationInputs,
    tokenizer: Any,
    live_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if set(bundles) != set(INVOCATION_ORDER):
        raise BoundaryAuthenticationError("calibration bundle inventory changed")
    for bundle in bundles.values():
        authenticate_bundle_engine_preflight(bundle, live_preflight)
    thought = authenticate_thought_bundle(
        bundles["calibration_thoughts"], inputs=inputs, tokenizer=tokenizer
    )
    cells: dict[str, dict[str, Any]] = {}
    condition_receipts = {}
    engine_bindings: list[str] = []
    for invocation in PAIR_INVOCATIONS:
        thinking_policy, prefix_condition = PAIR_CONDITION[invocation]
        bundle = bundles[invocation]
        receipt = authenticate_and_score_bundle(
            bundle["rows"],
            bundle["runner_metadata"],
            prefix_condition=prefix_condition,
            thinking_policy=thinking_policy,
            grammar_receipt=inputs.tokenizer_receipt,
            tokenizer=tokenizer,
            cap=int(inputs.config["interface"]["sampled_answer_cap"]),
        )
        if thinking_policy == "think512" and (
            bundle["runner_metadata"].get("thought_source_sha256")
            != thought["source_sha256"]
        ):
            raise BoundaryAuthenticationError("thinking cell used another thought source")
        if thinking_policy == "think512":
            authenticate_pair_thought_reuse(
                bundle, bundles["calibration_thoughts"]
            )
        condition = receipt["condition"]
        condition_receipts[condition] = receipt
        for boundary, metrics in receipt["cells"].items():
            cells[f"{boundary}_{condition}"] = metrics
        metadata = bundle["runner_metadata"]
        engine_bindings.append(
            canonical_sha256(
                {
                    "runner_sha256": metadata.get("runner_sha256"),
                    "engine": metadata.get("engine"),
                    "engine_args": metadata.get("engine_args"),
                    "resolved_cudagraph": metadata.get("resolved_cudagraph"),
                    "resolved_logprobs_mode": metadata.get("resolved_logprobs_mode"),
                    "adapter": metadata.get("adapter"),
                    "rng_isolation": metadata.get("rng_isolation"),
                }
            )
        )
    if len(set(engine_bindings)) != 1:
        raise BoundaryAuthenticationError("paired conditions used different engine geometry")
    gate_config = inputs.config["interface"]["calibration"]
    gate = {
        "rows": int(gate_config["rows_per_cell"]),
        "exact_echo_successes_min": int(gate_config["exact_echo_successes_min"]),
        "parse_successes_min": int(gate_config["parse_successes_min"]),
        "answer_cap_contacts_max": int(gate_config["answer_cap_contacts_max"]),
        "each_arity_rows": int(gate_config["each_arity_rows"]),
        "each_arity_exact_successes_min": int(
            gate_config["each_arity_exact_successes_min"]
        ),
        "each_arity_parse_successes_min": int(
            gate_config["each_arity_parse_successes_min"]
        ),
        "each_arity_answer_cap_contacts_max": int(
            gate_config["each_arity_answer_cap_contacts_max"]
        ),
    }
    selection = choose_interface(cells, gate)
    return {
        "schema_version": 1,
        "stage": "authenticated_tokenizer_eos_calibration_decision",
        "decision": selection["decision"],
        "winner": selection["winner"],
        "matched_hf_control": selection["matched_hf_control"],
        "qualification": selection["qualification"],
        "fixed_tokenizer_priority": selection["fixed_tokenizer_priority"],
        "gate": gate,
        "shared_thought": thought,
        "boundary_pairs": 192,
        "answer_requests": 384,
        "all_pair_authentication": "PASS",
        "condition_receipts": condition_receipts,
        "cells": cells,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "hidden_files_read": [],
        "mechanics_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }


def calibration_decision_value(
    *,
    inputs: CalibrationInputs,
    raw_dir: Path,
    tokenizer: Any,
    prepared_path: Path = DEFAULT_PREPARED_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
    live_preflight_path: Path = DEFAULT_LIVE_PREFLIGHT_PATH,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> dict[str, Any]:
    chain = authenticate_calibration_chain(
        inputs=inputs,
        raw_dir=raw_dir,
        prepared_path=prepared_path,
        implementation_lock_path=implementation_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
    )
    bundles = {
        name: read_canonical(artifact_paths(raw_dir, name)["bundle"])
        for name in INVOCATION_ORDER
    }
    try:
        scored = score_calibration_bundles(
            bundles,
            inputs=inputs,
            tokenizer=tokenizer,
            live_preflight=read_canonical(live_preflight_path),
        )
    except BoundaryAuthenticationError as error:
        scored = {
            "schema_version": 1,
            "stage": "authenticated_tokenizer_eos_calibration_decision",
            "decision": "BOUNDARY_PAIRING_INVALID",
            "winner": None,
            "matched_hf_control": None,
            "all_pair_authentication": "FAIL",
            "authentication_failure": str(error),
            "boundary_pairs": 192,
            "answer_requests": 384,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "hidden_files_read": [],
            "mechanics_files_read": [],
            "qualification_files_read": [],
            "confirmation_files_read": [],
            "benchmark_files_read": [],
        }
    return {
        **scored,
        "transaction_chain": chain,
        "calibration_input_receipt": inputs.read_receipt,
        "implementation_lock_sha256": sha256_file(implementation_lock_path),
        "live_preflight_sha256": sha256_file(live_preflight_path),
        "runner_sha256": sha256_file(runner_path),
    }

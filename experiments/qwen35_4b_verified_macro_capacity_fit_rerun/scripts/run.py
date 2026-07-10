#!/usr/bin/env python3
"""Run exactly one capacity-fit verified-macro smoke phase in one vLLM process.

Each invocation owns one engine with the preregistered concurrency for its rung,
writes a receipt last, and exits.  No invocation automatically advances to a
different budget or arm.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
DATA = EXP / "data"
ANALYSIS = EXP / "analysis"
CONFIG_PATH = EXP / "configs" / "default.yaml"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXP / "scripts"))

import analyze  # noqa: E402
import macro_domain as domain  # noqa: E402
import model_harness as harness  # noqa: E402
import scientific_artifacts as store  # noqa: E402


EXPECTED_HASHES = {
    "data/tasks.json": "82fbbd57e26fd392aa8f30ec6f26d370dc08dd78b3279bed6ee2e2174aea5073",
    "data/demonstrations.json": "1531b2722c5dc64530cbafda3e20a3de8a52ab537e50dc35ee4ec50a9fae06cf",
    "data/libraries.json": "a2ae3663753a3a0d0c9614a5d7c1d250506c74fd7879e11e99b66f5c1e43f865",
    "data/prompt_manifest.json": "f453b1ade590b3374be868dff2e8e7333e2d6e6044f6db9bf26dd033de98277f",
    "data/source_provenance.json": "0fbf805ccc38388044fe0a415eae83e907b4e9b239eb564f01b310cff86f0824",
    "src/macro_domain.py": "3a59b931faf42a6731ad73e31f9e8cdedf44c29423db4d8645b4e50a66ab21a7",
    "src/model_harness.py": "a43fb0e76f65819e5d1048f965e74c06409da870e77c7ce46f6df247257fa552",
    "src/vllm_runner.py": store.RUNNER_SHA256,
}
EXPECTED_RECORD_LIST_HASHES = {
    "base": "bd66aa64942f9e57e1fe55ae716c154ea1231480d6163f1811a07828ba364907",
    "designed_ceiling": "c5a6cd00d9600b7a63c8e2c132e202b25da30f30af299afb3735a8f5525d9e86",
}
MAX_SURFACE_CALLS = 5
MAX_EXPANDED_DEPTH = 5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(item) for item in inner.split(",")]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        _require(indent % 2 == 0, f"{path}:{line_number}: invalid indentation")
        line = raw.strip()
        _require(":" in line, f"{path}:{line_number}: expected mapping")
        key, value = line.split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        _require(key not in parent, f"{path}:{line_number}: duplicate key {key}")
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_bytes(path, payload)


def _primitive_descriptions() -> dict[str, str]:
    return {
        str(token): str(spec.description)
        for token, spec in domain.PRIMITIVES.items()
    }


def _smoke_tasks() -> list[dict[str, Any]]:
    payload = _load_json(DATA / "tasks.json")
    tasks = [
        dict(task)
        for task in payload["tasks"]
        if str(task.get("split", "")).startswith("smoke")
    ]
    tasks.sort(key=lambda task: str(task["id"]))
    expected_ids = [
        *(f"smoke-v2-no-reuse-{index:03d}" for index in range(6)),
        *(f"smoke-v2-reuse-{index:03d}" for index in range(6)),
    ]
    _require([str(task["id"]) for task in tasks] == expected_ids, "smoke-v2 task identity drift")
    return tasks


def solver_records(arm: str) -> list[dict[str, Any]]:
    _require(arm in store.SCIENTIFIC_MATRIX_ARMS, "unregistered matrix arm")
    libraries = _load_json(DATA / "libraries.json")["libraries"]
    demonstrations = _load_json(DATA / "demonstrations.json")["demonstrations"]
    library = libraries[arm]
    macros = [
        harness.MacroDefinition(
            token=str(macro["token"]),
            expansion=tuple(str(token) for token in macro["expansion"]),
        )
        for macro in library["macros"]
    ]
    records: list[dict[str, Any]] = []
    for task in _smoke_tasks():
        task_id = str(task["id"])
        records.append(
            harness.build_solver_record(
                f"{task_id}::{arm}",
                primitives=_primitive_descriptions(),
                macros=macros,
                macros_callable=arm != "base",
                solved_demonstrations=demonstrations,
                io_examples=task["visible"],
                max_surface_calls=MAX_SURFACE_CALLS,
                max_expanded_primitive_depth=MAX_EXPANDED_DEPTH,
                meta={
                    "task_id": task_id,
                    "split": task["split"],
                    "arm": arm,
                    "library_id": library["id"],
                    "max_surface_calls": MAX_SURFACE_CALLS,
                    "max_expanded_primitive_depth": MAX_EXPANDED_DEPTH,
                },
            )
        )
    _require(
        _sha256_value(records) == EXPECTED_RECORD_LIST_HASHES[arm],
        f"{arm} model-facing record payload drifted",
    )
    return records


def validate_frozen_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("model") == store.MODEL_ID, "only Qwen/Qwen3.5-4B is permitted")
    _require(config.get("model_revision") == store.MODEL_REVISION, "model revision drift")
    inference = config["inference"]
    _require(inference["backend"] == "vllm", "capacity-fit follow-up requires vLLM")
    _require(tuple(inference["thinking_budget_ladder"]) == store.SCIENTIFIC_BUDGETS, "budget drift")
    _require(
        tuple(inference["max_num_seqs_by_rung"])
        == tuple(store.MAX_NUM_SEQS_BY_BUDGET[budget] for budget in store.SCIENTIFIC_BUDGETS),
        "capacity-fit mapping drift",
    )
    _require(inference["max_model_len"] == 65536, "engine context drift")
    _require(inference["gpu_memory_utilization"] == 0.9, "GPU memory budget drift")
    _require(inference["max_num_batched_tokens"] == 32768, "scheduler token budget drift")
    _require(inference["enable_prefix_caching"] is False, "prefix caching must remain off")
    _require(inference["enforce_eager"] is False, "eager-mode setting drift")
    _require(inference["async_scheduling"] is False, "async scheduler must remain off")
    _require(inference["answer_max_tokens"] == 512, "answer allowance drift")
    _require(
        (inference["temperature"], inference["top_p"], inference["top_k"])
        == (0.6, 0.95, 20),
        "sampling distribution drift",
    )
    _require(inference["run_seed"] == 2701, "solver seed drift")
    _require(inference["probe_k"] == 4 and inference["matrix_k"] == 12, "K drift")
    _require(tuple(inference["matrix_arms"]) == store.SCIENTIFIC_MATRIX_ARMS, "arm drift")
    predecessor = config["predecessor"]
    _require(predecessor["artifacts_decision_eligible"] is False, "old artifacts became eligible")
    _require(
        predecessor["artifacts_may_be_pooled_or_promoted"] is False,
        "old artifacts became poolable",
    )
    decision = config["decision"]
    _require(
        (
            decision["max_unresolved_cap_rate"],
            decision["max_answer_limit_rate"],
            decision["max_periodic_loop_rate"],
        )
        == (0.05, 0.05, 0.25),
        "termination thresholds drift",
    )
    _require(
        (
            decision["smoke_matched_k"],
            decision["smoke_min_parse_rate"],
            decision["smoke_min_macro_reuse_tasks"],
            decision["designed_reuse_oracle_not_below_base"],
        )
        == (12, 0.5, 2, True),
        "semantic smoke thresholds drift",
    )
    artifacts = config["artifacts"]
    _require(
        artifacts["external_root"] == str(store.DEFAULT_ARTIFACT_ROOT)
        and artifacts["environment_override"] == store.ARTIFACT_ROOT_ENV
        and artifacts["predecessor_root_forbidden"]
        == str(store.PREDECESSOR_ARTIFACT_ROOT),
        "artifact namespace drift",
    )
    for relative, expected in EXPECTED_HASHES.items():
        _require(_sha256_file(EXP / relative) == expected, f"frozen copy drift: {relative}")
    records = {arm: solver_records(arm) for arm in store.SCIENTIFIC_MATRIX_ARMS}
    return {
        "model": store.MODEL_ID,
        "model_revision": store.MODEL_REVISION,
        "budgets": list(store.SCIENTIFIC_BUDGETS),
        "max_num_seqs_by_budget": dict(store.MAX_NUM_SEQS_BY_BUDGET),
        "record_hashes": {arm: _sha256_value(value) for arm, value in records.items()},
        "protocol_binding": store.build_protocol_binding(EXP),
    }


def _sampling(budget: int, k: int) -> Any:
    return harness.SamplingConfig(
        thinking="budget",
        thinking_budget=budget,
        n=k,
        max_tokens=ANSWER_MAX_TOKENS,
        answer_max_tokens=ANSWER_MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        run_seed=store.SCIENTIFIC_RUN_SEED,
    )


ANSWER_MAX_TOKENS = 512


def _prefix(phase: str, budget: int) -> tuple[str, str, int, bool]:
    if phase == "probe":
        return f"smoke_budget_probes/think_{budget}/base", "base", 4, True
    arm = "base" if phase == "base" else "designed_ceiling"
    return f"smoke_tiers/think_{budget}/{arm}", arm, 12, False


def _read_verified_metrics(root: Path, prefix: str, budget: int) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = store.verify_receipt(root, prefix)
    preflight = _load_json(store.bundle_paths(root, prefix).preflight)
    _require(
        preflight["protocol_binding"] == store.build_protocol_binding(EXP),
        "cached artifact belongs to a different protocol binding",
    )
    rows = analyze.read_rows(store.bundle_paths(root, prefix).rows)
    return receipt, analyze.termination_metrics(rows, budget=budget)


def _rung_outcome(root: Path, budget: int) -> str:
    probe_prefix = f"smoke_budget_probes/think_{budget}/base"
    probe_state = store.bundle_state(root, probe_prefix)
    if probe_state["status"] != "complete":
        return "needs_probe"
    _, probe = _read_verified_metrics(root, probe_prefix, budget)
    if not probe["adequate"]:
        return "probe_rejected"
    base_prefix = f"smoke_tiers/think_{budget}/base"
    base_state = store.bundle_state(root, base_prefix)
    if base_state["status"] != "complete":
        return "needs_base"
    _, base = _read_verified_metrics(root, base_prefix, budget)
    if not base["adequate"]:
        return "base_rejected"
    designed_prefix = f"smoke_tiers/think_{budget}/designed_ceiling"
    designed_state = store.bundle_state(root, designed_prefix)
    if designed_state["status"] != "complete":
        return "needs_designed"
    _, designed = _read_verified_metrics(root, designed_prefix, budget)
    return "matrix_adequate" if designed["adequate"] else "designed_rejected"


def authorize_phase(root: Path, phase: str, budget: int) -> None:
    first, second = store.SCIENTIFIC_BUDGETS
    if budget == second:
        lower = _rung_outcome(root, first)
        _require(
            lower in {"probe_rejected", "base_rejected", "designed_rejected"},
            f"61k probe requires a completed rejected 49k rung, found {lower}",
        )
    outcome = _rung_outcome(root, budget)
    allowed = {
        "probe": {"needs_probe"},
        "base": {"needs_base"},
        "designed": {"needs_designed"},
    }
    target_prefix, _, _, _ = _prefix(phase, budget)
    target_state = store.bundle_state(root, target_prefix)["status"]
    _require(
        outcome in allowed[phase] or target_state == "complete",
        f"phase {phase}@{budget} is not authorized from rung state {outcome}",
    )


@contextmanager
def _run_lock(root: Path) -> Iterator[None]:
    lock_path = store.LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another capacity-fit process owns {lock_path}") from exc
        root.mkdir(parents=True, exist_ok=True)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _preflight(runner: Any, records: Sequence[dict[str, Any]], sampling: Any) -> dict[str, Any]:
    prepared = runner.prepare(records, sampling.thinking, sampling.allow_custom_prompts)
    runner._check_context(prepared, sampling)
    reserve = int(sampling.thinking_budget) + len(runner.close_ids) + sampling.answer_max_tokens
    counts = [len(record.prompt_token_ids) for record in prepared]
    by_id = {str(record["id"]): record for record in records}
    result: dict[str, Any] = {
        "schema_version": 1,
        "pass": True,
        "protocol_binding": store.build_protocol_binding(EXP),
        "max_model_len": runner.config.max_model_len,
        "generation_reserve_tokens": reserve,
        "n_records": len(prepared),
        "min_prompt_tokens": min(counts),
        "max_prompt_tokens": max(counts),
        "max_prompt_plus_reserve_tokens": max(counts) + reserve,
        "records": [
            {
                "id": record.record_id,
                "input_record_sha256": _sha256_value(by_id[record.record_id]),
                "rendered_prompt_sha256": hashlib.sha256(record.prompt_text.encode()).hexdigest(),
                "prompt_tokens": len(record.prompt_token_ids),
                "prompt_plus_reserve_tokens": len(record.prompt_token_ids) + reserve,
            }
            for record in prepared
        ],
    }
    manifest = _load_json(DATA / "prompt_manifest.json")["arms"]
    arm = str(records[0]["meta"]["arm"])
    projected = [
        {key: row[key] for key in ("id", "input_record_sha256", "rendered_prompt_sha256", "prompt_tokens")}
        for row in result["records"]
    ]
    _require(projected == manifest[arm], f"{arm} prompt identity differs from predecessor")
    cache_config = runner.llm.llm_engine.vllm_config.cache_config
    live_model_len = runner.llm.llm_engine.vllm_config.model_config.max_model_len
    _require(live_model_len == 65536, "live vLLM model context differs from 65536")
    capacity = getattr(cache_config, "kv_cache_size_tokens", None)
    block_size = getattr(cache_config, "block_size", None)
    _require(isinstance(capacity, int) and capacity > 0, "vLLM did not expose KV token capacity")
    _require(isinstance(block_size, int) and block_size > 0, "vLLM did not expose KV block size")
    max_total = int(result["max_prompt_plus_reserve_tokens"])
    rounded = ((max_total + block_size - 1) // block_size) * block_size
    logical = len(records) * int(sampling.n)
    active = min(logical, runner.config.max_num_seqs)
    required = active * rounded
    _require(required <= capacity, "live KV cache cannot hold the registered active contexts")
    result["capacity_fit"] = {
        "source": "vllm_config.cache_config.kv_cache_size_tokens",
        "kv_cache_size_tokens": capacity,
        "block_size": block_size,
        "live_max_model_len": live_model_len,
        "max_num_seqs": runner.config.max_num_seqs,
        "logical_sequences": logical,
        "active_sequences": active,
        "max_prompt_plus_reserve_tokens": max_total,
        "rounded_tokens_per_sequence": rounded,
        "required_cache_tokens": required,
        "pass": True,
    }
    return result


def _normalized_runtime(runtime_value: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(runtime_value)
    runtime.pop("git_commit", None)
    runtime.pop("git_dirty", None)
    return json.loads(_canonical_bytes(runtime))


def _prerequisite_prefix(phase: str, budget: int) -> str | None:
    if phase == "base":
        return f"smoke_budget_probes/think_{budget}/base"
    if phase == "designed":
        return f"smoke_tiers/think_{budget}/base"
    return None


def _require_runtime_compatible(
    runtime: Mapping[str, Any], root: Path, phase: str, budget: int
) -> None:
    prefix = _prerequisite_prefix(phase, budget)
    if prefix is None:
        return
    receipt = store.verify_receipt(root, prefix)
    expected = receipt["identity"]["runtime_protocol"]
    _require(_normalized_runtime(runtime) == expected, "runtime changed within one rung protocol")


def _expected_identity(sampling: Any, engine: Any) -> dict[str, Any]:
    return {
        "model": store.MODEL_ID,
        "model_revision": store.MODEL_REVISION,
        "runner_sha256": store.RUNNER_SHA256,
        "sampling": json.loads(_canonical_bytes(dataclasses.asdict(sampling))),
        "engine": json.loads(_canonical_bytes(dataclasses.asdict(engine))),
    }


def _entry_artifacts(entry: Mapping[str, Any]) -> dict[str, str]:
    return {
        "rows": str(entry["files"]["rows"]["sha256"]),
        "meta": str(entry["files"]["metadata"]["sha256"]),
        "preflight": str(entry["files"]["preflight"]["sha256"]),
        "receipt": str(entry["receipt"]["sha256"]),
    }


def _selection_tier(
    root: Path, budget: int, entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    probe_prefix = f"smoke_budget_probes/think_{budget}/base"
    probe_receipt, probe_metrics = _read_verified_metrics(root, probe_prefix, budget)
    comparable = store.comparable_protocol_identity(probe_receipt)
    probe_entry = entries[f"probe/think_{budget}/base"]
    probe = {
        "status": "complete",
        "role": "termination_only_budget_probe",
        "budget": budget,
        "arm": "base",
        "k": 4,
        "records": 12,
        "termination": probe_metrics,
        "artifacts": _entry_artifacts(probe_entry),
    }
    arm_states: dict[str, Any] = {}
    arm_adequacy: list[bool] = []
    for arm in store.SCIENTIFIC_MATRIX_ARMS:
        prefix = f"smoke_tiers/think_{budget}/{arm}"
        state = store.bundle_state(root, prefix)
        if state["status"] == "complete":
            arm_receipt, metrics = _read_verified_metrics(root, prefix, budget)
            _require(
                store.comparable_protocol_identity(arm_receipt) == comparable,
                f"{arm} matrix runtime/engine differs from its same-rung probe",
            )
            entry = entries[f"matrix/think_{budget}/{arm}"]
            arm_states[arm] = {
                "status": "complete",
                "termination": metrics,
                "artifacts": _entry_artifacts(entry),
            }
            arm_adequacy.append(bool(metrics["adequate"]))
        else:
            arm_states[arm] = {"status": "skipped"}
    complete = all(state["status"] == "complete" for state in arm_states.values())
    adequate = complete and all(arm_adequacy)
    if not probe_metrics["adequate"]:
        status = "probe_only_rejected"
        tier_mode = "termination_probe_only"
    else:
        status = "selectable" if adequate else "rejected"
        tier_mode = "complete_k12_matrix"
    return {
        "budget": budget,
        "status": status,
        "tier_mode": tier_mode,
        "complete": complete,
        "adequate": adequate,
        "scientific_probe": probe,
        "arms": arm_states,
    }


def _assert_catalog_advancement(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    allowed_new_entry: str | None = None,
) -> None:
    for field in (
        "schema_version",
        "experiment_id",
        "storage",
        "checksum_scheme",
        "protocol_binding",
        "selected",
    ):
        _require(old.get(field) == new.get(field), f"catalog {field} changed unexpectedly")
    old_entries = {str(entry["id"]): entry for entry in old["entries"]}
    new_entries = {str(entry["id"]): entry for entry in new["entries"]}
    extras = set(new_entries) - set(old_entries)
    _require(
        extras == ({allowed_new_entry} if allowed_new_entry is not None else set()),
        f"unregistered external bundle appeared: {sorted(extras)}",
    )
    _require(not (set(old_entries) - set(new_entries)), "cataloged bundle disappeared")
    for entry_id, previous in old_entries.items():
        current = new_entries[entry_id]
        if previous == current:
            continue
        _require(
            previous.get("status") == "preflight_only"
            and current.get("status") == "complete",
            f"cataloged bundle changed outside preflight-to-receipt transition: {entry_id}",
        )
        _require(
            previous["files"]["preflight"] == current["files"]["preflight"],
            f"preflight identity changed while committing {entry_id}",
        )
        for field in (
            "relative_prefix",
            "role",
            "tier_mode",
            "thinking_budget",
            "arm",
            "k",
            "n_records",
        ):
            _require(
                previous.get(field) == current.get(field),
                f"bundle identity changed while committing {entry_id}",
            )
    if allowed_new_entry is not None:
        _require(
            new_entries[allowed_new_entry].get("status") == "preflight_only",
            "new checkpoint entry is not preflight-only",
        )


def _reconcile_inventory(root: Path) -> dict[str, Any]:
    """Validate every external byte and reconcile receipt-before-catalog crashes."""

    binding = store.build_protocol_binding(EXP)
    catalog_path = ANALYSIS / "scientific_smoke_artifact_catalog.json"
    selection_path = ANALYSIS / "smoke_budget_selection.json"
    stored = _load_json(catalog_path) if catalog_path.exists() else None
    if stored is not None:
        _require(
            stored.get("selected") is None or selection_path.exists(),
            "selected catalog cannot be downgraded without its selection file",
        )
    if selection_path.exists():
        selection = _load_json(selection_path)
        if selection.get("pass") is True:
            budget = int(selection["selected_thinking_budget"])
            catalog = store.build_catalog(
                root,
                protocol_binding=binding,
                selection_file=selection_path,
                selected_budget=budget,
                selected_entries={
                    arm: f"matrix/think_{budget}/{arm}"
                    for arm in store.SCIENTIFIC_MATRIX_ARMS
                },
            )
        else:
            _require(
                selection.get("pass") is False
                and selection.get("selected_thinking_budget") is None,
                "terminal failed selection is malformed",
            )
            catalog = store.build_catalog(root, protocol_binding=binding)
            store.validate_selection(
                selection_path,
                catalog,
                budget_ladder=store.SCIENTIFIC_BUDGETS,
                arms=store.SCIENTIFIC_MATRIX_ARMS,
            )
    else:
        catalog = store.build_catalog(root, protocol_binding=binding)
    if stored is None:
        _require(not catalog["entries"], "external artifacts exist without a catalog checkpoint")
    else:
        _assert_catalog_advancement(stored, catalog)
    store.write_catalog(catalog_path, catalog)
    return catalog


def _checkpoint_preflight(root: Path, prefix: str) -> None:
    catalog_path = ANALYSIS / "scientific_smoke_artifact_catalog.json"
    old = _load_json(catalog_path)
    new = store.build_catalog(root, protocol_binding=store.build_protocol_binding(EXP))
    namespace, think, arm = prefix.split("/")
    budget = int(think.removeprefix("think_"))
    entry_id = (
        f"probe/think_{budget}/{arm}"
        if namespace == "smoke_budget_probes"
        else f"matrix/think_{budget}/{arm}"
    )
    allowed = entry_id if entry_id not in {str(entry["id"]) for entry in old["entries"]} else None
    _assert_catalog_advancement(old, new, allowed_new_entry=allowed)
    store.write_catalog(catalog_path, new)


def _refresh_catalog(root: Path) -> None:
    binding = store.build_protocol_binding(EXP)
    catalog = _reconcile_inventory(root)
    entries = {str(entry["id"]): entry for entry in catalog["entries"]}
    selected: int | None = None
    for budget in store.SCIENTIFIC_BUDGETS:
        if _rung_outcome(root, budget) == "matrix_adequate":
            selected = budget
            break
    if selected is None:
        terminal = _rung_outcome(root, store.SCIENTIFIC_BUDGETS[-1]) in {
            "probe_rejected",
            "base_rejected",
            "designed_rejected",
        }
        if terminal:
            tiers = [
                _selection_tier(root, budget, entries)
                for budget in store.SCIENTIFIC_BUDGETS
            ]
            selection = {
                "schema_version": 1,
                "run": "smoke",
                "pass": False,
                "selected_thinking_budget": None,
                "selection_uses_output_content": False,
                "selection_uses_decoded_or_scored_content": False,
                "selection_uses_token_identity_for_periodicity": True,
                "lower_tiers_excluded_from_scoring": True,
                "scientific_probe_k": 4,
                "probes_excluded_from_promotion_scoring_and_prefix_pooling": True,
                "tiers": tiers,
            }
            selection_path = ANALYSIS / "smoke_budget_selection.json"
            _atomic_json(selection_path, selection)
            store.validate_selection(
                selection_path,
                catalog,
                budget_ladder=store.SCIENTIFIC_BUDGETS,
                arms=store.SCIENTIFIC_MATRIX_ARMS,
            )
        store.write_catalog(ANALYSIS / "scientific_smoke_artifact_catalog.json", catalog)
        return
    tiers = [
        _selection_tier(root, budget, entries)
        for budget in store.SCIENTIFIC_BUDGETS
        if budget <= selected
    ]
    _require(
        all(not tier["adequate"] for tier in tiers[:-1]) and tiers[-1]["adequate"],
        "selection is not the first adequate contiguous rung",
    )
    selection = {
        "schema_version": 1,
        "run": "smoke",
        "pass": True,
        "selected_thinking_budget": selected,
        "selection_uses_output_content": False,
        "selection_uses_decoded_or_scored_content": False,
        "selection_uses_token_identity_for_periodicity": True,
        "lower_tiers_excluded_from_scoring": True,
        "scientific_probe_k": 4,
        "probes_excluded_from_promotion_scoring_and_prefix_pooling": True,
        "tiers": tiers,
    }
    selection_path = ANALYSIS / "smoke_budget_selection.json"
    _atomic_json(selection_path, selection)
    selected_entries = {
        arm: f"matrix/think_{selected}/{arm}" for arm in store.SCIENTIFIC_MATRIX_ARMS
    }
    catalog = store.build_catalog(
        root,
        protocol_binding=binding,
        selection_file=selection_path,
        selected_budget=selected,
        selected_entries=selected_entries,
    )
    store.write_catalog(ANALYSIS / "scientific_smoke_artifact_catalog.json", catalog)


def run_phase(config: Mapping[str, Any], *, phase: str, budget: int) -> dict[str, Any]:
    root = store.resolve_artifact_root(
        os.environ.get(store.ARTIFACT_ROOT_ENV) or config["artifacts"]["external_root"]
    )
    with _run_lock(root):
        _reconcile_inventory(root)
        authorize_phase(root, phase, budget)
        prefix, arm, k, probe = _prefix(phase, budget)
        state = store.bundle_state(root, prefix)
        if state["status"] == "complete":
            _, metrics = _read_verified_metrics(root, prefix, budget)
            _refresh_catalog(root)
            return {"status": "already_complete", "prefix": prefix, "termination": metrics}

        records = solver_records(arm)
        sampling = _sampling(budget, k)
        engine = harness.EngineConfig(
            max_model_len=65536,
            gpu_memory_utilization=0.9,
            max_num_seqs=store.expected_max_num_seqs(budget),
            max_num_batched_tokens=32768,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        # Runtime identity is available without model construction; fail before
        # GPU allocation if a prerequisite receipt came from another environment.
        _require_runtime_compatible(
            harness.VLLMRunner.runtime_metadata(), root, phase, budget
        )
        runner = harness.VLLMRunner(engine)
        try:
            _require_runtime_compatible(
                runner.runtime_metadata(), root, phase, budget
            )
            preflight = _preflight(runner, records, sampling)
            store.write_preflight_only(root, prefix, preflight)
            _checkpoint_preflight(root, prefix)
            batch = harness.generate_vllm_batch(runner, records, sampling)
        finally:
            runner.close()
        paths = store.bundle_paths(root, prefix)
        _require(not paths.rows.exists() and not paths.metadata.exists(), "partial bundle collision")
        _atomic_jsonl(paths.rows, batch.rows)
        _atomic_json(paths.metadata, batch.summary)
        receipt = store.commit_receipt(
            root,
            prefix,
            role="termination_probe" if probe else "complete_matrix_arm",
            tier_mode="termination_probe_only" if probe else "complete_k12_matrix",
            thinking_budget=budget,
            arm=arm,
            k=k,
            expected_identity=_expected_identity(sampling, engine),
        )
        prerequisite = _prerequisite_prefix(phase, budget)
        if prerequisite is not None:
            previous = store.verify_receipt(root, prerequisite)
            _require(
                store.comparable_protocol_identity(previous)
                == store.comparable_protocol_identity(receipt),
                "probe/matrix same-rung protocol identities differ beyond K",
            )
        rows = analyze.read_rows(paths.rows)
        metrics = analyze.termination_metrics(rows, budget=budget)
        _refresh_catalog(root)
        return {"status": "complete", "prefix": prefix, "termination": metrics}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="run model-free frozen protocol checks")
    parser.add_argument("--phase", choices=("probe", "base", "designed"))
    parser.add_argument("--budget", type=int, choices=store.SCIENTIFIC_BUDGETS)
    args = parser.parse_args(argv)
    config = load_config()
    audit = validate_frozen_protocol(config)
    if args.validate:
        _require(args.phase is None and args.budget is None, "--validate is model-free and exclusive")
        print(json.dumps(audit, sort_keys=True))
        return 0
    _require(args.phase is not None and args.budget is not None, "GPU run needs --phase and --budget")
    result = run_phase(config, phase=args.phase, budget=args.budget)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

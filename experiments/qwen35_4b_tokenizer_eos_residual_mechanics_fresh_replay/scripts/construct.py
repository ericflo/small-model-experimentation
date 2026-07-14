#!/usr/bin/env python3
"""Build the fresh tasks and outcome-blind request inventory byte-identically."""

from __future__ import annotations

import hashlib
import json
import secrets
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from identity import (  # noqa: E402
    PARENT_COLLISION_MANIFEST,
    canonical_sha256,
    file_sha256,
    request_id,
    request_seed_key,
    verified_parent_collision_manifest,
)
from mechanics_protocol import (  # noqa: E402
    calibration_spec,
    direct_prompt,
    suffix_prompt,
    transport_spec,
)
from task_data import (  # noqa: E402
    CONCRETE_OPERATIONS,
    audit_task,
    build_tasks,
    canonical_operation,
    gold_task,
    public_task,
)


CONFIG = EXP / "configs" / "default.yaml"
DATA = EXP / "data" / "procedural"
PREPARED = EXP / "runs" / "prepared"
CONSTRUCTION = EXP / "runs" / "construction" / "summary.json"
PREOUTCOME = PREPARED / "preoutcome_receipt.json"
TOKENIZER_RECEIPT = EXP / "runs" / "tokenizer" / "receipt.json"
PARENT_COLLISION_REPAIR = (
    EXP / "runs" / "construction" / "parent_collision_receipt_repair.json"
)
HIDDEN_CIPHERTEXT = DATA / "mechanics_gold.jsonl.aesgcm"
HIDDEN_KEY = EXP / ".secrets" / "mechanics_gold.aes256.key"
HIDDEN_AAD = b"tokenizer-eos-residual-mechanics-fresh-replay-v1/mechanics-gold-v1"
HIDDEN_MAGIC = b"AESGCM1\0"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HEX_DIGITS = frozenset("0123456789abcdef")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def write_frozen(path: Path, value: Any, *, jsonl: bool = False) -> None:
    data = jsonl_bytes(value) if jsonl else json_bytes(value)
    if path.is_symlink():
        raise RuntimeError(f"frozen path is a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"frozen artifact differs on rebuild: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"unsafe frozen temporary path: {temporary}")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_hidden_ciphertext(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Seal hidden plaintext once; a pre-existing key is a terminal partial run."""

    plaintext = jsonl_bytes(rows)
    if HIDDEN_KEY.is_symlink() or HIDDEN_CIPHERTEXT.is_symlink():
        raise RuntimeError("hidden key/ciphertext path is a symlink")
    if HIDDEN_KEY.exists() or HIDDEN_CIPHERTEXT.exists():
        raise RuntimeError("hidden key/ciphertext already exists; refusing key reread")
    HIDDEN_KEY.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    with HIDDEN_KEY.open("xb") as handle:
        handle.write(key)
    if len(key) != 32:
        raise RuntimeError("hidden AES-256-GCM key length changed")
    nonce = secrets.token_bytes(12)
    blob = HIDDEN_MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, HIDDEN_AAD)
    HIDDEN_CIPHERTEXT.parent.mkdir(parents=True, exist_ok=True)
    with HIDDEN_CIPHERTEXT.open("xb") as handle:
        handle.write(blob)
    return {
        "algorithm": "AES-256-GCM",
        "aad_utf8": HIDDEN_AAD.decode("utf-8"),
        "ciphertext_path": str(HIDDEN_CIPHERTEXT.relative_to(ROOT)),
        "ciphertext_sha256": file_sha256(HIDDEN_CIPHERTEXT),
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        "plaintext_rows": len(rows),
        "local_key_path": str(HIDDEN_KEY.relative_to(ROOT)),
        "local_key_sha256": hashlib.sha256(key).hexdigest(),
        "key_tracked": False,
    }


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON is unsafe or absent: {path}")
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSONL is unsafe or absent: {path}")
    result = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(row, dict) for row in result):
        raise RuntimeError(f"non-object JSONL row: {path}")
    return result


def validate_completed_construction() -> dict[str, Any] | None:
    """Authenticate a completed rerun without opening or hashing the hidden key."""

    tracked_outputs = [
        DATA / "calibration_public.jsonl",
        DATA / "mechanics_public.jsonl",
        DATA / "mechanics_audit.jsonl",
        *sorted(PREPARED.glob("*_requests.jsonl")),
        HIDDEN_CIPHERTEXT,
        CONSTRUCTION,
        PREOUTCOME,
    ]
    state_exists = any(path.exists() or path.is_symlink() for path in tracked_outputs)
    state_exists = state_exists or HIDDEN_KEY.exists() or HIDDEN_KEY.is_symlink()
    if not state_exists:
        return None
    if any(path.is_symlink() for path in (*tracked_outputs, HIDDEN_KEY)):
        raise RuntimeError("existing construction state contains a symlink")
    if not CONSTRUCTION.is_file() or not PREOUTCOME.is_file():
        raise RuntimeError("partial construction exists; fresh key cannot be reopened")
    summary = read_json(CONSTRUCTION)
    receipt = read_json(PREOUTCOME)
    if not HIDDEN_KEY.is_file():
        raise RuntimeError("completed construction lacks its sealed local key")
    if receipt.get("construction_summary_sha256") != file_sha256(CONSTRUCTION):
        raise RuntimeError("completed preoutcome no longer binds construction")
    if receipt.get("config_sha256") != file_sha256(CONFIG):
        raise RuntimeError("completed preoutcome no longer binds config")
    verified_parent_collision_manifest(ROOT)
    expected_parent_read = {
        PARENT_COLLISION_MANIFEST["path"]: {
            "sha256": PARENT_COLLISION_MANIFEST["sha256"],
            "purpose": "authenticated_hash_only_parent_collision_domains",
        }
    }
    if summary.get("parent_read_receipt") != expected_parent_read:
        raise RuntimeError("completed construction no longer binds parent collision export")
    recorded_files = {
        **summary.get("data_files", {}),
        **summary.get("prepared_request_files", {}),
    }
    if recorded_files != {
        **receipt.get("request_files", {}),
        **summary.get("data_files", {}),
    }:
        raise RuntimeError("completed request inventories disagree")
    for relative, entry in recorded_files.items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"completed construction artifact changed: {relative}")
        if path.suffix == ".jsonl" and len(read_jsonl(path)) != entry.get("rows"):
            raise RuntimeError(f"completed construction row count changed: {relative}")
    hidden = summary.get("hidden_ciphertext", {})
    if hidden != receipt.get("hidden_ciphertext"):
        raise RuntimeError("completed hidden receipts disagree")
    ciphertext = ROOT / str(hidden.get("ciphertext_path", ""))
    if (
        ciphertext != HIDDEN_CIPHERTEXT
        or not ciphertext.is_file()
        or file_sha256(ciphertext) != hidden.get("ciphertext_sha256")
    ):
        raise RuntimeError("completed hidden ciphertext changed")
    if ROOT / str(hidden.get("local_key_path", "")) != HIDDEN_KEY:
        raise RuntimeError("completed hidden key path changed")
    if PARENT_COLLISION_REPAIR.exists() or PARENT_COLLISION_REPAIR.is_symlink():
        repair = read_json(PARENT_COLLISION_REPAIR)
        migrations = repair.get("receipt_migrations", {})
        expected_current = {
            str(CONSTRUCTION.relative_to(ROOT)): file_sha256(CONSTRUCTION),
            str(PREOUTCOME.relative_to(ROOT)): file_sha256(PREOUTCOME),
            str(TOKENIZER_RECEIPT.relative_to(ROOT)): file_sha256(TOKENIZER_RECEIPT),
        }
        if (
            repair.get("decision") != "PARENT_COLLISION_RECEIPT_REPAIR_PASS"
            or repair.get("new_manifest_sha256")
            != PARENT_COLLISION_MANIFEST["sha256"]
            or repair.get("collision_domains_exactly_unchanged") is not True
            or not isinstance(migrations, dict)
            or set(migrations) != set(expected_current)
            or any(
                not isinstance(migrations.get(relative), dict)
                or migrations[relative].get("new_sha256") != digest
                for relative, digest in expected_current.items()
            )
            or repair.get("hidden_files_read") != []
            or repair.get("local_key_files_read") != []
            or repair.get("benchmark_files_read") != []
            or repair.get("parent_raw_sampled_bundles_read") != []
            or repair.get("model_loaded") is not False
            or type(repair.get("model_calls")) is not int
            or repair.get("model_calls") != 0
            or type(repair.get("sampled_model_outputs_read")) is not int
            or repair.get("sampled_model_outputs_read") != 0
        ):
            raise RuntimeError("parent collision repair receipt changed")
    return summary


def request_row(
    *, request_key: list[str], prompt: str, meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": request_id(request_key),
        "messages": [{"role": "user", "content": prompt}],
        "meta": {**meta, "seed_key": request_key},
    }


def runner_seed(run_seed: int, record_id: str, sample_index: int, stage: str) -> int:
    payload = f"{run_seed}\0{record_id}\0{sample_index}\0{stage}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31
    )


def seed_inventory(
    rows: list[dict[str, Any]], config: dict[str, Any], *, parent: bool
) -> set[int]:
    representatives = {row["id"]: row for row in rows}
    if len(representatives) != len({row["id"] for row in rows}):
        raise RuntimeError("seed inventory representative collapse failed")
    result: set[int] = set()
    for row in representatives.values():
        family = row["meta"]["family"]
        run_seed = int(
            config["seeds"][
                "calibration"
                if family == "calibration"
                else "direct_pool"
                if family == "direct"
                else "mechanics"
                if parent or family == "suffix"
                else "transport"
            ]
        )
        for sample_index, domain in ((-1, "thought"), (0, "answer")):
            value = runner_seed(run_seed, row["id"], sample_index, domain)
            if value in result:
                raise RuntimeError("derived runner-seed collision within inventory")
            result.add(value)
    return result


def _hash_set(manifest: dict[str, Any], name: str, count: int) -> set[str]:
    values = manifest.get(name)
    if (
        not isinstance(values, list)
        or len(values) != count
        or values != sorted(values)
        or len(set(values)) != count
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in HEX_DIGITS for character in value)
            for value in values
        )
    ):
        raise RuntimeError(f"parent hash domain changed: {name}")
    return set(values)


def _string_set(manifest: dict[str, Any], name: str, count: int) -> set[str]:
    values = manifest.get(name)
    if (
        not isinstance(values, list)
        or len(values) != count
        or values != sorted(values)
        or len(set(values)) != count
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise RuntimeError(f"parent string domain changed: {name}")
    return set(values)


def parent_inventory() -> tuple[set[str], set[str], dict[str, Any]]:
    """Read only the exact hash-only parent export and validate it fail closed."""

    manifest_path = verified_parent_collision_manifest(ROOT)
    manifest = read_json(manifest_path)
    expected_keys = {
        "administrative_sources",
        "benchmark_files_read",
        "common_function_fingerprints",
        "derived_runner_seeds",
        "hidden_files_read",
        "model",
        "model_calls",
        "model_loaded",
        "parent_experiment",
        "parent_raw_sampled_bundles_read",
        "prompt_sha256s",
        "prompt_token_sequence_sha256s",
        "public_instance_fingerprints",
        "request_ids",
        "revision",
        "sampled_model_outputs_read",
        "schema_version",
        "seed_key_sha256s",
        "stage",
        "task_ids",
        "tokenizer_loaded",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RuntimeError("parent collision manifest schema changed")
    expected_scalars = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "parent_experiment": "qwen35_4b_tokenizer_eos_answer_commit_factorial",
        "stage": "hash_only_parent_collision_export",
        "schema_version": 1,
        "model_loaded": False,
        "model_calls": 0,
        "sampled_model_outputs_read": 0,
        "tokenizer_loaded": True,
    }
    if any(type(manifest.get(name)) is not type(value) or manifest.get(name) != value for name, value in expected_scalars.items()):
        raise RuntimeError("parent collision manifest scalar changed")
    for name in (
        "benchmark_files_read",
        "hidden_files_read",
        "parent_raw_sampled_bundles_read",
    ):
        if manifest.get(name) != []:
            raise RuntimeError(f"parent collision export crossed forbidden boundary: {name}")
    sources = manifest.get("administrative_sources")
    if (
        not isinstance(sources, dict)
        or len(sources) != 8
        or any(
            not isinstance(path, str)
            or not path.startswith(
                "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/"
            )
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in HEX_DIGITS for character in digest)
            for path, digest in sources.items()
        )
    ):
        raise RuntimeError("parent administrative source receipt changed")
    public_fingerprints = _hash_set(
        manifest, "public_instance_fingerprints", 72
    )
    function_fingerprints = _hash_set(
        manifest, "common_function_fingerprints", 72
    )
    request_ids = _hash_set(manifest, "request_ids", 2952)
    seed_key_sha256s = _hash_set(manifest, "seed_key_sha256s", 2952)
    prompt_sha256s = _hash_set(manifest, "prompt_sha256s", 1824)
    prompt_token_sha256s = _hash_set(
        manifest, "prompt_token_sequence_sha256s", 3648
    )
    task_ids = _string_set(manifest, "task_ids", 72)
    derived = manifest.get("derived_runner_seeds")
    if (
        not isinstance(derived, list)
        or len(derived) != 5904
        or derived != sorted(derived)
        or len(set(derived)) != 5904
        or any(type(value) is not int or not 0 <= value < 2**31 for value in derived)
    ):
        raise RuntimeError("parent derived-runner-seed domain changed")
    relative = str(manifest_path.relative_to(ROOT))
    return public_fingerprints, function_fingerprints, {
        "reads": {
            relative: {
                "sha256": PARENT_COLLISION_MANIFEST["sha256"],
                "purpose": "authenticated_hash_only_parent_collision_domains",
            }
        },
        "request_ids": request_ids,
        "seed_key_sha256s": seed_key_sha256s,
        "prompt_sha256s": prompt_sha256s,
        "prompt_token_sha256s": prompt_token_sha256s,
        "task_ids": task_ids,
        "derived_seeds": set(derived),
    }


def build_requests(
    tasks: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    calibration_rows: list[dict[str, Any]] = []
    calibration_requests: list[dict[str, Any]] = []
    for index, task in enumerate(tasks["calibration"]):
        public = public_task(task)
        spec = calibration_spec(public, index)
        calibration_rows.append(
            {
                "task_id": public["task_id"],
                "stratum": task["stratum"],
                "arity": spec["arity"],
                "expected": spec["expected"],
                "expected_aliases": spec["aliases"],
                "prompt": spec["prompt"],
            }
        )
        key = request_seed_key("calibration", public["task_id"])
        calibration_requests.append(
            request_row(
                request_key=key,
                prompt=spec["prompt"],
                meta={
                    "task_id": public["task_id"],
                    "family": "calibration",
                    "stratum": task["stratum"],
                    "arity": spec["arity"],
                    "expected": spec["expected"],
                },
            )
        )

    mechanics_public = [public_task(task) for task in tasks["mechanics"]]
    mechanics_audit = [audit_task(task) for task in tasks["mechanics"]]
    audit_by_id = {row["task_id"]: row for row in mechanics_audit}
    transport_requests: list[dict[str, Any]] = []
    suffix_requests = {name: [] for name in ("materialized", "name_only", "shuffled")}
    direct_requests: list[dict[str, Any]] = []
    for index, public in enumerate(mechanics_public):
        transport = transport_spec(public, audit_by_id[public["task_id"]], index)
        key = request_seed_key("transport", public["task_id"])
        transport_requests.append(
            request_row(
                request_key=key,
                prompt=transport["prompt"],
                meta={
                    "task_id": public["task_id"],
                    "family": "transport",
                    "arity": transport["arity"],
                    "expected": transport["expected"],
                },
            )
        )
        for candidate in CONCRETE_OPERATIONS:
            canonical = canonical_operation(candidate)
            key = request_seed_key(
                "suffix", public["task_id"], candidate=canonical
            )
            for name in suffix_requests:
                suffix_requests[name].append(
                    request_row(
                        request_key=key,
                        prompt=suffix_prompt(
                            public, candidate=candidate, representation=name
                        ),
                        meta={
                            "task_id": public["task_id"],
                            "family": "suffix",
                            "condition": name,
                            "candidate": {
                                "name": candidate[0],
                                "parameter": candidate[1],
                            },
                            "candidate_canonical": canonical,
                        },
                    )
                )
        for sample_index in range(96):
            key = request_seed_key(
                "direct", public["task_id"], sample_index=sample_index
            )
            direct_requests.append(
                request_row(
                    request_key=key,
                    prompt=direct_prompt(public),
                    meta={
                        "task_id": public["task_id"],
                        "family": "direct",
                        "sample_index": sample_index,
                    },
                )
            )
    return {
        "calibration": calibration_rows,
        "mechanics_public": mechanics_public,
        "mechanics_audit": mechanics_audit,
        "mechanics_gold": [gold_task(task) for task in tasks["mechanics"]],
    }, {
        "calibration": calibration_requests,
        "transport": transport_requests,
        **{f"suffix_{name}": rows for name, rows in suffix_requests.items()},
        "direct": direct_requests,
    }


def validate_request_freshness(
    requests: dict[str, list[dict[str, Any]]],
    parent: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    all_rows = [row for name in requests for row in requests[name]]
    ids = [row["id"] for row in all_rows]
    key_sha256s = [canonical_sha256(row["meta"]["seed_key"]) for row in all_rows]
    prompts = [row["messages"][0]["content"] for row in all_rows]
    prompt_sha256s = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
    ]
    task_ids = [row["meta"]["task_id"] for row in all_rows]
    if any(row["id"] != canonical_sha256(row["meta"]["seed_key"]) for row in all_rows):
        raise RuntimeError("request ID is not the canonical seed-key digest")
    if len(set(ids)) != 48 + 24 + 576 + 2304:
        # Three suffix arms deliberately share the same 576 causal IDs.
        raise RuntimeError("unique request-family geometry changed")
    suffix_orders = [
        [row["id"] for row in requests[f"suffix_{name}"]]
        for name in ("materialized", "name_only", "shuffled")
    ]
    if len({tuple(values) for values in suffix_orders}) != 1:
        raise RuntimeError("suffix causal arms do not share exact IDs/order")
    if set(ids) & parent["request_ids"]:
        raise RuntimeError("request IDs overlap the authenticated parent")
    if set(key_sha256s) & parent["seed_key_sha256s"]:
        raise RuntimeError("request seed keys overlap the authenticated parent")
    if set(prompt_sha256s) & parent["prompt_sha256s"]:
        raise RuntimeError("user prompts overlap the authenticated parent")
    if set(task_ids) & parent["task_ids"]:
        raise RuntimeError("task IDs overlap the authenticated parent")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    unique_prompts = sorted(set(prompts))
    prompt_token_sha256s: set[str] = set()
    for prompt in unique_prompts:
        for thinking in (False, True):
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            if isinstance(token_ids, Mapping):
                token_ids = token_ids.get("input_ids")
            if (
                not isinstance(token_ids, list)
                or any(type(value) is not int for value in token_ids)
            ):
                raise RuntimeError("rendered token sequence changed type")
            prompt_token_sha256s.add(canonical_sha256(token_ids))
    if len(prompt_token_sha256s) != 2 * len(unique_prompts):
        raise RuntimeError("current rendered prompt-token sequences collide")
    if prompt_token_sha256s & parent["prompt_token_sha256s"]:
        raise RuntimeError("rendered prompt-token sequences overlap the parent")
    representative_rows = list({row["id"]: row for row in all_rows}.values())
    derived_seeds = seed_inventory(representative_rows, config, parent=False)
    if derived_seeds & parent["derived_seeds"]:
        raise RuntimeError("derived runner seeds overlap the authenticated parent")
    transport_rows = requests["transport"]
    transport_namespace = config["identity"]["transport_request_namespace"]
    if any(
        row["meta"]["seed_key"][:2] != [transport_namespace, "transport"]
        for row in transport_rows
    ):
        raise RuntimeError("transport seed keys escaped their frozen namespace")
    nontransport_ids = {
        row["id"] for name, rows in requests.items() if name != "transport" for row in rows
    }
    if {row["id"] for row in transport_rows} & nontransport_ids:
        raise RuntimeError("transport request IDs overlap another current family")
    return {
        "total_rows_with_causal_duplicates": len(all_rows),
        "unique_request_ids": len(set(ids)),
        "unique_seed_keys": len(set(key_sha256s)),
        "unique_user_prompts": len(set(prompts)),
        "unique_rendered_prompt_token_sequences": len(prompt_token_sha256s),
        "same_suffix_ids_order": True,
        "parent_request_id_overlap": 0,
        "parent_seed_key_overlap": 0,
        "parent_user_prompt_overlap": 0,
        "parent_task_id_overlap": 0,
        "parent_rendered_prompt_token_sequence_overlap": 0,
        "derived_runner_seeds": len(derived_seeds),
        "parent_derived_runner_seed_overlap": 0,
        "tokenizer_loaded": True,
        "model_loaded": False,
        "model_calls": 0,
        "transport_request_namespace": transport_namespace,
        "transport_request_id_overlap_current_families": 0,
    }


def output_table(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        rows = read_jsonl(path) if path.suffix == ".jsonl" else None
        result[str(path.relative_to(ROOT))] = {
            "sha256": file_sha256(path),
            **({"rows": len(rows)} if rows is not None else {}),
        }
    return dict(sorted(result.items()))


def main() -> int:
    completed = validate_completed_construction()
    if completed is not None:
        print(json.dumps(completed, indent=2, sort_keys=True))
        return 0
    config = yaml.safe_load(CONFIG.read_text())
    if config.get("model", {}).get("id") != MODEL_ID or config.get("model", {}).get("revision") != MODEL_REVISION:
        raise RuntimeError("construction model/tokenizer identity changed")
    parent_public_fingerprints, parent_function_fingerprints, parent = parent_inventory()
    tasks, construction = build_tasks(
        config,
        excluded_public_fingerprints=parent_public_fingerprints,
        excluded_function_fingerprints=parent_function_fingerprints,
    )
    data_rows, requests = build_requests(tasks)
    freshness = validate_request_freshness(requests, parent, config)

    data_paths = {
        "calibration": DATA / "calibration_public.jsonl",
        "mechanics_public": DATA / "mechanics_public.jsonl",
        "mechanics_audit": DATA / "mechanics_audit.jsonl",
    }
    request_paths = {
        name: PREPARED / f"{name}_requests.jsonl" for name in requests
    }
    for name, path in data_paths.items():
        write_frozen(path, data_rows[name], jsonl=True)
    for name, path in request_paths.items():
        write_frozen(path, requests[name], jsonl=True)
    hidden_ciphertext = write_hidden_ciphertext(data_rows["mechanics_gold"])

    summary = {
        **construction,
        "decision": "CONSTRUCTION_PASS",
        "model": config["model"]["id"],
        "revision": config["model"]["revision"],
        "request_freshness": freshness,
        "parent_read_receipt": parent["reads"],
        "data_files": output_table(list(data_paths.values())),
        "hidden_ciphertext": hidden_ciphertext,
        "prepared_request_files": output_table(list(request_paths.values())),
        "calibration_expected_alias_position_counts": {
            str(arity): {
                str(position): dict(
                    sorted(
                        {
                            alias: sum(
                                row["arity"] == arity
                                and row["expected_aliases"][position] == alias
                                for row in data_rows["calibration"]
                            )
                            for alias in "ABCDEFGHIJKLMNOPQRSTUVWX"
                        }.items()
                    )
                )
                for position in range(arity)
            }
            for arity in (2, 3)
        },
        "calibration_strata_by_arity": {
            str(arity): dict(
                sorted(
                    {
                        stratum: sum(
                            row["arity"] == arity and row["stratum"] == stratum
                            for row in data_rows["calibration"]
                        )
                        for stratum in ("single", "double", "triple", "quad")
                    }.items()
                )
            )
            for arity in (2, 3)
        },
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
        "model_loaded": False,
        "model_calls": 0,
        "sampled_model_outputs": 0,
        "tokenizer_loaded": True,
    }
    write_frozen(CONSTRUCTION, summary)
    preoutcome = {
        "schema_version": 2,
        "stage": "preoutcome_prepare",
        "decision": "PREOUTCOME_PASS",
        "construction_summary_sha256": file_sha256(CONSTRUCTION),
        "config_sha256": file_sha256(CONFIG),
        "request_files": output_table(list(request_paths.values())),
        "hidden_ciphertext": hidden_ciphertext,
        "expected_invocation_rows": {
            "calibration_each_cell": 48,
            "calibration_answer_total": 384,
            "calibration_shared_thought": 48,
            "transport_selected_interface": 24,
            "suffix_materialized": 576,
            "suffix_name_only": 576,
            "suffix_shuffled": 576,
            "direct_master_pool": 2304,
        },
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
        "model_loaded": False,
        "model_calls": 0,
        "sampled_model_outputs": 0,
        "tokenizer_loaded": True,
    }
    write_frozen(PREOUTCOME, preoutcome)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

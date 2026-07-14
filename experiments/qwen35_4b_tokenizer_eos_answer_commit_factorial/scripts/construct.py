#!/usr/bin/env python3
"""Build the fresh tasks and outcome-blind request inventory byte-identically."""

from __future__ import annotations

import hashlib
import json
import secrets
import sys
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from identity import (  # noqa: E402
    PARENT_CONSTRUCTION,
    PARENT_CONFIG,
    PARENT_PREOUTCOME,
    canonical_sha256,
    file_sha256,
    request_id,
    request_seed_key,
    verified_parent_preoutcome,
    verified_parent_construction,
    verified_parent_config,
    verified_parent_public_files,
    verified_regular_file,
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
    public_instance_fingerprint,
    public_task,
)


CONFIG = EXP / "configs" / "default.yaml"
DATA = EXP / "data" / "procedural"
PREPARED = EXP / "runs" / "prepared"
CONSTRUCTION = EXP / "runs" / "construction" / "summary.json"
PREOUTCOME = PREPARED / "preoutcome_receipt.json"
HIDDEN_CIPHERTEXT = DATA / "mechanics_gold.jsonl.aesgcm"
HIDDEN_KEY = EXP / ".secrets" / "mechanics_gold.aes256.key"
HIDDEN_AAD = b"tokenizer-eos-answer-commit-factorial-v1/mechanics-gold-v1"
HIDDEN_MAGIC = b"AESGCM1\0"


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
    """Seal hidden plaintext behind a local untracked AES-256-GCM key."""

    plaintext = jsonl_bytes(rows)
    if HIDDEN_KEY.is_symlink() or HIDDEN_CIPHERTEXT.is_symlink():
        raise RuntimeError("hidden key/ciphertext path is a symlink")
    HIDDEN_KEY.parent.mkdir(parents=True, exist_ok=True)
    if HIDDEN_KEY.exists():
        if not HIDDEN_KEY.is_file():
            raise RuntimeError("hidden key path is not a regular file")
        key = HIDDEN_KEY.read_bytes()
    else:
        key = secrets.token_bytes(32)
        with HIDDEN_KEY.open("xb") as handle:
            handle.write(key)
    if len(key) != 32:
        raise RuntimeError("hidden AES-256-GCM key length changed")
    if HIDDEN_CIPHERTEXT.exists():
        blob = HIDDEN_CIPHERTEXT.read_bytes()
        if not blob.startswith(HIDDEN_MAGIC) or len(blob) <= len(HIDDEN_MAGIC) + 12:
            raise RuntimeError("hidden ciphertext envelope changed")
        nonce = blob[len(HIDDEN_MAGIC) : len(HIDDEN_MAGIC) + 12]
        observed = AESGCM(key).decrypt(
            nonce, blob[len(HIDDEN_MAGIC) + 12 :], HIDDEN_AAD
        )
        if observed != plaintext:
            raise RuntimeError("hidden ciphertext decrypts to different construction")
    else:
        nonce = secrets.token_bytes(12)
        blob = (
            HIDDEN_MAGIC
            + nonce
            + AESGCM(key).encrypt(nonce, plaintext, HIDDEN_AAD)
        )
        HIDDEN_CIPHERTEXT.parent.mkdir(parents=True, exist_ok=True)
        temporary = HIDDEN_CIPHERTEXT.with_name(HIDDEN_CIPHERTEXT.name + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise RuntimeError("unsafe hidden ciphertext temporary path")
        temporary.write_bytes(blob)
        temporary.replace(HIDDEN_CIPHERTEXT)
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


def parent_inventory() -> tuple[set[str], dict[str, Any]]:
    fingerprints: set[str] = set()
    read_receipt: dict[str, dict[str, Any]] = {}
    for split, path in verified_parent_public_files(ROOT).items():
        rows = read_jsonl(path)
        fingerprints.update(public_instance_fingerprint(row) for row in rows)
        read_receipt[str(path.relative_to(ROOT))] = {
            "sha256": file_sha256(path),
            "rows": len(rows),
            "purpose": "freshness_fingerprints_only",
        }

    construction_path = verified_parent_construction(ROOT)
    parent_construction = read_json(construction_path)
    construction_fingerprints = parent_construction.get(
        "public_instance_fingerprints"
    )
    if (
        not isinstance(construction_fingerprints, list)
        or len(construction_fingerprints) != 72
        or any(not isinstance(value, str) for value in construction_fingerprints)
    ):
        raise RuntimeError("parent construction fingerprint inventory changed")
    fingerprints.update(construction_fingerprints)
    read_receipt[str(construction_path.relative_to(ROOT))] = {
        "sha256": PARENT_CONSTRUCTION["sha256"],
        "purpose": "authenticated_72_public_fingerprint_inventory",
    }

    receipt_path = verified_parent_preoutcome(ROOT)
    receipt = read_json(receipt_path)
    if file_sha256(receipt_path) != PARENT_PREOUTCOME["sha256"]:
        raise RuntimeError("parent preoutcome receipt changed after authentication")
    parent_ids: set[str] = set()
    parent_seed_keys: set[str] = set()
    parent_prompts: set[str] = set()
    parent_request_rows: list[dict[str, Any]] = []
    for relative, row in sorted(receipt["request_files"].items()):
        if not relative.endswith("_requests.jsonl"):
            continue
        path = verified_regular_file(ROOT, relative, row["sha256"])
        values = read_jsonl(path)
        if len(values) != row["rows"]:
            raise RuntimeError("parent prepared row count changed")
        read_receipt[str(path.relative_to(ROOT))] = {
            "sha256": row["sha256"],
            "rows": len(values),
            "purpose": "request_identity_and_prompt_freshness_only",
        }
        for value in values:
            if value["id"] not in parent_ids:
                parent_request_rows.append(value)
            parent_ids.add(value["id"])
            parent_seed_keys.add(
                json.dumps(value["meta"]["seed_key"], sort_keys=True)
            )
            parent_prompts.add(value["messages"][0]["content"])
    read_receipt[str(receipt_path.relative_to(ROOT))] = {
        "sha256": PARENT_PREOUTCOME["sha256"],
        "purpose": "authenticated_parent_prepared_manifest",
    }
    parent_config_path = verified_parent_config(ROOT)
    parent_config = yaml.safe_load(parent_config_path.read_text())
    parent_derived_seeds = seed_inventory(
        parent_request_rows, parent_config, parent=True
    )
    read_receipt[str(parent_config_path.relative_to(ROOT))] = {
        "sha256": PARENT_CONFIG["sha256"],
        "purpose": "authenticated_parent_seed_inventory",
    }
    return fingerprints, {
        "reads": dict(sorted(read_receipt.items())),
        "request_ids": parent_ids,
        "seed_keys": parent_seed_keys,
        "prompts": parent_prompts,
        "derived_seeds": parent_derived_seeds,
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
    keys = [json.dumps(row["meta"]["seed_key"], sort_keys=True) for row in all_rows]
    prompts = [row["messages"][0]["content"] for row in all_rows]
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
    if set(keys) & parent["seed_keys"]:
        raise RuntimeError("request seed keys overlap the authenticated parent")
    if set(prompts) & parent["prompts"]:
        raise RuntimeError("user prompts overlap the authenticated parent")
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
        "unique_seed_keys": len(set(keys)),
        "unique_user_prompts": len(set(prompts)),
        "same_suffix_ids_order": True,
        "parent_request_id_overlap": 0,
        "parent_seed_key_overlap": 0,
        "parent_user_prompt_overlap": 0,
        "derived_runner_seeds": len(derived_seeds),
        "parent_derived_runner_seed_overlap": 0,
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
    config = yaml.safe_load(CONFIG.read_text())
    parent_fingerprints, parent = parent_inventory()
    tasks, construction = build_tasks(
        config, excluded_public_fingerprints=parent_fingerprints
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
    }
    write_frozen(PREOUTCOME, preoutcome)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

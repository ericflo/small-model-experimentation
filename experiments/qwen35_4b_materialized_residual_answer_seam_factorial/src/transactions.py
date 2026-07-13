"""Append-only durable generation transactions with fail-closed recovery."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SUFFIXES = (
    "started.json",
    "generated.json",
    "generated.receipt.json",
    "complete.json",
)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe file for hashing: {path}")
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_durable(path: Path, value: Any) -> None:
    data = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise RuntimeError(f"transaction path is a symlink: {path}")
    _fsync_directory(path.parent)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite transaction artifact: {path}") from error
    _fsync_directory(path.parent)


def redurable(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"cannot redurable unsafe artifact: {path}")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def read_canonical(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"transaction artifact is unsafe or absent: {path}")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in transaction artifact: {path}")
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid transaction JSON: {path}") from error
    if raw != json_bytes(value):
        raise RuntimeError(f"noncanonical transaction bytes: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"prepared request file is unsafe or absent: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"prepared row is not an object: {path}:{number}")
        rows.append(value)
    return rows


def artifact_paths(raw_dir: Path, invocation: str) -> dict[str, Path]:
    if not invocation or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in invocation):
        raise ValueError("invocation name must be lowercase snake_case")
    return {
        "started": raw_dir / f"{invocation}.started.json",
        "bundle": raw_dir / f"{invocation}.generated.json",
        "generated": raw_dir / f"{invocation}.generated.receipt.json",
        "complete": raw_dir / f"{invocation}.complete.json",
    }


def inventory_state(raw_dir: Path, invocation: str) -> str:
    paths = artifact_paths(raw_dir, invocation)
    present: set[str] = set()
    for name, path in paths.items():
        if path.is_symlink():
            raise RuntimeError(f"transaction artifact is a symlink: {path}")
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"transaction artifact is not regular: {path}")
            present.add(name)
    states = {
        frozenset(): "absent",
        frozenset({"started"}): "started_only",
        frozenset({"started", "bundle"}): "bundle_durable",
        frozenset({"started", "bundle", "generated"}): "generated_durable",
        frozenset({"started", "bundle", "generated", "complete"}): "complete",
    }
    try:
        return states[frozenset(present)]
    except KeyError as error:
        raise RuntimeError(
            f"invalid append-only transaction state for {invocation}: {sorted(present)}"
        ) from error


def audit_directory_inventory(raw_dir: Path, invocations: Sequence[str]) -> None:
    if raw_dir.is_symlink():
        raise RuntimeError("transaction directory is a symlink")
    if not raw_dir.exists():
        return
    if not raw_dir.is_dir():
        raise RuntimeError("transaction path is not a directory")
    allowed = {
        path.name
        for invocation in invocations
        for path in artifact_paths(raw_dir, invocation).values()
    }
    observed = {path.name for path in raw_dir.iterdir()}
    unknown = observed - allowed
    if unknown:
        raise RuntimeError(f"unknown transaction inventory: {sorted(unknown)}")
    for path in raw_dir.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"transaction inventory entry is a symlink: {path}")
        if not path.is_file():
            raise RuntimeError(f"unsafe transaction inventory entry: {path}")


def _validate_prepared(rows: Sequence[dict[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise RuntimeError("prepared request row count changed")
    ids = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "messages", "meta"}
            or not isinstance(row["id"], str)
            or not row["id"]
        ):
            raise RuntimeError("prepared request schema changed")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise RuntimeError("prepared request IDs collide within invocation")


def _started_value(
    *,
    invocation: str,
    prepared_path: Path,
    expected_rows: int,
    implementation_lock_path: Path,
    live_preflight_path: Path,
    runner_path: Path,
    sampling: Mapping[str, Any],
    predecessor_complete_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "STARTED",
        "invocation": invocation,
        "prepared_path": str(prepared_path),
        "prepared_sha256": sha256_file(prepared_path),
        "expected_rows": expected_rows,
        "implementation_lock_path": str(implementation_lock_path),
        "implementation_lock_sha256": sha256_file(implementation_lock_path),
        "live_preflight_path": str(live_preflight_path),
        "live_preflight_sha256": sha256_file(live_preflight_path),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "sampling": dict(sampling),
        "predecessor_complete_sha256": predecessor_complete_sha256,
    }


def _authenticate_started(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    observed = read_canonical(path)
    if observed != expected:
        raise RuntimeError("STARTED receipt differs from current locked invocation")
    return observed


def _validate_bundle(
    bundle: Any,
    *,
    invocation: str,
    prepared_rows: Sequence[dict[str, Any]],
    runner_sha256: str,
) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "invocation",
        "rows",
        "runner_metadata",
    }:
        raise RuntimeError("generated bundle schema changed")
    rows = bundle["rows"]
    metadata = bundle["runner_metadata"]
    if (
        bundle["schema_version"] != 1
        or bundle["invocation"] != invocation
        or not isinstance(rows, list)
        or not isinstance(metadata, dict)
        or [row.get("id") for row in rows]
        != [row["id"] for row in prepared_rows]
        or any(not isinstance(row.get("outputs"), list) or not row["outputs"] for row in rows)
    ):
        raise RuntimeError("generated bundle identity/order changed")
    if (
        metadata.get("model") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("runner_sha256") != runner_sha256
        or metadata.get("counts", {}).get("requests") != len(rows)
    ):
        raise RuntimeError("generated bundle runner metadata changed")
    return bundle


def _generated_receipt_value(
    *,
    invocation: str,
    started_path: Path,
    bundle_path: Path,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    bundle_bytes = bundle_path.stat().st_size
    outputs = sum(len(row["outputs"]) for row in bundle["rows"])
    return {
        "schema_version": 1,
        "state": "GENERATED",
        "invocation": invocation,
        "started_sha256": sha256_file(started_path),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": bundle_bytes,
        "rows": len(bundle["rows"]),
        "sampled_outputs": outputs,
    }


def _complete_value(
    *,
    invocation: str,
    started_path: Path,
    bundle_path: Path,
    generated_path: Path,
    predecessor_complete_sha256: str | None,
) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "state": "COMPLETE",
        "invocation": invocation,
        "started_sha256": sha256_file(started_path),
        "bundle_sha256": sha256_file(bundle_path),
        "generated_receipt_sha256": sha256_file(generated_path),
        "predecessor_complete_sha256": predecessor_complete_sha256,
    }
    return {
        **core,
        "chain_sha256": canonical_sha256(core),
    }


def _promote_bundle(
    *,
    invocation: str,
    paths: dict[str, Path],
    prepared_rows: Sequence[dict[str, Any]],
    runner_sha256: str,
    predecessor_complete_sha256: str | None,
    crash_after: str | None,
) -> dict[str, Any]:
    redurable(paths["bundle"])
    bundle = _validate_bundle(
        read_canonical(paths["bundle"]),
        invocation=invocation,
        prepared_rows=prepared_rows,
        runner_sha256=runner_sha256,
    )
    generated = _generated_receipt_value(
        invocation=invocation,
        started_path=paths["started"],
        bundle_path=paths["bundle"],
        bundle=bundle,
    )
    if paths["generated"].exists():
        if read_canonical(paths["generated"]) != generated:
            raise RuntimeError("GENERATED receipt authentication failed")
    else:
        write_exclusive_durable(paths["generated"], generated)
    if crash_after == "generated":
        raise RuntimeError("injected crash after GENERATED")
    complete = _complete_value(
        invocation=invocation,
        started_path=paths["started"],
        bundle_path=paths["bundle"],
        generated_path=paths["generated"],
        predecessor_complete_sha256=predecessor_complete_sha256,
    )
    if paths["complete"].exists():
        if read_canonical(paths["complete"]) != complete:
            raise RuntimeError("COMPLETE receipt authentication failed")
    else:
        write_exclusive_durable(paths["complete"], complete)
    if crash_after == "complete":
        raise RuntimeError("injected crash after COMPLETE")
    return complete


def run_transaction(
    *,
    raw_dir: Path,
    invocation: str,
    invocation_order: Sequence[str],
    prepared_path: Path,
    expected_rows: int,
    implementation_lock_path: Path,
    live_preflight_path: Path,
    runner_path: Path,
    sampling: Mapping[str, Any],
    generate: Callable[[Sequence[dict[str, Any]], Mapping[str, Any]], tuple[list[dict[str, Any]], dict[str, Any]]],
    crash_after: str | None = None,
) -> dict[str, Any]:
    if crash_after not in {None, "started", "bundle", "generated", "complete"}:
        raise ValueError("unknown crash injection boundary")
    if invocation not in invocation_order or len(set(invocation_order)) != len(invocation_order):
        raise ValueError("invocation order is invalid")
    audit_directory_inventory(raw_dir, invocation_order)
    position = list(invocation_order).index(invocation)
    predecessor_sha: str | None = None
    if position:
        predecessor_path = artifact_paths(raw_dir, invocation_order[position - 1])["complete"]
        if not predecessor_path.is_file() or predecessor_path.is_symlink():
            raise RuntimeError("predecessor invocation is not complete")
        predecessor = read_canonical(predecessor_path)
        if predecessor.get("state") != "COMPLETE":
            raise RuntimeError("predecessor COMPLETE receipt changed")
        predecessor_sha = sha256_file(predecessor_path)
    for later in invocation_order[position + 1 :]:
        if inventory_state(raw_dir, later) != "absent":
            raise RuntimeError("later invocation exists before its predecessor")

    prepared_rows = read_jsonl(prepared_path)
    _validate_prepared(prepared_rows, expected_rows)
    paths = artifact_paths(raw_dir, invocation)
    started = _started_value(
        invocation=invocation,
        prepared_path=prepared_path,
        expected_rows=expected_rows,
        implementation_lock_path=implementation_lock_path,
        live_preflight_path=live_preflight_path,
        runner_path=runner_path,
        sampling=sampling,
        predecessor_complete_sha256=predecessor_sha,
    )
    state = inventory_state(raw_dir, invocation)
    if state == "started_only":
        _authenticate_started(paths["started"], started)
        raise RuntimeError(
            f"terminal STARTED-only transaction; refusing to resample: {invocation}"
        )
    if state in {"bundle_durable", "generated_durable", "complete"}:
        _authenticate_started(paths["started"], started)
        return _promote_bundle(
            invocation=invocation,
            paths=paths,
            prepared_rows=prepared_rows,
            runner_sha256=started["runner_sha256"],
            predecessor_complete_sha256=predecessor_sha,
            crash_after=crash_after,
        )
    if state != "absent":
        raise RuntimeError(f"unhandled transaction state: {state}")

    write_exclusive_durable(paths["started"], started)
    if crash_after == "started":
        raise RuntimeError("injected crash after STARTED")
    rows, runner_metadata = generate(prepared_rows, sampling)
    bundle = {
        "schema_version": 1,
        "invocation": invocation,
        "rows": rows,
        "runner_metadata": runner_metadata,
    }
    _validate_bundle(
        bundle,
        invocation=invocation,
        prepared_rows=prepared_rows,
        runner_sha256=started["runner_sha256"],
    )
    write_exclusive_durable(paths["bundle"], bundle)
    if crash_after == "bundle":
        raise RuntimeError("injected crash after bundle")
    return _promote_bundle(
        invocation=invocation,
        paths=paths,
        prepared_rows=prepared_rows,
        runner_sha256=started["runner_sha256"],
        predecessor_complete_sha256=predecessor_sha,
        crash_after=crash_after,
    )


def _authenticate_complete_prefix(
    *,
    raw_dir: Path,
    invocation_order: Sequence[str],
    completed_count: int,
) -> dict[str, Any]:
    if (
        not invocation_order
        or len(set(invocation_order)) != len(invocation_order)
        or not 1 <= completed_count <= len(invocation_order)
    ):
        raise ValueError("authenticated prefix geometry is invalid")
    audit_directory_inventory(raw_dir, invocation_order)
    for invocation in invocation_order[completed_count:]:
        if inventory_state(raw_dir, invocation) != "absent":
            raise RuntimeError(
                f"authentication requires later invocation to be absent: {invocation}"
            )
    predecessor_sha: str | None = None
    rows = 0
    outputs = 0
    complete_files: dict[str, dict[str, Any]] = {}
    for invocation in invocation_order[:completed_count]:
        if inventory_state(raw_dir, invocation) != "complete":
            raise RuntimeError(f"authentication requires COMPLETE: {invocation}")
        paths = artifact_paths(raw_dir, invocation)
        started = read_canonical(paths["started"])
        bundle = read_canonical(paths["bundle"])
        generated = read_canonical(paths["generated"])
        complete = read_canonical(paths["complete"])
        if not isinstance(started, dict) or set(started) != {
            "schema_version",
            "state",
            "invocation",
            "prepared_path",
            "prepared_sha256",
            "expected_rows",
            "implementation_lock_path",
            "implementation_lock_sha256",
            "live_preflight_path",
            "live_preflight_sha256",
            "model",
            "revision",
            "runner_path",
            "runner_sha256",
            "sampling",
            "predecessor_complete_sha256",
        }:
            raise RuntimeError("STARTED receipt schema changed")
        if (
            started["schema_version"] != 1
            or started["state"] != "STARTED"
            or started["invocation"] != invocation
            or started["model"] != MODEL_ID
            or started["revision"] != MODEL_REVISION
            or started["predecessor_complete_sha256"] != predecessor_sha
            or not isinstance(started["sampling"], dict)
        ):
            raise RuntimeError("STARTED identity/predecessor changed")
        prepared_path = Path(started["prepared_path"])
        lock_path = Path(started["implementation_lock_path"])
        preflight_path = Path(started["live_preflight_path"])
        runner_path = Path(started["runner_path"])
        current_files = (
            (prepared_path, started["prepared_sha256"], "prepared"),
            (lock_path, started["implementation_lock_sha256"], "implementation lock"),
            (preflight_path, started["live_preflight_sha256"], "live preflight"),
            (runner_path, started["runner_sha256"], "runner"),
        )
        for path, expected_sha, label in current_files:
            if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
                raise RuntimeError(f"{label} changed after STARTED")
        prepared_rows = read_jsonl(prepared_path)
        expected_rows = started["expected_rows"]
        if not isinstance(expected_rows, int) or isinstance(expected_rows, bool):
            raise RuntimeError("STARTED expected row count changed")
        _validate_prepared(prepared_rows, expected_rows)
        _validate_bundle(
            bundle,
            invocation=invocation,
            prepared_rows=prepared_rows,
            runner_sha256=started["runner_sha256"],
        )
        expected_generated = _generated_receipt_value(
            invocation=invocation,
            started_path=paths["started"],
            bundle_path=paths["bundle"],
            bundle=bundle,
        )
        if generated != expected_generated:
            raise RuntimeError("GENERATED receipt changed during authentication")
        expected_complete = _complete_value(
            invocation=invocation,
            started_path=paths["started"],
            bundle_path=paths["bundle"],
            generated_path=paths["generated"],
            predecessor_complete_sha256=predecessor_sha,
        )
        if complete != expected_complete:
            raise RuntimeError("COMPLETE receipt changed during authentication")
        rows += len(bundle["rows"])
        outputs += sum(len(row["outputs"]) for row in bundle["rows"])
        complete_files[invocation] = {
            "complete_sha256": sha256_file(paths["complete"]),
            "chain_sha256": complete["chain_sha256"],
        }
        predecessor_sha = sha256_file(paths["complete"])
    return {
        "schema_version": 1,
        "decision": "TRANSACTION_PREFIX_AUTHENTICATED",
        "invocation_order": list(invocation_order),
        "authenticated_invocations": list(invocation_order[:completed_count]),
        "complete_files": complete_files,
        "rows": rows,
        "sampled_outputs": outputs,
        "terminal_complete_sha256": predecessor_sha,
    }


def authenticate_complete_prefix(
    *, raw_dir: Path, invocation_order: Sequence[str], through: str
) -> dict[str, Any]:
    if through not in invocation_order:
        raise ValueError("prefix endpoint is not in invocation order")
    return _authenticate_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=invocation_order,
        completed_count=list(invocation_order).index(through) + 1,
    )


def authenticate_complete_chain(
    *, raw_dir: Path, invocation_order: Sequence[str]
) -> dict[str, Any]:
    receipt = _authenticate_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=invocation_order,
        completed_count=len(invocation_order),
    )
    return {
        **receipt,
        "decision": "TRANSACTION_CHAIN_AUTHENTICATED",
    }

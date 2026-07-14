"""Exact-typed durable transactions for the post-calibration mechanics stage.

The calibration transaction implementation is already part of the immutable
calibration lock.  Mechanics therefore uses this additive implementation so
that durable JSON authentication cannot inherit Python's ``bool == int``
semantics without changing any calibration-locked byte.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from transactions import (
    MODEL_ID,
    MODEL_REVISION,
    artifact_paths,
    audit_directory_inventory,
    canonical_sha256,
    inventory_state,
    read_canonical,
    read_jsonl,
    redurable,
    sha256_file,
    write_exclusive_durable,
)


def json_native(value: Any) -> Any:
    """Normalize a serializable value to its durable JSON-domain form."""

    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )


def exact_json_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON-domain values recursively with exact Python types."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            exact_json_equal(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            exact_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _exact_int(value: Any, expected: int | None = None) -> bool:
    return type(value) is int and (expected is None or value == expected)


def _validate_prepared(rows: Sequence[dict[str, Any]], expected_rows: int) -> None:
    if not _exact_int(expected_rows) or expected_rows < 0 or len(rows) != expected_rows:
        raise RuntimeError("prepared request row count changed")
    ids: list[str] = []
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
    authorization_paths: Mapping[str, Path],
    predecessor_complete_sha256: str | None,
) -> dict[str, Any]:
    if any(
        not label
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in label
        )
        for label in authorization_paths
    ):
        raise ValueError("authorization labels must be lowercase snake_case")
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
        "sampling": json_native(dict(sampling)),
        "authorization_files": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(authorization_paths.items())
        },
        "predecessor_complete_sha256": predecessor_complete_sha256,
    }


def _authenticate_started(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    observed = read_canonical(path)
    if not exact_json_equal(observed, expected):
        raise RuntimeError("STARTED receipt differs from current locked invocation")
    return observed


def _validate_bundle(
    bundle: Any,
    *,
    invocation: str,
    prepared_rows: Sequence[dict[str, Any]],
    runner_sha256: str,
    sampling: Mapping[str, Any],
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
        not _exact_int(bundle["schema_version"], 1)
        or bundle["invocation"] != invocation
        or not isinstance(rows, list)
        or not isinstance(metadata, dict)
        or not exact_json_equal(
            [row.get("id") for row in rows],
            [row["id"] for row in prepared_rows],
        )
        or not exact_json_equal(
            [row.get("meta") for row in rows],
            [row["meta"] for row in prepared_rows],
        )
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("outputs"), list)
            or not row["outputs"]
            for row in rows
        )
    ):
        raise RuntimeError("generated bundle identity/order changed")
    counts = metadata.get("counts")
    expected_completions = sum(len(row["outputs"]) for row in rows)
    if (
        metadata.get("model") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("runner_sha256") != runner_sha256
        or not isinstance(counts, dict)
        or not _exact_int(counts.get("requests"), len(rows))
        or not _exact_int(counts.get("completions"), expected_completions)
        or not exact_json_equal(
            metadata.get("sampling"), json_native(dict(sampling))
        )
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
    return {
        "schema_version": 1,
        "state": "GENERATED",
        "invocation": invocation,
        "started_sha256": sha256_file(started_path),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "rows": len(bundle["rows"]),
        "sampled_outputs": sum(
            len(row["outputs"]) for row in bundle["rows"]
        ),
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
    return {**core, "chain_sha256": canonical_sha256(core)}


def _promote_bundle(
    *,
    invocation: str,
    paths: dict[str, Path],
    prepared_rows: Sequence[dict[str, Any]],
    runner_sha256: str,
    sampling: Mapping[str, Any],
    predecessor_complete_sha256: str | None,
    crash_after: str | None,
) -> dict[str, Any]:
    redurable(paths["bundle"])
    bundle = _validate_bundle(
        read_canonical(paths["bundle"]),
        invocation=invocation,
        prepared_rows=prepared_rows,
        runner_sha256=runner_sha256,
        sampling=sampling,
    )
    generated = _generated_receipt_value(
        invocation=invocation,
        started_path=paths["started"],
        bundle_path=paths["bundle"],
        bundle=bundle,
    )
    if paths["generated"].exists():
        if not exact_json_equal(read_canonical(paths["generated"]), generated):
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
        if not exact_json_equal(read_canonical(paths["complete"]), complete):
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
    authorization_paths: Mapping[str, Path],
    generate: Callable[
        [Sequence[dict[str, Any]], Mapping[str, Any]],
        tuple[list[dict[str, Any]], dict[str, Any]],
    ],
    crash_after: str | None = None,
) -> dict[str, Any]:
    if crash_after not in {None, "started", "bundle", "generated", "complete"}:
        raise ValueError("unknown crash injection boundary")
    if invocation not in invocation_order or len(set(invocation_order)) != len(
        invocation_order
    ):
        raise ValueError("invocation order is invalid")
    audit_directory_inventory(raw_dir, invocation_order)
    position = list(invocation_order).index(invocation)
    predecessor_sha: str | None = None
    if position:
        predecessor = _authenticate_complete_prefix(
            raw_dir=raw_dir,
            invocation_order=invocation_order,
            completed_count=position,
            require_later_absent=False,
        )
        predecessor_sha = predecessor["terminal_complete_sha256"]
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
        authorization_paths=authorization_paths,
        predecessor_complete_sha256=predecessor_sha,
    )

    def assert_predecessor_unchanged() -> None:
        if position:
            predecessor_path = artifact_paths(
                raw_dir, invocation_order[position - 1]
            )["complete"]
            if sha256_file(predecessor_path) != predecessor_sha:
                raise RuntimeError("predecessor changed after exact authentication")

    state = inventory_state(raw_dir, invocation)
    if state == "started_only":
        _authenticate_started(paths["started"], started)
        raise RuntimeError(
            f"terminal STARTED-only transaction; refusing to resample: {invocation}"
        )
    if state in {"bundle_durable", "generated_durable", "complete"}:
        _authenticate_started(paths["started"], started)
        assert_predecessor_unchanged()
        return _promote_bundle(
            invocation=invocation,
            paths=paths,
            prepared_rows=prepared_rows,
            runner_sha256=started["runner_sha256"],
            sampling=started["sampling"],
            predecessor_complete_sha256=predecessor_sha,
            crash_after=crash_after,
        )
    if state != "absent":
        raise RuntimeError(f"unhandled transaction state: {state}")

    write_exclusive_durable(paths["started"], started)
    if crash_after == "started":
        raise RuntimeError("injected crash after STARTED")
    assert_predecessor_unchanged()
    rows, runner_metadata = generate(prepared_rows, sampling)
    bundle = json_native(
        {
            "schema_version": 1,
            "invocation": invocation,
            "rows": rows,
            "runner_metadata": runner_metadata,
        }
    )
    _validate_bundle(
        bundle,
        invocation=invocation,
        prepared_rows=prepared_rows,
        runner_sha256=started["runner_sha256"],
        sampling=started["sampling"],
    )
    write_exclusive_durable(paths["bundle"], bundle)
    if crash_after == "bundle":
        raise RuntimeError("injected crash after bundle")
    return _promote_bundle(
        invocation=invocation,
        paths=paths,
        prepared_rows=prepared_rows,
        runner_sha256=started["runner_sha256"],
        sampling=started["sampling"],
        predecessor_complete_sha256=predecessor_sha,
        crash_after=crash_after,
    )


def _authenticate_complete_prefix(
    *,
    raw_dir: Path,
    invocation_order: Sequence[str],
    completed_count: int,
    require_later_absent: bool = True,
) -> dict[str, Any]:
    if (
        not invocation_order
        or len(set(invocation_order)) != len(invocation_order)
        or not _exact_int(completed_count)
        or not 1 <= completed_count <= len(invocation_order)
        or type(require_later_absent) is not bool
    ):
        raise ValueError("authenticated prefix geometry is invalid")
    audit_directory_inventory(raw_dir, invocation_order)
    if require_later_absent:
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
        started_keys = {
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
            "authorization_files",
            "predecessor_complete_sha256",
        }
        if not isinstance(started, dict) or set(started) != started_keys:
            raise RuntimeError("STARTED receipt schema changed")
        if (
            not _exact_int(started["schema_version"], 1)
            or started["state"] != "STARTED"
            or started["invocation"] != invocation
            or started["model"] != MODEL_ID
            or started["revision"] != MODEL_REVISION
            or started["predecessor_complete_sha256"] != predecessor_sha
            or not isinstance(started["sampling"], dict)
            or not isinstance(started["authorization_files"], dict)
            or not _exact_int(started["expected_rows"])
        ):
            raise RuntimeError("STARTED identity/predecessor changed")
        prepared_path = Path(started["prepared_path"])
        current_files = (
            (prepared_path, started["prepared_sha256"], "prepared"),
            (
                Path(started["implementation_lock_path"]),
                started["implementation_lock_sha256"],
                "implementation lock",
            ),
            (
                Path(started["live_preflight_path"]),
                started["live_preflight_sha256"],
                "live preflight",
            ),
            (Path(started["runner_path"]), started["runner_sha256"], "runner"),
        )
        for current_path, expected_sha, label in current_files:
            if not isinstance(expected_sha, str) or sha256_file(current_path) != expected_sha:
                raise RuntimeError(f"{label} changed after STARTED")
        for label, authorization in started["authorization_files"].items():
            if (
                not isinstance(label, str)
                or not label
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                    for character in label
                )
                or not isinstance(authorization, dict)
                or set(authorization) != {"path", "sha256"}
                or not isinstance(authorization["path"], str)
                or not isinstance(authorization["sha256"], str)
                or sha256_file(Path(authorization["path"]))
                != authorization["sha256"]
            ):
                raise RuntimeError(
                    f"authorization file changed after STARTED: {label}"
                )
        prepared_rows = read_jsonl(prepared_path)
        _validate_prepared(prepared_rows, started["expected_rows"])
        _validate_bundle(
            bundle,
            invocation=invocation,
            prepared_rows=prepared_rows,
            runner_sha256=started["runner_sha256"],
            sampling=started["sampling"],
        )
        expected_generated = _generated_receipt_value(
            invocation=invocation,
            started_path=paths["started"],
            bundle_path=paths["bundle"],
            bundle=bundle,
        )
        if not exact_json_equal(generated, expected_generated):
            raise RuntimeError("GENERATED receipt changed during authentication")
        expected_complete = _complete_value(
            invocation=invocation,
            started_path=paths["started"],
            bundle_path=paths["bundle"],
            generated_path=paths["generated"],
            predecessor_complete_sha256=predecessor_sha,
        )
        if not exact_json_equal(complete, expected_complete):
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


def authenticate_registered_complete_prefix(
    *,
    raw_dir: Path,
    invocation_order: Sequence[str],
    registrations: Mapping[str, Mapping[str, Any]],
    through: str,
) -> dict[str, Any]:
    if set(registrations) != set(invocation_order):
        raise ValueError("registered invocation inventory changed")
    if through not in invocation_order:
        raise ValueError("registered prefix endpoint is not in invocation order")
    completed_count = list(invocation_order).index(through) + 1
    receipt = _authenticate_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=invocation_order,
        completed_count=completed_count,
    )
    predecessor_sha: str | None = None
    for invocation in invocation_order[:completed_count]:
        registration = registrations[invocation]
        if set(registration) != {
            "prepared_path",
            "expected_rows",
            "implementation_lock_path",
            "live_preflight_path",
            "runner_path",
            "sampling",
            "authorization_paths",
        }:
            raise ValueError("registered invocation schema changed")
        if (
            not _exact_int(registration["expected_rows"])
            or not isinstance(registration["authorization_paths"], Mapping)
        ):
            raise ValueError("registered invocation values changed")
        expected = _started_value(
            invocation=invocation,
            prepared_path=Path(registration["prepared_path"]),
            expected_rows=registration["expected_rows"],
            implementation_lock_path=Path(
                registration["implementation_lock_path"]
            ),
            live_preflight_path=Path(registration["live_preflight_path"]),
            runner_path=Path(registration["runner_path"]),
            sampling=registration["sampling"],
            authorization_paths={
                label: Path(path)
                for label, path in registration["authorization_paths"].items()
            },
            predecessor_complete_sha256=predecessor_sha,
        )
        paths = artifact_paths(raw_dir, invocation)
        _authenticate_started(paths["started"], expected)
        predecessor_sha = sha256_file(paths["complete"])
    return {
        **receipt,
        "decision": "REGISTERED_TRANSACTION_PREFIX_AUTHENTICATED",
        "registered_invocations": list(invocation_order[:completed_count]),
    }


def authenticate_registered_complete_chain(
    *,
    raw_dir: Path,
    invocation_order: Sequence[str],
    registrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = authenticate_registered_complete_prefix(
        raw_dir=raw_dir,
        invocation_order=invocation_order,
        registrations=registrations,
        through=invocation_order[-1],
    )
    return {**receipt, "decision": "REGISTERED_TRANSACTION_CHAIN_AUTHENTICATED"}

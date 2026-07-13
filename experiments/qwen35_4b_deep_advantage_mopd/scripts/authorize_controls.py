#!/usr/bin/env python3
"""Semantically authorize all trained controls before sealed confirmation."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from authorize_benchmark import (  # noqa: E402
    _audit_controls,
    _audit_integration,
    _audit_preregistration,
    _integration_model,
)
from control_code_inventory import control_code_inventory  # noqa: E402
from io_utils import canonical_hash, load_config, sha256_file  # noqa: E402


CONFIG = EXP / "configs" / "default.yaml"
BENCHMARK_AUTHORIZER = EXP / "scripts" / "authorize_benchmark.py"
CONTROL_RECEIPTS = EXP / "src" / "control_receipts.py"


def _build_authorization(config: dict, config_path: Path) -> dict:
    inventory_before = control_code_inventory()
    preregistration = _audit_preregistration()
    integrations = []
    for seed_value in config["seeds"]["integration_training"]:
        seed = int(seed_value)
        integrations.append(
            _audit_integration(config, config_path, seed, _integration_model(config, seed))
        )
    controls, models = _audit_controls(config, config_path)
    inventory_after = control_code_inventory()
    if inventory_after != inventory_before:
        raise ValueError("control code changed during semantic authorization")
    result = {
        "schema_version": 1,
        "stage": "semantic_controls_confirmation_authorization",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "design_commit": preregistration["design_commit"],
        "integration_receipts": integrations,
        "controls_receipt": controls,
        "control_models": {
            name: {
                "model": str(model),
                "merge_receipt_sha256": sha256_file(model / "merge_receipt.json"),
            }
            for name, model in sorted(models.items())
        },
        "authorizer_sha256": sha256_file(Path(__file__)),
        "benchmark_control_audit_sha256": sha256_file(BENCHMARK_AUTHORIZER),
        "control_receipts_sha256": sha256_file(CONTROL_RECEIPTS),
        "control_code_inventory": inventory_after,
        "control_code_inventory_sha256": canonical_hash(inventory_after),
        "control_code_inventory_before_sha256": inventory_before["sha256"],
        "control_code_inventory_after_sha256": inventory_after["sha256"],
        "gate": {"passed": True},
        "downstream_authorization": "sealed_confirmation_evaluation",
    }
    if control_code_inventory() != inventory_after:
        raise ValueError("control code changed after semantic authorization")
    return result


def _validated_publication_path(path: Path) -> Path:
    """Return a lexical absolute output confined to the real analysis root."""

    output = Path(os.path.abspath(os.fspath(path)))
    analysis_root = Path(os.path.abspath(os.fspath(EXP / "analysis")))
    for label, candidate_path in (
        ("canonical analysis root", analysis_root),
        ("controls authorization output", output),
    ):
        for candidate in reversed((candidate_path, *candidate_path.parents)):
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                break
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"unsafe {label}: symlinked existing ancestor or output: "
                    f"{candidate}"
                )

    resolved_root = analysis_root.resolve(strict=False)
    resolved_parent = output.parent.resolve(strict=False)
    if not resolved_parent.is_relative_to(resolved_root):
        raise ValueError(
            "controls authorization output parent is outside the experiment "
            f"analysis root: {resolved_parent}"
        )
    return output


def _publish_no_clobber(path: Path, result: dict) -> None:
    """Publish once atomically; exact reruns validate without touching bytes."""

    try:
        encoded = json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("controls authorization is not canonical JSON") from exc
    path = _validated_publication_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _validated_publication_path(path)
    if path.is_file() and not path.is_symlink():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing controls authorization is unreadable: {path}") from exc
        try:
            existing_encoded = json.dumps(
                existing,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"existing controls authorization is not canonical JSON: {path}"
            ) from exc
        if existing_encoded != encoded:
            raise ValueError(f"refusing to overwrite stale controls authorization: {path}")
        return
    if path.exists() or path.is_symlink():
        raise ValueError(f"unsafe controls authorization output: {path}")

    temporary_name = None
    try:
        path = _validated_publication_path(path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            path = _validated_publication_path(path)
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise ValueError(
                f"controls authorization publication lost a race: {path}"
            ) from exc
        path = _validated_publication_path(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(CONFIG)
    try:
        result = _build_authorization(config, config_path)
    except (OSError, TypeError, ValueError) as error:
        result = {
            "schema_version": 1,
            "stage": "semantic_controls_confirmation_authorization",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "gate": {"passed": False},
            "error": str(error),
            "downstream_authorization": "stop_before_sealed_confirmation",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 4
    try:
        if control_code_inventory() != result["control_code_inventory"]:
            raise ValueError("control code changed before authorization publication")
        _publish_no_clobber(args.out, result)
    except (OSError, TypeError, ValueError) as error:
        failure = {
            "schema_version": 1,
            "stage": "semantic_controls_confirmation_authorization",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "gate": {"passed": False},
            "error": str(error),
            "downstream_authorization": "stop_before_sealed_confirmation",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 4
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Seal and summarize one pinned runtime without tokenizer, model, or GPU work."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


if sys.flags.no_site != 1:
    raise SystemExit("runtime audit requires static-launcher -I -B -S entry")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from runtime_contract import (  # noqa: E402
    _runtime_pin,
    bootstrap_runtime_environment,
    seal_runtime_environment,
)


def main() -> int:
    backend = os.environ.get("SME_RUNTIME_BACKEND")
    if backend not in {"training", "vllm"}:
        raise RuntimeError("runtime audit cannot identify one pinned launcher backend")
    _runtime_pin(backend)
    bootstrap_runtime_environment(ROOT, backend)
    cutlass_discoverable = None
    if backend == "vllm":
        cutlass_discoverable = importlib.util.find_spec("cutlass") is not None
        if not cutlass_discoverable:
            raise RuntimeError("authenticated vLLM CUTLASS path is not discoverable")
    receipt = seal_runtime_environment(ROOT, backend)
    guard = receipt["import_window_guard"]
    summary = {
        "backend": backend,
        "bootstrap_schema_version": receipt["schema_version"],
        "launcher_sha256": receipt["launcher_authentication"]["sha256"],
        "guard_schema_version": guard["schema_version"],
        "guard_decision": guard["decision"],
        "protected_files": guard["protected_files"],
        "unleased_files": 0,
        "preflight_protected_files": receipt["preflight_window_guard"][
            "protected_files"
        ],
        "loaded_native_mappings": len(
            receipt["post_import_loaded_native_closure"]["mappings"]
        ),
        "cutlass_discoverable": cutlass_discoverable,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

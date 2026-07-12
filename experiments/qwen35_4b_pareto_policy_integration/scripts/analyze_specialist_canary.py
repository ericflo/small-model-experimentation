#!/usr/bin/env python3
"""Gate full specialist merges on same-prefix, same-backend behavior changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, read_jsonl, sha256_file, write_json  # noqa: E402


def _rows(path: Path) -> dict[str, dict]:
    values = {str(row["id"]): row for row in read_jsonl(path)}
    if len(values) != 8:
        raise ValueError(f"expected eight canary rows in {path}, got {len(values)}")
    return values


def _tokens(row: dict) -> list[int]:
    outputs = row.get("outputs") or []
    if len(outputs) != 1:
        raise ValueError("canary requires one greedy output per prompt")
    return [int(value) for value in outputs[0]["token_ids"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--quick-model", type=Path, required=True)
    parser.add_argument("--deep-model", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "specialist_canary.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    rows = {
        "base": _rows(args.base), "quick": _rows(args.quick), "deep": _rows(args.deep),
    }
    keys = {tuple(sorted(values)) for values in rows.values()}
    input_hash = sha256_file(args.input)
    metadata = {
        name: json.loads(path.with_name(path.name + ".meta.json").read_text())
        for name, path in (("base", args.base), ("quick", args.quick), ("deep", args.deep))
    }
    quick_merge = json.loads((args.quick_model / "merge_receipt.json").read_text())
    deep_merge = json.loads((args.deep_model / "merge_receipt.json").read_text())
    changes = []
    for key in sorted(rows["base"]):
        base = _tokens(rows["base"][key])
        quick = _tokens(rows["quick"][key])
        deep = _tokens(rows["deep"][key])
        changes.append({
            "id": key,
            "quick_changed_from_base": quick != base,
            "deep_changed_from_base": deep != base,
            "quick_differs_from_deep": quick != deep,
        })
    checks = {
        "same_prompt_ids": len(keys) == 1,
        "same_input_hash": all(
            value.get("input", {}).get("sha256") == input_hash
            for value in metadata.values()
        ),
        "same_runner": len({value.get("runner_sha256") for value in metadata.values()}) == 1,
        "greedy": all(
            value.get("sampling", {}).get("greedy") is True
            and int(value.get("sampling", {}).get("n", -1)) == 1
            for value in metadata.values()
        ),
        "base_revision_pinned": metadata["base"].get("model_revision")
        == config["model"]["revision"],
        "quick_exact_merge": metadata["quick"].get(
            "local_model_provenance", {}
        ).get("merge_receipt_sha256")
        == sha256_file(args.quick_model / "merge_receipt.json"),
        "deep_exact_merge": metadata["deep"].get(
            "local_model_provenance", {}
        ).get("merge_receipt_sha256")
        == sha256_file(args.deep_model / "merge_receipt.json"),
        "quick_all_deltas_nonzero": int(quick_merge["nonzero_lora_modules"])
        == int(quick_merge["applied_lora_modules"]),
        "deep_all_deltas_nonzero": int(deep_merge["nonzero_lora_modules"])
        == int(deep_merge["applied_lora_modules"]),
        "quick_changes_behavior": any(row["quick_changed_from_base"] for row in changes),
        "deep_changes_behavior": any(row["deep_changed_from_base"] for row in changes),
        "specialists_behaviorally_distinct": any(
            row["quick_differs_from_deep"] for row in changes
        ),
    }
    result = {
        "stage": "full_specialist_installation_canary", "config": str(config_path),
        "input": str(args.input.resolve()), "input_sha256": input_hash,
        "quick_model": str(args.quick_model.resolve()),
        "deep_model": str(args.deep_model.resolve()),
        "changed_prompt_counts": {
            field: sum(bool(row[field]) for row in changes)
            for field in (
                "quick_changed_from_base", "deep_changed_from_base",
                "quick_differs_from_deep",
            )
        },
        "checks": checks, "changes": changes,
        "gate": {"passed": all(checks.values())},
        "downstream_authorization": (
            "calibration" if all(checks.values()) else "stop_before_calibration"
        ),
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

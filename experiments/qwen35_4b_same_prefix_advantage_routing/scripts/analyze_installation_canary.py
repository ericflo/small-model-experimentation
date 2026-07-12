#!/usr/bin/env python3
"""Require base/teachers/soup to be installed and behaviorally distinct."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, read_jsonl, sha256_file, write_json  # noqa: E402


def _rows(path: Path) -> dict[str, dict]:
    rows = {str(row["id"]): row for row in read_jsonl(path)}
    if len(rows) != 8:
        raise ValueError(f"expected eight canary rows in {path}")
    return rows


def _tokens(row: dict) -> list[int]:
    outputs = row.get("outputs") or []
    if len(outputs) != 1:
        raise ValueError("canary requires one greedy output")
    return [int(value) for value in outputs[0]["token_ids"]]


def _merge_nonzero(path: Path) -> bool:
    payload = json.loads((path / "merge_receipt.json").read_text(encoding="utf-8"))
    applied = int(payload.get("applied_lora_modules", 0))
    nonzero = int(payload.get("nonzero_lora_modules", 0))
    return applied > 0 and applied == nonzero


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--soup", type=Path, required=True)
    parser.add_argument("--quick-model", type=Path, required=True)
    parser.add_argument("--deep-model", type=Path, required=True)
    parser.add_argument("--soup-model", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "installation_canary.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    paths = {"base": args.base, "quick": args.quick, "deep": args.deep, "soup": args.soup}
    rows = {name: _rows(path) for name, path in paths.items()}
    ids = set(rows["base"])
    same_ids = all(set(value) == ids for value in rows.values())
    metadata = {
        name: json.loads(path.with_name(path.name + ".meta.json").read_text())
        for name, path in paths.items()
    }
    input_hash = sha256_file(args.input)
    local_models = {"quick": args.quick_model, "deep": args.deep_model, "soup": args.soup_model}
    changes = []
    for state_id in sorted(ids):
        values = {name: _tokens(policy_rows[state_id]) for name, policy_rows in rows.items()}
        changes.append(
            {
                "id": state_id,
                "quick_changed_from_base": values["quick"] != values["base"],
                "deep_changed_from_base": values["deep"] != values["base"],
                "soup_changed_from_base": values["soup"] != values["base"],
                "quick_differs_from_deep": values["quick"] != values["deep"],
                "soup_differs_from_quick": values["soup"] != values["quick"],
                "soup_differs_from_deep": values["soup"] != values["deep"],
            }
        )
    checks = {
        "same_prompt_ids": same_ids,
        "same_input_hash": all(
            payload.get("input", {}).get("sha256") == input_hash
            for payload in metadata.values()
        ),
        "same_runner": len({payload.get("runner_sha256") for payload in metadata.values()}) == 1,
        "greedy": all(
            payload.get("sampling", {}).get("greedy") is True
            and int(payload.get("sampling", {}).get("n", -1)) == 1
            for payload in metadata.values()
        ),
        "base_revision_pinned": metadata["base"].get("model_revision")
        == config["model"]["revision"],
        "all_local_models_exact": all(
            metadata[name].get("model") == str(model.resolve())
            and metadata[name].get("local_model_provenance", {}).get(
                "merge_receipt_sha256"
            )
            == sha256_file(model / "merge_receipt.json")
            for name, model in local_models.items()
        ),
        "all_merges_nonzero": all(_merge_nonzero(model) for model in local_models.values()),
        "quick_changes_behavior": any(row["quick_changed_from_base"] for row in changes),
        "deep_changes_behavior": any(row["deep_changed_from_base"] for row in changes),
        "soup_changes_behavior": any(row["soup_changed_from_base"] for row in changes),
        "teachers_distinct": any(row["quick_differs_from_deep"] for row in changes),
        "soup_distinct_from_quick": any(row["soup_differs_from_quick"] for row in changes),
        "soup_distinct_from_deep": any(row["soup_differs_from_deep"] for row in changes),
    }
    passed = all(checks.values())
    result = {
        "stage": "source_and_soup_installation_canary",
        "config": str(config_path),
        "input": str(args.input.resolve()),
        "input_sha256": input_hash,
        "models": {name: str(model.resolve()) for name, model in local_models.items()},
        "checks": checks,
        "changed_prompt_counts": {
            key: sum(bool(row[key]) for row in changes)
            for key in changes[0]
            if key != "id"
        },
        "changes": changes,
        "gate": {"passed": passed},
        "downstream_authorization": "route_qualification" if passed else "stop_before_routing",
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())


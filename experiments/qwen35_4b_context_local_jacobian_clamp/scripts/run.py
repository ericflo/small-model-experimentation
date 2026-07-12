#!/usr/bin/env python3
"""Stage-gated harness for the context-local Jacobian clamp experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io_utils import sha256_file, write_json, write_jsonl  # noqa: E402
from task_data import (  # noqa: E402
    CONCEPTS,
    consequence_prompt,
    direct_prompt,
    fingerprint,
    generate_splits,
    shared_prefix,
)


CONFIG_PATH = ROOT / "configs" / "default.yaml"
DATA_DIR = ROOT / "data" / "procedural"
RUNS_DIR = ROOT / "runs"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def run_smoke(config: dict) -> dict:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("the repository model boundary permits only Qwen/Qwen3.5-4B")
    splits = generate_splits(config)
    paths = {}
    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path

    prompt_receipts = {}
    for name, rows in splits.items():
        prompt_receipts[name] = {
            "items": len(rows),
            "unique_fingerprints": len({fingerprint(row) for row in rows}),
            "source_counts": {
                concept: sum(row["source"] == concept for row in rows)
                for concept in CONCEPTS
            },
            "example_character_lengths": {
                "shared_prefix": len(shared_prefix(rows[0])),
                "direct": len(direct_prompt(rows[0])),
                "consequence": len(consequence_prompt(rows[0])),
            },
        }
    manifest = {
        "schema_version": 1,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seeds": config["seeds"],
        "splits": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                **prompt_receipts[name],
            }
            for name, path in paths.items()
        },
        "dictionary": {"concepts": list(CONCEPTS), "count": len(CONCEPTS)},
        "scientific_result": False,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "scientific_result": False,
        "data_manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "candidate_bands": config["intervention"]["candidate_bands"],
        "alpha": config["intervention"]["alpha"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def unavailable(stage: str) -> None:
    raise RuntimeError(f"stage {stage!r} is not implemented yet; refusing a placeholder result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "fit-lens", "donor-gate", "confirmation", "full"),
        default="smoke",
    )
    args = parser.parse_args()
    config = load_config()
    if args.stage == "smoke":
        run_smoke(config)
        return 0
    unavailable(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

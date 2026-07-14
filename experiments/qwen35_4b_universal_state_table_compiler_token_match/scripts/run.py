#!/usr/bin/env python3
"""Intake harness for the natural-language state-table curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
EXPECTED = {
    "experiment_id": "qwen35_4b_universal_state_table_compiler_token_match",
    "model_id": "Qwen/Qwen3.5-4B",
    "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "parent_experiment": "qwen35_4b_universal_close_weight_token_match",
    "parent_arm": "close_xi",
    "parent_weights_sha256": "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179",
    "parent_config_sha256": "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff",
    "construction_seed": 77112,
    "training_seed": 46,
    "local_seed": 88008,
    "conditional_aggregate_seed": 78138,
    "status": "intake_cpu_only",
}


def read_flat_yaml(path: Path) -> dict[str, object]:
    """Read this experiment's deliberately flat scalar-only intake config."""
    parsed: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"non-scalar config line: {raw!r}")
        clean = value.strip().strip('"').strip("'")
        parsed[key.strip()] = int(clean) if clean.isdigit() else clean
    return parsed


def smoke() -> int:
    config = read_flat_yaml(CONFIG)
    if config != EXPECTED:
        missing = {key: value for key, value in EXPECTED.items() if config.get(key) != value}
        raise SystemExit(f"intake identity mismatch: {missing}")
    required = [
        ROOT / "idea_intake.md",
        ROOT / "README.md",
        ROOT / "reports" / "artifact_manifest.yaml",
        ROOT / "reports" / "report.md",
    ]
    absent = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if absent:
        raise SystemExit(f"missing intake artifacts: {absent}")
    print(
        json.dumps(
            {
                "authorized_stage": "cpu_design_only",
                "experiment_id": config["experiment_id"],
                "model_id": config["model_id"],
                "parent_arm": config["parent_arm"],
                "reserved_seeds": [
                    config["construction_seed"],
                    config["training_seed"],
                    config["local_seed"],
                    config["conditional_aggregate_seed"],
                ],
                "required_artifacts": len(required),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--smoke", action="store_true", help="validate intake identities without a model call"
    )
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    parser.error("scientific stages are unavailable until the design is frozen")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

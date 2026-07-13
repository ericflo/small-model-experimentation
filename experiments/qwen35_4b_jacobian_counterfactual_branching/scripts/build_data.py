#!/usr/bin/env python3
"""Generate fresh procedural splits and prove direct-ancestor disjointness."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from task_data import build_splits, task_fingerprint  # noqa: E402


ANCESTORS = (
    "qwen35_4b_native_thought_jacobian_value_transport",
    "qwen35_4b_native_thought_seam_budget_ladder",
    "qwen35_4b_forced_commit_jacobian_value_transport",
    "qwen35_4b_commit_slot_jacobian_value_transport",
    "qwen35_4b_commit_slot_semantic_power_replication",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    ancestor_fingerprints: set[str] = set()
    ancestor_files = []
    for experiment_id in ANCESTORS:
        directory = REPO / "experiments" / experiment_id / "data" / "procedural"
        for path in sorted(directory.glob("*.jsonl")):
            rows = load_jsonl(path)
            fingerprints = {task_fingerprint(row) for row in rows}
            ancestor_fingerprints.update(fingerprints)
            ancestor_files.append(
                {
                    "path": path.relative_to(REPO).as_posix(),
                    "sha256": sha256(path),
                    "rows": len(rows),
                    "unique_fingerprints": len(fingerprints),
                }
            )
    splits = build_splits(config)
    output_dir = EXP / "data" / "procedural"
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    split_receipts = {}
    for split, rows in splits.items():
        fingerprints = [task_fingerprint(row) for row in rows]
        if len(set(fingerprints)) != len(rows):
            raise RuntimeError(f"duplicate task fingerprint inside {split}")
        collision = set(fingerprints) & (ancestor_fingerprints | seen)
        if collision:
            raise RuntimeError(f"task fingerprint collision in {split}: {len(collision)}")
        seen.update(fingerprints)
        path = output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        split_receipts[split] = {
            "path": path.relative_to(REPO).as_posix(),
            "sha256": sha256(path),
            "rows": len(rows),
            "unique_fingerprints": len(set(fingerprints)),
            "first_operation_counts": dict(sorted(Counter(row["first_op"] for row in rows).items())),
            "ancestor_collisions": 0,
            "cross_split_collisions": 0,
        }
        if split == "mechanics":
            public_path = output_dir / "mechanics_public.jsonl"
            public_rows = [
                {"task_id": row["task_id"], "visible": row["visible"]}
                for row in rows
            ]
            public_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in public_rows)
            )
            split_receipts[split]["public_path"] = public_path.relative_to(REPO).as_posix()
            split_receipts[split]["public_sha256"] = sha256(public_path)
            split_receipts[split]["public_fields"] = ["task_id", "visible"]
    manifest = {
        "schema_version": 1,
        "generator": "src/task_data.py",
        "split_seed": int(config["seeds"]["split"]),
        "ancestor_files": ancestor_files,
        "ancestor_unique_fingerprints": len(ancestor_fingerprints),
        "splits": split_receipts,
        "total_new_unique_fingerprints": len(seen),
        "all_disjoint": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

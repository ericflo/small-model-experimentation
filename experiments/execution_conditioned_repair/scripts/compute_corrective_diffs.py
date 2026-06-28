#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.patching import apply_patch_to_files, unified_diff_for_files
from repair_experiment.runner import run_pytest


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    checked: list[dict] = []
    failures: list[dict] = []
    for row in rows:
        recomputed = unified_diff_for_files(row["current_files"], row["clean_files"])
        row["target_next_diff"] = recomputed
        applied, patched, apply_output = apply_patch_to_files(row["current_files"], recomputed)
        hidden = run_pytest(
            patched,
            row["visible_tests"],
            row["hidden_tests"],
            which="hidden",
        ) if applied else {"passed": False, "output": apply_output}
        row.setdefault("validation", {})
        row["validation"].update(
            {
                "target_diff_recomputed": True,
                "target_diff_applies": applied,
                "target_diff_hidden_passes": hidden["passed"],
                "target_diff_apply_output": apply_output,
                "target_diff_hidden_output": hidden["output"][-4000:],
            }
        )
        if not applied or not hidden["passed"]:
            failures.append(
                {
                    "episode_id": row["episode_id"],
                    "applied": applied,
                    "hidden_passed": hidden["passed"],
                    "output": (apply_output + "\n" + hidden["output"])[-2000:],
                }
            )
        checked.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in checked:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "records": len(rows),
        "target_diff_applies": sum(row["validation"]["target_diff_applies"] for row in checked),
        "target_diff_hidden_passes": sum(row["validation"]["target_diff_hidden_passes"] for row in checked),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

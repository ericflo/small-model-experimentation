#!/usr/bin/env python3
"""Strict full-heldout direct-Qwen diagnostic for the active crystallizer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

ROOT = Path("/workspace/experiments/qwen_active_crystallizer_public_gate")
ANALYSIS = ROOT / "analysis"
RUNS = ROOT / "runs"

sys.path.insert(0, str(ROOT / "src"))
from qwen_active_crystallizer_public_gate import (  # noqa: E402
    MODEL_NAME,
    load_qwen,
    load_tasks,
    qwen_answer,
    split_examples,
)


def run(args: argparse.Namespace) -> None:
    started = datetime.now(timezone.utc)
    qwen_details = pd.read_csv(ANALYSIS / "qwen_probe_details.csv")
    selected_ids = list(qwen_details["task_id"].head(args.tasks))
    task_map = {t.task_id: t for t in load_tasks(min_examples=5)}
    tok, model = load_qwen()
    rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    total = 0
    exact_total = 0
    for ti, task_id in enumerate(selected_ids, start=1):
        task = task_map[task_id]
        train, test = split_examples(task, args.train_n, args.heldout_cap)
        train_pairs = [(e.inputs, e.output) for e in train]
        ok = 0
        for ri, ex in enumerate(test, start=1):
            pred = qwen_answer(tok, model, train_pairs, ex.inputs, args.max_new_tokens)
            exact = pred == ex.output
            rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "features": ",".join(task.features),
                    "row_index": ri,
                    "target": ex.output,
                    "prediction": pred,
                    "exact": exact,
                }
            )
            ok += int(exact)
            exact_total += int(exact)
            total += 1
        task_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "heldout_rows": len(test),
                "row_exact": ok / len(test) if test else 0.0,
                "full_task_exact": ok == len(test) and bool(test),
            }
        )
        if ti == 1 or ti % 5 == 0 or ti == len(selected_ids):
            print(f"task {ti}/{len(selected_ids)} rows={total} row_exact={exact_total}/{total} full_tasks={sum(r['full_task_exact'] for r in task_rows)}/{len(task_rows)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame(rows)
    task_df = pd.DataFrame(task_rows)
    summary = pd.DataFrame(
        [
            {"metric": "row_exact", "tasks": len(task_df), "rows": len(row_df), "score": float(row_df["exact"].mean()) if len(row_df) else 0.0},
            {"metric": "full_task_exact", "tasks": len(task_df), "rows": len(row_df), "score": float(task_df["full_task_exact"].mean()) if len(task_df) else 0.0},
        ]
    )
    row_df.to_csv(ANALYSIS / "qwen_direct_full_rows.csv", index=False)
    task_df.to_csv(ANALYSIS / "qwen_direct_full_tasks.csv", index=False)
    summary.to_csv(ANALYSIS / "qwen_direct_full_summary.csv", index=False)
    meta = {
        "model": MODEL_NAME,
        "tasks": args.tasks,
        "actual_tasks": len(task_df),
        "rows": len(row_df),
        "train_n": args.train_n,
        "heldout_cap": args.heldout_cap,
        "max_new_tokens": args.max_new_tokens,
        "started_utc": started.isoformat(),
        "elapsed_sec": round(time.time() - started.timestamp(), 2),
    }
    (RUNS / "qwen_direct_full_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write("\n## Strict Direct-Qwen Full-Heldout Diagnostic\n\n")
        f.write(f"- Model: `{MODEL_NAME}`\n")
        f.write(f"- Tasks: `{len(task_df)}`; held-out rows: `{len(row_df)}`\n")
        f.write(f"- Row exact: {100 * float(row_df['exact'].mean()):.1f}% ({int(row_df['exact'].sum())}/{len(row_df)}).\n")
        f.write(f"- Full-task exact: {100 * float(task_df['full_task_exact'].mean()):.1f}% ({int(task_df['full_task_exact'].sum())}/{len(task_df)}).\n")
        f.write("- Full-task exact requires every held-out row for the task to be answered exactly.\n")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=40)
    p.add_argument("--train_n", type=int, default=4)
    p.add_argument("--heldout_cap", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

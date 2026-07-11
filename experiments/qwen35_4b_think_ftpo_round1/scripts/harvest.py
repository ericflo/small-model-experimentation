#!/usr/bin/env python3
"""Harvest P: n=8 verifier-scored think trajectories on learnable-band tasks.

Adaptive slices of `slice_prompts` prompts run until the projected mined-row
pool reaches `target_pool_rows` or the GPU-hour cap is hit. Every output keeps
exact stage-1 token IDs (the reproducibility anchor and the mining substrate).

../../.venv-vllm/bin/python scripts/harvest.py [--smoke] [--max-slices N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
import pivotmine  # noqa: E402
import tasks  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402

THINK_CLOSE = 248069
CAL_SEED_FLOOR = 72901  # calibration block; slice seeds must stay below


def think_sequence(stage1_token_ids: list[int]) -> list[int]:
    if THINK_CLOSE in stage1_token_ids:
        return stage1_token_ids[: stage1_token_ids.index(THINK_CLOSE)]
    return list(stage1_token_ids)


def load_band_cells(cfg: dict) -> list[dict]:
    table = json.loads((EXP / "runs" / "band_calibration.json").read_text())
    cells = [c for c in table["cells"] if c["in_band"]]
    if not cells:
        raise SystemExit("no in-band cells; band calibration must run first")
    return cells


def slice_items(cfg: dict, cells: list[dict], slice_index: int, n_prompts: int) -> list[tasks.TaskItem]:
    hp = cfg["harvest_pivot"]
    gym_cells = [c for c in cells if c["source"] == "gym"]
    code_cells = [c for c in cells if c["source"] == "code"]
    n_gym = int(round(n_prompts * float(hp["gym_fraction"]))) if gym_cells else 0
    n_code = n_prompts - n_gym if code_cells else 0
    if not code_cells:
        n_gym = n_prompts
    items: list[tasks.TaskItem] = []
    if gym_cells:
        # ceil-divide then trim so the slice reaches its full size (amendment 1.5)
        per_cell = -(-n_gym // len(gym_cells))
        for c_idx, cell in enumerate(gym_cells):
            seed = int(hp["gym_seed_range"][0]) + slice_index * 40 + c_idx
            if seed >= CAL_SEED_FLOOR:
                raise SystemExit("gym slice seed collided with calibration block")
            items.extend(tasks.make_gym_items(cell["family"], cell["level"], seed, per_cell))
        items = items[:n_gym]
    if code_cells:
        per_cell = -(-n_code // len(code_cells))
        code_items: list[tasks.TaskItem] = []
        for c_idx, cell in enumerate(code_cells):
            seed = int(hp["code_seed_range"][0]) + slice_index * 40 + c_idx
            if seed >= 73901:
                raise SystemExit("code slice seed collided with calibration block")
            code_items.extend(tasks.make_code_items(cell["level"], seed, per_cell))
        items.extend(code_items[:n_code])
    return items[:n_prompts]


def mine_group(cfg: dict, group: dict) -> list:
    m = cfg["mining"]
    sequences = [think_sequence(o["stage1_token_ids"]) for o in group["outputs"]]
    successes = [o["score"] >= 1.0 for o in group["outputs"]]
    return pivotmine.mine_pivots(
        sequences, successes,
        min_depth=int(m["pivot_min_depth"]),
        min_branch_rollouts=int(m["pivot_min_branch_rollouts"]),
        min_gap=float(m["pivot_min_gap"]),
        max_nodes=int(m["pivot_max_nodes_per_prompt"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="one 80-prompt slice")
    parser.add_argument("--max-slices", type=int, default=8)
    parser.add_argument("--config", default=None, help="alternate config (smoke probes only)")
    parser.add_argument("--start-slice", type=int, default=0,
                        help="first slice index (extension passes use fresh indices)")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else EXP / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    hp = cfg["harvest_pivot"]
    cells = load_band_cells(cfg)
    out_dir = EXP / "runs" / (("harvest_probe" if args.config else "harvest_smoke") if args.smoke else "harvest")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = harness.make_runner(cfg.get("engine_harvest", cfg["engine"]))
    sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(hp["think_budget"]),
        n=int(hp["n"]),
        answer_max_tokens=int(hp["answer_max_tokens"]),
        greedy=False,
        temperature=float(hp["temperature"]),
        top_p=float(hp["top_p"]),
        top_k=int(hp["top_k"]),
        run_seed=int(hp["run_seed"]),
    )

    started = time.time()
    cap_s = 0.25 * 3600 if args.smoke else float(hp["max_harvest_hours"]) * 3600
    slice_n = 80 if args.smoke else int(hp["slice_prompts"])
    target_rows = 0 if args.smoke else int(hp["target_pool_rows"])

    total_prompts = 0
    total_groups_with_node = 0
    total_mixed = 0
    total_rows = 0
    chosen_counts: list[int] = []
    slice_index = args.start_slice
    while slice_index < args.start_slice + args.max_slices:
        items = slice_items(cfg, cells, slice_index, slice_n)
        records = [{"id": it.item_id, "messages": [{"role": "user", "content": it.prompt}]}
                   for it in items]
        prepared = runner.prepare(records, "budget")
        rows, summary = runner.generate(records, sampling)

        slice_rows = []
        for it, prep, row in zip(items, prepared, rows):
            outputs = []
            for output in row["outputs"]:
                outputs.append({
                    "sample_index": output["sample_index"],
                    "stage1_token_ids": output["stage1_token_ids"],
                    "text": output["text"],
                    "score": tasks.score_item(it, output["text"]),
                    "forced_close": output["forced_close"],
                    "thinking_closed": output["thinking_closed"],
                    "n_thinking_tokens": output["n_thinking_tokens"],
                    "n_answer_tokens": output["n_answer_tokens"],
                    "n_sampled_tokens": output["n_sampled_tokens"],
                    "finish_reason": output["finish_reason"],
                })
            group = {
                "item_id": it.item_id, "source": it.source, "family": it.family,
                "level": it.level, "prompt": it.prompt,
                "prompt_token_ids": prep.prompt_token_ids,
                "outputs": outputs,
            }
            pivots = mine_group(cfg, group)
            group["n_pivots"] = len(pivots)
            wins = [o["score"] >= 1.0 for o in outputs]
            if any(wins) and not all(wins):
                total_mixed += 1
            if pivots:
                total_groups_with_node += 1
                total_rows += len(pivots)
                chosen_counts.extend(len(p.chosen_ids) for p in pivots)
            slice_rows.append(group)

        shard = out_dir / f"groups_slice{slice_index}.jsonl.gz"
        with gzip.open(shard, "wt", encoding="utf-8") as fh:
            for group in slice_rows:
                fh.write(json.dumps(group, ensure_ascii=False) + "\n")

        total_prompts += len(items)
        elapsed = time.time() - started
        census_rate = total_groups_with_node / max(total_prompts, 1)
        print(f"[slice {slice_index}] prompts={total_prompts} "
              f"census={census_rate:.3f} mined_rows={total_rows} "
              f"elapsed={elapsed/60:.1f}m", flush=True)

        slice_index += 1
        if args.smoke:
            break
        if total_rows >= target_rows:
            print("[harvest] pool target reached", flush=True)
            break
        if elapsed > cap_s:
            print("[harvest] GPU-hour cap reached", flush=True)
            break

    census_rate = total_groups_with_node / max(total_prompts, 1)
    mixed_rate = total_mixed / max(total_prompts, 1)
    p0_pass = (census_rate >= float(cfg["mining"]["census_gate_group_rate"])
               and mixed_rate >= float(cfg["mining"]["census_gate_mixed_rate"]))
    summary = {
        "prompts": total_prompts,
        "slices": slice_index,
        "groups_with_node": total_groups_with_node,
        "census_group_rate": census_rate,
        "mixed_outcome_rate": mixed_rate,
        "mined_rows_estimate": total_rows,
        "chosen_per_row_hist": {str(k): chosen_counts.count(k)
                                for k in sorted(set(chosen_counts))},
        "p0_gate": [float(cfg["mining"]["census_gate_group_rate"]),
                    float(cfg["mining"]["census_gate_mixed_rate"])],
        "p0_pass": bool(p0_pass),
        "wall_s": time.time() - started,
        "sampling": {k: hp[k] for k in
                     ("temperature", "top_p", "top_k", "n", "think_budget")},
        "band_cells_used": cells,
    }
    (out_dir / "harvest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("prompts", "census_group_rate", "mixed_outcome_rate",
                       "mined_rows_estimate", "p0_pass", "wall_s")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

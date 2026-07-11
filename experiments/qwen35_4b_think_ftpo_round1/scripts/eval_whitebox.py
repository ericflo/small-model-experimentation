#!/usr/bin/env python3
"""Whitebox evaluations. One engine (= one arm) per invocation.

Stages (each writes runs/whitebox_<arm>_<stage>.json):
  main      greedy grid over think@{1024,2048}: success (P1 substrate = in-band
            cells, fresh seeds), termination triple, natural-close, answer-parse
  formats   the P1 substrate re-rendered under 2 alternate scaffolds (@1024)
  coverage  base-only: n=8 sampling at harvest settings (NON-DEPLOYABLE oracle)
  collapse  code substrate greedy + pass@8 (C29 guard)
  nothink   no-think forced-answer accuracy on gym L1 atoms

  ../../.venv-vllm/bin/python scripts/eval_whitebox.py --arm base --stage main
  ../../.venv-vllm/bin/python scripts/eval_whitebox.py --arm pivot \
      --model <merged_dir> --stage main
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
import loopdetect  # noqa: E402
import tasks  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402

THINK_CLOSE = 248069


def band_cells() -> list[dict]:
    table = json.loads((EXP / "runs" / "band_calibration.json").read_text())
    return [c for c in table["cells"] if c["in_band"]]


def eval_items(cfg: dict, n_prompts: int, seed_offset: int = 0) -> list[tasks.TaskItem]:
    cells = band_cells()
    base_seed = int(cfg["eval"]["whitebox_seed_range"][0]) + seed_offset
    per_cell = max(1, n_prompts // len(cells))
    items: list[tasks.TaskItem] = []
    for idx, cell in enumerate(cells):
        seed = base_seed + idx
        if cell["source"] == "gym":
            items.extend(tasks.make_gym_items(cell["family"], cell["level"], seed, per_cell))
        else:
            items.extend(tasks.make_code_items(cell["level"], seed, per_cell))
    return items[:n_prompts]


def think_text_of(output: dict) -> str:
    text = output["text"]
    return text.split("</think>")[0] if "</think>" in text else text


def termination_triple(cfg: dict, output: dict) -> dict:
    det = cfg["detector"]
    think = think_text_of(output)
    loop = loopdetect.find_inner_repetition(
        think, min_repeats=int(det["min_repeats"]), max_period=int(det["max_period"]),
        min_period=int(det["min_period"]),
        min_total_repeated=int(det["min_total_repeated"]),
        sample_len=int(det["sample_len"]), sample_interval=int(det["sample_interval"]))
    forced = bool(output["forced_close"])
    answer_limited = (output.get("finish_reason") == "length"
                      or output["n_answer_tokens"] >= 512)
    return {
        "loop": loop is not None,
        "unresolved_contact": forced and loop is None,
        "answer_limit": answer_limited,
        "forced_close": forced,
    }


def run_batch(runner, items, sampling, cfg, prompts=None):
    records = [{"id": it.item_id,
                "messages": [{"role": "user",
                              "content": prompts[i] if prompts else it.prompt}]}
               for i, it in enumerate(items)]
    rows, _ = runner.generate(records, sampling)
    results = []
    for it, row in zip(items, rows):
        for output in row["outputs"]:
            triple = termination_triple(cfg, output)
            results.append({
                "item_id": it.item_id, "source": it.source, "family": it.family,
                "level": it.level, "sample_index": output["sample_index"],
                "score": tasks.score_item(it, output["text"]),
                "natural_close": not output["forced_close"],
                "n_thinking_tokens": output["n_thinking_tokens"],
                **triple,
            })
    return results


def aggregate(results: list[dict]) -> dict:
    n = max(len(results), 1)
    return {
        "n": len(results),
        "success": sum(r["score"] >= 1.0 for r in results) / n,
        "mean_score": sum(r["score"] for r in results) / n,
        "loop_rate": sum(r["loop"] for r in results) / n,
        "unresolved_rate": sum(r["unresolved_contact"] for r in results) / n,
        "answer_limit_rate": sum(r["answer_limit"] for r in results) / n,
        "natural_close_rate": sum(r["natural_close"] for r in results) / n,
        "mean_think_tokens": sum(r["n_thinking_tokens"] for r in results) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", default=None, help="merged checkpoint dir for trained arms")
    parser.add_argument("--stage", required=True,
                        choices=["main", "formats", "coverage", "collapse", "nothink"])
    parser.add_argument("--n", type=int, default=None, help="override N (smoke)")
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    ev = cfg["eval"]
    runner = harness.make_runner(cfg["engine"], model_override=args.model)
    started = time.time()
    out: dict = {"arm": args.arm, "stage": args.stage, "model": args.model or "base"}

    if args.stage == "main":
        n_prompts = args.n or int(ev["whitebox_n_prompts"])
        items = eval_items(cfg, n_prompts)
        for budget in ev["whitebox_budgets"]:
            sampling = SamplingConfig(thinking="budget", thinking_budget=int(budget),
                                      n=1, answer_max_tokens=512, greedy=True,
                                      run_seed=7500)
            results = run_batch(runner, items, sampling, cfg)
            out[f"think@{budget}"] = aggregate(results)
            per_source = {}
            for source in ("gym", "code"):
                sub = [r for r in results if r["source"] == source]
                if sub:
                    per_source[source] = aggregate(sub)
            out[f"think@{budget}_per_source"] = per_source

    elif args.stage == "formats":
        n_prompts = args.n or 150
        items = eval_items(cfg, n_prompts, seed_offset=200)
        sampling = SamplingConfig(thinking="budget", thinking_budget=1024, n=1,
                                  answer_max_tokens=512, greedy=True, run_seed=7501)
        for variant in (1, 2):
            prompts = [tasks.render_format_variant(it, variant) for it in items]
            results = run_batch(runner, items, sampling, cfg, prompts=prompts)
            out[f"variant{variant}"] = aggregate(results)

    elif args.stage == "coverage":
        if args.arm != "base":
            raise SystemExit("coverage reference is base-only (NON-DEPLOYABLE oracle)")
        n_prompts = args.n or 200
        items = eval_items(cfg, n_prompts, seed_offset=400)
        hp = cfg["harvest_pivot"]
        sampling = SamplingConfig(thinking="budget",
                                  thinking_budget=int(hp["think_budget"]),
                                  n=int(hp["n"]), answer_max_tokens=512, greedy=False,
                                  temperature=float(hp["temperature"]),
                                  top_p=float(hp["top_p"]), top_k=int(hp["top_k"]),
                                  run_seed=7502)
        results = run_batch(runner, items, sampling, cfg)
        by_item: dict[str, list[float]] = {}
        for r in results:
            by_item.setdefault(r["item_id"], []).append(r["score"])
        out["coverage_at_8"] = sum(1 for v in by_item.values() if max(v) >= 1.0) / len(by_item)
        out["label"] = "NON-DEPLOYABLE oracle ceiling (best-of-8, no selector)"

    elif args.stage == "collapse":
        n_tasks = args.n or int(ev["collapse_guard_n_tasks"])
        items = tasks.make_code_items(3, int(ev["collapse_guard_seed_base"]), n_tasks)
        greedy = SamplingConfig(thinking="budget", thinking_budget=1024, n=1,
                                answer_max_tokens=512, greedy=True, run_seed=7601)
        results = run_batch(runner, items, greedy, cfg)
        out["greedy"] = aggregate(results)
        k = int(ev["collapse_guard_pass_k"])
        hp = cfg["harvest_pivot"]
        sampled = SamplingConfig(thinking="budget", thinking_budget=1024, n=k,
                                 answer_max_tokens=512, greedy=False,
                                 temperature=float(hp["temperature"]),
                                 top_p=float(hp["top_p"]), top_k=int(hp["top_k"]),
                                 run_seed=7602)
        results_k = run_batch(runner, items, sampled, cfg)
        by_item = {}
        for r in results_k:
            by_item.setdefault(r["item_id"], []).append(r["score"])
        out[f"pass@{k}"] = sum(1 for v in by_item.values() if max(v) >= 1.0) / len(by_item)

    elif args.stage == "nothink":
        n_atoms = args.n or int(ev["nothink_guard_n_atoms"])
        per_family = max(1, n_atoms // 10)
        items: list[tasks.TaskItem] = []
        for fam, level in tasks.gym_cells([1]):
            items.extend(tasks.make_gym_items(fam, level, 77001, per_family))
        items = items[:n_atoms]
        sampling = SamplingConfig(thinking="off", n=1, max_tokens=256,
                                  greedy=True, run_seed=7701)
        results = run_batch(runner, items, sampling, cfg)
        out["nothink"] = aggregate(results)

    out["wall_s"] = time.time() - started
    dest = EXP / "runs" / f"whitebox_{args.arm}_{args.stage}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "history"},
                     indent=2, default=str)[:2000])
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

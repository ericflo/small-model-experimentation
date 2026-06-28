#!/usr/bin/env python3
"""Pairwise full-table judging for text transformation tasks.

This standalone experiment tests whether a model can choose the more
task-consistent candidate output table when shown examples, query rows, and two
candidate tables.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_pairwise_table_judge")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_pairwise_table_judge")
SOURCE_BENCH_ROOT = Path("/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks")
BENCH_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = BENCH_ROOT / "Transformation.Text"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CACHE_DIR = Path("/workspace/.cache/huggingface")
MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_CANDIDATE_SOURCE = Path("/workspace/experiments/qwen_full_table_consistency_reranker/runs/main_qwen_table_40")


@dataclass(frozen=True)
class Example:
    inputs: Tuple[str, ...]
    output: str


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    synthetic: bool
    features: Tuple[str, ...]
    examples: Tuple[Example, ...]
    source_path: str


def ensure_dirs() -> None:
    for d in [RUNS, ANALYSIS, FIGURES, REPORTS, LARGE_ROOT]:
        d.mkdir(parents=True, exist_ok=True)


def mirror_benchmark() -> None:
    if BENCH_ROOT.exists():
        return
    if not SOURCE_BENCH_ROOT.exists():
        raise FileNotFoundError(f"Missing benchmark source: {SOURCE_BENCH_ROOT}")
    LARGE_ROOT.mkdir(parents=True, exist_ok=True)
    BENCH_ROOT.symlink_to(SOURCE_BENCH_ROOT, target_is_directory=True)


def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", "" if s is None else str(s)).strip()


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


def norm_eq(a: Any, b: Any) -> bool:
    return clean(a) == clean(b)


def load_tasks(limit: Optional[int] = None, min_examples: int = 5) -> List[Task]:
    tasks: List[Task] = []
    for d in sorted(TRANSFORM_ROOT.iterdir()):
        if not d.is_dir() or not (d / "spec.json").exists():
            continue
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        examples: List[Example] = []
        for ex in spec.get("Examples", []):
            examples.append(Example(tuple(str(x) for x in ex.get("Input", [])), str(ex.get("Output", ""))))
        if len(examples) < min_examples:
            continue
        tasks.append(
            Task(
                task_id=d.name,
                family=d.name.split(".", 1)[0],
                synthetic=bool(meta.get("Synthetic", False)),
                features=tuple(str(x) for x in meta.get("Features", [])),
                examples=tuple(examples),
                source_path=str(d),
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def split_examples(task: Task, train_n: int, heldout_cap: int) -> Tuple[List[Example], List[Example]]:
    n = len(task.examples)
    k = min(train_n, max(2, n // 2))
    train = list(task.examples[:k])
    test = list(task.examples[k:])
    if heldout_cap > 0:
        test = test[:heldout_cap]
    return train, test


def choose_tasks(tasks: List[Task], limit: int, seed: int, train_n: int, heldout_cap: int, min_heldout: int) -> List[Task]:
    eligible: List[Task] = []
    for t in tasks:
        _, test = split_examples(t, train_n, heldout_cap)
        if len(test) >= min_heldout:
            eligible.append(t)
    rng = random.Random(seed)
    if limit and limit < len(eligible):
        chosen = rng.sample(eligible, limit)
    else:
        chosen = eligible
    return sorted(chosen, key=lambda t: t.task_id)


def parse_outputs(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return tuple(str(x) for x in parsed)
    except Exception:
        pass
    return tuple()


def copy_candidate_pool(run_dir: Path, source: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name in ["table_candidates.csv", "oracle_summary.csv", "selected_tables.csv"]:
        src = source / name
        if not src.exists():
            raise FileNotFoundError(f"Missing candidate source file: {src}")
        dst = run_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
    return (
        pd.read_csv(run_dir / "table_candidates.csv"),
        pd.read_csv(run_dir / "oracle_summary.csv"),
        pd.read_csv(run_dir / "selected_tables.csv"),
    )


def heuristic_score(row: pd.Series) -> float:
    return (
        1.5 * float(row.get("avg_vote_share", 0.0))
        + 0.8 * float(row.get("signature_mode_match_rate", 0.0))
        + 0.8 * float(row.get("simple_explainer", 0.0))
        - 0.02 * float(row.get("len_delta_abs", 0.0))
        - 0.4 * float(row.get("empty_rate", 0.0))
        - 0.1 * float(row.get("avg_vote_rank", 99.0))
    )


def row_exact(outputs: Sequence[str], targets: Sequence[str]) -> float:
    return sum(norm_eq(a, b) for a, b in zip(outputs, targets)) / max(1, len(targets))


def full_exact(outputs: Sequence[str], targets: Sequence[str]) -> bool:
    return len(outputs) == len(targets) and all(norm_eq(a, b) for a, b in zip(outputs, targets))


def table_lookup(table_df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {str(r["candidate_id"]): r for _, r in table_df.iterrows()}


def source_candidate(table_df: pd.DataFrame, task_id: str, source: str) -> Optional[pd.Series]:
    sub = table_df[table_df["task_id"].eq(task_id) & table_df["source"].eq(source)]
    if sub.empty:
        return None
    return sub.iloc[0]


def exact_candidate(table_df: pd.DataFrame, task_id: str) -> Optional[pd.Series]:
    sub = table_df[table_df["task_id"].eq(task_id) & table_df["exact"].astype(bool)]
    if sub.empty:
        return None
    return sub.sort_values(["row_exact", "avg_vote_share", "prior"], ascending=[False, False, False]).iloc[0]


def shortlist_candidates(table_df: pd.DataFrame, task_id: str, max_shortlist: int) -> List[pd.Series]:
    sub = table_df[table_df["task_id"].eq(task_id)].copy()
    if sub.empty:
        return []
    sub["local_heuristic"] = sub.apply(heuristic_score, axis=1)
    candidates: List[pd.Series] = []
    seen: set[str] = set()
    for source in ["row_greedy", "row_majority", "batch_plain", "batch_json", "row_consistency", "row_format"]:
        hit = source_candidate(sub, task_id, source)
        if hit is not None and str(hit["candidate_id"]) not in seen:
            candidates.append(hit)
            seen.add(str(hit["candidate_id"]))
    for _, row in sub.sort_values(["local_heuristic", "avg_vote_share", "prior"], ascending=[False, False, False]).iterrows():
        cid = str(row["candidate_id"])
        if cid not in seen:
            candidates.append(row)
            seen.add(cid)
        if len(candidates) >= max_shortlist:
            break
    return candidates[:max_shortlist]


def render_table_prompt(
    train: Sequence[Example],
    test: Sequence[Example],
    cand_a: Sequence[str],
    cand_b: Sequence[str],
    mode: str,
    rng: random.Random,
) -> str:
    if mode == "no_examples":
        examples: List[Example] = []
    elif mode == "shuffled_examples":
        outs = [e.output for e in train]
        rng.shuffle(outs)
        examples = [Example(e.inputs, out) for e, out in zip(train, outs)]
    else:
        examples = list(train)

    b_vals = list(cand_b)
    if mode == "row_shuffled_candidate":
        rng.shuffle(b_vals)

    lines = [
        "A task is defined by input-output examples. Two candidate output tables are proposed for the same query rows.",
        "Choose the candidate table that more consistently follows the same transformation as the examples.",
        "Return only A or B.",
        "",
    ]
    if examples:
        lines.append("Examples:")
        for ex in examples:
            lines.append(f"Input: {render_inputs(ex.inputs)}")
            lines.append(f"Output: {ex.output}")
        lines.append("")
    else:
        lines.append("Examples: [hidden]")
        lines.append("")
    lines.append("Query rows:")
    for i, ex in enumerate(test, start=1):
        lines.append(f"{i}. Input: {render_inputs(ex.inputs)}")
    lines.append("")
    lines.append("Candidate A:")
    for i, out in enumerate(cand_a, start=1):
        lines.append(f"{i}. {out}")
    lines.append("")
    lines.append("Candidate B:")
    for i, out in enumerate(b_vals, start=1):
        lines.append(f"{i}. {out}")
    lines.append("")
    lines.append("Which candidate table is more consistent with the examples? Answer A or B only.")
    return "\n".join(lines)


def parse_choice(raw: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    m = re.search(r"\b([AB])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if text[:1].upper() in {"A", "B"}:
        return text[:1].upper()
    return ""


def load_qwen() -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    return tok, model


def generate_text(tok: Any, model: Any, system: str, user: str, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def read_judgment_cache(path: Path) -> Dict[Tuple[str, str, str, str, str, str], Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str, str, str, str, str], Dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["task_id"], row["mode"], row["pair_kind"], row["cand_a_id"], row["cand_b_id"], row["order_tag"])
            out[key] = row
    return out


def judge_pair(
    task: Task,
    train: Sequence[Example],
    test: Sequence[Example],
    cand_a: pd.Series,
    cand_b: pd.Series,
    mode: str,
    pair_kind: str,
    order_tag: str,
    rng: random.Random,
    cache: Dict[Tuple[str, str, str, str, str, str], Dict[str, str]],
    writer: Optional[csv.DictWriter],
    tok: Any,
    model: Any,
    max_new_tokens: int,
    no_qwen: bool,
) -> Dict[str, Any]:
    key = (task.task_id, mode, pair_kind, str(cand_a["candidate_id"]), str(cand_b["candidate_id"]), order_tag)
    if key in cache:
        row = cache[key]
        choice = row["choice"]
        raw = row["raw"]
    else:
        if no_qwen:
            choice = ""
            raw = ""
        else:
            outs_a = parse_outputs(cand_a["outputs_json"])
            outs_b = parse_outputs(cand_b["outputs_json"])
            prompt = render_table_prompt(train, test, outs_a, outs_b, mode, rng)
            raw = generate_text(
                tok,
                model,
                "You are a careful judge of text transformation examples. Answer with one letter only.",
                prompt,
                max_new_tokens,
            )
            choice = parse_choice(raw)
        row = {
            "task_id": task.task_id,
            "mode": mode,
            "pair_kind": pair_kind,
            "cand_a_id": str(cand_a["candidate_id"]),
            "cand_b_id": str(cand_b["candidate_id"]),
            "order_tag": order_tag,
            "choice": choice,
            "raw": raw,
        }
        if writer is not None:
            writer.writerow(row)
    chosen = cand_a if choice == "A" else cand_b if choice == "B" else cand_a
    return {
        "task_id": task.task_id,
        "mode": mode,
        "pair_kind": pair_kind,
        "cand_a_id": str(cand_a["candidate_id"]),
        "cand_b_id": str(cand_b["candidate_id"]),
        "order_tag": order_tag,
        "choice": choice,
        "chosen_id": str(chosen["candidate_id"]),
        "chosen_source": str(chosen["source"]),
        "chosen_row_exact": float(chosen["row_exact"]),
        "chosen_full_exact": bool(chosen["exact"]),
        "raw": raw,
    }


def table_metrics(cand: pd.Series) -> Dict[str, Any]:
    return {
        "candidate_id": str(cand["candidate_id"]),
        "source": str(cand["source"]),
        "row_exact": float(cand["row_exact"]),
        "full_task_exact": bool(cand["exact"]),
        "outputs_json": str(cand["outputs_json"]),
    }


def run_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    table_df, oracle_df, selected_df = copy_candidate_pool(run_dir, Path(args.candidate_source))

    task_ids = sorted(table_df["task_id"].unique())
    all_tasks = {t.task_id: t for t in choose_tasks(load_tasks(min_examples=args.min_examples), args.task_limit, args.seed, args.train_n, args.heldout_cap, args.min_heldout)}
    tasks = [all_tasks[tid] for tid in task_ids if tid in all_tasks]
    train_map: Dict[str, List[Example]] = {}
    test_map: Dict[str, List[Example]] = {}
    for task in tasks:
        train, test = split_examples(task, args.train_n, args.heldout_cap)
        train_map[task.task_id] = train
        test_map[task.task_id] = test

    rng = random.Random(args.seed)
    cache_path = run_dir / "pairwise_judgments.csv"
    cache = read_judgment_cache(cache_path)
    file_exists = cache_path.exists()
    tok = model = None
    if not args.no_qwen:
        tok, model = load_qwen()

    fieldnames = ["task_id", "mode", "pair_kind", "cand_a_id", "cand_b_id", "order_tag", "choice", "raw"]
    judge_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    with cache_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        calls_planned = 0
        calls_done = 0
        # Baseline selections and deployable tournaments.
        for task in tasks:
            task_table = table_df[table_df["task_id"].eq(task.task_id)].copy()
            if task_table.empty:
                continue
            direct_hit = source_candidate(task_table, task.task_id, "row_greedy")
            direct = direct_hit if direct_hit is not None else task_table.iloc[0]
            oracle = exact_candidate(task_table, task.task_id)
            shortlist = shortlist_candidates(task_table, task.task_id, args.shortlist)

            for method, cand in [
                ("direct_row_greedy", direct),
                ("table_oracle", oracle if oracle is not None else task_table.sort_values("row_exact", ascending=False).iloc[0]),
            ]:
                selected_rows.append(
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "features": ",".join(task.features),
                        "method": method,
                        **table_metrics(cand),
                        "candidate_tables": len(task_table),
                        "table_candidate_oracle": oracle is not None,
                    }
                )

            # Deployable tournament starts from direct and challenges by shortlist order.
            current = direct
            tournament_path = [str(current["candidate_id"])]
            for challenger in shortlist:
                if str(challenger["candidate_id"]) == str(current["candidate_id"]):
                    continue
                calls_planned += 1
                res = judge_pair(
                    task,
                    train_map[task.task_id],
                    test_map[task.task_id],
                    current,
                    challenger,
                    "normal",
                    "tournament",
                    f"step{len(tournament_path)}",
                    rng,
                    cache,
                    writer,
                    tok,
                    model,
                    args.max_new_tokens,
                    args.no_qwen,
                )
                judge_rows.append(res)
                calls_done += 1
                if res["chosen_id"] == str(challenger["candidate_id"]):
                    current = challenger
                tournament_path.append(str(current["candidate_id"]))
            selected_rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "features": ",".join(task.features),
                    "method": "pairwise_tournament",
                    **table_metrics(current),
                    "candidate_tables": len(task_table),
                    "table_candidate_oracle": oracle is not None,
                    "tournament_path": json.dumps(tournament_path),
                }
            )

            if oracle is not None:
                # Diagnostic direct-vs-correct comparison. Randomize and also run swapped order.
                for mode in ["normal", "no_examples", "shuffled_examples", "row_shuffled_candidate"]:
                    for swap in [False, True] if mode == "normal" else [False]:
                        a, b = (oracle, direct) if swap else (direct, oracle)
                        order_tag = "swapped" if swap else "base"
                        calls_planned += 1
                        res = judge_pair(
                            task,
                            train_map[task.task_id],
                            test_map[task.task_id],
                            a,
                            b,
                            mode,
                            "direct_vs_correct",
                            order_tag,
                            rng,
                            cache,
                            writer,
                            tok,
                            model,
                            args.max_new_tokens,
                            args.no_qwen,
                        )
                        res["oracle_id"] = str(oracle["candidate_id"])
                        res["direct_id"] = str(direct["candidate_id"])
                        res["picked_oracle"] = res["chosen_id"] == str(oracle["candidate_id"])
                        res["direct_full_exact"] = bool(direct["exact"])
                        res["oracle_full_exact"] = bool(oracle["exact"])
                        diagnostic_rows.append(res)
                        judge_rows.append(res)
                        calls_done += 1
                # Row-level contrastive repair diagnostic: direct vs oracle rows for wrong direct rows.
                direct_out = parse_outputs(direct["outputs_json"])
                oracle_out = parse_outputs(oracle["outputs_json"])
                repaired = list(direct_out)
                for idx, (dout, oout) in enumerate(zip(direct_out, oracle_out)):
                    if norm_eq(dout, oout):
                        continue
                    # Represent a one-row replacement as a candidate table.
                    challenger = oracle.copy()
                    new_out = list(direct_out)
                    new_out[idx] = oout
                    challenger["candidate_id"] = f"{task.task_id}::rowrepair{idx}"
                    challenger["source"] = "row_repair_oracle_alt"
                    challenger["outputs_json"] = json.dumps(new_out, ensure_ascii=False)
                    challenger["row_exact"] = sum(norm_eq(a, b) for a, b in zip(new_out, [e.output for e in test_map[task.task_id]])) / max(1, len(new_out))
                    challenger["exact"] = all(norm_eq(a, b) for a, b in zip(new_out, [e.output for e in test_map[task.task_id]]))
                    calls_planned += 1
                    res = judge_pair(
                        task,
                        train_map[task.task_id],
                        test_map[task.task_id],
                        direct,
                        challenger,
                        "normal",
                        "row_repair",
                        f"row{idx}",
                        rng,
                        cache,
                        writer,
                        tok,
                        model,
                        args.max_new_tokens,
                        args.no_qwen,
                    )
                    judge_rows.append(res)
                    calls_done += 1
                    if res["chosen_id"] == str(challenger["candidate_id"]):
                        repaired[idx] = oout
                targets = [e.output for e in test_map[task.task_id]]
                selected_rows.append(
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "features": ",".join(task.features),
                        "method": "row_repair_diagnostic",
                        "candidate_id": f"{task.task_id}::row_repair_diagnostic",
                        "source": "row_repair_diagnostic",
                        "row_exact": row_exact(repaired, targets),
                        "full_task_exact": full_exact(repaired, targets),
                        "outputs_json": json.dumps(repaired, ensure_ascii=False),
                        "candidate_tables": len(task_table),
                        "table_candidate_oracle": oracle is not None,
                    }
                )
            if calls_done and (calls_done == 1 or calls_done % 50 == 0):
                print(f"pair judgments completed {calls_done}", flush=True)
        print(f"pair judgments completed {calls_done}", flush=True)

    judge_df = pd.DataFrame(judge_rows)
    selected_out = pd.DataFrame(selected_rows)
    diag_df = pd.DataFrame(diagnostic_rows)
    judge_df.to_csv(run_dir / "judge_details.csv", index=False)
    selected_out.to_csv(run_dir / "selected_tables.csv", index=False)
    diag_df.to_csv(run_dir / "diagnostic_direct_vs_correct.csv", index=False)
    summary, mode_summary, headroom_summary = summarize(selected_out, diag_df, judge_df)
    summary.to_csv(ANALYSIS / "summary.csv", index=False)
    mode_summary.to_csv(ANALYSIS / "diagnostic_summary.csv", index=False)
    headroom_summary.to_csv(ANALYSIS / "diagnostic_headroom_summary.csv", index=False)
    selected_out.to_csv(ANALYSIS / "selected_tables.csv", index=False)
    judge_df.to_csv(ANALYSIS / "judge_details.csv", index=False)
    diag_df.to_csv(ANALYSIS / "diagnostic_direct_vs_correct.csv", index=False)
    config = vars(args).copy()
    config.update(
        {
            "tasks": len(tasks),
            "candidate_rows": len(table_df),
            "judgment_rows": len(judge_df),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    tournament_changes = compute_tournament_changes(selected_out)
    tournament_changes.to_csv(ANALYSIS / "tournament_changes.csv", index=False)
    make_plots(summary, mode_summary, headroom_summary, selected_out, diag_df)
    write_reports(args.run_name, config, summary, mode_summary, headroom_summary, selected_out, diag_df, tournament_changes)


def summarize(selected: pd.DataFrame, diag: pd.DataFrame, judge: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        selected.groupby("method", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            row_exact=("row_exact", "mean"),
            full_task_exact=("full_task_exact", "mean"),
            table_oracle_rate=("table_candidate_oracle", "mean"),
            median_candidate_tables=("candidate_tables", "median"),
        )
        .sort_values("full_task_exact", ascending=False)
    )
    if diag.empty:
        mode_summary = pd.DataFrame()
        headroom_summary = pd.DataFrame()
    else:
        mode_summary = (
            diag.groupby(["mode", "pair_kind"], as_index=False)
            .agg(
                comparisons=("task_id", "count"),
                picked_oracle=("picked_oracle", "mean"),
                direct_full_exact=("direct_full_exact", "mean"),
                oracle_full_exact=("oracle_full_exact", "mean"),
            )
            .sort_values(["pair_kind", "mode"])
        )
        headroom = diag[(diag["oracle_full_exact"].astype(bool)) & (~diag["direct_full_exact"].astype(bool))].copy()
        if headroom.empty:
            headroom_summary = pd.DataFrame()
        else:
            headroom_summary = (
                headroom.groupby(["mode", "order_tag"], as_index=False)
                .agg(
                    comparisons=("task_id", "count"),
                    unique_tasks=("task_id", "nunique"),
                    picked_oracle=("picked_oracle", "mean"),
                    picked_candidate_a=("choice", lambda s: float((s == "A").mean())),
                )
                .sort_values(["mode", "order_tag"])
            )
    return summary, mode_summary, headroom_summary


def compute_tournament_changes(selected: pd.DataFrame) -> pd.DataFrame:
    direct = selected[selected.method.eq("direct_row_greedy")][
        ["task_id", "family", "candidate_id", "source", "row_exact", "full_task_exact", "outputs_json"]
    ].rename(
        columns={
            "candidate_id": "direct_id",
            "source": "direct_source",
            "row_exact": "direct_row_exact",
            "full_task_exact": "direct_full_exact",
            "outputs_json": "direct_outputs",
        }
    )
    tournament = selected[selected.method.eq("pairwise_tournament")][
        ["task_id", "candidate_id", "source", "row_exact", "full_task_exact", "outputs_json"]
    ].rename(
        columns={
            "candidate_id": "tournament_id",
            "source": "tournament_source",
            "row_exact": "tournament_row_exact",
            "full_task_exact": "tournament_full_exact",
            "outputs_json": "tournament_outputs",
        }
    )
    if direct.empty or tournament.empty:
        return pd.DataFrame()
    merged = direct.merge(tournament, on="task_id")
    merged["changed_output"] = merged.direct_outputs.ne(merged.tournament_outputs)
    merged["delta_full_exact"] = merged.tournament_full_exact.astype(float) - merged.direct_full_exact.astype(float)
    merged["delta_row_exact"] = merged.tournament_row_exact.astype(float) - merged.direct_row_exact.astype(float)
    return merged.sort_values(["changed_output", "delta_full_exact", "task_id"], ascending=[False, True, True])


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100 * float(x):.1f}%"


def md_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._"
    view = df[list(cols)].head(max_rows).copy()
    for c in view.columns:
        if view[c].dtype.kind in "fc":
            if "exact" in c or "rate" in c or "oracle" in c or "picked" in c or "candidate_a" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def make_plots(summary: pd.DataFrame, mode_summary: pd.DataFrame, headroom_summary: pd.DataFrame, selected: pd.DataFrame, diag: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = list(summary.sort_values("full_task_exact")["method"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    vals = [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order]
    ax.barh(order, vals, color="#2563eb")
    ax.set_xlabel("Full-task exact (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Selected-table accuracy")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "full_task_by_method.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(order))
    ax.plot(x, [float(summary[summary.method.eq(m)]["row_exact"].iloc[0]) * 100 for m in order], marker="o", label="row exact", color="#059669")
    ax.plot(x, vals, marker="o", label="full-task exact", color="#dc2626")
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Row accuracy versus full-task consistency")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "row_vs_task.png", dpi=160)
    plt.close(fig)

    if not mode_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        labels = mode_summary["mode"] + "/" + mode_summary["pair_kind"]
        ax.bar(labels, mode_summary["picked_oracle"] * 100, color="#7c3aed")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Picked hidden-correct table (%)")
        ax.set_title("Diagnostic direct-vs-correct judge accuracy")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "diagnostic_pick_oracle.png", dpi=160)
        plt.close(fig)

    if not headroom_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        labels = headroom_summary["mode"] + "/" + headroom_summary["order_tag"]
        ax.bar(labels, headroom_summary["picked_oracle"] * 100, color="#9333ea")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Picked hidden-correct table (%)")
        ax.set_title("Headroom-only diagnostic")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "diagnostic_headroom_pick_oracle.png", dpi=160)
        plt.close(fig)

    merged = compute_tournament_changes(selected)
    if not merged.empty:
        changed = merged.changed_output.astype(bool)
        vals2 = {
            "same output": int((~changed).sum()),
            "changed helped": int((changed & (merged.delta_full_exact > 0)).sum()),
            "changed tied": int((changed & (merged.delta_full_exact == 0)).sum()),
            "changed hurt": int((changed & (merged.delta_full_exact < 0)).sum()),
        }
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(vals2.keys(), vals2.values(), color=["#9ca3af", "#059669", "#2563eb", "#dc2626"])
        ax.set_ylabel("Tasks")
        ax.set_title("Pairwise tournament changes relative to direct")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "tournament_changes.png", dpi=160)
        plt.close(fig)

    if not selected.empty:
        family = selected.groupby(["method", "family"], as_index=False).agg(full_task_exact=("full_task_exact", "mean"))
        pivot = family.pivot_table(index="family", columns="method", values="full_task_exact", aggfunc="mean").fillna(0)
        pivot = pivot.loc[pivot.index.sort_values()]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(pivot))))
        im = ax.imshow(pivot.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("Full-task exact by family")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]*100:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="%")
        fig.tight_layout()
        fig.savefig(FIGURES / "family_heatmap.png", dpi=160)
        plt.close(fig)


def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    out: List[str] = []
    in_table = False
    for line in lines:
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set(cells) == {"---"}:
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
            if line.strip():
                escaped = html.escape(line)
                escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
                out.append(f"<p>{escaped}</p>")
    if in_table:
        out.append("</table>")
    css = """
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; margin: 40px; color: #111827; line-height: 1.5; }
    h1 { font-size: 32px; margin-bottom: 4px; }
    h2 { margin-top: 30px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }
    th { background: #f3f4f6; text-align: left; }
    img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 4px; }
    figure { margin: 18px 0 26px; }
    figcaption { color: #4b5563; font-size: 13px; margin-top: 6px; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
    """
    return "<!doctype html><html><head><meta charset='utf-8'><title>Pairwise Table Judge</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>"


def write_reports(
    run_name: str,
    config: Dict[str, Any],
    summary: pd.DataFrame,
    mode_summary: pd.DataFrame,
    headroom_summary: pd.DataFrame,
    selected: pd.DataFrame,
    diag: pd.DataFrame,
    tournament_changes: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    direct = summary[summary.method.eq("direct_row_greedy")]
    tournament = summary[summary.method.eq("pairwise_tournament")]
    oracle = summary[summary.method.eq("table_oracle")]
    repair = summary[summary.method.eq("row_repair_diagnostic")]
    direct_full = float(direct.full_task_exact.iloc[0]) if not direct.empty else float("nan")
    tournament_full = float(tournament.full_task_exact.iloc[0]) if not tournament.empty else float("nan")
    oracle_full = float(oracle.full_task_exact.iloc[0]) if not oracle.empty else float("nan")
    repair_full = float(repair.full_task_exact.iloc[0]) if not repair.empty else float("nan")
    normal_diag = mode_summary[(mode_summary["mode"].eq("normal")) & (mode_summary["pair_kind"].eq("direct_vs_correct"))] if not mode_summary.empty else pd.DataFrame()
    normal_pick = float(normal_diag.picked_oracle.mean()) if not normal_diag.empty else float("nan")
    changed = tournament_changes[tournament_changes.changed_output.astype(bool)].copy() if not tournament_changes.empty else pd.DataFrame()
    headroom_normal = (
        headroom_summary[headroom_summary["mode"].eq("normal")]
        if not headroom_summary.empty
        else pd.DataFrame()
    )
    headroom_base = headroom_normal[headroom_normal["order_tag"].eq("base")] if not headroom_normal.empty else pd.DataFrame()
    headroom_swapped = headroom_normal[headroom_normal["order_tag"].eq("swapped")] if not headroom_normal.empty else pd.DataFrame()
    headroom_base_pick = float(headroom_base.picked_oracle.iloc[0]) if not headroom_base.empty else float("nan")
    headroom_swapped_pick = float(headroom_swapped.picked_oracle.iloc[0]) if not headroom_swapped.empty else float("nan")
    report = f"""# Pairwise Table Judge

## Question

Can a model choose the more task-consistent full output table when shown examples, query rows, and two candidate tables?

This experiment evaluates pairwise table judging on public text-transformation tasks. The primary deployable method is a tournament over a non-label shortlist of candidate tables. A separate diagnostic compares direct greedy tables against hidden-correct tables when the hidden-correct table is present.

## Setup

- Benchmark root: `{BENCH_ROOT}`
- Run: `{run_name}`
- Tasks: {config.get('tasks')}
- Candidate table rows: {config.get('candidate_rows')}
- Pairwise judgment rows: {config.get('judgment_rows')}
- Shortlist size: {config.get('shortlist')}
- Train rows per task: {config.get('train_n')}
- Held-out cap per task: {config.get('heldout_cap')}

## Main Result

{md_table(summary, ['method', 'tasks', 'row_exact', 'full_task_exact', 'table_oracle_rate', 'median_candidate_tables'])}

## Interpretation

The deployable pairwise tournament changes full-task exact by {(tournament_full - direct_full) * 100:.1f} points relative to direct greedy. The table oracle is {(oracle_full - direct_full) * 100:.1f} points above direct greedy, so any gap between tournament and oracle is selection headroom. In the all-oracle-task direct-vs-hidden-correct diagnostic, the normal judge picks the hidden-correct table {normal_pick * 100:.1f}% of the time, but that aggregate includes saturated tasks where direct and oracle are identical.

On the actual headroom subset, where direct greedy is wrong and a hidden-correct table exists, the normal judge picks the hidden-correct table {headroom_base_pick * 100:.1f}% when direct is candidate A and {headroom_swapped_pick * 100:.1f}% when the hidden-correct table is candidate A. This is the critical diagnostic: a large base/swapped gap indicates position bias rather than semantic table judging.

The row-repair diagnostic reaches {repair_full * 100:.1f}% full-task exact. It is not deployable because it uses hidden oracle row alternatives; it measures whether the judge can accept correct row-level replacements when they are explicitly supplied.

## Charts

![Full-task exact by method](../analysis/figures/full_task_by_method.png)

![Row versus task accuracy](../analysis/figures/row_vs_task.png)

![Diagnostic pick oracle](../analysis/figures/diagnostic_pick_oracle.png)

![Headroom diagnostic pick oracle](../analysis/figures/diagnostic_headroom_pick_oracle.png)

![Tournament changes](../analysis/figures/tournament_changes.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

## Diagnostic Summary

{md_table(mode_summary, ['mode', 'pair_kind', 'comparisons', 'picked_oracle', 'direct_full_exact', 'oracle_full_exact'])}

## Headroom-Only Diagnostic

{md_table(headroom_summary, ['mode', 'order_tag', 'comparisons', 'unique_tasks', 'picked_oracle', 'picked_candidate_a'])}

## Selected Tables

{md_table(selected.sort_values(['method', 'task_id']), ['task_id', 'family', 'method', 'source', 'row_exact', 'full_task_exact', 'candidate_tables', 'table_candidate_oracle'], max_rows=160)}

## Tournament Changes

{md_table(changed, ['task_id', 'family', 'direct_source', 'tournament_source', 'direct_row_exact', 'tournament_row_exact', 'direct_full_exact', 'tournament_full_exact', 'delta_row_exact', 'delta_full_exact'], max_rows=120) if not changed.empty else "_The tournament selected the same output table as direct greedy for every task._"}

## Files

- `runs/{run_name}/table_candidates.csv`
- `runs/{run_name}/oracle_summary.csv`
- `runs/{run_name}/pairwise_judgments.csv`
- `runs/{run_name}/judge_details.csv`
- `runs/{run_name}/selected_tables.csv`
- `runs/{run_name}/diagnostic_direct_vs_correct.csv`
- `analysis/summary.csv`
- `analysis/diagnostic_summary.csv`
- `analysis/diagnostic_headroom_summary.csv`
- `analysis/tournament_changes.csv`
- `analysis/selected_tables.csv`
- `analysis/judge_details.csv`
"""
    (REPORTS / "qwen_pairwise_table_judge_report.md").write_text(report)
    (REPORTS / "qwen_pairwise_table_judge_report.html").write_text(markdown_to_html(report))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="main")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--task_limit", type=int, default=40)
    ap.add_argument("--train_n", type=int, default=4)
    ap.add_argument("--heldout_cap", type=int, default=6)
    ap.add_argument("--min_examples", type=int, default=5)
    ap.add_argument("--min_heldout", type=int, default=3)
    ap.add_argument("--candidate_source", default=str(DEFAULT_CANDIDATE_SOURCE))
    ap.add_argument("--shortlist", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--no_qwen", action="store_true")
    return ap.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

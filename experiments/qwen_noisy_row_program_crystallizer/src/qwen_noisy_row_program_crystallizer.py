#!/usr/bin/env python3
"""Noisy-row program crystallizer.

This standalone experiment asks whether noisy row-level model guesses can be
converted into a single deterministic task program. The model is used only as a
row-candidate proposer. Selection is deterministic: enumerate programs that fit
the visible examples, then choose the program whose held-out predictions are
most supported by the candidate pool.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsl_core import Example, Expr, Task, candidate_exprs


ROOT = Path("/workspace/experiments/qwen_noisy_row_program_crystallizer")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_noisy_row_program_crystallizer")
BENCH_ROOT = LARGE_ROOT / "prose-benchmarks"
SOURCE_BENCH_ROOT = BENCH_ROOT
TRANSFORM_ROOT = BENCH_ROOT / "Transformation.Text"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for d in [RUNS, ANALYSIS, FIGURES, REPORTS, LARGE_ROOT]:
        d.mkdir(parents=True, exist_ok=True)


def mirror_benchmark() -> None:
    if BENCH_ROOT.exists():
        return
    fallback = Path("/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks")
    source = SOURCE_BENCH_ROOT if SOURCE_BENCH_ROOT.exists() else fallback
    if not source.exists():
        raise FileNotFoundError(f"Missing benchmark source: {source}")
    LARGE_ROOT.mkdir(parents=True, exist_ok=True)
    BENCH_ROOT.symlink_to(source, target_is_directory=True)


def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", "" if s is None else str(s)).strip()


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
    for task in tasks:
        _, test = split_examples(task, train_n, heldout_cap)
        if len(test) >= min_heldout:
            eligible.append(task)
    rng = random.Random(seed)
    if limit and limit < len(eligible):
        chosen = rng.sample(eligible, limit)
    else:
        chosen = eligible
    return sorted(chosen, key=lambda t: t.task_id)


def candidate_row_sets(task_id: str, candidate_df: pd.DataFrame, n_rows: int) -> List[Counter[str]]:
    sub = candidate_df[candidate_df["task_id"].eq(task_id)].copy()
    out: List[Counter[str]] = []
    for i in range(n_rows):
        row = sub[sub["row_index"].astype(int).eq(i)]
        counts: Counter[str] = Counter()
        for _, r in row.iterrows():
            pred = clean(r.get("prediction", ""))
            if pred:
                counts[pred] += 1
        out.append(counts)
    return out


def method_outputs(task_id: str, candidate_df: pd.DataFrame, method: str, n_rows: int) -> Optional[Tuple[str, ...]]:
    sub = candidate_df[candidate_df["task_id"].eq(task_id) & candidate_df["method"].eq(method)]
    if sub.empty:
        return None
    vals: List[str] = []
    for i in range(n_rows):
        hit = sub[sub["row_index"].astype(int).eq(i)]
        if hit.empty:
            return None
        vals.append(clean(hit.iloc[0]["prediction"]))
    return tuple(vals)


def row_majority_outputs(row_sets: Sequence[Counter[str]]) -> Tuple[str, ...]:
    vals: List[str] = []
    for counts in row_sets:
        vals.append(counts.most_common(1)[0][0] if counts else "")
    return tuple(vals)


def ranked_row_sets(row_sets: Sequence[Counter[str]], topk: int) -> List[List[Tuple[str, int]]]:
    return [[(pred, count) for pred, count in counts.most_common(topk)] for counts in row_sets]


def table_vote_share(outputs: Sequence[str], row_sets: Sequence[Counter[str]]) -> float:
    shares: List[float] = []
    for pred, counts in zip(outputs, row_sets):
        total = sum(counts.values())
        shares.append(0.0 if total <= 0 else counts.get(clean(pred), 0) / total)
    return float(np.mean(shares)) if shares else 0.0


def enumerate_candidate_tables(task_id: str, candidate_df: pd.DataFrame, row_sets: Sequence[Counter[str]], max_tables: int, topk_per_row: int) -> List[Dict[str, Any]]:
    n_rows = len(row_sets)
    tables: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    methods = sorted(str(x) for x in candidate_df[candidate_df["task_id"].eq(task_id)]["method"].dropna().unique())
    for method in methods:
        vals = method_outputs(task_id, candidate_df, method, n_rows)
        if vals is None:
            continue
        tables.setdefault(vals, {"outputs": vals, "source": method, "avg_vote_share": table_vote_share(vals, row_sets)})
    majority = row_majority_outputs(row_sets)
    tables.setdefault(majority, {"outputs": majority, "source": "row_majority", "avg_vote_share": table_vote_share(majority, row_sets)})

    ranked = ranked_row_sets(row_sets, topk_per_row)
    if all(ranked):
        product = 1
        for rs in ranked:
            product *= len(rs)
        if product <= max_tables:
            import itertools

            combos = itertools.product(*ranked)
            for combo in combos:
                vals = tuple(pred for pred, _ in combo)
                tables.setdefault(vals, {"outputs": vals, "source": "row_combo", "avg_vote_share": table_vote_share(vals, row_sets)})
                if len(tables) >= max_tables:
                    break
        else:
            beam: List[Tuple[float, List[str]]] = [(0.0, [])]
            for rs, counts in zip(ranked, row_sets):
                total = sum(counts.values()) or 1
                new: List[Tuple[float, List[str]]] = []
                for score, prefix in beam:
                    for pred, count in rs:
                        new.append((score + math.log((count + 0.25) / (total + 0.25 * (len(counts) + 1))), prefix + [pred]))
                new.sort(key=lambda x: x[0], reverse=True)
                beam = new[: max_tables]
            for _, vals_list in beam[:max_tables]:
                vals = tuple(vals_list)
                tables.setdefault(vals, {"outputs": vals, "source": "row_combo", "avg_vote_share": table_vote_share(vals, row_sets)})
    out = list(tables.values())
    out.sort(key=lambda r: (r["avg_vote_share"], r["source"] == "row_greedy"), reverse=True)
    return out[:max_tables]


def eval_expr(expr: Optional[Expr], rows: Sequence[Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
    if expr is None:
        return None
    vals = expr.eval_many(rows)
    if vals is None:
        return None
    return tuple(clean(x) for x in vals)


def full_exact(outputs: Optional[Sequence[str]], targets: Sequence[str]) -> bool:
    if outputs is None or len(outputs) != len(targets):
        return False
    return all(norm_eq(a, b) for a, b in zip(outputs, targets))


def row_exact(outputs: Optional[Sequence[str]], targets: Sequence[str]) -> float:
    if outputs is None or not targets:
        return 0.0
    return sum(norm_eq(a, b) for a, b in zip(outputs, targets)) / len(targets)


def train_matching_programs(task: Task, train: Sequence[Example], max_candidates: int) -> Tuple[List[Expr], int]:
    train_rows = [e.inputs for e in train]
    train_y = tuple(clean(e.output) for e in train)
    num_cols = max(len(e.inputs) for e in task.examples)
    candidates = candidate_exprs(num_cols, train_rows, train_y, max_candidates)
    matches: List[Expr] = []
    for expr in candidates:
        vals = expr.eval_many(train_rows)
        if vals is not None and tuple(clean(x) for x in vals) == train_y:
            matches.append(expr)
    return matches, len(candidates)


def fit_pseudo_program(
    task: Task,
    train: Sequence[Example],
    test_rows: Sequence[Tuple[str, ...]],
    pseudo_outputs: Sequence[str],
    max_candidates: int,
    allow_maps: bool,
) -> Tuple[Optional[Expr], int]:
    rows = [e.inputs for e in train] + list(test_rows)
    ys = tuple(clean(e.output) for e in train) + tuple(clean(x) for x in pseudo_outputs)
    num_cols = max(len(e.inputs) for e in task.examples)
    candidates = candidate_exprs(num_cols, rows, ys, max_candidates)
    checked = 0
    for expr in sorted(candidates, key=lambda e: (e.depth, len(e.code), e.code)):
        checked += 1
        if not allow_maps and (expr.kind == "map" or expr.code.startswith("map[")):
            continue
        vals = expr.eval_many(rows)
        if vals is not None and tuple(clean(x) for x in vals) == ys:
            return expr, checked
    return None, checked


def select_table_by_pseudo_program(
    task: Task,
    train: Sequence[Example],
    test_rows: Sequence[Tuple[str, ...]],
    tables: Sequence[Dict[str, Any]],
    max_candidates: int,
    allow_maps: bool,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    scored: List[Tuple[float, int, int, str, Dict[str, Any], Expr]] = []
    checked_total = 0
    coherent = 0
    for table in tables:
        expr, checked = fit_pseudo_program(task, train, test_rows, table["outputs"], max_candidates, allow_maps)
        checked_total += checked
        if expr is None:
            continue
        coherent += 1
        scored.append((float(table["avg_vote_share"]), -expr.depth, -len(expr.code), expr.code, table, expr))
    if not scored:
        return None, {"coherent_tables": 0, "pseudo_checked_programs": checked_total}
    scored.sort(reverse=True)
    vote, _, _, _, table, expr = scored[0]
    out = dict(table)
    out["program"] = expr.code
    out["program_kind"] = expr.kind
    return out, {"coherent_tables": coherent, "pseudo_checked_programs": checked_total, "pseudo_avg_vote_share": vote}


def support_score(outputs: Optional[Sequence[str]], row_sets: Sequence[Counter[str]], alpha: float = 0.25) -> Tuple[float, float, int]:
    if outputs is None or len(outputs) != len(row_sets):
        return -1e9, 0.0, 0
    score = 0.0
    supported = 0
    vote_share_sum = 0.0
    for pred, counts in zip(outputs, row_sets):
        pred = clean(pred)
        total = sum(counts.values())
        unique = max(1, len(counts))
        count = counts.get(pred, 0)
        if total <= 0:
            score += math.log(alpha / (1.0 + alpha * unique))
            continue
        share = count / total
        vote_share_sum += share
        supported += int(count > 0)
        score += math.log((count + alpha) / (total + alpha * (unique + 1)))
    return score, vote_share_sum / max(1, len(row_sets)), supported


def select_program_by_support(matches: Sequence[Expr], test_rows: Sequence[Tuple[str, ...]], row_sets: Sequence[Counter[str]]) -> Tuple[Optional[Expr], Dict[str, Any]]:
    scored: List[Tuple[float, float, int, int, int, str, Expr, Optional[Tuple[str, ...]]]] = []
    for expr in matches:
        outputs = eval_expr(expr, test_rows)
        score, avg_share, supported = support_score(outputs, row_sets)
        # The final tuple order makes score primary and complexity a tiebreaker.
        scored.append((score, avg_share, supported, -expr.depth, -len(expr.code), expr.code, expr, outputs))
    if not scored:
        return None, {"support_score": -1e9, "avg_vote_share": 0.0, "supported_rows": 0, "candidate_programs": 0}
    scored.sort(reverse=True)
    score, avg_share, supported, _, _, _, expr, outputs = scored[0]
    return expr, {
        "support_score": float(score),
        "avg_vote_share": float(avg_share),
        "supported_rows": int(supported),
        "candidate_programs": len(scored),
        "selected_outputs": json.dumps(outputs or [], ensure_ascii=False),
    }


def shortest_program(matches: Sequence[Expr]) -> Optional[Expr]:
    if not matches:
        return None
    return min(matches, key=lambda e: (e.depth, len(e.code), e.code))


def oracle_program(matches: Sequence[Expr], test_rows: Sequence[Tuple[str, ...]], targets: Sequence[str]) -> Optional[Expr]:
    exact: List[Expr] = []
    for expr in matches:
        if full_exact(eval_expr(expr, test_rows), targets):
            exact.append(expr)
    return shortest_program(exact)


def rotate_row_sets(task_ids: Sequence[str], row_sets_by_task: Dict[str, List[Counter[str]]]) -> Dict[str, List[Counter[str]]]:
    ids = sorted(task_ids)
    out: Dict[str, List[Counter[str]]] = {}
    for i, tid in enumerate(ids):
        donor = ids[(i + 1) % len(ids)]
        mine = row_sets_by_task[tid]
        theirs = row_sets_by_task[donor]
        if not theirs:
            out[tid] = [Counter() for _ in mine]
            continue
        out[tid] = [theirs[j % len(theirs)] for j in range(len(mine))]
    return out


def load_candidate_cache(path: Path, tasks: Sequence[Task], run_dir: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate cache: {path}")
    df = pd.read_csv(path)
    ids = {t.task_id for t in tasks}
    df = df[df["task_id"].isin(ids)].copy()
    local = run_dir / "row_candidates.csv"
    df.to_csv(local, index=False)
    return df


def evaluate(tasks: Sequence[Task], candidate_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    precomputed_row_sets: Dict[str, List[Counter[str]]] = {}
    splits: Dict[str, Tuple[List[Example], List[Example]]] = {}
    for task in tasks:
        train, test = split_examples(task, args.train_n, args.heldout_cap)
        splits[task.task_id] = (train, test)
        precomputed_row_sets[task.task_id] = candidate_row_sets(task.task_id, candidate_df, len(test))
    shuffled_row_sets = rotate_row_sets([t.task_id for t in tasks], precomputed_row_sets)

    task_rows: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        train, test = splits[task.task_id]
        test_rows = [e.inputs for e in test]
        targets = tuple(clean(e.output) for e in test)
        row_sets = precomputed_row_sets[task.task_id]
        shuffled_sets = shuffled_row_sets[task.task_id]
        matches, candidate_count = train_matching_programs(task, train, args.max_candidates)
        tables = enumerate_candidate_tables(task.task_id, candidate_df, row_sets, args.max_tables, args.topk_per_row)
        shuffled_tables = enumerate_candidate_tables(task.task_id, candidate_df.iloc[0:0].copy(), shuffled_sets, args.shuffled_max_tables, args.topk_per_row)
        shortest = shortest_program(matches)
        support, support_meta = select_program_by_support(matches, test_rows, row_sets)
        shuffled, shuffled_meta = select_program_by_support(matches, test_rows, shuffled_sets)
        pseudo, pseudo_meta = select_table_by_pseudo_program(task, train, test_rows, tables, args.pseudo_max_candidates, args.allow_pseudo_maps)
        shuffled_pseudo, shuffled_pseudo_meta = select_table_by_pseudo_program(task, train, test_rows, shuffled_tables, args.shuffled_pseudo_max_candidates, args.allow_pseudo_maps)
        oracle = oracle_program(matches, test_rows, targets)
        direct = method_outputs(task.task_id, candidate_df, "row_greedy", len(test))
        majority = row_majority_outputs(row_sets)
        row_oracle = bool(row_sets and all(any(norm_eq(p, y) for p in counts for y in [target]) for counts, target in zip(row_sets, targets)))
        table_oracle = any(full_exact(t["outputs"], targets) for t in tables)
        program_oracle = oracle is not None
        pseudo_outputs = tuple(pseudo["outputs"]) if pseudo else None
        shuffled_pseudo_outputs = tuple(shuffled_pseudo["outputs"]) if shuffled_pseudo else None
        use_pseudo = bool(pseudo and pseudo_meta.get("pseudo_avg_vote_share", 0.0) >= args.pseudo_min_vote_share)
        pseudo_or_direct = pseudo_outputs if use_pseudo else direct

        programs = {
            "direct_row_greedy": (direct, "", "row_candidate", {}),
            "row_majority": (majority, "", "row_candidate", {}),
            "examples_shortest_program": (eval_expr(shortest, test_rows), shortest.code if shortest else "", shortest.kind if shortest else "", {}),
            "candidate_support_program": (eval_expr(support, test_rows), support.code if support else "", support.kind if support else "", support_meta),
            "shuffled_support_program": (eval_expr(shuffled, test_rows), shuffled.code if shuffled else "", shuffled.kind if shuffled else "", shuffled_meta),
            "pseudo_program_table": (pseudo_outputs, pseudo.get("program", "") if pseudo else "", pseudo.get("program_kind", "") if pseudo else "", pseudo_meta),
            "pseudo_program_or_direct": (pseudo_or_direct, pseudo.get("program", "") if use_pseudo and pseudo else "", pseudo.get("program_kind", "") if use_pseudo and pseudo else "direct_fallback", {**pseudo_meta, "used_pseudo": int(use_pseudo)}),
            "shuffled_pseudo_program_table": (shuffled_pseudo_outputs, shuffled_pseudo.get("program", "") if shuffled_pseudo else "", shuffled_pseudo.get("program_kind", "") if shuffled_pseudo else "", shuffled_pseudo_meta),
            "program_oracle": (eval_expr(oracle, test_rows), oracle.code if oracle else "", oracle.kind if oracle else "", {}),
            "row_candidate_oracle": (tuple(targets) if row_oracle else None, "hidden row-candidate oracle", "oracle", {}),
            "table_candidate_oracle": (tuple(targets) if table_oracle else None, "hidden table-candidate oracle", "oracle", {}),
        }
        for method, (outputs, code, kind, meta) in programs.items():
            row = {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "method": method,
                "row_exact": row_exact(outputs, targets),
                "full_task_exact": full_exact(outputs, targets),
                "program": code,
                "program_kind": kind,
                "outputs_json": json.dumps(outputs or [], ensure_ascii=False),
            }
            row.update(meta)
            method_rows.append(row)

        support_outputs = eval_expr(support, test_rows)
        shortest_outputs = eval_expr(shortest, test_rows)
        task_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "heldout_rows": len(test),
                "candidate_count": candidate_count,
                "train_match_count": len(matches),
                "program_oracle": program_oracle,
                "row_candidate_oracle": row_oracle,
                "table_candidate_oracle": table_oracle,
                "direct_full_exact": full_exact(direct, targets),
                "majority_full_exact": full_exact(majority, targets),
                "examples_full_exact": full_exact(shortest_outputs, targets),
                "support_full_exact": full_exact(support_outputs, targets),
                "shuffled_full_exact": full_exact(eval_expr(shuffled, test_rows), targets),
                "pseudo_full_exact": full_exact(pseudo_outputs, targets),
                "pseudo_or_direct_full_exact": full_exact(pseudo_or_direct, targets),
                "shuffled_pseudo_full_exact": full_exact(shuffled_pseudo_outputs, targets),
                "support_changed_from_examples": json.dumps(support_outputs or []) != json.dumps(shortest_outputs or []),
                "support_score": support_meta.get("support_score", -1e9),
                "support_avg_vote_share": support_meta.get("avg_vote_share", 0.0),
                "support_supported_rows": support_meta.get("supported_rows", 0),
                "candidate_tables": len(tables),
                "coherent_pseudo_tables": pseudo_meta.get("coherent_tables", 0),
                "pseudo_avg_vote_share": pseudo_meta.get("pseudo_avg_vote_share", 0.0),
                "pseudo_used_in_fallback": use_pseudo,
                "support_program": support.code if support else "",
                "examples_program": shortest.code if shortest else "",
                "pseudo_program": pseudo.get("program", "") if pseudo else "",
                "oracle_program": oracle.code if oracle else "",
                "targets_json": json.dumps(targets, ensure_ascii=False),
            }
        )
        if idx == 1 or idx % 10 == 0 or idx == len(tasks):
            solved = sum(r["support_full_exact"] for r in task_rows)
            print(f"eval {idx}/{len(tasks)} candidate_support={solved}/{len(task_rows)}", flush=True)
    return pd.DataFrame(task_rows), pd.DataFrame(method_rows)


def summarize(method_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        method_df.groupby("method", as_index=False)
        .agg(tasks=("task_id", "count"), row_exact=("row_exact", "mean"), full_task_exact=("full_task_exact", "mean"))
        .sort_values("full_task_exact", ascending=False)
    )
    family = (
        method_df.groupby(["method", "family"], as_index=False)
        .agg(tasks=("task_id", "count"), row_exact=("row_exact", "mean"), full_task_exact=("full_task_exact", "mean"))
        .sort_values(["method", "family"])
    )
    return summary, family


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100 * float(x):.1f}%"


def md_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df[list(cols)].head(max_rows).copy()
    for c in view.columns:
        if view[c].dtype.kind in "fc":
            if "exact" in c or "oracle" in c or "rate" in c or "share" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def plot_method_bars(summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("full_task_exact")["method"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    vals = [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order]
    ax.barh(order, vals, color="#2563eb")
    ax.set_xlabel("Strict full-task exact (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Program crystallizer versus controls")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_full_task_exact.png", dpi=170)
    plt.close(fig)


def plot_row_vs_full(summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("row_exact")["method"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, [float(summary[summary.method.eq(m)]["row_exact"].iloc[0]) * 100 for m in order], marker="o", label="row exact", color="#059669")
    ax.plot(x, [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], marker="o", label="full-task exact", color="#dc2626")
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Row accuracy versus task consistency")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "row_vs_full_task.png", dpi=170)
    plt.close(fig)


def plot_oracle_decomposition(task_df: pd.DataFrame) -> None:
    vals = {
        "direct solved": int(task_df["direct_full_exact"].astype(bool).sum()),
        "program headroom": int((~task_df["direct_full_exact"].astype(bool) & task_df["program_oracle"].astype(bool)).sum()),
        "no program": int((~task_df["program_oracle"].astype(bool)).sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(vals.keys(), vals.values(), color=["#059669", "#f59e0b", "#dc2626"])
    ax.set_ylabel("Tasks")
    ax.set_title("Direct solutions and deterministic-program headroom")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "program_headroom.png", dpi=170)
    plt.close(fig)


def plot_support_changes(task_df: pd.DataFrame) -> None:
    changed = task_df["support_changed_from_examples"].astype(bool)
    helped = changed & (~task_df["examples_full_exact"].astype(bool)) & task_df["support_full_exact"].astype(bool)
    hurt = changed & task_df["examples_full_exact"].astype(bool) & (~task_df["support_full_exact"].astype(bool))
    tied = changed & (task_df["examples_full_exact"].astype(bool) == task_df["support_full_exact"].astype(bool))
    vals = {
        "unchanged": int((~changed).sum()),
        "changed helped": int(helped.sum()),
        "changed tied": int(tied.sum()),
        "changed hurt": int(hurt.sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(vals.keys(), vals.values(), color=["#9ca3af", "#059669", "#2563eb", "#dc2626"])
    ax.set_ylabel("Tasks")
    ax.set_title("Candidate-support selection changes")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "support_selection_changes.png", dpi=170)
    plt.close(fig)


def plot_family_heatmap(family: pd.DataFrame) -> None:
    keep = ["direct_row_greedy", "examples_shortest_program", "candidate_support_program", "program_oracle"]
    data = family[family["method"].isin(keep)]
    pivot = data.pivot_table(index="family", columns="method", values="full_task_exact", aggfunc="mean").fillna(0)
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.index.sort_values()]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(pivot))))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Full-task exact by family")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j] * 100:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(FIGURES / "family_heatmap.png", dpi=170)
    plt.close(fig)


def plot_support_distribution(task_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ok = task_df[task_df["support_full_exact"].astype(bool)]["support_avg_vote_share"]
    bad = task_df[~task_df["support_full_exact"].astype(bool)]["support_avg_vote_share"]
    ax.hist([bad, ok], bins=10, label=["support selected wrong", "support selected exact"], color=["#dc2626", "#059669"], alpha=0.75)
    ax.set_xlabel("Average row-candidate vote share of selected program")
    ax.set_ylabel("Tasks")
    ax.set_title("Candidate support strength by outcome")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "support_vote_share_distribution.png", dpi=170)
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
            if not line.strip():
                out.append("")
            else:
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
    return "<!doctype html><html><head><meta charset='utf-8'><title>Noisy Row Program Crystallizer</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>"


def write_reports(run_name: str, config: Dict[str, Any], summary: pd.DataFrame, family: pd.DataFrame, task_df: pd.DataFrame, method_df: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    def metric(method: str, col: str) -> float:
        hit = summary[summary["method"].eq(method)]
        return float(hit[col].iloc[0]) if not hit.empty else float("nan")

    direct = metric("direct_row_greedy", "full_task_exact")
    support = metric("candidate_support_program", "full_task_exact")
    pseudo = metric("pseudo_program_table", "full_task_exact")
    pseudo_or_direct = metric("pseudo_program_or_direct", "full_task_exact")
    shuffled_pseudo = metric("shuffled_pseudo_program_table", "full_task_exact")
    shuffled = metric("shuffled_support_program", "full_task_exact")
    examples = metric("examples_shortest_program", "full_task_exact")
    program_oracle = metric("program_oracle", "full_task_exact")
    row_oracle = metric("row_candidate_oracle", "full_task_exact")
    support_delta = support - direct
    support_vs_examples = support - examples
    control_gap = support - shuffled
    oracle_gap = program_oracle - direct
    captured = (support - direct) / oracle_gap if oracle_gap > 1e-9 else None
    captured_text = f"{captured * 100:.1f}%" if captured is not None else "not defined because the deterministic-program oracle is below the direct row baseline"

    headroom = task_df[(~task_df["direct_full_exact"].astype(bool)) & task_df["program_oracle"].astype(bool)].copy()
    support_captured = int(headroom["support_full_exact"].astype(bool).sum()) if not headroom.empty else 0
    changed = task_df[task_df["support_changed_from_examples"].astype(bool)].copy()
    helped = changed[(~changed["examples_full_exact"].astype(bool)) & changed["support_full_exact"].astype(bool)]
    hurt = changed[changed["examples_full_exact"].astype(bool) & (~changed["support_full_exact"].astype(bool))]

    report = f"""# Noisy Row Program Crystallizer

## Question

Can noisy row-level candidate outputs be crystallized into one deterministic program for an entire text-transformation task?

The experiment uses the language model only to propose row outputs. It then enumerates deterministic programs that exactly match the visible examples and selects the program whose held-out predictions receive the strongest support from the row-candidate pool. Hidden outputs are used only for evaluation and oracle diagnostics.

## Setup

- Run: `{run_name}`
- Dataset: public text-transformation tasks.
- Tasks: `{config.get('tasks')}`
- Visible examples per task: `{config.get('train_n')}`
- Held-out cap per task: `{config.get('heldout_cap')}`
- Max deterministic programs enumerated per task: `{config.get('max_candidates')}`
- Max candidate tables per task: `{config.get('max_tables')}`
- Pseudo-program support threshold for fallback: `{config.get('pseudo_min_vote_share')}`
- Row-candidate rows used: `{config.get('candidate_rows')}`

## Main Result

{md_table(summary, ['method', 'tasks', 'row_exact', 'full_task_exact'])}

## Interpretation

Candidate support changes strict full-task exactness by `{support_delta * 100:.1f}` points relative to direct greedy row inference and by `{support_vs_examples * 100:.1f}` points relative to the shortest train-fitting deterministic program. The shuffled-support control is separated by `{control_gap * 100:.1f}` points.

The pseudo-label crystallizer reaches `{pseudo * 100:.1f}%` as a pure table selector and `{pseudo_or_direct * 100:.1f}%` with conservative direct fallback. Its shuffled-pseudo control reaches `{shuffled_pseudo * 100:.1f}%`.

The deterministic train-fitting program oracle solves `{program_oracle * 100:.1f}%` of tasks and the row-candidate oracle solves `{row_oracle * 100:.1f}%`. Relative to the direct-to-program oracle gap, candidate-support gap capture is {captured_text}.

## Diagnostics

- Direct row inference solves `{int(task_df['direct_full_exact'].astype(bool).sum())}` of `{len(task_df)}` tasks.
- A deterministic train-fitting program can solve `{int(task_df['program_oracle'].astype(bool).sum())}` of `{len(task_df)}` tasks.
- Candidate-support selection solves `{int(task_df['support_full_exact'].astype(bool).sum())}` of `{len(task_df)}` tasks.
- Pseudo-label crystallization solves `{int(task_df['pseudo_full_exact'].astype(bool).sum())}` tasks as a pure selector and `{int(task_df['pseudo_or_direct_full_exact'].astype(bool).sum())}` with direct fallback.
- The direct fallback uses the pseudo-program table on `{int(task_df['pseudo_used_in_fallback'].astype(bool).sum())}` tasks.
- There are `{len(headroom)}` direct-missed tasks with a hidden-valid deterministic program; candidate support captures `{support_captured}` of them.
- Candidate support changes the selected program on `{len(changed)}` tasks: `{len(helped)}` helped and `{len(hurt)}` hurt on strict full-task exactness.

## Charts

![Full-task exact by method](../analysis/figures/method_full_task_exact.png)

![Row versus full-task exact](../analysis/figures/row_vs_full_task.png)

![Program headroom](../analysis/figures/program_headroom.png)

![Support selection changes](../analysis/figures/support_selection_changes.png)

![Support vote-share distribution](../analysis/figures/support_vote_share_distribution.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

## Family Breakdown

{md_table(family, ['method', 'family', 'tasks', 'row_exact', 'full_task_exact'], max_rows=160)}

## Reachable Headroom Tasks

{md_table(headroom.sort_values(['support_full_exact', 'support_avg_vote_share', 'task_id'], ascending=[True, False, True]), ['task_id', 'family', 'features', 'direct_full_exact', 'examples_full_exact', 'support_full_exact', 'shuffled_full_exact', 'support_avg_vote_share', 'train_match_count', 'support_program', 'oracle_program'], max_rows=80) if not headroom.empty else '_No direct-missed deterministic-program headroom tasks._'}

## Candidate-Support Changes

{md_table(changed.sort_values(['support_full_exact', 'examples_full_exact', 'task_id']), ['task_id', 'family', 'features', 'examples_full_exact', 'support_full_exact', 'support_avg_vote_share', 'support_supported_rows', 'examples_program', 'support_program'], max_rows=80) if not changed.empty else '_Candidate support selected the same program as the shortest train-fitting baseline for every task._'}

## Task Details

{md_table(task_df.sort_values(['pseudo_or_direct_full_exact', 'support_full_exact', 'program_oracle', 'task_id'], ascending=[True, True, False, True]), ['task_id', 'family', 'heldout_rows', 'train_match_count', 'candidate_tables', 'coherent_pseudo_tables', 'program_oracle', 'table_candidate_oracle', 'direct_full_exact', 'examples_full_exact', 'support_full_exact', 'pseudo_full_exact', 'pseudo_or_direct_full_exact', 'shuffled_pseudo_full_exact', 'pseudo_avg_vote_share'], max_rows=120)}

## Files

- `runs/{run_name}/row_candidates.csv`
- `runs/{run_name}/task_details.csv`
- `runs/{run_name}/method_details.csv`
- `runs/{run_name}/summary.csv`
- `analysis/summary.csv`
- `analysis/task_details.csv`
- `analysis/method_details.csv`
- `analysis/family_summary.csv`
"""
    (REPORTS / "qwen_noisy_row_program_crystallizer_report.md").write_text(report)
    (REPORTS / "qwen_noisy_row_program_crystallizer_report.html").write_text(markdown_to_html(report))


def write_outputs(run_dir: Path, run_name: str, config: Dict[str, Any], task_df: pd.DataFrame, method_df: pd.DataFrame, summary: pd.DataFrame, family: pd.DataFrame) -> None:
    task_df.to_csv(run_dir / "task_details.csv", index=False)
    method_df.to_csv(run_dir / "method_details.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    family.to_csv(run_dir / "family_summary.csv", index=False)
    for path, df in [
        (ANALYSIS / "task_details.csv", task_df),
        (ANALYSIS / "method_details.csv", method_df),
        (ANALYSIS / "summary.csv", summary),
        (ANALYSIS / "family_summary.csv", family),
    ]:
        df.to_csv(path, index=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    plot_method_bars(summary)
    plot_row_vs_full(summary)
    plot_oracle_decomposition(task_df)
    plot_support_changes(task_df)
    plot_family_heatmap(family)
    plot_support_distribution(task_df)
    write_reports(run_name, config, summary, family, task_df, method_df)


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(text.rstrip() + "\n")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = choose_tasks(load_tasks(min_examples=args.min_examples), args.task_limit, args.seed, args.train_n, args.heldout_cap, args.min_heldout)
    candidate_df = load_candidate_cache(Path(args.candidate_cache), tasks, run_dir)
    task_ids_with_candidates = set(candidate_df["task_id"].astype(str).unique())
    tasks = [t for t in tasks if t.task_id in task_ids_with_candidates]
    if not tasks:
        raise RuntimeError("No selected tasks have row candidates.")
    task_df, method_df = evaluate(tasks, candidate_df, args)
    summary, family = summarize(method_df)
    config = {
        **vars(args),
        "tasks": len(tasks),
        "candidate_rows": len(candidate_df),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_outputs(run_dir, args.run_name, config, task_df, method_df, summary, family)
    append_log(
        f"\n### Run `{args.run_name}`\n"
        f"- Tasks: {len(tasks)}\n"
        f"- Candidate rows: {len(candidate_df)}\n"
        f"- Candidate-support full-task exact: {float(summary[summary.method.eq('candidate_support_program')]['full_task_exact'].iloc[0]) * 100:.1f}%\n"
        f"- Pseudo-program table full-task exact: {float(summary[summary.method.eq('pseudo_program_table')]['full_task_exact'].iloc[0]) * 100:.1f}%\n"
        f"- Pseudo-program with direct fallback full-task exact: {float(summary[summary.method.eq('pseudo_program_or_direct')]['full_task_exact'].iloc[0]) * 100:.1f}%\n"
        f"- Direct greedy full-task exact: {float(summary[summary.method.eq('direct_row_greedy')]['full_task_exact'].iloc[0]) * 100:.1f}%\n"
        f"- Program oracle full-task exact: {float(summary[summary.method.eq('program_oracle')]['full_task_exact'].iloc[0]) * 100:.1f}%\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="main_v1")
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--task_limit", type=int, default=40)
    p.add_argument("--train_n", type=int, default=4)
    p.add_argument("--heldout_cap", type=int, default=6)
    p.add_argument("--min_examples", type=int, default=5)
    p.add_argument("--min_heldout", type=int, default=3)
    p.add_argument("--max_candidates", type=int, default=40000)
    p.add_argument("--max_tables", type=int, default=128)
    p.add_argument("--shuffled_max_tables", type=int, default=16)
    p.add_argument("--topk_per_row", type=int, default=4)
    p.add_argument("--pseudo_max_candidates", type=int, default=12000)
    p.add_argument("--shuffled_pseudo_max_candidates", type=int, default=3000)
    p.add_argument("--pseudo_min_vote_share", type=float, default=0.75)
    p.add_argument("--allow_pseudo_maps", action="store_true")
    p.add_argument("--candidate_cache", default="/workspace/large_artifacts/qwen_noisy_row_program_crystallizer/qwen_row_candidates_main40.csv")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

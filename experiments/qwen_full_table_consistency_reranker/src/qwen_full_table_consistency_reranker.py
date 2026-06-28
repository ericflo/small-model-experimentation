#!/usr/bin/env python3
"""Full-table consistency reranker for text transformation tasks.

This standalone experiment asks whether row-level model guesses can be made
task-consistent by ranking whole candidate output tables.

The script:
  * loads public Transformation.Text tasks,
  * generates multiple Qwen candidates per row,
  * enumerates full-table candidates,
  * measures oracle reachability,
  * trains a task-held-out logistic consistency scorer,
  * compares against direct row inference, majority vote, heuristic scoring,
    shuffled-label control, and oracle selectors,
  * writes CSVs, figures, Markdown, and HTML reports.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import itertools
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("/workspace/experiments/qwen_full_table_consistency_reranker")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_full_table_consistency_reranker")
SOURCE_BENCH_ROOT = Path("/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks")
BENCH_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = BENCH_ROOT / "Transformation.Text"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CACHE_DIR = Path("/workspace/.cache/huggingface")
MODEL_NAME = "Qwen/Qwen3-4B"


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


@dataclass
class TableCandidate:
    task_id: str
    candidate_id: str
    source: str
    outputs: Tuple[str, ...]
    row_sources: Tuple[str, ...]
    score_prior: float
    exact: bool
    row_exact: float
    features: Dict[str, float]


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


def norm_eq(a: Any, b: Any) -> bool:
    return clean(a) == clean(b)


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


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


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def parse_json_array(text: str, expected_len: int) -> List[str]:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)
    try:
        val = json.loads(text)
    except Exception:
        try:
            val = ast.literal_eval(text)
        except Exception:
            return [""] * expected_len
    if not isinstance(val, list):
        return [""] * expected_len
    out = [str(x) for x in val[:expected_len]]
    while len(out) < expected_len:
        out.append("")
    return out


def row_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], query: Tuple[str, ...], variant: str) -> str:
    if variant == "format":
        intro = [
            "Infer the exact text transformation and the exact output format from the examples.",
            "Return only the transformed output for the query, with no explanation.",
        ]
    elif variant == "consistency":
        intro = [
            "The examples define one deterministic transformation.",
            "Apply that same transformation to the query. Return only the output string.",
        ]
    else:
        intro = [
            "Infer the text transformation from the examples.",
            "Return only the transformed output for the query. Do not explain.",
        ]
    lines = intro + ["", "Examples:"]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.extend(["", "Query:", f"Input: {render_inputs(query)}", "Output:"])
    return "\n".join(lines)


def batch_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], queries: Sequence[Tuple[str, ...]], variant: str) -> str:
    if variant == "json":
        payload = {
            "examples": [{"input": render_inputs(inp), "output": out} for inp, out in train_pairs],
            "queries": [{"index": i, "input": render_inputs(query)} for i, query in enumerate(queries)],
        }
        return "\n".join(
            [
                "Infer the transformation from the examples in this JSON object.",
                "Apply exactly the same transformation to every query.",
                "Return only a valid JSON array of strings in query index order.",
                f"The array must contain exactly {len(queries)} strings.",
                "",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "",
                "JSON array only:",
            ]
        )
    lines = [
        "Infer one deterministic text transformation from the examples.",
        "Apply it to every query row.",
        "Return only a valid JSON array of strings in query order.",
        "",
        "Examples:",
    ]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.extend(["", "Queries:"])
    for i, query in enumerate(queries):
        lines.append(f"{i}: {render_inputs(query)}")
    lines.append("")
    lines.append("JSON array:")
    return "\n".join(lines)


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


def generate_text(
    tok: Any,
    model: Any,
    system: str,
    user: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: int,
) -> str:
    import torch

    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": 0.9})
    with torch.inference_mode():
        out = model.generate(**enc, **kwargs)
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def read_candidate_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["task_id", "row_index", "method", "prediction", "raw"])


def generate_candidates(
    tasks: Sequence[Task],
    train_map: Dict[str, List[Example]],
    test_map: Dict[str, List[Example]],
    run_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cache_path = run_dir / "row_candidates.csv"
    cached = read_candidate_cache(cache_path)
    have = set()
    for _, r in cached.iterrows():
        have.add((str(r.task_id), int(r.row_index), str(r.method)))
    planned: List[Tuple[Task, int, Tuple[str, ...], str, str]] = []
    for task in tasks:
        train_pairs = [(e.inputs, e.output) for e in train_map[task.task_id]]
        tests = test_map[task.task_id]
        for idx, ex in enumerate(tests):
            for method in ["row_greedy", "row_format", "row_consistency"]:
                if (task.task_id, idx, method) not in have:
                    planned.append((task, idx, ex.inputs, "row", method))
            for s in range(args.row_samples):
                method = f"row_sample{s}"
                if (task.task_id, idx, method) not in have:
                    planned.append((task, idx, ex.inputs, "row", method))
        for method in ["batch_plain", "batch_json"]:
            # Stored later as one row per query index.
            if any((task.task_id, idx, method) not in have for idx in range(len(tests))):
                planned.append((task, -1, tuple(), "batch", method))
    if planned and not args.no_qwen:
        tok, model = load_qwen()
        file_exists = cache_path.exists()
        with cache_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "row_index", "method", "prediction", "raw"])
            if not file_exists:
                writer.writeheader()
            done = 0
            for task, row_idx, query, kind, method in planned:
                train_pairs = [(e.inputs, e.output) for e in train_map[task.task_id]]
                tests = test_map[task.task_id]
                if kind == "row":
                    if method == "row_greedy":
                        variant, sample, temp, seed = "plain", False, 0.0, -1
                    elif method == "row_format":
                        variant, sample, temp, seed = "format", False, 0.0, -1
                    elif method == "row_consistency":
                        variant, sample, temp, seed = "consistency", False, 0.0, -1
                    else:
                        variant, sample, temp = "plain", True, args.temperature
                        seed = args.seed * 100000 + (hash(task.task_id) & 0xFFFF) * 31 + row_idx * 7 + int(method.replace("row_sample", ""))
                    raw = generate_text(
                        tok,
                        model,
                        "You transform text exactly from examples.",
                        row_prompt(train_pairs, query, variant),
                        args.row_max_new_tokens,
                        sample,
                        temp,
                        seed,
                    )
                    pred = clean_prediction(raw)
                    writer.writerow({"task_id": task.task_id, "row_index": row_idx, "method": method, "prediction": pred, "raw": raw})
                    done += 1
                else:
                    variant = "json" if method == "batch_json" else "plain"
                    raw = generate_text(
                        tok,
                        model,
                        "You transform text exactly from examples.",
                        batch_prompt(train_pairs, [e.inputs for e in tests], variant),
                        args.batch_max_new_tokens,
                        False,
                        0.0,
                        -1,
                    )
                    preds = parse_json_array(raw, len(tests))
                    for i, pred in enumerate(preds):
                        if (task.task_id, i, method) not in have:
                            writer.writerow({"task_id": task.task_id, "row_index": i, "method": method, "prediction": pred, "raw": raw})
                    done += 1
                if done == 1 or done % 50 == 0 or done == len(planned):
                    print(f"generated candidate calls {done}/{len(planned)}", flush=True)
                f.flush()
    elif planned and args.no_qwen:
        print(f"missing {len(planned)} candidate calls but --no_qwen set; using cached subset only", flush=True)
    return read_candidate_cache(cache_path)


def signature(s: str) -> str:
    chars = []
    for ch in str(s):
        if ch.isalpha():
            chars.append("A")
        elif ch.isdigit():
            chars.append("9")
        elif ch.isspace():
            chars.append("_")
        else:
            chars.append(ch)
    compressed = []
    for ch in chars:
        if not compressed or compressed[-1] != ch:
            compressed.append(ch)
    return "".join(compressed)


def charclass_counts(s: str) -> Tuple[int, int, int, int, int]:
    text = str(s)
    return (
        sum(c.isalpha() for c in text),
        sum(c.isdigit() for c in text),
        sum(c.isspace() for c in text),
        sum((not c.isalnum()) and (not c.isspace()) for c in text),
        len(text),
    )


def output_in_input(inp: Tuple[str, ...], out: str) -> bool:
    return any(clean(out) and clean(out).lower() in clean(x).lower() for x in inp)


def deterministic_explainer_exists(train: Sequence[Example], tests: Sequence[Example], outputs: Sequence[str]) -> bool:
    """Small consistency feature: can a simple extractor explain examples and candidate outputs?"""
    rows = [e.inputs for e in train] + [e.inputs for e in tests]
    ys = [e.output for e in train] + list(outputs)
    if not rows:
        return False
    num_cols = max(len(r) for r in rows)
    funcs = []
    for c in range(num_cols):
        funcs.extend(
            [
                lambda r, j=c: r[j] if j < len(r) else "",
                lambda r, j=c: clean(r[j] if j < len(r) else "").lower(),
                lambda r, j=c: clean(r[j] if j < len(r) else "").upper(),
                lambda r, j=c: "".join(re.findall(r"\d+", r[j] if j < len(r) else "")),
                lambda r, j=c: " ".join(re.findall(r"[A-Za-z]+", r[j] if j < len(r) else "")),
                lambda r, j=c: (re.findall(r"[A-Za-z0-9]+", r[j] if j < len(r) else "") or [""])[0],
                lambda r, j=c: (re.findall(r"[A-Za-z0-9]+", r[j] if j < len(r) else "") or [""])[-1],
            ]
        )
    for fn in funcs:
        try:
            if all(clean(fn(r)) == clean(y) for r, y in zip(rows, ys)):
                return True
        except Exception:
            pass
    return False


def candidate_row_sets(task: Task, candidate_df: pd.DataFrame, test: Sequence[Example]) -> List[List[Tuple[str, str, int]]]:
    rows: List[List[Tuple[str, str, int]]] = []
    sub = candidate_df[candidate_df["task_id"].eq(task.task_id)]
    for i in range(len(test)):
        rs = sub[sub["row_index"].astype(int).eq(i)]
        counts: Counter[str] = Counter()
        first_method: Dict[str, str] = {}
        for _, r in rs.iterrows():
            pred = clean(r["prediction"])
            if pred == "":
                continue
            counts[pred] += 1
            first_method.setdefault(pred, str(r["method"]))
        ranked = [(pred, first_method[pred], count) for pred, count in counts.most_common()]
        rows.append(ranked)
    return rows


def method_table(task: Task, candidate_df: pd.DataFrame, method: str, n_rows: int) -> Optional[Tuple[str, ...]]:
    sub = candidate_df[candidate_df["task_id"].eq(task.task_id) & candidate_df["method"].eq(method)]
    if sub.empty:
        return None
    vals = []
    for i in range(n_rows):
        hit = sub[sub["row_index"].astype(int).eq(i)]
        if hit.empty:
            return None
        vals.append(clean(hit.iloc[0]["prediction"]))
    return tuple(vals)


def enumerate_table_candidates(
    task: Task,
    train: Sequence[Example],
    test: Sequence[Example],
    candidate_df: pd.DataFrame,
    targets: Sequence[str],
    max_tables: int,
) -> List[TableCandidate]:
    row_sets = candidate_row_sets(task, candidate_df, test)
    candidates: Dict[Tuple[Tuple[str, ...], str], Tuple[str, Tuple[str, ...], float]] = {}
    for method in ["row_greedy", "row_format", "row_consistency", "batch_plain", "batch_json"]:
        table = method_table(task, candidate_df, method, len(test))
        if table is not None:
            candidates.setdefault((table, method), (method, tuple([method] * len(test)), 10.0))
    for s in range(8):
        method = f"row_sample{s}"
        table = method_table(task, candidate_df, method, len(test))
        if table is not None:
            candidates.setdefault((table, method), (method, tuple([method] * len(test)), 5.0))
    # Majority vote table.
    majority = []
    majority_sources = []
    for rs in row_sets:
        if rs:
            majority.append(rs[0][0])
            majority_sources.append(f"vote:{rs[0][1]}")
        else:
            majority.append("")
            majority_sources.append("missing")
    candidates.setdefault((tuple(majority), "row_majority"), ("row_majority", tuple(majority_sources), 8.0))

    # Frequency-ranked combinations. Exhaustive if small, otherwise beam by row vote counts.
    if all(rs for rs in row_sets):
        product = 1
        for rs in row_sets:
            product *= min(len(rs), 5)
        combos: Iterable[Tuple[Tuple[str, str, int], ...]]
        if product <= max_tables:
            combos = itertools.product(*[rs[:5] for rs in row_sets])
        else:
            # Build top combinations with a small beam.
            beam: List[Tuple[float, List[Tuple[str, str, int]]]] = [(0.0, [])]
            for rs in row_sets:
                new = []
                total = sum(c for _, _, c in rs) or 1
                for score, prefix in beam:
                    for pred, method, count in rs[:5]:
                        new.append((score + math.log((count + 0.5) / total), prefix + [(pred, method, count)]))
                new.sort(key=lambda x: x[0], reverse=True)
                beam = new[: max(32, max_tables)]
            combos = [tuple(prefix) for _, prefix in beam[:max_tables]]
        for combo in combos:
            outs = tuple(x[0] for x in combo)
            srcs = tuple(x[1] for x in combo)
            prior = sum(x[2] for x in combo) / max(1, len(combo))
            candidates.setdefault((outs, "row_combo"), ("row_combo", srcs, prior))
            if len(candidates) >= max_tables:
                break

    out: List[TableCandidate] = []
    for idx, ((outs, _source_key), (source, row_sources, prior)) in enumerate(candidates.items()):
        row_exact = sum(norm_eq(p, t) for p, t in zip(outs, targets)) / max(1, len(targets))
        exact = bool(len(outs) == len(targets) and all(norm_eq(p, t) for p, t in zip(outs, targets)))
        features = table_features(task, train, test, outs, row_sets, source, prior)
        out.append(
            TableCandidate(
                task_id=task.task_id,
                candidate_id=f"{task.task_id}::cand{idx:04d}",
                source=source,
                outputs=outs,
                row_sources=row_sources,
                score_prior=prior,
                exact=exact,
                row_exact=row_exact,
                features=features,
            )
        )
    return out


def table_features(
    task: Task,
    train: Sequence[Example],
    test: Sequence[Example],
    outputs: Sequence[str],
    row_sets: Sequence[Sequence[Tuple[str, str, int]]],
    source: str,
    prior: float,
) -> Dict[str, float]:
    train_out = [e.output for e in train]
    all_out = list(outputs)
    lengths = [len(x) for x in all_out]
    train_lengths = [len(x) for x in train_out]
    sig_train = Counter(signature(x) for x in train_out)
    sig_mode = sig_train.most_common(1)[0][0] if sig_train else ""
    row_vote_shares = []
    row_vote_ranks = []
    for pred, rs in zip(outputs, row_sets):
        total = sum(c for _, _, c in rs) or 1
        found = False
        for rank, (cand, _, count) in enumerate(rs, start=1):
            if clean(cand) == clean(pred):
                row_vote_shares.append(count / total)
                row_vote_ranks.append(rank)
                found = True
                break
        if not found:
            row_vote_shares.append(0.0)
            row_vote_ranks.append(99.0)
    out_counts = [charclass_counts(x) for x in all_out]
    train_counts = [charclass_counts(x) for x in train_out]
    def avg_at(vals: Sequence[Tuple[int, int, int, int, int]], idx: int) -> float:
        return float(np.mean([v[idx] for v in vals])) if vals else 0.0
    in_input = [output_in_input(e.inputs, out) for e, out in zip(test, outputs)]
    train_in_input = [output_in_input(e.inputs, e.output) for e in train]
    return {
        "n_rows": float(len(outputs)),
        "source_row_combo": float(source == "row_combo"),
        "source_majority": float(source == "row_majority"),
        "source_batch": float(source.startswith("batch")),
        "source_greedy": float(source == "row_greedy"),
        "prior": float(prior),
        "avg_vote_share": float(np.mean(row_vote_shares)) if row_vote_shares else 0.0,
        "min_vote_share": float(np.min(row_vote_shares)) if row_vote_shares else 0.0,
        "avg_vote_rank": float(np.mean(row_vote_ranks)) if row_vote_ranks else 99.0,
        "unique_output_rate": len(set(clean(x) for x in outputs)) / max(1, len(outputs)),
        "empty_rate": sum(clean(x) == "" for x in outputs) / max(1, len(outputs)),
        "avg_len": float(np.mean(lengths)) if lengths else 0.0,
        "std_len": float(np.std(lengths)) if lengths else 0.0,
        "train_avg_len": float(np.mean(train_lengths)) if train_lengths else 0.0,
        "len_delta_abs": abs((float(np.mean(lengths)) if lengths else 0.0) - (float(np.mean(train_lengths)) if train_lengths else 0.0)),
        "signature_mode_match_rate": sum(signature(x) == sig_mode for x in outputs) / max(1, len(outputs)),
        "signature_diversity": len(set(signature(x) for x in outputs)) / max(1, len(outputs)),
        "alpha_delta": abs(avg_at(out_counts, 0) - avg_at(train_counts, 0)),
        "digit_delta": abs(avg_at(out_counts, 1) - avg_at(train_counts, 1)),
        "punct_delta": abs(avg_at(out_counts, 3) - avg_at(train_counts, 3)),
        "in_input_rate": sum(in_input) / max(1, len(in_input)),
        "train_in_input_rate": sum(train_in_input) / max(1, len(train_in_input)),
        "in_input_delta": abs((sum(in_input) / max(1, len(in_input))) - (sum(train_in_input) / max(1, len(train_in_input)))),
        "simple_explainer": float(deterministic_explainer_exists(train, test, outputs)),
    }


def build_candidate_tables(
    tasks: Sequence[Task],
    train_map: Dict[str, List[Example]],
    test_map: Dict[str, List[Example]],
    candidate_df: pd.DataFrame,
    max_tables: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table_rows = []
    oracle_rows = []
    for task in tasks:
        train = train_map[task.task_id]
        test = test_map[task.task_id]
        targets = [e.output for e in test]
        row_sets = candidate_row_sets(task, candidate_df, test)
        row_oracle = bool(row_sets and all(any(norm_eq(pred, target) for pred, _, _ in rs) for rs, target in zip(row_sets, targets)))
        cands = enumerate_table_candidates(task, train, test, candidate_df, targets, max_tables)
        table_oracle = any(c.exact for c in cands)
        direct = next((c for c in cands if c.source == "row_greedy"), None)
        majority = next((c for c in cands if c.source == "row_majority"), None)
        oracle_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "heldout_rows": len(test),
                "row_candidate_oracle": row_oracle,
                "table_candidate_oracle": table_oracle,
                "candidate_tables": len(cands),
                "row_candidate_median": float(np.median([len(rs) for rs in row_sets])) if row_sets else 0.0,
                "direct_row_exact": direct.row_exact if direct else 0.0,
                "direct_full_exact": direct.exact if direct else False,
                "majority_row_exact": majority.row_exact if majority else 0.0,
                "majority_full_exact": majority.exact if majority else False,
            }
        )
        for c in cands:
            row = {
                "task_id": c.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "candidate_id": c.candidate_id,
                "source": c.source,
                "outputs_json": json.dumps(c.outputs, ensure_ascii=False),
                "row_sources_json": json.dumps(c.row_sources),
                "exact": c.exact,
                "row_exact": c.row_exact,
            }
            row.update(c.features)
            table_rows.append(row)
    return pd.DataFrame(table_rows), pd.DataFrame(oracle_rows)


def feature_columns(table_df: pd.DataFrame) -> List[str]:
    excluded = {"task_id", "family", "features", "candidate_id", "source", "outputs_json", "row_sources_json", "exact", "row_exact"}
    cols = [c for c in table_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(table_df[c])]
    return cols


def select_by_score(table_df: pd.DataFrame, score_col: str, method: str) -> pd.DataFrame:
    rows = []
    for task_id, sub in table_df.groupby("task_id"):
        best = sub.sort_values([score_col, "avg_vote_share", "prior"], ascending=[False, False, False]).iloc[0]
        rows.append(
            {
                "task_id": task_id,
                "method": method,
                "candidate_id": best["candidate_id"],
                "source": best["source"],
                "row_exact": best["row_exact"],
                "full_task_exact": bool(best["exact"]),
                "score": best[score_col],
                "outputs_json": best["outputs_json"],
            }
        )
    return pd.DataFrame(rows)


def run_rerankers(table_df: pd.DataFrame, oracle_df: pd.DataFrame, seed: int, folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = feature_columns(table_df)
    table_df = table_df.copy()
    # Heuristic: interpretable consistency score, deliberately simple.
    table_df["heuristic_score"] = (
        1.5 * table_df["avg_vote_share"]
        + 0.8 * table_df["signature_mode_match_rate"]
        + 0.8 * table_df["simple_explainer"]
        - 0.02 * table_df["len_delta_abs"]
        - 0.4 * table_df["empty_rate"]
        - 0.1 * table_df["avg_vote_rank"]
    )
    selected = [select_by_score(table_df, "heuristic_score", "heuristic")]

    # Direct and majority selectors are selected from their source rows.
    for source, method in [("row_greedy", "direct_row_greedy"), ("row_majority", "row_majority")]:
        rows = []
        for task_id, sub in table_df.groupby("task_id"):
            hit = sub[sub["source"].eq(source)]
            if hit.empty:
                best = sub.iloc[0]
            else:
                best = hit.iloc[0]
            rows.append(
                {
                    "task_id": task_id,
                    "method": method,
                    "candidate_id": best["candidate_id"],
                    "source": best["source"],
                    "row_exact": best["row_exact"],
                    "full_task_exact": bool(best["exact"]),
                    "score": 0.0,
                    "outputs_json": best["outputs_json"],
                }
            )
        selected.append(pd.DataFrame(rows))

    # Oracle table selector.
    rows = []
    for task_id, sub in table_df.groupby("task_id"):
        exact = sub[sub["exact"].astype(bool)]
        best = exact.iloc[0] if not exact.empty else sub.sort_values("row_exact", ascending=False).iloc[0]
        rows.append(
            {
                "task_id": task_id,
                "method": "table_oracle",
                "candidate_id": best["candidate_id"],
                "source": best["source"],
                "row_exact": best["row_exact"],
                "full_task_exact": bool(best["exact"]),
                "score": 1.0,
                "outputs_json": best["outputs_json"],
            }
        )
    selected.append(pd.DataFrame(rows))

    task_ids = sorted(table_df["task_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    fold_of = {tid: i % folds for i, tid in enumerate(task_ids)}
    table_df["fold"] = table_df["task_id"].map(fold_of)
    model_rows = []
    diagnostics = []
    for fold in range(folds):
        train = table_df[table_df["fold"].ne(fold)].copy()
        test = table_df[table_df["fold"].eq(fold)].copy()
        X_train = train[feature_cols].fillna(0).to_numpy(dtype=float)
        y_train = train["exact"].astype(int).to_numpy()
        X_test = test[feature_cols].fillna(0).to_numpy(dtype=float)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        # If a fold has no positives in train, fall back to heuristic.
        if len(set(y_train)) < 2:
            test["learned_score"] = test["heuristic_score"]
            test["shuffled_score"] = test["heuristic_score"]
            auc = float("nan")
            shuf_auc = float("nan")
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + fold)
            clf.fit(X_train_s, y_train)
            test["learned_score"] = clf.predict_proba(X_test_s)[:, 1]
            y_shuf = y_train.copy()
            np_rng = np.random.default_rng(seed + 1000 + fold)
            np_rng.shuffle(y_shuf)
            if len(set(y_shuf)) < 2:
                test["shuffled_score"] = test["heuristic_score"]
            else:
                shuf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + 5000 + fold)
                shuf.fit(X_train_s, y_shuf)
                test["shuffled_score"] = shuf.predict_proba(X_test_s)[:, 1]
            try:
                auc = roc_auc_score(test["exact"].astype(int), test["learned_score"])
            except Exception:
                auc = float("nan")
            try:
                shuf_auc = roc_auc_score(test["exact"].astype(int), test["shuffled_score"])
            except Exception:
                shuf_auc = float("nan")
        diagnostics.append(
            {
                "fold": fold,
                "train_tasks": train["task_id"].nunique(),
                "test_tasks": test["task_id"].nunique(),
                "train_candidates": len(train),
                "test_candidates": len(test),
                "train_positive_rate": float(train["exact"].mean()),
                "test_positive_rate": float(test["exact"].mean()),
                "candidate_auc": auc,
                "shuffled_candidate_auc": shuf_auc,
            }
        )
        model_rows.append(test)
    scored = pd.concat(model_rows, ignore_index=True)
    selected.append(select_by_score(scored, "learned_score", "learned_reranker"))
    selected.append(select_by_score(scored, "shuffled_score", "shuffled_label_reranker"))
    selected_df = pd.concat(selected, ignore_index=True)
    selected_df = selected_df.merge(oracle_df[["task_id", "family", "features", "heldout_rows", "row_candidate_oracle", "table_candidate_oracle", "candidate_tables", "row_candidate_median"]], on="task_id", how="left")
    return selected_df, pd.DataFrame(diagnostics)


def summarize(selected_df: pd.DataFrame, oracle_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        selected_df.groupby("method", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            row_exact=("row_exact", "mean"),
            full_task_exact=("full_task_exact", "mean"),
            table_oracle_rate=("table_candidate_oracle", "mean"),
            row_oracle_rate=("row_candidate_oracle", "mean"),
            median_candidate_tables=("candidate_tables", "median"),
        )
        .sort_values("full_task_exact", ascending=False)
    )
    family = (
        selected_df.groupby(["method", "family"], as_index=False)
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
            if "exact" in c or "rate" in c or "oracle" in c or "auc" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def plot_summary(summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("full_task_exact")["method"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    vals = [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order]
    ax.barh(order, vals, color="#2563eb")
    ax.set_xlabel("Full-task exact (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Full-table selection accuracy")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "full_task_by_method.png", dpi=160)
    plt.close(fig)


def plot_row_vs_task(summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("row_exact")["method"])
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(list(x), [float(summary[summary.method.eq(m)]["row_exact"].iloc[0]) * 100 for m in order], marker="o", label="row exact", color="#059669")
    ax.plot(list(x), [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], marker="o", label="full-task exact", color="#dc2626")
    ax.set_xticks(list(x), order, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Row accuracy versus full-task consistency")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "row_vs_task.png", dpi=160)
    plt.close(fig)


def plot_oracles(oracle_df: pd.DataFrame) -> None:
    vals = {
        "direct": float(oracle_df["direct_full_exact"].mean()),
        "row oracle": float(oracle_df["row_candidate_oracle"].mean()),
        "table oracle": float(oracle_df["table_candidate_oracle"].mean()),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(vals.keys(), [v * 100 for v in vals.values()], color=["#6b7280", "#f59e0b", "#10b981"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Tasks (%)")
    ax.set_title("Reachability from generated candidates")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "oracle_reachability.png", dpi=160)
    plt.close(fig)


def plot_candidate_counts(oracle_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(oracle_df["candidate_tables"], bins=20, color="#7c3aed")
    ax.set_xlabel("Candidate tables per task")
    ax.set_ylabel("Tasks")
    ax.set_title("Enumerated candidate-table count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "candidate_table_counts.png", dpi=160)
    plt.close(fig)


def plot_fold_auc(fold_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(fold_df))
    ax.bar(x - 0.18, fold_df["candidate_auc"].fillna(0) * 100, width=0.36, label="learned", color="#2563eb")
    ax.bar(x + 0.18, fold_df["shuffled_candidate_auc"].fillna(0) * 100, width=0.36, label="shuffled labels", color="#ef4444")
    ax.set_xticks(x, fold_df["fold"].astype(str))
    ax.set_ylim(0, 105)
    ax.set_ylabel("Candidate AUC (%)")
    ax.set_xlabel("Held-out fold")
    ax.set_title("Reranker candidate discrimination")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fold_auc.png", dpi=160)
    plt.close(fig)


def plot_headroom(oracle_df: pd.DataFrame) -> None:
    direct = oracle_df["direct_full_exact"].astype(bool)
    oracle = oracle_df["table_candidate_oracle"].astype(bool)
    vals = {
        "direct solved": int((direct & oracle).sum()),
        "oracle headroom": int((~direct & oracle).sum()),
        "unreachable": int((~oracle).sum()),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(vals.keys(), vals.values(), color=["#059669", "#f59e0b", "#dc2626"])
    ax.set_ylabel("Tasks")
    ax.set_title("Direct solution, table-oracle headroom, and unreachable tasks")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "oracle_headroom.png", dpi=160)
    plt.close(fig)


def plot_selection_changes(selected: pd.DataFrame) -> None:
    direct = selected[selected["method"].eq("direct_row_greedy")][["task_id", "full_task_exact", "outputs_json"]].rename(
        columns={"full_task_exact": "direct_full", "outputs_json": "direct_outputs"}
    )
    learned = selected[selected["method"].eq("learned_reranker")][["task_id", "full_task_exact", "outputs_json"]].rename(
        columns={"full_task_exact": "learned_full", "outputs_json": "learned_outputs"}
    )
    if direct.empty or learned.empty:
        return
    merged = direct.merge(learned, on="task_id")
    changed = merged["direct_outputs"].ne(merged["learned_outputs"])
    vals = {
        "fallback/tied output": int((~changed).sum()),
        "changed helped": int((changed & (merged["learned_full"].astype(float) > merged["direct_full"].astype(float))).sum()),
        "changed tied": int((changed & (merged["learned_full"].astype(float) == merged["direct_full"].astype(float))).sum()),
        "changed hurt": int((changed & (merged["learned_full"].astype(float) < merged["direct_full"].astype(float))).sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(vals.keys(), vals.values(), color=["#9ca3af", "#059669", "#2563eb", "#dc2626"])
    ax.set_ylabel("Tasks")
    ax.set_title("Learned-reranker changes relative to direct greedy")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "learned_selection_changes.png", dpi=160)
    plt.close(fig)


def plot_family_heatmap(family: pd.DataFrame) -> None:
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
    return "<!doctype html><html><head><meta charset='utf-8'><title>Full-Table Consistency Reranker</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>"


def write_reports(
    run_name: str,
    config: Dict[str, Any],
    summary: pd.DataFrame,
    family: pd.DataFrame,
    selected: pd.DataFrame,
    oracle_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    direct = summary[summary["method"].eq("direct_row_greedy")]
    learned = summary[summary["method"].eq("learned_reranker")]
    table_oracle = summary[summary["method"].eq("table_oracle")]
    shuffled = summary[summary["method"].eq("shuffled_label_reranker")]
    direct_full = float(direct["full_task_exact"].iloc[0]) if not direct.empty else float("nan")
    learned_full = float(learned["full_task_exact"].iloc[0]) if not learned.empty else float("nan")
    oracle_full = float(table_oracle["full_task_exact"].iloc[0]) if not table_oracle.empty else float("nan")
    shuf_full = float(shuffled["full_task_exact"].iloc[0]) if not shuffled.empty else float("nan")
    row_oracle = float(oracle_df["row_candidate_oracle"].mean()) if not oracle_df.empty else float("nan")
    enum_oracle = float(oracle_df["table_candidate_oracle"].mean()) if not oracle_df.empty else float("nan")
    if not math.isnan(learned_full) and not math.isnan(direct_full):
        learned_delta = learned_full - direct_full
    else:
        learned_delta = float("nan")
    if not math.isnan(learned_full) and not math.isnan(shuf_full):
        control_delta = learned_full - shuf_full
    else:
        control_delta = float("nan")
    direct_sel = selected[selected["method"].eq("direct_row_greedy")][["task_id", "full_task_exact", "row_exact", "outputs_json"]].rename(
        columns={"full_task_exact": "direct_full_task_exact", "row_exact": "direct_row_exact", "outputs_json": "direct_outputs"}
    )
    learned_sel = selected[selected["method"].eq("learned_reranker")][["task_id", "source", "full_task_exact", "row_exact", "score", "outputs_json"]].rename(
        columns={"source": "learned_source", "full_task_exact": "learned_full_task_exact", "row_exact": "learned_row_exact", "outputs_json": "learned_outputs"}
    )
    oracle_sel = selected[selected["method"].eq("table_oracle")][["task_id", "full_task_exact", "row_exact", "source"]].rename(
        columns={"full_task_exact": "oracle_full_task_exact", "row_exact": "oracle_row_exact", "source": "oracle_source"}
    )
    comparison = direct_sel.merge(learned_sel, on="task_id").merge(oracle_sel, on="task_id").merge(
        oracle_df[["task_id", "family", "features", "candidate_tables", "row_candidate_median", "table_candidate_oracle"]],
        on="task_id",
        how="left",
    )
    comparison["learned_changed_output"] = comparison["direct_outputs"].ne(comparison["learned_outputs"])
    direct_solved = int(comparison["direct_full_task_exact"].astype(bool).sum()) if not comparison.empty else 0
    oracle_solved = int(comparison["oracle_full_task_exact"].astype(bool).sum()) if not comparison.empty else 0
    learned_solved = int(comparison["learned_full_task_exact"].astype(bool).sum()) if not comparison.empty else 0
    oracle_headroom = comparison[(~comparison["direct_full_task_exact"].astype(bool)) & comparison["oracle_full_task_exact"].astype(bool)] if not comparison.empty else pd.DataFrame()
    learned_headroom_capture = int((oracle_headroom["learned_full_task_exact"].astype(bool)).sum()) if not oracle_headroom.empty else 0
    changed = comparison[comparison["learned_changed_output"].astype(bool)] if not comparison.empty else pd.DataFrame()
    changed_helped = int((changed["learned_full_task_exact"].astype(float) > changed["direct_full_task_exact"].astype(float)).sum()) if not changed.empty else 0
    changed_hurt = int((changed["learned_full_task_exact"].astype(float) < changed["direct_full_task_exact"].astype(float)).sum()) if not changed.empty else 0
    changed_tied = int((changed["learned_full_task_exact"].astype(float) == changed["direct_full_task_exact"].astype(float)).sum()) if not changed.empty else 0
    auc_mean = float(fold_df["candidate_auc"].mean()) if not fold_df.empty else float("nan")
    shuf_auc_mean = float(fold_df["shuffled_candidate_auc"].mean()) if not fold_df.empty else float("nan")
    interp = [
        f"The generated row-candidate sets make the exact table reachable on {enum_oracle*100:.1f}% of tasks after enumeration, while the per-row oracle is {row_oracle*100:.1f}%.",
        f"The learned reranker changes full-task exact by {learned_delta*100:.1f} points relative to direct greedy row inference.",
        f"The learned reranker is separated from the shuffled-label control by {control_delta*100:.1f} points.",
        f"Mean candidate-level AUC is {auc_mean*100:.1f}% versus {shuf_auc_mean*100:.1f}% for shuffled labels, but that discrimination does not translate into task-level headroom capture.",
    ]
    report = f"""# Full-Table Consistency Reranker

## Question

Can multiple row-level model guesses be converted into task-level consistency by selecting an entire candidate output table?

The experiment generates several candidate outputs for each held-out row, enumerates full-table candidates, and trains a task-held-out consistency scorer to choose one table. The primary metric is strict full-task exactness: every held-out row for a task must be correct.

## Setup

- Benchmark root: `{BENCH_ROOT}`
- Run: `{run_name}`
- Tasks: {config.get('tasks')}
- Train rows per task: {config.get('train_n')}
- Held-out cap per task: {config.get('heldout_cap')}
- Row samples per row: {config.get('row_samples')}
- Max candidate tables per task: {config.get('max_tables')}
- Cross-validation folds: {config.get('folds')}

## Main Result

{md_table(summary, ['method', 'tasks', 'row_exact', 'full_task_exact', 'table_oracle_rate', 'row_oracle_rate', 'median_candidate_tables'])}

## Interpretation

{" ".join(interp)}

If oracle reachability is high but the learned reranker does not improve over direct row inference, the bottleneck is table selection. If oracle reachability is low, the bottleneck is generation diversity.

## Diagnostic Findings

- Direct greedy solves {direct_solved} of {len(comparison)} tasks.
- The table oracle solves {oracle_solved} of {len(comparison)} tasks, leaving {len(oracle_headroom)} tasks of reachable headroom beyond direct greedy.
- The learned reranker solves {learned_solved} of {len(comparison)} tasks and captures {learned_headroom_capture} of the {len(oracle_headroom)} reachable-headroom tasks.
- The learned reranker changes the selected output table on {len(changed)} tasks: {changed_helped} helped, {changed_tied} tied, and {changed_hurt} hurt on strict full-task exactness.

## Charts

![Full-task exact by method](../analysis/figures/full_task_by_method.png)

![Row versus task accuracy](../analysis/figures/row_vs_task.png)

![Oracle reachability](../analysis/figures/oracle_reachability.png)

![Oracle headroom](../analysis/figures/oracle_headroom.png)

![Candidate table counts](../analysis/figures/candidate_table_counts.png)

![Fold AUC](../analysis/figures/fold_auc.png)

![Learned selection changes](../analysis/figures/learned_selection_changes.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

## Fold Diagnostics

{md_table(fold_df, ['fold', 'train_tasks', 'test_tasks', 'train_candidates', 'test_candidates', 'train_positive_rate', 'test_positive_rate', 'candidate_auc', 'shuffled_candidate_auc'])}

## Family Breakdown

{md_table(family, ['method', 'family', 'tasks', 'row_exact', 'full_task_exact'], max_rows=140)}

## Task-Level Reachability

{md_table(oracle_df.sort_values(['table_candidate_oracle', 'row_candidate_oracle', 'direct_row_exact'], ascending=[True, True, False]), ['task_id', 'family', 'heldout_rows', 'direct_row_exact', 'direct_full_exact', 'row_candidate_oracle', 'table_candidate_oracle', 'candidate_tables', 'row_candidate_median'], max_rows=80)}

## Reachable Headroom Tasks

{md_table(oracle_headroom.sort_values(['learned_full_task_exact', 'direct_row_exact']), ['task_id', 'family', 'features', 'direct_row_exact', 'direct_full_task_exact', 'learned_row_exact', 'learned_full_task_exact', 'oracle_row_exact', 'oracle_full_task_exact', 'learned_source', 'oracle_source', 'candidate_tables'], max_rows=80) if not oracle_headroom.empty else "_No reachable headroom tasks._"}

## Learned Selection Changes

{md_table(changed.sort_values(['learned_full_task_exact', 'direct_full_task_exact', 'task_id']), ['task_id', 'family', 'direct_row_exact', 'learned_row_exact', 'direct_full_task_exact', 'learned_full_task_exact', 'learned_source', 'score', 'candidate_tables', 'table_candidate_oracle'], max_rows=80) if not changed.empty else "_The learned reranker selected the same output table as direct greedy for every task._"}

## Selected Tables

{md_table(selected.sort_values(['method', 'task_id']), ['task_id', 'method', 'source', 'row_exact', 'full_task_exact', 'score', 'candidate_tables', 'table_candidate_oracle'], max_rows=120)}

## Files

- `runs/{run_name}/row_candidates.csv`
- `runs/{run_name}/table_candidates.csv`
- `runs/{run_name}/selected_tables.csv`
- `runs/{run_name}/oracle_summary.csv`
- `runs/{run_name}/fold_diagnostics.csv`
- `analysis/summary.csv`
- `analysis/family_summary.csv`
- `analysis/selected_tables.csv`
- `analysis/oracle_summary.csv`
"""
    (REPORTS / "qwen_full_table_consistency_reranker_report.md").write_text(report)
    (REPORTS / "qwen_full_table_consistency_reranker_report.html").write_text(markdown_to_html(report))


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    tasks = choose_tasks(load_tasks(min_examples=args.min_examples), args.task_limit, args.seed, args.train_n, args.heldout_cap, args.min_heldout)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_map: Dict[str, List[Example]] = {}
    test_map: Dict[str, List[Example]] = {}
    for task in tasks:
        train, test = split_examples(task, args.train_n, args.heldout_cap)
        train_map[task.task_id] = train
        test_map[task.task_id] = test
    candidate_df = generate_candidates(tasks, train_map, test_map, run_dir, args)
    table_df, oracle_df = build_candidate_tables(tasks, train_map, test_map, candidate_df, args.max_tables)
    selected, fold_df = run_rerankers(table_df, oracle_df, args.seed, args.folds)
    summary, family = summarize(selected, oracle_df)

    table_df.to_csv(run_dir / "table_candidates.csv", index=False)
    oracle_df.to_csv(run_dir / "oracle_summary.csv", index=False)
    selected.to_csv(run_dir / "selected_tables.csv", index=False)
    fold_df.to_csv(run_dir / "fold_diagnostics.csv", index=False)
    summary.to_csv(ANALYSIS / "summary.csv", index=False)
    family.to_csv(ANALYSIS / "family_summary.csv", index=False)
    selected.to_csv(ANALYSIS / "selected_tables.csv", index=False)
    oracle_df.to_csv(ANALYSIS / "oracle_summary.csv", index=False)
    table_df.to_csv(ANALYSIS / "table_candidates.csv", index=False)
    config = vars(args).copy()
    config["tasks"] = len(tasks)
    config["created_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    plot_summary(summary)
    plot_row_vs_task(summary)
    plot_oracles(oracle_df)
    plot_headroom(oracle_df)
    plot_candidate_counts(oracle_df)
    plot_fold_auc(fold_df)
    plot_selection_changes(selected)
    plot_family_heatmap(family)
    write_reports(args.run_name, config, summary, family, selected, oracle_df, fold_df)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="main")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--task_limit", type=int, default=40)
    ap.add_argument("--train_n", type=int, default=4)
    ap.add_argument("--heldout_cap", type=int, default=6)
    ap.add_argument("--min_examples", type=int, default=5)
    ap.add_argument("--min_heldout", type=int, default=3)
    ap.add_argument("--row_samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--row_max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_max_new_tokens", type=int, default=420)
    ap.add_argument("--max_tables", type=int, default=512)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--no_qwen", action="store_true", help="Use cached row candidates only.")
    return ap.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

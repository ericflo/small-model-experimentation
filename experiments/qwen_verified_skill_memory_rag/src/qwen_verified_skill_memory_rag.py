#!/usr/bin/env python3
"""Verified skill memory retrieval experiment.

This standalone experiment tests whether analogous verified transformation
examples help Qwen solve public text-transformation tasks more consistently.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_verified_skill_memory_rag")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_verified_skill_memory_rag")
BENCH_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = BENCH_ROOT / "Transformation.Text"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CACHE_DIR = Path("/workspace/.cache/huggingface")
MODEL_NAME = "Qwen/Qwen3-4B"
SOURCE_BENCH = Path("/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks")


@dataclass(frozen=True)
class Example:
    inputs: Tuple[str, ...]
    output: str


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    features: Tuple[str, ...]
    examples: Tuple[Example, ...]
    source_path: str


def ensure_dirs() -> None:
    for p in [ROOT, LARGE_ROOT, RUNS, ANALYSIS, FIGURES, REPORTS]:
        p.mkdir(parents=True, exist_ok=True)


def mirror_benchmark() -> None:
    if BENCH_ROOT.exists():
        return
    if SOURCE_BENCH.exists():
        BENCH_ROOT.symlink_to(SOURCE_BENCH, target_is_directory=True)
        return
    raise FileNotFoundError(f"Benchmark source not found: {SOURCE_BENCH}")


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


def load_tasks(min_examples: int = 5) -> List[Task]:
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
                features=tuple(str(x) for x in meta.get("Features", [])),
                examples=tuple(examples),
                source_path=str(d),
            )
        )
    return tasks


def split_examples(task: Task, train_n: int, heldout_cap: int) -> Tuple[List[Example], List[Example]]:
    k = min(train_n, max(2, len(task.examples) // 2))
    train = list(task.examples[:k])
    heldout = list(task.examples[k:])
    if heldout_cap > 0:
        heldout = heldout[:heldout_cap]
    return train, heldout


def choose_eval_tasks(tasks: List[Task], limit: int, seed: int, train_n: int, heldout_cap: int, min_heldout: int) -> List[Task]:
    eligible = []
    for task in tasks:
        _, heldout = split_examples(task, train_n, heldout_cap)
        if len(heldout) >= min_heldout:
            eligible.append(task)
    rng = random.Random(seed)
    if limit and limit < len(eligible):
        eligible = rng.sample(eligible, limit)
    return sorted(eligible, key=lambda t: t.task_id)


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def ngram_counter(text: str, min_n: int = 3, max_n: int = 5) -> Counter[str]:
    text = f" {clean_text(text)} "
    counts: Counter[str] = Counter()
    for n in range(min_n, max_n + 1):
        for i in range(max(0, len(text) - n + 1)):
            counts[text[i : i + n]] += 1
    return counts


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return float(dot / max(1e-12, na * nb))


def signature_from_examples(examples: Sequence[Example]) -> str:
    parts = []
    for ex in examples:
        parts.append(f"IN {render_inputs(ex.inputs)} OUT {ex.output}")
    return "\n".join(parts)


def build_memory(eval_tasks: Sequence[Task], all_tasks: Sequence[Task], memory_limit: int, seed: int) -> List[Task]:
    eval_ids = {t.task_id for t in eval_tasks}
    memory = [t for t in all_tasks if t.task_id not in eval_ids]
    if memory_limit and memory_limit < len(memory):
        rng = random.Random(seed + 17)
        memory = rng.sample(memory, memory_limit)
    return sorted(memory, key=lambda t: t.task_id)


def retrieve_skills(
    query_train: Sequence[Example],
    memory: Sequence[Task],
    memory_vectors: Dict[str, Counter[str]],
    top_k: int,
    card_examples: int,
) -> List[Tuple[Task, float]]:
    qvec = ngram_counter(signature_from_examples(query_train[:card_examples]))
    scored = [(task, cosine(qvec, memory_vectors[task.task_id])) for task in memory]
    scored.sort(key=lambda x: (-x[1], x[0].task_id))
    return scored[:top_k]


def random_skills(task_id: str, memory: Sequence[Task], top_k: int, seed: int) -> List[Tuple[Task, float]]:
    rng = random.Random(stable_int(f"{seed}:{task_id}:random"))
    sample = rng.sample(list(memory), min(top_k, len(memory)))
    return [(task, 0.0) for task in sample]


def render_skill_card(task: Task, card_examples: int, corrupt: bool, seed_text: str) -> str:
    examples = list(task.examples[:card_examples])
    if corrupt and len(examples) > 1:
        rng = random.Random(stable_int(seed_text))
        outputs = [e.output for e in examples]
        rng.shuffle(outputs)
        if all(out == ex.output for out, ex in zip(outputs, examples)):
            outputs = outputs[1:] + outputs[:1]
        examples = [Example(ex.inputs, out) for ex, out in zip(examples, outputs)]
    lines = ["Reference transformation:"]
    for ex in examples:
        lines.append(f"Input: {render_inputs(ex.inputs)}")
        lines.append(f"Output: {ex.output}")
    return "\n".join(lines)


def direct_row_prompt(train: Sequence[Example], query: Example) -> str:
    lines = [
        "Infer the text transformation from the examples.",
        "Return only the transformed output for the query. Do not explain.",
        "",
        "Examples:",
    ]
    for ex in train:
        lines.append(f"Input: {render_inputs(ex.inputs)}")
        lines.append(f"Output: {ex.output}")
    lines.extend(["", "Query:", f"Input: {render_inputs(query.inputs)}", "Output:"])
    return "\n".join(lines)


def batch_prompt(train: Sequence[Example], heldout: Sequence[Example]) -> str:
    lines = [
        "Infer the text transformation from the examples.",
        "Apply one consistent transformation to every query.",
        "Return only a valid JSON array of strings in query order.",
        f"The array must contain exactly {len(heldout)} strings.",
        "",
        "Examples:",
    ]
    for ex in train:
        lines.append(f"Input: {render_inputs(ex.inputs)}")
        lines.append(f"Output: {ex.output}")
    lines.extend(["", "Queries:"])
    for i, ex in enumerate(heldout):
        lines.append(f"{i}: {render_inputs(ex.inputs)}")
    lines.append("")
    lines.append("JSON array only:")
    return "\n".join(lines)


def memory_prompt(
    train: Sequence[Example],
    heldout: Sequence[Example],
    skills: Sequence[Tuple[Task, float]],
    card_examples: int,
    corrupt: bool,
    seed_text: str,
) -> str:
    lines = [
        "Solve the target text transformation.",
        "The target examples are authoritative: follow their operation, formatting, casing, punctuation, and units exactly.",
        "Optional verified reference transformations are provided only as analogies.",
        "Ignore any reference that conflicts with the target examples.",
        "Return only a valid JSON array of strings in query order.",
        f"The array must contain exactly {len(heldout)} strings.",
        "",
        "Target examples:",
    ]
    for ex in train:
        lines.append(f"Input: {render_inputs(ex.inputs)}")
        lines.append(f"Output: {ex.output}")
    lines.extend(
        [
            "",
            "Optional verified reference transformations:",
        ]
    )
    for i, (task, _) in enumerate(skills, start=1):
        lines.append("")
        lines.append(f"[Reference {i}]")
        lines.append(render_skill_card(task, card_examples, corrupt, f"{seed_text}:{i}:{task.task_id}"))
    lines.extend(
        [
            "",
            "Target queries:",
        ]
    )
    for i, ex in enumerate(heldout):
        lines.append(f"{i}: {render_inputs(ex.inputs)}")
    lines.append("")
    lines.append("JSON array only:")
    return "\n".join(lines)


def parse_json_list(text: str, expected: int) -> Tuple[List[str], bool, str]:
    raw = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    candidates = [raw]
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        candidates.insert(0, m.group(0))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(candidate)
            except Exception:
                continue
            if isinstance(obj, list):
                vals = ["" if x is None else str(x) for x in obj]
                return vals, len(vals) == expected, "ok" if len(vals) == expected else "wrong_count"
    lines = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", x).strip().strip('"').strip("'") for x in raw.splitlines() if x.strip()]
    if len(lines) == expected:
        return lines, False, "line_fallback"
    return [], False, "parse_fail"


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def pad_outputs(outputs: Sequence[str], expected: int) -> List[str]:
    vals = list(outputs[:expected])
    while len(vals) < expected:
        vals.append("")
    return vals


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


def read_generation_cache(path: Path) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["task_id"], row["method"], row["item_key"], row["prompt_sha"])] = row
    return out


def cached_generate(
    task_id: str,
    method: str,
    item_key: str,
    prompt: str,
    max_new_tokens: int,
    cache: Dict[Tuple[str, str, str, str], Dict[str, str]],
    writer: Optional[csv.DictWriter],
    tok: Any,
    model: Any,
    no_qwen: bool,
) -> str:
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    key = (task_id, method, item_key, prompt_sha)
    if key in cache:
        return cache[key]["raw"]
    if no_qwen:
        raw = ""
    else:
        raw = generate_text(
            tok,
            model,
            "You are a precise text transformation function. Follow output format exactly.",
            prompt,
            max_new_tokens,
        )
    row = {
        "task_id": task_id,
        "method": method,
        "item_key": item_key,
        "prompt_sha": prompt_sha,
        "raw": raw,
    }
    cache[key] = row
    if writer is not None:
        writer.writerow(row)
    return raw


def exact_rows(preds: Sequence[str], targets: Sequence[str]) -> List[bool]:
    return [str(a) == str(b) for a, b in zip(preds, targets)]


def summarize(task_details: pd.DataFrame) -> pd.DataFrame:
    if task_details.empty:
        return pd.DataFrame()
    return (
        task_details.groupby("method", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            row_exact=("row_exact", "mean"),
            full_task_exact=("full_task_exact", "mean"),
            parse_ok=("parse_ok", "mean"),
            avg_outputs=("output_count", "mean"),
        )
        .sort_values("full_task_exact", ascending=False)
    )


def retrieval_summary(retrieval: pd.DataFrame) -> pd.DataFrame:
    if retrieval.empty:
        return pd.DataFrame()
    return (
        retrieval.groupby(["method", "rank"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            same_family=("same_family", "mean"),
        )
        .sort_values(["method", "rank"])
    )


def method_deltas(task_details: pd.DataFrame, baseline: str = "direct_row") -> pd.DataFrame:
    if task_details.empty:
        return pd.DataFrame()
    base = task_details[task_details.method.eq(baseline)][["task_id", "row_exact", "full_task_exact"]].rename(
        columns={"row_exact": "baseline_row_exact", "full_task_exact": "baseline_full_task_exact"}
    )
    rows: List[Dict[str, Any]] = []
    for method, sub in task_details.groupby("method"):
        if method == baseline:
            continue
        merged = sub.merge(base, on="task_id")
        if merged.empty:
            continue
        full_delta = merged.full_task_exact.astype(float) - merged.baseline_full_task_exact.astype(float)
        row_delta = merged.row_exact.astype(float) - merged.baseline_row_exact.astype(float)
        rows.append(
            {
                "method": method,
                "tasks": len(merged),
                "full_task_delta": float(full_delta.mean()),
                "row_exact_delta": float(row_delta.mean()),
                "tasks_helped": int((full_delta > 0).sum()),
                "tasks_hurt": int((full_delta < 0).sum()),
                "tasks_tied": int((full_delta == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("full_task_delta", ascending=False)


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
            if "exact" in c or "ok" in c or "family" in c or "delta" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def make_plots(summary_df: pd.DataFrame, task_details: pd.DataFrame, retrieval_df: pd.DataFrame, retr_summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not summary_df.empty:
        order = list(summary_df.sort_values("full_task_exact")["method"])
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(order, [float(summary_df[summary_df.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], color="#2563eb")
        ax.set_xlim(0, 105)
        ax.set_xlabel("Full-task exact (%)")
        ax.set_title("Strict full-task exact by method")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "method_full_task_exact.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = range(len(order))
        ax.plot(list(x), [float(summary_df[summary_df.method.eq(m)]["row_exact"].iloc[0]) * 100 for m in order], marker="o", label="row exact", color="#059669")
        ax.plot(list(x), [float(summary_df[summary_df.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], marker="o", label="full-task exact", color="#dc2626")
        ax.set_xticks(list(x), order, rotation=25, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Row accuracy versus full-task consistency")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "row_vs_full_task.png", dpi=160)
        plt.close(fig)

    if not task_details.empty:
        family = task_details.groupby(["method", "family"], as_index=False).agg(full_task_exact=("full_task_exact", "mean"))
        pivot = family.pivot_table(index="family", columns="method", values="full_task_exact", aggfunc="mean").fillna(0)
        pivot = pivot.loc[pivot.index.sort_values()]
        fig, ax = plt.subplots(figsize=(11, max(5, 0.28 * len(pivot))))
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

    if not retr_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, sub in retr_summary.groupby("method"):
            ax.plot(sub["rank"], sub["same_family"] * 100, marker="o", label=method)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Retrieved rank")
        ax.set_ylabel("Same-family retrieval (%)")
        ax.set_title("Retrieval family agreement")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "retrieval_family_agreement.png", dpi=160)
        plt.close(fig)

    if not retrieval_df.empty:
        top = retrieval_df[retrieval_df["rank"].eq(1)]
        if not top.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(top["score"], bins=12, color="#7c3aed", alpha=0.85)
            ax.set_xlabel("Top retrieval cosine score")
            ax.set_ylabel("Tasks")
            ax.set_title("Top retrieval score distribution")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES / "top_retrieval_scores.png", dpi=160)
            plt.close(fig)

    if not deltas.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(deltas))
        ax.bar([v - 0.18 for v in x], deltas["tasks_helped"], width=0.36, label="helped", color="#059669")
        ax.bar([v + 0.18 for v in x], -deltas["tasks_hurt"], width=0.36, label="hurt", color="#dc2626")
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks(list(x), deltas["method"], rotation=25, ha="right")
        ax.set_ylabel("Tasks versus direct_row")
        ax.set_title("Full-task wins and losses versus direct row baseline")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "wins_losses_vs_direct.png", dpi=160)
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
    return "<!doctype html><html><head><meta charset='utf-8'><title>Verified Skill Memory RAG</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>"


def write_reports(
    run_name: str,
    config: Dict[str, Any],
    summary_df: pd.DataFrame,
    task_details: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    retr_summary: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    def metric(method: str, col: str) -> float:
        row = summary_df[summary_df.method.eq(method)]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    direct_full = metric("direct_row", "full_task_exact")
    batch_full = metric("direct_batch", "full_task_exact")
    rag_full = metric("skill_rag", "full_task_exact")
    random_full = metric("random_skill_rag", "full_task_exact")
    corrupt_full = metric("corrupt_skill_rag", "full_task_exact")
    skill_delta = deltas[deltas.method.eq("skill_rag")]
    skill_helped = int(skill_delta.tasks_helped.iloc[0]) if not skill_delta.empty else 0
    skill_hurt = int(skill_delta.tasks_hurt.iloc[0]) if not skill_delta.empty else 0
    skill_retrieval = retr_summary[(retr_summary.method.eq("skill_rag")) & (retr_summary["rank"].eq(1))]
    skill_same_family = float(skill_retrieval.same_family.iloc[0]) if not skill_retrieval.empty else float("nan")

    report = f"""# Verified Skill Memory RAG

## Question

Can a frozen language model solve text-transformation tasks more consistently when it retrieves analogous verified transformation skills from a train-only memory?

The experiment evaluates strict full-task exact: a task is correct only if every held-out row is exactly correct.

## Setup

- Dataset root: `{BENCH_ROOT}`
- Run: `{run_name}`
- Model: `{MODEL_NAME}`
- Evaluation tasks: {config.get('eval_tasks')}
- Memory tasks: {config.get('memory_tasks')}
- Retrieved skills per task: {config.get('top_k')}
- Skill-card examples: {config.get('card_examples')}
- Train examples per target task: {config.get('train_n')}
- Held-out cap per task: {config.get('heldout_cap')}
- Generation rows: {config.get('generation_records')}

## Main Result

{md_table(summary_df, ['method', 'tasks', 'row_exact', 'full_task_exact', 'parse_ok', 'avg_outputs'])}

## Interpretation

The retrieved-skill method changes strict full-task exact by {(rag_full - direct_full) * 100:.1f} points relative to row-by-row direct inference and by {(rag_full - batch_full) * 100:.1f} points relative to direct batched inference.

The random-skill control scores {random_full * 100:.1f}% full-task exact and the corrupted-skill control scores {corrupt_full * 100:.1f}%. A retrieval-memory gain is only meaningful if `skill_rag` beats both controls and the direct baselines.

This run is negative for the tested retrieval-memory mechanism. Top-1 retrieval found a same-family verified skill {skill_same_family * 100:.1f}% of the time, so the retriever was not random, but `skill_rag` helped {skill_helped} tasks and hurt {skill_hurt} task relative to row-by-row direct inference. The failure is therefore not retrieval failure alone; adding a verified analogous skill card did not reliably improve the model's target transformation.

## Charts

![Full-task exact by method](../analysis/figures/method_full_task_exact.png)

![Row versus full-task accuracy](../analysis/figures/row_vs_full_task.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

![Retrieval family agreement](../analysis/figures/retrieval_family_agreement.png)

![Top retrieval scores](../analysis/figures/top_retrieval_scores.png)

![Wins and losses versus direct](../analysis/figures/wins_losses_vs_direct.png)

## Deltas Versus Direct Row Baseline

{md_table(deltas, ['method', 'tasks', 'full_task_delta', 'row_exact_delta', 'tasks_helped', 'tasks_hurt', 'tasks_tied'])}

## Retrieval Diagnostics

{md_table(retr_summary, ['method', 'rank', 'mean_score', 'same_family'])}

## Task Details

{md_table(task_details.sort_values(['method', 'task_id']), ['task_id', 'family', 'method', 'row_exact', 'full_task_exact', 'parse_ok', 'parse_status', 'output_count'], max_rows=220)}

## Files

- `runs/{run_name}/generations.csv`
- `runs/{run_name}/task_details.csv`
- `runs/{run_name}/row_details.csv`
- `runs/{run_name}/retrieval_details.csv`
- `analysis/summary.csv`
- `analysis/task_details.csv`
- `analysis/row_details.csv`
- `analysis/retrieval_details.csv`
- `analysis/retrieval_summary.csv`
- `analysis/method_deltas.csv`
"""
    (REPORTS / "qwen_verified_skill_memory_rag_report.md").write_text(report)
    (REPORTS / "qwen_verified_skill_memory_rag_report.html").write_text(markdown_to_html(report))


def run_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    all_tasks = load_tasks(min_examples=args.min_examples)
    eval_tasks = choose_eval_tasks(all_tasks, args.task_limit, args.seed, args.train_n, args.heldout_cap, args.min_heldout)
    memory_tasks = build_memory(eval_tasks, all_tasks, args.memory_limit, args.seed)
    memory_vectors = {
        task.task_id: ngram_counter(signature_from_examples(task.examples[: args.card_examples]))
        for task in memory_tasks
    }

    tok = model = None
    if not args.no_qwen:
        tok, model = load_qwen()

    cache_path = run_dir / "generations.csv"
    cache = read_generation_cache(cache_path)
    file_exists = cache_path.exists()
    fieldnames = ["task_id", "method", "item_key", "prompt_sha", "raw"]

    task_rows: List[Dict[str, Any]] = []
    row_rows: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []

    methods = ["direct_row", "direct_batch", "skill_rag", "random_skill_rag", "corrupt_skill_rag"]

    with cache_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        call_count = 0
        for task_i, task in enumerate(eval_tasks, start=1):
            train, heldout = split_examples(task, args.train_n, args.heldout_cap)
            targets = [ex.output for ex in heldout]
            retrieved = retrieve_skills(train, memory_tasks, memory_vectors, args.top_k, args.card_examples)
            random_retrieved = random_skills(task.task_id, memory_tasks, args.top_k, args.seed)

            for method, skills in [("skill_rag", retrieved), ("random_skill_rag", random_retrieved), ("corrupt_skill_rag", retrieved)]:
                for rank, (skill_task, score) in enumerate(skills, start=1):
                    retrieval_rows.append(
                        {
                            "task_id": task.task_id,
                            "family": task.family,
                            "method": method,
                            "rank": rank,
                            "retrieved_task_id": skill_task.task_id,
                            "retrieved_family": skill_task.family,
                            "same_family": skill_task.family == task.family,
                            "score": score,
                        }
                    )

            method_outputs: Dict[str, Tuple[List[str], bool, str, str]] = {}

            row_preds: List[str] = []
            row_raws: List[str] = []
            for idx, ex in enumerate(heldout):
                prompt = direct_row_prompt(train, ex)
                raw = cached_generate(
                    task.task_id,
                    "direct_row",
                    f"row{idx}",
                    prompt,
                    args.row_max_new_tokens,
                    cache,
                    writer,
                    tok,
                    model,
                    args.no_qwen,
                )
                row_raws.append(raw)
                row_preds.append(clean_prediction(raw))
            method_outputs["direct_row"] = (row_preds, True, "row_clean", "\n---\n".join(row_raws))
            call_count += len(heldout)

            batch_specs = [
                ("direct_batch", batch_prompt(train, heldout), None, False),
                ("skill_rag", memory_prompt(train, heldout, retrieved, args.card_examples, False, task.task_id), retrieved, False),
                ("random_skill_rag", memory_prompt(train, heldout, random_retrieved, args.card_examples, False, f"{task.task_id}:random"), random_retrieved, False),
                ("corrupt_skill_rag", memory_prompt(train, heldout, retrieved, args.card_examples, True, f"{task.task_id}:corrupt"), retrieved, True),
            ]
            for method, prompt, _, _ in batch_specs:
                raw = cached_generate(
                    task.task_id,
                    method,
                    "batch",
                    prompt,
                    args.batch_max_new_tokens,
                    cache,
                    writer,
                    tok,
                    model,
                    args.no_qwen,
                )
                outputs, parse_ok, status = parse_json_list(raw, len(heldout))
                method_outputs[method] = (pad_outputs(outputs, len(heldout)), parse_ok, status, raw)
                call_count += 1

            for method in methods:
                outputs, parse_ok, parse_status, raw = method_outputs[method]
                matches = exact_rows(outputs, targets)
                task_rows.append(
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "features": ",".join(task.features),
                        "method": method,
                        "row_exact": sum(matches) / max(1, len(matches)),
                        "full_task_exact": all(matches),
                        "parse_ok": parse_ok,
                        "parse_status": parse_status,
                        "output_count": len(outputs),
                        "outputs_json": json.dumps(outputs, ensure_ascii=False),
                        "targets_json": json.dumps(targets, ensure_ascii=False),
                        "raw": raw,
                    }
                )
                for idx, (ex, pred, target, exact) in enumerate(zip(heldout, outputs, targets, matches)):
                    row_rows.append(
                        {
                            "task_id": task.task_id,
                            "family": task.family,
                            "method": method,
                            "row_index": idx,
                            "input": render_inputs(ex.inputs),
                            "prediction": pred,
                            "target": target,
                            "exact": exact,
                        }
                    )

            if task_i == 1 or task_i % 5 == 0 or task_i == len(eval_tasks):
                print(f"tasks {task_i}/{len(eval_tasks)} generation_units={call_count}", flush=True)

    task_details = pd.DataFrame(task_rows)
    row_details = pd.DataFrame(row_rows)
    retrieval_details = pd.DataFrame(retrieval_rows)
    summary_df = summarize(task_details)
    retr_summary = retrieval_summary(retrieval_details)
    deltas = method_deltas(task_details)

    task_details.to_csv(run_dir / "task_details.csv", index=False)
    row_details.to_csv(run_dir / "row_details.csv", index=False)
    retrieval_details.to_csv(run_dir / "retrieval_details.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    retr_summary.to_csv(run_dir / "retrieval_summary.csv", index=False)
    deltas.to_csv(run_dir / "method_deltas.csv", index=False)

    task_details.to_csv(ANALYSIS / "task_details.csv", index=False)
    row_details.to_csv(ANALYSIS / "row_details.csv", index=False)
    retrieval_details.to_csv(ANALYSIS / "retrieval_details.csv", index=False)
    summary_df.to_csv(ANALYSIS / "summary.csv", index=False)
    retr_summary.to_csv(ANALYSIS / "retrieval_summary.csv", index=False)
    deltas.to_csv(ANALYSIS / "method_deltas.csv", index=False)

    config = vars(args).copy()
    config.update(
        {
            "model": MODEL_NAME,
            "eval_tasks": len(eval_tasks),
            "memory_tasks": len(memory_tasks),
            "generation_records": len(read_generation_cache(cache_path)),
            "elapsed_seconds": round(time.time() - started, 2),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    make_plots(summary_df, task_details, retrieval_details, retr_summary, deltas)
    write_reports(args.run_name, config, summary_df, task_details, retrieval_details, retr_summary, deltas)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="main")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--task_limit", type=int, default=40)
    ap.add_argument("--memory_limit", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=4)
    ap.add_argument("--heldout_cap", type=int, default=6)
    ap.add_argument("--min_examples", type=int, default=5)
    ap.add_argument("--min_heldout", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--card_examples", type=int, default=4)
    ap.add_argument("--row_max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_max_new_tokens", type=int, default=384)
    ap.add_argument("--no_qwen", action="store_true")
    return ap.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

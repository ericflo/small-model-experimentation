#!/usr/bin/env python3
"""Episodic soft-prompt task-vector experiment.

For each task, this runner freezes Qwen and optimizes a short continuous prefix
on leave-one-out training examples. The learned prefix is then prepended during
held-out inference.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_episodic_soft_prompt_task_vectors")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_episodic_soft_prompt_task_vectors")
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


@dataclass
class TrainCase:
    prompt_ids: List[int]
    target_ids: List[int]


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


def choose_tasks(tasks: List[Task], limit: int, seed: int, train_n: int, heldout_cap: int, min_heldout: int) -> List[Task]:
    eligible: List[Task] = []
    for task in tasks:
        _, heldout = split_examples(task, train_n, heldout_cap)
        if len(heldout) >= min_heldout:
            eligible.append(task)
    rng = random.Random(seed)
    if limit and limit < len(eligible):
        eligible = rng.sample(eligible, limit)
    return sorted(eligible, key=lambda t: t.task_id)


def row_user_prompt(train: Sequence[Example], query: Example) -> str:
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


def batch_user_prompt(train: Sequence[Example], heldout: Sequence[Example]) -> str:
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


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


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


def pad_outputs(outputs: Sequence[str], expected: int) -> List[str]:
    vals = list(outputs[:expected])
    while len(vals) < expected:
        vals.append("")
    return vals


def render_chat(tok: Any, user: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise text transformation function. Follow output format exactly."},
        {"role": "user", "content": user},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def direct_generate(tok: Any, model: Any, user: str, max_new_tokens: int) -> str:
    import torch

    rendered = render_chat(tok, user)
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


def soft_generate(tok: Any, model: Any, prefix: Any, user: str, max_new_tokens: int) -> str:
    import torch

    rendered = render_chat(tok, user)
    enc = tok(rendered, return_tensors="pt").to(model.device)
    embed = model.get_input_embeddings()
    input_ids = enc["input_ids"]
    prompt_embeds = embed(input_ids)
    pref = prefix.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype).unsqueeze(0)
    inputs_embeds = torch.cat([pref, prompt_embeds], dim=1)
    dummy_prefix = torch.full((1, pref.shape[1]), tok.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
    gen_input_ids = torch.cat([dummy_prefix, input_ids], dim=1)
    attention_mask = torch.ones(gen_input_ids.shape, device=input_ids.device, dtype=enc["attention_mask"].dtype)
    with torch.inference_mode():
        out = model.generate(
            input_ids=gen_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    if out.shape[1] > gen_input_ids.shape[1]:
        gen = out[0, gen_input_ids.shape[1] :]
    else:
        gen = out[0]
    return tok.decode(gen, skip_special_tokens=True).strip()


def make_train_cases(tok: Any, train: Sequence[Example], shuffle_outputs: bool, seed: int) -> List[TrainCase]:
    outputs = [ex.output for ex in train]
    if shuffle_outputs and len(outputs) > 1:
        rng = random.Random(seed)
        rng.shuffle(outputs)
        if all(a == b.output for a, b in zip(outputs, train)):
            outputs = outputs[1:] + outputs[:1]
    cases: List[TrainCase] = []
    for i, ex in enumerate(train):
        context = [train[j] for j in range(len(train)) if j != i]
        user = row_user_prompt(context, ex)
        prompt_ids = tok(render_chat(tok, user), add_special_tokens=False)["input_ids"]
        target_text = outputs[i] + (tok.eos_token or "")
        target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        cases.append(TrainCase(prompt_ids=prompt_ids, target_ids=target_ids))
    return cases


def init_prefix(tok: Any, model: Any, soft_tokens: int, seed: int) -> Any:
    import torch

    embed = model.get_input_embeddings()
    template = tok(" task vector ", add_special_tokens=False)["input_ids"]
    if not template:
        template = [tok.eos_token_id]
    base_ids = (template * ((soft_tokens + len(template) - 1) // len(template)))[:soft_tokens]
    ids = torch.tensor(base_ids, device=model.device, dtype=torch.long)
    with torch.no_grad():
        base = embed(ids).detach().float()
        g = torch.Generator(device=base.device)
        g.manual_seed(seed)
        noise = torch.randn(base.shape, generator=g, device=base.device, dtype=base.dtype) * 0.01
    return (base + noise).detach()


def prefix_loss(tok: Any, model: Any, prefix: Any, cases: Sequence[TrainCase]) -> Any:
    import torch

    embed = model.get_input_embeddings()
    max_len = max(len(c.prompt_ids) + len(c.target_ids) for c in cases)
    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    mask_rows: List[List[int]] = []
    for c in cases:
        ids = c.prompt_ids + c.target_ids
        labels = [-100] * len(c.prompt_ids) + c.target_ids
        pad = max_len - len(ids)
        input_rows.append(ids + [tok.pad_token_id] * pad)
        label_rows.append(labels + [-100] * pad)
        mask_rows.append([1] * len(ids) + [0] * pad)
    input_ids = torch.tensor(input_rows, device=model.device, dtype=torch.long)
    labels = torch.tensor(label_rows, device=model.device, dtype=torch.long)
    mask = torch.tensor(mask_rows, device=model.device, dtype=torch.long)
    prompt_embeds = embed(input_ids)
    pref = prefix.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype).unsqueeze(0).expand(input_ids.shape[0], -1, -1)
    inputs_embeds = torch.cat([pref, prompt_embeds], dim=1)
    prefix_mask = torch.ones((input_ids.shape[0], pref.shape[1]), device=model.device, dtype=mask.dtype)
    attention_mask = torch.cat([prefix_mask, mask], dim=1)
    prefix_labels = torch.full((input_ids.shape[0], pref.shape[1]), -100, device=model.device, dtype=labels.dtype)
    labels = torch.cat([prefix_labels, labels], dim=1)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    return out.loss


def optimize_prefix(
    tok: Any,
    model: Any,
    train: Sequence[Example],
    soft_tokens: int,
    steps: int,
    lr: float,
    seed: int,
    shuffle_outputs: bool,
) -> Tuple[Any, List[Dict[str, Any]], Any]:
    import torch

    init = init_prefix(tok, model, soft_tokens, seed)
    prefix = torch.nn.Parameter(init.clone())
    cases = make_train_cases(tok, train, shuffle_outputs=shuffle_outputs, seed=seed + 313)
    opt = torch.optim.AdamW([prefix], lr=lr, weight_decay=0.0)
    logs: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_prefix = prefix.detach().clone()
    for step in range(steps + 1):
        loss = prefix_loss(tok, model, prefix, cases)
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_prefix = prefix.detach().clone()
        logs.append({"step": step, "loss": loss_value, "best_loss": best_loss})
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prefix], 1.0)
        opt.step()
    return best_prefix.detach(), logs, init.detach()


def exact_rows(preds: Sequence[str], targets: Sequence[str]) -> List[bool]:
    return [str(a) == str(b) for a, b in zip(preds, targets)]


def method_summary(task_details: pd.DataFrame) -> pd.DataFrame:
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


def md_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 160) -> str:
    if df.empty:
        return "_No rows._"
    view = df[list(cols)].head(max_rows).copy()
    for c in view.columns:
        if view[c].dtype.kind in "fc":
            if "exact" in c or "ok" in c or "delta" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def make_plots(summary: pd.DataFrame, deltas: pd.DataFrame, task_details: pd.DataFrame, train_log: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        order = list(summary.sort_values("full_task_exact")["method"])
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(order, [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], color="#2563eb")
        ax.set_xlim(0, 105)
        ax.set_xlabel("Full-task exact (%)")
        ax.set_title("Strict full-task exact by method")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "method_full_task_exact.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = range(len(order))
        ax.plot(list(x), [float(summary[summary.method.eq(m)]["row_exact"].iloc[0]) * 100 for m in order], marker="o", label="row exact", color="#059669")
        ax.plot(list(x), [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], marker="o", label="full-task exact", color="#dc2626")
        ax.set_xticks(list(x), order, rotation=25, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Row accuracy versus full-task consistency")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "row_vs_full_task.png", dpi=160)
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

    if not train_log.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arm, sub in train_log.groupby("arm"):
            mean = sub.groupby("step", as_index=False).agg(loss=("loss", "mean"))
            ax.plot(mean["step"], mean["loss"], marker="o", label=arm)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Mean leave-one-out train loss")
        ax.set_title("Soft-prompt optimization curves")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "train_loss_curves.png", dpi=160)
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
    return "<!doctype html><html><head><meta charset='utf-8'><title>Episodic Soft-Prompt Task Vectors</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>"


def write_reports(run_name: str, config: Dict[str, Any], summary: pd.DataFrame, deltas: pd.DataFrame, task_details: pd.DataFrame, train_log: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    def metric(method: str, col: str) -> float:
        row = summary[summary.method.eq(method)]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    direct_full = metric("direct_row", "full_task_exact")
    learned_full = metric("learned_soft_row", "full_task_exact")
    learned_row = metric("learned_soft_row", "row_exact")
    direct_row_acc = metric("direct_row", "row_exact")
    init_full = metric("init_soft_row", "full_task_exact")
    shuffled_full = metric("shuffled_soft_row", "full_task_exact")
    learned_delta = deltas[deltas.method.eq("learned_soft_row")]
    helped = int(learned_delta.tasks_helped.iloc[0]) if not learned_delta.empty else 0
    hurt = int(learned_delta.tasks_hurt.iloc[0]) if not learned_delta.empty else 0

    report = f"""# Episodic Soft-Prompt Task Vectors

## Question

Can a frozen language model become more consistent on text-transformation tasks by learning a small continuous task vector from that task's examples?

Each task receives its own optimized soft prefix. The model weights are frozen. The prefix is trained on leave-one-out versions of the task's training rows, then evaluated on held-out rows.

## Setup

- Dataset root: `{BENCH_ROOT}`
- Run: `{run_name}`
- Model: `{MODEL_NAME}`
- Tasks: {config.get('tasks')}
- Soft tokens: {config.get('soft_tokens')}
- Optimization steps: {config.get('steps')}
- Learning rate: {config.get('lr')}
- Train examples per task: {config.get('train_n')}
- Held-out cap per task: {config.get('heldout_cap')}

## Main Result

{md_table(summary, ['method', 'tasks', 'row_exact', 'full_task_exact', 'parse_ok', 'avg_outputs'])}

## Interpretation

The learned soft-prompt row method changes strict full-task exact by {(learned_full - direct_full) * 100:.1f} points relative to direct row-by-row inference. It changes full-task exact by {(learned_full - init_full) * 100:.1f} points relative to the untrained initialized prefix and by {(learned_full - shuffled_full) * 100:.1f} points relative to a prefix optimized on shuffled training outputs.

The learned prefix changes row exact by {(learned_row - direct_row_acc) * 100:.1f} points relative to direct row-by-row inference. It helps {helped} tasks and hurts {hurt} tasks on strict full-task exact. A positive result requires `learned_soft_row` to beat direct inference and both soft-prefix controls.

This run is neutral for strict full-task exact and weakly positive only at row level. The learned prefix beats the shuffled-label control, showing that the optimization target matters, but it does not improve the number of fully solved tasks over direct row-by-row inference.

## Charts

![Full-task exact by method](../analysis/figures/method_full_task_exact.png)

![Row versus full-task accuracy](../analysis/figures/row_vs_full_task.png)

![Wins and losses versus direct](../analysis/figures/wins_losses_vs_direct.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

![Train loss curves](../analysis/figures/train_loss_curves.png)

## Deltas Versus Direct Row Baseline

{md_table(deltas, ['method', 'tasks', 'full_task_delta', 'row_exact_delta', 'tasks_helped', 'tasks_hurt', 'tasks_tied'])}

## Task Details

{md_table(task_details.sort_values(['method', 'task_id']), ['task_id', 'family', 'method', 'row_exact', 'full_task_exact', 'parse_ok', 'parse_status', 'output_count'], max_rows=220)}

## Train Loss Log

{md_table(train_log.groupby(['arm', 'step'], as_index=False).agg(loss=('loss', 'mean')).sort_values(['arm', 'step']), ['arm', 'step', 'loss'], max_rows=120)}

## Files

- `runs/{run_name}/task_details.csv`
- `runs/{run_name}/row_details.csv`
- `runs/{run_name}/train_log.csv`
- `runs/{run_name}/summary.csv`
- `runs/{run_name}/method_deltas.csv`
- `analysis/summary.csv`
- `analysis/method_deltas.csv`
- `analysis/task_details.csv`
- `analysis/row_details.csv`
- `analysis/train_log.csv`
"""
    (REPORTS / "qwen_episodic_soft_prompt_task_vectors_report.md").write_text(report)
    (REPORTS / "qwen_episodic_soft_prompt_task_vectors_report.html").write_text(markdown_to_html(report))


def evaluate_outputs(
    task: Task,
    method: str,
    outputs: Sequence[str],
    targets: Sequence[str],
    parse_ok: bool,
    parse_status: str,
    raw: str,
) -> Dict[str, Any]:
    matches = exact_rows(outputs, targets)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "features": ",".join(task.features),
        "method": method,
        "row_exact": sum(matches) / max(1, len(matches)),
        "full_task_exact": all(matches),
        "parse_ok": parse_ok,
        "parse_status": parse_status,
        "output_count": len(outputs),
        "outputs_json": json.dumps(list(outputs), ensure_ascii=False),
        "targets_json": json.dumps(list(targets), ensure_ascii=False),
        "raw": raw,
    }


def add_row_details(rows: List[Dict[str, Any]], task: Task, method: str, heldout: Sequence[Example], outputs: Sequence[str], targets: Sequence[str]) -> None:
    for idx, (ex, pred, target, exact) in enumerate(zip(heldout, outputs, targets, exact_rows(outputs, targets))):
        rows.append(
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


def run_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        task_details = pd.read_csv(run_dir / "task_details.csv")
        row_details = pd.read_csv(run_dir / "row_details.csv")
        train_log = pd.read_csv(run_dir / "train_log.csv")
        summary = method_summary(task_details)
        deltas = method_deltas(task_details)
        summary.to_csv(ANALYSIS / "summary.csv", index=False)
        deltas.to_csv(ANALYSIS / "method_deltas.csv", index=False)
        task_details.to_csv(ANALYSIS / "task_details.csv", index=False)
        row_details.to_csv(ANALYSIS / "row_details.csv", index=False)
        train_log.to_csv(ANALYSIS / "train_log.csv", index=False)
        config = json.loads((run_dir / "config.json").read_text())
        make_plots(summary, deltas, task_details, train_log)
        write_reports(args.run_name, config, summary, deltas, task_details, train_log)
        return
    started = time.time()
    tasks = choose_tasks(load_tasks(args.min_examples), args.task_limit, args.seed, args.train_n, args.heldout_cap, args.min_heldout)

    tok = model = None
    if not args.no_qwen:
        tok, model = load_qwen()

    task_rows: List[Dict[str, Any]] = []
    row_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    for task_i, task in enumerate(tasks, start=1):
        train, heldout = split_examples(task, args.train_n, args.heldout_cap)
        targets = [ex.output for ex in heldout]

        prefixes: Dict[str, Any] = {}
        init_prefix_tensor = None
        if args.no_qwen:
            learned_logs = [{"step": 0, "loss": 0.0}]
            shuffled_logs = [{"step": 0, "loss": 0.0}]
        else:
            prefix, learned_logs, init_prefix_tensor = optimize_prefix(
                tok, model, train, args.soft_tokens, args.steps, args.lr, args.seed + task_i * 1009, shuffle_outputs=False
            )
            shuffled_prefix, shuffled_logs, _ = optimize_prefix(
                tok, model, train, args.soft_tokens, args.steps, args.lr, args.seed + task_i * 1009, shuffle_outputs=True
            )
            prefixes["learned"] = prefix
            prefixes["shuffled"] = shuffled_prefix
            prefixes["init"] = init_prefix_tensor

        for arm, logs in [("learned", learned_logs), ("shuffled", shuffled_logs)]:
            for row in logs:
                train_rows.append(
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "arm": arm,
                        "step": row["step"],
                        "loss": row["loss"],
                    }
                )

        # Direct row-by-row.
        direct_row_outputs: List[str] = []
        direct_raw: List[str] = []
        for ex in heldout:
            if args.no_qwen:
                raw = ""
            else:
                raw = direct_generate(tok, model, row_user_prompt(train, ex), args.row_max_new_tokens)
            direct_raw.append(raw)
            direct_row_outputs.append(clean_prediction(raw))
        task_rows.append(evaluate_outputs(task, "direct_row", direct_row_outputs, targets, True, "row_clean", "\n---\n".join(direct_raw)))
        add_row_details(row_rows, task, "direct_row", heldout, direct_row_outputs, targets)

        # Direct batch.
        if args.no_qwen:
            raw = ""
        else:
            raw = direct_generate(tok, model, batch_user_prompt(train, heldout), args.batch_max_new_tokens)
        outs, parse_ok, status = parse_json_list(raw, len(heldout))
        direct_batch_outputs = pad_outputs(outs, len(heldout))
        task_rows.append(evaluate_outputs(task, "direct_batch", direct_batch_outputs, targets, parse_ok, status, raw))
        add_row_details(row_rows, task, "direct_batch", heldout, direct_batch_outputs, targets)

        # Soft-prompt row methods.
        for method, key in [("init_soft_row", "init"), ("learned_soft_row", "learned"), ("shuffled_soft_row", "shuffled")]:
            outputs: List[str] = []
            raws: List[str] = []
            for ex in heldout:
                if args.no_qwen:
                    raw = ""
                else:
                    raw = soft_generate(tok, model, prefixes[key], row_user_prompt(train, ex), args.row_max_new_tokens)
                raws.append(raw)
                outputs.append(clean_prediction(raw))
            task_rows.append(evaluate_outputs(task, method, outputs, targets, True, "row_clean", "\n---\n".join(raws)))
            add_row_details(row_rows, task, method, heldout, outputs, targets)

        # Learned soft-prompt batch method.
        if args.no_qwen:
            raw = ""
        else:
            raw = soft_generate(tok, model, prefixes["learned"], batch_user_prompt(train, heldout), args.batch_max_new_tokens)
        outs, parse_ok, status = parse_json_list(raw, len(heldout))
        learned_batch_outputs = pad_outputs(outs, len(heldout))
        task_rows.append(evaluate_outputs(task, "learned_soft_batch", learned_batch_outputs, targets, parse_ok, status, raw))
        add_row_details(row_rows, task, "learned_soft_batch", heldout, learned_batch_outputs, targets)

        if task_i == 1 or task_i % 5 == 0 or task_i == len(tasks):
            print(f"tasks {task_i}/{len(tasks)}", flush=True)

    task_details = pd.DataFrame(task_rows)
    row_details = pd.DataFrame(row_rows)
    train_log = pd.DataFrame(train_rows)
    summary = method_summary(task_details)
    deltas = method_deltas(task_details)

    task_details.to_csv(run_dir / "task_details.csv", index=False)
    row_details.to_csv(run_dir / "row_details.csv", index=False)
    train_log.to_csv(run_dir / "train_log.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    deltas.to_csv(run_dir / "method_deltas.csv", index=False)

    task_details.to_csv(ANALYSIS / "task_details.csv", index=False)
    row_details.to_csv(ANALYSIS / "row_details.csv", index=False)
    train_log.to_csv(ANALYSIS / "train_log.csv", index=False)
    summary.to_csv(ANALYSIS / "summary.csv", index=False)
    deltas.to_csv(ANALYSIS / "method_deltas.csv", index=False)

    config = vars(args).copy()
    config.update(
        {
            "model": MODEL_NAME,
            "tasks": len(tasks),
            "elapsed_seconds": round(time.time() - started, 2),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    make_plots(summary, deltas, task_details, train_log)
    write_reports(args.run_name, config, summary, deltas, task_details, train_log)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="main")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--task_limit", type=int, default=20)
    ap.add_argument("--train_n", type=int, default=4)
    ap.add_argument("--heldout_cap", type=int, default=4)
    ap.add_argument("--min_examples", type=int, default=5)
    ap.add_argument("--min_heldout", type=int, default=3)
    ap.add_argument("--soft_tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--row_max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_max_new_tokens", type=int, default=320)
    ap.add_argument("--no_qwen", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

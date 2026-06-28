#!/usr/bin/env python3
"""Support-contrastive meta-ICL posttraining for Qwen.

The experiment trains small LoRA adapters on synthetic text-transformation
episodes and evaluates whether the learned behavior transfers to public
few-shot transformation tasks. The distinctive arm adds margin losses that
make the same answer less likely when the support examples are corrupted or
removed.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


EXP_NAME = "qwen_support_contrastive_meta_icl"
ROOT = Path("/workspace/experiments") / EXP_NAME
LARGE_ROOT = Path("/workspace/large_artifacts") / EXP_NAME
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGS = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
PROSE_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = PROSE_ROOT / "Transformation.Text"


def ensure_dirs() -> None:
    for p in [ROOT, LARGE_ROOT, RUNS, ANALYSIS, FIGS, REPORTS, LARGE_ROOT / "checkpoints"]:
        p.mkdir(parents=True, exist_ok=True)
    if not PROSE_ROOT.exists():
        src_env = os.environ.get("PROSE_BENCHMARKS")
        candidates = [Path(src_env)] if src_env else []
        candidates.extend(Path("/workspace/large_artifacts").glob("*/prose-benchmarks"))
        for candidate in candidates:
            if candidate.exists() and (candidate / "Transformation.Text").exists():
                shutil.copytree(candidate, PROSE_ROOT, symlinks=True)
                break
    if not TRANSFORM_ROOT.exists():
        raise FileNotFoundError(f"Could not locate public benchmark at {TRANSFORM_ROOT}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def clean_text(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def exact(a: str, b: str) -> bool:
    return clean_text(a) == clean_text(b)


def pct(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{100 * float(x):.1f}%"


def parse_first_line(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.split("<|endoftext|>")[0]
    text = text.split("<|im_end|>")[0]
    text = re.split(
        r"\s+(?:Okay|Wait|To solve|Let's|I need|The pattern|The user|For example)\b",
        text,
        maxsplit=1,
    )[0]
    for marker in ["\nInput:", "\nQuery", "\nExample", "\nOutput:"]:
        if marker in text:
            text = text.split(marker)[0]
    text = text.strip()
    if text.startswith("?"):
        text = ""
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    return clean_text(text)


@dataclass(frozen=True)
class Example:
    inp: str
    out: str


@dataclass(frozen=True)
class Episode:
    task_id: str
    family: str
    rule_name: str
    support: tuple[Example, ...]
    queries: tuple[Example, ...]
    contrast_support: tuple[Example, ...] = tuple()
    contrast_rule: str | None = None


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    fn: Callable[[str], str]


NAMES = [
    "Aysu Polat",
    "Hone Albert",
    "Artem Kuznetsov",
    "Fatma Yilmaz",
    "Sonia Rees",
    "Elise Vadeboncoeur",
    "Akila Kadija",
    "Muralix Hasmik",
    "Fiamma Greco",
    "Noah Martinez",
    "Ava Johnson",
    "Lucas Brown",
]
WORDS = [
    "atlas",
    "binary",
    "cobalt",
    "delta",
    "ember",
    "fable",
    "granite",
    "harbor",
    "ivory",
    "jupiter",
    "kernel",
    "lambda",
    "matrix",
    "nectar",
    "orbit",
    "prairie",
]
DOMAINS = ["example.com", "northwind.io", "contoso.net", "fabrikam.org", "mail.test"]
CITIES = ["Denver", "Seattle", "Austin", "Boston", "Phoenix", "Yonkers", "Curitiba"]
STATES = ["CA", "NY", "TX", "WA", "OH", "AK", "OK"]


def split_words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", s)


def digits(s: str) -> str:
    return "".join(re.findall(r"\d", s))


def letters(s: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", s))


def first_word(s: str) -> str:
    toks = split_words(s)
    return toks[0] if toks else ""


def second_word(s: str) -> str:
    toks = split_words(s)
    return toks[1] if len(toks) > 1 else ""


def last_word(s: str) -> str:
    toks = split_words(s)
    return toks[-1] if toks else ""


def initials(s: str) -> str:
    toks = split_words(s)
    return "".join(t[0].upper() for t in toks if t)


def first_three(s: str) -> str:
    return clean_text(s)[:3]


def last_three(s: str) -> str:
    return clean_text(s)[-3:]


def field(delim: str, idx: int) -> Callable[[str], str]:
    def _f(s: str) -> str:
        parts = [p.strip() for p in s.split(delim)]
        if not parts:
            return ""
        j = idx if idx >= 0 else len(parts) + idx
        return parts[j] if 0 <= j < len(parts) else ""

    return _f


def email_user(s: str) -> str:
    m = re.search(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)", s)
    return m.group(1) if m else ""


def email_domain(s: str) -> str:
    m = re.search(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)", s)
    return m.group(2) if m else ""


def number_plus_1(s: str) -> str:
    m = re.search(r"-?\d+", s)
    return str(int(m.group(0)) + 1) if m else ""


def number_times_2(s: str) -> str:
    m = re.search(r"-?\d+", s)
    return str(int(m.group(0)) * 2) if m else ""


def number_last2(s: str) -> str:
    d = digits(s)
    return d[-2:] if len(d) >= 2 else d


RULES: tuple[Rule, ...] = (
    Rule("first_word", "words", first_word),
    Rule("second_word", "words", second_word),
    Rule("last_word", "words", last_word),
    Rule("initials", "words", initials),
    Rule("upper", "case", lambda s: clean_text(s).upper()),
    Rule("lower", "case", lambda s: clean_text(s).lower()),
    Rule("title", "case", lambda s: clean_text(s).title()),
    Rule("first_three", "span", first_three),
    Rule("last_three", "span", last_three),
    Rule("digits", "class", digits),
    Rule("letters", "class", letters),
    Rule("pipe_0", "delim", field("|", 0)),
    Rule("pipe_1", "delim", field("|", 1)),
    Rule("pipe_2", "delim", field("|", 2)),
    Rule("dash_0", "delim", field("-", 0)),
    Rule("dash_1", "delim", field("-", 1)),
    Rule("comma_0", "delim", field(",", 0)),
    Rule("comma_1", "delim", field(",", 1)),
    Rule("email_user", "email", email_user),
    Rule("email_domain", "email", email_domain),
    Rule("number_plus_1", "number", number_plus_1),
    Rule("number_times_2", "number", number_times_2),
    Rule("number_last2", "number", number_last2),
)


def make_input(rng: random.Random, family: str) -> str:
    if family == "words":
        return f"{rng.choice(NAMES)} {rng.choice(WORDS).title()}"
    if family == "case":
        text = f"{rng.choice(WORDS)} {rng.choice(WORDS)}"
        mode = rng.choice(["lower", "upper", "title", "mixed"])
        if mode == "upper":
            return text.upper()
        if mode == "title":
            return text.title()
        if mode == "mixed":
            return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(text))
        return text.lower()
    if family == "span":
        return rng.choice(WORDS) + rng.choice(WORDS)
    if family == "class":
        return f"{rng.choice(WORDS).title()}-{rng.randint(100, 999)}-{rng.choice(STATES)}"
    if family == "delim":
        parts = [rng.choice(CITIES), rng.choice(STATES), str(rng.randint(100, 999))]
        return rng.choice(["|", "-", ","]).join(parts)
    if family == "email":
        user = f"{rng.choice(WORDS)}.{rng.choice(WORDS)}{rng.randint(1,99)}"
        return f"{user}@{rng.choice(DOMAINS)}"
    if family == "number":
        return f"{rng.choice(WORDS).upper()} {rng.randint(10, 9999)}"
    return f"{rng.choice(WORDS)} {rng.randint(1,999)}"


def make_episode(
    rng: random.Random,
    task_id: str,
    support_n: int,
    query_n: int,
    rule: Rule | None = None,
) -> Episode:
    rule = rule or rng.choice(RULES)
    used: set[str] = set()

    def unique_input() -> str:
        for _ in range(200):
            s = make_input(rng, rule.family)
            if s not in used and rule.fn(s) != "":
                used.add(s)
                return s
        s = make_input(rng, rule.family)
        used.add(s)
        return s

    support = tuple(Example(x, rule.fn(x)) for x in (unique_input() for _ in range(support_n)))
    queries = tuple(Example(x, rule.fn(x)) for x in (unique_input() for _ in range(query_n)))
    return Episode(task_id=task_id, family=rule.family, rule_name=rule.name, support=support, queries=queries)


def counterfactual_eval_episodes(seed: int, pairs: int, support_n: int, query_n: int) -> list[Episode]:
    rng = random.Random(seed)
    out: list[Episode] = []
    by_family: dict[str, list[Rule]] = {}
    for r in RULES:
        by_family.setdefault(r.family, []).append(r)
    eligible = [f for f, rs in by_family.items() if len(rs) >= 2]
    for i in range(pairs):
        inputs: list[str] | None = None
        r1: Rule | None = None
        r2: Rule | None = None
        fam = ""
        for pair_attempt in range(400):
            fam = rng.choice(eligible)
            cand1, cand2 = rng.sample(by_family[fam], 2)
            base_rng = random.Random(seed * 100000 + i * 997 + pair_attempt)
            found: list[str] = []
            for _ in range(2500):
                s = make_input(base_rng, fam)
                if s in found:
                    continue
                y1 = cand1.fn(s)
                y2 = cand2.fn(s)
                if y1 and y2 and y1 != y2:
                    found.append(s)
                    if len(found) >= support_n + query_n:
                        break
            if len(found) >= support_n + query_n:
                inputs, r1, r2 = found, cand1, cand2
                break
        if inputs is None or r1 is None or r2 is None:
            raise RuntimeError(f"Could not construct counterfactual pair {i}")
        support_inputs = inputs[:support_n]
        query_inputs = inputs[support_n:]
        for r, other in [(r1, r2), (r2, r1)]:
            support = tuple(Example(x, r.fn(x)) for x in support_inputs)
            contrast_support = tuple(Example(x, other.fn(x)) for x in support_inputs)
            queries = tuple(Example(x, r.fn(x)) for x in query_inputs)
            out.append(
                Episode(
                    task_id=f"cf_{i:04d}_{r.name}",
                    family=fam,
                    rule_name=r.name,
                    support=support,
                    queries=queries,
                    contrast_support=contrast_support,
                    contrast_rule=other.name,
                )
            )
    return out


def train_episodes(seed: int, n: int, support_n: int, query_n: int, train_mode: str) -> list[Episode]:
    if train_mode in {"counterfactual", "shuffled_labels"}:
        return counterfactual_eval_episodes(seed, math.ceil(n / 2), support_n, query_n)[:n]
    if train_mode != "ordinary":
        raise ValueError(f"unknown train mode: {train_mode}")
    rng = random.Random(seed)
    return [make_episode(rng, f"train_{i:05d}", support_n, query_n, rng.choice(RULES)) for i in range(n)]


def shuffled_support(support: tuple[Example, ...], rng: random.Random) -> tuple[Example, ...]:
    outs = [ex.out for ex in support]
    rng.shuffle(outs)
    return tuple(Example(ex.inp, out) for ex, out in zip(support, outs))


def format_prompt(support: Iterable[Example], query_input: str) -> str:
    lines = ["Infer the text transformation from the examples. Return only the output for the query.", "Examples:"]
    for ex in support:
        lines.append(f"Input: {ex.inp}")
        lines.append(f"Output: {ex.out}")
    lines.append(f"Query Input: {query_input}")
    lines.append("Output:")
    return "\n".join(lines)


def load_prose_tasks(limit: int, train_n: int, heldout_n: int, seed: int) -> list[Episode]:
    tasks: list[Episode] = []
    for d in sorted(TRANSFORM_ROOT.iterdir()):
        if not d.is_dir() or not (d / "spec.json").exists():
            continue
        try:
            spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        rows: list[Example] = []
        for ex in spec.get("Examples", []):
            inp = ex.get("Input", [])
            inp_s = " | ".join(clean_text(x) for x in inp) if isinstance(inp, list) else clean_text(inp)
            out_s = clean_text(ex.get("Output", ""))
            if inp_s and out_s:
                rows.append(Example(inp_s, out_s))
        seen: set[tuple[str, str]] = set()
        dedup: list[Example] = []
        for r in rows:
            key = (r.inp, r.out)
            if key not in seen:
                seen.add(key)
                dedup.append(r)
        if len(dedup) < train_n + heldout_n:
            continue
        tasks.append(
            Episode(
                task_id=d.name,
                family=d.name.split(".")[0],
                rule_name="public",
                support=tuple(dedup[:train_n]),
                queries=tuple(dedup[train_n : train_n + heldout_n]),
            )
        )
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks[:limit]


@dataclass(frozen=True)
class TrainRow:
    prompt: str
    answer: str
    negative_prompts: tuple[str, ...]


def build_train_rows(
    episodes: list[Episode],
    objective: str,
    train_mode: str,
    rows_per_episode: int,
    seed: int,
) -> list[TrainRow]:
    rng = random.Random(seed)
    rows: list[TrainRow] = []
    for ep in episodes:
        if train_mode == "shuffled_labels":
            support = shuffled_support(ep.support, rng)
        else:
            support = ep.support
        for q in ep.queries[:rows_per_episode]:
            negs: list[str] = []
            if objective == "support_contrastive":
                negs.append(format_prompt(shuffled_support(ep.support, rng), q.inp))
                negs.append(format_prompt(tuple(), q.inp))
                if ep.contrast_support:
                    negs.append(format_prompt(ep.contrast_support, q.inp))
            rows.append(TrainRow(format_prompt(support, q.inp), q.out, tuple(negs)))
    rng.shuffle(rows)
    return rows


def encode_answer(tokenizer, prompt: str, answer: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    if len(target_ids) >= max_length:
        target_ids = target_ids[-max_length + 1 :]
    keep_prompt = max(0, max_length - len(target_ids))
    prompt_ids = prompt_ids[-keep_prompt:]
    ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
        torch.ones((1, len(ids)), dtype=torch.long),
    )


def answer_nll(model, tokenizer, prompt: str, answer: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, labels, attention_mask = encode_answer(tokenizer, prompt, answer, max_length)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    if not mask.any():
        raise RuntimeError("empty target mask")
    nll = F.cross_entropy(logits[mask], shifted_labels[mask], reduction="sum")
    return nll, mask.sum().to(nll.dtype)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_first_line(gen)


def support_for_eval(ep: Episode, support_mode: str, rng: random.Random) -> tuple[Example, ...]:
    if support_mode == "normal":
        return ep.support
    if support_mode == "shuffled":
        return shuffled_support(ep.support, rng)
    if support_mode == "none":
        return tuple()
    if support_mode == "contrast":
        return ep.contrast_support if ep.contrast_support else shuffled_support(ep.support, rng)
    raise ValueError(support_mode)


def evaluate_episodes(
    model,
    tokenizer,
    episodes: list[Episode],
    method: str,
    split: str,
    support_modes: list[str],
    max_new_tokens: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    support_seed_offsets = {"normal": 11, "shuffled": 23, "none": 37, "contrast": 41}
    for support_mode in support_modes:
        rng = random.Random(seed + support_seed_offsets[support_mode])
        for ep in episodes:
            support = support_for_eval(ep, support_mode, rng)
            for qi, q in enumerate(ep.queries):
                prompt = format_prompt(support, q.inp)
                pred = generate_one(model, tokenizer, prompt, max_new_tokens)
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "support_mode": support_mode,
                        "task_id": ep.task_id,
                        "family": ep.family,
                        "rule_name": ep.rule_name,
                        "contrast_rule": ep.contrast_rule,
                        "query_index": qi,
                        "input": q.inp,
                        "target": q.out,
                        "prediction": pred,
                        "exact": exact(pred, q.out),
                    }
                )
    row_df = pd.DataFrame(rows)
    task_df = (
        row_df.groupby(["method", "split", "support_mode", "task_id", "family"], as_index=False)
        .agg(row_exact=("exact", "mean"), full_task_exact=("exact", "all"), rows=("exact", "size"))
    )
    return row_df, task_df


def summarize(task_df: pd.DataFrame, row_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in task_df.groupby(["method", "split", "support_mode"]):
        rg = row_df
        for k, v in zip(["method", "split", "support_mode"], key):
            rg = rg[rg[k] == v]
        rows.append(
            {
                "method": key[0],
                "split": key[1],
                "support_mode": key[2],
                "tasks": len(g),
                "rows": len(rg),
                "row_exact": float(rg["exact"].mean()) if len(rg) else 0.0,
                "full_task_exact": float(g["full_task_exact"].mean()) if len(g) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "method", "support_mode"]).reset_index(drop=True)


def train_adapter(model, tokenizer, args, train_eps: list[Episode], run_dir: Path) -> pd.DataFrame:
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()
    rows = build_train_rows(train_eps, args.objective, args.train_mode, args.train_rows_per_episode, args.seed + 901)
    rng = random.Random(args.seed + 902)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, args.warmup_steps),
        num_training_steps=args.train_steps,
    )
    opt.zero_grad(set_to_none=True)
    log_rows: list[dict[str, float]] = []
    micro = 0
    step = 0
    accum_total = 0.0
    accum_pos = 0.0
    accum_margin = 0.0
    while step < args.train_steps:
        rng.shuffle(rows)
        for row in rows:
            pos_nll, pos_tok = answer_nll(model, tokenizer, row.prompt, row.answer, args.max_length)
            pos_loss = pos_nll / pos_tok
            scaled_pos = pos_loss / args.grad_accum
            scaled_pos.backward()
            total_loss = float(pos_loss.detach().cpu())
            margin_loss_value = 0.0
            if args.objective == "support_contrastive" and row.negative_prompts:
                pos_lp = (-pos_loss).detach()
                for neg_prompt in row.negative_prompts:
                    neg_nll, neg_tok = answer_nll(model, tokenizer, neg_prompt, row.answer, args.max_length)
                    neg_lp = -(neg_nll / neg_tok)
                    margin_loss = F.softplus(torch.tensor(args.margin, device=neg_lp.device, dtype=neg_lp.dtype) - (pos_lp - neg_lp))
                    this_loss = args.contrast_weight * margin_loss / max(1, len(row.negative_prompts))
                    (this_loss / args.grad_accum).backward()
                    total_loss += float(this_loss.detach().cpu())
                    margin_loss_value += float(this_loss.detach().cpu())
            micro += 1
            accum_total += total_loss
            accum_pos += float(pos_loss.detach().cpu())
            accum_margin += margin_loss_value
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % args.log_every == 0:
                    denom = args.log_every * args.grad_accum if step > 1 else args.grad_accum
                    rec = {
                        "step": step,
                        "loss": accum_total / max(1, denom),
                        "pos_loss": accum_pos / max(1, denom),
                        "margin_loss": accum_margin / max(1, denom),
                        "lr": sched.get_last_lr()[0],
                    }
                    print(
                        f"step {step}/{args.train_steps} loss {rec['loss']:.4f} "
                        f"pos {rec['pos_loss']:.4f} margin {rec['margin_loss']:.4f} lr {rec['lr']:.2e}",
                        flush=True,
                    )
                    log_rows.append(rec)
                    accum_total = 0.0
                    accum_pos = 0.0
                    accum_margin = 0.0
                if step >= args.train_steps:
                    break
    model.eval()
    ckpt = LARGE_ROOT / "checkpoints" / args.run_name / "adapter"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    train_log = pd.DataFrame(log_rows)
    train_log.to_csv(run_dir / "training_log.csv", index=False)
    return train_log, model


def md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if cols:
        df = df[cols]
    if max_rows is not None:
        df = df.head(max_rows)
    tmp = df.copy()
    for c in tmp.columns:
        if "exact" in c or c.endswith("_std") or c.endswith("_mean") or c.endswith("_delta"):
            tmp[c] = tmp[c].map(lambda x: pct(float(x)) if pd.notna(x) else "")
    return tmp.to_markdown(index=False) if len(tmp) else "_No rows._"


def plot_run(summary: pd.DataFrame, train_log: pd.DataFrame, task_df: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    labels = summary["method"] + "\n" + summary["split"] + "\n" + summary["support_mode"]
    order = summary.sort_values("full_task_exact", ascending=False).index
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(order)), summary.loc[order, "full_task_exact"], color="#2f6f73")
    plt.xticks(range(len(order)), labels.loc[order], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Full-task exact")
    plt.ylim(0, 1)
    plt.title("Run Full-Task Exact")
    plt.tight_layout()
    plt.savefig(FIGS / "latest_run_full_task_exact.png", dpi=180)
    plt.close()
    if len(train_log):
        plt.figure(figsize=(9, 5))
        plt.plot(train_log["step"], train_log["loss"], label="total", marker="o")
        if "pos_loss" in train_log:
            plt.plot(train_log["step"], train_log["pos_loss"], label="positive CE", marker="o")
        if "margin_loss" in train_log:
            plt.plot(train_log["step"], train_log["margin_loss"], label="margin", marker="o")
        plt.xlabel("Optimizer step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(FIGS / "latest_run_training_loss.png", dpi=180)
        plt.close()


def write_run_report(args, run_dir: Path, summary: pd.DataFrame, task_df: pd.DataFrame, row_df: pd.DataFrame, train_log: pd.DataFrame, elapsed: float) -> None:
    report_md = REPORTS / f"{EXP_NAME}_latest_run_report.md"
    lines: list[str] = []
    lines.append("# Support-Contrastive Meta-ICL Run Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Run: `{args.run_name}`")
    lines.append(f"- Objective: `{args.objective}`")
    lines.append(f"- Train mode: `{args.train_mode}`")
    lines.append(f"- Train steps: `{args.train_steps}`")
    lines.append(f"- Elapsed seconds: `{elapsed:.1f}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(md_table(summary))
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for name, caption in [("latest_run_full_task_exact.png", "Run full-task exact"), ("latest_run_training_loss.png", "Training loss")]:
        if (FIGS / name).exists():
            lines.append(f"![{caption}](../analysis/figures/{name})")
            lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Checkpoint: `{LARGE_ROOT / 'checkpoints' / args.run_name / 'adapter'}`")
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_one(args) -> None:
    ensure_dirs()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    append_log(
        f"\n## Run `{args.run_name}`\n\n"
        f"- Time UTC: `{now_iso()}`\n"
        f"- Config: `{json.dumps(vars(args), sort_keys=True)}`"
    )
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_eps = train_episodes(args.seed, args.train_episodes, args.support_n, args.query_n, args.train_mode)
    syn_eval = counterfactual_eval_episodes(args.eval_seed + 17, args.synthetic_eval_pairs, args.support_n, args.query_n)
    public_eval = load_prose_tasks(args.public_task_limit, args.support_n, args.public_heldout_n, args.eval_seed + 33)
    pd.DataFrame(
        [
            {
                "task_id": ep.task_id,
                "family": ep.family,
                "rule_name": ep.rule_name,
                "contrast_rule": ep.contrast_rule,
                "support_json": json.dumps([ex.__dict__ for ex in ep.support]),
                "contrast_support_json": json.dumps([ex.__dict__ for ex in ep.contrast_support]),
                "queries_json": json.dumps([ex.__dict__ for ex in ep.queries]),
            }
            for ep in syn_eval + public_eval
        ]
    ).to_csv(run_dir / "eval_episodes.csv", index=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    all_rows: list[pd.DataFrame] = []
    all_tasks: list[pd.DataFrame] = []
    if not args.skip_base_eval:
        for split, eps, modes in [
            ("synthetic_counterfactual", syn_eval, ["normal", "shuffled", "none", "contrast"]),
            ("public_prose", public_eval, ["normal", "shuffled", "none"]),
        ]:
            r, t = evaluate_episodes(model, tokenizer, eps, "base", split, modes, args.max_new_tokens, args.eval_seed + 100)
            all_rows.append(r)
            all_tasks.append(t)

    train_log = pd.DataFrame()
    if args.no_train:
        adapter_model = model
    else:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        train_log, adapter_model = train_adapter(model, tokenizer, args, train_eps, run_dir)
        if hasattr(adapter_model.config, "use_cache"):
            adapter_model.config.use_cache = True

    for split, eps, modes in [
        ("synthetic_counterfactual", syn_eval, ["normal", "shuffled", "none", "contrast"]),
        ("public_prose", public_eval, ["normal", "shuffled", "none"]),
    ]:
        r, t = evaluate_episodes(adapter_model, tokenizer, eps, "adapter", split, modes, args.max_new_tokens, args.eval_seed + 200)
        all_rows.append(r)
        all_tasks.append(t)

    row_df = pd.concat(all_rows, ignore_index=True)
    task_df = pd.concat(all_tasks, ignore_index=True)
    summary = summarize(task_df, row_df)
    row_df.to_csv(run_dir / "row_predictions.csv", index=False)
    task_df.to_csv(run_dir / "task_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    train_log.to_csv(run_dir / "training_log.csv", index=False)
    plot_run(summary, train_log, task_df)
    elapsed = time.time() - start
    write_run_report(args, run_dir, summary, task_df, row_df, train_log, elapsed)
    append_log(
        f"- Elapsed seconds: `{elapsed:.1f}`\n"
        f"- Summary rows: `{len(summary)}`\n"
        f"- Latest run report: `{REPORTS / (EXP_NAME + '_latest_run_report.md')}`"
    )
    print(summary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="main_v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--eval_seed", type=int, default=20260702)
    ap.add_argument("--objective", choices=["ce", "support_contrastive"], default="support_contrastive")
    ap.add_argument("--train_mode", choices=["counterfactual", "ordinary", "shuffled_labels"], default="counterfactual")
    ap.add_argument("--train_episodes", type=int, default=800)
    ap.add_argument("--train_steps", type=int, default=100)
    ap.add_argument("--support_n", type=int, default=4)
    ap.add_argument("--query_n", type=int, default=2)
    ap.add_argument("--train_rows_per_episode", type=int, default=2)
    ap.add_argument("--synthetic_eval_pairs", type=int, default=30)
    ap.add_argument("--public_task_limit", type=int, default=45)
    ap.add_argument("--public_heldout_n", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--contrast_weight", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=0.75)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--no_train", action="store_true")
    ap.add_argument("--skip_base_eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_episodes = min(args.train_episodes, 16)
        args.train_steps = min(args.train_steps, 2)
        args.synthetic_eval_pairs = min(args.synthetic_eval_pairs, 2)
        args.public_task_limit = min(args.public_task_limit, 2)
        args.public_heldout_n = min(args.public_heldout_n, 3)
    run_one(args)


if __name__ == "__main__":
    main()

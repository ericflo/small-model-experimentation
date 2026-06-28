#!/usr/bin/env python3
"""Counterfactual episodic ICL posttraining for Qwen.

This standalone experiment trains a small LoRA adapter on synthetic
few-shot text-transformation episodes where the same input distribution
supports multiple incompatible rules. The evaluation asks whether the
adapter improves task induction from support examples, not just family
memorization.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
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
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


EXP_NAME = "qwen_counterfactual_episodic_icl"
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def clean_text(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def exact(a: str, b: str) -> bool:
    return clean_text(a) == clean_text(b)


def parse_first_line(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.split("<|endoftext|>")[0]
    text = text.split("<|im_end|>")[0]
    # Qwen3 often emits the answer and immediately starts a rationale in the
    # same line. These markers strip that rationale without using the target.
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


def pct(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{100 * float(x):.1f}%"


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
    counterfactual_rule: str | None = None


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


def split_alnum(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s)


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
        w = [rng.choice(WORDS), rng.choice(WORDS)]
        mode = rng.choice(["lower", "upper", "title", "mixed"])
        text = " ".join(w)
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
        delim = rng.choice(["|", "-", ","])
        return delim.join(parts)
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
    contrast_rule: Rule | None = None,
) -> Episode:
    if rule is None:
        rule = rng.choice(RULES)
    used: set[str] = set()

    def unique_input() -> str:
        for _ in range(100):
            s = make_input(rng, rule.family)
            if s not in used and rule.fn(s) != "":
                used.add(s)
                return s
        s = make_input(rng, rule.family)
        used.add(s)
        return s

    support = tuple(Example(x, rule.fn(x)) for x in (unique_input() for _ in range(support_n)))
    queries = tuple(Example(x, rule.fn(x)) for x in (unique_input() for _ in range(query_n)))
    return Episode(
        task_id=task_id,
        family=rule.family,
        rule_name=rule.name,
        support=support,
        queries=queries,
        counterfactual_rule=contrast_rule.name if contrast_rule else None,
    )


def counterfactual_eval_episodes(seed: int, pairs: int, support_n: int, query_n: int) -> list[Episode]:
    rng = random.Random(seed)
    out: list[Episode] = []
    by_family: dict[str, list[Rule]] = {}
    for r in RULES:
        by_family.setdefault(r.family, []).append(r)
    eligible_families = [f for f, rs in by_family.items() if len(rs) >= 2]
    for i in range(pairs):
        inputs: list[str] | None = None
        r1: Rule | None = None
        r2: Rule | None = None
        fam = ""
        # Some rule pairs, especially delimiter pairs with different delimiters,
        # cannot both produce nonempty distinct outputs on the same input. Bound
        # the search and pick another pair instead of spinning forever.
        for pair_attempt in range(300):
            fam = rng.choice(eligible_families)
            cand1, cand2 = rng.sample(by_family[fam], 2)
            base_rng = random.Random(seed * 100000 + i * 997 + pair_attempt)
            found: list[str] = []
            for _ in range(2000):
                s = make_input(base_rng, fam)
                if s in found:
                    continue
                y1 = cand1.fn(s)
                y2 = cand2.fn(s)
                if y1 != "" and y2 != "" and y1 != y2:
                    found.append(s)
                    if len(found) >= support_n + query_n:
                        break
            if len(found) >= support_n + query_n:
                inputs = found
                r1, r2 = cand1, cand2
                break
        if inputs is None or r1 is None or r2 is None:
            raise RuntimeError(f"Could not construct counterfactual pair {i} after bounded search")
        support_inputs = inputs[:support_n]
        query_inputs = inputs[support_n:]
        for j, r in enumerate([r1, r2]):
            support = tuple(Example(x, r.fn(x)) for x in support_inputs)
            queries = tuple(Example(x, r.fn(x)) for x in query_inputs)
            other = r2 if j == 0 else r1
            out.append(
                Episode(
                    task_id=f"cf_{i:04d}_{r.name}",
                    family=fam,
                    rule_name=r.name,
                    support=support,
                    queries=queries,
                    counterfactual_rule=other.name,
                )
            )
    return out


def train_episodes(seed: int, n: int, support_n: int, query_n: int) -> list[Episode]:
    rng = random.Random(seed)
    episodes: list[Episode] = []
    by_family: dict[str, list[Rule]] = {}
    for r in RULES:
        by_family.setdefault(r.family, []).append(r)
    for i in range(n):
        if i % 2 == 0:
            fam = rng.choice([f for f, rs in by_family.items() if len(rs) >= 2])
            r, other = rng.sample(by_family[fam], 2)
        else:
            r = rng.choice(RULES)
            candidates = [x for x in by_family[r.family] if x.name != r.name]
            other = rng.choice(candidates) if candidates else None
        episodes.append(make_episode(rng, f"train_{i:05d}", support_n, query_n, r, other))
    return episodes


def format_prompt(support: Iterable[Example], query_input: str, include_instruction: bool = True) -> str:
    lines: list[str] = []
    if include_instruction:
        lines.append("Infer the text transformation from the examples. Return only the output for the query.")
    lines.append("Examples:")
    for ex in support:
        lines.append(f"Input: {ex.inp}")
        lines.append(f"Output: {ex.out}")
    lines.append(f"Query Input: {query_input}")
    lines.append("Output:")
    return "\n".join(lines)


def shuffled_support(support: tuple[Example, ...], rng: random.Random) -> tuple[Example, ...]:
    outs = [ex.out for ex in support]
    rng.shuffle(outs)
    return tuple(Example(ex.inp, out) for ex, out in zip(support, outs))


class AnswerOnlyDataset(Dataset):
    def __init__(self, episodes: list[Episode], tokenizer, max_length: int, rows_per_episode: int):
        self.items: list[tuple[str, str]] = []
        for ep in episodes:
            for q in ep.queries[:rows_per_episode]:
                self.items.append((format_prompt(ep.support, q.inp), q.out))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt, answer = self.items[idx]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(" " + answer, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(ids) > self.max_length:
            cut = len(ids) - self.max_length
            ids = ids[cut:]
            labels = labels[cut:]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, mask = [], [], []
    for item in batch:
        n = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([torch.full((n,), pad_id, dtype=torch.long), item["input_ids"]]))
        labels.append(torch.cat([torch.full((n,), -100, dtype=torch.long), item["labels"]]))
        mask.append(torch.cat([torch.zeros(n, dtype=torch.long), item["attention_mask"]]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(mask),
    }


def load_prose_tasks(limit: int, train_n: int, heldout_n: int, seed: int) -> list[Episode]:
    tasks: list[Episode] = []
    for d in sorted(TRANSFORM_ROOT.iterdir()):
        if not d.is_dir() or not (d / "spec.json").exists():
            continue
        try:
            spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        except Exception:
            continue
        examples = spec.get("Examples", [])
        rows: list[Example] = []
        for ex in examples:
            inp = ex.get("Input", [])
            if isinstance(inp, list):
                inp_s = " | ".join(clean_text(x) for x in inp)
            else:
                inp_s = clean_text(inp)
            out_s = clean_text(ex.get("Output", ""))
            if inp_s and out_s:
                rows.append(Example(inp_s, out_s))
        # Deduplicate while preserving order; public synthetic data often repeats rows.
        seen: set[tuple[str, str]] = set()
        dedup: list[Example] = []
        for r in rows:
            key = (r.inp, r.out)
            if key not in seen:
                seen.add(key)
                dedup.append(r)
        if len(dedup) < train_n + heldout_n:
            continue
        family = d.name.split(".")[0]
        tasks.append(
            Episode(
                task_id=d.name,
                family=family,
                rule_name="public",
                support=tuple(dedup[:train_n]),
                queries=tuple(dedup[train_n : train_n + heldout_n]),
                counterfactual_rule=",".join(meta.get("Features", [])) if isinstance(meta.get("Features"), list) else None,
            )
        )
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks[:limit]


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


def evaluate_episodes(
    model,
    tokenizer,
    episodes: list[Episode],
    method: str,
    split: str,
    max_new_tokens: int,
    support_mode: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rng = random.Random(seed)
    for ep in episodes:
        if support_mode == "normal":
            support = ep.support
        elif support_mode == "shuffled":
            support = shuffled_support(ep.support, rng)
        elif support_mode == "none":
            support = tuple()
        else:
            raise ValueError(support_mode)
        for qi, q in enumerate(ep.queries):
            prompt = format_prompt(support, q.inp)
            pred = generate_one(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "support_mode": support_mode,
                    "task_id": ep.task_id,
                    "family": ep.family,
                    "rule_name": ep.rule_name,
                    "counterfactual_rule": ep.counterfactual_rule,
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
    keys = ["method", "split", "support_mode"]
    for key, g in task_df.groupby(keys):
        rg = row_df
        for k, v in zip(keys, key):
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
    ds = AnswerOnlyDataset(train_eps, tokenizer, args.max_length, rows_per_episode=args.train_rows_per_episode)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = args.train_steps
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, args.warmup_steps), num_training_steps=total_updates)
    log_rows = []
    step = 0
    micro_step = 0
    accum_loss = 0.0
    opt.zero_grad(set_to_none=True)
    while step < total_updates:
        for batch in dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            micro_step += 1
            accum_loss += float(loss.detach().cpu()) * args.grad_accum
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == 1:
                    lr = sched.get_last_lr()[0]
                    denom = args.log_every * args.grad_accum if step > 1 else args.grad_accum
                    avg_loss = accum_loss / max(1, denom)
                    print(f"step {step}/{total_updates} loss {avg_loss:.4f} lr {lr:.2e}", flush=True)
                    log_rows.append({"step": step, "loss": avg_loss, "lr": lr})
                    accum_loss = 0.0
                if step >= total_updates:
                    break
    model.eval()
    ckpt = LARGE_ROOT / "checkpoints" / args.run_name / "adapter"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)
    return pd.DataFrame(log_rows), model


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if cols:
        df = df[cols]
    tmp = df.copy()
    for c in tmp.columns:
        if "exact" in c or c.endswith("_rate") or c.endswith("_delta"):
            tmp[c] = tmp[c].map(lambda x: pct(x) if pd.notna(x) else "")
    return tmp.to_markdown(index=False)


def plot_results(summary: pd.DataFrame, train_log: pd.DataFrame, task_df: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    label = summary["method"] + "\n" + summary["split"] + "\n" + summary["support_mode"]
    order = summary.sort_values("full_task_exact", ascending=False).index
    plt.figure(figsize=(13, 7))
    plt.bar(range(len(order)), summary.loc[order, "full_task_exact"], color="#3a6ea5")
    plt.xticks(range(len(order)), label.loc[order], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Full-task exact")
    plt.ylim(0, 1)
    plt.title("Strict Full-Task Exact by Method")
    plt.tight_layout()
    plt.savefig(FIGS / "full_task_exact_by_method.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for split, g in summary.groupby("split"):
        plt.scatter(g["row_exact"], g["full_task_exact"], s=90, label=split)
        for _, r in g.iterrows():
            plt.annotate(r["method"].replace("_", " "), (r["row_exact"], r["full_task_exact"]), fontsize=7)
    plt.xlabel("Row exact")
    plt.ylabel("Full-task exact")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Row Accuracy Versus Task Consistency")
    plt.tight_layout()
    plt.savefig(FIGS / "row_vs_full_task.png", dpi=180)
    plt.close()

    if len(train_log):
        plt.figure(figsize=(9, 5))
        plt.plot(train_log["step"], train_log["loss"], marker="o")
        plt.xlabel("Optimizer step")
        plt.ylabel("Training loss")
        plt.title("LoRA Training Loss")
        plt.tight_layout()
        plt.savefig(FIGS / "training_loss.png", dpi=180)
        plt.close()

    piv = task_df.pivot_table(index="task_id", columns=["method", "support_mode"], values="full_task_exact", aggfunc="mean")
    if ("base", "normal") in piv.columns and ("adapter", "normal") in piv.columns:
        delta = piv[("adapter", "normal")].fillna(0).astype(float) - piv[("base", "normal")].fillna(0).astype(float)
        counts = pd.Series({"helped": (delta > 0).sum(), "hurt": (delta < 0).sum(), "tied": (delta == 0).sum()})
        plt.figure(figsize=(7, 5))
        plt.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51", "#9aa0a6"])
        plt.ylabel("Tasks")
        plt.title("Adapter Task Flips Versus Base")
        plt.tight_layout()
        plt.savefig(FIGS / "adapter_task_flips.png", dpi=180)
        plt.close()


def write_report(args, run_dir: Path, summary: pd.DataFrame, task_df: pd.DataFrame, row_df: pd.DataFrame, train_log: pd.DataFrame, elapsed: float) -> None:
    report_md = REPORTS / f"{EXP_NAME}_report.md"
    report_html = REPORTS / f"{EXP_NAME}_report.html"

    def metric(method: str, split: str, support: str, col: str) -> float | None:
        m = summary[(summary.method == method) & (summary.split == split) & (summary.support_mode == support)]
        if len(m):
            return float(m.iloc[0][col])
        return None

    base_syn = metric("base", "synthetic_counterfactual", "normal", "full_task_exact")
    ad_syn = metric("adapter", "synthetic_counterfactual", "normal", "full_task_exact")
    shuf_syn = metric("adapter", "synthetic_counterfactual", "shuffled", "full_task_exact")
    base_pub = metric("base", "public_prose", "normal", "full_task_exact")
    ad_pub = metric("adapter", "public_prose", "normal", "full_task_exact")
    shuf_pub = metric("adapter", "public_prose", "shuffled", "full_task_exact")

    lines: list[str] = []
    lines.append("# Counterfactual Episodic ICL Posttraining")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Can answer-only LoRA posttraining on counterfactual few-shot episodes improve a 4B model's ability to infer a task from support examples, rather than relying on a task-family prior?")
    lines.append("")
    lines.append("The training episodes are synthetic and deliberately counterfactual: the same kind of query input can require incompatible outputs depending on the support examples. Public benchmark outputs are used only for evaluation.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Run: `{args.run_name}`")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Train episodes: `{args.train_episodes}`")
    lines.append(f"- Train steps: `{args.train_steps}`")
    lines.append(f"- LoRA rank: `{args.lora_r}`")
    lines.append(f"- Synthetic eval counterfactual pairs: `{args.synthetic_eval_pairs}`")
    lines.append(f"- Public PROSE tasks: `{args.public_task_limit}`")
    lines.append(f"- Elapsed seconds: `{elapsed:.1f}`")
    lines.append("")
    lines.append("## Main Result")
    lines.append("")
    lines.append(md_table(summary, ["method", "split", "support_mode", "tasks", "rows", "row_exact", "full_task_exact"]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if ad_syn is not None and base_syn is not None:
        lines.append(f"On held-out synthetic counterfactual episodes, the adapter changes full-task exactness from `{pct(base_syn)}` to `{pct(ad_syn)}`. With shuffled support examples, the adapter scores `{pct(shuf_syn)}`.")
    if ad_pub is not None and base_pub is not None:
        lines.append(f"On the public text-transformation sample, the adapter changes full-task exactness from `{pct(base_pub)}` to `{pct(ad_pub)}`. With shuffled support examples, the adapter scores `{pct(shuf_pub)}`.")
    lines.append("")
    if ad_syn is not None and base_syn is not None and ad_pub is not None and base_pub is not None:
        syn_gain = ad_syn - base_syn
        pub_gain = ad_pub - base_pub
        if syn_gain > 0.05 and (shuf_syn is not None and shuf_syn < ad_syn - 0.05):
            lines.append("The synthetic split shows a real support-conditioning effect: performance improves and shuffled support degrades it. The public split determines whether that learned behavior transfers outside the synthetic generator.")
        elif syn_gain > 0.05:
            lines.append("The adapter improves the synthetic split, but the shuffled-support control is too close to the normal-support result. That pattern is not enough to prove genuine task induction.")
        else:
            lines.append("The adapter does not produce a clear synthetic counterfactual gain. That is a negative result for this posttraining recipe at the tested scale.")
        if pub_gain > 0.05:
            lines.append("The public benchmark transfer is positive at the tested scale.")
        elif abs(pub_gain) <= 0.025:
            lines.append("The public benchmark transfer is roughly flat at the tested scale.")
        else:
            lines.append("The public benchmark transfer is negative at the tested scale.")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for name, caption in [
        ("full_task_exact_by_method.png", "Strict full-task exact by method"),
        ("row_vs_full_task.png", "Row exact versus full-task exact"),
        ("training_loss.png", "Training loss"),
        ("adapter_task_flips.png", "Adapter task flips versus base"),
    ]:
        if (FIGS / name).exists():
            lines.append(f"![{caption}](../analysis/figures/{name})")
            lines.append("")
    lines.append("## Task-Level Details")
    lines.append("")
    lines.append(md_table(task_df.sort_values(["split", "task_id", "method", "support_mode"]).head(120)))
    lines.append("")
    lines.append("## Error Examples")
    lines.append("")
    misses = row_df[(row_df.method == "adapter") & (row_df.support_mode == "normal") & (~row_df.exact)].head(40)
    cols = ["split", "task_id", "family", "rule_name", "input", "target", "prediction"]
    lines.append(md_table(misses[cols] if len(misses) else pd.DataFrame(columns=cols)))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Adapter checkpoint: `{LARGE_ROOT / 'checkpoints' / args.run_name / 'adapter'}`")
    lines.append(f"- Row predictions: `{run_dir / 'row_predictions.csv'}`")
    lines.append(f"- Task metrics: `{run_dir / 'task_metrics.csv'}`")
    lines.append(f"- Summary: `{run_dir / 'summary.csv'}`")
    lines.append(f"- Training log: `{run_dir / 'training_log.csv'}`")
    lines.append(f"- Large artifacts directory: `{LARGE_ROOT}`")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This run trains on synthetic counterfactual transformations, so public benchmark transfer is the decisive external signal. The public evaluation is capped for runtime. Exact-match scoring is intentionally strict and does not award partial credit for near-correct formats.")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    html_body = "\n".join(
        f"<p>{html.escape(line)}</p>" if line and not line.startswith("|") and not line.startswith("#") and not line.startswith("![") and not line.startswith("- ")
        else line
        for line in lines
    )
    # Use markdown conversion when available; fallback keeps readable HTML with images.
    try:
        import markdown  # type: ignore

        html_body = markdown.markdown("\n".join(lines), extensions=["tables"])
    except Exception:
        html_body = html_body.replace("\n", "\n<br>")
        for name in ["full_task_exact_by_method.png", "row_vs_full_task.png", "training_loss.png", "adapter_task_flips.png"]:
            html_body = html_body.replace(
                f"![{name}](../analysis/figures/{name})",
                f"<img src='../analysis/figures/{name}' alt='{name}'>",
            )
    report_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Counterfactual Episodic ICL Posttraining</title>"
        "<style>body{font-family:Inter,Arial,sans-serif;max-width:1180px;margin:32px auto;line-height:1.45;color:#1f2937}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d1d5db;padding:4px 7px}th{background:#f3f4f6}"
        "img{max-width:100%;margin:14px 0;border:1px solid #e5e7eb}</style></head><body>"
        + html_body
        + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="main_v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--train_episodes", type=int, default=900)
    ap.add_argument("--train_steps", type=int, default=240)
    ap.add_argument("--support_n", type=int, default=4)
    ap.add_argument("--query_n", type=int, default=2)
    ap.add_argument("--train_rows_per_episode", type=int, default=2)
    ap.add_argument("--synthetic_eval_pairs", type=int, default=36)
    ap.add_argument("--public_task_limit", type=int, default=36)
    ap.add_argument("--public_heldout_n", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--no_train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.train_episodes = min(args.train_episodes, 16)
        args.train_steps = min(args.train_steps, 2)
        args.synthetic_eval_pairs = min(args.synthetic_eval_pairs, 3)
        args.public_task_limit = min(args.public_task_limit, 4)

    ensure_dirs()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    append_log(
        f"\n## Run `{args.run_name}`\n\n"
        f"- Time UTC: `{now_iso()}`\n"
        f"- Config: `{json.dumps(vars(args), sort_keys=True)}`"
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_eps = train_episodes(args.seed, args.train_episodes, args.support_n, args.query_n)
    syn_eval = counterfactual_eval_episodes(args.seed + 17, args.synthetic_eval_pairs, args.support_n, args.query_n)
    public_eval = load_prose_tasks(args.public_task_limit, args.support_n, args.public_heldout_n, args.seed + 33)

    pd.DataFrame(
        [
            {
                "task_id": ep.task_id,
                "family": ep.family,
                "rule_name": ep.rule_name,
                "counterfactual_rule": ep.counterfactual_rule,
                "support_json": json.dumps([ex.__dict__ for ex in ep.support]),
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
    for split, eps in [("synthetic_counterfactual", syn_eval), ("public_prose", public_eval)]:
        for support_mode in ["normal", "shuffled"]:
            r, t = evaluate_episodes(model, tokenizer, eps, "base", split, args.max_new_tokens, support_mode, args.seed + 100)
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

    for split, eps in [("synthetic_counterfactual", syn_eval), ("public_prose", public_eval)]:
        for support_mode in ["normal", "shuffled", "none"]:
            r, t = evaluate_episodes(adapter_model, tokenizer, eps, "adapter", split, args.max_new_tokens, support_mode, args.seed + 200)
            all_rows.append(r)
            all_tasks.append(t)

    row_df = pd.concat(all_rows, ignore_index=True)
    task_df = pd.concat(all_tasks, ignore_index=True)
    summary = summarize(task_df, row_df)
    row_df.to_csv(run_dir / "row_predictions.csv", index=False)
    task_df.to_csv(run_dir / "task_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    shutil.copy(run_dir / "summary.csv", ANALYSIS / "summary.csv")
    train_log.to_csv(run_dir / "training_log.csv", index=False)

    plot_results(summary, train_log, task_df)
    elapsed = time.time() - start
    write_report(args, run_dir, summary, task_df, row_df, train_log, elapsed)
    append_log(
        f"- Elapsed seconds: `{elapsed:.1f}`\n"
        f"- Synthetic adapter full-task exact: `{pct(float(summary[(summary.method == 'adapter') & (summary.split == 'synthetic_counterfactual') & (summary.support_mode == 'normal')]['full_task_exact'].iloc[0]))}`\n"
        f"- Public adapter full-task exact: `{pct(float(summary[(summary.method == 'adapter') & (summary.split == 'public_prose') & (summary.support_mode == 'normal')]['full_task_exact'].iloc[0]))}`\n"
        f"- Report: `{REPORTS / (EXP_NAME + '_report.md')}`"
    )
    print(summary.to_string(index=False))
    print(f"Report: {REPORTS / (EXP_NAME + '_report.md')}")


if __name__ == "__main__":
    main()

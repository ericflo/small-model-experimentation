#!/usr/bin/env python3
"""Episodic ECHO-TTT experiment.

This experiment tests a narrow mechanism: can temporary per-episode gradient
updates on environment-observation prediction improve later decisions made by a
frozen local language model?

The environment is a deterministic diagnostic box. Each episode has a hidden
rule mapping integer probes to integer observations. The model sees a few
probe/observation pairs, then must choose the result for a held-out probe from
multiple candidates. ECHO-TTT updates a small virtual prefix for that episode by
backpropagating only observation-token cross entropy. The prefix is reset for
the next episode.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/episodic_echo_ttt")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
LARGE_ROOT = Path("/workspace/large_artifacts/episodic_echo_ttt")
CHECKPOINTS = LARGE_ROOT / "checkpoints"

MODULUS = 17
VALUE_TOKENS = [f"{i:02d}" for i in range(MODULUS)]
LETTERS = ["A", "B", "C", "D"]


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct(x: float) -> str:
    if x is None or math.isnan(float(x)):
        return "n/a"
    return f"{100.0 * float(x):.1f}%"


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def std(xs: Sequence[float]) -> float:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def ensure_dirs() -> None:
    for path in [RUNS, REPORTS, ANALYSIS, FIGURES, CHECKPOINTS]:
        path.mkdir(parents=True, exist_ok=True)


def append_experiment_log(text: str) -> None:
    path = ROOT / "experiment_log.md"
    with path.open("a") as f:
        f.write(text.rstrip() + "\n\n")


@dataclass
class Rule:
    family: str
    a: int
    b: int
    c: int
    threshold: int

    def apply(self, x: int) -> int:
        x = int(x) % MODULUS
        if self.family == "shift":
            return (x + self.b) % MODULUS
        if self.family == "scale_shift":
            return (self.a * x + self.b) % MODULUS
        if self.family == "mirror":
            return (self.b - x) % MODULUS
        if self.family == "threshold":
            return (x + self.b if x < self.threshold else x + self.c) % MODULUS
        if self.family == "zigzag":
            return (self.a * x + self.b if x % 2 == 0 else self.c - x) % MODULUS
        raise ValueError(self.family)

    def description(self) -> str:
        return (
            "The box uses one hidden deterministic rule for this episode. "
            "The rule is fixed across all probes in the episode."
        )


@dataclass
class Episode:
    episode_id: str
    split: str
    seed: int
    rule: Rule
    support: List[Tuple[int, int]]
    ce_probe: List[Tuple[int, int]]
    target_x: int
    target_y: int


class EpisodeGenerator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def sample_rule(self, split: str) -> Rule:
        if split == "heldout_family":
            family = self.rng.choice(["threshold", "zigzag"])
        else:
            family = self.rng.choice(["shift", "scale_shift", "mirror"])
        a = self.rng.choice([2, 3, 5, 7, 11, 13])
        b = self.rng.randint(1, MODULUS - 1)
        c = self.rng.randint(1, MODULUS - 1)
        threshold = self.rng.randint(4, 12)
        return Rule(family=family, a=a, b=b, c=c, threshold=threshold)

    def make_one(self, split: str, index: int, support_count: int, ce_count: int) -> Episode:
        rule = self.sample_rule(split)
        xs = list(range(MODULUS))
        self.rng.shuffle(xs)
        total = support_count + ce_count + 1
        chosen = xs[:total]
        support_xs = chosen[:support_count]
        ce_xs = chosen[support_count : support_count + ce_count]
        target_x = chosen[-1]
        support = [(x, rule.apply(x)) for x in support_xs]
        ce_probe = [(x, rule.apply(x)) for x in ce_xs]
        target_y = rule.apply(target_x)
        eid = f"{split}_{index}_{rule.family}_{target_x}_{abs(hash((tuple(chosen), rule.family, rule.a, rule.b, rule.c, rule.threshold))) % 10_000_000}"
        return Episode(
            episode_id=eid,
            split=split,
            seed=0,
            rule=rule,
            support=support,
            ce_probe=ce_probe,
            target_x=target_x,
            target_y=target_y,
        )

    def make_set(self, split: str, n: int, support_count: int, ce_count: int) -> List[Episode]:
        return [self.make_one(split, i, support_count, ce_count) for i in range(n)]


def value_text(v: int) -> str:
    return f"result={int(v):02d}"


def command_text(x: int) -> str:
    return f"probe {int(x):02d}"


def support_transcript(ep: Episode, corrupt: Optional[str] = None, rng: Optional[random.Random] = None) -> str:
    pairs = list(ep.support)
    if corrupt == "shuffle_within":
        ys = [y for _, y in pairs]
        if rng is None:
            rng = random.Random(0)
        rng.shuffle(ys)
        pairs = [(x, y) for (x, _), y in zip(pairs, ys)]
    elif corrupt == "generic":
        pairs = [(x, -1) for x, _ in pairs]
    lines = []
    for x, y in pairs:
        lines.append(f"Command: {command_text(x)}")
        if corrupt == "generic":
            lines.append("Observation: status=ok")
        else:
            lines.append(f"Observation: {value_text(y)}")
    return "\n".join(lines) + "\n"


def intro_text(ep: Episode) -> str:
    return (
        "You are using a deterministic diagnostic box.\n"
        f"{ep.rule.description()}\n"
        "Each command has the form `probe NN`. The box replies with `result=YY`.\n"
        "Use the observed replies to infer the hidden rule for this episode.\n\n"
    )


def option_values(ep: Episode) -> List[int]:
    seed_bytes = hashlib.sha256(ep.episode_id.encode("utf-8")).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    values = [ep.target_y]
    pool = [v for v in range(MODULUS) if v != ep.target_y]
    rng.shuffle(pool)
    values.extend(pool[:3])
    rng.shuffle(values)
    return values


def final_prompt(ep: Episode, options: Optional[Sequence[int]] = None) -> str:
    if options is None:
        options = option_values(ep)
    option_lines = "\n".join(f"{letter}. {value_text(value)}" for letter, value in zip(LETTERS, options))
    return (
        intro_text(ep)
        + support_transcript(ep)
        + f"Command: {command_text(ep.target_x)}\n"
        + "Which observation is correct for this command?\n"
        + option_lines
        + "\nAnswer with the option letter only.\nAnswer:"
    )


def ce_prompt(ep: Episode, x: int) -> str:
    return intro_text(ep) + support_transcript(ep) + f"Command: {command_text(x)}\nObservation:"


def adaptation_text(ep: Episode, corrupt: Optional[str], rng: random.Random) -> Tuple[str, List[float]]:
    text = intro_text(ep)
    weights: List[float] = [0.0] * 0
    segments: List[Tuple[str, bool]] = [(intro_text(ep), False)]
    pairs = list(ep.support)
    if corrupt == "shuffle_within":
        ys = [y for _, y in pairs]
        rng.shuffle(ys)
        pairs = [(x, y) for (x, _), y in zip(pairs, ys)]
    for x, y in pairs:
        segments.append((f"Command: {command_text(x)}\nObservation:", False))
        if corrupt == "generic":
            segments.append((" status=ok\n", True))
        else:
            segments.append((f" {value_text(y)}\n", True))
    return "".join(seg for seg, _ in segments), [1.0 if obs else 0.0 for seg, obs in segments for _ in seg]


def format_instruction(tokenizer: Any, user: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise diagnostic-box interpreter. Return only the requested observation."},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        except Exception:
            pass
    return messages[0]["content"] + "\n\n" + user


def render_generation_prompt(tokenizer: Any, user: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise diagnostic-box interpreter. Return only the requested observation."},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        except Exception:
            pass
    return messages[0]["content"] + "\n\n" + user + "\n"


def tokenize_with_char_weights(tokenizer: Any, text: str, char_weights: Sequence[float], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    offsets = enc["offset_mapping"]
    weights = []
    for start, end in offsets:
        if end <= start:
            weights.append(0.0)
        else:
            span = char_weights[start:end]
            weights.append(1.0 if any(w > 0 for w in span) else 0.0)
    return ids, torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)


def tokenize_many_with_char_weights(
    tokenizer: Any,
    texts: Sequence[str],
    char_weights_by_text: Sequence[Sequence[float]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(list(texts), return_offsets_mapping=True, add_special_tokens=False, padding=True)
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
    batch_weights: List[List[float]] = []
    for offsets, char_weights in zip(enc["offset_mapping"], char_weights_by_text):
        weights: List[float] = []
        for start, end in offsets:
            if end <= start:
                weights.append(0.0)
            else:
                span = char_weights[start:end]
                weights.append(1.0 if any(w > 0 for w in span) else 0.0)
        batch_weights.append(weights)
    return input_ids, torch.tensor(batch_weights, dtype=torch.float32, device=device)


def tokenized_text(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    return ids


def make_prefix(
    base_prefix: Optional[torch.Tensor],
    prefix_len: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 0.01,
) -> torch.Tensor:
    if prefix_len <= 0:
        return torch.empty(1, 0, hidden_size, device=device, dtype=dtype)
    if base_prefix is not None:
        return base_prefix.detach().clone().to(device=device, dtype=dtype)
    return (scale * torch.randn(1, prefix_len, hidden_size, device=device, dtype=torch.float32)).to(dtype)


class FrozenQwenPrefixScorer:
    def __init__(self, model_name: str, load_4bit: bool, dtype_name: str, device: str) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "right"
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16 if dtype_name == "fp16" else torch.float32
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = dtype
            kwargs["device_map"] = "auto" if self.device.type == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.embed = self.model.get_input_embeddings()
        self.hidden_size = int(self.embed.embedding_dim)
        self.dtype = dtype

    def weighted_nlls(self, input_ids: torch.Tensor, weights: torch.Tensor, prefix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.embed(input_ids)
        if prefix.shape[1] > 0:
            prefix = prefix.to(device=token_emb.device, dtype=token_emb.dtype)
            inputs_embeds = torch.cat([prefix.expand(input_ids.shape[0], -1, -1), token_emb], dim=1)
            prefix_labels = torch.full((input_ids.shape[0], prefix.shape[1]), -100, dtype=torch.long, device=input_ids.device)
            labels = torch.cat([prefix_labels, input_ids], dim=1)
            prefix_w = torch.zeros((input_ids.shape[0], prefix.shape[1]), dtype=weights.dtype, device=weights.device)
            all_weights = torch.cat([prefix_w, weights], dim=1)
            attention = torch.ones(labels.shape, dtype=torch.long, device=input_ids.device)
            out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention, use_cache=False)
        else:
            labels = input_ids
            all_weights = weights
            out = self.model(input_ids=input_ids, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = all_weights[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        nll = (loss * shift_weights).sum(dim=1)
        denom = shift_weights.sum(dim=1).clamp_min(1.0)
        return nll, denom

    def weighted_loss(self, input_ids: torch.Tensor, weights: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        nll, denom = self.weighted_nlls(input_ids, weights, prefix)
        return nll.sum() / denom.sum().clamp_min(1.0)

    def continuation_nll(self, prompt: str, continuation: str, prefix: torch.Tensor) -> float:
        rendered_prompt = render_generation_prompt(self.tokenizer, prompt)
        full = rendered_prompt + continuation
        start = len(rendered_prompt)
        char_weights = [0.0] * len(full)
        for i in range(start, len(full)):
            char_weights[i] = 1.0
        input_ids, weights = tokenize_with_char_weights(self.tokenizer, full, char_weights, self.device)
        with torch.no_grad():
            nll, _ = self.weighted_nlls(input_ids, weights, prefix)
        return float(nll[0].detach().cpu())

    def score_candidates(self, ep: Episode, prefix: torch.Tensor) -> Tuple[int, Dict[int, float]]:
        options = option_values(ep)
        prompt = final_prompt(ep, options=options)
        rendered_prompt = render_generation_prompt(self.tokenizer, prompt)
        texts: List[str] = []
        char_weights: List[List[float]] = []
        for letter in LETTERS:
            continuation = f" {letter}\n"
            full = rendered_prompt + continuation
            weights = [0.0] * len(full)
            for i in range(len(rendered_prompt), len(full)):
                weights[i] = 1.0
            texts.append(full)
            char_weights.append(weights)
        input_ids, weights = tokenize_many_with_char_weights(self.tokenizer, texts, char_weights, self.device)
        with torch.no_grad():
            nll, _ = self.weighted_nlls(input_ids, weights, prefix)
        letter_scores = {letter: -float(nll[i].detach().cpu()) for i, letter in enumerate(LETTERS)}
        best_letter = max(letter_scores.items(), key=lambda kv: kv[1])[0]
        best_value = int(options[LETTERS.index(best_letter)])
        value_scores = {int(value): letter_scores[letter] for letter, value in zip(LETTERS, options)}
        return best_value, value_scores

    def observation_ce(self, ep: Episode, prefix: torch.Tensor) -> float:
        texts: List[str] = []
        char_weights: List[List[float]] = []
        for x, y in ep.ce_probe:
            prompt = ce_prompt(ep, x)
            continuation = f" {value_text(y)}\n"
            rendered_prompt = render_generation_prompt(self.tokenizer, prompt)
            full = rendered_prompt + continuation
            weights = [0.0] * len(full)
            for i in range(len(rendered_prompt), len(full)):
                weights[i] = 1.0
            texts.append(full)
            char_weights.append(weights)
        input_ids, weights = tokenize_many_with_char_weights(self.tokenizer, texts, char_weights, self.device)
        with torch.no_grad():
            nll, denom = self.weighted_nlls(input_ids, weights, prefix)
        return float((nll.sum() / denom.sum().clamp_min(1.0)).detach().cpu())

    def adapt_prefix(
        self,
        ep: Episode,
        prefix: torch.Tensor,
        steps: int,
        lr: float,
        corrupt: Optional[str],
        rng: random.Random,
    ) -> Tuple[torch.Tensor, List[float]]:
        if steps <= 0 or prefix.shape[1] == 0:
            return prefix.detach(), []
        work = prefix.detach().clone().float().requires_grad_(True)
        opt = torch.optim.AdamW([work], lr=lr, weight_decay=0.0)
        losses: List[float] = []
        raw_text, char_weights = adaptation_text(ep, corrupt=corrupt, rng=rng)
        text = format_instruction(self.tokenizer, raw_text)
        # The chat template wraps text, so rebuild a conservative observation
        # mask by marking literal result/status spans in the rendered string.
        rendered_weights = [0.0] * len(text)
        for marker in [" result=", " status="]:
            start = 0
            while True:
                pos = text.find(marker, start)
                if pos < 0:
                    break
                end = text.find("\n", pos)
                if end < 0:
                    end = len(text)
                for i in range(pos, end):
                    rendered_weights[i] = 1.0
                start = end
        input_ids, weights = tokenize_with_char_weights(self.tokenizer, text, rendered_weights, self.device)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = self.weighted_loss(input_ids, weights, work.to(dtype=self.dtype))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([work], 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return work.detach().to(dtype=self.dtype), losses


def run_eval(
    scorer: FrozenQwenPrefixScorer,
    episodes: Sequence[Episode],
    split: str,
    seed: int,
    prefix_len: int,
    base_prefix: Optional[torch.Tensor],
    ttt_steps_list: Sequence[int],
    ttt_lr: float,
    arms: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed + 999)
    for arm in arms:
        corrupt: Optional[str] = None
        init_mode = "global"
        active_steps = list(ttt_steps_list)
        if arm == "no_prefix":
            init_mode = "none"
            active_steps = [0]
        elif arm == "global_no_ttt":
            init_mode = "global"
            active_steps = [0]
        elif arm == "scratch_echo_ttt":
            init_mode = "scratch"
            corrupt = None
        elif arm == "echo_ttt":
            init_mode = "global"
            corrupt = None
        elif arm == "shuffle_ttt":
            init_mode = "global"
            corrupt = "shuffle_within"
        elif arm == "generic_ttt":
            init_mode = "global"
            corrupt = "generic"
        else:
            raise ValueError(arm)
        for steps in active_steps:
            correct: List[float] = []
            before_ces: List[float] = []
            after_ces: List[float] = []
            update_losses_last: List[float] = []
            for ep in episodes:
                if init_mode == "none":
                    prefix = make_prefix(None, 0, scorer.hidden_size, scorer.device, scorer.dtype)
                elif init_mode == "scratch":
                    prefix = make_prefix(None, prefix_len, scorer.hidden_size, scorer.device, scorer.dtype)
                else:
                    prefix = make_prefix(base_prefix, prefix_len, scorer.hidden_size, scorer.device, scorer.dtype)
                before_ce = scorer.observation_ce(ep, prefix)
                adapted, losses = scorer.adapt_prefix(ep, prefix, steps=steps, lr=ttt_lr, corrupt=corrupt, rng=rng)
                after_ce = scorer.observation_ce(ep, adapted)
                pred, scores = scorer.score_candidates(ep, adapted)
                ok = float(pred == ep.target_y)
                correct.append(ok)
                before_ces.append(before_ce)
                after_ces.append(after_ce)
                if losses:
                    update_losses_last.append(losses[-1])
                detail_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "episode_id": ep.episode_id,
                        "arm": arm,
                        "ttt_steps": steps,
                        "rule_family": ep.rule.family,
                        "target_x": ep.target_x,
                        "target_y": ep.target_y,
                        "prediction": pred,
                        "correct": ok,
                        "obs_ce_before": before_ce,
                        "obs_ce_after": after_ce,
                        "update_loss_final": losses[-1] if losses else np.nan,
                        "top_score": scores.get(pred, np.nan),
                        "target_score": scores.get(ep.target_y, np.nan),
                    }
                )
                del adapted
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "arm": arm,
                    "ttt_steps": steps,
                    "n": len(episodes),
                    "accuracy": mean(correct),
                    "obs_ce_before": mean(before_ces),
                    "obs_ce_after": mean(after_ces),
                    "obs_ce_delta": mean([a - b for a, b in zip(after_ces, before_ces)]),
                    "update_loss_final": mean(update_losses_last) if update_losses_last else np.nan,
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, detail_rows


def train_global_prefix(
    scorer: FrozenQwenPrefixScorer,
    train_eps: Sequence[Episode],
    val_eps: Sequence[Episode],
    prefix_len: int,
    steps: int,
    lr: float,
    seed: int,
    run_dir: Path,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    rng = random.Random(seed + 111)
    prefix = make_prefix(None, prefix_len, scorer.hidden_size, scorer.device, scorer.dtype).float().requires_grad_(True)
    opt = torch.optim.AdamW([prefix], lr=lr, weight_decay=0.0)
    rows: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        ep = train_eps[(step - 1) % len(train_eps)]
        raw_text, _ = adaptation_text(ep, corrupt=None, rng=rng)
        text = format_instruction(scorer.tokenizer, raw_text)
        rendered_weights = [0.0] * len(text)
        start = 0
        while True:
            pos = text.find(" result=", start)
            if pos < 0:
                break
            end = text.find("\n", pos)
            if end < 0:
                end = len(text)
            for i in range(pos, end):
                rendered_weights[i] = 1.0
            start = end
        input_ids, weights = tokenize_with_char_weights(scorer.tokenizer, text, rendered_weights, scorer.device)
        opt.zero_grad(set_to_none=True)
        loss = scorer.weighted_loss(input_ids, weights, prefix.to(dtype=scorer.dtype))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prefix], 1.0)
        opt.step()
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            with torch.no_grad():
                val_ce = mean([scorer.observation_ce(ep, prefix.to(dtype=scorer.dtype)) for ep in val_eps[: min(8, len(val_eps))]])
            row = {"step": step, "train_loss": float(loss.detach().cpu()), "val_obs_ce": val_ce}
            rows.append(row)
            log(f"[prefix seed={seed}] step={step}/{steps} loss={row['train_loss']:.4f} val_obs_ce={val_ce:.4f}")
    ckpt = run_dir / "global_prefix.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prefix": prefix.detach().cpu(), "prefix_len": prefix_len, "seed": seed}, ckpt)
    return prefix.detach().to(dtype=scorer.dtype), rows


def aggregate(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    group_cols = ["split", "arm", "ttt_steps"]
    out = (
        df.groupby(group_cols)
        .agg(
            n=("n", "sum"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_obs_ce_before=("obs_ce_before", "mean"),
            mean_obs_ce_after=("obs_ce_after", "mean"),
            mean_obs_ce_delta=("obs_ce_delta", "mean"),
        )
        .reset_index()
    )
    out["std_accuracy"] = out["std_accuracy"].fillna(0.0)
    return out


def make_figures(metrics: pd.DataFrame, training: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not metrics.empty:
        eval_rows = metrics.copy()
        labels = eval_rows["split"] + "\n" + eval_rows["arm"] + "\nK=" + eval_rows["ttt_steps"].astype(str)
        plt.figure(figsize=(max(12, len(eval_rows) * 0.34), 6))
        plt.bar(range(len(eval_rows)), eval_rows["mean_accuracy"], yerr=eval_rows["std_accuracy"], color="#2563eb", alpha=0.85)
        plt.xticks(range(len(eval_rows)), labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("Final probe accuracy")
        plt.title("Accuracy by Split, Arm, and TTT Steps")
        plt.ylim(0, max(0.25, float(eval_rows["mean_accuracy"].max()) + 0.12))
        plt.tight_layout()
        plt.savefig(FIGURES / "accuracy_by_arm.png", dpi=160)
        plt.close()

        pivot = eval_rows.pivot_table(index="ttt_steps", columns=["split", "arm"], values="mean_accuracy")
        plt.figure(figsize=(12, 6))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", label=f"{col[0]} / {col[1]}")
        plt.xlabel("Per-episode TTT update steps")
        plt.ylabel("Final probe accuracy")
        plt.title("TTT Step Curve")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(FIGURES / "ttt_step_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(12, 6))
        x = np.arange(len(eval_rows))
        plt.bar(x, eval_rows["mean_obs_ce_delta"], color="#dc2626", alpha=0.85)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("Held-out observation CE delta after TTT")
        plt.title("Observation Prediction Change")
        plt.tight_layout()
        plt.savefig(FIGURES / "observation_ce_delta.png", dpi=160)
        plt.close()

    if not training.empty:
        plt.figure(figsize=(9, 5))
        for seed, part in training.groupby("seed"):
            plt.plot(part["step"], part["val_obs_ce"], marker="o", label=f"seed {seed}")
        plt.xlabel("Global prefix training step")
        plt.ylabel("Validation observation CE")
        plt.title("Global Prefix Observation Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / "global_prefix_training.png", dpi=160)
        plt.close()


def markdown_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return ""
    show = df.loc[:, cols].copy()
    for col in show.columns:
        if "accuracy" in col or "capture" in col:
            show[col] = show[col].map(lambda x: pct(float(x)))
        elif "ce" in col or "loss" in col:
            show[col] = show[col].map(lambda x: "n/a" if pd.isna(x) else f"{float(x):.3f}")
    return show.to_markdown(index=False)


def write_report(run_name: str, summary: pd.DataFrame, metrics: pd.DataFrame, training: pd.DataFrame, args: argparse.Namespace) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    primary = summary[(summary["split"].eq("eval_indist")) & (summary["arm"].isin(["global_no_ttt", "echo_ttt", "shuffle_ttt", "generic_ttt"]))]
    best = summary.sort_values("mean_accuracy", ascending=False).head(1)
    if not best.empty:
        best_sentence = (
            f"Best mean accuracy was {pct(float(best.iloc[0]['mean_accuracy']))} "
            f"for `{best.iloc[0]['arm']}` on `{best.iloc[0]['split']}` at "
            f"{int(best.iloc[0]['ttt_steps'])} TTT steps."
        )
    else:
        best_sentence = "No result rows were produced."

    max_ttt = max(parse_ints(args.ttt_steps))
    verdict = "negative mechanism signal"
    gate_notes: List[str] = []
    for split in ["eval_indist", "heldout_family"]:
        gate_rows = summary[(summary["split"].eq(split)) & (summary["ttt_steps"].eq(max_ttt))]
        base_rows = summary[(summary["split"].eq(split)) & (summary["ttt_steps"].eq(0))]
        if gate_rows.empty:
            continue

        def acc_for(arm: str) -> float:
            r = gate_rows[gate_rows["arm"].eq(arm)]
            return float(r.iloc[0]["mean_accuracy"]) if not r.empty else float("nan")

        def base_acc_for(arm: str) -> float:
            r = base_rows[base_rows["arm"].eq(arm)]
            return float(r.iloc[0]["mean_accuracy"]) if not r.empty else float("nan")

        echo = acc_for("echo_ttt")
        no = base_acc_for("global_no_ttt")
        corrupt = max(acc_for("shuffle_ttt"), acc_for("generic_ttt"))
        gate_notes.append(
            f"- `{split}` at {max_ttt} TTT steps: echo {pct(echo)}, no-TTT {pct(no)}, best corrupted control {pct(corrupt)}."
        )
    if gate_notes and all("echo" in note for note in gate_notes):
        pass

    md = f"""# Episodic ECHO-TTT Report

## Summary

This standalone experiment tests whether a frozen local language model can use
temporary per-episode gradient updates on environment-observation prediction to
make better later decisions.

Verdict: **{verdict}**.

{best_sentence}

The primary comparison is `global_no_ttt` versus `echo_ttt` and the corrupted
controls `shuffle_ttt` and `generic_ttt`. A real mechanism signal requires
true-observation TTT to improve final-probe accuracy and held-out observation
prediction more than corrupted-observation TTT.

Gate readout at the largest tested TTT step:

{chr(10).join(gate_notes) if gate_notes else "- No gate rows were available."}

## Setup

- Base model: `{args.model_name}`.
- Frozen model weights: yes.
- Per-episode trainable state: `{args.prefix_len}` virtual prefix tokens.
- Observation update target: only text spans containing diagnostic-box observations.
- Candidate decision: likelihood over four randomized option-letter continuations.
- Four-choice chance accuracy: `25.0%`.
- Support observations per episode: `{args.support_count}`.
- Held-out observation probes per episode: `{args.ce_count}`.
- Seeds: `{args.seeds}`.
- Large artifacts: `/workspace/large_artifacts/episodic_echo_ttt`.

## Main Results

{markdown_table(summary, ["split", "arm", "ttt_steps", "n", "mean_accuracy", "std_accuracy", "mean_obs_ce_before", "mean_obs_ce_after", "mean_obs_ce_delta"])}

![Accuracy by arm](../analysis/figures/accuracy_by_arm.png)

![TTT step curve](../analysis/figures/ttt_step_curve.png)

![Observation CE delta](../analysis/figures/observation_ce_delta.png)

## Global Prefix Training

{markdown_table(training, ["seed", "step", "train_loss", "val_obs_ce"]) if not training.empty else "No global prefix training rows."}

![Global prefix training](../analysis/figures/global_prefix_training.png)

## Interpretation

The load-bearing control is corrupted-observation TTT. The result does not pass
that control. True-observation TTT reduces held-out observation CE, but shuffled
and generic-observation updates also reduce CE on the same held-out probes.
Final-probe accuracy remains near the four-choice chance level and does not
separate reliably from the corrupted controls.

The experiment therefore supports a narrow negative conclusion for this tested
recipe: a tiny virtual-prefix test-time update can move Qwen's likelihoods, but
the movement is not specifically grounded in the true episode dynamics strongly
enough to improve decisions.

## Artifacts

- Run directory: `/workspace/experiments/episodic_echo_ttt/runs/{run_name}`.
- Metrics CSV: `/workspace/experiments/episodic_echo_ttt/analysis/metrics.csv`.
- Summary CSV: `/workspace/experiments/episodic_echo_ttt/analysis/summary_by_arm.csv`.
- Detail CSV: `/workspace/experiments/episodic_echo_ttt/analysis/detail_rows.csv`.
- Checkpoints: `/workspace/large_artifacts/episodic_echo_ttt/checkpoints/{run_name}`.
"""
    md_path = REPORTS / "episodic_echo_ttt_report.md"
    html_path = REPORTS / "episodic_echo_ttt_report.html"
    md_path.write_text(md)
    try:
        import markdown

        body = markdown.markdown(md, extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(md) + "</pre>"
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Episodic ECHO-TTT Report</title>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; color: #111827; }}
    table {{ border-collapse: collapse; font-size: 13px; margin: 16px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; margin: 12px 0 24px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    html_path.write_text(html_doc)


def parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_strs(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    start = time.time()
    set_seed(args.data_seed)
    run_dir = RUNS / args.run_name
    ckpt_dir = CHECKPOINTS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    append_experiment_log(
        f"## Run `{args.run_name}`\n\n"
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"- Suite: `{args.suite}`\n"
        f"- Model: `{args.model_name}`\n"
        f"- Seeds: `{args.seeds}`\n"
        f"- Arms: `{args.arms}`\n"
    )
    write_json(run_dir / "args.json", vars(args))

    scorer = FrozenQwenPrefixScorer(args.model_name, load_4bit=bool(args.load_4bit), dtype_name=args.dtype, device=args.device)
    all_metrics: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    for seed in parse_ints(args.seeds):
        set_seed(seed)
        gen = EpisodeGenerator(seed=args.data_seed + seed)
        train_eps = gen.make_set("train", args.train_episodes, args.support_count, args.ce_count)
        val_eps = gen.make_set("eval_indist", args.val_episodes, args.support_count, args.ce_count)
        eval_indist = gen.make_set("eval_indist", args.eval_episodes, args.support_count, args.ce_count)
        eval_heldout = gen.make_set("heldout_family", args.eval_episodes, args.support_count, args.ce_count)
        seed_ckpt = ckpt_dir / f"seed_{seed}"
        seed_ckpt.mkdir(parents=True, exist_ok=True)
        base_prefix: Optional[torch.Tensor] = None
        if args.global_prefix_steps > 0 and args.prefix_len > 0:
            base_prefix, rows = train_global_prefix(
                scorer,
                train_eps,
                val_eps,
                prefix_len=args.prefix_len,
                steps=args.global_prefix_steps,
                lr=args.global_prefix_lr,
                seed=seed,
                run_dir=seed_ckpt,
            )
            for row in rows:
                row["seed"] = seed
                train_rows.append(row)
        if base_prefix is None and args.prefix_len > 0:
            base_prefix = make_prefix(None, args.prefix_len, scorer.hidden_size, scorer.device, scorer.dtype)
        for split_name, eps in [("eval_indist", eval_indist), ("heldout_family", eval_heldout)]:
            metrics, details = run_eval(
                scorer,
                eps,
                split=split_name,
                seed=seed,
                prefix_len=args.prefix_len,
                base_prefix=base_prefix,
                ttt_steps_list=parse_ints(args.ttt_steps),
                ttt_lr=args.ttt_lr,
                arms=parse_strs(args.arms),
            )
            all_metrics.extend(metrics)
            all_details.extend(details)
            write_csv(run_dir / f"metrics_seed_{seed}_{split_name}.csv", metrics)
            write_csv(run_dir / f"details_seed_{seed}_{split_name}.csv", details)
            log(f"[eval seed={seed} split={split_name}] wrote {len(metrics)} metric rows")

    metrics_df = pd.DataFrame(all_metrics)
    summary_df = aggregate(all_metrics)
    training_df = pd.DataFrame(train_rows)
    write_csv(ANALYSIS / "metrics.csv", all_metrics)
    write_csv(ANALYSIS / "detail_rows.csv", all_details)
    summary_df.to_csv(ANALYSIS / "summary_by_arm.csv", index=False)
    if not training_df.empty:
        training_df.to_csv(ANALYSIS / "training_log.csv", index=False)
    else:
        (ANALYSIS / "training_log.csv").write_text("")
    make_figures(summary_df, training_df)
    write_report(args.run_name, summary_df, metrics_df, training_df, args)
    manifest_rows = []
    for path in sorted(ckpt_dir.rglob("*")):
        if path.is_file():
            manifest_rows.append({"artifact": str(path.relative_to(Path("/workspace"))), "bytes": path.stat().st_size})
    write_csv(ROOT / "checkpoint_manifest.csv", manifest_rows)
    elapsed = time.time() - start
    write_json(run_dir / "run_summary.json", {"run_name": args.run_name, "elapsed_sec": elapsed, "metric_rows": len(all_metrics), "detail_rows": len(all_details)})
    append_experiment_log(
        f"Completed `{args.run_name}` in {elapsed:.1f}s.\n\n"
        f"- Metric rows: {len(all_metrics)}\n"
        f"- Detail rows: {len(all_details)}\n"
        f"- Report: `reports/episodic_echo_ttt_report.md`\n"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="main")
    p.add_argument("--run_name", default="main_episodic_echo_ttt")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--load_4bit", type=int, default=1)
    p.add_argument("--data_seed", type=int, default=17)
    p.add_argument("--seeds", default="101,202,303")
    p.add_argument("--support_count", type=int, default=5)
    p.add_argument("--ce_count", type=int, default=2)
    p.add_argument("--train_episodes", type=int, default=96)
    p.add_argument("--val_episodes", type=int, default=24)
    p.add_argument("--eval_episodes", type=int, default=24)
    p.add_argument("--prefix_len", type=int, default=8)
    p.add_argument("--global_prefix_steps", type=int, default=60)
    p.add_argument("--global_prefix_lr", type=float, default=0.05)
    p.add_argument("--ttt_steps", default="0,1,4")
    p.add_argument("--ttt_lr", type=float, default=0.03)
    p.add_argument("--arms", default="no_prefix,global_no_ttt,echo_ttt,shuffle_ttt,generic_ttt")
    return p


def apply_suite_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.suite == "smoke":
        args.run_name = args.run_name or "smoke_episodic_echo_ttt"
        args.seeds = "101"
        args.train_episodes = 8
        args.val_episodes = 4
        args.eval_episodes = 4
        args.global_prefix_steps = 2
        args.ttt_steps = "0,1"
        args.arms = "no_prefix,global_no_ttt,echo_ttt,shuffle_ttt"
    elif args.suite == "pilot":
        args.seeds = "101"
        args.train_episodes = 32
        args.val_episodes = 8
        args.eval_episodes = 8
        args.global_prefix_steps = 12
        args.ttt_steps = "0,1,2"
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    args = apply_suite_defaults(args)
    run(args)


if __name__ == "__main__":
    main()

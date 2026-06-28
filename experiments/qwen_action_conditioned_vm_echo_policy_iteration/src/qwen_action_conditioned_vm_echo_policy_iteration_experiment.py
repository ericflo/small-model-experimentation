#!/usr/bin/env python3
"""Action-conditioned VM-ECHO policy iteration for a frozen-Qwen compiler.

The compiler proposes typed bytecode programs. The VM executes each candidate,
producing action-conditioned observations. A consequence model learns from those
candidate observations, reranks future candidates, and supplies targets for a
distilled compiler.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from typed_bytecode_core import (
    CHECKPOINT_ROOT,
    MAX_PROGRAM_LEN,
    MAX_PROMPT_LEN,
    MODULUS,
    OPCODES,
    OP_TO_ID,
    ROOT,
    RUNS,
    BytecodeProgram,
    TaskExample,
    TaskGenerator,
    choose_answer_verified_candidate,
    execute_program,
    generate_candidates,
    normalize_program,
    program_equal,
    program_from_logits,
    program_logprob,
    set_seed,
)


@dataclass
class PolicyEvalResult:
    run: str
    phase: str
    split: str
    n: int
    direct_accuracy: float
    answer_search_accuracy: float
    oracle_accuracy: float
    learned_accuracy: Optional[float]
    learned_valid_rate: Optional[float]
    direct_valid_rate: float
    candidate_valid_rate: float
    mean_candidates: float
    found_rate: float
    program_exact: float
    learned_program_exact: Optional[float]
    learned_gap_recovered: Optional[float]


@dataclass
class CandidateGroup:
    ex_idx: int
    ops: torch.Tensor
    args: torch.Tensor
    correct: torch.Tensor
    valid: torch.Tensor
    final: torch.Tensor
    trace_top: torch.Tensor
    trace_depth: torch.Tensor
    trace_mask: torch.Tensor
    logprob: torch.Tensor
    base_index: int


class FeatureSet(Dataset):
    def __init__(self, examples: Sequence[TaskExample], hidden: torch.Tensor, mask: torch.Tensor) -> None:
        self.examples = list(examples)
        self.hidden = hidden
        self.mask = mask
        weights = mask.float().unsqueeze(-1)
        pooled = (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        self.prompt_features = pooled.to(torch.float16)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        program = normalize_program(ex.program)
        return {
            "hidden": self.hidden[idx],
            "attention_mask": self.mask[idx],
            "prompt_features": self.prompt_features[idx],
            "ops": torch.tensor(program.ops, dtype=torch.long),
            "args": torch.tensor(program.args, dtype=torch.long),
            "answer": torch.tensor(ex.answer, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class QwenFeatureCompiler(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        max_program_len: int = MAX_PROGRAM_LEN,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(hidden_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.slot_queries = nn.Parameter(torch.randn(max_program_len, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.op_head = nn.Linear(d_model, len(OPCODES))
        self.arg_head = nn.Linear(d_model, MODULUS)
        self.answer_head = nn.Linear(d_model, MODULUS)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.input_norm(self.input_proj(hidden.float()))
        key_padding_mask = ~attention_mask.bool()
        bsz = hidden.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)
        pooled = (memory * attention_mask.unsqueeze(-1).float()).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return {
            "op_logits": self.op_head(decoded),
            "arg_logits": self.arg_head(decoded),
            "answer_logits": self.answer_head(pooled),
        }


class CandidateConsequenceModel(nn.Module):
    def __init__(self, hidden_size: int, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.prompt_proj = nn.Linear(hidden_size, d_model)
        self.op_embed = nn.Embedding(len(OPCODES), d_model)
        self.arg_embed = nn.Embedding(MODULUS, d_model)
        self.pos_embed = nn.Parameter(torch.randn(MAX_PROGRAM_LEN + 1, d_model) * 0.02)
        self.kind_embed = nn.Embedding(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.correct_head = nn.Linear(d_model, 2)
        self.valid_head = nn.Linear(d_model, 2)
        self.final_head = nn.Linear(d_model, MODULUS)
        self.trace_top_head = nn.Linear(d_model, MODULUS)
        self.trace_depth_head = nn.Linear(d_model, MAX_PROGRAM_LEN + 1)

    def forward(self, prompt_features: torch.Tensor, ops: torch.Tensor, args: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = ops.shape[0]
        prompt = self.prompt_proj(prompt_features.float()).unsqueeze(1)
        prompt = prompt + self.kind_embed(torch.zeros(bsz, 1, dtype=torch.long, device=ops.device))
        slots = self.op_embed(ops) + self.arg_embed(args % MODULUS)
        slots = slots + self.kind_embed(torch.ones(bsz, MAX_PROGRAM_LEN, dtype=torch.long, device=ops.device))
        x = torch.cat([prompt, slots], dim=1) + self.pos_embed.unsqueeze(0)
        encoded = self.encoder(x)
        pooled = encoded[:, 0]
        slot_states = encoded[:, 1:]
        return {
            "correct_logits": self.correct_head(pooled),
            "valid_logits": self.valid_head(pooled),
            "final_logits": self.final_head(pooled),
            "trace_top_logits": self.trace_top_head(slot_states),
            "trace_depth_logits": self.trace_depth_head(slot_states),
        }


class CandidateGroupDataset(Dataset):
    def __init__(self, groups: Sequence[CandidateGroup], prompt_features: torch.Tensor) -> None:
        self.groups = list(groups)
        self.prompt_features = prompt_features

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> CandidateGroup:
        return self.groups[idx]


def collate_candidate_groups(groups: Sequence[CandidateGroup], prompt_features: torch.Tensor) -> Dict[str, torch.Tensor]:
    prompt_parts: List[torch.Tensor] = []
    ops_parts: List[torch.Tensor] = []
    args_parts: List[torch.Tensor] = []
    correct_parts: List[torch.Tensor] = []
    valid_parts: List[torch.Tensor] = []
    final_parts: List[torch.Tensor] = []
    top_parts: List[torch.Tensor] = []
    depth_parts: List[torch.Tensor] = []
    mask_parts: List[torch.Tensor] = []
    group_ids: List[torch.Tensor] = []
    for gid, group in enumerate(groups):
        n = group.ops.shape[0]
        prompt_parts.append(prompt_features[group.ex_idx].repeat(n, 1))
        ops_parts.append(group.ops)
        args_parts.append(group.args)
        correct_parts.append(group.correct)
        valid_parts.append(group.valid)
        final_parts.append(group.final)
        top_parts.append(group.trace_top)
        depth_parts.append(group.trace_depth)
        mask_parts.append(group.trace_mask)
        group_ids.append(torch.full((n,), gid, dtype=torch.long))
    return {
        "prompt_features": torch.cat(prompt_parts, dim=0),
        "ops": torch.cat(ops_parts, dim=0),
        "args": torch.cat(args_parts, dim=0),
        "correct": torch.cat(correct_parts, dim=0),
        "valid": torch.cat(valid_parts, dim=0),
        "final": torch.cat(final_parts, dim=0),
        "trace_top": torch.cat(top_parts, dim=0),
        "trace_depth": torch.cat(depth_parts, dim=0),
        "trace_mask": torch.cat(mask_parts, dim=0),
        "group_ids": torch.cat(group_ids, dim=0),
        "num_groups": torch.tensor(len(groups), dtype=torch.long),
    }


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def load_qwen(model_name: str, device: torch.device) -> Tuple[Any, Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quant, device_map="auto")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model, int(model.config.hidden_size)


def extract_features(
    examples: Sequence[TaskExample],
    tokenizer: Any,
    qwen: Any,
    batch_size: int,
    device: torch.device,
    max_prompt_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_hidden: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    prompts = [ex.prompt for ex in examples]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, padding="max_length", truncation=True, max_length=max_prompt_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out = qwen.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            all_hidden.append(out.last_hidden_state.detach().to(torch.float16).cpu())
            all_mask.append(mask.detach().bool().cpu())
    return torch.cat(all_hidden, dim=0), torch.cat(all_mask, dim=0)


def make_splits(args: argparse.Namespace) -> Dict[str, List[TaskExample]]:
    gen = TaskGenerator(seed=args.seed, max_arith_steps=args.max_arith_steps)
    return {
        "seed_train": gen.make_set(args.seed_train_size, template="mixed", hard=False),
        "unlabeled_train": gen.make_set(args.unlabeled_train_size, template="mixed", hard=False),
        "full_supervised_train": gen.make_set(args.full_supervised_size, template="mixed", hard=False),
        "val_mixed": gen.make_set(args.val_size, template="mixed", hard=False),
        "fresh_standard": gen.make_set(args.fresh_size, template="standard", hard=False),
        "fresh_paraphrase": gen.make_set(args.fresh_size, template="paraphrase", hard=False),
        "fresh_paired": gen.make_paired_set(max(1, args.fresh_size // 2), hard=False),
        "hard_composition": gen.make_set(args.hard_size, template="mixed", hard=True),
    }


def append_csv(path: Path, row: Dict[str, Any], rewrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and not rewrite
    with path.open("a" if exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_eval_results(path: Path, results: Sequence[PolicyEvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def masked_slot_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_slot = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    per_slot = per_slot.view_as(targets)
    return (per_slot * mask).sum() / mask.sum().clamp_min(1.0)


def active_observation(program: BytecodeProgram, answer: int) -> Dict[str, torch.Tensor]:
    program = normalize_program(program)
    valid, final, trace = execute_program(program)
    active_len = MAX_PROGRAM_LEN
    for idx, op in enumerate(program.ops[:MAX_PROGRAM_LEN]):
        if int(op) == OP_TO_ID["END"]:
            active_len = idx + 1
            break
    active_len = min(active_len, len(trace))
    top: List[int] = []
    depth: List[int] = []
    mask: List[float] = []
    for slot in range(MAX_PROGRAM_LEN):
        if slot < active_len:
            stack = trace[slot]
            mask.append(1.0)
            depth.append(min(len(stack), MAX_PROGRAM_LEN))
            top.append(int(stack[-1] % MODULUS) if stack else 0)
        else:
            mask.append(0.0)
            depth.append(0)
            top.append(0)
    return {
        "correct": torch.tensor(1 if valid and int(final) == int(answer) else 0, dtype=torch.long),
        "valid": torch.tensor(1 if valid else 0, dtype=torch.long),
        "final": torch.tensor(int(final % MODULUS) if valid else 0, dtype=torch.long),
        "trace_top": torch.tensor(top, dtype=torch.long),
        "trace_depth": torch.tensor(depth, dtype=torch.long),
        "trace_mask": torch.tensor(mask, dtype=torch.float32),
    }


def compiler_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], answer_weight: float) -> torch.Tensor:
    op_loss = F.cross_entropy(outputs["op_logits"].reshape(-1, len(OPCODES)), batch["ops"].reshape(-1))
    arg_loss = F.cross_entropy(outputs["arg_logits"].reshape(-1, MODULUS), batch["args"].reshape(-1))
    ans_loss = F.cross_entropy(outputs["answer_logits"], batch["answer"])
    return op_loss + arg_loss + answer_weight * ans_loss


def train_compiler(
    model: QwenFeatureCompiler,
    dataset: FeatureSet,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    answer_weight: float,
    log_path: Path,
    phase: str,
    quick_val: Optional[FeatureSet] = None,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            outputs = model(batch["hidden"], batch["attention_mask"])
            loss = compiler_loss(outputs, batch, answer_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * batch["hidden"].shape[0]
            count += batch["hidden"].shape[0]
        row: Dict[str, Any] = {"phase": phase, "epoch": epoch, "loss": total / max(1, count), "train_examples": len(dataset)}
        if quick_val is not None:
            res = evaluate_compiler(model, quick_val, device, run="quick", phase=phase, search_topk=1, max_two_arg_pairs=0, consequence_model=None, limit=min(96, len(quick_val)))
            row["quick_val_direct_accuracy"] = res.direct_accuracy
            row["quick_val_answer_search_accuracy"] = res.answer_search_accuracy
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)


def tensor_program(ops: torch.Tensor, args: torch.Tensor) -> BytecodeProgram:
    return normalize_program(BytecodeProgram(ops=[int(x) for x in ops.tolist()], args=[int(x) for x in args.tolist()]))


def build_candidate_groups(
    model: QwenFeatureCompiler,
    dataset: FeatureSet,
    device: torch.device,
    search_topk: int,
    max_two_arg_pairs: int,
    max_candidates: int,
    limit: Optional[int] = None,
) -> List[CandidateGroup]:
    model.eval()
    n_items = len(dataset) if limit is None else min(limit, len(dataset))
    groups: List[CandidateGroup] = []
    with torch.no_grad():
        for idx in range(n_items):
            item = dataset[idx]
            hidden = item["hidden"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            ex = dataset.examples[idx]
            outputs = model(hidden, mask)
            op_logits = outputs["op_logits"][0]
            arg_logits = outputs["arg_logits"][0]
            base = program_from_logits(op_logits, arg_logits)
            base_key = (tuple(base.ops), tuple(base.args))
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            scored = [(program_logprob(c, op_logits, arg_logits), c) for c in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            if max_candidates > 0:
                scored = scored[:max_candidates]
                if not any((tuple(c.ops), tuple(c.args)) == base_key for _, c in scored):
                    scored[-1] = (program_logprob(base, op_logits, arg_logits), base)
            obs = [active_observation(c, ex.answer) for _, c in scored]
            ops = torch.tensor([normalize_program(c).ops for _, c in scored], dtype=torch.long)
            args = torch.tensor([normalize_program(c).args for _, c in scored], dtype=torch.long)
            base_index = 0
            for cand_idx, (_, cand) in enumerate(scored):
                if (tuple(cand.ops), tuple(cand.args)) == base_key:
                    base_index = cand_idx
                    break
            groups.append(
                CandidateGroup(
                    ex_idx=idx,
                    ops=ops,
                    args=args,
                    correct=torch.stack([o["correct"] for o in obs]),
                    valid=torch.stack([o["valid"] for o in obs]),
                    final=torch.stack([o["final"] for o in obs]),
                    trace_top=torch.stack([o["trace_top"] for o in obs]),
                    trace_depth=torch.stack([o["trace_depth"] for o in obs]),
                    trace_mask=torch.stack([o["trace_mask"] for o in obs]),
                    logprob=torch.tensor([float(s) for s, _ in scored], dtype=torch.float32),
                    base_index=base_index,
                )
            )
    return groups


def consequence_loss_terms(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    class_weight: torch.Tensor,
    group_weight: float,
    valid_weight: float,
    final_weight: float,
    trace_top_weight: float,
    trace_depth_weight: float,
) -> Dict[str, torch.Tensor]:
    correct_loss = F.cross_entropy(outputs["correct_logits"], batch["correct"], weight=class_weight)
    valid_loss = F.cross_entropy(outputs["valid_logits"], batch["valid"])
    valid_mask = batch["valid"].float()
    final_ce = F.cross_entropy(outputs["final_logits"], batch["final"], reduction="none")
    final_loss = (final_ce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    trace_top_loss = masked_slot_ce(outputs["trace_top_logits"], batch["trace_top"], batch["trace_mask"])
    trace_depth_loss = masked_slot_ce(outputs["trace_depth_logits"], batch["trace_depth"], batch["trace_mask"])
    scores = outputs["correct_logits"][:, 1]
    group_losses: List[torch.Tensor] = []
    for gid in range(int(batch["num_groups"].item())):
        mask = batch["group_ids"].eq(gid)
        positives = batch["correct"][mask].bool()
        if bool(positives.any()):
            group_scores = scores[mask]
            group_losses.append(-(torch.logsumexp(group_scores[positives], dim=0) - torch.logsumexp(group_scores, dim=0)))
    group_loss = torch.stack(group_losses).mean() if group_losses else scores.sum() * 0.0
    total = (
        correct_loss
        + group_weight * group_loss
        + valid_weight * valid_loss
        + final_weight * final_loss
        + trace_top_weight * trace_top_loss
        + trace_depth_weight * trace_depth_loss
    )
    return {
        "loss": total,
        "correct_loss": correct_loss,
        "group_loss": group_loss,
        "valid_loss": valid_loss,
        "final_loss": final_loss,
        "trace_top_loss": trace_top_loss,
        "trace_depth_loss": trace_depth_loss,
    }


def candidate_label_stats(groups: Sequence[CandidateGroup]) -> Dict[str, float]:
    total = sum(int(g.correct.numel()) for g in groups)
    positives = sum(int(g.correct.sum().item()) for g in groups)
    with_pos = sum(1 for g in groups if int(g.correct.sum().item()) > 0)
    return {
        "groups": float(len(groups)),
        "candidates": float(total),
        "positive_candidates": float(positives),
        "positive_rate": positives / max(1, total),
        "oracle_found_rate": with_pos / max(1, len(groups)),
        "mean_candidates": total / max(1, len(groups)),
    }


def train_consequence_model(
    model: CandidateConsequenceModel,
    groups: Sequence[CandidateGroup],
    prompt_features: torch.Tensor,
    device: torch.device,
    epochs: int,
    group_batch_size: int,
    lr: float,
    log_path: Path,
    phase: str,
    quick_groups: Optional[Sequence[CandidateGroup]] = None,
    quick_prompt_features: Optional[torch.Tensor] = None,
    group_weight: float = 1.0,
    valid_weight: float = 0.2,
    final_weight: float = 0.2,
    trace_top_weight: float = 0.1,
    trace_depth_weight: float = 0.05,
) -> None:
    stats = candidate_label_stats(groups)
    pos = max(1.0, stats["positive_candidates"])
    neg = max(1.0, stats["candidates"] - stats["positive_candidates"])
    pos_weight = min(12.0, neg / pos)
    class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    ds = CandidateGroupDataset(groups, prompt_features)
    loader = DataLoader(ds, batch_size=group_batch_size, shuffle=True, collate_fn=lambda xs: collate_candidate_groups(xs, prompt_features))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_score = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {k: 0.0 for k in ["loss", "correct_loss", "group_loss", "valid_loss", "final_loss", "trace_top_loss", "trace_depth_loss"]}
        count = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            outputs = model(batch["prompt_features"], batch["ops"], batch["args"])
            terms = consequence_loss_terms(outputs, batch, class_weight, group_weight, valid_weight, final_weight, trace_top_weight, trace_depth_weight)
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int(batch["correct"].numel())
            for key in totals:
                totals[key] += float(terms[key].detach().cpu()) * n
            count += n
        row: Dict[str, Any] = {"phase": phase, "epoch": epoch, "candidate_examples": int(stats["candidates"]), "positive_rate": stats["positive_rate"], "pos_weight": pos_weight}
        for key, val in totals.items():
            row[key] = val / max(1, count)
        if quick_groups is not None:
            quick_features = prompt_features if quick_prompt_features is None else quick_prompt_features
            quick = evaluate_consequence_on_groups(model, quick_groups, quick_features, device)
            for key, val in quick.items():
                row[f"quick_{key}"] = val
            score = float(quick.get("learned_accuracy", 0.0))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)
    if best_state is not None:
        model.load_state_dict(best_state)


def score_group(
    model: CandidateConsequenceModel,
    group: CandidateGroup,
    prompt_features: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    scores: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, group.ops.shape[0], batch_size):
            end = min(group.ops.shape[0], start + batch_size)
            prompt = prompt_features[group.ex_idx].unsqueeze(0).repeat(end - start, 1).to(device)
            ops = group.ops[start:end].to(device)
            args = group.args[start:end].to(device)
            logits = model(prompt, ops, args)["correct_logits"]
            scores.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu())
    return torch.cat(scores, dim=0)


def evaluate_consequence_on_groups(
    model: CandidateConsequenceModel,
    groups: Sequence[CandidateGroup],
    prompt_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    selected = 0
    base = 0
    oracle = 0
    selected_valid = 0
    score_sum = 0.0
    for group in groups:
        scores = score_group(model, group, prompt_features, device)
        idx = int(torch.argmax(scores).item())
        selected += int(group.correct[idx].item())
        selected_valid += int(group.valid[idx].item())
        base += int(group.correct[group.base_index].item())
        oracle += int(group.correct.sum().item() > 0)
        score_sum += float(scores[idx].item())
    n = max(1, len(groups))
    base_acc = base / n
    oracle_acc = oracle / n
    learned_acc = selected / n
    return {
        "base_accuracy": base_acc,
        "learned_accuracy": learned_acc,
        "oracle_accuracy": oracle_acc,
        "learned_valid_rate": selected_valid / n,
        "mean_selected_score": score_sum / n,
        "gap_recovered": 0.0 if oracle_acc <= base_acc else (learned_acc - base_acc) / (oracle_acc - base_acc),
    }


def select_targets_with_consequence(
    model: CandidateConsequenceModel,
    groups: Sequence[CandidateGroup],
    source_set: FeatureSet,
    prompt_features: torch.Tensor,
    device: torch.device,
    min_score: float,
) -> Tuple[List[TaskExample], Dict[str, Any]]:
    targets: List[TaskExample] = []
    selected_correct = 0
    selected_valid = 0
    selected_scores = 0.0
    oracle = 0
    changed = 0
    for group in groups:
        scores = score_group(model, group, prompt_features, device)
        idx = int(torch.argmax(scores).item())
        score = float(scores[idx].item())
        oracle += int(group.correct.sum().item() > 0)
        if score < min_score:
            continue
        ex = source_set.examples[group.ex_idx]
        program = tensor_program(group.ops[idx], group.args[idx])
        base = tensor_program(group.ops[group.base_index], group.args[group.base_index])
        targets.append(TaskExample(ex.prompt, ex.domain, ex.answer, program, ex.template, ex.length))
        selected_correct += int(group.correct[idx].item())
        selected_valid += int(group.valid[idx].item())
        selected_scores += score
        changed += int(not program_equal(program, base))
    return targets, {
        "source_examples": len(groups),
        "targets": len(targets),
        "oracle_found_rate": oracle / max(1, len(groups)),
        "selected_correct_rate": selected_correct / max(1, len(targets)),
        "selected_valid_rate": selected_valid / max(1, len(targets)),
        "changed_rate": changed / max(1, len(targets)),
        "mean_selected_score": selected_scores / max(1, len(targets)),
        "min_score": min_score,
    }


def select_answer_verified_targets(groups: Sequence[CandidateGroup], source_set: FeatureSet) -> Tuple[List[TaskExample], Dict[str, Any]]:
    targets: List[TaskExample] = []
    found = 0
    changed = 0
    for group in groups:
        pos = torch.nonzero(group.correct.bool(), as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        # Candidate groups are sorted by compiler logprob, so first positive is
        # the answer-verified logprob choice.
        idx = int(pos[0].item())
        ex = source_set.examples[group.ex_idx]
        program = tensor_program(group.ops[idx], group.args[idx])
        base = tensor_program(group.ops[group.base_index], group.args[group.base_index])
        targets.append(TaskExample(ex.prompt, ex.domain, ex.answer, program, ex.template, ex.length))
        found += 1
        changed += int(not program_equal(program, base))
    return targets, {
        "source_examples": len(groups),
        "targets": len(targets),
        "oracle_found_rate": found / max(1, len(groups)),
        "selected_correct_rate": 1.0 if targets else 0.0,
        "selected_valid_rate": 1.0 if targets else 0.0,
        "changed_rate": changed / max(1, found),
        "mean_selected_score": "",
        "min_score": "",
    }


def rebuild_feature_set_from_cache(seed_set: FeatureSet, source_set: FeatureSet, targets: Sequence[TaskExample]) -> FeatureSet:
    examples = list(seed_set.examples)
    hidden_parts = [seed_set.hidden]
    mask_parts = [seed_set.mask]
    prompt_to_idx = {ex.prompt: idx for idx, ex in enumerate(source_set.examples)}
    hidden_targets: List[torch.Tensor] = []
    mask_targets: List[torch.Tensor] = []
    for ex in targets:
        idx = prompt_to_idx[ex.prompt]
        examples.append(ex)
        hidden_targets.append(source_set.hidden[idx])
        mask_targets.append(source_set.mask[idx])
    if hidden_targets:
        hidden_parts.append(torch.stack(hidden_targets, dim=0))
        mask_parts.append(torch.stack(mask_targets, dim=0))
    return FeatureSet(examples, torch.cat(hidden_parts, dim=0), torch.cat(mask_parts, dim=0))


def evaluate_compiler(
    model: QwenFeatureCompiler,
    dataset: FeatureSet,
    device: torch.device,
    run: str,
    phase: str,
    search_topk: int,
    max_two_arg_pairs: int,
    consequence_model: Optional[CandidateConsequenceModel],
    limit: Optional[int] = None,
) -> PolicyEvalResult:
    model.eval()
    n_items = len(dataset) if limit is None else min(limit, len(dataset))
    direct_ok = search_ok = oracle_ok = program_exact = direct_valid = 0
    total_candidates = valid_candidates = found_any = 0
    learned_ok = learned_valid = learned_exact = 0
    with torch.no_grad():
        for idx in range(n_items):
            item = dataset[idx]
            ex = dataset.examples[idx]
            hidden = item["hidden"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            outputs = model(hidden, mask)
            op_logits = outputs["op_logits"][0]
            arg_logits = outputs["arg_logits"][0]
            base = program_from_logits(op_logits, arg_logits)
            base_valid, base_answer, _ = execute_program(base)
            direct_valid += int(base_valid)
            direct_ok += int(base_valid and base_answer == ex.answer)
            program_exact += int(program_equal(base, ex.program))
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            search_ok += int(chosen is not None)
            oracle_ok += int(found > 0)
            found_any += int(found > 0)
            valid_candidates += valid_count
            if consequence_model is not None:
                scored = [(program_logprob(c, op_logits, arg_logits), c) for c in candidates]
                scored.sort(key=lambda x: x[0], reverse=True)
                obs = [active_observation(c, ex.answer) for _, c in scored]
                group = CandidateGroup(
                    ex_idx=idx,
                    ops=torch.tensor([normalize_program(c).ops for _, c in scored], dtype=torch.long),
                    args=torch.tensor([normalize_program(c).args for _, c in scored], dtype=torch.long),
                    correct=torch.stack([o["correct"] for o in obs]),
                    valid=torch.stack([o["valid"] for o in obs]),
                    final=torch.stack([o["final"] for o in obs]),
                    trace_top=torch.stack([o["trace_top"] for o in obs]),
                    trace_depth=torch.stack([o["trace_depth"] for o in obs]),
                    trace_mask=torch.stack([o["trace_mask"] for o in obs]),
                    logprob=torch.tensor([s for s, _ in scored], dtype=torch.float32),
                    base_index=0,
                )
                scores = score_group(consequence_model, group, dataset.prompt_features, device)
                learned_idx = int(torch.argmax(scores).item())
                learned_program = tensor_program(group.ops[learned_idx], group.args[learned_idx])
                learned_valid += int(group.valid[learned_idx].item())
                learned_ok += int(group.correct[learned_idx].item())
                learned_exact += int(program_equal(learned_program, ex.program))
    n = max(1, n_items)
    direct = direct_ok / n
    oracle = oracle_ok / n
    learned_acc = None if consequence_model is None else learned_ok / n
    return PolicyEvalResult(
        run=run,
        phase=phase,
        split="",
        n=n_items,
        direct_accuracy=direct,
        answer_search_accuracy=search_ok / n,
        oracle_accuracy=oracle,
        learned_accuracy=learned_acc,
        learned_valid_rate=None if consequence_model is None else learned_valid / n,
        direct_valid_rate=direct_valid / n,
        candidate_valid_rate=valid_candidates / max(1, total_candidates),
        mean_candidates=total_candidates / n,
        found_rate=found_any / n,
        program_exact=program_exact / n,
        learned_program_exact=None if consequence_model is None else learned_exact / n,
        learned_gap_recovered=None if consequence_model is None or oracle <= direct else (learned_acc - direct) / (oracle - direct),
    )


def eval_splits(
    model: QwenFeatureCompiler,
    feature_cache: Dict[str, FeatureSet],
    device: torch.device,
    run: str,
    phase: str,
    eval_names: Sequence[str],
    search_topk: int,
    max_two_arg_pairs: int,
    consequence_model: Optional[CandidateConsequenceModel],
) -> List[PolicyEvalResult]:
    out: List[PolicyEvalResult] = []
    for split in eval_names:
        res = evaluate_compiler(model, feature_cache[split], device, run, phase, search_topk, max_two_arg_pairs, consequence_model)
        res.split = split
        out.append(res)
    return out


def save_compiler(model: QwenFeatureCompiler, path: Path, extra: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path / "compiler_head.pt")


def save_consequence(model: CandidateConsequenceModel, path: Path, extra: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path / "consequence_model.pt")


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer, qwen, hidden_size = load_qwen(args.model_name, device)
    splits = make_splits(args)
    with (run_dir / "dataset_manifest.json").open("w") as f:
        json.dump(
            {
                "run": args.run_name,
                "model_name": args.model_name,
                "seed": args.seed,
                "sizes": {k: len(v) for k, v in splits.items()},
                "hidden_size": hidden_size,
                "backend": "frozen_qwen_hidden_states",
                "candidate_topk": args.search_topk,
                "max_two_arg_pairs": args.max_two_arg_pairs,
                "max_candidates": args.max_candidates,
                "distill_min_score": args.distill_min_score,
            },
            f,
            indent=2,
        )
    feature_cache: Dict[str, FeatureSet] = {}
    for name, examples in splits.items():
        hidden, mask = extract_features(examples, tokenizer, qwen, args.qwen_batch_size, device, MAX_PROMPT_LEN)
        feature_cache[name] = FeatureSet(examples, hidden, mask)
    del qwen
    torch.cuda.empty_cache()

    eval_names = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    train_log = run_dir / "compiler_train_log.csv"
    consequence_log = run_dir / "consequence_train_log.csv"
    target_log = run_dir / "target_selection.csv"
    for path in [train_log, consequence_log, target_log]:
        if path.exists():
            path.unlink()

    set_seed(args.seed + 1000)
    init_model = QwenFeatureCompiler(hidden_size, args.d_model, args.compiler_layers, args.heads, args.dropout)
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}

    def fresh_compiler() -> QwenFeatureCompiler:
        model = QwenFeatureCompiler(hidden_size, args.d_model, args.compiler_layers, args.heads, args.dropout).to(device)
        model.load_state_dict(init_state)
        return model

    results: List[PolicyEvalResult] = []

    seed_compiler = fresh_compiler()
    train_compiler(seed_compiler, feature_cache["seed_train"], device, args.seed_epochs, args.batch_size, args.lr, args.answer_weight, train_log, "seed_supervised", feature_cache["val_mixed"])
    seed_state = {k: v.detach().cpu().clone() for k, v in seed_compiler.state_dict().items()}
    results.extend(eval_splits(seed_compiler, feature_cache, device, args.run_name, "seed_supervised", eval_names, args.search_topk, args.max_two_arg_pairs, None))
    save_compiler(seed_compiler, CHECKPOINT_ROOT / args.run_name / "seed_supervised", {"phase": "seed_supervised", "model_name": args.model_name})

    train_groups = build_candidate_groups(seed_compiler, feature_cache["unlabeled_train"], device, args.search_topk, args.max_two_arg_pairs, args.max_candidates)
    val_groups = build_candidate_groups(seed_compiler, feature_cache["val_mixed"], device, args.search_topk, args.max_two_arg_pairs, args.max_candidates)
    train_stats = candidate_label_stats(train_groups)
    val_stats = candidate_label_stats(val_groups)
    append_csv(run_dir / "candidate_group_stats.csv", {"phase": "train_candidates", **train_stats}, rewrite=True)
    append_csv(run_dir / "candidate_group_stats.csv", {"phase": "val_candidates", **val_stats})

    consequence = CandidateConsequenceModel(hidden_size, args.consequence_d_model, args.consequence_layers, args.heads, args.dropout).to(device)
    train_consequence_model(
        consequence,
        train_groups,
        feature_cache["unlabeled_train"].prompt_features,
        device,
        args.consequence_epochs,
        args.group_batch_size,
        args.consequence_lr,
        consequence_log,
        "candidate_consequence",
        quick_groups=val_groups,
        quick_prompt_features=feature_cache["val_mixed"].prompt_features,
        group_weight=args.group_loss_weight,
        valid_weight=args.valid_loss_weight,
        final_weight=args.final_loss_weight,
        trace_top_weight=args.trace_top_loss_weight,
        trace_depth_weight=args.trace_depth_loss_weight,
    )
    save_consequence(consequence, CHECKPOINT_ROOT / args.run_name / "candidate_consequence", {"phase": "candidate_consequence", "model_name": args.model_name})
    learned_train_eval = evaluate_consequence_on_groups(consequence, train_groups, feature_cache["unlabeled_train"].prompt_features, device)
    learned_val_eval = evaluate_consequence_on_groups(consequence, val_groups, feature_cache["val_mixed"].prompt_features, device)
    append_csv(run_dir / "consequence_selection_eval.csv", {"phase": "train_candidates", **learned_train_eval}, rewrite=True)
    append_csv(run_dir / "consequence_selection_eval.csv", {"phase": "val_candidates", **learned_val_eval})

    results.extend(eval_splits(seed_compiler, feature_cache, device, args.run_name, "learned_rerank_seed", eval_names, args.search_topk, args.max_two_arg_pairs, consequence))

    learned_targets, learned_stats = select_targets_with_consequence(consequence, train_groups, feature_cache["unlabeled_train"], feature_cache["unlabeled_train"].prompt_features, device, args.distill_min_score)
    learned_stats["phase"] = "learned_policy_targets"
    append_csv(target_log, learned_stats, rewrite=True)
    answer_targets, answer_stats = select_answer_verified_targets(train_groups, feature_cache["unlabeled_train"])
    answer_stats["phase"] = "answer_verified_targets"
    append_csv(target_log, answer_stats)

    set_seed(args.seed + 2000)
    learned_compiler = fresh_compiler()
    learned_compiler.load_state_dict(seed_state)
    learned_set = rebuild_feature_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"], learned_targets)
    train_compiler(learned_compiler, learned_set, device, args.distill_epochs, args.batch_size, args.distill_lr, args.answer_weight, train_log, "learned_policy_distill", feature_cache["val_mixed"])
    results.extend(eval_splits(learned_compiler, feature_cache, device, args.run_name, "learned_policy_distill", eval_names, args.search_topk, args.max_two_arg_pairs, consequence))
    save_compiler(learned_compiler, CHECKPOINT_ROOT / args.run_name / "learned_policy_distill", {"phase": "learned_policy_distill", "targets": learned_stats, "model_name": args.model_name})

    set_seed(args.seed + 3000)
    answer_compiler = fresh_compiler()
    answer_compiler.load_state_dict(seed_state)
    answer_set = rebuild_feature_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"], answer_targets)
    train_compiler(answer_compiler, answer_set, device, args.distill_epochs, args.batch_size, args.distill_lr, args.answer_weight, train_log, "answer_verified_distill", feature_cache["val_mixed"])
    results.extend(eval_splits(answer_compiler, feature_cache, device, args.run_name, "answer_verified_distill", eval_names, args.search_topk, args.max_two_arg_pairs, consequence))
    save_compiler(answer_compiler, CHECKPOINT_ROOT / args.run_name / "answer_verified_distill", {"phase": "answer_verified_distill", "targets": answer_stats, "model_name": args.model_name})

    if args.full_supervised_epochs > 0:
        set_seed(args.seed + 4000)
        full_compiler = fresh_compiler()
        train_compiler(full_compiler, feature_cache["full_supervised_train"], device, args.full_supervised_epochs, args.batch_size, args.lr, args.answer_weight, train_log, "full_supervised", feature_cache["val_mixed"])
        results.extend(eval_splits(full_compiler, feature_cache, device, args.run_name, "full_supervised", eval_names, args.search_topk, args.max_two_arg_pairs, consequence))
        save_compiler(full_compiler, CHECKPOINT_ROOT / args.run_name / "full_supervised", {"phase": "full_supervised", "model_name": args.model_name})

    write_eval_results(run_dir / "metrics.csv", results)
    with (run_dir / "results.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "results": [asdict(r) for r in results]}, f, indent=2)
    manifest = ROOT / "checkpoint_manifest.csv"
    exists = manifest.exists()
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"action-conditioned VM-ECHO policy iteration; base={args.model_name}"})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--device", default="")
    p.add_argument("--seed_train_size", type=int, default=192)
    p.add_argument("--unlabeled_train_size", type=int, default=1024)
    p.add_argument("--full_supervised_size", type=int, default=1024)
    p.add_argument("--val_size", type=int, default=128)
    p.add_argument("--fresh_size", type=int, default=128)
    p.add_argument("--hard_size", type=int, default=128)
    p.add_argument("--max_arith_steps", type=int, default=4)
    p.add_argument("--qwen_batch_size", type=int, default=16)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--compiler_layers", type=int, default=3)
    p.add_argument("--consequence_d_model", type=int, default=256)
    p.add_argument("--consequence_layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--group_batch_size", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--distill_lr", type=float, default=2.5e-4)
    p.add_argument("--consequence_lr", type=float, default=4e-4)
    p.add_argument("--answer_weight", type=float, default=0.2)
    p.add_argument("--seed_epochs", type=int, default=12)
    p.add_argument("--consequence_epochs", type=int, default=5)
    p.add_argument("--distill_epochs", type=int, default=6)
    p.add_argument("--full_supervised_epochs", type=int, default=18)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--max_two_arg_pairs", type=int, default=8)
    p.add_argument("--max_candidates", type=int, default=256)
    p.add_argument("--distill_min_score", type=float, default=0.0)
    p.add_argument("--group_loss_weight", type=float, default=1.0)
    p.add_argument("--valid_loss_weight", type=float, default=0.2)
    p.add_argument("--final_loss_weight", type=float, default=0.2)
    p.add_argument("--trace_top_loss_weight", type=float, default=0.1)
    p.add_argument("--trace_depth_loss_weight", type=float, default=0.05)
    return p


def main() -> None:
    run_experiment(parser().parse_args())


if __name__ == "__main__":
    main()

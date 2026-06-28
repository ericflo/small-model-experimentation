#!/usr/bin/env python3
"""Counterfactual trace-preference distillation for a Qwen-attached VM compiler.

The compiler emits typed bytecode from frozen Qwen hidden states. The experiment
builds candidate sets around the current compiler output, executes every
candidate in a typed VM, and trains a candidate preference model on hard
counterfactual groups: prompts where a better executable candidate exists than
the base decode. Preference quality is ordered as invalid < valid-wrong <
answer-correct < trace-consistent < canonical program. The learned preference
selector is then used as a no-answer repair teacher and compared against
answer-verified and best-quality distillation controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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


QUALITY_LABELS = ["invalid", "valid_wrong", "answer_correct", "trace_consistent", "canonical"]


@dataclass
class PolicyEvalResult:
    run: str
    phase: str
    split: str
    n: int
    direct_accuracy: float
    answer_search_accuracy: float
    oracle_accuracy: float
    preference_rerank_accuracy: Optional[float]
    preference_rerank_valid_rate: Optional[float]
    preference_rerank_program_exact: Optional[float]
    preference_rerank_quality: Optional[float]
    direct_valid_rate: float
    candidate_valid_rate: float
    mean_candidates: float
    found_rate: float
    program_exact: float
    gap_recovered: Optional[float]


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
    quality: torch.Tensor
    program_exact: torch.Tensor
    trace_match: torch.Tensor
    logprob: torch.Tensor
    candidate_features: torch.Tensor
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


class QwenPreferenceCompiler(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_model: int,
        compiler_layers: int,
        preference_layers: int,
        heads: int,
        dropout: float,
        max_program_len: int = MAX_PROGRAM_LEN,
    ) -> None:
        super().__init__()
        self.compiler_proj = nn.Linear(hidden_size, d_model)
        self.compiler_norm = nn.LayerNorm(d_model)
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
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=compiler_layers)
        self.op_head = nn.Linear(d_model, len(OPCODES))
        self.arg_head = nn.Linear(d_model, MODULUS)
        self.answer_head = nn.Linear(d_model, MODULUS)

        self.pref_prompt_proj = nn.Linear(hidden_size, d_model)
        self.pref_prompt_norm = nn.LayerNorm(d_model)
        self.pref_op_embed = nn.Embedding(len(OPCODES), d_model)
        self.pref_arg_embed = nn.Embedding(MODULUS, d_model)
        self.pref_feature_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pref_kind_embed = nn.Embedding(2, d_model)
        self.pref_pos_embed = nn.Parameter(torch.randn(MAX_PROGRAM_LEN + 1, d_model) * 0.02)
        pref_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.pref_encoder = nn.TransformerEncoder(pref_layer, num_layers=preference_layers)
        self.pref_score_head = nn.Linear(d_model, 1)
        self.pref_quality_head = nn.Linear(d_model, len(QUALITY_LABELS))
        self.pref_valid_head = nn.Linear(d_model, 2)
        self.pref_final_head = nn.Linear(d_model, MODULUS)
        self.pref_trace_top_head = nn.Linear(d_model, MODULUS)
        self.pref_trace_depth_head = nn.Linear(d_model, MAX_PROGRAM_LEN + 1)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.compiler_norm(self.compiler_proj(hidden.float()))
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

    def preference_forward(
        self,
        prompt_features: torch.Tensor,
        ops: torch.Tensor,
        args: torch.Tensor,
        candidate_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz = ops.shape[0]
        prompt = self.pref_prompt_norm(self.pref_prompt_proj(prompt_features.float())).unsqueeze(1)
        if candidate_features is not None:
            prompt = prompt + self.pref_feature_proj(candidate_features.float()).unsqueeze(1)
        prompt = prompt + self.pref_kind_embed(torch.zeros(bsz, 1, dtype=torch.long, device=ops.device))
        slots = self.pref_op_embed(ops) + self.pref_arg_embed(args % MODULUS)
        slots = slots + self.pref_kind_embed(torch.ones(bsz, MAX_PROGRAM_LEN, dtype=torch.long, device=ops.device))
        x = torch.cat([prompt, slots], dim=1) + self.pref_pos_embed.unsqueeze(0)
        encoded = self.pref_encoder(x)
        pooled = encoded[:, 0]
        slot_states = encoded[:, 1:]
        return {
            "score": self.pref_score_head(pooled).squeeze(-1),
            "quality_logits": self.pref_quality_head(pooled),
            "valid_logits": self.pref_valid_head(pooled),
            "final_logits": self.pref_final_head(pooled),
            "trace_top_logits": self.pref_trace_top_head(slot_states),
            "trace_depth_logits": self.pref_trace_depth_head(slot_states),
        }

    def preference_parameters(self) -> List[nn.Parameter]:
        return [p for name, p in self.named_parameters() if name.startswith("pref_")]


class CandidateGroupDataset(Dataset):
    def __init__(self, groups: Sequence[CandidateGroup], prompt_features: torch.Tensor) -> None:
        self.groups = list(groups)
        self.prompt_features = prompt_features

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> CandidateGroup:
        return self.groups[idx]


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def load_qwen(model_name: str) -> Tuple[Any, Any, int]:
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
            print(f"[features] {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)
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
    if path.exists() and not rewrite:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            keys = list(reader.fieldnames or [])
        for key in row:
            if key not in keys:
                keys.append(key)
        rows.append(row)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def write_eval_results(path: Path, results: Sequence[PolicyEvalResult]) -> None:
    write_rows(path, [asdict(r) for r in results])


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


def trace_consistent(candidate_obs: Dict[str, torch.Tensor], gold_obs: Dict[str, torch.Tensor]) -> bool:
    if int(candidate_obs["correct"].item()) != 1:
        return False
    cand_mask = candidate_obs["trace_mask"].bool()
    gold_mask = gold_obs["trace_mask"].bool()
    if not torch.equal(cand_mask, gold_mask):
        return False
    if not torch.equal(candidate_obs["trace_top"][gold_mask], gold_obs["trace_top"][gold_mask]):
        return False
    return torch.equal(candidate_obs["trace_depth"][gold_mask], gold_obs["trace_depth"][gold_mask])


def candidate_quality(program: BytecodeProgram, obs: Dict[str, torch.Tensor], ex: TaskExample, gold_obs: Dict[str, torch.Tensor]) -> Tuple[int, bool, bool]:
    is_exact = program_equal(program, ex.program)
    is_trace = trace_consistent(obs, gold_obs)
    if is_exact:
        return 4, True, is_trace
    if is_trace:
        return 3, False, True
    if int(obs["correct"].item()) == 1:
        return 2, False, False
    if int(obs["valid"].item()) == 1:
        return 1, False, False
    return 0, False, False


def compiler_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], answer_weight: float) -> torch.Tensor:
    op_loss = F.cross_entropy(outputs["op_logits"].reshape(-1, len(OPCODES)), batch["ops"].reshape(-1))
    arg_loss = F.cross_entropy(outputs["arg_logits"].reshape(-1, MODULUS), batch["args"].reshape(-1))
    ans_loss = F.cross_entropy(outputs["answer_logits"], batch["answer"])
    return op_loss + arg_loss + answer_weight * ans_loss


def tensor_program(ops: torch.Tensor, args: torch.Tensor) -> BytecodeProgram:
    return normalize_program(BytecodeProgram(ops=[int(x) for x in ops.tolist()], args=[int(x) for x in args.tolist()]))


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
    quality_parts: List[torch.Tensor] = []
    exact_parts: List[torch.Tensor] = []
    trace_parts: List[torch.Tensor] = []
    logprob_parts: List[torch.Tensor] = []
    feature_parts: List[torch.Tensor] = []
    group_ids: List[torch.Tensor] = []
    base_indices: List[int] = []
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
        quality_parts.append(group.quality)
        exact_parts.append(group.program_exact)
        trace_parts.append(group.trace_match)
        logprob_parts.append(group.logprob)
        feature_parts.append(group.candidate_features)
        group_ids.append(torch.full((n,), gid, dtype=torch.long))
        base_indices.append(group.base_index)
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
        "quality": torch.cat(quality_parts, dim=0),
        "program_exact": torch.cat(exact_parts, dim=0),
        "trace_match": torch.cat(trace_parts, dim=0),
        "logprob": torch.cat(logprob_parts, dim=0),
        "candidate_features": torch.cat(feature_parts, dim=0),
        "group_ids": torch.cat(group_ids, dim=0),
        "base_indices": torch.tensor(base_indices, dtype=torch.long),
        "num_groups": torch.tensor(len(groups), dtype=torch.long),
    }


def candidate_label_stats(groups: Sequence[CandidateGroup]) -> Dict[str, float]:
    total = sum(int(g.quality.numel()) for g in groups)
    correct = sum(int(g.correct.sum().item()) for g in groups)
    valid = sum(int(g.valid.sum().item()) for g in groups)
    exact = sum(int(g.program_exact.sum().item()) for g in groups)
    trace = sum(int(g.trace_match.sum().item()) for g in groups)
    with_correct = sum(1 for g in groups if int(g.correct.sum().item()) > 0)
    counterfactual = sum(1 for g in groups if int(g.quality.max().item()) > int(g.quality[g.base_index].item()))
    base_correct = sum(1 for g in groups if int(g.correct[g.base_index].item()) == 1)
    return {
        "groups": float(len(groups)),
        "candidates": float(total),
        "valid_candidates": float(valid),
        "correct_candidates": float(correct),
        "trace_consistent_candidates": float(trace),
        "canonical_candidates": float(exact),
        "valid_rate": valid / max(1, total),
        "correct_rate": correct / max(1, total),
        "trace_consistent_rate": trace / max(1, total),
        "canonical_rate": exact / max(1, total),
        "oracle_found_rate": with_correct / max(1, len(groups)),
        "base_correct_rate": base_correct / max(1, len(groups)),
        "counterfactual_group_rate": counterfactual / max(1, len(groups)),
        "mean_candidates": total / max(1, len(groups)),
    }


def build_candidate_groups(
    model: QwenPreferenceCompiler,
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
            answer_logp = F.log_softmax(outputs["answer_logits"][0], dim=-1).detach().cpu()
            base = program_from_logits(op_logits, arg_logits)
            base_key = (tuple(base.ops), tuple(base.args))
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            scored = [(program_logprob(c, op_logits, arg_logits), c) for c in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            if max_candidates > 0 and len(scored) > max_candidates:
                scored = scored[:max_candidates]
                if not any((tuple(c.ops), tuple(c.args)) == base_key for _, c in scored):
                    scored[-1] = (program_logprob(base, op_logits, arg_logits), base)
            gold_obs = active_observation(ex.program, ex.answer)
            obs = [active_observation(c, ex.answer) for _, c in scored]
            candidate_features: List[List[float]] = []
            qualities: List[int] = []
            exacts: List[bool] = []
            traces: List[bool] = []
            for (score, cand), cand_obs in zip(scored, obs):
                q, exact, trace = candidate_quality(cand, cand_obs, ex, gold_obs)
                qualities.append(q)
                exacts.append(exact)
                traces.append(trace)
                valid = float(cand_obs["valid"].item())
                final = int(cand_obs["final"].item())
                hint = float(answer_logp[final].item()) if valid else -12.0
                candidate_features.append([float(score) / float(2 * MAX_PROGRAM_LEN), hint / 12.0, valid, final / float(MODULUS - 1)])
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
                    quality=torch.tensor(qualities, dtype=torch.long),
                    program_exact=torch.tensor(exacts, dtype=torch.bool),
                    trace_match=torch.tensor(traces, dtype=torch.bool),
                    logprob=torch.tensor([float(s) for s, _ in scored], dtype=torch.float32),
                    candidate_features=torch.tensor(candidate_features, dtype=torch.float32),
                    base_index=base_index,
                )
            )
            if (idx + 1) % 64 == 0 or idx + 1 == n_items:
                print(f"[candidates] {idx + 1}/{n_items}", flush=True)
    return groups


def filter_counterfactual_groups(groups: Sequence[CandidateGroup]) -> List[CandidateGroup]:
    out: List[CandidateGroup] = []
    for group in groups:
        best = int(group.quality.max().item())
        base = int(group.quality[group.base_index].item())
        if best >= 2 and best > base:
            out.append(group)
    if out:
        return out
    return [g for g in groups if int(g.quality.max().item()) >= 2]


def preference_loss_terms(
    model: QwenPreferenceCompiler,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    outputs = model.preference_forward(batch["prompt_features"], batch["ops"], batch["args"], batch["candidate_features"])
    scores = outputs["score"]
    quality = batch["quality"]
    quality_loss = F.cross_entropy(outputs["quality_logits"], quality)
    valid_loss = F.cross_entropy(outputs["valid_logits"], batch["valid"])
    valid_mask = batch["valid"].float()
    final_ce = F.cross_entropy(outputs["final_logits"], batch["final"], reduction="none")
    final_loss = (final_ce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    trace_top_loss = masked_slot_ce(outputs["trace_top_logits"], batch["trace_top"], batch["trace_mask"])
    trace_depth_loss = masked_slot_ce(outputs["trace_depth_logits"], batch["trace_depth"], batch["trace_mask"])

    listwise_losses: List[torch.Tensor] = []
    pair_losses: List[torch.Tensor] = []
    base_losses: List[torch.Tensor] = []
    for gid in range(int(batch["num_groups"].item())):
        group_mask = batch["group_ids"].eq(gid)
        group_scores = scores[group_mask]
        group_quality = quality[group_mask]
        group_logprob = batch["logprob"][group_mask]
        best_quality = int(group_quality.max().item())
        if best_quality < 2:
            continue
        pos_mask = group_quality.eq(best_quality)
        neg_mask = group_quality.lt(best_quality)
        listwise_losses.append(torch.logsumexp(group_scores, dim=0) - torch.logsumexp(group_scores[pos_mask], dim=0))
        if bool(neg_mask.any()):
            neg_indices = torch.nonzero(neg_mask, as_tuple=False).flatten()
            neg_rank = torch.argsort(group_logprob[neg_indices], descending=True)
            neg_indices = neg_indices[neg_rank[: args.hard_negative_count]]
            pos_scores = group_scores[pos_mask]
            neg_scores = group_scores[neg_indices]
            pair_losses.append(F.softplus(neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1) + args.preference_margin).mean())
        base_index = int(batch["base_indices"][gid].item())
        base_quality = int(group_quality[base_index].item())
        if base_quality < best_quality:
            best_score = torch.logsumexp(group_scores[pos_mask], dim=0) - math.log(max(1, int(pos_mask.sum().item())))
            base_losses.append(F.softplus(group_scores[base_index] - best_score + args.base_margin))
    listwise_loss = torch.stack(listwise_losses).mean() if listwise_losses else scores.sum() * 0.0
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else scores.sum() * 0.0
    base_loss = torch.stack(base_losses).mean() if base_losses else scores.sum() * 0.0
    total = (
        args.listwise_weight * listwise_loss
        + args.pairwise_weight * pair_loss
        + args.base_contrast_weight * base_loss
        + args.quality_loss_weight * quality_loss
        + args.valid_loss_weight * valid_loss
        + args.final_loss_weight * final_loss
        + args.trace_top_loss_weight * trace_top_loss
        + args.trace_depth_loss_weight * trace_depth_loss
    )
    return {
        "loss": total,
        "listwise_loss": listwise_loss,
        "pairwise_loss": pair_loss,
        "base_contrast_loss": base_loss,
        "quality_loss": quality_loss,
        "valid_loss": valid_loss,
        "final_loss": final_loss,
        "trace_top_loss": trace_top_loss,
        "trace_depth_loss": trace_depth_loss,
    }


def infinite(loader: DataLoader) -> Iterator[Any]:
    while True:
        for item in loader:
            yield item


def train_supervised(
    model: QwenPreferenceCompiler,
    dataset: FeatureSet,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    answer_weight: float,
    log_path: Path,
    phase: str,
    quick_val: Optional[FeatureSet],
    search_topk: int,
    max_two_arg_pairs: int,
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
            n = int(batch["hidden"].shape[0])
            total += float(loss.detach().cpu()) * n
            count += n
        row: Dict[str, Any] = {"phase": phase, "epoch": epoch, "loss": total / max(1, count), "train_examples": len(dataset)}
        if quick_val is not None:
            quick = evaluate_compiler(model, quick_val, device, "quick", phase, search_topk, max_two_arg_pairs, preference_model=None, limit=min(96, len(quick_val)))
            row["quick_val_direct_accuracy"] = quick.direct_accuracy
            row["quick_val_search_accuracy"] = quick.answer_search_accuracy
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)
        print(f"[train:{phase}] epoch={epoch} loss={row['loss']:.4f}", flush=True)


def train_preference(
    model: QwenPreferenceCompiler,
    train_groups: Sequence[CandidateGroup],
    prompt_features: torch.Tensor,
    val_groups: Sequence[CandidateGroup],
    val_prompt_features: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    log_path: Path,
    phase: str,
) -> None:
    loader = DataLoader(
        CandidateGroupDataset(train_groups, prompt_features),
        batch_size=args.group_batch_size,
        shuffle=True,
        collate_fn=lambda xs: collate_candidate_groups(xs, prompt_features),
    )
    opt = torch.optim.AdamW(model.preference_parameters(), lr=args.preference_lr, weight_decay=0.01)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -1.0
    for epoch in range(1, args.preference_epochs + 1):
        model.train()
        totals: Dict[str, float] = {
            "loss": 0.0,
            "listwise_loss": 0.0,
            "pairwise_loss": 0.0,
            "base_contrast_loss": 0.0,
            "quality_loss": 0.0,
            "valid_loss": 0.0,
            "final_loss": 0.0,
            "trace_top_loss": 0.0,
            "trace_depth_loss": 0.0,
        }
        count = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            terms = preference_loss_terms(model, batch, args)
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.preference_parameters(), 1.0)
            opt.step()
            n = int(batch["num_groups"].item())
            count += n
            for key in totals:
                totals[key] += float(terms[key].detach().cpu()) * n
        val_metrics = evaluate_preference_groups(model, val_groups, val_prompt_features, device)
        row: Dict[str, Any] = {
            "phase": phase,
            "epoch": epoch,
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            **{key: value / max(1, count) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)
        print(
            f"[train:{phase}] epoch={epoch} loss={row['loss']:.4f} "
            f"val_pref={100.0 * row['val_selected_correct_rate']:.1f}% "
            f"val_oracle={100.0 * row['val_oracle_found_rate']:.1f}%",
            flush=True,
        )
        if row["val_selected_correct_rate"] > best_val:
            best_val = float(row["val_selected_correct_rate"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def score_group_with_preference(
    model: QwenPreferenceCompiler,
    group: CandidateGroup,
    prompt_features: torch.Tensor,
    device: torch.device,
    valid_only: bool = True,
    batch_size: int = 512,
) -> Tuple[int, torch.Tensor]:
    model.eval()
    scores: List[torch.Tensor] = []
    for start in range(0, group.ops.shape[0], batch_size):
        end = min(group.ops.shape[0], start + batch_size)
        prompt = prompt_features[group.ex_idx].unsqueeze(0).repeat(end - start, 1).to(device)
        ops = group.ops[start:end].to(device)
        args = group.args[start:end].to(device)
        features = group.candidate_features[start:end].to(device)
        out = model.preference_forward(prompt, ops, args, features)
        scores.append(out["score"].detach().cpu())
    all_scores = torch.cat(scores, dim=0)
    if valid_only and bool(group.valid.any()):
        masked = all_scores.clone()
        masked[~group.valid.bool()] = -1e9
        return int(masked.argmax().item()), all_scores
    return int(all_scores.argmax().item()), all_scores


def evaluate_preference_groups(
    model: QwenPreferenceCompiler,
    groups: Sequence[CandidateGroup],
    prompt_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    selected_correct = 0
    selected_valid = 0
    selected_exact = 0
    selected_trace = 0
    selected_quality = 0
    oracle = 0
    base_correct = 0
    counterfactual = 0
    for group in groups:
        idx, _ = score_group_with_preference(model, group, prompt_features, device, valid_only=True)
        selected_correct += int(group.correct[idx].item())
        selected_valid += int(group.valid[idx].item())
        selected_exact += int(group.program_exact[idx].item())
        selected_trace += int(group.trace_match[idx].item())
        selected_quality += int(group.quality[idx].item())
        oracle += int(group.correct.any().item())
        base_correct += int(group.correct[group.base_index].item())
        counterfactual += int(group.quality.max().item() > group.quality[group.base_index].item())
    n = max(1, len(groups))
    oracle_rate = oracle / n
    base_rate = base_correct / n
    selected_rate = selected_correct / n
    return {
        "selected_correct_rate": selected_rate,
        "selected_valid_rate": selected_valid / n,
        "selected_program_exact_rate": selected_exact / n,
        "selected_trace_consistent_rate": selected_trace / n,
        "selected_quality_mean": selected_quality / n,
        "oracle_found_rate": oracle_rate,
        "base_correct_rate": base_rate,
        "counterfactual_group_rate": counterfactual / n,
        "gap_recovered": 0.0 if oracle_rate <= base_rate else (selected_rate - base_rate) / (oracle_rate - base_rate),
    }


def select_index_by_quality(group: CandidateGroup) -> int:
    best_quality = int(group.quality.max().item())
    idxs = torch.nonzero(group.quality.eq(best_quality), as_tuple=False).flatten()
    if idxs.numel() == 1:
        return int(idxs[0].item())
    best = idxs[torch.argmax(group.logprob[idxs])]
    return int(best.item())


def target_stats_from_indices(groups: Sequence[CandidateGroup], indices: Sequence[Optional[int]]) -> Dict[str, Any]:
    selected = [idx for idx in indices if idx is not None]
    correct = valid = exact = trace = changed = quality_sum = 0
    for group, idx in zip(groups, indices):
        if idx is None:
            continue
        correct += int(group.correct[idx].item())
        valid += int(group.valid[idx].item())
        exact += int(group.program_exact[idx].item())
        trace += int(group.trace_match[idx].item())
        quality_sum += int(group.quality[idx].item())
        changed += int(idx != group.base_index)
    n = max(1, len(selected))
    return {
        "source_examples": len(groups),
        "targets": len(selected),
        "selected_correct_rate": correct / n,
        "selected_valid_rate": valid / n,
        "selected_program_exact_rate": exact / n,
        "selected_trace_consistent_rate": trace / n,
        "selected_quality_mean": quality_sum / n,
        "changed_rate": changed / n,
    }


def targets_from_indices(groups: Sequence[CandidateGroup], source_set: FeatureSet, indices: Sequence[Optional[int]]) -> List[TaskExample]:
    targets: List[TaskExample] = []
    for group, idx in zip(groups, indices):
        if idx is None:
            continue
        ex = source_set.examples[group.ex_idx]
        program = tensor_program(group.ops[idx], group.args[idx])
        targets.append(TaskExample(ex.prompt, ex.domain, ex.answer, program, ex.template, ex.length))
    return targets


def select_answer_verified_targets(groups: Sequence[CandidateGroup], source_set: FeatureSet) -> Tuple[List[TaskExample], Dict[str, Any]]:
    indices: List[Optional[int]] = []
    for group in groups:
        pos = torch.nonzero(group.correct.bool(), as_tuple=False).flatten()
        if pos.numel() == 0:
            indices.append(None)
            continue
        idx = int(pos[torch.argmax(group.logprob[pos])].item())
        indices.append(idx)
    return targets_from_indices(groups, source_set, indices), target_stats_from_indices(groups, indices)


def select_best_quality_targets(groups: Sequence[CandidateGroup], source_set: FeatureSet) -> Tuple[List[TaskExample], Dict[str, Any]]:
    indices: List[Optional[int]] = []
    for group in groups:
        if int(group.quality.max().item()) < 2:
            indices.append(None)
        else:
            indices.append(select_index_by_quality(group))
    return targets_from_indices(groups, source_set, indices), target_stats_from_indices(groups, indices)


def select_preference_targets(
    model: QwenPreferenceCompiler,
    groups: Sequence[CandidateGroup],
    source_set: FeatureSet,
    device: torch.device,
) -> Tuple[List[TaskExample], Dict[str, Any]]:
    indices: List[Optional[int]] = []
    for group in groups:
        if not bool(group.valid.any()):
            indices.append(None)
            continue
        idx, _ = score_group_with_preference(model, group, source_set.prompt_features, device, valid_only=True)
        indices.append(idx)
    return targets_from_indices(groups, source_set, indices), target_stats_from_indices(groups, indices)


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


def rebuild_gold_set_from_cache(seed_set: FeatureSet, source_set: FeatureSet) -> FeatureSet:
    examples = list(seed_set.examples) + list(source_set.examples)
    return FeatureSet(examples, torch.cat([seed_set.hidden, source_set.hidden], dim=0), torch.cat([seed_set.mask, source_set.mask], dim=0))


def evaluate_compiler(
    model: QwenPreferenceCompiler,
    dataset: FeatureSet,
    device: torch.device,
    run: str,
    phase: str,
    search_topk: int,
    max_two_arg_pairs: int,
    preference_model: Optional[QwenPreferenceCompiler],
    limit: Optional[int] = None,
) -> PolicyEvalResult:
    model.eval()
    n_items = len(dataset) if limit is None else min(limit, len(dataset))
    direct_ok = search_ok = oracle_ok = program_exact_count = direct_valid = 0
    total_candidates = valid_candidates = found_any = 0
    pref_ok = pref_valid = pref_exact = pref_quality = 0
    with torch.no_grad():
        for idx in range(n_items):
            item = dataset[idx]
            ex = dataset.examples[idx]
            hidden = item["hidden"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            outputs = model(hidden, mask)
            op_logits = outputs["op_logits"][0]
            arg_logits = outputs["arg_logits"][0]
            answer_logp = F.log_softmax(outputs["answer_logits"][0], dim=-1).detach().cpu()
            base = program_from_logits(op_logits, arg_logits)
            base_valid, base_answer, _ = execute_program(base)
            direct_valid += int(base_valid)
            direct_ok += int(base_valid and base_answer == ex.answer)
            program_exact_count += int(program_equal(base, ex.program))
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            search_ok += int(chosen is not None)
            oracle_ok += int(found > 0)
            found_any += int(found > 0)
            valid_candidates += valid_count
            if preference_model is not None:
                scored = [(program_logprob(c, op_logits, arg_logits), c) for c in candidates]
                scored.sort(key=lambda x: x[0], reverse=True)
                obs = [active_observation(c, ex.answer) for _, c in scored]
                gold_obs = active_observation(ex.program, ex.answer)
                candidate_features: List[List[float]] = []
                qualities: List[int] = []
                exacts: List[bool] = []
                traces: List[bool] = []
                for (score, cand), cand_obs in zip(scored, obs):
                    q, exact, trace = candidate_quality(cand, cand_obs, ex, gold_obs)
                    qualities.append(q)
                    exacts.append(exact)
                    traces.append(trace)
                    valid_float = float(cand_obs["valid"].item())
                    final_value = int(cand_obs["final"].item())
                    hint = float(answer_logp[final_value].item()) if valid_float else -12.0
                    candidate_features.append([float(score) / float(2 * MAX_PROGRAM_LEN), hint / 12.0, valid_float, final_value / float(MODULUS - 1)])
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
                    quality=torch.tensor(qualities, dtype=torch.long),
                    program_exact=torch.tensor(exacts, dtype=torch.bool),
                    trace_match=torch.tensor(traces, dtype=torch.bool),
                    logprob=torch.tensor([float(s) for s, _ in scored], dtype=torch.float32),
                    candidate_features=torch.tensor(candidate_features, dtype=torch.float32),
                    base_index=0,
                )
                pref_idx, _ = score_group_with_preference(preference_model, group, dataset.prompt_features, device, valid_only=True)
                pref_ok += int(group.correct[pref_idx].item())
                pref_valid += int(group.valid[pref_idx].item())
                pref_exact += int(group.program_exact[pref_idx].item())
                pref_quality += int(group.quality[pref_idx].item())
    n = max(1, n_items)
    direct = direct_ok / n
    oracle = oracle_ok / n
    pref_acc = None if preference_model is None else pref_ok / n
    return PolicyEvalResult(
        run=run,
        phase=phase,
        split="",
        n=n_items,
        direct_accuracy=direct,
        answer_search_accuracy=search_ok / n,
        oracle_accuracy=oracle,
        preference_rerank_accuracy=pref_acc,
        preference_rerank_valid_rate=None if preference_model is None else pref_valid / n,
        preference_rerank_program_exact=None if preference_model is None else pref_exact / n,
        preference_rerank_quality=None if preference_model is None else pref_quality / n,
        direct_valid_rate=direct_valid / n,
        candidate_valid_rate=valid_candidates / max(1, total_candidates),
        mean_candidates=total_candidates / n,
        found_rate=found_any / n,
        program_exact=program_exact_count / n,
        gap_recovered=None if preference_model is None or oracle <= direct else (pref_acc - direct) / (oracle - direct),
    )


def eval_splits(
    model: QwenPreferenceCompiler,
    feature_cache: Dict[str, FeatureSet],
    device: torch.device,
    run: str,
    phase: str,
    eval_names: Sequence[str],
    search_topk: int,
    max_two_arg_pairs: int,
    preference_model: Optional[QwenPreferenceCompiler],
) -> List[PolicyEvalResult]:
    out: List[PolicyEvalResult] = []
    for split in eval_names:
        res = evaluate_compiler(model, feature_cache[split], device, run, phase, search_topk, max_two_arg_pairs, preference_model)
        res.split = split
        out.append(res)
    return out


def save_compiler(model: QwenPreferenceCompiler, path: Path, extra: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path / "compiler_head.pt")


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer, qwen, hidden_size = load_qwen(args.model_name)
    splits = make_splits(args)
    with (run_dir / "dataset_manifest.json").open("w") as f:
        json.dump(
            {
                "run": args.run_name,
                "model_name": args.model_name,
                "seed": args.seed,
                "sizes": {k: len(v) for k, v in splits.items()},
                "hidden_size": hidden_size,
                "backend": "frozen_qwen_hidden_states_with_counterfactual_trace_preference",
                "candidate_topk": args.search_topk,
                "max_two_arg_pairs": args.max_two_arg_pairs,
                "max_candidates": args.max_candidates,
                "quality_labels": QUALITY_LABELS,
            },
            f,
            indent=2,
        )
    feature_cache: Dict[str, FeatureSet] = {}
    for name, examples in splits.items():
        print(f"[features] split={name} n={len(examples)}", flush=True)
        hidden, mask = extract_features(examples, tokenizer, qwen, args.qwen_batch_size, device, MAX_PROMPT_LEN)
        feature_cache[name] = FeatureSet(examples, hidden, mask)
    del qwen
    torch.cuda.empty_cache()

    eval_names = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    train_log = run_dir / "train_log.csv"
    target_log = run_dir / "target_selection.csv"
    candidate_log = run_dir / "candidate_group_stats.csv"
    for path in [train_log, target_log, candidate_log, run_dir / "metrics.csv"]:
        if path.exists():
            path.unlink()

    set_seed(args.seed + 1000)
    init_model = QwenPreferenceCompiler(hidden_size, args.d_model, args.compiler_layers, args.preference_layers, args.heads, args.dropout)
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}

    def fresh_compiler() -> QwenPreferenceCompiler:
        model = QwenPreferenceCompiler(hidden_size, args.d_model, args.compiler_layers, args.preference_layers, args.heads, args.dropout).to(device)
        model.load_state_dict(init_state)
        return model

    results: List[PolicyEvalResult] = []

    seed_compiler = fresh_compiler()
    train_supervised(
        seed_compiler,
        feature_cache["seed_train"],
        device,
        args.seed_epochs,
        args.batch_size,
        args.lr,
        args.answer_weight,
        train_log,
        "seed_supervised",
        feature_cache["val_mixed"],
        args.search_topk,
        args.max_two_arg_pairs,
    )
    seed_state = {k: v.detach().cpu().clone() for k, v in seed_compiler.state_dict().items()}
    results.extend(eval_splits(seed_compiler, feature_cache, device, args.run_name, "seed_supervised", eval_names, args.search_topk, args.max_two_arg_pairs, preference_model=None))
    save_compiler(seed_compiler, CHECKPOINT_ROOT / args.run_name / "seed_supervised", {"phase": "seed_supervised", "model_name": args.model_name})

    print("[groups] collect unlabeled candidate groups", flush=True)
    train_groups = build_candidate_groups(seed_compiler, feature_cache["unlabeled_train"], device, args.search_topk, args.max_two_arg_pairs, args.max_candidates)
    stats = candidate_label_stats(train_groups)
    append_csv(candidate_log, {"phase": "seed_unlabeled_candidates", "round": 0, **stats}, rewrite=True)
    preference_train_groups = filter_counterfactual_groups(train_groups)
    pref_stats = candidate_label_stats(preference_train_groups)
    append_csv(candidate_log, {"phase": "counterfactual_preference_train_groups", "round": 0, **pref_stats})

    print("[groups] collect validation preference groups", flush=True)
    val_groups = build_candidate_groups(seed_compiler, feature_cache["val_mixed"], device, args.search_topk, args.max_two_arg_pairs, args.max_candidates, limit=args.preference_val_limit)
    append_csv(candidate_log, {"phase": "preference_val_candidates", "round": 0, **candidate_label_stats(val_groups)})

    preference_model = fresh_compiler()
    preference_model.load_state_dict(seed_state)
    train_preference(
        preference_model,
        preference_train_groups,
        feature_cache["unlabeled_train"].prompt_features,
        val_groups,
        feature_cache["val_mixed"].prompt_features,
        device,
        args,
        train_log,
        "counterfactual_preference",
    )
    save_compiler(preference_model, CHECKPOINT_ROOT / args.run_name / "counterfactual_preference_selector", {"phase": "counterfactual_preference_selector", "model_name": args.model_name})
    results.extend(eval_splits(seed_compiler, feature_cache, device, args.run_name, "counterfactual_preference_selector", eval_names, args.search_topk, args.max_two_arg_pairs, preference_model=preference_model))

    answer_targets, answer_target_stats = select_answer_verified_targets(train_groups, feature_cache["unlabeled_train"])
    append_csv(target_log, {"phase": "answer_verified_targets", "round": 1, **answer_target_stats}, rewrite=True)

    quality_targets, quality_target_stats = select_best_quality_targets(train_groups, feature_cache["unlabeled_train"])
    append_csv(target_log, {"phase": "best_quality_targets", "round": 1, **quality_target_stats})

    preference_targets, preference_target_stats = select_preference_targets(preference_model, train_groups, feature_cache["unlabeled_train"], device)
    append_csv(target_log, {"phase": "preference_selected_targets", "round": 1, **preference_target_stats})

    answer_compiler = fresh_compiler()
    answer_compiler.load_state_dict(seed_state)
    answer_set = rebuild_feature_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"], answer_targets)
    train_supervised(
        answer_compiler,
        answer_set,
        device,
        args.distill_epochs,
        args.batch_size,
        args.distill_lr,
        args.answer_weight,
        train_log,
        "answer_verified_distill",
        feature_cache["val_mixed"],
        args.search_topk,
        args.max_two_arg_pairs,
    )
    results.extend(eval_splits(answer_compiler, feature_cache, device, args.run_name, "answer_verified_distill", eval_names, args.search_topk, args.max_two_arg_pairs, preference_model=None))
    save_compiler(answer_compiler, CHECKPOINT_ROOT / args.run_name / "answer_verified_distill", {"phase": "answer_verified_distill", "model_name": args.model_name})

    pref_compiler = fresh_compiler()
    pref_compiler.load_state_dict(seed_state)
    pref_set = rebuild_feature_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"], preference_targets)
    train_supervised(
        pref_compiler,
        pref_set,
        device,
        args.distill_epochs,
        args.batch_size,
        args.distill_lr,
        args.answer_weight,
        train_log,
        "preference_selected_distill",
        feature_cache["val_mixed"],
        args.search_topk,
        args.max_two_arg_pairs,
    )
    results.extend(eval_splits(pref_compiler, feature_cache, device, args.run_name, "preference_selected_distill", eval_names, args.search_topk, args.max_two_arg_pairs, preference_model=None))
    save_compiler(pref_compiler, CHECKPOINT_ROOT / args.run_name / "preference_selected_distill", {"phase": "preference_selected_distill", "model_name": args.model_name})

    quality_compiler = fresh_compiler()
    quality_compiler.load_state_dict(seed_state)
    quality_set = rebuild_feature_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"], quality_targets)
    train_supervised(
        quality_compiler,
        quality_set,
        device,
        args.distill_epochs,
        args.batch_size,
        args.distill_lr,
        args.answer_weight,
        train_log,
        "best_quality_distill",
        feature_cache["val_mixed"],
        args.search_topk,
        args.max_two_arg_pairs,
    )
    results.extend(eval_splits(quality_compiler, feature_cache, device, args.run_name, "best_quality_distill", eval_names, args.search_topk, args.max_two_arg_pairs, preference_model=None))
    save_compiler(quality_compiler, CHECKPOINT_ROOT / args.run_name / "best_quality_distill", {"phase": "best_quality_distill", "model_name": args.model_name})

    if args.full_supervised_epochs > 0:
        full_compiler = fresh_compiler()
        gold_set = rebuild_gold_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"])
        train_supervised(
            full_compiler,
            gold_set,
            device,
            args.full_supervised_epochs,
            args.batch_size,
            args.lr,
            args.answer_weight,
            train_log,
            "full_supervised",
            feature_cache["val_mixed"],
            args.search_topk,
            args.max_two_arg_pairs,
        )
        results.extend(eval_splits(full_compiler, feature_cache, device, args.run_name, "full_supervised", eval_names, args.search_topk, args.max_two_arg_pairs, preference_model=None))
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
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"counterfactual trace preference distillation; base={args.model_name}"})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--seed", type=int, default=131)
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
    p.add_argument("--preference_layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--group_batch_size", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--distill_lr", type=float, default=2.5e-4)
    p.add_argument("--preference_lr", type=float, default=3e-4)
    p.add_argument("--answer_weight", type=float, default=0.2)
    p.add_argument("--seed_epochs", type=int, default=12)
    p.add_argument("--distill_epochs", type=int, default=6)
    p.add_argument("--preference_epochs", type=int, default=8)
    p.add_argument("--full_supervised_epochs", type=int, default=18)
    p.add_argument("--preference_val_limit", type=int, default=128)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--max_two_arg_pairs", type=int, default=8)
    p.add_argument("--max_candidates", type=int, default=256)
    p.add_argument("--hard_negative_count", type=int, default=32)
    p.add_argument("--preference_margin", type=float, default=0.25)
    p.add_argument("--base_margin", type=float, default=0.5)
    p.add_argument("--listwise_weight", type=float, default=1.0)
    p.add_argument("--pairwise_weight", type=float, default=1.0)
    p.add_argument("--base_contrast_weight", type=float, default=0.5)
    p.add_argument("--quality_loss_weight", type=float, default=0.5)
    p.add_argument("--valid_loss_weight", type=float, default=0.1)
    p.add_argument("--final_loss_weight", type=float, default=0.1)
    p.add_argument("--trace_top_loss_weight", type=float, default=0.05)
    p.add_argument("--trace_depth_loss_weight", type=float, default=0.03)
    return p


def main() -> None:
    run_experiment(parser().parse_args())


if __name__ == "__main__":
    main()

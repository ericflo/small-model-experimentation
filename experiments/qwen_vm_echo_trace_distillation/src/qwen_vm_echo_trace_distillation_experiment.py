#!/usr/bin/env python3
"""Frozen-Qwen VM-ECHO trace-distillation experiment.

This standalone run attaches a typed-bytecode compiler head to frozen
`Qwen/Qwen3-4B` hidden states. The baseline learns only to emit bytecode and a
final answer. The VM-ECHO arm receives the same program loss plus an auxiliary
environment-observation loss: predict the VM trace produced by the target
program, slot by slot.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    TaskExample,
    TaskGenerator,
    choose_answer_verified_candidate,
    execute_program,
    generate_candidates,
    normalize_program,
    program_equal,
    program_from_logits,
    set_seed,
)


@dataclass
class EchoEvalResult:
    run: str
    arm: str
    phase: str
    split: str
    n: int
    direct_accuracy: float
    search_accuracy: float
    oracle_accuracy: float
    program_exact: float
    valid_rate: float
    direct_valid_rate: float
    mean_candidates: float
    found_rate: float
    gap_recovered: float
    echo_final_accuracy: float
    echo_valid_accuracy: float
    echo_trace_top_accuracy: float
    echo_trace_depth_accuracy: float


def vm_observation_targets(program: Any) -> Dict[str, torch.Tensor]:
    """Convert VM execution observations into dense tensor labels."""
    program = normalize_program(program)
    valid, final, trace = execute_program(program)
    active_len = MAX_PROGRAM_LEN
    for idx, op in enumerate(program.ops[:MAX_PROGRAM_LEN]):
        if int(op) == OP_TO_ID["END"]:
            active_len = idx + 1
            break
    active_len = min(active_len, len(trace))
    trace_top: List[int] = []
    trace_depth: List[int] = []
    trace_mask: List[int] = []
    for slot in range(MAX_PROGRAM_LEN):
        if slot < active_len:
            stack = trace[slot]
            trace_mask.append(1)
            trace_depth.append(min(len(stack), MAX_PROGRAM_LEN))
            trace_top.append(int(stack[-1] % MODULUS) if stack else 0)
        else:
            trace_mask.append(0)
            trace_depth.append(0)
            trace_top.append(0)
    return {
        "vm_valid": torch.tensor(1 if valid else 0, dtype=torch.long),
        "vm_final": torch.tensor(int(final % MODULUS) if valid else 0, dtype=torch.long),
        "trace_top": torch.tensor(trace_top, dtype=torch.long),
        "trace_depth": torch.tensor(trace_depth, dtype=torch.long),
        "trace_mask": torch.tensor(trace_mask, dtype=torch.float32),
    }


class FeatureSet(Dataset):
    def __init__(self, examples: Sequence[TaskExample], hidden: torch.Tensor, mask: torch.Tensor) -> None:
        self.examples = list(examples)
        self.hidden = hidden
        self.mask = mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        program = normalize_program(ex.program)
        obs = vm_observation_targets(program)
        item = {
            "hidden": self.hidden[idx],
            "attention_mask": self.mask[idx],
            "ops": torch.tensor(program.ops, dtype=torch.long),
            "args": torch.tensor(program.args, dtype=torch.long),
            "answer": torch.tensor(ex.answer, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        item.update(obs)
        return item


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
        self.trace_top_head = nn.Linear(d_model, MODULUS)
        self.trace_depth_head = nn.Linear(d_model, MAX_PROGRAM_LEN + 1)
        self.vm_valid_head = nn.Linear(d_model, 2)
        self.vm_final_head = nn.Linear(d_model, MODULUS)

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
            "trace_top_logits": self.trace_top_head(decoded),
            "trace_depth_logits": self.trace_depth_head(decoded),
            "vm_valid_logits": self.vm_valid_head(pooled),
            "vm_final_logits": self.vm_final_head(pooled),
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
    hidden_size = int(model.config.hidden_size)
    return tokenizer, model, hidden_size


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
            enc = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out = qwen.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            hidden = out.last_hidden_state.detach().to(torch.float16).cpu()
            all_hidden.append(hidden)
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


def write_eval_results(path: Path, results: Sequence[EchoEvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))


def masked_slot_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_slot = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    per_slot = per_slot.view_as(targets)
    return (per_slot * mask).sum() / mask.sum().clamp_min(1.0)


def loss_terms(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    answer_weight: float,
    echo_weight: float,
    echo_trace_weight: float,
    echo_depth_weight: float,
    echo_final_weight: float,
    echo_valid_weight: float,
) -> Dict[str, torch.Tensor]:
    ops = batch["ops"]
    args = batch["args"]
    answers = batch["answer"]
    op_loss = F.cross_entropy(outputs["op_logits"].reshape(-1, len(OPCODES)), ops.reshape(-1))
    arg_loss = F.cross_entropy(outputs["arg_logits"].reshape(-1, MODULUS), args.reshape(-1))
    answer_loss = F.cross_entropy(outputs["answer_logits"], answers)
    program_total = op_loss + arg_loss + answer_weight * answer_loss

    trace_mask = batch["trace_mask"]
    trace_top_loss = masked_slot_ce(outputs["trace_top_logits"], batch["trace_top"], trace_mask)
    trace_depth_loss = masked_slot_ce(outputs["trace_depth_logits"], batch["trace_depth"], trace_mask)
    final_loss = F.cross_entropy(outputs["vm_final_logits"], batch["vm_final"])
    valid_loss = F.cross_entropy(outputs["vm_valid_logits"], batch["vm_valid"])
    echo_total = (
        echo_trace_weight * trace_top_loss
        + echo_depth_weight * trace_depth_loss
        + echo_final_weight * final_loss
        + echo_valid_weight * valid_loss
    )
    total = program_total + echo_weight * echo_total
    return {
        "loss": total,
        "program_loss": program_total,
        "op_loss": op_loss,
        "arg_loss": arg_loss,
        "answer_loss": answer_loss,
        "echo_loss": echo_total,
        "trace_top_loss": trace_top_loss,
        "trace_depth_loss": trace_depth_loss,
        "vm_final_loss": final_loss,
        "vm_valid_loss": valid_loss,
    }


def train_head(
    model: QwenFeatureCompiler,
    dataset: FeatureSet,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    answer_weight: float,
    echo_weight: float,
    echo_trace_weight: float,
    echo_depth_weight: float,
    echo_final_weight: float,
    echo_valid_weight: float,
    log_path: Path,
    arm: str,
    phase: str,
    quick_val: Optional[FeatureSet] = None,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "program_loss": 0.0,
            "op_loss": 0.0,
            "arg_loss": 0.0,
            "answer_loss": 0.0,
            "echo_loss": 0.0,
            "trace_top_loss": 0.0,
            "trace_depth_loss": 0.0,
            "vm_final_loss": 0.0,
            "vm_valid_loss": 0.0,
        }
        count = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            hidden = batch["hidden"]
            mask = batch["attention_mask"]
            opt.zero_grad(set_to_none=True)
            outputs = model(hidden, mask)
            terms = loss_terms(
                outputs,
                batch,
                answer_weight,
                echo_weight,
                echo_trace_weight,
                echo_depth_weight,
                echo_final_weight,
                echo_valid_weight,
            )
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for key in totals:
                totals[key] += float(terms[key].detach().cpu()) * hidden.shape[0]
            count += hidden.shape[0]
        row: Dict[str, Any] = {
            "arm": arm,
            "phase": phase,
            "epoch": epoch,
            "train_examples": len(dataset),
            "echo_weight": echo_weight,
        }
        for key, value in totals.items():
            row[key] = value / max(1, count)
        if quick_val is not None:
            res = evaluate_head(model, quick_val, device, run="quick", arm=arm, phase=phase, split="quick_val", search_topk=1, max_two_arg_pairs=0, limit=min(96, len(quick_val)))
            row["quick_val_direct_accuracy"] = res.direct_accuracy
            row["quick_val_valid_rate"] = res.direct_valid_rate
            row["quick_val_echo_trace_top_accuracy"] = res.echo_trace_top_accuracy
            row["quick_val_echo_final_accuracy"] = res.echo_final_accuracy
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)


def evaluate_head(
    model: QwenFeatureCompiler,
    dataset: FeatureSet,
    device: torch.device,
    run: str,
    arm: str,
    phase: str,
    split: str,
    search_topk: int,
    max_two_arg_pairs: int,
    limit: Optional[int] = None,
) -> EchoEvalResult:
    model.eval()
    n_items = len(dataset) if limit is None else min(limit, len(dataset))
    direct_ok = search_ok = oracle_ok = prog_exact = direct_valid = valid_total = found_any = total_candidates = 0
    echo_final_ok = echo_valid_ok = 0
    echo_top_ok = echo_depth_ok = 0.0
    echo_slots = 0.0
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
            base_valid, base_answer, _ = execute_program(base)
            direct_valid += int(base_valid)
            direct_ok += int(base_valid and base_answer == ex.answer)
            prog_exact += int(program_equal(base, ex.program))
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            valid_total += valid_count
            found_any += int(found > 0)
            search_ok += int(chosen is not None)
            oracle_ok += int(found > 0)

            echo_final_ok += int(outputs["vm_final_logits"][0].argmax().item() == int(item["vm_final"].item()))
            echo_valid_ok += int(outputs["vm_valid_logits"][0].argmax().item() == int(item["vm_valid"].item()))
            trace_mask = item["trace_mask"].to(device)
            top_pred = outputs["trace_top_logits"][0].argmax(dim=-1)
            depth_pred = outputs["trace_depth_logits"][0].argmax(dim=-1)
            top_target = item["trace_top"].to(device)
            depth_target = item["trace_depth"].to(device)
            echo_top_ok += float(((top_pred == top_target).float() * trace_mask).sum().detach().cpu())
            echo_depth_ok += float(((depth_pred == depth_target).float() * trace_mask).sum().detach().cpu())
            echo_slots += float(trace_mask.sum().detach().cpu())
    n = max(1, n_items)
    base_acc = direct_ok / n
    oracle = oracle_ok / n
    search = search_ok / n
    gap = 0.0 if oracle <= base_acc else (search - base_acc) / (oracle - base_acc)
    return EchoEvalResult(
        run=run,
        arm=arm,
        phase=phase,
        split=split,
        n=n_items,
        direct_accuracy=base_acc,
        search_accuracy=search,
        oracle_accuracy=oracle,
        program_exact=prog_exact / n,
        valid_rate=valid_total / max(1, total_candidates),
        direct_valid_rate=direct_valid / n,
        mean_candidates=total_candidates / n,
        found_rate=found_any / n,
        gap_recovered=gap,
        echo_final_accuracy=echo_final_ok / n,
        echo_valid_accuracy=echo_valid_ok / n,
        echo_trace_top_accuracy=echo_top_ok / max(1.0, echo_slots),
        echo_trace_depth_accuracy=echo_depth_ok / max(1.0, echo_slots),
    )


def collect_targets(
    model: QwenFeatureCompiler,
    dataset: FeatureSet,
    device: torch.device,
    search_topk: int,
    max_two_arg_pairs: int,
) -> Tuple[List[TaskExample], Dict[str, Any]]:
    model.eval()
    targets: List[TaskExample] = []
    total_candidates = valid_total = found = changed = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            hidden = item["hidden"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            ex = dataset.examples[idx]
            outputs = model(hidden, mask)
            op_logits = outputs["op_logits"][0]
            arg_logits = outputs["arg_logits"][0]
            base = program_from_logits(op_logits, arg_logits)
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found_count, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            valid_total += valid_count
            if chosen is not None:
                found += 1
                changed += int(not program_equal(chosen, base))
                targets.append(TaskExample(ex.prompt, ex.domain, ex.answer, chosen, ex.template, ex.length))
    return targets, {
        "source_examples": len(dataset),
        "targets": len(targets),
        "found_rate": found / max(1, len(dataset)),
        "changed_rate": changed / max(1, found),
        "mean_candidates": total_candidates / max(1, len(dataset)),
        "candidate_valid_rate": valid_total / max(1, total_candidates),
    }


def save_checkpoint(model: QwenFeatureCompiler, path: Path, extra: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path / "compiler_head.pt")


def rebuild_feature_set_from_cache(seed_set: FeatureSet, unlabeled_set: FeatureSet, targets: Sequence[TaskExample]) -> FeatureSet:
    """Build a feature set for seed examples plus selected unlabeled targets.

    Target examples reuse prompt strings from the unlabeled set. This lets us
    reuse frozen Qwen features instead of rerunning the base model each round.
    """
    seed_hidden = seed_set.hidden
    seed_mask = seed_set.mask
    examples = list(seed_set.examples)
    hidden_parts = [seed_hidden]
    mask_parts = [seed_mask]
    prompt_to_indices: Dict[str, int] = {ex.prompt: i for i, ex in enumerate(unlabeled_set.examples)}
    target_hidden: List[torch.Tensor] = []
    target_masks: List[torch.Tensor] = []
    for ex in targets:
        idx = prompt_to_indices[ex.prompt]
        target_hidden.append(unlabeled_set.hidden[idx])
        target_masks.append(unlabeled_set.mask[idx])
        examples.append(ex)
    if target_hidden:
        hidden_parts.append(torch.stack(target_hidden, dim=0))
        mask_parts.append(torch.stack(target_masks, dim=0))
    return FeatureSet(examples, torch.cat(hidden_parts, dim=0), torch.cat(mask_parts, dim=0))


def run_fixed(args: argparse.Namespace) -> None:
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
                "arms": args.arms,
                "echo_weights": {
                    "echo_weight": args.echo_weight,
                    "trace": args.echo_trace_weight,
                    "depth": args.echo_depth_weight,
                    "final": args.echo_final_weight,
                    "valid": args.echo_valid_weight,
                },
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
    train_log = run_dir / "train_log.csv"
    if train_log.exists():
        train_log.unlink()
    set_seed(args.seed + 1000)
    init_model = QwenFeatureCompiler(hidden_size, args.d_model, args.layers, args.heads, args.dropout)
    initial_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}

    def make_head() -> QwenFeatureCompiler:
        model = QwenFeatureCompiler(hidden_size, args.d_model, args.layers, args.heads, args.dropout).to(device)
        model.load_state_dict(initial_state)
        return model

    def eval_phase(model: QwenFeatureCompiler, arm: str, phase: str, out: List[EchoEvalResult]) -> None:
        for split in eval_names:
            out.append(evaluate_head(model, feature_cache[split], device, args.run_name, arm, phase, split, args.search_topk, args.max_two_arg_pairs))

    results: List[EchoEvalResult] = []
    target_log = run_dir / "expert_targets.csv"
    if target_log.exists():
        target_log.unlink()

    for arm in args.arms:
        echo_weight = args.echo_weight if arm == "vm_echo" else 0.0
        head = make_head()
        phase = f"{arm}_seed_supervised"
        train_head(
            head,
            feature_cache["seed_train"],
            device,
            args.seed_epochs,
            args.batch_size,
            args.lr,
            args.answer_weight,
            echo_weight,
            args.echo_trace_weight,
            args.echo_depth_weight,
            args.echo_final_weight,
            args.echo_valid_weight,
            train_log,
            arm,
            phase,
            feature_cache["val_mixed"],
        )
        eval_phase(head, arm, phase, results)
        save_checkpoint(head, CHECKPOINT_ROOT / args.run_name / phase, {"phase": phase, "arm": arm, "model_name": args.model_name, "echo_weight": echo_weight})

        for round_idx in range(1, args.expert_rounds + 1):
            targets, stats = collect_targets(head, feature_cache["unlabeled_train"], device, args.search_topk, args.max_two_arg_pairs)
            stats["arm"] = arm
            stats["round"] = round_idx
            stats["phase"] = f"{arm}_expert_round_{round_idx}"
            append_csv(target_log, stats, rewrite=not target_log.exists())
            expert_set = rebuild_feature_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"], targets)
            phase = f"{arm}_expert_round_{round_idx}"
            train_head(
                head,
                expert_set,
                device,
                args.expert_epochs,
                args.batch_size,
                args.expert_lr,
                args.answer_weight,
                echo_weight,
                args.echo_trace_weight,
                args.echo_depth_weight,
                args.echo_final_weight,
                args.echo_valid_weight,
                train_log,
                arm,
                phase,
                feature_cache["val_mixed"],
            )
            eval_phase(head, arm, phase, results)
            save_checkpoint(head, CHECKPOINT_ROOT / args.run_name / phase, {"phase": phase, "arm": arm, "targets": stats, "model_name": args.model_name, "echo_weight": echo_weight})

        if args.full_supervised_epochs > 0:
            full = make_head()
            phase = f"{arm}_full_supervised"
            train_head(
                full,
                feature_cache["full_supervised_train"],
                device,
                args.full_supervised_epochs,
                args.batch_size,
                args.lr,
                args.answer_weight,
                echo_weight,
                args.echo_trace_weight,
                args.echo_depth_weight,
                args.echo_final_weight,
                args.echo_valid_weight,
                train_log,
                arm,
                phase,
                feature_cache["val_mixed"],
            )
            eval_phase(full, arm, phase, results)
            save_checkpoint(full, CHECKPOINT_ROOT / args.run_name / phase, {"phase": phase, "arm": arm, "model_name": args.model_name, "echo_weight": echo_weight})

    write_eval_results(run_dir / "metrics.csv", results)
    with (run_dir / "results.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "results": [asdict(r) for r in results]}, f, indent=2)
    manifest = ROOT / "checkpoint_manifest.csv"
    exists = manifest.exists()
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"frozen Qwen VM-ECHO head; base={args.model_name}; arms={','.join(args.arms)}"})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--device", default="")
    p.add_argument("--seed_train_size", type=int, default=192)
    p.add_argument("--unlabeled_train_size", type=int, default=1024)
    p.add_argument("--full_supervised_size", type=int, default=1024)
    p.add_argument("--val_size", type=int, default=128)
    p.add_argument("--fresh_size", type=int, default=128)
    p.add_argument("--hard_size", type=int, default=128)
    p.add_argument("--max_arith_steps", type=int, default=4)
    p.add_argument("--qwen_batch_size", type=int, default=8)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--expert_lr", type=float, default=2.5e-4)
    p.add_argument("--answer_weight", type=float, default=0.2)
    p.add_argument("--arms", nargs="+", choices=["baseline", "vm_echo"], default=["baseline", "vm_echo"])
    p.add_argument("--echo_weight", type=float, default=0.35)
    p.add_argument("--echo_trace_weight", type=float, default=1.0)
    p.add_argument("--echo_depth_weight", type=float, default=0.4)
    p.add_argument("--echo_final_weight", type=float, default=0.5)
    p.add_argument("--echo_valid_weight", type=float, default=0.2)
    p.add_argument("--seed_epochs", type=int, default=12)
    p.add_argument("--expert_rounds", type=int, default=2)
    p.add_argument("--expert_epochs", type=int, default=6)
    p.add_argument("--full_supervised_epochs", type=int, default=18)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--max_two_arg_pairs", type=int, default=8)
    return p


def main() -> None:
    run_fixed(parser().parse_args())


if __name__ == "__main__":
    main()

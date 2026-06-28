#!/usr/bin/env python3
"""Frozen-Qwen typed-bytecode compiler-head experiment.

This companion run attaches the typed-bytecode compiler head to frozen
`Qwen/Qwen3-4B` hidden states. Qwen is used as the text feature extractor; only
the bytecode head is trained. The bytecode VM, search, and expert-iteration
logic are shared with the compact controlled experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_typed_bytecode_expert_iteration_experiment import (
    CHECKPOINT_ROOT,
    MAX_PROGRAM_LEN,
    MAX_PROMPT_LEN,
    MODULUS,
    OPCODES,
    ROOT,
    RUNS,
    TaskExample,
    TaskGenerator,
    EvalResult,
    choose_answer_verified_candidate,
    execute_program,
    generate_candidates,
    normalize_program,
    program_equal,
    program_from_logits,
    program_loss,
    set_seed,
    write_eval_results,
)


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
        return {
            "hidden": self.hidden[idx],
            "attention_mask": self.mask[idx],
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


def train_head(
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
            hidden = batch["hidden"].to(device)
            mask = batch["attention_mask"].to(device)
            ops = batch["ops"].to(device)
            args = batch["args"].to(device)
            answers = batch["answer"].to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(hidden, mask)
            loss = program_loss(outputs, ops, args, answers, answer_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * hidden.shape[0]
            count += hidden.shape[0]
        row: Dict[str, Any] = {"phase": phase, "epoch": epoch, "loss": total / max(1, count), "train_examples": len(dataset)}
        if quick_val is not None:
            res = evaluate_head(model, quick_val, device, run="quick", phase=phase, split="quick_val", search_topk=1, max_two_arg_pairs=0, limit=min(96, len(quick_val)))
            row["quick_val_direct_accuracy"] = res.direct_accuracy
            row["quick_val_valid_rate"] = res.direct_valid_rate
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)


def evaluate_head(
    model: QwenFeatureCompiler,
    dataset: FeatureSet,
    device: torch.device,
    run: str,
    phase: str,
    split: str,
    search_topk: int,
    max_two_arg_pairs: int,
    limit: Optional[int] = None,
) -> EvalResult:
    model.eval()
    n_items = len(dataset) if limit is None else min(limit, len(dataset))
    direct_ok = search_ok = oracle_ok = prog_exact = direct_valid = valid_total = found_any = total_candidates = 0
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
    n = max(1, n_items)
    base_acc = direct_ok / n
    oracle = oracle_ok / n
    search = search_ok / n
    gap = 0.0 if oracle <= base_acc else (search - base_acc) / (oracle - base_acc)
    return EvalResult(
        run=run,
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
    results: List[EvalResult] = []
    head = QwenFeatureCompiler(hidden_size, args.d_model, args.layers, args.heads, args.dropout).to(device)
    train_head(head, feature_cache["seed_train"], device, args.seed_epochs, args.batch_size, args.lr, args.answer_weight, train_log, "seed_supervised", feature_cache["val_mixed"])
    for split in eval_names:
        results.append(evaluate_head(head, feature_cache[split], device, args.run_name, "seed_supervised", split, args.search_topk, args.max_two_arg_pairs))
    save_checkpoint(head, CHECKPOINT_ROOT / args.run_name / "seed_supervised", {"phase": "seed_supervised", "model_name": args.model_name})

    for round_idx in range(1, args.expert_rounds + 1):
        targets, stats = collect_targets(head, feature_cache["unlabeled_train"], device, args.search_topk, args.max_two_arg_pairs)
        stats["round"] = round_idx
        stats["phase"] = f"expert_round_{round_idx}"
        append_csv(run_dir / "expert_targets.csv", stats, rewrite=round_idx == 1)
        expert_set = rebuild_feature_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"], targets)
        train_head(head, expert_set, device, args.expert_epochs, args.batch_size, args.expert_lr, args.answer_weight, train_log, f"expert_round_{round_idx}", feature_cache["val_mixed"])
        for split in eval_names:
            results.append(evaluate_head(head, feature_cache[split], device, args.run_name, f"expert_round_{round_idx}", split, args.search_topk, args.max_two_arg_pairs))
        save_checkpoint(head, CHECKPOINT_ROOT / args.run_name / f"expert_round_{round_idx}", {"phase": f"expert_round_{round_idx}", "targets": stats, "model_name": args.model_name})

    if args.full_supervised_epochs > 0:
        full = QwenFeatureCompiler(hidden_size, args.d_model, args.layers, args.heads, args.dropout).to(device)
        train_head(full, feature_cache["full_supervised_train"], device, args.full_supervised_epochs, args.batch_size, args.lr, args.answer_weight, train_log, "full_supervised", feature_cache["val_mixed"])
        for split in eval_names:
            results.append(evaluate_head(full, feature_cache[split], device, args.run_name, "full_supervised", split, args.search_topk, args.max_two_arg_pairs))
        save_checkpoint(full, CHECKPOINT_ROOT / args.run_name / "full_supervised", {"phase": "full_supervised", "model_name": args.model_name})

    write_eval_results(run_dir / "metrics.csv", results)
    with (run_dir / "results.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "results": [asdict(r) for r in results]}, f, indent=2)
    manifest = ROOT / "checkpoint_manifest.csv"
    exists = manifest.exists()
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"frozen Qwen head; base={args.model_name}"})


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

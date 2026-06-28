#!/usr/bin/env python3
"""Dense-state DAgger VM agent for Qwen.

The policy treats one Qwen forward pass as one recurrent transition. Each turn
receives prompt tokens plus learned dense tokens describing the current typed VM
state, then predicts one edit action or STOP from direct heads. DAgger exposes
the policy to states it actually visits and labels those states with the oracle
next action.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from typed_bytecode_core import (
    CHECKPOINT_ROOT,
    MAX_PROGRAM_LEN,
    MODULUS,
    NO_ARG,
    OPCODES,
    OP_TO_ID,
    ROOT,
    RUNS,
    BytecodeProgram,
    TaskExample,
    TaskGenerator,
    execute_program,
    normalize_program,
    program_equal,
    set_seed,
    allowed_ops_for_depth,
    stack_delta,
)


STOP_ID = 0
OP_OFFSET = 1
ARG_OFFSET = OP_OFFSET + MAX_PROGRAM_LEN * len(OPCODES)
ACTION_SIZE = ARG_OFFSET + MAX_PROGRAM_LEN * MODULUS


def blank_program() -> BytecodeProgram:
    ops = [OP_TO_ID["PUSH"], OP_TO_ID["END"]] + [OP_TO_ID["PAD"]] * (MAX_PROGRAM_LEN - 2)
    args = [0, NO_ARG] + [NO_ARG] * (MAX_PROGRAM_LEN - 2)
    return normalize_program(BytecodeProgram(ops, args))


def action_op(slot: int, op: int) -> int:
    return OP_OFFSET + int(slot) * len(OPCODES) + int(op)


def action_arg(slot: int, arg: int) -> int:
    return ARG_OFFSET + int(slot) * MODULUS + int(arg % MODULUS)


def decode_action(action: int) -> Tuple[str, int, int]:
    action = int(action)
    if action == STOP_ID:
        return "STOP", -1, 0
    if OP_OFFSET <= action < ARG_OFFSET:
        x = action - OP_OFFSET
        return "OP", x // len(OPCODES), x % len(OPCODES)
    x = action - ARG_OFFSET
    return "ARG", x // MODULUS, x % MODULUS


def action_to_text(action: int) -> str:
    kind, slot, value = decode_action(action)
    if kind == "STOP":
        return "STOP"
    if kind == "OP":
        return f"OP {slot} {OPCODES[value]}"
    return f"ARG {slot} {value}"


def action_components(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kinds: List[int] = []
    slots: List[int] = []
    ops: List[int] = []
    args: List[int] = []
    for action in actions.detach().cpu().tolist():
        kind, slot, value = decode_action(int(action))
        if kind == "STOP":
            kinds.append(0)
            slots.append(0)
            ops.append(0)
            args.append(0)
        elif kind == "OP":
            kinds.append(1)
            slots.append(int(slot))
            ops.append(int(value))
            args.append(0)
        else:
            kinds.append(2)
            slots.append(int(slot))
            ops.append(0)
            args.append(int(value))
    device = actions.device
    return (
        torch.tensor(kinds, dtype=torch.long, device=device),
        torch.tensor(slots, dtype=torch.long, device=device),
        torch.tensor(ops, dtype=torch.long, device=device),
        torch.tensor(args, dtype=torch.long, device=device),
    )


def full_action_logits(outputs: Dict[str, torch.Tensor], arg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    bsz = outputs["kind_logits"].shape[0]
    device = outputs["kind_logits"].device
    logits = torch.full((bsz, ACTION_SIZE), -1e9, dtype=outputs["kind_logits"].dtype, device=device)
    logits[:, STOP_ID] = outputs["kind_logits"][:, 0]
    op_scores = (
        outputs["kind_logits"][:, 1].view(bsz, 1, 1)
        + outputs["slot_logits"].view(bsz, MAX_PROGRAM_LEN, 1)
        + outputs["op_logits"].view(bsz, 1, len(OPCODES))
    )
    logits[:, OP_OFFSET:ARG_OFFSET] = op_scores.reshape(bsz, MAX_PROGRAM_LEN * len(OPCODES))
    arg_scores = (
        outputs["kind_logits"][:, 2].view(bsz, 1, 1)
        + outputs["slot_logits"].view(bsz, MAX_PROGRAM_LEN, 1)
        + outputs["arg_logits"].view(bsz, 1, MODULUS)
    )
    if arg_mask is not None:
        arg_scores = arg_scores.masked_fill(~arg_mask.bool().view(bsz, 1, MODULUS), -1e9)
    logits[:, ARG_OFFSET:] = arg_scores.reshape(bsz, MAX_PROGRAM_LEN * MODULUS)
    return logits


def compose_actions(outputs: Dict[str, torch.Tensor], forbid_stop: bool = False, arg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    logits = full_action_logits(outputs, arg_mask=arg_mask)
    if forbid_stop:
        logits = logits.clone()
        logits[:, STOP_ID] = -1e9
    return logits.argmax(dim=-1)


def apply_action(program: BytecodeProgram, action: int) -> Tuple[BytecodeProgram, bool]:
    prog = normalize_program(program)
    kind, slot, value = decode_action(action)
    if kind == "STOP":
        return prog, True
    ops = list(prog.ops)
    args = list(prog.args)
    if not (0 <= slot < MAX_PROGRAM_LEN):
        return prog, False
    if kind == "OP":
        ops[slot] = int(value)
        if int(value) != OP_TO_ID["PUSH"]:
            args[slot] = NO_ARG
    else:
        args[slot] = int(value) % MODULUS
    return normalize_program(BytecodeProgram(ops, args)), False


def is_answer_correct(program: BytecodeProgram, answer: int) -> bool:
    valid, final, _ = execute_program(normalize_program(program))
    return bool(valid and int(final) == int(answer))


def oracle_next_action(current: BytecodeProgram, target: BytecodeProgram, answer: int, stop_on_answer: bool = True) -> int:
    current = normalize_program(current)
    target = normalize_program(target)
    if program_equal(current, target):
        return STOP_ID
    if stop_on_answer and is_answer_correct(current, answer):
        return STOP_ID
    for slot, (cur_op, tgt_op) in enumerate(zip(current.ops, target.ops)):
        if int(cur_op) != int(tgt_op):
            return action_op(slot, int(tgt_op))
    for slot, (cur_arg, tgt_arg) in enumerate(zip(current.args, target.args)):
        if int(cur_arg) != int(tgt_arg):
            return action_arg(slot, int(tgt_arg))
    return STOP_ID


def oracle_distance(current: BytecodeProgram, target: BytecodeProgram, answer: int, stop_on_answer: bool = True) -> int:
    current = normalize_program(current)
    target = normalize_program(target)
    if stop_on_answer and is_answer_correct(current, answer):
        return 0
    dist = 0
    for cur_op, tgt_op in zip(current.ops, target.ops):
        dist += int(int(cur_op) != int(tgt_op))
    for cur_arg, tgt_arg in zip(current.args, target.args):
        dist += int(int(cur_arg) != int(tgt_arg))
    return dist


def observation(program: BytecodeProgram) -> Tuple[int, int, List[int], List[int], List[float]]:
    valid, final, trace = execute_program(normalize_program(program))
    top: List[int] = []
    depth: List[int] = []
    mask: List[float] = []
    for slot in range(MAX_PROGRAM_LEN):
        if slot < len(trace):
            stack = trace[slot]
            mask.append(1.0)
            depth.append(min(len(stack), MAX_PROGRAM_LEN))
            top.append(int(stack[-1] % MODULUS) if stack else 0)
        else:
            mask.append(0.0)
            depth.append(0)
            top.append(0)
    return int(valid), int(final % MODULUS if valid else 0), top, depth, mask


def prefix_depth(program: BytecodeProgram, slot: int) -> Tuple[int, bool]:
    depth = 0
    ended = False
    for op in normalize_program(program).ops[:slot]:
        name = OPCODES[int(op)] if 0 <= int(op) < len(OPCODES) else "BAD"
        if ended:
            if name != "PAD":
                return depth, False
            continue
        if name == "PAD":
            return depth, False
        if name == "END":
            ended = True
            continue
        depth += stack_delta(int(op))
        if depth < 0:
            return 0, False
    return depth, True


def make_prompt(ex: TaskExample) -> str:
    return (
        "You are controlling a small typed bytecode VM. Use the dense VM state, "
        "not text observations, to choose exactly one repair action or STOP.\n"
        "Only STOP when the current VM program is a valid solution for the task.\n"
        f"Task: {ex.prompt}\n"
    )


def prompt_arg_mask(ex: TaskExample) -> torch.Tensor:
    """Mask VM constants to values visible in the prompt.

    The bytecode ABI uses numeric constants copied from the task text. Giving
    the action head this copy prior keeps it from treating argument prediction
    as an unconstrained 97-way classification problem.
    """
    mask = torch.zeros(MODULUS, dtype=torch.bool)
    mask[0] = True
    mask[7] = True
    for match in re.finditer(r"-?\d+", ex.prompt):
        mask[int(match.group(0)) % MODULUS] = True
    return mask


@dataclass
class StateSample:
    ex_idx: int
    ops: List[int]
    args: List[int]
    trace_top: List[int]
    trace_depth: List[int]
    trace_mask: List[float]
    valid: int
    final: int
    step: int
    action: int
    solved: int
    distance: float
    weight: float
    source: str


@dataclass
class TrajectoryStats:
    run: str
    phase: str
    n_examples: int
    states: int
    stop_labels: int
    edit_labels: int
    false_stop_states: int
    rollout_success_rate: float
    mean_states_per_example: float
    mean_policy_steps: float


@dataclass
class TrainRow:
    run: str
    phase: str
    epoch: int
    loss: float
    action_accuracy: float
    kind_accuracy: float
    slot_accuracy: float
    op_accuracy: float
    arg_accuracy: float
    stop_accuracy: float
    solved_bce: float
    distance_mae: float
    train_states: int
    stop_labels: int


@dataclass
class EvalRow:
    run: str
    phase: str
    split: str
    mode: str
    k: int
    n: int
    accuracy: float
    valid_rate: float
    program_exact: float
    stop_rate: float
    false_stop_rate: float
    mean_steps: float
    blank_accuracy: float
    oracle_accuracy: float
    native_accuracy: float


class PromptSet(Dataset):
    def __init__(self, tokenizer: Any, examples: Sequence[TaskExample], max_prompt_len: int) -> None:
        self.examples = list(examples)
        enc = tokenizer(
            [make_prompt(ex) for ex in self.examples],
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.arg_masks = torch.stack([prompt_arg_mask(ex) for ex in self.examples], dim=0)

    def __len__(self) -> int:
        return len(self.examples)


class StateDataset(Dataset):
    def __init__(self, samples: Sequence[StateSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "ex_idx": torch.tensor(s.ex_idx, dtype=torch.long),
            "ops": torch.tensor(s.ops, dtype=torch.long),
            "args": torch.tensor(s.args, dtype=torch.long),
            "trace_top": torch.tensor(s.trace_top, dtype=torch.long),
            "trace_depth": torch.tensor(s.trace_depth, dtype=torch.long),
            "trace_mask": torch.tensor(s.trace_mask, dtype=torch.float32),
            "valid": torch.tensor(s.valid, dtype=torch.long),
            "final": torch.tensor(s.final, dtype=torch.long),
            "step": torch.tensor(s.step, dtype=torch.long),
            "action": torch.tensor(s.action, dtype=torch.long),
            "solved": torch.tensor(s.solved, dtype=torch.float32),
            "distance": torch.tensor(s.distance, dtype=torch.float32),
            "weight": torch.tensor(s.weight, dtype=torch.float32),
        }


def collate_states(batch: Sequence[Dict[str, torch.Tensor]], prompts: PromptSet) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ["ex_idx", "ops", "args", "trace_top", "trace_depth", "trace_mask", "valid", "final", "step", "action", "solved", "distance", "weight"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["input_ids"] = prompts.input_ids[out["ex_idx"]]
    out["attention_mask"] = prompts.attention_mask[out["ex_idx"]]
    out["arg_mask"] = prompts.arg_masks[out["ex_idx"]]
    return out


class DenseStateEncoder(nn.Module):
    def __init__(self, hidden_size: int, state_tokens: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_tokens = state_tokens
        self.op_embed = nn.Embedding(len(OPCODES), hidden_size)
        self.arg_embed = nn.Embedding(MODULUS, hidden_size)
        self.trace_top_embed = nn.Embedding(MODULUS, hidden_size)
        self.trace_depth_embed = nn.Embedding(MAX_PROGRAM_LEN + 1, hidden_size)
        self.valid_embed = nn.Embedding(2, hidden_size)
        self.final_embed = nn.Embedding(MODULUS, hidden_size)
        self.step_embed = nn.Embedding(64, hidden_size)
        self.slot_pos = nn.Parameter(torch.randn(MAX_PROGRAM_LEN, hidden_size) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        self.kind_embed = nn.Embedding(2, hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        trace_top: torch.Tensor,
        trace_depth: torch.Tensor,
        trace_mask: torch.Tensor,
        valid: torch.Tensor,
        final: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        bsz = ops.shape[0]
        slot_x = (
            self.op_embed(ops.long())
            + self.arg_embed((args % MODULUS).long())
            + self.trace_top_embed((trace_top % MODULUS).long())
            + self.trace_depth_embed(trace_depth.long().clamp(0, MAX_PROGRAM_LEN))
            + self.slot_pos.unsqueeze(0)
            + self.kind_embed(torch.ones(bsz, MAX_PROGRAM_LEN, dtype=torch.long, device=ops.device))
        )
        slot_x = slot_x + trace_mask.unsqueeze(-1).float()
        global_x = self.global_token.unsqueeze(0).expand(bsz, -1, -1).clone()
        global_x = (
            global_x
            + self.valid_embed(valid.long()).unsqueeze(1)
            + self.final_embed(final.long().clamp(0, MODULUS - 1)).unsqueeze(1)
            + self.step_embed(step.long().clamp(0, 63)).unsqueeze(1)
            + self.kind_embed(torch.zeros(bsz, 1, dtype=torch.long, device=ops.device))
        )
        x = torch.cat([global_x, slot_x], dim=1)
        if x.shape[1] > self.state_tokens:
            x = x[:, : self.state_tokens]
        return self.proj(x)


class DenseStateQwenAgent(nn.Module):
    def __init__(self, base_model: Any, hidden_size: int, state_tokens: int, dropout: float) -> None:
        super().__init__()
        self.base_model = base_model
        self.state_encoder = DenseStateEncoder(hidden_size, state_tokens, dropout)
        self.head_norm = nn.LayerNorm(hidden_size)
        self.kind_head = nn.Linear(hidden_size, 3)
        self.slot_head = nn.Linear(hidden_size, MAX_PROGRAM_LEN)
        self.op_head = nn.Linear(hidden_size, len(OPCODES))
        self.arg_head = nn.Linear(hidden_size, MODULUS)
        self.solved_head = nn.Linear(hidden_size, 1)
        self.distance_head = nn.Linear(hidden_size, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_emb = self.base_model.get_input_embeddings()(input_ids)
        state_emb = self.state_encoder(
            batch["ops"],
            batch["args"],
            batch["trace_top"],
            batch["trace_depth"],
            batch["trace_mask"],
            batch["valid"],
            batch["final"],
            batch["step"],
        ).to(token_emb.dtype)
        inputs_embeds = torch.cat([token_emb, state_emb], dim=1)
        state_mask = torch.ones(state_emb.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([attention_mask, state_mask], dim=1)
        out = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        pooled = self.head_norm(out.hidden_states[-1][:, -1, :].float())
        return {
            "kind_logits": self.kind_head(pooled),
            "slot_logits": self.slot_head(pooled),
            "op_logits": self.op_head(pooled),
            "arg_logits": self.arg_head(pooled),
            "solved_logits": self.solved_head(pooled).squeeze(-1),
            "distance": self.distance_head(pooled).squeeze(-1),
        }


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def load_agent(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, state_tokens: int, dropout: float) -> Tuple[Any, DenseStateQwenAgent, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quant, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    hidden_size = int(model.config.hidden_size)
    agent = DenseStateQwenAgent(model, hidden_size, state_tokens, dropout)
    model.print_trainable_parameters()
    return tokenizer, agent, hidden_size


def make_splits(args: argparse.Namespace) -> Dict[str, List[TaskExample]]:
    gen = TaskGenerator(seed=args.seed, max_arith_steps=args.max_arith_steps)
    return {
        "train": gen.make_set(args.train_size, template="mixed", hard=False),
        "val_mixed": gen.make_set(args.val_size, template="mixed", hard=False),
        "fresh_standard": gen.make_set(args.fresh_size, template="standard", hard=False),
        "fresh_paraphrase": gen.make_set(args.fresh_size, template="paraphrase", hard=False),
        "fresh_paired": gen.make_paired_set(max(1, args.fresh_size // 2), hard=False),
        "hard_composition": gen.make_set(args.hard_size, template="mixed", hard=True),
    }


def state_from_program(ex_idx: int, ex: TaskExample, program: BytecodeProgram, step: int, action: int, source: str, weight: float = 1.0) -> StateSample:
    prog = normalize_program(program)
    valid, final, top, depth, mask = observation(prog)
    solved = int(valid and final == int(ex.answer))
    dist = oracle_distance(prog, ex.program, ex.answer, stop_on_answer=True)
    return StateSample(
        ex_idx=ex_idx,
        ops=list(prog.ops),
        args=list(prog.args),
        trace_top=top,
        trace_depth=depth,
        trace_mask=mask,
        valid=valid,
        final=final,
        step=step,
        action=int(action),
        solved=solved,
        distance=float(dist),
        weight=float(weight),
        source=source,
    )


def collect_teacher_states(examples: Sequence[TaskExample], max_steps: int, phase: str) -> Tuple[List[StateSample], TrajectoryStats]:
    samples: List[StateSample] = []
    stop_labels = edit_labels = false_stop_states = policy_steps = success = 0
    for idx, ex in enumerate(examples):
        current = blank_program()
        for step in range(max_steps + 1):
            action = oracle_next_action(current, ex.program, ex.answer, stop_on_answer=True)
            samples.append(state_from_program(idx, ex, current, step, action, phase))
            if action == STOP_ID:
                stop_labels += 1
                break
            edit_labels += 1
            policy_steps += 1
            current, _ = apply_action(current, action)
        success += int(is_answer_correct(current, ex.answer))
    n = max(1, len(examples))
    return samples, TrajectoryStats("", phase, len(examples), len(samples), stop_labels, edit_labels, false_stop_states, success / n, len(samples) / n, policy_steps / n)


def sample_to_batch(prompt_set: PromptSet, sample: StateSample, device: torch.device) -> Dict[str, torch.Tensor]:
    item = StateDataset([sample])[0]
    batch = collate_states([item], prompt_set)
    return {k: v.to(device) for k, v in batch.items()}


def masked_action_from_outputs(
    outputs: Dict[str, torch.Tensor],
    program: BytecodeProgram,
    arg_mask: Optional[torch.Tensor],
    forbid_stop: bool,
    value_gate: bool,
    value_threshold: float,
    typed_action_mask: bool,
) -> int:
    logits = full_action_logits(outputs, arg_mask=arg_mask)[0].detach().float().cpu()
    solved_prob = torch.sigmoid(outputs["solved_logits"][0].detach().float()).item()
    if forbid_stop or (value_gate and solved_prob < float(value_threshold)):
        logits[STOP_ID] = -1e9
    prog = normalize_program(program)
    if typed_action_mask:
        for slot in range(MAX_PROGRAM_LEN):
            depth, ok = prefix_depth(prog, slot)
            allowed = set(allowed_ops_for_depth(depth, slot, MAX_PROGRAM_LEN)) if ok else set()
            for op in range(len(OPCODES)):
                idx = action_op(slot, op)
                if op not in allowed or int(prog.ops[slot]) == op:
                    logits[idx] = -1e9
            if int(prog.ops[slot]) != OP_TO_ID["PUSH"]:
                logits[action_arg(slot, 0) : action_arg(slot, 0) + MODULUS] = -1e9
            else:
                current_arg = int(prog.args[slot]) % MODULUS
                logits[action_arg(slot, current_arg)] = -1e9
    return int(torch.argmax(logits).item())


@torch.no_grad()
def predict_action(
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    ex_idx: int,
    program: BytecodeProgram,
    step: int,
    device: torch.device,
    forbid_stop: bool = False,
    value_gate: bool = False,
    value_threshold: float = 0.5,
    typed_action_mask: bool = False,
) -> int:
    ex = prompt_set.examples[ex_idx]
    sample = state_from_program(ex_idx, ex, program, step, STOP_ID, "predict")
    batch = sample_to_batch(prompt_set, sample, device)
    outputs = agent(batch)
    return masked_action_from_outputs(outputs, program, batch.get("arg_mask"), forbid_stop, value_gate, value_threshold, typed_action_mask)


@torch.no_grad()
def collect_dagger_states(
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    device: torch.device,
    rollout_steps: int,
    phase: str,
    value_gate: bool,
    value_threshold: float,
    typed_action_mask: bool,
    false_stop_weight: float,
) -> Tuple[List[StateSample], TrajectoryStats]:
    agent.eval()
    samples: List[StateSample] = []
    stop_labels = edit_labels = false_stop_states = policy_steps = success = 0
    for idx, ex in enumerate(prompt_set.examples):
        current = blank_program()
        for step in range(rollout_steps + 1):
            oracle = oracle_next_action(current, ex.program, ex.answer, stop_on_answer=True)
            pred = predict_action(
                agent,
                prompt_set,
                idx,
                current,
                step,
                device,
                value_gate=value_gate,
                value_threshold=value_threshold,
                typed_action_mask=typed_action_mask,
            )
            false_stop = int(pred == STOP_ID and oracle != STOP_ID)
            weight = false_stop_weight if false_stop else 1.0
            samples.append(state_from_program(idx, ex, current, step, oracle, phase, weight=weight))
            false_stop_states += false_stop
            if oracle == STOP_ID:
                stop_labels += 1
                break
            edit_labels += 1
            if pred == STOP_ID:
                break
            current, _ = apply_action(current, pred)
            policy_steps += 1
        success += int(is_answer_correct(current, ex.answer))
    n = max(1, len(prompt_set.examples))
    return samples, TrajectoryStats("", phase, len(prompt_set.examples), len(samples), stop_labels, edit_labels, false_stop_states, success / n, len(samples) / n, policy_steps / n)


def train_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stop_loss_weight: float, value_loss_weight: float, distance_loss_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    action = batch["action"]
    kind_t, slot_t, op_t, arg_t = action_components(action)
    sample_w = batch["weight"].float()
    action_logits = full_action_logits(outputs, arg_mask=batch.get("arg_mask"))
    action_loss_items = F.cross_entropy(action_logits, action, reduction="none")
    action_w = sample_w * torch.where(kind_t.eq(0), torch.full_like(sample_w, float(stop_loss_weight)), torch.ones_like(sample_w))
    action_loss = (action_loss_items * action_w).sum() / action_w.sum().clamp_min(1.0)
    kind_loss_items = F.cross_entropy(outputs["kind_logits"], kind_t, reduction="none")
    stop_w = torch.where(kind_t.eq(0), torch.full_like(sample_w, float(stop_loss_weight)), torch.ones_like(sample_w))
    kind_loss = (kind_loss_items * sample_w * stop_w).sum() / (sample_w * stop_w).sum().clamp_min(1.0)
    edit_mask = kind_t.ne(0)
    op_mask = kind_t.eq(1)
    arg_mask = kind_t.eq(2)
    slot_loss = F.cross_entropy(outputs["slot_logits"][edit_mask], slot_t[edit_mask]) if bool(edit_mask.any()) else action_loss * 0.0
    op_loss = F.cross_entropy(outputs["op_logits"][op_mask], op_t[op_mask]) if bool(op_mask.any()) else action_loss * 0.0
    masked_arg_logits = outputs["arg_logits"]
    if "arg_mask" in batch:
        masked_arg_logits = masked_arg_logits.masked_fill(~batch["arg_mask"].bool(), -1e9)
    arg_loss = F.cross_entropy(masked_arg_logits[arg_mask], arg_t[arg_mask]) if bool(arg_mask.any()) else action_loss * 0.0
    solved_bce = F.binary_cross_entropy_with_logits(outputs["solved_logits"], batch["solved"].float(), reduction="none")
    solved_loss = (solved_bce * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    dist_target = torch.log1p(batch["distance"].float())
    dist_loss_items = F.smooth_l1_loss(outputs["distance"], dist_target, reduction="none")
    dist_loss = (dist_loss_items * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    aux_loss = kind_loss + slot_loss + op_loss + arg_loss
    loss = action_loss + 0.25 * aux_loss + float(value_loss_weight) * solved_loss + float(distance_loss_weight) * dist_loss
    return loss, {
        "action_loss": float(action_loss.detach().cpu()),
        "kind_loss": float(kind_loss.detach().cpu()),
        "slot_loss": float(slot_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
        "solved_bce": float(solved_loss.detach().cpu()),
        "distance_mae": float(torch.mean(torch.abs(torch.expm1(outputs["distance"].detach().float()).clamp_min(0) - batch["distance"].float())).detach().cpu()),
    }


def train_agent(
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    samples: Sequence[StateSample],
    device: torch.device,
    phase: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    stop_loss_weight: float,
    value_loss_weight: float,
    distance_loss_weight: float,
    log_path: Path,
) -> List[TrainRow]:
    dataset = StateDataset(samples)
    weights = torch.tensor([max(1e-3, s.weight) for s in samples], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda xs: collate_states(xs, prompt_set))
    opt = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    rows: List[TrainRow] = []
    accum = max(1, grad_accum)
    for epoch in range(1, epochs + 1):
        agent.train()
        total_loss = total_n = action_correct = kind_correct = stop_correct = stop_total = 0
        slot_correct = op_correct = arg_correct = edit_total = op_total = arg_total = 0
        solved_bce_total = dist_mae_total = 0.0
        opt.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = agent(batch)
            loss, parts = train_loss(outputs, batch, stop_loss_weight, value_loss_weight, distance_loss_weight)
            (loss / accum).backward()
            if batch_idx % accum == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in agent.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            pred = compose_actions(outputs, arg_mask=batch.get("arg_mask"))
            kind_t, slot_t, op_t, arg_t = action_components(batch["action"])
            pred_kind, pred_slot, pred_op, pred_arg = action_components(pred)
            n = int(batch["action"].numel())
            total_loss += float(loss.detach().cpu()) * n
            total_n += n
            action_correct += int(pred.eq(batch["action"]).sum().item())
            kind_correct += int(pred_kind.eq(kind_t).sum().item())
            stop_mask = batch["action"].eq(STOP_ID)
            stop_total += int(stop_mask.sum().item())
            if bool(stop_mask.any()):
                stop_correct += int(pred[stop_mask].eq(STOP_ID).sum().item())
            edit_mask = kind_t.ne(0)
            edit_total += int(edit_mask.sum().item())
            if bool(edit_mask.any()):
                slot_correct += int(pred_slot[edit_mask].eq(slot_t[edit_mask]).sum().item())
            op_mask = kind_t.eq(1)
            op_total += int(op_mask.sum().item())
            if bool(op_mask.any()):
                op_correct += int(pred_op[op_mask].eq(op_t[op_mask]).sum().item())
            arg_mask = kind_t.eq(2)
            arg_total += int(arg_mask.sum().item())
            if bool(arg_mask.any()):
                arg_correct += int(pred_arg[arg_mask].eq(arg_t[arg_mask]).sum().item())
            solved_bce_total += parts["solved_bce"] * n
            dist_mae_total += parts["distance_mae"] * n
        row = TrainRow(
            run=run_name,
            phase=phase,
            epoch=epoch,
            loss=total_loss / max(1, total_n),
            action_accuracy=action_correct / max(1, total_n),
            kind_accuracy=kind_correct / max(1, total_n),
            slot_accuracy=slot_correct / max(1, edit_total),
            op_accuracy=op_correct / max(1, op_total),
            arg_accuracy=arg_correct / max(1, arg_total),
            stop_accuracy=stop_correct / max(1, stop_total),
            solved_bce=solved_bce_total / max(1, total_n),
            distance_mae=dist_mae_total / max(1, total_n),
            train_states=len(samples),
            stop_labels=sum(1 for s in samples if s.action == STOP_ID),
        )
        rows.append(row)
        append_csv(log_path, asdict(row), rewrite=not log_path.exists() and epoch == 1)
        print(f"[train:{phase}] epoch={epoch} loss={row.loss:.4f} action_acc={100*row.action_accuracy:.1f}% stop_acc={100*row.stop_accuracy:.1f}%", flush=True)
    return rows


def run_oracle(program: BytecodeProgram, ex: TaskExample, k: int) -> Tuple[BytecodeProgram, bool, int]:
    current = normalize_program(program)
    stopped = False
    steps = 0
    for _ in range(k):
        action = oracle_next_action(current, ex.program, ex.answer, stop_on_answer=True)
        if action == STOP_ID:
            stopped = True
            break
        current, _ = apply_action(current, action)
        steps += 1
    return current, stopped, steps


@torch.no_grad()
def run_policy(
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    ex_idx: int,
    k: int,
    device: torch.device,
    mode: str,
    value_threshold: float,
    typed_action_mask: bool,
) -> Tuple[BytecodeProgram, bool, int, List[str]]:
    current = blank_program()
    stopped = False
    steps = 0
    actions: List[str] = []
    for step in range(k):
        action = predict_action(
            agent,
            prompt_set,
            ex_idx,
            current,
            step,
            device,
            forbid_stop=mode == "forced",
            value_gate=mode == "value_gated",
            value_threshold=value_threshold,
            typed_action_mask=typed_action_mask,
        )
        actions.append(action_to_text(action))
        if action == STOP_ID:
            stopped = True
            break
        current, _ = apply_action(current, action)
        steps += 1
    return current, stopped, steps, actions


@torch.no_grad()
def evaluate_native_qwen(base_model: Any, tokenizer: Any, examples: Sequence[TaskExample], max_new_tokens: int) -> Tuple[float, float]:
    int_re = re.compile(r"-?\d+")
    correct = parsed = 0
    for ex in examples:
        prompt = f"Answer with only one integer from 0 to 96.\nTask: {ex.prompt}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        out = base_model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        pred: Optional[int] = None
        for match in int_re.finditer(text):
            value = int(match.group(0))
            if 0 <= value <= 96:
                pred = value
                break
        parsed += int(pred is not None)
        correct += int(pred == int(ex.answer))
    n = max(1, len(examples))
    return correct / n, parsed / n


def evaluate_split(
    agent: Optional[DenseStateQwenAgent],
    prompt_set: PromptSet,
    native_accuracy: float,
    run_name: str,
    phase: str,
    split: str,
    k_values: Sequence[int],
    device: torch.device,
    value_threshold: float,
    typed_action_mask: bool,
) -> List[EvalRow]:
    blank_ok = sum(int(is_answer_correct(blank_program(), ex.answer)) for ex in prompt_set.examples)
    n = max(1, len(prompt_set.examples))
    blank_acc = blank_ok / n
    rows: List[EvalRow] = []
    modes = ["blank", "oracle_teacher"]
    if agent is not None:
        modes.extend(["learned", "value_gated", "forced"])
    for k in k_values:
        for mode in modes:
            if mode == "blank" and k != 0:
                continue
            correct = valid_count = exact = stop_count = false_stop = total_steps = 0
            for idx, ex in enumerate(prompt_set.examples):
                if mode == "blank":
                    prog, stopped, steps = blank_program(), False, 0
                elif mode == "oracle_teacher":
                    prog, stopped, steps = run_oracle(blank_program(), ex, k)
                elif mode == "learned":
                    assert agent is not None
                    prog, stopped, steps, _ = run_policy(agent, prompt_set, idx, k, device, "learned", value_threshold, typed_action_mask)
                elif mode == "value_gated":
                    assert agent is not None
                    prog, stopped, steps, _ = run_policy(agent, prompt_set, idx, k, device, "value_gated", value_threshold, typed_action_mask)
                else:
                    assert agent is not None
                    prog, stopped, steps, _ = run_policy(agent, prompt_set, idx, k, device, "forced", value_threshold, typed_action_mask)
                valid, final, _ = execute_program(prog)
                ok = bool(valid and final == int(ex.answer))
                correct += int(ok)
                valid_count += int(valid)
                exact += int(program_equal(prog, ex.program))
                stop_count += int(stopped)
                false_stop += int(stopped and not ok)
                total_steps += int(steps)
            rows.append(EvalRow(
                run=run_name,
                phase=phase,
                split=split,
                mode=mode,
                k=int(k),
                n=len(prompt_set.examples),
                accuracy=correct / n,
                valid_rate=valid_count / n,
                program_exact=exact / n,
                stop_rate=stop_count / n,
                false_stop_rate=false_stop / n,
                mean_steps=total_steps / n,
                blank_accuracy=blank_acc,
                oracle_accuracy=0.0,
                native_accuracy=native_accuracy,
            ))
    # Fill oracle accuracy for each row with the max-K oracle value for easy summaries.
    max_k = max(k_values)
    oracle_row = [r for r in rows if r.mode == "oracle_teacher" and r.k == max_k]
    oracle_acc = oracle_row[0].accuracy if oracle_row else 0.0
    for row in rows:
        row.oracle_accuracy = oracle_acc
    return rows


@torch.no_grad()
def write_rollout_samples(
    path: Path,
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    phase: str,
    split: str,
    device: torch.device,
    k: int,
    value_threshold: float,
    typed_action_mask: bool,
    count: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(prompt_set.examples[:count]):
        for mode in ["learned", "value_gated", "forced"]:
            prog, stopped, steps, actions = run_policy(agent, prompt_set, idx, k, device, mode, value_threshold, typed_action_mask)
            valid, final, _ = execute_program(prog)
            rows.append({
                "phase": phase,
                "split": split,
                "mode": mode,
                "idx": idx,
                "prompt": ex.prompt,
                "answer": ex.answer,
                "actions": " | ".join(actions),
                "stopped": int(stopped),
                "steps": steps,
                "valid": int(valid),
                "final": int(final) if valid else "",
                "correct": int(valid and final == ex.answer),
                "final_ops": " ".join(OPCODES[int(op)] for op in normalize_program(prog).ops),
                "final_args": " ".join(str(int(arg)) for arg in normalize_program(prog).args),
            })
    existing: List[Dict[str, Any]] = []
    if path.exists():
        with path.open(newline="") as f:
            existing = list(csv.DictReader(f))
    write_rows(path, existing + rows)


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


def evaluate_all(
    agent: Optional[DenseStateQwenAgent],
    prompt_sets: Dict[str, PromptSet],
    native_metrics: Dict[str, float],
    run_name: str,
    phase: str,
    k_values: Sequence[int],
    device: torch.device,
    value_threshold: float,
    typed_action_mask: bool,
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for split in ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]:
        print(f"[eval] phase={phase} split={split}", flush=True)
        rows.extend(evaluate_split(agent, prompt_sets[split], native_metrics.get(split, 0.0), run_name, phase, split, k_values, device, value_threshold, typed_action_mask))
    return rows


def save_agent(agent: DenseStateQwenAgent, path: Path, args: argparse.Namespace) -> None:
    path.mkdir(parents=True, exist_ok=True)
    agent.base_model.save_pretrained(path / "qwen_lora")
    torch.save(
        {
            "state_encoder": agent.state_encoder.state_dict(),
            "head_norm": agent.head_norm.state_dict(),
            "kind_head": agent.kind_head.state_dict(),
            "slot_head": agent.slot_head.state_dict(),
            "op_head": agent.op_head.state_dict(),
            "arg_head": agent.arg_head.state_dict(),
            "solved_head": agent.solved_head.state_dict(),
            "distance_head": agent.distance_head.state_dict(),
            "args": vars(args),
        },
        path / "dense_heads.pt",
    )


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    for filename in ["metrics.csv", "train_log.csv", "trajectory_stats.csv", "native_qwen_metrics.csv", "rollout_samples.csv"]:
        path = run_dir / filename
        if path.exists():
            path.unlink()

    splits = make_splits(args)
    tokenizer, agent, hidden_size = load_agent(args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout, args.state_tokens, args.dropout)
    agent.to(device)
    prompt_sets = {name: PromptSet(tokenizer, examples, args.max_prompt_len) for name, examples in splits.items()}
    with (run_dir / "dataset_manifest.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "sizes": {k: len(v) for k, v in splits.items()}, "hidden_size": hidden_size, "action_size": ACTION_SIZE}, f, indent=2)

    native_metrics: Dict[str, float] = {}
    native_rows: List[Dict[str, Any]] = []
    if args.eval_native:
        agent.base_model.eval()
        for split in ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]:
            acc, parse = evaluate_native_qwen(agent.base_model, tokenizer, prompt_sets[split].examples, args.native_max_new_tokens)
            native_metrics[split] = acc
            native_rows.append({"run": args.run_name, "split": split, "accuracy": acc, "parse_rate": parse})
            print(f"[native] split={split} acc={acc:.3f} parse={parse:.3f}", flush=True)
        write_rows(run_dir / "native_qwen_metrics.csv", native_rows)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        for split in ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]:
            native_metrics[split] = 0.0

    metrics: List[EvalRow] = []
    teacher_states, teacher_stats = collect_teacher_states(prompt_sets["train"].examples, args.teacher_max_steps, "bc_teacher")
    teacher_stats.run = args.run_name
    append_csv(run_dir / "trajectory_stats.csv", asdict(teacher_stats), rewrite=True)
    print(f"[states] bc_teacher states={len(teacher_states)} mean={teacher_stats.mean_states_per_example:.2f}", flush=True)

    train_agent(
        agent,
        prompt_sets["train"],
        teacher_states,
        device,
        "bc_policy",
        args.run_name,
        args.bc_epochs,
        args.batch_size,
        args.grad_accum,
        args.lr,
        args.stop_loss_weight,
        args.value_loss_weight,
        args.distance_loss_weight,
        run_dir / "train_log.csv",
    )
    save_agent(agent, CHECKPOINT_ROOT / args.run_name / "bc_policy", args)
    metrics.extend(evaluate_all(agent, prompt_sets, native_metrics, args.run_name, "bc_policy", args.k_values, device, args.value_threshold, bool(args.typed_action_mask)))
    write_rows(run_dir / "metrics.csv", [asdict(r) for r in metrics])
    write_rollout_samples(run_dir / "rollout_samples.csv", agent, prompt_sets["val_mixed"], "bc_policy", "val_mixed", device, max(args.k_values), args.value_threshold, bool(args.typed_action_mask), min(3, len(prompt_sets["val_mixed"])))

    all_dagger: List[StateSample] = []
    for round_idx in range(1, args.dagger_rounds + 1):
        traj_phase = f"dagger_r{round_idx}_states"
        policy_phase = f"dagger_r{round_idx}_policy"
        states, stats = collect_dagger_states(
            agent,
            prompt_sets["train"],
            device,
            args.dagger_rollout_steps,
            traj_phase,
            value_gate=bool(args.dagger_value_gate),
            value_threshold=args.value_threshold,
            typed_action_mask=bool(args.typed_action_mask),
            false_stop_weight=args.false_stop_weight,
        )
        stats.run = args.run_name
        append_csv(run_dir / "trajectory_stats.csv", asdict(stats))
        print(f"[states] {traj_phase} states={len(states)} false_stop={stats.false_stop_states} rollout_success={stats.rollout_success_rate:.3f}", flush=True)
        all_dagger.extend(states)
        combined = list(teacher_states) + list(all_dagger)
        train_agent(
            agent,
            prompt_sets["train"],
            combined,
            device,
            policy_phase,
            args.run_name,
            args.dagger_epochs,
            args.batch_size,
            args.grad_accum,
            args.dagger_lr,
            args.stop_loss_weight,
            args.value_loss_weight,
            args.distance_loss_weight,
            run_dir / "train_log.csv",
        )
        save_agent(agent, CHECKPOINT_ROOT / args.run_name / policy_phase, args)
        metrics.extend(evaluate_all(agent, prompt_sets, native_metrics, args.run_name, policy_phase, args.k_values, device, args.value_threshold, bool(args.typed_action_mask)))
        write_rows(run_dir / "metrics.csv", [asdict(r) for r in metrics])
        write_rollout_samples(run_dir / "rollout_samples.csv", agent, prompt_sets["val_mixed"], policy_phase, "val_mixed", device, max(args.k_values), args.value_threshold, bool(args.typed_action_mask), min(3, len(prompt_sets["val_mixed"])))

    with (run_dir / "results.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "metrics": [asdict(r) for r in metrics]}, f, indent=2)
    with (ROOT / "checkpoint_manifest.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"dense-state DAgger VM agent; base={args.model_name}"})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--seed", type=int, default=431)
    p.add_argument("--train_size", type=int, default=512)
    p.add_argument("--val_size", type=int, default=64)
    p.add_argument("--fresh_size", type=int, default=64)
    p.add_argument("--hard_size", type=int, default=64)
    p.add_argument("--max_arith_steps", type=int, default=4)
    p.add_argument("--max_prompt_len", type=int, default=96)
    p.add_argument("--state_tokens", type=int, default=17)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dagger_lr", type=float, default=5e-5)
    p.add_argument("--bc_epochs", type=int, default=2)
    p.add_argument("--dagger_epochs", type=int, default=1)
    p.add_argument("--dagger_rounds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--teacher_max_steps", type=int, default=12)
    p.add_argument("--dagger_rollout_steps", type=int, default=8)
    p.add_argument("--stop_loss_weight", type=float, default=4.0)
    p.add_argument("--value_loss_weight", type=float, default=0.5)
    p.add_argument("--distance_loss_weight", type=float, default=0.05)
    p.add_argument("--false_stop_weight", type=float, default=6.0)
    p.add_argument("--value_threshold", type=float, default=0.55)
    p.add_argument("--dagger_value_gate", type=int, default=0)
    p.add_argument("--typed_action_mask", type=int, default=1)
    p.add_argument("--eval_native", type=int, default=1)
    p.add_argument("--native_max_new_tokens", type=int, default=16)
    p.add_argument("--k_values", type=int, nargs="+", default=[0, 2, 4, 8, 12])
    return p


def main() -> None:
    run_experiment(parser().parse_args())


if __name__ == "__main__":
    main()

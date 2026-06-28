#!/usr/bin/env python3
"""Fuyu-style whole-network VM loop with GRPO-style ECHO training.

One Qwen forward pass is one private recurrent VM transition. Each turn receives
prompt tokens plus learned dense tokens describing the current typed VM state via
``inputs_embeds``. The model emits no natural-language action tokens; direct
heads predict one structured VM edit action or STOP.

Training starts with behavior cloning from oracle traces, then uses sampled
complete rollouts with group-normalized rewards. A structured ECHO auxiliary
loss trains the same hidden state to predict the VM observation caused by each
sampled action.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
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
NEG_ACTION_PAD = -1
MAX_NEG_ACTIONS = 8


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


def edit_distance(a: BytecodeProgram, b: BytecodeProgram) -> int:
    a = normalize_program(a)
    b = normalize_program(b)
    return sum(int(x != y) for x, y in zip(a.ops, b.ops)) + sum(int(x != y) for x, y in zip(a.args, b.args))


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


def legal_edit_actions(program: BytecodeProgram, arg_mask: torch.Tensor, include_stop: bool = False) -> List[int]:
    prog = normalize_program(program)
    actions: List[int] = [STOP_ID] if include_stop else []
    constants = [i for i, ok in enumerate(arg_mask.tolist()) if ok]
    for slot in range(MAX_PROGRAM_LEN):
        depth, ok = prefix_depth(prog, slot)
        allowed = allowed_ops_for_depth(depth, slot, MAX_PROGRAM_LEN) if ok else []
        for op in allowed:
            if int(prog.ops[slot]) != int(op):
                actions.append(action_op(slot, int(op)))
        if int(prog.ops[slot]) == OP_TO_ID["PUSH"]:
            cur_arg = int(prog.args[slot]) % MODULUS
            for arg in constants:
                if int(arg) != cur_arg:
                    actions.append(action_arg(slot, int(arg)))
    return actions


def first_edit_toward(current: BytecodeProgram, target: BytecodeProgram) -> int:
    current = normalize_program(current)
    target = normalize_program(target)
    for slot, (cur_op, tgt_op) in enumerate(zip(current.ops, target.ops)):
        if int(cur_op) != int(tgt_op):
            return action_op(slot, int(tgt_op))
    for slot, (cur_arg, tgt_arg) in enumerate(zip(current.args, target.args)):
        if int(cur_arg) != int(tgt_arg):
            return action_arg(slot, int(tgt_arg))
    return STOP_ID


def complete_toward_target(
    current: BytecodeProgram,
    target: BytecodeProgram,
    answer: int,
    max_edits: int,
) -> Tuple[BytecodeProgram, List[int], bool]:
    """Greedily complete toward a target program, then verify only by answer."""
    program = normalize_program(current)
    actions: List[int] = []
    for _ in range(max(0, int(max_edits))):
        if is_answer_correct(program, answer):
            return program, actions, True
        action = first_edit_toward(program, target)
        if action == STOP_ID:
            return program, actions, is_answer_correct(program, answer)
        program, _ = apply_action(program, action)
        actions.append(action)
    return program, actions, is_answer_correct(program, answer)


def repair_search(
    current: BytecodeProgram,
    ex: TaskExample,
    arg_mask: torch.Tensor,
    max_edits: int,
    max_first_actions: int,
    max_second_actions: int,
) -> Tuple[Optional[BytecodeProgram], List[int], List[int], int, int]:
    """Bounded answer-verified repair from the current policy state."""
    current = normalize_program(current)
    if is_answer_correct(current, ex.answer):
        return current, [STOP_ID], [], 1, 1

    actions = legal_edit_actions(current, arg_mask, include_stop=False)
    if not actions:
        return None, [], [], 0, 0
    actions = sorted(actions, key=lambda a: edit_distance(apply_action(current, a)[0], ex.program))
    first_actions = actions[:max_first_actions]
    searched = found = 0
    negatives: List[int] = []
    best: Optional[Tuple[int, int, BytecodeProgram, List[int]]] = None

    for a1 in first_actions:
        p1, _ = apply_action(current, a1)
        searched += 1
        if is_answer_correct(p1, ex.answer):
            found += 1
            cand = (1, edit_distance(p1, ex.program), p1, [a1])
            if best is None or cand[:2] < best[:2]:
                best = cand
            continue
        negatives.append(a1)
        if max_edits < 2:
            continue
        a2s = legal_edit_actions(p1, arg_mask, include_stop=False)
        a2s = sorted(a2s, key=lambda a: edit_distance(apply_action(p1, a)[0], ex.program))[:max_second_actions]
        for a2 in a2s:
            p2, _ = apply_action(p1, a2)
            searched += 1
            if is_answer_correct(p2, ex.answer):
                found += 1
                cand = (2, edit_distance(p2, ex.program), p2, [a1, a2])
                if best is None or cand[:2] < best[:2]:
                    best = cand
            elif len(negatives) < 64:
                negatives.append(a1)

    # The shallow verifier above only finds states that are already within one
    # or two edits of a correct program. For early on-policy states, judge each
    # candidate first action by whether a bounded completion can still reach a
    # verified answer within the shortest target-edit budget. This keeps the
    # label attached to the policy's own state while turning global trajectory
    # choice into a direct ranking problem.
    remaining_budget = min(int(max_edits), max(1, oracle_distance(current, ex.program, ex.answer, stop_on_answer=True)))
    if remaining_budget > 2:
        gold_first = first_edit_toward(current, ex.program)
        guided_actions = list(first_actions)
        if gold_first != STOP_ID and gold_first not in guided_actions:
            guided_actions = [gold_first] + guided_actions[: max(0, max_first_actions - 1)]
        for a1 in guided_actions:
            p1, _ = apply_action(current, a1)
            completed, suffix, ok = complete_toward_target(p1, ex.program, ex.answer, remaining_budget - 1)
            searched += 1 + len(suffix)
            if ok:
                found += 1
                cand = (1 + len(suffix), edit_distance(completed, ex.program), completed, [a1] + suffix)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            elif len(negatives) < 64:
                negatives.append(a1)

    if best is None:
        return normalize_program(ex.program), [first_edit_toward(current, ex.program)], list(dict.fromkeys(negatives))[:32], found, searched
    return best[2], best[3], list(dict.fromkeys(negatives))[:32], found, searched


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
    next_trace_top: List[int]
    next_trace_depth: List[int]
    next_trace_mask: List[float]
    next_valid: int
    next_final: int
    next_solved: int
    neg_actions: List[int]
    solved: int
    distance: float
    repair_found: int
    searched: int
    weight: float
    bc_weight: float
    reward: float
    advantage: float
    rollout_id: int
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
    repair_found_states: int
    search_states: int
    mean_search_candidates: float
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
    rank_pair_accuracy: float
    solved_bce: float
    distance_mae: float
    pg_loss: float
    echo_loss: float
    mean_reward: float
    mean_advantage: float
    next_valid_accuracy: float
    next_final_accuracy: float
    train_states: int
    stop_labels: int


@dataclass
class RolloutStats:
    run: str
    phase: str
    n_examples: int
    rollouts: int
    states: int
    success_rate: float
    mean_reward: float
    reward_std: float
    false_stop_rate: float
    invalid_final_rate: float
    stop_rate: float
    mean_steps: float
    mean_advantage_abs: float
    reachable_after_rate: float
    destroyed_reachability_rate: float
    mean_shaping_reward: float
    shuffled_rewards: int


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
        neg = list(s.neg_actions[:MAX_NEG_ACTIONS])
        while len(neg) < MAX_NEG_ACTIONS:
            neg.append(NEG_ACTION_PAD)
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
            "next_trace_top": torch.tensor(s.next_trace_top, dtype=torch.long),
            "next_trace_depth": torch.tensor(s.next_trace_depth, dtype=torch.long),
            "next_trace_mask": torch.tensor(s.next_trace_mask, dtype=torch.float32),
            "next_valid": torch.tensor(s.next_valid, dtype=torch.long),
            "next_final": torch.tensor(s.next_final, dtype=torch.long),
            "next_solved": torch.tensor(s.next_solved, dtype=torch.float32),
            "neg_actions": torch.tensor(neg, dtype=torch.long),
            "solved": torch.tensor(s.solved, dtype=torch.float32),
            "distance": torch.tensor(s.distance, dtype=torch.float32),
            "repair_found": torch.tensor(s.repair_found, dtype=torch.float32),
            "searched": torch.tensor(s.searched, dtype=torch.float32),
            "weight": torch.tensor(s.weight, dtype=torch.float32),
            "bc_weight": torch.tensor(s.bc_weight, dtype=torch.float32),
            "reward": torch.tensor(s.reward, dtype=torch.float32),
            "advantage": torch.tensor(s.advantage, dtype=torch.float32),
            "rollout_id": torch.tensor(s.rollout_id, dtype=torch.long),
        }


def collate_states(batch: Sequence[Dict[str, torch.Tensor]], prompts: PromptSet) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in [
        "ex_idx",
        "ops",
        "args",
        "trace_top",
        "trace_depth",
        "trace_mask",
        "valid",
        "final",
        "step",
        "action",
        "next_trace_top",
        "next_trace_depth",
        "next_trace_mask",
        "next_valid",
        "next_final",
        "next_solved",
        "neg_actions",
        "solved",
        "distance",
        "repair_found",
        "searched",
        "weight",
        "bc_weight",
        "reward",
        "advantage",
        "rollout_id",
    ]:
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
        self.next_valid_head = nn.Linear(hidden_size, 2)
        self.next_final_head = nn.Linear(hidden_size, MODULUS)
        self.next_trace_top_head = nn.Linear(hidden_size, MAX_PROGRAM_LEN * MODULUS)
        self.next_trace_depth_head = nn.Linear(hidden_size, MAX_PROGRAM_LEN * (MAX_PROGRAM_LEN + 1))
        self.next_trace_mask_head = nn.Linear(hidden_size, MAX_PROGRAM_LEN)
        self.next_solved_head = nn.Linear(hidden_size, 1)

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
            "next_valid_logits": self.next_valid_head(pooled),
            "next_final_logits": self.next_final_head(pooled),
            "next_trace_top_logits": self.next_trace_top_head(pooled).view(-1, MAX_PROGRAM_LEN, MODULUS),
            "next_trace_depth_logits": self.next_trace_depth_head(pooled).view(-1, MAX_PROGRAM_LEN, MAX_PROGRAM_LEN + 1),
            "next_trace_mask_logits": self.next_trace_mask_head(pooled),
            "next_solved_logits": self.next_solved_head(pooled).squeeze(-1),
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


def state_from_program(
    ex_idx: int,
    ex: TaskExample,
    program: BytecodeProgram,
    step: int,
    action: int,
    source: str,
    weight: float = 1.0,
    neg_actions: Optional[Sequence[int]] = None,
    repair_found: int = 1,
    searched: int = 0,
    bc_weight: float = 1.0,
    reward: float = 0.0,
    advantage: float = 0.0,
    rollout_id: int = -1,
) -> StateSample:
    prog = normalize_program(program)
    valid, final, top, depth, mask = observation(prog)
    next_prog, _ = apply_action(prog, action)
    next_valid, next_final, next_top, next_depth, next_mask = observation(next_prog)
    next_solved = int(next_valid and next_final == int(ex.answer))
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
        next_trace_top=next_top,
        next_trace_depth=next_depth,
        next_trace_mask=next_mask,
        next_valid=next_valid,
        next_final=next_final,
        next_solved=next_solved,
        neg_actions=list(neg_actions or [])[:MAX_NEG_ACTIONS],
        solved=solved,
        distance=float(dist),
        repair_found=int(repair_found),
        searched=int(searched),
        weight=float(weight),
        bc_weight=float(bc_weight),
        reward=float(reward),
        advantage=float(advantage),
        rollout_id=int(rollout_id),
        source=source,
    )


def collect_teacher_states(examples: Sequence[TaskExample], max_steps: int, phase: str) -> Tuple[List[StateSample], TrajectoryStats]:
    samples: List[StateSample] = []
    stop_labels = edit_labels = false_stop_states = policy_steps = success = 0
    for idx, ex in enumerate(examples):
        current = blank_program()
        for step in range(max_steps + 1):
            action = oracle_next_action(current, ex.program, ex.answer, stop_on_answer=True)
            samples.append(state_from_program(idx, ex, current, step, action, phase, repair_found=1, searched=0))
            if action == STOP_ID:
                stop_labels += 1
                break
            edit_labels += 1
            policy_steps += 1
            current, _ = apply_action(current, action)
        success += int(is_answer_correct(current, ex.answer))
    n = max(1, len(examples))
    return samples, TrajectoryStats("", phase, len(examples), len(samples), stop_labels, edit_labels, false_stop_states, len(samples), 0, 0.0, success / n, len(samples) / n, policy_steps / n)


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


def masked_action_logits_for_program(
    outputs: Dict[str, torch.Tensor],
    program: BytecodeProgram,
    arg_mask: Optional[torch.Tensor],
    forbid_stop: bool,
    value_gate: bool,
    value_threshold: float,
    typed_action_mask: bool,
) -> torch.Tensor:
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
    return logits


def apply_typed_action_mask_to_batch(
    action_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    typed_action_mask: bool,
) -> torch.Tensor:
    if not typed_action_mask:
        return action_logits
    logits = action_logits.clone()
    ops_batch = batch["ops"].detach().cpu().tolist()
    args_batch = batch["args"].detach().cpu().tolist()
    for row, (ops, args) in enumerate(zip(ops_batch, args_batch)):
        prog = normalize_program(BytecodeProgram(list(map(int, ops)), list(map(int, args))))
        for slot in range(MAX_PROGRAM_LEN):
            depth, ok = prefix_depth(prog, slot)
            allowed = set(allowed_ops_for_depth(depth, slot, MAX_PROGRAM_LEN)) if ok else set()
            for op in range(len(OPCODES)):
                idx = action_op(slot, op)
                if op not in allowed or int(prog.ops[slot]) == op:
                    logits[row, idx] = -1e9
            if int(prog.ops[slot]) != OP_TO_ID["PUSH"]:
                logits[row, action_arg(slot, 0) : action_arg(slot, 0) + MODULUS] = -1e9
            else:
                current_arg = int(prog.args[slot]) % MODULUS
                logits[row, action_arg(slot, current_arg)] = -1e9
    return logits


def sample_action_from_outputs(
    outputs: Dict[str, torch.Tensor],
    program: BytecodeProgram,
    arg_mask: Optional[torch.Tensor],
    temperature: float,
    typed_action_mask: bool,
) -> int:
    logits = masked_action_logits_for_program(
        outputs,
        program,
        arg_mask,
        forbid_stop=False,
        value_gate=False,
        value_threshold=0.0,
        typed_action_mask=typed_action_mask,
    )
    if float(temperature) <= 0.0:
        return int(torch.argmax(logits).item())
    finite = torch.isfinite(logits)
    if not bool(finite.any()):
        return STOP_ID
    scaled = logits / max(float(temperature), 1e-4)
    probs = torch.softmax(scaled, dim=-1)
    if not bool(torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits).item())
    return int(torch.distributions.Categorical(probs=probs).sample().item())


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
def collect_search_augmented_states(
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    device: torch.device,
    rollout_steps: int,
    phase: str,
    value_gate: bool,
    value_threshold: float,
    typed_action_mask: bool,
    false_stop_weight: float,
    repair_max_edits: int,
    repair_first_actions: int,
    repair_second_actions: int,
) -> Tuple[List[StateSample], TrajectoryStats]:
    agent.eval()
    samples: List[StateSample] = []
    stop_labels = edit_labels = false_stop_states = policy_steps = success = 0
    repair_found_states = search_states = searched_total = 0
    for idx, ex in enumerate(prompt_set.examples):
        current = blank_program()
        arg_mask = prompt_set.arg_masks[idx]
        for step in range(rollout_steps + 1):
            repair_target, repair_actions, negatives, found, searched = repair_search(
                current,
                ex,
                arg_mask,
                max_edits=repair_max_edits,
                max_first_actions=repair_first_actions,
                max_second_actions=repair_second_actions,
            )
            search_states += 1
            searched_total += int(searched)
            repair_found = int(found > 0)
            repair_found_states += repair_found
            if repair_actions:
                oracle = int(repair_actions[0])
            elif repair_target is not None:
                oracle = first_edit_toward(current, repair_target)
            else:
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
            false_stop = int(pred == STOP_ID and not is_answer_correct(current, ex.answer))
            weight = false_stop_weight if false_stop else 1.0
            samples.append(
                state_from_program(
                    idx,
                    ex,
                    current,
                    step,
                    oracle,
                    phase,
                    weight=weight,
                    neg_actions=negatives,
                    repair_found=repair_found,
                    searched=searched,
                )
            )
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
    mean_searched = searched_total / max(1, search_states)
    return samples, TrajectoryStats("", phase, len(prompt_set.examples), len(samples), stop_labels, edit_labels, false_stop_states, repair_found_states, search_states, mean_searched, success / n, len(samples) / n, policy_steps / n)


def terminal_reward(
    program: BytecodeProgram,
    ex: TaskExample,
    stopped: bool,
    steps: int,
    reward_correct: float,
    reward_exact_bonus: float,
    reward_false_stop_penalty: float,
    reward_invalid_penalty: float,
    reward_step_penalty: float,
) -> Tuple[float, bool, bool, bool]:
    valid, final, _ = execute_program(normalize_program(program))
    ok = bool(valid and int(final) == int(ex.answer))
    false_stop = bool(stopped and not ok)
    invalid = bool(not valid)
    reward = float(reward_correct) if ok else 0.0
    if program_equal(program, ex.program):
        reward += float(reward_exact_bonus)
    if false_stop:
        reward -= float(reward_false_stop_penalty)
    if invalid:
        reward -= float(reward_invalid_penalty)
    reward -= float(reward_step_penalty) * float(steps)
    return reward, ok, false_stop, invalid


def reachability_signal(
    program: BytecodeProgram,
    ex: TaskExample,
    arg_mask: torch.Tensor,
    budget: int,
    repair_max_edits: int,
    repair_first_actions: int,
    repair_second_actions: int,
) -> Tuple[bool, int, int]:
    prog = normalize_program(program)
    if is_answer_correct(prog, ex.answer):
        return True, 0, 0
    dist = oracle_distance(prog, ex.program, ex.answer, stop_on_answer=True)
    budget = max(0, int(budget))
    if dist <= budget:
        _, _, ok = complete_toward_target(prog, ex.program, ex.answer, dist)
        if ok:
            return True, int(dist), 0
    if repair_max_edits > 0 and budget > 0:
        _, actions, _, found, searched = repair_search(
            prog,
            ex,
            arg_mask,
            max_edits=min(int(repair_max_edits), budget),
            max_first_actions=int(repair_first_actions),
            max_second_actions=int(repair_second_actions),
        )
        if found > 0:
            return True, min(int(dist), len(actions) if actions else 0), int(searched)
        return False, int(dist), int(searched)
    return False, int(dist), 0


@torch.no_grad()
def collect_grpo_rollouts(
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    device: torch.device,
    rollout_steps: int,
    rollouts_per_prompt: int,
    phase: str,
    temperature: float,
    typed_action_mask: bool,
    reward_correct: float,
    reward_exact_bonus: float,
    reward_false_stop_penalty: float,
    reward_invalid_penalty: float,
    reward_step_penalty: float,
    reward_shaping: bool,
    reward_reachable_bonus: float,
    reward_destroy_reachability_penalty: float,
    reward_progress_weight: float,
    reward_regress_penalty: float,
    shape_repair_max_edits: int,
    shape_repair_first_actions: int,
    shape_repair_second_actions: int,
    shuffle_rollout_rewards: bool,
) -> Tuple[List[StateSample], RolloutStats]:
    agent.eval()
    all_samples: List[StateSample] = []
    rollout_rewards: List[float] = []
    rollout_steps_taken: List[int] = []
    advantage_abs: List[float] = []
    shaping_rewards: List[float] = []
    success = false_stop_count = invalid_count = stop_count = rollout_count = 0
    reachable_after_count = destroyed_reachability_count = transition_count = 0
    for idx, ex in enumerate(prompt_set.examples):
        prompt_rollouts: List[Tuple[List[StateSample], float, int, bool, bool, bool, bool]] = []
        for _ in range(max(1, int(rollouts_per_prompt))):
            rollout_id = rollout_count
            rollout_count += 1
            current = blank_program()
            traj: List[StateSample] = []
            stopped = False
            steps_taken = 0
            shaped_return = 0.0
            arg_mask = prompt_set.arg_masks[idx]
            for step in range(max(1, int(rollout_steps))):
                probe = state_from_program(idx, ex, current, step, STOP_ID, "rollout_probe", rollout_id=rollout_id)
                batch = sample_to_batch(prompt_set, probe, device)
                outputs = agent(batch)
                before_reachable, before_dist, _ = reachability_signal(
                    current,
                    ex,
                    arg_mask,
                    budget=max(0, int(rollout_steps) - step),
                    repair_max_edits=shape_repair_max_edits if reward_shaping else 0,
                    repair_first_actions=shape_repair_first_actions,
                    repair_second_actions=shape_repair_second_actions,
                )
                action = sample_action_from_outputs(
                    outputs,
                    current,
                    batch.get("arg_mask"),
                    temperature=temperature,
                    typed_action_mask=typed_action_mask,
                )
                traj.append(state_from_program(idx, ex, current, step, action, phase, bc_weight=0.0, rollout_id=rollout_id))
                if action == STOP_ID:
                    stopped = True
                    ok_now = is_answer_correct(current, ex.answer)
                    if reward_shaping:
                        step_reward = float(reward_reachable_bonus) if ok_now else -float(reward_false_stop_penalty)
                        shaped_return += step_reward
                        shaping_rewards.append(step_reward)
                    break
                next_program, _ = apply_action(current, action)
                if reward_shaping:
                    after_reachable, after_dist, _ = reachability_signal(
                        next_program,
                        ex,
                        arg_mask,
                        budget=max(0, int(rollout_steps) - step - 1),
                        repair_max_edits=shape_repair_max_edits,
                        repair_first_actions=shape_repair_first_actions,
                        repair_second_actions=shape_repair_second_actions,
                    )
                    progress = float(before_dist - after_dist)
                    step_reward = 0.0
                    if after_reachable:
                        step_reward += float(reward_reachable_bonus)
                    if before_reachable and not after_reachable:
                        step_reward -= float(reward_destroy_reachability_penalty)
                        destroyed_reachability_count += 1
                    if progress > 0:
                        step_reward += float(reward_progress_weight) * min(1.0, progress)
                    elif progress < 0:
                        step_reward -= float(reward_regress_penalty) * min(1.0, abs(progress))
                    shaped_return += step_reward
                    shaping_rewards.append(step_reward)
                    reachable_after_count += int(after_reachable)
                    transition_count += 1
                current = next_program
                steps_taken += 1
            reward, ok, false_stop, invalid = terminal_reward(
                current,
                ex,
                stopped,
                steps_taken,
                reward_correct,
                reward_exact_bonus,
                reward_false_stop_penalty,
                reward_invalid_penalty,
                reward_step_penalty,
            )
            reward += shaped_return
            prompt_rollouts.append((traj, reward, steps_taken, stopped, ok, false_stop, invalid))
        rewards = torch.tensor([r[1] for r in prompt_rollouts], dtype=torch.float32)
        train_rewards = rewards.clone()
        if shuffle_rollout_rewards and train_rewards.numel() > 1:
            train_rewards = train_rewards[torch.randperm(train_rewards.numel())]
        mean = train_rewards.mean()
        std = train_rewards.std(unbiased=False)
        if float(std.item()) > 1e-6:
            advantages = (train_rewards - mean) / std
        else:
            advantages = train_rewards - mean
        for rollout_idx, (traj, reward, steps_taken, stopped, ok, false_stop, invalid) in enumerate(prompt_rollouts):
            adv = float(advantages[rollout_idx].item())
            for sample in traj:
                sample.reward = float(train_rewards[rollout_idx].item())
                sample.advantage = adv
                all_samples.append(sample)
                advantage_abs.append(abs(adv))
            rollout_rewards.append(float(reward))
            rollout_steps_taken.append(int(steps_taken))
            success += int(ok)
            false_stop_count += int(false_stop)
            invalid_count += int(invalid)
            stop_count += int(stopped)
    reward_tensor = torch.tensor(rollout_rewards or [0.0], dtype=torch.float32)
    stats = RolloutStats(
        run="",
        phase=phase,
        n_examples=len(prompt_set.examples),
        rollouts=rollout_count,
        states=len(all_samples),
        success_rate=success / max(1, rollout_count),
        mean_reward=float(reward_tensor.mean().item()),
        reward_std=float(reward_tensor.std(unbiased=False).item()),
        false_stop_rate=false_stop_count / max(1, rollout_count),
        invalid_final_rate=invalid_count / max(1, rollout_count),
        stop_rate=stop_count / max(1, rollout_count),
        mean_steps=sum(rollout_steps_taken) / max(1, len(rollout_steps_taken)),
        mean_advantage_abs=sum(advantage_abs) / max(1, len(advantage_abs)),
        reachable_after_rate=reachable_after_count / max(1, transition_count),
        destroyed_reachability_rate=destroyed_reachability_count / max(1, transition_count),
        mean_shaping_reward=sum(shaping_rewards) / max(1, len(shaping_rewards)),
        shuffled_rewards=int(bool(shuffle_rollout_rewards)),
    )
    return all_samples, stats


def train_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    stop_loss_weight: float,
    value_loss_weight: float,
    distance_loss_weight: float,
    rank_loss_weight: float,
    bc_loss_weight: float,
    pg_loss_weight: float,
    echo_loss_weight: float,
    typed_action_mask: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    action = batch["action"]
    kind_t, slot_t, op_t, arg_t = action_components(action)
    sample_w = batch["weight"].float()
    bc_sample_w = sample_w * batch["bc_weight"].float()
    raw_action_logits = full_action_logits(outputs, arg_mask=batch.get("arg_mask"))
    action_logits = apply_typed_action_mask_to_batch(raw_action_logits, batch, typed_action_mask)
    target_scores = action_logits.gather(1, action.view(-1, 1))
    raw_target_scores = raw_action_logits.gather(1, action.view(-1, 1))
    action_logits = action_logits.scatter(
        1,
        action.view(-1, 1),
        torch.where(target_scores.lt(-1e8), raw_target_scores, target_scores),
    )
    action_loss_items = F.cross_entropy(action_logits, action, reduction="none")
    action_w = bc_sample_w * torch.where(kind_t.eq(0), torch.full_like(sample_w, float(stop_loss_weight)), torch.ones_like(sample_w))
    action_loss = (action_loss_items * action_w).sum() / action_w.sum().clamp_min(1.0)
    kind_loss_items = F.cross_entropy(outputs["kind_logits"], kind_t, reduction="none")
    stop_w = torch.where(kind_t.eq(0), torch.full_like(sample_w, float(stop_loss_weight)), torch.ones_like(sample_w))
    kind_loss = (kind_loss_items * bc_sample_w * stop_w).sum() / (bc_sample_w * stop_w).sum().clamp_min(1.0)
    edit_mask = kind_t.ne(0)
    op_mask = kind_t.eq(1)
    arg_mask = kind_t.eq(2)
    bc_edit_mask = edit_mask & batch["bc_weight"].float().gt(0)
    bc_op_mask = op_mask & batch["bc_weight"].float().gt(0)
    bc_arg_mask = arg_mask & batch["bc_weight"].float().gt(0)
    slot_loss = F.cross_entropy(outputs["slot_logits"][bc_edit_mask], slot_t[bc_edit_mask]) if bool(bc_edit_mask.any()) else action_loss * 0.0
    op_loss = F.cross_entropy(outputs["op_logits"][bc_op_mask], op_t[bc_op_mask]) if bool(bc_op_mask.any()) else action_loss * 0.0
    masked_arg_logits = outputs["arg_logits"]
    if "arg_mask" in batch:
        masked_arg_logits = masked_arg_logits.masked_fill(~batch["arg_mask"].bool(), -1e9)
    arg_loss = F.cross_entropy(masked_arg_logits[bc_arg_mask], arg_t[bc_arg_mask]) if bool(bc_arg_mask.any()) else action_loss * 0.0
    solved_bce = F.binary_cross_entropy_with_logits(outputs["solved_logits"], batch["solved"].float(), reduction="none")
    solved_loss = (solved_bce * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    dist_target = torch.log1p(batch["distance"].float())
    dist_loss_items = F.smooth_l1_loss(outputs["distance"], dist_target, reduction="none")
    dist_loss = (dist_loss_items * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    neg_actions = batch["neg_actions"].long()
    neg_mask = neg_actions.ge(0)
    safe_neg = neg_actions.clamp_min(0)
    pos_scores = action_logits.gather(1, action.view(-1, 1))
    neg_scores = action_logits.gather(1, safe_neg)
    rank_items = F.softplus(-(pos_scores - neg_scores))
    rank_w = sample_w.view(-1, 1) * neg_mask.float()
    rank_loss = (rank_items * rank_w).sum() / rank_w.sum().clamp_min(1.0)
    rank_acc = ((pos_scores > neg_scores).float() * neg_mask.float()).sum() / neg_mask.float().sum().clamp_min(1.0)
    chosen_logp = F.log_softmax(action_logits, dim=-1).gather(1, action.view(-1, 1)).squeeze(1)
    advantage = batch["advantage"].float().detach()
    pg_loss = -((chosen_logp * advantage) * sample_w).sum() / sample_w.sum().clamp_min(1.0)

    next_valid_loss = F.cross_entropy(outputs["next_valid_logits"], batch["next_valid"].long(), reduction="none")
    next_valid_loss = (next_valid_loss * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    next_final_loss = F.cross_entropy(outputs["next_final_logits"], batch["next_final"].long(), reduction="none")
    next_final_loss = (next_final_loss * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    next_mask_loss_items = F.binary_cross_entropy_with_logits(
        outputs["next_trace_mask_logits"],
        batch["next_trace_mask"].float(),
        reduction="none",
    ).mean(dim=1)
    next_mask_loss = (next_mask_loss_items * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    trace_mask = batch["next_trace_mask"].bool()
    if bool(trace_mask.any()):
        next_top_loss = F.cross_entropy(outputs["next_trace_top_logits"][trace_mask], batch["next_trace_top"].long()[trace_mask])
        next_depth_loss = F.cross_entropy(outputs["next_trace_depth_logits"][trace_mask], batch["next_trace_depth"].long()[trace_mask])
        next_top_acc = outputs["next_trace_top_logits"].argmax(dim=-1)[trace_mask].eq(batch["next_trace_top"].long()[trace_mask]).float().mean()
    else:
        next_top_loss = action_loss * 0.0
        next_depth_loss = action_loss * 0.0
        next_top_acc = action_loss.detach() * 0.0
    next_solved_loss_items = F.binary_cross_entropy_with_logits(outputs["next_solved_logits"], batch["next_solved"].float(), reduction="none")
    next_solved_loss = (next_solved_loss_items * sample_w).sum() / sample_w.sum().clamp_min(1.0)
    echo_loss = (
        next_valid_loss
        + next_final_loss
        + 0.5 * next_mask_loss
        + 0.5 * next_top_loss
        + 0.5 * next_depth_loss
        + 0.5 * next_solved_loss
    )
    aux_loss = kind_loss + slot_loss + op_loss + arg_loss
    bc_loss = action_loss + 0.25 * aux_loss + float(rank_loss_weight) * rank_loss
    loss = (
        float(bc_loss_weight) * bc_loss
        + float(pg_loss_weight) * pg_loss
        + float(echo_loss_weight) * echo_loss
        + float(value_loss_weight) * solved_loss
        + float(distance_loss_weight) * dist_loss
    )
    next_valid_acc = outputs["next_valid_logits"].argmax(dim=-1).eq(batch["next_valid"].long()).float().mean()
    next_final_acc = outputs["next_final_logits"].argmax(dim=-1).eq(batch["next_final"].long()).float().mean()
    return loss, {
        "action_loss": float(action_loss.detach().cpu()),
        "kind_loss": float(kind_loss.detach().cpu()),
        "slot_loss": float(slot_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "rank_pair_accuracy": float(rank_acc.detach().cpu()),
        "solved_bce": float(solved_loss.detach().cpu()),
        "distance_mae": float(torch.mean(torch.abs(torch.expm1(outputs["distance"].detach().float()).clamp_min(0) - batch["distance"].float())).detach().cpu()),
        "pg_loss": float(pg_loss.detach().cpu()),
        "echo_loss": float(echo_loss.detach().cpu()),
        "mean_reward": float(batch["reward"].float().mean().detach().cpu()),
        "mean_advantage": float(batch["advantage"].float().mean().detach().cpu()),
        "next_valid_accuracy": float(next_valid_acc.detach().cpu()),
        "next_final_accuracy": float(next_final_acc.detach().cpu()),
        "next_trace_top_accuracy": float(next_top_acc.detach().cpu()),
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
    rank_loss_weight: float,
    bc_loss_weight: float,
    pg_loss_weight: float,
    echo_loss_weight: float,
    typed_action_mask: bool,
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
        solved_bce_total = dist_mae_total = rank_acc_total = 0.0
        pg_loss_total = echo_loss_total = reward_total = advantage_total = 0.0
        next_valid_acc_total = next_final_acc_total = 0.0
        opt.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = agent(batch)
            loss, parts = train_loss(
                outputs,
                batch,
                stop_loss_weight,
                value_loss_weight,
                distance_loss_weight,
                rank_loss_weight,
                bc_loss_weight,
                pg_loss_weight,
                echo_loss_weight,
                typed_action_mask,
            )
            (loss / accum).backward()
            if batch_idx % accum == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in agent.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            pred_logits = apply_typed_action_mask_to_batch(full_action_logits(outputs, arg_mask=batch.get("arg_mask")), batch, typed_action_mask)
            pred = pred_logits.argmax(dim=-1)
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
            rank_acc_total += parts["rank_pair_accuracy"] * n
            pg_loss_total += parts["pg_loss"] * n
            echo_loss_total += parts["echo_loss"] * n
            reward_total += parts["mean_reward"] * n
            advantage_total += parts["mean_advantage"] * n
            next_valid_acc_total += parts["next_valid_accuracy"] * n
            next_final_acc_total += parts["next_final_accuracy"] * n
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
            rank_pair_accuracy=rank_acc_total / max(1, total_n),
            solved_bce=solved_bce_total / max(1, total_n),
            distance_mae=dist_mae_total / max(1, total_n),
            pg_loss=pg_loss_total / max(1, total_n),
            echo_loss=echo_loss_total / max(1, total_n),
            mean_reward=reward_total / max(1, total_n),
            mean_advantage=advantage_total / max(1, total_n),
            next_valid_accuracy=next_valid_acc_total / max(1, total_n),
            next_final_accuracy=next_final_acc_total / max(1, total_n),
            train_states=len(samples),
            stop_labels=sum(1 for s in samples if s.action == STOP_ID),
        )
        rows.append(row)
        append_csv(log_path, asdict(row), rewrite=not log_path.exists() and epoch == 1)
        print(
            f"[train:{phase}] epoch={epoch} loss={row.loss:.4f} "
            f"action_acc={100*row.action_accuracy:.1f}% stop_acc={100*row.stop_accuracy:.1f}% "
            f"pg={row.pg_loss:.4f} echo={row.echo_loss:.4f} "
            f"next_valid={100*row.next_valid_accuracy:.1f}% next_final={100*row.next_final_accuracy:.1f}%",
            flush=True,
        )
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
def run_value_beam(
    agent: DenseStateQwenAgent,
    prompt_set: PromptSet,
    ex_idx: int,
    k: int,
    device: torch.device,
    value_threshold: float,
    typed_action_mask: bool,
    beam_width: int,
    expand_actions: int,
) -> Tuple[BytecodeProgram, bool, int, List[str]]:
    beams: List[Tuple[float, BytecodeProgram, bool, List[str]]] = [(0.0, blank_program(), False, [])]
    ex = prompt_set.examples[ex_idx]
    for step in range(k):
        expanded: List[Tuple[float, BytecodeProgram, bool, List[str]]] = []
        for score, program, stopped, actions in beams:
            if stopped:
                expanded.append((score, program, stopped, actions))
                continue
            sample = state_from_program(ex_idx, ex, program, step, STOP_ID, "beam")
            batch = sample_to_batch(prompt_set, sample, device)
            outputs = agent(batch)
            logits = masked_action_logits_for_program(outputs, program, batch.get("arg_mask"), False, False, value_threshold, typed_action_mask)
            top = torch.topk(logits, k=min(expand_actions, logits.numel())).indices.tolist()
            solved_logit = float(outputs["solved_logits"][0].detach().float().cpu())
            distance_penalty = float(torch.expm1(outputs["distance"][0].detach().float().cpu()).clamp_min(0).item())
            for action in top:
                action = int(action)
                if action == STOP_ID:
                    expanded.append((score + float(logits[action]) + solved_logit, program, True, actions + [action_to_text(action)]))
                    continue
                next_program, _ = apply_action(program, action)
                expanded.append((score + float(logits[action]) + 0.25 * solved_logit - 0.02 * distance_penalty, next_program, False, actions + [action_to_text(action)]))
        beams = sorted(expanded, key=lambda x: x[0], reverse=True)[:beam_width]
    # Re-score final states by current value estimate rather than answer.
    rescored: List[Tuple[float, BytecodeProgram, bool, List[str]]] = []
    for score, program, stopped, actions in beams:
        sample = state_from_program(ex_idx, ex, program, k, STOP_ID, "beam_final")
        batch = sample_to_batch(prompt_set, sample, device)
        outputs = agent(batch)
        solved_logit = float(outputs["solved_logits"][0].detach().float().cpu())
        distance_penalty = float(torch.expm1(outputs["distance"][0].detach().float().cpu()).clamp_min(0).item())
        rescored.append((score + 2.0 * solved_logit - 0.05 * distance_penalty, program, stopped, actions))
    best = max(rescored, key=lambda x: x[0])
    return best[1], best[2], min(k, len(best[3])), best[3]


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
    beam_width: int,
    beam_expand_actions: int,
    eval_modes: Sequence[str],
) -> List[EvalRow]:
    blank_ok = sum(int(is_answer_correct(blank_program(), ex.answer)) for ex in prompt_set.examples)
    n = max(1, len(prompt_set.examples))
    blank_acc = blank_ok / n
    rows: List[EvalRow] = []
    modes = ["blank", "oracle_teacher"]
    if agent is not None:
        modes.extend([mode for mode in eval_modes if mode in {"learned", "value_gated", "forced", "beam_value"}])
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
                elif mode == "forced":
                    assert agent is not None
                    prog, stopped, steps, _ = run_policy(agent, prompt_set, idx, k, device, "forced", value_threshold, typed_action_mask)
                else:
                    assert agent is not None
                    prog, stopped, steps, _ = run_value_beam(agent, prompt_set, idx, k, device, value_threshold, typed_action_mask, beam_width, beam_expand_actions)
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
    beam_width: int,
    beam_expand_actions: int,
    eval_modes: Sequence[str],
    count: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(prompt_set.examples[:count]):
        for mode in [m for m in eval_modes if m in {"learned", "value_gated", "forced", "beam_value"}]:
            if mode == "beam_value":
                prog, stopped, steps, actions = run_value_beam(agent, prompt_set, idx, k, device, value_threshold, typed_action_mask, beam_width, beam_expand_actions)
            else:
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
    beam_width: int,
    beam_expand_actions: int,
    eval_modes: Sequence[str],
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for split in ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]:
        print(f"[eval] phase={phase} split={split}", flush=True)
        rows.extend(evaluate_split(agent, prompt_sets[split], native_metrics.get(split, 0.0), run_name, phase, split, k_values, device, value_threshold, typed_action_mask, beam_width, beam_expand_actions, eval_modes))
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
            "next_valid_head": agent.next_valid_head.state_dict(),
            "next_final_head": agent.next_final_head.state_dict(),
            "next_trace_top_head": agent.next_trace_top_head.state_dict(),
            "next_trace_depth_head": agent.next_trace_depth_head.state_dict(),
            "next_trace_mask_head": agent.next_trace_mask_head.state_dict(),
            "next_solved_head": agent.next_solved_head.state_dict(),
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
    for filename in ["metrics.csv", "train_log.csv", "trajectory_stats.csv", "rollout_stats.csv", "native_qwen_metrics.csv", "rollout_samples.csv"]:
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
        args.rank_loss_weight,
        1.0,
        0.0,
        args.bc_echo_weight,
        bool(args.typed_action_mask),
        run_dir / "train_log.csv",
    )
    save_agent(agent, CHECKPOINT_ROOT / args.run_name / "bc_policy", args)
    metrics.extend(evaluate_all(agent, prompt_sets, native_metrics, args.run_name, "bc_policy", args.k_values, device, args.value_threshold, bool(args.typed_action_mask), args.beam_width, args.beam_expand_actions, args.eval_modes))
    write_rows(run_dir / "metrics.csv", [asdict(r) for r in metrics])
    write_rollout_samples(run_dir / "rollout_samples.csv", agent, prompt_sets["val_mixed"], "bc_policy", "val_mixed", device, max(args.k_values), args.value_threshold, bool(args.typed_action_mask), args.beam_width, args.beam_expand_actions, args.eval_modes, min(3, len(prompt_sets["val_mixed"])))

    for round_idx in range(1, args.rl_rounds + 1):
        teacher_anchor = [
            replace(s, source=f"teacher_anchor_r{round_idx}", weight=float(args.rl_teacher_anchor_weight), reward=0.0, advantage=0.0, rollout_id=-1)
            for s in teacher_states
        ]
        if args.algorithm == "process_dpo":
            traj_phase = f"process_dpo_r{round_idx}_states"
            policy_phase = f"process_dpo_r{round_idx}_policy"
            states, stats = collect_search_augmented_states(
                agent,
                prompt_sets["train"],
                device,
                args.rl_rollout_steps,
                traj_phase,
                value_gate=bool(args.preference_value_gate),
                value_threshold=args.value_threshold,
                typed_action_mask=bool(args.typed_action_mask),
                false_stop_weight=args.false_stop_weight,
                repair_max_edits=args.preference_repair_max_edits,
                repair_first_actions=args.preference_repair_first_actions,
                repair_second_actions=args.preference_repair_second_actions,
            )
            stats.run = args.run_name
            append_csv(run_dir / "trajectory_stats.csv", asdict(stats))
            print(
                f"[states:{traj_phase}] states={len(states)} false_stop={stats.false_stop_states} "
                f"repair_found={stats.repair_found_states}/{stats.search_states} "
                f"mean_candidates={stats.mean_search_candidates:.1f} rollout_success={stats.rollout_success_rate:.3f}",
                flush=True,
            )
            combined_states = list(states) + teacher_anchor
            bc_weight = 1.0
            pg_weight = 0.0
        else:
            traj_phase = f"grpo_echo_r{round_idx}_rollouts"
            policy_phase = f"grpo_echo_r{round_idx}_policy"
            states, stats = collect_grpo_rollouts(
                agent,
                prompt_sets["train"],
                device,
                args.rl_rollout_steps,
                args.rollouts_per_prompt,
                traj_phase,
                temperature=args.temperature,
                typed_action_mask=bool(args.typed_action_mask),
                reward_correct=args.reward_correct,
                reward_exact_bonus=args.reward_exact_bonus,
                reward_false_stop_penalty=args.reward_false_stop_penalty,
                reward_invalid_penalty=args.reward_invalid_penalty,
                reward_step_penalty=args.reward_step_penalty,
                reward_shaping=bool(args.reward_shaping),
                reward_reachable_bonus=args.reward_reachable_bonus,
                reward_destroy_reachability_penalty=args.reward_destroy_reachability_penalty,
                reward_progress_weight=args.reward_progress_weight,
                reward_regress_penalty=args.reward_regress_penalty,
                shape_repair_max_edits=args.shape_repair_max_edits,
                shape_repair_first_actions=args.shape_repair_first_actions,
                shape_repair_second_actions=args.shape_repair_second_actions,
                shuffle_rollout_rewards=bool(args.shuffle_rollout_rewards),
            )
            stats.run = args.run_name
            append_csv(run_dir / "rollout_stats.csv", asdict(stats), rewrite=not (run_dir / "rollout_stats.csv").exists())
            print(
                f"[rollout:{traj_phase}] states={len(states)} rollouts={stats.rollouts} "
                f"success={stats.success_rate:.3f} reward={stats.mean_reward:.3f} "
                f"std={stats.reward_std:.3f} false_stop={stats.false_stop_rate:.3f} "
                f"|adv|={stats.mean_advantage_abs:.3f}",
                flush=True,
            )
            combined_states = list(states) + teacher_anchor
            bc_weight = args.rl_bc_loss_weight
            pg_weight = args.pg_loss_weight
        train_agent(
            agent,
            prompt_sets["train"],
            combined_states,
            device,
            policy_phase,
            args.run_name,
            args.rl_epochs,
            args.batch_size,
            args.grad_accum,
            args.rl_lr,
            args.stop_loss_weight,
            args.value_loss_weight,
            args.distance_loss_weight,
            args.rank_loss_weight,
            bc_weight,
            pg_weight,
            args.echo_loss_weight,
            bool(args.typed_action_mask),
            run_dir / "train_log.csv",
        )
        save_agent(agent, CHECKPOINT_ROOT / args.run_name / policy_phase, args)
        metrics.extend(evaluate_all(agent, prompt_sets, native_metrics, args.run_name, policy_phase, args.k_values, device, args.value_threshold, bool(args.typed_action_mask), args.beam_width, args.beam_expand_actions, args.eval_modes))
        write_rows(run_dir / "metrics.csv", [asdict(r) for r in metrics])
        write_rollout_samples(run_dir / "rollout_samples.csv", agent, prompt_sets["val_mixed"], policy_phase, "val_mixed", device, max(args.k_values), args.value_threshold, bool(args.typed_action_mask), args.beam_width, args.beam_expand_actions, args.eval_modes, min(3, len(prompt_sets["val_mixed"])))

    with (run_dir / "results.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "metrics": [asdict(r) for r in metrics]}, f, indent=2)
    with (ROOT / "checkpoint_manifest.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"Fuyu-style whole-network VM agent; algorithm={args.algorithm}; base={args.model_name}"})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--algorithm", choices=["grpo_echo", "process_dpo"], default="grpo_echo")
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
    p.add_argument("--rl_lr", type=float, default=5e-5)
    p.add_argument("--bc_epochs", type=int, default=2)
    p.add_argument("--rl_epochs", type=int, default=1)
    p.add_argument("--rl_rounds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--teacher_max_steps", type=int, default=12)
    p.add_argument("--rl_rollout_steps", type=int, default=8)
    p.add_argument("--rollouts_per_prompt", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--stop_loss_weight", type=float, default=4.0)
    p.add_argument("--value_loss_weight", type=float, default=0.5)
    p.add_argument("--distance_loss_weight", type=float, default=0.05)
    p.add_argument("--rank_loss_weight", type=float, default=0.5)
    p.add_argument("--bc_echo_weight", type=float, default=0.05)
    p.add_argument("--echo_loss_weight", type=float, default=0.1)
    p.add_argument("--pg_loss_weight", type=float, default=1.0)
    p.add_argument("--rl_bc_loss_weight", type=float, default=0.25)
    p.add_argument("--rl_teacher_anchor_weight", type=float, default=0.25)
    p.add_argument("--false_stop_weight", type=float, default=6.0)
    p.add_argument("--preference_value_gate", type=int, default=0)
    p.add_argument("--preference_repair_max_edits", type=int, default=4)
    p.add_argument("--preference_repair_first_actions", type=int, default=64)
    p.add_argument("--preference_repair_second_actions", type=int, default=48)
    p.add_argument("--reward_correct", type=float, default=1.0)
    p.add_argument("--reward_exact_bonus", type=float, default=0.25)
    p.add_argument("--reward_false_stop_penalty", type=float, default=0.25)
    p.add_argument("--reward_invalid_penalty", type=float, default=0.1)
    p.add_argument("--reward_step_penalty", type=float, default=0.01)
    p.add_argument("--reward_shaping", type=int, default=1)
    p.add_argument("--reward_reachable_bonus", type=float, default=0.05)
    p.add_argument("--reward_destroy_reachability_penalty", type=float, default=0.35)
    p.add_argument("--reward_progress_weight", type=float, default=0.08)
    p.add_argument("--reward_regress_penalty", type=float, default=0.05)
    p.add_argument("--shape_repair_max_edits", type=int, default=2)
    p.add_argument("--shape_repair_first_actions", type=int, default=48)
    p.add_argument("--shape_repair_second_actions", type=int, default=24)
    p.add_argument("--shuffle_rollout_rewards", type=int, default=0)
    p.add_argument("--value_threshold", type=float, default=0.55)
    p.add_argument("--typed_action_mask", type=int, default=1)
    p.add_argument("--beam_width", type=int, default=3)
    p.add_argument("--beam_expand_actions", type=int, default=4)
    p.add_argument("--eval_native", type=int, default=1)
    p.add_argument("--native_max_new_tokens", type=int, default=16)
    p.add_argument("--k_values", type=int, nargs="+", default=[0, 2, 4, 8, 12])
    p.add_argument("--eval_modes", nargs="+", default=["learned", "value_gated", "forced", "beam_value"])
    return p


def main() -> None:
    run_experiment(parser().parse_args())


if __name__ == "__main__":
    main()

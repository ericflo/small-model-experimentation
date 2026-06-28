#!/usr/bin/env python3
"""Train a Qwen-attached hidden VM compiler with canonical on-policy repair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("experiments/qwen_hidden_vm_onpolicy_canonical_repair")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_hidden_vm_onpolicy_canonical_repair/checkpoints")

VM_OPS = ["ADD", "SUB", "MUL", "ADD7", "SUB7", "SET", "MAX", "MIN", "XOR", "GT"]
OP_TO_ID = {name: idx for idx, name in enumerate(VM_OPS)}
DOMAIN_NAMES = ["arithmetic", "calendar", "unit", "list", "boolean", "lookup"]
VERY_NEG = -1e4


@dataclass
class VMProgramSpec:
    domain: str
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    answer: int
    metadata: Dict[str, Any]


@dataclass
class VMExample:
    prompt: str
    domain: str
    template_mode: str
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    answer: int
    init_pos: int
    step_op_pos: List[int]
    step_arg_pos: List[int]
    answer_pos: int
    input_ids: List[int]
    num_values: List[int]
    op_values: List[int]


@dataclass
class ExampleSet:
    examples: List[VMExample]
    paired: bool = False
    pair_size: int = 1

    def __len__(self) -> int:
        return len(self.examples)


@dataclass
class RepairOutcome:
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    answer: int
    found: bool
    changed: bool
    program_exact: bool
    num_candidates: int
    num_verified: int


@dataclass
class OnPolicyTargetSet:
    examples: List[VMExample]
    init: torch.Tensor
    ops: torch.Tensor
    args: torch.Tensor
    active_mask: torch.Tensor
    found: torch.Tensor
    changed: torch.Tensor
    program_exact: torch.Tensor
    candidate_count: torch.Tensor
    verified_count: torch.Tensor
    source_counts: Dict[str, int]


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)


def tokenize_no_special(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    return list(ids)


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def apply_vm_op(x: int, op: int, arg: int, modulus: int) -> int:
    op_name = VM_OPS[int(op)]
    x = int(x)
    arg = int(arg)
    if op_name == "ADD":
        return (x + arg) % modulus
    if op_name == "SUB":
        return (x - arg) % modulus
    if op_name == "MUL":
        return (x * arg) % modulus
    if op_name == "ADD7":
        return (x + arg) % 7
    if op_name == "SUB7":
        return (x - arg) % 7
    if op_name == "SET":
        return arg % modulus
    if op_name == "MAX":
        return max(x, arg) % modulus
    if op_name == "MIN":
        return min(x, arg) % modulus
    if op_name == "XOR":
        return (x ^ arg) % modulus
    if op_name == "GT":
        return 1 if x > arg else 0
    raise ValueError(op_name)


def execute_program(init_value: int, ops: Sequence[int], args: Sequence[int], length: int, max_steps: int, modulus: int) -> Tuple[int, List[int]]:
    x = int(init_value)
    states = [-100] * max_steps
    for t in range(int(length)):
        x = apply_vm_op(x, int(ops[t]), int(args[t]), modulus)
        states[t] = x
    return x, states


class MixedDomainGenerator:
    def __init__(self, tokenizer: Any, modulus: int, max_steps: int, seed: int) -> None:
        self.tokenizer = tokenizer
        self.modulus = int(modulus)
        self.max_steps = int(max_steps)
        self.rng = random.Random(seed)

    def _choice_for_mode(self, mode: str, standard: Sequence[Any], paraphrase: Sequence[Any]) -> Any:
        if mode == "standard":
            return standard[0]
        if mode == "paraphrase":
            return self.rng.choice(list(paraphrase))
        if mode == "mixed":
            return self.rng.choice(list(standard) + list(paraphrase))
        raise ValueError(mode)

    def _run(self, init_value: int, ops: Sequence[int], args: Sequence[int]) -> Tuple[int, List[int]]:
        return execute_program(init_value, ops, args, len(ops), self.max_steps, self.modulus)

    def make_spec(self, domain: str, min_len: int, max_len: int) -> VMProgramSpec:
        if domain not in DOMAIN_NAMES:
            raise ValueError(domain)
        length = self.rng.randint(max(1, min_len), max(1, max_len))
        if domain == "arithmetic":
            init = self.rng.randrange(self.modulus)
            ops: List[int] = []
            args: List[int] = []
            for _ in range(length):
                op = self.rng.choice([OP_TO_ID["ADD"], OP_TO_ID["SUB"], OP_TO_ID["MUL"]])
                arg = self.rng.randint(2, 12) if op == OP_TO_ID["MUL"] else self.rng.randint(1, 40)
                ops.append(op)
                args.append(arg)
            answer, states = self._run(init, ops, args)
            return VMProgramSpec(domain, length, init, ops, args, states, answer, {})

        if domain == "calendar":
            init = self.rng.randrange(7)
            ops = []
            args = []
            for _ in range(length):
                ops.append(self.rng.choice([OP_TO_ID["ADD7"], OP_TO_ID["SUB7"]]))
                args.append(self.rng.randint(1, 14))
            answer, states = self._run(init, ops, args)
            return VMProgramSpec(domain, length, init, ops, args, states, answer, {})

        if domain == "unit":
            init = self.rng.randint(1, 48)
            ops = []
            args = []
            for _ in range(length):
                op = self.rng.choice([OP_TO_ID["ADD"], OP_TO_ID["SUB"], OP_TO_ID["MUL"]])
                if op == OP_TO_ID["MUL"]:
                    arg = self.rng.choice([2, 3])
                else:
                    arg = self.rng.randint(1, 25)
                ops.append(op)
                args.append(arg)
            answer, states = self._run(init, ops, args)
            return VMProgramSpec(domain, length, init, ops, args, states, answer, {})

        if domain == "list":
            init = self.rng.randint(0, self.modulus - 1)
            ops = [self.rng.choice([OP_TO_ID["MAX"], OP_TO_ID["MIN"]]) for _ in range(length)]
            args = [self.rng.randint(0, self.modulus - 1) for _ in range(length)]
            answer, states = self._run(init, ops, args)
            return VMProgramSpec(domain, length, init, ops, args, states, answer, {})

        if domain == "boolean":
            init = self.rng.randint(0, self.modulus - 1)
            ops = []
            args = []
            for _ in range(max(0, length - 1)):
                op = self.rng.choice([OP_TO_ID["ADD"], OP_TO_ID["SUB"], OP_TO_ID["XOR"]])
                arg = self.rng.randint(1, 31) if op == OP_TO_ID["XOR"] else self.rng.randint(1, 20)
                ops.append(op)
                args.append(arg)
            ops.append(OP_TO_ID["GT"])
            args.append(self.rng.randint(10, 86))
            answer, states = self._run(init, ops, args)
            return VMProgramSpec(domain, len(ops), init, ops, args, states, answer, {})

        init = 0
        keys = ["A", "B", "C", "D"]
        values = [self.rng.randint(0, self.modulus - 1) for _ in keys]
        selected = self.rng.randrange(len(keys))
        ops = [OP_TO_ID["SET"]]
        args = [values[selected]]
        for _ in range(max(0, length - 1)):
            op = self.rng.choice([OP_TO_ID["ADD"], OP_TO_ID["SUB"], OP_TO_ID["MAX"]])
            arg = self.rng.randint(1, 35) if op != OP_TO_ID["MAX"] else self.rng.randint(0, self.modulus - 1)
            ops.append(op)
            args.append(arg)
        answer, states = self._run(init, ops, args)
        return VMProgramSpec(
            domain,
            len(ops),
            init,
            ops,
            args,
            states,
            answer,
            {"keys": keys, "values": values, "selected": selected},
        )

    def render(self, spec: VMProgramSpec, template_mode: str) -> VMExample:
        input_ids: List[int] = []
        num_values: List[int] = []
        op_values: List[int] = []
        parts: List[str] = []

        def add_text(text: str, num_value: int = -1, op_value: int = -1) -> int:
            ids = tokenize_no_special(self.tokenizer, text)
            if not ids:
                raise RuntimeError(f"empty tokenization for {text!r}")
            input_ids.extend(ids)
            num_values.extend([num_value] * len(ids))
            op_values.extend([op_value] * len(ids))
            parts.append(text)
            return len(input_ids) - 1

        def add_number(value: int) -> int:
            return add_text(str(int(value)), num_value=int(value) % self.modulus)

        def add_op_word(text: str, op: int) -> int:
            return add_text(text, op_value=int(op))

        op_pos: List[int] = []
        arg_pos: List[int] = []
        init_pos = -1

        add_text(self._choice_for_mode(template_mode, ["Solve the task with the hidden VM.\n"], ["Use the hidden machine to solve this.\n", "Compute the requested value using the private VM.\n"]))

        if spec.domain == "arithmetic":
            add_text(self._choice_for_mode(template_mode, ["Domain: arithmetic chain.\nInitial x = "], ["Arithmetic task. Start x at ", "Track an integer. Initial value is "]))
            init_pos = add_number(spec.init_value)
            add_text(".\n")
            for op, arg in zip(spec.ops, spec.args):
                if op == OP_TO_ID["ADD"]:
                    prefix, word, between, suffix = self._choice_for_mode(template_mode, [("Step: ", "add", " ", ".\n")], [("Then ", "increase", " by ", ".\n"), ("Apply ", "add", " with ", ".\n")])
                elif op == OP_TO_ID["SUB"]:
                    prefix, word, between, suffix = self._choice_for_mode(template_mode, [("Step: ", "subtract", " ", ".\n")], [("Then ", "decrease", " by ", ".\n"), ("Apply ", "subtract", " with ", ".\n")])
                else:
                    prefix, word, between, suffix = self._choice_for_mode(template_mode, [("Step: ", "multiply", " by ", ".\n")], [("Then ", "scale", " by ", ".\n"), ("Apply ", "multiply", " using ", ".\n")])
                add_text(prefix)
                op_pos.append(add_op_word(word, op))
                add_text(between)
                arg_pos.append(add_number(arg))
                add_text(suffix)

        elif spec.domain == "calendar":
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            add_text(self._choice_for_mode(template_mode, ["Domain: calendar shift. Day codes use Mon=0 through Sun=6.\nStart day code "], ["Calendar task. Encode Monday as 0 and Sunday as 6. Begin on code ", "Shift a weekday number where Mon is 0 and Sun is 6. Initial code: "]))
            init_pos = add_number(spec.init_value)
            add_text(f" ({day_names[spec.init_value]}).\n")
            for op, arg in zip(spec.ops, spec.args):
                if op == OP_TO_ID["ADD7"]:
                    prefix, word, between, suffix = self._choice_for_mode(template_mode, [("Step: move ", "forward", " ", " days.\n")], [("Then go ", "ahead", " by ", " days.\n"), ("Shift ", "forward", " ", " days.\n")])
                else:
                    prefix, word, between, suffix = self._choice_for_mode(template_mode, [("Step: move ", "back", " ", " days.\n")], [("Then go ", "backward", " by ", " days.\n"), ("Shift ", "back", " ", " days.\n")])
                add_text(prefix)
                op_pos.append(add_op_word(word, op))
                add_text(between)
                arg_pos.append(add_number(arg))
                add_text(suffix)

        elif spec.domain == "unit":
            add_text(self._choice_for_mode(template_mode, ["Domain: unit conversion.\nInitial reading = "], ["Convert a machine reading. Start with ", "Unit task. The beginning reading is "]))
            init_pos = add_number(spec.init_value)
            add_text(".\n")
            for op, arg in zip(spec.ops, spec.args):
                if op == OP_TO_ID["MUL"]:
                    word = "double" if arg == 2 else "triple"
                    prefix, between, suffix = self._choice_for_mode(template_mode, [("Step: ", " by ", ".\n")], [("Then ", " using factor ", ".\n"), ("Apply ", " with multiplier ", ".\n")])
                    add_text(prefix)
                    op_pos.append(add_op_word(word, op))
                    add_text(between)
                    arg_pos.append(add_number(arg))
                    add_text(suffix)
                elif op == OP_TO_ID["ADD"]:
                    add_text(self._choice_for_mode(template_mode, ["Step: add offset "], ["Then include offset ", "Increase by offset "]))
                    op_pos.append(add_op_word("add", op))
                    add_text(" ")
                    arg_pos.append(add_number(arg))
                    add_text(".\n")
                else:
                    add_text(self._choice_for_mode(template_mode, ["Step: subtract offset "], ["Then remove offset ", "Decrease by offset "]))
                    op_pos.append(add_op_word("subtract", op))
                    add_text(" ")
                    arg_pos.append(add_number(arg))
                    add_text(".\n")

        elif spec.domain == "list":
            add_text(self._choice_for_mode(template_mode, ["Domain: list aggregation.\nFirst value = "], ["Aggregate a list. Begin with ", "List rule. The first item is "]))
            init_pos = add_number(spec.init_value)
            add_text(".\n")
            for op, arg in zip(spec.ops, spec.args):
                if op == OP_TO_ID["MAX"]:
                    add_text(self._choice_for_mode(template_mode, ["Step: take "], ["Then keep the ", "Update with the "]))
                    op_pos.append(add_op_word("max", op))
                    add_text(" with ")
                else:
                    add_text(self._choice_for_mode(template_mode, ["Step: take "], ["Then keep the ", "Update with the "]))
                    op_pos.append(add_op_word("min", op))
                    add_text(" with ")
                arg_pos.append(add_number(arg))
                add_text(".\n")

        elif spec.domain == "boolean":
            add_text(self._choice_for_mode(template_mode, ["Domain: boolean threshold.\nInitial score = "], ["Boolean rule. Start score ", "Threshold task. Initial score is "]))
            init_pos = add_number(spec.init_value)
            add_text(".\n")
            for op, arg in zip(spec.ops, spec.args):
                if op == OP_TO_ID["ADD"]:
                    add_text(self._choice_for_mode(template_mode, ["Step: add "], ["Adjust upward by ", "Increase score by "]))
                    op_pos.append(add_op_word("add", op))
                    add_text(" ")
                    arg_pos.append(add_number(arg))
                    add_text(".\n")
                elif op == OP_TO_ID["SUB"]:
                    add_text(self._choice_for_mode(template_mode, ["Step: subtract "], ["Adjust downward by ", "Decrease score by "]))
                    op_pos.append(add_op_word("subtract", op))
                    add_text(" ")
                    arg_pos.append(add_number(arg))
                    add_text(".\n")
                elif op == OP_TO_ID["XOR"]:
                    add_text(self._choice_for_mode(template_mode, ["Step: xor "], ["Mix with xor value ", "Apply xor mask "]))
                    op_pos.append(add_op_word("xor", op))
                    add_text(" ")
                    arg_pos.append(add_number(arg))
                    add_text(".\n")
                else:
                    add_text(self._choice_for_mode(template_mode, ["Step: greater than "], ["Return whether score is above ", "Test if score is greater than "]))
                    op_pos.append(add_op_word("greater", op))
                    add_text(" ")
                    arg_pos.append(add_number(arg))
                    add_text(".\n")

        else:
            keys = spec.metadata["keys"]
            values = spec.metadata["values"]
            selected = int(spec.metadata["selected"])
            add_text(self._choice_for_mode(template_mode, ["Domain: lookup and adjust.\nStart x = "], ["Lookup task. Begin x at ", "Use a table. Initial x is "]))
            init_pos = add_number(spec.init_value)
            add_text(".\n")
            add_text("Rows: ")
            for i, (key, value) in enumerate(zip(keys, values)):
                add_text(f"{key}->")
                if i == selected:
                    selected_value_pos = add_number(value)
                else:
                    add_number(value)
                add_text("; " if i + 1 < len(keys) else ".\n")
            add_text(f"Use key {keys[selected]}.\n")
            add_text(self._choice_for_mode(template_mode, ["Step: set selected value "], ["Step: choose selected value ", "Set x to the matched value "]))
            op_pos.append(add_op_word("set", OP_TO_ID["SET"]))
            add_text(" ")
            arg_pos.append(add_number(values[selected]))
            add_text(".\n")
            for op, arg in zip(spec.ops[1:], spec.args[1:]):
                if op == OP_TO_ID["ADD"]:
                    add_text(self._choice_for_mode(template_mode, ["Step: add "], ["Then increase by ", "After lookup add "]))
                    op_pos.append(add_op_word("add", op))
                    add_text(" ")
                elif op == OP_TO_ID["SUB"]:
                    add_text(self._choice_for_mode(template_mode, ["Step: subtract "], ["Then decrease by ", "After lookup subtract "]))
                    op_pos.append(add_op_word("subtract", op))
                    add_text(" ")
                else:
                    add_text(self._choice_for_mode(template_mode, ["Step: max with "], ["Then take max with ", "After lookup keep max with "]))
                    op_pos.append(add_op_word("max", op))
                    add_text(" ")
                arg_pos.append(add_number(arg))
                add_text(".\n")

        add_text(self._choice_for_mode(template_mode, ["Return the final VM value.\n"], ["Report the final hidden-machine value.\n", "Give the final integer.\n"]))
        answer_pos = add_text(self._choice_for_mode(template_mode, ["Answer:\n"], ["Final answer:\n", "Result:\n"]))
        if init_pos < 0 or len(op_pos) != spec.length or len(arg_pos) != spec.length:
            raise RuntimeError(f"bad render for {spec.domain}: init={init_pos} ops={len(op_pos)} args={len(arg_pos)} length={spec.length}")
        return VMExample(
            prompt="".join(parts).rstrip("\n"),
            domain=spec.domain,
            template_mode=template_mode,
            length=spec.length,
            init_value=spec.init_value,
            ops=spec.ops,
            args=spec.args,
            states=spec.states,
            answer=spec.answer,
            init_pos=init_pos,
            step_op_pos=op_pos,
            step_arg_pos=arg_pos,
            answer_pos=answer_pos,
            input_ids=input_ids,
            num_values=num_values,
            op_values=op_values,
        )

    def dataset(self, n: int, min_len: int, max_len: int, template_mode: str, domains: Sequence[str]) -> ExampleSet:
        examples = []
        domain_list = list(domains)
        for i in range(n):
            domain = domain_list[i % len(domain_list)]
            examples.append(self.render(self.make_spec(domain, min_len, max_len), template_mode))
        self.rng.shuffle(examples)
        return ExampleSet(examples)

    def domain_dataset(self, n: int, min_len: int, max_len: int, template_mode: str, domain: str) -> ExampleSet:
        return ExampleSet([self.render(self.make_spec(domain, min_len, max_len), template_mode) for _ in range(n)])

    def paired_dataset(self, n_pairs: int, min_len: int, max_len: int, domains: Sequence[str]) -> ExampleSet:
        examples: List[VMExample] = []
        domain_list = list(domains)
        for i in range(n_pairs):
            spec = self.make_spec(domain_list[i % len(domain_list)], min_len, max_len)
            examples.append(self.render(spec, "standard"))
            examples.append(self.render(spec, "paraphrase"))
        return ExampleSet(examples, paired=True, pair_size=2)


def collate_examples(
    examples: Sequence[VMExample],
    pad_id: int,
    max_steps: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    seq_len = min(max(len(ex.input_ids) for ex in examples), max_length)
    ids = torch.full((len(examples), seq_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(examples), seq_len), dtype=torch.long, device=device)
    num_values: List[List[int]] = []
    op_values: List[List[int]] = []
    ops: List[List[int]] = []
    args: List[List[int]] = []
    states: List[List[int]] = []
    op_pos: List[List[int]] = []
    arg_pos: List[List[int]] = []
    lengths: List[int] = []
    init_values: List[int] = []
    answers: List[int] = []
    init_pos: List[int] = []
    answer_pos: List[int] = []
    domains: List[str] = []
    for i, ex in enumerate(examples):
        if ex.answer_pos >= seq_len:
            raise RuntimeError(f"max_length={max_length} truncates required answer position; prompt length={len(ex.input_ids)}")
        cur = ex.input_ids[:seq_len]
        ids[i, : len(cur)] = torch.tensor(cur, dtype=torch.long, device=device)
        mask[i, : len(cur)] = 1
        num_values.append(ex.num_values[:seq_len] + [-1] * (seq_len - len(cur)))
        op_values.append(ex.op_values[:seq_len] + [-1] * (seq_len - len(cur)))
        ops.append(ex.ops + [-100] * (max_steps - len(ex.ops)))
        args.append(ex.args + [-100] * (max_steps - len(ex.args)))
        states.append(ex.states + [-100] * (max_steps - len(ex.states)))
        op_pos.append(ex.step_op_pos + [-100] * (max_steps - len(ex.step_op_pos)))
        arg_pos.append(ex.step_arg_pos + [-100] * (max_steps - len(ex.step_arg_pos)))
        lengths.append(ex.length)
        init_values.append(ex.init_value)
        answers.append(ex.answer)
        init_pos.append(ex.init_pos)
        answer_pos.append(ex.answer_pos)
        domains.append(ex.domain)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "hidden_mask": mask.bool(),
        "num_values": torch.tensor(num_values, dtype=torch.long, device=device),
        "op_values": torch.tensor(op_values, dtype=torch.long, device=device),
        "ops": torch.tensor(ops, dtype=torch.long, device=device),
        "args": torch.tensor(args, dtype=torch.long, device=device),
        "states": torch.tensor(states, dtype=torch.long, device=device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "init_value": torch.tensor(init_values, dtype=torch.long, device=device),
        "answer": torch.tensor(answers, dtype=torch.long, device=device),
        "init_pos": torch.tensor(init_pos, dtype=torch.long, device=device),
        "answer_pos": torch.tensor(answer_pos, dtype=torch.long, device=device),
        "op_pos": torch.tensor(op_pos, dtype=torch.long, device=device),
        "arg_pos": torch.tensor(arg_pos, dtype=torch.long, device=device),
        "domains": domains,
    }


def copy_logits_from_token_map(scores: torch.Tensor, token_values: torch.Tensor, num_classes: int) -> torch.Tensor:
    classes = torch.arange(num_classes, device=scores.device)
    value_matches = token_values[:, :, None].eq(classes[None, None, :])
    value_matches = value_matches & token_values[:, :, None].ge(0)
    if scores.dim() == 2:
        masked = scores[:, :, None].masked_fill(~value_matches, VERY_NEG)
        return torch.logsumexp(masked, dim=1)
    if scores.dim() == 3:
        masked = scores[:, :, :, None].masked_fill(~value_matches[:, None, :, :], VERY_NEG)
        return torch.logsumexp(masked, dim=2)
    raise ValueError(tuple(scores.shape))


class HiddenVMCompiler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        value_modulus: int,
        width: int,
        max_steps: int,
        rank_temperature: float,
        arg_window: int,
        arg_distance_temperature: float,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.rank_temperature = float(rank_temperature)
        self.arg_window = int(arg_window)
        self.arg_distance_temperature = float(arg_distance_temperature)
        self.value_modulus = int(value_modulus)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.token_ff = nn.Sequential(nn.Linear(hidden_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.role_head = nn.Linear(width, 3)
        nn.init.constant_(self.role_head.bias, -2.5)

    def _monotonic_slot_scores(self, role_logits: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
        mask = hidden_mask.bool()
        role_prob = torch.sigmoid(role_logits).masked_fill(~mask, 0.0)
        cumulative = torch.cumsum(role_prob, dim=1)
        centers = torch.arange(1, self.max_steps + 1, device=role_logits.device, dtype=role_logits.dtype)
        distance = cumulative[:, None, :] - centers[None, :, None]
        scores = torch.log(role_prob.clamp_min(1e-6))[:, None, :] - distance.square() / self.rank_temperature
        return scores.masked_fill(~mask[:, None, :], VERY_NEG)

    def _after_op_arg_scores(self, arg_role_logits: torch.Tensor, hidden_mask: torch.Tensor, op_slot_scores: torch.Tensor) -> torch.Tensor:
        mask = hidden_mask.bool()
        seq_len = arg_role_logits.shape[1]
        arg_role_score = torch.log(torch.sigmoid(arg_role_logits).clamp_min(1e-6)).masked_fill(~mask, VERY_NEG)
        op_log_w = F.log_softmax(op_slot_scores, dim=-1)
        positions = torch.arange(seq_len, device=arg_role_logits.device)
        distance = positions[None, :] - positions[:, None]
        valid = (distance > 0) & (distance <= self.arg_window)
        distance_penalty = -((distance.to(arg_role_logits.dtype) - 1.0).square() / self.arg_distance_temperature)
        distance_penalty = distance_penalty.masked_fill(~valid, VERY_NEG)
        anchored_scores = torch.logsumexp(op_log_w[:, :, :, None] + distance_penalty[None, None, :, :], dim=2)
        return (anchored_scores + arg_role_score[:, None, :]).masked_fill(~mask[:, None, :], VERY_NEG)

    def forward(
        self,
        hidden: torch.Tensor,
        hidden_mask: torch.Tensor,
        num_values: torch.Tensor,
        op_values: torch.Tensor,
        return_scores: bool = False,
    ) -> Any:
        x = self.token_ff(self.hidden_norm(hidden.float()))
        mask = hidden_mask.bool()
        role_logits = self.role_head(x)
        init_scores = role_logits[:, :, 0].masked_fill(~mask, VERY_NEG)
        op_slot_scores = self._monotonic_slot_scores(role_logits[:, :, 1], hidden_mask)
        arg_slot_scores = self._after_op_arg_scores(role_logits[:, :, 2], hidden_mask, op_slot_scores)
        init_logits = copy_logits_from_token_map(init_scores, num_values, self.value_modulus)
        op_logits = copy_logits_from_token_map(op_slot_scores, op_values, len(VM_OPS))
        arg_logits = copy_logits_from_token_map(arg_slot_scores, num_values, self.value_modulus)
        if return_scores:
            return init_logits, op_logits, arg_logits, {
                "role_logits": role_logits,
                "init_scores": init_scores,
                "op_slot_scores": op_slot_scores,
                "arg_slot_scores": arg_slot_scores,
            }
        return init_logits, op_logits, arg_logits


class HiddenVMExecutor(nn.Module):
    def __init__(self, value_modulus: int, device: torch.device) -> None:
        super().__init__()
        table = torch.zeros(len(VM_OPS), value_modulus, value_modulus, value_modulus, dtype=torch.float32)
        for op in range(len(VM_OPS)):
            for arg in range(value_modulus):
                for old in range(value_modulus):
                    table[op, arg, old, apply_vm_op(old, op, arg, value_modulus)] = 1.0
        self.register_buffer("table", table.to(device))

    def soft_trajectory(self, init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        state = F.softmax(init_logits.float(), dim=-1)
        op_probs = F.softmax(op_logits.float(), dim=-1)
        arg_probs = F.softmax(arg_logits.float(), dim=-1)
        states: List[torch.Tensor] = []
        for t in range(op_logits.shape[1]):
            cand = torch.einsum("bp,oapq->boaq", state, self.table)
            weights = op_probs[:, t, :, None] * arg_probs[:, t, None, :]
            next_state = (cand * weights[:, :, :, None]).sum(dim=(1, 2))
            active = (lengths > t).float().unsqueeze(-1)
            state = active * next_state + (1.0 - active) * state
            states.append(state.clamp_min(1e-9))
        return torch.stack(states, dim=1)

    def soft_forward(self, init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states = self.soft_trajectory(init_logits, op_logits, arg_logits, lengths)
        gather_idx = (lengths.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, states.shape[-1])
        return states.gather(1, gather_idx).squeeze(1).clamp_min(1e-9)


@torch.no_grad()
def argmax_execute_with_states(
    init_pred: torch.Tensor,
    op_pred: torch.Tensor,
    arg_pred: torch.Tensor,
    lengths: torch.Tensor,
    max_steps: int,
    modulus: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outs: List[int] = []
    state_rows: List[List[int]] = []
    for i in range(init_pred.shape[0]):
        answer, states = execute_program(
            int(init_pred[i].item()),
            [int(x) for x in op_pred[i].tolist()],
            [int(x) for x in arg_pred[i].tolist()],
            int(lengths[i].item()),
            max_steps,
            modulus,
        )
        outs.append(answer)
        state_rows.append(states)
    return torch.tensor(outs, dtype=torch.long, device=init_pred.device), torch.tensor(state_rows, dtype=torch.long, device=init_pred.device)


def candidate_log_score(
    init_value: int,
    ops: Sequence[int],
    args: Sequence[int],
    init_log_probs: torch.Tensor,
    op_log_probs: torch.Tensor,
    arg_log_probs: torch.Tensor,
    length: int,
) -> float:
    score = float(init_log_probs[int(init_value)].item())
    for t in range(length):
        score += float(op_log_probs[t, int(ops[t])].item())
        score += float(arg_log_probs[t, int(args[t])].item())
    return score


def repair_single_program(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    length: int,
    answer: int,
    target_states: Optional[Sequence[int]],
    gold_init: Optional[int],
    gold_ops: Optional[Sequence[int]],
    gold_args: Optional[Sequence[int]],
    max_steps: int,
    modulus: int,
    topk: int,
    max_edits: int,
    verifier_mode: str,
) -> RepairOutcome:
    init_log_probs = F.log_softmax(init_logits.float(), dim=-1).detach().cpu()
    op_log_probs = F.log_softmax(op_logits.float(), dim=-1).detach().cpu()
    arg_log_probs = F.log_softmax(arg_logits.float(), dim=-1).detach().cpu()
    init_top = torch.topk(init_log_probs, k=min(topk, init_log_probs.numel())).indices.tolist()
    op_top = [torch.topk(op_log_probs[t], k=min(topk, op_log_probs.shape[-1])).indices.tolist() for t in range(length)]
    arg_top = [torch.topk(arg_log_probs[t], k=min(topk, arg_log_probs.shape[-1])).indices.tolist() for t in range(length)]
    base_init = int(init_top[0])
    base_ops = [int(op_top[t][0]) for t in range(length)]
    base_args = [int(arg_top[t][0]) for t in range(length)]

    edits: List[Tuple[str, int, int, float]] = []
    for value in init_top[1:]:
        edits.append(("init", -1, int(value), float(init_log_probs[int(value)].item())))
    for t in range(length):
        for value in op_top[t][1:]:
            edits.append(("op", t, int(value), float(op_log_probs[t, int(value)].item())))
        for value in arg_top[t][1:]:
            edits.append(("arg", t, int(value), float(arg_log_probs[t, int(value)].item())))

    candidates: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], Tuple[int, float]] = {}

    def add_candidate(edit_set: Sequence[Tuple[str, int, int, float]]) -> None:
        cand_init = base_init
        cand_ops = list(base_ops)
        cand_args = list(base_args)
        changed = 0
        seen_slots = set()
        for kind, t, value, _ in edit_set:
            slot = (kind, t)
            if slot in seen_slots:
                return
            seen_slots.add(slot)
            if kind == "init":
                if cand_init != value:
                    changed += 1
                cand_init = value
            elif kind == "op":
                if cand_ops[t] != value:
                    changed += 1
                cand_ops[t] = value
            else:
                if cand_args[t] != value:
                    changed += 1
                cand_args[t] = value
        key = (cand_init, tuple(cand_ops), tuple(cand_args))
        if key not in candidates:
            score = candidate_log_score(cand_init, cand_ops, cand_args, init_log_probs, op_log_probs, arg_log_probs, length)
            candidates[key] = (changed, score)

    add_candidate([])
    if max_edits >= 1:
        for edit in edits:
            add_candidate([edit])
    if max_edits >= 2:
        for i in range(len(edits)):
            for j in range(i + 1, len(edits)):
                add_candidate([edits[i], edits[j]])

    ordered = sorted(candidates.items(), key=lambda item: (item[1][0], -item[1][1]))
    if verifier_mode not in {"state", "answer"}:
        raise ValueError(verifier_mode)
    gold_ops_short = list(gold_ops[:length]) if gold_ops is not None else None
    gold_args_short = list(gold_args[:length]) if gold_args is not None else None
    target_states_short = list(target_states[:length]) if target_states is not None else None

    num_verified = 0
    best_verified: Optional[RepairOutcome] = None
    best_fallback: Optional[RepairOutcome] = None
    for (cand_init, cand_ops_tuple, cand_args_tuple), (changed, _) in ordered:
        cand_ops = list(cand_ops_tuple)
        cand_args = list(cand_args_tuple)
        cand_answer, cand_states = execute_program(cand_init, cand_ops, cand_args, length, max_steps, modulus)
        program_exact = (
            gold_init is not None
            and gold_ops_short is not None
            and gold_args_short is not None
            and int(gold_init) == int(cand_init)
            and gold_ops_short == cand_ops[:length]
            and gold_args_short == cand_args[:length]
        )
        if best_fallback is None:
            best_fallback = RepairOutcome(
                cand_init,
                cand_ops,
                cand_args,
                cand_states,
                cand_answer,
                False,
                bool(changed),
                bool(program_exact),
                len(ordered),
                0,
            )
        if verifier_mode == "state":
            verified = target_states_short is not None and cand_states[:length] == target_states_short
        else:
            verified = cand_answer == int(answer)
        if verified:
            num_verified += 1
            if best_verified is None:
                best_verified = RepairOutcome(
                    cand_init,
                    cand_ops,
                    cand_args,
                    cand_states,
                    cand_answer,
                    True,
                    bool(changed),
                    bool(program_exact),
                    len(ordered),
                    0,
                )
    assert best_fallback is not None
    if best_verified is not None:
        best_verified.num_verified = num_verified
        return best_verified
    best_fallback.num_verified = num_verified
    return best_fallback


def forward_hidden_and_logits(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("model did not return hidden_states")
    logits = getattr(out, "logits", None)
    if logits is None:
        raise RuntimeError("model did not return logits")
    return hidden_states[-1], logits


def trace_losses(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    init_weight: float,
    op_weight: float,
    arg_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["ops"].ne(-100)
    init_loss = F.cross_entropy(init_logits, batch["init_value"])
    op_loss = F.cross_entropy(op_logits[active], batch["ops"][active])
    arg_loss = F.cross_entropy(arg_logits[active], batch["args"][active])
    return init_weight * init_loss + op_weight * op_loss + arg_weight * arg_loss, {
        "init_loss": float(init_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
    }


def selection_losses(scores: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["ops"].ne(-100)
    init_loss = F.cross_entropy(scores["init_scores"], batch["init_pos"])
    op_loss = F.cross_entropy(scores["op_slot_scores"][active], batch["op_pos"][active])
    arg_loss = F.cross_entropy(scores["arg_slot_scores"][active], batch["arg_pos"][active])
    role_targets = torch.zeros_like(scores["role_logits"])
    rows = torch.arange(role_targets.shape[0], device=role_targets.device)
    role_targets[rows, batch["init_pos"], 0] = 1.0
    for col, pos in [(1, batch["op_pos"]), (2, batch["arg_pos"])]:
        mask = pos.ne(-100)
        b_idx, _ = mask.nonzero(as_tuple=True)
        role_targets[b_idx, pos[mask], col] = 1.0
    valid = batch["hidden_mask"].bool()
    role_logits = scores["role_logits"][valid]
    role_targets_flat = role_targets[valid]
    pos_weight = torch.full((3,), args.role_pos_weight, dtype=role_logits.dtype, device=role_logits.device)
    role_loss = F.binary_cross_entropy_with_logits(role_logits, role_targets_flat, pos_weight=pos_weight)
    total = args.init_selection_loss_weight * init_loss + args.op_selection_loss_weight * op_loss + args.arg_selection_loss_weight * arg_loss + role_loss
    return total, {
        "init_attn_loss": float(init_loss.detach().cpu()),
        "op_attn_loss": float(op_loss.detach().cpu()),
        "arg_attn_loss": float(arg_loss.detach().cpu()),
        "role_loss": float(role_loss.detach().cpu()),
    }


def state_trajectory_loss(state_probs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["states"].ne(-100)
    if not active.any():
        zero = state_probs.sum() * 0.0
        return zero, {"state_loss": 0.0, "state_train_accuracy": 0.0}
    loss = F.nll_loss(state_probs[active].log(), batch["states"][active])
    pred = state_probs.argmax(dim=-1)
    acc = pred[active].eq(batch["states"][active]).float().mean()
    return loss, {"state_loss": float(loss.detach().cpu()), "state_train_accuracy": float(acc.detach().cpu())}


def answer_token_ids(tokenizer: Any, modulus: int, device: torch.device) -> torch.Tensor:
    ids = []
    for value in range(modulus):
        toks = tokenize_no_special(tokenizer, str(value))
        if not toks:
            raise RuntimeError(value)
        ids.append(int(toks[0]))
    return torch.tensor(ids, dtype=torch.long, device=device)


def direct_answer_logits(lm_logits: torch.Tensor, batch: Dict[str, torch.Tensor], answer_ids: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(lm_logits.shape[0], device=lm_logits.device)
    logits_at_answer = lm_logits[rows, batch["answer_pos"]]
    return logits_at_answer.index_select(-1, answer_ids)


def make_datasets(tokenizer: Any, args: argparse.Namespace) -> Dict[str, ExampleSet]:
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    for domain in domains:
        if domain not in DOMAIN_NAMES:
            raise ValueError(f"unknown domain {domain!r}")
    gen_train = MixedDomainGenerator(tokenizer, args.value_modulus, args.max_steps, args.seed + 101)
    gen_eval = MixedDomainGenerator(tokenizer, args.value_modulus, args.max_steps, args.eval_seed)
    datasets: Dict[str, ExampleSet] = {
        "train_mixed": gen_train.dataset(args.train_examples, args.train_min_len, args.train_max_len, args.train_template_mode, domains),
        "val_mixed": gen_eval.dataset(args.val_examples, args.eval_length, args.eval_length, "mixed", domains),
        "fresh_standard_mixed": gen_eval.dataset(args.eval_examples, args.eval_length, args.eval_length, "standard", domains),
        "fresh_paraphrase_mixed": gen_eval.dataset(args.eval_examples, args.eval_length, args.eval_length, "paraphrase", domains),
        "fresh_paired_mixed": gen_eval.paired_dataset(args.eval_pairs, args.eval_length, args.eval_length, domains),
    }
    if args.hard_length > args.eval_length:
        datasets["hard_standard_mixed"] = gen_eval.dataset(args.eval_examples, args.hard_length, args.hard_length, "standard", domains)
        datasets["hard_paraphrase_mixed"] = gen_eval.dataset(args.eval_examples, args.hard_length, args.hard_length, "paraphrase", domains)
    if args.harder_length > max(args.eval_length, args.hard_length):
        datasets["harder_standard_mixed"] = gen_eval.dataset(args.eval_examples, args.harder_length, args.harder_length, "standard", domains)
        datasets["harder_paraphrase_mixed"] = gen_eval.dataset(args.eval_examples, args.harder_length, args.harder_length, "paraphrase", domains)
    if args.domain_eval_examples > 0:
        for domain in domains:
            datasets[f"domain_{domain}"] = gen_eval.domain_dataset(args.domain_eval_examples, args.eval_length, args.eval_length, "mixed", domain)
    return datasets


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device) -> Tuple[Any, nn.Module, int]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    dtype = dtype_from_string(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    common: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map if torch.cuda.is_available() else None,
    }
    if quantization_config is not None:
        common["quantization_config"] = quantization_config
    common = {k: v for k, v in common.items() if v is not None}
    print(f"[load] {args.model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **common)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(args.gradient_checkpointing))
    if args.train_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    probe = tokenizer("probe\nAnswer:\n", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"], output_hidden_states=True, use_cache=False)
    hidden_dim = int(out.hidden_states[-1].shape[-1])
    return tokenizer, model, hidden_dim


@torch.no_grad()
def evaluate(
    split: str,
    dataset: ExampleSet,
    tokenizer: Any,
    model: nn.Module,
    compiler: HiddenVMCompiler,
    executor: HiddenVMExecutor,
    answer_ids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    compiler.eval()
    pad_id = int(tokenizer.pad_token_id)
    total = 0
    direct_correct = 0
    exec_correct = 0
    init_correct = 0
    op_correct = 0
    arg_correct = 0
    op_total = 0
    program_exact = 0
    state_correct = 0
    state_total = 0
    prefix_correct_total = 0
    prefix_possible_total = 0
    repair_correct = 0
    repair_program_exact = 0
    repair_state_correct = 0
    repair_state_total = 0
    repair_prefix_correct_total = 0
    repair_prefix_possible_total = 0
    repair_found_total = 0
    repair_changed_total = 0
    repair_candidate_total = 0
    repair_verified_total = 0
    answer_values: List[torch.Tensor] = []
    exec_values: List[torch.Tensor] = []
    repair_exec_values: List[torch.Tensor] = []
    init_values: List[torch.Tensor] = []
    op_values: List[torch.Tensor] = []
    arg_values: List[torch.Tensor] = []
    state_values: List[torch.Tensor] = []
    repair_state_values: List[torch.Tensor] = []
    length_values: List[torch.Tensor] = []
    domain_counts = {domain: 0 for domain in DOMAIN_NAMES}
    domain_exec = {domain: 0 for domain in DOMAIN_NAMES}
    domain_direct = {domain: 0 for domain in DOMAIN_NAMES}
    domain_repair = {domain: 0 for domain in DOMAIN_NAMES}

    for start in range(0, len(dataset), args.eval_batch_size):
        examples = dataset.examples[start : start + args.eval_batch_size]
        batch = collate_examples(examples, pad_id, args.max_steps, args.max_length, device)
        hidden, lm_logits = forward_hidden_and_logits(model, batch)
        init_logits, op_logits, arg_logits = compiler(hidden, batch["hidden_mask"], batch["num_values"], batch["op_values"])
        direct_pred = direct_answer_logits(lm_logits, batch, answer_ids).argmax(dim=-1)
        init_pred = init_logits.argmax(dim=-1)
        op_pred = op_logits.argmax(dim=-1)
        arg_pred = arg_logits.argmax(dim=-1)
        exec_pred, states_pred = argmax_execute_with_states(init_pred, op_pred, arg_pred, batch["lengths"], args.max_steps, args.value_modulus)
        repair_preds: List[int] = []
        repair_states_rows: List[List[int]] = []
        repair_exact_flags: List[bool] = []
        if args.repair_eval_topk > 0:
            for row_idx in range(len(examples)):
                length = int(batch["lengths"][row_idx].item())
                repair = repair_single_program(
                    init_logits[row_idx],
                    op_logits[row_idx],
                    arg_logits[row_idx],
                    length,
                    int(batch["answer"][row_idx].item()),
                    [int(x) for x in batch["states"][row_idx].tolist()],
                    int(batch["init_value"][row_idx].item()),
                    [int(x) for x in batch["ops"][row_idx].tolist()],
                    [int(x) for x in batch["args"][row_idx].tolist()],
                    args.max_steps,
                    args.value_modulus,
                    args.repair_eval_topk,
                    args.repair_max_edits,
                    args.repair_verifier_mode,
                )
                repair_preds.append(repair.answer)
                repair_states_rows.append(repair.states)
                repair_found_total += int(repair.found)
                repair_changed_total += int(repair.changed)
                repair_candidate_total += int(repair.num_candidates)
                repair_verified_total += int(repair.num_verified)
                repair_exact_flags.append(bool(repair.program_exact))
            repair_pred = torch.tensor(repair_preds, dtype=torch.long, device=device)
            repair_states = torch.tensor(repair_states_rows, dtype=torch.long, device=device)
        else:
            repair_pred = exec_pred
            repair_states = states_pred
            repair_exact_flags = [False] * len(examples)

        active = batch["ops"].ne(-100)
        state_active = batch["states"].ne(-100)
        total += len(examples)
        direct_correct += int(direct_pred.eq(batch["answer"]).sum().item())
        exec_correct += int(exec_pred.eq(batch["answer"]).sum().item())
        if args.repair_eval_topk > 0:
            repair_correct += int(repair_pred.eq(batch["answer"]).sum().item())
            repair_program_exact += int(sum(repair_exact_flags))
            repair_state_correct += int(repair_states[state_active].eq(batch["states"][state_active]).sum().item())
            repair_state_total += int(state_active.sum().item())
        init_correct += int(init_pred.eq(batch["init_value"]).sum().item())
        op_correct += int(op_pred[active].eq(batch["ops"][active]).sum().item())
        arg_correct += int(arg_pred[active].eq(batch["args"][active]).sum().item())
        op_total += int(active.sum().item())
        exact = init_pred.eq(batch["init_value"]) & ((op_pred.eq(batch["ops"]) | ~active).all(dim=1)) & ((arg_pred.eq(batch["args"]) | ~active).all(dim=1))
        program_exact += int(exact.sum().item())
        state_correct += int(states_pred[state_active].eq(batch["states"][state_active]).sum().item())
        state_total += int(state_active.sum().item())
        state_ok = states_pred.eq(batch["states"]) | ~state_active
        for row_idx in range(states_pred.shape[0]):
            length = int(batch["lengths"][row_idx].item())
            prefix = 0
            for t in range(length):
                if bool(state_ok[row_idx, t].item()):
                    prefix += 1
                else:
                    break
            prefix_correct_total += prefix
            prefix_possible_total += length
            if args.repair_eval_topk > 0:
                repair_prefix = 0
                repair_state_ok = repair_states[row_idx].eq(batch["states"][row_idx]) | ~state_active[row_idx]
                for t in range(length):
                    if bool(repair_state_ok[t].item()):
                        repair_prefix += 1
                    else:
                        break
                repair_prefix_correct_total += repair_prefix
                repair_prefix_possible_total += length
        for row_idx, domain in enumerate(batch["domains"]):
            domain_counts[domain] += 1
            domain_exec[domain] += int(bool(exec_pred[row_idx].eq(batch["answer"][row_idx]).item()))
            domain_direct[domain] += int(bool(direct_pred[row_idx].eq(batch["answer"][row_idx]).item()))
            if args.repair_eval_topk > 0:
                domain_repair[domain] += int(bool(repair_pred[row_idx].eq(batch["answer"][row_idx]).item()))

        answer_values.append(batch["answer"].detach().cpu())
        exec_values.append(exec_pred.detach().cpu())
        if args.repair_eval_topk > 0:
            repair_exec_values.append(repair_pred.detach().cpu())
        init_values.append(init_pred.detach().cpu())
        op_values.append(op_pred.detach().cpu())
        arg_values.append(arg_pred.detach().cpu())
        state_values.append(states_pred.detach().cpu())
        if args.repair_eval_topk > 0:
            repair_state_values.append(repair_states.detach().cpu())
        length_values.append(batch["lengths"].detach().cpu())

    metrics: Dict[str, float] = {
        "n": float(total),
        "direct_accuracy": direct_correct / total if total else math.nan,
        "executor_accuracy": exec_correct / total if total else math.nan,
        "init_accuracy": init_correct / total if total else math.nan,
        "op_accuracy": op_correct / op_total if op_total else math.nan,
        "arg_accuracy": arg_correct / op_total if op_total else math.nan,
        "program_exact": program_exact / total if total else math.nan,
        "state_accuracy": state_correct / state_total if state_total else math.nan,
        "state_prefix_fraction": prefix_correct_total / prefix_possible_total if prefix_possible_total else math.nan,
    }
    if args.repair_eval_topk > 0:
        metrics.update(
            {
                "repair_executor_accuracy": repair_correct / total if total else math.nan,
                "repair_program_exact": repair_program_exact / total if total else math.nan,
                "repair_state_accuracy": repair_state_correct / repair_state_total if repair_state_total else math.nan,
                "repair_state_prefix_fraction": repair_prefix_correct_total / repair_prefix_possible_total if repair_prefix_possible_total else math.nan,
                "repair_found_fraction": repair_found_total / total if total else math.nan,
                "repair_changed_fraction": repair_changed_total / total if total else math.nan,
                "repair_avg_candidates": repair_candidate_total / total if total else math.nan,
                "repair_avg_verified": repair_verified_total / total if total else math.nan,
            }
        )
    for domain in DOMAIN_NAMES:
        n = domain_counts[domain]
        if n:
            metrics[f"domain_{domain}_n"] = float(n)
            metrics[f"domain_{domain}_executor_accuracy"] = domain_exec[domain] / n
            metrics[f"domain_{domain}_direct_accuracy"] = domain_direct[domain] / n
            if args.repair_eval_topk > 0:
                metrics[f"domain_{domain}_repair_executor_accuracy"] = domain_repair[domain] / n

    if dataset.paired and dataset.pair_size == 2 and total >= 2:
        usable = (total // 2) * 2
        pair_count = usable // 2
        answers = torch.cat(answer_values)[:usable].view(pair_count, 2)
        exec_preds = torch.cat(exec_values)[:usable].view(pair_count, 2)
        repair_preds_pair = torch.cat(repair_exec_values)[:usable].view(pair_count, 2) if args.repair_eval_topk > 0 else None
        init_preds = torch.cat(init_values)[:usable].view(pair_count, 2)
        op_preds = torch.cat(op_values)[:usable].view(pair_count, 2, -1)
        arg_preds = torch.cat(arg_values)[:usable].view(pair_count, 2, -1)
        state_preds = torch.cat(state_values)[:usable].view(pair_count, 2, -1)
        repair_state_preds = torch.cat(repair_state_values)[:usable].view(pair_count, 2, -1) if args.repair_eval_topk > 0 else None
        lengths = torch.cat(length_values)[:usable].view(pair_count, 2)
        program_same = []
        state_same = []
        repair_state_same = []
        for pair_idx in range(pair_count):
            length = int(lengths[pair_idx, 0].item())
            same = bool(init_preds[pair_idx, 0].eq(init_preds[pair_idx, 1]).item())
            same = same and bool(op_preds[pair_idx, 0, :length].eq(op_preds[pair_idx, 1, :length]).all().item())
            same = same and bool(arg_preds[pair_idx, 0, :length].eq(arg_preds[pair_idx, 1, :length]).all().item())
            program_same.append(same)
            state_same.append(bool(state_preds[pair_idx, 0, :length].eq(state_preds[pair_idx, 1, :length]).all().item()))
            if repair_state_preds is not None:
                repair_state_same.append(bool(repair_state_preds[pair_idx, 0, :length].eq(repair_state_preds[pair_idx, 1, :length]).all().item()))
        metrics["executor_pair_answer_consistency"] = float(exec_preds[:, 0].eq(exec_preds[:, 1]).float().mean().item())
        metrics["executor_pair_both_correct"] = float(exec_preds.eq(answers).all(dim=1).float().mean().item())
        metrics["compiler_pair_program_consistency"] = sum(program_same) / len(program_same)
        metrics["compiler_pair_state_consistency"] = sum(state_same) / len(state_same)
        if repair_preds_pair is not None:
            metrics["repair_pair_answer_consistency"] = float(repair_preds_pair[:, 0].eq(repair_preds_pair[:, 1]).float().mean().item())
            metrics["repair_pair_both_correct"] = float(repair_preds_pair.eq(answers).all(dim=1).float().mean().item())
            metrics["repair_pair_state_consistency"] = sum(repair_state_same) / len(repair_state_same)
    suffix = ""
    if args.repair_eval_topk > 0:
        suffix = f" repair={100.0 * metrics['repair_executor_accuracy']:.1f}%"
    print(f"[eval] {split} exec={100.0 * metrics['executor_accuracy']:.1f}% direct={100.0 * metrics['direct_accuracy']:.1f}%{suffix}", flush=True)
    return metrics


def flatten_metrics(metrics_by_split: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split, metrics in metrics_by_split.items():
        for key, value in metrics.items():
            if key != "n":
                out[f"{split}_{key}"] = value
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def parse_curriculum_schedule(schedule: str, default_max_len: int, train_steps: int) -> List[Tuple[int, int]]:
    if not schedule.strip():
        return [(int(train_steps), int(default_max_len))]
    stages: List[Tuple[int, int]] = []
    for item in schedule.split(","):
        if not item.strip():
            continue
        max_len_s, until_s = item.split(":", 1)
        stages.append((int(until_s), int(max_len_s)))
    stages.sort(key=lambda x: x[0])
    if not stages or stages[-1][0] < train_steps:
        stages.append((int(train_steps), int(default_max_len)))
    return stages


def curriculum_max_len(step: int, stages: Sequence[Tuple[int, int]]) -> int:
    for until_step, max_len in stages:
        if step <= until_step:
            return int(max_len)
    return int(stages[-1][1])


def loss_for_batch(
    model: nn.Module,
    compiler: HiddenVMCompiler,
    executor: HiddenVMExecutor,
    answer_ids: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    selection_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    hidden, lm_logits = forward_hidden_and_logits(model, batch)
    init_logits, op_logits, arg_logits, scores = compiler(hidden, batch["hidden_mask"], batch["num_values"], batch["op_values"], return_scores=True)
    loss = init_logits.sum() * 0.0
    parts: Dict[str, float] = {}

    if args.variant == "trace":
        tr_loss, tr_parts = trace_losses(
            init_logits,
            op_logits,
            arg_logits,
            batch,
            args.init_trace_loss_weight,
            args.op_trace_loss_weight,
            args.arg_trace_loss_weight,
        )
        loss = loss + args.trace_loss_weight * tr_loss
        parts.update(tr_parts)
        if selection_weight > 0:
            sel_loss, sel_parts = selection_losses(scores, batch, args)
            loss = loss + selection_weight * sel_loss
            parts.update(sel_parts)
    elif args.variant != "answer_only":
        raise ValueError(args.variant)

    if args.executor_loss_weight > 0:
        final_probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
        exec_loss = F.nll_loss(final_probs.log(), batch["answer"])
        loss = loss + args.executor_loss_weight * exec_loss
        parts["executor_loss"] = float(exec_loss.detach().cpu())
    if args.state_loss_weight > 0:
        state_probs = executor.soft_trajectory(init_logits, op_logits, arg_logits, batch["lengths"])
        st_loss, st_parts = state_trajectory_loss(state_probs, batch)
        loss = loss + args.state_loss_weight * st_loss
        parts.update(st_parts)
    if args.direct_answer_loss_weight > 0:
        direct_logits = direct_answer_logits(lm_logits, batch, answer_ids)
        direct_loss = F.cross_entropy(direct_logits, batch["answer"])
        loss = loss + args.direct_answer_loss_weight * direct_loss
        parts["direct_answer_loss"] = float(direct_loss.detach().cpu())
    return loss, parts


@torch.no_grad()
def make_onpolicy_targets(
    source: ExampleSet,
    tokenizer: Any,
    model: nn.Module,
    compiler: HiddenVMCompiler,
    args: argparse.Namespace,
    device: torch.device,
    split_name: str,
) -> Tuple[OnPolicyTargetSet, Dict[str, float]]:
    model.eval()
    compiler.eval()
    pad_id = int(tokenizer.pad_token_id)
    limit = len(source.examples) if args.onpolicy_source_examples <= 0 else min(args.onpolicy_source_examples, len(source.examples))
    examples = source.examples[:limit]

    init_rows: List[torch.Tensor] = []
    op_rows: List[torch.Tensor] = []
    arg_rows: List[torch.Tensor] = []
    active_rows: List[torch.Tensor] = []
    found_rows: List[torch.Tensor] = []
    changed_rows: List[torch.Tensor] = []
    program_exact_rows: List[torch.Tensor] = []
    candidate_rows: List[torch.Tensor] = []
    verified_rows: List[torch.Tensor] = []
    source_counts = {"repair": 0, "gold": 0, "gold_fallback": 0, "base_fallback": 0, "skipped": 0}

    for start in range(0, len(examples), args.eval_batch_size):
        chunk = examples[start : start + args.eval_batch_size]
        batch = collate_examples(chunk, pad_id, args.max_steps, args.max_length, device)
        hidden, _ = forward_hidden_and_logits(model, batch)
        init_logits, op_logits, arg_logits = compiler(hidden, batch["hidden_mask"], batch["num_values"], batch["op_values"])
        base_init = init_logits.argmax(dim=-1)
        base_ops = op_logits.argmax(dim=-1)
        base_args = arg_logits.argmax(dim=-1)

        target_init = batch["init_value"].detach().clone()
        target_ops = batch["ops"].detach().clone()
        target_args = batch["args"].detach().clone()
        use_mask = torch.ones((len(chunk),), dtype=torch.bool, device=device)
        found = torch.zeros((len(chunk),), dtype=torch.bool, device=device)
        changed = torch.zeros((len(chunk),), dtype=torch.bool, device=device)
        program_exact = torch.zeros((len(chunk),), dtype=torch.bool, device=device)
        candidate_count = torch.zeros((len(chunk),), dtype=torch.float32, device=device)
        verified_count = torch.zeros((len(chunk),), dtype=torch.float32, device=device)

        for row_idx, ex in enumerate(chunk):
            if args.target_mode == "gold_only":
                source_counts["gold"] += 1
                found[row_idx] = True
                changed[row_idx] = False
                program_exact[row_idx] = True
                candidate_count[row_idx] = 0
                verified_count[row_idx] = 0
                continue

            repair = repair_single_program(
                init_logits[row_idx],
                op_logits[row_idx],
                arg_logits[row_idx],
                int(ex.length),
                int(ex.answer),
                list(ex.states),
                int(ex.init_value),
                list(ex.ops),
                list(ex.args),
                args.max_steps,
                args.value_modulus,
                args.repair_train_topk,
                args.repair_max_edits,
                args.repair_verifier_mode,
            )
            found[row_idx] = bool(repair.found)
            changed[row_idx] = bool(repair.changed)
            program_exact[row_idx] = bool(repair.program_exact)
            candidate_count[row_idx] = float(repair.num_candidates)
            verified_count[row_idx] = float(repair.num_verified)

            if repair.found:
                target_init[row_idx] = int(repair.init_value)
                target_ops[row_idx, : ex.length] = torch.tensor(repair.ops[: ex.length], dtype=torch.long, device=device)
                target_args[row_idx, : ex.length] = torch.tensor(repair.args[: ex.length], dtype=torch.long, device=device)
                source_counts["repair"] += 1
            elif args.target_mode == "repair_or_gold":
                source_counts["gold_fallback"] += 1
            elif args.target_mode == "repair_or_base":
                target_init[row_idx] = base_init[row_idx]
                target_ops[row_idx, : ex.length] = base_ops[row_idx, : ex.length]
                target_args[row_idx, : ex.length] = base_args[row_idx, : ex.length]
                source_counts["base_fallback"] += 1
            elif args.target_mode == "repair_only":
                use_mask[row_idx] = False
                source_counts["skipped"] += 1
            else:
                raise ValueError(args.target_mode)

        init_rows.append(target_init.detach().cpu())
        op_rows.append(target_ops.detach().cpu())
        arg_rows.append(target_args.detach().cpu())
        active_rows.append(use_mask.detach().cpu())
        found_rows.append(found.detach().cpu())
        changed_rows.append(changed.detach().cpu())
        program_exact_rows.append(program_exact.detach().cpu())
        candidate_rows.append(candidate_count.detach().cpu())
        verified_rows.append(verified_count.detach().cpu())
        print(f"[targets] {split_name} {min(start + len(chunk), len(examples))}/{len(examples)}", flush=True)

    targets = OnPolicyTargetSet(
        examples=examples,
        init=torch.cat(init_rows, dim=0) if init_rows else torch.empty(0, dtype=torch.long),
        ops=torch.cat(op_rows, dim=0) if op_rows else torch.empty(0, args.max_steps, dtype=torch.long),
        args=torch.cat(arg_rows, dim=0) if arg_rows else torch.empty(0, args.max_steps, dtype=torch.long),
        active_mask=torch.cat(active_rows, dim=0) if active_rows else torch.empty(0, dtype=torch.bool),
        found=torch.cat(found_rows, dim=0) if found_rows else torch.empty(0, dtype=torch.bool),
        changed=torch.cat(changed_rows, dim=0) if changed_rows else torch.empty(0, dtype=torch.bool),
        program_exact=torch.cat(program_exact_rows, dim=0) if program_exact_rows else torch.empty(0, dtype=torch.bool),
        candidate_count=torch.cat(candidate_rows, dim=0) if candidate_rows else torch.empty(0, dtype=torch.float32),
        verified_count=torch.cat(verified_rows, dim=0) if verified_rows else torch.empty(0, dtype=torch.float32),
        source_counts=source_counts,
    )
    stats = {
        "target_source_n": float(len(examples)),
        "target_active_fraction": float(targets.active_mask.float().mean().item()) if len(examples) else math.nan,
        "target_repair_found_fraction": float(targets.found.float().mean().item()) if len(examples) else math.nan,
        "target_repair_changed_fraction": float(targets.changed.float().mean().item()) if len(examples) else math.nan,
        "target_program_exact_fraction": float(targets.program_exact.float().mean().item()) if len(examples) else math.nan,
        "target_avg_candidates": float(targets.candidate_count.float().mean().item()) if len(examples) else math.nan,
        "target_avg_verified": float(targets.verified_count.float().mean().item()) if len(examples) else math.nan,
        **{f"target_source_{key}": float(value) for key, value in source_counts.items()},
    }
    return targets, stats


def onpolicy_slot_loss(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    target_init: torch.Tensor,
    target_ops: torch.Tensor,
    target_args: torch.Tensor,
    target_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_mask = target_mask.bool()
    active = batch["ops"].ne(-100) & target_mask.view(-1, 1)
    if not bool(target_mask.any().item()):
        zero = init_logits.sum() * 0.0 + op_logits.sum() * 0.0 + arg_logits.sum() * 0.0
        return zero, {"repair_init_loss": 0.0, "repair_op_loss": 0.0, "repair_arg_loss": 0.0}
    init_loss = F.cross_entropy(init_logits[target_mask], target_init[target_mask])
    op_loss = F.cross_entropy(op_logits[active], target_ops[active]) if active.any() else op_logits.sum() * 0.0
    arg_loss = F.cross_entropy(arg_logits[active], target_args[active]) if active.any() else arg_logits.sum() * 0.0
    loss = args.repair_trace_loss_weight * (
        args.init_repair_loss_weight * init_loss
        + args.op_repair_loss_weight * op_loss
        + args.arg_repair_loss_weight * arg_loss
    )
    return loss, {
        "repair_init_loss": float(init_loss.detach().cpu()),
        "repair_op_loss": float(op_loss.detach().cpu()),
        "repair_arg_loss": float(arg_loss.detach().cpu()),
    }


def onpolicy_loss_for_batch(
    model: nn.Module,
    compiler: HiddenVMCompiler,
    executor: HiddenVMExecutor,
    answer_ids: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    target_init: torch.Tensor,
    target_ops: torch.Tensor,
    target_args: torch.Tensor,
    target_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    hidden, lm_logits = forward_hidden_and_logits(model, batch)
    init_logits, op_logits, arg_logits, scores = compiler(hidden, batch["hidden_mask"], batch["num_values"], batch["op_values"], return_scores=True)
    loss, parts = onpolicy_slot_loss(init_logits, op_logits, arg_logits, batch, target_init, target_ops, target_args, target_mask, args)

    if args.gold_trace_loss_weight > 0:
        gold_loss, gold_parts = trace_losses(
            init_logits,
            op_logits,
            arg_logits,
            batch,
            args.init_gold_loss_weight,
            args.op_gold_loss_weight,
            args.arg_gold_loss_weight,
        )
        loss = loss + args.gold_trace_loss_weight * gold_loss
        parts["gold_trace_loss"] = float(gold_loss.detach().cpu())
        for key, value in gold_parts.items():
            parts[f"gold_{key}"] = value

    if args.gold_selection_loss_weight > 0:
        sel_loss, sel_parts = selection_losses(scores, batch, args)
        loss = loss + args.gold_selection_loss_weight * sel_loss
        parts["gold_selection_loss"] = float(sel_loss.detach().cpu())
        for key, value in sel_parts.items():
            parts[f"gold_{key}"] = value

    if args.executor_loss_weight > 0:
        final_probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
        exec_loss = F.nll_loss(final_probs.log(), batch["answer"])
        loss = loss + args.executor_loss_weight * exec_loss
        parts["executor_loss"] = float(exec_loss.detach().cpu())
    if args.state_loss_weight > 0:
        state_probs = executor.soft_trajectory(init_logits, op_logits, arg_logits, batch["lengths"])
        st_loss, st_parts = state_trajectory_loss(state_probs, batch)
        loss = loss + args.state_loss_weight * st_loss
        parts.update(st_parts)
    if args.direct_answer_loss_weight > 0:
        direct_logits = direct_answer_logits(lm_logits, batch, answer_ids)
        direct_loss = F.cross_entropy(direct_logits, batch["answer"])
        loss = loss + args.direct_answer_loss_weight * direct_loss
        parts["direct_answer_loss"] = float(direct_loss.detach().cpu())
    return loss, parts


def save_checkpoint(run_dir: Path, model: nn.Module, compiler: HiddenVMCompiler, args: argparse.Namespace, hidden_dim: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_root = CHECKPOINT_ROOT / run_dir.name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_root / "adapter"
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_dir)
    heads_path = ckpt_root / "heads.pt"
    torch.save(
        {
            "variant": args.variant,
            "args": vars(args),
            "vm_ops": VM_OPS,
            "domains": DOMAIN_NAMES,
            "hidden_dim": hidden_dim,
            "compiler": compiler.state_dict(),
            "metrics": metrics,
        },
        heads_path,
    )
    return {"run": run_dir.name, "adapter_path": str(adapter_dir), "heads_path": str(heads_path)}


def train(args: argparse.Namespace) -> None:
    if args.max_steps < max(args.train_max_len, args.eval_length, args.hard_length, args.harder_length):
        raise ValueError(
            f"max_steps={args.max_steps} must cover train_max_len={args.train_max_len}, "
            f"eval_length={args.eval_length}, hard_length={args.hard_length}, and harder_length={args.harder_length}"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    tokenizer, model, hidden_dim = load_model_and_tokenizer(args, device)
    compiler = HiddenVMCompiler(
        hidden_dim,
        args.value_modulus,
        args.head_width,
        args.max_steps,
        args.rank_temperature,
        args.arg_window,
        args.arg_distance_temperature,
    ).to(device)
    executor = HiddenVMExecutor(args.value_modulus, device)
    answer_ids = answer_token_ids(tokenizer, args.value_modulus, device)
    datasets = make_datasets(tokenizer, args)
    train_set = datasets["train_mixed"]
    eval_sets = {key: value for key, value in datasets.items() if key != "train_mixed"}
    pad_id = int(tokenizer.pad_token_id)
    curriculum = parse_curriculum_schedule(args.curriculum_schedule, args.train_max_len, args.train_steps)
    train_indices_by_len: Dict[int, List[int]] = {}
    for _, max_len in curriculum:
        train_indices_by_len[max_len] = [i for i, ex in enumerate(train_set.examples) if ex.length <= max_len]
        if not train_indices_by_len[max_len]:
            raise RuntimeError(f"no training examples available for curriculum max_len={max_len}")

    trainable = [p for p in model.parameters() if p.requires_grad] + [p for p in compiler.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 9000)
    train_rows: List[Dict[str, Any]] = []

    metrics_by_split = {
        split: evaluate(split, dataset, tokenizer, model, compiler, executor, answer_ids, args, device)
        for split, dataset in eval_sets.items()
    }
    train_rows.append({"step": 0, "phase": "baseline", **flatten_metrics(metrics_by_split)})

    for step in range(1, args.train_steps + 1):
        stage_max_len = curriculum_max_len(step, curriculum)
        pool = train_indices_by_len[stage_max_len]
        idxs = [pool[rng.randrange(len(pool))] for _ in range(args.train_batch_size)]
        batch = collate_examples([train_set.examples[i] for i in idxs], pad_id, args.max_steps, args.max_length, device)
        model.train()
        compiler.train()
        loss, parts = loss_for_batch(model, compiler, executor, answer_ids, batch, args, args.selection_loss_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == args.train_steps:
            row = {"step": step, "phase": "curriculum_train", "stage_max_len": stage_max_len, "loss": float(loss.detach().cpu())}
            row.update(parts)
            train_rows.append(row)
            print(f"[train] step={step} loss={row['loss']:.4f}", flush=True)
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.train_steps):
            metrics_by_split = {
                split: evaluate(split, dataset, tokenizer, model, compiler, executor, answer_ids, args, device)
                for split, dataset in eval_sets.items()
            }
            train_rows.append({"step": step, "phase": "eval", **flatten_metrics(metrics_by_split)})

    onpolicy_step = args.train_steps
    if args.onpolicy_rounds > 0:
        if not args.onpolicy_train_lora:
            for p in model.parameters():
                p.requires_grad_(False)
        onpolicy_trainable = [p for p in model.parameters() if p.requires_grad] + [p for p in compiler.parameters() if p.requires_grad]
        onpolicy_lr = args.lr * args.onpolicy_lr_multiplier
        if args.onpolicy_reset_optimizer:
            trainable = onpolicy_trainable
            opt = torch.optim.AdamW(trainable, lr=onpolicy_lr, weight_decay=args.weight_decay)
        else:
            trainable = onpolicy_trainable
            for group in opt.param_groups:
                group["lr"] = onpolicy_lr
        print(
            f"[onpolicy] lr={onpolicy_lr:.6g} train_lora={bool(args.onpolicy_train_lora)} "
            f"reset_optimizer={bool(args.onpolicy_reset_optimizer)} trainable_tensors={len(trainable)}",
            flush=True,
        )
    for round_index in range(1, args.onpolicy_rounds + 1):
        targets, target_stats = make_onpolicy_targets(
            train_set,
            tokenizer,
            model,
            compiler,
            args,
            device,
            f"round{round_index}",
        )
        train_rows.append({"step": onpolicy_step, "round": round_index, "epoch": 0, "phase": "onpolicy_targets", **target_stats})
        print(
            "[onpolicy-targets] "
            f"round={round_index} active={100.0 * target_stats['target_active_fraction']:.1f}% "
            f"found={100.0 * target_stats['target_repair_found_fraction']:.1f}% "
            f"changed={100.0 * target_stats['target_repair_changed_fraction']:.1f}% "
            f"program_exact={100.0 * target_stats['target_program_exact_fraction']:.1f}%",
            flush=True,
        )
        if len(targets.examples) == 0 or not bool(targets.active_mask.any().item()):
            print(f"[onpolicy-targets] round={round_index} has no active rows; skipping", flush=True)
            continue

        order = list(range(len(targets.examples)))
        for epoch in range(1, args.epochs_per_round + 1):
            rng.shuffle(order)
            totals: Dict[str, float] = {}
            trained_batches = 0
            model.train()
            compiler.train()
            for start in range(0, len(order), args.train_batch_size):
                if args.onpolicy_max_batches > 0 and trained_batches >= args.onpolicy_max_batches:
                    break
                idxs = order[start : start + args.train_batch_size]
                examples = [targets.examples[i] for i in idxs]
                batch = collate_examples(examples, pad_id, args.max_steps, args.max_length, device)
                target_init = targets.init[idxs].to(device)
                target_ops = targets.ops[idxs].to(device)
                target_args = targets.args[idxs].to(device)
                target_mask = targets.active_mask[idxs].to(device)
                if not bool(target_mask.any().item()):
                    continue
                loss, parts = onpolicy_loss_for_batch(
                    model,
                    compiler,
                    executor,
                    answer_ids,
                    batch,
                    target_init,
                    target_ops,
                    target_args,
                    target_mask,
                    args,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()
                totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu())
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                trained_batches += 1

            onpolicy_step += 1
            metrics_by_split = {
                split: evaluate(split, dataset, tokenizer, model, compiler, executor, answer_ids, args, device)
                for split, dataset in eval_sets.items()
            }
            row = {
                "step": onpolicy_step,
                "round": round_index,
                "epoch": epoch,
                "phase": "onpolicy_train",
                "trained_batches": trained_batches,
                **{key: value / max(1, trained_batches) for key, value in totals.items()},
                **target_stats,
                **flatten_metrics(metrics_by_split),
            }
            train_rows.append(row)
            paired_key = "fresh_paired_mixed"
            paired_acc = metrics_by_split.get(paired_key, metrics_by_split["val_mixed"])["executor_accuracy"]
            hard_key = "hard_standard_mixed" if "hard_standard_mixed" in metrics_by_split else "val_mixed"
            hard_acc = metrics_by_split[hard_key]["executor_accuracy"]
            print(
                f"[onpolicy-train] round={round_index} epoch={epoch} loss={row.get('loss', 0.0):.4f} "
                f"paired={100.0 * paired_acc:.1f}% hard={100.0 * hard_acc:.1f}%",
                flush=True,
            )

    metrics_by_split = {
        split: evaluate(split, dataset, tokenizer, model, compiler, executor, answer_ids, args, device)
        for split, dataset in eval_sets.items()
    }
    metric_rows = []
    for split, metrics in metrics_by_split.items():
        row = {"run": args.run_name, "variant": args.variant, "phase": "final", "split": split}
        row.update(metrics)
        metric_rows.append(row)
    checkpoint_row = save_checkpoint(run_dir, model, compiler, args, hidden_dim, flatten_metrics(metrics_by_split))

    write_csv(run_dir / "train_log.csv", train_rows)
    write_csv(run_dir / "metrics.csv", metric_rows)
    write_csv(run_dir / "checkpoints.csv", [checkpoint_row])
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "hidden_dim": hidden_dim,
        "args": vars(args),
        "train_seconds": time.time() - t0,
    }
    json_dump(run_dir / "results.json", {"metadata": metadata, "metrics": metrics_by_split, "train_log": train_rows, "checkpoint": checkpoint_row})
    write_csv(ANALYSIS / "final_metrics.csv", metric_rows)
    write_csv(ROOT / "checkpoint_manifest.csv", [checkpoint_row])
    print(f"[done] {run_dir} seconds={metadata['train_seconds']:.1f}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Qwen hidden VM compiler with canonical on-policy repair")
    p.add_argument("--run_name", type=str, default="main_onpolicy_canonical_repair_s512")
    p.add_argument("--variant", type=str, default="trace", choices=["trace", "answer_only"])
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch_dtype", type=str, default="bf16")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--train_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value_modulus", type=int, default=97)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=6)
    p.add_argument("--eval_length", type=int, default=6)
    p.add_argument("--hard_length", type=int, default=8)
    p.add_argument("--harder_length", type=int, default=10)
    p.add_argument("--curriculum_schedule", type=str, default="4:240,6:700")
    p.add_argument("--domains", type=str, default="arithmetic,calendar,unit,list,boolean,lookup")
    p.add_argument("--train_template_mode", type=str, default="mixed", choices=["standard", "paraphrase", "mixed"])
    p.add_argument("--train_examples", type=int, default=512)
    p.add_argument("--val_examples", type=int, default=128)
    p.add_argument("--eval_examples", type=int, default=192)
    p.add_argument("--eval_pairs", type=int, default=128)
    p.add_argument("--domain_eval_examples", type=int, default=32)
    p.add_argument("--train_steps", type=int, default=700)
    p.add_argument("--onpolicy_rounds", type=int, default=1)
    p.add_argument("--epochs_per_round", type=int, default=1)
    p.add_argument("--onpolicy_lr_multiplier", type=float, default=1.0)
    p.add_argument("--onpolicy_train_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--onpolicy_reset_optimizer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--onpolicy_max_batches", type=int, default=0)
    p.add_argument("--target_mode", type=str, default="repair_or_gold", choices=["repair_or_gold", "repair_or_base", "repair_only", "gold_only"])
    p.add_argument("--onpolicy_source_examples", type=int, default=0)
    p.add_argument("--repair_train_topk", type=int, default=3)
    p.add_argument("--repair_eval_topk", type=int, default=3)
    p.add_argument("--repair_max_edits", type=int, default=2)
    p.add_argument("--repair_verifier_mode", type=str, default="state", choices=["state", "answer"])
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--head_width", type=int, default=512)
    p.add_argument("--rank_temperature", type=float, default=1.0)
    p.add_argument("--arg_window", type=int, default=16)
    p.add_argument("--arg_distance_temperature", type=float, default=4.0)
    p.add_argument("--trace_loss_weight", type=float, default=1.0)
    p.add_argument("--init_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--op_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--repair_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--init_repair_loss_weight", type=float, default=4.0)
    p.add_argument("--op_repair_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_repair_loss_weight", type=float, default=4.0)
    p.add_argument("--gold_trace_loss_weight", type=float, default=0.25)
    p.add_argument("--init_gold_loss_weight", type=float, default=4.0)
    p.add_argument("--op_gold_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_gold_loss_weight", type=float, default=4.0)
    p.add_argument("--gold_selection_loss_weight", type=float, default=0.25)
    p.add_argument("--selection_loss_weight", type=float, default=1.0)
    p.add_argument("--role_pos_weight", type=float, default=20.0)
    p.add_argument("--init_selection_loss_weight", type=float, default=1.0)
    p.add_argument("--op_selection_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_selection_loss_weight", type=float, default=4.0)
    p.add_argument("--executor_loss_weight", type=float, default=0.2)
    p.add_argument("--state_loss_weight", type=float, default=0.05)
    p.add_argument("--direct_answer_loss_weight", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--eval_seed", type=int, default=91001)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())

#!/usr/bin/env python3
"""Qwen latent beam program compiler experiment.

The model receives a prompt plus several latent register banks. Each bank is a
candidate program. Training uses set-level losses: at least one beam should
compile the correct program and execution trace. Evaluation reports the selected
beam, the first beam, and an oracle over all beams.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception as exc:
    raise SystemExit(f"transformers is required: {exc}")

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:
    raise SystemExit(f"peft is required: {exc}")


ROOT = Path("experiments/qwen_latent_beam_program_compiler")
CHECKPOINT_ROOT = Path("large_artifacts/qwen_latent_beam_program_compiler/checkpoints")
OP_NAMES = ["ADD", "SUB", "MUL"]


@dataclass
class ProgramSpec:
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    answer: int


@dataclass
class ProgramExample:
    prompt: str
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    answer: int
    answer_pos: int
    register_pos: List[List[int]]
    input_ids: List[int]
    template_mode: str


@dataclass
class ExampleSet:
    examples: List[ProgramExample]
    paired: bool = False
    pair_size: int = 1

    def __len__(self) -> int:
        return len(self.examples)


@dataclass
class CurriculumStage:
    name: str
    min_len: int
    max_len: int
    steps: int


def parse_curriculum_stages(spec: str, fallback_min: int, fallback_max: int, fallback_steps: int) -> List[CurriculumStage]:
    if not spec.strip():
        return [CurriculumStage("train", fallback_min, fallback_max, fallback_steps)]
    stages: List[CurriculumStage] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 4:
            raise ValueError(f"curriculum stage must be name:min_len:max_len:steps, got {part!r}")
        name, min_len, max_len, steps = fields
        stages.append(CurriculumStage(name=name, min_len=int(min_len), max_len=int(max_len), steps=int(steps)))
    if not stages:
        raise ValueError("curriculum_stages produced no stages")
    return stages


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)


def tokenize_no_special(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    ids = list(ids)
    if not ids:
        raise RuntimeError(f"empty tokenization for {text!r}")
    return ids


def apply_op(x: int, op: int, arg: int, modulus: int) -> int:
    if op == 0:
        return (x + arg) % modulus
    if op == 1:
        return (x - arg) % modulus
    if op == 2:
        return (x * arg) % modulus
    raise ValueError(op)


class TextProgramGenerator:
    def __init__(
        self,
        tokenizer: Any,
        modulus: int,
        max_steps: int,
        beam_count: int,
        seed: int,
        template_mode: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.modulus = int(modulus)
        self.max_steps = int(max_steps)
        self.beam_count = int(beam_count)
        self.rng = random.Random(seed)
        self.template_mode = template_mode

    def _choice_for_mode(self, mode: str, standard: Sequence[Any], paraphrase: Sequence[Any]) -> Any:
        if mode == "standard":
            return standard[0]
        if mode == "paraphrase":
            return self.rng.choice(list(paraphrase))
        if mode == "mixed":
            return self.rng.choice(list(standard) + list(paraphrase))
        raise ValueError(mode)

    def make_spec(self, min_len: int, max_len: int) -> ProgramSpec:
        length = self.rng.randint(min_len, max_len)
        init_value = self.rng.randrange(self.modulus)
        x = init_value
        ops: List[int] = []
        args: List[int] = []
        states: List[int] = []
        for _ in range(length):
            op = self.rng.randrange(len(OP_NAMES))
            if op == 2:
                arg = self.rng.randint(2, min(self.modulus - 1, 12))
            else:
                arg = self.rng.randint(1, min(self.modulus - 1, 40))
            ops.append(op)
            args.append(arg)
            x = apply_op(x, op, arg, self.modulus)
            states.append(x)
        return ProgramSpec(length=length, init_value=init_value, ops=ops, args=args, states=states, answer=x)

    def render_spec(self, spec: ProgramSpec, template_mode: str) -> ProgramExample:
        input_ids: List[int] = []
        parts: List[str] = []

        def add_text(text: str) -> int:
            ids = tokenize_no_special(self.tokenizer, text)
            input_ids.extend(ids)
            parts.append(text)
            return len(input_ids) - 1

        intro = self._choice_for_mode(
            template_mode,
            [f"Compute a hidden value modulo {self.modulus}.\n"],
            [
                f"Work in arithmetic modulo {self.modulus}.\n",
                f"Track x using modulus {self.modulus}.\n",
                f"Every update below is modulo {self.modulus}.\n",
            ],
        )
        init_prefix = self._choice_for_mode(
            template_mode,
            ["Initial x = "],
            ["Start with x equal to ", "Let x be ", "The starting value of x is "],
        )
        init_suffix = self._choice_for_mode(template_mode, [".\n"], [".\n", " before any updates.\n", " at the start.\n"])
        final_line = self._choice_for_mode(
            template_mode,
            ["Return x after the final step.\n"],
            ["Report x after all updates.\n", "Give the final value of x.\n", "What is x after the listed updates?\n"],
        )
        answer_marker = self._choice_for_mode(template_mode, ["Answer:\n"], ["Final answer:\n", "Result:\n", "Value:\n"])

        add_text(intro)
        add_text(init_prefix)
        add_text(str(spec.init_value))
        add_text(init_suffix)
        for op, arg in zip(spec.ops, spec.args):
            if op == 0:
                prefix, op_word, between, number, suffix = self._choice_for_mode(
                    template_mode,
                    [("Step: ", "add", " ", str(arg), ".")],
                    [
                        ("Next, ", "increase", " x by ", str(arg), "."),
                        ("Now ", "add", " ", str(arg), " to x."),
                        ("Use an ", "add", " update of ", str(arg), "."),
                    ],
                )
            elif op == 1:
                prefix, op_word, between, number, suffix = self._choice_for_mode(
                    template_mode,
                    [("Step: ", "subtract", " ", str(arg), ".")],
                    [
                        ("Next, ", "decrease", " x by ", str(arg), "."),
                        ("Now ", "subtract", " ", str(arg), " from x."),
                        ("Use a ", "subtract", " update of ", str(arg), "."),
                    ],
                )
            else:
                prefix, op_word, between, number, suffix = self._choice_for_mode(
                    template_mode,
                    [("Step: ", "multiply", " by ", str(arg), ".")],
                    [
                        ("Next, ", "multiply", " x by ", str(arg), "."),
                        ("Now ", "scale", " x by ", str(arg), "."),
                        ("Use a ", "multiply", " update of ", str(arg), "."),
                    ],
                )
            add_text(prefix)
            add_text(op_word)
            add_text(between)
            add_text(number)
            add_text(suffix + "\n")
        add_text(final_line)
        answer_pos = add_text(answer_marker)

        register_pos: List[List[int]] = []
        add_text("\nLatent candidate program banks:\n")
        for beam in range(self.beam_count):
            cur: List[int] = []
            add_text(f"Bank {beam:02d}: ")
            cur.append(add_text(f"<B{beam:02d}I>"))
            add_text("\n")
            for step in range(self.max_steps):
                add_text(f"{beam:02d}:{step:02d} ")
                cur.append(add_text(f"<B{beam:02d}O{step:02d}>"))
                add_text(" ")
                cur.append(add_text(f"<B{beam:02d}A{step:02d}>"))
                add_text("\n")
            register_pos.append(cur)

        return ProgramExample(
            prompt="".join(parts).rstrip("\n"),
            length=spec.length,
            init_value=spec.init_value,
            ops=list(spec.ops),
            args=list(spec.args),
            states=list(spec.states),
            answer=spec.answer,
            answer_pos=answer_pos,
            register_pos=register_pos,
            input_ids=input_ids,
            template_mode=template_mode,
        )

    def make(self, min_len: int, max_len: int) -> ProgramExample:
        return self.render_spec(self.make_spec(min_len, max_len), self.template_mode)

    def dataset(self, n: int, min_len: int, max_len: int) -> ExampleSet:
        return ExampleSet([self.make(min_len, max_len) for _ in range(n)])

    def paired_dataset(self, n_pairs: int, min_len: int, max_len: int, template_modes: Sequence[str]) -> ExampleSet:
        examples: List[ProgramExample] = []
        for _ in range(n_pairs):
            spec = self.make_spec(min_len, max_len)
            for mode in template_modes:
                examples.append(self.render_spec(spec, mode))
        return ExampleSet(examples, paired=True, pair_size=len(template_modes))


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def collate_examples(
    examples: Sequence[ProgramExample],
    pad_id: int,
    bank_count: int,
    max_steps: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    seq_len = min(max(len(ex.input_ids) for ex in examples), max_length)
    ids = torch.full((len(examples), seq_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(examples), seq_len), dtype=torch.long, device=device)
    reg_count = 1 + 2 * max_steps
    lengths: List[int] = []
    init_values: List[int] = []
    answers: List[int] = []
    answer_pos: List[int] = []
    register_pos: List[List[List[int]]] = []
    ops: List[List[int]] = []
    args: List[List[int]] = []
    states: List[List[int]] = []
    for row, ex in enumerate(examples):
        if len(ex.register_pos) != bank_count:
            raise RuntimeError(f"example has {len(ex.register_pos)} prompt banks but expected {bank_count}")
        required = [ex.answer_pos] + [p for beam in ex.register_pos for p in beam]
        if max(required) >= seq_len:
            raise RuntimeError(
                f"max_length={max_length} truncated a required answer/register position; "
                f"needed {max(required) + 1} tokens"
            )
        for beam in ex.register_pos:
            if len(beam) != reg_count:
                raise RuntimeError(f"beam has {len(beam)} registers but expected {reg_count}")
        cur = ex.input_ids[:seq_len]
        ids[row, : len(cur)] = torch.tensor(cur, dtype=torch.long, device=device)
        mask[row, : len(cur)] = 1
        lengths.append(ex.length)
        init_values.append(ex.init_value)
        answers.append(ex.answer)
        answer_pos.append(ex.answer_pos)
        register_pos.append(ex.register_pos)
        ops.append(ex.ops + [-100] * (max_steps - len(ex.ops)))
        args.append(ex.args + [-100] * (max_steps - len(ex.args)))
        states.append(ex.states + [-100] * (max_steps - len(ex.states)))
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "init_value": torch.tensor(init_values, dtype=torch.long, device=device),
        "ops": torch.tensor(ops, dtype=torch.long, device=device),
        "args": torch.tensor(args, dtype=torch.long, device=device),
        "states": torch.tensor(states, dtype=torch.long, device=device),
        "answer": torch.tensor(answers, dtype=torch.long, device=device),
        "answer_pos": torch.tensor(answer_pos, dtype=torch.long, device=device),
        "register_pos": torch.tensor(register_pos, dtype=torch.long, device=device),
    }


def sample_batch(
    dataset: ExampleSet,
    batch_size: int,
    pad_id: int,
    bank_count: int,
    max_steps: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    idxs = torch.randint(0, len(dataset), (batch_size,)).tolist()
    return collate_examples([dataset.examples[i] for i in idxs], pad_id, bank_count, max_steps, max_length, device)


def sample_paired_batch(
    dataset: ExampleSet,
    batch_size: int,
    pair_size: int,
    pad_id: int,
    bank_count: int,
    max_steps: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if pair_size < 2:
        raise ValueError("pair_size must be at least 2")
    if batch_size % pair_size != 0:
        raise ValueError("batch size must be divisible by pair size")
    n_pairs = len(dataset) // pair_size
    pair_idxs = torch.randint(0, n_pairs, (batch_size // pair_size,)).tolist()
    examples: List[ProgramExample] = []
    for pair_idx in pair_idxs:
        start = pair_idx * pair_size
        examples.extend(dataset.examples[start : start + pair_size])
    return collate_examples(examples, pad_id, bank_count, max_steps, max_length, device)


class BeamProgramCompiler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        modulus: int,
        max_steps: int,
        beam_count: int,
        width: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.modulus = int(modulus)
        self.beam_count = int(beam_count)
        self.register_count = 1 + 2 * self.max_steps
        self.input = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))
        self.slot_pos = nn.Parameter(torch.randn(self.register_count, width) * 0.02)
        self.beam_pos = nn.Parameter(torch.randn(self.beam_count, width) * 0.02)
        if layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        else:
            self.encoder = nn.Identity()
        self.init_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, modulus))
        self.op_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, len(OP_NAMES)))
        self.arg_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, modulus))
        self.selector = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, register_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # register_h: [batch, beam, register, hidden]
        bsz, beams, regs, _ = register_h.shape
        if beams != self.beam_count or regs != self.register_count:
            raise RuntimeError(f"expected [batch,{self.beam_count},{self.register_count},hidden], got {tuple(register_h.shape)}")
        x = self.input(register_h.float())
        x = x + self.slot_pos[None, None, :, :] + self.beam_pos[None, :, None, :]
        x = x.reshape(bsz * beams, regs, x.shape[-1])
        x = self.encoder(x)
        init = x[:, 0]
        step = x[:, 1:].view(bsz * beams, self.max_steps, 2, x.shape[-1])
        op_h = step[:, :, 0]
        arg_h = step[:, :, 1]
        init_logits = self.init_head(init).view(bsz, beams, self.modulus)
        op_logits = self.op_head(op_h).view(bsz, beams, self.max_steps, len(OP_NAMES))
        arg_logits = self.arg_head(arg_h).view(bsz, beams, self.max_steps, self.modulus)
        selector_logits = self.selector(x.mean(dim=1)).view(bsz, beams)
        return init_logits, op_logits, arg_logits, selector_logits


class StructuredCyclicRuntime(nn.Module):
    def __init__(self, modulus: int, device: torch.device) -> None:
        super().__init__()
        table = torch.zeros(len(OP_NAMES), modulus, modulus, modulus, dtype=torch.float32)
        for op in range(len(OP_NAMES)):
            for arg in range(modulus):
                for old in range(modulus):
                    table[op, arg, old, apply_op(old, op, arg, modulus)] = 1.0
        self.register_buffer("table", table.to(device))

    def soft_trajectory(self, init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        flat = init_logits.dim() == 2
        if init_logits.dim() == 3:
            bsz, beams, mod = init_logits.shape
            init_logits = init_logits.reshape(bsz * beams, mod)
            op_logits = op_logits.reshape(bsz * beams, op_logits.shape[2], op_logits.shape[3])
            arg_logits = arg_logits.reshape(bsz * beams, arg_logits.shape[2], arg_logits.shape[3])
            lengths = lengths[:, None].expand(bsz, beams).reshape(-1)
        state = F.softmax(init_logits.float(), dim=-1)
        op_probs = F.softmax(op_logits.float(), dim=-1)
        arg_probs = F.softmax(arg_logits.float(), dim=-1)
        states: List[torch.Tensor] = []
        for step in range(op_logits.shape[1]):
            cand = torch.einsum("bp,oapq->boaq", state, self.table)
            weights = op_probs[:, step, :, None] * arg_probs[:, step, None, :]
            next_state = (cand * weights[:, :, :, None]).sum(dim=(1, 2))
            active = (lengths > step).float().unsqueeze(-1)
            state = active * next_state + (1.0 - active) * state
            states.append(state.clamp_min(1e-9))
        out = torch.stack(states, dim=1)
        if not flat:
            out = out.view(bsz, beams, out.shape[1], out.shape[2])
        return out

    def soft_forward(self, init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states = self.soft_trajectory(init_logits, op_logits, arg_logits, lengths)
        if states.dim() == 4:
            bsz, beams, steps, mod = states.shape
            idx = (lengths.clamp_min(1) - 1).view(bsz, 1, 1, 1).expand(bsz, beams, 1, mod)
            final = states.gather(2, idx).squeeze(2)
            no_step = lengths.eq(0).view(bsz, 1, 1)
            if no_step.any():
                init_state = F.softmax(init_logits.float(), dim=-1).clamp_min(1e-9)
                final = torch.where(no_step, init_state, final)
            return final.clamp_min(1e-9)
        idx = (lengths.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, states.shape[-1])
        final = states.gather(1, idx).squeeze(1)
        no_step = lengths.eq(0).view(-1, 1)
        if no_step.any():
            init_state = F.softmax(init_logits.float(), dim=-1).clamp_min(1e-9)
            final = torch.where(no_step, init_state, final)
        return final.clamp_min(1e-9)


def extract_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(last):
        return last
    raise RuntimeError("model did not return hidden states")


def forward_hidden(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return extract_hidden(outputs)


def gather_registers(hidden: torch.Tensor, register_pos: torch.Tensor) -> torch.Tensor:
    # register_pos: [batch, beam, register]
    bsz, beams, regs = register_pos.shape
    b_idx = torch.arange(hidden.shape[0], device=hidden.device).view(bsz, 1, 1).expand(bsz, beams, regs)
    return hidden[b_idx, register_pos]


def prompt_bank_count(args: argparse.Namespace) -> int:
    return 1 if args.beam_register_mode == "latent" else int(args.beam_count)


def prepare_register_h(hidden: torch.Tensor, batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    reg_h = gather_registers(hidden, batch["register_pos"])
    if args.beam_register_mode == "latent":
        if reg_h.shape[1] != 1:
            raise RuntimeError(f"latent mode expects one prompt bank, got {reg_h.shape[1]}")
        reg_h = reg_h.expand(-1, int(args.beam_count), -1, -1)
    return reg_h


def softmin(values: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        return values.min(dim=1).values
    return -tau * torch.logsumexp(-values / tau, dim=1)


def beam_trace_nll(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    init_weight: float,
    op_weight: float,
    arg_weight: float,
) -> torch.Tensor:
    bsz, beams, steps, _ = op_logits.shape
    init_targets = batch["init_value"][:, None].expand(bsz, beams).reshape(-1)
    init_loss = F.cross_entropy(init_logits.reshape(bsz * beams, -1), init_targets, reduction="none").view(bsz, beams)

    active = batch["ops"].ne(-100)
    active_f = active.float()
    denom = active_f.sum(dim=1).clamp_min(1.0)[:, None]
    op_targets = batch["ops"].clamp_min(0)[:, None, :].expand(bsz, beams, steps).reshape(-1)
    arg_targets = batch["args"].clamp_min(0)[:, None, :].expand(bsz, beams, steps).reshape(-1)
    op_loss = F.cross_entropy(op_logits.reshape(bsz * beams * steps, -1), op_targets, reduction="none").view(bsz, beams, steps)
    arg_loss = F.cross_entropy(arg_logits.reshape(bsz * beams * steps, -1), arg_targets, reduction="none").view(bsz, beams, steps)
    mask = active_f[:, None, :]
    op_term = (op_loss * mask).sum(dim=2) / denom
    arg_term = (arg_loss * mask).sum(dim=2) / denom
    return init_weight * init_loss + op_weight * op_term + arg_weight * arg_term


def beam_state_nll(state_probs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    bsz, beams, steps, mod = state_probs.shape
    active = batch["states"].ne(-100)
    active_f = active.float()
    denom = active_f.sum(dim=1).clamp_min(1.0)[:, None]
    targets = batch["states"].clamp_min(0)[:, None, :].expand(bsz, beams, steps)
    target_probs = state_probs.gather(3, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
    return -(target_probs.log() * active_f[:, None, :]).sum(dim=2) / denom


def final_target_probs(final_probs: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    return final_probs.gather(2, answers[:, None, None].expand(-1, final_probs.shape[1], 1)).squeeze(-1).clamp_min(1e-9)


def diversity_loss(final_probs: torch.Tensor) -> torch.Tensor:
    if final_probs.shape[1] <= 1:
        return final_probs.sum() * 0.0
    probs = final_probs.float()
    sim = torch.einsum("bim,bjm->bij", probs, probs)
    beams = probs.shape[1]
    mask = ~torch.eye(beams, dtype=torch.bool, device=probs.device)[None, :, :]
    return sim[mask.expand_as(sim)].mean()


def pair_set_consistency_loss(final_probs: torch.Tensor, batch: Dict[str, torch.Tensor], pair_size: int, tau: float) -> torch.Tensor:
    if pair_size != 2 or final_probs.shape[0] < 2 or final_probs.shape[0] % 2 != 0:
        return final_probs.sum() * 0.0
    pairs = final_probs.shape[0] // 2
    probs = final_probs.view(pairs, 2, final_probs.shape[1], final_probs.shape[2]).float().clamp_min(1e-9)
    left = probs[:, 0]
    right = probs[:, 1]
    kl_lr = (left[:, :, None, :] * (left[:, :, None, :].log() - right[:, None, :, :].log())).sum(dim=-1)
    kl_rl = (right[:, :, None, :] * (right[:, :, None, :].log() - left[:, None, :, :].log())).sum(dim=-1)
    soft_lr = softmin(kl_lr.reshape(pairs * left.shape[1], right.shape[1]), tau).mean()
    soft_rl = softmin(kl_rl.reshape(pairs * right.shape[1], left.shape[1]), tau).mean()
    return 0.5 * (soft_lr + soft_rl)


def train_losses(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    executor: StructuredCyclicRuntime,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    trace_nll = beam_trace_nll(
        init_logits,
        op_logits,
        arg_logits,
        batch,
        args.init_trace_loss_weight,
        args.op_trace_loss_weight,
        args.arg_trace_loss_weight,
    )
    state_probs = executor.soft_trajectory(init_logits, op_logits, arg_logits, batch["lengths"])
    final_probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
    state_nll = beam_state_nll(state_probs, batch)
    target_probs = final_target_probs(final_probs, batch["answer"])
    answer_loss = -(target_probs.mean(dim=1).clamp_min(1e-9).log()).mean()
    trace_set_loss = softmin(trace_nll, args.set_loss_tau).mean()
    state_set_loss = softmin(state_nll, args.set_loss_tau).mean()
    target_beam = trace_nll.detach().argmin(dim=1)
    selector_loss = F.cross_entropy(selector_logits, target_beam)
    div_loss = diversity_loss(final_probs)
    pair_loss = pair_set_consistency_loss(final_probs, batch, args.pair_size, args.pair_set_tau) if args.paired_train else div_loss * 0.0
    loss = (
        args.trace_loss_weight * trace_set_loss
        + args.executor_loss_weight * answer_loss
        + args.state_loss_weight * state_set_loss
        + args.selector_loss_weight * selector_loss
        + args.diversity_loss_weight * div_loss
        + args.pair_set_consistency_loss_weight * pair_loss
    )
    with torch.no_grad():
        best_trace = trace_nll.argmin(dim=1)
        best_answer = target_probs.argmax(dim=1)
        selector_choice = selector_logits.argmax(dim=1)
    return loss, {
        "trace_set_loss": float(trace_set_loss.detach().cpu()),
        "executor_loss": float(answer_loss.detach().cpu()),
        "state_set_loss": float(state_set_loss.detach().cpu()),
        "selector_loss": float(selector_loss.detach().cpu()),
        "diversity_loss": float(div_loss.detach().cpu()),
        "pair_set_loss": float(pair_loss.detach().cpu()),
        "selector_matches_best_trace": float(selector_choice.eq(best_trace).float().mean().detach().cpu()),
        "best_soft_answer_mass": float(target_probs.max(dim=1).values.mean().detach().cpu()),
        "mean_soft_answer_mass": float(target_probs.mean().detach().cpu()),
        "best_target_prob_beam_fraction": float(best_answer.eq(best_trace).float().mean().detach().cpu()),
    }


@torch.no_grad()
def execute_beams(
    init_pred: torch.Tensor,
    op_pred: torch.Tensor,
    arg_pred: torch.Tensor,
    lengths: torch.Tensor,
    modulus: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, beams, max_steps = op_pred.shape
    answers = torch.empty((bsz, beams), dtype=torch.long, device=init_pred.device)
    states = torch.full((bsz, beams, max_steps), -100, dtype=torch.long, device=init_pred.device)
    for row in range(bsz):
        length = int(lengths[row].item())
        for beam in range(beams):
            x = int(init_pred[row, beam].item())
            for step in range(length):
                x = apply_op(x, int(op_pred[row, beam, step].item()), int(arg_pred[row, beam, step].item()), modulus)
                states[row, beam, step] = x
            answers[row, beam] = x
    return answers, states


def prefix_fraction(states: torch.Tensor, target_states: torch.Tensor, lengths: torch.Tensor, selected: torch.Tensor) -> float:
    total = 0
    got = 0
    for row in range(states.shape[0]):
        beam = int(selected[row].item())
        length = int(lengths[row].item())
        total += length
        for step in range(length):
            if int(states[row, beam, step].item()) == int(target_states[row, step].item()):
                got += 1
            else:
                break
    return got / max(1, total)


def program_exact_matrix(
    init_pred: torch.Tensor,
    op_pred: torch.Tensor,
    arg_pred: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    bsz, beams, steps = op_pred.shape
    active = batch["ops"].ne(-100)[:, None, :]
    init_ok = init_pred.eq(batch["init_value"][:, None])
    op_ok = op_pred.eq(batch["ops"][:, None, :]) | ~active
    arg_ok = arg_pred.eq(batch["args"][:, None, :]) | ~active
    return init_ok & op_ok.all(dim=2) & arg_ok.all(dim=2)


def distinct_count_rows(values: torch.Tensor) -> float:
    counts = []
    for row in values.detach().cpu().tolist():
        counts.append(len(set(row)))
    return sum(counts) / max(1, len(counts))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    compiler: BeamProgramCompiler,
    executor: StructuredCyclicRuntime,
    dataset: ExampleSet,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    compiler.eval()
    totals: Dict[str, float] = {
        "n": 0.0,
        "beam0_correct": 0.0,
        "selected_correct": 0.0,
        "prior_correct": 0.0,
        "oracle_correct": 0.0,
        "beam0_program": 0.0,
        "selected_program": 0.0,
        "prior_program": 0.0,
        "oracle_program": 0.0,
        "selected_prefix_weighted": 0.0,
        "oracle_prefix_weighted": 0.0,
        "length_total": 0.0,
        "distinct_answers_total": 0.0,
        "selector_entropy_total": 0.0,
    }
    pair_answers: List[torch.Tensor] = []
    pair_lengths: List[torch.Tensor] = []
    pair_selected: List[torch.Tensor] = []
    pair_oracle_ok: List[torch.Tensor] = []
    pair_selected_states: List[torch.Tensor] = []
    pair_oracle_states: List[torch.Tensor] = []

    for start in range(0, len(dataset), args.eval_batch_size):
        chunk = dataset.examples[start : start + args.eval_batch_size]
        batch = collate_examples(chunk, int(tokenizer.pad_token_id), prompt_bank_count(args), args.max_steps, args.max_length, device)
        hidden = forward_hidden(model, batch)
        reg_h = prepare_register_h(hidden, batch, args)
        init_logits, op_logits, arg_logits, selector_logits = compiler(reg_h)
        final_probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
        init_pred = init_logits.argmax(dim=-1)
        op_pred = op_logits.argmax(dim=-1)
        arg_pred = arg_logits.argmax(dim=-1)
        pred_answers, pred_states = execute_beams(init_pred, op_pred, arg_pred, batch["lengths"], args.modulus)
        prog_exact = program_exact_matrix(init_pred, op_pred, arg_pred, batch)
        correct = pred_answers.eq(batch["answer"][:, None])

        target_probs = final_target_probs(final_probs, batch["answer"])
        selector_idx = selector_logits.argmax(dim=1)
        prior_score = (
            F.log_softmax(init_logits.float(), dim=-1).max(dim=-1).values
            + F.log_softmax(op_logits.float(), dim=-1).max(dim=-1).values.sum(dim=-1)
            + F.log_softmax(arg_logits.float(), dim=-1).max(dim=-1).values.sum(dim=-1)
        )
        prior_idx = prior_score.argmax(dim=1)
        oracle_idx = torch.where(correct.any(dim=1), correct.float().argmax(dim=1), target_probs.argmax(dim=1))
        row_idx = torch.arange(pred_answers.shape[0], device=device)

        bsz = pred_answers.shape[0]
        totals["n"] += bsz
        totals["beam0_correct"] += float(correct[:, 0].sum().item())
        totals["selected_correct"] += float(correct[row_idx, selector_idx].sum().item())
        totals["prior_correct"] += float(correct[row_idx, prior_idx].sum().item())
        totals["oracle_correct"] += float(correct.any(dim=1).sum().item())
        totals["beam0_program"] += float(prog_exact[:, 0].sum().item())
        totals["selected_program"] += float(prog_exact[row_idx, selector_idx].sum().item())
        totals["prior_program"] += float(prog_exact[row_idx, prior_idx].sum().item())
        totals["oracle_program"] += float(prog_exact.any(dim=1).sum().item())
        totals["selected_prefix_weighted"] += prefix_fraction(pred_states, batch["states"], batch["lengths"], selector_idx) * float(batch["lengths"].sum().item())
        totals["oracle_prefix_weighted"] += prefix_fraction(pred_states, batch["states"], batch["lengths"], oracle_idx) * float(batch["lengths"].sum().item())
        totals["length_total"] += float(batch["lengths"].sum().item())
        totals["distinct_answers_total"] += distinct_count_rows(pred_answers) * bsz
        selector_probs = F.softmax(selector_logits.float(), dim=-1).clamp_min(1e-9)
        totals["selector_entropy_total"] += float((-(selector_probs * selector_probs.log()).sum(dim=1)).sum().item())

        if dataset.paired and dataset.pair_size == 2:
            pair_answers.append(batch["answer"].detach().cpu())
            pair_lengths.append(batch["lengths"].detach().cpu())
            pair_selected.append(pred_answers[row_idx, selector_idx].detach().cpu())
            pair_oracle_ok.append(correct.any(dim=1).detach().cpu())
            pair_selected_states.append(pred_states[row_idx, selector_idx].detach().cpu())
            pair_oracle_states.append(pred_states[row_idx, oracle_idx].detach().cpu())

    n = max(1.0, totals["n"])
    out = {
        "n": totals["n"],
        "beam_count": float(args.beam_count),
        "beam0_accuracy": totals["beam0_correct"] / n,
        "selected_accuracy": totals["selected_correct"] / n,
        "prior_accuracy": totals["prior_correct"] / n,
        "oracle_accuracy": totals["oracle_correct"] / n,
        "beam0_program_exact": totals["beam0_program"] / n,
        "selected_program_exact": totals["selected_program"] / n,
        "prior_program_exact": totals["prior_program"] / n,
        "oracle_program_exact": totals["oracle_program"] / n,
        "selected_state_prefix_fraction": totals["selected_prefix_weighted"] / max(1.0, totals["length_total"]),
        "oracle_state_prefix_fraction": totals["oracle_prefix_weighted"] / max(1.0, totals["length_total"]),
        "avg_distinct_answers": totals["distinct_answers_total"] / n,
        "selector_entropy": totals["selector_entropy_total"] / n,
    }
    denom = out["oracle_accuracy"] - out["beam0_accuracy"]
    out["selected_oracle_gap_recovered"] = (out["selected_accuracy"] - out["beam0_accuracy"]) / denom if abs(denom) > 1e-9 else math.nan
    if dataset.paired and pair_answers:
        answers = torch.cat(pair_answers)
        selected = torch.cat(pair_selected)
        oracle_ok = torch.cat(pair_oracle_ok)
        lengths = torch.cat(pair_lengths)
        sel_states = torch.cat(pair_selected_states)
        oracle_states = torch.cat(pair_oracle_states)
        usable = (answers.numel() // 2) * 2
        pairs = usable // 2
        ans = answers[:usable].view(pairs, 2)
        sel = selected[:usable].view(pairs, 2)
        ok = oracle_ok[:usable].view(pairs, 2)
        lens = lengths[:usable].view(pairs, 2)
        sel_state = sel_states[:usable].view(pairs, 2, -1)
        oracle_state = oracle_states[:usable].view(pairs, 2, -1)
        out["selected_pair_answer_consistency"] = float(sel[:, 0].eq(sel[:, 1]).float().mean().item())
        out["selected_pair_both_correct"] = float(sel.eq(ans).all(dim=1).float().mean().item())
        out["oracle_pair_both_correct"] = float(ok.all(dim=1).float().mean().item())
        state_same = []
        oracle_state_same = []
        for i in range(pairs):
            length = int(lens[i, 0].item())
            state_same.append(bool(sel_state[i, 0, :length].eq(sel_state[i, 1, :length]).all().item()))
            oracle_state_same.append(bool(oracle_state[i, 0, :length].eq(oracle_state[i, 1, :length]).all().item()))
        out["selected_pair_state_consistency"] = sum(state_same) / max(1, len(state_same))
        out["oracle_pair_state_consistency"] = sum(oracle_state_same) / max(1, len(oracle_state_same))
    return out


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, nn.Module, str]:
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
    print(f"[load] Loading {args.model_id} with AutoModelForCausalLM", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **common)
    model.config.use_cache = False
    if args.use_lora and args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        target_modules: str | List[str]
        if args.lora_target_modules == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    return tokenizer, model, "AutoModelForCausalLM"


def optimizer_for(params: List[nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if not params:
        raise RuntimeError("no trainable parameters")
    if args.optimizer == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
        except Exception as exc:
            print(f"[optim] PagedAdamW8bit unavailable ({exc}); falling back to AdamW", flush=True)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return CHECKPOINT_ROOT / Path(args.output_dir).name


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


def flatten_final_metrics(run_name: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, metrics in results.get("final_metrics", {}).items():
        rows.append({"run": run_name, "split": split, **metrics})
    return rows


def collect_metadata(args: argparse.Namespace, loader: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers_loader": loader,
        "peft_installed": importlib.util.find_spec("peft") is not None,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    return meta


def train_run(
    model: nn.Module,
    hidden_dim: int,
    train_sets: List[Tuple[CurriculumStage, ExampleSet]],
    eval_sets: Dict[str, ExampleSet],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    compiler = BeamProgramCompiler(
        hidden_dim,
        args.modulus,
        args.max_steps,
        args.beam_count,
        args.register_width,
        args.register_layers,
        args.register_heads,
        args.register_dropout,
    ).to(device)
    executor = StructuredCyclicRuntime(args.modulus, device)
    model_trainable = any(p.requires_grad for p in model.parameters())
    modules: List[nn.Module] = [model, compiler]
    trainable = [p for module in modules for p in module.parameters() if p.requires_grad]
    opt = optimizer_for(trainable, args)
    pad_id = int(tokenizer.pad_token_id)
    log_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    global_step = 0

    for stage, dataset in train_sets:
        for local_step in range(1, stage.steps + 1):
            global_step += 1
            if model_trainable:
                model.train()
            else:
                model.eval()
            compiler.train()
            if args.paired_train and args.paired_batches:
                batch = sample_paired_batch(
                    dataset,
                    args.train_batch_size,
                    args.pair_size,
                    pad_id,
                    prompt_bank_count(args),
                    args.max_steps,
                    args.max_length,
                    device,
                )
            else:
                batch = sample_batch(dataset, args.train_batch_size, pad_id, prompt_bank_count(args), args.max_steps, args.max_length, device)
            if model_trainable:
                hidden = forward_hidden(model, batch)
            else:
                with torch.no_grad():
                    hidden = forward_hidden(model, batch)
            reg_h = prepare_register_h(hidden, batch, args)
            init_logits, op_logits, arg_logits, selector_logits = compiler(reg_h)
            loss, aux = train_losses(init_logits, op_logits, arg_logits, selector_logits, executor, batch, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()

            if local_step == 1 or local_step == stage.steps or global_step % args.eval_every == 0 or local_step % args.stage_eval_every == 0:
                row: Dict[str, Any] = {
                    "stage": stage.name,
                    "local_step": local_step,
                    "step": global_step,
                    "loss": float(loss.detach().cpu()),
                    **aux,
                }
                for split, eval_set in eval_sets.items():
                    metrics = evaluate(model, compiler, executor, eval_set, tokenizer, args, device)
                    for key, val in metrics.items():
                        row[f"{split}_{key}"] = val
                log_rows.append(row)
                write_csv(Path(args.output_dir) / "train_log.partial.csv", log_rows)
                with (Path(args.output_dir) / "latest_status.json").open("w") as f:
                    json.dump(row, f, indent=2)
                msg = f"[beam{args.beam_count}] {stage.name} step {global_step} loss={row['loss']:.4f}"
                for split in eval_sets:
                    key = f"{split}_selected_accuracy"
                    if key in row:
                        msg += f" {split}={100.0 * float(row[key]):.1f}%"
                    oracle_key = f"{split}_oracle_accuracy"
                    if oracle_key in row:
                        msg += f"/{100.0 * float(row[oracle_key]):.1f}%oracle"
                print(msg, flush=True)

    final_metrics = {split: evaluate(model, compiler, executor, eval_set, tokenizer, args, device) for split, eval_set in eval_sets.items()}
    ckpt_root = checkpoint_dir(args)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    if args.use_lora and hasattr(model, "save_pretrained"):
        adapter_dir = ckpt_root / "adapter"
        model.save_pretrained(adapter_dir)
        model_ckpt = str(adapter_dir)
    else:
        marker = ckpt_root / "frozen_backbone.txt"
        marker.write_text(f"Frozen backbone: {args.model_id}\n")
        model_ckpt = str(marker)
    heads_path = ckpt_root / "heads.pt"
    torch.save(
        {
            "args": vars(args),
            "compiler": compiler.state_dict(),
            "hidden_dim": hidden_dim,
            "beam_count": args.beam_count,
            "register_count": 1 + 2 * args.max_steps,
        },
        heads_path,
    )
    return {
        "train_seconds": time.time() - t0,
        "checkpoints": [model_ckpt, str(heads_path)],
        "train_log": log_rows,
        "final_metrics": final_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen latent beam program compiler experiment")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch_dtype", type=str, default="bf16")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--optimizer", type=str, default="paged_adamw_8bit", choices=["adamw", "paged_adamw_8bit"])
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--max_steps", type=int, default=24)
    p.add_argument("--beam_count", type=int, default=4)
    p.add_argument("--beam_register_mode", type=str, default="latent", choices=["latent", "prompt"])
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=12)
    p.add_argument("--curriculum_stages", type=str, default="short:1:4:100,medium:1:8:100,train:1:12:100,long:8:24:120")
    p.add_argument("--eval_lengths", type=str, default="4,8,12,24")
    p.add_argument("--train_template_mode", type=str, default="mixed", choices=["standard", "mixed", "paraphrase"])
    p.add_argument("--paired_train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--paired_batches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--paired_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--pair_size", type=int, default=2)
    p.add_argument("--eval_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--paired_eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--paired_eval_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--train_size", type=int, default=256)
    p.add_argument("--eval_size", type=int, default=128)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=120)
    p.add_argument("--stage_eval_every", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--register_width", type=int, default=512)
    p.add_argument("--register_layers", type=int, default=1)
    p.add_argument("--register_heads", type=int, default=4)
    p.add_argument("--register_dropout", type=float, default=0.05)
    p.add_argument("--trace_loss_weight", type=float, default=1.0)
    p.add_argument("--init_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--op_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--executor_loss_weight", type=float, default=1.0)
    p.add_argument("--state_loss_weight", type=float, default=1.0)
    p.add_argument("--selector_loss_weight", type=float, default=0.2)
    p.add_argument("--diversity_loss_weight", type=float, default=0.03)
    p.add_argument("--pair_set_consistency_loss_weight", type=float, default=0.05)
    p.add_argument("--set_loss_tau", type=float, default=0.5)
    p.add_argument("--pair_set_tau", type=float, default=0.5)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output_dir", type=str, default=str(ROOT / "runs/default"))
    p.add_argument("--checkpoint_dir", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, loader = load_model_and_tokenizer(args)
    if not args.load_in_4bit:
        model.to(device)
    metadata = collect_metadata(args, loader)

    stages = parse_curriculum_stages(args.curriculum_stages, args.train_min_len, args.train_max_len, 0)
    paired_modes = [x.strip() for x in args.paired_template_modes.split(",") if x.strip()]
    train_sets: List[Tuple[CurriculumStage, ExampleSet]] = []
    for idx, stage in enumerate(stages):
        gen = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, prompt_bank_count(args), args.seed + 31 * idx, args.train_template_mode)
        if args.paired_train:
            if len(paired_modes) != args.pair_size:
                raise ValueError("paired_template_modes count must match pair_size")
            stage_set = gen.paired_dataset(args.train_size, stage.min_len, stage.max_len, paired_modes)
        else:
            stage_set = gen.dataset(args.train_size, stage.min_len, stage.max_len)
        train_sets.append((stage, stage_set))

    eval_sets: Dict[str, ExampleSet] = {}
    eval_modes = [x.strip() for x in args.eval_template_modes.split(",") if x.strip()]
    paired_eval_modes = [x.strip() for x in args.paired_eval_template_modes.split(",") if x.strip()]
    eval_lengths = [int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()]
    for mode_idx, mode in enumerate(eval_modes):
        for length in eval_lengths:
            gen = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, prompt_bank_count(args), args.seed + 1000 + 97 * mode_idx + length, mode)
            eval_sets[f"{mode}_len{length}"] = gen.dataset(args.eval_size, length, length)
    if args.paired_eval:
        if len(paired_eval_modes) != 2:
            raise ValueError("paired_eval currently expects exactly two template modes")
        for length in eval_lengths:
            gen = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, prompt_bank_count(args), args.seed + 2000 + length, "mixed")
            eval_sets[f"paired_len{length}"] = gen.paired_dataset(args.eval_size, length, length, paired_eval_modes)

    probe = collate_examples(train_sets[0][1].examples[:1], int(tokenizer.pad_token_id), prompt_bank_count(args), args.max_steps, args.max_length, device)
    with torch.no_grad():
        hidden_dim = int(forward_hidden(model, probe).shape[-1])
    print(
        f"[setup] hidden_dim={hidden_dim} beams={args.beam_count} prompt_banks={prompt_bank_count(args)} "
        f"registers_per_bank={1 + 2 * args.max_steps} "
        f"seq_len={probe['input_ids'].shape[1]} eval_splits={len(eval_sets)}",
        flush=True,
    )

    results = {
        "args": vars(args),
        "metadata": metadata,
        "hidden_dim": hidden_dim,
        "beam_count": args.beam_count,
        "prompt_bank_count": prompt_bank_count(args),
        "beam_register_mode": args.beam_register_mode,
        "register_count": 1 + 2 * args.max_steps,
        "dataset": {
            "train_size": args.train_size,
            "curriculum_stages": [stage.__dict__ for stage in stages],
            "eval_size": args.eval_size,
            "eval_lengths": eval_lengths,
            "train_template_mode": args.train_template_mode,
            "paired_train": args.paired_train,
            "paired_template_modes": paired_modes,
            "eval_template_modes": eval_modes,
            "paired_eval": args.paired_eval,
            "paired_eval_template_modes": paired_eval_modes,
        },
    }
    run_result = train_run(model, hidden_dim, train_sets, eval_sets, tokenizer, args, device)
    results.update(run_result)
    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_csv(out / "train_log.csv", run_result["train_log"])
    rows = flatten_final_metrics(out.name, results)
    write_csv(out / "metrics.csv", rows)
    manifest = [
        {"run": out.name, "artifact": "adapter_or_backbone", "path": run_result["checkpoints"][0]},
        {"run": out.name, "artifact": "compiler_heads", "path": run_result["checkpoints"][1]},
    ]
    write_csv(ROOT / "checkpoint_manifest.csv", manifest)
    print(f"[done] wrote {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

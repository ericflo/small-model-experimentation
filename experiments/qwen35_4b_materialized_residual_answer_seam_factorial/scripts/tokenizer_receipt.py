#!/usr/bin/env python3
"""Build the model-free real-tokenizer and rendered-prompt parity receipt."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig, AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from identity import (  # noqa: E402
    PARENT_PREOUTCOME,
    file_sha256,
    verified_parent_preoutcome,
    verified_regular_file,
)


CONFIG = EXP / "configs" / "default.yaml"
OUTPUT = EXP / "runs" / "tokenizer" / "receipt_v2.json"
PREPARED = EXP / "runs" / "prepared"
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ALIASES = tuple("ABCDEFGHIJKLMNOPQRSTUVWX")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON is unsafe or absent: {path}")
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSONL is unsafe or absent: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"non-object JSONL row: {path}")
    return rows


def write_frozen(path: Path, value: Any) -> None:
    data = canonical_bytes(value)
    if path.is_symlink():
        raise RuntimeError("tokenizer receipt path is a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise RuntimeError("frozen tokenizer receipt differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("unsafe tokenizer receipt temporary path")
    temporary.write_bytes(data)
    temporary.replace(path)


def render_ids(tokenizer: Any, messages: list[dict[str, str]], thinking: bool) -> tuple[int, ...]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("chat template returned non-text")
    return tuple(tokenizer.encode(rendered, add_special_tokens=False))


def parent_rendered_inventory(tokenizer: Any) -> tuple[set[tuple[int, ...]], dict[str, Any]]:
    receipt_path = verified_parent_preoutcome(ROOT)
    receipt = read_json(receipt_path)
    rendered: set[tuple[int, ...]] = set()
    reads: dict[str, dict[str, Any]] = {
        str(receipt_path.relative_to(ROOT)): {
            "sha256": PARENT_PREOUTCOME["sha256"],
            "purpose": "authenticated_rendered_prompt_freshness",
        }
    }
    for name, row in sorted(receipt["files"].items()):
        if not name.endswith("_requests.jsonl"):
            continue
        path = verified_regular_file(ROOT, row["path"], row["sha256"])
        rows = read_jsonl(path)
        if len(rows) != row["rows"]:
            raise RuntimeError("parent request row count changed")
        thinking = name.startswith("suffix_") or name == "direct_requests.jsonl"
        rendered.update(render_ids(tokenizer, value["messages"], thinking) for value in rows)
        reads[str(path.relative_to(ROOT))] = {
            "sha256": row["sha256"],
            "rows": len(rows),
            "thinking": thinking,
            "purpose": "authenticated_rendered_prompt_freshness",
        }
    return rendered, dict(sorted(reads.items()))


def canonical_answer_receipt(tokenizer: Any, prefix_ids: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    compositional_failures: list[str] = []
    for arity in (2, 3):
        if arity == 2:
            programs = ((a, b) for a in ALIASES for b in ALIASES)
        else:
            programs = (
                (a, b, c) for a in ALIASES for b in ALIASES for c in ALIASES
            )
        for program in programs:
            line = "PROGRAM: " + " | ".join(program)
            tail = " " + " | ".join(program)
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            tail_ids = tokenizer.encode(tail, add_special_tokens=False)
            if line_ids != prefix_ids + tail_ids:
                compositional_failures.append(line)
            rows.append(
                {
                    "arity": arity,
                    "line": line,
                    "line_ids": line_ids,
                    "tail_ids": tail_ids,
                }
            )
    if compositional_failures:
        raise RuntimeError(
            "PROGRAM prefix is not token-compositional for canonical lines: "
            + repr(compositional_failures[:3])
        )
    return {
        "rows": len(rows),
        "arity_rows": dict(sorted(Counter(row["arity"] for row in rows).items())),
        "line_token_length_histogram": dict(
            sorted(Counter(len(row["line_ids"]) for row in rows).items())
        ),
        "tail_token_length_histogram": dict(
            sorted(Counter(len(row["tail_ids"]) for row in rows).items())
        ),
        "minimum_terminal_slack_tokens": 2,
        "all_fit_sampled_answer_cap_24_with_terminal_slack": all(
            len(row["tail_ids"]) + 2 <= 24 for row in rows
        ),
        "prefix_tail_token_compositional_failures": 0,
        "canonical_rows_sha256": canonical_sha256(rows),
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text())
    if config["model"]["id"] != MODEL_ID or config["model"]["revision"] != REVISION:
        raise RuntimeError("one-model boundary changed")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    hf_eos = int(model_config.text_config.eos_token_id)
    tokenizer_eos = int(tokenizer.eos_token_id)
    if hf_eos != 248044 or tokenizer_eos != 248046 or tokenizer.eos_token != "<|im_end|>":
        raise RuntimeError("Qwen termination identity changed")
    close_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    prefix_text = str(config["interface"]["answer_prefix_text"])
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if close_ids != [248069, 271]:
        raise RuntimeError(f"forced-close IDs changed: {close_ids}")
    if tokenizer.decode(close_ids + prefix_ids, skip_special_tokens=False) != "</think>\n\nPROGRAM:":
        raise RuntimeError("close plus PROGRAM token sequence does not decode compositionally")

    thinking_suffix = render_ids(
        tokenizer, [{"role": "user", "content": "receipt-probe"}], True
    )
    no_thinking_suffix = render_ids(
        tokenizer, [{"role": "user", "content": "receipt-probe"}], False
    )
    expected_thinking_tail = tuple(
        tokenizer.encode(
            "<|im_start|>assistant\n<think>\n", add_special_tokens=False
        )
    )
    expected_no_thinking_tail = tuple(
        tokenizer.encode(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
    )
    if thinking_suffix[-len(expected_thinking_tail) :] != expected_thinking_tail:
        raise RuntimeError("thinking chat suffix changed")
    if no_thinking_suffix[-len(expected_no_thinking_tail) :] != expected_no_thinking_tail:
        raise RuntimeError("no-thinking chat suffix changed")

    parent_rendered, parent_reads = parent_rendered_inventory(tokenizer)
    new_base_thinking: set[tuple[int, ...]] = set()
    new_base_no_thinking: set[tuple[int, ...]] = set()
    new_no_think_slot: set[tuple[int, ...]] = set()
    max_context = {
        "think512_freeform": 0,
        "think512_program_slot": 0,
        "no_think_freeform": 0,
        "no_think_program_slot": 0,
    }
    new_read_receipt: dict[str, dict[str, Any]] = {}
    for path in sorted(PREPARED.glob("*_requests.jsonl")):
        rows = read_jsonl(path)
        new_read_receipt[str(path.relative_to(ROOT))] = {
            "sha256": file_sha256(path),
            "rows": len(rows),
            "purpose": "rendered_prompt_freshness_and_context_fit",
        }
        for row in rows:
            thinking_ids = render_ids(tokenizer, row["messages"], True)
            no_thinking_ids = render_ids(tokenizer, row["messages"], False)
            new_base_thinking.add(thinking_ids)
            new_base_no_thinking.add(no_thinking_ids)
            new_no_think_slot.add(no_thinking_ids + tuple(prefix_ids))
            max_context["think512_freeform"] = max(
                max_context["think512_freeform"],
                len(thinking_ids) + 512 + len(close_ids) + 24,
            )
            max_context["think512_program_slot"] = max(
                max_context["think512_program_slot"],
                len(thinking_ids) + 512 + len(close_ids) + len(prefix_ids) + 24,
            )
            max_context["no_think_freeform"] = max(
                max_context["no_think_freeform"], len(no_thinking_ids) + 24
            )
            max_context["no_think_program_slot"] = max(
                max_context["no_think_program_slot"],
                len(no_thinking_ids) + len(prefix_ids) + 24,
            )
    overlaps = {
        "think_base": len(new_base_thinking & parent_rendered),
        "no_think_base": len(new_base_no_thinking & parent_rendered),
        "no_think_program_slot": len(new_no_think_slot & parent_rendered),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"rendered prompt token overlap with parent: {overlaps}")
    if max(max_context.values()) > int(config["generation"]["max_model_len"]):
        raise RuntimeError(f"registered context limit is exceeded: {max_context}")

    alias_receipt = {
        alias: {
            "plain": tokenizer.encode(alias, add_special_tokens=False),
            "leading_space": tokenizer.encode(" " + alias, add_special_tokens=False),
        }
        for alias in ALIASES
    }
    if any(
        len(value["plain"]) != 1 or len(value["leading_space"]) != 1
        for value in alias_receipt.values()
    ):
        raise RuntimeError("A-X aliases are not single tokens in both registered forms")

    receipt = {
        "schema_version": 2,
        "stage": "real_tokenizer_shared_thought_model_free_receipt",
        "decision": "TOKENIZER_AND_RENDERED_FRESHNESS_PASS",
        "model": MODEL_ID,
        "revision": REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": len(tokenizer),
        "config_sha256": file_sha256(CONFIG),
        "runner_sha256": file_sha256(SRC / "vllm_runner.py"),
        "tests_sha256": file_sha256(EXP / "tests" / "test_vllm_runner.py"),
        "termination": {
            "hf_model_eos_token_id": hf_eos,
            "tokenizer_eos_token_id": tokenizer_eos,
            "tokenizer_eos_token": tokenizer.eos_token,
        },
        "think_token_ids": {
            "open": tokenizer.encode("<think>", add_special_tokens=False),
            "close": tokenizer.encode("</think>", add_special_tokens=False),
            "forced_close_sequence": close_ids,
            "thinking_prompt_suffix": list(expected_thinking_tail),
            "no_thinking_prompt_suffix": list(expected_no_thinking_tail),
        },
        "answer_prefix": {
            "text": prefix_text,
            "token_ids": prefix_ids,
            "close_plus_prefix_decode_exact": True,
        },
        "aliases": alias_receipt,
        "canonical_answers": canonical_answer_receipt(tokenizer, prefix_ids),
        "rendered_prompt_inventory": {
            "parent_unique": len(parent_rendered),
            "new_thinking_unique": len(new_base_thinking),
            "new_no_thinking_unique": len(new_base_no_thinking),
            "new_no_think_slot_unique": len(new_no_think_slot),
            "parent_overlap": overlaps,
        },
        "registered_max_context_tokens": max_context,
        "max_model_len": int(config["generation"]["max_model_len"]),
        "read_receipt": {
            **parent_reads,
            **dict(sorted(new_read_receipt.items())),
        },
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
        "model_loaded": False,
        "model_calls": 0,
        "sampled_model_outputs": 0,
    }
    write_frozen(OUTPUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

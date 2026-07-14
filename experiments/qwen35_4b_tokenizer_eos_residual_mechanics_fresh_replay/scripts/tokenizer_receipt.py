#!/usr/bin/env python3
"""Build the frozen token grammar, prompt, termination, and freshness receipt."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig, AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from identity import (  # noqa: E402
    PARENT_COLLISION_MANIFEST,
    file_sha256,
    verified_parent_collision_manifest,
)


CONFIG = EXP / "configs/default.yaml"
PREOUTCOME = EXP / "runs/prepared/preoutcome_receipt.json"
PREPARED = EXP / "runs/prepared"
OUTPUT = EXP / "runs/tokenizer/receipt.json"
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ALIASES = tuple("ABCDEFGHIJKLMNOPQRSTUVWX")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
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


def render(
    tokenizer: Any, messages: list[dict[str, str]], *, thinking: bool
) -> tuple[str, tuple[int, ...]]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    if not isinstance(text, str):
        raise RuntimeError("chat template returned non-text")
    return text, tuple(tokenizer.encode(text, add_special_tokens=False))


def parent_rendered_inventory() -> tuple[set[str], dict[str, Any]]:
    path = verified_parent_collision_manifest(ROOT)
    manifest = read_json(path)
    rendered = manifest.get("prompt_token_sequence_sha256s")
    if (
        manifest.get("stage") != "hash_only_parent_collision_export"
        or not isinstance(rendered, list)
        or len(rendered) != 3648
        or len(set(rendered)) != 3648
        or any(not isinstance(value, str) or len(value) != 64 for value in rendered)
    ):
        raise RuntimeError("authenticated parent prompt-token domain changed")
    relative = str(path.relative_to(ROOT))
    return set(rendered), {
        relative: {
            "sha256": PARENT_COLLISION_MANIFEST["sha256"],
            "purpose": "authenticated_hash_only_parent_prompt_token_freshness",
        }
    }


def grammar_inventory(
    tokenizer: Any, prefix_ids: list[int], cap: int
) -> tuple[dict[str, Any], dict[tuple[str, ...], dict[str, list[int]]]]:
    inventory: dict[str, Any] = {"freeform": {}, "program_slot": {}}
    lookup: dict[tuple[str, ...], dict[str, list[int]]] = {}
    for arity in (2, 3):
        programs = list(product(ALIASES, repeat=arity))
        freeform: list[list[int]] = []
        slot_composed: list[list[int]] = []
        slot_remainders: list[list[int]] = []
        lines: list[str] = []
        equal_segmentation = 0
        for program in programs:
            line = "PROGRAM: " + " | ".join(program)
            tail = " " + " | ".join(program)
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            tail_ids = tokenizer.encode(tail, add_special_tokens=False)
            composed_ids = prefix_ids + tail_ids
            if (
                tokenizer.decode(line_ids, skip_special_tokens=False) != line
                or tokenizer.decode(tail_ids, skip_special_tokens=False) != tail
                or tokenizer.decode(composed_ids, skip_special_tokens=False) != line
            ):
                raise RuntimeError("registered grammar does not round-trip exactly")
            if line_ids == composed_ids:
                equal_segmentation += 1
            lines.append(line)
            freeform.append(line_ids)
            slot_composed.append(composed_ids)
            slot_remainders.append(tail_ids)
            lookup[tuple(program)] = {
                "freeform_composed": line_ids,
                "freeform_sampled": line_ids,
                "program_slot_composed": composed_ids,
                "program_slot_sampled": tail_ids,
            }
        if len({tuple(values) for values in freeform}) != 24**arity:
            raise RuntimeError("freeform grammar token sequences collide")
        if len({tuple(values) for values in slot_composed}) != 24**arity:
            raise RuntimeError("program-slot grammar token sequences collide")
        if max(map(len, freeform)) + 1 > cap or max(map(len, slot_remainders)) + 1 > cap:
            raise RuntimeError("registered grammar does not fit answer cap plus stop")
        line_hash = canonical_sha256(lines)
        inventory["freeform"][str(arity)] = {
            "rows": len(freeform),
            "semantic_lines_sha256": line_hash,
            "token_id_sequences": freeform,
            "token_id_sequences_sha256": canonical_sha256(freeform),
            "sampled_remainder_token_id_sequences": freeform,
            "sampled_remainder_token_id_sequences_sha256": canonical_sha256(
                freeform
            ),
            "sampled_token_length_histogram": dict(
                sorted(Counter(map(len, freeform)).items())
            ),
            "max_sampled_tokens_including_terminal": max(map(len, freeform)) + 1,
        }
        inventory["program_slot"][str(arity)] = {
            "rows": len(slot_composed),
            "semantic_lines_sha256": line_hash,
            "token_id_sequences": slot_composed,
            "token_id_sequences_sha256": canonical_sha256(slot_composed),
            "sampled_remainder_token_id_sequences": slot_remainders,
            "sampled_remainder_token_id_sequences_sha256": canonical_sha256(
                slot_remainders
            ),
            "sampled_token_length_histogram": dict(
                sorted(Counter(map(len, slot_remainders)).items())
            ),
            "max_sampled_tokens_including_terminal": max(map(len, slot_remainders))
            + 1,
            "same_segmentation_as_freeform_rows": equal_segmentation,
            "different_segmentation_from_freeform_rows": len(programs)
            - equal_segmentation,
        }
    return inventory, lookup


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text())
    preoutcome = read_json(PREOUTCOME)
    if (
        config["model"]["id"] != MODEL_ID
        or config["model"]["revision"] != REVISION
        or preoutcome.get("schema_version") != 2
        or preoutcome.get("decision") != "PREOUTCOME_PASS"
        or preoutcome.get("config_sha256") != file_sha256(CONFIG)
        or preoutcome.get("model_calls") != 0
        or preoutcome.get("sampled_model_outputs") != 0
    ):
        raise RuntimeError("construction/preoutcome boundary changed")
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
    if (
        hf_eos != 248044
        or tokenizer_eos != 248046
        or tokenizer.eos_token != "<|im_end|>"
    ):
        raise RuntimeError("Qwen termination identity changed")
    close_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    prefix_text = str(config["interface"]["answer_prefix_text"])
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if close_ids != [248069, 271] or not prefix_ids:
        raise RuntimeError("forced close or PROGRAM prefix token IDs changed")
    if (
        tokenizer.decode(close_ids + prefix_ids, skip_special_tokens=False)
        != "</think>\n\nPROGRAM:"
    ):
        raise RuntimeError("forced close plus answer prefix does not compose")

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
    probe_messages = [{"role": "user", "content": "receipt-probe"}]
    if not render(tokenizer, probe_messages, thinking=True)[1][
        -len(expected_thinking_tail) :
    ] == expected_thinking_tail:
        raise RuntimeError("thinking chat suffix changed")
    if not render(tokenizer, probe_messages, thinking=False)[1][
        -len(expected_no_thinking_tail) :
    ] == expected_no_thinking_tail:
        raise RuntimeError("no-thinking chat suffix changed")

    cap = int(config["interface"]["sampled_answer_cap"])
    inventories, grammar_lookup = grammar_inventory(tokenizer, prefix_ids, cap)
    parent_rendered, parent_reads = parent_rendered_inventory()
    new_thinking: set[tuple[int, ...]] = set()
    new_no_thinking: set[tuple[int, ...]] = set()
    new_no_think_slot: set[tuple[int, ...]] = set()
    prompt_reads: dict[str, dict[str, Any]] = {}
    calibration_prompts: dict[str, Any] = {}
    max_context = {
        "think512_freeform": 0,
        "think512_program_slot": 0,
        "no_think_freeform": 0,
        "no_think_program_slot": 0,
    }
    for path in sorted(PREPARED.glob("*_requests.jsonl")):
        rows = read_jsonl(path)
        relative = str(path.relative_to(ROOT))
        expected_entry = preoutcome["request_files"].get(relative)
        if expected_entry != {"rows": len(rows), "sha256": file_sha256(path)}:
            raise RuntimeError("prepared request differs from preoutcome receipt")
        prompt_reads[relative] = {
            "sha256": file_sha256(path),
            "rows": len(rows),
            "purpose": "prompt_token_freshness_and_context_fit",
        }
        for row in rows:
            thinking_text, thinking_ids = render(
                tokenizer, row["messages"], thinking=True
            )
            no_think_text, no_think_ids = render(
                tokenizer, row["messages"], thinking=False
            )
            new_thinking.add(thinking_ids)
            new_no_thinking.add(no_think_ids)
            new_no_think_slot.add(no_think_ids + tuple(prefix_ids))
            max_context["think512_freeform"] = max(
                max_context["think512_freeform"],
                len(thinking_ids) + 512 + len(close_ids) + cap,
            )
            max_context["think512_program_slot"] = max(
                max_context["think512_program_slot"],
                len(thinking_ids) + 512 + len(close_ids) + len(prefix_ids) + cap,
            )
            max_context["no_think_freeform"] = max(
                max_context["no_think_freeform"], len(no_think_ids) + cap
            )
            max_context["no_think_program_slot"] = max(
                max_context["no_think_program_slot"],
                len(no_think_ids) + len(prefix_ids) + cap,
            )
            if path.name == "calibration_requests.jsonl":
                calibration_prompts[row["id"]] = {
                    "think512": {
                        "token_ids": list(thinking_ids),
                        "prompt_text_sha256": hashlib.sha256(
                            thinking_text.encode("utf-8")
                        ).hexdigest(),
                    },
                    "no_think": {
                        "token_ids": list(no_think_ids),
                        "prompt_text_sha256": hashlib.sha256(
                            no_think_text.encode("utf-8")
                        ).hexdigest(),
                    },
                }
    overlaps = {
        "thinking_base": sum(canonical_sha256(list(row)) in parent_rendered for row in new_thinking),
        "no_think_base": sum(canonical_sha256(list(row)) in parent_rendered for row in new_no_thinking),
        "no_think_program_slot": sum(canonical_sha256(list(row)) in parent_rendered for row in new_no_think_slot),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"rendered prompt overlap with predecessor: {overlaps}")
    if max(max_context.values()) > int(config["generation"]["max_model_len"]):
        raise RuntimeError("registered context exceeds max_model_len")

    calibration_rows = read_jsonl(PREPARED / "calibration_requests.jsonl")
    calibration_expected: dict[str, dict[str, list[int]]] = {}
    calibration_sampled: dict[str, dict[str, list[int]]] = {}
    for row in calibration_rows:
        aliases = tuple(row["meta"]["expected"].removeprefix("PROGRAM: ").split(" | "))
        if aliases not in grammar_lookup or len(aliases) != int(row["meta"]["arity"]):
            raise RuntimeError("calibration expected answer left grammar")
        registered = grammar_lookup[aliases]
        calibration_expected[row["id"]] = {
            "freeform": registered["freeform_composed"],
            "program_slot": registered["program_slot_composed"],
        }
        calibration_sampled[row["id"]] = {
            "freeform": registered["freeform_sampled"],
            "program_slot": registered["program_slot_sampled"],
        }

    alias_receipt = {
        alias: {
            "plain": tokenizer.encode(alias, add_special_tokens=False),
            "leading_space": tokenizer.encode(" " + alias, add_special_tokens=False),
        }
        for alias in ALIASES
    }
    receipt = {
        "schema_version": 1,
        "stage": "token_native_boundary_grammar_and_prompt_receipt",
        "decision": "TOKENIZER_GRAMMAR_PROMPT_FRESHNESS_PASS",
        "model": MODEL_ID,
        "revision": REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": len(tokenizer),
        "config_sha256": file_sha256(CONFIG),
        "preoutcome_sha256": file_sha256(PREOUTCOME),
        "runner_sha256": file_sha256(SRC / "vllm_runner.py"),
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
        "program_slot_prefix_text": prefix_text,
        "program_slot_prefix_token_ids": prefix_ids,
        "aliases": alias_receipt,
        "grammar_inventories": inventories,
        "calibration_expected_token_ids": calibration_expected,
        "calibration_expected_sampled_token_ids": calibration_sampled,
        "calibration_prompt_token_ids": calibration_prompts,
        "rendered_prompt_inventory": {
            "predecessor_unique": len(parent_rendered),
            "new_thinking_unique": len(new_thinking),
            "new_no_thinking_unique": len(new_no_thinking),
            "new_no_think_slot_unique": len(new_no_think_slot),
            "predecessor_overlap": overlaps,
        },
        "registered_max_context_tokens": max_context,
        "max_model_len": int(config["generation"]["max_model_len"]),
        "read_receipt": {
            str(CONFIG.relative_to(ROOT)): {
                "sha256": file_sha256(CONFIG),
                "purpose": "frozen_config",
            },
            str(PREOUTCOME.relative_to(ROOT)): {
                "sha256": file_sha256(PREOUTCOME),
                "purpose": "preoutcome_authorization",
            },
            **parent_reads,
            **dict(sorted(prompt_reads.items())),
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
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "output": str(OUTPUT.relative_to(ROOT)),
                "sha256": file_sha256(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "grammar_rows": sum(
                    entry["rows"]
                    for condition in inventories.values()
                    for entry in condition.values()
                ),
                "model_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT, add_usage, empty_usage, estimate_text_tokens, sample_prompt_with_usage, write_manifest  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, load_quant_model, load_tokenizer  # noqa: E402


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def judge_prompt(tokenizer: Any, user_content: str) -> str:
    system = (
        "You are a careful Python code judge. Choose the candidate most likely to pass hidden tests. "
        "Use only the task, public tests, and candidate code. Return exactly one letter."
    )
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user_content}\n\nLetter:"


def build_user_prompt(row: dict[str, Any], visible_candidates: list[dict[str, Any]], code_by_id: dict[str, str]) -> tuple[str, dict[str, str]]:
    letter_map: dict[str, str] = {}
    chunks = [
        "Choose the candidate most likely to solve the task. Return one letter only.",
        "",
        f"Task:\n{row['task_text']}",
        "",
        "Public probe calls used for agreement only:",
        "\n".join(row.get("probe_calls", [])[:12]) or "(none)",
        "",
        "Candidates:",
    ]
    for index, candidate in enumerate(visible_candidates[: len(LETTERS)]):
        letter = LETTERS[index]
        candidate_id = candidate["pool_candidate_id"]
        letter_map[letter] = candidate_id
        code = code_by_id.get(candidate_id, "")
        chunks.append(f"\nCandidate {letter} ({candidate.get('pool_name')} / {candidate.get('source')}):\n{code[:2400]}")
    chunks.append("\nAnswer with exactly one candidate letter.")
    return "\n".join(chunks), letter_map


def parse_choice(text: str, valid_letters: set[str]) -> str | None:
    match = re.search(r"\b([A-Z])\b", text.strip().upper())
    if match and match.group(1) in valid_letters:
        return match.group(1)
    for char in text.strip().upper():
        if char in valid_letters:
            return char
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-records", type=Path, required=True)
    parser.add_argument("--candidate-records", type=Path, nargs="+", required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    code_by_id: dict[str, str] = {}
    for path in args.candidate_records:
        for record in load_jsonl(path):
            pool_name = record.get("arm_name", path.stem.replace("_records", ""))
            for idx, candidate in enumerate(record.get("candidates", [])):
                candidate_id = f"{pool_name}:{candidate.get('candidate_id', idx)}:{idx}"
                code_by_id[candidate_id] = candidate.get("code", "")

    rows = load_jsonl(args.selector_records)
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_quant_model(args.model_path, for_training=False)
    model.eval()

    out_rows: list[dict[str, Any]] = []
    usage = empty_usage()
    for row_index, row in enumerate(rows):
        visible = [candidate for candidate in row.get("candidates", []) if candidate.get("visible_all_pass")]
        if not visible:
            out_rows.append(
                {
                    "record_id": row["record_id"],
                    "task_id": row["task_id"],
                    "selected": False,
                    "choice": None,
                    "pool_candidate_id": None,
                    "full_pass": False,
                    "raw_completion": "",
                    "visible_candidate_count": 0,
                }
            )
            continue
        user_prompt, letter_map = build_user_prompt(row, visible, code_by_id)
        prompt = judge_prompt(tokenizer, user_prompt)
        completions, batch_usage = sample_prompt_with_usage(
            model,
            tokenizer,
            prompt,
            count=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=1,
            seed=args.seed + row_index * 1009,
        )
        usage = add_usage(usage, batch_usage)
        raw = completions[0] if completions else ""
        choice = parse_choice(raw, set(letter_map))
        chosen_id = letter_map.get(choice or "")
        chosen = next((candidate for candidate in visible if candidate["pool_candidate_id"] == chosen_id), None)
        out_rows.append(
            {
                "record_id": row["record_id"],
                "task_id": row["task_id"],
                "selected": chosen is not None,
                "choice": choice,
                "pool_candidate_id": chosen_id,
                "pool_name": chosen.get("pool_name") if chosen else None,
                "source": chosen.get("source") if chosen else None,
                "visible_all_pass": bool(chosen.get("visible_all_pass")) if chosen else False,
                "full_pass": bool(chosen.get("full_pass")) if chosen else False,
                "raw_completion": raw,
                "visible_candidate_count": len(visible),
                "prompt_tokens_estimate": estimate_text_tokens(tokenizer, prompt),
            }
        )

    commits = [row for row in out_rows if row.get("selected")]
    summary = {
        "experiment": EXPERIMENT,
        "selector": "frozen_qwen_visible_rerank",
        "records": len(out_rows),
        "commit_count": len(commits),
        "selected_hidden_correct": sum(1 for row in commits if row.get("full_pass")),
        "selected_recovery_rate": sum(1 for row in commits if row.get("full_pass")) / len(out_rows) if out_rows else 0.0,
        "selected_visible_hidden_wrong": sum(1 for row in commits if row.get("visible_all_pass") and not row.get("full_pass")),
        "selected_false_pass_rate": (
            sum(1 for row in commits if row.get("visible_all_pass") and not row.get("full_pass")) / len(commits) if commits else 0.0
        ),
        "token_usage": usage,
        "out": str(args.out),
    }
    write_jsonl(args.out, out_rows)
    write_json(args.summary, summary)
    write_manifest(args.out.with_suffix(".manifest.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

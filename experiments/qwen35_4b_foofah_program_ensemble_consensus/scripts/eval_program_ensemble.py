#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import direct_prompt, program_prompt, repair_prompt  # noqa: E402
from src.table_utils import compact_table, diff_tables, equal_table, extract_code, extract_json_table, table_key  # noqa: E402

MODEL_PATH = "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SAFE_EXEC = ROOT / "src" / "safe_exec.py"
DEFAULT_VARIANTS = ["verified_structural", "structural_python", "row_column_rule"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def load_model(load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    model.eval()
    return tokenizer, model


def render_prompt(tokenizer, system: str, prompt: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate_one(tokenizer, model, system: str, prompt: str, max_new_tokens: int, temperature: float | None = None, seed: int | None = None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    text = render_prompt(tokenizer, system, prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    do_sample = temperature is not None and temperature > 0
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
    else:
        kwargs.update({"do_sample": False, "temperature": None, "top_p": None})
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def run_generated_code(code: str, table: list[list[str]], timeout_s: float) -> dict[str, Any]:
    payload = json.dumps({"code": code, "table": table}, ensure_ascii=False)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(SAFE_EXEC)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "TimeoutExpired", "message": f">{timeout_s}s"}
    if proc.returncode != 0:
        return {"ok": False, "error": "ProcessError", "message": proc.stderr[-500:]}
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {"ok": False, "error": "BadRunnerJson", "message": proc.stdout[-500:]}


def failure_feedback(row: dict[str, Any], attempt: dict[str, Any]) -> str:
    if not attempt["code_found"]:
        return (
            "No valid transform(table) function was found.\n"
            f"Parse reason: {attempt['code_parse_reason']}\n"
            "Return only a complete Python function named transform(table)."
        )
    if not attempt["visible_exec_ok"]:
        return (
            "The function raised an error on the visible example.\n"
            f"Error type: {attempt.get('visible_exec_error')}\n"
            f"Error message: {attempt.get('visible_exec_message')}\n"
            "Repair it so it runs on the visible input and returns the expected visible output."
        )
    return (
        "The function ran on the visible input but returned the wrong table.\n"
        f"{diff_tables(row['output_table'], attempt.get('visible_output'))}\n\n"
        f"Actual visible output:\n{compact_table(attempt.get('visible_output'))}\n"
        "Repair the transformation logic so the visible output exactly matches the expected visible output."
    )


def evaluate_code(code_raw: str, row: dict[str, Any], variant: str, stage: str, exec_timeout: float) -> dict[str, Any]:
    code_found, code, code_parse_reason = extract_code(code_raw)
    visible_exec = {"ok": False, "error": "CodeParse", "message": code_parse_reason}
    hidden_exec = {"ok": False, "error": "Skipped", "message": "visible_exec_failed_or_code_missing"}
    visible_output = None
    hidden_output = None
    visible_pass = False
    hidden_exact = False
    if code_found:
        visible_exec = run_generated_code(code, row["input_table"], exec_timeout)
        if visible_exec.get("ok"):
            visible_output = visible_exec["result"]
            visible_pass = equal_table(visible_output, row["output_table"])
        hidden_exec = run_generated_code(code, row["testing_table"], exec_timeout)
        if hidden_exec.get("ok"):
            hidden_output = hidden_exec["result"]
            hidden_exact = equal_table(hidden_output, row["test_answer"])
    return {
        "variant": variant,
        "stage": stage,
        "code_raw": code_raw,
        "code_found": code_found,
        "code_parse_reason": code_parse_reason,
        "code": code,
        "visible_exec_ok": bool(visible_exec.get("ok")),
        "visible_exec_error": visible_exec.get("error"),
        "visible_exec_message": visible_exec.get("message"),
        "visible_output": visible_output,
        "visible_pass": visible_pass,
        "hidden_exec_ok": bool(hidden_exec.get("ok")),
        "hidden_exec_error": hidden_exec.get("error"),
        "hidden_exec_message": hidden_exec.get("message"),
        "hidden_output": hidden_output,
        "hidden_output_key": table_key(hidden_output),
        "hidden_exact": hidden_exact,
        "visible_verified_hidden_exact": bool(visible_pass and hidden_exact),
    }


def generate_program_candidate(tokenizer, model, row: dict[str, Any], variant: str, args, variant_index: int) -> dict[str, Any]:
    raw = generate_one(
        tokenizer,
        model,
        "You write executable Python table transformers. Output only Python code.",
        program_prompt(row, variant),
        args.max_code_tokens,
        temperature=None,
    )
    attempts = [evaluate_code(raw, row, variant, "initial", args.exec_timeout)]
    current = attempts[-1]
    for repair_idx in range(1, args.max_repairs + 1):
        if current["visible_pass"]:
            break
        feedback = failure_feedback(row, current)
        repaired_raw = generate_one(
            tokenizer,
            model,
            "You repair executable Python table transformers. Output only Python code.",
            repair_prompt(row, variant, current.get("code") or current.get("code_raw") or "", feedback, repair_idx),
            args.max_code_tokens,
            temperature=None,
        )
        current = evaluate_code(repaired_raw, row, variant, f"repair_{repair_idx}", args.exec_timeout)
        current["feedback_used"] = feedback
        attempts.append(current)
    final = next((a for a in attempts if a["visible_pass"]), attempts[-1])
    return {
        "variant": variant,
        "variant_index": variant_index,
        "attempts": attempts,
        "final": final,
    }


def choose_output(record: dict[str, Any], policy: str) -> tuple[str, str | None, bool]:
    direct_key = table_key(record.get("direct_table"))
    direct_ok = bool(record["direct_exact"])
    visible = [c for c in record["program_candidates"] if c["final"]["visible_pass"] and c["final"]["hidden_output_key"] is not None]
    if policy == "direct":
        return "direct", direct_key, direct_ok
    if policy == "first_visible_program":
        if visible:
            c = visible[0]
            return f"program:{c['variant']}", c["final"]["hidden_output_key"], bool(c["final"]["hidden_exact"])
        return "direct", direct_key, direct_ok
    if policy == "consensus_2":
        counts = Counter(c["final"]["hidden_output_key"] for c in visible)
        if counts:
            key, count = counts.most_common(1)[0]
            if count >= 2:
                correct = any(c["final"]["hidden_output_key"] == key and c["final"]["hidden_exact"] for c in visible)
                return "consensus_2", key, correct
        return "direct", direct_key, direct_ok
    if policy == "consensus_3":
        counts = Counter(c["final"]["hidden_output_key"] for c in visible)
        if counts:
            key, count = counts.most_common(1)[0]
            if count >= 3:
                correct = any(c["final"]["hidden_output_key"] == key and c["final"]["hidden_exact"] for c in visible)
                return "consensus_3", key, correct
        return "direct", direct_key, direct_ok
    if policy == "direct_or_program_agreement":
        if direct_key is not None:
            agree = [c for c in visible if c["final"]["hidden_output_key"] == direct_key]
            if agree:
                return "direct_program_agreement", direct_key, direct_ok
        return "direct", direct_key, direct_ok
    raise ValueError(f"unknown policy: {policy}")


POLICIES = ["direct", "first_visible_program", "consensus_2", "consensus_3", "direct_or_program_agreement"]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    policies: dict[str, Any] = {}
    for policy in POLICIES:
        exact = 0
        program_commits = 0
        program_commit_correct = 0
        direct_miss_recoveries = 0
        direct_correct_losses = 0
        consensus_commits = 0
        for record in records:
            source, _key, ok = choose_output(record, policy)
            exact += int(ok)
            direct_ok = bool(record["direct_exact"])
            source_is_program = source.startswith("program") or source.startswith("consensus")
            if source_is_program:
                program_commits += 1
                program_commit_correct += int(ok)
                direct_miss_recoveries += int((not direct_ok) and ok)
                direct_correct_losses += int(direct_ok and not ok)
            if source.startswith("consensus"):
                consensus_commits += 1
        policies[policy] = {
            "exact": exact,
            "accuracy": exact / n if n else 0,
            "program_commits": program_commits,
            "program_commit_correct": program_commit_correct,
            "program_commit_precision": program_commit_correct / program_commits if program_commits else None,
            "direct_miss_recoveries": direct_miss_recoveries,
            "direct_correct_losses": direct_correct_losses,
            "consensus_commits": consensus_commits,
        }
    visible_counts = [sum(c["final"]["visible_pass"] for c in r["program_candidates"]) for r in records]
    correct_visible_counts = [sum(c["final"]["visible_verified_hidden_exact"] for c in r["program_candidates"]) for r in records]
    unique_visible_outputs = [
        len({c["final"]["hidden_output_key"] for c in r["program_candidates"] if c["final"]["visible_pass"] and c["final"]["hidden_output_key"] is not None})
        for r in records
    ]
    oracle_program = sum(any(c["final"]["visible_verified_hidden_exact"] for c in r["program_candidates"]) for r in records)
    oracle_union = sum(r["direct_exact"] or any(c["final"]["visible_verified_hidden_exact"] for c in r["program_candidates"]) for r in records)
    return {
        "n": n,
        "direct_exact": sum(r["direct_exact"] for r in records),
        "direct_parse_ok": sum(r["direct_parse_ok"] for r in records),
        "oracle_visible_program": oracle_program,
        "oracle_union": oracle_union,
        "mean_visible_programs": sum(visible_counts) / n if n else 0,
        "mean_correct_visible_programs": sum(correct_visible_counts) / n if n else 0,
        "mean_unique_visible_outputs": sum(unique_visible_outputs) / n if n else 0,
        "tasks_with_visible_program": sum(v > 0 for v in visible_counts),
        "tasks_with_two_plus_visible_programs": sum(v >= 2 for v in visible_counts),
        "tasks_with_consensus_2": sum(
            bool([count for count in Counter(c["final"]["hidden_output_key"] for c in r["program_candidates"] if c["final"]["visible_pass"] and c["final"]["hidden_output_key"] is not None).values() if count >= 2])
            for r in records
        ),
        "policies": policies,
    }


def evaluate_record(tokenizer, model, row: dict[str, Any], args) -> dict[str, Any]:
    direct_raw = generate_one(
        tokenizer,
        model,
        "You transform tables exactly. Output only JSON. Do not explain.",
        direct_prompt(row),
        args.max_direct_tokens,
    )
    direct_parse_ok, direct_table, direct_fragment = extract_json_table(direct_raw)
    direct_exact = bool(direct_parse_ok and equal_table(direct_table, row["test_answer"]))
    candidates = [
        generate_program_candidate(tokenizer, model, row, variant, args, idx)
        for idx, variant in enumerate(args.variants)
    ]
    record = {
        "file": row["file"],
        "family": row["family"],
        "num_samples": row["num_samples"],
        "direct_raw": direct_raw,
        "direct_parse_ok": direct_parse_ok,
        "direct_table": direct_table,
        "direct_fragment": direct_fragment,
        "direct_output_key": table_key(direct_table),
        "direct_exact": direct_exact,
        "target_table": row["test_answer"],
        "program_candidates": candidates,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--records-out", type=Path, default=ROOT / "reports" / "ensemble_records.jsonl")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports" / "ensemble_summary.json")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-direct-tokens", type=int, default=768)
    parser.add_argument("--max-code-tokens", type=int, default=768)
    parser.add_argument("--exec-timeout", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.records_out.exists():
        args.records_out.unlink()
    rows = load_jsonl(args.cases)
    selected = []
    for idx, row in enumerate(rows):
        if idx % args.stride != 0:
            continue
        selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break

    existing_records = []
    completed_files = set()
    if args.resume and args.records_out.exists():
        existing_records = load_jsonl(args.records_out)
        completed_files = {r["file"] for r in existing_records}
        selected = [row for row in selected if row["file"] not in completed_files]
    tokenizer, model = load_model(args.load_in_4bit)
    records = list(existing_records)
    for i, row in enumerate(selected, start=1):
        record = evaluate_record(tokenizer, model, row, args)
        records.append(record)
        append_jsonl(args.records_out, record)
        if args.progress_every and (i % args.progress_every == 0 or i == len(selected)):
            partial = summarize(records)
            best = max(partial["policies"].items(), key=lambda kv: kv[1]["accuracy"])
            print(
                f"[{i}/{len(selected)} new, {partial['n']} total] best={best[0]} {best[1]['exact']}/{partial['n']} "
                f"direct={partial['direct_exact']} oracle_union={partial['oracle_union']} "
                f"visible_tasks={partial['tasks_with_visible_program']}",
                flush=True,
            )

    summary = summarize(records)
    write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

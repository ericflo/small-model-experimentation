#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.foofah import (  # noqa: E402
    compact_table,
    diff_tables,
    equal_table,
    extract_code,
    extract_json_table,
    repair_prompt,
)


MODEL_PATH = "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SAFE_EXEC = ROOT / "src" / "safe_exec.py"


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


def generate_one(tokenizer, model, system: str, prompt: str, max_new_tokens: int) -> str:
    text = render_prompt(tokenizer, system, prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
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


def failure_feedback(row: dict[str, Any], round_record: dict[str, Any]) -> str:
    if not round_record["code_found"]:
        return (
            "No valid transform(table) function was found in the generated text.\n"
            f"Parse reason: {round_record['code_parse_reason']}\n"
            "Return only a complete Python function named transform(table)."
        )
    if not round_record["visible_exec_ok"]:
        return (
            "The function raised an error on the visible example.\n"
            f"Error type: {round_record.get('visible_exec_error')}\n"
            f"Error message: {round_record.get('visible_exec_message')}\n"
            "Repair the function so it runs on the visible input and returns the expected visible output."
        )
    return (
        "The function ran on the visible input but returned the wrong table.\n"
        f"{diff_tables(row['output_table'], round_record.get('visible_output'))}\n\n"
        f"Actual visible output:\n{compact_table(round_record.get('visible_output'))}\n\n"
        "Repair the transformation logic so the visible output exactly matches the expected visible output."
    )


def evaluate_code_round(code_raw: str, row: dict[str, Any], round_index: int, exec_timeout: float) -> dict[str, Any]:
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
        "round": round_index,
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
        "hidden_exact": hidden_exact,
        "visible_verified_hidden_exact": bool(visible_pass and hidden_exact),
    }


def evaluate_record(tokenizer, model, row: dict[str, Any], args) -> dict[str, Any]:
    direct_raw = generate_one(
        tokenizer,
        model,
        "You transform tables exactly. Output only JSON. Do not explain.",
        row["direct_prompt"],
        args.max_direct_tokens,
    )
    direct_parse_ok, direct_table, direct_fragment = extract_json_table(direct_raw)
    direct_exact = bool(direct_parse_ok and direct_table is not None and equal_table(direct_table, row["test_answer"]))

    rounds = []
    initial_raw = generate_one(
        tokenizer,
        model,
        "You write executable Python table transformers. Output only Python code.",
        row["initial_program_prompt"],
        args.max_code_tokens,
    )
    current = evaluate_code_round(initial_raw, row, 0, args.exec_timeout)
    rounds.append(current)
    stopped_reason = "visible_pass" if current["visible_pass"] else "max_repairs"

    for repair_idx in range(1, args.max_repairs + 1):
        if current["visible_pass"]:
            break
        feedback = failure_feedback(row, current)
        prompt = repair_prompt(row, current.get("code") or current.get("code_raw") or "", feedback, repair_idx)
        raw = generate_one(
            tokenizer,
            model,
            "You repair executable Python table transformers. Output only Python code.",
            prompt,
            args.max_code_tokens,
        )
        current = evaluate_code_round(raw, row, repair_idx, args.exec_timeout)
        current["feedback_used"] = feedback
        rounds.append(current)
        if current["visible_pass"]:
            stopped_reason = "visible_pass"
            break

    final = rounds[-1]
    first_visible_round = next((r for r in rounds if r["visible_pass"]), None)
    if first_visible_round is not None:
        final = first_visible_round
        stopped_reason = "visible_pass"
    final_program_exact = bool(final["visible_pass"] and final["hidden_exact"])
    any_round_hidden_exact = any(r["hidden_exact"] for r in rounds)
    any_visible_verified_hidden_exact = any(r["visible_verified_hidden_exact"] for r in rounds)
    direct_program_agreement = bool(
        direct_parse_ok
        and final.get("hidden_output") is not None
        and equal_table(direct_table, final["hidden_output"])
    )
    agreement_correct = bool(direct_program_agreement and direct_exact and final_program_exact)
    hybrid_exact = direct_exact
    hybrid_source = "direct"
    if not direct_parse_ok and final["visible_pass"]:
        hybrid_exact = final_program_exact
        hybrid_source = "program_parse_fallback"

    return {
        "file": row["file"],
        "family": row["family"],
        "num_samples": row["num_samples"],
        "direct_raw": direct_raw,
        "direct_parse_ok": direct_parse_ok,
        "direct_fragment": direct_fragment,
        "direct_table": direct_table,
        "direct_exact": direct_exact,
        "rounds": rounds,
        "round_count": len(rounds),
        "stopped_reason": stopped_reason,
        "final_round": final["round"],
        "final_visible_pass": final["visible_pass"],
        "final_hidden_exact": final_program_exact,
        "final_hidden_output": final.get("hidden_output"),
        "initial_visible_pass": rounds[0]["visible_pass"],
        "initial_hidden_exact": bool(rounds[0]["visible_pass"] and rounds[0]["hidden_exact"]),
        "any_round_hidden_exact": any_round_hidden_exact,
        "any_visible_verified_hidden_exact": any_visible_verified_hidden_exact,
        "direct_program_agreement": direct_program_agreement,
        "agreement_correct": agreement_correct,
        "hybrid_exact": hybrid_exact,
        "hybrid_source": hybrid_source,
        "target_table": row["test_answer"],
    }


def summarize(records: list[dict[str, Any]], max_repairs: int) -> dict[str, Any]:
    def c(key: str, group: list[dict[str, Any]]) -> int:
        return sum(bool(row[key]) for row in group)

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        if not n:
            return {"n": 0}
        agreement = c("direct_program_agreement", group)
        final_visible = c("final_visible_pass", group)
        final_exact = c("final_hidden_exact", group)
        union = sum(row["direct_exact"] or row["final_hidden_exact"] for row in group)
        return {
            "n": n,
            "direct_exact": c("direct_exact", group),
            "direct_accuracy": c("direct_exact", group) / n,
            "direct_parse_ok": c("direct_parse_ok", group),
            "direct_parse_rate": c("direct_parse_ok", group) / n,
            "initial_visible_pass": c("initial_visible_pass", group),
            "initial_visible_pass_rate": c("initial_visible_pass", group) / n,
            "initial_hidden_exact": c("initial_hidden_exact", group),
            "initial_accuracy": c("initial_hidden_exact", group) / n,
            "final_visible_pass": final_visible,
            "final_visible_pass_rate": final_visible / n,
            "final_hidden_exact": final_exact,
            "final_program_accuracy": final_exact / n,
            "visible_false_pass": final_visible - final_exact,
            "visible_false_pass_rate_among_visible": (final_visible - final_exact) / final_visible if final_visible else 0.0,
            "program_only": sum((not row["direct_exact"]) and row["final_hidden_exact"] for row in group),
            "direct_only": sum(row["direct_exact"] and not row["final_hidden_exact"] for row in group),
            "both_direct_and_program": sum(row["direct_exact"] and row["final_hidden_exact"] for row in group),
            "neither": sum((not row["direct_exact"]) and (not row["final_hidden_exact"]) for row in group),
            "oracle_union": union,
            "oracle_union_rate": union / n,
            "hybrid_exact": c("hybrid_exact", group),
            "hybrid_accuracy": c("hybrid_exact", group) / n,
            "agreement": agreement,
            "agreement_rate": agreement / n,
            "agreement_correct": c("agreement_correct", group),
            "agreement_precision": c("agreement_correct", group) / agreement if agreement else 0.0,
            "any_round_hidden_exact": c("any_round_hidden_exact", group),
            "any_round_hidden_exact_rate": c("any_round_hidden_exact", group) / n,
            "any_visible_verified_hidden_exact": c("any_visible_verified_hidden_exact", group),
            "any_visible_verified_hidden_exact_rate": c("any_visible_verified_hidden_exact", group) / n,
            "mean_round_count": sum(row["round_count"] for row in group) / n,
        }

    families = sorted({row["family"] for row in records})
    samples = sorted({row["num_samples"] for row in records})
    round_stats = {}
    for r in range(max_repairs + 1):
        seen = [row for row in records if any(rr["round"] == r for rr in row["rounds"])]
        round_stats[str(r)] = {
            "attempted": len(seen),
            "visible_pass": sum(any(rr["round"] == r and rr["visible_pass"] for rr in row["rounds"]) for row in records),
            "visible_verified_hidden_exact": sum(any(rr["round"] == r and rr["visible_verified_hidden_exact"] for rr in row["rounds"]) for row in records),
            "hidden_exact": sum(any(rr["round"] == r and rr["hidden_exact"] for rr in row["rounds"]) for row in records),
        }
    return {
        "overall": metrics(records),
        "round_stats": round_stats,
        "by_family": {fam: metrics([r for r in records if r["family"] == fam]) for fam in families},
        "by_num_samples": {str(s): metrics([r for r in records if r["num_samples"] == s]) for s in samples},
    }


def output_paths(limit: int, tag: str) -> tuple[Path, Path]:
    suffix = (f"_{tag}" if tag else "") + (f"_limit{limit}" if limit else "")
    return ROOT / "reports" / f"eval_records{suffix}.jsonl", ROOT / "reports" / f"eval_summary{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--max-direct-tokens", type=int, default=768)
    parser.add_argument("--max-code-tokens", type=int, default=768)
    parser.add_argument("--exec-timeout", type=float, default=1.5)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    cases = cases[args.start_index :: args.stride]
    if args.max_cases:
        cases = cases[: args.max_cases]
    if args.limit:
        cases = cases[: args.limit]
    records_path, summary_path = output_paths(args.limit or args.max_cases, args.tag)
    done_files = set()
    if args.resume and records_path.exists():
        done_files = {row["file"] for row in load_jsonl(records_path)}
    elif records_path.exists():
        records_path.unlink()
    tokenizer, model = load_model(args.load_in_4bit)
    for idx, row in enumerate(cases):
        if row["file"] in done_files:
            continue
        record = evaluate_record(tokenizer, model, row, args)
        append_jsonl(records_path, record)
        if (idx + 1) % args.progress_every == 0:
            print(
                f"evaluated {idx + 1}/{len(cases)} "
                f"direct={int(record['direct_exact'])} "
                f"init={int(record['initial_hidden_exact'])} "
                f"final={int(record['final_hidden_exact'])} "
                f"visible={int(record['final_visible_pass'])} "
                f"rounds={record['round_count']}",
                flush=True,
            )
    records = load_jsonl(records_path)
    summary = summarize(records, args.max_repairs)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

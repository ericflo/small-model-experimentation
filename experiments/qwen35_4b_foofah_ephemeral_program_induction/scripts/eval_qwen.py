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

from src.foofah import equal_table, extract_code, extract_json_table  # noqa: E402


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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
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


def evaluate_record(tokenizer, model, row: dict[str, Any], program_prompt: str, args) -> dict[str, Any]:
    direct_raw = generate_one(
        tokenizer,
        model,
        "You transform tables exactly. Output only JSON. Do not explain.",
        row["direct_prompt"],
        args.max_direct_tokens,
    )
    direct_parse_ok, direct_table, direct_fragment = extract_json_table(direct_raw)
    direct_exact = bool(direct_parse_ok and direct_table is not None and equal_table(direct_table, row["test_answer"]))

    code_raw = generate_one(
        tokenizer,
        model,
        "You write executable Python table transformers. Output only Python code.",
        row[f"program_prompt_{program_prompt}"],
        args.max_code_tokens,
    )
    code_found, code, code_parse_reason = extract_code(code_raw)
    visible_exec = {"ok": False, "error": "CodeParse", "message": code_parse_reason}
    hidden_exec = {"ok": False, "error": "Skipped", "message": "visible_failed_or_code_missing"}
    visible_pass = False
    program_hidden_exact = False
    program_output = None
    if code_found:
        visible_exec = run_generated_code(code, row["input_table"], args.exec_timeout)
        if visible_exec.get("ok"):
            visible_pass = equal_table(visible_exec["result"], row["output_table"])
        if visible_pass:
            hidden_exec = run_generated_code(code, row["testing_table"], args.exec_timeout)
            if hidden_exec.get("ok"):
                program_output = hidden_exec["result"]
                program_hidden_exact = equal_table(program_output, row["test_answer"])

    agreement = bool(direct_parse_ok and visible_pass and program_output is not None and equal_table(direct_table, program_output))
    agreement_correct = bool(agreement and direct_exact and program_hidden_exact)
    hybrid_exact = direct_exact
    hybrid_source = "direct"
    if not direct_parse_ok and visible_pass and program_output is not None:
        hybrid_exact = program_hidden_exact
        hybrid_source = "program_parse_fallback"

    return {
        "file": row["file"],
        "family": row["family"],
        "num_samples": row["num_samples"],
        "direct_parse_ok": direct_parse_ok,
        "direct_exact": direct_exact,
        "direct_fragment": direct_fragment,
        "direct_table": direct_table,
        "direct_raw": direct_raw,
        "code_found": code_found,
        "code_parse_reason": code_parse_reason,
        "code": code,
        "code_raw": code_raw,
        "visible_exec_ok": bool(visible_exec.get("ok")),
        "visible_exec_error": visible_exec.get("error"),
        "visible_exec_message": visible_exec.get("message"),
        "visible_pass": visible_pass,
        "hidden_exec_ok": bool(hidden_exec.get("ok")),
        "hidden_exec_error": hidden_exec.get("error"),
        "hidden_exec_message": hidden_exec.get("message"),
        "program_hidden_exact": program_hidden_exact,
        "program_output": program_output,
        "agreement": agreement,
        "agreement_correct": agreement_correct,
        "hybrid_exact": hybrid_exact,
        "hybrid_source": hybrid_source,
        "target_table": row["test_answer"],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key: str, group: list[dict[str, Any]]) -> int:
        return sum(bool(row[key]) for row in group)

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        if not n:
            return {"n": 0}
        agreement = count("agreement", group)
        agreement_correct = count("agreement_correct", group)
        return {
            "n": n,
            "direct_exact": count("direct_exact", group),
            "direct_accuracy": count("direct_exact", group) / n,
            "direct_parse_ok": count("direct_parse_ok", group),
            "direct_parse_rate": count("direct_parse_ok", group) / n,
            "code_found": count("code_found", group),
            "code_found_rate": count("code_found", group) / n,
            "visible_exec_ok": count("visible_exec_ok", group),
            "visible_exec_ok_rate": count("visible_exec_ok", group) / n,
            "visible_pass": count("visible_pass", group),
            "visible_pass_rate": count("visible_pass", group) / n,
            "program_hidden_exact": count("program_hidden_exact", group),
            "program_accuracy": count("program_hidden_exact", group) / n,
            "agreement": agreement,
            "agreement_rate": agreement / n,
            "agreement_correct": agreement_correct,
            "agreement_precision": agreement_correct / agreement if agreement else 0.0,
            "hybrid_exact": count("hybrid_exact", group),
            "hybrid_accuracy": count("hybrid_exact", group) / n,
            "oracle_union": sum(row["direct_exact"] or row["program_hidden_exact"] for row in group),
            "oracle_union_rate": sum(row["direct_exact"] or row["program_hidden_exact"] for row in group) / n,
        }

    families = sorted({row["family"] for row in records})
    samples = sorted({row["num_samples"] for row in records})
    return {
        "overall": metrics(records),
        "by_family": {family: metrics([r for r in records if r["family"] == family]) for family in families},
        "by_num_samples": {str(k): metrics([r for r in records if r["num_samples"] == k]) for k in samples},
    }


def output_paths(program_prompt: str, limit: int, tag: str) -> tuple[Path, Path]:
    suffix = f"{program_prompt}" + (f"_{tag}" if tag else "") + (f"_limit{limit}" if limit else "")
    return ROOT / "reports" / f"eval_records_{suffix}.jsonl", ROOT / "reports" / f"eval_summary_{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--program-prompt", choices=["induce", "context", "context_v2"], default="induce")
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
    records_path, summary_path = output_paths(args.program_prompt, args.limit or args.max_cases, args.tag)
    done_files = set()
    if args.resume and records_path.exists():
        done_files = {row["file"] for row in load_jsonl(records_path)}
    elif records_path.exists():
        records_path.unlink()
    tokenizer, model = load_model(args.load_in_4bit)
    for idx, row in enumerate(cases):
        if row["file"] in done_files:
            continue
        record = evaluate_record(tokenizer, model, row, args.program_prompt, args)
        append_jsonl(records_path, record)
        if (idx + 1) % args.progress_every == 0:
            print(
                f"evaluated {idx + 1}/{len(cases)} "
                f"direct={int(record['direct_exact'])} visible={int(record['visible_pass'])} "
                f"program={int(record['program_hidden_exact'])} agree={int(record['agreement'])}",
                flush=True,
            )
    records = load_jsonl(records_path)
    summary = summarize(records)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

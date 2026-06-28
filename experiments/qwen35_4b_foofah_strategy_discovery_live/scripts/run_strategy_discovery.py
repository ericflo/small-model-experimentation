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

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.table_utils import compact_table, diff_tables, equal_table, extract_code, table_key  # noqa: E402


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


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


def generate_one(
    tokenizer,
    model,
    system: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float | None = None,
) -> dict[str, Any]:
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
    return {
        "text": tokenizer.decode(gen, skip_special_tokens=True),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(gen.shape[0]),
        "total_tokens": int(inputs["input_ids"].shape[1] + gen.shape[0]),
    }


def fallback_cards() -> list[dict[str, str]]:
    return [
        {
            "id": "discovered_column_contraction_group_filter",
            "name": "column contraction with row grouping/filtering",
            "instruction": (
                "Assume the transformation may reduce columns by selecting identifier/measure columns, filtering repeated groups, "
                "or keeping only representative rows. Infer which columns survive from the example output, preserve row order unless "
                "the example clearly sorts, and implement the general column-selection plus grouping/filtering rule without hard-coding values."
            ),
        },
        {
            "id": "discovered_unpivot_split_fold",
            "name": "unpivot, split, and fold reshape",
            "instruction": (
                "Assume the transformation may reshape a wide table into a longer table or split compound cells into multiple rows. "
                "Map headers and repeated cell positions into output rows, handle blank cells carefully, and implement the reshape with explicit loops."
            ),
        },
        {
            "id": "discovered_header_semantic_aggregation",
            "name": "header-aware grouping and aggregation",
            "instruction": (
                "Use headers and row labels to identify key columns, category columns, and value columns. If the example groups, counts, "
                "deduplicates, or aggregates rows, implement that grouping logic generically with Counter/defaultdict-style structures."
            ),
        },
    ]


def calibration_examples(records: list[dict[str, Any]], cases: dict[str, dict[str, Any]], max_examples: int = 8) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        visible_correct = [c for c in record["program_candidates"] if c["final"].get("visible_verified_hidden_exact")]
        if record.get("direct_exact") or not visible_correct:
            continue
        case = cases[record["file"]]
        selected.append(
            {
                "family": record["family"],
                "file": record["file"],
                "visible_input": case["input_table"],
                "visible_output": case["output_table"],
                "new_input_shape": [len(case["testing_table"]), max([len(r) for r in case["testing_table"]] or [0])],
                "winning_strategy": visible_correct[0]["variant"],
                "visible_output_shape": [len(case["output_table"]), max([len(r) for r in case["output_table"]] or [0])],
                "visible_input_shape": [len(case["input_table"]), max([len(r) for r in case["input_table"]] or [0])],
            }
        )
    return selected[:max_examples]


def discovery_prompt(examples: list[dict[str, Any]], max_cards: int) -> str:
    compact_examples = []
    for ex in examples:
        compact_examples.append(
            {
                "family": ex["family"],
                "input_shape": ex["visible_input_shape"],
                "output_shape": ex["visible_output_shape"],
                "winning_strategy": ex["winning_strategy"],
                "visible_input": ex["visible_input"][:4],
                "visible_output": ex["visible_output"][:4],
            }
        )
    return (
        "Design reusable strategy prompts for generating Python table transformation programs.\n"
        "You are given calibration examples where a direct JSON answer failed but an executable program strategy succeeded.\n"
        "Propose strategies that should transfer to unseen table transformations. Do not write task-specific solutions.\n\n"
        f"Return exactly {max_cards} JSON objects in a JSON array. Each object must have keys: id, name, instruction.\n"
        "The instruction should be 2-4 sentences that can be prepended to a Python code generation prompt.\n\n"
        "Constraints for every strategy: use only plain Python lists, loops, string methods, re, math, Counter, and defaultdict. "
        "Do not mention pandas, numpy, files, imports, or any library outside those helpers. "
        "Do not reuse existing generic names like verified_structural or cell_parser; invent a descriptive reusable strategy name.\n\n"
        f"Calibration examples:\n{json.dumps(compact_examples, ensure_ascii=False)}\n\n"
        "JSON strategy cards:"
    )


def parse_cards(text: str, max_cards: int) -> list[dict[str, str]]:
    start = text.find("[")
    end = text.rfind("]")
    candidates = []
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, list):
            continue
        cards = []
        for i, row in enumerate(obj):
            if not isinstance(row, dict):
                continue
            instruction = str(row.get("instruction", "")).strip()
            instruction = instruction.replace("pandas indexing", "plain Python list indexing")
            instruction = instruction.replace("pandas", "plain Python lists")
            instruction = instruction.replace("numpy", "plain Python lists")
            if len(instruction) < 40:
                continue
            raw_id = str(row.get("id") or f"discovered_{i}").strip().lower()
            safe_id = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_id)[:64]
            cards.append(
                {
                    "id": safe_id or f"discovered_{i}",
                    "name": str(row.get("name") or safe_id),
                    "instruction": instruction,
                }
            )
        if cards:
            return cards[:max_cards]
    return fallback_cards()[:max_cards]


def discover_strategy_cards(
    tokenizer,
    model,
    records: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    max_cards: int,
    out_path: Path,
) -> list[dict[str, str]]:
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return data["cards"]
    examples = calibration_examples(records, cases)
    if not examples:
        cards = fallback_cards()[:max_cards]
        write_json(out_path, {"source": "fallback_no_calibration_examples", "cards": cards})
        return cards
    raw = generate_one(
        tokenizer,
        model,
        "You design reusable prompts for program synthesis. Output only JSON.",
        discovery_prompt(examples, max_cards),
        max_new_tokens=900,
        temperature=0.2,
    )
    parsed = parse_cards(raw["text"], max_cards)
    write_json(
        out_path,
        {
            "source": "qwen_discovery",
            "raw": raw,
            "calibration_examples": examples,
            "cards": parsed,
        },
    )
    return parsed


def common_program_context(row: dict[str, Any]) -> str:
    return (
        "Output only Python code. Define exactly one function named transform(table). "
        "The function must return a list of rows, where each row is a list of strings. "
        "Do not print, do not read files, and do not call input(). "
        "Do not include import statements; helper names are already available: re, math, Counter, defaultdict.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
    )


def program_prompt(row: dict[str, Any], card: dict[str, str]) -> str:
    return (
        f"Reusable strategy: {card['name']}\n"
        f"{card['instruction']}\n\n"
        "Apply that reusable strategy only if it fits the example. If it does not fit, infer the most plausible general table transformation.\n\n"
        f"{common_program_context(row)}"
        "Python code:"
    )


def repair_prompt(row: dict[str, Any], card: dict[str, str], code: str, feedback: str, repair_round: int) -> str:
    return (
        "Repair the Python function below so it implements the table transformation generally. "
        "It failed on the visible example. Return a full replacement implementation of exactly one function named transform(table). "
        "Output only Python code, with no markdown, no explanation, and no comments. "
        "Do not include import statements; helper names are already available: re, math, Counter, defaultdict. "
        "Do not hard-code the visible input, visible output, new input, or specific row values.\n\n"
        f"Reusable strategy: {card['name']}\n{card['instruction']}\n"
        f"Repair round: {repair_round}\n"
        f"Visible input shape: rows={len(row['input_table'])}, columns={[len(r) for r in row['input_table'][:8]]}\n"
        f"Visible output shape: rows={len(row['output_table'])}, columns={[len(r) for r in row['output_table'][:8]]}\n"
        f"New input shape: rows={len(row['testing_table'])}, columns={[len(r) for r in row['testing_table'][:12]]}\n\n"
        f"Visible example input:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Expected visible output:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table for shape reference:\n{compact_table(row['testing_table'], 3500)}\n\n"
        f"Failure feedback:\n{feedback}\n\n"
        f"Previous code:\n{code[:6000]}\n\n"
        "Replacement Python code:"
    )


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


def evaluate_code(code_raw: str, row: dict[str, Any], variant: str, stage: str, exec_timeout: float, gen_stats: dict[str, Any]) -> dict[str, Any]:
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
        "input_tokens": gen_stats["input_tokens"],
        "output_tokens": gen_stats["output_tokens"],
        "total_tokens": gen_stats["total_tokens"],
    }


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


def generate_candidate(tokenizer, model, row: dict[str, Any], card: dict[str, str], args) -> dict[str, Any]:
    raw = generate_one(
        tokenizer,
        model,
        "You write executable Python table transformers. Output only Python code.",
        program_prompt(row, card),
        args.max_code_tokens,
    )
    variant = card["id"]
    attempts = [evaluate_code(raw["text"], row, variant, "initial", args.exec_timeout, raw)]
    current = attempts[-1]
    for repair_idx in range(1, args.max_repairs + 1):
        if current["visible_pass"]:
            break
        feedback = failure_feedback(row, current)
        repaired_raw = generate_one(
            tokenizer,
            model,
            "You repair executable Python table transformers. Output only Python code.",
            repair_prompt(row, card, current.get("code") or current.get("code_raw") or "", feedback, repair_idx),
            args.max_code_tokens,
        )
        current = evaluate_code(repaired_raw["text"], row, variant, f"repair_{repair_idx}", args.exec_timeout, repaired_raw)
        current["feedback_used"] = feedback
        attempts.append(current)
    final = next((attempt for attempt in attempts if attempt["visible_pass"]), attempts[-1])
    return {
        "variant": variant,
        "name": card["name"],
        "instruction": card["instruction"],
        "attempts": attempts,
        "final": final,
        "total_tokens": sum(attempt["total_tokens"] for attempt in attempts),
    }


def evaluate_row(tokenizer, model, row: dict[str, Any], cards: list[dict[str, str]], args) -> dict[str, Any]:
    candidates = [generate_candidate(tokenizer, model, row, card, args) for card in cards]
    return {
        "file": row["file"],
        "family": row["family"],
        "num_samples": row["num_samples"],
        "target_table": row["test_answer"],
        "program_candidates": candidates,
    }


def visible_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in record["program_candidates"]
        if candidate["final"]["visible_pass"] and candidate["final"].get("hidden_output_key") is not None
    ]


def first_visible_ok(record: dict[str, Any], direct_ok: bool) -> tuple[str, bool]:
    visible = visible_candidates(record)
    if visible:
        candidate = visible[0]
        return f"program:{candidate['variant']}", bool(candidate["final"]["hidden_exact"])
    return "direct", direct_ok


def visible_out_less_cols(record: dict[str, Any], cases: dict[str, dict[str, Any]]) -> bool:
    row = cases[record["file"]]
    in_cols = max([len(r) for r in row["input_table"]] or [0])
    out_cols = max([len(r) for r in row["output_table"]] or [0])
    return out_cols < in_cols


def summarize(records: list[dict[str, Any]], baseline_by_file: dict[str, dict[str, Any]], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    direct_exact = 0
    discovered_exact = 0
    discovered_commits = 0
    discovered_commit_correct = 0
    discovered_recoveries = 0
    discovered_losses = 0
    triggered_exact = 0
    triggered_commits = 0
    triggered_commit_correct = 0
    triggered_recoveries = 0
    triggered_losses = 0
    discovered_oracle = 0
    baseline_first = 0
    baseline_oracle = 0
    new_over_baseline_oracle = 0
    total_tokens = 0
    variant_rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"visible": 0, "hidden": 0, "tokens": 0})
    by_family = defaultdict(lambda: {"n": 0, "direct": 0, "discovered": 0, "triggered": 0, "baseline_first": 0, "discovered_oracle": 0})
    for record in records:
        baseline = baseline_by_file[record["file"]]
        direct_ok = bool(baseline["direct_exact"])
        source, ok = first_visible_ok(record, direct_ok)
        if visible_out_less_cols(record, cases):
            triggered_source, triggered_ok = first_visible_ok(record, direct_ok)
        else:
            triggered_source, triggered_ok = "direct", direct_ok
        visible_correct = any(candidate["final"]["visible_verified_hidden_exact"] for candidate in record["program_candidates"])
        base_source, base_ok = baseline_first_visible_ok(baseline)
        base_oracle = bool(direct_ok or any(c["final"]["visible_verified_hidden_exact"] for c in baseline["program_candidates"]))
        direct_exact += int(direct_ok)
        discovered_exact += int(ok)
        triggered_exact += int(triggered_ok)
        baseline_first += int(base_ok)
        baseline_oracle += int(base_oracle)
        discovered_oracle += int(direct_ok or visible_correct)
        new_over_baseline_oracle += int((not base_oracle) and visible_correct)
        if source.startswith("program"):
            discovered_commits += 1
            discovered_commit_correct += int(ok)
            discovered_recoveries += int((not direct_ok) and ok)
            discovered_losses += int(direct_ok and not ok)
        if triggered_source.startswith("program"):
            triggered_commits += 1
            triggered_commit_correct += int(triggered_ok)
            triggered_recoveries += int((not direct_ok) and triggered_ok)
            triggered_losses += int(direct_ok and not triggered_ok)
        total_tokens += baseline.get("direct_total_tokens", 0) + sum(c.get("total_tokens", 0) for c in record["program_candidates"])
        fam = by_family[record["family"]]
        fam["n"] += 1
        fam["direct"] += int(direct_ok)
        fam["discovered"] += int(ok)
        fam["triggered"] += int(triggered_ok)
        fam["baseline_first"] += int(base_ok)
        fam["discovered_oracle"] += int(direct_ok or visible_correct)
        for candidate in record["program_candidates"]:
            row = variant_rows[candidate["variant"]]
            row["visible"] += int(candidate["final"]["visible_pass"])
            row["hidden"] += int(candidate["final"]["hidden_exact"])
            row["tokens"] += int(candidate["total_tokens"])
    return {
        "n": n,
        "direct_exact": direct_exact,
        "direct_accuracy": direct_exact / n if n else 0,
        "discovered_first_visible_exact": discovered_exact,
        "discovered_first_visible_accuracy": discovered_exact / n if n else 0,
        "discovered_program_commits": discovered_commits,
        "discovered_commit_correct": discovered_commit_correct,
        "discovered_commit_precision": discovered_commit_correct / discovered_commits if discovered_commits else None,
        "discovered_direct_miss_recoveries": discovered_recoveries,
        "discovered_direct_correct_losses": discovered_losses,
        "discovered_shape_trigger_exact": triggered_exact,
        "discovered_shape_trigger_accuracy": triggered_exact / n if n else 0,
        "discovered_shape_trigger_program_commits": triggered_commits,
        "discovered_shape_trigger_commit_correct": triggered_commit_correct,
        "discovered_shape_trigger_commit_precision": triggered_commit_correct / triggered_commits if triggered_commits else None,
        "discovered_shape_trigger_recoveries": triggered_recoveries,
        "discovered_shape_trigger_losses": triggered_losses,
        "discovered_oracle_union": discovered_oracle,
        "discovered_oracle_union_accuracy": discovered_oracle / n if n else 0,
        "baseline_first_visible_exact": baseline_first,
        "baseline_first_visible_accuracy": baseline_first / n if n else 0,
        "baseline_oracle_union": baseline_oracle,
        "baseline_oracle_union_accuracy": baseline_oracle / n if n else 0,
        "new_visible_correct_over_baseline_oracle": new_over_baseline_oracle,
        "total_forward_tokens": total_tokens,
        "avg_forward_tokens": total_tokens / n if n else 0,
        "variant_summary": dict(sorted(variant_rows.items())),
        "by_family": {
            family: {
                **row,
                "direct_accuracy": row["direct"] / row["n"],
                "discovered_accuracy": row["discovered"] / row["n"],
                "triggered_accuracy": row["triggered"] / row["n"],
                "baseline_first_accuracy": row["baseline_first"] / row["n"],
                "discovered_oracle_accuracy": row["discovered_oracle"] / row["n"],
            }
            for family, row in sorted(by_family.items())
        },
    }


def baseline_first_visible_ok(record: dict[str, Any]) -> tuple[str, bool]:
    visible = [
        candidate
        for candidate in record["program_candidates"]
        if candidate["final"]["visible_pass"] and candidate["final"].get("hidden_output_key") is not None
    ]
    if visible:
        candidate = visible[0]
        return f"baseline:{candidate['variant']}", bool(candidate["final"]["hidden_exact"])
    return "direct", bool(record["direct_exact"])


def save_charts(root: Path, summary: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = ["Direct", "Discovered first-visible", "Discovered shape-trigger", "Discovered oracle", "Baseline first-visible", "Baseline oracle"]
    values = [
        summary["direct_accuracy"],
        summary["discovered_first_visible_accuracy"],
        summary["discovered_shape_trigger_accuracy"],
        summary["discovered_oracle_union_accuracy"],
        summary["baseline_first_visible_accuracy"],
        summary["baseline_oracle_union_accuracy"],
    ]
    ax.bar(labels, values, color=["tab:gray", "tab:blue", "tab:green", "tab:cyan", "tab:orange", "tab:red"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Held-out exact accuracy")
    ax.set_title("Discovered strategies vs included baselines")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "reports/figures/accuracy_bars.png", dpi=180)
    plt.close(fig)

    variants = summary["variant_summary"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = list(variants)
    visible = [variants[v]["visible"] / summary["n"] for v in labels]
    hidden = [variants[v]["hidden"] / summary["n"] for v in labels]
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], visible, width, label="Visible pass")
    ax.bar([i + width / 2 for i in x], hidden, width, label="Hidden exact")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Discovered strategy candidate quality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "reports/figures/strategy_quality.png", dpi=180)
    plt.close(fig)

    fam = summary["by_family"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    labels = list(fam)
    direct = [fam[f]["direct_accuracy"] for f in labels]
    discovered = [fam[f]["discovered_accuracy"] for f in labels]
    triggered = [fam[f]["triggered_accuracy"] for f in labels]
    baseline = [fam[f]["baseline_first_accuracy"] for f in labels]
    x = list(range(len(labels)))
    width = 0.2
    ax.bar([i - 1.5 * width for i in x], direct, width, label="Direct")
    ax.bar([i - 0.5 * width for i in x], discovered, width, label="Discovered")
    ax.bar([i + 0.5 * width for i in x], triggered, width, label="Shape-triggered")
    ax.bar([i + 1.5 * width for i in x], baseline, width, label="Baseline")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out accuracy by family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "reports/figures/family_accuracy.png", dpi=180)
    plt.close(fig)


def make_report(root: Path, cards: list[dict[str, str]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Live Strategy Discovery",
        "",
        "## Summary",
        "",
        "This experiment uses Qwen3.5-4B to propose reusable program-generation strategy prompts from calibration examples, freezes those strategies, and evaluates fresh executable-program generations on held-out Foofah-style table transformations.",
        "",
        "## Discovered Strategy Cards",
        "",
    ]
    for card in cards:
        lines += [
            f"### `{card['id']}`",
            "",
            f"Name: {card['name']}",
            "",
            card["instruction"],
            "",
        ]
    lines += [
        "## Held-Out Result",
        "",
        "| Policy | Exact | Accuracy | Tokens | Recoveries | Losses | Commit precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Direct JSON | {summary['direct_exact']}/{summary['n']} | {pct(summary['direct_accuracy'])} | included | 0 | 0 | n/a |",
        f"| Discovered first-visible | {summary['discovered_first_visible_exact']}/{summary['n']} | {pct(summary['discovered_first_visible_accuracy'])} | {summary['total_forward_tokens']:,} | {summary['discovered_direct_miss_recoveries']} | {summary['discovered_direct_correct_losses']} | {pct(summary['discovered_commit_precision'])} |",
        f"| Discovered shape-triggered | {summary['discovered_shape_trigger_exact']}/{summary['n']} | {pct(summary['discovered_shape_trigger_accuracy'])} | {summary['total_forward_tokens']:,} | {summary['discovered_shape_trigger_recoveries']} | {summary['discovered_shape_trigger_losses']} | {pct(summary['discovered_shape_trigger_commit_precision'])} |",
        f"| Discovered oracle union | {summary['discovered_oracle_union']}/{summary['n']} | {pct(summary['discovered_oracle_union_accuracy'])} | {summary['total_forward_tokens']:,} | n/a | n/a | n/a |",
        f"| Included baseline first-visible | {summary['baseline_first_visible_exact']}/{summary['n']} | {pct(summary['baseline_first_visible_accuracy'])} | included | n/a | n/a | n/a |",
        f"| Included baseline oracle union | {summary['baseline_oracle_union']}/{summary['n']} | {pct(summary['baseline_oracle_union_accuracy'])} | included | n/a | n/a | n/a |",
        "",
        f"New visible-correct tasks over the included baseline oracle: `{summary['new_visible_correct_over_baseline_oracle']}`.",
        "",
        "## Strategy Quality",
        "",
        "| Strategy | Visible pass | Hidden exact | Tokens |",
        "|---|---:|---:|---:|",
    ]
    for variant, row in summary["variant_summary"].items():
        lines.append(f"| `{variant}` | {row['visible']}/{summary['n']} | {row['hidden']}/{summary['n']} | {row['tokens']:,} |")
    lines += [
        "",
        "## Family Breakdown",
        "",
        "| Family | n | Direct | Discovered | Shape-triggered | Baseline first-visible | Discovered oracle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in summary["by_family"].items():
        lines.append(
            f"| `{family}` | {row['n']} | {row['direct']}/{row['n']} | {row['discovered']}/{row['n']} | "
            f"{row['triggered']}/{row['n']} | {row['baseline_first']}/{row['n']} | {row['discovered_oracle']}/{row['n']} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "![Accuracy](figures/accuracy_bars.png)",
        "",
        "![Strategy quality](figures/strategy_quality.png)",
        "",
        "![Family accuracy](figures/family_accuracy.png)",
        "",
        "## Interpretation",
        "",
        "The decisive question is whether discovered strategies add held-out recoveries that are not already available to the included baseline pool. The first-visible row measures deployable commitment if any discovered program passes the public example; the shape-triggered row avoids committing discovered programs outside a simple public column-contraction trigger; the oracle row measures coverage if selection were perfect.",
        "",
        "## Limitations",
        "",
        f"- Evaluation used `max_discovered={args.max_discovered}`, `max_repairs={args.max_repairs}`, and `limit_test={args.limit_test or 'all'}`.",
        "- Direct and included baseline metrics are read from local records packaged with this experiment; discovered strategy programs are freshly generated in this run.",
        "- This is still a small held-out benchmark and should be repeated across additional family splits before any strategy card is treated as robust.",
    ]
    write_text(root / "reports/report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--max-discovered", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-code-tokens", type=int, default=640)
    parser.add_argument("--exec-timeout", type=float, default=2.0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    root = args.root
    records_path = root / "reports/discovered_test_records.jsonl"
    cards_path = root / "reports/discovered_strategy_cards.json"
    if args.overwrite:
        for path in [records_path, cards_path, root / "reports/final_summary.json", root / "reports/report.md"]:
            if path.exists():
                path.unlink()
    cases = {row["file"]: row for row in load_jsonl(root / "data/cases.jsonl")}
    calibration_records = load_jsonl(root / "data/calibration_baseline_records.jsonl")
    baseline_records = load_jsonl(root / "data/test_baseline_records.jsonl")
    if args.limit_test:
        baseline_records = baseline_records[: args.limit_test]
    baseline_by_file = {record["file"]: record for record in baseline_records}

    tokenizer, model = load_model(args.load_in_4bit)
    cards = discover_strategy_cards(tokenizer, model, calibration_records, cases, args.max_discovered, cards_path)

    existing = []
    completed = set()
    if args.resume and records_path.exists():
        existing = load_jsonl(records_path)
        completed = {record["file"] for record in existing}
    generated = list(existing)
    remaining = [record for record in baseline_records if record["file"] not in completed]
    for idx, baseline in enumerate(remaining, 1):
        row = cases[baseline["file"]]
        record = evaluate_row(tokenizer, model, row, cards, args)
        generated.append(record)
        append_jsonl(records_path, record)
        if args.progress_every and (idx % args.progress_every == 0 or idx == len(remaining)):
            partial = summarize(generated, baseline_by_file, cases)
            print(
                f"[{idx}/{len(remaining)} new, {partial['n']} total] "
                f"discovered={partial['discovered_first_visible_exact']}/{partial['n']} "
                f"direct={partial['direct_exact']} baseline={partial['baseline_first_visible_exact']} "
                f"new_over_baseline_oracle={partial['new_visible_correct_over_baseline_oracle']} "
                f"tokens={partial['total_forward_tokens']}",
                flush=True,
            )

    summary = summarize(generated, baseline_by_file, cases)
    write_json(root / "reports/final_summary.json", {"cards": cards, "summary": summary})
    save_charts(root, summary)
    make_report(root, cards, summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

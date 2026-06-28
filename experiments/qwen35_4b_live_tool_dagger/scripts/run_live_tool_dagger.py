#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
MODEL_PATH = "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

sys.path.insert(0, str(ROOT_DEFAULT))
from src.foofah import compact_table, diff_tables, equal_table, extract_code, extract_json_table, repair_prompt  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table_dims(table: Any) -> tuple[int, int, int, int]:
    if not isinstance(table, list):
        return 0, 0, 0, 0
    rows = len(table)
    cols = max([len(row) for row in table if isinstance(row, list)] or [0])
    cells = sum(len(row) for row in table if isinstance(row, list))
    chars = sum(len(str(cell)) for row in table if isinstance(row, list) for cell in row)
    return rows, cols, cells, chars


def direct_prompt(row: dict[str, Any]) -> str:
    return row["direct_prompt"] if "direct_prompt" in row else (
        "You are given one or more examples of a table transformation and a new input table. "
        "Return only the transformed table as valid JSON, with no explanation, no markdown, and no prose. "
        "The first character of your answer must be '[' and the last character must be ']'. "
        "The table must be a JSON array of rows, where each row is an array of strings.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
        "Transformed output table:"
    )


def initial_program_prompt(row: dict[str, Any]) -> str:
    return row["initial_program_prompt"] if "initial_program_prompt" in row else (
        "Write a Python function that performs the table transformation shown by the example. "
        "Output only Python code, with no markdown and no explanation. "
        "Define exactly one function named transform(table). "
        "The function must return a list of rows, where each row is a list of strings. "
        "Do not read files, do not print, and do not use input(). "
        "Do not include import statements; helper names are already available as globals: re, math, Counter, defaultdict.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        "The function will be executed on this new input table after it is verified on the example. "
        "Use the new table only to understand the shape of the data; still write a general transform(table) function.\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
        "Python code:"
    )


def add_prompts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in cases:
        row = dict(row)
        row["direct_prompt"] = direct_prompt(row)
        row["initial_program_prompt"] = initial_program_prompt(row)
        out.append(row)
    return out


def prior_program_only(row: dict[str, Any]) -> bool:
    return bool(row.get("final_visible_pass") and row.get("final_hidden_exact") and not row.get("direct_exact"))


def select_balanced_cases(root: Path, cases_per_family: int, limit_total: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = add_prompts(load_jsonl(root / "data" / "cases.jsonl"))
    by_file = {row["file"]: row for row in cases}
    prior = load_jsonl(root / "data" / "prior_trace_index_for_split.jsonl")
    fams = sorted({row["family"] for row in cases})
    fam_score = defaultdict(int)
    for row in prior:
        fam_score[row["family"]] += int(prior_program_only(row))
    positive = [fam for fam in fams if fam_score[fam] > 0]
    neutral = [fam for fam in fams if fam_score[fam] == 0]
    positive.sort(key=lambda f: (-fam_score[f], f))
    neutral.sort()
    # Keep every split seeded with families that previously exposed program-only headroom.
    # Round-robin assignment prevents the held-out test split from becoming a zero-headroom slice.
    positive_bins = {"train": [], "dev": [], "test": []}
    cycle = ["train", "dev", "test"]
    for idx, fam in enumerate(positive):
        positive_bins[cycle[idx % len(cycle)]].append(fam)
    neutral_iter = iter(neutral)

    def fill(split: str, target: int) -> list[str]:
        fams = list(positive_bins[split])
        while len(fams) < target:
            try:
                fams.append(next(neutral_iter))
            except StopIteration:
                break
        return fams

    train_fams = fill("train", 12)
    dev_fams = fill("dev", 4)
    test_fams = fill("test", 4)
    selected_fams = {"train": train_fams, "dev": dev_fams, "test": test_fams}
    rng = random.Random(seed)
    selected = []
    split_rows = {}
    for split, families in selected_fams.items():
        rows = []
        for fam in families:
            fam_rows = sorted([row for row in cases if row["family"] == fam], key=lambda r: r["file"])
            # Prefer files that were informative in the prior index, but still generate fresh traces.
            prior_files = {
                row["file"] for row in prior
                if row["family"] == fam and (prior_program_only(row) or row.get("direct_exact"))
            }
            ordered = [row for row in fam_rows if row["file"] in prior_files] + [row for row in fam_rows if row["file"] not in prior_files]
            rows.extend(ordered[:cases_per_family])
        split_rows[split] = rows
        selected.extend([{**row, "split": split} for row in rows])
    if limit_total:
        interleaved = []
        split_tagged = {
            split: [{**row, "split": split} for row in rows]
            for split, rows in split_rows.items()
        }
        max_len = max((len(rows) for rows in split_tagged.values()), default=0)
        for idx in range(max_len):
            for split in ["train", "dev", "test"]:
                if idx < len(split_tagged.get(split, [])):
                    interleaved.append(split_tagged[split][idx])
        selected = interleaved[:limit_total]
    meta = {
        "cases_per_family": cases_per_family,
        "limit_total": limit_total,
        "family_program_only_prior_counts": dict(sorted(fam_score.items())),
        "split_families": selected_fams,
        "selected_counts": {split: len(rows) for split, rows in split_rows.items()},
    }
    return selected, meta


def load_model(load_in_4bit: bool):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

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


def render_chat(tokenizer: Any, system: str, prompt: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one(tokenizer: Any, model: Any, system: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    text = render_chat(tokenizer, system, prompt)
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


def run_generated_code(root: Path, code: str, table: list[list[str]], timeout_s: float) -> dict[str, Any]:
    payload = json.dumps({"code": code, "table": table}, ensure_ascii=False)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(root / "src" / "safe_exec.py")],
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


def evaluate_code_round(root: Path, code_raw: str, row: dict[str, Any], round_index: int, exec_timeout: float) -> dict[str, Any]:
    code_found, code, code_parse_reason = extract_code(code_raw)
    visible_exec = {"ok": False, "error": "CodeParse", "message": code_parse_reason}
    hidden_exec = {"ok": False, "error": "Skipped", "message": "visible_exec_failed_or_code_missing"}
    visible_output = None
    hidden_output = None
    visible_pass = False
    hidden_exact = False
    if code_found:
        visible_exec = run_generated_code(root, code, row["input_table"], exec_timeout)
        if visible_exec.get("ok"):
            visible_output = visible_exec["result"]
            visible_pass = equal_table(visible_output, row["output_table"])
            hidden_exec = run_generated_code(root, code, row["testing_table"], exec_timeout)
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


def failure_feedback(row: dict[str, Any], round_record: dict[str, Any]) -> str:
    if not round_record["code_found"]:
        return "No valid transform(table) function was found. Return only a complete function named transform(table)."
    if not round_record["visible_exec_ok"]:
        return (
            "The function raised an error on the visible example.\n"
            f"Error type: {round_record.get('visible_exec_error')}\n"
            f"Error message: {round_record.get('visible_exec_message')}"
        )
    return (
        "The function returned the wrong visible table.\n"
        f"{diff_tables(row['output_table'], round_record.get('visible_output'))}\n"
        f"Actual visible output:\n{compact_table(round_record.get('visible_output'))}"
    )


def evaluate_record(root: Path, tokenizer: Any, model: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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
    current = evaluate_code_round(root, initial_raw, row, 0, args.exec_timeout)
    rounds.append(current)
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
        current = evaluate_code_round(root, raw, row, repair_idx, args.exec_timeout)
        current["feedback_used"] = feedback
        rounds.append(current)
    first_visible = next((rr for rr in rounds if rr["visible_pass"]), None)
    final = first_visible or rounds[-1]
    direct_program_agreement = bool(
        direct_parse_ok and final.get("hidden_output") is not None and equal_table(direct_table, final["hidden_output"])
    )
    return {
        "file": row["file"],
        "family": row["family"],
        "split": row["split"],
        "num_samples": row.get("num_samples", 1),
        "direct_raw": direct_raw,
        "direct_parse_ok": direct_parse_ok,
        "direct_fragment": direct_fragment,
        "direct_table": direct_table,
        "direct_exact": direct_exact,
        "rounds": rounds,
        "round_count": len(rounds),
        "final_round": final["round"],
        "final_visible_pass": final["visible_pass"],
        "final_hidden_exact": bool(final["visible_pass"] and final["hidden_exact"]),
        "final_hidden_output": final.get("hidden_output"),
        "direct_program_agreement": direct_program_agreement,
        "target_table": row["test_answer"],
    }


def trace_path(root: Path, tag: str) -> Path:
    return root / "reports" / (f"fresh_traces_{tag}.jsonl" if tag else "fresh_traces.jsonl")


def generate_traces(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    selected, meta = select_balanced_cases(root, args.cases_per_family, args.limit_total, args.split_seed)
    write_json(root / "reports" / "split_manifest.json", meta)
    path = trace_path(root, args.tag)
    if path.exists() and not args.resume:
        path.unlink()
    done = {row["file"] for row in load_jsonl(path)} if args.resume else set()
    tokenizer, model = load_model(args.load_in_4bit)
    for idx, row in enumerate(selected, start=1):
        if row["file"] in done:
            continue
        record = evaluate_record(root, tokenizer, model, row, args)
        append_jsonl(path, record)
        if idx % args.progress_every == 0:
            print(
                f"generated {idx}/{len(selected)} split={row['split']} "
                f"direct={int(record['direct_exact'])} program={int(record['final_hidden_exact'])} "
                f"visible={int(record['final_visible_pass'])} rounds={record['round_count']}",
                flush=True,
            )
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return load_jsonl(path)


def final_program_exact(record: dict[str, Any]) -> bool:
    return bool(record.get("final_visible_pass") and record.get("final_hidden_exact"))


def valid_actions(state: dict[str, Any]) -> list[str]:
    if state["stage"] == "start":
        return ["DIRECT", "WRITE"]
    rr = state["round"]
    if rr.get("visible_pass"):
        return ["DIRECT", "PROGRAM"]
    if state["round_index"] + 1 < len(state["record"].get("rounds", [])):
        return ["DIRECT", "FIX"]
    return ["DIRECT"]


def oracle_action_for_state(state: dict[str, Any]) -> str:
    record = state["record"]
    if state["stage"] == "start":
        if bool(record.get("direct_exact")):
            return "DIRECT"
        if any(rr.get("visible_pass") and rr.get("hidden_exact") for rr in record.get("rounds", [])):
            return "WRITE"
        return "DIRECT"
    rr = state["round"]
    if rr.get("visible_pass"):
        if rr.get("hidden_exact") and not bool(record.get("direct_exact")):
            return "PROGRAM"
        return "DIRECT"
    later = record.get("rounds", [])[state["round_index"] + 1 :]
    if any(x.get("visible_pass") and x.get("hidden_exact") for x in later) and not bool(record.get("direct_exact")):
        return "FIX"
    return "DIRECT"


def output_exact(record: dict[str, Any], action: str, round_index: int | None = None) -> bool:
    if action == "DIRECT":
        return bool(record.get("direct_exact"))
    if action == "PROGRAM" and round_index is not None:
        rr = record["rounds"][round_index]
        return bool(rr.get("visible_pass") and rr.get("hidden_exact"))
    return False


def state_features(state: dict[str, Any]) -> dict[str, Any]:
    record = state["record"]
    direct_r, direct_c, _, direct_chars = table_dims(record.get("direct_table"))
    target_r, target_c, _, target_chars = table_dims(record.get("target_table"))
    features = {
        "stage_start": state["stage"] == "start",
        "stage_round": state["stage"] == "round",
        "direct_parse_ok": bool(record.get("direct_parse_ok")),
        "direct_output_empty": direct_r == 0 or direct_c == 0,
        "direct_program_agreement": bool(record.get("direct_program_agreement")),
        "direct_shape_matches_visible": (direct_r, direct_c) == (target_r, target_c),
        "direct_char_close_visible": abs(direct_chars - target_chars) <= 5,
    }
    if state["stage"] == "round":
        rr = state["round"]
        out_r, out_c, _, out_chars = table_dims(rr.get("hidden_output"))
        vis_r, vis_c, _, vis_chars = table_dims(rr.get("visible_output"))
        features.update(
            {
                "visible_pass": bool(rr.get("visible_pass")),
                "visible_exec_ok": bool(rr.get("visible_exec_ok")),
                "round_zero": state["round_index"] == 0,
                "has_next_round": state["round_index"] + 1 < len(record.get("rounds", [])),
                "program_direct_same_shape": (out_r, out_c) == (direct_r, direct_c),
                "program_visible_shape_matches": (vis_r, vis_c) == (target_r, target_c),
                "program_hidden_shape_matches_direct": (out_r, out_c) == (direct_r, direct_c),
                "program_disagrees_direct": not bool(record.get("direct_program_agreement")),
                "program_char_close_direct": abs(out_chars - direct_chars) <= 5,
            }
        )
    return features


def render_state(state: dict[str, Any]) -> str:
    record = state["record"]
    payload: dict[str, Any] = {
        "split": record.get("split"),
        "family": record.get("family"),
        "stage": state["stage"],
        "valid_actions": valid_actions(state),
        "direct_parse_ok": bool(record.get("direct_parse_ok")),
        "direct_shape": table_dims(record.get("direct_table"))[:2],
        "visible_target_shape": table_dims(record.get("target_table"))[:2],
    }
    if state["stage"] == "round":
        rr = state["round"]
        payload.update(
            {
                "round_index": state["round_index"],
                "visible_exec_ok": bool(rr.get("visible_exec_ok")),
                "visible_pass": bool(rr.get("visible_pass")),
                "program_new_output_shape": table_dims(rr.get("hidden_output"))[:2],
                "program_visible_output_shape": table_dims(rr.get("visible_output"))[:2],
                "direct_program_agreement": bool(record.get("direct_program_agreement")),
                "has_next_repair": state["round_index"] + 1 < len(record.get("rounds", [])),
            }
        )
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def policy_prompt(state: dict[str, Any]) -> str:
    return (
        "You are a controller for a table-transformation tool loop. Choose the next action.\n"
        "Actions: DIRECT, WRITE, FIX, PROGRAM.\n"
        "DIRECT commits the direct JSON answer. WRITE uses the first generated program. "
        "FIX uses one repair step. PROGRAM commits the current visible-passing program output.\n"
        "Choose only one action from valid_actions. Return exactly one action token.\n\n"
        f"TOOL_STATE: {render_state(state)}\n\nACTION:"
    )


def states_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    states = [{"stage": "start", "record": record, "round": None, "round_index": None}]
    for idx, rr in enumerate(record.get("rounds", [])):
        states.append({"stage": "round", "record": record, "round": rr, "round_index": idx})
    return states


def build_examples(records: list[dict[str, Any]], seed: int) -> list[dict[str, str]]:
    examples = []
    for record in records:
        for state in states_for_record(record):
            examples.append({"prompt": policy_prompt(state), "label": oracle_action_for_state(state), "file": record["file"]})
    by_label = defaultdict(list)
    for ex in examples:
        by_label[ex["label"]].append(ex)
    rng = random.Random(seed)
    max_count = max(len(v) for v in by_label.values()) if by_label else 0
    balanced = []
    for label, group in by_label.items():
        reps = max(1, math.ceil(max_count / len(group)))
        pool = (group * reps)[:max_count]
        balanced.extend(pool)
    rng.shuffle(balanced)
    return balanced


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = defaultdict(list)
    for row in records:
        out[row["split"]].append(row)
    return dict(out)


def simulate_policy(records: list[dict[str, Any]], chooser: Any, name: str, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    exact = commits_program = program_correct = recoveries = losses = generated_program = repairs = 0
    for record in records:
        actions_taken = []
        state = {"stage": "start", "record": record, "round": None, "round_index": None}
        action = chooser(state)
        actions_taken.append(action)
        if action == "DIRECT":
            ok = output_exact(record, "DIRECT")
            source = "DIRECT"
        else:
            generated_program += 1
            idx = 0
            ok = output_exact(record, "DIRECT")
            source = "DIRECT"
            while idx < len(record.get("rounds", [])):
                rr_state = {"stage": "round", "record": record, "round": record["rounds"][idx], "round_index": idx}
                action = chooser(rr_state)
                actions_taken.append(action)
                if action == "PROGRAM":
                    source = "PROGRAM"
                    ok = output_exact(record, "PROGRAM", idx)
                    break
                if action == "FIX" and idx + 1 < len(record["rounds"]):
                    repairs += 1
                    idx += 1
                    continue
                source = "DIRECT"
                ok = output_exact(record, "DIRECT")
                break
        exact += int(ok)
        if source == "PROGRAM":
            commits_program += 1
            program_correct += int(ok)
            recoveries += int(ok and not record.get("direct_exact"))
            losses += int((not ok) and record.get("direct_exact"))
        rows.append(
            {
                "policy": name,
                "split": split,
                "file": record["file"],
                "family": record["family"],
                "actions": actions_taken,
                "source": source,
                "correct": ok,
                "direct_exact": bool(record.get("direct_exact")),
                "program_exact": final_program_exact(record),
            }
        )
    n = len(records)
    result = {
        "policy": name,
        "split": split,
        "n": n,
        "exact": exact,
        "accuracy": exact / n if n else 0.0,
        "program_commits": commits_program,
        "program_correct": program_correct,
        "program_precision": program_correct / commits_program if commits_program else None,
        "direct_miss_recoveries": recoveries,
        "direct_correct_losses": losses,
        "program_generations": generated_program,
        "repair_actions": repairs,
    }
    return result, rows


def fixed_chooser(name: str):
    def choose(state: dict[str, Any]) -> str:
        if name == "direct_only":
            return "DIRECT"
        if name == "always_tool_visible":
            if state["stage"] == "start":
                return "WRITE"
            rr = state["round"]
            if rr.get("visible_pass"):
                return "PROGRAM"
            return "FIX" if "FIX" in valid_actions(state) else "DIRECT"
        if name == "visible_disagree_rule":
            if state["stage"] == "start":
                return "WRITE"
            rr = state["round"]
            if rr.get("visible_pass"):
                return "PROGRAM" if not state["record"].get("direct_program_agreement") else "DIRECT"
            return "FIX" if "FIX" in valid_actions(state) else "DIRECT"
        if name == "oracle_seq":
            return oracle_action_for_state(state)
        raise KeyError(name)
    return choose


def train_boolean_rules(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train_states = [s for r in records_by_split["train"] for s in states_for_record(r)]
    dev_records = records_by_split["dev"]
    bool_features = sorted({k for s in train_states for k, v in state_features(s).items() if isinstance(v, bool)})
    candidates = []
    for feature in bool_features:
        for action_true in ["DIRECT", "WRITE", "FIX", "PROGRAM"]:
            for action_false in ["DIRECT", "WRITE", "FIX", "PROGRAM"]:
                if action_true == action_false:
                    continue

                def chooser(state: dict[str, Any], ft=feature, at=action_true, af=action_false) -> str:
                    action = at if bool(state_features(state).get(ft)) else af
                    return action if action in valid_actions(state) else "DIRECT"

                result, _ = simulate_policy(dev_records, chooser, "rule", "dev")
                candidates.append({"feature": feature, "true": action_true, "false": action_false, "dev": result})
    candidates.sort(key=lambda c: (c["dev"]["exact"], -c["dev"]["direct_correct_losses"], c["dev"]["direct_miss_recoveries"]), reverse=True)
    selected = candidates[0] if candidates else {"feature": "none", "true": "DIRECT", "false": "DIRECT"}

    def selected_chooser(state: dict[str, Any]) -> str:
        if selected["feature"] == "none":
            return "DIRECT"
        action = selected["true"] if bool(state_features(state).get(selected["feature"])) else selected["false"]
        return action if action in valid_actions(state) else "DIRECT"

    results, preds = {}, []
    for split, records in records_by_split.items():
        result, rows = simulate_policy(records, selected_chooser, "learned_rule", split)
        results[split] = result
        preds.extend(rows)
    return {"selected": selected, "top_candidates": candidates[:20], "results": results, "predictions": preds}


def maybe_import_torch():
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    return torch, AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training, DataLoader


def render_policy_chat(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise controller. Return only one action token."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class ActionDataset:
    def __init__(self, tokenizer: Any, examples: list[dict[str, str]], max_length: int):
        self.items = []
        eos = tokenizer.eos_token or ""
        for ex in examples:
            prompt = render_policy_chat(tokenizer, ex["prompt"])
            full = prompt + " " + ex["label"] + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            tok = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
            ids = tok["input_ids"]
            if len(ids) <= len(prompt_ids):
                continue
            labels = [-100] * min(len(prompt_ids), len(ids)) + ids[min(len(prompt_ids), len(ids)) :]
            self.items.append({"input_ids": ids, "attention_mask": tok["attention_mask"], "labels": labels[: len(ids)]})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


def collate(tokenizer: Any, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
    import torch

    max_len = max(len(x["input_ids"]) for x in batch)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        n = max_len - len(item["input_ids"])
        out["input_ids"].append(item["input_ids"] + [pad] * n)
        out["attention_mask"].append(item["attention_mask"] + [0] * n)
        out["labels"].append(item["labels"] + [-100] * n)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def load_policy_model(args: argparse.Namespace):
    torch, AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training, DataLoader = maybe_import_torch()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
    )
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    return torch, tokenizer, model, DataLoader


def score_action(tokenizer: Any, model: Any, state: dict[str, Any]) -> str:
    import torch

    action_ids = {a: tokenizer(" " + a, add_special_tokens=False)["input_ids"][0] for a in ["DIRECT", "WRITE", "FIX", "PROGRAM"]}
    prompt = render_policy_chat(tokenizer, policy_prompt(state))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1]
    valid = valid_actions(state)
    return max(valid, key=lambda action: float(logits[action_ids[action]].detach().cpu()))


def train_lora_policy(args: argparse.Namespace, records_by_split: dict[str, list[dict[str, Any]]], name: str, shuffle: bool) -> dict[str, Any]:
    torch, tokenizer, model, DataLoader = load_policy_model(args)
    examples = build_examples(records_by_split["train"], args.split_seed)
    if shuffle:
        rng = random.Random(args.split_seed + 99)
        labels = [ex["label"] for ex in examples]
        rng.shuffle(labels)
        for ex, label in zip(examples, labels):
            ex["label"] = label
    dataset = ActionDataset(tokenizer, examples, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(tokenizer, b))
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses, step = [], 0
    opt.zero_grad(set_to_none=True)
    model.train()
    while step < args.max_steps:
        for i, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                losses.append(float(loss.detach().cpu()) * args.grad_accum)
                if step == 1 or step % max(1, args.max_steps // 5) == 0:
                    print(f"{name} step {step}/{args.max_steps} loss={losses[-1]:.4f}", flush=True)
                if step >= args.max_steps:
                    break
    model.eval()

    def chooser(state: dict[str, Any]) -> str:
        return score_action(tokenizer, model, state)

    results, predictions = {}, []
    for split, records in records_by_split.items():
        result, rows = simulate_policy(records, chooser, name, split)
        results[split] = result
        predictions.extend(rows)
    adapter_dir = Path(args.root) / "reports" / "adapters" / name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    write_jsonl(Path(args.root) / "reports" / "predictions" / f"{name}.jsonl", predictions)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"name": name, "shuffle": shuffle, "num_examples": len(examples), "num_tokenized": len(dataset), "losses": losses, "results": results, "adapter_dir": str(adapter_dir)}


def evaluate_all(root: Path, args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    records_by_split = split_records(records)
    policy_results, prediction_rows = {}, []
    for name in ["direct_only", "always_tool_visible", "visible_disagree_rule", "oracle_seq"]:
        results = {}
        for split, rows in records_by_split.items():
            result, preds = simulate_policy(rows, fixed_chooser(name), name, split)
            results[split] = result
            prediction_rows.extend(preds)
        policy_results[name] = results
    rule = train_boolean_rules(records_by_split)
    policy_results["learned_rule"] = rule["results"]
    write_jsonl(root / "reports" / "predictions" / "fixed_and_rule.jsonl", prediction_rows + rule["predictions"])
    summary = {
        "experiment": "qwen35_4b_live_tool_dagger",
        "trace_file": str(trace_path(root, args.tag)),
        "n_records": len(records),
        "splits": {
            split: {
                "n": len(rows),
                "families": sorted({r["family"] for r in rows}),
                "direct_exact": sum(r.get("direct_exact") for r in rows),
                "program_exact": sum(final_program_exact(r) for r in rows),
                "oracle_union": sum(bool(r.get("direct_exact")) or final_program_exact(r) for r in rows),
            }
            for split, rows in records_by_split.items()
        },
        "policy_results": policy_results,
        "learned_rule": {k: v for k, v in rule.items() if k != "predictions"},
    }
    if args.train_lora:
        lora = train_lora_policy(args, records_by_split, "lora_seq_policy", False)
        policy_results["lora_seq_policy"] = lora["results"]
        summary["lora"] = {k: v for k, v in lora.items() if k != "results"}
    if args.train_shuffled_lora:
        shuf = train_lora_policy(args, records_by_split, "lora_shuffled_seq", True)
        policy_results["lora_shuffled_seq"] = shuf["results"]
        summary["lora_shuffled"] = {k: v for k, v in shuf.items() if k != "results"}
    write_json(root / "reports" / "final_summary.json", summary)
    make_figures(root, summary)
    write_report(root, summary)
    return summary


def row_for_policy(name: str, result: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {result['exact']}/{result['n']} | {pct(result['accuracy'])} | "
        f"{result['program_generations']} | {result['repair_actions']} | {result['program_commits']} | "
        f"{result['direct_miss_recoveries']} | {result['direct_correct_losses']} | {pct(result['program_precision'])} |"
    )


def make_figures(root: Path, summary: dict[str, Any]) -> None:
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = [(name, res["test"]) for name, res in summary["policy_results"].items() if "test" in res]
    labels = [name for name, _ in rows]
    acc = [r["accuracy"] for _, r in rows]
    y = list(range(len(rows)))
    plt.figure(figsize=(10.5, max(5.5, 0.45 * len(rows))))
    plt.barh(y, acc)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Held-out exact accuracy")
    plt.title("Sequential tool controller accuracy")
    plt.xlim(0, min(1.0, max(acc + [0.1]) + 0.2))
    plt.tight_layout()
    plt.savefig(fig_dir / "heldout_accuracy.png", dpi=180)
    plt.close()

    rec = [r["direct_miss_recoveries"] for _, r in rows]
    loss = [r["direct_correct_losses"] for _, r in rows]
    plt.figure(figsize=(10.5, max(5.5, 0.45 * len(rows))))
    plt.barh([i - 0.18 for i in y], rec, height=0.36, label="Recoveries")
    plt.barh([i + 0.18 for i in y], loss, height=0.36, label="Losses")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Tasks")
    plt.title("Recoveries and losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "recoveries_losses.png", dpi=180)
    plt.close()

    gens = [r["program_generations"] for _, r in rows]
    repairs = [r["repair_actions"] for _, r in rows]
    plt.figure(figsize=(10.5, max(5.5, 0.45 * len(rows))))
    plt.barh([i - 0.18 for i in y], gens, height=0.36, label="Program generations")
    plt.barh([i + 0.18 for i in y], repairs, height=0.36, label="Repair actions")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Actions on held-out tasks")
    plt.title("Tool budget usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "tool_budget.png", dpi=180)
    plt.close()


def write_report(root: Path, summary: dict[str, Any]) -> None:
    test_order = ["direct_only", "always_tool_visible", "visible_disagree_rule", "learned_rule", "lora_seq_policy", "lora_shuffled_seq", "oracle_seq"]
    lines = [
        "# Live Tool-State DAgger-Style Controller",
        "",
        "## Summary",
        "",
        "This standalone experiment generates fresh tool-environment traces and trains/evaluates a sequential controller over visible tool state.",
        "",
        "The controller chooses among `DIRECT`, `WRITE`, `FIX`, and `PROGRAM`. It is evaluated by simulating those actions on the freshly generated traces.",
        "",
        "## Split And Trace Counts",
        "",
    ]
    for split, info in sorted(summary["splits"].items()):
        lines.append(
            f"- {split}: {info['n']} records, direct {info['direct_exact']}, program {info['program_exact']}, oracle union {info['oracle_union']}, families: {', '.join(info['families'])}"
        )
    lines += [
        "",
        "## Held-Out Test Result",
        "",
        "| Policy | Exact | Accuracy | Program gens | Repairs | Program commits | Recoveries | Losses | Program precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in test_order:
        if name in summary["policy_results"] and "test" in summary["policy_results"][name]:
            lines.append(row_for_policy(name, summary["policy_results"][name]["test"]))
    rule_test = summary["policy_results"].get("learned_rule", {}).get("test")
    lora_test = summary["policy_results"].get("lora_seq_policy", {}).get("test")
    shuf_test = summary["policy_results"].get("lora_shuffled_seq", {}).get("test")
    lines += ["", "## Gate Verdict", ""]
    if rule_test:
        lines.append(
            f"The best deployable rule reached {rule_test['exact']}/{rule_test['n']} with "
            f"{rule_test['direct_miss_recoveries']} recoveries and {rule_test['direct_correct_losses']} losses."
        )
    if lora_test:
        lines.append(
            f"The sequential LoRA policy reached {lora_test['exact']}/{lora_test['n']} with "
            f"{lora_test['direct_miss_recoveries']} recoveries and {lora_test['direct_correct_losses']} losses."
        )
    if shuf_test and lora_test:
        lines.append(
            f"The shuffled-label control reached {shuf_test['exact']}/{shuf_test['n']}, providing the label-noise control for the LoRA arm."
        )
    lines += [
        "",
        "## Learned Rule",
        "",
        "```json",
        json.dumps(summary["learned_rule"]["selected"], indent=2, sort_keys=True)[:4000],
        "```",
        "",
        "## Figures",
        "",
        "![Accuracy](figures/heldout_accuracy.png)",
        "",
        "![Recoveries and losses](figures/recoveries_losses.png)",
        "",
        "![Tool budget](figures/tool_budget.png)",
        "",
        "## Limitations",
        "",
        "- This is a balanced pilot split, not a full benchmark run.",
        "- The policy is evaluated on fresh precomputed traces; `WRITE` and `FIX` reveal the corresponding fresh generated tool outputs from those traces.",
        "- Hidden labels are used only for oracle labels and evaluation, not in policy state.",
    ]
    write_text(root / "reports" / "report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--generate-traces", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cases-per-family", type=int, default=3)
    parser.add_argument("--limit-total", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260628)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--max-direct-tokens", type=int, default=768)
    parser.add_argument("--max-code-tokens", type=int, default=768)
    parser.add_argument("--exec-timeout", type=float, default=1.5)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--progress-every", type=int, default=3)
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--train-shuffled-lora", action="store_true")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    args.root = str(args.root)
    root = Path(args.root)
    if args.generate_traces:
        records = generate_traces(root, args)
    else:
        records = load_jsonl(trace_path(root, args.tag))
    if not records:
        raise SystemExit("No trace records available; run with --generate-traces first.")
    summary = evaluate_all(root, args, records)
    print(json.dumps(summary, indent=2, sort_keys=True)[:12000])


if __name__ == "__main__":
    main()

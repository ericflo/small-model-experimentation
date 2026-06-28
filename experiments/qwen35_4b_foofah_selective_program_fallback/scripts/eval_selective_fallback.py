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

from src.probes import make_probe_inputs  # noqa: E402
from src.table_utils import direct_prompt_for_table, equal_table, extract_json_table  # noqa: E402

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


def get_final_code(candidate: dict[str, Any]) -> str | None:
    if not candidate.get("rounds"):
        return None
    idx = int(candidate.get("final_round", len(candidate["rounds"]) - 1))
    if idx < 0 or idx >= len(candidate["rounds"]):
        return None
    code = candidate["rounds"][idx].get("code")
    return code if isinstance(code, str) and code.strip() else None


def eligible_for_probes(candidate: dict[str, Any], mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "visible":
        return bool(candidate["final_visible_pass"])
    if mode == "visible_disagree":
        return bool(candidate["final_visible_pass"] and not candidate["direct_program_agreement"])
    if mode == "all":
        return True
    raise ValueError(f"unknown probe eligibility mode: {mode}")


def probe_case(
    case: dict[str, Any],
    candidate: dict[str, Any],
    tokenizer,
    model,
    args,
) -> list[dict[str, Any]]:
    code = get_final_code(candidate)
    if code is None:
        return []
    out = []
    for probe in make_probe_inputs(case, max_probes=args.max_probes):
        program_exec = run_generated_code(code, probe["table"], args.exec_timeout)
        program_output = program_exec.get("result") if program_exec.get("ok") else None
        direct_raw = None
        direct_parse_ok = False
        direct_table = None
        direct_fragment = None
        if args.model_probes:
            direct_raw = generate_one(
                tokenizer,
                model,
                "You transform tables exactly. Output only JSON. Do not explain.",
                direct_prompt_for_table(case, probe["table"]),
                args.max_probe_tokens,
            )
            direct_parse_ok, direct_table, direct_fragment = extract_json_table(direct_raw)
        comparable = bool(program_exec.get("ok") and direct_parse_ok and direct_table is not None)
        out.append(
            {
                "name": probe["name"],
                "input_table": probe["table"],
                "program_exec_ok": bool(program_exec.get("ok")),
                "program_exec_error": program_exec.get("error"),
                "program_output": program_output,
                "direct_raw": direct_raw,
                "direct_parse_ok": direct_parse_ok,
                "direct_table": direct_table,
                "direct_fragment": direct_fragment,
                "comparable": comparable,
                "direct_program_agree": bool(comparable and equal_table(program_output, direct_table)),
            }
        )
    return out


def summarize_probe_features(record: dict[str, Any]) -> dict[str, Any]:
    probes = record["probes"]
    comparable = [p for p in probes if p["comparable"]]
    agree = [p for p in comparable if p["direct_program_agree"]]
    program_ok = [p for p in probes if p["program_exec_ok"]]
    unique_program_outputs = {
        json.dumps(p.get("program_output"), sort_keys=True, ensure_ascii=False)
        for p in program_ok
    }
    return {
        "probe_count": len(probes),
        "probe_program_ok": len(program_ok),
        "probe_direct_parse_ok": sum(p["direct_parse_ok"] for p in probes),
        "probe_comparable": len(comparable),
        "probe_agree": len(agree),
        "probe_agree_rate": len(agree) / len(comparable) if comparable else None,
        "probe_program_unique_outputs": len(unique_program_outputs),
    }


def choose_policy(record: dict[str, Any], policy: str) -> tuple[str, bool]:
    c = record["candidate"]
    direct = bool(c["direct_exact"])
    program = bool(c["final_hidden_exact"])
    visible = bool(c["final_visible_pass"])
    agree = bool(c["direct_program_agreement"])
    direct_parse_ok = bool(c["direct_parse_ok"])
    support = record["probe_features"].get("probe_agree_rate")

    choose_program = False
    if policy == "direct":
        choose_program = False
    elif policy == "program_if_visible":
        choose_program = visible
    elif policy == "program_if_visible_disagree":
        choose_program = bool(visible and not agree)
    elif policy == "program_if_direct_parse_fail":
        choose_program = bool(visible and not direct_parse_ok)
    elif policy == "program_if_probe_support_050":
        choose_program = bool(visible and not agree and support is not None and support >= 0.50)
    elif policy == "program_if_probe_support_067":
        choose_program = bool(visible and not agree and support is not None and support >= 0.67)
    elif policy == "program_if_probe_support_100":
        choose_program = bool(visible and not agree and support is not None and support >= 1.00)
    elif policy == "visible_program_veto_probe_lt_050":
        choose_program = bool(visible and (support is None or support >= 0.50))
    elif policy == "visible_program_veto_probe_lt_067":
        choose_program = bool(visible and (support is None or support >= 0.67))
    else:
        raise ValueError(f"unknown policy: {policy}")

    return ("program" if choose_program else "direct", program if choose_program else direct)


POLICIES = [
    "direct",
    "program_if_visible",
    "program_if_visible_disagree",
    "program_if_direct_parse_fail",
    "program_if_probe_support_050",
    "program_if_probe_support_067",
    "program_if_probe_support_100",
    "visible_program_veto_probe_lt_050",
    "visible_program_veto_probe_lt_067",
]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    by_policy = {}
    for policy in POLICIES:
        exact = 0
        program_commits = 0
        program_correct = 0
        direct_miss_recoveries = 0
        direct_correct_losses = 0
        for record in records:
            source, ok = choose_policy(record, policy)
            exact += int(ok)
            c = record["candidate"]
            if source == "program":
                program_commits += 1
                program_correct += int(c["final_hidden_exact"])
                direct_miss_recoveries += int((not c["direct_exact"]) and c["final_hidden_exact"])
                direct_correct_losses += int(c["direct_exact"] and not c["final_hidden_exact"])
        by_policy[policy] = {
            "exact": exact,
            "accuracy": exact / n if n else 0.0,
            "program_commits": program_commits,
            "program_commit_correct": program_correct,
            "program_commit_precision": program_correct / program_commits if program_commits else None,
            "direct_miss_recoveries": direct_miss_recoveries,
            "direct_correct_losses": direct_correct_losses,
        }

    visible = [r for r in records if r["candidate"]["final_visible_pass"]]
    visible_disagree = [r for r in visible if not r["candidate"]["direct_program_agreement"]]
    probe_comparable = [r for r in records if r["probe_features"]["probe_comparable"]]
    return {
        "n": n,
        "direct_exact": sum(r["candidate"]["direct_exact"] for r in records),
        "final_program_exact": sum(r["candidate"]["final_hidden_exact"] for r in records),
        "final_visible_pass": len(visible),
        "visible_disagree": len(visible_disagree),
        "visible_disagree_program_correct": sum(r["candidate"]["final_hidden_exact"] for r in visible_disagree),
        "visible_disagree_direct_correct": sum(r["candidate"]["direct_exact"] for r in visible_disagree),
        "probe_records": sum(bool(r["probes"]) for r in records),
        "probe_comparable_records": len(probe_comparable),
        "mean_probe_agree_rate": (
            sum(r["probe_features"]["probe_agree_rate"] for r in probe_comparable if r["probe_features"]["probe_agree_rate"] is not None)
            / len(probe_comparable)
            if probe_comparable
            else None
        ),
        "policies": by_policy,
    }


def family_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["candidate"]["family"]].append(record)
    out = []
    for family, rows in sorted(grouped.items()):
        item = summarize(rows)
        item["family"] = family
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--candidates", type=Path, default=ROOT / "data" / "candidate_records.jsonl")
    parser.add_argument("--records-out", type=Path, default=ROOT / "reports" / "selective_records.jsonl")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports" / "selective_summary.json")
    parser.add_argument("--family-out", type=Path, default=ROOT / "reports" / "selective_family_summary.json")
    parser.add_argument("--focus", choices=["all", "visible", "visible_disagree"], default="all")
    parser.add_argument("--probe-eligible", choices=["none", "visible", "visible_disagree", "all"], default="visible")
    parser.add_argument("--model-probes", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-probes", type=int, default=3)
    parser.add_argument("--max-probe-tokens", type=int, default=768)
    parser.add_argument("--exec-timeout", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = {r["file"]: r for r in load_jsonl(args.cases)}
    candidates = load_jsonl(args.candidates)
    selected = []
    for idx, candidate in enumerate(candidates):
        if idx % args.stride != 0:
            continue
        if args.focus == "visible" and not candidate["final_visible_pass"]:
            continue
        if args.focus == "visible_disagree" and not (candidate["final_visible_pass"] and not candidate["direct_program_agreement"]):
            continue
        selected.append(candidate)
        if args.limit is not None and len(selected) >= args.limit:
            break

    if args.overwrite and args.records_out.exists():
        args.records_out.unlink()

    tokenizer = model = None
    if args.model_probes:
        tokenizer, model = load_model(args.load_in_4bit)

    records = []
    for i, candidate in enumerate(selected, start=1):
        case = cases[candidate["file"]]
        probes = []
        if eligible_for_probes(candidate, args.probe_eligible):
            probes = probe_case(case, candidate, tokenizer, model, args)
        record = {
            "file": candidate["file"],
            "case": case,
            "candidate": candidate,
            "probes": probes,
        }
        record["probe_features"] = summarize_probe_features(record)
        records.append(record)
        append_jsonl(args.records_out, record)
        if args.progress_every and (i % args.progress_every == 0 or i == len(selected)):
            partial = summarize(records)
            best = max(partial["policies"].items(), key=lambda kv: kv[1]["accuracy"])
            print(
                f"[{i}/{len(selected)}] best={best[0]} {best[1]['exact']}/{partial['n']} "
                f"probed={partial['probe_records']}",
                flush=True,
            )

    summary = summarize(records)
    write_json(args.summary_out, summary)
    write_json(args.family_out, family_summary(records))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


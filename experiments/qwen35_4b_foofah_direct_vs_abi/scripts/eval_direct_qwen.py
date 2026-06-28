#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.foofah import equal_table, extract_json_table  # noqa: E402


MODEL_PATH = "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


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


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You transform tables exactly. Output only JSON. Do not explain."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate_one(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    text = render_prompt(tokenizer, prompt)
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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "parse_ok": sum(r["parse_ok"] for r in group),
            "parse_rate": sum(r["parse_ok"] for r in group) / len(group),
            "exact": sum(r["exact"] for r in group),
            "exact_accuracy": sum(r["exact"] for r in group) / len(group),
        }

    families = sorted({r["family"] for r in records})
    samples = sorted({r["num_samples"] for r in records})
    return {
        "overall": metrics(records),
        "by_family": {fam: metrics([r for r in records if r["family"] == fam]) for fam in families},
        "by_num_samples": {str(n): metrics([r for r in records if r["num_samples"] == n]) for n in samples},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    tokenizer, model = load_model(args.load_in_4bit)
    records = []
    for idx, row in enumerate(cases):
        raw = generate_one(tokenizer, model, row["prompt"], args.max_new_tokens)
        parse_ok, table, parsed_fragment = extract_json_table(raw)
        exact = bool(parse_ok and table is not None and equal_table(table, row["test_answer"]))
        records.append(
            {
                "file": row["file"],
                "family": row["family"],
                "num_samples": row["num_samples"],
                "parse_ok": parse_ok,
                "exact": exact,
                "raw_generation": raw,
                "parsed_fragment": parsed_fragment,
                "predicted_table": table,
                "target_table": row["test_answer"],
            }
        )
        if (idx + 1) % args.progress_every == 0:
            print(f"evaluated {idx + 1}/{len(cases)}", flush=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    write_jsonl(ROOT / "reports" / f"direct_qwen_records{suffix}.jsonl", records)
    summary = summarize(records)
    write_json(ROOT / "reports" / f"direct_qwen_summary{suffix}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

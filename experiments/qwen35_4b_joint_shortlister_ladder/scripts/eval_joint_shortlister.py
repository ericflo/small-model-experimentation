#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.operator_library import build_operator_library  # noqa: E402
from src.prompts import pair_prompt  # noqa: E402
from src.search import designed_probe_cases, observation_summary  # noqa: E402


RECALL_KS = [1, 4, 8, 16, 32]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_records(records: list[dict[str, Any]], max_per_cell: int | None) -> list[dict[str, Any]]:
    if max_per_cell is None:
        return records
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: (row["library_size"], row["template"], row["id"])):
        key = (int(record["library_size"]), record["template"])
        if len(buckets[key]) < max_per_cell:
            buckets[key].append(record)
            selected.append(record)
    return selected


def parse_pair(text: str, library_size: int) -> tuple[int, int] | None:
    match = re.search(r"(\d{3})\s*,\s*(\d{3})", text)
    if not match:
        digits = "".join(char for char in text if char.isdigit())
        if len(digits) < 6:
            return None
        left = int(digits[:3])
        right = int(digits[3:6])
    else:
        left = int(match.group(1))
        right = int(match.group(2))
    if 0 <= left < library_size and 0 <= right < library_size:
        return left, right
    return None


def code_strings(library_size: int) -> list[str]:
    return [f"{code:03d}" for code in range(library_size)]


def constrained_pair_beams(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    library_size: int,
    beam_width: int,
    max_length: int,
) -> list[tuple[int, int]]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    digit_ids = {str(digit): tokenizer(str(digit), add_special_tokens=False)["input_ids"][0] for digit in range(10)}
    comma_ids = tokenizer(",", add_special_tokens=False)["input_ids"]
    if len(comma_ids) != 1:
        raise RuntimeError("comma is not a single token for this tokenizer")
    comma_id = comma_ids[0]
    id_to_digit = {value: key for key, value in digit_ids.items()}
    valid_codes = code_strings(library_size)

    def allowed_code_digit(prefix: str) -> list[int]:
        allowed_digits = sorted({code[len(prefix)] for code in valid_codes if code.startswith(prefix) and len(prefix) < 3})
        return [digit_ids[digit] for digit in allowed_digits]

    def allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated_ids = input_ids[prompt_len:].tolist()
        chars: list[str] = []
        for token_id in generated_ids:
            if token_id == comma_id:
                chars.append(",")
            elif token_id in id_to_digit:
                chars.append(id_to_digit[token_id])
            else:
                return []
        generated = "".join(chars)
        if len(generated) < 3:
            return allowed_code_digit(generated)
        if len(generated) == 3:
            return [comma_id] if generated in valid_codes else []
        if len(generated) < 7:
            right_prefix = generated[4:]
            return allowed_code_digit(right_prefix)
        return []

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=7,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            do_sample=False,
            prefix_allowed_tokens_fn=allowed,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for sequence in output:
        suffix = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
        pair = parse_pair(suffix, library_size)
        if pair is not None and pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    return pairs


def pair_hits(record: dict[str, Any], pairs: list[tuple[int, int]]) -> dict[str, Any]:
    left = int(record["target_left_code"])
    right = int(record["target_right_code"])
    out: dict[str, Any] = {
        "target_left_code": left,
        "target_right_code": right,
        "top1_pair": f"{pairs[0][0]:03d},{pairs[0][1]:03d}" if pairs else None,
        "top1_pair_hit": bool(pairs and pairs[0] == (left, right)),
        "returned_pairs": [f"{pair_left:03d},{pair_right:03d}" for pair_left, pair_right in pairs],
    }
    for k in RECALL_KS:
        subset = pairs[:k]
        out[f"pair_recall_at_{k}"] = (left, right) in subset
        out[f"left_recall_at_{k}"] = any(pair_left == left for pair_left, _pair_right in subset)
        out[f"right_recall_at_{k}"] = any(pair_right == right for _pair_left, pair_right in subset)
    return out


def evaluate_records(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    operators: list[Any],
    control: str,
    observation: str,
    shuffled_inventory: bool,
    beam_width: int,
    max_length: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    desc = f"{control}:{observation}"
    for index, record in enumerate(tqdm(records, desc=desc)):
        cases = None
        if observation == "designed":
            cases, _summary = designed_probe_cases(record, operators[: record["library_size"]], budget=len(record["visible"]))
        prompt = pair_prompt(
            record,
            operators[: record["library_size"]],
            cases=cases,
            shuffled_inventory=shuffled_inventory,
            shuffle_seed=10_001 + index,
        )
        pairs = constrained_pair_beams(
            model,
            tokenizer,
            prompt,
            library_size=int(record["library_size"]),
            beam_width=beam_width,
            max_length=max_length,
        )
        rows.append(
            {
                "id": record["id"],
                "control": control,
                "observation": observation,
                "library_size": int(record["library_size"]),
                "template": record["template"],
                "beam_width": beam_width,
                "returned_pair_count": len(pairs),
                **pair_hits(record, pairs),
            }
        )
    return rows


def metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(1 for row in rows if row.get(key))
    return {"successes": successes, "records": len(rows), "rate": successes / len(rows) if rows else 0.0}


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / len(rows) if rows else 0.0


def summarize_predictions(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, subset in sorted(groups.items()):
        data = dict(zip(keys, key_values))
        data["records"] = len(subset)
        data["returned_pair_count_avg"] = round(avg(subset, "returned_pair_count"), 3)
        for k in RECALL_KS:
            data[f"pair_recall_at_{k}"] = metric(subset, f"pair_recall_at_{k}")
            data[f"left_recall_at_{k}"] = metric(subset, f"left_recall_at_{k}")
            data[f"right_recall_at_{k}"] = metric(subset, f"right_recall_at_{k}")
        out.append(data)
    return out


def probe_rows(records: list[dict[str, Any]], operators: list[Any], *, limit_per_cell: int | None) -> list[dict[str, Any]]:
    selected = select_records(records, limit_per_cell)
    rows: list[dict[str, Any]] = []
    for record in tqdm(selected, desc="observation-diagnostic"):
        rows.append(observation_summary(record, operators[: record["library_size"]]))
    return rows


def summarize_observations(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, subset in sorted(groups.items()):
        data = dict(zip(keys, key_values))
        data.update(
            {
                "records": len(subset),
                "random_consistent_avg": round(avg(subset, "random_consistent_count"), 3),
                "designed_consistent_avg": round(avg(subset, "designed_consistent_count"), 3),
                "random_selected_hidden_all": metric(subset, "random_selected_hidden_all"),
                "designed_selected_hidden_all": metric(subset, "designed_selected_hidden_all"),
                "random_target_in_set": metric(subset, "random_target_in_set"),
                "designed_target_in_set": metric(subset, "designed_target_in_set"),
            }
        )
        out.append(data)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-records", type=Path, default=ROOT / "data" / "eval_records.jsonl")
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("/workspace/large_artifacts/qwen35_4b_joint_shortlister_ladder/models/joint_pair_lora"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "joint_shortlister_results.json")
    parser.add_argument("--max-records-per-cell", type=int, default=2)
    parser.add_argument("--probe-records-per-cell", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-designed-shuffled", action="store_true")
    args = parser.parse_args()
    global RECALL_KS
    RECALL_KS = [k for k in RECALL_KS if k <= args.beam_width]
    if not RECALL_KS:
        raise RuntimeError("beam width must be at least 1")

    records = select_records(load_jsonl(args.eval_records), args.max_records_per_cell)
    operators = build_operator_library(512)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.eval()

    prediction_rows: list[dict[str, Any]] = []
    if not args.skip_base:
        prediction_rows.extend(
            evaluate_records(
                model=base_model,
                tokenizer=tokenizer,
                records=records,
                operators=operators,
                control="base_model",
                observation="random",
                shuffled_inventory=False,
                beam_width=args.beam_width,
                max_length=args.max_length,
            )
        )

    trained_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    trained_model.eval()
    prediction_rows.extend(
        evaluate_records(
            model=trained_model,
            tokenizer=tokenizer,
            records=records,
            operators=operators,
            control="trained_model",
            observation="random",
            shuffled_inventory=False,
            beam_width=args.beam_width,
            max_length=args.max_length,
        )
    )
    prediction_rows.extend(
        evaluate_records(
            model=trained_model,
            tokenizer=tokenizer,
            records=records,
            operators=operators,
            control="trained_model",
            observation="designed",
            shuffled_inventory=False,
            beam_width=args.beam_width,
            max_length=args.max_length,
        )
    )
    prediction_rows.extend(
        evaluate_records(
            model=trained_model,
            tokenizer=tokenizer,
            records=records,
            operators=operators,
            control="trained_model_shuffled_inventory",
            observation="random",
            shuffled_inventory=True,
            beam_width=args.beam_width,
            max_length=args.max_length,
        )
    )
    if not args.skip_designed_shuffled:
        prediction_rows.extend(
            evaluate_records(
                model=trained_model,
                tokenizer=tokenizer,
                records=records,
                operators=operators,
                control="trained_model_shuffled_inventory",
                observation="designed",
                shuffled_inventory=True,
                beam_width=args.beam_width,
                max_length=args.max_length,
            )
        )

    observation_rows = probe_rows(load_jsonl(args.eval_records), operators, limit_per_cell=args.probe_records_per_cell)
    result = {
        "records": len(records),
        "beam_width": args.beam_width,
        "max_length": args.max_length,
        "recall_ks": RECALL_KS,
        "model_eval_max_records_per_cell": args.max_records_per_cell,
        "probe_records_per_cell": args.probe_records_per_cell,
        "prediction_summary_by_control": summarize_predictions(prediction_rows, ["control", "observation"]),
        "prediction_summary_by_ladder": summarize_predictions(prediction_rows, ["control", "observation", "library_size", "template"]),
        "prediction_rows": prediction_rows,
        "observation_summary_by_ladder": summarize_observations(observation_rows, ["library_size", "template"]),
        "observation_summary_overall": summarize_observations(observation_rows, []),
        "observation_rows": observation_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": result["records"],
                "beam_width": result["beam_width"],
                "prediction_summary_by_control": result["prediction_summary_by_control"],
                "observation_summary_overall": result["observation_summary_overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

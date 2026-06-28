#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
from src.prompts import slot_prompt  # noqa: E402
from src.search import designed_probe_summary, exhaustive_summary  # noqa: E402


BUDGET_TO_SLOT_K = {1024: 32, 4096: 64, 16384: 128}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def alias_from_digits(text: str) -> str | None:
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) < 3:
        return None
    value = int(digits[:3])
    if 0 <= value < 512:
        return f"op_{value:03d}"
    return None


def constrained_alias_beams(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    beam_width: int,
    max_length: int,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    digit_ids = {str(digit): tokenizer(str(digit), add_special_tokens=False)["input_ids"][0] for digit in range(10)}
    id_to_digit = {value: key for key, value in digit_ids.items()}

    def allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = input_ids[prompt_len:].tolist()
        if len(generated) == 0:
            return [digit_ids[str(digit)] for digit in range(6)]
        if len(generated) == 1:
            first = id_to_digit.get(generated[0])
            if first is None:
                return []
            if int(first) < 5:
                return [digit_ids[str(digit)] for digit in range(10)]
            return [digit_ids["0"], digit_ids["1"]]
        if len(generated) == 2:
            first = id_to_digit.get(generated[0])
            second = id_to_digit.get(generated[1])
            if first is None or second is None:
                return []
            if int(first) < 5:
                return [digit_ids[str(digit)] for digit in range(10)]
            if second == "0":
                return [digit_ids[str(digit)] for digit in range(10)]
            if second == "1":
                return [digit_ids["0"], digit_ids["1"]]
            return []
        return []

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=3,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            do_sample=False,
            prefix_allowed_tokens_fn=allowed,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    aliases: list[str] = []
    seen: set[str] = set()
    for sequence in output:
        suffix = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
        alias = alias_from_digits(suffix)
        if alias and alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    return aliases


def evaluate_control(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    operators: list[Any],
    control: str,
    shuffled_inventory: bool,
    beam_width: int,
    max_length: int,
) -> list[dict[str, Any]]:
    rows = []
    for idx, record in enumerate(tqdm(records, desc=control)):
        prompt_left = slot_prompt(record, operators, "LEFT", shuffled_inventory=shuffled_inventory, shuffle_seed=991 + idx)
        prompt_right = slot_prompt(record, operators, "RIGHT", shuffled_inventory=shuffled_inventory, shuffle_seed=1991 + idx)
        left_aliases = constrained_alias_beams(model, tokenizer, prompt_left, beam_width=beam_width, max_length=max_length)
        right_aliases = constrained_alias_beams(model, tokenizer, prompt_right, beam_width=beam_width, max_length=max_length)
        row = {
            "id": record["id"],
            "control": control,
            "template": record["template"],
            "target_left_alias": record["target_left_alias"],
            "target_right_alias": record["target_right_alias"],
            "left_top1": left_aliases[0] if left_aliases else None,
            "right_top1": right_aliases[0] if right_aliases else None,
            "left_beams": left_aliases,
            "right_beams": right_aliases,
            "left_top1_hit": bool(left_aliases and left_aliases[0] == record["target_left_alias"]),
            "right_top1_hit": bool(right_aliases and right_aliases[0] == record["target_right_alias"]),
            "pair_top1_hit": bool(
                left_aliases
                and right_aliases
                and left_aliases[0] == record["target_left_alias"]
                and right_aliases[0] == record["target_right_alias"]
            ),
        }
        for budget, slot_k in BUDGET_TO_SLOT_K.items():
            left_hit = record["target_left_alias"] in left_aliases[:slot_k]
            right_hit = record["target_right_alias"] in right_aliases[:slot_k]
            row[f"left_top{slot_k}_hit"] = left_hit
            row[f"right_top{slot_k}_hit"] = right_hit
            row[f"pair_budget{budget}_hit"] = left_hit and right_hit
        rows.append(row)
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
    out = []
    for key_values, subset in sorted(groups.items()):
        data = dict(zip(keys, key_values))
        data.update(
            {
                "records": len(subset),
                "left_top1_hit": metric(subset, "left_top1_hit"),
                "right_top1_hit": metric(subset, "right_top1_hit"),
                "pair_top1_hit": metric(subset, "pair_top1_hit"),
            }
        )
        for budget in BUDGET_TO_SLOT_K:
            data[f"pair_budget{budget}_hit"] = metric(subset, f"pair_budget{budget}_hit")
        out.append(data)
    return out


def probe_diagnostic(records: list[dict[str, Any]], operators: list[Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    selected = [record for record in records if record["template"] == "pair_compare_gate"]
    if limit:
        selected = selected[:limit]
    for record in tqdm(selected, desc="probe-design"):
        base = exhaustive_summary(record, operators)
        designed = designed_probe_summary(record, operators, budget=6)
        rows.append({"id": record["id"], "template": record["template"], **base, **designed})
    return rows


def summarize_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "random_visible_consistent_count": round(avg(rows, "visible_consistent_count"), 3),
        "designed_visible_consistent_count": round(avg(rows, "designed_visible_consistent_count"), 3),
        "random_selected_hidden_all": metric(rows, "selected_hidden_all"),
        "designed_selected_hidden_all": metric(rows, "designed_selected_hidden_all"),
        "random_target_in_visible": metric(rows, "target_in_visible"),
        "designed_target_in_visible": metric(rows, "designed_target_in_visible"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-records", type=Path, default=ROOT / "data" / "eval_records.jsonl")
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("/workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/qwen35_lora"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "shortlister_results.json")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--probe-limit", type=int, default=24)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.eval_records)
    if args.max_records:
        records = records[: args.max_records]
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
            evaluate_control(
                model=base_model,
                tokenizer=tokenizer,
                records=records,
                operators=operators,
                control="base_model",
                shuffled_inventory=False,
                beam_width=args.beam_width,
                max_length=args.max_length,
            )
        )

    trained_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    trained_model.eval()
    prediction_rows.extend(
        evaluate_control(
            model=trained_model,
            tokenizer=tokenizer,
            records=records,
            operators=operators,
            control="trained_model",
            shuffled_inventory=False,
            beam_width=args.beam_width,
            max_length=args.max_length,
        )
    )
    prediction_rows.extend(
        evaluate_control(
            model=trained_model,
            tokenizer=tokenizer,
            records=records,
            operators=operators,
            control="trained_model_shuffled_inventory",
            shuffled_inventory=True,
            beam_width=args.beam_width,
            max_length=args.max_length,
        )
    )

    probe_rows = probe_diagnostic(records, operators, limit=args.probe_limit)
    result = {
        "records": len(records),
        "beam_width": args.beam_width,
        "max_length": args.max_length,
        "candidate_budgets": list(BUDGET_TO_SLOT_K),
        "slot_topk_for_budgets": BUDGET_TO_SLOT_K,
        "prediction_summary_by_control": summarize_predictions(prediction_rows, ["control"]),
        "prediction_summary_by_control_template": summarize_predictions(prediction_rows, ["control", "template"]),
        "prediction_rows": prediction_rows,
        "probe_summary": summarize_probe(probe_rows),
        "probe_rows": probe_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["records", "beam_width", "probe_summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

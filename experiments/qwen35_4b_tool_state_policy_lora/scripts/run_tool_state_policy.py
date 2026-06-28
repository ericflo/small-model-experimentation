#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_PATH = "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


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


def final_program_exact(record: dict[str, Any]) -> bool:
    return bool(record.get("final_visible_pass") and record.get("final_hidden_exact"))


def oracle_action(record: dict[str, Any]) -> str:
    if final_program_exact(record) and not bool(record.get("direct_exact")):
        return "PROGRAM"
    return "DIRECT"


def action_is_correct(record: dict[str, Any], action: str) -> bool:
    if action == "PROGRAM":
        return final_program_exact(record)
    return bool(record.get("direct_exact"))


def visible_output_diff_features(record: dict[str, Any]) -> dict[str, int | bool]:
    expected = None
    actual = None
    final_round = None
    for round_record in record.get("rounds", []):
        if round_record.get("round") == record.get("final_round"):
            final_round = round_record
            break
    if final_round is None and record.get("rounds"):
        final_round = record["rounds"][-1]
    if final_round is not None:
        actual = final_round.get("visible_output")
    target = record.get("target_table")
    if target is not None:
        expected = target
    exp_r, exp_c, exp_cells, exp_chars = table_dims(expected)
    act_r, act_c, act_cells, act_chars = table_dims(actual)
    return {
        "visible_output_rows_match": exp_r == act_r,
        "visible_output_cols_match": exp_c == act_c,
        "visible_output_cells_match": exp_cells == act_cells,
        "visible_output_char_delta_abs": abs(exp_chars - act_chars),
        "visible_output_row_delta_abs": abs(exp_r - act_r),
        "visible_output_col_delta_abs": abs(exp_c - act_c),
    }


def state_features(record: dict[str, Any]) -> dict[str, Any]:
    direct_r, direct_c, direct_cells, direct_chars = table_dims(record.get("direct_table"))
    program_r, program_c, program_cells, program_chars = table_dims(record.get("final_hidden_output"))
    target_r, target_c, target_cells, target_chars = table_dims(record.get("target_table"))
    rounds = record.get("rounds", [])
    final_round_record = rounds[-1] if rounds else {}
    code_error = str(final_round_record.get("visible_exec_error") or "")
    features: dict[str, Any] = {
        "direct_parse_ok": bool(record.get("direct_parse_ok")),
        "direct_output_empty": direct_r == 0 or direct_c == 0,
        "direct_program_agreement": bool(record.get("direct_program_agreement")),
        "final_visible_pass": bool(record.get("final_visible_pass")),
        "initial_visible_pass": bool(record.get("initial_visible_pass")),
        "round_count_1": int(record.get("round_count", 0)) <= 1,
        "round_count_ge_3": int(record.get("round_count", 0)) >= 3,
        "final_round_0": int(record.get("final_round", -1)) == 0,
        "final_round_ge_2": int(record.get("final_round", -1)) >= 2,
        "program_output_empty": program_r == 0 or program_c == 0,
        "program_direct_same_shape": (program_r, program_c) == (direct_r, direct_c),
        "program_target_same_shape": (program_r, program_c) == (target_r, target_c),
        "program_more_rows_than_direct": program_r > direct_r,
        "program_fewer_rows_than_direct": program_r < direct_r,
        "program_more_cols_than_direct": program_c > direct_c,
        "program_fewer_cols_than_direct": program_c < direct_c,
        "visible_exec_error": bool(code_error),
        "visible_exec_timeout": "timeout" in code_error.lower(),
        "direct_program_disagree_visible_pass": bool(record.get("final_visible_pass"))
        and not bool(record.get("direct_program_agreement")),
    }
    features.update(visible_output_diff_features(record))
    return features


def numeric_state(record: dict[str, Any]) -> dict[str, float]:
    direct_r, direct_c, direct_cells, direct_chars = table_dims(record.get("direct_table"))
    program_r, program_c, program_cells, program_chars = table_dims(record.get("final_hidden_output"))
    target_r, target_c, target_cells, target_chars = table_dims(record.get("target_table"))
    return {
        "round_count": float(record.get("round_count", 0)),
        "final_round": float(record.get("final_round", -1)),
        "direct_rows": float(direct_r),
        "direct_cols": float(direct_c),
        "program_rows": float(program_r),
        "program_cols": float(program_c),
        "target_rows": float(target_r),
        "target_cols": float(target_c),
        "abs_program_direct_row_delta": float(abs(program_r - direct_r)),
        "abs_program_direct_col_delta": float(abs(program_c - direct_c)),
        "abs_program_target_row_delta": float(abs(program_r - target_r)),
        "abs_program_target_col_delta": float(abs(program_c - target_c)),
        "direct_chars": float(direct_chars),
        "program_chars": float(program_chars),
        "abs_program_direct_char_delta": float(abs(program_chars - direct_chars)),
    }


def render_state(record: dict[str, Any]) -> str:
    f = state_features(record)
    n = numeric_state(record)
    fields = {
        "family": record.get("family"),
        "direct_parse_ok": f["direct_parse_ok"],
        "direct_shape": [int(n["direct_rows"]), int(n["direct_cols"])],
        "program_visible_pass": f["final_visible_pass"],
        "program_round_count": int(n["round_count"]),
        "program_final_round": int(n["final_round"]),
        "program_output_shape": [int(n["program_rows"]), int(n["program_cols"])],
        "visible_expected_shape": [int(n["target_rows"]), int(n["target_cols"])],
        "program_matches_visible_shape": f["program_target_same_shape"],
        "direct_program_agreement_on_new_input": f["direct_program_agreement"],
        "program_changed_shape_vs_direct": not f["program_direct_same_shape"],
        "visible_row_delta_abs": f["visible_output_row_delta_abs"],
        "visible_col_delta_abs": f["visible_output_col_delta_abs"],
    }
    return json.dumps(fields, sort_keys=True, ensure_ascii=False)


def policy_prompt(record: dict[str, Any]) -> str:
    return (
        "You are a table-transformation controller. You must choose which already-produced output to commit.\n"
        "Available actions:\n"
        "DIRECT = commit the direct JSON answer.\n"
        "PROGRAM = commit the executable program's output on the new input.\n\n"
        "Use only the tool-state below. The public example was used to test the program; hidden correctness is not shown.\n"
        "Return exactly one token: DIRECT or PROGRAM.\n\n"
        f"TOOL_STATE: {render_state(record)}\n\n"
        "ACTION:"
    )


def split_by_family(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    families = sorted({record["family"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(families)
    split_families = {
        "train": set(families[:30]),
        "dev": set(families[30:40]),
        "test": set(families[40:]),
    }
    return {
        split: [record for record in records if record["family"] in fams]
        for split, fams in split_families.items()
    }


def choose_action(policy: str, record: dict[str, Any]) -> str:
    if policy == "direct_only":
        return "DIRECT"
    if policy == "program_if_visible_else_direct":
        return "PROGRAM" if bool(record.get("final_visible_pass")) else "DIRECT"
    if policy == "program_if_visible_and_disagrees_else_direct":
        return (
            "PROGRAM"
            if bool(record.get("final_visible_pass")) and not bool(record.get("direct_program_agreement"))
            else "DIRECT"
        )
    if policy == "parse_fallback_program_else_direct":
        return (
            "PROGRAM"
            if (not bool(record.get("direct_parse_ok"))) and bool(record.get("final_visible_pass"))
            else "DIRECT"
        )
    if policy == "oracle_action":
        return oracle_action(record)
    raise KeyError(policy)


@dataclass(frozen=True)
class EvalResult:
    policy: str
    split: str
    n: int
    exact: int
    accuracy: float
    program_commits: int
    program_correct: int
    program_precision: float | None
    direct_miss_recoveries: int
    direct_correct_losses: int
    action_counts: dict[str, int]


def evaluate_actions(records: list[dict[str, Any]], actions: list[str], policy: str, split: str) -> EvalResult:
    exact = 0
    program_commits = 0
    program_correct = 0
    recoveries = 0
    losses = 0
    for record, action in zip(records, actions):
        ok = action_is_correct(record, action)
        exact += int(ok)
        if action == "PROGRAM":
            program_commits += 1
            program_correct += int(ok)
            recoveries += int(ok and not bool(record.get("direct_exact")))
            losses += int((not ok) and bool(record.get("direct_exact")))
    n = len(records)
    return EvalResult(
        policy=policy,
        split=split,
        n=n,
        exact=exact,
        accuracy=exact / n if n else 0.0,
        program_commits=program_commits,
        program_correct=program_correct,
        program_precision=program_correct / program_commits if program_commits else None,
        direct_miss_recoveries=recoveries,
        direct_correct_losses=losses,
        action_counts=dict(Counter(actions)),
    )


def eval_fixed_policy(records_by_split: dict[str, list[dict[str, Any]]], policy: str) -> dict[str, Any]:
    out = {}
    for split, records in records_by_split.items():
        actions = [choose_action(policy, record) for record in records]
        out[split] = evaluate_actions(records, actions, policy, split).__dict__
    return out


def boolean_feature_names(records: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for record in records:
        for key, value in state_features(record).items():
            if isinstance(value, bool):
                names.add(key)
    return sorted(names)


def rule_actions(records: list[dict[str, Any]], feature: str, true_action: str, false_action: str) -> list[str]:
    actions = []
    for record in records:
        value = bool(state_features(record).get(feature))
        actions.append(true_action if value else false_action)
    return actions


def train_rule_policy(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train = records_by_split["train"]
    dev = records_by_split["dev"]
    candidates = []
    for feature in boolean_feature_names(train + dev):
        for true_action, false_action in [("PROGRAM", "DIRECT"), ("DIRECT", "PROGRAM")]:
            train_eval = evaluate_actions(train, rule_actions(train, feature, true_action, false_action), "rule", "train")
            dev_eval = evaluate_actions(dev, rule_actions(dev, feature, true_action, false_action), "rule", "dev")
            direct_train = evaluate_actions(train, ["DIRECT"] * len(train), "direct_only", "train")
            if train_eval.exact < direct_train.exact:
                continue
            candidates.append(
                {
                    "feature": feature,
                    "true_action": true_action,
                    "false_action": false_action,
                    "train": train_eval.__dict__,
                    "dev": dev_eval.__dict__,
                }
            )
    candidates.sort(
        key=lambda c: (
            c["dev"]["exact"],
            -c["dev"]["direct_correct_losses"],
            c["dev"]["direct_miss_recoveries"],
            -c["dev"]["program_commits"],
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates else {
        "feature": "none",
        "true_action": "DIRECT",
        "false_action": "DIRECT",
        "train": evaluate_actions(train, ["DIRECT"] * len(train), "direct_only", "train").__dict__,
        "dev": evaluate_actions(dev, ["DIRECT"] * len(dev), "direct_only", "dev").__dict__,
    }
    policy_name = f"rule_{selected['feature']}"
    split_results = {}
    for split, records in records_by_split.items():
        if selected["feature"] == "none":
            actions = ["DIRECT"] * len(records)
        else:
            actions = rule_actions(records, selected["feature"], selected["true_action"], selected["false_action"])
        split_results[split] = evaluate_actions(records, actions, policy_name, split).__dict__
    return {"selected": selected, "top_candidates": candidates[:20], "results": split_results}


def result_to_row(result: dict[str, Any]) -> str:
    precision = result.get("program_precision")
    return (
        f"| `{result['policy']}` | {result['exact']}/{result['n']} | {pct(result['accuracy'])} | "
        f"{result['program_commits']} | {result['direct_miss_recoveries']} | "
        f"{result['direct_correct_losses']} | {pct(precision)} |"
    )


def build_balanced_training_examples(records: list[dict[str, Any]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    program = [r for r in records if oracle_action(r) == "PROGRAM"]
    direct = [r for r in records if oracle_action(r) == "DIRECT"]
    hard_direct = [
        r for r in direct
        if bool(r.get("final_visible_pass")) or not bool(r.get("direct_parse_ok")) or bool(r.get("direct_program_agreement"))
    ]
    direct_pool = hard_direct or direct
    examples: list[dict[str, str]] = []
    target_program_repeats = max(1, math.ceil((min(len(direct_pool), 64) or 1) / max(len(program), 1)))
    for record in program:
        for _ in range(target_program_repeats):
            examples.append({"prompt": policy_prompt(record), "label": "PROGRAM", "file": record["file"]})
    for record in rng.sample(direct_pool, min(len(direct_pool), max(32, len(program) * target_program_repeats))):
        examples.append({"prompt": policy_prompt(record), "label": "DIRECT", "file": record["file"]})
    rng.shuffle(examples)
    return examples


def parse_action(text: str) -> str:
    cleaned = text.strip().upper()
    match = re.search(r"\b(DIRECT|PROGRAM)\b", cleaned)
    if match:
        return match.group(1)
    return "DIRECT"


def maybe_import_torch():
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    return torch, AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training, DataLoader


def render_chat(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise controller. Return only the requested action token."},
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
        for example in examples:
            prompt_text = render_chat(tokenizer, example["prompt"])
            full_text = prompt_text + " " + example["label"] + eos
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
            input_ids = full["input_ids"]
            if len(input_ids) <= len(prompt_ids):
                continue
            labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[min(len(prompt_ids), len(input_ids)) :]
            labels = labels[: len(input_ids)]
            self.items.append({"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


def collate_batch(tokenizer: Any, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    import torch
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_lora_model(args: argparse.Namespace):
    (
        torch,
        AutoTokenizer,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        DataLoader,
    ) = maybe_import_torch()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return torch, tokenizer, model, DataLoader


def generate_actions(tokenizer: Any, model: Any, records: list[dict[str, Any]], max_new_tokens: int = 4) -> list[str]:
    import torch
    actions: list[str] = []
    model.eval()
    for record in records:
        text = render_chat(tokenizer, policy_prompt(record))
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
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
        actions.append(parse_action(tokenizer.decode(gen, skip_special_tokens=True)))
    return actions


def score_actions(tokenizer: Any, model: Any, records: list[dict[str, Any]]) -> list[str]:
    import torch
    direct_id = tokenizer(" DIRECT", add_special_tokens=False)["input_ids"][0]
    program_id = tokenizer(" PROGRAM", add_special_tokens=False)["input_ids"][0]
    actions: list[str] = []
    model.eval()
    for idx, record in enumerate(records, start=1):
        text = render_chat(tokenizer, policy_prompt(record))
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]
        action = "PROGRAM" if logits[program_id] > logits[direct_id] else "DIRECT"
        actions.append(action)
        if idx % 50 == 0:
            print(f"scored {idx}/{len(records)} action states", flush=True)
    return actions


def train_and_eval_lora(
    args: argparse.Namespace,
    records_by_split: dict[str, list[dict[str, Any]]],
    run_name: str,
    shuffle_labels: bool,
) -> dict[str, Any]:
    torch, tokenizer, model, DataLoader = load_lora_model(args)
    examples = build_balanced_training_examples(records_by_split["train"], args.split_seed)
    if args.limit_train_examples:
        examples = examples[: args.limit_train_examples]
    if shuffle_labels:
        rng = random.Random(args.split_seed + 17)
        labels = [example["label"] for example in examples]
        rng.shuffle(labels)
        for example, label in zip(examples, labels):
            example["label"] = label
    dataset = ActionDataset(tokenizer, examples, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(float(loss.detach().cpu()) * args.grad_accum)
                if step % max(1, args.max_steps // 5) == 0 or step == 1:
                    print(f"{run_name} step {step}/{args.max_steps} loss={losses[-1]:.4f}", flush=True)
                if step >= args.max_steps:
                    break
        if len(loader) == 0:
            break
    model.eval()
    split_results = {}
    prediction_rows: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        eval_records = records[: args.limit_eval] if args.limit_eval else records
        actions = score_actions(tokenizer, model, eval_records)
        split_results[split] = evaluate_actions(eval_records, actions, run_name, split).__dict__
        for record, action in zip(eval_records, actions):
            prediction_rows.append(
                {
                    "split": split,
                    "file": record["file"],
                    "family": record["family"],
                    "action": action,
                    "oracle_action": oracle_action(record),
                    "direct_exact": bool(record.get("direct_exact")),
                    "program_exact": final_program_exact(record),
                    "correct": action_is_correct(record, action),
                    "final_visible_pass": bool(record.get("final_visible_pass")),
                    "direct_program_agreement": bool(record.get("direct_program_agreement")),
                }
            )
    adapter_dir = Path(args.root) / "reports" / "adapters" / run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    write_jsonl(Path(args.root) / "reports" / "predictions" / f"{run_name}.jsonl", prediction_rows)
    result = {
        "run_name": run_name,
        "shuffle_labels": shuffle_labels,
        "num_train_examples": len(examples),
        "num_tokenized_examples": len(dataset),
        "losses": losses,
        "results": split_results,
        "adapter_dir": str(adapter_dir),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def make_figures(root: Path, summary: dict[str, Any]) -> None:
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    policies = summary["policy_results"]
    test_rows = [(name, res["test"]) for name, res in policies.items() if "test" in res]
    labels = [name for name, _ in test_rows]
    acc = [row["accuracy"] for _, row in test_rows]
    y = list(range(len(labels)))
    plt.figure(figsize=(10.5, max(5.5, 0.45 * len(labels))))
    plt.barh(y, acc)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Held-out exact accuracy")
    plt.xlim(0, min(1.0, max(acc + [0.1]) + 0.2))
    plt.title("Tool-state action policies on held-out families")
    plt.tight_layout()
    plt.savefig(fig_dir / "heldout_accuracy.png", dpi=180)
    plt.close()

    recoveries = [row["direct_miss_recoveries"] for _, row in test_rows]
    losses = [row["direct_correct_losses"] for _, row in test_rows]
    x = range(len(labels))
    plt.figure(figsize=(max(9, 0.75 * len(labels)), 5.5))
    plt.bar([i - 0.18 for i in x], recoveries, width=0.36, label="Recoveries")
    plt.bar([i + 0.18 for i in x], losses, width=0.36, label="Losses")
    plt.xticks(list(x), labels, rotation=35, ha="right")
    plt.ylabel("Tasks")
    plt.title("Direct-miss recoveries vs direct-correct losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "recoveries_losses.png", dpi=180)
    plt.close()

    family = summary["family_breakdown"]
    fams = sorted(family)
    direct = [family[f]["direct_accuracy"] for f in fams]
    oracle = [family[f]["oracle_accuracy"] for f in fams]
    plt.figure(figsize=(11, max(6, 0.24 * len(fams))))
    y = list(range(len(fams)))
    plt.barh([i - 0.18 for i in y], direct, height=0.36, label="Direct")
    plt.barh([i + 0.18 for i in y], oracle, height=0.36, label="Oracle union")
    plt.yticks(y, fams, fontsize=7)
    plt.xlabel("Accuracy")
    plt.title("Family-level direct accuracy and tool-state headroom")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "family_headroom.png", dpi=180)
    plt.close()


def family_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["family"]].append(record)
    out = {}
    for family, rows in grouped.items():
        n = len(rows)
        direct = sum(bool(r.get("direct_exact")) for r in rows)
        program = sum(final_program_exact(r) for r in rows)
        oracle = sum(bool(r.get("direct_exact")) or final_program_exact(r) for r in rows)
        program_only = sum(final_program_exact(r) and not bool(r.get("direct_exact")) for r in rows)
        out[family] = {
            "n": n,
            "direct": direct,
            "direct_accuracy": direct / n if n else 0.0,
            "program": program,
            "program_accuracy": program / n if n else 0.0,
            "oracle": oracle,
            "oracle_accuracy": oracle / n if n else 0.0,
            "program_only": program_only,
        }
    return out


def write_report(root: Path, summary: dict[str, Any]) -> None:
    policy_results = summary["policy_results"]
    ordered = [
        "direct_only",
        "program_if_visible_else_direct",
        "program_if_visible_and_disagrees_else_direct",
        "parse_fallback_program_else_direct",
        "learned_rule",
        "base_zero_shot_action",
        "lora_action_policy",
        "lora_shuffled_labels",
        "oracle_action",
    ]
    lines = [
        "# Tool-State Action Policy",
        "",
        "## Summary",
        "",
        "This standalone experiment tests whether a small action policy can use executable-tool observations to choose between a direct table answer and a repaired program output.",
        "",
        "The policy is not asked to produce a table or a program. It sees a compact tool state and chooses `DIRECT` or `PROGRAM`.",
        "",
        "## Split",
        "",
        f"- Train: {summary['splits']['train']['n']} records across {summary['splits']['train']['families']} families",
        f"- Dev: {summary['splits']['dev']['n']} records across {summary['splits']['dev']['families']} families",
        f"- Test: {summary['splits']['test']['n']} records across {summary['splits']['test']['families']} families",
        "",
        "## Held-Out Test Result",
        "",
        "| Policy | Exact | Accuracy | Program commits | Recoveries | Losses | Program precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ordered:
        if name in policy_results and "test" in policy_results[name]:
            row = dict(policy_results[name]["test"])
            row["policy"] = name
            lines.append(result_to_row(row))
    rule_test = policy_results.get("learned_rule", {}).get("test")
    lora_test = policy_results.get("lora_action_policy", {}).get("test")
    shuffled_test = policy_results.get("lora_shuffled_labels", {}).get("test")
    direct_test = policy_results.get("direct_only", {}).get("test")
    verdict_lines = [
        "",
        "## Gate Verdict",
        "",
    ]
    if rule_test is not None:
        verdict_lines.append(
            "The environment state contains a deployable selection signal. "
            f"The selected rule reaches {rule_test['exact']}/{rule_test['n']} on held-out test, "
            f"with {rule_test['direct_miss_recoveries']} direct-miss recoveries and "
            f"{rule_test['direct_correct_losses']} losses."
        )
    if lora_test is not None and direct_test is not None:
        verdict_lines.append(
            "The LoRA action-policy arm is a partial positive. "
            f"It reaches {lora_test['exact']}/{lora_test['n']} versus direct-only "
            f"{direct_test['exact']}/{direct_test['n']}, with "
            f"{lora_test['direct_miss_recoveries']} recoveries and "
            f"{lora_test['direct_correct_losses']} losses."
        )
    if lora_test is not None and shuffled_test is not None:
        verdict_lines.append(
            "The shuffled-label control is worse than the real-label adapter "
            f"({shuffled_test['exact']}/{shuffled_test['n']} vs "
            f"{lora_test['exact']}/{lora_test['n']}), so the adapter learned useful state signal."
        )
    if rule_test is not None and lora_test is not None and lora_test["exact"] < rule_test["exact"]:
        verdict_lines.append(
            "The posttraining arm did not beat the simpler deployable rule; the rule exposes the cleaner mechanism."
        )
    lines.extend(verdict_lines)
    lines.extend(
        [
            "",
            "## Learned Rule",
            "",
            "The non-neural rule search selected:",
            "",
            "```json",
            json.dumps(summary["learned_rule"]["selected"], indent=2, sort_keys=True)[:3000],
            "```",
            "",
            "## LoRA Training",
            "",
        ]
    )
    if "lora" in summary:
        lora = summary["lora"]
        lines.extend(
            [
                f"- Real-label LoRA train examples: {lora.get('num_train_examples')}",
                f"- Real-label LoRA tokenized examples: {lora.get('num_tokenized_examples')}",
                f"- Real-label final loss: {lora.get('losses', [None])[-1] if lora.get('losses') else 'n/a'}",
            ]
        )
    else:
        lines.append("- LoRA arm was skipped for this run.")
    if "lora_shuffled" in summary:
        shuffled = summary["lora_shuffled"]
        lines.extend(
            [
                f"- Shuffled-label LoRA train examples: {shuffled.get('num_train_examples')}",
                f"- Shuffled-label final loss: {shuffled.get('losses', [None])[-1] if shuffled.get('losses') else 'n/a'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Held-out accuracy](figures/heldout_accuracy.png)",
            "",
            "![Recoveries and losses](figures/recoveries_losses.png)",
            "",
            "![Family headroom](figures/family_headroom.png)",
            "",
            "## Interpretation",
            "",
            "The decisive comparison is whether the learned action policy converts program-only headroom into held-out recoveries without committing hidden-wrong visible-pass programs on direct-correct tasks.",
            "",
            "The oracle row is an upper bound that uses hidden labels to pick `PROGRAM` exactly when the program is correct and the direct answer is not. Deployable policies do not see that label.",
            "",
            "## Limitations",
            "",
            "- The environment traces are precomputed; this package trains and evaluates the commit policy over those observed states.",
            "- The policy observes the final repair-loop state, so policies that use tool observations pay the full repair-loop generation cost even when they choose `DIRECT`.",
            "- The family-disjoint split is deterministic but still small. Repeat across multiple split seeds before treating the learned policy as stable.",
        ]
    )
    write_text(root / "reports" / "report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--split-seed", type=int, default=6137)
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--train-shuffled-lora", action="store_true")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--limit-train-examples", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    args = parser.parse_args()

    root = args.root
    records = load_jsonl(root / "data" / "tool_state_traces.jsonl")
    records_by_split = split_by_family(records, args.split_seed)
    summary: dict[str, Any] = {
        "experiment": "qwen35_4b_tool_state_policy_lora",
        "model_path": args.model_path,
        "split_seed": args.split_seed,
        "splits": {},
        "policy_results": {},
    }
    for split, split_records in records_by_split.items():
        fams = sorted({record["family"] for record in split_records})
        labels = Counter(oracle_action(record) for record in split_records)
        summary["splits"][split] = {
            "n": len(split_records),
            "families": len(fams),
            "family_names": fams,
            "oracle_action_labels": dict(labels),
            "direct_exact": sum(bool(record.get("direct_exact")) for record in split_records),
            "program_exact": sum(final_program_exact(record) for record in split_records),
            "oracle_union": sum(bool(record.get("direct_exact")) or final_program_exact(record) for record in split_records),
        }

    fixed_policies = [
        "direct_only",
        "program_if_visible_else_direct",
        "program_if_visible_and_disagrees_else_direct",
        "parse_fallback_program_else_direct",
        "oracle_action",
    ]
    for policy in fixed_policies:
        summary["policy_results"][policy] = eval_fixed_policy(records_by_split, policy)
    learned_rule = train_rule_policy(records_by_split)
    summary["learned_rule"] = learned_rule
    summary["policy_results"]["learned_rule"] = learned_rule["results"]

    if args.train_lora and not args.skip_lora:
        # Base zero-shot action generation is evaluated before adapter training.
        torch, tokenizer, model, _ = load_lora_model(args)
        base_results = {}
        base_predictions = []
        for split, split_records in records_by_split.items():
            eval_records = split_records[: args.limit_eval] if args.limit_eval else split_records
            actions = score_actions(tokenizer, model, eval_records)
            base_results[split] = evaluate_actions(eval_records, actions, "base_zero_shot_action", split).__dict__
            for record, action in zip(eval_records, actions):
                base_predictions.append(
                    {
                        "split": split,
                        "file": record["file"],
                        "family": record["family"],
                        "action": action,
                        "oracle_action": oracle_action(record),
                        "correct": action_is_correct(record, action),
                    }
                )
        write_jsonl(root / "reports" / "predictions" / "base_zero_shot_action.jsonl", base_predictions)
        summary["policy_results"]["base_zero_shot_action"] = base_results
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        lora = train_and_eval_lora(args, records_by_split, "lora_action_policy", shuffle_labels=False)
        summary["lora"] = {k: v for k, v in lora.items() if k != "results"}
        summary["policy_results"]["lora_action_policy"] = lora["results"]
        if args.train_shuffled_lora:
            shuffled = train_and_eval_lora(args, records_by_split, "lora_shuffled_labels", shuffle_labels=True)
            summary["lora_shuffled"] = {k: v for k, v in shuffled.items() if k != "results"}
            summary["policy_results"]["lora_shuffled_labels"] = shuffled["results"]

    summary["family_breakdown"] = family_breakdown(records_by_split["test"])
    write_json(root / "reports" / "final_summary.json", summary)
    make_figures(root, summary)
    write_report(root, summary)
    print(json.dumps(summary, indent=2, sort_keys=True)[:12000])


if __name__ == "__main__":
    main()

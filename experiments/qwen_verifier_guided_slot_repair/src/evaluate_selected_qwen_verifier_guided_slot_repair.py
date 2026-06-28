#!/usr/bin/env python3
"""Retest selected verifier-guided slot-repair checkpoints on fresh programs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from qwen_verifier_guided_slot_repair_experiment import (
    DirectAnswerHead,
    ExampleSet,
    ProgramCompiler,
    TextProgramGenerator,
    TransitionExecutor,
    dtype_from_string,
    ensure_pad_token,
    evaluate,
)


ROOT = Path("experiments/qwen_verifier_guided_slot_repair")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_selected_run(run_name: str) -> Dict[str, Any]:
    path = RUNS / run_name / "results.json"
    data = json.loads(path.read_text())
    if len(data.get("variants", {})) != 1:
        raise ValueError(f"{path} must contain exactly one variant")
    variant, result = next(iter(data["variants"].items()))
    return {
        "run": run_name,
        "variant": variant,
        "args": data["args"],
        "metadata": data.get("metadata", {}),
        "hidden_dim": data["hidden_dim"],
        "selected": result["selected_checkpoint"],
    }


def load_model_and_tokenizer(run_args: Dict[str, Any], adapter_dir: str, device: torch.device) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(run_args["model_id"], trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    dtype = dtype_from_string(run_args.get("torch_dtype", "bf16"))
    quantization_config = None
    if run_args.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    common: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": run_args.get("device_map", "auto") if torch.cuda.is_available() else None,
    }
    if quantization_config is not None:
        common["quantization_config"] = quantization_config
    common = {k: v for k, v in common.items() if v is not None}
    print(f"[load] {run_args['model_id']} + {adapter_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(run_args["model_id"], **common)
    model.config.use_cache = False
    if run_args.get("use_lora", False):
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    else:
        if not run_args.get("load_in_4bit", False):
            model.to(device)
    model.eval()
    return tokenizer, model


def build_eval_sets(tokenizer: Any, run_args: Dict[str, Any], eval_size: int, eval_lengths: List[int], eval_seed: int) -> Dict[str, ExampleSet]:
    eval_sets: Dict[str, ExampleSet] = {}
    for mode_index, mode in enumerate(["standard", "paraphrase"]):
        for length in eval_lengths:
            gen = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], eval_seed + 97 * mode_index + length, mode)
            eval_sets[f"fresh_{mode}_len{length}"] = gen.dataset(eval_size, length, length)
    for length in eval_lengths:
        gen = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], eval_seed + 2000 + length, "mixed")
        eval_sets[f"fresh_paired_len{length}"] = gen.paired_dataset(eval_size, length, length, ["standard", "paraphrase"])
    return eval_sets


def retest_one(
    spec: Dict[str, Any],
    eval_size: int,
    eval_lengths: List[int],
    eval_seed: int,
    eval_batch_size: int,
    device: torch.device,
    repair_verifier_mode: str,
    repair_topk: int,
    repair_max_edits: int,
) -> List[Dict[str, Any]]:
    run_args = dict(spec["args"])
    run_args["eval_batch_size"] = eval_batch_size
    if repair_verifier_mode:
        run_args["repair_verifier_mode"] = repair_verifier_mode
    if repair_topk > 0:
        run_args["repair_topk"] = repair_topk
    if repair_max_edits > 0:
        run_args["repair_max_edits"] = repair_max_edits
    args_ns = SimpleNamespace(**run_args)
    selected = spec["selected"]
    tokenizer, model = load_model_and_tokenizer(run_args, selected["model_checkpoint"], device)
    heads = torch.load(selected["heads_checkpoint"], map_location=device)
    direct = None
    compiler = None
    if heads.get("direct") is not None:
        direct = DirectAnswerHead(heads["hidden_dim"], run_args["modulus"], run_args["head_width"]).to(device)
        direct.load_state_dict(heads["direct"])
        direct.eval()
    if heads.get("compiler") is not None:
        compiler = ProgramCompiler(
            heads["hidden_dim"],
            run_args["modulus"],
            run_args["head_width"],
            run_args["max_steps"],
            run_args["rank_temperature"],
            run_args["arg_reader_mode"],
            run_args["arg_window"],
            run_args["arg_distance_temperature"],
        ).to(device)
        compiler.load_state_dict(heads["compiler"])
        compiler.eval()
    executor = TransitionExecutor(run_args["modulus"], device)
    eval_sets = build_eval_sets(tokenizer, run_args, eval_size, eval_lengths, eval_seed)
    rows: List[Dict[str, Any]] = []
    for split, dataset in eval_sets.items():
        metrics = evaluate(spec["variant"], model, direct, compiler, executor, dataset, tokenizer, args_ns, device)
        row = {
            "run": spec["run"],
            "variant": spec["variant"],
            "selected_step": selected.get("step"),
            "selected_tag": selected.get("tag"),
            "state_loss_schedule": run_args.get("state_loss_schedule", ""),
            "repair_verifier_mode": run_args.get("repair_verifier_mode", ""),
            "repair_topk": run_args.get("repair_topk", ""),
            "repair_max_edits": run_args.get("repair_max_edits", ""),
            "selection_metric": selected.get("selection_metric", ""),
            "selection_value": selected.get("selection_value", ""),
            "split": split,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str)) and k not in {"variant"}},
        }
        rows.append(row)
        print(
            f"[eval] {spec['run']} {split} "
            f"executor={metrics.get('executor_accuracy')} repair={metrics.get('repair_executor_accuracy')}",
            flush=True,
        )
    del model, direct, compiler, executor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retest selected verifier-guided slot-repair checkpoints")
    p.add_argument(
        "--runs",
        type=str,
        default="main_state_w025_repair_s900",
    )
    p.add_argument("--eval_size", type=int, default=256)
    p.add_argument("--eval_lengths", type=str, default="24")
    p.add_argument("--eval_seed", type=int, default=91001)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--repair_verifier_mode", type=str, default="")
    p.add_argument("--repair_topk", type=int, default=0)
    p.add_argument("--repair_max_edits", type=int, default=0)
    p.add_argument("--output_csv", type=str, default=str(ANALYSIS / "selected_retest_metrics.csv"))
    p.add_argument("--output_json", type=str, default=str(ANALYSIS / "selected_retest_results.json"))
    p.add_argument("--append_existing", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    lengths = [int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: List[Dict[str, Any]] = []
    if args.append_existing:
        all_rows.extend(read_csv(Path(args.output_csv)))
    for run_name in runs:
        spec = load_selected_run(run_name)
        all_rows.extend(
            retest_one(
                spec,
                args.eval_size,
                lengths,
                args.eval_seed,
                args.eval_batch_size,
                device,
                args.repair_verifier_mode,
                args.repair_topk,
                args.repair_max_edits,
            )
        )
    write_csv(Path(args.output_csv), all_rows)
    Path(args.output_json).write_text(json.dumps({"args": vars(args), "rows": all_rows}, indent=2))
    print(args.output_csv, flush=True)


if __name__ == "__main__":
    main()

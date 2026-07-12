#!/usr/bin/env python3
"""Cache full-softmax teacher top-k distributions on exact student rollouts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, read_jsonl, sha256_file  # noqa: E402


def _score_subset(model_path: Path, rows: list[dict], top_k: int, max_length: int) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            output = row["outputs"][0]
            prompt = [int(value) for value in row["prompt_token_ids"]]
            completion = [int(value) for value in output["token_ids"]]
            if not completion:
                continue
            ids = prompt + completion
            if len(ids) > max_length:
                raise RuntimeError(f"teacher-scoring input exceeds max_length: {row['id']} {len(ids)}")
            inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
            logits = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                logits_to_keep=len(completion) + 1,
                use_cache=False,
            ).logits
            if logits.shape[1] < len(completion) + 1:
                raise RuntimeError(f"insufficient logits for {row['id']}: {tuple(logits.shape)}")
            prediction = logits[0, -(len(completion) + 1):-1].float()
            log_probs = torch.log_softmax(prediction, dim=-1)
            values, indices = torch.topk(log_probs, k=top_k, dim=-1)
            mask = torch.ones(len(completion), dtype=torch.bool)
            injected = output.get("injected_token_ids") or []
            retained = output.get("retained_thinking_token_ids") or []
            if injected:
                start = len(retained)
                mask[start:start + len(injected)] = False
            results.append({
                "id": str(row["id"]),
                "meta": dict(row.get("meta") or {}),
                "prompt_ids": torch.tensor(prompt, dtype=torch.int32),
                "completion_ids": torch.tensor(completion, dtype=torch.int32),
                "policy_mask": mask,
                "teacher_indices": indices.to(device="cpu", dtype=torch.int32),
                "teacher_log_probs": values.to(device="cpu", dtype=torch.float16),
                "teacher": str(model_path.resolve()),
                "forced_close": bool(output.get("forced_close")),
            })
            del logits, prediction, log_probs, values, indices, inputs
            if (index + 1) % 10 == 0:
                print(f"[teacher-score] {model_path.name}: {index + 1}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--quick-teacher", type=Path, required=True)
    parser.add_argument("--deep-teacher", type=Path, required=True)
    parser.add_argument("--routing", choices=("correct", "wrong"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    rows = read_jsonl(args.rollouts)
    if not rows:
        raise SystemExit("empty student rollout file")
    by_stratum = {
        name: [row for row in rows if row.get("meta", {}).get("stratum") == name]
        for name in ("quick", "deep")
    }
    if any(not values for values in by_stratum.values()):
        raise SystemExit("rollout file must contain both quick and deep strata")
    paths = {
        "quick": args.quick_teacher.resolve(),
        "deep": args.deep_teacher.resolve(),
    }
    if args.routing == "wrong":
        paths = {"quick": paths["deep"], "deep": paths["quick"]}
    started = time.perf_counter()
    samples = []
    for stratum in ("quick", "deep"):
        samples.extend(
            _score_subset(
                paths[stratum], by_stratum[stratum], int(config["mopd"]["top_k"]),
                int(config["mopd"]["max_length"]),
            )
        )
    samples.sort(key=lambda row: row["id"])
    payload = {
        "schema_version": 1,
        "method": "cached_full_softmax_teacher_topk",
        "routing": args.routing,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": sha256_file(args.rollouts),
        "top_k": int(config["mopd"]["top_k"]),
        "teachers": {
            stratum: {
                "path": str(path),
                "config_sha256": sha256_file(path / "config.json"),
                "merge_receipt_sha256": sha256_file(path / "merge_receipt.json"),
            }
            for stratum, path in paths.items()
        },
        "sample_count": len(samples),
        "quick_count": sum(row["meta"]["stratum"] == "quick" for row in samples),
        "deep_count": sum(row["meta"]["stratum"] == "deep" for row in samples),
        "active_positions": sum(int(row["policy_mask"].sum()) for row in samples),
        "wall_seconds": time.perf_counter() - started,
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    receipt = {key: value for key, value in payload.items() if key != "samples"}
    receipt["cache_sha256"] = digest
    receipt["cache_bytes"] = args.out.stat().st_size
    args.out.with_suffix(args.out.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

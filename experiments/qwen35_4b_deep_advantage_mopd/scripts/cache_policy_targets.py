#!/usr/bin/env python3
"""Cache all matched policy top-k distributions on exact student tokens."""

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

from io_utils import load_config, sha256_file  # noqa: E402
from training_units import make_sparse_sample  # noqa: E402


def _score_policy(model_path: Path, samples: list[dict], policy: str, top_k: int) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    forward_tokens = 0
    positions_scored = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            if sample["meta"]["role"] == "anchor" and policy != "soup":
                continue
            prompt = sample["prompt_ids"].to(dtype=torch.long).tolist()
            completion = sample["completion_ids"].to(dtype=torch.long).tolist()
            positions = sample["positions"].to(dtype=torch.long).tolist()
            first = min(positions)
            tail = len(completion) - first
            ids = torch.tensor([prompt + completion], dtype=torch.long, device=model.device)
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                logits_to_keep=tail + 1,
                use_cache=False,
            ).logits
            prediction = logits[0, -(tail + 1):-1].float()
            relative = torch.tensor(
                [position - first for position in positions],
                dtype=torch.long,
                device=model.device,
            )
            selected = prediction.index_select(0, relative)
            log_probs = torch.log_softmax(selected, dim=-1)
            values, indices = torch.topk(log_probs, k=top_k, dim=-1)
            sample["targets"][policy] = {
                "indices": indices.to(device="cpu", dtype=torch.int32),
                "log_probs": values.to(device="cpu", dtype=torch.float32),
            }
            forward_tokens += int(ids.shape[1])
            positions_scored += len(positions)
            del ids, logits, prediction, relative, selected, log_probs, values, indices
            if (index + 1) % 10 == 0:
                print(f"[cache] {policy}: {index + 1}/{len(samples)}", flush=True)
    elapsed = time.perf_counter() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "forward_input_tokens": forward_tokens,
        "positions_scored": positions_scored,
        "wall_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--round-manifest", type=Path, required=True)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--soup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    manifest = json.loads(args.round_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "online_advantage_training_round":
        raise SystemExit("invalid training-round manifest")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise SystemExit("training-round config mismatch")
    units = [*manifest["units"], *manifest.get("control_units", [])]
    expected = int(config["mopd"]["updates_per_round"]) * int(config["mopd"]["grad_accum"])
    expected_control = int(config["mopd"]["capability_units_per_round"])
    if (
        len(manifest.get("units") or []) != expected
        or len(manifest.get("control_units") or []) != expected_control
        or len({row["state_id"] for row in units}) != expected + expected_control
    ):
        raise SystemExit("training-round unit/identity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        args.soup, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    samples = [
        make_sparse_sample(
            unit,
            tokenizer,
            max_positions=int(config["mopd"]["max_target_positions"]),
            max_length=int(config["mopd"]["max_length"]),
        )
        for unit in units
    ]
    models = {"quick": args.quick.resolve(), "deep": args.deep.resolve(), "soup": args.soup.resolve()}
    for path in models.values():
        if not (path / "merge_receipt.json").is_file():
            raise SystemExit(f"target policy is not an explicit composite: {path}")
    ledgers = {}
    for policy in ("quick", "deep", "soup"):
        ledgers[policy] = _score_policy(
            models[policy], samples, policy, int(config["mopd"]["top_k"])
        )
    for sample in samples:
        expected_targets = (
            {"soup"}
            if sample["meta"]["role"] == "anchor"
            else {"quick", "deep", "soup"}
        )
        if set(sample["targets"]) != expected_targets:
            raise SystemExit(f"target cache incomplete for {sample['id']}")
    payload = {
        "schema_version": 1,
        "stage": "matched_all_policy_topk_cache",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "round_manifest": str(args.round_manifest.resolve()),
        "round_manifest_sha256": sha256_file(args.round_manifest),
        "round": int(manifest["round"]),
        "top_k": int(config["mopd"]["top_k"]),
        "models": {
            policy: {
                "path": str(path),
                "config_sha256": sha256_file(path / "config.json"),
                "merge_receipt_sha256": sha256_file(path / "merge_receipt.json"),
            }
            for policy, path in models.items()
        },
        "sample_count": len(samples),
        "active_positions": sum(int(sample["positions"].numel()) for sample in samples),
        "prompt_truncation": {
            "sample_count": sum(
                int(sample["meta"]["prompt_tokens_truncated"]) > 0
                for sample in samples
            ),
            "total_tokens": sum(
                int(sample["meta"]["prompt_tokens_truncated"])
                for sample in samples
            ),
            "maximum_tokens": max(
                int(sample["meta"]["prompt_tokens_truncated"])
                for sample in samples
            ),
            "state_ids_sha256": hashlib.sha256(
                "\n".join(
                    sorted(
                        sample["id"]
                        for sample in samples
                        if int(sample["meta"]["prompt_tokens_truncated"]) > 0
                    )
                ).encode("utf-8")
            ).hexdigest(),
        },
        "ledgers": ledgers,
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

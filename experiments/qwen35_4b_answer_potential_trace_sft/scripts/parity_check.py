#!/usr/bin/env python3
"""Small HF plumbing parity check for the vLLM targeted likelihood readout.

No HF-scored value enters a scientific comparison. This verifies only that
the registered token span and raw conditional log-probabilities agree across
the two implementations within bf16 backend tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from io_utils import read_jsonl, write_json  # noqa: E402
from model_ops import ANSWER_BOUNDARY  # noqa: E402
from vllm_runner import MODEL_ID, MODEL_REVISION  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=EXP / "runs" / "smoke" / "traces.jsonl")
    parser.add_argument("--scores", type=Path, default=EXP / "runs" / "smoke" / "scores.jsonl")
    parser.add_argument("--tasks", type=Path, default=EXP / "data" / "procedural" / "calibration.jsonl")
    parser.add_argument("--output", type=Path, default=EXP / "runs" / "smoke" / "hf_parity.json")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-abs-token-delta", type=float, default=0.15)
    args = parser.parse_args()

    traces = {row["trace_id"]: row for row in read_jsonl(args.traces)}
    scores = read_jsonl(args.scores)[: args.n]
    tasks = {row["id"]: row for row in read_jsonl(args.tasks)}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    print(f"[parity] loading {MODEL_ID} as HF text model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()

    rows = []
    with torch.inference_mode():
        for score in scores:
            trace = traces[score["trace_id"]]
            item = tasks[score["task_id"]]
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            boundary_ids = tokenizer.encode(ANSWER_BOUNDARY, add_special_tokens=False)
            answer_ids = tokenizer.encode(item["canonical_answer"], add_special_tokens=False)
            prefix_ids = prompt_ids + trace["token_ids"] + boundary_ids
            full_ids = prefix_ids + answer_ids
            inputs = torch.tensor([full_ids], dtype=torch.long, device="cuda")
            logits = model(input_ids=inputs, use_cache=False).logits[0]
            positions = torch.arange(
                len(prefix_ids) - 1,
                len(full_ids) - 1,
                dtype=torch.long,
                device="cuda",
            )
            target = torch.tensor(answer_ids, dtype=torch.long, device="cuda")
            selected_logits = logits.index_select(0, positions).float()
            hf_logprobs = selected_logits.log_softmax(dim=-1).gather(
                1, target[:, None]
            )[:, 0]
            hf_values = [float(value) for value in hf_logprobs.cpu()]
            canonical_variant = next(
                row
                for row in score["variant_scores"]
                if row["answer_text"] == item["canonical_answer"]
            )
            hf_sum = sum(hf_values)
            vllm_sum = float(score["canonical_ll_sum"])
            rows.append(
                {
                    "trace_id": score["trace_id"],
                    "answer_token_ids": answer_ids,
                    "hf_token_logprobs": hf_values,
                    "vllm_token_logprobs": canonical_variant.get(
                        "answer_token_logprobs"
                    ),
                    "hf_ll_sum": hf_sum,
                    "vllm_ll_sum": vllm_sum,
                    "sum_delta_hf_minus_vllm": hf_sum - vllm_sum,
                    "abs_mean_token_delta": abs(hf_sum - vllm_sum) / len(answer_ids),
                }
            )
    maximum = max(row["abs_mean_token_delta"] for row in rows)
    result = {
        "schema_version": 1,
        "purpose": "plumbing_only_no_scientific_rows",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "hf_backend": "transformers_bf16_sdpa",
        "vllm_backend": "vllm_bf16_targeted_logprob_token_ids",
        "n": len(rows),
        "max_abs_mean_token_delta": maximum,
        "threshold": args.max_abs_token_delta,
        "passed": math.isfinite(maximum) and maximum <= args.max_abs_token_delta,
        "rows": rows,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("HF/vLLM likelihood plumbing parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

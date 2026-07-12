#!/usr/bin/env python3
"""Validate the pinned Transformers training path against the vLLM smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import bitsandbytes
import peft
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TASKS = (
    ("Reply with exactly VLLM_OK.", "VLLM_OK"),
    ("Return only the integer equal to 17 + 25.", "42"),
    ("Write the reverse of the string abcdef, with no explanation.", "fedcba"),
    ("Return only a JSON array containing the first five positive odd integers.", "[1, 3, 5, 7, 9]"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-output",
        type=Path,
        default=EXP / "runs" / "model_smoke" / "base.jsonl",
    )
    parser.add_argument(
        "--out", type=Path, default=EXP / "runs" / "model_smoke" / "hf.json"
    )
    args = parser.parse_args()
    rows = _read_jsonl(args.vllm_output)
    if len(rows) != len(TASKS):
        raise SystemExit(f"expected {len(TASKS)} vLLM rows, found {len(rows)}")
    if not is_causal_conv1d_available() or not is_flash_linear_attention_available():
        raise SystemExit("Qwen3.5 Transformers fast-path extensions are unavailable")
    metadata_path = args.vllm_output.with_name(args.vllm_output.name + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model") != MODEL_ID or metadata.get("model_revision") != MODEL_REVISION:
        raise SystemExit("vLLM smoke did not use the pinned model revision")
    graphs = metadata.get("resolved_cudagraph", {})
    if not graphs.get("has_full_cudagraphs") or graphs.get("cudagraph_capture_sizes") != [1, 2, 4, 8, 16]:
        raise SystemExit(f"vLLM CUDA-graph resolution failed: {graphs}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.encode("<think>", add_special_tokens=False) != [248068]:
        raise SystemExit("pinned <think> token changed")
    if tokenizer.encode("</think>", add_special_tokens=False) != [248069]:
        raise SystemExit("pinned </think> token changed")
    prompt_counts = []
    semantic_checks = []
    for index, ((task, expected), row) in enumerate(zip(TASKS, rows)):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": task}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        prompt_counts.append(count)
        if count != int(row["n_prompt_tokens"]):
            raise SystemExit(
                f"HF/vLLM prompt-token mismatch at row {index}: {count} != {row['n_prompt_tokens']}"
            )
        text = str(row["outputs"][0]["text"]).replace("<|im_end|>", "").strip()
        passed = text == expected
        semantic_checks.append({"id": row["id"], "expected": expected, "observed": text, "passed": passed})
    if not all(row["passed"] for row in semantic_checks):
        raise SystemExit("vLLM semantic smoke failed")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        quantization_config=bnb,
        attn_implementation="sdpa",
    )
    training_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return only OK."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not training_prompt.endswith("<think>\n"):
        raise SystemExit("training chat template no longer opens the thinking channel")
    inputs = tokenizer(training_prompt, add_special_tokens=False, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, logits_to_keep=1, use_cache=False)
    logits = outputs.logits
    if logits.shape[0] != 1 or logits.shape[1] != 1 or logits.shape[-1] < len(tokenizer):
        raise SystemExit(f"unexpected logits shape: {tuple(logits.shape)}")
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise SystemExit("non-finite pinned-model forward logits")

    lock = REPO / "requirements-training.lock.txt"
    receipt = {
        "status": "pass",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vllm_output": str(args.vllm_output.resolve()),
        "vllm_output_sha256": _sha256(args.vllm_output),
        "vllm_metadata": str(metadata_path.resolve()),
        "vllm_metadata_sha256": _sha256(metadata_path),
        "vllm_resolved_cudagraph": graphs,
        "prompt_token_counts": prompt_counts,
        "semantic_checks": semantic_checks,
        "training_prompt_thinking_channel": True,
        "fast_path": {
            "causal_conv1d": is_causal_conv1d_available(),
            "flash_linear_attention": is_flash_linear_attention_available(),
        },
        "forward_logits_shape": list(logits.shape),
        "forward_logits_finite": finite,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "training_lock": {"path": str(lock), "sha256": _sha256(lock)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

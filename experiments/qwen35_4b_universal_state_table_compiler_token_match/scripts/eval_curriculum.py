#!/usr/bin/env python3
"""Greedy held-out installability gate on fresh synthetic curriculum tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
sys.path.insert(0, str(EXP / "scripts"))

from gen_curriculum import SMOKE_MIX, generate_curriculum  # noqa: E402


ANSWER_RE = re.compile(r"(?:^|\n)ANSWER:\s*(.*?)(?=\n|<\||</|$)", re.DOTALL)


def parse_answer(text: str) -> str | None:
    matches = [match.group(1).strip() for match in ANSWER_RE.finditer(text)]
    return matches[-1] if matches and matches[-1] else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", action="append", required=True,
        help="label=/path/to/adapter; repeat for compared adapters",
    )
    parser.add_argument("--seed", type=int, default=88008)
    parser.add_argument("--mix", default=SMOKE_MIX)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite local evaluation output")

    adapters: list[tuple[str, Path]] = []
    for specification in args.adapter:
        label, separator, raw_path = specification.partition("=")
        path = Path(raw_path).resolve()
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label):
            parser.error(f"invalid adapter specification: {specification}")
        if not (path / "adapter_model.safetensors").is_file():
            parser.error(f"missing adapter weights: {path}")
        adapters.append((label, path))
    if len({label for label, _ in adapters}) != len(adapters):
        parser.error("adapter labels must be unique")

    rows = generate_curriculum(args.mix, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        for row in rows
    ]
    if not all(prompt.endswith("<think>\n") for prompt in prompts):
        raise SystemExit("unexpected thinking-template boundary")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    )
    first_label, first_path = adapters[0]
    model = PeftModel.from_pretrained(base, str(first_path), adapter_name=first_label)
    for label, path in adapters[1:]:
        model.load_adapter(str(path), adapter_name=label)
    model.eval()

    output_rows = []
    summaries = {}
    with torch.inference_mode():
        for label, _ in adapters:
            model.set_adapter(label)
            for start in range(0, len(rows), args.batch_size):
                stop = min(start + args.batch_size, len(rows))
                print(
                    f"[local-eval] adapter={label} rows={start}:{stop}/{len(rows)}",
                    file=sys.stderr,
                    flush=True,
                )
                encoded = tokenizer(
                    prompts[start:stop], return_tensors="pt", padding=True,
                    add_special_tokens=False,
                ).to(model.device)
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
                prefix_length = encoded["input_ids"].shape[1]
                for offset, token_ids in enumerate(generated[:, prefix_length:]):
                    row = rows[start + offset]
                    text = tokenizer.decode(token_ids, skip_special_tokens=False)
                    parsed = parse_answer(text)
                    expected = row["answer"].removeprefix("ANSWER: ").strip()
                    token_list = token_ids.tolist()
                    eos = model.generation_config.eos_token_id
                    eos_values = set(eos if isinstance(eos, list) else [eos])
                    first_eos = next(
                        (index for index, token in enumerate(token_list) if token in eos_values),
                        None,
                    )
                    n_tokens = first_eos + 1 if first_eos is not None else len(token_list)
                    output_rows.append({
                        "adapter": label,
                        "task_id": row["task_id"],
                        "kind": row["kind"],
                        "surface": row["surface"],
                        "expected": expected,
                        "parsed": parsed,
                        "correct": parsed == expected,
                        "n_generated_tokens": n_tokens,
                        "cap_contact": n_tokens >= args.max_new_tokens,
                        "completion": text,
                    })
            selected = [row for row in output_rows if row["adapter"] == label]
            by_kind: dict[str, list[dict]] = defaultdict(list)
            for row in selected:
                by_kind[row["kind"]].append(row)
            summaries[label] = {
                "rows": len(selected),
                "parse_rate": sum(row["parsed"] is not None for row in selected) / len(selected),
                "accuracy": sum(row["correct"] for row in selected) / len(selected),
                "cap_contacts": sum(row["cap_contact"] for row in selected),
                "mean_generated_tokens": sum(row["n_generated_tokens"] for row in selected) / len(selected),
                "per_kind": {
                    kind: {
                        "n": len(kind_rows),
                        "parse_rate": sum(row["parsed"] is not None for row in kind_rows) / len(kind_rows),
                        "accuracy": sum(row["correct"] for row in kind_rows) / len(kind_rows),
                    }
                    for kind, kind_rows in sorted(by_kind.items())
                },
                "answer_counts": dict(sorted(Counter(
                    row["parsed"] if row["parsed"] is not None else "<NONE>"
                    for row in selected
                ).items())),
            }

    payload = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": args.seed,
        "mix": args.mix,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "adapters": {label: str(path) for label, path in adapters},
        "summaries": summaries,
        "rows": output_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

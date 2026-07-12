#!/usr/bin/env python3
"""Audit SFT inclusion/skips without loading model weights or writing examples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from io_utils import sha256_file, write_json  # noqa: E402
from train_specialist import MODEL_ID, MODEL_REVISION, encode_row  # noqa: E402


def _quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    index = round(probability * (len(ordered) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--w-think", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model_source = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    if model_source != MODEL_ID:
        if not Path(model_source, "config.json").exists():
            raise SystemExit("local source model is missing config.json")
        source_kwargs = {"local_files_only": True}
    else:
        source_kwargs = {"revision": MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, **source_kwargs, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    family_included: Counter[str] = Counter()
    family_skipped: Counter[str] = Counter()
    kind_included: Counter[str] = Counter()
    kind_skipped: Counter[str] = Counter()
    lengths: list[int] = []
    input_rows = 0
    for path in args.train:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            input_rows += 1
            row = json.loads(line)
            row.setdefault("kind", "atom")
            encoded = encode_row(row, tokenizer, args.max_length, args.w_think)
            family = str(row.get("family", "unknown"))
            kind = str(row.get("kind", "unknown"))
            if encoded is None:
                family_skipped[family] += 1
                kind_skipped[kind] += 1
            else:
                family_included[family] += 1
                kind_included[kind] += 1
                lengths.append(len(encoded["input_ids"]))
    skipped = input_rows - len(lengths)
    result = {
        "stage": "sft_encoding_audit",
        "model": model_source,
        "model_revision": MODEL_REVISION if model_source == MODEL_ID else None,
        "train_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.train
        ],
        "max_length": args.max_length,
        "think_loss_weight": args.w_think,
        "input_rows": input_rows,
        "encoded_rows": len(lengths),
        "skipped_rows": skipped,
        "skip_rate": skipped / input_rows if input_rows else None,
        "encoded_family_counts": dict(sorted(family_included.items())),
        "skipped_family_counts": dict(sorted(family_skipped.items())),
        "encoded_kind_counts": dict(sorted(kind_included.items())),
        "skipped_kind_counts": dict(sorted(kind_skipped.items())),
        "encoded_length_tokens": {
            "min": min(lengths),
            "p50": _quantile(lengths, 0.50),
            "p90": _quantile(lengths, 0.90),
            "p95": _quantile(lengths, 0.95),
            "p99": _quantile(lengths, 0.99),
            "max": max(lengths),
            "sum": sum(lengths),
        },
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

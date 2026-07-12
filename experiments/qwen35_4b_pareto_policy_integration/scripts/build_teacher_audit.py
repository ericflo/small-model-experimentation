#!/usr/bin/env python3
"""Build exact student-prefix atom continuations for the teacher-routing audit."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, write_jsonl  # noqa: E402


PREFIX_TOKENS = 128


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--atom-rows", type=Path, required=True)
    parser.add_argument("--input-out", type=Path, required=True)
    parser.add_argument("--items-out", type=Path, required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"],
        trust_remote_code=True, use_fast=True,
    )
    close_id = int(tokenizer.convert_tokens_to_ids("</think>"))
    candidates: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    with gzip.open(args.atom_rows, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            level = int(row["level"])
            stratum = (
                "quick" if level in config["strata"]["quick_atom_levels"] else "deep"
            )
            output = row["outputs"][0]
            token_ids = [int(value) for value in output["token_ids"]]
            close_index = token_ids.index(close_id) if close_id in token_ids else len(token_ids)
            thinking = token_ids[:close_index]
            if len(thinking) < PREFIX_TOKENS:
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            audit_id = f"audit-{row['id']}"
            item = {
                "audit_id": audit_id,
                "source_id": row["id"],
                "family": row["family"],
                "kind": "atom",
                "level": level,
                "stratum": stratum,
                "prompt": row["prompt"],
                "gold": row["gold"],
                "answer_domain": row.get("answer_domain"),
                "prefix_ids": thinking[:PREFIX_TOKENS],
                "exact_prompt_ids": prompt_ids + thinking[:PREFIX_TOKENS],
            }
            candidates[(stratum, row["family"], level)].append(item)
    rng = random.Random(int(config["seeds"]["qualification_blocks"][0]) + 777)
    selected = []
    required = int(config["teacher_audit"]["prefixes_per_stratum"])
    for stratum in ("quick", "deep"):
        cells = [key for key in sorted(candidates) if key[0] == stratum]
        for key in cells:
            rng.shuffle(candidates[key])
        cursors = {key: 0 for key in cells}
        chosen = []
        while len(chosen) < required:
            progressed = False
            for key in cells:
                cursor = cursors[key]
                if cursor < len(candidates[key]):
                    chosen.append(candidates[key][cursor])
                    cursors[key] += 1
                    progressed = True
                    if len(chosen) == required:
                        break
            if not progressed:
                raise SystemExit(f"only found {len(chosen)} usable {stratum} prefixes")
        selected.extend(chosen)
    selected.sort(key=lambda row: row["audit_id"])
    inputs = [
        {
            "id": row["audit_id"],
            "prompt_token_ids": row["exact_prompt_ids"],
            "meta": {
                "family": row["family"], "level": row["level"],
                "stratum": row["stratum"], "prefix_tokens": PREFIX_TOKENS,
            },
        }
        for row in selected
    ]
    # Keep exact prompt ids in the input but not duplicate them in the compact
    # scoring item artifact.
    compact = [
        {key: value for key, value in row.items() if key != "exact_prompt_ids"}
        for row in selected
    ]
    write_jsonl(args.input_out, inputs)
    write_jsonl(args.items_out, compact)
    print(json.dumps({
        "items": len(selected),
        "quick": sum(row["stratum"] == "quick" for row in selected),
        "deep": sum(row["stratum"] == "deep" for row in selected),
        "prefix_tokens": PREFIX_TOKENS,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

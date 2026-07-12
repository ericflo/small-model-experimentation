#!/usr/bin/env python3
"""Build balanced visible-only quick/deep prompts for one MOPD rollout round."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, resolve_repo_path, write_jsonl  # noqa: E402


def _key(messages: list[dict]) -> str:
    data = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    seeds = config["seeds"]["rollout_rounds"]
    if not 0 <= args.round < len(seeds):
        raise SystemExit(f"round must be in [0, {len(seeds) - 1}]")
    seed = int(seeds[args.round])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"],
        trust_remote_code=True, use_fast=True,
    )
    max_length = int(config["mopd"]["max_length"])
    reserve = int(config["mopd"]["rollout_thinking_budget"]) + int(
        config["mopd"]["rollout_answer_max_tokens"]
    ) + 8
    transfer = set(config["strata"]["transfer_families"])
    buckets: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    seen = set()
    for source_name in ("quick_data", "deep_data"):
        path = resolve_repo_path(config["model"][source_name])
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            family = str(row.get("family"))
            level = int(row.get("level", -1))
            kind = str(row.get("kind", "unknown"))
            if family in transfer or kind == "skin_trace":
                continue
            if kind == "episode":
                stratum = "deep"
            elif level in config["strata"]["quick_atom_levels"]:
                stratum = "quick"
            elif level in config["strata"]["deep_atom_levels"]:
                stratum = "deep"
            else:
                continue
            messages = row["messages"]
            digest = _key(messages)
            if digest in seen:
                continue
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            prompt_tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if prompt_tokens + reserve > max_length:
                continue
            seen.add(digest)
            buckets[(stratum, family, kind, level)].append({
                "id": digest[:24], "messages": messages,
                "meta": {
                    "stratum": stratum, "family": family, "level": level,
                    "source_kind": kind, "round": args.round, "round_seed": seed,
                    "prompt_tokens": prompt_tokens,
                },
            })
    rng = random.Random(seed)
    selected = []
    count = int(config["mopd"]["rollout_prompts_per_stratum"])
    for stratum in ("quick", "deep"):
        cells = [key for key in sorted(buckets) if key[0] == stratum and buckets[key]]
        for key in cells:
            rng.shuffle(buckets[key])
        cursors = {key: 0 for key in cells}
        while len([row for row in selected if row["meta"]["stratum"] == stratum]) < count:
            progressed = False
            for key in cells:
                cursor = cursors[key]
                if cursor >= len(buckets[key]):
                    continue
                selected.append(buckets[key][cursor])
                cursors[key] += 1
                progressed = True
                if len([row for row in selected if row["meta"]["stratum"] == stratum]) >= count:
                    break
            if not progressed:
                raise SystemExit(f"only found {sum(len(buckets[key]) for key in cells)} usable {stratum} prompts")
    rng.shuffle(selected)
    written = write_jsonl(args.out, selected)
    print(json.dumps({
        "round": args.round, "seed": seed, "written": written,
        "quick": sum(row["meta"]["stratum"] == "quick" for row in selected),
        "deep": sum(row["meta"]["stratum"] == "deep" for row in selected),
        "max_prompt_tokens": max(row["meta"]["prompt_tokens"] for row in selected),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

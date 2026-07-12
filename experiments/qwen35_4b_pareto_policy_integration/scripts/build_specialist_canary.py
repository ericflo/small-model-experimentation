#!/usr/bin/env python3
"""Build eight target-free visible prompts for specialist installation canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, resolve_repo_path, write_jsonl  # noqa: E402


def _digest(messages: list[dict]) -> str:
    encoded = json.dumps(
        messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"],
        trust_remote_code=True, use_fast=True,
    )
    buckets = {"quick": {}, "deep": {}}
    for source in ("quick_data", "deep_data"):
        path = resolve_repo_path(config["model"][source])
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            level = int(row.get("level", -1))
            kind = str(row.get("kind", ""))
            if kind == "skin_trace":
                continue
            if kind == "episode" and level in config["strata"]["deep_episode_levels"]:
                stratum = "deep"
            elif kind == "episode":
                continue
            elif level in config["strata"]["deep_atom_levels"]:
                stratum = "deep"
            elif level in config["strata"]["quick_atom_levels"]:
                stratum = "quick"
            else:
                continue
            messages = row["messages"]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_tokens = len(
                tokenizer(rendered, add_special_tokens=False)["input_ids"]
            )
            if prompt_tokens + 256 + 64 + 2 > 4096:
                continue
            digest = _digest(messages)
            buckets[stratum].setdefault(digest, {
                "id": f"canary-{stratum}-{digest[:16]}",
                "messages": messages,
                "meta": {
                    "stratum": stratum, "family": str(row.get("family")),
                    "kind": kind, "level": level,
                    "prompt_tokens": prompt_tokens,
                },
            })
    rng = random.Random(int(config["seeds"]["model_smoke"]) + 91)
    selected = []
    for stratum in ("quick", "deep"):
        values = [buckets[stratum][key] for key in sorted(buckets[stratum])]
        rng.shuffle(values)
        if len(values) < 4:
            raise SystemExit(f"insufficient {stratum} canary prompts")
        selected.extend(values[:4])
    selected.sort(key=lambda row: row["id"])
    write_jsonl(args.out, selected)
    print(json.dumps({
        "rows": len(selected),
        "quick": sum(row["meta"]["stratum"] == "quick" for row in selected),
        "deep": sum(row["meta"]["stratum"] == "deep" for row in selected),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

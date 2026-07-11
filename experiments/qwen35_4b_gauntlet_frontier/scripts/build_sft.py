#!/usr/bin/env python3
"""Build the think-channel SFT dataset from a scored harvest.

CPU-only. Filters (per gym_design.md): verified score == 1.0, naturally-closed
thinking, clean parse, think length <= cap; at most N samples per atom item
(shortest thinking first); per-family cap to balance the mixture. Episode
rollouts with score == 1.0 contribute one example per naturally-closed
assistant turn, with the context rendered exactly as at generation time.

Emits JSONL rows: {family, level, kind, messages, think, answer, n_think_tokens}
where `messages` is the chat context (ending with a user turn) and
`think`/`answer` are the training targets for the next assistant turn.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym import base  # noqa: E402


def read_gz_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_target(text: str) -> tuple[str, str] | None:
    """Split a generation into (think, answer_region); None if malformed."""
    think, answer = base.split_think(text)
    think = think.strip()
    answer = answer.strip()
    if not think or not answer:
        return None
    if "<think>" in answer or "</think>" in answer:
        return None
    return think, answer


def atom_examples(rows: list[dict], cfg: dict) -> list[dict]:
    examples = []
    for row in rows:
        # Lucky-guess gate (C28 rationalization guard): items with a small
        # answer domain admit correct-by-chance samples whose thinking is
        # rationalization. Require strong sample agreement before keeping any.
        domain = row.get("answer_domain") or 10**9
        n_correct = sum(1 for output in row["outputs"] if output["score"] >= 1.0)
        if domain < 5 and n_correct < 3:
            continue
        keepers = []
        fc_keepers = []
        for output in row["outputs"]:
            if output["score"] < 1.0 or output["truncated"]:
                continue
            if output["forced_close"]:
                # Forced-close recovery arm: the chain was cut at budget and
                # the model STILL answered correctly. Training
                # truncated_think + </think> + terse ANSWER puts the
                # deployment-critical post-force-close state in-distribution.
                target = split_target(output["text"])
                value = base.extract_answer(output["text"])
                if target is not None and value is not None and cfg.get("fc_arm", True):
                    think_ctx = target[0]
                    # Trim the truncated-chain CONTEXT so recovery examples fit
                    # the encode window (rounds 2-3 silently trained on none of
                    # them; C50 attribution correction). Keep the tail: the
                    # state nearest the forced close.
                    trim = int(cfg.get("fc_trim_chars", 0))
                    if trim and len(think_ctx) > trim:
                        tail = think_ctx[-trim:]
                        cut = tail.find("\n")
                        think_ctx = tail[cut + 1:] if 0 <= cut < 400 else tail
                    fc_keepers.append(
                        (output["n_thinking_tokens"], (think_ctx, f"ANSWER: {value}"))
                    )
                continue
            if output["n_thinking_tokens"] > cfg["max_think_tokens"]:
                continue
            target = split_target(output["text"])
            if target is None:
                continue
            value = base.extract_answer(output["text"])
            if value is None:
                continue
            # Canonicalize the atom target to the terse deployable shape:
            # the think chain is the model's own verbatim; the answer region
            # becomes exactly the final ANSWER line (the model's own verified
            # value). Deployment scorers read a short answer window, so the
            # verbose re-explanation the base model emits is the one part of
            # its output we do NOT want to reinforce.
            think, _ = target
            answer = f"ANSWER: {value}"
            keepers.append((output["n_thinking_tokens"], (think, answer)))
        keepers.sort(key=lambda pair: pair[0])
        for n_think, (think, answer) in keepers[: cfg["max_per_item"]]:
            examples.append(
                {
                    "family": row["family"],
                    "level": row["level"],
                    "kind": "atom",
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "think": think,
                    "answer": answer,
                    "n_think_tokens": n_think,
                }
            )
        # At most one recovery example per item, and only when the item has
        # few/no naturally-closed keepers (the recovery state matters most
        # exactly where natural closure is rare).
        if fc_keepers and len(keepers) < cfg["max_per_item"]:
            fc_keepers.sort(key=lambda pair: pair[0])
            n_think, (think, answer) = fc_keepers[0]
            examples.append(
                {
                    "family": row["family"],
                    "level": row["level"],
                    "kind": "atom_fc",
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "think": think,
                    "answer": answer,
                    "n_think_tokens": n_think,
                }
            )
    return examples


def episode_examples(rows: list[dict], cfg: dict) -> list[dict]:
    # Cap rollouts per episode instance so K near-identical successful
    # rollouts of the same hidden world don't over-weight it.
    by_instance: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["score"] >= 1.0:
            by_instance[(row["family"], row["level"], row["ep_seed"])].append(row)
    kept_rows: list[dict] = []
    for instance_rows in by_instance.values():
        instance_rows.sort(key=lambda r: r["n_turns"])  # most efficient first
        kept_rows.extend(instance_rows[: cfg["max_rollouts_per_episode"]])

    examples = []
    for row in kept_rows:
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["initial_observation"]},
        ]
        for turn in row["turns"]:
            usable = (
                not turn["forced_close"]
                and not turn["truncated"]
                and turn.get("action_ok", True)
                and turn["n_thinking_tokens"] <= cfg["max_think_tokens"]
                and turn["n_answer_tokens"] <= cfg["max_answer_tokens"]
                and turn["action"]
            )
            if usable:
                target = split_target(turn["text"]) if "text" in turn else None
                if target is not None:
                    think, _ = target
                    examples.append(
                        {
                            "family": row["family"],
                            "level": row["level"],
                            "kind": "episode",
                            "messages": [dict(m) for m in messages],
                            "think": think,
                            "answer": turn["action"],
                            "n_think_tokens": turn["n_thinking_tokens"],
                        }
                    )
            messages.append({"role": "assistant", "content": turn["action"]})
            messages.append({"role": "user", "content": turn["observation"]})
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--harvest-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    sft_cfg = {
        "max_think_tokens": int(config["sft"]["max_think_tokens"]),
        "max_answer_tokens": int(config["sft"].get("max_answer_tokens", 96)),
        "max_per_item": int(config["sft"]["max_per_item"]),
        "max_rollouts_per_episode": int(config["sft"].get("max_rollouts_per_episode", 2)),
        "family_cap": int(config["sft"]["family_cap"]),
        "fc_trim_chars": int(config["sft"].get("fc_trim_chars", 0)),
    }
    harvest_dir = args.harvest_dir or (EXP / "runs" / f"harvest_round{config['round']}")
    out_path = args.out or (EXP / config["sft"]["out"])

    examples: list[dict] = []
    atom_rows: list[dict] = []
    episode_rows: list[dict] = []
    for path in sorted(harvest_dir.glob("atoms_rows*.jsonl.gz")):
        atom_rows += read_gz_jsonl(path)
    for path in sorted(harvest_dir.glob("episodes_rows*.jsonl.gz")):
        episode_rows += read_gz_jsonl(path)
    if atom_rows:
        examples += atom_examples(atom_rows, sft_cfg)
    if episode_rows:
        examples += episode_examples(episode_rows, sft_cfg)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_family[example["family"]].append(example)

    final: list[dict] = []
    stats = {}
    for family, family_examples in sorted(by_family.items()):
        # True stratified round-robin across (level, kind) cells, shortest
        # thinking first WITHIN a cell, so the cap trims every stratum evenly
        # instead of deleting the hardest supervision (design-review must-fix).
        cells: dict[tuple, list[dict]] = defaultdict(list)
        for example in family_examples:
            cells[(example["level"], example["kind"])].append(example)
        for cell in cells.values():
            cell.sort(key=lambda e: e["n_think_tokens"])
        kept = []
        cell_keys = sorted(cells.keys())
        while len(kept) < sft_cfg["family_cap"] and any(cells[k] for k in cell_keys):
            for key in cell_keys:
                if cells[key] and len(kept) < sft_cfg["family_cap"]:
                    kept.append(cells[key].pop(0))
        final.extend(kept)
        stats[family] = {
            "candidates": len(family_examples),
            "kept": len(kept),
            "atoms": sum(1 for e in kept if e["kind"] == "atom"),
            "episode_turns": sum(1 for e in kept if e["kind"] == "episode"),
            "mean_think_tokens": round(
                sum(e["n_think_tokens"] for e in kept) / max(1, len(kept)), 1
            ),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in final:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    summary = {"total": len(final), "families": stats}
    (out_path.with_suffix(".stats.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"[build_sft] wrote {len(final)} examples to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

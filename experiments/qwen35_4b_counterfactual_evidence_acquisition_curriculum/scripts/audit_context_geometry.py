#!/usr/bin/env python3
"""Prove registered multi-turn agent histories fit the frozen model context."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402
import harness  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_output(tokenizer, action: dict, think_budget: int) -> dict:
    seed_ids = tokenizer(" inspect evidence and preserve verified state" * 600)[
        "input_ids"
    ][:think_budget]
    thought = tokenizer.decode(seed_ids, skip_special_tokens=True)
    return {
        "text": f"{thought}\n</think>\n\n{repo_agent.action_text(action)}",
        "n_sampled_tokens": len(seed_ids) + 32,
        "n_thinking_tokens": len(seed_ids),
        "n_answer_tokens": 32,
        "n_stage1_prompt_tokens": 0,
        "n_stage2_prompt_tokens": 0,
        "n_injected_tokens": 0,
        "thinking_closed": True,
        "forced_close": False,
        "finish_reason": "stop",
        "stage1_finish_reason": "stop",
        "stage2_finish_reason": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--out", type=Path, default=EXP / "reports" / "context_geometry_receipt.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    model_path = resolve(cfg["model"]["start_checkpoint"])
    tokenizer_provenance = harness.tokenizer_provenance(model_path)
    if (
        tokenizer_provenance["tokenizer_manifest_sha256"]
        != cfg["model"]["start_tokenizer_manifest_sha256"]
        or tokenizer_provenance["tokenizer_compatibility_sha256"]
        != cfg["model"]["tokenizer_compatibility_sha256"]
    ):
        raise SystemExit("context-geometry tokenizer differs from frozen config")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    think_budget = int(cfg["evaluation"]["think_budget"])
    answer_budget = max(
        int(value) for value in cfg["evaluation"]["interface_answer_rungs"]
    )
    model_limit = int(cfg["engine"]["max_model_len"])
    tasks = repo_tasks.make_pairs(
        repo_tasks.ALL_FAMILIES, 1, 93950, "context_geometry"
    )
    scenario_turns = {
        "normal": int(cfg["evaluation"]["normal"]["deep_turns"]),
        "ambiguous_source": int(cfg["evaluation"]["controlled"]["deep_turns"]),
        "rejected_patch": int(cfg["evaluation"]["recovery"]["deep_turns"]),
        "failed_test": int(cfg["evaluation"]["recovery"]["deep_turns"]),
    }
    rows = []
    for task in tasks:
        source_path = task.oracle_patches[0].path
        for scenario, turns in scenario_turns.items():
            episode = repo_agent.Episode(task, 0, scenario=scenario)
            prompt_lengths = []
            try:
                actions = [
                    {"tool": "read", "path": source_path},
                    {"tool": "search", "query": task.acquisition_query},
                    {"tool": "read", "path": task.evidence_path},
                    {"tool": "test"},
                ]
                for turn in range(turns):
                    prompt = tokenizer.apply_chat_template(
                        episode.messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    prompt_tokens = len(
                        tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    )
                    prompt_lengths.append(prompt_tokens)
                    episode.consume(
                        synthetic_output(
                            tokenizer, actions[turn % len(actions)], think_budget
                        )
                    )
                final_prompt = tokenizer.apply_chat_template(
                    episode.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompt_lengths.append(
                    len(tokenizer(final_prompt, add_special_tokens=False)["input_ids"])
                )
            finally:
                episode.env.close()
            maximum = max(prompt_lengths) + think_budget + answer_budget
            rows.append({
                "task_id": task.task_id,
                "scenario": scenario,
                "turns": turns,
                "max_prompt_tokens": max(prompt_lengths),
                "max_prompt_plus_generation_allowance": maximum,
                "headroom_tokens": model_limit - maximum,
            })
    worst = max(rows, key=lambda row: row["max_prompt_plus_generation_allowance"])
    checks = {
        "all_registered_histories_fit": all(
            row["max_prompt_plus_generation_allowance"] <= model_limit for row in rows
        ),
        "minimum_512_token_safety_margin": all(
            row["headroom_tokens"] >= 512 for row in rows
        ),
    }
    result = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(args.config),
        "model": str(model_path.resolve()),
        **tokenizer_provenance,
        "model_max_length": model_limit,
        "think_budget": think_budget,
        "answer_budget": answer_budget,
        "history_policy": "canonical_first_valid_tool_call",
        "checks": checks,
        "worst_case": worst,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if all(checks.values()) else 4


if __name__ == "__main__":
    raise SystemExit(main())

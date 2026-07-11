#!/usr/bin/env python3
"""Stage orchestrator and CPU scientific smoke for specialist integration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from curriculum import expert_decision  # noqa: E402
from gym.families import load  # noqa: E402
from io_utils import canonical_hash, load_config, write_json  # noqa: E402

COMPOUND_FAMILIES = ("cipherkiln", "mazeferry", "patchferry", "tripleforge")


def _expert_score(family_name: str, seed: int, level: int) -> float:
    family = load(family_name)
    episode = family.Episode(seed, level)
    messages = [
        {"role": "system", "content": episode.system_prompt()},
        {"role": "user", "content": episode.initial_observation()},
    ]
    for _ in range(episode.max_turns):
        decision = expert_decision(family_name, episode, messages)
        observation, done = episode.step(decision.action)
        if not episode.last_action_ok:
            raise AssertionError((family_name, level, decision.action, observation))
        messages.extend(
            [
                {"role": "assistant", "content": decision.action},
                {"role": "user", "content": observation},
            ]
        )
        if done:
            break
    return float(episode.score())


def scientific_smoke(config: dict, config_path: Path) -> dict:
    command = [
        sys.executable,
        str(EXP / "scripts" / "selftest_gym.py"),
        "--families",
        *COMPOUND_FAMILIES,
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    expert_scores = {}
    for family_name in COMPOUND_FAMILIES:
        family = load(family_name)
        expert_scores[family_name] = {
            str(level): _expert_score(family_name, 99000 + level, level)
            for level in family.LEVELS
        }
    train = set(config["split"]["train_families"])
    transfer = set(config["split"]["transfer_families"])
    replay_excluded = set(config["split"]["replay_excluded_families"])
    if train & transfer:
        raise AssertionError(f"train/transfer overlap: {sorted(train & transfer)}")
    if transfer != replay_excluded:
        raise AssertionError("every transfer family must be excluded from replay")
    if any(score < 0.999 for row in expert_scores.values() for score in row.values()):
        raise AssertionError("a state-aware compound expert failed")
    payload = {
        "status": "pass",
        "config": str(config_path.relative_to(EXP)),
        "config_sha256": canonical_hash(config),
        "compound_families": list(COMPOUND_FAMILIES),
        "expert_scores": expert_scores,
        "train_families": sorted(train),
        "transfer_families": sorted(transfer),
        "selftest_stdout": completed.stdout.strip().splitlines(),
    }
    write_json(EXP / "runs" / "smoke" / "summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("smoke", "incumbent", "specialists", "teacher-audit", "integration", "evaluate", "analyze"),
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    stage = "smoke" if args.smoke else args.stage
    if stage == "smoke":
        print(json.dumps(scientific_smoke(config, config_path), indent=2, sort_keys=True))
        return 0
    parser.error(
        f"stage {stage!r} is not wired yet; use the explicit commands in README until the next implementation checkpoint"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

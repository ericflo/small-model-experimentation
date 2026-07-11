#!/usr/bin/env python3
"""Smoke orchestrator: CPU selftests + config-freeze assertion.

python3 scripts/run.py --smoke

Asserts that configs/default.yaml still carries the preregistered constants
(reports/preregistration.md v2) so the freeze is enforced, not remembered.
GPU smoke stages live in band_calibrate.py / harvest.py --smoke.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]

# Frozen literals from reports/preregistration.md (v2). The config must match.
FROZEN = {
    ("harvest_pivot", "temperature"): 0.6,
    ("harvest_pivot", "top_p"): 0.95,
    ("harvest_pivot", "top_k"): 20,
    ("harvest_pivot", "n"): 16,  # amendment 2
    ("harvest_pivot", "think_budget"): 1024,
    ("harvest_pivot", "band_low"): 0.1,
    ("harvest_pivot", "band_high"): 0.9,
    ("harvest_pivot", "target_pool_rows"): 1200,
    ("harvest_pivot", "max_harvest_hours"): 5.0,
    ("detector", "min_repeats"): 4,
    ("detector", "min_total_repeated"): 60,
    ("detector", "max_period"): 1024,
    ("mining", "pivot_min_depth"): 2,  # amendment 1
    ("mining", "pivot_min_branch_rollouts"): 2,
    ("mining", "pivot_min_gap"): 0.5,
    ("mining", "pivot_max_nodes_per_prompt"): 3,  # amendment 1
    ("mining", "context_cap_tokens"): 6144,
    ("mining", "census_gate_group_rate"): 0.15,  # amendment 1
    ("mining", "census_gate_mixed_rate"): 0.30,  # amendment 1
    ("regularization", "rejected_strength"): 0.3,
    ("regularization", "chosen_strength"): 0.5,
    ("regularization", "max_train_fraction"): 0.70,
    ("regularization", "min_train_rows"): 600,
    ("regularization", "negative_label_min_rows"): 1200,
    ("train", "lora_r"): 256,
    ("train", "lora_alpha"): 128,
    ("train", "learning_rate"): 1.5e-5,
    ("train", "clip_epsilon_logits"): 2.0,
    ("train", "lambda_mse"): 0.4,
    ("train", "lambda_mse_target"): 0.05,
    ("train", "tau_mse_target"): 0.5,
    ("train", "early_stopping_chosen_win"): 0.85,  # amendment 3
    ("train", "early_stop_min_progress"): 0.2,  # amendment 3
    ("train", "max_seq_length"): 6144,
    ("eval", "whitebox_n_prompts"): 500,
    ("eval", "p1_success_gain"): 0.05,
    ("eval", "collapse_guard_max_rel_drop"): 0.10,
    ("menagerie", "quick_seeds_needed"): 3,
    ("menagerie", "quick_positive_floor"): 0.03,
    ("menagerie", "quick_negative_ceiling"): 0.01,
    ("menagerie", "medium_positive_floor"): 0.02,
}


def check_config() -> int:
    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    errors = 0
    for (section, key), frozen in FROZEN.items():
        actual = cfg.get(section, {}).get(key)
        if actual != frozen:
            print(f"FROZEN-MISMATCH {section}.{key}: config={actual!r} prereg={frozen!r}")
            errors += 1
    if (EXP / "configs" / "default.yaml").read_text().count("harvest_loop"):
        print("FROZEN-MISMATCH: stale harvest_loop section present (v1 residue)")
        errors += 1
    if "lm_head" in cfg["train"]["target_modules"]:
        print("FROZEN-MISMATCH: lm_head must not be a target module (merge-path constraint)")
        errors += 1
    print(f"config freeze: {'OK' if errors == 0 else f'{errors} mismatches'}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        print("Full pipeline stages are separate scripts; see README Run section.")
        return 0

    failures = check_config()
    for test in ["test_loopdetect.py", "test_pivotmine.py"]:
        result = subprocess.run([sys.executable, str(EXP / "tests" / test)])
        failures += result.returncode
    # Task-source sanity: oracle scores 1.0, garbage scores 0.0 on both sources.
    sys.path.insert(0, str(EXP / "src"))
    import tasks
    from gym.families import load as load_family
    gym_items = tasks.make_gym_items("caravan", 1, 70001, 1)
    oracle = load_family("caravan").oracle_atom(gym_items[0].payload)
    assert tasks.score_item(gym_items[0], f"<think>x</think>\n\n{oracle}") == 1.0
    assert tasks.score_item(gym_items[0], "<think>x</think>\n\nANSWER: junk") == 0.0
    code_items = tasks.make_code_items(2, 73001, 1)
    assert tasks.score_item(code_items[0], "<think>x</think>\n\nnothing") == 0.0
    print("task-source sanity: OK")
    print(f"SMOKE {'PASS' if failures == 0 else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

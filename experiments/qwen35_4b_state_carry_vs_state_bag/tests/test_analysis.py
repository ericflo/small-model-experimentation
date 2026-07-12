from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import (
    _deployment_comparison,
    _evaluation_bundles,
    _prefer_full_bundles,
    analyze_runs,
)
from src.config import config_sha256, load_config


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def test_relative_row_receipt_resolves_next_to_summary(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "eval"
            rows_path = run / "rows.jsonl"
            write_jsonl(rows_path, [{"id": "one", "k": 1, "split": "validation"}])
            summary = {
                "status": "EVALUATION_COMPLETE",
                "config_sha256": config_sha256(config),
                "row_file": "rows.jsonl",
                "train_arm": "carry",
                "eval_mode": "carry",
                "setup": {"checkpoint_metadata": {"train_seed": 1}},
            }
            (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            bundles = _evaluation_bundles(Path(directory), config_sha256(config))
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0]["rows"][0]["id"], "one")

    def test_full_evaluations_exclude_retained_pilots(self) -> None:
        bundles = [
            {"pilot": True, "name": "pilot"},
            {"pilot": False, "name": "full"},
        ]
        self.assertEqual(_prefer_full_bundles(bundles), [bundles[1]])

    def test_deployment_comparator_is_paired_and_seed_complete(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml")
        config["training"]["train_seeds"] = [1, 2, 3]
        config["evaluation"]["bootstrap_resamples"] = 2000
        expected_hash = config_sha256(config)
        bundles = []
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            for seed in (1, 2, 3):
                carry_rows = [
                    {
                        "id": f"item-{item}",
                        "split": "depth_extrapolation",
                        "depth": 5,
                        "k": 5,
                        "correct": True,
                    }
                    for item in range(20)
                ]
                bundles.append(
                    {
                        "train_seed": seed,
                        "train_arm": "carry",
                        "eval_mode": "carry",
                        "rows": carry_rows,
                    }
                )
                run = runs / f"sample_{seed}"
                sample_rows = [
                    {"id": f"item-{item}", "pass_at_n": False}
                    for item in range(20)
                ]
                write_jsonl(run / "rows.jsonl", sample_rows)
                (run / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "SAMPLE_MORE_COMPLETE",
                            "config_sha256": expected_hash,
                            "text_train_seed": seed,
                            "rows": "rows.jsonl",
                        }
                    ),
                    encoding="utf-8",
                )
            result = _deployment_comparison(
                bundles, runs, expected_hash, config
            )
            self.assertTrue(result["available"])
            self.assertEqual(result["training_seed_pairs"], 3)
            self.assertGreater(result["ci95"][0], 0)

    def test_fail_closed_ladder_reaches_mechanistic_positive(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml")
        config["training"]["train_seeds"] = [1, 2, 3]
        config["evaluation"]["bootstrap_resamples"] = 2000
        config["evaluation"]["primary_depths"] = [5]
        config["gates"]["min_positive_primary_depths"] = 1
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            for seed in (1, 2, 3):
                for arm in ("carry", "bag"):
                    run = runs / f"eval_{arm}_{seed}"
                    rows = []
                    for item in range(20):
                        item_id = f"item-{item}"
                        rows.append(
                            {
                                "id": item_id,
                                "split": "depth_extrapolation",
                                "depth": 5,
                                "k": 4,
                                "correct": False,
                                "node_step_accuracy": 1.0,
                                "joint_step_accuracy": 1.0,
                            }
                        )
                        rows.append(
                            {
                                "id": item_id,
                                "split": "depth_extrapolation",
                                "depth": 5,
                                "k": 5,
                                "correct": arm == "carry",
                                "node_step_accuracy": 1.0,
                                "joint_step_accuracy": 1.0,
                            }
                        )
                    rows_path = run / "rows.jsonl"
                    write_jsonl(rows_path, rows)
                    summary = {
                        "status": "EVALUATION_COMPLETE",
                        "config_sha256": config_sha256(config),
                        "row_file": str(rows_path),
                        "train_arm": arm,
                        "eval_mode": arm,
                        "checkpoint_k1_max_logit_abs_error": 0.0,
                        "setup": {
                            "checkpoint_metadata": {
                                "train_seed": seed,
                                "data_manifest_sha256": "same-data",
                                "training_prompt_tokens": 100,
                                "training_layer_token_applications": 1000,
                                "trainable_parameters": {
                                    "total": 10,
                                    "names_sha256": "same-names",
                                    "values_sha256": f"same-init-{seed}",
                                },
                            }
                        },
                        "counterfactual_swaps": (
                            {
                                "pairs": 20,
                                "baseline_accuracy": 0.8,
                                "donor_follow_rate": 0.8,
                                "recipient_preserve_rate": 0.1,
                            }
                            if arm == "carry"
                            else None
                        ),
                    }
                    (run / "summary.json").write_text(
                        json.dumps(summary), encoding="utf-8"
                    )
                    if arm == "carry":
                        edge_run = runs / f"edge_cut_{seed}"
                        edge_rows = edge_run / "rows.jsonl"
                        write_jsonl(edge_rows, rows)
                        edge_summary = {
                            **summary,
                            "row_file": str(edge_rows),
                            "eval_mode": "bag",
                            "counterfactual_swaps": None,
                        }
                        (edge_run / "summary.json").write_text(
                            json.dumps(edge_summary), encoding="utf-8"
                        )
            result = analyze_runs(config, runs, runs / "analysis" / "summary.json")
            self.assertEqual(result["verdict"], "MECHANISTIC_DEPTH_POSITIVE")
            self.assertEqual(result["carry_vs_bag"]["train_seed_pairs"], 3)
            self.assertEqual(result["carry_vs_bag"]["complete_primary_depths"], 1)
            self.assertGreater(result["unseen_k_scaling"]["ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()

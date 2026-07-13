from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import (  # noqa: E402
    _edge_cut_summary,
    _evaluation_bundles,
    _prefer_full_bundles,
    analyze_runs,
)
from src.config import config_sha256, load_config, source_contract_sha256  # noqa: E402


MANIFEST_HASH = "a" * 64
SOURCE_HASH = source_contract_sha256(ROOT)
REQUIREMENTS_HASH = hashlib.sha256(
    (ROOT.parents[1] / "requirements-training.lock.txt").read_bytes()
).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def scientific_config() -> dict:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["substrate"]["extrapolation_depths"] = [5]
    config["substrate"]["evaluation_examples_per_split"] = 20
    config["substrate"]["pilot_examples_per_split"] = 20
    config["evaluation"]["primary_depths"] = [5]
    config["evaluation"]["holdout_items_per_depth"] = 20
    config["evaluation"]["bootstrap_resamples"] = 2000
    config["gates"]["min_positive_primary_depths"] = 1
    return config


def checkpoint_metadata(
    config: dict,
    *,
    arm: str,
    seed: int,
    pilot: bool = False,
) -> dict:
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": SOURCE_HASH,
        "requirements_training_lock_sha256": REQUIREMENTS_HASH,
        "phase": "pilot" if pilot else "full",
        "train_arm": arm,
        "train_seed": seed,
        "step": int(
            config["training"]["pilot_steps" if pilot else "train_steps"]
        ),
        "pilot": pilot,
        "data_manifest_sha256": MANIFEST_HASH,
        "training_prompt_tokens": 100,
        "training_layer_token_applications": 1000,
        "training_order_sha256": f"same-order-{seed}",
        "gate_lineage": {
            "model_smoke": {
                "path": "/tmp/model-smoke.json",
                "sha256": "1" * 64,
                "receipt_identity_sha256": "2" * 64,
                "status": "MODEL_SMOKE_PASS",
                "phase": "g0",
            },
            **(
                {}
                if pilot
                else {
                    "pilot_promotion": {
                        "path": "/tmp/pilot-promotion.json",
                        "sha256": "3" * 64,
                        "receipt_identity_sha256": "4" * 64,
                        "status": "PILOT_PROMOTION_READY",
                        "phase": "pilot",
                    }
                }
            ),
        },
        "trainable_parameters": {
            "total": 10,
            "names_sha256": "same-names",
            "values_sha256": f"same-init-{seed}",
        },
        "delta_parameters": config["architecture"]["full_rank_delta"][
            "expected_parameters"
        ],
        "delta_target_manifest_sha256": "d" * 64,
        "delta_state_sha256": hashlib.sha256(
            f"delta-{arm}-{seed}".encode()
        ).hexdigest(),
        "loop_state_sha256": hashlib.sha256(
            f"loop-{arm}-{seed}".encode()
        ).hexdigest(),
    }
    metadata["checkpoint_identity_sha256"] = canonical_sha256(metadata)
    return metadata


def swap_rows(pair_count: int) -> list[dict]:
    return [
        {
            "pair_id": f"pair-{pair_index}",
            "direction": direction,
            "geometry_equal": True,
            "baseline_prediction": 0,
            "swapped_prediction": 1,
            "recipient_choice": 0,
            "donor_choice_in_recipient": 1,
            "baseline_correct": True,
            "baseline_donor_follow": False,
            "baseline_recipient_correct": True,
            "donor_follow": True,
            "recipient_preserve": False,
        }
        for pair_index in range(pair_count)
        for direction in ("a_to_b", "b_to_a")
    ]


def write_evaluation(
    run: Path,
    config: dict,
    *,
    metadata: dict,
    eval_mode: str,
    rows: list[dict],
    pilot: bool = False,
    swaps: list[dict] | None = None,
) -> None:
    rows_path = run / "rows.jsonl"
    write_jsonl(rows_path, rows)
    swap_summary = None
    if swaps is not None:
        swap_path = run / "counterfactual_swaps.jsonl"
        write_jsonl(swap_path, swaps)
        swap_summary = {
            "pairs": len(swaps) // 2,
            "directions": len(swaps),
            "baseline_accuracy": 1.0,
            "baseline_donor_follow_rate": 0.0,
            "donor_follow_rate": 1.0,
            "recipient_preserve_rate": 0.0,
            "counterfactual_swap_row_file": swap_path.name,
            "counterfactual_swap_row_file_sha256": sha256(swap_path),
        }
    summary = {
        "status": "EVALUATION_COMPLETE",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": SOURCE_HASH,
        "requirements_training_lock_sha256": REQUIREMENTS_HASH,
        "data_manifest_sha256": MANIFEST_HASH,
        "pilot": pilot,
        "phase": "pilot" if pilot else "full",
        "row_file": rows_path.name,
        "row_file_sha256": sha256(rows_path),
        "train_arm": metadata["train_arm"],
        "eval_mode": eval_mode,
        "expected_seed": metadata["train_seed"],
        "checkpoint_identity_sha256": metadata["checkpoint_identity_sha256"],
        "checkpoint_k1_max_logit_abs_error": 0.0,
        "setup": {"checkpoint_metadata": metadata},
        "counterfactual_swaps": swap_summary,
    }
    summary["receipt_identity_sha256"] = canonical_sha256(summary)
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def evaluation_rows(*, carry: bool) -> list[dict]:
    rows = []
    for item in range(20):
        query_kind = "node" if item % 2 == 0 else "checksum"
        common = {
            "family": "family",
            "template": "template",
            "depth": 5,
            "query_kind": query_kind,
            "correct_choice": 0,
            "prompt_tokens": 10,
            "layer_token_applications": 100,
            "full_top_is_answer": True,
            "node_step_accuracy": 1.0,
            "joint_step_accuracy": 1.0,
        }
        rows.append(
            {
                **common,
                "id": f"depth-{item}",
                "split": "depth_extrapolation",
                "k": 4,
                "correct": False,
            }
        )
        rows.append(
            {
                **common,
                "id": f"depth-{item}",
                "split": "depth_extrapolation",
                "k": 5,
                "correct": carry,
            }
        )
        rows.append(
            {
                **common,
                "id": f"joint-{item}",
                "split": "joint_holdout",
                "k": 5,
                "correct": carry,
            }
        )
    return rows


def pilot_rows(
    *, carry: bool, joint_step_accuracy: float = 1.0, primary_correct: bool | None = None
) -> list[dict]:
    converted_rows = []
    for row in evaluation_rows(carry=carry):
        converted = dict(row)
        converted["split"] = (
            "pilot_joint" if row["split"] == "joint_holdout" else "pilot_depth"
        )
        converted["joint_step_accuracy"] = joint_step_accuracy
        if (
            primary_correct is not None
            and converted["split"] == "pilot_depth"
            and converted["k"] == converted["depth"]
        ):
            converted["correct"] = primary_correct
        converted_rows.append(converted)
    return converted_rows


class AnalysisTests(unittest.TestCase):
    def test_pilot_analysis_never_enters_full_deployment_comparator(self) -> None:
        config = scientific_config()
        pilot_rows_carry = pilot_rows(carry=True)
        pilot_rows_bag = pilot_rows(carry=False)
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            carry_metadata = checkpoint_metadata(
                config, arm="carry", seed=7401, pilot=True
            )
            bag_metadata = checkpoint_metadata(
                config, arm="bag", seed=7401, pilot=True
            )
            write_evaluation(
                runs / "pilot_carry",
                config,
                metadata=carry_metadata,
                eval_mode="carry",
                rows=pilot_rows_carry,
                pilot=True,
                swaps=swap_rows(config["substrate"]["pilot_counterfactual_pairs"]),
            )
            write_evaluation(
                runs / "pilot_bag",
                config,
                metadata=bag_metadata,
                eval_mode="bag",
                rows=pilot_rows_bag,
                pilot=True,
            )
            with patch("src.analysis.is_confirmatory_config", return_value=True):
                result = analyze_runs(config, runs, runs / "analysis.json")
        self.assertEqual(result["verdict"], "PILOT_PROMOTION_READY")
        self.assertFalse(result["deployment_comparison"]["available"])

    def test_incomplete_pilot_is_non_scientific_and_cannot_close_capacity(self) -> None:
        config = scientific_config()
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            write_evaluation(
                runs / "pilot_carry",
                config,
                metadata=checkpoint_metadata(
                    config, arm="carry", seed=7401, pilot=True
                ),
                eval_mode="carry",
                rows=pilot_rows(carry=True),
                pilot=True,
                swaps=swap_rows(config["substrate"]["pilot_counterfactual_pairs"]),
            )
            with patch("src.analysis.is_confirmatory_config", return_value=True):
                result = analyze_runs(config, runs, runs / "analysis.json")
        self.assertEqual(result["verdict"], "PILOT_INCOMPLETE")
        self.assertFalse(result["pilot_gate"]["complete"])
        self.assertFalse(result["pilot_gate"]["capacity_branch_closed"])
        self.assertEqual(result["pilot_gate"]["capacity_conclusion"], "not_licensed")

    def test_only_valid_joint_state_miss_closes_capacity_branch(self) -> None:
        config = scientific_config()

        def run_case(*, joint_accuracy: float, carry_correct: bool) -> dict:
            with tempfile.TemporaryDirectory() as directory:
                runs = Path(directory)
                write_evaluation(
                    runs / "pilot_carry",
                    config,
                    metadata=checkpoint_metadata(
                        config, arm="carry", seed=7401, pilot=True
                    ),
                    eval_mode="carry",
                    rows=pilot_rows(
                        carry=True,
                        joint_step_accuracy=joint_accuracy,
                        primary_correct=carry_correct,
                    ),
                    pilot=True,
                    swaps=swap_rows(
                        config["substrate"]["pilot_counterfactual_pairs"]
                    ),
                )
                write_evaluation(
                    runs / "pilot_bag",
                    config,
                    metadata=checkpoint_metadata(
                        config, arm="bag", seed=7401, pilot=True
                    ),
                    eval_mode="bag",
                    rows=pilot_rows(carry=False),
                    pilot=True,
                )
                with patch("src.analysis.is_confirmatory_config", return_value=True):
                    return analyze_runs(config, runs, runs / "analysis.json")

        state_miss = run_case(joint_accuracy=0.0, carry_correct=True)
        self.assertEqual(state_miss["verdict"], "PILOT_STATE_FORMATION_MISS")
        self.assertTrue(state_miss["pilot_gate"]["complete"])
        self.assertTrue(state_miss["pilot_gate"]["capacity_branch_closed"])

        unrelated_miss = run_case(joint_accuracy=1.0, carry_correct=False)
        self.assertEqual(unrelated_miss["verdict"], "PILOT_PROMOTION_BLOCKED")
        self.assertFalse(unrelated_miss["pilot_gate"]["capacity_branch_closed"])
        self.assertIn(
            "joint_state_formed",
            unrelated_miss["pilot_gate"]["capacity_conclusion"],
        )

    def test_reduced_config_cannot_emit_scientific_verdict(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = analyze_runs(config, root / "runs", root / "summary.json")
        self.assertEqual(result["verdict"], "NONCONFIRMATORY_SMOKE_ONLY")
        self.assertEqual(result["phase"], "setup")

    def test_relative_row_receipt_resolves_next_to_summary(self) -> None:
        config = scientific_config()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "eval"
            metadata = checkpoint_metadata(config, arm="carry", seed=7411)
            row = {
                "id": "one",
                "k": 1,
                "split": "validation",
                "depth": 1,
                "correct": True,
                "family": "family",
                "template": "template",
                "query_kind": "node",
                "correct_choice": 0,
            }
            write_evaluation(
                run,
                config,
                metadata=metadata,
                eval_mode="carry",
                rows=[row],
            )
            bundles = _evaluation_bundles(
                Path(directory), config_sha256(config), config
            )
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0]["rows"][0]["id"], "one")

    def test_duplicate_evaluation_rows_are_rejected(self) -> None:
        config = scientific_config()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "eval"
            metadata = checkpoint_metadata(config, arm="carry", seed=7411)
            row = {
                "id": "duplicate",
                "k": 1,
                "split": "validation",
                "depth": 1,
                "correct": True,
            }
            write_evaluation(
                run,
                config,
                metadata=metadata,
                eval_mode="carry",
                rows=[row, dict(row)],
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate evaluation row key"):
                _evaluation_bundles(Path(directory), config_sha256(config), config)

    def test_checkpoint_phase_and_identity_are_enforced(self) -> None:
        config = scientific_config()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "eval"
            metadata = checkpoint_metadata(config, arm="carry", seed=7411)
            metadata["step"] = 1
            # Deliberately recompute the digest so step enforcement, not digest
            # enforcement, is the reason this fixture is rejected.
            metadata["checkpoint_identity_sha256"] = canonical_sha256(
                {k: v for k, v in metadata.items() if k != "checkpoint_identity_sha256"}
            )
            row = {
                "id": "one",
                "k": 1,
                "split": "validation",
                "depth": 1,
                "correct": True,
            }
            write_evaluation(
                run,
                config,
                metadata=metadata,
                eval_mode="carry",
                rows=[row],
            )
            with self.assertRaisesRegex(RuntimeError, "registered final full step"):
                _evaluation_bundles(Path(directory), config_sha256(config), config)

    def test_full_evaluations_exclude_retained_pilots(self) -> None:
        bundles = [{"pilot": True, "name": "pilot"}, {"pilot": False, "name": "full"}]
        self.assertEqual(_prefer_full_bundles(bundles), [bundles[1]])

    def test_identical_edge_cut_can_never_pass(self) -> None:
        config = scientific_config()
        bundles = []
        rows = [
            {
                "id": f"depth-{item}",
                "split": "depth_extrapolation",
                "depth": 5,
                "k": 5,
                "correct": True,
                "family": "family",
                "template": "template",
                "query_kind": "node" if item % 2 == 0 else "checksum",
                "correct_choice": 0,
                "prompt_tokens": 10,
                "layer_token_applications": 100,
            }
            for item in range(20)
        ]
        for seed in config["training"]["train_seeds"]:
            identity = hashlib.sha256(f"checkpoint-{seed}".encode()).hexdigest()
            for mode in ("carry", "bag"):
                bundles.append(
                    {
                        "train_seed": seed,
                        "train_arm": "carry",
                        "eval_mode": mode,
                        "rows": [dict(row) for row in rows],
                        "checkpoint_identity_sha256": identity,
                    }
                )
        result = _edge_cut_summary(bundles, config)
        self.assertTrue(result["available"])
        self.assertEqual(result["intact_minus_edge_cut"], 0.0)
        self.assertFalse(result["passes"])

    def test_fail_closed_ladder_reaches_mechanistic_positive(self) -> None:
        config = scientific_config()
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            for seed in config["training"]["train_seeds"]:
                carry_metadata = checkpoint_metadata(config, arm="carry", seed=seed)
                bag_metadata = checkpoint_metadata(config, arm="bag", seed=seed)
                write_evaluation(
                    runs / f"eval_carry_{seed}",
                    config,
                    metadata=carry_metadata,
                    eval_mode="carry",
                    rows=evaluation_rows(carry=True),
                    swaps=swap_rows(config["substrate"]["counterfactual_pairs"]),
                )
                write_evaluation(
                    runs / f"eval_bag_{seed}",
                    config,
                    metadata=bag_metadata,
                    eval_mode="bag",
                    rows=evaluation_rows(carry=False),
                )
                edge_rows = [
                    row
                    for row in evaluation_rows(carry=False)
                    if row["split"] == "depth_extrapolation" and row["k"] == row["depth"]
                ]
                write_evaluation(
                    runs / f"edge_cut_{seed}",
                    config,
                    metadata=carry_metadata,
                    eval_mode="bag",
                    rows=edge_rows,
                )
            with patch("src.analysis.is_confirmatory_config", return_value=True):
                result = analyze_runs(config, runs, runs / "analysis" / "summary.json")
            self.assertEqual(result["verdict"], "FULLRANK_CAUSAL_DEPTH_POSITIVE")
            self.assertEqual(result["carry_vs_bag"]["model_seed_count"], 3)
            self.assertEqual(result["carry_vs_bag"]["unique_tasks"], 20)
            self.assertTrue(result["trained_checkpoint_edge_cut"]["passes"])
            self.assertTrue(result["counterfactual_swaps"]["passes"])
            self.assertTrue(
                all(
                    seed["bootstrap_unit"]
                    == "counterfactual_pair_mean_over_two_directions"
                    for seed in result["counterfactual_swaps"]["seeds"]
                )
            )


if __name__ == "__main__":
    unittest.main()

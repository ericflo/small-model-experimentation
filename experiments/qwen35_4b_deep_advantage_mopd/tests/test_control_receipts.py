from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import authorize_benchmark  # noqa: E402
import control_receipts  # noqa: E402
import run as run_script  # noqa: E402
from control_receipts import validate_control_training_receipt  # noqa: E402
from io_utils import sha256_file  # noqa: E402


CONFIG = {
    "mopd": {
        "updates_per_round": 20,
        "grad_accum": 4,
        "capability_units_per_round": 60,
        "anchor_units_per_round": 20,
        "max_length": 1024,
        "max_target_positions": 256,
    }
}
TARGET_PRESSURE = 0.05


def _receipt(arm: str) -> dict:
    capability_role = "route_control" if arm == "non_advantage_route" else "capability"
    capability_target = {
        "non_advantage_route": "deep",
        "wrong_teacher": "quick",
        "offpolicy_sft": "deep",
    }[arm]
    target_field = "target_policy" if arm == "offpolicy_sft" else "target"
    rows = []
    for index in range(60):
        rows.append(
            {
                "sample_id": f"capability-{index:03d}",
                "role": capability_role,
                "kind": "atom",
                target_field: capability_target,
                "prompt_tokens_truncated": 0,
                "target_positions": 32,
            }
        )
    for index in range(20):
        rows.append(
            {
                "sample_id": f"anchor-{index:03d}",
                "role": "anchor",
                "kind": "atom",
                target_field: "student_anchor" if arm == "offpolicy_sft" else "soup",
                "prompt_tokens_truncated": 0,
                "target_positions": 16,
            }
        )
    for index, row in enumerate(rows):
        row["micro_step"] = index + 1
    probe_ids = [f"capability-{index:03d}" for index in range(6)] + [
        f"anchor-{index:03d}" for index in range(2)
    ]
    if arm == "offpolicy_sft":
        target_counts = {"deep": 60, "student_anchor": 20}
        pressure_probe = {
            "unit_ids": probe_ids,
            "role_counts": {"capability": 6, "anchor": 2},
            "target_counts": {"deep": 6, "student_anchor": 2},
            "geometry": "6_capability_2_anchor",
            "matching_statistic": "initial_mean_objective_loss",
        }
        schema = 1
        method = "offpolicy_best_selection_continuation_sft"
        assignment = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    else:
        target_counts = {
            "quick": 60 if arm == "wrong_teacher" else 0,
            "deep": 60 if arm == "non_advantage_route" else 0,
            "soup": 20,
        }
        pressure_probe = {
            "unit_ids": probe_ids,
            "role_counts": {
                "capability": 6 if arm == "wrong_teacher" else 0,
                "route_control": 6 if arm == "non_advantage_route" else 0,
                "anchor": 2,
            },
            "target_counts": {
                "quick": 6 if arm == "wrong_teacher" else 0,
                "deep": 6 if arm == "non_advantage_route" else 0,
                "soup": 2,
            },
            "geometry": "6_teacher_2_anchor",
            "matching_statistic": "initial_mean_objective_loss",
        }
        schema = 2
        method = "deep_advantage_routed_corrected_teacher_topk_reverse_kl"
        assignment = hashlib.sha256(
            json.dumps(
                sorted((row["sample_id"], row["target"]) for row in rows),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    return {
        "schema_version": schema,
        "method": method,
        "arm": arm,
        "round": 0,
        "seed": {"non_advantage_route": 64, "wrong_teacher": 65, "offpolicy_sft": 66}[arm],
        "requested_updates": 20,
        "completed_updates": 20,
        "consume_once_units": 80,
        "consume_once_verified": True,
        "assignment_sha256": assignment,
        "target_counts": target_counts,
        "initial_probe": {"mean_loss": 0.2, "unit_count": 8},
        "final_probe": {"mean_loss": 0.1, "unit_count": 8},
        "pressure_probe": pressure_probe,
        "target_initial_loss": TARGET_PRESSURE,
        "backward_loss_scale": 0.25,
        "round_gate": {"passed": True, "completed_all_updates": True},
        "unit_ledger": rows,
    }


def _validate(receipt: dict, arm: str, *, pressure: float = TARGET_PRESSURE) -> None:
    helper = (
        "_canonical_offpolicy_ledger"
        if arm == "offpolicy_sft"
        else "_canonical_mopd_ledger"
    )
    kwargs = {
        "source_manifest": Path("round.json"),
        "round_index": int(receipt["round"]),
        "seed": int(receipt["seed"]),
    }
    if arm == "offpolicy_sft":
        kwargs["base_model"] = Path("base")
    else:
        kwargs["target_cache"] = Path("targets.pt")
    with mock.patch.object(
        control_receipts, helper, return_value=receipt["unit_ledger"]
    ):
        validate_control_training_receipt(
            receipt,
            config=CONFIG,
            arm=arm,
            expected_target_initial_loss=pressure,
            **kwargs,
        )


def _mopd_bound_fixture(root: Path, arm: str) -> tuple[dict, dict]:
    receipt = _receipt(arm)
    config_sha = "c" * 64
    manifest = root / "round.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "online_advantage_training_round",
                "config_sha256": config_sha,
                "round": 0,
            }
        ),
        encoding="utf-8",
    )
    role = "route_control" if arm == "non_advantage_route" else "capability"
    samples = []
    for index in range(60):
        samples.append(
            {
                "id": f"capability-{index:03d}",
                "meta": {
                    "role": role,
                    "kind": "atom",
                    "level": 5,
                    "prompt_tokens_truncated": 0,
                },
                "positions": torch.arange(32, dtype=torch.int32),
                "targets": {"quick": {}, "deep": {}},
            }
        )
    for index in range(20):
        samples.append(
            {
                "id": f"anchor-{index:03d}",
                "meta": {
                    "role": "anchor",
                    "kind": "atom",
                    "level": 5,
                    "prompt_tokens_truncated": 0,
                },
                "positions": torch.arange(16, dtype=torch.int32),
                "targets": {"soup": {}},
            }
        )
    target_cache = root / "targets.pt"
    torch.save(
        {
            "stage": "matched_all_policy_topk_cache",
            "config_sha256": config_sha,
            "round": 0,
            "round_manifest_sha256": sha256_file(manifest),
            "samples": samples,
        },
        target_cache,
    )
    units = []
    for sample in samples:
        target = (
            "soup"
            if sample["meta"]["role"] == "anchor"
            else "quick" if arm == "wrong_teacher" else "deep"
        )
        units.append((sample, target))
    random.Random(receipt["seed"]).shuffle(units)
    ledger = [
        {
            "micro_step": index + 1,
            "sample_id": sample["id"],
            "target": target,
            "role": sample["meta"]["role"],
            "kind": sample["meta"]["kind"],
            "level": sample["meta"]["level"],
            "prompt_tokens_truncated": 0,
            "target_positions": int(sample["positions"].numel()),
        }
        for index, (sample, target) in enumerate(units)
    ]
    receipt.update(
        {
            "config_sha256": config_sha,
            "target_cache": str(target_cache.resolve()),
            "target_cache_sha256": sha256_file(target_cache),
            "unit_ledger": ledger,
            "assignment_sha256": hashlib.sha256(
                json.dumps(
                    sorted((row["sample_id"], row["target"]) for row in ledger),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    )
    return receipt, {
        "source_manifest": manifest,
        "target_cache": target_cache,
        "round_index": 0,
        "seed": receipt["seed"],
    }


def _offpolicy_bound_fixture(root: Path) -> tuple[dict, dict]:
    receipt = _receipt("offpolicy_sft")
    config_sha = "d" * 64
    base = root / "base"
    base.mkdir()
    (base / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
    source_units = []
    canonical = []
    for index in range(60):
        state_id = f"capability-{index:03d}"
        score = float(index) / 100.0
        source_units.append(
            {
                "state_id": state_id,
                "role": "capability",
                "kind": "atom",
                "level": 5,
                "state": {
                    "kind": "atom",
                    "exact_prompt_token_ids": [1, 2],
                },
                "offpolicy_target": {
                    "policy": "deep",
                    "terminal_score": score,
                    "completion_ids": list(range(10, 42)),
                    "injected_token_ids": [],
                },
            }
        )
        canonical.append(
            {
                "sample_id": state_id,
                "role": "capability",
                "kind": "atom",
                "target_policy": "deep",
                "prompt_tokens_truncated": 0,
                "target_positions": 32,
                "target_terminal_score": score,
            }
        )
    for index in range(20):
        state_id = f"anchor-{index:03d}"
        score = 0.5 + float(index) / 100.0
        source_units.append(
            {
                "state_id": state_id,
                "role": "anchor",
                "kind": "atom",
                "level": 5,
                "state": {
                    "kind": "atom",
                    "exact_prompt_token_ids": [1, 2],
                    "student_suffix_ids": list(range(50, 66)),
                    "student_output": {
                        "injected_token_ids": [],
                        "n_thinking_tokens": 0,
                    },
                    "prefix_length": 0,
                    "student_terminal_score": score,
                },
            }
        )
        canonical.append(
            {
                "sample_id": state_id,
                "role": "anchor",
                "kind": "atom",
                "target_policy": "student_anchor",
                "prompt_tokens_truncated": 0,
                "target_positions": 16,
                "target_terminal_score": score,
            }
        )
    manifest = root / "round.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "online_advantage_training_round",
                "config_sha256": config_sha,
                "round": 0,
                "units": source_units,
            }
        ),
        encoding="utf-8",
    )
    random.Random(receipt["seed"]).shuffle(canonical)
    ledger = [
        {"micro_step": index + 1, **row}
        for index, row in enumerate(canonical)
    ]
    receipt.update(
        {
            "config_sha256": config_sha,
            "base_model": str(base.resolve()),
            "base_merge_receipt_sha256": sha256_file(base / "merge_receipt.json"),
            "round_manifest": str(manifest.resolve()),
            "round_manifest_sha256": sha256_file(manifest),
            "unit_ledger": ledger,
            "assignment_sha256": hashlib.sha256(
                json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    return receipt, {
        "source_manifest": manifest,
        "base_model": base,
        "round_index": 0,
        "seed": receipt["seed"],
    }


class ControlReceiptTests(unittest.TestCase):
    def test_all_control_receipts_realize_exact_frozen_semantics(self):
        for arm in ("non_advantage_route", "wrong_teacher", "offpolicy_sft"):
            with self.subTest(arm=arm):
                _validate(_receipt(arm), arm)

    def test_semantic_mutations_fail_closed(self):
        mutations = {
            "pressure": lambda row: row.update(target_initial_loss=0.051),
            "scale": lambda row: row.update(backward_loss_scale=0.251),
            "target_counts": lambda row: row["target_counts"].update(deep=59),
            "probe": lambda row: row["pressure_probe"].update(geometry="unregistered"),
            "truncation": lambda row: row["unit_ledger"][0].update(
                prompt_tokens_truncated=1
            ),
            "duplicate": lambda row: row["unit_ledger"][1].update(
                sample_id=row["unit_ledger"][0]["sample_id"]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = _receipt("non_advantage_route")
                mutate(receipt)
                with self.assertRaises(ValueError):
                    _validate(receipt, "non_advantage_route")

    def test_mopd_ledger_is_replayed_from_bound_cache_not_self_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt, inputs = _mopd_bound_fixture(
                Path(temporary), "non_advantage_route"
            )
            validate_control_training_receipt(
                receipt,
                config=CONFIG,
                arm="non_advantage_route",
                expected_target_initial_loss=TARGET_PRESSURE,
                **inputs,
            )
            for mutation in ("sample_id", "kind", "order"):
                forged = copy.deepcopy(receipt)
                if mutation == "sample_id":
                    forged["unit_ledger"][0]["sample_id"] = "forged-state"
                elif mutation == "kind":
                    forged["unit_ledger"][0]["kind"] = "forged-kind"
                else:
                    forged["unit_ledger"][0], forged["unit_ledger"][1] = (
                        forged["unit_ledger"][1],
                        forged["unit_ledger"][0],
                    )
                    forged["unit_ledger"][0]["micro_step"] = 1
                    forged["unit_ledger"][1]["micro_step"] = 2
                forged["assignment_sha256"] = hashlib.sha256(
                    json.dumps(
                        sorted(
                            (row["sample_id"], row["target"])
                            for row in forged["unit_ledger"]
                        ),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ValueError, "canonical source replay"
                ):
                    validate_control_training_receipt(
                        forged,
                        config=CONFIG,
                        arm="non_advantage_route",
                        expected_target_initial_loss=TARGET_PRESSURE,
                        **inputs,
                    )

    def test_offpolicy_ledger_is_replayed_from_bound_manifest_not_self_hash(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=object()
        ):
            receipt, inputs = _offpolicy_bound_fixture(Path(temporary))
            validate_control_training_receipt(
                receipt,
                config=CONFIG,
                arm="offpolicy_sft",
                expected_target_initial_loss=TARGET_PRESSURE,
                **inputs,
            )
            for mutation in ("sample_id", "target_terminal_score", "order"):
                forged = copy.deepcopy(receipt)
                if mutation == "sample_id":
                    forged["unit_ledger"][0]["sample_id"] = "forged-state"
                elif mutation == "target_terminal_score":
                    forged["unit_ledger"][0]["target_terminal_score"] += 0.25
                else:
                    forged["unit_ledger"][0], forged["unit_ledger"][1] = (
                        forged["unit_ledger"][1],
                        forged["unit_ledger"][0],
                    )
                    forged["unit_ledger"][0]["micro_step"] = 1
                    forged["unit_ledger"][1]["micro_step"] = 2
                forged["assignment_sha256"] = hashlib.sha256(
                    json.dumps(
                        forged["unit_ledger"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ValueError, "canonical source replay"
                ):
                    validate_control_training_receipt(
                        forged,
                        config=CONFIG,
                        arm="offpolicy_sft",
                        expected_target_initial_loss=TARGET_PRESSURE,
                        **inputs,
                    )

    def test_runner_rejects_reused_mopd_adapter_with_wrong_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            target_cache = root / "targets.pt"
            target_cache.write_bytes(b"targets")
            base = root / "base"
            base.mkdir()
            receipt = _receipt("non_advantage_route")
            receipt.update(
                {
                    "base_model": str(base.resolve()),
                    "target_cache_sha256": sha256_file(target_cache),
                }
            )
            (adapter / "training_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            with mock.patch.object(run_script, "_adapter_complete", return_value=True), mock.patch.object(
                run_script, "_merge_round_adapter"
            ) as merge, mock.patch.object(
                control_receipts,
                "_canonical_mopd_ledger",
                return_value=receipt["unit_ledger"],
            ):
                with self.assertRaisesRegex(SystemExit, "target initial pressure"):
                    run_script._train_mopd_checkpoint(
                        CONFIG,
                        root / "config.yaml",
                        base_model=base,
                        target_cache=target_cache,
                        adapter=adapter,
                        merged=root / "merged",
                        round_index=0,
                        seed=64,
                        arm="non_advantage_route",
                        source_manifest=root / "round.json",
                        target_initial_loss=0.06,
                    )
            merge.assert_not_called()

    def test_runner_rejects_reused_offpolicy_adapter_with_wrong_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            manifest = root / "round.json"
            manifest.write_text("{}\n", encoding="utf-8")
            base = root / "base"
            base.mkdir()
            receipt = _receipt("offpolicy_sft")
            receipt.update(
                {
                    "base_model": str(base.resolve()),
                    "round_manifest_sha256": sha256_file(manifest),
                }
            )
            (adapter / "training_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            with mock.patch.object(run_script, "_adapter_complete", return_value=True), mock.patch.object(
                run_script, "_merge_round_adapter"
            ) as merge, mock.patch.object(
                control_receipts,
                "_canonical_offpolicy_ledger",
                return_value=receipt["unit_ledger"],
            ):
                with self.assertRaisesRegex(SystemExit, "target initial pressure"):
                    run_script._train_offpolicy_checkpoint(
                        CONFIG,
                        root / "config.yaml",
                        base_model=base,
                        manifest=manifest,
                        adapter=adapter,
                        merged=root / "merged",
                        round_index=0,
                        seed=66,
                        target_initial_loss=0.06,
                    )
            merge.assert_not_called()

    def test_runner_preserves_a_semantically_bound_failed_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            target_cache = root / "targets.pt"
            target_cache.write_bytes(b"targets")
            base = root / "base"
            base.mkdir()
            receipt = _receipt("non_advantage_route")
            receipt.update(
                {
                    "base_model": str(base.resolve()),
                    "target_cache_sha256": sha256_file(target_cache),
                    "completed_updates": 0,
                    "round_gate": {
                        "passed": False,
                        "completed_all_updates": False,
                        "unsafe_reason": "non_finite_gradient",
                    },
                }
            )
            (adapter / "training_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            with mock.patch.object(run_script, "_adapter_complete", return_value=True), mock.patch.object(
                run_script, "_merge_round_adapter"
            ) as merge, mock.patch.object(
                control_receipts,
                "_canonical_mopd_ledger",
                return_value=receipt["unit_ledger"],
            ):
                observed, merged = run_script._train_mopd_checkpoint(
                    CONFIG,
                    root / "config.yaml",
                    base_model=base,
                    target_cache=target_cache,
                    adapter=adapter,
                    merged=root / "merged",
                    round_index=0,
                    seed=64,
                    arm="non_advantage_route",
                    source_manifest=root / "round.json",
                    target_initial_loss=TARGET_PRESSURE,
                )
            self.assertIsNone(merged)
            self.assertEqual(observed["round_gate"]["unsafe_reason"], "non_finite_gradient")
            merge.assert_not_called()

    def test_benchmark_audit_rejects_wrong_control_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text("synthetic\n", encoding="utf-8")
            base = root / "base"
            base.mkdir()
            (base / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
            target = root / "targets.pt"
            target.write_bytes(b"targets")
            receipt_path = root / "training_receipt.json"
            receipt = _receipt("wrong_teacher")
            receipt.update(
                {
                    "config": str(config_path.resolve()),
                    "config_sha256": sha256_file(config_path),
                    "base_model": str(base.resolve()),
                    "base_merge_receipt_sha256": sha256_file(
                        base / "merge_receipt.json"
                    ),
                    "target_cache": str(target.resolve()),
                    "target_cache_sha256": sha256_file(target),
                }
            )
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch.object(
                control_receipts,
                "_canonical_mopd_ledger",
                return_value=receipt["unit_ledger"],
            ), self.assertRaisesRegex(ValueError, "target initial pressure"):
                authorize_benchmark._audit_mopd_training_receipt(
                    receipt_path,
                    config=CONFIG,
                    config_path=config_path,
                    base_model=base,
                    round_manifest=root / "round.json",
                    target_cache=target,
                    round_index=0,
                    seed=65,
                    arm="wrong_teacher",
                    recorded_gate=receipt["round_gate"],
                    expected_target_initial_loss=0.06,
                )


if __name__ == "__main__":
    unittest.main()

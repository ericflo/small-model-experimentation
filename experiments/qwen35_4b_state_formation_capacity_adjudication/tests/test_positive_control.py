from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import gpu_runner  # noqa: E402
from src.adaptation import microbatch_dropout_seed  # noqa: E402
from src.config import load_config  # noqa: E402
from src.gpu_runner import (  # noqa: E402
    _parameter_delta_baseline,
    _parameter_delta_norm_receipt,
    _parameter_norm_receipt,
    _positive_control_diagnostic_context,
    _positive_control_probe_steps,
    _positive_control_rows,
    _positive_control_schedule,
    _state_accuracy_counts,
    _summarize_positive_control_records,
)


class FreshPositiveControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    @staticmethod
    def manifest(fingerprints=()) -> dict:
        return {
            "files": {
                "train": {"structural_fingerprints": list(fingerprints)},
                "validation": {"structural_fingerprints": []},
            }
        }

    def test_fresh_seed_73991_builds_exact_balanced_48_row_factorial(self) -> None:
        rows, receipt = _positive_control_rows(self.config, self.manifest())
        repeated, repeated_receipt = _positive_control_rows(self.config, self.manifest())
        self.assertEqual(rows, repeated)
        self.assertEqual(receipt, repeated_receipt)
        self.assertEqual(receipt["seed"], 73991)
        self.assertEqual(receipt["rows"], 48)
        self.assertEqual(
            receipt["canonical_rows_sha256"],
            "581dadcb7bba053d94a849e42e0490127c7e0de199311d053e7585adcc78ef41",
        )
        self.assertEqual(receipt["cross_result_structural_overlap"], 0)
        self.assertEqual({row["split"] for row in rows}, {"setup_positive_control"})
        grid = Counter(
            (row["depth"], row["query_kind"], row["family"], row["template"])
            for row in rows
        )
        self.assertEqual(len(grid), 3 * 2 * 2 * 2)
        self.assertEqual(set(grid.values()), {2})
        self.assertEqual({depth for depth, *_ in grid}, {2, 3, 4})
        self.assertEqual({query for _, query, *_ in grid}, {"node", "checksum"})
        self.assertEqual(
            {family for _, _, family, _ in grid},
            set(self.config["substrate"]["train_families"]),
        )
        self.assertEqual(
            {template for _, _, _, template in grid},
            set(self.config["substrate"]["train_templates"]),
        )
        self.assertEqual(
            len({row["structural_fingerprint"] for row in rows}),
            len(rows),
        )

    def test_any_structural_overlap_with_result_data_fails_closed(self) -> None:
        rows, _ = _positive_control_rows(self.config, self.manifest())
        with self.assertRaisesRegex(RuntimeError, "overlap result data"):
            _positive_control_rows(
                self.config,
                self.manifest([rows[0]["structural_fingerprint"]]),
            )

    def test_confirmatory_schedule_is_exact_256_by_16_global_cycle(self) -> None:
        rows, _ = _positive_control_rows(self.config, self.manifest())
        events = _positive_control_schedule(
            rows, updates=256, accumulation=16, model_seed=7411
        )
        self.assertEqual(len(events), 4096)
        self.assertEqual(
            [event["microbatch_index"] for event in events], list(range(1, 4097))
        )
        by_step = Counter(event["optimizer_step"] for event in events)
        self.assertEqual(set(by_step), set(range(1, 257)))
        self.assertEqual(set(by_step.values()), {16})
        self.assertEqual(
            [event["microbatch_in_step"] for event in events[:16]],
            list(range(1, 17)),
        )
        exposures = Counter(event["row_index"] for event in events)
        self.assertEqual([exposures[index] for index in range(16)], [86] * 16)
        self.assertEqual([exposures[index] for index in range(16, 48)], [85] * 32)
        self.assertEqual(
            Counter(event["k"] for event in events), {2: 1368, 3: 1368, 4: 1360}
        )
        self.assertEqual(
            [event["id"] for event in events[:256]],
            [str(rows[index % 48]["id"]) for index in range(256)],
        )
        for event in (events[0], events[255], events[2047], events[-1]):
            self.assertEqual(
                event["dropout_seed"],
                microbatch_dropout_seed(
                    7411, event["microbatch_index"], event["id"], event["k"]
                ),
            )
        schedule_digest = hashlib.sha256(
            b"".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
                for event in events
            )
        ).hexdigest()
        self.assertEqual(
            schedule_digest,
            "6dbe928a0b4818e079e6f5b8acab1b8408cbeb7f0a9e3b89a3cd7948b006f06f",
        )
        self.assertEqual(
            [events[index]["dropout_seed"] for index in (0, 255, 2047, 4095)],
            [
                6232283538495441173,
                8820665304680524663,
                7568421441250656730,
                665791658563906471,
            ],
        )

    def test_smoke_schedule_honors_two_by_two_geometry(self) -> None:
        rows, _ = _positive_control_rows(self.config, self.manifest())
        events = _positive_control_schedule(
            rows, updates=2, accumulation=2, model_seed=7411
        )
        self.assertEqual(len(events), 4)
        self.assertEqual(
            [(event["optimizer_step"], event["microbatch_in_step"]) for event in events],
            [(1, 1), (1, 2), (2, 1), (2, 2)],
        )

    def test_fixed_probe_positions_are_nonselective(self) -> None:
        self.assertEqual(_positive_control_probe_steps(256), (0, 1, 16, 64, 128, 256))
        self.assertEqual(_positive_control_probe_steps(2), (0, 1, 2))
        with self.assertRaisesRegex(RuntimeError, "must be positive"):
            _positive_control_probe_steps(0)

    @staticmethod
    def logits(predictions: torch.Tensor, classes: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(predictions, num_classes=classes).float() * 10.0

    def test_state_scorer_separates_trajectory_terminal_and_joint(self) -> None:
        targets = {
            "node": torch.tensor([[1, 2, 3]]),
            "phase": torch.tensor([[0, 1, 0]]),
            "checksum": torch.tensor([[4, 5, 6]]),
        }
        node = self.logits(torch.tensor([[1, 2, 7]]), 16)
        phase = self.logits(torch.tensor([[0, 1, 0]]), 2)
        checksum = self.logits(torch.tensor([[4, 5, 6]]), 8)
        scored = _state_accuracy_counts(node, phase, checksum, targets)
        self.assertEqual(scored["terminal"], {
            "node": 0, "phase": 1, "checksum": 1, "joint": 0, "rows": 1,
        })
        self.assertEqual(scored["trajectory"], {
            "node": 2, "phase": 3, "checksum": 3, "joint": 2, "steps": 3,
        })
        one_head_wrong = _state_accuracy_counts(
            self.logits(targets["node"], 16),
            self.logits(targets["phase"], 2),
            self.logits(torch.tensor([[4, 5, 7]]), 8),
            targets,
        )
        self.assertEqual(one_head_wrong["terminal"]["node"], 1)
        self.assertEqual(one_head_wrong["terminal"]["phase"], 1)
        self.assertEqual(one_head_wrong["terminal"]["checksum"], 0)
        self.assertEqual(one_head_wrong["terminal"]["joint"], 0)

    def test_state_scorer_rejects_broadcast_class_dtype_range_and_nonfinite(self) -> None:
        targets = {
            "node": torch.tensor([[1, 2]]),
            "phase": torch.tensor([[0, 1]]),
            "checksum": torch.tensor([[4, 5]]),
        }
        valid = (
            self.logits(targets["node"], 16),
            self.logits(targets["phase"], 2),
            self.logits(targets["checksum"], 8),
        )
        bad_cases = []
        bad_cases.append((valid[0][:, :1], valid[1], valid[2], targets))
        bad_cases.append((valid[0][..., :15], valid[1], valid[2], targets))
        float_targets = dict(targets, node=targets["node"].float())
        bad_cases.append((*valid, float_targets))
        out_of_range = dict(targets, checksum=torch.tensor([[4, 8]]))
        bad_cases.append((*valid, out_of_range))
        nonfinite = valid[0].clone()
        nonfinite[0, 0, 0] = float("nan")
        bad_cases.append((nonfinite, valid[1], valid[2], targets))
        for case in bad_cases:
            with self.subTest(case=len(case)):
                with self.assertRaises(RuntimeError):
                    _state_accuracy_counts(*case)

    def test_source_geometry_has_no_inner_zero_clip_step_or_early_stop(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "positive_control"
        )
        loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
        update_loop = next(
            node for node in loops
            if isinstance(node.target, ast.Name) and node.target.id == "update"
        )
        inner = next(node for node in update_loop.body if isinstance(node, ast.For))

        def called_names(node: ast.AST) -> list[str]:
            names = []
            for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                if isinstance(call.func, ast.Attribute):
                    names.append(call.func.attr)
                elif isinstance(call.func, ast.Name):
                    names.append(call.func.id)
            return names

        inner_calls = called_names(inner)
        self.assertNotIn("zero_grad", inner_calls)
        self.assertNotIn("clip_grad_norm_", inner_calls)
        self.assertNotIn("step", inner_calls)
        update_calls = called_names(update_loop)
        self.assertEqual(update_calls.count("clip_grad_norm_"), 2)
        self.assertEqual(update_calls.count("step"), 1)
        self.assertEqual(update_calls.count("zero_grad"), 1)
        self.assertNotIn("break", [type(node).__name__.lower() for node in ast.walk(function)])
        self.assertIn('accumulation = int(training["gradient_accumulation"])', source)
        self.assertIn("scaled_loss = objective_loss / accumulation", source)
        self.assertIn('dropout_seed = int(event["dropout_seed"])', source)
        self.assertIn("wrapper.zero_grad(set_to_none=True)", source)

    def test_parameter_receipts_measure_zero_output_and_common_deltas(self) -> None:
        class TinyWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adaptation = torch.nn.Module()
                self.adaptation.down = torch.nn.ModuleDict({"d000": torch.nn.Linear(3, 2, bias=False)})
                self.adaptation.up = torch.nn.ModuleDict({"d000": torch.nn.Linear(2, 3, bias=False)})
                torch.nn.init.zeros_(self.adaptation.up["d000"].weight)
                self.common = torch.nn.Linear(3, 3, bias=False)

        wrapper = TinyWrapper()
        baseline = _parameter_delta_baseline(wrapper)
        self.assertIn("adaptation.down.d000.weight", baseline)
        self.assertNotIn("adaptation.up.d000.weight", baseline)
        initial = _parameter_delta_norm_receipt(wrapper, baseline)
        self.assertEqual(initial["adaptation_input"]["l2_delta_norm"], 0.0)
        self.assertEqual(initial["adaptation_output"]["l2_delta_norm"], 0.0)
        self.assertEqual(initial["common_state"]["l2_delta_norm"], 0.0)
        with torch.no_grad():
            wrapper.adaptation.up["d000"].weight[0, 0] = 0.25
            wrapper.adaptation.down["d000"].weight[0, 0] += 0.5
            wrapper.common.weight[0, 0] += 0.75
        changed = _parameter_delta_norm_receipt(wrapper, baseline)
        self.assertGreater(changed["adaptation_input"]["l2_delta_norm"], 0.0)
        self.assertEqual(changed["adaptation_output"]["l2_delta_norm"], 0.25)
        self.assertGreater(changed["common_state"]["l2_delta_norm"], 0.0)
        norms = _parameter_norm_receipt(wrapper)
        self.assertEqual(norms["adaptation_output"]["nonzero_tensors"], 1)

    def test_diagnostic_context_restores_cpu_rng_mode_and_suspension(self) -> None:
        class FakeAdaptation(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.suspended_depth = 0

            @contextlib.contextmanager
            def suspended(self):
                self.suspended_depth += 1
                try:
                    yield
                finally:
                    self.suspended_depth -= 1

        class FakeWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.adaptation = FakeAdaptation()

        wrapper = FakeWrapper()
        wrapper.train()
        torch.manual_seed(9817)
        rng_before = torch.get_rng_state().clone()
        with _positive_control_diagnostic_context(
            wrapper, adaptation_enabled=False
        ):
            self.assertFalse(wrapper.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(wrapper.adaptation.suspended_depth, 1)
            torch.rand(7)
        self.assertTrue(wrapper.training)
        self.assertEqual(wrapper.adaptation.suspended_depth, 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

        wrapper.eval()
        rng_before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "synthetic diagnostic"):
            with _positive_control_diagnostic_context(
                wrapper, adaptation_enabled=True
            ):
                torch.rand(3)
                raise RuntimeError("synthetic diagnostic")
        self.assertFalse(wrapper.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_mixed_depth_summary_uses_exact_trajectory_denominator(self) -> None:
        widths = {"node": 16, "phase": 2, "checksum": 8}

        def record(depth: int, joint_steps: int, joint_final: int) -> dict:
            terminal = {
                "node": joint_final,
                "phase": joint_final,
                "checksum": joint_final,
                "joint": joint_final,
                "rows": 1,
            }
            trajectory = {
                "node": joint_steps,
                "phase": joint_steps,
                "checksum": joint_steps,
                "joint": joint_steps,
                "steps": depth,
            }
            histograms = {
                head: {"prediction": [0] * width, "target": [0] * width}
                for head, width in widths.items()
            }
            return {
                "depth": depth,
                "state": {
                    "terminal": terminal,
                    "trajectory": trajectory,
                    "histograms": histograms,
                },
                "objective_loss": float(depth),
                "state_loss": float(depth + 1),
                "fixed_point_loss": 0.0,
            }

        records = [record(2, 2, 1), record(4, 1, 0)]
        overall = _summarize_positive_control_records(records)
        self.assertEqual(overall["rows"], 2)
        self.assertEqual(overall["trajectory_steps"], 6)
        self.assertEqual(overall["joint_final_accuracy"], 0.5)
        self.assertEqual(overall["joint_trajectory_accuracy"], 3 / 6)
        by_depth = _summarize_positive_control_records(records, field="depth")
        self.assertEqual(by_depth["2"]["joint_trajectory_accuracy"], 1.0)
        self.assertEqual(by_depth["4"]["joint_trajectory_accuracy"], 0.25)

    def test_any_early_failure_writes_identity_valid_fail_closed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            with mock.patch.object(
                gpu_runner, "_read_receipt", side_effect=RuntimeError("synthetic G0 failure")
            ), mock.patch.object(gpu_runner, "_atomic_copy") as copy:
                with self.assertRaisesRegex(RuntimeError, "synthetic G0 failure"):
                    gpu_runner.positive_control(
                        self.config,
                        output,
                        capacity="lora",
                        model_seed=7411,
                        initialization_bundle=Path(directory) / "init.pt",
                        model_smoke_receipt=Path(directory) / "g0.json",
                        authorization_receipt=None,
                    )
            payload = json.loads(output.read_text(encoding="utf-8"))
            claimed = payload.pop("receipt_identity_sha256")
            self.assertEqual(gpu_runner._canonical_sha256(payload), claimed)
            self.assertEqual(payload["status"], "SETUP_CONTROL_FAILED")
            self.assertEqual(payload["failure_stage"], "receipt_preflight")
            self.assertFalse(payload["authorizes_training"])
            self.assertFalse(payload["authorizes_result_training"])
            self.assertFalse(payload["scientific_evidence"])
            copy.assert_called_once()

    def test_training_receipt_loader_rejects_setup_control_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "receipt.json"
            payload = gpu_runner._with_identity({
                "schema_version": 1,
                "status": "SETUP_CONTROL_FAILED",
                **gpu_runner._identity(self.config, phase="lora_positive_control"),
                "scientific_evidence": False,
            })
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not authorize"):
                gpu_runner._read_receipt(
                    path,
                    self.config,
                    statuses={"POSITIVE_CONTROL_PASS"},
                    phases={"lora_positive_control"},
                    label="synthetic positive control",
                )


if __name__ == "__main__":
    unittest.main()

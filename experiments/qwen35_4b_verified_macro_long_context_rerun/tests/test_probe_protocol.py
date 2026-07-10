"""Focused model-free tests for amendment-6 workload-budget probes."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verified_macro_probe_protocol_run",
    EXP / "scripts" / "run.py",
)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


class TerminationContentBlindnessTests(unittest.TestCase):
    def test_decoded_content_and_correctness_mutations_cannot_change_gate(self) -> None:
        config = run.load_config()
        budget = 49152
        rows: list[dict[str, object]] = []
        for record_index in range(12):
            outputs: list[dict[str, object]] = []
            for sample_index in range(4):
                output_index = 4 * record_index + sample_index
                output: dict[str, object] = {
                    "sample_index": sample_index,
                    "n_thinking_tokens": 1000 + output_index,
                    "n_answer_tokens": 20 + output_index,
                    "forced_close": False,
                    "stage1_finish_reason": "stop",
                    "finish_reason": "stop",
                    "truncated": False,
                    "text": "baseline decoded answer",
                    "thinking_text": "baseline decoded reasoning",
                    "parsed_program": ["ADD1"],
                    "parse_error": None,
                    "correct": True,
                    "visible_correct": True,
                    "hidden_correct": True,
                    "oracle_success": True,
                    "macro_used": False,
                }
                outputs.append(output)
            rows.append(
                {
                    "id": f"smoke-task-{record_index}::base",
                    "meta": {
                        "task_id": f"smoke-task-{record_index}",
                        "arm": "base",
                        "correct": True,
                    },
                    "score": 1.0,
                    "outputs": outputs,
                }
            )

        flat_outputs = [
            output
            for row in rows
            for output in row["outputs"]  # type: ignore[index, union-attr]
        ]
        flat_outputs[45].update(
            {
                "n_thinking_tokens": budget,
                "forced_close": True,
                "stage1_finish_reason": "length",
                "retained_thinking_token_ids": list(range(8192)),
            }
        )
        flat_outputs[46].update(
            {
                "n_thinking_tokens": budget,
                "forced_close": True,
                "stage1_finish_reason": "length",
                "retained_thinking_token_ids": [101, 202, 303, 404] * 2048,
            }
        )
        # Amendment 7 treats equality with the answer allowance as contact even
        # after a natural stage-1 close.
        flat_outputs[47]["n_answer_tokens"] = 512

        mutated = copy.deepcopy(rows)
        for record_index, row in enumerate(mutated):
            row["score"] = -1000 - record_index
            row["evaluation"] = {
                "correct": False,
                "hidden_grade": "arbitrarily bad",
            }
            meta = row["meta"]
            assert isinstance(meta, dict)
            meta["correct"] = False
            for sample_index, output in enumerate(row["outputs"]):
                assert isinstance(output, dict)
                output.update(
                    {
                        "text": f"mutated answer {record_index}:{sample_index}",
                        "thinking_text": "completely different decoded reasoning",
                        "parsed_program": ["INVALID", "PROGRAM"],
                        "parse_error": "forced mutation",
                        "correct": False,
                        "visible_correct": False,
                        "hidden_correct": False,
                        "oracle_success": False,
                        "macro_used": True,
                    }
                )

        kwargs = {
            "budget": budget,
            "answer_cap": 512,
            "config": config,
            "require_headroom": False,
        }
        baseline = run._termination_metrics(rows, **kwargs)
        changed = run._termination_metrics(mutated, **kwargs)

        self.assertEqual(changed, baseline)
        self.assertTrue(baseline["adequate"])
        self.assertEqual(baseline["unresolved_cap_contacts"], 1)
        self.assertEqual(baseline["periodic_loop_contacts"], 1)
        self.assertEqual(baseline["answer_limit_contacts"], 1)
        self.assertFalse(baseline["selection_uses_output_content"])


class ProbeTransitionTests(unittest.TestCase):
    def _run_scenario(
        self,
        *,
        matrix_adequacy: dict[tuple[int, str], bool],
        probe_adequacy: dict[int, bool],
    ) -> dict[str, object]:
        config = run.load_config()
        validation_calls: dict[Path, int] = {}
        generated: list[tuple[str, int, int]] = []
        probe_budgets: list[int] = []

        def validate(path: Path, **_: object) -> bool:
            validation_calls[path] = validation_calls.get(path, 0) + 1
            return validation_calls[path] > 1

        def validate_scientific(**kwargs: object) -> bool:
            paths = run.scientific_store.bundle_paths(
                Path(kwargs["root"]), str(kwargs["relative_prefix"])
            )
            return validate(paths.rows)

        def rows(path: Path) -> list[dict[str, object]]:
            return [
                {
                    "budget": int(path.parent.name.removeprefix("think_")),
                    "arm": path.stem,
                }
            ]

        def termination(
            fixture: list[dict[str, object]], **_: object
        ) -> dict[str, object]:
            key = (int(fixture[0]["budget"]), str(fixture[0]["arm"]))
            return {
                "adequate": matrix_adequacy[key],
                "selection_uses_output_content": False,
            }

        def probe(**kwargs: object) -> dict[str, object]:
            budget = int(kwargs["budget"])
            probe_budgets.append(budget)
            adequate = probe_adequacy[budget]
            return {
                "status": "complete",
                "role": "termination_only_budget_probe",
                "budget": budget,
                "arm": "base",
                "k": 4,
                "termination": {"adequate": adequate},
                "artifacts": {"rows": f"probe-{budget}"},
                "eligible_for_promotion": False,
                "eligible_for_scoring": False,
                "eligible_for_prefix_pooling": False,
                "selection_uses_output_content": False,
            }

        class FakeHarness:
            @staticmethod
            def generate_vllm_batch(
                runner: object,
                records: list[dict[str, object]],
                sampling: types.SimpleNamespace,
            ) -> object:
                del runner
                generated.append(
                    (
                        str(records[0]["id"]),
                        int(sampling.thinking_budget),
                        int(sampling.n),
                    )
                )
                return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(
                    run, "_scientific_external_root", return_value=root / "external"
                ),
                mock.patch.object(run, "_validate_cached_arm", side_effect=validate),
                mock.patch.object(
                    run,
                    "_validate_cached_scientific_arm",
                    side_effect=validate_scientific,
                ),
                mock.patch.object(
                    run,
                    "_solver_records",
                    side_effect=lambda **kwargs: [{"id": kwargs["arm"]}],
                ),
                mock.patch.object(
                    run,
                    "_solver_sampling",
                    side_effect=lambda _harness, _config, *, budget, n: types.SimpleNamespace(
                        thinking_budget=budget,
                        n=n,
                    ),
                ),
                mock.patch.object(run, "_preflight_records", return_value={"pass": True}),
                mock.patch.object(run, "_write_batch"),
                mock.patch.object(run.scientific_store, "write_preflight_only"),
                mock.patch.object(run, "_commit_scientific_receipt"),
                mock.patch.object(run, "_write_scientific_catalog") as catalog_write,
                mock.patch.object(run, "_read_jsonl", side_effect=rows),
                mock.patch.object(run, "_termination_metrics", side_effect=termination),
                mock.patch.object(
                    run,
                    "_stage1_prefix_audit",
                    return_value={"pass": True, "samples": 1, "failures": []},
                ),
                mock.patch.object(
                    run,
                    "_scientific_artifact_hashes",
                    return_value={"rows": "matrix"},
                ),
                mock.patch.object(run, "_run_smoke_budget_probe", side_effect=probe),
            ):
                selected: int | None = None
                error: ValueError | None = None
                try:
                    selected = run._run_scientific_stage(
                        run="smoke",
                        runner=object(),
                        harness=FakeHarness,
                        domain=object(),
                        config=config,
                        tasks=[],
                        libraries={"base": {}, "designed_ceiling": {}},
                        demonstrations=[],
                        starting_budget=32768,
                    )
                except ValueError as caught:
                    error = caught

                selection = run._read_json(
                    root / "analysis" / "smoke_budget_selection.json"
                )

        return {
            "selected": selected,
            "error": error,
            "generated": generated,
            "probe_budgets": probe_budgets,
            "selection": selection,
            "catalog_calls": catalog_write.call_args_list,
        }

    def test_designed_32k_rejection_activates_higher_rung_probe(self) -> None:
        result = self._run_scenario(
            matrix_adequacy={
                (32768, "base"): True,
                (32768, "designed_ceiling"): False,
                (49152, "base"): True,
                (49152, "designed_ceiling"): True,
            },
            probe_adequacy={49152: True},
        )

        self.assertEqual(result["selected"], 49152)
        self.assertIsNone(result["error"])
        self.assertEqual(result["probe_budgets"], [49152])
        self.assertEqual(
            result["generated"],
            [
                ("base", 32768, 12),
                ("designed_ceiling", 32768, 12),
                ("base", 49152, 12),
                ("designed_ceiling", 49152, 12),
            ],
        )
        tiers = result["selection"]["tiers"]
        self.assertEqual(tiers[0]["rejecting_arm"], "designed_ceiling")
        self.assertEqual(tiers[1]["status"], "selectable")
        self.assertTrue(tiers[1]["scientific_probe"]["termination"]["adequate"])

    def test_adequate_49k_probe_matrix_rejection_routes_through_61k(self) -> None:
        cases = (
            ("inadequate_probe", False, None, False),
            ("inadequate_matrix", True, False, False),
            ("adequate_matrix", True, True, True),
        )
        for name, final_probe_adequate, final_matrix_adequate, selectable in cases:
            with self.subTest(branch=name):
                matrix = {
                    (32768, "base"): False,
                    (49152, "base"): True,
                    (49152, "designed_ceiling"): False,
                }
                if final_probe_adequate:
                    matrix[(61440, "base")] = bool(final_matrix_adequate)
                    if final_matrix_adequate:
                        matrix[(61440, "designed_ceiling")] = True
                result = self._run_scenario(
                    matrix_adequacy=matrix,
                    probe_adequacy={
                        49152: True,
                        61440: final_probe_adequate,
                    },
                )

                self.assertEqual(result["probe_budgets"], [49152, 61440])
                tiers = result["selection"]["tiers"]
                self.assertTrue(tiers[1]["scientific_probe"]["termination"]["adequate"])
                self.assertEqual(tiers[1]["status"], "rejected")
                self.assertEqual(tiers[1]["rejecting_arm"], "designed_ceiling")
                self.assertEqual(bool(result["selected"]), selectable)
                self.assertTrue(result["catalog_calls"])
                self.assertEqual(result["selection"]["pass"], selectable)
                if selectable:
                    self.assertIsNone(result["error"])
                    self.assertEqual(result["selected"], 61440)
                    self.assertEqual(tiers[2]["status"], "selectable")
                    self.assertTrue(tiers[2]["complete"])
                else:
                    self.assertIsInstance(result["error"], ValueError)
                    self.assertIn("setup-inconclusive", str(result["error"]))
                    self.assertIsNone(result["selected"])
                    self.assertFalse(tiers[2]["adequate"])


if __name__ == "__main__":
    unittest.main()

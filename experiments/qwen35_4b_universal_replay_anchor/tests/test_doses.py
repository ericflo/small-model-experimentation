from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPT = EXP / "scripts" / "materialize_doses.py"
SPEC = importlib.util.spec_from_file_location("materialize_doses", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def lines(value: bytes) -> list[str]:
    return [line for line in value.decode("utf-8").splitlines() if line]


class DoseConstructionTests(unittest.TestCase):
    def test_frozen_doses_are_deterministic_and_nested(self) -> None:
        first, manifest = MODULE.build_outputs()
        second, second_manifest = MODULE.build_outputs()
        self.assertEqual(first, second)
        self.assertEqual(manifest, second_manifest)

        candidate = lines(first["warm_union.jsonl"])
        control = lines(first["replay_refresh.jsonl"])
        self.assertEqual(len(candidate), 1520)
        self.assertEqual(len(control), 1520)
        self.assertEqual(candidate[400:], control[:1120])
        self.assertEqual(len(set(candidate)), len(candidate))
        self.assertEqual(len(set(control)), len(control))
        self.assertEqual(manifest["selection"], {
            "designed_rows": 400,
            "shared_replay_rows": 1120,
            "extra_control_replay_rows": 400,
            "candidate_total_rows": 1520,
            "control_total_rows": 1520,
            "nested_replay_control": True,
        })

    def test_designed_half_preserves_every_kind_count(self) -> None:
        outputs, manifest = MODULE.build_outputs()
        designed = [json.loads(line) for line in lines(outputs["warm_union.jsonl"])[:400]]
        self.assertEqual(Counter(row["kind"] for row in designed), {
            "u_induct": 40,
            "u_execute": 30,
            "u_select": 25,
            "u_trace": 30,
            "u_verify": 30,
            "u_count": 15,
            "u_repair": 45,
            "u_optimize": 35,
            "u_abstain": 35,
            "u_state": 40,
            "u_order": 25,
            "u_probe": 25,
            "u_route": 25,
        })
        self.assertEqual(
            manifest["outputs"]["warm_union.jsonl"]["sha256"],
            "f209c677a734308525a0feb04a14c1e1e3773bea750ef3ee50172687e67a61aa",
        )
        self.assertEqual(
            manifest["outputs"]["replay_refresh.jsonl"]["sha256"],
            "5d5d7c4b8a4b0a4f270fe8b2ecaebe356c771948d71b0f7bbeead6bfc04308b6",
        )

    def test_checked_in_derived_bytes_match_builder(self) -> None:
        outputs, manifest = MODULE.build_outputs()
        for name, expected in outputs.items():
            self.assertEqual((EXP / "data" / name).read_bytes(), expected)
        expected_manifest = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self.assertEqual((EXP / "data" / "dose_manifest.json").read_bytes(), expected_manifest)


if __name__ == "__main__":
    unittest.main()

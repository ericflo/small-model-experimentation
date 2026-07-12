from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPT = EXP / "scripts" / "assemble_training_round.py"
CONFIG = EXP / "configs" / "default.yaml"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RoundAssemblyTests(unittest.TestCase):
    def test_exact_equal_teacher_and_anchor_quota(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states = []
            anchors = []
            branches = {policy: [] for policy in ("quick", "deep", "student")}
            for index in range(60):
                state_id = f"state-{index:03d}"
                teacher = "quick" if index < 30 else "deep"
                states.append(
                    {
                        "state_id": state_id,
                        "family": "caravan",
                        "kind": "atom",
                        "level": 1 + index % 6,
                    }
                )
                for policy in branches:
                    for branch_index in range(4):
                        score = 1.0 if policy == teacher else 0.0
                        branches[policy].append(
                            {
                                "state_id": state_id,
                                "branch_index": branch_index,
                                "policy": policy,
                                "kind": "atom",
                                "score": score,
                                "output": {
                                    "token_ids": [10 + branch_index],
                                    "injected_token_ids": [],
                                },
                            }
                        )
            for index in range(20):
                anchors.append(
                    {
                        "state_id": f"anchor-{index:03d}",
                        "family": "ferrier",
                        "kind": "atom",
                        "level": 1,
                    }
                )
            state_path = root / "states.jsonl"
            anchor_path = root / "anchors.jsonl"
            _write_jsonl(state_path, states)
            _write_jsonl(anchor_path, anchors)
            branch_paths = {}
            for policy, rows in branches.items():
                branch_paths[policy] = root / f"{policy}.jsonl"
                _write_jsonl(branch_paths[policy], rows)
            out = root / "round.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config", str(CONFIG),
                    "--states", str(state_path),
                    "--anchors", str(anchor_path),
                    "--quick", str(branch_paths["quick"]),
                    "--deep", str(branch_paths["deep"]),
                    "--student", str(branch_paths["student"]),
                    "--round", "0",
                    "--out", str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["unit_counts"],
                {"total": 80, "quick": 30, "deep": 30, "soup_anchor": 20},
            )
            self.assertEqual(len({row["state_id"] for row in payload["units"]}), 80)
            capability = [row for row in payload["units"] if row["role"] == "capability"]
            self.assertTrue(all(row["offpolicy_target"] for row in capability))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from confirmation_artifacts import (  # noqa: E402
    RAW_FILENAMES,
    commit_confirmation_score,
    configured_confirmation_raw_root,
    confirmation_raw_dir,
    prepare_confirmation_output,
    validate_confirmation_geometry,
    validate_confirmation_score_artifacts,
)


class ConfirmationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.score_root = root / "visible"
        self.raw_root = root / "large"
        self.score_path = self.score_root / "block_0" / "arm" / "scores.json"
        self.tag = "block_0_arm"

    def tearDown(self):
        self.temporary.cleanup()

    def _commit(self) -> dict:
        return commit_confirmation_score(
            self.score_path,
            {
                "stage": "policy_eval",
                "tag": self.tag,
                "items": [
                    {
                        "key": "a",
                        "family": "family",
                        "kind": "atom",
                        "level": 1,
                        "samples": 1,
                        "score": 0.25,
                    },
                    {
                        "key": "b",
                        "family": "family",
                        "kind": "atom",
                        "level": 2,
                        "samples": 1,
                        "score": 0.5,
                    },
                    {
                        "key": "family/episode/L3/s7",
                        "family": "family",
                        "kind": "episode",
                        "level": 3,
                        "samples": 1,
                        "score": 0.75,
                    },
                ],
            },
            atom_rows=[
                {
                    "id": "a",
                    "family": "family",
                    "level": 1,
                    "outputs": [{"sample_index": 0, "score": 0.25}],
                },
                {
                    "id": "b",
                    "family": "family",
                    "level": 2,
                    "outputs": [{"sample_index": 0, "score": 0.5}],
                },
            ],
            episode_rows=[
                {
                    "family": "family",
                    "level": 3,
                    "ep_seed": 7,
                    "rollout": 0,
                    "score": 0.75,
                }
            ],
            score_root=self.score_root,
            raw_root=self.raw_root,
        )

    def _validate(self, tag: str | None = None) -> dict:
        return validate_confirmation_score_artifacts(
            self.score_path,
            expected_tag=tag or self.tag,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )

    def test_configured_raw_root_is_frozen_to_this_experiment(self):
        expected = EXP.parents[1] / "large_artifacts" / EXP.name / "confirmation"
        self.assertEqual(
            configured_confirmation_raw_root(
                {"model": {"artifacts_root": f"large_artifacts/{EXP.name}"}}
            ),
            expected.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "frozen experiment root"):
            configured_confirmation_raw_root(
                {"model": {"artifacts_root": self.raw_root}}
            )

    def test_full_geometry_rejects_smoke_or_family_subset(self):
        config = {
            "strata": {
                "trained_families": ["trained"],
                "transfer_families": ["transfer"],
                "quick_atom_levels": [1],
                "deep_atom_levels": [2],
                "deep_episode_levels": [3],
            },
            "confirmation": {
                "atoms_per_family_level": 2,
                "episodes_per_family_level": 1,
            },
            "controls": {"sample_more_k": 8},
        }
        items = []
        for family in ("trained", "transfer"):
            for level, stratum in ((1, "quick"), (2, "deep")):
                for index in range(2):
                    items.append(
                        {
                            "key": f"{family}-L{level}-{index}",
                            "family": family,
                            "kind": "atom",
                            "level": level,
                            "stratum": stratum,
                            "score": 0.5,
                            "samples": 1,
                        }
                    )
            items.append(
                {
                    "key": f"{family}/episode/L3/s7",
                    "family": family,
                    "kind": "episode",
                    "level": 3,
                    "stratum": "deep",
                    "score": 0.5,
                    "samples": 1,
                }
            )
        payload = {
            "scope": "confirmatory",
            "families": ["trained", "transfer"],
            "atoms_per_level": 2,
            "episodes_per_level": 1,
            "decode": "greedy",
            "k": 1,
            "items": items,
        }
        family = SimpleNamespace(LEVELS={1, 2, 3}, HAS_EPISODES=True)
        with mock.patch("confirmation_artifacts.load_family", return_value=family):
            validate_confirmation_geometry(payload, config)
            smoke = {**payload, "atoms_per_level": 1}
            with self.assertRaisesRegex(ValueError, "frozen full geometry"):
                validate_confirmation_geometry(smoke, config)
            subset = {**payload, "families": ["trained"]}
            with self.assertRaisesRegex(ValueError, "frozen full geometry"):
                validate_confirmation_geometry(subset, config)
            missing_cell = {**payload, "items": items[:-1]}
            with self.assertRaisesRegex(ValueError, "cells"):
                validate_confirmation_geometry(missing_cell, config)

    def test_score_is_last_commit_marker_and_descriptors_are_exact(self):
        payload = self._commit()
        raw_dir = confirmation_raw_dir(
            self.score_path,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        self.assertEqual(
            set(path.name for path in raw_dir.iterdir()), set(RAW_FILENAMES.values())
        )
        self.assertEqual(payload, self._validate())
        self.assertEqual(payload["raw_artifacts"]["atom_rows"]["rows"], 2)
        self.assertEqual(payload["raw_artifacts"]["episode_rows"]["rows"], 1)
        for key, filename in RAW_FILENAMES.items():
            descriptor = payload["raw_artifacts"][key]
            path = raw_dir / filename
            self.assertEqual(descriptor["path"], str(path))
            self.assertEqual(descriptor["bytes"], path.stat().st_size)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), descriptor["rows"])

    def test_failed_raw_write_leaves_no_score_and_known_orphans_recover(self):
        with self.assertRaises(ValueError):
            commit_confirmation_score(
                self.score_path,
                {
                    "stage": "policy_eval",
                    "tag": self.tag,
                    "items": [
                        {
                            "key": "a",
                            "family": "family",
                            "kind": "atom",
                            "level": 1,
                            "samples": 1,
                            "score": 0.25,
                        },
                        {
                            "key": "family/episode/L3/s7",
                            "family": "family",
                            "kind": "episode",
                            "level": 3,
                            "samples": 2,
                            "score": 0.75,
                        },
                    ],
                },
                atom_rows=[
                    {
                        "id": "a",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 0, "score": 0.25}],
                    }
                ],
                episode_rows=[
                    {
                        "family": "family",
                        "level": 3,
                        "ep_seed": 7,
                        "rollout": 0,
                        "score": 0.75,
                    },
                    "not-an-object",
                ],
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertFalse(self.score_path.exists())
        raw_dir = prepare_confirmation_output(
            self.score_path,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        self.assertEqual(list(raw_dir.iterdir()), [])
        self.assertEqual(list(self.score_path.parent.iterdir()), [])

    def test_raw_row_count_must_match_scored_item_geometry(self):
        with self.assertRaisesRegex(ValueError, "item geometry"):
            commit_confirmation_score(
                self.score_path,
                {
                    "stage": "policy_eval",
                    "tag": self.tag,
                    "items": [
                        {
                            "key": "a",
                            "family": "family",
                            "kind": "atom",
                            "level": 1,
                            "samples": 1,
                            "score": 0.25,
                        }
                    ],
                },
                atom_rows=[
                    {
                        "id": "a",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 0, "score": 0.25}],
                    },
                    {
                        "id": "b",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 0, "score": 0.5}],
                    },
                ],
                episode_rows=[],
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertFalse(self.score_path.exists())

    def test_score_item_semantics_are_recomputed_from_raw_rows(self):
        self._commit()
        payload = json.loads(self.score_path.read_text(encoding="utf-8"))
        payload["items"][0]["score"] = 999.0
        self.score_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "raw semantics"):
            self._validate()

    def test_raw_sample_indexes_must_be_contiguous(self):
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            commit_confirmation_score(
                self.score_path,
                {
                    "stage": "policy_eval",
                    "tag": self.tag,
                    "items": [
                        {
                            "key": "a",
                            "family": "family",
                            "kind": "atom",
                            "level": 1,
                            "samples": 1,
                            "score": 0.25,
                        }
                    ],
                },
                atom_rows=[
                    {
                        "id": "a",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 1, "score": 0.25}],
                    }
                ],
                episode_rows=[],
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertFalse(self.score_path.exists())

    def test_missing_or_tampered_raw_fails_closed(self):
        payload = self._commit()
        raw = Path(payload["raw_artifacts"]["atom_rows"]["path"])
        raw.write_bytes(raw.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "descriptor is stale"):
            self._validate()
        raw.unlink()
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            self._validate()

    def test_tag_and_descriptor_path_are_validated(self):
        self._commit()
        with self.assertRaisesRegex(ValueError, "tag"):
            self._validate("block_0_other")
        payload = json.loads(self.score_path.read_text(encoding="utf-8"))
        payload["raw_artifacts"]["atom_rows"]["path"] = str(
            Path(self.temporary.name) / "escaped.jsonl.gz"
        )
        self.score_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "raw path"):
            self._validate()

    def test_unknown_partials_are_never_recovered(self):
        self.score_path.parent.mkdir(parents=True)
        (self.score_path.parent / "mystery.bin").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "unknown partial"):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_known_visible_and_external_orphans_recover_without_score(self):
        self.score_path.parent.mkdir(parents=True)
        raw_dir = confirmation_raw_dir(
            self.score_path,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        raw_dir.mkdir(parents=True)
        (self.score_path.parent / "atom_rows.jsonl.gz").write_bytes(b"old")
        (self.score_path.parent / ".scores.json.crash.tmp").write_bytes(b"partial")
        (raw_dir / "episode_rows.jsonl.gz").write_bytes(b"old")
        (raw_dir / ".atom_rows.jsonl.gz.crash.tmp").write_bytes(b"partial")
        prepare_confirmation_output(
            self.score_path,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        self.assertEqual(list(self.score_path.parent.iterdir()), [])
        self.assertEqual(list(raw_dir.iterdir()), [])

    def test_committed_score_rejects_extras_and_cannot_be_auto_recovered(self):
        self._commit()
        (self.score_path.parent / ".scores.json.crash.tmp").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "unknown partials"):
            self._validate()
        with self.assertRaisesRegex(ValueError, "commit marker already exists"):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_score_path_escape_is_rejected(self):
        outside = Path(self.temporary.name) / "outside" / "scores.json"
        with self.assertRaisesRegex(ValueError, "escaped"):
            prepare_confirmation_output(
                outside,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_score_path_wrong_depth_is_rejected(self):
        shallow = self.score_root / "arm" / "scores.json"
        with self.assertRaisesRegex(ValueError, "block/arm"):
            prepare_confirmation_output(
                shallow,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_mirrored_path_cannot_be_redirected_by_symlink(self):
        self.raw_root.mkdir()
        redirected = self.raw_root / "redirected"
        redirected.mkdir()
        (self.raw_root / "block_0").symlink_to(redirected, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )


if __name__ == "__main__":
    unittest.main()

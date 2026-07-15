#!/usr/bin/env python3
"""Recompute the model-free re-adjudication design receipt.

This experiment trains and merges NOTHING. It inherits three published
composites (parent, replay-squared control, axis-on-replay candidate),
re-judges them on a fresh gate instrument with the prospectively corrected
breadth bar, and conditionally runs one medium-tier pilot. This check
authenticates the frozen inherited corpora, the committed external merge
receipts of all three arms (cheap identity checks; the full tree manifests
are recomputed by gen_local_gate.py), and the frozen lifecycle contracts of
the harness and both eval consumers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_axis_curriculum as axis  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
TREATMENT_PATH = EXP / "data" / "sft_axis160.jsonl"
TREATMENT_SHA256 = "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e"
TREATMENT_ROWS = 160
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_goal_gap_axis_curriculum_target_match"
STACK_EXP = ROOT / "experiments" / "qwen35_4b_axis_replay_stack_medium_target_match"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = ("replay_parent", "replay_squared", "axis_on_replay")
CANDIDATE_ARMS = ("axis_on_replay",)
PARENT_EVAL_LABEL = "replay_parent"
AXIS_KINDS = ("u_explore", "u_hygiene", "u_protocol", "u_tracefix")
PUBLIC_FAMILIES = (
    "chronicle", "lockpick", "menders", "mirage", "rites",
    "siftstack", "sirens", "stockade", "toolsmith", "warren",
)
# Inherited published composites (fail-closed pins). "tracked" is the
# committed run receipt in the donor experiment; "external" is the
# merge_receipt.json inside the composite tree itself.
COMPOSITES = {
    "replay_parent": {
        "name": "replay_repeat",
        "experiment": "qwen35_4b_goal_gap_axis_curriculum_target_match",
        "tracked_receipt": PARENT_EXP / "runs" / "merges" / "replay_repeat.json",
        "tracked_receipt_sha256": (
            "22384463d7825ec2a0b95faeaeb273264d7331f4584f8b7e9e58a60545398af1"
        ),
        "external_receipt_sha256": (
            "d3b184010f0470078e77e25796c572f41c177451f0157ced35d4e4d818a11b5b"
        ),
        "merged": (
            ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "replay_repeat"
        ),
        "tree_sha256": (
            "4c4f3561efbcafe1b9f777f4bd21bf4949ff89177f77946d0fa0f88cafafacd7"
        ),
        "weights_sha256": (
            "3df45004fcf42519ce28cdcfedcbb39b0907662f8ecfb8a87b13b416087d0072"
        ),
        "weights_size_bytes": 9_078_620_536,
    },
    "replay_squared": {
        "name": "replay_squared",
        "experiment": "qwen35_4b_axis_replay_stack_medium_target_match",
        "tracked_receipt": STACK_EXP / "runs" / "merges" / "replay_squared.json",
        "tracked_receipt_sha256": (
            "3d36542e0ea91b07e94dbe9e16551f97f33faf194ed1b5e6c87526e5248ee777"
        ),
        "external_receipt_sha256": (
            "2a4be2fcbb26cfeb2eb5f616bc8eef207c6f3aa99061ba79c84bd2e61b5845d6"
        ),
        "merged": (
            ROOT / "large_artifacts" / STACK_EXP.name / "merged" / "replay_squared"
        ),
        "tree_sha256": (
            "01108a985d2179561656141b2b824ee15a1d7a8a260da5d9e83387ebdc3a777d"
        ),
        "weights_sha256": (
            "e43b885c47ecc7046c3c741b48afd53dd1cb96d2d98426b714714d5ac271069e"
        ),
        "weights_size_bytes": 9_078_620_536,
    },
    "axis_on_replay": {
        "name": "axis_on_replay",
        "experiment": "qwen35_4b_axis_replay_stack_medium_target_match",
        "tracked_receipt": STACK_EXP / "runs" / "merges" / "axis_on_replay.json",
        "tracked_receipt_sha256": (
            "f58c946259090c0647fe607b87a884a11be913f7bdc71219a005907478c4201d"
        ),
        "external_receipt_sha256": (
            "3b13f5d7e893edf1e4af1476d0296478d198e192518afdcfdffd4df6682bdb5d"
        ),
        "merged": (
            ROOT / "large_artifacts" / STACK_EXP.name / "merged" / "axis_on_replay"
        ),
        "tree_sha256": (
            "77e4858fe6ddade7a8446a0c561c3c18d07c338d4dea2f0b8193693fcca264ea"
        ),
        "weights_sha256": (
            "7ebcad397c820196fb2271fe4c608a62a578465152b48e3fcee2c8d3b46fd0e4"
        ),
        "weights_size_bytes": 9_078_620_536,
    },
}
BASE_COMPOSITE = (
    ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
    / "merged" / "base_reserialized"
)
BASE_WEIGHTS_SHA256 = "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen_corpora() -> None:
    for path, expected, rows in (
        (TREATMENT_PATH, TREATMENT_SHA256, TREATMENT_ROWS),
        (REPLAY_PATH, REPLAY_SHA256, REPLAY_ROWS),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen corpus artifact is absent or changed: {path}")
        if len(path.read_text(encoding="utf-8").splitlines()) != rows:
            raise ValueError(f"frozen corpus row count changed: {path}")
    # The stack predecessor holds the byte-identical originals.
    for donor_path, expected in (
        (STACK_EXP / "data" / "sft_axis160.jsonl", TREATMENT_SHA256),
        (STACK_EXP / "data" / "sft_blend.jsonl", REPLAY_SHA256),
    ):
        if not donor_path.is_file() or sha256_file(donor_path) != expected:
            raise ValueError(
                f"donor corpus no longer matches the inherited copy: {donor_path}"
            )


def check_composite_identity(label: str) -> None:
    pins = COMPOSITES[label]
    tracked = pins["tracked_receipt"]
    if not tracked.is_file() or sha256_file(tracked) != pins["tracked_receipt_sha256"]:
        raise ValueError(f"published composite run receipt changed: {tracked}")
    payload = json.loads(tracked.read_text(encoding="utf-8"))
    external = pins["merged"] / "merge_receipt.json"
    weights = pins["merged"] / "model.safetensors"
    if (
        payload.get("name") != pins["name"]
        or payload.get("experiment_id") != pins["experiment"]
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != pins["merged"].resolve()
        or payload.get("merge_receipt_sha256") != pins["external_receipt_sha256"]
        or payload.get("output_tree_sha256") != pins["tree_sha256"]
        or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
        != {"model.safetensors": pins["weights_sha256"]}
        or not external.is_file()
        or sha256_file(external) != pins["external_receipt_sha256"]
        or not weights.is_file()
        or weights.stat().st_size != pins["weights_size_bytes"]
    ):
        raise ValueError(f"authenticated inherited composite identity changed: {label}")


def check_lifecycle_contracts() -> None:
    """Substring contracts pin the frozen stage/gate constants in place."""
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    local_eval = (EXP / "scripts" / "eval_local_vllm.py").read_text(encoding="utf-8")
    benchmark = (EXP / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
    gate = (EXP / "scripts" / "check_local.py").read_text(encoding="utf-8")
    required_harness = (
        "def require_pushed_checkpoint",
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88016",
        "AGGREGATE_SEED = 78146",
        "PASS_LOCAL_EVENT",
    )
    forbidden_harness = (
        "train-control",
        "train-candidate",
        "merge-arms",
        "train_trial",
        "merge_trained_arm",
    )
    required_local_eval = (
        "SEED = 88016",
        "AGGREGATE_SEED = 78146",
        "ROWS = 144",
        '"01108a985d2179561656141b2b824ee15a1d7a8a260da5d9e83387ebdc3a777d"',
        '"77e4858fe6ddade7a8446a0c561c3c18d07c338d4dea2f0b8193693fcca264ea"',
        '"4c4f3561efbcafe1b9f777f4bd21bf4949ff89177f77946d0fa0f88cafafacd7"',
    )
    required_benchmark = (
        'FROZEN_NAME = "pilot"',
        'FROZEN_TIER = "medium"',
        "FROZEN_THINK_BUDGET = 1024",
        "FROZEN_SEED = 78146",
        "LOCAL_SEED = 88016",
        '"b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db"',
        '"3df45004fcf42519ce28cdcfedcbb39b0907662f8ecfb8a87b13b416087d0072"',
        '"e43b885c47ecc7046c3c741b48afd53dd1cb96d2d98426b714714d5ac271069e"',
        '"7ebcad397c820196fb2271fe4c608a62a578465152b48e3fcee2c8d3b46fd0e4"',
    )
    required_gate = (
        "SEED = 88016",
        "AGGREGATE_SEED = 78146",
        "DETECTABILITY_CEILING = 9",
        "GATE_UNDETECTABLE",
        "def required_kind_wins",
    )
    if any(value not in harness for value in required_harness):
        raise ValueError("frozen stage-harness contract changed")
    if any(value in harness for value in forbidden_harness):
        raise ValueError("training/merge lifecycle leaked into the harness")
    if any(value not in local_eval for value in required_local_eval):
        raise ValueError("frozen local-eval contract changed")
    if any(value not in benchmark for value in required_benchmark):
        raise ValueError("frozen benchmark wrapper contract changed")
    if any(value not in gate for value in required_gate):
        raise ValueError("frozen promotion-gate contract changed")
    forbidden = "benchmarks" + "/"
    if any(forbidden in text for text in (harness, local_eval, benchmark, gate)):
        raise ValueError("benchmark content leaked into the lifecycle scripts")


def build_receipt() -> dict:
    check_frozen_corpora()
    for label in ARMS:
        check_composite_identity(label)
    check_lifecycle_contracts()
    banned = tuple(axis.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "axis_stack_readjudication_design",
        "training_free": True,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "seeds": {
            "construction_inherited": 77117,
            "construction_inherited_from": PARENT_EXP.name,
            "local_gate": 88016,
            "conditional_aggregate": 78146,
        },
        "corpora": {
            "purpose": (
                "frozen sources for the gate's zero-overlap receipt only; "
                "nothing is trained on them here"
            ),
            "treatment": {
                "path": TREATMENT_PATH.relative_to(EXP).as_posix(),
                "rows": TREATMENT_ROWS,
                "sha256": TREATMENT_SHA256,
                "inheritance": {
                    "from_experiment": STACK_EXP.name,
                    "byte_identical": True,
                },
            },
            "replay": {
                "path": REPLAY_PATH.relative_to(EXP).as_posix(),
                "rows": REPLAY_ROWS,
                "sha256": REPLAY_SHA256,
                "inheritance": {
                    "from_experiment": STACK_EXP.name,
                    "byte_identical": True,
                },
            },
            "banned_vocabulary": {
                "tokens": len(banned),
                "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
                "checked_by_generator": True,
            },
        },
        "arms": {
            label: {
                "composite_name": COMPOSITES[label]["name"],
                "experiment": COMPOSITES[label]["experiment"],
                "deployment": "explicit_merged_composite",
                "tracked_receipt": (
                    COMPOSITES[label]["tracked_receipt"].relative_to(ROOT).as_posix()
                ),
                "tracked_receipt_sha256": COMPOSITES[label]["tracked_receipt_sha256"],
                "external_receipt_sha256": COMPOSITES[label]["external_receipt_sha256"],
                "composite": COMPOSITES[label]["merged"].relative_to(ROOT).as_posix(),
                "tree_sha256": COMPOSITES[label]["tree_sha256"],
                "weights_sha256": COMPOSITES[label]["weights_sha256"],
                "weights_size_bytes": COMPOSITES[label]["weights_size_bytes"],
                "runtime_lora_forbidden": True,
            }
            for label in ARMS
        },
        "arm_roles": {
            "parent_eval_label": PARENT_EVAL_LABEL,
            "control": "replay_squared",
            "candidate": "axis_on_replay",
        },
        "training_plan": {
            "training_authorized": False,
            "reason": (
                "training-free re-adjudication: the arms are inherited "
                "published composites; no corpus, streams, training, or "
                "merging exist in this experiment"
            ),
        },
        "local_gate": {
            "seed": 88016,
            "rows": 144,
            "instruments": {
                "axis_holdout": {
                    "rows": 40,
                    "per_kind": 10,
                    "kinds": list(AXIS_KINDS),
                    "generator": "scripts/gen_axis_curriculum.py",
                },
                "retention": {
                    "rows": 104,
                    "per_kind": 8,
                    "skills": 13,
                    "generator": "scripts/gen_curriculum.py",
                },
            },
            "arms": list(ARMS),
            "promotion": {
                "axis_total_strictly_beats": [PARENT_EVAL_LABEL, "replay_squared"],
                "detectable_kind_definition": (
                    "an axis kind where NEITHER control scores >= 9 of 10 "
                    "on the holdout"
                ),
                "detectability_ceiling_correct_of_10": 9,
                "undetectable_kinds_excluded_and_reported_as_not_detectable": True,
                "axis_kind_wins_at_least": "ceil(2/3 * detectable_kinds)",
                "axis_kind_win_requires_strictly_above_max_of_both_controls": True,
                "zero_detectable_kinds_fails_closed_as": "GATE_UNDETECTABLE",
                "retention_correct_band": 5,
                "retention_cap_contact_band": 3,
                "retention_parsed_band": 3,
                "route_abstentions_at_most": 4,
                "no_absolute_per_kind_floors": True,
                "single_candidate": "axis_on_replay",
                "no_passing_candidate_keeps_aggregate_seed_sealed": True,
            },
        },
        "benchmark_plan": {
            "seed": 78146,
            "tier": "medium",
            "think_budget": 1024,
            "name": "pilot",
            "gateway": "scripts/run_benchmark_aggregate.py",
            "models": {
                "base": BASE_COMPOSITE.relative_to(ROOT).as_posix(),
                "parent": f"large_artifacts/{PARENT_EXP.name}/merged/replay_repeat",
                "control": f"large_artifacts/{STACK_EXP.name}/merged/replay_squared",
                "candidate": f"large_artifacts/{STACK_EXP.name}/merged/axis_on_replay",
            },
            "base_weights_sha256": BASE_WEIGHTS_SHA256,
            "public_families": list(PUBLIC_FAMILIES),
            "gates": {
                "candidate_aggregate_strictly_above_base": True,
                "candidate_aggregate_strictly_above_control": True,
                "candidate_aggregate_strictly_above_parent": True,
            },
            "goal_gate": {
                "every_public_family_strictly_above_base": (
                    "recorded and reported from the same event as the goal "
                    "gate; medium-tier family scores have finer granularity "
                    "than quick and the all-families gate is empirically "
                    "reachable there, but it is NOT part of the pilot pass"
                ),
            },
        },
        "checkpoint_policy": {
            "next_authorized_stage": "local",
            "one_stage_per_invocation": True,
            "clean_pushed_main_required": True,
            "preceding_receipt_committed_at_head": True,
            "full_check_rebase_push_two_workflow_gate_between_expensive_stages": True,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "code": {
            "curriculum_generator_sha256": sha256_file(EXP / "scripts" / "gen_curriculum.py"),
            "axis_curriculum_generator_sha256": sha256_file(
                EXP / "scripts" / "gen_axis_curriculum.py"
            ),
            "gate_sha256": sha256_file(EXP / "scripts" / "check_local.py"),
            "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = (json.dumps(build_receipt(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("design receipt is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite design receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(json.dumps({"out": str(args.out), "sha256": sha256_bytes(value)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

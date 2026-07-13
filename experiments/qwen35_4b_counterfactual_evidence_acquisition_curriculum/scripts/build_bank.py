#!/usr/bin/env python3
"""Build exact-mass candidate, redundant-evidence, and shuffled banks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import harness  # noqa: E402
import repo_tasks  # noqa: E402


LEGACY_TRANSITIONS = {
    "start_to_inspect": "start_to_inspect_source",
    "inspect_to_patch": "explicit_source_to_patch",
}


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_file(path)


def select_prior_blocks(rows: list[dict], n_tasks: int) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    by_family: dict[str, list[str]] = defaultdict(list)
    for task_id, task_rows in by_task.items():
        by_family[task_rows[0]["family"]].append(task_id)
    selected: list[str] = []
    depth = 0
    while len(selected) < n_tasks:
        added = False
        for family in sorted(by_family):
            task_ids = sorted(by_family[family])
            if depth < len(task_ids):
                selected.append(task_ids[depth])
                added = True
                if len(selected) == n_tasks:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != n_tasks:
        raise AssertionError(f"could select only {len(selected)} prior task blocks")
    result = []
    for task_id in selected:
        task_rows = sorted(by_task[task_id], key=lambda row: row["id"])
        transitions = {row["transition"] for row in task_rows}
        if len(task_rows) != 7 or transitions != {
            "start_to_inspect",
            "inspect_to_patch",
            "rejected_patch_to_changed_patch",
            "failed_test_to_diagnose",
            "diagnosis_to_changed_patch",
            "patch_ok_to_verify",
            "passed_test_to_commit",
        }:
            raise AssertionError((task_id, len(task_rows), transitions))
        for row in task_rows:
            normalized = copy.deepcopy(row)
            old_transition = normalized["transition"]
            normalized["transition"] = LEGACY_TRANSITIONS.get(
                old_transition, old_transition
            )
            normalized["id"] = (
                f"prior-{normalized['task_id']}-{normalized['transition']}"
            )
            normalized["kind"] = "repo_prior_complete_replay"
            normalized["conditioning"] = "prior_complete_replay"
            normalized["think_weight"] = 0.0
            normalized.pop("row_weight", None)
            normalized.pop("token_counts", None)
            result.append(normalized)
    return result


def evidence_rows(tasks: list[repo_tasks.RepoTask]) -> tuple[list[dict], list[dict]]:
    rows = []
    receipts = []
    for task in tasks:
        task_rows, receipt = bank.evidence_transition_rows(task)
        rows.extend(task_rows)
        receipts.append(receipt)
    return rows, receipts


def balance_transition_counts(rows: list[dict]) -> list[dict]:
    """Duplicate whole rows until every conditional stratum has equal exposure."""
    by_transition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_transition[row["transition"]].append(row)
    if set(by_transition) != set(bank.TRANSITIONS):
        raise AssertionError(f"missing transition strata: {set(bank.TRANSITIONS) - set(by_transition)}")
    target = max(len(items) for items in by_transition.values())
    balanced = copy.deepcopy(rows)
    for transition in bank.TRANSITIONS:
        items = by_transition[transition]
        for index in range(target - len(items)):
            copied = copy.deepcopy(items[index % len(items)])
            copied["id"] = f"{copied['id']}::transition-balance-pad-{index}"
            copied["transition_balance_padding"] = True
            balanced.append(copied)
    counts = Counter(row["transition"] for row in balanced)
    if set(counts.values()) != {target}:
        raise AssertionError(counts)
    return balanced


def shuffled_rows(candidate_rows: list[dict]) -> list[dict]:
    rows = copy.deepcopy(candidate_rows)
    targets_by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["transition"] == "evidence_to_policy_patch":
            targets_by_pair[row["pair_id"]].append(row)
    for pair_id, pair_rows in targets_by_pair.items():
        by_branch = {
            branch: sorted(
                (row for row in pair_rows if row["branch"] == branch),
                key=lambda row: row["id"],
            )
            for branch in (0, 1)
        }
        if not by_branch[0] or len(by_branch[0]) != len(by_branch[1]):
            raise AssertionError(f"bad shuffled dyad: {pair_id}")
        for a, b in zip(by_branch[0], by_branch[1]):
            a_answer, a_target = a["answer"], copy.deepcopy(a["target_action"])
            a["answer"], a["target_action"] = b["answer"], copy.deepcopy(b["target_action"])
            b["answer"], b["target_action"] = a_answer, a_target
            a["label_alignment"] = "counterfactual_shuffled"
            b["label_alignment"] = "counterfactual_shuffled"
            a["conditioning"] = "shuffled_binding"
            b["conditioning"] = "shuffled_binding"
    for row in rows:
        row["id"] = f"shuffled-{row['id']}"
        row["kind"] = (
            "repo_shuffled_binding"
            if row["transition"] == "evidence_to_policy_patch"
            else row["kind"]
        )
    return rows


def assert_exact_within_dyad_shuffle(
    candidate_rows: list[dict], shuffled: list[dict]
) -> dict:
    candidate = {
        row["id"]: row for row in candidate_rows
        if row["transition"] == "evidence_to_policy_patch"
    }
    shuffled_by_source = {
        row["id"].removeprefix("shuffled-"): row for row in shuffled
        if row["transition"] == "evidence_to_policy_patch"
    }
    if set(candidate) != set(shuffled_by_source):
        raise AssertionError("shuffled evidence rows do not preserve the candidate row set")
    checks = 0
    by_pair: dict[str, dict[int, list[dict]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for row in candidate.values():
        by_pair[row["pair_id"]][int(row["branch"])].append(row)
    for pair_id, branches in by_pair.items():
        left = sorted(branches[0], key=lambda row: row["id"])
        right = sorted(branches[1], key=lambda row: row["id"])
        if len(left) != len(right):
            raise AssertionError(f"unbalanced shuffle proof pair: {pair_id}")
        for a, b in zip(left, right):
            shuffled_a = shuffled_by_source[a["id"]]
            shuffled_b = shuffled_by_source[b["id"]]
            if (
                shuffled_a["answer"] != b["answer"]
                or shuffled_b["answer"] != a["answer"]
                or shuffled_a["messages"] != a["messages"]
                or shuffled_b["messages"] != b["messages"]
            ):
                raise AssertionError(f"imperfect within-dyad target swap: {pair_id}")
            checks += 1
    return {
        "status": "PASS",
        "dyads": len(by_pair),
        "paired_copy_swaps": checks,
        "prompt_multiset_preserved": True,
        "target_multiset_preserved": sorted(
            row["answer"] for row in candidate.values()
        ) == sorted(row["answer"] for row in shuffled_by_source.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    bcfg = cfg["bank"]
    source = resolve(bcfg["prior_source"])
    observed = sha256_file(source)
    if observed != bcfg["prior_source_sha256"]:
        raise SystemExit(f"prior bank hash mismatch: {observed}")
    prior = select_prior_blocks(read_jsonl(source), int(bcfg["prior_tasks_per_arm"]))
    inferred_tasks = repo_tasks.make_pairs(
        tuple(cfg["families"]["acquisition_train"]),
        int(bcfg["inferred_pairs_per_train_family"]),
        int(bcfg["seed"]),
        str(bcfg["split"]),
    )
    explicit_tasks = repo_tasks.make_pairs(
        tuple(cfg["families"]["acquisition_train"]),
        int(bcfg["inferred_pairs_per_train_family"]),
        int(bcfg["seed"]),
        str(bcfg["split"]),
        explicit_contract=True,
    )
    if len(inferred_tasks) != int(bcfg["new_tasks_per_arm"]):
        raise AssertionError("inferred bank task count mismatch")
    inferred, inferred_receipts = evidence_rows(inferred_tasks)
    explicit, explicit_receipts = evidence_rows(explicit_tasks)
    candidate = balance_transition_counts(copy.deepcopy(prior) + inferred)
    redundant = balance_transition_counts(copy.deepcopy(prior) + explicit)
    for row in candidate:
        if row.get("pair_id"):
            row["conditioning"] = "evidence_binding"
            row["kind"] = "repo_evidence_binding"
    for row in redundant:
        if row.get("pair_id"):
            row["conditioning"] = "explicit_redundant"
            row["kind"] = "repo_explicit_redundant"
    shuffled = shuffled_rows(candidate)
    shuffle_proof = assert_exact_within_dyad_shuffle(candidate, shuffled)
    arms = {
        "evidence_binding": candidate,
        "explicit_redundant": redundant,
        "shuffled_binding": shuffled,
    }
    for arm, rows in arms.items():
        for row in rows:
            row["source_kind"] = row.get("kind")
            row["kind"] = f"repo_{arm}"
    tokenizer = AutoTokenizer.from_pretrained(
        resolve(cfg["model"]["start_checkpoint"]), trust_remote_code=True
    )
    calibrations = {}
    for arm, rows in arms.items():
        if len(rows) != int(bcfg["rows_per_arm"]):
            raise AssertionError((arm, len(rows), bcfg["rows_per_arm"]))
        calibrations[arm] = bank.calibrate_transition_loss_mass(
            rows,
            tokenizer,
            target_transition_action_mass=float(
                bcfg["target_transition_action_mass"]
            ),
            plan_mass_fraction=0.0,
            max_length=int(cfg["training"]["max_length"]),
        )
        if any(row["think_weight"] != 0.0 for row in rows):
            raise AssertionError(f"think loss is nonzero in {arm}")
    target_mass = float(bcfg["target_transition_action_mass"])
    for arm, receipt in calibrations.items():
        for transition, mass in receipt["weighted_action_mass_by_transition"].items():
            if abs(mass - target_mass) > 1e-6:
                raise AssertionError((arm, transition, mass, target_mass))
    bank.assert_firewall_clean(candidate, inferred_tasks)
    bank.assert_firewall_clean(redundant, explicit_tasks)
    root = resolve(cfg["artifacts"]["root"]) / "bank"
    hashes = {
        arm: write_jsonl(root / f"{arm}.jsonl", rows)
        for arm, rows in arms.items()
    }
    start_checkpoint = resolve(cfg["model"]["start_checkpoint"])
    try:
        tokenizer = harness.tokenizer_provenance(start_checkpoint)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"bank tokenizer provenance is invalid: {exc}") from exc
    if (
        tokenizer["tokenizer_manifest_sha256"]
        != cfg["model"]["start_tokenizer_manifest_sha256"]
        or tokenizer["tokenizer_compatibility_sha256"]
        != cfg["model"]["tokenizer_compatibility_sha256"]
    ):
        raise SystemExit("bank tokenizer differs from the frozen config identity")
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "prior_source_sha256": observed,
        "config_sha256": sha256_file(args.config),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "bank_library_sha256": sha256_file(EXP / "src" / "bank.py"),
        "task_generator_sha256": sha256_file(EXP / "src" / "repo_tasks.py"),
        **tokenizer,
        "prior_task_blocks": len({row["task_id"] for row in prior}),
        "new_inferred_tasks": len(inferred_tasks),
        "new_explicit_tasks": len(explicit_tasks),
        "inferred_task_manifest_sha256": repo_tasks.manifest_digest(inferred_tasks),
        "explicit_task_manifest_sha256": repo_tasks.manifest_digest(explicit_tasks),
        "inferred_pair_static_manifest_sha256": hashlib.sha256(
            json.dumps(
                sorted(repo_tasks.pair_static_digest(task) for task in inferred_tasks),
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "rows_per_arm": {arm: len(rows) for arm, rows in arms.items()},
        "tasks_per_arm": {
            arm: len({row["task_id"] for row in rows}) for arm, rows in arms.items()
        },
        "transition_counts": {
            arm: dict(Counter(row["transition"] for row in rows))
            for arm, rows in arms.items()
        },
        "operator_counts": {
            arm: dict(Counter(row["operator"] for row in rows))
            for arm, rows in arms.items()
        },
        "family_counts": {
            arm: dict(Counter(row["family"] for row in rows))
            for arm, rows in arms.items()
        },
        "evidence_channel_counts": {
            arm: dict(Counter(
                row["evidence_channel"] for row in rows
                if row.get("evidence_channel")
            ))
            for arm, rows in arms.items()
        },
        "evidence_path_regime_counts": {
            arm: dict(Counter(
                row["evidence_path_regime"] for row in rows
                if row.get("evidence_path_regime")
            ))
            for arm, rows in arms.items()
        },
        "transition_balance_padding_rows": {
            arm: sum(bool(row.get("transition_balance_padding")) for row in rows)
            for arm, rows in arms.items()
        },
        "weighted_action_mass_by_operator": {
            arm: row["weighted_action_mass_by_operator"]
            for arm, row in calibrations.items()
        },
        "weighted_action_mass_by_transition": {
            arm: row["weighted_action_mass_by_transition"]
            for arm, row in calibrations.items()
        },
        "max_total_tokens": {
            arm: row["max_total_tokens"] for arm, row in calibrations.items()
        },
        "think_loss_zero": True,
        "within_dyad_shuffle_proof": shuffle_proof,
        "candidate_replay_receipts": inferred_receipts,
        "explicit_replay_receipts": explicit_receipts,
        "bank_sha256": hashes,
    }
    (root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

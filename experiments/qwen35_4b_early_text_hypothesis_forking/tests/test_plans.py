from __future__ import annotations

import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from plans import branch_operations, branch_plan, freeze_resource_plan  # noqa: E402
from task_data import (  # noqa: E402
    CONCRETE_OPERATIONS,
    build_splits,
    operation_from_record,
)


def config():
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


@lru_cache(maxsize=1)
def splits():
    return build_splits(config())


PANEL = [[3, -1, 2, 0, 5], [-2, 4, 1, -3, 6]]


def test_branch_compositions_are_noncyclic_unique_and_gold_slots_balanced():
    value = config()
    task_splits = splits()
    seed = int(value["seeds"]["branch_permutation"])
    for split, tasks in task_splits.items():
        plans = [branch_plan(task["task_id"], seed, behavior_panel=PANEL) for task in tasks]
        assert len({plan["composed_map_sha256"] for plan in plans}) == len(tasks)
        slots = Counter(plan["gold_slot_from_public_schedule"] for plan in plans)
        assert set(slots) == set(range(24))
        assert max(slots.values()) - min(slots.values()) <= 1
        for task, plan in zip(tasks, plans, strict=True):
            order = branch_operations(task["task_id"], seed)
            assert set(order) == set(CONCRETE_OPERATIONS)
            assert len(order) == len(set(order)) == 24
            assert order[plan["gold_slot_from_public_schedule"]] == operation_from_record(
                task["first_op"]
            )
        if len(plans) > 1:
            first = [row["canonical_operation"] for row in plans[0]["rows"]]
            second = [row["canonical_operation"] for row in plans[1]["rows"]]
            assert not any(second == first[offset:] + first[:offset] for offset in range(24))


def test_branch_seed_changes_plan_without_changing_task_bytes():
    task = splits()["qualification"][0]
    before = json.dumps(task, sort_keys=True)
    first = branch_plan(task["task_id"], 101, behavior_panel=PANEL)
    second = branch_plan(task["task_id"], 202, behavior_panel=PANEL)
    assert first["composed_map_sha256"] != second["composed_map_sha256"]
    assert json.dumps(task, sort_keys=True) == before


def test_resource_matching_uses_order_and_no_outcome_fields():
    rows = [
        {"id": f"sample-{index}", "sampled_tokens": 10, "logical_model_tokens": 25}
        for index in range(1, 49)
    ]
    plan = freeze_resource_plan(
        rows,
        target_sampled_tokens=235,
        target_logical_model_tokens=610,
    )
    assert plan["sampled"]["under_count"] == 23
    assert plan["sampled"]["over_count"] == 24
    assert plan["logical"]["under_count"] == 24
    assert plan["logical"]["over_count"] == 25
    assert len(plan["full_k_ids"]) == 24
    assert not plan["sampled"]["pool_exhausted"]
    assert not plan["logical"]["pool_exhausted"]

    contaminated = [dict(row) for row in rows]
    contaminated[0]["correct"] = True
    try:
        freeze_resource_plan(
            contaminated,
            target_sampled_tokens=235,
            target_logical_model_tokens=610,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("resource matcher accepted an outcome-bearing row")

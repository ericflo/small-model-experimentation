#!/usr/bin/env python3
"""Generate fresh exact-depth task splits and compact semantic-oracle receipts."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as F  # noqa: E402
import experiment_common as C  # noqa: E402
import oracle_data as O  # noqa: E402


def _behavior_signature(task: dict[str, Any], probes: list[list[int]]) -> tuple[Any, ...]:
    pipeline = F.normalize_pipeline(task["target_pipeline"])
    return tuple(tuple(F.execute_pipeline(pipeline, value) or ()) for value in probes)


def _generate_one(payload: tuple[str, int, int, int]) -> dict[str, Any]:
    name, depth, candidate_index, task_seed = payload
    return F.generate_task(
        task_id=f"{name}-candidate-{candidate_index:05d}",
        depth=depth,
        seed=task_seed,
        n_visible=8,
        n_label_probe=6,
        n_hidden=6,
        max_attempts=500,
    )


def _generate_split(
    name: str, depth: int, count: int, seed: int, workers: int
) -> list[dict[str, Any]]:
    seed_rng = random.Random(seed)
    probe_rng = random.Random(440_000 + depth)
    probes = [[probe_rng.randint(-9, 9) for _ in range(probe_rng.randint(5, 8))] for _ in range(20)]
    seen: set[tuple[Any, ...]] = set()
    tasks: list[dict[str, Any]] = []
    attempts = 0
    while len(tasks) < count:
        batch_n = max(workers, min(count - len(tasks) + workers, 2 * workers))
        payloads = []
        for _ in range(batch_n):
            attempts += 1
            payloads.append((name, depth, attempts, seed_rng.randrange(2**63)))
        if attempts > count * 30:
            raise RuntimeError(f"could not build {count} behavior-distinct {name} tasks")
        if workers == 1:
            generated = [_generate_one(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                generated = list(executor.map(_generate_one, payloads))
        for task in generated:
            signature = _behavior_signature(task, probes)
            if signature in seen:
                continue
            seen.add(signature)
            task["task_id"] = f"{name}-d{depth}-{len(tasks):04d}"
            task["behavior_signature_sha256"] = __import__("hashlib").sha256(
                json.dumps(signature, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            task["split"] = name
            if not C.prompt_boundary_audit(task, task["target_skeleton"][:1]):
                raise AssertionError("prompt whitelist changed after oracle-field deletion")
            tasks.append(task)
            print(
                f"[data] {name} {len(tasks)}/{count}: seed={task['seed']} "
                f"audit_states={task['min_depth_audit']['unique_behaviors_seen']}",
                flush=True,
            )
            if len(tasks) == count:
                break
    return tasks


def _compact_oracle(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    oracle = O.ExactSemanticOracle(task).build()
    successes = oracle.successful_skeletons()
    record = {
        "schema_version": 1,
        "task_id": str(task["task_id"]),
        "depth": int(task["depth"]),
        "label_source_splits": ["visible", "label_probe"],
        "hidden_cases_used_for_labels": False,
        "successful_skeleton_count": len(successes),
        "successful_parameter_fill_count": sum(row.parameter_fill_count for row in successes),
        "successful_skeletons": [row.to_dict() for row in successes],
        "behavior_states_by_layer": [len(layer) for layer in oracle.layers],
        "live_behavior_states_by_layer": [len(layer) for layer in oracle.live_states],
        "accounting": oracle.accounting.to_dict(),
        "wall_seconds": time.perf_counter() - started,
    }
    if not successes:
        raise RuntimeError(f"semantic oracle lost the serialized solution for {task['task_id']}")
    return record


def _oracle_split(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [_compact_oracle(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_compact_oracle, tasks))


def split_specs(cfg: dict[str, Any], smoke: bool) -> list[tuple[str, int, int, int]]:
    """Frozen task partitions; development is depth-matched to primary but disjoint."""

    smoke_n = int(cfg["smoke_n"])
    count_cal = int(smoke_n if smoke else cfg["calibration_n"])
    count_development = int(smoke_n if smoke else cfg["oracle_development_n"])
    count_primary = int(smoke_n if smoke else cfg["primary_n"])
    return [
        ("calibration", int(cfg["calibration_depth"]), count_cal, int(cfg["calibration_seed"])),
        (
            "development",
            int(cfg["primary_depth"]),
            count_development,
            int(cfg["oracle_development_seed"]),
        ),
        ("primary", int(cfg["primary_depth"]), count_primary, int(cfg["primary_seed"])),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = C.load_config()["task"]
    suffix = "_smoke" if args.smoke else ""
    for name, depth, count, seed in split_specs(cfg, args.smoke):
        task_path = EXP / "data" / f"{name}_tasks{suffix}.jsonl"
        oracle_path = EXP / "data" / f"{name}_oracle{suffix}.jsonl"
        if task_path.exists() and oracle_path.exists():
            print(f"[data] {name}: cached", flush=True)
            continue
        tasks = _generate_split(name, depth, count, seed, max(1, args.workers))
        C.write_jsonl(task_path, tasks)
        started = time.perf_counter()
        oracle = _oracle_split(tasks, max(1, args.workers))
        C.write_jsonl(oracle_path, oracle)
        print(
            json.dumps(
                {
                    "split": name,
                    "tasks": len(tasks),
                    "depth": depth,
                    "oracle_wall_seconds": time.perf_counter() - started,
                    "successful_skeletons": sum(row["successful_skeleton_count"] for row in oracle),
                    "hidden_used_for_labels": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

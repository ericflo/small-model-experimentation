#!/usr/bin/env python3
"""Select bounded-compute parent failures and emit clean oracle restarts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SOURCE = EXP / "data" / "rollout_tasks_seed77114.jsonl"
MANIFEST = EXP / "data" / "rollout_task_manifest.json"
ROLLOUT = EXP / "runs" / "parent_rollout" / "seed66114.jsonl"
ROLLOUT_RECEIPT = EXP / "runs" / "parent_rollout" / "seed66114.receipt.json"
INVENTORY = EXP / "data" / "failure_inventory_seed66114.json"
RESTART_SOURCE = EXP / "data" / "counterfactual_restart_source.jsonl"
SELECTION_RECEIPT = EXP / "data" / "restart_selection_receipt.json"
SELECTION_SEED = 55114
QUOTA_PER_SKILL = 4
THINK_BUDGET = 128
EXPECTED_SKILLS = (
    "induct", "execute", "select", "trace", "verify", "count", "repair",
    "optimize", "abstain", "state", "order", "probe", "route",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def jsonl_bytes(rows: list[dict]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for row in rows
        )
        + "\n"
    ).encode()


def expected_answer(source: dict) -> str:
    prefix = "ANSWER: "
    value = source.get("answer", "")
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"malformed oracle answer: {source.get('task_id')}")
    return value.removeprefix(prefix).strip()


def extract_answer(text: str) -> str | None:
    tail = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    matches = re.findall(r"(?:^|\n)ANSWER:\s*([^\n<]+)", tail)
    if not matches:
        return None
    return matches[-1].strip()


def classify(source: dict, rollout: dict) -> dict:
    outputs = rollout.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError(f"expected one rollout output: {source.get('task_id')}")
    output = outputs[0]
    text = output.get("text")
    think_tokens = output.get("n_thinking_tokens")
    if not isinstance(text, str) or not isinstance(think_tokens, int):
        raise ValueError(f"rollout lacks text/token accounting: {source.get('task_id')}")
    observed = extract_answer(text)
    target = expected_answer(source)
    cap_contact = bool(
        output.get("truncated") is True
        or output.get("finish_reason") == "length"
        or output.get("thinking_closed") is not True
    )
    missing_answer = observed is None
    wrong_answer = observed is not None and observed != target
    over_budget = think_tokens > THINK_BUDGET
    reasons = [
        name
        for name, active in (
            ("cap_contact", cap_contact),
            ("missing_answer", missing_answer),
            ("wrong_answer", wrong_answer),
            ("over_think_budget", over_budget),
        )
        if active
    ]
    hard_failure = cap_contact or missing_answer or wrong_answer
    return {
        "task_id": source["task_id"],
        "skill": source["selection_skill"],
        "kind": source["kind"],
        "surface": source["surface"],
        "level": source["level"],
        "eligible": bool(reasons),
        "hard_failure": hard_failure,
        "reasons": reasons,
        "expected_answer": target,
        "observed_answer": observed,
        "n_thinking_tokens": think_tokens,
        "n_sampled_tokens": output.get("n_sampled_tokens"),
        "thinking_closed": output.get("thinking_closed"),
        "finish_reason": output.get("finish_reason"),
        "parent_output_sha256": sha256_bytes(text.encode()),
    }


def deterministic_rank(item: dict) -> tuple:
    severity = 0 if item["hard_failure"] else 1
    tie = hashlib.sha256(
        f"{SELECTION_SEED}:{item['skill']}:{item['task_id']}".encode()
    ).hexdigest()
    return (severity, -int(item["n_thinking_tokens"]), tie)


def select_inventory(items: list[dict]) -> tuple[list[dict], dict[str, int]]:
    by_skill: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if item["skill"] not in EXPECTED_SKILLS:
            raise ValueError(f"unexpected skill in inventory: {item['skill']}")
        if item["eligible"]:
            by_skill[item["skill"]].append(item)
    availability = {skill: len(by_skill[skill]) for skill in EXPECTED_SKILLS}
    if any(availability[skill] < QUOTA_PER_SKILL for skill in EXPECTED_SKILLS):
        return [], availability
    selected = [
        item
        for skill in EXPECTED_SKILLS
        for item in sorted(by_skill[skill], key=deterministic_rank)[:QUOTA_PER_SKILL]
    ]
    return selected, availability


def restart_row(source: dict, item: dict, rollout_sha256: str) -> dict:
    skill = source["selection_skill"]
    row = {
        "messages": source["messages"],
        "think": source["think"],
        "answer": source["answer"],
        "kind": f"u_counterfactual_restart_{skill}",
        "family": "universal",
        "surface": source["surface"],
        "level": source["level"],
        "n_think_tokens": source["n_think_tokens"],
        "row_weight": 1.0,
        "task_id": f"restart_{source['task_id']}",
        "failure_selection": {
            "parent_task_id": source["task_id"],
            "parent_rollout_sha256": rollout_sha256,
            "parent_output_sha256": item["parent_output_sha256"],
            "parent_failure_reasons": item["reasons"],
            "parent_n_thinking_tokens": item["n_thinking_tokens"],
            "selected_from_original_prompt": True,
            "parent_prefix_in_training_context": False,
            "oracle_truth_valid": source.get("_audit", {}).get("truth_valid") is True,
        },
    }
    if "assistant_prefix_token_ids" in row or not row["failure_selection"]["oracle_truth_valid"]:
        raise ValueError(f"invalid counterfactual restart: {source['task_id']}")
    return row


def authenticate_inputs() -> tuple[list[dict], list[dict], dict, dict]:
    for path in (SOURCE, MANIFEST, ROLLOUT, ROLLOUT_RECEIPT):
        if not path.is_file():
            raise ValueError(f"required selection input is absent: {path}")
    manifest = load_json(MANIFEST)
    receipt = load_json(ROLLOUT_RECEIPT)
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != 77114
        or manifest.get("rows") != 624
        or manifest.get("source", {}).get("sha256") != sha256_file(SOURCE)
        or receipt.get("experiment_id") != EXP.name
        or receipt.get("seed") != 66114
        or receipt.get("rows") != 624
        or receipt.get("rollouts_sha256") != sha256_file(ROLLOUT)
        or receipt.get("benchmark_data_read") is not False
    ):
        raise ValueError("selection input authentication failed")
    source = load_jsonl(SOURCE)
    rollout = load_jsonl(ROLLOUT)
    if len(source) != 624 or len(rollout) != 624:
        raise ValueError("selection input row count changed")
    return source, rollout, manifest, receipt


def build_payloads() -> tuple[dict, bytes | None, dict]:
    source_rows, rollout_rows, manifest, receipt = authenticate_inputs()
    source_by_id = {row["task_id"]: row for row in source_rows}
    rollout_by_id = {row.get("id"): row for row in rollout_rows}
    if set(source_by_id) != set(rollout_by_id):
        raise ValueError("rollout task identities differ from source identities")
    items = [classify(source_by_id[task_id], rollout_by_id[task_id]) for task_id in source_by_id]
    selected, availability = select_inventory(items)
    selected_ids = {item["task_id"] for item in selected}
    selected_by_skill = Counter(item["skill"] for item in selected)
    reasons = Counter(reason for item in items for reason in item["reasons"])
    inventory = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "failure_inventory",
        "selection_seed": SELECTION_SEED,
        "think_budget": THINK_BUDGET,
        "quota_per_skill": QUOTA_PER_SKILL,
        "skills": list(EXPECTED_SKILLS),
        "rows": len(items),
        "eligible_rows": sum(item["eligible"] for item in items),
        "hard_failure_rows": sum(item["hard_failure"] for item in items),
        "availability_by_skill": availability,
        "failure_reasons": dict(sorted(reasons.items())),
        "quota_pass": bool(selected),
        "selected_task_ids": sorted(selected_ids),
        "items": items,
        "source_sha256": manifest["source"]["sha256"],
        "rollout_sha256": receipt["rollouts_sha256"],
        "benchmark_data_read": False,
    }
    restart_bytes = None
    if selected:
        restart_rows = [
            restart_row(source_by_id[item["task_id"]], item, receipt["rollouts_sha256"])
            for item in selected
        ]
        restart_bytes = jsonl_bytes(restart_rows)
        if selected_by_skill != Counter({skill: QUOTA_PER_SKILL for skill in EXPECTED_SKILLS}):
            raise ValueError("selected restart curriculum lost its skill balance")
    selection_receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "counterfactual_restart_selection",
        "outcome": "PASS_RESTART_QUOTAS" if selected else "STOP_INSUFFICIENT_FAILURES",
        "inventory_sha256": sha256_bytes(
            (json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        ),
        "selected_rows": len(selected),
        "selected_rows_by_skill": dict(sorted(selected_by_skill.items())),
        "restart_source_sha256": sha256_bytes(restart_bytes) if restart_bytes else None,
        "full_oracle_restart_from_original_prompt": True,
        "parent_prefix_in_training_context": False,
        "target_exposure_match_pending": bool(selected),
        "training_authorized": False,
        "next_required_review": "exact three-axis exposure feasibility and adversarial compute review" if selected else None,
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }
    return inventory, restart_bytes, selection_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory, restart_bytes, receipt = build_payloads()
    outputs: dict[Path, bytes] = {
        INVENTORY: (json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
        SELECTION_RECEIPT: (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    }
    if restart_bytes is not None:
        outputs[RESTART_SOURCE] = restart_bytes
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                parser.error(f"derived restart artifact is absent or changed: {path}")
        if restart_bytes is None and RESTART_SOURCE.exists():
            parser.error("restart source exists after a failed quota gate")
    else:
        conflict = next((path for path in (INVENTORY, RESTART_SOURCE, SELECTION_RECEIPT) if path.exists()), None)
        if conflict is not None:
            parser.error(f"refusing to overwrite selection artifact: {conflict}")
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if restart_bytes is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

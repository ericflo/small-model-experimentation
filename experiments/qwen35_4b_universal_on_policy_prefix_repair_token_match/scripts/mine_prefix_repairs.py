#!/usr/bin/env python3
"""Mine masked corrective continuations from authenticated parent failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
TASKS = EXP / "data" / "rollout_tasks.jsonl"
TASK_MANIFEST = EXP / "data" / "rollout_task_manifest.json"
ROLLOUTS = EXP / "runs" / "parent_rollout" / "seed66113.jsonl"
ROLLOUT_METADATA = EXP / "runs" / "parent_rollout" / "seed66113.meta.json"
ROLLOUT_RECEIPT = EXP / "runs" / "parent_rollout" / "seed66113.receipt.json"
OUT = EXP / "data" / "prefix_repair_source.jsonl"
INVENTORY = EXP / "data" / "prefix_failure_inventory.json"
MERGED_PARENT = ROOT / "large_artifacts" / EXP.name / "merged" / "close_xi_parent"
RUNNER = EXP / "src" / "vllm_runner.py"
SELECTION_SEED = 77113
ROLLOUT_SEED = 66113
QUOTA_PER_CLASS = 10
MAX_NEW_TOKENS = 1024
COMMIT_THINK_LIMIT = 32
THINK_CLOSE_ID = 248069
FORBIDDEN_PREFIX_IDS = {248044, 248046, THINK_CLOSE_ID}
ANSWER_RE = re.compile(r"(?:^|\n)ANSWER:\s*(.*?)(?=\n|<\||</|$)", re.DOTALL)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or malformed JSONL: {path}")
    return rows


def parse_answer(text: str) -> str | None:
    matches = [match.group(1).strip() for match in ANSWER_RE.finditer(text)]
    return matches[-1] if matches and matches[-1] else None


def canonical_answer(text: str, expected: str) -> bool:
    _, separator, tail = text.partition("</think>")
    if not separator:
        return False
    for token in ("<|im_end|>", "<|endoftext|>"):
        tail = tail.replace(token, "")
    return tail.strip() == f"ANSWER: {expected}"


def failure_reasons(task: dict, output: dict) -> list[str]:
    expected = task["answer"].removeprefix("ANSWER: ").strip()
    parsed = parse_answer(output["text"])
    reasons: list[str] = []
    if output.get("truncated") or output.get("finish_reason") == "length":
        reasons.append("generation_cap")
    if parsed is None:
        reasons.append("missing_answer")
    elif parsed != expected:
        reasons.append("wrong_answer")
    elif not canonical_answer(output["text"], expected):
        reasons.append("noncanonical_serialization")

    failure_class = task["failure_class"]
    thought = output["text"].partition("</think>")[0].lower()
    if failure_class == "commit_serialization" and int(
        output.get("n_thinking_tokens", 0)
    ) > COMMIT_THINK_LIMIT:
        reasons.append("delayed_commit")
    if failure_class == "declaration_operation" and any(
        phrase in thought
        for phrase in (
            "advance every item",
            "advance each item",
            "apply the cycle",
            "cycle operation",
            "shift every item",
        )
    ):
        reasons.append("declaration_executed_as_operation")
    return list(dict.fromkeys(reasons))


def thought_prefix(output: dict, reasons: list[str]) -> tuple[list[int], str, str]:
    token_ids = list(output["token_ids"])
    close_index = token_ids.index(THINK_CLOSE_ID) if THINK_CLOSE_ID in token_ids else len(token_ids)
    thought_ids = token_ids[:close_index]
    thought_text = output["text"].partition("</think>")[0]
    boundary = "answer_boundary"
    if "delayed_commit" in reasons:
        cutoff = min(COMMIT_THINK_LIMIT + 1, len(thought_ids))
        thought_ids = thought_ids[:cutoff]
        # Exact text is authenticated later by decoding these IDs in the trainer.
        # Keep a null sentinel here when the policy boundary is a strict token cut.
        thought_text = ""
        boundary = "first_token_beyond_commit_budget"
    elif "generation_cap" in reasons:
        boundary = "generation_cap_boundary"
    elif "declaration_executed_as_operation" in reasons:
        boundary = "first_observable_declaration_misuse_by_close"
    if not thought_ids or any(token in FORBIDDEN_PREFIX_IDS for token in thought_ids):
        raise ValueError("failure has no clean thinking-channel parent prefix")
    return thought_ids, thought_text, boundary


def correction_text(task: dict) -> str:
    lead = {
        "declaration_operation": (
            "RECOVER: the cycle line is reference data, not an extra operation. "
            "Execute only the listed procedure and carry state forward."
        ),
        "state_transition": (
            "RECOVER: restart from the last trusted state, apply each listed operation once, "
            "and verify the carried state before committing."
        ),
        "bounded_induction": (
            "RECOVER: bound the search. Test a complete two-operation hypothesis against every "
            "probe, reject it on the first contradiction, and stop at a verified decomposition."
        ),
        "probe_scoring": (
            "RECOVER: recompute every hypothesis output independently, then count distinct "
            "outputs exactly once per candidate probe."
        ),
        "repair_propagation": (
            "RECOVER: locate the first bad transition and propagate the corrected state through "
            "every remaining operation."
        ),
        "commit_serialization": (
            "RECOVER: the verified final state is already available. Perform no operation and "
            "commit it exactly now."
        ),
    }[task["failure_class"]]
    return lead + "\n" + task["oracle_think"].strip()


def severity(reasons: list[str]) -> int:
    weights = {
        "generation_cap": 6,
        "missing_answer": 5,
        "wrong_answer": 4,
        "declaration_executed_as_operation": 3,
        "noncanonical_serialization": 2,
        "delayed_commit": 1,
    }
    return max((weights[value] for value in reasons), default=0)


def analyze(tasks: list[dict], rollouts: list[dict]) -> tuple[list[dict], dict]:
    by_task = {row["task_id"]: row for row in tasks}
    by_rollout = {row["id"]: row for row in rollouts}
    if len(by_task) != len(tasks) or len(by_rollout) != len(rollouts):
        raise ValueError("duplicate task or rollout id")
    if set(by_task) != set(by_rollout):
        raise ValueError("rollout ids do not exactly cover the frozen task source")

    candidates: dict[str, list[dict]] = defaultdict(list)
    inventory_rows: list[dict] = []
    for task_id in sorted(by_task):
        task = by_task[task_id]
        rollout = by_rollout[task_id]
        if rollout.get("meta", {}).get("failure_class") != task["failure_class"]:
            raise ValueError(f"rollout metadata mismatch: {task_id}")
        outputs = rollout.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError(f"expected exactly one parent output: {task_id}")
        output = outputs[0]
        reasons = failure_reasons(task, output)
        row = {
            "task_id": task_id,
            "failure_class": task["failure_class"],
            "reasons": reasons,
            "failed": bool(reasons),
            "parsed": parse_answer(output["text"]),
            "expected": task["answer"].removeprefix("ANSWER: ").strip(),
            "n_thinking_tokens": output.get("n_thinking_tokens"),
            "n_answer_tokens": output.get("n_answer_tokens"),
            "truncated": output.get("truncated"),
        }
        inventory_rows.append(row)
        if not reasons:
            continue
        try:
            prefix_ids, prefix_text, boundary = thought_prefix(output, reasons)
        except ValueError:
            row["reachable_prefix"] = False
            continue
        row["reachable_prefix"] = True
        tie = sha256_bytes(
            f"{SELECTION_SEED}:{task_id}:{sha256_bytes(bytes(str(output['token_ids']), 'utf-8'))}".encode()
        )
        candidates[task["failure_class"]].append(
            {
                "rank": (-severity(reasons), tie),
                "task": task,
                "output": output,
                "reasons": reasons,
                "prefix_ids": prefix_ids,
                "prefix_text": prefix_text,
                "boundary": boundary,
            }
        )

    selected: list[dict] = []
    availability = {}
    for failure_class in sorted({task["failure_class"] for task in tasks}):
        ranked = sorted(candidates.get(failure_class, []), key=lambda row: row["rank"])
        availability[failure_class] = len(ranked)
        for candidate in ranked[:QUOTA_PER_CLASS]:
            task = candidate["task"]
            output = candidate["output"]
            prefix_ids = candidate["prefix_ids"]
            prefix_text = candidate["prefix_text"]
            selected.append(
                {
                    "messages": task["messages"],
                    "assistant_prefix_token_ids": prefix_ids,
                    **({"assistant_prefix_text": prefix_text} if prefix_text else {}),
                    "prefix_loss_masked": True,
                    "think": correction_text(task),
                    "answer": task["answer"],
                    "kind": f"u_prefix_repair_{task['failure_class']}",
                    "family": "universal_on_policy_prefix_repair",
                    "failure_class": task["failure_class"],
                    "row_weight": 1.0,
                    "task_id": f"repair_{task['task_id']}",
                    "provenance": {
                        "source_task_id": task["task_id"],
                        "source_output_token_ids_sha256": sha256_bytes(
                            json.dumps(output["token_ids"], separators=(",", ":")).encode()
                        ),
                        "prefix_token_ids_sha256": sha256_bytes(
                            json.dumps(prefix_ids, separators=(",", ":")).encode()
                        ),
                        "prefix_tokens": len(prefix_ids),
                        "failure_reasons": candidate["reasons"],
                        "first_observable_boundary": candidate["boundary"],
                        "parent_seed_stage1": output.get("seed_stage1"),
                    },
                }
            )
    selected.sort(key=lambda row: row["task_id"])
    counts = Counter(row["failure_class"] for row in selected)
    inventory = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "selection_seed": SELECTION_SEED,
        "rollout_seed": ROLLOUT_SEED,
        "quota_per_failure_class": QUOTA_PER_CLASS,
        "available_reachable_failures": availability,
        "selected": dict(sorted(counts.items())),
        "quota_satisfied": all(
            counts[name] == QUOTA_PER_CLASS for name in availability
        ),
        "task_outcomes": inventory_rows,
    }
    return selected, inventory


def validate_authentication() -> tuple[list[dict], list[dict]]:
    manifest = load_json(TASK_MANIFEST)
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != SELECTION_SEED
        or manifest.get("rows") != 288
        or manifest.get("runner_input_excludes_hidden_oracle_fields") is not True
        or manifest.get("benchmark_data_read") is not False
        or manifest.get("source", {}).get("sha256") != sha256_file(TASKS)
    ):
        raise ValueError("rollout task manifest failed authentication")
    receipt = load_json(ROLLOUT_RECEIPT)
    metadata = load_json(ROLLOUT_METADATA)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("experiment_id") != EXP.name
        or receipt.get("seed") != ROLLOUT_SEED
        or receipt.get("rollouts_sha256") != sha256_file(ROLLOUTS)
        or receipt.get("metadata_sha256") != sha256_file(ROLLOUT_METADATA)
        or receipt.get("runner_sha256") != sha256_file(RUNNER)
        or Path(receipt.get("model", "")).resolve() != MERGED_PARENT.resolve()
        or metadata.get("schema_version") != 4
        or Path(metadata.get("model", "")).resolve() != MERGED_PARENT.resolve()
        or metadata.get("model_revision") is not None
        or metadata.get("adapter") is not None
        or metadata.get("sampling", {}).get("thinking") != "natural"
        or metadata.get("sampling", {}).get("n") != 1
        or metadata.get("sampling", {}).get("max_tokens") != MAX_NEW_TOKENS
        or metadata.get("sampling", {}).get("greedy") is not True
        or metadata.get("sampling", {}).get("run_seed") != ROLLOUT_SEED
        or metadata.get("input", {}).get("sha256")
        != manifest.get("runner_input", {}).get("sha256")
    ):
        raise ValueError("parent rollout receipt or vLLM metadata failed authentication")
    return load_jsonl(TASKS), load_jsonl(ROLLOUTS)


def render_rows(rows: list[dict]) -> bytes:
    return (
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
        + "\n"
    ).encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tasks, rollouts = validate_authentication()
    rows, inventory = analyze(tasks, rollouts)
    source = render_rows(rows)
    inventory.update(
        {
            "source_path": OUT.relative_to(EXP).as_posix(),
            "source_rows": len(rows),
            "source_sha256": sha256_bytes(source),
        }
    )
    inventory_bytes = (
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if args.check:
        if (
            not OUT.is_file()
            or OUT.read_bytes() != source
            or not INVENTORY.is_file()
            or INVENTORY.read_bytes() != inventory_bytes
        ):
            parser.error("prefix-repair source or inventory is absent or changed")
    else:
        if OUT.exists() or INVENTORY.exists():
            parser.error("refusing to overwrite prefix-repair source or inventory")
        INVENTORY.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY.write_bytes(inventory_bytes)
        if inventory["quota_satisfied"]:
            OUT.write_bytes(source)
    print(json.dumps({key: value for key, value in inventory.items() if key != "task_outcomes"}, indent=2, sort_keys=True))
    if not inventory["quota_satisfied"]:
        raise SystemExit("on-policy failure inventory did not satisfy every frozen class quota")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

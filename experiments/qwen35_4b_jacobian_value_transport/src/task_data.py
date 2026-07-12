"""Fresh procedural tasks and prompt-local causal positive controls.

This file is intentionally self-contained. It neither imports nor reads anything
under ``benchmarks/``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from io_utils import artifact_receipt, write_json, write_jsonl


CONCEPT_CANDIDATES = (
    "cat", "dog", "horse", "tiger", "apple", "lemon", "river", "ocean",
    "silver", "gold", "circle", "square", "winter", "summer", "music", "dance",
    "bread", "glass", "stone", "cloud", "green", "purple", "north", "south",
)


def _reverse(xs: list[int], _: int | None) -> list[int]:
    return xs[::-1]


def _sort_asc(xs: list[int], _: int | None) -> list[int]:
    return sorted(xs)


def _sort_desc(xs: list[int], _: int | None) -> list[int]:
    return sorted(xs, reverse=True)


def _abs_all(xs: list[int], _: int | None) -> list[int]:
    return [abs(x) for x in xs]


def _square(xs: list[int], _: int | None) -> list[int]:
    return [x * x for x in xs]


def _negate(xs: list[int], _: int | None) -> list[int]:
    return [-x for x in xs]


def _running_sum(xs: list[int], _: int | None) -> list[int]:
    return [sum(xs[: index + 1]) for index in range(len(xs))]


def _adjacent_diff(xs: list[int], _: int | None) -> list[int]:
    return [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]


def _add_k(xs: list[int], k: int | None) -> list[int]:
    assert k is not None
    return [x + k for x in xs]


def _mul_k(xs: list[int], k: int | None) -> list[int]:
    assert k is not None
    return [x * k for x in xs]


def _take_k(xs: list[int], k: int | None) -> list[int]:
    assert k is not None
    return xs[:k]


def _rotate_k(xs: list[int], k: int | None) -> list[int]:
    assert k is not None
    return xs[k % len(xs):] + xs[: k % len(xs)] if xs else []


Operation = tuple[Callable[[list[int], int | None], list[int]], tuple[int | None, ...]]
OPERATIONS: dict[str, Operation] = {
    "reverse": (_reverse, (None,)),
    "sort_asc": (_sort_asc, (None,)),
    "sort_desc": (_sort_desc, (None,)),
    "abs_all": (_abs_all, (None,)),
    "square": (_square, (None,)),
    "negate": (_negate, (None,)),
    "running_sum": (_running_sum, (None,)),
    "adjacent_diff": (_adjacent_diff, (None,)),
    "add_k": (_add_k, (-3, -2, -1, 1, 2, 3)),
    "mul_k": (_mul_k, (-2, 2, 3)),
    "take_k": (_take_k, (1, 2, 3, 4)),
    "rotate_k": (_rotate_k, (1, 2, 3)),
}


def _string_reverse(value: str, _: int | None) -> str:
    return value[::-1]


def _string_sort(value: str, _: int | None) -> str:
    return "".join(sorted(value))


def _string_remove_vowels(value: str, _: int | None) -> str:
    return "".join(character for character in value if character not in "aeiou")


def _string_dedup(value: str, _: int | None) -> str:
    return "".join(dict.fromkeys(value))


def _string_shift(value: str, k: int | None) -> str:
    assert k is not None
    return "".join(chr((ord(character) - 97 + k) % 26 + 97) for character in value)


def _string_take(value: str, k: int | None) -> str:
    assert k is not None
    return value[:k]


def _string_rotate(value: str, k: int | None) -> str:
    assert k is not None
    return value[k % len(value):] + value[: k % len(value)] if value else value


STRING_OPERATIONS: dict[str, tuple[Callable[[str, int | None], str], tuple[int | None, ...]]] = {
    "reverse": (_string_reverse, (None,)),
    "sort_chars": (_string_sort, (None,)),
    "remove_vowels": (_string_remove_vowels, (None,)),
    "dedup_all": (_string_dedup, (None,)),
    "shift_k": (_string_shift, (1, 2, 3, 13)),
    "take_k": (_string_take, (1, 2, 3, 4)),
    "rotate_k": (_string_rotate, (1, 2, 3)),
}


def _reg_a_plus_b(value: list[int], _: int | None) -> list[int]:
    return [value[0] + value[1], value[1], value[2]]


def _reg_b_plus_c(value: list[int], _: int | None) -> list[int]:
    return [value[0], value[1] + value[2], value[2]]


def _reg_neg_a(value: list[int], _: int | None) -> list[int]:
    return [-value[0], value[1], value[2]]


def _reg_swap_ab(value: list[int], _: int | None) -> list[int]:
    return [value[1], value[0], value[2]]


def _reg_rotate(value: list[int], _: int | None) -> list[int]:
    return [value[2], value[0], value[1]]


def _reg_inc_a(value: list[int], k: int | None) -> list[int]:
    assert k is not None
    return [value[0] + k, value[1], value[2]]


def _reg_mul_a(value: list[int], k: int | None) -> list[int]:
    assert k is not None
    return [value[0] * k, value[1], value[2]]


REGISTER_OPERATIONS: dict[str, Operation] = {
    "a_plus_b": (_reg_a_plus_b, (None,)),
    "b_plus_c": (_reg_b_plus_c, (None,)),
    "neg_a": (_reg_neg_a, (None,)),
    "swap_ab": (_reg_swap_ab, (None,)),
    "rotate": (_reg_rotate, (None,)),
    "inc_a": (_reg_inc_a, (-2, -1, 1, 2, 3)),
    "mul_a": (_reg_mul_a, (-1, 2, 3)),
}


def op_text(name: str, parameter: int | None) -> str:
    return name if parameter is None else f"{name}({parameter})"


def apply_pipeline(xs: list[int], pipeline: list[tuple[str, int | None]]) -> list[int]:
    state = list(xs)
    for name, parameter in pipeline:
        state = OPERATIONS[name][0](state, parameter)
        if len(state) > 64 or any(abs(value) > 10**7 for value in state):
            raise ValueError("pipeline state exceeded safety bound")
    return state


def _random_input(rng: random.Random) -> list[int]:
    return [rng.randint(-9, 9) for _ in range(rng.randint(4, 8))]


def _pipeline_signature(pipeline: list[tuple[str, int | None]], inputs: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(apply_pipeline(xs, pipeline)) for xs in inputs)


def make_task(rng: random.Random, *, task_id: str, depth: int, visible: int, hidden: int) -> dict[str, Any]:
    names = tuple(OPERATIONS)
    for _ in range(500):
        pipeline = []
        for _step in range(depth):
            name = rng.choice(names)
            parameter = rng.choice(OPERATIONS[name][1])
            pipeline.append((name, parameter))
        inputs: list[list[int]] = []
        seen = set()
        while len(inputs) < visible + hidden:
            value = _random_input(rng)
            key = tuple(value)
            if key not in seen:
                seen.add(key)
                inputs.append(value)
        try:
            signature = _pipeline_signature(pipeline, inputs)
        except ValueError:
            continue
        if signature == tuple(tuple(xs) for xs in inputs) or len(set(signature)) < 3:
            continue
        if depth > 1:
            shallow_match = False
            for name, (_function, parameters) in OPERATIONS.items():
                for parameter in parameters:
                    if _pipeline_signature([(name, parameter)], inputs) == signature:
                        shallow_match = True
                        break
                if shallow_match:
                    break
            if shallow_match:
                continue
        examples = [
            {"input": xs, "output": list(output)}
            for xs, output in zip(inputs, signature, strict=True)
        ]
        return {
            "task_id": task_id,
            "depth": depth,
            "target_ops": [op_text(name, parameter) for name, parameter in pipeline],
            "first_op": pipeline[0][0],
            "visible": examples[:visible],
            "hidden": examples[visible:],
        }
    raise RuntimeError(f"failed to construct {task_id}")


def _apply_generic(value: Any, pipeline: list[tuple[str, int | None]], operations: dict[str, tuple[Callable, tuple]]) -> Any:
    state = value[:] if isinstance(value, list) else value
    for name, parameter in pipeline:
        state = operations[name][0](state, parameter)
        if hasattr(state, "__len__") and len(state) > 64:
            raise ValueError("generic state exceeded length bound")
        if isinstance(state, list) and any(abs(item) > 10**7 for item in state):
            raise ValueError("generic state exceeded numeric bound")
    return state


def make_held_family_task(
    rng: random.Random,
    *,
    family: str,
    task_id: str,
    depth: int,
    visible: int,
    hidden: int,
) -> dict[str, Any]:
    if family == "string":
        operations = STRING_OPERATIONS
        make_input: Callable[[], Any] = lambda: "".join(
            rng.choice("abcdefghijklmnop") for _ in range(rng.randint(4, 8))
        )
    elif family == "register":
        operations = REGISTER_OPERATIONS
        make_input = lambda: [rng.randint(-9, 9) for _ in range(3)]
    else:
        raise ValueError(f"unknown held family {family!r}")
    names = tuple(operations)
    for _ in range(500):
        pipeline = []
        for _step in range(depth):
            name = rng.choice(names)
            pipeline.append((name, rng.choice(operations[name][1])))
        inputs = []
        seen = set()
        while len(inputs) < visible + hidden:
            value = make_input()
            key = tuple(value) if isinstance(value, list) else value
            if key not in seen:
                seen.add(key)
                inputs.append(value)
        try:
            outputs = [_apply_generic(value, pipeline, operations) for value in inputs]
        except ValueError:
            continue
        input_keys = [tuple(value) if isinstance(value, list) else value for value in inputs]
        output_keys = [tuple(value) if isinstance(value, list) else value for value in outputs]
        if input_keys == output_keys or len(set(output_keys)) < 3:
            continue
        shallow_match = False
        for name, (_function, parameters) in operations.items():
            for parameter in parameters:
                candidate = [_apply_generic(value, [(name, parameter)], operations) for value in inputs]
                candidate_keys = [tuple(value) if isinstance(value, list) else value for value in candidate]
                if candidate_keys == output_keys:
                    shallow_match = True
                    break
            if shallow_match:
                break
        if shallow_match:
            continue
        examples = [
            {"input": value, "output": output}
            for value, output in zip(inputs, outputs, strict=True)
        ]
        return {
            "task_id": task_id,
            "family": family,
            "depth": depth,
            "target_ops": [op_text(name, parameter) for name, parameter in pipeline],
            "first_op": pipeline[0][0],
            "visible": examples[:visible],
            "hidden": examples[visible:],
        }
    raise RuntimeError(f"failed to construct {task_id}")


def first_op_prompt(task: dict[str, Any]) -> str:
    examples = "\n".join(
        f"transform({row['input']!r}) == {row['output']!r}" for row in task["visible"]
    )
    names = ", ".join(OPERATIONS)
    return (
        "Infer the hidden sequence of list operations from these examples:\n"
        f"{examples}\n\nAvailable operation types: {names}.\n"
        "Think briefly inside the thinking section. Then answer with exactly one final line "
        "formatted `First: <operation_type>`."
    )


def parse_first_op(text: str) -> str | None:
    tail = text.split("First:")[-1].strip().split()
    if tail and tail[0].strip("`.,:;\n") in OPERATIONS:
        return tail[0].strip("`.,:;\n")
    return None


def verify_first_op(task: dict[str, Any], text: str) -> bool:
    return parse_first_op(text) == task["first_op"]


def make_positive_controls(rng: random.Random, count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        source, target = rng.sample(CONCEPT_CANDIDATES, 2)
        source_value, target_value = rng.sample(range(10), 2)
        rows.append({
            "item_id": f"pc-{index:04d}",
            "source": source,
            "target": target,
            "source_value": source_value,
            "target_value": target_value,
            "direct_prompt": (
                f"Hold the selected concept in mind. The selected concept is {source}. "
                f"The alternative concept is {target}. Think silently, then answer exactly "
                f"`Concept: {source}`."
            ),
            "consequence_prompt": (
                f"Use this temporary mapping: {source} maps to {source_value}; "
                f"{target} maps to {target_value}. The selected concept is {source}. "
                "Think silently about the selection, then answer with exactly one line "
                f"formatted `Value: {source_value}`."
            ),
        })
    return rows


def make_lens_corpus(rng: random.Random, count: int) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        words = rng.sample(CONCEPT_CANDIDATES, 8)
        order = rng.sample(tuple(OPERATIONS), 6)
        text = (
            f"Procedure note {index}: compare {', '.join(words[:4])} with "
            f"{', '.join(words[4:])}. The available transformations are "
            f"{', '.join(order)}. First inspect the input, then retain the selected "
            "concept, apply each declared operation in order, verify the intermediate "
            "state, and report only the requested consequence."
        )
        rows.append({"corpus_id": f"lens-{index:04d}", "text": text})
    return rows


def _digest_rows(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def build_splits(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seeds"]["split"])
    data = config["data"]
    specifications = {
        "value_calibration": (int(data["value_calibration_tasks"]), int(data["anchor_depth"]), seed + 11),
        "iid_eval": (int(data["iid_eval_tasks"]), int(data["anchor_depth"]), seed + 23),
        "hard_eval": (int(data["hard_eval_tasks"]), int(data["hard_depth"]), seed + 37),
    }
    all_ids: set[str] = set()
    receipts: dict[str, Any] = {}
    split_digests: dict[str, str] = {}
    for split, (count, depth, split_seed) in specifications.items():
        rng = random.Random(split_seed)
        rows = [
            make_task(
                rng,
                task_id=f"{split}-{index:05d}",
                depth=depth,
                visible=int(data["visible_examples"]),
                hidden=int(data["hidden_examples"]),
            )
            for index in range(count)
        ]
        ids = {str(row["task_id"]) for row in rows}
        if all_ids & ids:
            raise AssertionError("task ID overlap")
        all_ids |= ids
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        receipts[split] = artifact_receipt(path, rows=len(rows))
        split_digests[split] = _digest_rows(rows)

    held_count = int(data["held_family_tasks_per_family"])
    for offset, family in enumerate(("string", "register"), start=1):
        split = f"held_{family}_eval"
        rng = random.Random(seed + 100 + offset)
        rows = [
            make_held_family_task(
                rng,
                family=family,
                task_id=f"{split}-{index:05d}",
                depth=int(data["anchor_depth"]),
                visible=int(data["visible_examples"]),
                hidden=int(data["hidden_examples"]),
            )
            for index in range(held_count)
        ]
        ids = {str(row["task_id"]) for row in rows}
        if all_ids & ids:
            raise AssertionError("held-family task ID overlap")
        all_ids |= ids
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        receipts[split] = artifact_receipt(path, rows=len(rows))
        split_digests[split] = _digest_rows(rows)

    lens_rng = random.Random(int(config["seeds"]["lens_corpus"]))
    lens_rows = make_lens_corpus(lens_rng, int(data["lens_fit_prompts"]))
    lens_path = output_dir / "lens_fit.jsonl"
    write_jsonl(lens_path, lens_rows)
    receipts["lens_fit"] = artifact_receipt(lens_path, rows=len(lens_rows))

    control_rng = random.Random(seed + 51)
    controls = make_positive_controls(control_rng, int(data["positive_control_items"]))
    control_path = output_dir / "positive_control.jsonl"
    write_jsonl(control_path, controls)
    receipts["positive_control"] = artifact_receipt(control_path, rows=len(controls))

    manifest = {
        "schema_version": 1,
        "generator": "src/task_data.py",
        "split_seed": seed,
        "firewall": {"benchmark_content_used": False, "fresh_procedural_only": True},
        "counts": {name: value["rows"] for name, value in receipts.items()},
        "task_split_digests": split_digests,
        "artifacts": receipts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest

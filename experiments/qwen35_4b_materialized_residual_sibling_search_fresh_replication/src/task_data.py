"""Exact-depth-three multi-label list tasks for materialized residual search."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

from identity import namespaced_task_id


BoundOperation = tuple[str, int | None]
Program = tuple[BoundOperation, ...]


@dataclass(frozen=True)
class InvalidState:
    """Typed result shared by every partial-DSL execution path."""

    reason: str = "INVALID"


INVALID = InvalidState()
State = list[int] | InvalidState


def _reverse(xs: list[int], _: int | None) -> list[int]:
    return xs[::-1]


def _sort_asc(xs: list[int], _: int | None) -> list[int]:
    return sorted(xs)


def _sort_desc(xs: list[int], _: int | None) -> list[int]:
    return sorted(xs, reverse=True)


def _abs_all(xs: list[int], _: int | None) -> list[int]:
    return [abs(value) for value in xs]


def _square(xs: list[int], _: int | None) -> list[int]:
    return [value * value for value in xs]


def _negate(xs: list[int], _: int | None) -> list[int]:
    return [-value for value in xs]


def _running_sum(xs: list[int], _: int | None) -> list[int]:
    total = 0
    result: list[int] = []
    for value in xs:
        total += value
        result.append(total)
    return result


def _adjacent_diff(xs: list[int], _: int | None) -> list[int]:
    return [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]


def _add_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None:
        raise ValueError("add_k requires a parameter")
    return [value + parameter for value in xs]


def _mul_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None:
        raise ValueError("mul_k requires a parameter")
    return [value * parameter for value in xs]


def _take_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None or parameter < 1:
        raise ValueError("take_k requires a positive parameter")
    return xs[:parameter]


def _rotate_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None or not xs:
        raise ValueError("rotate_k requires a parameter and nonempty input")
    shift = parameter % len(xs)
    return xs[shift:] + xs[:shift]


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

CONCRETE_OPERATIONS: tuple[BoundOperation, ...] = tuple(
    (name, parameter)
    for name, (_function, parameters) in OPERATIONS.items()
    for parameter in parameters
)
ALIASES: tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(24))
OPERATION_TO_ALIAS = dict(zip(CONCRETE_OPERATIONS, ALIASES, strict=True))
ALIAS_TO_OPERATION = dict(zip(ALIASES, CONCRETE_OPERATIONS, strict=True))
DEPTH_TWO_PROGRAMS: tuple[Program, ...] = tuple(
    tuple(values) for values in product(CONCRETE_OPERATIONS, repeat=2)
)
DEPTH_THREE_PROGRAMS: tuple[Program, ...] = tuple(
    tuple(values) for values in product(CONCRETE_OPERATIONS, repeat=3)
)

if len(CONCRETE_OPERATIONS) != 24 or len(DEPTH_THREE_PROGRAMS) != 24**3:
    raise RuntimeError("the frozen 24-operation grammar changed")


def _validate_operation(operation: BoundOperation) -> None:
    if not isinstance(operation, tuple) or len(operation) != 2:
        raise ValueError("operation must be a (name, parameter) tuple")
    name, parameter = operation
    if name not in OPERATIONS or parameter not in OPERATIONS[name][1]:
        raise ValueError(f"invalid bound operation: {operation!r}")


def operation_record(operation: BoundOperation) -> dict[str, int | str | None]:
    _validate_operation(operation)
    return {"name": operation[0], "parameter": operation[1]}


def operation_from_record(value: dict[str, Any]) -> BoundOperation:
    if not isinstance(value, dict) or set(value) != {"name", "parameter"}:
        raise ValueError("operation record requires exact name/parameter keys")
    operation = (value["name"], value["parameter"])
    _validate_operation(operation)
    return operation


def canonical_operation(operation: BoundOperation) -> str:
    _validate_operation(operation)
    name, parameter = operation
    return name if parameter is None else f"{name}({parameter})"


def canonical_program(program: Sequence[BoundOperation]) -> str:
    values = tuple(program)
    if len(values) not in {2, 3}:
        raise ValueError("canonical programs have two or three operations")
    return " | ".join(canonical_operation(operation) for operation in values)


def alias_program(program: Sequence[BoundOperation]) -> str:
    values = tuple(program)
    if len(values) not in {2, 3}:
        raise ValueError("alias programs have two or three operations")
    return " | ".join(OPERATION_TO_ALIAS[operation] for operation in values)


def apply_pipeline(values: list[int], program: Sequence[BoundOperation]) -> State:
    """Execute through the one frozen partial-semantics contract."""

    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, int) or isinstance(value, bool) for value in values)
    ):
        return INVALID
    state = list(values)
    for operation in program:
        try:
            _validate_operation(operation)
            name, parameter = operation
            state = OPERATIONS[name][0](state, parameter)
        except (TypeError, ValueError, OverflowError):
            return INVALID
        if not state or len(state) > 64 or any(abs(value) > 10**7 for value in state):
            return INVALID
    return state


def _state_key(state: State) -> tuple[str | int, ...]:
    return ("INVALID",) if state is INVALID else tuple(state)


def output_vector(program: Sequence[BoundOperation], inputs: Sequence[list[int]]) -> tuple[tuple[str | int, ...], ...]:
    return tuple(_state_key(apply_pipeline(values, program)) for values in inputs)


def valid_output_vector(program: Sequence[BoundOperation], inputs: Sequence[list[int]]) -> tuple[tuple[int, ...], ...] | None:
    rows: list[tuple[int, ...]] = []
    for values in inputs:
        state = apply_pipeline(values, program)
        if state is INVALID:
            return None
        rows.append(tuple(state))
    return tuple(rows)


def behavior_fingerprint(outputs: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(outputs, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _draw_input(rng: random.Random, data: dict[str, Any], *, length: int | None = None) -> list[int]:
    if length is None:
        length = rng.randint(int(data["input_min_length"]), int(data["input_max_length"]))
    return [
        rng.randint(int(data["input_value_min"]), int(data["input_value_max"]))
        for _ in range(length)
    ]


def build_common_panel(config: dict[str, Any]) -> list[list[int]]:
    data = config["data"]
    rng = random.Random(int(config["seeds"]["common_panel"]))
    panel: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    per_length = int(data["common_panel_per_length"])
    for length in range(int(data["input_min_length"]), int(data["input_max_length"]) + 1):
        while sum(len(values) == length for values in panel) < per_length:
            values = _draw_input(rng, data, length=length)
            key = tuple(values)
            if key not in seen:
                seen.add(key)
                panel.append(values)
    if len(panel) != int(data["common_fingerprint_panel_inputs"]):
        raise RuntimeError("common-panel geometry disagrees with configuration")
    return panel


def enumerate_exact_depth_three_functions(
    common_inputs: Sequence[list[int]],
) -> tuple[dict[str, list[Program]], set[str], dict[str, tuple[tuple[int, ...], ...]]]:
    """Return exact-d3 program groups keyed by common-panel function hash."""

    shallow: set[str] = set()
    identity = tuple(tuple(values) for values in common_inputs)
    shallow.add(behavior_fingerprint(identity))
    for operation in CONCRETE_OPERATIONS:
        outputs = valid_output_vector((operation,), common_inputs)
        if outputs is not None:
            shallow.add(behavior_fingerprint(outputs))
    for program in DEPTH_TWO_PROGRAMS:
        outputs = valid_output_vector(program, common_inputs)
        if outputs is not None:
            shallow.add(behavior_fingerprint(outputs))

    groups: dict[str, list[Program]] = defaultdict(list)
    representatives: dict[str, tuple[tuple[int, ...], ...]] = {}
    for program in DEPTH_THREE_PROGRAMS:
        outputs = valid_output_vector(program, common_inputs)
        if outputs is None:
            continue
        fingerprint = behavior_fingerprint(outputs)
        if fingerprint in shallow:
            continue
        if fingerprint in representatives and representatives[fingerprint] != outputs:
            raise RuntimeError("SHA-256 collision in common-panel functions")
        representatives.setdefault(fingerprint, outputs)
        groups[fingerprint].append(program)
    return dict(groups), shallow, representatives


def _visible_has_shallow_match(inputs: Sequence[list[int]], outputs: Sequence[Sequence[int]]) -> bool:
    expected = tuple(tuple(row) for row in outputs)
    if tuple(tuple(values) for values in inputs) == expected:
        return True
    for operation in CONCRETE_OPERATIONS:
        if valid_output_vector((operation,), inputs) == expected:
            return True
    for program in DEPTH_TWO_PROGRAMS:
        if valid_output_vector(program, inputs) == expected:
            return True
    return False


def candidate_state_tables(inputs: Sequence[list[int]]) -> dict[BoundOperation, tuple[tuple[int, ...], ...]]:
    tables: dict[BoundOperation, tuple[tuple[int, ...], ...]] = {}
    for operation in CONCRETE_OPERATIONS:
        outputs = valid_output_vector((operation,), inputs)
        if outputs is None:
            raise RuntimeError("a one-step candidate was invalid on legal input")
        tables[operation] = outputs
    return tables


def fitting_suffixes(
    inputs: Sequence[list[int]], outputs: Sequence[Sequence[int]]
) -> dict[BoundOperation, tuple[Program, ...]]:
    """Enumerate every publicly live sibling and every visible-fitting suffix."""

    expected = tuple(tuple(row) for row in outputs)
    states = candidate_state_tables(inputs)
    result: dict[BoundOperation, tuple[Program, ...]] = {}
    for operation in CONCRETE_OPERATIONS:
        suffix_matches: list[Program] = []
        state_rows = states[operation]
        for suffix in DEPTH_TWO_PROGRAMS:
            matched = True
            for state, target in zip(state_rows, expected, strict=True):
                candidate = apply_pipeline(list(state), suffix)
                if candidate is INVALID or tuple(candidate) != target:
                    matched = False
                    break
            if matched:
                suffix_matches.append(suffix)
        if suffix_matches:
            result[operation] = tuple(suffix_matches)
    return result


def _draw_unique_inputs(
    *,
    seed: int,
    data: dict[str, Any],
    count: int,
    forbidden: set[tuple[int, ...]],
    target: Program | None = None,
) -> list[list[int]]:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    limit = int(data["max_input_draws_per_partition"])
    for _ in range(limit):
        if len(rows) == count:
            break
        values = _draw_input(rng, data)
        key = tuple(values)
        if key in forbidden:
            continue
        if target is not None and apply_pipeline(values, target) is INVALID:
            continue
        forbidden.add(key)
        rows.append(values)
    if len(rows) != count:
        raise RuntimeError("input draw limit exhausted")
    return rows


def _stratum_name(live_count: int, data: dict[str, Any]) -> str | None:
    for name, bounds in data["live_count_strata"].items():
        if int(bounds[0]) <= live_count <= int(bounds[1]):
            return str(name)
    return None


def _visible_candidate(
    *,
    fingerprint: str,
    program: Program,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    data = config["data"]
    base = int(config["seeds"]["data"])
    for attempt in range(int(data["max_visible_panels_per_program"])):
        seed = _stable_seed(base, "visible", fingerprint, canonical_program(program), str(attempt))
        visible_inputs = _draw_unique_inputs(
            seed=seed,
            data=data,
            count=int(data["visible_examples"]),
            forbidden=set(),
            target=program,
        )
        target_outputs = valid_output_vector(program, visible_inputs)
        if target_outputs is None or _visible_has_shallow_match(visible_inputs, target_outputs):
            continue
        if len(set(target_outputs)) != len(target_outputs):
            # Makes the token-preserving target derangement total and prevents
            # value-level fixed pairs without consulting any model outcome.
            continue
        state_tables = candidate_state_tables(visible_inputs)
        if len(set(state_tables.values())) != len(CONCRETE_OPERATIONS):
            continue
        live = fitting_suffixes(visible_inputs, target_outputs)
        stratum = _stratum_name(len(live), data)
        if stratum is None:
            continue
        return {
            "common_fingerprint": fingerprint,
            "target_pipeline": program,
            "visible_inputs": visible_inputs,
            "visible_outputs": target_outputs,
            "live_suffixes": live,
            "stratum": stratum,
            "visible_attempt": attempt,
            "state_tables": state_tables,
        }
    return None


def _required_stratum_counts(config: dict[str, Any]) -> dict[str, int]:
    data = config["data"]
    total = sum(
        int(data[field])
        for field in ("mechanics_tasks", "qualification_tasks", "confirmation_tasks")
    )
    per_block = {
        str(name): int(count)
        for name, count in data["tasks_per_stratum_per_block"].items()
    }
    block_size = int(data["tasks_per_balance_block"])
    if total % block_size or sum(per_block.values()) != block_size:
        raise ValueError("live-count block geometry is inconsistent")
    if set(per_block) != set(data["live_count_strata"]):
        raise ValueError("live-count strata and per-block quotas disagree")
    return {
        name: total // block_size * per_block[name]
        for name in data["live_count_strata"]
    }


def _collect_candidates(
    config: dict[str, Any],
    groups: dict[str, list[Program]],
    excluded_fingerprints: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    required = _required_stratum_counts(config)
    candidates = {name: [] for name in required}
    used_suffixes: set[Program] = set()
    used_triples: set[Program] = set()
    order = sorted(set(groups) - excluded_fingerprints)
    random.Random(int(config["seeds"]["data"])).shuffle(order)
    order_index = {fingerprint: index for index, fingerprint in enumerate(order)}
    used_fingerprints: set[str] = set()
    scanned_pairs: set[tuple[str, str]] = set()
    construction_order = [str(value) for value in config["data"]["stratum_construction_order"]]
    if set(construction_order) != set(required):
        raise ValueError("stratum construction order is incomplete")
    for desired in construction_order:
        low, high = (int(value) for value in config["data"]["live_count_strata"][desired])
        prioritized = sorted(
            order,
            key=lambda fingerprint: (
                not low <= len({program[0] for program in groups[fingerprint]}) <= high,
                order_index[fingerprint],
            ),
        )
        for fingerprint in prioritized:
            if len(candidates[desired]) >= required[desired]:
                break
            if fingerprint in used_fingerprints:
                continue
            accepted: dict[str, Any] | None = None
            for program in sorted(groups[fingerprint], key=alias_program):
                suffix = tuple(program[1:])
                if program in used_triples or suffix in used_suffixes:
                    continue
                scanned_pairs.add((fingerprint, canonical_program(program)))
                candidate = _visible_candidate(
                    fingerprint=fingerprint,
                    program=program,
                    config=config,
                )
                if candidate is not None and candidate["stratum"] == desired:
                    accepted = candidate
                    break
            if accepted is None:
                continue
            candidates[desired].append(accepted)
            target = accepted["target_pipeline"]
            used_fingerprints.add(fingerprint)
            used_triples.add(target)
            used_suffixes.add(tuple(target[1:]))
    missing = {
        name: required[name] - len(candidates[name])
        for name in required
        if len(candidates[name]) < required[name]
    }
    if missing:
        raise RuntimeError(f"exact task pool exhausted before quotas: {missing}")
    return candidates, {
        "eligible_function_fingerprints": len(groups),
        "excluded_prior_fingerprints": len(set(groups) & excluded_fingerprints),
        "fingerprints_scanned": len({value[0] for value in scanned_pairs}),
        "fingerprint_program_pairs_scanned": len(scanned_pairs),
        "required_strata": required,
        "selected_unique_suffixes": len(used_suffixes),
        "selected_unique_triples": len(used_triples),
    }


def _stratum_schedule(
    split: str, count: int, config: dict[str, Any]
) -> list[tuple[str, str]]:
    data = config["data"]
    block_size = int(data["tasks_per_balance_block"])
    if count % block_size:
        raise ValueError(f"{split} is not a complete balance-block multiple")
    schedule: list[tuple[str, str]] = []
    for block in range(count // block_size):
        values: list[tuple[str, str]] = []
        for name in data["live_count_strata"]:
            quota = int(data["tasks_per_stratum_per_block"][name])
            if quota % 2:
                raise ValueError("each stratum quota must balance A/B orientation")
            values.extend((name, "A") for _ in range(quota // 2))
            values.extend((name, "B") for _ in range(quota // 2))
        random.Random(
            _stable_seed(int(config["seeds"]["data"]), "strata", split, str(block))
        ).shuffle(values)
        schedule.extend(values)
    return schedule


def _materialize_task(
    *,
    task_id: str,
    candidate: dict[str, Any],
    viability_live_alias: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    data = config["data"]
    target: Program = candidate["target_pipeline"]
    visible_inputs: list[list[int]] = candidate["visible_inputs"]
    visible_outputs: tuple[tuple[int, ...], ...] = candidate["visible_outputs"]
    forbidden = {tuple(values) for values in visible_inputs}
    base = int(config["seeds"]["data"])
    hidden_inputs = _draw_unique_inputs(
        seed=_stable_seed(base, "hidden", task_id),
        data=data,
        count=int(data["hidden_examples"]),
        forbidden=forbidden,
        target=target,
    )
    probe_inputs = _draw_unique_inputs(
        seed=_stable_seed(base, "probe", task_id),
        data=data,
        count=int(data["unlabeled_probe_inputs"]),
        forbidden=forbidden,
        target=target,
    )
    hidden_outputs = valid_output_vector(target, hidden_inputs)
    if hidden_outputs is None:
        raise RuntimeError("validity-only hidden resampling failed")
    live_suffixes: dict[BoundOperation, tuple[Program, ...]] = candidate["live_suffixes"]
    return {
        "task_id": task_id,
        "depth": 3,
        "stratum": candidate["stratum"],
        "viability_live_alias": viability_live_alias,
        "common_fingerprint": candidate["common_fingerprint"],
        "target_pipeline": [operation_record(operation) for operation in target],
        "visible": [
            {"input": values, "output": list(output)}
            for values, output in zip(visible_inputs, visible_outputs, strict=True)
        ],
        "hidden": [
            {"input": values, "output": list(output)}
            for values, output in zip(hidden_inputs, hidden_outputs, strict=True)
        ],
        "unlabeled_probe_inputs": probe_inputs,
        "public_live": [
            {
                "operation": operation_record(operation),
                "canonical": canonical_operation(operation),
                "visible_fitting_suffix_count": len(live_suffixes[operation]),
                "first_fitting_suffix": [
                    operation_record(value) for value in live_suffixes[operation][0]
                ],
            }
            for operation in CONCRETE_OPERATIONS
            if operation in live_suffixes
        ],
        "visible_attempt": int(candidate["visible_attempt"]),
    }


def build_splits(
    config: dict[str, Any], *, excluded_fingerprints: set[str] | None = None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[list[int]]]:
    common = build_common_panel(config)
    groups, shallow, _representatives = enumerate_exact_depth_three_functions(common)
    queues, pool_receipt = _collect_candidates(
        config, groups, excluded_fingerprints or set()
    )
    offsets = Counter()
    split_specs = (
        ("mechanics", int(config["data"]["mechanics_tasks"])),
        ("qualification", int(config["data"]["qualification_tasks"])),
        ("confirmation", int(config["data"]["confirmation_tasks"])),
    )
    splits: dict[str, list[dict[str, Any]]] = {}
    task_namespace = str(config["identity"]["task_namespace"])
    for split, count in split_specs:
        rows: list[dict[str, Any]] = []
        for index, (stratum, live_alias) in enumerate(
            _stratum_schedule(split, count, config)
        ):
            candidate = queues[stratum][offsets[stratum]]
            offsets[stratum] += 1
            rows.append(
                _materialize_task(
                    task_id=namespaced_task_id(task_namespace, split, index),
                    candidate=candidate,
                    viability_live_alias=live_alias,
                    config=config,
                )
            )
        splits[split] = rows
    validate_splits(splits, config, common_inputs=common)
    receipt = {
        **pool_receipt,
        "common_panel_sha256": hashlib.sha256(
            json.dumps(common, separators=(",", ":")).encode()
        ).hexdigest(),
        "shallow_function_fingerprints": len(shallow),
        "split_rows": {name: len(rows) for name, rows in splits.items()},
    }
    return splits, receipt, common


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "depth": task["depth"],
        "viability_live_alias": task["viability_live_alias"],
        "visible": task["visible"],
        "unlabeled_probe_inputs": task["unlabeled_probe_inputs"],
    }


def gold_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "common_fingerprint": task["common_fingerprint"],
        "target_pipeline": task["target_pipeline"],
        "hidden": task["hidden"],
    }


def audit_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "stratum": task["stratum"],
        "viability_live_alias": task["viability_live_alias"],
        "public_live": task["public_live"],
        "visible_attempt": task["visible_attempt"],
    }


def validate_splits(
    splits: dict[str, Sequence[dict[str, Any]]],
    config: dict[str, Any],
    *,
    common_inputs: Sequence[list[int]],
    exhaustive_live_task_ids: set[str] | None = None,
) -> None:
    data = config["data"]
    seen_ids: set[str] = set()
    seen_functions: set[str] = set()
    seen_triples: set[Program] = set()
    seen_suffixes: set[Program] = set()
    for split, tasks in splits.items():
        counts = Counter(task["stratum"] for task in tasks)
        blocks = len(tasks) // int(data["tasks_per_balance_block"])
        expected = {
            name: blocks * int(data["tasks_per_stratum_per_block"][name])
            for name in data["live_count_strata"]
        }
        if any(counts[name] != expected[name] for name in data["live_count_strata"]):
            raise ValueError(f"{split} live-count strata are unbalanced: {counts}")
        orientations = Counter(
            (task["stratum"], task["viability_live_alias"]) for task in tasks
        )
        if any(
            orientations[(name, "A")] != orientations[(name, "B")]
            for name in data["live_count_strata"]
        ):
            raise ValueError(f"{split} viability aliases are unbalanced")
        for task in tasks:
            task_id = task["task_id"]
            fingerprint = task["common_fingerprint"]
            target = tuple(operation_from_record(value) for value in task["target_pipeline"])
            suffix = tuple(target[1:])
            if task_id in seen_ids or fingerprint in seen_functions:
                raise ValueError("duplicate task ID or function fingerprint")
            if target in seen_triples or suffix in seen_suffixes:
                raise ValueError("duplicate target triple or suffix")
            seen_ids.add(task_id)
            seen_functions.add(fingerprint)
            seen_triples.add(target)
            seen_suffixes.add(suffix)
            common_outputs = valid_output_vector(target, common_inputs)
            if common_outputs is None or behavior_fingerprint(common_outputs) != fingerprint:
                raise ValueError("target disagrees with common-panel fingerprint")
            all_inputs = [
                *(row["input"] for row in task["visible"]),
                *(row["input"] for row in task["hidden"]),
                *task["unlabeled_probe_inputs"],
            ]
            if len({tuple(values) for values in all_inputs}) != len(all_inputs):
                raise ValueError("task input partitions overlap")
            for row in [*task["visible"], *task["hidden"]]:
                output = apply_pipeline(row["input"], target)
                if output is INVALID or output != row["output"]:
                    raise ValueError("stored target row disagrees with interpreter")
            for values in task["unlabeled_probe_inputs"]:
                if apply_pipeline(values, target) is INVALID:
                    raise ValueError("target invalid on selector probe")
            visible_inputs = [row["input"] for row in task["visible"]]
            visible_outputs = [row["output"] for row in task["visible"]]
            if _visible_has_shallow_match(visible_inputs, visible_outputs):
                raise ValueError("visible task has a shallow match")
            tables = candidate_state_tables(visible_inputs)
            if len(set(tables.values())) != 24:
                raise ValueError("candidate state tables are not distinct")
            stored = [row["canonical"] for row in task["public_live"]]
            if len(stored) != len(set(stored)):
                raise ValueError("stored public-live set contains duplicates")
            for live_row in task["public_live"]:
                operation = operation_from_record(live_row["operation"])
                witness = tuple(
                    operation_from_record(value)
                    for value in live_row["first_fitting_suffix"]
                )
                if len(witness) != 2:
                    raise ValueError("stored fitting suffix has wrong arity")
                for values, output in zip(
                    visible_inputs, visible_outputs, strict=True
                ):
                    state = apply_pipeline(values, (operation, *witness))
                    if state is INVALID or state != output:
                        raise ValueError("stored public-live witness does not fit")
            if _stratum_name(len(stored), data) != task["stratum"]:
                raise ValueError("live count disagrees with stored stratum")
            if exhaustive_live_task_ids and task_id in exhaustive_live_task_ids:
                live = fitting_suffixes(visible_inputs, visible_outputs)
                expected_live = [
                    canonical_operation(operation)
                    for operation in CONCRETE_OPERATIONS
                    if operation in live
                ]
                if stored != expected_live:
                    raise ValueError(
                        "stored public-live set disagrees with independent enumeration"
                    )


def count_live_operations(tasks: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(
        row["canonical"] for task in tasks for row in task["public_live"]
    )

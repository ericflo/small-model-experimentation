"""Fresh, exactly verifiable state-transition substrate.

The model sees a randomly skinned finite world and must track a joint state
``(node, phase, checksum)``.  Each transition depends on the previous state, so
the only generic solution is serial state update.  Every item is generated
locally; this module never imports or reads ``benchmarks/``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


FAMILIES = ("phase_branch", "checksum_branch", "braided_branch")
TEMPLATES = ("ledger", "prose", "compact")
LETTERS = "ABCD"
SYLLABLES = (
    "ba", "be", "bi", "bo", "da", "de", "di", "do", "fa", "fe", "fi", "fo",
    "ga", "ge", "gi", "go", "ka", "ke", "ki", "ko", "la", "le", "li", "lo",
    "ma", "me", "mi", "mo", "na", "ne", "ni", "no", "pa", "pe", "pi", "po",
    "ra", "re", "ri", "ro", "sa", "se", "si", "so", "ta", "te", "ti", "to",
    "va", "ve", "vi", "vo", "za", "ze", "zi", "zo",
)


@dataclass(frozen=True)
class Node:
    left: int
    right: int
    toggle: int
    weight: int


@dataclass(frozen=True)
class State:
    node: int
    phase: int
    checksum: int


@dataclass(frozen=True)
class World:
    family: str
    nodes: tuple[Node, ...]
    labels: tuple[str, ...]
    checksum_modulus: int


def transition(world: World, state: State) -> State:
    record = world.nodes[state.node]
    if world.family == "phase_branch":
        go_left = state.phase == 0
    elif world.family == "checksum_branch":
        go_left = state.checksum % 2 == 0
    elif world.family == "braided_branch":
        go_left = state.phase == (state.checksum % 2)
    else:  # pragma: no cover - protected by construction and config validation
        raise ValueError(f"unknown family: {world.family}")
    destination = record.left if go_left else record.right
    target = world.nodes[destination]
    if world.family == "phase_branch":
        phase = state.phase ^ target.toggle
        checksum = state.checksum + target.weight
    elif world.family == "checksum_branch":
        phase = state.phase ^ target.toggle
        checksum = state.checksum + target.weight + state.phase
    else:
        phase = state.phase ^ target.toggle ^ (target.weight % 2)
        checksum = state.checksum + target.weight + state.phase
    return State(destination, phase % 2, checksum % world.checksum_modulus)


def unroll(world: World, initial: State, steps: int) -> tuple[State, ...]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    states = [initial]
    for _ in range(steps):
        states.append(transition(world, states[-1]))
    return tuple(states)


def state_at(trajectory: Sequence[State], step: int) -> State:
    if step < 0:
        raise ValueError("step must be non-negative")
    return trajectory[min(step, len(trajectory) - 1)]


def _labels(rng: random.Random, count: int) -> tuple[str, ...]:
    labels: set[str] = set()
    while len(labels) < count:
        labels.add(rng.choice(SYLLABLES) + rng.choice(SYLLABLES) + rng.choice(SYLLABLES))
    # Set iteration for strings depends on PYTHONHASHSEED. Canonicalize before
    # applying the seeded permutation so generation is stable across processes.
    result = sorted(labels)
    rng.shuffle(result)
    return tuple(result)


def _world(rng: random.Random, family: str, count: int, modulus: int) -> World:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    nodes: list[Node] = []
    for index in range(count):
        choices = [candidate for candidate in range(count) if candidate != index]
        left, right = rng.sample(choices, 2)
        nodes.append(Node(left, right, rng.randrange(2), rng.randrange(1, modulus)))
    return World(family, tuple(nodes), _labels(rng, count), modulus)


def family_rule(family: str, modulus: int) -> str:
    if family == "phase_branch":
        return (
            f"If phase is 0 follow LEFT, otherwise follow RIGHT. At the destination, XOR phase "
            f"with its toggle and add its weight to checksum modulo {modulus}."
        )
    if family == "checksum_branch":
        return (
            f"If checksum is even follow LEFT, otherwise follow RIGHT. At the destination, XOR phase "
            f"with its toggle and add its weight plus the old phase to checksum modulo {modulus}."
        )
    if family == "braided_branch":
        return (
            f"Follow LEFT exactly when phase equals checksum parity; otherwise follow RIGHT. At the "
            f"destination, XOR phase with its toggle and weight parity, then add its weight plus the "
            f"old phase to checksum modulo {modulus}."
        )
    raise ValueError(family)


def _render_table(world: World, order: Sequence[int], template: str) -> str:
    rows = []
    for index in order:
        node = world.nodes[index]
        if template == "ledger":
            rows.append(
                f"{world.labels[index]} | LEFT {world.labels[node.left]} | RIGHT {world.labels[node.right]} "
                f"| toggle {node.toggle} | weight {node.weight}"
            )
        elif template == "prose":
            rows.append(
                f"From {world.labels[index]}, left reaches {world.labels[node.left]} and right reaches "
                f"{world.labels[node.right]}; its toggle is {node.toggle} and its weight is {node.weight}."
            )
        elif template == "compact":
            rows.append(
                f"{world.labels[index]}: L={world.labels[node.left]}, R={world.labels[node.right]}, "
                f"T={node.toggle}, W={node.weight}"
            )
        else:
            raise ValueError(template)
    return "\n".join(rows)


def _choice_values(
    rng: random.Random,
    world: World,
    query_kind: str,
    target: State,
    num_choices: int,
    forced_values: Iterable[int] = (),
) -> tuple[list[int], int]:
    true_value = target.node if query_kind == "node" else target.checksum
    universe = range(len(world.nodes)) if query_kind == "node" else range(world.checksum_modulus)
    values = {true_value, *map(int, forced_values)}
    candidates = [value for value in universe if value not in values]
    rng.shuffle(candidates)
    while len(values) < num_choices:
        if not candidates:
            raise ValueError("choice universe is too small")
        values.add(candidates.pop())
    chosen = list(values)
    rng.shuffle(chosen)
    return chosen, chosen.index(true_value)


def _render_prompt(
    world: World,
    initial: State,
    depth: int,
    query_kind: str,
    choices: Sequence[int],
    order: Sequence[int],
    template: str,
    state_token: str,
    state_slots: int,
) -> str:
    display = [world.labels[value] if query_kind == "node" else str(value) for value in choices]
    query = "which node are you at" if query_kind == "node" else "what is the checksum"
    choice_lines = "\n".join(f"{LETTERS[i]}) {value}" for i, value in enumerate(display))
    workspace = " ".join([state_token] * state_slots)
    return (
        "Track the finite-state process exactly.\n"
        f"Rule: {family_rule(world.family, world.checksum_modulus)}\n"
        "World table:\n"
        f"{_render_table(world, order, template)}\n"
        f"Initial state: node={world.labels[initial.node]}, phase={initial.phase}, checksum={initial.checksum}.\n"
        f"Requested transition count: {depth}.\n"
        "Internal state workspace (the question deliberately comes later):\n"
        f"{workspace}\n"
        f"Query: After exactly {depth} transitions, {query}?\n"
        f"Choices:\n{choice_lines}\n"
        "Answer with only the choice letter.\nAnswer:"
    )


def _canonical_world(world: World) -> dict[str, Any]:
    return {
        "family": world.family,
        "checksum_modulus": world.checksum_modulus,
        "labels": list(world.labels),
        "nodes": [dataclasses.asdict(node) for node in world.nodes],
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def generate_example(
    *,
    seed: int,
    split: str,
    family: str,
    template: str,
    depth: int,
    node_count: int = 16,
    checksum_modulus: int = 8,
    num_choices: int = 4,
    state_token: str = "<|fim_pad|>",
    state_slots: int = 8,
    max_attempts: int = 500,
    pair_id: str | None = None,
    world: World | None = None,
    initial: State | None = None,
    query_kind: str | None = None,
    forced_choices: Iterable[int] = (),
    fixed_choices: Sequence[int] | None = None,
    table_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    if family not in FAMILIES or template not in TEMPLATES:
        raise ValueError("unknown family or template")
    if query_kind not in {None, "node", "checksum"}:
        raise ValueError("query_kind must be node or checksum")
    if depth < 1:
        raise ValueError("depth must be positive")
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        candidate_world = world or _world(rng, family, node_count, checksum_modulus)
        candidate_initial = initial or State(
            rng.randrange(node_count), rng.randrange(2), rng.randrange(checksum_modulus)
        )
        trajectory = unroll(candidate_world, candidate_initial, depth)
        # Repeated joint states permit a shorter equivalent computation and are rejected.
        if len(set(trajectory)) != len(trajectory):
            if world is not None or initial is not None:
                raise ValueError("provided counterfactual trajectory is not minimum-depth")
            continue
        selected_query = query_kind or rng.choice(("node", "checksum"))
        query_values = [
            state.node if selected_query == "node" else state.checksum
            for state in trajectory
        ]
        # A unique joint state is not sufficient: the requested terminal field
        # could still have appeared earlier, making the answer compatible with
        # a shallower computation. Reject that shortcut explicitly.
        if query_values[-1] in query_values[:-1]:
            if world is not None or initial is not None:
                raise ValueError("provided trajectory repeats its terminal queried value")
            continue
        true_value = (
            trajectory[-1].node
            if selected_query == "node"
            else trajectory[-1].checksum
        )
        if fixed_choices is None:
            choices, correct = _choice_values(
                rng,
                candidate_world,
                selected_query,
                trajectory[-1],
                num_choices,
                forced_choices,
            )
        else:
            choices = list(map(int, fixed_choices))
            if len(choices) != num_choices or len(set(choices)) != num_choices:
                raise ValueError("fixed choices must contain the exact number of unique values")
            if true_value not in choices:
                raise ValueError("fixed choices omit the exact answer")
            correct = choices.index(true_value)
        order = list(table_order) if table_order is not None else list(range(node_count))
        if sorted(order) != list(range(node_count)):
            raise ValueError("table_order must be a permutation of every node")
        if table_order is None:
            rng.shuffle(order)
        prompt = _render_prompt(
            candidate_world,
            candidate_initial,
            depth,
            selected_query,
            choices,
            order,
            template,
            state_token,
            state_slots,
        )
        structural_core = {
            "family": candidate_world.family,
            "checksum_modulus": candidate_world.checksum_modulus,
            "nodes": [dataclasses.asdict(node) for node in candidate_world.nodes],
            "initial": dataclasses.asdict(candidate_initial),
            "depth": depth,
        }
        core = {
            "world": _canonical_world(candidate_world),
            "initial": dataclasses.asdict(candidate_initial),
            "depth": depth,
            "query_kind": selected_query,
            "choices": choices,
        }
        fingerprint = _fingerprint(core)
        return {
            "id": f"{split}-{seed}-{fingerprint[:12]}",
            "split": split,
            "family": family,
            "template": template,
            "depth": depth,
            "world": core["world"],
            "table_order": order,
            "initial": core["initial"],
            "trajectory": [dataclasses.asdict(state) for state in trajectory],
            "query_kind": selected_query,
            "choices": choices,
            "correct_choice": correct,
            "answer_letter": LETTERS[correct],
            "prompt": prompt,
            "fingerprint": fingerprint,
            "structural_fingerprint": _fingerprint(structural_core),
            "pair_id": pair_id,
            "generation_attempt": attempt,
        }
    raise RuntimeError(f"failed to generate a minimum-depth item after {max_attempts} attempts")


def deserialize_world(payload: dict[str, Any]) -> World:
    return World(
        family=payload["family"],
        nodes=tuple(Node(**record) for record in payload["nodes"]),
        labels=tuple(payload["labels"]),
        checksum_modulus=int(payload["checksum_modulus"]),
    )


def verify_example(row: dict[str, Any], state_token: str, state_slots: int) -> None:
    if row.get("query_kind") not in {"node", "checksum"}:
        raise AssertionError("query kind is not a registered target field")
    world = deserialize_world(row["world"])
    initial = State(**row["initial"])
    trajectory = unroll(world, initial, int(row["depth"]))
    stored = tuple(State(**state) for state in row["trajectory"])
    if trajectory != stored:
        raise AssertionError("stored trajectory does not match exact execution")
    if len(set(trajectory)) != len(trajectory):
        raise AssertionError("trajectory repeats a joint state and is not minimum-depth")
    query_values = [
        state.node if row["query_kind"] == "node" else state.checksum
        for state in trajectory
    ]
    if query_values[-1] in query_values[:-1]:
        raise AssertionError("terminal queried value occurs at an earlier depth")
    target = trajectory[-1].node if row["query_kind"] == "node" else trajectory[-1].checksum
    if row["choices"][row["correct_choice"]] != target:
        raise AssertionError("answer choice does not equal exact terminal state")
    if row["answer_letter"] != LETTERS[row["correct_choice"]]:
        raise AssertionError("answer letter mismatch")
    if row["prompt"].count(state_token) != state_slots:
        raise AssertionError("prompt does not contain the exact workspace-slot count")
    if row["prompt"].index(state_token) > row["prompt"].index("Query:"):
        raise AssertionError("query appears before the causal state bottleneck")


def generate_counterfactual_pair(
    *,
    seed: int,
    split: str,
    family: str,
    template: str,
    depth: int,
    node_count: int,
    checksum_modulus: int,
    num_choices: int,
    state_token: str,
    state_slots: int,
    max_attempts: int,
    query_kind: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if query_kind not in {None, "node", "checksum"}:
        raise ValueError("query_kind must be node or checksum")
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        world = _world(rng, family, node_count, checksum_modulus)
        initial_node = rng.randrange(node_count)
        first_initial = State(initial_node, rng.randrange(2), rng.randrange(checksum_modulus))
        second_checksum = rng.randrange(checksum_modulus - 1)
        if second_checksum >= first_initial.checksum:
            second_checksum += 1
        second_initial = State(initial_node, first_initial.phase ^ 1, second_checksum)
        first_path = unroll(world, first_initial, depth)
        second_path = unroll(world, second_initial, depth)
        if len(set(first_path)) != len(first_path) or len(set(second_path)) != len(second_path):
            continue
        selected_query = query_kind or rng.choice(("node", "checksum"))
        first_query_path = [
            state.node if selected_query == "node" else state.checksum
            for state in first_path
        ]
        second_query_path = [
            state.node if selected_query == "node" else state.checksum
            for state in second_path
        ]
        if (
            first_query_path[-1] in first_query_path[:-1]
            or second_query_path[-1] in second_query_path[:-1]
        ):
            continue
        first_value = (
            first_path[-1].node
            if selected_query == "node"
            else first_path[-1].checksum
        )
        second_value = (
            second_path[-1].node
            if selected_query == "node"
            else second_path[-1].checksum
        )
        if first_value == second_value:
            continue
        pair_id = f"cf-{seed}-{attempt}"
        universe = (
            list(range(node_count))
            if selected_query == "node"
            else list(range(checksum_modulus))
        )
        shared_choices = [first_value, second_value]
        candidates = [value for value in universe if value not in shared_choices]
        rng.shuffle(candidates)
        while len(shared_choices) < num_choices:
            shared_choices.append(candidates.pop())
        rng.shuffle(shared_choices)
        shared_order = list(range(node_count))
        rng.shuffle(shared_order)
        first = generate_example(
            seed=seed * 1000 + attempt * 2,
            split=split,
            family=family,
            template=template,
            depth=depth,
            node_count=node_count,
            checksum_modulus=checksum_modulus,
            num_choices=num_choices,
            state_token=state_token,
            state_slots=state_slots,
            max_attempts=max_attempts,
            pair_id=pair_id,
            world=world,
            initial=first_initial,
            query_kind=selected_query,
            fixed_choices=shared_choices,
            table_order=shared_order,
        )
        second = generate_example(
            seed=seed * 1000 + attempt * 2 + 1,
            split=split,
            family=family,
            template=template,
            depth=depth,
            node_count=node_count,
            checksum_modulus=checksum_modulus,
            num_choices=num_choices,
            state_token=state_token,
            state_slots=state_slots,
            max_attempts=max_attempts,
            pair_id=pair_id,
            world=world,
            initial=second_initial,
            query_kind=selected_query,
            fixed_choices=shared_choices,
            table_order=shared_order,
        )
        return first, second
    raise RuntimeError("failed to generate a usable counterfactual pair")


def trajectory_targets(row: dict[str, Any], k: int) -> dict[str, list[int]]:
    trajectory = [State(**state) for state in row["trajectory"]]
    selected = [state_at(trajectory, step) for step in range(1, k + 1)]
    row_position = {
        int(node_index): position
        for position, node_index in enumerate(row["table_order"])
    }
    return {
        # Canonical node IDs are hidden and randomly skinned. Table position is
        # the visible, task-relative identity a shared decoder can learn.
        "node": [row_position[state.node] for state in selected],
        "phase": [state.phase for state in selected],
        "checksum": [state.checksum for state in selected],
    }

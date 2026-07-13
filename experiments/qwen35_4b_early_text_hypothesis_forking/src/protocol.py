"""Strict prompt, parse, execution, and visible-only selection contracts.

Model-produced Python is treated only as a small serialized DSL.  It is parsed
with :mod:`ast`, converted to a canonical pair of bound operations, and then
executed by :func:`task_data.apply_pipeline`; model text is never executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from task_data import OPERATIONS, apply_pipeline, canonical_operation
from task_data import canonical_program as canonical_pipeline


BoundOperation = tuple[str, int | None]
BoundOperationInput = BoundOperation | Sequence[Any] | Mapping[str, Any]

PUBLIC_TASK_KEYS = {"task_id", "depth", "visible", "unlabeled_probe_inputs"}
SELECTOR_POLICY_VERSION = "canonical-visible-probe-cluster-v1"
RESULT_RE = re.compile(r"(?im)^\s*RESULT\s*:\s*(\[[^\r\n]*\])\s*$")
FENCED_PROGRAM_RE = re.compile(
    r"\A```(?:python)?[ \t]*\r?\n(?P<source>.*?)\r?\n```[ \t]*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


def stable_index(task_id: str, size: int, *, salt: str) -> int:
    if size <= 0:
        raise ValueError("size must be positive")
    digest = hashlib.blake2b(f"{task_id}\0{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


def operation_text(name: str) -> str:
    descriptions = {
        "reverse": "reverse the list",
        "sort_asc": "sort ascending",
        "sort_desc": "sort descending",
        "abs_all": "replace every value by its absolute value",
        "square": "square every value",
        "negate": "negate every value",
        "running_sum": "replace the list by its running sums",
        "adjacent_diff": "replace it by adjacent differences next-minus-current",
        "add_k": "add k to every value",
        "mul_k": "multiply every value by k",
        "take_k": "keep the first k values",
        "rotate_k": "rotate left by k positions",
    }
    if name not in OPERATIONS or name not in descriptions:
        raise ValueError(f"unknown operation {name!r}")
    return descriptions[name]


def _parameter_literal(parameter: int) -> str:
    if not isinstance(parameter, int) or isinstance(parameter, bool):
        raise ValueError("operation parameter must be an integer literal")
    return str(parameter)


def _coerce_bound_operation(operation: BoundOperationInput) -> BoundOperation:
    if isinstance(operation, Mapping):
        if set(operation) != {"name", "parameter"}:
            raise ValueError("bound-operation mapping requires exact name/parameter schema")
        name = operation["name"]
        parameter = operation["parameter"]
    elif (
        isinstance(operation, Sequence)
        and not isinstance(operation, (str, bytes))
        and len(operation) == 2
    ):
        name, parameter = operation
    else:
        raise ValueError("bound operation must be a (name, parameter) pair")
    if not isinstance(name, str) or name not in OPERATIONS:
        raise ValueError("unknown operation")
    if parameter is not None and (
        not isinstance(parameter, int) or isinstance(parameter, bool)
    ):
        raise ValueError("operation parameter must be an integer or None")
    if parameter not in OPERATIONS[name][1]:
        raise ValueError(f"invalid parameter for {name}: {parameter!r}")
    return name, parameter


def helper_call(operation: BoundOperationInput, *, argument: str = "xs") -> str:
    """Return the canonical helper call for one validated bound operation."""

    name, parameter = _coerce_bound_operation(operation)
    if not argument.isidentifier():
        raise ValueError("helper argument must be an identifier")
    if parameter is None:
        return f"{name}({argument})"
    return f"{name}({argument}, {_parameter_literal(parameter)})"


def program_source(pipeline: Sequence[BoundOperationInput]) -> str:
    """Serialize exactly two bound operations to the frozen natural-Python ABI."""

    if len(pipeline) != 2:
        raise ValueError("a program must contain exactly two operations")
    first, second = (_coerce_bound_operation(operation) for operation in pipeline)
    return (
        "def transform(xs):\n"
        f"    xs = {helper_call(first)}\n"
        f"    xs = {helper_call(second)}\n"
        "    return xs"
    )


def helper_menu() -> str:
    """Describe the already-defined helper ABI and every allowed bound literal."""

    rows: list[str] = []
    for name, (_function, parameters) in OPERATIONS.items():
        if parameters == (None,):
            signature = f"{name}(xs)"
        else:
            literals = ", ".join(_parameter_literal(value) for value in parameters)
            signature = f"{name}(xs, k), where k is exactly one of: {literals}"
        rows.append(f"- {signature}: {operation_text(name)}")
    return "\n".join(rows)


def _validate_integer_list(value: Any, *, field: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{field} must be a list of integers")


def _validate_public_task(task: dict[str, Any]) -> None:
    if not isinstance(task, dict) or set(task) != PUBLIC_TASK_KEYS:
        raise ValueError("public task requires exact public-task schema")
    if not isinstance(task["task_id"], str) or not task["task_id"]:
        raise ValueError("task_id must be a nonempty string")
    if task["depth"] != 2:
        raise ValueError("public task depth must be exactly two")
    if not isinstance(task["visible"], list) or not task["visible"]:
        raise ValueError("visible must be a nonempty list")
    for index, row in enumerate(task["visible"]):
        if not isinstance(row, dict) or set(row) != {"input", "output"}:
            raise ValueError("visible rows require exact input/output schema")
        _validate_integer_list(row["input"], field=f"visible[{index}].input")
        _validate_integer_list(row["output"], field=f"visible[{index}].output")
    if not isinstance(task["unlabeled_probe_inputs"], list):
        raise ValueError("unlabeled_probe_inputs must be a list")
    for index, values in enumerate(task["unlabeled_probe_inputs"]):
        _validate_integer_list(values, field=f"unlabeled_probe_inputs[{index}]")


def task_prompt(task: dict[str, Any]) -> str:
    """Build the public user prompt; hypothesis text is deliberately separate."""

    _validate_public_task(task)
    examples = "\n".join(
        f"transform({row['input']!r}) = {row['output']!r}" for row in task["visible"]
    )
    return (
        "Infer the hidden program of exactly two list operations. The helper "
        "functions below are already defined; do not define or call anything else.\n\n"
        "Allowed helpers and bound integer literals:\n"
        f"{helper_menu()}\n\n"
        f"Visible examples:\n{examples}\n\n"
        "Reason privately. Return only one raw Python function (or exactly one "
        "Python fenced code block) with this exact shape:\n"
        "def transform(xs):\n"
        "    xs = allowed_helper(xs[, allowed_bound_integer_literal])\n"
        "    xs = allowed_helper(xs[, allowed_bound_integer_literal])\n"
        "    return xs\n"
        "Imports, attributes, control flow, arithmetic expressions, helper "
        "definitions, extra statements, and alternate candidates are forbidden."
    )


def candidate_injection(operation: BoundOperationInput) -> str:
    """Return hypothesis text intended for insertion inside an open ``<think>``.

    This is a continuation fragment, not part of :func:`task_prompt` and not a
    chat message.  It intentionally describes the bound operation as provisional.
    """

    bound = _coerce_bound_operation(operation)
    return (
        "\nHypothesis fork — provisional; test it against every example and revise "
        "if contradicted.\n"
        f"Concrete first operation: {canonical_operation(bound)}\n"
    )


def mechanics_prompt(values: list[int], *, candidate: BoundOperationInput) -> str:
    """Ask for the computed consequence of one supplied *bound* operation."""

    _validate_integer_list(values, field="mechanics input")
    operation = _coerce_bound_operation(candidate)
    return (
        "Apply exactly this supplied bound operation once:\n"
        f"xs = {helper_call(operation)}\n"
        f"Operation meaning: {operation_text(operation[0])}.\n"
        f"Input xs: {values!r}\n"
        "Reason privately, then give exactly one final line: "
        "RESULT: [comma-separated integers]"
    )


def _extract_program_source(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty_program")
    stripped = text.strip()
    # These are runner terminal tokens, not part of the model-facing Python ABI.
    for marker in ("<|endoftext|>", "<|im_end|>"):
        if stripped.endswith(marker):
            stripped = stripped[: -len(marker)].rstrip()
    if "```" not in stripped:
        return stripped
    match = FENCED_PROGRAM_RE.fullmatch(stripped)
    if match is None or "```" in match.group("source"):
        raise ValueError("fenced_program_shape")
    source = match.group("source").strip()
    if not source:
        raise ValueError("empty_program")
    return source


def _integer_literal(node: ast.expr) -> int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("parameter_not_integer_literal")
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    raise ValueError("parameter_not_integer_literal")


def _parse_helper_call(node: ast.expr) -> BoundOperation:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("assignment_not_allowed_helper_call")
    if node.keywords:
        raise ValueError("helper_keywords_forbidden")
    name = node.func.id
    if name not in OPERATIONS:
        raise ValueError("unknown_operation")
    if not node.args or not isinstance(node.args[0], ast.Name) or node.args[0].id != "xs":
        raise ValueError("helper_first_argument_must_be_xs")
    if OPERATIONS[name][1] == (None,):
        if len(node.args) != 1:
            raise ValueError("invalid_operation_arity")
        parameter: int | None = None
    else:
        if len(node.args) != 2:
            raise ValueError("invalid_operation_arity")
        parameter = _integer_literal(node.args[1])
    return _coerce_bound_operation((name, parameter))


def _parse_assignment(node: ast.stmt) -> BoundOperation:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        raise ValueError("expected_xs_assignment")
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != "xs":
        raise ValueError("assignment_target_must_be_xs")
    return _parse_helper_call(node.value)


def _parse_function(module: ast.Module) -> list[BoundOperation]:
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError("exactly_one_transform_function_required")
    function = module.body[0]
    if function.name != "transform":
        raise ValueError("function_must_be_named_transform")
    arguments = function.args
    if (
        arguments.posonlyargs
        or len(arguments.args) != 1
        or arguments.args[0].arg != "xs"
        or arguments.args[0].annotation is not None
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kw_defaults
        or arguments.kwarg is not None
        or arguments.defaults
    ):
        raise ValueError("transform_signature_must_be_exact")
    if function.decorator_list or function.returns is not None or function.type_comment:
        raise ValueError("transform_signature_must_be_exact")
    if len(function.body) != 3:
        raise ValueError("transform_body_must_have_exactly_three_statements")
    pipeline = [
        _parse_assignment(function.body[0]),
        _parse_assignment(function.body[1]),
    ]
    returned = function.body[2]
    if (
        not isinstance(returned, ast.Return)
        or not isinstance(returned.value, ast.Name)
        or returned.value.id != "xs"
    ):
        raise ValueError("transform_must_return_xs")
    return pipeline


def parse_program(text: str) -> dict[str, Any]:
    """Parse the strict Python ABI into canonical bound operations.

    No AST from model output is compiled or executed.  Downstream evaluation
    receives only the validated operation names and bound integer literals.
    """

    if not isinstance(text, str):
        return {
            "parsed": False,
            "error": "program_not_string",
            "pipeline": None,
            "canonical": None,
            "canonical_source": None,
        }
    try:
        # Generation may contain arbitrary reasoning, including provisional code,
        # inside ``<think>``.  Once a close marker exists, only the answer after
        # the final close marker is part of the parser ABI.
        answer = text.rsplit("</think>", 1)[1] if "</think>" in text else text
        source = _extract_program_source(answer)
        module = ast.parse(source, mode="exec")
        pipeline = _parse_function(module)
    except (SyntaxError, ValueError, OverflowError) as error:
        message = str(error) or error.__class__.__name__
        return {
            "parsed": False,
            "error": message,
            "pipeline": None,
            "canonical": None,
            "canonical_source": None,
        }
    canonical = canonical_pipeline(pipeline)
    return {
        "parsed": True,
        "error": None,
        "pipeline": pipeline,
        "canonical": canonical,
        "canonical_source": program_source(pipeline),
    }


def parse_result(text: str) -> dict[str, Any]:
    matches = list(RESULT_RE.finditer(text))
    if len(matches) != 1:
        return {"parsed": False, "error": "result_line_count", "result": None}
    try:
        value = ast.literal_eval(matches[0].group(1))
    except (SyntaxError, ValueError):
        return {"parsed": False, "error": "invalid_result_literal", "result": None}
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        return {"parsed": False, "error": "result_not_integer_list", "result": None}
    return {"parsed": True, "error": None, "result": value}


def score_candidate(task: dict[str, Any], text: str) -> dict[str, Any]:
    """Parse and score via the canonical DSL executor, never model Python."""

    _validate_public_task(task)
    parsed = parse_program(text)
    if not parsed["parsed"]:
        return {**parsed, "visible_pass": False, "probe_vector": None}
    pipeline = parsed["pipeline"]
    try:
        visible_pass = all(
            apply_pipeline(row["input"], pipeline) == row["output"]
            for row in task["visible"]
        )
        probe_vector = [
            apply_pipeline(values, pipeline) for values in task["unlabeled_probe_inputs"]
        ]
    except ValueError as error:
        return {
            **parsed,
            "visible_pass": False,
            "probe_vector": None,
            "execution_error": str(error),
        }
    return {**parsed, "visible_pass": visible_pass, "probe_vector": probe_vector}


def _row_id(row: Mapping[str, Any]) -> str | None:
    key = "candidate_id" if "candidate_id" in row else "id" if "id" in row else None
    if key is None or not isinstance(row[key], str) or not row[key]:
        return None
    return row[key]


def _abstention(
    *, reason: str, scored: list[dict[str, Any]], invalid_candidates: int
) -> dict[str, Any]:
    return {
        "selector_policy": SELECTOR_POLICY_VERSION,
        "abstained": True,
        "abstain_reason": reason,
        "selected": None,
        "selected_candidate_id": None,
        "selected_cluster_size": 0,
        "eligible": 0,
        "eligible_rows": 0,
        "eligible_unique_programs": 0,
        "invalid_candidates": invalid_candidates,
        "scored": scored,
    }


def select_visible(task: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select using only visible labels and frozen unlabeled-probe behavior.

    Canonical programs are deduplicated *before* behavior-cluster support is
    counted, so duplicated samples cannot manufacture consensus.  All ties are
    broken by stable hashes of public task IDs, behavior vectors, canonical
    programs, and preserved candidate row IDs.
    """

    _validate_public_task(task)
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")

    identifiers = [
        _row_id(row) if isinstance(row, Mapping) else None for row in candidates
    ]
    duplicate_ids = {
        identifier
        for identifier, count in Counter(
            identifier for identifier in identifiers if identifier is not None
        ).items()
        if count > 1
    }
    scored: list[dict[str, Any]] = []
    invalid_candidates = 0
    for row, identifier in zip(candidates, identifiers, strict=True):
        if not isinstance(row, Mapping):
            scored.append(
                {
                    "raw_candidate": row,
                    "parsed": False,
                    "error": "candidate_not_mapping",
                    "pipeline": None,
                    "canonical": None,
                    "canonical_source": None,
                    "visible_pass": False,
                    "probe_vector": None,
                }
            )
            invalid_candidates += 1
            continue
        preserved = dict(row)
        if identifier is None:
            value = {
                "parsed": False,
                "error": "missing_candidate_id",
                "pipeline": None,
                "canonical": None,
                "canonical_source": None,
                "visible_pass": False,
                "probe_vector": None,
            }
            invalid_candidates += 1
        elif identifier in duplicate_ids:
            value = {
                "parsed": False,
                "error": "duplicate_candidate_id",
                "pipeline": None,
                "canonical": None,
                "canonical_source": None,
                "visible_pass": False,
                "probe_vector": None,
            }
            invalid_candidates += 1
        elif not isinstance(row.get("text"), str):
            value = {
                "parsed": False,
                "error": "candidate_text_not_string",
                "pipeline": None,
                "canonical": None,
                "canonical_source": None,
                "visible_pass": False,
                "probe_vector": None,
            }
            invalid_candidates += 1
        else:
            value = score_candidate(task, row["text"])
            if not value["parsed"]:
                invalid_candidates += 1
        preserved.update(value)
        scored.append(preserved)

    parsed_rows = [row for row in scored if row["parsed"]]
    if not parsed_rows:
        return _abstention(
            reason="no_valid_candidate",
            scored=scored,
            invalid_candidates=invalid_candidates,
        )
    eligible_rows = [row for row in parsed_rows if row["visible_pass"]]
    if not eligible_rows:
        result = _abstention(
            reason="no_visible_passer",
            scored=scored,
            invalid_candidates=invalid_candidates,
        )
        result["eligible_rows"] = 0
        return result

    # Pick one stable, ID-preserving representative per canonical program.
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        by_canonical[row["canonical"]].append(row)
    representatives: list[dict[str, Any]] = []
    for canonical, rows in by_canonical.items():
        representative = min(
            rows,
            key=lambda row: hashlib.sha256(
                f"{task['task_id']}\0row\0{_row_id(row)}\0{canonical}".encode()
            ).hexdigest(),
        )
        representatives.append(representative)

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        key = json.dumps(row["probe_vector"], separators=(",", ":"))
        clusters[key].append(row)
    cluster_key = min(
        clusters,
        key=lambda key: (
            -len(clusters[key]),
            hashlib.sha256(
                f"{task['task_id']}\0behavior-cluster\0{key}".encode()
            ).hexdigest(),
        ),
    )
    selected = min(
        clusters[cluster_key],
        key=lambda row: hashlib.sha256(
            f"{task['task_id']}\0canonical-program\0{row['canonical']}".encode()
        ).hexdigest(),
    )
    return {
        "selector_policy": SELECTOR_POLICY_VERSION,
        "abstained": False,
        "abstain_reason": None,
        "selected": selected,
        "selected_candidate_id": _row_id(selected),
        "selected_cluster_size": len(clusters[cluster_key]),
        "eligible": len(representatives),
        "eligible_rows": len(eligible_rows),
        "eligible_unique_programs": len(representatives),
        "invalid_candidates": invalid_candidates,
        "scored": scored,
    }

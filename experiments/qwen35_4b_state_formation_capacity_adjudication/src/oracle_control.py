"""Exact, standard-library oracle-analysis control for the 48 setup rows.

The learned positive control and this oracle analysis answer different
mechanical questions.  The learned control checks whether the registered
state path can overfit.  This module checks that the row-level analyzer maps
an exact prediction table to the registered task/depth counts and terminal
joint accuracy without relying on model code or mutable global state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from .substrate import generate_example, trajectory_targets, verify_example


SCHEMA_VERSION = 1
STATUS = "ORACLE_ANALYSIS_PASS"
CONTROL_ROWS = 48
CONTROL_SEED = 73991
CONTROL_ROWS_SHA256 = "581dadcb7bba053d94a849e42e0490127c7e0de199311d053e7585adcc78ef41"
ORACLE_TABLE_SHA256 = "c42a546ca67992dea7e17309c29ccdd4663cd250803e5d488094c2d7ad108f92"
ORACLE_ANALYZER_OUTPUT_SHA256 = "493bc4cb6a592564d2f1ad8ab6187d209348011acb0c4f0d9a64b5430b658a9c"
CONTROL_DEPTH_COUNTS = {"2": 16, "3": 16, "4": 16}
CONTROL_DEPTHS = (2, 3, 4)
CONTROL_FAMILIES = ("phase_branch", "checksum_branch")
CONTROL_TEMPLATES = ("ledger", "prose")
CONTROL_QUERY_KINDS = ("node", "checksum")
EXAMPLES_PER_CELL = 2
NODE_COUNT = 16
CHECKSUM_MODULUS = 8
NUM_CHOICES = 4
MAX_GENERATION_ATTEMPTS = 500
STATE_TOKEN = "<|fim_pad|>"
STATE_SLOTS = 8
ORACLE_THRESHOLD = 0.99
RESULT_SPLITS = frozenset(
    {
        "train",
        "validation",
        "depth_extrapolation",
        "joint_holdout",
        "contrast_validation",
        "contrast_depth",
        "contrast_joint",
    }
)
CONTROL_ROW_FIELDS = frozenset(
    {
        "id",
        "split",
        "family",
        "template",
        "depth",
        "world",
        "table_order",
        "initial",
        "trajectory",
        "query_kind",
        "choices",
        "correct_choice",
        "answer_letter",
        "prompt",
        "fingerprint",
        "structural_fingerprint",
        "generation_attempt",
    }
)
HEAD_WIDTHS = {"node": NODE_COUNT, "phase": 2, "checksum": CHECKSUM_MODULUS}
RECORD_FIELDS = frozenset(
    {
        "id",
        "depth",
        "family",
        "template",
        "query_kind",
        "state",
        "objective_loss",
        "state_loss",
        "fixed_point_loss",
    }
)
STATE_FIELDS = frozenset(
    {
        "batch_size",
        "steps",
        "terminal",
        "trajectory",
        "histograms",
        "predictions",
        "targets",
    }
)

TABLE_FIELDS = frozenset(
    {
        "id",
        "depth",
        "node_target",
        "phase_target",
        "checksum_target",
        "node_prediction",
        "phase_prediction",
        "checksum_prediction",
        "node_final_correct",
        "phase_final_correct",
        "checksum_final_correct",
        "joint_final_correct",
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "rows",
        "unique_tasks",
        "depth_counts",
        "terminal_joint_correct",
        "terminal_joint_accuracy",
        "threshold",
        "control_seed",
        "canonical_control_rows_sha256",
        "canonical_table_sha256",
        "analyzer_output_sha256",
        "receipt_identity_sha256",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OracleControlError(RuntimeError):
    """The oracle table or its receipt violates the frozen control contract."""


def _fail(message: str) -> None:
    raise OracleControlError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(f"{label} must be a string-keyed mapping")
    return value


def _exact_int(value: Any, label: str, *, low: int | None = None) -> int:
    if type(value) is not int or (low is not None and value < low):
        _fail(f"{label} must be an exact integer")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OracleControlError("oracle payload is not finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _registered_generation_config(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Fail closed unless generation uses the frozen setup-control geometry."""

    root = _mapping(config, "experiment config")
    training = _mapping(root.get("training"), "training config")
    control = _mapping(training.get("positive_control"), "positive-control config")
    substrate = _mapping(root.get("substrate"), "substrate config")
    architecture = _mapping(root.get("architecture"), "architecture config")

    exact_scalars = (
        (control, "seed", CONTROL_SEED),
        (control, "rows", CONTROL_ROWS),
        (control, "examples_per_cell", EXAMPLES_PER_CELL),
        (substrate, "node_count", NODE_COUNT),
        (substrate, "checksum_modulus", CHECKSUM_MODULUS),
        (substrate, "num_choices", NUM_CHOICES),
        (substrate, "max_generation_attempts", MAX_GENERATION_ATTEMPTS),
        (architecture, "state_token", STATE_TOKEN),
        (architecture, "state_slots", STATE_SLOTS),
    )
    for section, field, expected in exact_scalars:
        observed = section.get(field)
        if type(expected) is int:
            if type(observed) is not int or observed != expected:
                _fail(f"registered control generation requires {field}={expected!r}")
        elif type(observed) is not str or observed != expected:
            _fail(f"registered control generation requires {field}={expected!r}")

    ordered_fields = (
        (control, "depths", list(CONTROL_DEPTHS)),
        (substrate, "train_families", list(CONTROL_FAMILIES)),
        (substrate, "train_templates", list(CONTROL_TEMPLATES)),
    )
    for section, field, expected in ordered_fields:
        observed = section.get(field)
        if type(observed) is not list or observed != expected:
            _fail(
                f"registered control generation requires exact ordered {field}={expected!r}"
            )
    return control, substrate, architecture


def _result_structural_fingerprints(manifest: Mapping[str, Any]) -> set[str]:
    """Collect every result fingerprint while rejecting lossy manifest shapes."""

    root = _mapping(manifest, "result data manifest")
    files = _mapping(root.get("files"), "result data manifest files")
    if set(files) != RESULT_SPLITS:
        _fail("result data manifest must contain the exact seven registered splits")
    fingerprints: set[str] = set()
    for split in sorted(RESULT_SPLITS):
        raw_metadata = files[split]
        metadata = _mapping(raw_metadata, f"result data manifest file {split}")
        rows = metadata.get("rows")
        if type(rows) is not int or rows < 0:
            _fail(f"result data manifest file {split} has a malformed row count")
        values = metadata.get("structural_fingerprints")
        if type(values) is not list or len(values) != rows:
            _fail(
                f"result data manifest file {split} has a malformed fingerprint index"
            )
        if values != sorted(values):
            _fail(f"result data manifest file {split} fingerprint index is not canonical")
        for index, value in enumerate(values):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                _fail(
                    f"result data manifest file {split} fingerprint {index} is malformed"
                )
            if value in fingerprints:
                _fail("result data manifest repeats a structural fingerprint")
            fingerprints.add(value)
    return fingerprints


def _canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        try:
            encoded = json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise OracleControlError(
                "generated control rows are not finite canonical JSON"
            ) from exc
        digest.update(encoded + b"\n")
    return digest.hexdigest()


def generate_control_rows(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate and verify the exact seed-73991 setup-control corpus.

    The returned receipt is deliberately schema-compatible with the historical
    ``gpu_runner._positive_control_rows`` receipt so callers can replace that
    model-bearing import without changing persisted positive-control evidence.
    """

    control, substrate, architecture = _registered_generation_config(config)
    result_fingerprints = _result_structural_fingerprints(manifest)
    seed = int(control["seed"])
    examples_per_cell = int(control["examples_per_cell"])
    depths = tuple(map(int, control["depths"]))
    families = tuple(map(str, substrate["train_families"]))
    templates = tuple(map(str, substrate["train_templates"]))
    node_count = int(substrate["node_count"])
    checksum_modulus = int(substrate["checksum_modulus"])
    num_choices = int(substrate["num_choices"])
    max_attempts = int(substrate["max_generation_attempts"])
    state_token = str(architecture["state_token"])
    state_slots = int(architecture["state_slots"])
    rows: list[dict[str, Any]] = []
    expected_order: list[tuple[int, str, str, str]] = []
    index = 0
    for _repeat in range(examples_per_cell):
        for depth in depths:
            for family in families:
                for template in templates:
                    for query_kind in CONTROL_QUERY_KINDS:
                        try:
                            row = generate_example(
                                seed=seed * 10_000_000 + index,
                                split="setup_positive_control",
                                family=family,
                                template=template,
                                depth=depth,
                                node_count=node_count,
                                checksum_modulus=checksum_modulus,
                                num_choices=num_choices,
                                state_token=state_token,
                                state_slots=state_slots,
                                max_attempts=max_attempts,
                                query_kind=query_kind,
                            )
                            verify_example(row, state_token, state_slots)
                        except Exception as exc:
                            raise OracleControlError(
                                f"generated control row {index} failed exact verification"
                            ) from exc
                        rows.append(row)
                        expected_order.append((depth, family, template, query_kind))
                        index += 1

    if len(rows) != CONTROL_ROWS or len(rows) != int(control["rows"]):
        _fail("positive-control factorial grid has the wrong row count")
    observed_order = [
        (
            row.get("depth"),
            row.get("family"),
            row.get("template"),
            row.get("query_kind"),
        )
        for row in rows
    ]
    if observed_order != expected_order:
        _fail("positive-control rows changed their exact registered order")
    if any(row.get("split") != "setup_positive_control" for row in rows):
        _fail("positive-control rows changed their registered split")

    ids = [row.get("id") for row in rows]
    if any(type(value) is not str or not value for value in ids) or len(set(ids)) != CONTROL_ROWS:
        _fail("positive-control rows have duplicate or malformed id values")
    for field in ("fingerprint", "structural_fingerprint"):
        values = [row.get(field) for row in rows]
        if (
            any(type(value) is not str or _SHA256.fullmatch(value) is None for value in values)
            or len(set(values)) != CONTROL_ROWS
        ):
            _fail(f"positive-control rows have duplicate or malformed {field} values")
    control_fingerprints = {str(row["structural_fingerprint"]) for row in rows}
    overlap = sorted(control_fingerprints & result_fingerprints)
    if overlap:
        _fail("positive-control rows overlap result data")

    grid = Counter(
        f"{row['family']}|{row['template']}|depth={row['depth']}|query={row['query_kind']}"
        for row in rows
    )
    expected_grid = {
        f"{family}|{template}|depth={depth}|query={query_kind}": examples_per_cell
        for depth in depths
        for family in families
        for template in templates
        for query_kind in CONTROL_QUERY_KINDS
    }
    if dict(grid) != expected_grid:
        _fail("positive-control grid is not the exact registered factorial")

    receipt = validate_control_rows(rows)
    if receipt["canonical_rows_sha256"] != CONTROL_ROWS_SHA256:
        _fail("positive-control canonical rows differ from the frozen seed-73991 corpus")
    if receipt["grid"] != dict(sorted(grid.items())):
        _fail("generated and consumed positive-control grids differ")
    return rows, receipt


def _expected_control_order() -> list[tuple[int, str, str, str]]:
    return [
        (depth, family, template, query_kind)
        for _repeat in range(EXAMPLES_PER_CELL)
        for depth in CONTROL_DEPTHS
        for family in CONTROL_FAMILIES
        for template in CONTROL_TEMPLATES
        for query_kind in CONTROL_QUERY_KINDS
    ]


def _validated_control_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the complete frozen corpus, not merely its target projection."""

    if type(rows) is not list or len(rows) != CONTROL_ROWS:
        _fail(f"oracle control requires exactly {CONTROL_ROWS} generated rows")
    expected_order = _expected_control_order()
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    fingerprints: set[str] = set()
    structural_fingerprints: set[str] = set()
    depth_counts: Counter[int] = Counter()
    grid: Counter[tuple[int, str, str, str]] = Counter()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"control row {index}")
        if set(row) != CONTROL_ROW_FIELDS:
            _fail(f"control row {index} has noncanonical schema")
        copied = copy.deepcopy(dict(row))
        try:
            verify_example(copied, STATE_TOKEN, STATE_SLOTS)
        except Exception as exc:
            raise OracleControlError(
                f"control row {index} failed full verify_example"
            ) from exc

        observed_cell = (
            copied.get("depth"),
            copied.get("family"),
            copied.get("template"),
            copied.get("query_kind"),
        )
        if observed_cell != expected_order[index]:
            _fail(f"control row {index} changed the exact seed/order grid")
        if copied.get("split") != "setup_positive_control":
            _fail(f"control row {index} has the wrong split")
        depth = _exact_int(copied.get("depth"), f"control row {index} depth")
        row_id = copied.get("id")
        if type(row_id) is not str or not row_id or row_id in identifiers:
            _fail("control row IDs must be nonempty and unique")
        identifiers.add(row_id)
        for field, seen in (
            ("fingerprint", fingerprints),
            ("structural_fingerprint", structural_fingerprints),
        ):
            value = copied.get(field)
            if type(value) is not str or _SHA256.fullmatch(value) is None or value in seen:
                _fail(f"control row {field} values must be exact unique SHA-256 digests")
            seen.add(value)
        depth_counts[depth] += 1
        grid[
            (
                depth,
                str(copied["family"]),
                str(copied["template"]),
                str(copied["query_kind"]),
            )
        ] += 1
        normalized.append(copied)

    observed_depths = {
        str(depth): count for depth, count in sorted(depth_counts.items())
    }
    if observed_depths != CONTROL_DEPTH_COUNTS:
        _fail("control rows have the wrong depth counts")
    expected_grid = {
        (depth, family, template, query_kind): EXAMPLES_PER_CELL
        for depth in CONTROL_DEPTHS
        for family in CONTROL_FAMILIES
        for template in CONTROL_TEMPLATES
        for query_kind in CONTROL_QUERY_KINDS
    }
    if dict(grid) != expected_grid:
        _fail("control rows do not form the exact frozen factorial grid")
    if _canonical_rows_sha256(normalized) != CONTROL_ROWS_SHA256:
        _fail("control rows differ from the frozen seed-73991 canonical corpus")
    return normalized


def _terminal_targets(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        depth = int(row["depth"])
        try:
            targets = trajectory_targets(dict(row), depth)
        except Exception as exc:
            raise OracleControlError(
                f"control row {index} trajectory target derivation failed"
            ) from exc
        if set(targets) != set(HEAD_WIDTHS):
            _fail(f"control row {index} trajectory targets changed schema")
        result.append(
            {
                "id": str(row["id"]),
                "depth": depth,
                "node_target": targets["node"][-1],
                "phase_target": targets["phase"][-1],
                "checksum_target": targets["checksum"][-1],
            }
        )
    return result


def _control_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _terminal_targets(_validated_control_rows(rows))


def validate_control_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Public exact-corpus consumer receipt used by non-model gate paths."""

    validated = _validated_control_rows(rows)
    grid = Counter(
        f"{row['family']}|{row['template']}|depth={row['depth']}|query={row['query_kind']}"
        for row in validated
    )
    return {
        "seed": CONTROL_SEED,
        "rows": CONTROL_ROWS,
        "grid": dict(sorted(grid.items())),
        "canonical_rows_sha256": CONTROL_ROWS_SHA256,
        "cross_result_structural_overlap": 0,
    }


def build_oracle_prediction_table(
    control_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Construct the one deterministic oracle-perfect terminal prediction table."""

    targets = _control_targets(control_rows)
    table = [
        {
            **target,
            "node_prediction": target["node_target"],
            "phase_prediction": target["phase_target"],
            "checksum_prediction": target["checksum_target"],
            "node_final_correct": True,
            "phase_final_correct": True,
            "checksum_final_correct": True,
            "joint_final_correct": True,
        }
        for target in targets
    ]
    if _canonical_sha256({"rows": table}) != ORACLE_TABLE_SHA256:
        _fail("oracle table differs from the frozen canonical prediction table")
    return table


def _validate_oracle_prediction_table(
    targets: Sequence[Mapping[str, Any]],
    prediction_table: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if type(prediction_table) is not list or len(prediction_table) != CONTROL_ROWS:
        _fail(f"oracle prediction table must contain exactly {CONTROL_ROWS} rows")
    observed_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, (target, raw) in enumerate(zip(targets, prediction_table)):
        row = _mapping(raw, f"oracle prediction row {index}")
        if set(row) != TABLE_FIELDS:
            _fail(f"oracle prediction row {index} has noncanonical fields")
        row_id = row.get("id")
        if type(row_id) is not str or not row_id or row_id in observed_ids:
            _fail("oracle prediction IDs must be nonempty and unique")
        observed_ids.add(row_id)
        if row_id != target["id"]:
            _fail("oracle prediction IDs/order differ from the exact control rows")
        if type(row.get("depth")) is not int or row["depth"] != target["depth"]:
            _fail(f"oracle prediction row {index} has the wrong depth")

        component_correct: list[bool] = []
        for component in HEAD_WIDTHS:
            target_field = f"{component}_target"
            prediction_field = f"{component}_prediction"
            flag_field = f"{component}_final_correct"
            expected_target = target[target_field]
            if type(row.get(target_field)) is not int or row[target_field] != expected_target:
                _fail(f"oracle prediction row {index} changed {target_field}")
            prediction = row.get(prediction_field)
            if type(prediction) is not int:
                _fail(f"oracle prediction row {index} has a non-integer prediction")
            flag = row.get(flag_field)
            if type(flag) is not bool:
                _fail(f"oracle prediction row {index} has a non-boolean correctness flag")
            expected_flag = prediction == expected_target
            if flag is not expected_flag:
                _fail(f"oracle prediction row {index} has an inconsistent correctness flag")
            if not expected_flag:
                _fail(f"oracle prediction row {index} is not oracle-perfect")
            component_correct.append(expected_flag)
        joint = row.get("joint_final_correct")
        if type(joint) is not bool:
            _fail(f"oracle prediction row {index} has a non-boolean joint flag")
        if joint is not all(component_correct) or not joint:
            _fail(f"oracle prediction row {index} has an incorrect terminal joint")
        normalized.append(copy.deepcopy(dict(row)))

    digest = _canonical_sha256({"rows": normalized})
    if digest != ORACLE_TABLE_SHA256:
        _fail("oracle prediction table differs from its frozen canonical digest")
    return normalized


def _histogram(values: Sequence[int], width: int) -> list[int]:
    result = [0] * width
    for value in values:
        result[value] += 1
    return result


def _production_state(
    predictions: Mapping[str, Sequence[int]],
    targets: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    depth = len(targets["node"])
    correct = {
        head: [
            prediction == target
            for prediction, target in zip(predictions[head], targets[head])
        ]
        for head in HEAD_WIDTHS
    }
    joint = [
        all(correct[head][step] for head in HEAD_WIDTHS)
        for step in range(depth)
    ]
    terminal = {head: int(values[-1]) for head, values in correct.items()}
    terminal.update({"joint": int(joint[-1]), "rows": 1})
    trajectory = {head: sum(values) for head, values in correct.items()}
    trajectory.update({"joint": sum(joint), "steps": depth})
    return {
        "batch_size": 1,
        "steps": depth,
        "terminal": terminal,
        "trajectory": trajectory,
        "histograms": {
            head: {
                "prediction": _histogram(predictions[head], width),
                "target": _histogram(targets[head], width),
            }
            for head, width in HEAD_WIDTHS.items()
        },
        "predictions": {head: [list(predictions[head])] for head in HEAD_WIDTHS},
        "targets": {head: [list(targets[head])] for head in HEAD_WIDTHS},
    }


def _oracle_records_from_validated(
    control_rows: Sequence[Mapping[str, Any]],
    prediction_table: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    targets = _terminal_targets(control_rows)
    table = _validate_oracle_prediction_table(targets, prediction_table)
    records: list[dict[str, Any]] = []
    for row, table_row in zip(control_rows, table):
        depth = int(row["depth"])
        sequences = trajectory_targets(dict(row), depth)
        predictions = {head: list(values) for head, values in sequences.items()}
        for head in HEAD_WIDTHS:
            predictions[head][-1] = int(table_row[f"{head}_prediction"])
        records.append(
            {
                "id": str(row["id"]),
                "depth": depth,
                "family": str(row["family"]),
                "template": str(row["template"]),
                "query_kind": str(row["query_kind"]),
                "state": _production_state(predictions, sequences),
                "objective_loss": 0.0,
                "state_loss": 0.0,
                "fixed_point_loss": 0.0,
            }
        )
    return records


def build_oracle_positive_control_records(
    control_rows: Sequence[Mapping[str, Any]],
    prediction_table: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build oracle-perfect records with the exact live evaluator schema."""

    validated = _validated_control_rows(control_rows)
    table = (
        build_oracle_prediction_table(validated)
        if prediction_table is None
        else prediction_table
    )
    return _oracle_records_from_validated(validated, table)


def _exact_sequence(
    value: Any, *, length: int, width: int, label: str
) -> list[int]:
    if type(value) is not list or len(value) != length:
        _fail(f"{label} has the wrong sequence geometry")
    if any(type(item) is not int or not 0 <= item < width for item in value):
        _fail(f"{label} has an invalid class value")
    return list(value)


def _validate_positive_control_records(
    records: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _validated_control_rows(expected_rows)
    if type(records) is not list or len(records) != CONTROL_ROWS:
        _fail(f"positive-control analyzer requires exactly {CONTROL_ROWS} records")
    normalized: list[dict[str, Any]] = []
    for index, (raw_record, expected_row) in enumerate(zip(records, rows)):
        record = _mapping(raw_record, f"positive-control record {index}")
        if set(record) != RECORD_FIELDS:
            _fail(f"positive-control record {index} has noncanonical schema")
        for field in ("id", "depth", "family", "template", "query_kind"):
            if record.get(field) != expected_row[field] or type(
                record.get(field)
            ) is not type(expected_row[field]):
                _fail(f"positive-control record {index} changed expected {field}")
        depth = int(expected_row["depth"])
        state = _mapping(record.get("state"), f"positive-control record {index} state")
        if set(state) != STATE_FIELDS:
            _fail(f"positive-control record {index} state has noncanonical schema")
        if type(state.get("batch_size")) is not int or state["batch_size"] != 1:
            _fail(f"positive-control record {index} batch size changed")
        if type(state.get("steps")) is not int or state["steps"] != depth:
            _fail(f"positive-control record {index} step count changed")

        expected_targets = trajectory_targets(dict(expected_row), depth)
        targets_payload = _mapping(
            state.get("targets"), f"positive-control record {index} targets"
        )
        predictions_payload = _mapping(
            state.get("predictions"), f"positive-control record {index} predictions"
        )
        if set(targets_payload) != set(HEAD_WIDTHS) or set(
            predictions_payload
        ) != set(HEAD_WIDTHS):
            _fail(f"positive-control record {index} state heads changed")
        targets: dict[str, list[int]] = {}
        predictions: dict[str, list[int]] = {}
        for head, width in HEAD_WIDTHS.items():
            raw_targets = targets_payload[head]
            raw_predictions = predictions_payload[head]
            if type(raw_targets) is not list or len(raw_targets) != 1:
                _fail(f"positive-control record {index} {head} targets changed batch geometry")
            if type(raw_predictions) is not list or len(raw_predictions) != 1:
                _fail(f"positive-control record {index} {head} predictions changed batch geometry")
            targets[head] = _exact_sequence(
                raw_targets[0],
                length=depth,
                width=width,
                label=f"positive-control record {index} {head} targets",
            )
            predictions[head] = _exact_sequence(
                raw_predictions[0],
                length=depth,
                width=width,
                label=f"positive-control record {index} {head} predictions",
            )
            if targets[head] != expected_targets[head]:
                _fail(f"positive-control record {index} changed exact {head} targets")

        expected_state = _production_state(predictions, targets)
        terminal = _mapping(
            state.get("terminal"), f"positive-control record {index} terminal"
        )
        trajectory = _mapping(
            state.get("trajectory"), f"positive-control record {index} trajectory"
        )
        if set(terminal) != {*HEAD_WIDTHS, "joint", "rows"}:
            _fail(f"positive-control record {index} terminal schema changed")
        if set(trajectory) != {*HEAD_WIDTHS, "joint", "steps"}:
            _fail(f"positive-control record {index} trajectory schema changed")
        if any(type(value) is not int or value < 0 for value in terminal.values()):
            _fail(f"positive-control record {index} terminal counts are invalid")
        if any(type(value) is not int or value < 0 for value in trajectory.values()):
            _fail(f"positive-control record {index} trajectory counts are invalid")
        histograms = _mapping(
            state.get("histograms"), f"positive-control record {index} histograms"
        )
        if set(histograms) != set(HEAD_WIDTHS):
            _fail(f"positive-control record {index} histogram heads changed")
        for head, width in HEAD_WIDTHS.items():
            histogram = _mapping(
                histograms[head], f"positive-control record {index} {head} histogram"
            )
            if set(histogram) != {"prediction", "target"}:
                _fail(f"positive-control record {index} {head} histogram schema changed")
            for kind in ("prediction", "target"):
                values = histogram[kind]
                if (
                    type(values) is not list
                    or len(values) != width
                    or any(type(value) is not int or value < 0 for value in values)
                ):
                    _fail(
                        f"positive-control record {index} {head} {kind} histogram is invalid"
                    )
        for section, observed in (
            ("terminal", terminal),
            ("trajectory", trajectory),
            ("histograms", histograms),
        ):
            if observed != expected_state[section]:
                _fail(f"positive-control record {index} has inconsistent {section}")
        losses: dict[str, float] = {}
        for field in ("objective_loss", "state_loss", "fixed_point_loss"):
            value = record.get(field)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                _fail(f"positive-control record {index} {field} is nonfinite")
            if float(value) < 0.0:
                _fail(f"positive-control record {index} {field} is negative")
            losses[field] = float(value)
        normalized.append(
            {
                "id": str(record["id"]),
                "depth": depth,
                "family": str(record["family"]),
                "template": str(record["template"]),
                "query_kind": str(record["query_kind"]),
                "state": expected_state,
                **losses,
            }
        )
    return normalized


def _summarize_validated_records(
    records: Sequence[Mapping[str, Any]], *, field: str | None
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {"overall": list(records)}
    if field is not None:
        if field not in {"depth", "family", "template", "query_kind"}:
            _fail(f"positive-control analyzer cannot group by {field!r}")
        grouped = {}
        for record in records:
            grouped.setdefault(str(record[field]), []).append(record)
    summaries: dict[str, dict[str, Any]] = {}
    for key, selected in sorted(grouped.items()):
        terminal_rows = len(selected)
        trajectory_steps = sum(record["state"]["trajectory"]["steps"] for record in selected)
        terminal_counts = {
            name: sum(record["state"]["terminal"][name] for record in selected)
            for name in (*HEAD_WIDTHS, "joint")
        }
        trajectory_counts = {
            name: sum(record["state"]["trajectory"][name] for record in selected)
            for name in (*HEAD_WIDTHS, "joint")
        }
        histograms = {
            head: {
                kind: [
                    sum(record["state"]["histograms"][head][kind][index] for record in selected)
                    for index in range(width)
                ]
                for kind in ("prediction", "target")
            }
            for head, width in HEAD_WIDTHS.items()
        }
        summaries[key] = {
            "rows": terminal_rows,
            "trajectory_steps": trajectory_steps,
            "terminal_correct_counts": terminal_counts,
            "trajectory_correct_counts": trajectory_counts,
            **{
                f"{name}_final_accuracy": terminal_counts[name] / terminal_rows
                for name in terminal_counts
            },
            **{
                f"{name}_trajectory_accuracy": trajectory_counts[name] / trajectory_steps
                for name in trajectory_counts
            },
            "mean_objective_loss": sum(record["objective_loss"] for record in selected)
            / terminal_rows,
            "mean_state_loss": sum(record["state_loss"] for record in selected)
            / terminal_rows,
            "mean_fixed_point_loss": sum(record["fixed_point_loss"] for record in selected)
            / terminal_rows,
            "histograms": histograms,
        }
    return summaries["overall"] if field is None else summaries


def summarize_positive_control_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    field: str | None = None,
) -> dict[str, Any]:
    """Strict drop-in summarizer for live positive-control evaluation records."""

    validated = _validate_positive_control_records(records, expected_rows)
    return _summarize_validated_records(validated, field=field)


def analyze_positive_control_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return every registered production positive-control summary slice."""

    validated = _validate_positive_control_records(records, expected_rows)
    return {
        "overall": _summarize_validated_records(validated, field=None),
        "by_depth": _summarize_validated_records(validated, field="depth"),
        "by_family": _summarize_validated_records(validated, field="family"),
        "by_template": _summarize_validated_records(validated, field="template"),
        "by_query_kind": _summarize_validated_records(validated, field="query_kind"),
    }


def analyze_oracle_control(
    control_rows: Sequence[Mapping[str, Any]],
    prediction_table: Sequence[Mapping[str, Any]],
    *,
    threshold: float = ORACLE_THRESHOLD,
) -> dict[str, Any]:
    """Validate an oracle table and return the exact identity-bound pass receipt."""

    if type(threshold) not in (int, float) or isinstance(threshold, bool):
        _fail("oracle threshold must be numeric")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value != ORACLE_THRESHOLD:
        _fail(f"oracle threshold must remain exactly {ORACLE_THRESHOLD}")
    rows = _validated_control_rows(control_rows)
    targets = _terminal_targets(rows)
    normalized_table = _validate_oracle_prediction_table(targets, prediction_table)
    records = _oracle_records_from_validated(rows, normalized_table)
    analysis = analyze_positive_control_records(records, expected_rows=rows)
    overall = analysis["overall"]
    correct = overall["terminal_correct_counts"]["joint"]
    accuracy = overall["joint_final_accuracy"]
    if accuracy < threshold_value:
        _fail("oracle analysis is below its frozen accuracy threshold")
    if any(
        overall[f"{head}_{kind}_accuracy"] != 1.0
        for head in (*HEAD_WIDTHS, "joint")
        for kind in ("final", "trajectory")
    ):
        _fail("oracle production-record analysis is not perfect")
    depth_counts = {
        depth: summary["rows"] for depth, summary in analysis["by_depth"].items()
    }
    if depth_counts != CONTROL_DEPTH_COUNTS:
        _fail("oracle analyzer produced the wrong exact depth counts")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "rows": CONTROL_ROWS,
        "unique_tasks": len({record["id"] for record in records}),
        "depth_counts": depth_counts,
        "terminal_joint_correct": correct,
        "terminal_joint_accuracy": accuracy,
        "threshold": threshold_value,
        "control_seed": CONTROL_SEED,
        "canonical_control_rows_sha256": CONTROL_ROWS_SHA256,
        "canonical_table_sha256": _canonical_sha256({"rows": normalized_table}),
        "analyzer_output_sha256": _canonical_sha256(analysis),
    }
    if receipt["analyzer_output_sha256"] != ORACLE_ANALYZER_OUTPUT_SHA256:
        _fail("oracle shared-analyzer output differs from its frozen canonical digest")
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    return receipt


def produce_oracle_analysis_receipt(
    control_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Producer convenience: construct and analyze the exact perfect table."""

    table = build_oracle_prediction_table(control_rows)
    return analyze_oracle_control(control_rows, table)


def validate_oracle_analysis_receipt(
    control_rows: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the oracle table from exact rows and require one canonical receipt."""

    observed = _mapping(receipt, "oracle analysis receipt")
    if set(observed) != RECEIPT_FIELDS:
        _fail("oracle analysis receipt has noncanonical fields")
    claimed = observed.get("receipt_identity_sha256")
    if type(claimed) is not str or _SHA256.fullmatch(claimed) is None:
        _fail("oracle analysis receipt identity is malformed")
    payload = {
        key: value
        for key, value in observed.items()
        if key != "receipt_identity_sha256"
    }
    if claimed != _canonical_sha256(payload):
        _fail("oracle analysis receipt identity mismatch")
    expected = produce_oracle_analysis_receipt(control_rows)
    if dict(observed) != expected:
        _fail("oracle analysis receipt differs from exact recomputation")
    return copy.deepcopy(expected)


__all__ = [
    "OracleControlError",
    "analyze_positive_control_records",
    "analyze_oracle_control",
    "build_oracle_positive_control_records",
    "build_oracle_prediction_table",
    "generate_control_rows",
    "produce_oracle_analysis_receipt",
    "summarize_positive_control_records",
    "validate_control_rows",
    "validate_oracle_analysis_receipt",
]

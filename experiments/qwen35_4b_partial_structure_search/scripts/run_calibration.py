#!/usr/bin/env python3
"""Score the frozen calibration frontier with all model-based recognition controls."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import experiment_common as C  # noqa: E402
import families as F  # noqa: E402
import model_scoring as M  # noqa: E402
import vllm_runner as VR  # noqa: E402
from vllm_runner import EngineConfig, MODEL_ID, MODEL_REVISION, VLLMRunner  # noqa: E402


RECEIPT_SCHEMA_VERSION = 2
RECEIPT_STATUS = "complete"


class CacheValidationError(RuntimeError):
    """A cache exists but cannot be proved complete for the current inputs."""


@dataclasses.dataclass(frozen=True)
class OutputSpec:
    tag: str
    path: Path
    expected_ids: tuple[str, ...]


def _write_rows(tag: str, suffix: str, rows: list[dict[str, Any]]) -> None:
    C.write_jsonl(EXP / "runs" / f"calibration_{tag}{suffix}.jsonl", rows)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise CacheValidationError(f"required cache file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _file_fingerprint(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(path, root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _calibration_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the config namespaces that can affect calibration scoring."""
    missing = [name for name in ("model", "judge", "calibration") if name not in config]
    if missing:
        raise CacheValidationError(
            f"config is missing calibration-relevant namespaces: {missing}"
        )
    return _json_normalized(
        {
            "model": config["model"],
            "judge": config["judge"],
            "calibration": config["calibration"],
        }
    )


def _load_config_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CacheValidationError(f"required config file is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CacheValidationError(f"config is not valid YAML: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheValidationError(f"config must be a mapping: {path}")
    return value


def _config_projection_fingerprint(path: Path, root: Path) -> dict[str, Any]:
    projection = _calibration_config_projection(_load_config_mapping(path))
    return {
        "path": _relative_path(path, root),
        "projection": projection,
        "projection_sha256": _sha256_json(projection),
    }


def _full_config_context(path: Path, root: Path) -> dict[str, Any]:
    return {
        **_file_fingerprint(path, root),
        "validates_calibration_cache": False,
        "observed_at": "receipt_seal_or_upgrade",
        "note": (
            "Context only. Calibration validity is bound to the canonical "
            "model/judge/calibration projection, so unrelated search-only edits "
            "neither invalidate nor become attributed to model scoring."
        ),
    }


def _input_fingerprints(
    *,
    root: Path,
    config_path: Path,
    candidate_path: Path,
    scorer_path: Path | None = None,
    runner_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "config": _config_projection_fingerprint(config_path, root),
        "candidates": _file_fingerprint(candidate_path, root),
        "model_scorer": _file_fingerprint(
            scorer_path or Path(M.__file__).resolve(), root
        ),
        "vllm_runner": _file_fingerprint(
            runner_path or Path(VR.__file__).resolve(), root
        ),
    }


def _refresh_input_fingerprints(
    root: Path, fingerprints: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    refreshed: dict[str, dict[str, Any]] = {}
    for name, fingerprint in fingerprints.items():
        raw_path = Path(str(fingerprint.get("path", "")))
        path = raw_path if raw_path.is_absolute() else root / raw_path
        refreshed[name] = (
            _config_projection_fingerprint(path, root)
            if name == "config"
            else _file_fingerprint(path, root)
        )
    return refreshed


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CacheValidationError(f"receipt is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CacheValidationError(f"receipt is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheValidationError(f"receipt must be a JSON object: {path}")
    return value


def _strict_jsonl_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise CacheValidationError(f"required output is missing: {path}")
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CacheValidationError(
                    f"invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict) or not str(row.get("id", "")):
                raise CacheValidationError(
                    f"output {path} line {line_number} has no non-empty id"
                )
            ids.append(str(row["id"]))
    return ids


def _validated_output_fingerprints(
    specs: Sequence[OutputSpec], *, root: Path
) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for spec in specs:
        ids = _strict_jsonl_ids(spec.path)
        expected = list(spec.expected_ids)
        if ids != expected:
            mismatch = next(
                (
                    index
                    for index, (actual, wanted) in enumerate(zip(ids, expected))
                    if actual != wanted
                ),
                min(len(ids), len(expected)),
            )
            raise CacheValidationError(
                f"output {spec.tag!r} IDs/count do not match the frozen input: "
                f"actual={len(ids)}, expected={len(expected)}, first_mismatch={mismatch}"
            )
        fingerprints[spec.tag] = {
            **_file_fingerprint(spec.path, root),
            "row_count": len(ids),
            "ordered_ids_sha256": _sha256_json(ids),
        }
    return fingerprints


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish the completion sentinel only after bytes and directory are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parent_records(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for child in candidates:
        group = str(child["parent_group"])
        if group in seen:
            continue
        seen.add(group)
        prefix = list(child["candidate_prefix"][:-1])
        rows.append(
            {
                "id": group,
                "task_text": child["task_text"],
                "visible_examples": child["visible_examples"],
                "candidate_prefix": prefix,
                "remaining_steps": int(child["remaining_steps"]) + 1,
                "choices": list(F.TYPES),
                "task_id": child["task_id"],
                "prefix_len": len(prefix),
                "parent_group": group,
            }
        )
    return rows


def _shuffled_records(
    candidates: list[dict[str, Any]], max_tasks: int
) -> list[dict[str, Any]]:
    tasks = sorted({str(row["task_id"]) for row in candidates})[:max_tasks]
    if len(tasks) < 2:
        return []
    visible = {}
    for row in candidates:
        visible.setdefault(str(row["task_id"]), row["visible_examples"])
    donor = {task: tasks[(index + 1) % len(tasks)] for index, task in enumerate(tasks)}
    rows = []
    for row in candidates:
        task = str(row["task_id"])
        if task not in donor:
            continue
        changed = dict(row)
        changed["id"] = "shuffle:" + str(row["id"])
        changed["visible_examples"] = visible[donor[task]]
        changed["original_id"] = row["id"]
        changed["visible_donor_task"] = donor[task]
        rows.append(changed)
    return rows


def _output_specs(
    candidates: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    suffix: str,
    *,
    root: Path = EXP,
) -> tuple[OutputSpec, ...]:
    candidate_ids = tuple(str(row["id"]) for row in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise CacheValidationError("calibration candidate IDs are not unique")
    parent_ids = tuple(str(row["id"]) for row in _parent_records(candidates))
    shuffled_ids = tuple(
        str(row["id"])
        for row in _shuffled_records(
            candidates, int(cfg["calibration"]["shuffle_canary_tasks"])
        )
    )
    specs = [
        OutputSpec(
            "thinking",
            root / "runs" / f"calibration_thinking{suffix}.jsonl",
            candidate_ids,
        ),
        OutputSpec(
            "nothink",
            root / "runs" / f"calibration_nothink{suffix}.jsonl",
            candidate_ids,
        ),
        OutputSpec(
            "nextop",
            root / "runs" / f"calibration_nextop{suffix}.jsonl",
            parent_ids,
        ),
    ]
    if shuffled_ids:
        specs.append(
            OutputSpec(
                "task_shuffled_thinking",
                root / "runs" / f"calibration_task_shuffled{suffix}.jsonl",
                shuffled_ids,
            )
        )
    return tuple(specs)


def _expected_engine_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dataclasses.asdict(
        EngineConfig(
            max_model_len=int(cfg["judge"]["max_model_len"]),
            gpu_memory_utilization=float(cfg["judge"]["gpu_memory_utilization"]),
            max_num_seqs=int(cfg["judge"]["max_num_seqs"]),
            max_num_batched_tokens=8192,
        )
    )


def _scoring_plan(
    cfg: Mapping[str, Any], specs: Sequence[OutputSpec]
) -> dict[str, dict[str, Any]]:
    by_tag = {spec.tag: len(spec.expected_ids) for spec in specs}
    run_seed = int(cfg["judge"]["run_seed"])
    thinking_budget = int(cfg["judge"]["thinking_budget"])
    plan = {
        "thinking": {
            "method": "thinking_p_viable",
            "run_seed": run_seed,
            "thinking_budget": thinking_budget,
            "rows": by_tag["thinking"],
            "logical_requests": 2 * by_tag["thinking"],
        },
        "nothink": {
            "method": "no_think_p_viable",
            "run_seed": run_seed + 1,
            "thinking_budget": None,
            "rows": by_tag["nothink"],
            "logical_requests": by_tag["nothink"],
        },
        "nextop": {
            "method": "next_operation_likelihood",
            "run_seed": run_seed + 2,
            "thinking_budget": None,
            "rows": by_tag["nextop"],
            "logical_requests": by_tag["nextop"],
        },
    }
    if "task_shuffled_thinking" in by_tag:
        plan["task_shuffled_thinking"] = {
            "method": "thinking_p_viable",
            "run_seed": run_seed + 3,
            "thinking_budget": thinking_budget,
            "rows": by_tag["task_shuffled_thinking"],
            "logical_requests": 2 * by_tag["task_shuffled_thinking"],
        }
    return plan


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _validate_identity_and_engine(
    receipt: Mapping[str, Any], cfg: Mapping[str, Any]
) -> None:
    expected = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "vllm",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise CacheValidationError(
                f"receipt {key} mismatch: {receipt.get(key)!r} != {value!r}"
            )
    actual_engine = _json_normalized(receipt.get("engine_config"))
    expected_engine = _json_normalized(_expected_engine_config(cfg))
    if actual_engine != expected_engine:
        raise CacheValidationError(
            f"receipt engine_config mismatch: {actual_engine!r} != {expected_engine!r}"
        )


def _validate_legacy_summaries(
    receipt: Mapping[str, Any], plan: Mapping[str, Mapping[str, Any]]
) -> None:
    summaries = receipt.get("scoring_summaries")
    if not isinstance(summaries, Mapping):
        raise CacheValidationError("legacy receipt has no scoring_summaries object")
    if set(summaries) != set(plan):
        raise CacheValidationError(
            f"legacy scoring summary tags mismatch: {sorted(summaries)} != {sorted(plan)}"
        )
    for tag, expected in plan.items():
        summary = summaries.get(tag)
        if not isinstance(summary, Mapping):
            raise CacheValidationError(f"legacy summary {tag!r} is not an object")
        for key in ("method", "run_seed", "thinking_budget"):
            if summary.get(key) != expected[key]:
                raise CacheValidationError(
                    f"legacy summary {tag!r} {key} mismatch: "
                    f"{summary.get(key)!r} != {expected[key]!r}"
                )
        accounting = summary.get("accounting")
        if not isinstance(accounting, Mapping) or accounting.get("requests") != expected[
            "logical_requests"
        ]:
            raise CacheValidationError(
                f"legacy summary {tag!r} logical request count mismatch"
            )


def _seal_receipt(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    cfg: Mapping[str, Any],
    specs: Sequence[OutputSpec],
    input_fingerprints: Mapping[str, Mapping[str, Any]],
    legacy_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    current_inputs = _refresh_input_fingerprints(root, input_fingerprints)
    if _json_normalized(current_inputs) != _json_normalized(input_fingerprints):
        raise CacheValidationError(
            "config, candidates, scorer, or runner changed during calibration; "
            "refusing to publish a completion receipt"
        )
    outputs = _validated_output_fingerprints(specs, root=root)
    plan = _scoring_plan(cfg, specs)
    sealed = dict(receipt)
    sealed.update(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "completion_status": RECEIPT_STATUS,
            "input_fingerprints": _json_normalized(input_fingerprints),
            "full_config_context": _full_config_context(
                root / "configs" / "default.yaml", root
            ),
            "output_fingerprints": outputs,
            "expected_outputs": {
                spec.tag: {
                    "row_count": len(spec.expected_ids),
                    "ordered_ids_sha256": _sha256_json(list(spec.expected_ids)),
                }
                for spec in specs
            },
            "scoring_plan": plan,
        }
    )
    if legacy_receipt_sha256 is not None:
        sealed["legacy_upgrade"] = {
            "source_schema_version": 1,
            "legacy_receipt_sha256": legacy_receipt_sha256,
            "note": (
                "The legacy receipt predated code/output fingerprints. It was upgraded "
                "only after its identity, engine, candidate hash, scoring settings, "
                "request counts, and exact ordered output IDs validated."
            ),
        }
    return sealed


def _validate_v2_receipt(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    cfg: Mapping[str, Any],
    specs: Sequence[OutputSpec],
    input_fingerprints: Mapping[str, Mapping[str, Any]],
) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise CacheValidationError("receipt is not schema v2")
    if receipt.get("completion_status") != RECEIPT_STATUS:
        raise CacheValidationError("receipt does not carry the complete sentinel")
    _validate_identity_and_engine(receipt, cfg)
    if _json_normalized(receipt.get("input_fingerprints")) != _json_normalized(
        input_fingerprints
    ):
        raise CacheValidationError("receipt input fingerprints are stale")
    plan = _scoring_plan(cfg, specs)
    if _json_normalized(receipt.get("scoring_plan")) != _json_normalized(plan):
        raise CacheValidationError("receipt scoring plan is stale")
    expected_outputs = {
        spec.tag: {
            "row_count": len(spec.expected_ids),
            "ordered_ids_sha256": _sha256_json(list(spec.expected_ids)),
        }
        for spec in specs
    }
    if _json_normalized(receipt.get("expected_outputs")) != expected_outputs:
        raise CacheValidationError("receipt expected output IDs/counts are stale")
    actual_outputs = _validated_output_fingerprints(specs, root=root)
    if _json_normalized(receipt.get("output_fingerprints")) != _json_normalized(
        actual_outputs
    ):
        raise CacheValidationError("one or more calibration output fingerprints changed")
    if _json_normalized(_refresh_input_fingerprints(root, input_fingerprints)) != (
        _json_normalized(input_fingerprints)
    ):
        raise CacheValidationError("cache inputs changed during validation")


def _accept_or_upgrade_cache(
    receipt_path: Path,
    *,
    root: Path,
    cfg: Mapping[str, Any],
    specs: Sequence[OutputSpec],
    input_fingerprints: Mapping[str, Mapping[str, Any]],
    permit_legacy_upgrade: bool = True,
) -> str | None:
    if not receipt_path.exists():
        return None
    legacy_sha256 = _sha256_file(receipt_path)
    receipt = _strict_json(receipt_path)
    if receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION:
        _validate_v2_receipt(
            receipt,
            root=root,
            cfg=cfg,
            specs=specs,
            input_fingerprints=input_fingerprints,
        )
        return "valid"
    if receipt.get("schema_version") != 1:
        raise CacheValidationError(
            f"unsupported receipt schema: {receipt.get('schema_version')!r}"
        )
    if not permit_legacy_upgrade:
        raise CacheValidationError("legacy receipt upgrade was not permitted")
    _validate_identity_and_engine(receipt, cfg)
    candidate_sha = input_fingerprints["candidates"]["sha256"]
    if receipt.get("candidate_file_sha256") != candidate_sha:
        raise CacheValidationError("legacy receipt candidate hash is stale")
    plan = _scoring_plan(cfg, specs)
    _validate_legacy_summaries(receipt, plan)
    # Validate output bytes and exact ordered IDs before attributing current code
    # fingerprints to a legacy artifact.
    _validated_output_fingerprints(specs, root=root)
    upgraded = _seal_receipt(
        receipt,
        root=root,
        cfg=cfg,
        specs=specs,
        input_fingerprints=input_fingerprints,
        legacy_receipt_sha256=legacy_sha256,
    )
    _atomic_write_json(receipt_path, upgraded)
    _validate_v2_receipt(
        upgraded,
        root=root,
        cfg=cfg,
        specs=specs,
        input_fingerprints=input_fingerprints,
    )
    return "upgraded"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--recompute",
        action="store_true",
        help="invalidate any receipt and intentionally regenerate every model score",
    )
    cache_group.add_argument(
        "--upgrade-receipt",
        action="store_true",
        help="validate/upgrade a legacy receipt without loading the model",
    )
    args = parser.parse_args(argv)
    suffix = "_smoke" if args.smoke else ""
    cfg = C.load_config()
    config_path = EXP / "configs" / "default.yaml"
    candidate_path = EXP / "data" / f"calibration_candidates{suffix}.jsonl"
    receipt_path = EXP / "runs" / f"calibration_model_receipt{suffix}.json"
    candidates = C.load_jsonl(candidate_path)
    if not candidates:
        raise RuntimeError("build calibration candidates before model scoring")
    specs = _output_specs(candidates, cfg, suffix, root=EXP)
    input_fingerprints = _input_fingerprints(
        root=EXP,
        config_path=config_path,
        candidate_path=candidate_path,
    )
    if args.recompute:
        # The receipt is the only completion sentinel. Remove it before any
        # mutable work so a killed recomputation cannot expose old outputs as a
        # valid current cache.
        receipt_path.unlink(missing_ok=True)
    else:
        try:
            cache_state = _accept_or_upgrade_cache(
                receipt_path,
                root=EXP,
                cfg=cfg,
                specs=specs,
                input_fingerprints=input_fingerprints,
            )
        except CacheValidationError as exc:
            raise RuntimeError(
                f"calibration cache failed closed: {exc}. "
                "Inspect the artifacts or pass --recompute to regenerate them."
            ) from exc
        if cache_state is not None:
            print(f"[calibration-model] cached ({cache_state} receipt)", flush=True)
            return 0
        existing_outputs = [str(spec.path) for spec in specs if spec.path.exists()]
        if existing_outputs:
            raise RuntimeError(
                "calibration outputs exist without a completion receipt; refusing to "
                "guess whether a process is still running. Wait for it, use "
                "--upgrade-receipt after its legacy receipt appears, or pass --recompute. "
                f"Existing outputs: {existing_outputs}"
            )
        if args.upgrade_receipt:
            raise RuntimeError(
                f"no receipt exists to validate or upgrade: {receipt_path}"
            )

    engine_cfg = EngineConfig(**_expected_engine_config(cfg))
    started = time.perf_counter()
    summaries: dict[str, Any] = {}
    with VLLMRunner(engine_cfg) as runner:
        scorer = M.ModelScorer(runner)
        think, summaries["thinking"] = scorer.score_thinking_viability(
            candidates,
            thinking_budget=int(cfg["judge"]["thinking_budget"]),
            run_seed=int(cfg["judge"]["run_seed"]),
        )
        _write_rows("thinking", suffix, think)
        no_think, summaries["nothink"] = scorer.score_no_think_viability(
            candidates, run_seed=int(cfg["judge"]["run_seed"]) + 1
        )
        _write_rows("nothink", suffix, no_think)
        nextop, summaries["nextop"] = scorer.score_next_operation_likelihood(
            _parent_records(candidates), run_seed=int(cfg["judge"]["run_seed"]) + 2
        )
        _write_rows("nextop", suffix, nextop)
        canary_records = _shuffled_records(
            candidates, int(cfg["calibration"]["shuffle_canary_tasks"])
        )
        if canary_records:
            shuffled, summaries["task_shuffled_thinking"] = scorer.score_thinking_viability(
                canary_records,
                thinking_budget=int(cfg["judge"]["thinking_budget"]),
                run_seed=int(cfg["judge"]["run_seed"]) + 3,
            )
            _write_rows("task_shuffled", suffix, shuffled)
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "backend": "vllm",
            "engine_config": dataclasses.asdict(engine_cfg),
            "engine_args": runner.engine_args,
            "resolved_cudagraph": runner.resolved_cudagraph,
            "runtime": runner.runtime_metadata(),
            "scoring_summaries": summaries,
            "candidate_file_sha256": input_fingerprints["candidates"]["sha256"],
            "wall_seconds": time.perf_counter() - started,
        }
    try:
        sealed_receipt = _seal_receipt(
            receipt,
            root=EXP,
            cfg=cfg,
            specs=specs,
            input_fingerprints=input_fingerprints,
        )
    except CacheValidationError as exc:
        raise RuntimeError(f"refusing to publish calibration receipt: {exc}") from exc
    _atomic_write_json(receipt_path, sealed_receipt)
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "groups": len(_parent_records(candidates)),
                "wall_seconds": receipt["wall_seconds"],
                "thinking_forced_close_rate": sum(row["forced_close"] for row in think) / len(think),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

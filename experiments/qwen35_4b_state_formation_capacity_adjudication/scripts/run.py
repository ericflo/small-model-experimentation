#!/usr/bin/env python3
"""Run the phase-gated state-formation capacity adjudication."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    load_config,
    resolved_config_receipt,
    source_contract_sha256,
)
from src.data_pipeline import build_datasets  # noqa: E402
from src.design_boundary import (  # noqa: E402
    authorized_source_execution_snapshot,
    freeze_design,
)
from src.initialization import prepare_initialization_bundle  # noqa: E402
from src.attempt_receipts import locked_regular  # noqa: E402
from src.safe_io import publish_new_bytes  # noqa: E402
from src.substrate import generate_example, verify_example  # noqa: E402


LOADED_SOURCE_CONTRACT_SHA256 = source_contract_sha256(ROOT)
EXECUTION_GENERATION_LOCK = ROOT / "runs" / "run.lock"


def _absolute_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _lexical_path(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = os.fspath(value)
    pieces = raw.split("/")
    content_start = 1 if raw.startswith("/") else 0
    if (
        not raw
        or "\x00" in raw
        or "\\" in raw
        or any(part in {".", ".."} for part in pieces)
        or any(part == "" for part in pieces[content_start:])
    ):
        raise SystemExit(f"{label} must be a canonical lexical path")
    return _absolute_path(raw)


def _require_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if not os.path.lexists(current):
            continue
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise SystemExit(f"{label} uses a symlink alias: {path}")


def _same_lexical_path(value: str | os.PathLike[str], expected: Path, *, label: str) -> bool:
    actual = _lexical_path(value, label=label)
    canonical = _absolute_path(expected)
    _require_no_symlink_components(actual, label=label)
    return actual == canonical


def _write_json(path: Path, payload: object) -> None:
    """Install a setup receipt once, without following or overwriting aliases."""

    path = _absolute_path(path)
    _require_no_symlink_components(path, label="JSON output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(path.parent, label="JSON output parent")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    publish_new_bytes(REPO, path, encoded, mode=0o644)


ANALYSIS_PHASES = {
    "lora_joint",
    "lora_control",
    "stage_b_seal",
    "fullrank_joint",
    "fullrank_control",
}


def _validate_registered_cell(args: argparse.Namespace) -> None:
    if args.stage == "analyze":
        if args.phase not in ANALYSIS_PHASES:
            raise SystemExit("--phase must select a registered analysis phase")
    elif args.phase is not None:
        raise SystemExit("--phase is only valid for analyze")
    if args.stage == "evaluate-state":
        if args.eval_set == "contrast" and args.objective != "joint":
            raise SystemExit("contrast evaluation requires --objective joint")
    elif args.eval_set != "trigger":
        raise SystemExit("--eval-set is only valid for evaluate-state")


def _canonical_output(args: argparse.Namespace, config: dict) -> Path:
    _validate_registered_cell(args)
    analysis_outputs = {
        "lora_joint": ROOT / "analysis" / "lora_joint_trigger.json",
        "lora_control": ROOT / "analysis" / "lora_control.json",
        "stage_b_seal": ROOT / "analysis" / "stage_b_seal.json",
        "fullrank_joint": ROOT / "analysis" / "fullrank_joint.json",
        "fullrank_control": ROOT / "analysis" / "summary.json",
    }
    if args.stage == "analyze":
        return _absolute_path(analysis_outputs[args.phase])
    outputs = {
        "cpu-smoke": ROOT / "runs" / "cpu_smoke" / "receipt.json",
        "design-boundary": ROOT / config["paths"]["design_receipt"],
        "prepare-data": ROOT / config["paths"]["data_dir"],
        "prepare-init": ROOT / config["paths"]["large_artifacts_dir"]
        / f"initialization_seed{args.seed}.pt",
        "model-smoke": ROOT / "runs" / "setup"
        / f"g0_{args.capacity}_seed{args.seed}.json",
        "positive-control": ROOT / "runs" / "setup"
        / f"positive_control_{args.capacity}_seed{args.seed}.json",
        "train": ROOT / config["paths"]["large_artifacts_dir"]
        / f"{args.capacity}_{args.objective}_seed{args.seed}",
        "evaluate-state": ROOT / "runs"
        / f"{args.capacity}_{args.objective}_seed{args.seed}_{args.eval_set}",
    }
    return _absolute_path(outputs[args.stage])


def _require_canonical_inputs(args: argparse.Namespace, config: dict) -> None:
    large = _absolute_path(ROOT / config["paths"]["large_artifacts_dir"])
    if args.seed is not None and args.stage in {
        "model-smoke", "positive-control", "train"
    }:
        expected_initialization = large / f"initialization_seed{args.seed}.pt"
        if not _same_lexical_path(
            args.initialization_bundle,
            expected_initialization,
            label="--initialization-bundle",
        ):
            raise SystemExit(
                f"--initialization-bundle must be canonical: {expected_initialization}"
            )
    expected_g0 = ROOT / "runs" / "setup" / f"g0_{args.capacity}_seed{args.seed}.json"
    if args.stage in {"positive-control", "train"} and not _same_lexical_path(
        args.model_smoke_receipt,
        expected_g0,
        label="--model-smoke-receipt",
    ):
        raise SystemExit(f"--model-smoke-receipt must be canonical: {expected_g0}")
    expected_control = (
        ROOT / "runs" / "setup" / f"positive_control_{args.capacity}_seed{args.seed}.json"
    )
    if args.stage == "train" and not _same_lexical_path(
        args.positive_control_receipt,
        expected_control,
        label="--positive-control-receipt",
    ):
        raise SystemExit(
            f"--positive-control-receipt must be canonical: {expected_control}"
        )
    if args.stage == "evaluate-state":
        expected_checkpoint = (
            large
            / f"{args.capacity}_{args.objective}_seed{args.seed}"
            / f"checkpoint_{int(config['training']['train_steps']):06d}"
        )
        if not args.checkpoint or not _same_lexical_path(
            args.checkpoint,
            expected_checkpoint,
            label="--checkpoint",
        ):
            raise SystemExit(f"--checkpoint must be the canonical fixed final: {expected_checkpoint}")

    lora_miss = _absolute_path(ROOT / "analysis" / "lora_joint_trigger.json")
    stage_b = _absolute_path(ROOT / "analysis" / "stage_b_seal.json")
    fullrank_joint = _absolute_path(ROOT / "analysis" / "fullrank_joint.json")
    allowed_authorizations: set[Path]
    if args.stage in {"cpu-smoke", "design-boundary", "prepare-data", "prepare-init"}:
        allowed_authorizations = set()
    elif args.stage == "analyze":
        allowed_authorizations = {
            "lora_joint": set(),
            "lora_control": {lora_miss},
            "stage_b_seal": {lora_miss},
            "fullrank_joint": {stage_b},
            "fullrank_control": {stage_b, fullrank_joint},
        }[args.phase]
    elif args.stage == "evaluate-state" and args.eval_set == "contrast":
        if args.objective != "joint":
            raise SystemExit("contrast evaluation requires --objective joint")
        allowed_authorizations = {stage_b}
    elif args.stage in {"model-smoke", "positive-control"} and args.capacity == "lora":
        allowed_authorizations = set()
    elif args.stage in {"model-smoke", "positive-control"} and args.capacity == "fullrank":
        allowed_authorizations = {lora_miss}
    elif args.capacity == "lora" and args.objective == "joint":
        allowed_authorizations = set()
    elif args.capacity == "fullrank" and args.objective == "state_only":
        allowed_authorizations = {stage_b, fullrank_joint}
    elif args.stage in {"model-smoke", "positive-control", "train", "evaluate-state"}:
        allowed_authorizations = {lora_miss}
    else:
        raise SystemExit("registered CLI cell has no authorization policy")
    supplied = None
    if args.authorization_receipt:
        supplied = _lexical_path(
            args.authorization_receipt, label="--authorization-receipt"
        )
        _require_no_symlink_components(supplied, label="--authorization-receipt")
    if (
        supplied is not None
        if not allowed_authorizations
        else supplied not in allowed_authorizations
    ):
        rendered = ", ".join(map(str, sorted(allowed_authorizations))) or "none"
        raise SystemExit(f"--authorization-receipt must be one of: {rendered}")


def cpu_smoke(config: dict, output: Path) -> dict:
    substrate = config["substrate"]
    architecture = config["architecture"]
    rows = []
    cells = (
        (substrate["train_families"][0], substrate["train_templates"][0], 2),
        (substrate["train_families"][1], substrate["train_templates"][1], 4),
        (substrate["heldout_family"], substrate["heldout_template"], 8),
    )
    for index, (family, template, depth) in enumerate(cells):
        row = generate_example(
            seed=88300 + index,
            split="cpu_smoke",
            family=family,
            template=template,
            depth=depth,
            node_count=int(substrate["node_count"]),
            checksum_modulus=int(substrate["checksum_modulus"]),
            num_choices=int(substrate["num_choices"]),
            state_token=architecture["state_token"],
            state_slots=int(architecture["state_slots"]),
            max_attempts=int(substrate["max_generation_attempts"]),
            query_kind=("node", "checksum")[index % 2],
        )
        verify_example(row, architecture["state_token"], int(architecture["state_slots"]))
        rows.append(row)
    receipt = {
        "status": "CPU_SMOKE_PASS",
        "config": resolved_config_receipt(config),
        "generated_families": [row["family"] for row in rows],
        "generated_depths": [row["depth"] for row in rows],
        "benchmark_files_read": 0,
        "gpu_model_loaded": False,
        "scientific_evidence": False,
    }
    _write_json(output, receipt)
    return receipt


def _gpu_stage(args: argparse.Namespace, config: dict) -> int:
    try:
        from src import gpu_runner
    except Exception as exc:
        raise SystemExit(
            "GPU stages require the pinned requirements-training.lock.txt environment.\n"
            f"Import failure: {exc}"
        ) from exc
    authorization = Path(args.authorization_receipt) if args.authorization_receipt else None
    if args.stage == "model-smoke":
        gpu_runner.model_smoke(
            config,
            Path(args.output),
            capacity=args.capacity,
            model_seed=args.seed,
            initialization_bundle=Path(args.initialization_bundle),
            authorization_receipt=authorization,
        )
    elif args.stage == "positive-control":
        gpu_runner.positive_control(
            config,
            Path(args.output),
            capacity=args.capacity,
            model_seed=args.seed,
            initialization_bundle=Path(args.initialization_bundle),
            model_smoke_receipt=Path(args.model_smoke_receipt),
            authorization_receipt=authorization,
        )
    elif args.stage == "train":
        gpu_runner.train(
            config,
            capacity=args.capacity,
            objective=args.objective,
            model_seed=args.seed,
            output_dir=Path(args.output),
            initialization_bundle=Path(args.initialization_bundle),
            model_smoke_receipt=Path(args.model_smoke_receipt),
            positive_control_receipt=Path(args.positive_control_receipt),
            authorization_receipt=authorization,
        )
    elif args.stage == "evaluate-state":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for evaluate-state")
        gpu_runner.evaluate_state(
            config,
            checkpoint=Path(args.checkpoint),
            capacity=args.capacity,
            objective=args.objective,
            model_seed=args.seed,
            eval_set=args.eval_set,
            output_dir=Path(args.output),
            authorization_receipt=authorization,
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--stage",
        choices=(
            "cpu-smoke", "design-boundary", "prepare-data", "prepare-init", "model-smoke",
            "positive-control", "train", "evaluate-state", "analyze",
        ),
        default="cpu-smoke",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--capacity", choices=("lora", "fullrank"), default="lora")
    parser.add_argument("--objective", choices=("joint", "state_only"), default="joint")
    parser.add_argument(
        "--phase",
        choices=(
            "lora_joint", "lora_control", "stage_b_seal",
            "fullrank_joint", "fullrank_control",
        ),
    )
    parser.add_argument("--eval-set", choices=("trigger", "contrast"), default="trigger")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--initialization-bundle")
    parser.add_argument("--model-smoke-receipt")
    parser.add_argument("--positive-control-receipt")
    parser.add_argument("--authorization-receipt")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke and args.stage != "cpu-smoke":
        raise SystemExit("--smoke cannot override a non-cpu-smoke --stage")
    config = load_config(args.config)
    _validate_registered_cell(args)
    model_stages = {"prepare-init", "model-smoke", "positive-control", "train", "evaluate-state"}
    if args.stage in model_stages and args.seed is None:
        raise SystemExit(f"--seed is required for {args.stage}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.initialization_bundle is None and args.seed is not None:
        args.initialization_bundle = str(
            (ROOT / config["paths"]["large_artifacts_dir"] / f"initialization_seed{args.seed}.pt").resolve()
        )
    if args.model_smoke_receipt is None:
        args.model_smoke_receipt = str(
            ROOT / "runs" / "setup" / f"g0_{args.capacity}_seed{args.seed}.json"
        )
    if args.positive_control_receipt is None:
        args.positive_control_receipt = str(
            ROOT / "runs" / "setup" / f"positive_control_{args.capacity}_seed{args.seed}.json"
        )
    canonical_output = _canonical_output(args, config)
    if args.output is None:
        args.output = str(canonical_output)
    elif not _same_lexical_path(args.output, canonical_output, label="--output"):
        raise SystemExit(
            f"--output must be the canonical path for {args.stage}: {canonical_output}"
        )
    _require_no_symlink_components(canonical_output, label="--output")
    _require_canonical_inputs(args, config)
    # Import every stage implementation before opening the reviewed snapshot.
    # A later import from a replaced pathname could otherwise execute bytes
    # other than those held and authorized by that snapshot.
    if args.stage in model_stages:
        try:
            from src import gpu_runner as _preloaded_gpu_runner  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "GPU stages require the pinned requirements-training.lock.txt environment.\n"
                f"Import failure: {exc}"
            ) from exc
    elif args.stage == "analyze":
        from src import analysis as _preloaded_analysis  # noqa: F401
    loaded_source = source_contract_sha256()
    if loaded_source != LOADED_SOURCE_CONTRACT_SHA256:
        raise RuntimeError("source contract changed while CLI modules were loading")
    setup_lock_stages = {
        "cpu-smoke",
        "prepare-data",
        "prepare-init",
        "model-smoke",
        "positive-control",
        "train",
        "evaluate-state",
        "analyze",
    }
    with ExitStack() as execution_locks:
        if args.stage in setup_lock_stages:
            execution_locks.enter_context(
                locked_regular(EXECUTION_GENERATION_LOCK)
            )
        with authorized_source_execution_snapshot() as authorized_source:
            if authorized_source != loaded_source:
                raise RuntimeError(
                    "loaded CLI modules do not match the reviewed source-contract snapshot"
                )
            if args.stage == "cpu-smoke":
                print(json.dumps(cpu_smoke(config, Path(args.output)), indent=2, sort_keys=True))
                return 0
            if args.stage == "design-boundary":
                print(json.dumps(freeze_design(config, Path(args.output)), indent=2, sort_keys=True))
                return 0
            if args.stage == "prepare-data":
                print(json.dumps(build_datasets(config, Path(args.output)), indent=2, sort_keys=True))
                return 0
            if args.stage == "prepare-init":
                print(json.dumps(
                    prepare_initialization_bundle(config, args.seed, Path(args.output)),
                    indent=2, sort_keys=True,
                ))
                return 0
            if args.stage == "analyze":
                from src.analysis import analyze_phase

                print(json.dumps(
                    analyze_phase(
                        config,
                        ROOT / "runs",
                        args.phase,
                        Path(args.output),
                        Path(args.authorization_receipt) if args.authorization_receipt else None,
                    ),
                    indent=2, sort_keys=True,
                ))
                return 0
            return _gpu_stage(args, config)


if __name__ == "__main__":
    raise SystemExit(main())

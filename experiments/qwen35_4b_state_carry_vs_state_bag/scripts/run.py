#!/usr/bin/env python3
"""Orchestrate CPU preparation, GPU mechanics, training, evaluation, and analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolved_config_receipt  # noqa: E402
from src.data_pipeline import build_datasets  # noqa: E402
from src.mechanics import bag_unroll, carry_unroll, recurrent_compute_receipt  # noqa: E402
from src.substrate import generate_counterfactual_pair, generate_example, verify_example  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cpu_smoke(config: dict, output: Path) -> dict:
    substrate = config["substrate"]
    architecture = config["architecture"]
    rows = []
    for index, family in enumerate((*substrate["train_families"], substrate["heldout_family"])):
        row = generate_example(
            seed=88000 + index,
            split="cpu_smoke",
            family=family,
            template=((*substrate["train_templates"], substrate["heldout_template"])[index]),
            depth=(1, 4, 8)[index],
            node_count=int(substrate["node_count"]),
            checksum_modulus=int(substrate["checksum_modulus"]),
            num_choices=int(substrate["num_choices"]),
            state_token=architecture["state_token"],
            state_slots=int(architecture["state_slots"]),
            max_attempts=int(substrate["max_generation_attempts"]),
        )
        verify_example(row, architecture["state_token"], int(architecture["state_slots"]))
        rows.append(row)
    first, second = generate_counterfactual_pair(
        seed=88101,
        split="cpu_smoke_counterfactual",
        family=substrate["train_families"][0],
        template=substrate["train_templates"][0],
        depth=5,
        node_count=int(substrate["node_count"]),
        checksum_modulus=int(substrate["checksum_modulus"]),
        num_choices=int(substrate["num_choices"]),
        state_token=architecture["state_token"],
        state_slots=int(architecture["state_slots"]),
        max_attempts=int(substrate["max_generation_attempts"]),
    )
    verify_example(first, architecture["state_token"], int(architecture["state_slots"]))
    verify_example(second, architecture["state_token"], int(architecture["state_slots"]))

    carry = carry_unroll(1, lambda state, step: state + step, 4)
    bag = bag_unroll(1, lambda state, step: state + step, 4)
    if carry == bag:
        raise AssertionError("reference carry and bag mechanics unexpectedly coincide")
    loop_layers = int(architecture["loop_end"]) - int(architecture["loop_start"])
    carry_compute = recurrent_compute_receipt(
        sequence_tokens=512,
        total_layers=int(architecture["expected_num_layers"]),
        loop_layers=loop_layers,
        k=4,
    )
    bag_compute = recurrent_compute_receipt(
        sequence_tokens=512,
        total_layers=int(architecture["expected_num_layers"]),
        loop_layers=loop_layers,
        k=4,
    )
    if carry_compute != bag_compute:
        raise AssertionError("carry and bag compute receipts differ")
    receipt = {
        "status": "CPU_SMOKE_PASS",
        "config": resolved_config_receipt(config),
        "generated_families": [row["family"] for row in rows],
        "generated_depths": [row["depth"] for row in rows],
        "counterfactual_pair_id": first["pair_id"],
        "counterfactual_distinct_answers": first["answer_letter"] != second["answer_letter"],
        "reference_carry": carry,
        "reference_bag": bag,
        "compute_receipt": carry_compute.__dict__,
        "benchmark_files_read": 0,
        "gpu_model_loaded": False,
        "scientific_evidence": False,
    }
    _write_json(output, receipt)
    return receipt


def _gpu_stage(stage: str, args: argparse.Namespace, config: dict) -> int:
    try:
        from src import gpu_runner
    except Exception as exc:
        raise SystemExit(
            "GPU stages require the root requirements-training.lock.txt environment. "
            "Rebuild .venv exactly as docs/compute_environment.md specifies.\n"
            f"Import failure: {exc}"
        ) from exc
    if stage == "model-smoke":
        gpu_runner.model_smoke(config, Path(args.output))
    elif stage == "train":
        if args.arm not in {"carry", "bag"}:
            raise SystemExit("--arm must be carry or bag for recurrent training")
        gpu_runner.train(
            config,
            arm=args.arm,
            seed=args.seed,
            output_dir=Path(args.output),
            pilot=args.pilot,
            model_smoke_receipt=Path(args.model_smoke_receipt),
            promotion_receipt=(
                Path(args.promotion_receipt) if args.promotion_receipt else None
            ),
        )
    elif stage == "evaluate":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for evaluation")
        gpu_runner.evaluate(
            config,
            checkpoint=Path(args.checkpoint),
            arm=args.arm,
            expected_seed=args.seed,
            output_dir=Path(args.output),
            pilot=args.pilot,
        )
    elif stage == "text-baseline":
        gpu_runner.train_text_baseline(
            config,
            seed=args.seed,
            output_dir=Path(args.output),
            mechanism_receipt=Path(args.mechanism_receipt),
        )
    elif stage == "sample-more":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for sample-more")
        gpu_runner.evaluate_sample_more(
            config,
            checkpoint=Path(args.checkpoint),
            expected_seed=args.seed,
            output_dir=Path(args.output),
            mechanism_receipt=Path(args.mechanism_receipt),
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--stage",
        choices=(
            "cpu-smoke",
            "prepare-data",
            "model-smoke",
            "train",
            "evaluate",
            "text-baseline",
            "sample-more",
            "analyze",
        ),
        default="cpu-smoke",
    )
    parser.add_argument("--smoke", action="store_true", help="alias for --stage cpu-smoke")
    parser.add_argument("--arm", default="carry")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--pilot", action="store_true", help="run the phase-one limited evaluation")
    parser.add_argument(
        "--model-smoke-receipt",
        default=str(ROOT / "runs" / "model_smoke" / "receipt.json"),
        help="matching MODEL_SMOKE_PASS receipt required by recurrent training",
    )
    parser.add_argument(
        "--promotion-receipt",
        default=str(ROOT / "analysis" / "summary.json"),
        help="matching PILOT_PROMOTION_READY receipt required by full recurrent training",
    )
    parser.add_argument(
        "--mechanism-receipt",
        default=str(ROOT / "analysis" / "summary.json"),
        help="matching MECHANISTIC_DEPTH_POSITIVE receipt required by text/sample-more stages",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.stage = "cpu-smoke"
    config = load_config(args.config)
    if args.stage in {"train", "evaluate", "text-baseline", "sample-more"} and args.seed is None:
        raise SystemExit(f"--seed is required for the {args.stage} stage")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.output is None:
        defaults = {
            "cpu-smoke": ROOT / "runs" / "cpu_smoke" / "receipt.json",
            "prepare-data": ROOT / config["paths"]["data_dir"],
            "model-smoke": ROOT / "runs" / "model_smoke" / "receipt.json",
            "train": ROOT
            / config["paths"]["large_artifacts_dir"]
            / f"{'pilot_' if args.pilot else ''}{args.arm}_seed{args.seed}",
            "evaluate": ROOT
            / "runs"
            / f"{'pilot_' if args.pilot else 'full_'}{args.arm}_seed{args.seed}",
            "text-baseline": ROOT / config["paths"]["large_artifacts_dir"] / f"text_seed{args.seed}",
            "sample-more": ROOT / "runs" / f"sample_more_seed{args.seed}",
            "analyze": ROOT / "analysis" / "summary.json",
        }
        args.output = str(defaults[args.stage].resolve())
    if args.stage == "cpu-smoke":
        receipt = cpu_smoke(config, Path(args.output))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.stage == "prepare-data":
        manifest = build_datasets(config, Path(args.output))
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.stage == "analyze":
        from src.analysis import analyze_runs

        summary = analyze_runs(config, ROOT / "runs", Path(args.output))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return _gpu_stage(args.stage, args, config)


if __name__ == "__main__":
    raise SystemExit(main())

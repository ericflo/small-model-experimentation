#!/usr/bin/env python3
"""Restartable, immutable-design-gated context-local clamp orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from io_utils import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from task_data import (  # noqa: E402
    CONCEPTS,
    DIGITS,
    consequence_prompt,
    direct_prompt,
    fingerprint,
    generate_splits,
    shared_prefix,
)


CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = str(boundary["commit"])
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode == 0
    paths = {
        "preregistration": (
            "experiments/qwen35_4b_context_local_jacobian_clamp/reports/preregistration.md"
        ),
        "readme": "experiments/qwen35_4b_context_local_jacobian_clamp/README.md",
    }
    observed = {
        name: hashlib.sha256(_git(["show", f"{commit}:{path}"]).stdout.encode()).hexdigest()
        for name, path in paths.items()
    }
    expected = {
        "preregistration": str(boundary["preregistration_sha256"]),
        "readme": str(boundary["readme_sha256"]),
    }
    passed = bool(ancestor and observed == expected)
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "scientific_result": False,
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"immutable design boundary failed: {receipt}")
    return receipt


def _write_splits(config: dict[str, Any]) -> dict[str, Any]:
    splits = generate_splits(config)
    paths = {}
    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    manifest = {
        "schema_version": 1,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "seeds": config["seeds"],
        "splits": {
            name: {
                "path": str(path.relative_to(EXP)),
                "sha256": sha256_file(path),
                "items": len(splits[name]),
                "unique_fingerprints": len({fingerprint(row) for row in splits[name]}),
                "source_counts": {
                    concept: sum(row["source"] == concept for row in splits[name])
                    for concept in CONCEPTS
                },
                "example_character_lengths": {
                    "shared_prefix": len(shared_prefix(splits[name][0])),
                    "direct_user": len(direct_prompt(splits[name][0])),
                    "consequence_user": len(consequence_prompt(splits[name][0])),
                },
            }
            for name, path in paths.items()
        },
        "dictionary": {"concepts": list(CONCEPTS), "digits": list(DIGITS)},
        "scientific_result": False,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    return manifest


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("the repository model boundary permits only Qwen/Qwen3.5-4B")
    manifest = _write_splits(config)
    receipt = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "scientific_result": False,
        "data_manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "split_sizes": {
            name: details["items"] for name, details in manifest["splits"].items()
        },
        "candidate_bands": config["intervention"]["candidate_bands"],
        "alpha": config["intervention"]["alpha"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _prepare_item(model, row: dict[str, Any], *, kind: str, selected: str, max_length: int):
    builder = direct_prompt if kind == "direct" else consequence_prompt
    return model.prepare(
        builder(row, selected=selected),
        kind=kind,
        selected_concept=selected,
        max_length=max_length,
    )


def run_model_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    if not (DATA_DIR / "manifest.json").exists():
        _write_splits(config)
    import time

    import torch
    import transformers

    from coordinates import dictionary_stats
    from model_ops import CoordinateClampPatcher, FullActivationPatcher, QwenClampModel

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = QwenClampModel(config)
    max_length = int(config["lens"]["max_sequence_tokens"])
    token_contract = {
        "concepts": {concept: model.concept_token_id(concept) for concept in CONCEPTS},
        "digits": {digit: model.bare_token_id(digit) for digit in DIGITS},
    }
    rows = read_jsonl(DATA_DIR / "lens_fit.jsonl")[:2]
    row = rows[0]
    prepared = {
        (kind, selected): _prepare_item(
            model, row, kind=kind, selected=selected, max_length=max_length
        )
        for kind in ("direct", "consequence")
        for selected in (row["source"], row["target"])
    }
    source_direct = prepared[("direct", row["source"])]
    target_direct = prepared[("direct", row["target"])]
    source_consequence = prepared[("consequence", row["source"])]
    layers = (4, 16, 28)
    direct_capture = model.capture(source_direct, layers=layers)
    consequence_capture = model.capture(source_consequence, layers=layers)
    target_capture = model.capture(target_direct, layers=layers)
    causal_max_abs = max(
        float((direct_capture["activations"][layer] - consequence_capture["activations"][layer]).abs().max())
        for layer in layers
    )

    if source_direct["sequence_tokens"] != target_direct["sequence_tokens"]:
        raise RuntimeError("source and target donor prompt lengths differ")
    batch_ids = torch.cat([source_direct["input_ids"], target_direct["input_ids"]], dim=0)
    with torch.no_grad():
        batch_logits = model.model(
            input_ids=batch_ids, use_cache=False, logits_to_keep=1
        ).logits[:, -1, :].float().cpu()
    batch_max_abs = max(
        float((batch_logits[0] - direct_capture["logits"]).abs().max()),
        float((batch_logits[1] - target_capture["logits"]).abs().max()),
    )

    smoke_concepts = CONCEPTS[:4]
    prepared_fit = [
        _prepare_item(
            model,
            fit_row,
            kind="direct",
            selected=fit_row["source"],
            max_length=max_length,
        )
        for fit_row in rows
    ]
    lens, prompt_receipts = model.fit_context_lens(
        prepared_fit,
        smoke_concepts,
        source_layers=layers,
        concept_batch=4,
    )
    stats = {layer: dictionary_stats(lens.directions[layer], rtol=1e-5) for layer in layers}
    desired = model.donor_coordinates(
        {16: target_capture["activations"][16]},
        {16: lens.directions[16]},
        rtol=1e-5,
    )
    coordinate_patcher = CoordinateClampPatcher(
        model.layers,
        source_direct["position"],
        {16: lens.directions[16]},
        desired,
        rtol=1e-5,
    )
    model.score(source_direct, patcher=coordinate_patcher)
    donor_patcher = FullActivationPatcher(
        model.layers,
        source_direct["position"],
        {16: target_capture["activations"][16]},
    )
    model.score(source_direct, patcher=donor_patcher)

    smoke_dir = RUNS_DIR / "model_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lens.state_dict(), smoke_dir / "context_lens.pt")
    position_contract = {
        f"{kind}_{selected}": {
            "position": value["position"],
            "sequence_tokens": value["sequence_tokens"],
            "token_id": value["selected_token_id"],
        }
        for (kind, selected), value in prepared.items()
    }
    passed = bool(
        design["passed"]
        and model.n_layers == 32
        and model.d_model == 2560
        and all(stat.effective_rank == 4 for stat in stats.values())
        and causal_max_abs <= float(config["intervention"]["causal_activation_atol"])
        and batch_max_abs <= float(config["intervention"]["clean_batch_logit_atol"])
        and source_direct["position"] == source_consequence["position"]
        and source_direct["position"] == target_direct["position"]
        and float(coordinate_patcher.deltas[16].norm()) > 0
        and float(donor_patcher.deltas[16].norm()) > 0
    )
    result = {
        "schema_version": 1,
        "stage": "model_smoke",
        "passed": passed,
        "scientific_result": False,
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "layers": model.n_layers,
            "hidden_size": model.d_model,
            "vocab_size": model.vocab_size,
            "load_seconds": model.load_seconds,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "token_contract": token_contract,
        "position_contract": position_contract,
        "causal_activation_max_abs": causal_max_abs,
        "clean_batch_logit_max_abs": batch_max_abs,
        "smoke_lens": {
            "layers": list(layers),
            "n_prompts": lens.n_prompts,
            "effective_ranks": {str(layer): stats[layer].effective_rank for layer in layers},
            "condition_numbers": {str(layer): stats[layer].condition_number for layer in layers},
            "prompt_receipts": prompt_receipts,
            "artifact_sha256": sha256_file(smoke_dir / "context_lens.pt"),
        },
        "coordinate_delta_norm": float(coordinate_patcher.deltas[16].norm()),
        "donor_delta_norm": float(donor_patcher.deltas[16].norm()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(smoke_dir / "result.json", result)
    if not passed:
        raise RuntimeError(f"model smoke failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_fit_lens(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    import time

    import torch

    from coordinates import dictionary_stats
    from model_ops import QwenClampModel

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = QwenClampModel(config)
    lens_config = config["lens"]
    source_layers = tuple(int(value) for value in lens_config["source_layers"])
    max_length = int(lens_config["max_sequence_tokens"])
    rows = read_jsonl(DATA_DIR / "lens_fit.jsonl")
    prepared = [
        _prepare_item(
            model,
            row,
            kind="direct",
            selected=row["source"],
            max_length=max_length,
        )
        for row in rows
    ]
    lens, prompt_receipts = model.fit_context_lens(
        prepared,
        CONCEPTS,
        source_layers=source_layers,
        concept_batch=int(lens_config["concept_batch"]),
    )
    rtol = float(lens_config["pseudoinverse_rtol"])
    stats = {layer: dictionary_stats(lens.directions[layer], rtol=rtol) for layer in source_layers}
    artifact = RUNS_DIR / "context_lens.pt"
    torch.save(lens.state_dict(), artifact)
    minimum_rank = int(lens_config["minimum_effective_rank"])
    passed = bool(
        design["passed"]
        and len(rows) == int(config["data"]["lens_fit_items"])
        and all(
            stats[layer].effective_rank >= minimum_rank
            and bool(torch.isfinite(lens.directions[layer]).all())
            and bool((lens.directions[layer].norm(dim=0) > 0).all())
            for layer in source_layers
        )
    )
    result = {
        "schema_version": 1,
        "stage": "fit_lens",
        "passed": passed,
        "scientific_result": False,
        "model_revision": config["model"]["revision"],
        "concepts": list(lens.concepts),
        "token_ids": list(lens.token_ids),
        "source_layers": list(lens.source_layers),
        "n_prompts": lens.n_prompts,
        "estimator": lens.estimator,
        "pseudoinverse_rtol": rtol,
        "minimum_effective_rank": minimum_rank,
        "layer_stats": {
            str(layer): {
                "effective_rank": stats[layer].effective_rank,
                "condition_number": stats[layer].condition_number,
                "singular_values": list(stats[layer].singular_values),
                "direction_norms": lens.directions[layer].norm(dim=0).tolist(),
            }
            for layer in source_layers
        },
        "prompt_receipts": prompt_receipts,
        "artifact": {
            "path": str(artifact.relative_to(ROOT)),
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(RUNS_DIR / "lens_fit.json", result)
    if not passed:
        raise RuntimeError(f"context-local lens fit failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def unavailable(stage: str) -> None:
    raise RuntimeError(f"stage {stage!r} is not implemented yet; refusing a placeholder result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "fit-lens", "donor-gate", "confirmation", "full"),
        default="smoke",
    )
    args = parser.parse_args()
    config = load_config()
    if args.stage == "smoke":
        run_smoke(config)
        return 0
    if args.stage == "model-smoke":
        run_model_smoke(config)
        return 0
    if args.stage == "fit-lens":
        run_fit_lens(config)
        return 0
    design_boundary_receipt(config)
    unavailable(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

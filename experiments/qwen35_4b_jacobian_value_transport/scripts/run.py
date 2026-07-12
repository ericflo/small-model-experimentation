#!/usr/bin/env python3
"""Restartable, immutable-design-gated Jacobian value-transport orchestrator."""

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
sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from io_utils import read_jsonl, sha256_file, write_json  # noqa: E402
from task_data import CONCEPT_CANDIDATES, build_splits  # noqa: E402

CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = boundary.get("commit")
    expected_prereg = boundary.get("preregistration_sha256")
    expected_readme = boundary.get("readme_sha256")
    if not commit or not expected_prereg or not expected_readme:
        raise RuntimeError("design boundary has not been anchored to a commit")
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = _git(["merge-base", "--is-ancestor", str(commit), head], check=False).returncode == 0
    paths = {
        "preregistration": "experiments/qwen35_4b_jacobian_value_transport/reports/preregistration.md",
        "readme": "experiments/qwen35_4b_jacobian_value_transport/README.md",
    }
    observed = {}
    for name, path in paths.items():
        payload = _git(["show", f"{commit}:{path}"]).stdout.encode()
        observed[name] = hashlib.sha256(payload).hexdigest()
    passed = bool(
        ancestor
        and observed["preregistration"] == expected_prereg
        and observed["readme"] == expected_readme
    )
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": {
            "preregistration": expected_prereg,
            "readme": expected_readme,
        },
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"immutable design boundary failed: {receipt}")
    return receipt


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    manifest = build_splits(DATA_DIR, config)
    required = {
        "lens_fit", "positive_control", "value_calibration", "iid_eval",
        "held_string_eval", "held_register_eval", "hard_eval",
    }
    passed = required == set(manifest["counts"]) and all(manifest["counts"][name] > 0 for name in required)
    receipt = {
        "schema_version": 1,
        "scientific_evidence": False,
        "passed": passed,
        "manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "counts": manifest["counts"],
        "benchmark_content_used": manifest["firewall"]["benchmark_content_used"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"data smoke failed: {receipt}")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def run_model_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    if not (DATA_DIR / "manifest.json").exists():
        build_splits(DATA_DIR, config)
    import torch
    import transformers
    from model_ops import QwenTransportModel

    started = __import__("time").perf_counter()
    model = QwenTransportModel(config)
    concepts = tuple(CONCEPT_CANDIDATES[:4])
    concept_ids = model.audit_concepts(concepts)
    corpus = [row["text"] for row in read_jsonl(DATA_DIR / "lens_fit.jsonl")[:2]]
    lens, prompt_receipts = model.fit_targeted_lens(
        corpus,
        concepts,
        source_layers=(8, 16, 24),
        target_layer=int(config["lens"]["target_layer"]),
        concept_batch=4,
        max_sequence_tokens=64,
        skip_first_positions=int(config["lens"]["skip_first_positions"]),
    )
    smoke_dir = RUNS_DIR / "model_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lens.state_dict(), smoke_dir / "targeted_lens.pt")
    rendered = model.render(
        "Think briefly, then answer with exactly `Concept: cat`.", enable_thinking=True
    )
    generation = model.generate_full_recompute(
        rendered,
        max_new_tokens=8,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=int(config["seeds"]["generation"]),
    )
    patched_generation = model.generate_full_recompute(
        rendered,
        max_new_tokens=2,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=int(config["seeds"]["generation"]),
        layer_directions={
            16: (lens.directions[16][0], lens.directions[16][1]),
        },
        alpha=1.0,
    )
    finite = all(
        bool(torch.isfinite(value).all()) and float(value.norm()) > 0
        for value in lens.directions.values()
    )
    result = {
        "schema_version": 1,
        "scientific_evidence": False,
        "passed": bool(
            design["passed"]
            and model.n_layers == 32
            and model.d_model == 2560
            and len(concept_ids) == 4
            and finite
            and generation["sampled_tokens"] > 0
            and patched_generation["sampled_tokens"] > 0
            and patched_generation["mean_patch_delta_norm"] > 0
        ),
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "layers": model.n_layers,
            "hidden_size": model.d_model,
            "vocab_size": model.vocab_size,
            "dtype": config["model"]["dtype"],
            "attention": config["model"]["attention"],
            "load_seconds": model.load_seconds,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "concept_token_ids": concept_ids,
        "lens": {
            "source_layers": list(lens.source_layers),
            "target_layer": lens.target_layer,
            "n_prompts": lens.n_prompts,
            "pair_weighting": lens.pair_weighting,
            "direction_norms": {
                str(layer): lens.directions[layer].norm(dim=1).tolist()
                for layer in lens.source_layers
            },
            "prompt_receipts": prompt_receipts,
            "artifact_sha256": sha256_file(smoke_dir / "targeted_lens.pt"),
        },
        "generation": generation,
        "patched_generation": patched_generation,
        "elapsed_seconds": __import__("time").perf_counter() - started,
    }
    write_json(smoke_dir / "result.json", result)
    if not result["passed"]:
        raise RuntimeError(f"model smoke failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def unavailable_stage(stage: str) -> None:
    raise RuntimeError(
        f"stage {stage!r} is not implemented yet; refusing to emit a placeholder result"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "fit-lens", "positive-control", "prefix-value", "causal-patch", "full"),
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
    design_boundary_receipt(config)
    unavailable_stage(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
